from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.feature_extraction import FeatureHasher

from anomaly_poc.data import to_dates


CATEGORICAL_NAME_HINTS = (
    "ip", "addr", "address", "port", "protocol", "proto", "flag", "vlan", "asn", "class",
    "status", "reason", "interface", "ifindex", "domain", "name", "type", "sequence", "code", "id",
    "subnet", "socket",
)


def _as_text(value) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "<NULL>"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_numeric(dtype: pa.DataType) -> bool:
    return (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype))


def _is_temporal(dtype: pa.DataType) -> bool:
    return pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)


@dataclass
class ColumnPlan:
    numeric: list[str]
    temporal: list[str]
    categorical: list[str]
    excluded: list[str]
    source_columns: list[str]


class AllColumnEncoder:
    """Numeric robust scaling + cyclic time terms + fixed-width categorical hashing."""

    def __init__(self, schema: pa.Schema, timestamp_column: str, timestamp_unit: str | None,
                 exclude_columns: list[str], hash_features: int):
        self.schema = schema
        self.timestamp_column = timestamp_column
        self.timestamp_unit = timestamp_unit
        self.hash_features = int(hash_features)
        excluded = set(exclude_columns)
        unknown = excluded - set(schema.names)
        if unknown:
            raise ValueError(f"data.exclude_columns not found in Parquet schema: {sorted(unknown)}")
        numeric, temporal, categorical = [], [], []
        for field in schema:
            name = field.name
            if name in excluded:
                continue
            lower_name = name.casefold()
            if (name == timestamp_column or _is_temporal(field.type)
                    or lower_name.endswith(("_time", "_timestamp", "_stamp"))):
                temporal.append(name)
            elif (_is_numeric(field.type)
                  and not any(hint in lower_name for hint in CATEGORICAL_NAME_HINTS)
                  and not lower_name.endswith("_as")):
                numeric.append(name)
            else:
                categorical.append(name)
        if timestamp_column not in schema.names:
            raise ValueError(f"Timestamp column {timestamp_column!r} is absent")
        if timestamp_column in excluded:
            raise ValueError("The timestamp column cannot be excluded")
        self.plan = ColumnPlan(numeric, temporal, categorical, sorted(excluded), list(schema.names))
        self.numeric_dim = len(numeric)
        self.temporal_dim = len(temporal) * 6
        self.continuous_dim = self.numeric_dim + self.temporal_dim
        self.missing_dim = self.numeric_dim
        self.medians = np.zeros(self.continuous_dim, dtype=np.float32)
        self.scales = np.ones(self.continuous_dim, dtype=np.float32)
        self.hasher = FeatureHasher(n_features=self.hash_features, input_type="string",
                                    alternate_sign=False, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self.continuous_dim + self.missing_dim + self.hash_features

    def _numeric_values(self, batch: pa.RecordBatch, name: str) -> np.ndarray:
        array = batch.column(batch.schema.get_field_index(name))
        try:
            values = array.cast(pa.float64(), safe=False).to_numpy(zero_copy_only=False).astype(np.float32)
        except Exception:
            values = pd.to_numeric(pd.Series(array.to_pylist()), errors="coerce").to_numpy(dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        return values

    def _time_features(self, batch: pa.RecordBatch, name: str, n: int) -> np.ndarray:
        array = batch.column(batch.schema.get_field_index(name))
        if name == self.timestamp_column:
            # Reuse the configured unit for numeric timestamp columns.
            dates = to_dates(array, self.timestamp_unit)
            if pa.types.is_timestamp(array.type):
                raw = array.to_pylist()
                values = pd.to_datetime(raw, utc=True, errors="coerce")
            elif pa.types.is_date(array.type):
                values = pd.to_datetime(array.to_pylist(), utc=True, errors="coerce")
            else:
                raw = array.to_numpy(zero_copy_only=False) if _is_numeric(array.type) else array.to_pylist()
                values = pd.to_datetime(raw, unit=self.timestamp_unit if _is_numeric(array.type) else None,
                                        utc=True, errors="coerce")
        elif pa.types.is_timestamp(array.type) or pa.types.is_date(array.type):
            values = pd.to_datetime(array.to_pylist(), utc=True, errors="coerce")
        else:
            values = pd.to_datetime(array.to_pylist(), utc=True, errors="coerce")
        idx = pd.DatetimeIndex(values)
        valid = ~idx.isna()
        hour = np.asarray(idx.hour, dtype=np.float64) + np.asarray(idx.minute, dtype=np.float64) / 60.0
        weekday = np.asarray(idx.dayofweek, dtype=np.float64)
        day_year = np.asarray(idx.dayofyear, dtype=np.float64)
        cols = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
                np.sin(2 * np.pi * day_year / 366), np.cos(2 * np.pi * day_year / 366)]
        out = np.column_stack(cols).astype(np.float32, copy=False)
        out[~valid, :] = np.nan
        if len(out) != n:
            raise ValueError(f"Timestamp transform returned {len(out)} rows; expected {n}")
        return out

    def transform_unscaled(self, batch: pa.RecordBatch) -> np.ndarray:
        n = batch.num_rows
        numeric = []
        missing = []
        for name in self.plan.numeric:
            values = self._numeric_values(batch, name)
            missing.append(np.isnan(values).astype(np.float32))
            # Compress very heavy tails without imposing an upper cap.
            values = np.sign(values) * np.log1p(np.abs(values))
            numeric.append(values)
        numeric_matrix = np.column_stack(numeric).astype(np.float32) if numeric else np.empty((n, 0), np.float32)
        missing_matrix = np.column_stack(missing).astype(np.float32) if missing else np.empty((n, 0), np.float32)
        temporal = [self._time_features(batch, name, n) for name in self.plan.temporal]
        temporal_matrix = np.column_stack(temporal).astype(np.float32) if temporal else np.empty((n, 0), np.float32)
        continuous = np.column_stack([numeric_matrix, temporal_matrix]).astype(np.float32, copy=False)

        rows: list[list[str]] = [[] for _ in range(n)]
        for name in self.plan.categorical:
            values = batch.column(batch.schema.get_field_index(name)).to_pylist()
            prefix = f"{name}="
            for i, value in enumerate(values):
                rows[i].append(prefix + _as_text(value))
        hashed = self.hasher.transform(rows).toarray().astype(np.float32, copy=False)
        return np.concatenate([continuous, missing_matrix, hashed], axis=1)

    def fit_scaler(self, training_sample: np.ndarray) -> None:
        if training_sample.ndim != 2 or training_sample.shape[1] != self.output_dim:
            raise ValueError(f"Training sample shape {training_sample.shape} does not match encoder dimension "
                             f"{self.output_dim}")
        if training_sample.shape[0] == 0:
            raise ValueError("Training split produced no encodable rows")
        if self.continuous_dim:
            continuous = training_sample[:, :self.continuous_dim].astype(np.float64)
            medians = np.nanmedian(continuous, axis=0)
            medians[~np.isfinite(medians)] = 0.0
            q25, q75 = np.nanpercentile(continuous, [25, 75], axis=0)
            scales = q75 - q25
            scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
            self.medians = medians.astype(np.float32)
            self.scales = scales.astype(np.float32)

    def transform(self, batch: pa.RecordBatch) -> np.ndarray:
        matrix = self.transform_unscaled(batch)
        if self.continuous_dim:
            cont = matrix[:, :self.continuous_dim]
            cont = np.where(np.isfinite(cont), cont, self.medians)
            matrix[:, :self.continuous_dim] = (cont - self.medians) / self.scales
        return matrix.astype(np.float32, copy=False)

    def manifest(self) -> dict:
        return {
            "source_columns": self.plan.source_columns,
            "numeric_columns": self.plan.numeric,
            "timestamp_columns_as_cyclic_features": self.plan.temporal,
            "categorical_hashed_columns": self.plan.categorical,
            "excluded_columns": self.plan.excluded,
            "hash_features": self.hash_features,
            "output_dimensions": self.output_dim,
            "timestamp_policy": "cyclic hour, weekday and day-of-year; absolute epoch not used",
            "numeric_policy": "signed log1p, train-sample median imputation, train-sample IQR scaling",
        }
