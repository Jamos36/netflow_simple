from __future__ import annotations

import glob
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class DataInfo:
    files: list[Path]
    schema: pa.Schema
    min_date: date
    max_date: date
    train_end: date
    validation_end: date
    rows: int
    split_rows: dict[str, int]


def resolve_files(patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for raw in patterns:
        path = Path(raw)
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*.parquet") if p.is_file())
        elif path.is_file():
            found.add(path.resolve())
        else:
            found.update(Path(p).resolve() for p in glob.glob(raw, recursive=True) if Path(p).is_file())
    files = sorted(found)
    if not files:
        raise FileNotFoundError(f"No Parquet files found for data.paths={patterns}")
    return files


def unified_schema(files: list[Path]) -> pa.Schema:
    schemas = [pq.read_schema(path) for path in files]
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except Exception as exc:
        raise ValueError(f"Parquet files do not have compatible schemas: {exc}") from exc


def iter_file_batches(files: list[Path], schema: pa.Schema, batch_rows: int):
    """Yield (file path, zero-based file row offset, batch) with columns aligned by name."""
    for path in files:
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=True):
            arrays = []
            for field in schema:
                index = batch.schema.get_field_index(field.name)
                if index < 0:
                    arrays.append(pa.nulls(batch.num_rows, type=field.type))
                    continue
                array = batch.column(index)
                if array.type != field.type:
                    try:
                        array = pc.cast(array, field.type, safe=False)
                    except Exception as exc:
                        raise ValueError(f"Column {field.name!r} has incompatible types in {path}: "
                                         f"{array.type} vs {field.type}") from exc
                arrays.append(array)
            aligned = pa.RecordBatch.from_arrays(arrays, schema=schema)
            yield path, offset, aligned
            offset += batch.num_rows


def to_dates(array: pa.Array, timestamp_unit: str | None = None) -> np.ndarray:
    if pa.types.is_timestamp(array.type):
        raw = pc.cast(array, pa.int64()).to_numpy(zero_copy_only=False)
        unit = array.type.unit
        divisor = {"s": 86_400, "ms": 86_400_000, "us": 86_400_000_000,
                   "ns": 86_400_000_000_000}[unit]
        days = np.floor_divide(raw, divisor).astype("int64", copy=False)
        days = np.where(np.asarray(array.is_null()), np.iinfo(np.int64).min, days)
        return days.astype("datetime64[D]")
    if pa.types.is_date32(array.type):
        raw = pc.cast(array, pa.int32()).to_numpy(zero_copy_only=False).astype("int64")
        raw = np.where(np.asarray(array.is_null()), np.iinfo(np.int64).min, raw)
        return raw.astype("datetime64[D]")
    if pa.types.is_date64(array.type):
        raw = pc.cast(array, pa.int64()).to_numpy(zero_copy_only=False)
        days = np.floor_divide(raw, 86_400_000)
        days = np.where(np.asarray(array.is_null()), np.iinfo(np.int64).min, days)
        return days.astype("datetime64[D]")
    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        if timestamp_unit not in {"s", "ms", "us", "ns"}:
            raise ValueError("Numeric timestamps require data.timestamp_unit: s, ms, us, or ns")
        values = pd.to_datetime(array.to_numpy(zero_copy_only=False), unit=timestamp_unit, utc=True,
                                errors="coerce")
    else:
        values = pd.to_datetime(array.to_pylist(), utc=True, errors="coerce")
    return values.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]")


def _boundaries(first: date, last: date, cfg: dict) -> tuple[date, date]:
    split = cfg["split"]
    if split.get("train_end") and split.get("validation_end"):
        train_end = date.fromisoformat(str(split["train_end"]))
        validation_end = date.fromisoformat(str(split["validation_end"]))
    else:
        n_days = (last - first).days + 1
        train_days = max(1, int(n_days * float(split["train_fraction"])))
        validation_days = max(1, int(n_days * float(split["validation_fraction"])))
        if train_days + validation_days >= n_days:
            raise ValueError(f"Need at least 3 UTC dates for 60/20/20 split; found {n_days}")
        train_end = first + timedelta(days=train_days)
        validation_end = train_end + timedelta(days=validation_days)
    if not first < train_end < validation_end <= last + timedelta(days=1):
        raise ValueError("Split boundaries must satisfy first_date < train_end < validation_end <= day_after_last")
    return train_end, validation_end


def inspect_data(files: list[Path], schema: pa.Schema, cfg: dict) -> DataInfo:
    timestamp = cfg["data"]["timestamp_column"]
    if timestamp not in schema.names:
        raise ValueError(f"Timestamp column {timestamp!r} not found. Available: {schema.names}")
    first = None
    last = None
    rows = 0
    counts_by_day: dict[date, int] = {}
    for _, _, batch in iter_file_batches(files, schema, int(cfg["data"]["batch_rows"])):
        dates = to_dates(batch.column(schema.get_field_index(timestamp)), cfg["data"].get("timestamp_unit"))
        valid = dates[~np.isnat(dates)]
        if len(valid):
            lo = pd.Timestamp(valid.min()).date()
            hi = pd.Timestamp(valid.max()).date()
            first = lo if first is None or lo < first else first
            last = hi if last is None or hi > last else last
            unique, counts = np.unique(valid, return_counts=True)
            for day_value, count in zip(unique, counts):
                day = pd.Timestamp(day_value).date()
                counts_by_day[day] = counts_by_day.get(day, 0) + int(count)
        rows += batch.num_rows
    if first is None or last is None:
        raise ValueError(f"Timestamp column {timestamp!r} has no parseable values")
    train_end, validation_end = _boundaries(first, last, cfg)
    split_rows = {"train": 0, "validation": 0, "test": 0, "invalid_timestamp": rows - sum(counts_by_day.values())}
    for day, count in counts_by_day.items():
        name = "train" if day < train_end else "validation" if day < validation_end else "test"
        split_rows[name] += count
    return DataInfo(files, schema, first, last, train_end, validation_end, rows, split_rows)


def split_dates(dates: np.ndarray, info: DataInfo) -> np.ndarray:
    result = np.full(len(dates), "invalid", dtype="U12")
    valid = ~np.isnat(dates)
    train_end = np.datetime64(info.train_end, "D")
    validation_end = np.datetime64(info.validation_end, "D")
    result[valid & (dates < train_end)] = "train"
    result[valid & (dates >= train_end) & (dates < validation_end)] = "validation"
    result[valid & (dates >= validation_end)] = "test"
    return result
