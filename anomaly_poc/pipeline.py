from __future__ import annotations

import csv
import heapq
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from anomaly_poc.data import inspect_data, iter_file_batches, resolve_files, split_dates, to_dates, unified_schema
from anomaly_poc.encoding import AllColumnEncoder
from anomaly_poc.reporting import finish_model_outputs, write_comparison
from anomaly_poc.settings import load_config
from anomaly_poc.models import gmm, hbos, iforest, pca

LOG = logging.getLogger("all_columns_poc")
MODEL_MODULES = {"iforest": iforest, "pca": pca, "hbos": hbos, "gmm": gmm}


class MatrixSample:
    """Keep the rows with the smallest seeded random keys; bounded and independent of file size."""

    def __init__(self, limit: int, seed: int):
        self.limit = int(limit)
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0, dtype=np.float64)
        self.rows: np.ndarray | None = None

    def update(self, matrix: np.ndarray) -> None:
        if len(matrix) == 0:
            return
        keys = self.rng.random(len(matrix))
        if self.rows is None:
            all_keys, all_rows = keys, matrix
        else:
            all_keys = np.concatenate([self.keys, keys])
            all_rows = np.vstack([self.rows, matrix])
        if len(all_keys) > self.limit:
            keep = np.argpartition(all_keys, self.limit - 1)[:self.limit]
            self.keys = all_keys[keep]
            self.rows = all_rows[keep]
        else:
            self.keys = all_keys
            self.rows = all_rows


class ScoreReservoir:
    def __init__(self, limit: int, seed: int):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(0, dtype=np.float64)
        self.keys = np.empty(0, dtype=np.float64)
        self.seen = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        self.seen += len(values)
        keys = np.concatenate((self.keys, self.rng.random(len(values))))
        values = np.concatenate((self.values, values))
        if len(keys) > self.limit:
            keep = np.argpartition(keys, self.limit - 1)[:self.limit]
            self.keys, self.values = keys[keep], values[keep]
        else:
            self.keys, self.values = keys, values

    def array(self) -> np.ndarray:
        if not len(self.values):
            raise ValueError("Validation produced no scores; review the chronological split dates")
        return np.sort(np.asarray(self.values, dtype=np.float64))


class CandidateHeap:
    def __init__(self, limit: int):
        self.limit = limit
        self.heaps: dict[tuple[str, str], list[tuple[float, int, dict]]] = {}
        self.sequence = 0

    def offer(self, model: str, split: str, day: str, score: float, row: dict) -> None:
        key = (split, day)
        heap = self.heaps.setdefault(key, [])
        self.sequence += 1
        item = (float(score), self.sequence, row)
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    def rows(self, model: str, split: str, day: str) -> list[dict]:
        return [item[2] for item in sorted(self.heaps.get((split, day), []), reverse=True)]


def _take_batch(batch: pa.RecordBatch, indices: np.ndarray) -> pa.RecordBatch:
    return batch.take(pa.array(indices, type=pa.int64()))


def _training_sample(info, cfg, encoder: AllColumnEncoder) -> np.ndarray:
    sample = MatrixSample(int(cfg["data"]["train_sample_rows"]), int(cfg["models"]["seed"]))
    timestamp = cfg["data"]["timestamp_column"]
    for path, _, batch in iter_file_batches(info.files, info.schema, int(cfg["data"]["batch_rows"])):
        dates = to_dates(batch.column(info.schema.get_field_index(timestamp)), cfg["data"].get("timestamp_unit"))
        split = split_dates(dates, info)
        indices = np.flatnonzero(split == "train")
        if len(indices):
            training_batch = _take_batch(batch, indices)
            sample.update(encoder.transform_unscaled(training_batch))
        LOG.info("read training sample from %s", path.name)
    if sample.rows is None or not len(sample.rows):
        raise ValueError("No training rows were selected")
    encoder.fit_scaler(sample.rows)
    return _scale_sample(sample.rows, encoder)


def _scale_sample(sample: np.ndarray, encoder: AllColumnEncoder) -> np.ndarray:
    matrix = sample.copy()
    if encoder.continuous_dim:
        cont = matrix[:, :encoder.continuous_dim]
        cont = np.where(np.isfinite(cont), cont, encoder.medians)
        matrix[:, :encoder.continuous_dim] = (cont - encoder.medians) / encoder.scales
    return matrix.astype(np.float32, copy=False)


def _raw_schema(model_names: list[str]) -> pa.Schema:
    return pa.schema([("date", pa.string()), ("split", pa.string()), ("source_file", pa.string()),
                      ("row_in_file", pa.int64()), *[(f"score_{name}", pa.float64()) for name in model_names]])


def _date_strings(dates: np.ndarray) -> list[str]:
    return ["" if np.isnat(d) else str(d) for d in dates]


def _row_record(batch: pa.RecordBatch, i: int) -> dict:
    row = batch.slice(i, 1).to_pylist()[0]
    return {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.items()}


def _score_pass(info, cfg, encoder, fitted: dict, model_names: list[str], model_dirs: dict[ str, Path],
                context_dir: Path):
    raw_schema = _raw_schema(model_names)
    writers = {name: pq.ParquetWriter(model_dirs[name] / "_scores_raw.parquet", raw_schema,
                                       compression="zstd") for name in model_names}
    validation = {name: ScoreReservoir(250_000, int(cfg["models"]["seed"]) + i)
                  for i, name in enumerate(model_names)}
    candidates = {name: CandidateHeap(int(cfg["review"]["top_per_day"])) for name in model_names}
    timestamp = cfg["data"]["timestamp_column"]
    try:
        for path, row_offset, batch in iter_file_batches(info.files, info.schema, int(cfg["data"]["batch_rows"])):
            dates = to_dates(batch.column(info.schema.get_field_index(timestamp)), cfg["data"].get("timestamp_unit"))
            splits = split_dates(dates, info)
            valid = splits != "invalid"
            if not np.any(valid):
                continue
            idx = np.flatnonzero(valid)
            selected = _take_batch(batch, idx)
            x = encoder.transform(selected)
            model_scores: dict[str, np.ndarray] = {}
            for name in model_names:
                scores = np.asarray(MODEL_MODULES[name].score(fitted[name], x), dtype=np.float64)
                model_scores[name] = scores
                val_mask = splits[idx] == "validation"
                validation[name].update(scores[val_mask])
                # Keep only the top rows per day for review; full score vectors go to Parquet.
                review_mask = np.isin(splits[idx], ["validation", "test"])
                review_groups = np.flatnonzero(review_mask)
                group_keys = {(str(splits[idx[i]]), str(dates[idx[i]])) for i in review_groups}
                for split_name, day in sorted(group_keys):
                    group = np.asarray([i for i in review_groups
                                        if str(splits[idx[i]]) == split_name and str(dates[idx[i]]) == day],
                                       dtype=np.int64)
                    take = min(len(group), candidates[name].limit)
                    local_top = group[np.argpartition(scores[group], len(group) - take)[-take:]]
                    for local_i in local_top:
                        original_i = int(idx[local_i])
                        score_value = float(scores[local_i])
                        heap = candidates[name].heaps.get((split_name, day), [])
                        if len(heap) >= candidates[name].limit and score_value <= heap[0][0]:
                            continue
                        row = _row_record(batch, original_i)
                        row.update({"model": name, "date": day, "split": split_name,
                                    "source_file": str(path), "row_in_file": row_offset + original_i,
                                    "anomaly_score": score_value})
                        candidates[name].offer(name, split_name, day, score_value, row)
            dates_valid = dates[idx]
            date_text = _date_strings(dates_valid)
            split_text = splits[idx].tolist()
            source_names = [str(path)] * len(idx)
            row_numbers = (row_offset + idx).astype(np.int64)
            for name in model_names:
                table = pa.Table.from_arrays(
                    [pa.array(date_text), pa.array(split_text), pa.array(source_names), pa.array(row_numbers),
                     pa.array(model_scores[name], type=pa.float64())], schema=raw_schema)
                writers[name].write_table(table)
            LOG.info("scored %s rows from %s", f"{len(idx):,}", path.name)
    finally:
        for writer in writers.values():
            writer.close()
    return validation, candidates


def _write_candidates(path: Path, records: list[dict], validation_scores: np.ndarray,
                      cfg: dict) -> None:
    qcfg = cfg["review"]
    quantiles = {name: float(np.quantile(validation_scores, float(qcfg[f"{name.lower()}_quantile"])))
                 for name in ("Critical", "High", "Medium", "Low")}
    threshold = float(np.quantile(validation_scores, float(qcfg["threshold_quantile"])))
    for row in records:
        score = float(row["anomaly_score"])
        row["validation_percentile"] = float(np.searchsorted(validation_scores, score, side="right") /
                                             len(validation_scores))
        row["review_band"] = next((name for name in ("Critical", "High", "Medium", "Low")
                                   if score >= quantiles[name]), "Below threshold")
        row["above_validation_threshold"] = score >= threshold
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _input_manifest(info, cfg, encoder) -> dict:
    return {
        "files": [{"path": str(path), "size_bytes": path.stat().st_size,
                   "modified_ns": path.stat().st_mtime_ns} for path in info.files],
        "input_rows": info.rows,
        "schema": [{"name": field.name, "type": str(field.type)} for field in info.schema],
        "columns": encoder.manifest(),
        "model_settings": cfg["models"],
        "split_row_counts": info.split_rows,
        "split": {"first_date": info.min_date.isoformat(), "last_date": info.max_date.isoformat(),
                  "train_end_exclusive": info.train_end.isoformat(),
                  "validation_end_exclusive": info.validation_end.isoformat(),
                  "test_start": info.validation_end.isoformat(),
                  "method": "chronological whole UTC dates"},
    }


def run(config_path: Path, requested_model: str = "all", run_id: str | None = None) -> Path:
    cfg, _ = load_config(config_path)
    files = resolve_files(cfg["data"]["paths"])
    schema = unified_schema(files)
    info = inspect_data(files, schema, cfg)
    LOG.info("Parquet files: %d | rows: %s | UTC span: %s to %s", len(files), f"{info.rows:,}",
             info.min_date, info.max_date)
    LOG.info("Chronological split: train before %s; validation before %s; test from %s",
             info.train_end, info.validation_end, info.validation_end)
    model_names = list(MODEL_MODULES) if requested_model == "all" else [requested_model]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"allcols-{stamp}"
    out_root = Path(cfg["output"]["directory"]) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    model_dirs = {name: out_root / name for name in model_names}
    for directory in model_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    encoder = AllColumnEncoder(schema, cfg["data"]["timestamp_column"], cfg["data"].get("timestamp_unit"),
                               cfg["data"]["exclude_columns"], int(cfg["data"]["hash_features"]))
    training_sample = _training_sample(info, cfg, encoder)
    LOG.info("training sample: %s rows x %s encoded columns", f"{len(training_sample):,}",
             training_sample.shape[1])
    fitted = {}
    for name in model_names:
        LOG.info("fitting %s", name)
        fitted[name] = MODEL_MODULES[name].fit(training_sample, cfg["models"][name],
                                               int(cfg["models"]["seed"]))
        joblib.dump({"model": fitted[name], "encoder": encoder, "name": name},
                    model_dirs[name] / "model.joblib", compress=3)

    validation, candidate_heaps = _score_pass(info, cfg, encoder, fitted, model_names, model_dirs, out_root)
    for name in model_names:
        reference = validation[name].array()
        threshold = float(np.quantile(reference, float(cfg["review"]["threshold_quantile"])))
        records = []
        for split_name in ("validation", "test"):
            for day in sorted(day for split, day in candidate_heaps[name].heaps if split == split_name):
                records.extend(candidate_heaps[name].rows(name, split_name, day))
        _write_candidates(model_dirs[name] / "candidates.csv", records, reference, cfg)
        thresholds = {band: float(np.quantile(reference, float(cfg["review"][f"{band.lower()}_quantile"])))
                      for band in ("Critical", "High", "Medium", "Low")}
        thresholds["configured_review_threshold"] = threshold
        finish_model_outputs(model_dirs[name], info, cfg, reference, thresholds)
        bundle = joblib.load(model_dirs[name] / "model.joblib")
        bundle["review_thresholds"] = thresholds
        bundle["validation_score_sample"] = reference
        joblib.dump(bundle, model_dirs[name] / "model.joblib", compress=3)
        (model_dirs[name] / "manifest.json").write_text(json.dumps({
            "model": name, "created_utc": stamp, "thresholds": thresholds,
            "validation_score_rows_seen": validation[name].seen,
            **_input_manifest(info, cfg, encoder),
        }, indent=2, default=str), encoding="utf-8")
        (model_dirs[name] / "_scores_raw.parquet").unlink(missing_ok=True)

    if len(model_names) > 1:
        write_comparison(out_root, model_names, cfg)
    (out_root / "run.json").write_text(json.dumps({"run_id": run_id, "models": model_names,
                                                     "created_utc": stamp, "source_files": len(files),
                                                     "rows": info.rows}, indent=2), encoding="utf-8")
    return out_root
