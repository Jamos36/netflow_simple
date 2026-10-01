from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class _Reservoir:
    """Small deterministic sample for approximate daily quantiles and plots."""

    def __init__(self, limit: int, seed: int):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0, dtype=np.float64)
        self.values = np.empty(0, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        keys = self.rng.random(len(values))
        keys = np.concatenate((self.keys, keys))
        values = np.concatenate((self.values, values))
        if len(keys) > self.limit:
            keep = np.argpartition(keys, self.limit - 1)[:self.limit]
            keys, values = keys[keep], values[keep]
        self.keys, self.values = keys, values


def _band(score: float, thresholds: dict) -> str:
    for name in ("Critical", "High", "Medium", "Low"):
        if score >= float(thresholds[name]):
            return name
    return "Benign"


def _pentest_span(cfg: dict):
    start = cfg.get("context", {}).get("pentest_start")
    end = cfg.get("context", {}).get("pentest_end")
    if not start or not end:
        return None, None
    return np.datetime64(str(start), "D"), np.datetime64(str(end), "D")


def _style_daily_axis(ax, cfg: dict, split_spans: dict[str, tuple[str, str]]) -> None:
    colors = {"train": "#dceef8", "validation": "#fff0c2", "test": "#e9e0f5"}
    for split, (first, after_last) in split_spans.items():
        ax.axvspan(np.datetime64(first), np.datetime64(after_last), color=colors[split], alpha=0.22,
                   label=f"{split.title()} period")
    start, end = _pentest_span(cfg)
    if start is not None:
        ax.axvspan(start, end, color="#e45756", alpha=0.11, label="Pentest date range (context only)")
    else:
        months = sorted(set(int(month) for month in cfg.get("context", {}).get("pentest_months", [])))
        if months and split_spans:
            all_dates = [value for span in split_spans.values() for value in span]
            years = range(min(int(value[:4]) for value in all_dates),
                          max(int(value[:4]) for value in all_dates) + 1)
            groups: list[list[int]] = []
            for month in months:
                if not 1 <= month <= 12:
                    continue
                if not groups or month > groups[-1][-1] + 1:
                    groups.append([month])
                else:
                    groups[-1].append(month)
            label_used = False
            for year in years:
                for group in groups:
                    start_month = group[0]
                    after_month = group[-1] + 1
                    start_date = np.datetime64(f"{year:04d}-{start_month:02d}-01", "D")
                    end_date = (np.datetime64(f"{year:04d}-{after_month:02d}-01", "D")
                                if after_month <= 12 else np.datetime64(f"{year + 1:04d}-01-01", "D"))
                    ax.axvspan(start_date, end_date, color="#e45756", alpha=0.11,
                               label="Pentest months (context only)" if not label_used else None)
                    label_used = True
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, axis="y", alpha=0.25)


def finish_model_outputs(model_dir: Path, info, cfg: dict, validation_scores: np.ndarray,
                         thresholds: dict) -> None:
    """Convert streamed raw scores to final Parquet and write compact stats/plots."""
    raw_path = model_dir / "_scores_raw.parquet"
    final_path = model_dir / "scores.parquet"
    parquet = pq.ParquetFile(raw_path)
    output_schema = pa.schema([
        ("date", pa.string()), ("split", pa.string()), ("source_file", pa.string()),
        ("row_in_file", pa.int64()), ("anomaly_score", pa.float64()),
        ("validation_percentile", pa.float64()), ("review_band", pa.string()),
        ("above_validation_threshold", pa.bool_()),
    ])
    writer = pq.ParquetWriter(final_path, output_schema, compression="zstd")
    daily = {}
    split_totals = defaultdict(lambda: _Reservoir(40_000, 18))
    split_spans: dict[str, tuple[str, str]] = {}
    threshold = float(thresholds["configured_review_threshold"])
    try:
        for batch in parquet.iter_batches(batch_size=int(cfg["data"]["batch_rows"])):
            table = pa.Table.from_batches([batch])
            dates = np.asarray(table["date"].to_pylist(), dtype=object)
            splits = np.asarray(table["split"].to_pylist(), dtype=object)
            scores = table.column("score_" + model_dir.name).to_numpy(zero_copy_only=False)
            percentiles = np.searchsorted(validation_scores, scores, side="right") / len(validation_scores)
            bands = [_band(float(score), thresholds) for score in scores]
            above = scores >= threshold
            writer.write_table(pa.Table.from_arrays([
                table["date"], table["split"], table["source_file"], table["row_in_file"],
                pa.array(scores, type=pa.float64()), pa.array(percentiles, type=pa.float64()),
                pa.array(bands, type=pa.string()), pa.array(above, type=pa.bool_()),
            ], schema=output_schema))

            for split in np.unique(splits):
                mask = splits == split
                if not np.any(mask):
                    continue
                split_totals[str(split)].add(scores[mask])
                day_values = dates[mask]
                if len(day_values):
                    first, last = sorted(day_values)[0], sorted(day_values)[-1]
                    existing = split_spans.get(str(split))
                    split_spans[str(split)] = (min(first, existing[0]) if existing else first,
                                               max(last, existing[1]) if existing else last)
            # Daily counts and threshold rates are exact; daily quantiles use a bounded sample.
            for day in np.unique(dates):
                for split in np.unique(splits[dates == day]):
                    mask = (dates == day) & (splits == split)
                    key = (str(day), str(split))
                    seed = int.from_bytes(hashlib.blake2b("|".join(key).encode(), digest_size=4).digest(), "little")
                    item = daily.setdefault(key, {"count": 0, "above": 0, "sum": 0.0, "min": np.inf,
                                                  "max": -np.inf, "sample": _Reservoir(5000, seed)})
                    values = scores[mask]
                    item["count"] += int(len(values))
                    item["above"] += int(np.count_nonzero(values >= threshold))
                    item["sum"] += float(np.sum(values, dtype=np.float64))
                    item["min"] = min(item["min"], float(np.min(values)))
                    item["max"] = max(item["max"], float(np.max(values)))
                    item["sample"].add(values)
    finally:
        writer.close()

    rows = []
    for (day, split), item in sorted(daily.items()):
        sample = item["sample"].values
        rows.append({"date": day, "split": split, "rows": item["count"],
                     "score_mean": item["sum"] / item["count"], "score_min": item["min"],
                     "score_max": item["max"], "score_p50_approx": float(np.quantile(sample, .5)),
                     "score_p95_approx": float(np.quantile(sample, .95)),
                     "score_p99_approx": float(np.quantile(sample, .99)),
                     "above_validation_threshold": item["above"],
                     "above_validation_threshold_rate": item["above"] / item["count"]})
    stats_path = model_dir / "daily_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                    ["date", "split", "rows", "score_mean", "score_min", "score_max",
                                     "score_p50_approx", "score_p95_approx", "score_p99_approx",
                                     "above_validation_threshold", "above_validation_threshold_rate"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    (model_dir / "review_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    _plot_daily(model_dir, rows, cfg, split_spans)
    _plot_distribution(model_dir, split_totals, thresholds)


def _plot_daily(model_dir: Path, rows: list[dict], cfg: dict,
                split_spans: dict[str, tuple[str, str]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    _style_daily_axis(ax, cfg, split_spans)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        if selected:
            ax.plot([np.datetime64(row["date"], "D") for row in selected],
                    [row["score_p95_approx"] for row in selected], marker=".", markersize=4,
                    linewidth=1.3, label=f"{split.title()} daily p95")
    ax.set_title("Daily anomaly score trend (higher means stranger)")
    ax.set_ylabel("Raw model score")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(model_dir / "daily_score_trend.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    _style_daily_axis(ax, cfg, split_spans)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        if selected:
            ax.plot([np.datetime64(row["date"], "D") for row in selected],
                    [100 * row["above_validation_threshold_rate"] for row in selected],
                    marker=".", markersize=4, linewidth=1.3, label=f"{split.title()} above cutoff")
    ax.set_title("Daily share above the validation review threshold")
    ax.set_ylabel("Rows above threshold (%)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(model_dir / "daily_alert_rate.png", dpi=160)
    plt.close(fig)


def _plot_distribution(model_dir: Path, reservoirs: dict, thresholds: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for split, color in (("train", "#4c78a8"), ("validation", "#f2a541"), ("test", "#7a65a8")):
        values = reservoirs[split].values if split in reservoirs else np.empty(0)
        if len(values):
            ax.hist(values, bins=70, alpha=0.48, density=True, color=color, label=split.title())
    ax.axvline(float(thresholds["configured_review_threshold"]), color="#d62728", linestyle="--",
               label="Validation review threshold")
    ax.set_title("Anomaly score distributions")
    ax.set_xlabel("Raw model score (higher means stranger)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(model_dir / "score_distribution.png", dpi=160)
    plt.close(fig)


def write_comparison(out_root: Path, model_names: list[str], cfg: dict) -> None:
    comparison = out_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    daily_by_model = {}
    candidates = {}
    for name in model_names:
        model_dir = out_root / name
        with (model_dir / "daily_stats.csv").open(encoding="utf-8", newline="") as handle:
            daily_by_model[name] = list(csv.DictReader(handle))
        selected = set()
        candidate_path = model_dir / "candidates.csv"
        if candidate_path.exists():
            with candidate_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("split") == "test":
                        selected.add((row.get("source_file", ""), row.get("row_in_file", "")))
        candidates[name] = selected

    all_days = sorted({row["date"] for values in daily_by_model.values() for row in values
                       if row["split"] == "test"})
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for name, rows in daily_by_model.items():
        lookup = {row["date"]: float(row["above_validation_threshold_rate"]) * 100
                  for row in rows if row["split"] == "test"}
        ax.plot([np.datetime64(day, "D") for day in all_days],
                [lookup.get(day, np.nan) for day in all_days], marker=".", markersize=4,
                linewidth=1.2, label=name)
    test_spans = {}
    for values in daily_by_model.values():
        days = [row["date"] for row in values if row["split"] == "test"]
        if days:
            test_spans["test"] = (min(days), str(np.datetime64(max(days), "D") + np.timedelta64(1, "D")))
            break
    _style_daily_axis(ax, cfg, test_spans)
    ax.set_title("Model comparison: test-day share above each validation cutoff")
    ax.set_ylabel("Rows above model-specific cutoff (%)")
    ax.legend()
    fig.savefig(comparison / "test_alert_rate_comparison.png", dpi=160)
    plt.close(fig)

    matrix = []
    for first in model_names:
        row = {"model": first}
        for second in model_names:
            union = candidates[first] | candidates[second]
            row[second] = len(candidates[first] & candidates[second]) / len(union) if union else 1.0
        matrix.append(row)
    with (comparison / "top_candidate_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *model_names])
        writer.writeheader()
        writer.writerows(matrix)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    values = np.asarray([[row[name] for name in model_names] for row in matrix], dtype=float)
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(model_names)), model_names, rotation=35, ha="right")
    ax.set_yticks(range(len(model_names)), model_names)
    ax.set_title("Overlap of top test candidates (Jaccard similarity)")
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="#17324d")
    fig.colorbar(image, ax=ax, label="Shared / union")
    fig.savefig(comparison / "top_candidate_overlap.png", dpi=160)
    plt.close(fig)
