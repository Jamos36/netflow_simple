"""Command line entry point.

    python run.py develop   build train/validation features, fit, calibrate, save model.joblib, development report
    python run.py test      score the configured held-out test period with the saved model (no fitting)
    python run.py score --input 'later/*.parquet' --out results_dir [--start YYYY-MM-DD --end YYYY-MM-DD]

Each analysis writes to its own named folder; a rerun replaces that folder's generated results.
test and score only read model.joblib; only develop writes it.
"""

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import config
import features
import model
import report

OUTPUT_MARKER = ".netflow_simple_output"  # only folders carrying this marker (or empty) are replaced
PACKAGES = ["duckdb", "pyarrow", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"]


def prepare_output(folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        if not (folder / OUTPUT_MARKER).exists() or (folder / "model.joblib").exists():
            raise SystemExit(f"Refusing to replace {folder}: not an output folder created by this tool")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / OUTPUT_MARKER).write_text("Generated results; a rerun replaces this folder.\n")
    return folder


def periods_from_config():
    names = {"train": config.TRAIN_PERIOD, "validation": config.VALIDATION_PERIOD, "test": config.TEST_PERIOD}
    periods = {name: (features.parse_day(a), features.parse_day(b)) for name, (a, b) in names.items()}
    order = [periods["train"], periods["validation"], periods["test"]]
    for (start, end), (next_start, _) in zip(order, order[1:] + [(None, None)]):
        if start >= end or (next_start is not None and end > next_start):
            raise SystemExit("Periods must be non-empty, chronological and non-overlapping: check config.py")
    return periods


def stream_raw_scores(con, bundle, features_glob, out_dir):
    """Validation pass 1: raw scores for every row, streamed to disk in batches, plus a full histogram."""
    out_dir.mkdir(parents=True)
    histogram = np.zeros(len(model.HIST_BINS) - 1, dtype=np.int64)
    reader = con.execute(f"SELECT * FROM read_parquet({features.sql_text(features_glob)})"
                         ).to_arrow_reader(config.SCORE_BATCH_ROWS)
    for number, batch in enumerate(reader):
        raw = model.raw_scores(bundle, model.feature_matrix(batch))
        histogram += np.histogram(raw, bins=model.HIST_BINS)[0]
        table = pa.table({"src_ip": batch.column("src_ip"), "window_start": batch.column("window_start"),
                          "raw_score": pa.array(raw, pa.float64())})
        pq.write_table(table, out_dir / f"part-{number:05d}.parquet")
    return histogram


def score_period(con, bundle, features_glob, out_dir, batch_rows):
    """Score every host-window with the frozen bundle, batch by batch, into out_dir/scores/. Returns histogram."""
    scores_dir = Path(out_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(len(model.HIST_BINS) - 1, dtype=np.int64)
    reader = con.execute(f"SELECT * FROM read_parquet({features.sql_text(features_glob)}) "
                         f"ORDER BY window_start, src_ip").to_arrow_reader(batch_rows)
    for number, batch in enumerate(reader):
        table = model.score_batch(bundle, batch)
        histogram += np.histogram(table.column("raw_score").to_numpy(), bins=model.HIST_BINS)[0]
        pq.write_table(table, scores_dir / f"part-{number:05d}.parquet")
    return histogram


def develop():
    started = time.time()
    periods = periods_from_config()
    con = features.connect()
    files = features.check_input(con, config.INPUT_GLOB)
    out = prepare_output(config.WORK_DIR / "development")
    feature_dir = out / "features"
    train = features.build_period_features(con, config.INPUT_GLOB, "train", *periods["train"], feature_dir)
    valid = features.build_period_features(con, config.INPUT_GLOB, "validation", *periods["validation"], feature_dir)

    sample = model.sample_training_rows(con, str(feature_dir / "train" / "*.parquet"), *periods["train"],
                                        config.TRAIN_SAMPLE_ROWS, config.SAMPLE_SEED)
    bundle = model.fit_bundle(sample)
    del sample
    valid_glob = str(feature_dir / "validation" / "*.parquet")
    bundle["validation_histogram"] = stream_raw_scores(con, bundle, valid_glob, out / "validation_raw_scores")
    reference = model.reference_sample(con, str(out / "validation_raw_scores" / "*.parquet"),
                                       config.VALIDATION_REFERENCE_ROWS, config.SAMPLE_SEED)
    bundle.update(model.calibrate(reference, valid["host_windows"], valid["days"]))
    bundle.update({
        "reference_scores": reference,
        "reference_method": (f"all {valid['host_windows']:,} validation scores" if len(reference) == valid["host_windows"]
                             else f"hash sample of {len(reference):,} of {valid['host_windows']:,} validation scores"),
        "large_deviation": config.LARGE_DEVIATION, "hist_bins": model.HIST_BINS,
        "forest_settings": dict(config.FOREST_SETTINGS), "raw_columns": dict(features.RAW_COLUMNS),
        "window_minutes": config.WINDOW_MINUTES,
        "periods": {name: (f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}") for name, (a, b) in periods.items()},
        "train_info": train, "validation_info": valid, "development_input_files": files,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), **{name: version(name) for name in PACKAGES}},
    })
    save_bundle(bundle)
    finish(con, bundle, out, "development (validation period)", valid_glob, [train, valid], files, started)


def save_bundle(bundle):
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    settings = {key: bundle[key] for key in ["model_features", "feature_notes", "thresholds", "reference_band_counts",
                                             "reference_method", "forest_settings", "periods", "window_minutes",
                                             "train_sample_rows", "train_sample_days", "raw_columns",
                                             "large_deviation", "created_utc", "versions"]}
    (config.MODEL_PATH.parent / "model_settings.json").write_text(json.dumps(settings, indent=2))


def load_bundle():
    if not config.MODEL_PATH.exists():
        raise SystemExit(f"No saved model at {config.MODEL_PATH}; run `python run.py develop` first")
    return joblib.load(config.MODEL_PATH)


def test():
    started = time.time()
    bundle = load_bundle()
    con = features.connect()
    files = features.check_input(con, config.INPUT_GLOB)
    out = prepare_output(config.WORK_DIR / "test")
    start, end = (features.parse_day(day) for day in bundle["periods"]["test"])  # frozen dates, not config.py
    info = features.build_period_features(con, config.INPUT_GLOB, "test", start, end, out / "features")
    finish(con, bundle, out, "final test (held-out period)", info["features_dir"] + "/*.parquet", [info], files, started)


def score(input_glob, out_dir, start_text=None, end_text=None):
    started = time.time()
    bundle = load_bundle()
    con = features.connect()
    files = features.check_input(con, input_glob)
    out = prepare_output(out_dir)
    data_start, data_end = features.input_day_range(con, input_glob)
    start = features.parse_day(start_text) if start_text else data_start
    end = features.parse_day(end_text) if end_text else data_end
    info = features.build_period_features(con, input_glob, "new", start, end, out / "features",
                                          bounded_end=end_text is not None)
    if start < features.parse_day(bundle["periods"]["validation"][1]):
        info["note"] = "This range overlaps the development periods; it is not later, independent data."
    finish(con, bundle, out, "new data", info["features_dir"] + "/*.parquet", [info], files, started)


def finish(con, bundle, out, label, features_glob, period_infos, files, started):
    histogram = score_period(con, bundle, features_glob, out, config.SCORE_BATCH_ROWS)
    report.write_report(con, bundle, out, label, period_infos, files, histogram, started)
    print(f"{label}: report at {out / 'report.html'}")


def main():
    parser = argparse.ArgumentParser(description="Small NetFlow IsolationForest prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("develop")
    commands.add_parser("test")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--input", required=True, help="Parquet glob, quoted")
    score_parser.add_argument("--out", required=True, help="output folder for this analysis")
    score_parser.add_argument("--start", help="first UTC day to score (default: first day in the input)")
    score_parser.add_argument("--end", help="day after the last UTC day (default: unbounded, data's last day)")
    args = parser.parse_args()
    if args.command == "develop":
        develop()
    elif args.command == "test":
        test()
    else:
        score(args.input, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
