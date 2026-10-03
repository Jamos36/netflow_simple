"""Build, compare, backtest, or score with frozen models."""

import argparse
import copy
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from research.features import build
from research.pipeline import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "run", "backtest", "score"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--models", default=None, help="comma-separated names or all")
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--model-file")
    p.add_argument("--input")
    p.add_argument("--output")
    args = p.parse_args()
    if args.command == "score":
        bundle = joblib.load(args.model_file)
        writer = None
        for batch in pq.ParquetFile(args.input).iter_batches(batch_size=20000):
            frame = batch.to_pandas()
            score = bundle["model"].score(bundle["transform"].apply(frame))
            frame["score"] = score
            frame["alert"] = score > bundle["threshold"]
            frame["percentile"] = np.searchsorted(
                bundle["calibration_scores"], score, side="right"
            ) / len(bundle["calibration_scores"])
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema)
            writer.write_table(table)
        writer.close()
        return
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base = config_path.parent
    cfg["data"]["paths"] = [
        str(base / path) if not Path(path).is_absolute() else path
        for path in cfg["data"]["paths"]
    ]
    for key in ["events_csv", "reviews_csv"]:
        path = cfg["labels"].get(key)
        if path:
            cfg["labels"][key] = (
                str(base / path) if not Path(path).is_absolute() else path
            )
    if args.models:
        cfg["models"]["names"] = args.models.split(",")
    out = Path(args.output or cfg["output"])
    if not out.is_absolute():
        out = base / out
    feature_dir = out / "features"
    if args.command in ["build", "run"]:
        print("Building host-window features...", flush=True)
        paths = build(cfg, feature_dir)
        (feature_dir / "input_manifest.json").write_text(
            json.dumps(
                [
                    dict(
                        path=p,
                        bytes=Path(p).stat().st_size,
                        modified_ns=Path(p).stat().st_mtime_ns,
                    )
                    for p in paths
                ],
                indent=2,
            )
        )
    if args.command == "run":
        run(cfg, feature_dir, out / "analysis", args.ablations)
    elif args.command == "backtest":
        if not cfg["backtests"]:
            raise ValueError("Add chronological backtests in config.yaml")
        rows = []
        for fold in cfg["backtests"]:
            local = copy.deepcopy(cfg)
            local["split"] = {
                k: fold[k] for k in ["train_end", "validation_end", "calibration_end"]
            }
            rows.extend(
                [
                    dict(fold=fold["name"], **r)
                    for r in run(local, feature_dir, out / "backtests" / fold["name"])
                ]
            )
        pd.DataFrame(rows).to_csv(out / "backtest_comparison.csv", index=False)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
