from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anomaly_poc.pipeline import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train and compare CPU anomaly models on Parquet flow rows.")
    parser.add_argument("--config", default="config.yaml", help="YAML settings file")
    parser.add_argument("--model", choices=["all", "iforest", "pca", "hbos", "gmm"], default="all")
    parser.add_argument("--run-id", help="Optional output folder name")
    args = parser.parse_args()
    result = run(Path(args.config), args.model, args.run_id)
    print(f"Finished. Results: {result}")


if __name__ == "__main__":
    main()
