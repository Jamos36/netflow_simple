"""Small SYNTHETIC smoke experiment; not realistic detection validation."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def main():
    root = Path(__file__).parent
    data = root / "data"
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    # Sparse activity across 28 days, so all split dates and weekly context exist.
    for day in pd.date_range("2026-01-01", periods=28, tz="UTC"):
        for h in range(8):
            host = f"10.0.0.{h + 1}"
            for minute in range(30):
                t = day + pd.Timedelta(hours=9, minutes=minute)
                for j in range(int(rng.integers(2, 6))):
                    duration = float(rng.exponential(3))
                    rows.append(
                        dict(
                            flow_start_time=t + pd.Timedelta(seconds=j * 7),
                            flow_end_time=t + pd.Timedelta(seconds=j * 7 + duration),
                            src_ip=host,
                            dst_ip=f"10.1.0.{rng.integers(1, 6)}",
                            dst_port=int(rng.choice([443, 53, 22])),
                            bytes=float(rng.lognormal(7, 1)),
                            packets=float(rng.integers(2, 40)),
                            duration_seconds=duration,
                            failure=0,
                        )
                    )
    start = pd.Timestamp("2026-01-26 09:10", tz="UTC")
    for minute in range(5):
        for j in range(100):
            t = start + pd.Timedelta(minutes=minute, seconds=j * 0.5)
            rows.append(
                dict(
                    flow_start_time=t,
                    flow_end_time=t + pd.Timedelta(seconds=0.05),
                    src_ip="10.0.0.1",
                    dst_ip=f"192.0.2.{j + 1}",
                    dst_port=10000 + j,
                    bytes=60.0,
                    packets=1.0,
                    duration_seconds=0.05,
                    failure=1,
                )
            )
    frame = pd.DataFrame(rows)
    # Shuffle across files to test aggregation independent of file boundaries.
    frame = frame.sample(frac=1, random_state=42)
    for i, indices in enumerate(np.array_split(np.arange(len(frame)), 4)):
        part = frame.iloc[indices]
        part.to_parquet(data / f"demo_{i}.parquet", index=False)
    pd.DataFrame(
        [
            dict(
                event_id="synthetic_scan",
                host="10.0.0.1",
                start=start,
                end=start + pd.Timedelta(minutes=5),
            )
        ]
    ).to_csv(data / "events.csv", index=False)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cfg["data"]["paths"] = ["data/demo_*.parquet"]
    cfg["columns"]["failure"] = "failure"
    cfg["split"] = dict(
        train_end="2026-01-15",
        validation_end="2026-01-20",
        calibration_end="2026-01-24",
    )
    cfg["labels"] = dict(events_csv="data/events.csv", complete=True, reviews_csv=None)
    cfg["context"] = dict(pentest_start="2026-01-26", pentest_end="2026-01-27")
    cfg["backtests"] = [
        dict(
            name="earlier",
            train_end="2026-01-12",
            validation_end="2026-01-17",
            calibration_end="2026-01-22",
        )
    ]
    (root / "demo.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Created synthetic inputs and demo.yaml")


if __name__ == "__main__":
    main()
