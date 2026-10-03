from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from research.features import build
from research.models import Detector
from research.evaluation import incidents, event_metrics

ROOT = Path(__file__).parents[1]


def config(paths):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["data"]["paths"] = list(map(str, paths))
    cfg["features"]["min_history_days"] = 1
    return cfg


def flows():
    rows = []
    for day in range(1, 5):
        for j in range(4):
            t = pd.Timestamp(f"2026-01-{day:02d} 09:00", tz="UTC") + pd.Timedelta(
                seconds=j * 10
            )
            rows.append(
                dict(
                    flow_start_time=t,
                    flow_end_time=t,
                    src_ip="a",
                    dst_ip="b",
                    dst_port=443,
                    bytes=100.0 * day,
                    packets=5.0,
                    duration_seconds=1.0,
                )
            )
    return pd.DataFrame(rows)


def test_files_and_causal_history(tmp_path):
    f = flows()
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    one = tmp_path / "one.parquet"
    f.to_parquet(one, index=False)
    f.iloc[::2].to_parquet(a, index=False)
    f.iloc[1::2].to_parquet(b, index=False)
    build(config([one]), tmp_path / "one")
    build(config([a, b]), tmp_path / "many")
    first = pd.read_parquet(tmp_path / "one/windows.parquet")
    pd.testing.assert_frame_equal(
        first, pd.read_parquet(tmp_path / "many/windows.parquet")
    )
    assert first.new_ip_rate.tolist() == [1, 0, 0, 0]
    assert pd.isna(first.bytes_robust_z.iloc[0])
    f.loc[f.flow_start_time.dt.day == 4, "bytes"] = 1e10
    f.to_parquet(one, index=False)
    build(config([one]), tmp_path / "future")
    pd.testing.assert_frame_equal(
        first.iloc[:3], pd.read_parquet(tmp_path / "future/windows.parquet").iloc[:3]
    )


def test_hbos_outside_range_and_pca_distance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    h = Detector("hbos", {"bins": 10}).fit(x)
    assert h.score(np.array([[100, 0, 0]]))[0] > h.score(np.array([[10, 0, 0]]))[0]
    p = Detector("pca", {"components_pca": 1}).fit(x)
    assert np.isfinite(p.score(x)).all()
    assert p.score(np.array([[100, 100, 100]]))[0] > np.median(p.score(x))


def test_incidents_and_incomplete_labels():
    time = pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        dict(
            host=["a"] * 4,
            time=time,
            alert=[True, True, False, True],
            score=[4.0, 3.0, 0.0, 2.0],
        )
    )
    assert len(incidents(frame, 2)) == 1
    events = pd.DataFrame([dict(event_id="e", host="a", start=time[0], end=time[2])])
    result, detail, _ = event_metrics(frame, events, 1, False)
    assert result["event_recall"] == 1
    assert "precision" not in result


def test_csv_and_parquet_equivalent(tmp_path):
    frame = flows()
    a, b = tmp_path / "input.csv", tmp_path / "input.parquet"
    frame.to_csv(a, index=False)
    frame.to_parquet(b, index=False)
    build(config([a]), tmp_path / "csv")
    build(config([b]), tmp_path / "parquet")
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "csv/windows.parquet"),
        pd.read_parquet(tmp_path / "parquet/windows.parquet"),
    )


def test_complete_labels_enable_metrics():
    time = pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        dict(
            host=["a"] * 4,
            time=time,
            alert=[True, True, False, False],
            score=[4.0, 3.0, 0.0, 1.0],
        )
    )
    events = pd.DataFrame([dict(event_id="e", host="a", start=time[0], end=time[2])])
    result, _, _ = event_metrics(frame, events, 1, True)
    assert result["precision"] == result["recall"] == result["f1"] == 1
