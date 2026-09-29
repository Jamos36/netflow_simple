"""A few substantive checks on tiny temporary Parquet fixtures (not a data generator for users)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import features  # noqa: E402
import model  # noqa: E402
import report  # noqa: E402
import run  # noqa: E402

DAY0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def write_flows(path, rows):
    """rows: list of (src, dst, port, protocol, bytes, packets, start, end)."""
    names = list(features.RAW_COLUMNS)
    columns = list(zip(*rows))
    types = [pa.string(), pa.string(), pa.int64(), pa.int64(), pa.int64(), pa.int64(),
             pa.timestamp("us", tz="UTC"), pa.timestamp("us", tz="UTC")]
    pq.write_table(pa.table({n: pa.array(c, t) for n, c, t in zip(names, columns, types)}), path)


def random_flows(rng, day, count):
    rows = []
    for _ in range(count):
        start = day + timedelta(seconds=float(rng.uniform(0, 86_000)))
        packets = int(rng.integers(1, 50))
        rows.append((f"10.0.0.{rng.integers(1, 20)}", f"10.1.0.{rng.integers(1, 40)}", int(rng.choice([53, 80, 443])),
                     int(rng.choice([6, 17])), packets * int(rng.integers(60, 1400)), packets,
                     start, start + timedelta(seconds=float(rng.exponential(5)))))
    return rows


def use_settings(monkeypatch, tmp_path, input_glob):
    monkeypatch.setattr(config, "INPUT_GLOB", input_glob)
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "work" / "model.joblib")
    monkeypatch.setattr(config, "DUCKDB_SPILL_DIR", tmp_path / "work" / "spill")
    monkeypatch.setattr(config, "TRAIN_PERIOD", ("2026-01-01", "2026-01-03"))
    monkeypatch.setattr(config, "VALIDATION_PERIOD", ("2026-01-03", "2026-01-04"))
    monkeypatch.setattr(config, "TEST_PERIOD", ("2026-01-04", "2026-01-05"))


def develop_on(monkeypatch, tmp_path, add_test_rows):
    rng = np.random.default_rng(7)
    data = tmp_path / "data"
    data.mkdir(parents=True)
    for day in range(3):
        write_flows(data / f"day{day}.parquet", random_flows(rng, DAY0 + timedelta(days=day), 1500))
    # a training-period flow that only completes in validation: must be excluded from training
    write_flows(data / "late.parquet", [("10.0.0.99", "10.1.0.1", 22, 6, 10**9, 10**6,
                                         DAY0 + timedelta(hours=47), DAY0 + timedelta(hours=49))])
    if add_test_rows:
        extreme = [("10.0.0.1", f"10.9.{i}.1", i, 1, 10**8, 10, DAY0 + timedelta(days=3, minutes=i),
                    DAY0 + timedelta(days=3, minutes=i + 1)) for i in range(200)]
        write_flows(data / "test.parquet", random_flows(rng, DAY0 + timedelta(days=3), 1500) + extreme)
    use_settings(monkeypatch, tmp_path, str(data / "*.parquet"))
    run.develop()
    return joblib.load(config.MODEL_PATH)


def test_host_window_split_across_files_is_aggregated_once(tmp_path, monkeypatch):
    t = DAY0 + timedelta(minutes=3)
    write_flows(tmp_path / "a.parquet", [("h1", "d1", 80, 6, 100, 2, t, t), ("h1", "d2", 80, 6, 100, 2, t, t)])
    write_flows(tmp_path / "b.parquet", [("h1", "d2", 443, 17, 300, 1, t, t), ("h1", "d3", 80, 6, 100, 5, t, t)])
    use_settings(monkeypatch, tmp_path, str(tmp_path / "*.parquet"))
    con = features.connect()
    info = features.build_period_features(con, config.INPUT_GLOB, "train", DAY0, DAY0 + timedelta(days=1), tmp_path)
    row = con.execute(f"SELECT * FROM read_parquet('{tmp_path / 'train' / '*.parquet'}')").fetchdf().iloc[0]
    assert info["host_windows"] == 1
    assert (row.flows, row.uniq_dst_ip, row.uniq_dst_port, row.bytes_total) == (4, 3, 2, 600)
    assert row.bytes_per_packet == pytest.approx(600 / 10)
    assert row.udp_share == pytest.approx(0.25)


def test_future_rows_do_not_change_training_or_thresholds(tmp_path, monkeypatch):
    without = develop_on(monkeypatch, tmp_path / "a", add_test_rows=False)
    with_test = develop_on(monkeypatch, tmp_path / "b", add_test_rows=True)
    assert without["train_sample_rows"] == with_test["train_sample_rows"]
    assert without["thresholds"] == with_test["thresholds"]
    assert np.array_equal(without["reference_scores"], with_test["reference_scores"])
    assert without["train_info"]["excluded_flows_ending_after_period"] == 1


def test_batch_size_does_not_change_scores_and_empty_candidates_still_report(tmp_path, monkeypatch):
    bundle = develop_on(monkeypatch, tmp_path, add_test_rows=False)
    con = features.connect()
    glob = str(config.WORK_DIR / "development" / "features" / "validation" / "*.parquet")
    for size in (7, 100_000):
        run.score_period(con, bundle, glob, tmp_path / f"batch{size}", size)
    same = con.execute(f"""SELECT count(*), max(abs(a.raw_score - b.raw_score))
        FROM read_parquet('{tmp_path / 'batch7' / 'scores' / '*.parquet'}') a
        JOIN read_parquet('{tmp_path / 'batch100000' / 'scores' / '*.parquet'}') b USING (src_ip, window_start)""")
    rows, max_difference = same.fetchone()
    assert rows == bundle["validation_info"]["host_windows"] and max_difference == 0

    quiet = dict(bundle, thresholds={band: np.inf for band in model.BANDS}, large_deviation=np.inf)
    out = tmp_path / "quiet"
    histogram = run.score_period(con, quiet, glob, out, 1000)
    report.write_report(con, quiet, out, "empty check", [bundle["validation_info"]], [], histogram, 0)
    candidates = pq.read_table(out / "candidates" / "candidates.parquet")
    assert candidates.num_rows == 0 and "raw_score" in candidates.column_names
    assert (out / "top_candidates.csv").read_text().startswith("src_ip,")
    assert "No candidates in this period" in (out / "report.html").read_text(encoding="utf-8")


def test_tied_thresholds_and_percentile_saturation():
    reference = np.sort(np.array([0.5] * 995 + [0.6] * 5))
    calibration = model.calibrate(reference, validation_rows=1000, validation_days=1)
    assert calibration["thresholds"]["Critical"] == calibration["thresholds"]["High"] == 0.6
    assert calibration["reference_band_counts"]["Critical"] == 5 and calibration["reference_band_counts"]["High"] == 0
    assert list(model.assign_bands(np.array([0.6, 0.55, 0.4]), calibration["thresholds"])) == \
        ["Critical", "Medium", model.BELOW]
    assert 100 * np.searchsorted(reference, 0.9, side="right") / len(reference) == 100


def test_zero_spread_deviation_reference():
    X = np.column_stack([np.zeros(10), [0.0] * 6 + [1, 2, 3, 4]])
    stats = model.training_stats(X)
    q1, q3 = np.percentile(X[:, 1], [25, 75])
    assert stats["scale"][0] == 0 and stats["scale"][1] == pytest.approx((q3 - q1) / 1.349)
    deviation, constant_departure, beyond = model.robust_deviations(np.array([[5.0, 9.0], [0.0, np.nan]]), stats)
    assert np.isnan(deviation[0, 0]) and constant_departure[0, 0] and beyond[0, 0]
    assert not constant_departure[1, 0] and np.isnan(deviation[1, 1]) and not beyond[1, 1]


def test_missing_required_column_fails_clearly(tmp_path):
    pq.write_table(pa.table({"src_id_addr": ["h1"]}), tmp_path / "bad.parquet")
    with pytest.raises(SystemExit, match="missing required columns"):
        features.check_input(duckdb.connect(), str(tmp_path / "*.parquet"))
