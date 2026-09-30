"""A few substantive checks on tiny temporary Parquet fixtures (not a data generator for users)."""

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import joblib
import matplotlib.image
import pandas as pd
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
    info = features.build_period_features(con, config.INPUT_GLOB, "train", DAY0, DAY0 + timedelta(days=1),
                                          tmp_path, 15)
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
    calibration = model.calibrate(reference, 1000, 1, config.REVIEW_QUANTILES)
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


def test_saved_window_and_quantiles_survive_config_changes(tmp_path, monkeypatch):
    bundle = develop_on(monkeypatch, tmp_path, add_test_rows=True)
    model_bytes = config.MODEL_PATH.read_bytes()
    run.test()
    first = tmp_path / "first_test"
    shutil.copytree(config.WORK_DIR / "test", first)
    monkeypatch.setattr(config, "WINDOW_MINUTES", 60)
    monkeypatch.setattr(config, "REVIEW_QUANTILES", {"Critical": 0.9, "High": 0.8, "Medium": 0.7})
    run.test()
    second = config.WORK_DIR / "test"

    columns = ", ".join(["src_ip", "window_start", "window_end"] + features.FEATURES + ["raw_score", "band", "selected"])
    con = features.connect()
    before, after = (con.execute(f"SELECT {columns} FROM read_parquet('{folder / 'scores' / '*.parquet'}') "
                                 f"ORDER BY src_ip, window_start").fetchdf() for folder in (first, second))
    pd.testing.assert_frame_equal(before, after)
    assert ((after["window_end"] - after["window_start"]) == pd.Timedelta(minutes=15)).all()
    summary = json.loads((second / "summary.json").read_text())
    assert summary["settings"]["window_minutes"] == 15
    assert summary["model"]["review_quantiles"] == bundle["review_quantiles"] == {"Critical": 0.999, "High": 0.995,
                                                                                    "Medium": 0.99}
    assert summary["model"]["reference_band_counts"] == bundle["reference_band_counts"]
    page = (second / "report.html").read_text(encoding="utf-8")
    assert "<td>0.995</td>" in page and "<td>0.8</td>" not in page and "15-minute" in page
    assert config.MODEL_PATH.read_bytes() == model_bytes

    joblib.dump({key: value for key, value in bundle.items() if key != "review_quantiles"}, config.MODEL_PATH)
    with pytest.raises(SystemExit, match="rerun `python run.py develop`"):
        run.load_bundle()


@pytest.mark.parametrize("days, weekly", [(5, False), (365, True)])
def test_plots_stay_bounded_and_label_weekly_spans(tmp_path, days, weekly):
    """Small summary fixtures only: no raw NetFlow is generated for the chart layout check."""
    rng = np.random.default_rng(1)
    calendar = pd.date_range("2026-01-01", periods=days, freq="D")
    active = calendar[rng.random(days) > 0.2]  # inactive days must stay blank, not become zero scores
    daily = pd.DataFrame({"day": active, "observed_flows": 1000, "host_windows": rng.integers(50, 150, len(active)),
                          "critical": rng.integers(0, 3, len(active)), "high": rng.integers(0, 5, len(active))})
    starts = [day.tz_localize("UTC") + pd.Timedelta(hours=h) for day in active for h in range(25)]
    pq.write_table(pa.table({
        "src_ip": [f"10.0.0.{h}" for _ in active for h in range(25)],
        "window_start": pa.array(starts, pa.timestamp("us", tz="UTC")),
        "raw_score": rng.uniform(0.4, 0.7, len(starts)), "selected": rng.random(len(starts)) < 0.05,
    }), tmp_path / "scores.parquet")
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")

    assert report.plot_timeline(daily, tmp_path / "timeline.png") is weekly
    assert report.plot_heatmap(con, f"read_parquet('{tmp_path / 'scores.parquet'}')", tmp_path / "heatmap.png") is weekly
    for name in ("timeline.png", "heatmap.png"):
        assert matplotlib.image.imread(tmp_path / name).shape[1] <= 1500
    table, _ = report.timeline_table(daily)
    first_period = daily[daily["day"] < table.index[1]]
    expected = 1000 * (first_period["critical"] + first_period["high"]).sum() / first_period["host_windows"].sum()
    assert table["flagged_per_1000"].iloc[0] == pytest.approx(expected)
    assert (table.index.weekday == 0).all() if weekly else len(table) == len(active)
