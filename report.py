"""Stage 4: candidate tables, daily summary, six plots, summary.json and a small static report.html.

All global sorting and top-N extraction happens in DuckDB over the scored Parquet; Python only receives small
summaries. The same code serves development, final test and new-data scoring.
"""

import html
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import model  # noqa: E402
from features import FEATURES, sql_text  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
BAND_COLORS = {"Critical": "#b2182b", "High": "#ef8a62", "Medium": "#999999"}
WEEKLY_AFTER_DAYS = 60  # timeline and heatmap switch to weekly columns above this calendar span
CANDIDATE_ORDER = "raw_score DESC, max_abs_deviation DESC NULLS LAST, src_ip, window_start"


def write_tables(con, out):
    """candidates/candidates.parquet (complete set), top_candidates.csv, daily_summary.csv."""
    scores = f"read_parquet({sql_text(out / 'scores' / '*.parquet')})"
    (out / "candidates").mkdir(exist_ok=True)
    candidates = f"""SELECT * EXCLUDE (selected),
        CASE WHEN deviation_only THEN 'deviation_only: large deviation beyond training range'
             ELSE 'model band ' || band END AS candidate_reason
        FROM {scores} WHERE selected ORDER BY {CANDIDATE_ORDER}"""
    con.execute(f"COPY ({candidates}) TO {sql_text(out / 'candidates' / 'candidates.parquet')} (FORMAT parquet)")
    candidates_file = f"read_parquet({sql_text(out / 'candidates' / 'candidates.parquet')})"
    con.execute(f"COPY (SELECT * FROM {candidates_file} ORDER BY {CANDIDATE_ORDER} LIMIT {config.TOP_CSV_ROWS}) "
                f"TO {sql_text(out / 'top_candidates.csv')} (HEADER)")
    daily = con.execute(f"""
        SELECT CAST(window_start AS DATE) AS day, count(*) AS host_windows, sum(flows) AS observed_flows,
            count(DISTINCT src_ip) AS active_hosts,
            count(*) FILTER (WHERE band = 'Critical') AS critical, count(*) FILTER (WHERE band = 'High') AS high,
            count(*) FILTER (WHERE band = 'Medium') AS medium,
            count(*) FILTER (WHERE band = '{model.BELOW}') AS below_threshold,
            1000.0 * count(*) FILTER (WHERE band IN ('Critical', 'High')) / count(*) AS high_critical_per_1000,
            count(*) FILTER (WHERE selected) AS candidates,
            min(raw_score) AS raw_min, median(raw_score) AS raw_median,
            quantile_cont(raw_score, 0.99) AS raw_p99, max(raw_score) AS raw_max
        FROM {scores} GROUP BY 1 ORDER BY 1""").fetchdf()
    daily.to_csv(out / "daily_summary.csv", index=False)
    return daily, candidates_file, scores


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def use_weeks(first_day, last_day):
    """Weekly display when the calendar span (not the number of active days) exceeds WEEKLY_AFTER_DAYS."""
    return (pd.Timestamp(last_day) - pd.Timestamp(first_day)).days + 1 > WEEKLY_AFTER_DAYS


def monday_of(days):
    """Monday of each day's week, matching DuckDB date_trunc('week') in UTC."""
    return days - pd.to_timedelta(days.weekday, unit="D")


def date_ticks(axis, span_days):
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, span_days // 10)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


def timeline_table(daily):
    """Daily rows, or Monday-start weekly totals for long spans. The rate uses totals, never averaged daily rates."""
    days = pd.DatetimeIndex(pd.to_datetime(daily["day"]))
    weekly = use_weeks(days.min(), days.max())
    data = daily[["observed_flows", "host_windows", "critical", "high"]].set_axis(monday_of(days) if weekly else days)
    data = data.groupby(level=0).sum()
    data["flagged"] = data["critical"] + data["high"]
    data["flagged_per_1000"] = 1000 * data["flagged"] / data["host_windows"]
    return data, weekly


def plot_timeline(daily, path):
    data, weekly = timeline_table(daily)
    days = pd.to_datetime(daily["day"])
    unit = "week" if weekly else "day"
    flagged = data["flagged"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    top.bar(data.index, data["observed_flows"], color="#9ecae1", width=0.8 * (7 if weekly else 1),
            align="edge" if weekly else "center", label=f"observed flows per {unit}")
    top.set_ylabel(f"observed flow records per {unit}")
    twin = top.twinx()
    twin.plot(data.index, flagged, color=BAND_COLORS["Critical"], marker="o", label=f"High+Critical windows per {unit}")
    twin.set_ylabel(f"High+Critical host-windows per {unit}")
    bottom.plot(data.index, data["flagged_per_1000"], color="black", marker="o")
    bottom.set_ylabel(f"High+Critical per 1,000\nactive host-windows ({'weekly' if weekly else 'daily'} totals)")
    for start, end in config.PENTEST_INTERVALS:
        for axis in (top, bottom):
            axis.axvspan(np.datetime64(start), np.datetime64(end), color="orange", alpha=0.15)
    title = "Weekly (Monday-start) volume and review load (UTC)" if weekly else "Daily volume and review load (UTC)"
    top.set_title(title + (" - shaded: broad pentest context, not labels" if config.PENTEST_INTERVALS else ""))
    top.legend(loc="upper left")
    twin.legend(loc="upper right")
    date_ticks(bottom, (days.max() - days.min()).days + 1)
    fig.autofmt_xdate()
    save(fig, path)
    return weekly


def plot_distributions(bundle, histogram, label, path):
    bins = bundle["hist_bins"]
    width = np.diff(bins)
    fig, axis = plt.subplots(figsize=(10, 4))
    for counts, name in [(bundle["validation_histogram"], "validation (all rows)"), (histogram, label)]:
        axis.stairs(counts / counts.sum() / width, bins, label=f"{name}, n={counts.sum():,}")
    for band, value in bundle["thresholds"].items():
        axis.axvline(value, color=BAND_COLORS[band], linestyle="--", label=f"{band} >= {value:.4f}")
    used = np.flatnonzero(bundle["validation_histogram"] + histogram)
    axis.set_xlim(bins[used[0]], bins[used[-1] + 1])
    axis.set_yscale("log")
    axis.set_xlabel("raw score (larger = more unusual)")
    axis.set_ylabel("density (normalized per period)")
    axis.set_title("Raw-score distributions, identical bins")
    axis.legend(fontsize=8)
    save(fig, path)


def plot_workload(bundle, path):
    rows = bundle["workload"]
    labels = [f"q{100 * row['quantile']:g}" for row in rows]
    fig, axis = plt.subplots(figsize=(8, 3.8))
    bars = axis.bar(labels, [row["expected_per_day"] for row in rows], color="#6baed6")
    for bar, row in zip(bars, rows):
        axis.annotate(f"{row['per_1000_host_windows']:.1f}/1000", (bar.get_x() + bar.get_width() / 2,
                      bar.get_height()), ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("validation raw-score threshold")
    axis.set_ylabel("host-windows per day at or above")
    axis.set_title("Review workload by threshold on validation (workload, not accuracy)")
    save(fig, path)


def plot_heatmap(con, scores, path):
    """Top 20 hosts by candidate count, ties by max raw score then src_ip. Cells: max raw score per host and day,
    or per Monday-start week for long spans. Blank = no active window of that host in that day/week."""
    first, last = con.execute(f"SELECT CAST(min(window_start) AS DATE), CAST(max(window_start) AS DATE) "
                              f"FROM {scores}").fetchone()
    weekly = use_weeks(first, last)
    bucket = "CAST(date_trunc('week', window_start) AS DATE)" if weekly else "CAST(window_start AS DATE)"
    hosts = [row[0] for row in con.execute(f"""
        SELECT src_ip FROM {scores} GROUP BY src_ip
        ORDER BY count(*) FILTER (WHERE selected) DESC, max(raw_score) DESC, src_ip LIMIT 20""").fetchall()]
    cells = con.execute(f"SELECT src_ip, {bucket} AS period, max(raw_score) AS max_raw FROM {scores} "
                        f"WHERE list_contains(?, src_ip) GROUP BY 1, 2", [hosts]).fetchdf()
    cells["period"] = pd.to_datetime(cells["period"])
    first_column = monday_of(pd.DatetimeIndex([pd.Timestamp(first)]))[0] if weekly else pd.Timestamp(first)
    columns = pd.date_range(first_column, pd.Timestamp(last), freq="7D" if weekly else "D")
    grid = cells.pivot(index="src_ip", columns="period", values="max_raw").reindex(index=hosts, columns=columns)
    fig, axis = plt.subplots(figsize=(13, 6))
    image = axis.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), aspect="auto", cmap="viridis")
    step = int(np.ceil(len(columns) / 12))
    axis.set_yticks(range(len(hosts)), hosts, fontsize=7)
    axis.set_xticks(range(0, len(columns), step), [f"{day:%Y-%m-%d}" for day in columns[::step]],
                    rotation=45, ha="right", fontsize=8)
    unit = "week (Monday start, UTC)" if weekly else "UTC day"
    fig.colorbar(image, label=f"{'weekly' if weekly else 'daily'} max raw score")
    axis.set_title(f"Top 20 hosts by candidate count: max raw score per host and {unit}\n"
                   f"(blank = no active window in that {'week' if weekly else 'day'})")
    axis.grid(False)
    save(fig, path)
    return weekly


def plot_detail(con, bundle, scores, top, path):
    fig, axis = plt.subplots(figsize=(10, 3.8))
    if top is None:
        axis.text(0.5, 0.5, "No candidates in this period", ha="center", va="center", transform=axis.transAxes)
    else:
        rows = con.execute(f"""SELECT window_start, raw_score FROM {scores} WHERE src_ip = ?
            AND window_start BETWEEN ?::TIMESTAMPTZ - INTERVAL 24 HOUR AND ?::TIMESTAMPTZ + INTERVAL 24 HOUR
            ORDER BY window_start""", [top["src_ip"], top["window_start"].isoformat(),
                                       top["window_start"].isoformat()]).fetchdf()
        axis.plot(rows["window_start"], rows["raw_score"], marker=".", linestyle="-", color="#3182bd")
        axis.axvline(top["window_start"], color="black", alpha=0.4)
        for band, value in bundle["thresholds"].items():
            axis.axhline(value, color=BAND_COLORS[band], linestyle="--", label=band)
        axis.set_title(f"+/-24 h around the highest-scoring candidate: {top['src_ip']} at "
                       f"{top['window_start']:%Y-%m-%d %H:%M} UTC\n(active windows of this host within the scored period only)")
        axis.set_xlim(top["window_start"] - np.timedelta64(24, "h"), top["window_start"] + np.timedelta64(24, "h"))
        axis.set_ylabel("raw score")
        axis.legend(fontsize=8)
        fig.autofmt_xdate()
    save(fig, path)


def plot_context(con, bundle, candidates_file, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = bundle["model_features"]
    image = left.imshow(bundle["spearman"], vmin=-1, vmax=1, cmap="RdBu_r")
    left.set_xticks(range(len(names)), names, rotation=60, ha="right")
    left.set_yticks(range(len(names)), names)
    left.set_title("Spearman correlation, training sample (redundancy)")
    left.grid(False)
    fig.colorbar(image, ax=left, shrink=0.8)
    top = con.execute(f"SELECT * FROM {candidates_file} ORDER BY {CANDIDATE_ORDER} LIMIT 15").to_arrow_table()
    if top.num_rows == 0:
        right.text(0.5, 0.5, "No candidates in this period", ha="center", va="center")
    else:
        deviation = model.robust_deviations(model.feature_matrix(top), bundle["training_stats"])[0]
        image = right.imshow(np.ma.masked_invalid(np.clip(deviation, -10, 10)), vmin=-10, vmax=10,
                             cmap="PuOr_r", aspect="auto")
        labels = [f"{ip} {ts:%m-%d %H:%M}" for ip, ts in zip(top.column("src_ip").to_pylist(),
                                                              top.column("window_start").to_pylist())]
        right.set_yticks(range(len(labels)), labels, fontsize=7)
        right.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right")
        fig.colorbar(image, ax=right, shrink=0.8, label="robust deviation (clipped at +/-10)")
    right.set_title("Top candidates: deviation from training median (context, not importance)")
    right.grid(False)
    save(fig, path)


def table_html(header, rows):
    head = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def write_report(con, bundle, out, label, period_infos, files, histogram, started):
    out = Path(out)
    daily, candidates_file, scores = write_tables(con, out)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    top_rows = con.execute(f"SELECT src_ip, window_start, band, raw_score, reference_percentile, priority_candidate, "
                           f"candidate_reason, deviation_context FROM {candidates_file} ORDER BY {CANDIDATE_ORDER} "
                           f"LIMIT {config.REPORT_TOP_ROWS}").fetchdf().itertuples(index=False, name=None)
    top_rows = list(top_rows)
    top = {"src_ip": top_rows[0][0], "window_start": top_rows[0][1]} if top_rows else None
    plot_timeline(daily, plots / "1_timeline.png")
    plot_distributions(bundle, histogram, label, plots / "2_score_distribution.png")
    plot_workload(bundle, plots / "3_threshold_workload.png")
    plot_heatmap(con, scores, plots / "4_host_day_heatmap.png")
    plot_detail(con, bundle, scores, top, plots / "5_top_candidate_detail.png")
    plot_context(con, bundle, candidates_file, plots / "6_feature_context.png")

    stats = con.execute(f"SELECT count(*), min(raw_score), median(raw_score), quantile_cont(raw_score, 0.99), "
                        f"max(raw_score) FROM {scores}").fetchone()
    counts = dict(con.execute(f"SELECT candidate_reason, count(*) FROM {candidates_file} GROUP BY 1").fetchall())
    n_candidates = sum(counts.values())
    notes = list(bundle["feature_notes"]) + [info["note"] for info in period_infos if "note" in info]
    if n_candidates > config.TOP_CSV_ROWS:
        notes.append(f"{n_candidates:,} candidates; top_candidates.csv holds the top {config.TOP_CSV_ROWS}, "
                     f"candidates/ holds all.")
    summary = {
        "analysis": label, "periods": period_infos, "input_files": files,
        "scored_host_windows": stats[0], "candidates": n_candidates, "candidates_by_reason": counts,
        "raw_score_exact": dict(zip(["min", "median", "p99", "max"], stats[1:])),
        "model": {key: bundle[key] for key in ["model_features", "train_sample_rows", "train_sample_days",
                                               "reference_method", "review_quantiles", "thresholds",
                                               "reference_band_counts"]},
        "model_periods_frozen": bundle["periods"], "model_created_utc": bundle["created_utc"],
        "settings": {"window_minutes": bundle["window_minutes"], "forest": bundle["forest_settings"],
                     "large_deviation": bundle["large_deviation"], "score_batch_rows": config.SCORE_BATCH_ROWS,
                     "duckdb_memory_limit": config.DUCKDB_MEMORY_LIMIT, "duckdb_threads": config.DUCKDB_THREADS},
        "notes": notes, "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.html").write_text(report_html(bundle, label, period_infos, summary, top_rows), encoding="utf-8")


def report_html(bundle, label, period_infos, summary, top_rows):
    periods = table_html(["period", "UTC range [start, end)", "host-windows", "flows", "excluded: ended after period"],
                         [[p["period"], f"{p['start']} .. {p['end']}", f"{p['host_windows']:,}", f"{p['flows']:,}",
                           p["excluded_flows_ending_after_period"]] for p in period_infos])
    bands = table_html(["band", "validation quantile", "raw score >=", "validation reference count"],
                       [[b, bundle["review_quantiles"][b], f"{bundle['thresholds'][b]:.6f}",
                         bundle["reference_band_counts"][b]] for b in model.BANDS])
    candidates = table_html(["source", "window start (UTC)", "band", "raw score", "validation pct", "priority",
                             "reason", "largest deviations (original units)"],
                            [[r[0], f"{r[1]:%Y-%m-%d %H:%M}", r[2], f"{r[3]:.5f}", f"{r[4]:.2f}", r[5], r[6],
                              r[7] or ""] for r in top_rows]) if top_rows else "<p>No candidates in this period.</p>"
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in summary["notes"]) or "<li>none</li>"
    image = '<img src="plots/{}" alt="{}">'.format
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NetFlow anomaly candidates</title>
<style>body{{font-family:sans-serif;max-width:1150px;margin:auto;padding:0 16px}}img{{max-width:100%}}
table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:left}}</style>
</head><body><h1>NetFlow anomaly candidates: {html.escape(label)}</h1>
<h2>1. Run and score explanation</h2>{periods}
<p>{summary['scored_host_windows']:,} host-windows scored ({bundle['window_minutes']}-minute UTC windows);
{summary['candidates']:,} candidates. Model trained on {bundle['train_sample_rows']:,} sampled training host-windows
from {bundle['train_sample_days']} days; features: {html.escape(', '.join(bundle['model_features']))}.</p>
<p><b>raw score</b> = -IsolationForest.score_samples: larger means more unusual; it ranks rows and is not a
probability. <b>validation pct</b> is the position within the frozen validation reference
({html.escape(bundle['reference_method'])}), not a chance of attack. Bands are review priorities from validation
quantiles; "Below threshold" does not mean safe. Candidates = all High/Critical windows plus
<i>deviation_only</i> windows (|deviation| &gt;= {bundle['large_deviation']} and outside the training range).</p>
<h2>2. Overview and score distribution</h2>{image('1_timeline.png', 'timeline')}
{image('2_score_distribution.png', 'score distribution')}
<h2>3. Threshold workload (validation)</h2>{bands}{image('3_threshold_workload.png', 'workload')}
<h2>4. Top {config.REPORT_TOP_ROWS} candidates</h2>{candidates}
<h2>5. Hosts, detail and feature context</h2>{image('4_host_day_heatmap.png', 'heatmap')}
{image('5_top_candidate_detail.png', 'detail')}{image('6_feature_context.png', 'feature context')}
<h2>6. Outputs and limitations</h2><ul>
<li><a href="candidates/candidates.parquet">candidates/candidates.parquet</a> (all candidates),
<a href="top_candidates.csv">top_candidates.csv</a>, <a href="daily_summary.csv">daily_summary.csv</a>,
<a href="summary.json">summary.json</a>, scores/ (every scored host-window)</li>
<li>These are anomalous candidates to check against independent records; no detection rate is claimed.
Low-volume or normal-looking activity can be missed. Deviations are descriptive context, not model attribution.</li>
<li>Pentest dates, if shown, are broad context only and were not used to choose features, dates or thresholds.</li>
</ul><h3>Notes</h3><ul>{notes}</ul></body></html>"""
