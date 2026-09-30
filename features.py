"""Stage 1: raw NetFlow Parquet -> one feature row per source host per active 15-minute UTC window.

One DuckDB query runs per UTC day over ALL matching input files, filtered by date. A file is not a time
partition: flows of one host-window may sit in several files, so they are aggregated together, never per file.
Assumptions: host identifiers are stable and the export holds no duplicate flow records.
Timestamps: zone-aware values are converted to UTC; zone-less values are read as UTC (session time zone).
"""

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

import config

# Fixed input mapping: raw column -> internal name. Only these eight raw columns are read.
RAW_COLUMNS = {
    "src_id_addr": "src_ip",
    "dist_id_addr": "dst_ip",
    "dist_port": "dst_port",
    "ip_protocol_id": "protocol",
    "num_bytes": "bytes",
    "num_packets": "packets",
    "flow_start_time": "flow_start",
    "flow_end_time": "flow_end",
}

# Ordered model features. Definitions are the SELECT below; the first seven get log1p before modelling.
FEATURES = [
    "flows",             # count(*)
    "bytes_total",       # sum(bytes)
    "packets_total",     # sum(packets)
    "bytes_per_packet",  # sum(bytes) / NULLIF(sum(packets), 0)
    "uniq_dst_ip",       # count(DISTINCT dst_ip)
    "uniq_dst_port",     # count(DISTINCT dst_port)
    "mean_duration_s",   # avg(flow_end - flow_start) in seconds
    "tcp_share",         # share of flows with protocol 6
    "udp_share",         # share of flows with protocol 17
    "icmp_share",        # share of flows with protocol 1
]
LOG_FEATURES = FEATURES[:7]
KEY_COLUMNS = ["src_ip", "window_start", "window_end", "period"]

DAY_FEATURE_SQL = """
WITH flows AS (
    SELECT
        src_id_addr AS src_ip,
        dist_id_addr AS dst_ip,
        dist_port AS dst_port,
        ip_protocol_id AS protocol,
        num_bytes AS bytes,
        num_packets AS packets,
        CAST(flow_start_time AS TIMESTAMPTZ) AS flow_start,
        CAST(flow_end_time AS TIMESTAMPTZ) AS flow_end,
        to_timestamp(floor(epoch(CAST(flow_start_time AS TIMESTAMPTZ)) / {window_s}) * {window_s}) AS window_start
    FROM read_parquet({files})
    WHERE flow_start_time >= {day_start} AND flow_start_time < {day_end}
      AND src_id_addr IS NOT NULL
      AND {completion_rule}
)
SELECT
    src_ip,
    window_start,
    window_start + INTERVAL '{window_minutes} minutes' AS window_end,
    '{period}' AS period,
    count(*) AS flows,
    sum(bytes) AS bytes_total,
    sum(packets) AS packets_total,
    sum(bytes) / NULLIF(sum(packets), 0) AS bytes_per_packet,
    count(DISTINCT dst_ip) AS uniq_dst_ip,
    count(DISTINCT dst_port) AS uniq_dst_port,
    avg(CASE WHEN flow_end >= flow_start THEN epoch(flow_end) - epoch(flow_start) END) AS mean_duration_s,
    avg(CAST(protocol = 6 AS DOUBLE)) AS tcp_share,
    avg(CAST(protocol = 17 AS DOUBLE)) AS udp_share,
    avg(CAST(protocol = 1 AS DOUBLE)) AS icmp_share
FROM flows
GROUP BY src_ip, window_start
"""


def sql_text(value):
    """Quote a path or string for SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_time(moment):
    return f"TIMESTAMPTZ '{moment.isoformat()}'"


def parse_day(text):
    """'YYYY-MM-DD' -> UTC midnight."""
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def connect():
    config.DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET memory_limit = {sql_text(config.DUCKDB_MEMORY_LIMIT)}")
    con.execute(f"SET temp_directory = {sql_text(config.DUCKDB_SPILL_DIR)}")
    con.execute(f"SET threads = {int(config.DUCKDB_THREADS)}")
    con.execute("SET preserve_insertion_order = false")
    return con


def check_input(con, input_glob):
    """Return the sorted input file list; fail clearly on no files or missing required columns."""
    files = [row[0] for row in con.execute(f"SELECT file FROM glob({sql_text(input_glob)}) ORDER BY file").fetchall()]
    if not files:
        raise SystemExit(f"No Parquet files match {input_glob}")
    columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_text(input_glob)})").fetchall()}
    missing = sorted(set(RAW_COLUMNS) - columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {', '.join(missing)}")
    return files


def input_day_range(con, input_glob):
    """First UTC day and the day after the last UTC day that have flows (used by `score` without dates)."""
    first, last = con.execute(
        f"SELECT CAST(min(flow_start_time::TIMESTAMPTZ) AS DATE), CAST(max(flow_start_time::TIMESTAMPTZ) AS DATE) "
        f"FROM read_parquet({sql_text(input_glob)})").fetchone()
    if first is None:
        raise SystemExit(f"No flows in {input_glob}")
    return parse_day(first.isoformat()), parse_day(last.isoformat()) + timedelta(days=1)


def build_period_features(con, input_glob, period, start, end, out_dir, window_minutes, bounded_end=True):
    """Write one feature Parquet per UTC day of [start, end) to out_dir/period/. Replaces earlier output.

    window_minutes comes from config.py in develop and from the saved model in test/score.

    With bounded_end, flows that end at or after `end` are excluded (and counted), so activity completed in a
    later period cannot enter this one. A completed long flow is attributed to its start window (retrospective).
    """
    if (24 * 60) % window_minutes:
        raise SystemExit(f"Window of {window_minutes} minutes must divide 24 hours")
    period_dir = Path(out_dir) / period
    if period_dir.exists():
        shutil.rmtree(period_dir)
    period_dir.mkdir(parents=True)
    completion_rule = f"(flow_end_time IS NULL OR flow_end_time < {sql_time(end)})" if bounded_end else "TRUE"
    day = start
    while day < end:
        query = DAY_FEATURE_SQL.format(
            files=sql_text(input_glob), day_start=sql_time(day), day_end=sql_time(day + timedelta(days=1)),
            completion_rule=completion_rule, window_s=window_minutes * 60,
            window_minutes=window_minutes, period=period)
        con.execute(f"COPY ({query}) TO {sql_text(period_dir / f'{day:%Y-%m-%d}.parquet')} (FORMAT parquet)")
        day += timedelta(days=1)

    rows, flows = con.execute(
        f"SELECT count(*), coalesce(sum(flows), 0) FROM read_parquet({sql_text(period_dir / '*.parquet')})").fetchone()
    if rows == 0:
        raise SystemExit(f"No usable host-window rows for {period} [{start:%Y-%m-%d}, {end:%Y-%m-%d})")
    excluded = 0
    if bounded_end:
        excluded = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_text(input_glob)}) WHERE flow_start_time >= {sql_time(start)} "
            f"AND flow_start_time < {sql_time(end)} AND flow_end_time >= {sql_time(end)}").fetchone()[0]
    return {"period": period, "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}",
            "days": (end - start).days, "host_windows": int(rows), "flows": int(flows),
            "excluded_flows_ending_after_period": int(excluded), "features_dir": str(period_dir)}
