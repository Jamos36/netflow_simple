"""Disk-backed aggregation. Input contracts are explicit; malformed rows fail."""

from pathlib import Path
import glob
import duckdb

FAMILIES = {
    "volume": ["flows", "bytes", "packets"],
    "spread": ["unique_ips", "unique_ports", "top_destination_share"],
    "shape": ["median_duration", "median_bytes", "short_fraction", "failure_fraction"],
    "novelty": ["new_ip_rate", "new_port_rate"],
    "baseline": ["bytes_robust_z", "destinations_robust_z", "seasonal_bytes_z"],
    "timing": ["interarrival_cv"],
}
FLOW_FEATURES = [
    "bytes",
    "packets",
    "duration",
    "bytes_per_packet",
    "failure",
    "external",
]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def ident(value):
    return '"' + value.replace('"', '""') + '"'


def export(con, query, path):
    con.execute(f"COPY ({query}) TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)")


def build(cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        {
            str(Path(p).resolve())
            for pattern in cfg["data"]["paths"]
            for p in glob.glob(pattern, recursive=True)
        }
    )
    if not paths:
        raise ValueError("No input files matched data.paths")
    con = duckdb.connect(str(out / "features.duckdb"))
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit={literal(cfg['data'].get('memory_limit', '2GB'))}")
    con.execute(f"SET threads={int(cfg['data'].get('threads', 4))}")
    con.execute(f"SET temp_directory={literal(out / 'spill')}")
    # Read each format as a relation; all selected input columns must agree.
    relations = []
    for ext in [".parquet", ".csv"]:
        selected = [p for p in paths if Path(p).suffix == ext]
        if selected:
            names = "[" + ",".join(map(literal, selected)) + "]"
            reader = (
                f"read_parquet({names}, union_by_name=true, filename=true, file_row_number=true)"
                if ext == ".parquet"
                else f"read_csv({names}, union_by_name=true, filename=true)"
            )
            row = (
                "file_row_number"
                if ext == ".parquet"
                else "row_number() OVER (PARTITION BY filename)-1"
            )
            con.execute(
                f"CREATE OR REPLACE VIEW input_{ext[1:]} AS SELECT *, {row} AS source_row FROM {reader}"
            )
            relations.append(f"input_{ext[1:]}")
    if not relations:
        raise ValueError("Use .csv or .parquet inputs")
    m = cfg["columns"]

    def col(name, default="NULL"):
        return ident(m[name]) if m.get(name) else default

    timestamp = col("timestamp")
    unit = cfg["data"].get("timestamp_unit")
    timestamp = (
        f"to_timestamp({timestamp} / {dict(s=1, ms=1000, us=1000000, ns=1000000000)[unit]})"
        if unit
        else f"CAST({timestamp} AS TIMESTAMPTZ)"
    )
    end = f"CAST({col('end')} AS TIMESTAMPTZ)" if m.get("end") else timestamp
    # Timestamp is event/observation time. End/start distinction is documented.
    select = f"""SELECT filename AS source_file, source_row, {timestamp} AS time,
      {end} AS end_time, CAST({col("src_ip")} AS VARCHAR) AS host,
      CAST({col("dst_ip")} AS VARCHAR) AS dst, CAST({col("dst_port")} AS VARCHAR) AS port,
      CAST({col("bytes")} AS DOUBLE) AS bytes, CAST({col("packets")} AS DOUBLE) AS packets,
      CAST({col("duration", "0")} AS DOUBLE) AS duration,
      CAST({col("failure")} AS DOUBLE) AS failure,
      CAST({col("external")} AS DOUBLE) AS external FROM {{relation}}"""
    con.execute(
        "CREATE OR REPLACE TABLE flows AS "
        + " UNION ALL ".join(select.format(relation=r) for r in relations)
    )
    bad = con.execute(
        "SELECT count(*) FROM flows WHERE time IS NULL OR host IS NULL OR dst IS NULL OR port IS NULL OR bytes < 0 OR packets < 0 OR duration < 0"
    ).fetchone()[0]
    if bad:
        raise ValueError(
            f"{bad} rows violate the input contract; fix the selected columns"
        )
    minutes = int(cfg["features"]["window_minutes"])
    con.execute(
        f"CREATE OR REPLACE TABLE f AS SELECT *, time_bucket(INTERVAL '{minutes} minutes', time) AS window_start FROM flows"
    )
    con.execute(
        "CREATE OR REPLACE TABLE first_ip AS SELECT host,dst,min(window_start) first_window FROM f GROUP BY ALL"
    )
    con.execute(
        "CREATE OR REPLACE TABLE first_port AS SELECT host,port,min(window_start) first_window FROM f GROUP BY ALL"
    )
    con.execute(
        """CREATE OR REPLACE TABLE destination_counts AS SELECT host,window_start,dst,count(*) n FROM f GROUP BY ALL"""
    )
    short = float(cfg["features"].get("short_seconds", 1))
    con.execute(f"""CREATE OR REPLACE TABLE w AS SELECT f.host,f.window_start,
      count(*)::DOUBLE flows, sum(bytes) bytes, sum(packets) packets,
      count(DISTINCT f.dst)::DOUBLE unique_ips, count(DISTINCT f.port)::DOUBLE unique_ports,
      median(duration) median_duration, median(bytes) median_bytes,
      avg((duration <= {short})::DOUBLE) short_fraction, avg(failure) failure_fraction,
      avg((i.first_window = f.window_start)::DOUBLE) new_ip_rate,
      avg((p.first_window = f.window_start)::DOUBLE) new_port_rate
      FROM f JOIN first_ip i USING(host,dst) JOIN first_port p USING(host,port) GROUP BY f.host,f.window_start""")
    # Daily summaries give hosts equal day_date weight; all baselines exclude the current day_date.
    con.execute("""CREATE OR REPLACE TABLE daily AS SELECT host,CAST(window_start AS DATE) day_date,
      median(ln(1+bytes)) b, median(ln(1+unique_ips)) d FROM w GROUP BY ALL""")
    days = int(cfg["features"].get("history_days", 7))
    con.execute(f"""CREATE OR REPLACE TABLE centers AS SELECT a.host,a.day_date,count(b.day_date) history_days,
      median(b.b) mb,median(b.d) md FROM daily a LEFT JOIN daily b ON a.host=b.host
      AND b.day_date>=a.day_date-INTERVAL '{days} days' AND b.day_date<a.day_date GROUP BY a.host,a.day_date""")
    con.execute(f"""CREATE OR REPLACE TABLE baseline AS SELECT a.*,median(abs(b.b-a.mb)) mad_b,
      median(abs(b.d-a.md)) mad_d FROM centers a LEFT JOIN daily b ON a.host=b.host
      AND b.day_date>=a.day_date-INTERVAL '{days} days' AND b.day_date<a.day_date GROUP BY ALL""")
    # Same hour-of-week context: use prior whole days only, never future windows.
    con.execute("""CREATE OR REPLACE TABLE seasonal_daily AS SELECT host,CAST(window_start AS DATE) day_date,
      extract(hour FROM window_start) AS hour_number,extract(isodow FROM window_start) AS weekday_number,
      median(ln(1+bytes)) b FROM w GROUP BY ALL""")
    con.execute(f"""CREATE OR REPLACE TABLE seasonal AS SELECT a.host,a.day_date,a.hour_number,a.weekday_number,
      count(b.day_date) n,median(b.b) center,list(b.b) history
      FROM seasonal_daily a LEFT JOIN seasonal_daily b ON a.host=b.host AND a.hour_number=b.hour_number
      AND a.weekday_number=b.weekday_number AND b.day_date<a.day_date AND b.day_date>=a.day_date-INTERVAL '{int(cfg["features"].get("seasonal_days", 35))} days'
      GROUP BY a.host,a.day_date,a.hour_number,a.weekday_number""")
    # Gaps are calculated within each host/destination pair. Aggregate moments of gaps
    # ending in earlier minutes; the current minute cannot contribute to its own timing.
    con.execute("""CREATE OR REPLACE TABLE gaps AS SELECT host,dst,window_start,
      epoch(time-lag(time) OVER(PARTITION BY host,dst ORDER BY time)) gap FROM f""")
    hours = int(cfg["features"].get("timing_hours", 2))
    con.execute(f"""CREATE OR REPLACE TABLE pair_minutes AS SELECT host,dst,window_start,
      count(gap) n,sum(gap) s,sum(gap*gap) ss FROM gaps WHERE gap IS NULL OR gap<={hours * 3600} GROUP BY ALL""")
    hours = int(cfg["features"].get("timing_hours", 2))
    con.execute(f"""CREATE OR REPLACE TABLE pair_history AS SELECT *,
      sum(n) OVER hist hn,sum(s) OVER hist hs,sum(ss) OVER hist hss FROM pair_minutes
      WINDOW hist AS(PARTITION BY host,dst ORDER BY window_start RANGE BETWEEN INTERVAL '{hours} hours' PRECEDING AND INTERVAL '{minutes} minutes' PRECEDING)""")
    con.execute("""CREATE OR REPLACE TABLE timing AS SELECT host,window_start,
      median(CASE WHEN hn>=3 AND hs>0 THEN sqrt(greatest(0,hss/hn-pow(hs/hn,2)))/(hs/hn) END) interarrival_cv
      FROM pair_history GROUP BY ALL""")
    floor = float(cfg["features"].get("scale_floor", 0.1))
    min_days = int(cfg["features"].get("min_history_days", 3))
    query = f"""SELECT w.*, t.top_n/w.flows top_destination_share,
      CASE WHEN b.history_days>={min_days} THEN (ln(1+w.bytes)-b.mb)/greatest(1.4826*b.mad_b,{floor}) END bytes_robust_z,
      CASE WHEN b.history_days>={min_days} THEN (ln(1+w.unique_ips)-b.md)/greatest(1.4826*b.mad_d,{floor}) END destinations_robust_z,
      CASE WHEN s.n>=2 THEN (ln(1+w.bytes)-s.center)/greatest(1.4826*list_median(list_transform(s.history,x->abs(x-s.center))),{floor}) END seasonal_bytes_z,
      timing.interarrival_cv,b.history_days,
      sin(2*pi()*extract(hour FROM w.window_start)/24) hour_sin,
      cos(2*pi()*extract(hour FROM w.window_start)/24) hour_cos,
      sin(2*pi()*extract(isodow FROM w.window_start)/7) weekday_sin,
      cos(2*pi()*extract(isodow FROM w.window_start)/7) weekday_cos
      FROM w JOIN (SELECT host,window_start,max(n) top_n FROM destination_counts GROUP BY ALL) t USING(host,window_start)
      LEFT JOIN baseline b ON w.host=b.host AND CAST(w.window_start AS DATE)=b.day_date
      LEFT JOIN seasonal s ON w.host=s.host AND CAST(w.window_start AS DATE)=s.day_date
        AND extract(hour FROM w.window_start)=s.hour_number AND extract(isodow FROM w.window_start)=s.weekday_number
      LEFT JOIN timing USING(host,window_start) ORDER BY w.window_start,w.host"""
    con.execute("CREATE OR REPLACE TABLE windows AS " + query)
    export(con, "SELECT * FROM windows", out / "windows.parquet")
    export(
        con,
        """SELECT *, bytes/greatest(packets,1) bytes_per_packet FROM flows ORDER BY time,host""",
        out / "flows.parquet",
    )
    export(
        con,
        "SELECT host,window_start,source_file,source_row FROM f",
        out / "window_members.parquet",
    )
    # Missing minute inventory is not zero-filled: without telemetry coverage, absence
    # cannot be distinguished from inactivity.
    export(
        con,
        f"""SELECT host,window_start,epoch(window_start-lag(window_start) OVER(PARTITION BY host ORDER BY window_start))/60-{minutes} AS unobserved_minutes FROM w""",
        out / "coverage.parquet",
    )
    con.close()
    return paths
