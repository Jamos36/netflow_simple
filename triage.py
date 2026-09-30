## I want to further narrow down the potential candidates - hopefully this does the trick.

"""In order to run this, use: python3 triage.py. Put this file beside candidates.parquet (or top_candidates.csv).

Only Parquet input needs pyarrow, already included in the prototype's requirements.
This is a transparent review aid, not a maliciousness classifier or attack-probability model.
"""

import csv
import heapq
import html
import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_FILE = None  # Optional: Path("/full/path/to/candidates.parquet") or a candidate CSV
OUTPUT_DIR = HERE / "triage_results"
# Starter review heuristics, not empirically validated detection thresholds.
PORTS_PER_15_MIN = 20
HOSTS_PER_15_MIN = 20
ICMP_SHARE_MIN = 0.8
SMALL_FLOW_BYTES_MAX = 4096
TRANSFER_BYTES_PER_15_MIN = 100_000_000
DEVIATION_MIN = 3.0
EXTRA = ["triage_rank", "triage_priority", "behavior_hypothesis", "attack_hypotheses",
         "triage_evidence", "benign_explanations", "check_next", "maliciousness_assessment"]
REQUIRED = {"src_ip", "window_start", "window_end", "raw_score", "flows", "bytes_total",
            "uniq_dst_ip", "uniq_dst_port", "icmp_share"}
NOTE = ("Priorities describe review order, not maliciousness probabilities. ATT&CK mappings are tentative "
        "behavior hypotheses, not confirmed techniques. No rule match does not mean benign. "
        "Only the existing candidate set is assessed; rows excluded by the detector cannot be recovered. "
        "Candidate exports contain at most three numerical feature deviations, so an unlisted deviation "
        "is unknown, not zero. A pattern and its deviation are not independent evidence.")


def number(value):
    try:
        return float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def assess(row):
    """Inspect the fixed candidate-export fields. No model, network calls or repository imports."""
    get = lambda name: number(row.get(name))
    dates = [datetime.fromisoformat(str(row[name]).replace("Z", "+00:00"))
             for name in ("window_start", "window_end")]
    minutes = (dates[1] - dates[0]).total_seconds() / 60
    if minutes <= 0:
        raise ValueError("Candidate window_end must be after window_start")
    factor = minutes / 15
    departures = {row.get(f"dev{k}_feature"): get(f"dev{k}_value") for k in (1, 2, 3)}
    patterns, ids, evidence, benign, checks, ranks = [], [], [], [], [], []

    def add(rank, pattern, attack, key, extra, alternative, check):
        deviation = departures.get(key, float("nan"))
        context = (f"exported transformed deviation={deviation:+.1f}" if math.isfinite(deviation)
                   else "numerical deviation not listed in candidate export")
        ranks.append(rank); patterns.append(pattern)
        if attack:
            ids.append(attack)
        evidence.append(f"{key}={get(key):.4g} in {minutes:g} minutes; {context}{extra}")
        benign.append(alternative); checks.append(check)

    if get("uniq_dst_port") >= PORTS_PER_15_MIN * factor:
        rank = 3 if departures.get("uniq_dst_port", 0) >= DEVIATION_MIN else 2
        add(rank, "Possible service/port sweep", "T1046: Network Service Discovery (possible)",
            "uniq_dst_port", "", "Authorized vulnerability scans; service checks; peer-to-peer traffic",
            "Inspect destination/port pairs and connection outcomes; verify scanner ownership and authorized scope.")
    bytes_per_flow = get("bytes_total") / get("flows") if get("flows") > 0 else float("nan")
    if (get("uniq_dst_ip") >= HOSTS_PER_15_MIN * factor and get("icmp_share") >= ICMP_SHARE_MIN
            and bytes_per_flow <= SMALL_FLOW_BYTES_MAX):
        rank = 3 if departures.get("uniq_dst_ip", 0) >= DEVIATION_MIN else 2
        add(rank, "Possible ICMP host-discovery sweep", "T1018: Remote System Discovery (possible)",
            "uniq_dst_ip", f"; ICMP share={get('icmp_share'):.2f}; mean bytes/flow={bytes_per_flow:.0f}",
            "Monitoring probes; authorized inventory; network troubleshooting",
            "Check ICMP types and target sequence; identify the originating process and scheduled job. "
            "Host-level aggregates do not prove the ICMP flows contacted all counted destinations.")
    if get("bytes_total") >= TRANSFER_BYTES_PER_15_MIN * factor:
        add(1, "Unusual candidate with bulk-transfer volume; intent unknown", "", "bytes_total", "",
            "Backups; replication; software distribution; legitimate downloads",
            "Check destination ownership, transfer direction, schedules and endpoint file/process activity. "
            "Volume alone cannot establish exfiltration or a C2 channel.")
    if not ranks:
        ranks.append(0); patterns.append("Unclassified statistical anomaly")
        evidence.append("No configured behavior rule matched these candidate features.")
        benign.append("Host-role differences; rare legitimate workloads; an unmodelled behavior")
        checks.append("Inspect the existing deviations and surrounding traffic; no rule match is not a clearance.")
    rank = max(ranks)
    priority = ["Unclassified", "Volume lead", "Discovery pattern", "Discovery pattern + listed departure"][rank]
    return dict(zip(EXTRA, [rank, priority, " | ".join(patterns), "; ".join(ids), " | ".join(evidence),
                           " | ".join(benign), " | ".join(checks), "Unknown: independent corroboration needed"]))


def input_rows(path):
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        rows = (row for batch in parquet.iter_batches(batch_size=10_000) for row in batch.to_pylist())
        return parquet.schema_arrow.names, rows
    if path.suffix.lower() == ".csv":
        def rows():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                yield from csv.DictReader(stream)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            names = csv.DictReader(stream).fieldnames or []
        return names, rows()
    raise ValueError("Input must be the prototype's candidate Parquet or CSV export")


def write_html(rows, count, source, target):
    columns = ["src_ip", "window_start", "raw_score", "band", "triage_priority", "behavior_hypothesis",
               "attack_hypotheses", "triage_evidence", "benign_explanations", "check_next"]
    esc = lambda x: html.escape(str(x if x is not None else ""))
    head = "".join(f"<th>{esc(name)}</th>" for name in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(row.get(name))}</td>" for name in columns) + "</tr>" for row in rows)
    csv_note = "CSV may be only the detector's top-500 preview; this is not necessarily its full candidate set." if source.suffix.lower() == ".csv" else "Input is the supplied candidate Parquet, not all scored traffic."
    target.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Candidate triage</title>
<style>body{{font:15px sans-serif;margin:24px;color:#193047}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccd6df;padding:8px;text-align:left;vertical-align:top}}th{{background:#edf5f6}}</style>
</head><body><h1>Candidate triage</h1><p>Input: {esc(source.name)}. {count:,} candidates; showing {len(rows)}.</p>
<p><b>Maliciousness remains unknown.</b> {esc(NOTE)}</p><p>{csv_note}</p>
<p>Order: discovery pattern with a listed upward deviation; discovery pattern; bulk-volume lead; unclassified.
Within each group, larger raw anomaly scores come first. This is a heuristic, not estimated attack likelihood.</p>
<p><a href="triage.csv">All enriched candidates</a> | <a href="top_triage.csv">Top 500 by triage order</a> |
<a href="triage_settings.json">Rule settings</a></p>
<div style="overflow-x:auto"><table><tr>{head}</tr>{body}</table></div>
{'<p>No candidates in the input file.</p>' if count == 0 else ''}</body></html>""", encoding="utf-8")


def triage(source, out_dir):
    source, out_dir = Path(source), Path(out_dir)
    names, rows = input_rows(source)
    if REQUIRED - set(names):
        raise ValueError(f"Missing candidate-export fields: {sorted(REQUIRED - set(names))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = names + [name for name in EXTRA if name not in names]
    best, counts, count = [], {}, 0
    with (out_dir / "triage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for count, row in enumerate(rows, start=1):
            row.update(assess(row)); writer.writerow(row)
            counts[row["triage_priority"]] = counts.get(row["triage_priority"], 0) + 1
            score = number(row.get("raw_score"))
            key = (row["triage_rank"], score if math.isfinite(score) else -math.inf, -count)
            item = (key, row)
            if len(best) < 500:
                heapq.heappush(best, item)
            elif key > best[0][0]:
                heapq.heapreplace(best, item)
    top = [row for _, row in sorted(best, key=lambda item: item[0], reverse=True)]
    with (out_dir / "top_triage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(top)
    write_html(top[:30], count, source, out_dir / "triage.html")
    settings = {"input": str(source), "candidate_count": count, "priority_counts": counts, "interpretation": NOTE,
                "rules": {name: globals()[name] for name in ["PORTS_PER_15_MIN", "HOSTS_PER_15_MIN",
                "ICMP_SHARE_MIN", "SMALL_FLOW_BYTES_MAX", "TRANSFER_BYTES_PER_15_MIN", "DEVIATION_MIN"]},
                "sources": ["https://attack.mitre.org/techniques/T1046/", "https://attack.mitre.org/techniques/T1018/"]}
    (out_dir / "triage_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Triaged {count:,} candidates. Open {out_dir / 'triage.html'}")


if __name__ == "__main__":
    source = Path(INPUT_FILE) if INPUT_FILE is not None else next(
        (HERE / name for name in ("candidates.parquet", "top_candidates.csv") if (HERE / name).exists()), None)
    if source is None:
        raise SystemExit("Place candidates.parquet or top_candidates.csv beside triage.py, then run python3 triage.py")
    triage(source, OUTPUT_DIR)
