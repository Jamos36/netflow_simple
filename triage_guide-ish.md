# Standalone candidate triage

Place **triage.py beside candidates.parquet**, then run:

```bash
python3 triage.py
```

Use the full file from the detector's `test/candidates/candidates.parquet` or corresponding new-data run. You can copy both files into a separate folder, or put the script into that candidates folder. The script finds input beside itself even when invoked from another working directory. It does not import the detector, load a model, access the network, retrain anything, or change the input.

Parquet input requires **pyarrow**, which the detector already uses. On a separate machine without it, install it once with `python3 -m pip install pyarrow`. Python 3.10+ is recommended. CSV input uses only the Python standard library.

If `candidates.parquet` is absent, the script uses `top_candidates.csv` beside it. That CSV may contain only the detector's top 500 candidates, so use Parquet for complete coverage. If both are present, Parquet wins. To use another location, edit the `INPUT_FILE` constant near the top of the script; no command-line arguments are required.

## What it produces

Open `triage_results/triage.html` after running. The same folder contains:

- `triage.csv`: every supplied candidate, with its original columns and the new explanations. Written in batches/input order, not globally ranked.
- `top_triage.csv`: up to 500 candidates sorted by triage priority and then descending raw anomaly score. Ties preserve input order.
- `triage.html`: the first 30 ranked candidates, evidence, benign alternatives, and investigation questions.
- `triage_settings.json`: the applied heuristics, input path, priority counts, limitations and source links.

Rerunning replaces these generated triage files. Memory use is bounded by a Parquet batch plus at most 500 ranked candidates; CSV input streams one row at a time. Full CSV output can still use substantial disk space.

## What the priorities mean

| Review order | Meaning |
|---|---|
| Discovery pattern + listed departure | A discovery-like pattern matches and the candidate export explicitly lists a relevant upward deviation of at least 3. |
| Discovery pattern | The pattern matches, but the export does not establish that additional departure criterion. |
| Volume lead | A previously selected candidate has substantial observed byte volume; transfer purpose is unknown. |
| Unclassified | None of these three starter rules matches; the anomaly still needs investigation. |

These are **rule-based priorities, not confidence percentages or estimated probabilities of maliciousness**. The order is an explicit review preference. It can move a lower-scoring discovery lead ahead of a higher-scoring volume anomaly. All candidates retain their original score and band, and every row's maliciousness assessment remains unknown pending independent corroboration. No match does not imply benign.

## The three starter rules

**Possible service/port sweep:** at least 20 distinct destination ports per 15 minutes. If `uniq_dst_port` appears among the exported top-three deviations with a positive value of at least 3, it receives the higher discovery priority. Tentative mapping: T1046, Network Service Discovery. Legitimate scanners, service checks and peer-to-peer traffic can produce similar patterns. The counts do not show which ports were contacted on each host, connection success or scanning intent.

**Possible ICMP host-discovery sweep:** at least 20 distinct destinations per 15 minutes, at least 80% ICMP flows and at most 4,096 mean bytes per flow. A listed upward `uniq_dst_ip` deviation of at least 3 increases review priority. Tentative mapping: T1018, Remote System Discovery. Monitoring and inventory jobs are common benign alternatives. Because these are host-window aggregates, the ICMP traffic and destination count may refer to different subsets of flows; the mapping is a hypothesis to investigate, not a verified ping sweep.

**Bulk-transfer volume:** at least 100,000,000 observed bytes per 15 minutes in an existing candidate. No ATT&CK technique is assigned. Backups, replication and ordinary downloads are alternatives. Neither destination ownership, external direction, data sensitivity nor a C2 channel is known, so this rule does not identify exfiltration.

Count and volume floors scale linearly with the candidate's recorded window duration. This is a transparent starting heuristic, not a claim that distinct counts scale linearly in real traffic. The constants are at the top of the file; tune them against reviewed traffic, not pentest-month overlap. There is no validated universal cutoff. The exported training deviations describe the global training sample, not a host-specific baseline.

## Limits that matter

The full candidate export includes ten features, but only the three largest numerical feature deviations. This standalone script cannot reconstruct omitted deviations without the saved model/reference statistics. An unlisted deviation is therefore unknown, not zero. It may under-prioritize a discovery lead whose relevant departure is missing from the export.

It only examines previously selected candidates. A pentest never selected by the original detector will not appear here. It does not establish brute force, authentication failures, command-and-control, lateral movement, execution, exploitation, exfiltration, or an actor's intent. Some malicious activity will have no matching rule. Authorized testing and malicious scanning can look identical in these aggregates.

To judge maliciousness, check the source's role and owner, approved scan/change schedules, destination/port relationships and relevant firewall/endpoint/authentication records. Those are suggested follow-up checks, not additional inputs required to run the script. A future probability model would need credible confirmed outcomes and evaluation on separate representative data; neither anomaly percentiles nor rule matches supply those labels.

## Reference basis and checks

The technique definitions come from MITRE ATT&CK; the numeric rules are original, unvalidated starter heuristics and are not MITRE detection rules:

- [T1046 - Network Service Discovery](https://attack.mitre.org/techniques/T1046/)
- [T1018 - Remote System Discovery](https://attack.mitre.org/techniques/T1018/)

Seven development tests cover missing deviation evidence, conservative volume interpretation, ICMP rule conditions, window scaling, unclassified candidates, complete output beyond the 500-row preview, Parquet/CSV consistency and empty outputs. The script also processed all 91 candidates from the detector's synthetic example export, preserving the input and resolving all report links. Tests and synthetic examples verify mechanics, not security detection quality.
