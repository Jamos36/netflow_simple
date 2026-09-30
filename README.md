# netflow_simple

A small feasibility prototype. It ranks unusual **source host x 15-minute UTC window** activity in NetFlow Parquet
files with one CPU scikit-learn IsolationForest. Scores are rankings for review, not attack probabilities.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings: input glob, work folder, UTC periods, DuckDB limits, sample sizes, forest, bands |
| `features.py` | Fixed column mapping, the ten features, one DuckDB aggregation per UTC day -> daily feature Parquet |
| `model.py` | Training sample, IsolationForest fit, validation thresholds, deviations, batch scoring |
| `report.py` | Candidate tables, daily summary, six plots, `summary.json`, `report.html` |
| `run.py` | The three commands |
| `tests/test_core.py` | A few checks on tiny temporary fixtures |

## Run

```bash
python -m pip install -r requirements.txt
python run.py develop      # train + validation only: fit, calibrate, save model.joblib, development report
python run.py test         # score the held-out test period once with the saved model
python run.py score --input '/path/to/later/*.parquet' --out '/path/to/new_results'
python -m pytest -q
```

Requires DuckDB 1.5.6 or newer (older versions lack the Arrow methods used).

`score` can take `--start YYYY-MM-DD --end YYYY-MM-DD` to bound the days. Without `--end`, flows are not cut at
the period end.

## What to edit for real data (config.py)

- `INPUT_GLOB`: your Parquet files. Keep real data **outside** this source folder.
- `WORK_DIR`: where features, scores, reports and `model.joblib` go. Keep it outside this folder too.
- `TRAIN_PERIOD`, `VALIDATION_PERIOD`, `TEST_PERIOD`: UTC days, half-open `[start, end)`, chronological and
  non-overlapping. The example dates fit the bundled example files only. Choose real dates from your own
  knowledge of the data. Do not pick them from the pentest month. If a broad pentest interval is known, keep it
  out of train/validation when chronology allows. If it cannot be kept out, the model may have learned the
  activity you are looking for.
- `DUCKDB_MEMORY_LIMIT` (starts at 2GB) and `DUCKDB_THREADS` (2). Raise them on a bigger machine. The limit is for
  DuckDB only, not for the whole Python process.
- `PENTEST_INTERVALS`: leave empty until the model and thresholds are frozen. The intervals are drawn only as
  broad shading on the timeline and are never used as labels.

## Outputs

Each analysis has its own folder: `WORK_DIR/development`, `WORK_DIR/test`, or the `--out` folder. **A rerun
replaces that folder's generated results.** Only folders created by this tool are replaced. `test` and `score` only
read `model.joblib`; only `develop` writes it, together with `model_settings.json` (settings + package versions).
`test` and `score` use the window size, test dates and band quantiles saved in the model, so editing `config.py`
after `develop` does not change them. Rerun `develop` to apply new values; older model files are refused.

`scores/` (every scored host-window), `candidates/candidates.parquet` (all candidates), `top_candidates.csv`
(top 500), `daily_summary.csv`, `summary.json`, `report.html` with `plots/` (keep them together), and `features/`.

## How it works, briefly

- Eight raw columns are read: `src_id_addr`, `dist_id_addr`, `dist_port`, `ip_protocol_id`, `num_bytes`,
  `num_packets`, `flow_start_time` and `flow_end_time`. Times with a zone are converted to UTC. Times without a zone
  are read as UTC. `flow_length`, `tcp_flag` and `packet_length` are not used.
- Ten features per host-window: flows, bytes_total, packets_total, bytes_per_packet, uniq_dst_ip, uniq_dst_port,
  mean_duration_s (log1p), and tcp/udp/icmp_share. These count observed one-direction flow records, not net
  traffic. The code assumes stable host IDs and no duplicate exports.
- Each UTC day is aggregated over **all** input files, so a host-window spread across files is counted once.
  Flows that end after a train/validation/test period's end are excluded from that period and counted.
- Training uses up to 100,000 rows drawn by hash across the whole training period. Busy periods therefore weigh
  more. Features that are constant or all missing in training are dropped, with a note. The median imputer and
  training statistics come from training rows only.
- `raw_score = -score_samples`. Bands come from validation raw-score quantiles: Critical >= q99.9,
  High >= q99.5, Medium >= q99, otherwise "Below threshold", which does **not** mean safe.
  `reference_percentile` is a position in the validation scores, not a probability.
- Deviation = (x - training median) / robust scale, on the model's transformed values. It is context, not
  feature importance. Candidates are all High/Critical windows plus `deviation_only` windows (|deviation| >= 5
  and outside the training range).

## Limits

- Tested on the bundled example data only: 388,749 flows over 6 days (about 6 s per command). Results on
  synthetic data say nothing about real detection performance.
- Each day's query scans every input file. With unpartitioned files this repeats reads, and it is the first
  thing to get slow on a year of data. The later fix is to partition the files by date once. Many distinct
  destinations per day can still need a lot of DuckDB memory; the limits above are not a measured guarantee.
- Low-volume or normal-looking pentest activity can be missed. If the test period is used to tune anything, it
  becomes exploratory; keep another later period for an independent check.
