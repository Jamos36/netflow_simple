# NetFlow all-column anomaly comparison

A small CPU-only prototype that reads Parquet files directly with PyArrow, transforms every available flow column by type, and compares four unsupervised anomaly models. It writes Parquet/CSV/JSON/PNG outputs.

## What it does

1. Scans every Parquet file listed by `config.yaml`, in sorted order.
2. Finds the global timestamp range, then divides whole UTC dates chronologically: first 60% train, next 20% validation, final 20% test. No row-level random split.
3. Learns numeric medians/scales and fits models on a bounded sample of training rows only.
4. Scores train, validation, and test rows in batches. Validation scores set each model's review threshold; test scores do not affect fitting or calibration.
5. Saves daily trend plots, score distributions, daily statistics, scored Parquet, top candidate CSV, fitted model and a manifest for each selected model.
6. Shades August-September in plots as context only. It is not a label and is not used in fitting.

The default chronological 60/20/20 split should place training largely in first portion of data and validation/test in later dates when the supplied data covers later evenly. Inspect the printed date boundaries and `manifest.json`. If the automatic boundary puts candidate dates into training, set explicit split dates in the YAML before running. The broad August-September period may overlap validation and test; these are comparison/calibration periods, not trusted-clean data.

## Models

- `iforest.py`: Isolation Forest; tree isolation score, larger means stranger.
- `pca.py`: PCA reconstruction error; large reconstruction error is unusual.
- `hbos.py`: histogram based outlier score; sums per-column rarity and assumes approximate feature independence.
- `gmm.py`: diagonal Gaussian mixture negative log likelihood; low-density combinations score high.

The four modules share the same encoded rows and `fit` / `score` interface. `iforest` is the fast baseline. PCA and HBOS are lightweight comparisons. GMM can take longer as dimensions and training sample size grow. All fit limits are in `config.yaml`.

One-Class SVM and autoencoders are deliberately not included. A CNN is not a natural first model for unordered tabular flow rows. Each model reports unfamiliarity, not probability of maliciousness.

## Run it

Python 3.11+ is recommended. Install from this folder:

```bash
python -m pip install -e .
```

Edit `config.yaml`: point `data.paths` at the Parquet files and set the actual timestamp column. Check the printed date boundaries before interpreting results. Then run one model or all four:

```bash
python run.py --model iforest
python run.py --model pca
python run.py --model hbos
python run.py --model gmm
python run.py --model all
```

Results go to `../analysis_outputs/<run_id>/<model>/` by default, outside this project. `python run.py --help` lists options. The core API can also be imported from `anomaly_poc.pipeline` in a notebook.

## Interpreting output

- `scores.parquet`: one row per flow with a valid timestamp, source filename and zero-based row number, date, split, raw anomaly score, validation percentile and review band.
- `candidates.csv`: top-ranked rows per day, including original source columns for investigation.
- `daily_stats.csv`: row counts, score percentiles and fraction over the validation threshold by day.
- `daily_score_trend.png`: daily 95th-percentile scores with split and pentest-context markers. `daily_stats.csv` also gives daily median and 99th percentile. A rising raw score trend is treated as possible distribution drift, not automatically as a confirmed attack.
- `daily_alert_rate.png`: share of rows above the validation threshold by day.
- `score_distribution.png`: train, validation and test score distributions.
- `manifest.json`: split boundaries and row counts, schema, column treatment, thresholds, model parameters and input file size/modified-time fingerprints.
- With `--model all`, `outputs/<run_id>/comparison/` includes cross-model daily trends and top-candidate overlap.

The band is a rank against validation: `Critical`, `High`, `Medium`, `Low`, or `Benign`. “Benign” means only that the score fell below the configured review bands; it is not a confirmed-clean label. These are review buckets, not probability or confidence. There are no accuracy/F1 claims because the exact pentest flows are unlabeled.

## All-column handling

All Parquet data columns are retained unless listed under `data.exclude_columns`. Numeric fields are log-compressed with sign preserved, then median-imputed and robust-scaled using the training sample. String, boolean, and identifier-like numeric fields are categorical and hashed into a fixed-size representation; IPs and ports are never treated as numeric magnitudes. Timestamp fields become cyclical calendar components, not a monotonically increasing epoch feature. Missing numeric values have separate indicator columns. The manifest records the treatment of every source column.

The hash representation limits memory and handles unseen categories, but collisions are possible. High-cardinality IDs can add noise; if they dominate after inspection, exclude them explicitly and compare a new run. No input data is included in this project. Outputs default to a sibling folder outside the project and should remain outside the source-data directory.

## Important limits

- Unsupervised score ranks unfamiliarity. It cannot establish intent or maliciousness by itself.
- The validation period may include pentest activity, which can inflate the reference threshold. Review its daily score plot before interpreting candidate counts.
- A global threshold can miss anomalies during seasonal or operational drift; compare daily percentiles and rates.
- Raw flow-row scoring has not yet added host/window behavior features. If candidates are weak, engineer behavioral aggregates next instead of adding a CNN by default.
- Training uses a deterministic bounded sample for CPU/RAM control. `scores.parquet` is streamed over every row.
- Scoring all input rows still takes time. GMM has the greatest fit cost; adjust per-model sample limits if necessary.
- Date splitting requires a usable timestamp column and comparable Parquet schemas. Review automatic boundaries before trusting results.

## Project files

- `run.py`: CLI entry point.
- `anomaly_poc/data.py`: Parquet discovery, unified schema, scanning, timestamps and chronological split.
- `anomaly_poc/encoding.py`: type-aware all-column transformation and training-only scaling.
- `anomaly_poc/pipeline.py`: sample, fit, score, persist outputs and coordinate the models.
- `anomaly_poc/reporting.py`: CSV/JSON/PNG reporting and cross-model comparison.
- `anomaly_poc/models/`: four separate detector modules.
- `config.yaml`: data, split, memory, model and plot settings.
