# NetFlow behavior research

A standalone, CPU-only add-on for `netflow_simple`. Copy this entire folder into
that repo as `research_experiment/`, or use it as a new project. It does not alter
the old pipeline. No source traffic or trained company models are included.

## First run

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python make_demo.py
python run_research.py run --config demo.yaml --models all --ablations
```

The synthetic example verifies execution, not realistic malicious-traffic accuracy.
For your data, edit `config.yaml` paths and exact column names, then run:

```bash
python run_research.py run --config config.yaml
```

Start with the default four window models and two flow models. Use `--models all`
for the six window detectors, including robust PCA and autoencoder. No GPU or
PyTorch is needed; the small autoencoder uses scikit-learn's MLPRegressor.

All dates and thresholds are UTC. Explicit split dates are strongly recommended:
train ends before validation; validation ends before calibration; calibration
ends before the untouched test/pentest period. End boundaries are exclusive.
Automatic boundaries use whole dates, 50/20/15/15, and require enough days.

## What is implemented

- Multi-file CSV/Parquet aggregation using a disk-backed DuckDB database.
- One row per source host per fixed minute; change `window_minutes` for 5/15/60.
- Flow-level scores plus separate behavioral-window scores.
- Volume, destination spread, concentration, shape, outcomes, novelty, prior-host
  baselines, same-hour-of-week baseline, and prior-pair timing variability.
- Training-only feature selection, log transforms, median imputation, IQR scaling,
  missing indicators, correlation reports, optional correlated-feature pruning.
- Independent Isolation Forest, PCA, HBOS, GMM, robust PCA, autoencoder models.
- Validation PCA rank comparison; GMM component/covariance comparison by training BIC;
  autoencoder early stopping on validation reconstruction loss.
- Calibration threshold sweeps and exact full-calibration incident-budget counts.
- Frozen test thresholds, incident grouping, candidate evidence, saved joblib models.
- Feature-family ablations, chronological backtests, seed rank stability for IF,
  model candidate overlap, PSI, score/feature distributions, and correlation heatmaps.
- Host/time heatmap, local scores, feature traces, hourly alert statistics, weekly
  statistics, daily drift context, and train/validation/calibration/test distributions.
- Optional known-event recall/delay, complete-label PR-AUC/ROC-AUC/F1/precision/recall,
  and reviewed-candidate precision. Unlabeled runs do not invent those metrics.

## Input assumptions—read before interpretation

Required mapped fields: timestamp, source IP, destination IP, destination port,
bytes and packets. The input is one consistent exporter schema; malformed values
fail. Map only columns you understand. Duration is seconds; map it if available.
If absent, remove `shape` from feature families. Failure/external are optional
explicit 0/1 fields. Failure is not blindly inferred from TCP flags. IPs and ports
are identifiers for grouping/novelty; their numeric magnitudes are not inputs.

Bytes describe the selected exporter record, not automatically outbound or
bidirectional traffic. `src_ip` means the record's source, not necessarily its
initiator. Confirm exporter direction, sampling, duplicates, and counters yourself.
The tool does not deduplicate or correct exporter sampling.

`columns.timestamp` is the observation/event time. By default it is flow start,
and entire record counters are assigned to its start minute: this is retrospective
analysis, NOT a real-time detector. If completed counters become available only at
flow end, map timestamp to the end/export time for availability-aware evaluation.
There is no assumed uniform byte allocation across a long flow. Such allocation
would invent within-flow timing. The optional end column is retained for evidence.
For numeric end timestamps, map end to null unless converted to a timestamp first.
Offset-free timestamps mean UTC; calendar baselines are UTC, including DST effects.

Files are aggregated together before windows are finalized; a minute split across
files still has one vector. `window_members.parquet` links host/window to input
file and row. Parquet row numbers are physical zero-based rows. CSV row numbers
are scan indices within a file for this run, not guaranteed physical line numbers.

## Historical features

Novelty uses all earlier observed windows per host. Every contact to a destination
first observed in the current minute contributes to that minute's novelty rate.
The beginning of collection has a cold start; novelty there is not proof of attack.

Host volume/spread baselines compare logged current values with the median of
prior daily median active-window values. MAD is computed across those daily
medians. Current day is excluded; minimum 3 prior observed days by default.
This measures deviation from typical active-minute behavior, not daily byte totals.
`scale_floor` prevents zero-MAD explosion; inspect it, particularly for quiet hosts.

Seasonal baseline uses earlier whole dates with the same UTC weekday/hour, up to
35 days. It needs at least two matching dates. Short training histories can leave
it entirely unavailable; such features are excluded until a later retraining.
Timing uses prior host/destination gaps ending in earlier windows, over the past
2 hours. It excludes the current minute and needs at least 3 gaps. Sparse history
stays missing. Median pair CV is a starting summary; it can dilute one unusual pair.

No future observations enter history features. Earlier test traffic may enter later
test histories, as in deployment, while models/transforms/thresholds remain frozen.
`coverage.parquet` records gaps between active windows. Absence is never zero-filled:
without independent telemetry coverage, silence and collection failure are ambiguous.

## Model details

All raw scores increase with unusualness. Calibration percentiles are ranks,
not probabilities or maliciousness confidence. There are no 'benign' guarantees.

PCA combines mean-square reconstruction error and variance-normalized retained-
component distance, using the maximum of training-p95-normalized terms. Rank is
selected by train/validation tail-rate consistency: a stability heuristic, not
proof of detection quality. Do not select rank using the final pentest/test labels.

HBOS sums smoothed per-feature histogram rarity. Outside-range values receive a
separate tail penalty increasing with distance; training-constant dimensions also
penalize deviations. It still assumes feature independence. Duplicate information
can overweight a behavior; compare ablations and correlation diagnostics.

GMM evaluates configured component counts and diagonal/full covariance by training
BIC. BIC assesses density fit, not attack detection. Check convergence in metrics.

Robust PCA runs Principal Component Pursuit ONLY on a bounded training sample,
then learns a frozen PCA basis from its low-rank component. Test scoring uses
iterative sparse-residual projection plus retained-component distance. This is an
inductive research approximation, NOT a joint PCP decomposition of the test data.
Inspect its reported training reconstruction error; low-rank/sparse assumptions
may be inappropriate. It is capped at 2,000 training rows because repeated SVDs cost.

Autoencoder architecture: input → max(8,input width) → bottleneck → symmetric
hidden layer → input. Adam, regularization, validation early stopping, best epoch
restored. Score is reconstruction MSE; contribution table shows squared residuals.
A low training/validation loss alone does not establish useful detection.

For IF/HBOS/GMM, explanations are standardized feature deviations, NOT formal
model attributions. For reconstruction models they are per-dimension residuals.
`candidate_feature_contributions.csv` rows match the first candidate rows in order.

## Calibration and incidents

Calibration quantiles come from a bounded deterministic sample. Every calibration
observation is then evaluated for each candidate threshold, with exact incident
counts. Pick the least strict configured cutoff meeting incidents/day budget; if
none meets it, pick the strictest and report `budget_met=false`. Ties use `score >
threshold`, so discrete scores need not produce the nominal quantile alert rate.

Flagged observations on the same host merge if separated by at most
`incident_gap_minutes`. Budget counts use raw observations; flow plots/report
metrics use host-minute maximum flow scores, so their grouped counts can differ
at minute boundaries. Flow raw scores and flags remain in scores.parquet.
Window and flow cutoffs are separate. No model scores are averaged as probabilities.

Candidate CSVs contain the top test rows, including below-cutoff candidates; use
`alert` to distinguish threshold detections from ranking-only retrieval.

## Labels and review

Optional `events_csv`:

```csv
event_id,host,start,end
scan1,10.0.0.1,2026-09-05T10:00:00Z,2026-09-05T10:10:00Z
```

Intervals are [start,end); host `*` matches all hosts. Window overlap establishes
known-event membership. Event recall counts all supplied test-overlapping events,
including those with no observed matching window; `represented` exposes coverage.
Detection delay is from event start to first flagged overlapping window start,
clamped at zero. The event report is NOT exact malicious-flow labeling.

Set `labels.complete: true` only if every observed test host-window outside those
intervals is reliably negative. Otherwise only known-event recall and delay are
reported. ROC-AUC/PR-AUC/F1 then describe observed host-windows, not all absent
minutes and not individual-flow ground truth. Broad pentest month shading is context
only; it never supplies labels or influences fitting/calibration.

Optional reviews CSV has `model,level,host,time,malicious` (0/1), where level is
`window` or `flow`. Match candidate host/time exactly. For flow reviews, review a
host/time candidate once; all flows at that same time can share the review. Reported
precision applies only to matched reviewed candidates and can be selection-biased.
Do not present it as population precision. Prioritize review across several models.

## Ablations and backtests

```bash
python run_research.py run --config config.yaml --ablations
python run_research.py backtest --config config.yaml
```

Ablations add families cumulatively in the configured order and refit IF for each.
Compare `ablation_comparison.csv` at the shared incident budget. Alerts alone do not
prove an improvement; reviewed findings or known-event recall are stronger evidence.
Every backtest refits preprocessing/models and recalibrates its threshold using
that fold's historical splits. Build features first. Add folds to `backtests`, with
name/train_end/validation_end/calibration_end. Each fold tests through the dataset
end, so test spans can overlap; they are not statistically independent estimates.

To compare window sizes, change `window_minutes`, use a new output directory, and
rerun. Compare 1/5/15 minutes on the SAME chronological periods and incident budget.
Small windows offer localization; larger windows can reveal low-rate patterns.

## Frozen scoring

```bash
python run_research.py score \
  --model-file results/analysis/window/iforest/model.joblib \
  --input results/features/windows.parquet --output rescored.parquet
```

Use the same feature contract and window size. For new raw files, build features
including the historical records needed for novelty/baselines, then score the new
feature rows. A new-data-only feature build cold-starts history. Joblib files are
trusted local development artifacts; do not load untrusted models.

## Scale and outputs

Raw reads, feature aggregation, model scoring and score writes use DuckDB/PyArrow;
model fits are bounded to `fit_rows`. All raw flow scores stream to Parquet. Flow
reporting first collapses scores to host-minute maxima. Compact host-window scores
are loaded into pandas for plots/evaluation: enormous host counts can still require
substantial RAM. This research version does not claim unlimited scale or that a
DuckDB memory setting limits the Python process. Disk holds canonical flows,
intermediate aggregates, membership, scores, and optional spill files.

`results/features/`: windows, flows, membership, coverage, input fingerprints,
DuckDB feature tables. `results/analysis/window/<model>/`: scores, thresholds,
candidates, contributions, incidents, metrics, model bundle, plots and summaries.
`analysis/flow/` has separate flow detectors. Parent model_comparison and overlap
files compare detectors; feature diagnostics use bounded split samples, so PSI,
quantiles and correlations are approximate. Plotted heatmap cells show hourly
maximum rank and omit unobserved cells; they do not turn hourly bins into model inputs.

Keep source data and trained models outside Git. Never copy real company traffic
into a public repository. Synthetic tests are runnable with:

```bash
python -m pip install pytest
python -m pytest -q
```
