# Claude Code prompt: a small NetFlow anomaly prototype

Build a NEW, deliberately small Python prototype from scratch. The existing detector checkout is reference material, not the architecture to preserve. Put the new project in a separate sibling directory called `netflow_simple`; leave the existing checkout intact. If that directory already contains work, inspect it before changing anything. Do not include any person's name, username, email address, or identifying repository URL in generated code, documentation, metadata, or reports.

## Goal and priorities

Read multiple Parquet files containing the fixed NetFlow schema, build behavioral features, fit ONE CPU-based scikit-learn IsolationForest, calibrate review thresholds on validation data, and score a later test period. Save useful plots, a simple HTML report, and ranked anomaly candidates. Also support scoring subsequent Parquet files with the saved model without refitting.

The code will be manually typed into another environment. Prioritize short, readable, ordinary Python. This is a feasibility prototype, not a production system. Use descriptive names and small functions, not compressed one-liners. Prefer about five Python files and roughly 450-650 total application lines as a design budget, not a correctness-breaking limit. Avoid files over roughly 200 lines when responsibilities can be separated naturally. Count configuration and reporting code in the budget; report the actual total. Do not inflate scope to meet old repository rules that only concern its existing architecture.

Explicit exclusions: One-Class SVM, autoencoders, GPU support, live ingestion, APIs, databases as services, Docker, distributed processing, experiment registries, parameter search frameworks, schema contracts, Pydantic, plugin systems, dynamic feature registries, source-destination history state, automated feedback, attack labels inferred from dates, synthetic generators, separate synthetic/real modes, and general-purpose input recovery. No CSV input support. CSV outputs are welcome. Use the same pipeline for every compatible Parquet input.

## First inspect the reference

Inspect the entire repository tree, including `src/netanomaly`, tests, both Parquet directories, root configuration, and `report.html`. Read README.md, SCHEMA.md, FEATURES.md, ARCHITECTURE.md and PROJECT_STATUS.md; consult the current portions of DECISIONS.md. Inspect the actual Parquet schema and timestamps. Inspect the feature construction, model fitting, scoring, bands, outputs, charts, and report implementations. Do not conclude something is absent after checking only the root.

The reviewed reference snapshot is commit `ed7f756fc8073de1b78e254182ab222db324ddef`: 24 Python source files totaling 4,830 lines before tests. It includes six mock Parquet files totaling 60,000 rows and six more structured example files totaling 388,749 rows. Both use the same 42 raw column names. Inspect the current checkout rather than treating these counts as a contract. The root HTML report refers to external `charts/*.svg` assets that were absent from the reviewed snapshot; recreate the chosen charts. Never claim old synthetic results establish real detection performance.

Give a short scope summary, then implement the prototype through the bounded stages below. Do not stop after producing a plan and do not request approval for routine reversible implementation decisions.

## Project layout and commands

Use `config.py`, `features.py`, `model.py`, `report.py`, `run.py`, `requirements.txt`, a short `README.md`, and one small `tests/test_core.py`. Add `.gitignore` for data, output, models, and spill files. No package installation scaffolding is needed.

Use ordinary Python constants in config.py, not YAML. Include input glob, output directory, explicit train/validation/test UTC ranges, 15-minute window size, DuckDB memory limit and spill path, thread count, training cap, scoring batch size, validation-reference cap, forest settings, and review quantiles. Put the ordered feature names and the list of log-transformed features beside their definitions, without a registry class.

Provide these commands:

```bash
python -m pip install -r requirements.txt
python run.py develop
python run.py test
python run.py score --input '/path/to/later/*.parquet' --out '/path/to/new_results'
```

`develop` builds only train/validation features, trains, calibrates, saves `model.joblib`, and produces a development report. `test` loads the saved artifact, builds/scans only the configured held-out range, scores, and produces a separate test report. `score` does the same for subsequent input with the frozen artifact. Reuse the same scoring/report functions. A short argparse dispatch is enough. Do not build a separate orchestration framework or holdout ledger. Use a named output folder per analysis and document that a rerun replaces that folder's generated results. Do not silently overwrite the model during test/score. Store external real data and its outputs outside the source directory through simple configured paths; no repository-boundary detection framework.

Use the reference's more structured example Parquet folder for the smoke run, without reading its truth files. Example dates can be filled in from those files for a runnable example; clearly explain that real-data dates are edited in config.py. Do not infer real split dates from the pentest month.

## Fixed input mapping

Read only these eight raw columns for model features:

| Raw column | Internal name | Use |
|---|---|---|
| src_id_addr | src_ip | Source host / grouping key |
| dist_id_addr | dst_ip | Destination diversity and drill-down |
| dist_port | dst_port | Distinct destination services |
| ip_protocol_id | protocol | Protocol proportions |
| num_bytes | bytes | Volume |
| num_packets | packets | Volume and packet-size ratio |
| flow_start_time | flow_start | UTC bucket and split assignment |
| flow_end_time | flow_end | Duration in seconds |

Use the observed timestamp types with one explicit UTC convention. Do not implement a universal timestamp parser. Treat `flow_length` as unused; it represents milliseconds in the example, not bytes. Exclude uncertain `tcp_flag` and `packet_length` semantics from this version. Do not label-encode IPs, ports or protocol IDs as ordered magnitudes. Do not read every raw column into the training matrix. Use only the ten selected features. Do not add an all-42-column monitor, evidence viewer, feature expansion workflow, or additional feature groups. The original Parquet files remain unchanged; broader column analysis is outside this proof of concept.

## Features and bounded preprocessing

One model row = one source host in one active 15-minute UTC window. No rows are generated for inactive host-windows. Use one readable DuckDB aggregation query in features.py, run for each UTC day over ALL matching input files with a date predicate. A file is not a time partition: flows for one host-window can occur in several files. Never aggregate each file or Arrow batch separately and concatenate partial windows; never sum partial distinct counts.

Create these ten model features:

| Feature | Definition | Transform |
|---|---|---|
| flows | Number of flows | log1p |
| bytes_total | Sum of bytes | log1p |
| packets_total | Sum of packets | log1p |
| bytes_per_packet | Sum of bytes / sum of packets | log1p |
| uniq_dst_ip | Distinct destination IPs | log1p |
| uniq_dst_port | Distinct destination ports | log1p |
| mean_duration_s | Mean of (flow_end - flow_start) in seconds | log1p |
| tcp_share | Fraction of flows with protocol 6 | None |
| udp_share | Fraction with protocol 17 | None |
| icmp_share | Fraction with protocol 1 | None |

Keep `src_ip`, `window_start`, `window_end` and a period designation as metadata, never inputs. The sums count observed directional flow records; do not call them Internet uploads or net bidirectional volume. Assume the fixed export has stable host identifiers and no duplicate exports; state these assumptions briefly rather than building infrastructure for them.

Use `NULLIF(sum(packets), 0)` for the ratio. Missing numerical values use training medians. Drop all-missing or constant training features with a short recorded note, preserving the retained feature order for future scoring. Fail clearly on missing required columns or no usable rows rather than silently altering the schema. No detailed rejects ledger.

Use half-open UTC ranges [start, end), disjoint chronological periods, and windows aligned to midnight. For a train or validation period, exclude completed-flow records with flow_end at or beyond that period's end, and record the count, so a duration/byte total completed in a later split cannot enter the earlier split. All processing is retrospective: a completed long flow is attributed to its start window, not claimed detectable at that start instant. Apply the documented range rule consistently to test and explicitly bounded later scoring. No online claim.

Write daily feature Parquet files to disk. Rebuilding features may simply replace that command's feature outputs; no content-addressed cache, fingerprint, or resume ledger. Set DuckDB `memory_limit`, `temp_directory`, `threads`, and `preserve_insertion_order=false`. Keep the spill path under the configured work directory. No server or imported raw database is necessary. Start with a 2 GB DuckDB limit and two threads on a modest machine, with documentation to adjust them. This is not a total-process RAM guarantee.

Daily processing bounds group state, but unpartitioned files can be scanned repeatedly when time predicates cannot prune row groups. Explain the tradeoff. If measured input size makes this too slow, propose one-time date partitioning as a later improvement; do not build a second ingestion system now. High-cardinality distinct aggregations can still require substantial memory. Never claim a hard cap without measurement.

## Train, calibrate, test

Train only on the earlier training period. Use a bounded deterministic sample of at most 100,000 host-window rows drawn across the WHOLE training period, for example order by hash(src_ip, window_start, seed), with stable key tie-breaking, then LIMIT. Filter the training dates BEFORE sampling. Report the sample's rows and represented days; do not sample the first N rows/files or retain all rows in Python. A bounded global sample weights busy periods more heavily; document that briefly. Day-balanced sampling is a possible manual follow-up, not a default framework.

Apply log1p to the seven nonnegative count/volume/duration features, then fit a median imputer on training data. Use float32 for the model matrix where practical. No StandardScaler is required for this baseline. Train:

```python
IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=2,
)
```

These are starter settings, not optimized settings. `max_samples` controls rows per tree; it does not make an unbounded X_fit safe. IsolationForest is not an incremental partial_fit estimator. Fit once on the bounded sample and score every eligible validation/test/new-data row in batches of about 50,000 using Arrow. Never refit per file. Do not concatenate all scores or features into pandas.

Validation is for inspecting rankings, setting cutoffs, and checking review volume; it is not supervised model validation. Fit transforms and deviation reference statistics on training only. Calibrate on a deterministic representative sample of at most 200,000 validation scores, drawn across the full validation period, after all scores have been streamed to disk. If validation is smaller, use all its scores. Save reference size and sampling method. Freeze the model and calibration before `test`. Never choose features, dates, or thresholds to maximize pentest-month overlap. If the final test is used for later tuning, say it has become exploratory and reserve another period for independent evaluation.

If the broad pentest interval is known, keep it out of baseline development when chronology permits. The exact days, hosts and flows remain unknown. An earlier representative baseline is preferable; if none exists, disclose that training may contain the activity being sought. Do not automatically remove high scores and retrain.

## Scores, review bands, and deviations

Set `raw_score = -model.score_samples(X)`: larger means more unusual. Save it without rounding away ranking information. Save an additional `reference_percentile = 100 * searchsorted(sorted_validation_scores, raw_score, side="right") / n_reference`. This is a position in the validation distribution, NOT a probability of attack. It can saturate at 100; sort candidates primarily by raw score, then deviation, then source/time for deterministic ties.

Use saved raw-score quantiles from validation for bands: Critical >= q99.9, High >= q99.5 but below Critical, Medium >= q99 but below High, otherwise Below threshold. Check the highest band first. These are review priorities, not calibrated incident severities. Tied scores can collapse thresholds and change achieved rates; show actual validation counts. Do not force a fixed percentage of every test day into each band. Never label lower scores Safe or Benign.

For feature context, store training-sample medians, min/max and robust scales on the same transformed feature values used by the model. Compute signed deviation d_j = (x_j - median_j) / (1.4826 * MAD_j). If MAD is zero, use IQR/1.349 if positive. If both are zero, leave that deviation undefined and separately mark a departure from the constant reference; do not divide by epsilon and invent huge significance. Imputed values are not evidence of a measured deviation. Show each candidate's three largest defined absolute deviations, with feature values and training medians in original units where possible. This is descriptive context, not model attribution, a p-value, or proof of causation. Global training context is enough in version one; do not implement per-host rolling baselines.

Set `large_deviation` when max absolute defined deviation >= 5; this is a configurable review heuristic. Save ALL High/Critical windows, even those without a large marginal deviation: unusual combinations can matter. Add `priority_candidate` for High/Critical AND large_deviation. Separately retain windows with large_deviation AND a feature outside its training-sample range even if their forest score is lower, tagged `deviation_only`. Preserve the model band. Training-range departure alone does not prove an attack. This extra route helps expose extremes that tree scores may not order by magnitude. No opaque weighted composite score is needed.

## Saved results and evidence

Save a single joblib dictionary containing the fitted model/imputer, feature order, log-feature names, training statistics, sorted validation reference, raw thresholds, model settings, fixed mapping, window size and split dates. Reuse these values in test/score even if config.py is later edited. Store a small JSON/text settings summary and dependency versions; no registry, checksum framework, or artifact classes.

Each report directory contains:

- `scores/`: batch/day Parquet for every scored host-window, including feature values and raw score/percentile/band.
- `candidates/`: Parquet for the complete selected candidate set, not only the displayed top rows.
- `top_candidates.csv`: highest 500 selected rows; state if the total exceeds 500.
- `daily_summary.csv`: host-windows, observed flows, distinct active hosts, band counts, High/Critical rate per 1,000 active host-windows, raw-score summary statistics.
- `summary.json`: dates, sizes, model/reference counts, thresholds, score statistics, settings, elapsed time, and any missing-data notes. Label approximate/sample statistics as such.
- `report.html` and `plots/*.png`.

Candidate columns: source host, window start/end in UTC, period, raw score, validation percentile, band, large_deviation, priority_candidate, deviation_only reason if applicable, the feature values, largest deviation names/values, and beyond-training-range fields. Keep complete Parquet outputs on disk; global sorting/top-N extraction happens in DuckDB. Preserve zero-candidate outputs with headers/schema and a clear report message.

For simple traceability, keep the input file list once with the run settings and source-host/window keys in every candidate. That is sufficient for this prototype. Do not build an all-column drill-down tool, evidence export subsystem, row-ID/provenance framework, or new relationship features. Do not use `flow_sequence` as a global unique key.

## Plots and a short HTML report

Use matplotlib and a tiny static HTML string with tables generated from small summaries. No JavaScript, dashboard framework, templating engine, or recreation of the old 11-section report. One reusable plotting style is sufficient. Use original-unit table labels, readable axes and UTC labels. Plot only bounded summaries or clearly labelled samples; never every raw flow.

Produce six figures:

1. Timeline: daily observed-flow volume and High/Critical counts plus rate per 1,000 active host-windows. Weekly overview for long spans may be added by simple resampling of daily summaries; keep daily data on disk.
2. Raw-score distributions for validation and the current scored period, normalized for unequal row counts, using identical bins with threshold lines. Prefer streamed histogram counts for all scored rows. Keep the frozen validation histogram or clearly identify a validation-reference sample.
3. Threshold-versus-review-volume curve on validation at q95, q97.5, q99, q99.5 and q99.9. Label as workload, not accuracy. Do not use test to select a threshold.
4. Top-20 source-host by day heatmap of maximum raw score. Missing/inactive cells are blank, not zero. Select hosts by candidate counts with documented tie-breaking.
5. A fixed +/-24-hour detail view around the highest raw-score candidate in the scored period, with thresholds and source/time labels; show no-candidate text if none exists. Select without supplied pentest dates.
6. Feature context panel: a Spearman correlation matrix from the bounded training sample and a transformed-feature deviation view for the top candidates. Describe correlation as redundancy, and deviations as context, not learned feature importance.

HTML order: run counts/periods and score explanation; overview and distribution; threshold workload; ranked top-25 candidates; heatmap/detail/context; output links and brief limitations. Escape table text. The report and its `plots` directory travel together. Reuse the same template for development and final scoring. Keep development and test candidates distinct. No model-comparison plots, robustness suite, drift framework or relationship-history report.

Keep supplied pentest intervals empty by default. If later provided, overlay them only AFTER model and ranking choices are frozen, and label them broad context. Do not use bundled truth files. Do not calculate F1, precision, recall, ROC AUC, PR AUC, accuracy or a confusion matrix from month membership. The supported conclusion is: these hosts and times are anomalous candidates, which can later be checked against independent pentest records. Low-volume or normal-looking pentests may be missed.

## Implementation stages and acceptance

Work in this order: (1) fixed features and daily Parquet output; (2) bounded training/calibration and model artifact; (3) frozen test/new-data scoring and candidate storage; (4) six plots and small report; (5) smoke run and concise README. Keep explaining the purpose of each file so the code is easy to retype, but do not generate a documentation suite.

Add only a few substantive tests, using tiny fixtures: a host-window split across files aggregates once with correct distinct counts; future test rows cannot change the training sample/thresholds; changing scoring batch size does not change row scores; empty candidates still yield usable outputs; tied percentile thresholds and zero-spread deviation references behave as documented. A temporary test fixture is fine; a user-facing synthetic generation subsystem is not.

Run the prototype on the compatible bundled Parquet data using the SAME code path as real data. Verify output row counts and keys, frozen model reuse, candidate ordering, report links and visual readability. Report runtime and observed peak RSS for the example run if measurement is available, explicitly including what was measured; do not extrapolate a guarantee for months of real data. State any untested scale limits. The absence of a GPU should not block completion.

Finish with the project location, exact commands, settings the user must edit, file line counts, outputs produced, and test results. Do not push, publish, modify the old repository, or add unrequested systems. If an optional feature threatens simplicity, list it as deferred and deliver the working core.
