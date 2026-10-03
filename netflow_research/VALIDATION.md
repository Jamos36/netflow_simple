# Validation

- Five targeted tests passed: cross-file aggregation, future-history invariance,
  novelty cold start, CSV/Parquet equivalence, HBOS tail ordering, PCA scoring,
  incident grouping, and incomplete/complete-label metric behavior (some tests
  cover several assertions).
- Syntax/static checks passed; all source formatted.
- Synthetic end-to-end run completed with six window detectors, two flow
  detectors, and six cumulative feature-family ablations.
- Chronological backtest completed for IF/PCA window models and IF/HBOS flows.
- Saved-model rescoring reproduced IF scores and flags exactly.
- The final plotting additions completed in an IF window/flow smoke run.
- Each of the six window detectors detected the deliberately loud synthetic
  scan. This is an execution check, not real-data effectiveness evidence.
- Tested on a small 28-day, eight-host dataset. Large real-data RAM/runtime
  performance has not been benchmarked. Compact report tables load into pandas.

## Tested versions

- duckdb: 1.5.6
- numpy: 2.3.5
- pandas: 2.2.3
- pyarrow: 25.0.1
- scikit-learn: 1.8.0
- matplotlib: 3.10.8
