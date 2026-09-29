"""Settings for the NetFlow anomaly prototype. Edit this file; there is no YAML.

All dates are UTC calendar days. Ranges are half-open: [start, end).
The example dates below match the bundled example Parquet files (2026-09-01 .. 2026-09-06).
For real data, replace INPUT_GLOB, WORK_DIR and the three periods with your own dates.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Input Parquet files (fixed NetFlow schema). Real data should live outside this source folder.
INPUT_GLOB = str(PROJECT_DIR.parent / "running_a_test" / "data" / "synth" / "raw" / "*.parquet")

# Everything generated (features, scores, reports, model, DuckDB spill) goes under WORK_DIR.
WORK_DIR = PROJECT_DIR.parent / "netflow_simple_work"
MODEL_PATH = WORK_DIR / "model.joblib"

# Chronological, non-overlapping periods: (first day, day after the last day).
TRAIN_PERIOD = ("2026-09-01", "2026-09-04")
VALIDATION_PERIOD = ("2026-09-04", "2026-09-06")
TEST_PERIOD = ("2026-09-06", "2026-09-07")

WINDOW_MINUTES = 15  # must divide 24 hours; windows are aligned to UTC midnight

# DuckDB resources. The memory limit applies to DuckDB only, not to the whole Python process.
DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2
DUCKDB_SPILL_DIR = WORK_DIR / "duckdb_spill"

# Bounded sizes
TRAIN_SAMPLE_ROWS = 100_000  # max host-windows used to fit the forest
SCORE_BATCH_ROWS = 50_000  # host-windows scored per Arrow batch
VALIDATION_REFERENCE_ROWS = 200_000  # max validation scores kept as the frozen reference
SAMPLE_SEED = 42

FOREST_SETTINGS = {
    "n_estimators": 200,
    "max_samples": 256,
    "contamination": "auto",
    "random_state": 42,
    "n_jobs": 2,
}

# Review bands: validation raw-score quantiles. Checked from the highest band down.
REVIEW_QUANTILES = {"Critical": 0.999, "High": 0.995, "Medium": 0.99}
WORKLOAD_QUANTILES = [0.95, 0.975, 0.99, 0.995, 0.999]  # for the workload plot only
LARGE_DEVIATION = 5.0  # |robust deviation| that counts as large (review heuristic)

TOP_CSV_ROWS = 500
REPORT_TOP_ROWS = 25

# Broad pentest context, e.g. [("2026-09-05", "2026-09-06")]. Keep empty during development.
# Only drawn on the timeline AFTER the model and thresholds are frozen; never used as labels.
PENTEST_INTERVALS = []
