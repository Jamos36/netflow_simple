"""Stages 2-3 math: fit one IsolationForest on a bounded training sample, calibrate review bands on validation
scores, and score feature rows with the frozen bundle. Nothing here refits during test or new-data scoring.

raw_score = -score_samples(X): larger is more unusual. It is a ranking, not a probability of attack.
Deviations are robust z-like distances from training-sample medians: descriptive context, not attribution.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import config
from features import FEATURES, LOG_FEATURES, sql_text, sql_time

BANDS = ["Critical", "High", "Medium"]
BELOW = "Below threshold"
HIST_BINS = np.linspace(0.0, 1.0, 201)  # raw_score of an IsolationForest lies in (0, 1]


def sample_training_rows(con, features_glob, start, end, limit, seed):
    """Deterministic sample across the WHOLE training period: filter dates first, order by a hash, then LIMIT."""
    return con.execute(f"""
        SELECT * FROM read_parquet({sql_text(features_glob)})
        WHERE window_start >= {sql_time(start)} AND window_start < {sql_time(end)}
        ORDER BY hash(src_ip, window_start, {int(seed)}), src_ip, window_start
        LIMIT {int(limit)}""").to_arrow_table()


def feature_matrix(table):
    """Arrow table/batch -> float64 matrix in FEATURES order, log1p applied. Nulls and invalid values -> NaN."""
    columns = []
    for name in FEATURES:
        values = pc.cast(table.column(name), pa.float64()).to_numpy(zero_copy_only=False)
        if name in LOG_FEATURES:
            values = np.log1p(np.where(values >= 0, values, np.nan))
        columns.append(values)
    return np.column_stack(columns)


def training_stats(X):
    """Per-feature median, min, max and robust scale on non-missing transformed training values."""
    stats = {"median": [], "min": [], "max": [], "scale": []}
    for j in range(X.shape[1]):
        values = X[~np.isnan(X[:, j]), j]
        if values.size == 0:
            for key in stats:
                stats[key].append(np.nan)
            continue
        median = np.median(values)
        mad_scale = 1.4826 * np.median(np.abs(values - median))
        q1, q3 = np.percentile(values, [25, 75])
        iqr_scale = (q3 - q1) / 1.349
        stats["median"].append(median)
        stats["min"].append(values.min())
        stats["max"].append(values.max())
        stats["scale"].append(mad_scale if mad_scale > 0 else (iqr_scale if iqr_scale > 0 else 0.0))
    return {key: np.array(value, dtype=float) for key, value in stats.items()}


def fit_bundle(sample):
    """Screen features, fit the median imputer and the forest on training rows only."""
    X_all = feature_matrix(sample)
    model_features, notes = [], []
    for j, name in enumerate(FEATURES):
        values = X_all[~np.isnan(X_all[:, j]), j]
        if values.size == 0:
            notes.append(f"{name} dropped: all values missing in the training sample")
        elif np.all(values == values[0]):
            notes.append(f"{name} dropped: constant ({values[0]:g}, transformed) in the training sample")
        else:
            model_features.append(name)
    if not model_features:
        raise SystemExit("Every feature is missing or constant in the training sample")
    columns = [FEATURES.index(name) for name in model_features]
    imputer = SimpleImputer(strategy="median").fit(X_all[:, columns])
    X_fit = imputer.transform(X_all[:, columns]).astype(np.float32)
    forest = IsolationForest(**config.FOREST_SETTINGS).fit(X_fit)
    ranks = np.column_stack([rankdata(X_fit[:, j]) for j in range(X_fit.shape[1])])
    days = np.unique(sample.column("window_start").to_numpy().astype("datetime64[D]"))
    return {
        "model": forest, "imputer": imputer, "feature_order": FEATURES, "log_features": LOG_FEATURES,
        "model_features": model_features, "model_columns": columns, "feature_notes": notes,
        "training_stats": training_stats(X_all), "spearman": np.corrcoef(ranks, rowvar=False),
        "train_sample_rows": int(sample.num_rows), "train_sample_days": int(days.size),
    }


def raw_scores(bundle, X_all):
    X = bundle["imputer"].transform(X_all[:, bundle["model_columns"]]).astype(np.float32)
    return -bundle["model"].score_samples(X)


def reference_sample(con, raw_glob, limit, seed):
    """Deterministic sample (or all) of streamed validation raw scores, sorted, as the frozen reference."""
    scores = con.execute(f"""
        SELECT raw_score FROM read_parquet({sql_text(raw_glob)})
        ORDER BY hash(src_ip, window_start, {int(seed)}), src_ip, window_start
        LIMIT {int(limit)}""").fetchnumpy()["raw_score"]
    return np.sort(np.asarray(scores, dtype=np.float64))


def calibrate(reference, validation_rows, validation_days):
    """Raw-score thresholds at validation quantiles, plus achieved counts and the workload table."""
    thresholds = {band: float(np.quantile(reference, q, method="higher"))
                  for band, q in config.REVIEW_QUANTILES.items()}
    bands = assign_bands(reference, thresholds)
    workload = []
    for q in config.WORKLOAD_QUANTILES:
        cutoff = float(np.quantile(reference, q, method="higher"))
        share = float(np.mean(reference >= cutoff))
        workload.append({"quantile": q, "raw_threshold": cutoff, "share_at_or_above": share,
                         "per_1000_host_windows": 1000 * share,
                         "expected_per_day": share * validation_rows / validation_days})
    return {"thresholds": thresholds,
            "reference_band_counts": {band: int(np.sum(bands == band)) for band in BANDS + [BELOW]},
            "workload": workload}


def assign_bands(raw, thresholds):
    """Highest band first. Tied thresholds can leave a band empty; counts are reported, not forced."""
    conditions = [raw >= thresholds[band] for band in BANDS]
    return np.select(conditions, BANDS, default=BELOW).astype(object)


def robust_deviations(X_all, stats):
    """Signed (x - median) / scale; undefined where scale is 0 or the value is missing."""
    median, scale = stats["median"], stats["scale"]
    present = ~np.isnan(X_all)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.where(scale > 0, (X_all - median) / scale, np.nan)
    constant_departure = present & (scale == 0) & ~np.isnan(median) & (X_all != median)
    beyond_range = present & ((X_all < stats["min"]) | (X_all > stats["max"]))
    return deviation, constant_departure, beyond_range


def original_units(name, value):
    return float(np.expm1(value)) if name in LOG_FEATURES else float(value)


def names_where(mask_row):
    return ",".join(FEATURES[j] for j in np.flatnonzero(mask_row)) or None


def score_batch(bundle, batch):
    """Score one Arrow batch of feature rows with the frozen bundle. Returns an Arrow table with a fixed schema."""
    X_all = feature_matrix(batch)
    raw = raw_scores(bundle, X_all)
    reference = bundle["reference_scores"]
    percentile = 100.0 * np.searchsorted(reference, raw, side="right") / len(reference)
    band = assign_bands(raw, bundle["thresholds"])
    deviation, constant_departure, beyond = robust_deviations(X_all, bundle["training_stats"])
    abs_filled = np.where(np.isnan(deviation), -1.0, np.abs(deviation))
    max_abs = abs_filled.max(axis=1)
    max_abs = np.where(max_abs < 0, np.nan, max_abs)
    large = max_abs >= bundle["large_deviation"]
    high = np.isin(band, ["Critical", "High"])
    deviation_only = large & beyond.any(axis=1) & ~high
    order = np.argsort(-abs_filled, axis=1, kind="stable")[:, :3]

    columns = {name: batch.column(name) for name in ["src_ip", "window_start", "window_end", "period"] + FEATURES}
    columns.update({
        "raw_score": pa.array(raw, pa.float64()),
        "reference_percentile": pa.array(percentile, pa.float64()),
        "band": pa.array(band.tolist(), pa.string()),
        "max_abs_deviation": pa.array(max_abs, pa.float64(), from_pandas=True),
        "large_deviation": pa.array(large),
        "priority_candidate": pa.array(high & large),
        "deviation_only": pa.array(deviation_only),
        "selected": pa.array(high | deviation_only),
    })
    for k in range(3):
        rank_cols = order[:, k]
        values = np.take_along_axis(deviation, order[:, k:k + 1], axis=1)[:, 0]
        names = [FEATURES[c] if not np.isnan(v) else None for c, v in zip(rank_cols, values)]
        columns[f"dev{k + 1}_feature"] = pa.array(names, pa.string())
        columns[f"dev{k + 1}_value"] = pa.array(values, pa.float64(), from_pandas=True)
    columns["beyond_training_range"] = pa.array([names_where(row) for row in beyond], pa.string())
    columns["constant_reference_departure"] = pa.array([names_where(row) for row in constant_departure], pa.string())
    columns["deviation_context"] = pa.array(
        [deviation_text(bundle, X_all[i], deviation[i], order[i]) if high[i] or deviation_only[i] else None
         for i in range(len(raw))], pa.string())
    return pa.table(columns)


def deviation_text(bundle, x_row, deviation_row, order_row):
    """'feature=value (train median m, d=+z)' for the three largest defined deviations, in original units."""
    parts = []
    for j in order_row:
        if np.isnan(deviation_row[j]):
            continue
        name = FEATURES[j]
        value = original_units(name, x_row[j])
        median = original_units(name, bundle["training_stats"]["median"][j])
        parts.append(f"{name}={value:.4g} (train median {median:.4g}, d={deviation_row[j]:+.1f})")
    return "; ".join(parts) or None
