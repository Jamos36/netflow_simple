import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


def incidents(frame, gap_minutes):
    flagged = frame.loc[frame.alert].sort_values(["host", "time"]).copy()
    columns = ["host", "start", "end", "peak_score", "observations"]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    delta = flagged.groupby("host").time.diff().dt.total_seconds() / 60
    flagged["group"] = ((delta > gap_minutes) | delta.isna()).cumsum()
    return (
        flagged.groupby(["host", "group"])
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            peak_score=("score", "max"),
            observations=("score", "size"),
        )
        .reset_index()
        .drop(columns="group")
    )


def threshold_table(frame, quantiles, gap, budget):
    rows = []
    days = max(
        1, (frame.time.max().normalize() - frame.time.min().normalize()).days + 1
    )
    for q in quantiles:
        threshold = float(frame.score.quantile(q))
        view = frame.assign(alert=frame.score > threshold)
        count = len(incidents(view, gap))
        rows.append(
            dict(
                quantile=q,
                threshold=threshold,
                flagged_rate=float(view.alert.mean()),
                incidents=count,
                incidents_per_day=count / days,
            )
        )
    table = pd.DataFrame(rows)
    feasible = table[table.incidents_per_day <= budget]
    selected = feasible.iloc[0] if len(feasible) else table.iloc[-1]
    return table, selected.to_dict()


def event_metrics(frame, events, window_minutes, complete):
    truth = np.zeros(len(frame), dtype=bool)
    rows = []
    for e in events.itertuples():
        start, end = pd.to_datetime(e.start, utc=True), pd.to_datetime(e.end, utc=True)
        host = (
            np.ones(len(frame), bool)
            if e.host == "*"
            else frame.host.to_numpy() == str(e.host)
        )
        overlap = (
            host
            & (frame.time < end).to_numpy()
            & ((frame.time + pd.Timedelta(minutes=window_minutes)) > start).to_numpy()
        )
        truth |= overlap
        detected = frame.loc[overlap & frame.alert.to_numpy(), "time"]
        represented = bool(overlap.any())
        rows.append(
            dict(
                event_id=e.event_id,
                host=e.host,
                represented=represented,
                detected=bool(len(detected)),
                delay_seconds=max(0, (detected.min() - start).total_seconds())
                if len(detected)
                else np.nan,
            )
        )
    detail = pd.DataFrame(rows)
    result = {
        "known_events": len(rows),
        "represented_events": int(detail.represented.sum()),
        "event_recall": float(detail.detected.mean()) if rows else None,
    }
    # All supplied events count in recall, even when telemetry has no matching row.
    if complete and truth.any() and (~truth).any():
        p, r, f, _ = precision_recall_fscore_support(
            truth, frame.alert, average="binary", zero_division=0
        )
        result.update(
            precision=float(p),
            recall=float(r),
            f1=float(f),
            pr_auc=float(average_precision_score(truth, frame.score)),
            roc_auc=float(roc_auc_score(truth, frame.score)),
            false_positive_rate=float(frame.loc[~truth, "alert"].mean()),
        )
    return result, detail, truth


def population_stats(frame, gap):
    inc = incidents(frame, gap)
    span = max(
        1, (frame.time.max().normalize() - frame.time.min().normalize()).days + 1
    )
    return dict(
        rows=len(frame),
        flagged_rate=float(frame.alert.mean()),
        affected_hosts=int(frame.loc[frame.alert, "host"].nunique()),
        incidents=len(inc),
        incidents_per_day=len(inc) / span,
        median_incident_span_seconds=float(
            (inc.end - inc.start).dt.total_seconds().median()
        )
        if len(inc)
        else 0,
        score_p50=float(frame.score.quantile(0.5)),
        score_p95=float(frame.score.quantile(0.95)),
        score_p99=float(frame.score.quantile(0.99)),
    )


def psi(reference, current):
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    a, b = (
        np.histogram(reference.dropna(), edges)[0],
        np.histogram(current.dropna(), edges)[0],
    )
    a, b = np.append(a, reference.isna().sum()), np.append(b, current.isna().sum())
    a, b = (a + 0.5) / (a.sum() + 0.5 * len(a)), (b + 0.5) / (b.sum() + 0.5 * len(b))
    return float(np.sum((b - a) * np.log(b / a)))
