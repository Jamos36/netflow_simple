import json
import joblib
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .features import FAMILIES, FLOW_FEATURES, literal, export
from .models import Transform, Detector
from .evaluation import threshold_table, incidents, population_stats, event_metrics
from .reporting import plots, diagnostics

ALL_MODELS = ["iforest", "pca", "hbos", "gmm", "robust_pca", "autoencoder"]


def boundaries(cfg, path):
    con = duckdb.connect()
    first, last = con.execute(
        f"SELECT min(window_start),max(window_start) FROM read_parquet({literal(path)})"
    ).fetchone()
    first, last = pd.Timestamp(first).normalize(), pd.Timestamp(last).normalize()
    n = (last - first).days + 1
    supplied = cfg["split"]
    ends = [
        pd.to_datetime(supplied.get(key), utc=True)
        if supplied.get(key)
        else first + pd.Timedelta(days=max(1, int(n * f)))
        for key, f in zip(
            ["train_end", "validation_end", "calibration_end"], [0.5, 0.7, 0.85]
        )
    ]
    if not first < ends[0] < ends[1] < ends[2] <= last:
        raise ValueError(
            "Need nonempty train/validation/calibration/test dates; set explicit boundaries"
        )
    return ends


def split_frame(frame, ends):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["split"] = np.select(
        [frame.time < ends[0], frame.time < ends[1], frame.time < ends[2]],
        ["train", "validation", "calibration"],
        default="test",
    )
    return frame


def sample(path, time_col, ends, cfg):
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW data AS SELECT *,{time_col} AS time FROM read_parquet({literal(path)})"
    )
    result = []
    limit = int(cfg["data"]["fit_rows"])
    for lower, upper in [
        (None, ends[0]),
        (ends[0], ends[1]),
        (ends[1], ends[2]),
        (ends[2], None),
    ]:
        conditions = []
        if lower is not None:
            conditions.append(f"time>={literal(lower)}::TIMESTAMPTZ")
        if upper is not None:
            conditions.append(f"time<{literal(upper)}::TIMESTAMPTZ")
        where = " AND ".join(conditions)
        frame = con.execute(
            f"SELECT * FROM data WHERE {where} ORDER BY hash(host,time,bytes) LIMIT {limit}"
        ).df()
        if frame.empty:
            raise ValueError("An evaluation split has no rows")
        result.append(split_frame(frame, ends))
    con.close()
    return result


def select_features(train, names, cfg, out):
    kept = [
        n
        for n in names
        if n not in cfg["features"].get("drop", [])
        and train[n].notna().any()
        and train[n].nunique() > 1
    ]
    removed = {
        n: "explicit drop, all missing, or constant in training"
        for n in names
        if n not in kept
    }
    corr = train[kept].corr(method="spearman").abs()
    if cfg["features"].get("prune_correlated", False):
        chosen = []
        for n in kept:
            if any(
                corr.loc[n, k] >= cfg["features"]["correlation_cutoff"] for k in chosen
            ):
                removed[n] = "correlated with earlier retained feature"
            else:
                chosen.append(n)
        kept = chosen
    if len(kept) < 2:
        raise ValueError("Need at least two variable training features")
    (out / "feature_selection.json").write_text(
        json.dumps({"kept": kept, "removed": removed}, indent=2)
    )
    return kept


def run_level(cfg, feature_dir, out, level, names, model_names, ends, make_plots=True):
    out.mkdir(parents=True, exist_ok=True)
    path = feature_dir / ("windows.parquet" if level == "window" else "flows.parquet")
    time_col = "window_start" if level == "window" else "time"
    train, val, cal, test = sample(path, time_col, ends, cfg)
    names = select_features(train, names, cfg, out)
    transform = Transform().fit(train, names)
    x, v, c = map(transform.apply, [train, val, cal])
    diagnostics(train, test, names, out)
    summaries = []
    references = {}
    for name in model_names:
        print(f"{out.name}/{level}: {name}", flush=True)
        folder = out / name
        folder.mkdir(exist_ok=True)
        settings = dict(cfg["models"])
        fit_x = x[: settings.get("rpca_fit_rows", 2000)] if name == "robust_pca" else x
        if name == "pca":
            # Select by train/validation tail-rate consistency without test labels.
            choices = []
            for rank in sorted(
                set(
                    min(int(r), x.shape[1] - 1)
                    for r in settings.get("pca_ranks", [2, 4, 8])
                )
            ):
                local = dict(settings, components_pca=max(1, rank))
                d = Detector(name, local, settings["seed"]).fit(fit_x, v)
                cutoff = np.quantile(d.score(x), 0.995)
                # Quantile discrepancy across train and validation is a drift diagnostic,
                # not evidence of attack detection. Freeze selection before calibration.
                discrepancy = abs(float(np.mean(d.score(v) > cutoff)) - 0.005)
                choices.append((discrepancy, rank, d))
            _, _, detector = min(choices, key=lambda z: (z[0], z[1]))
            pd.DataFrame(
                [{"rank": r, "validation_rate_discrepancy": s} for s, r, _ in choices]
            ).to_csv(folder / "rank_selection.csv", index=False)
        else:
            detector = Detector(name, settings, settings["seed"]).fit(fit_x, v)
        reference = np.sort(detector.score(c))
        cal_frame = cal[["host", "time"]].assign(score=detector.score(c))
        table, selected = threshold_table(
            cal_frame,
            cfg["review"]["quantiles"],
            cfg["review"]["incident_gap_minutes"],
            cfg["review"]["incidents_per_day"],
        )
        threshold = selected["threshold"]
        table.to_csv(folder / "threshold_sweep_sample.csv", index=False)
        # Stream all observations; cutoff candidates come from a bounded calibration sample.
        writer = None
        top = []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=cfg["data"]["batch_rows"]
        ):
            raw = batch.to_pandas()
            raw["time"] = pd.to_datetime(raw[time_col], utc=True)
            raw = split_frame(raw, ends)
            z = transform.apply(raw)
            scores = detector.score(z)
            metadata = ["host", "time", "split"] + (
                ["source_file", "source_row", "dst", "port"] if level == "flow" else []
            )
            scored = raw[metadata].copy()
            scored["score"] = scores
            scored["percentile"] = np.searchsorted(
                reference, scores, side="right"
            ) / len(reference)
            scored["alert"] = scores > threshold
            scored["threshold"] = threshold
            arrow = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    folder / "scores.parquet", arrow.schema, compression="zstd"
                )
            writer.write_table(arrow)
            candidates = raw.loc[raw.split == "test"].copy()
            if len(candidates):
                candidates["score"] = scores[raw.split == "test"]
                candidates = candidates.nlargest(cfg["review"]["top_windows"], "score")
                top.append(candidates)
                top = [pd.concat(top).nlargest(cfg["review"]["top_windows"], "score")]
        writer.close()
        db = duckdb.connect()
        db.execute(
            f"CREATE VIEW scored AS SELECT * FROM read_parquet({literal(folder / 'scores.parquet')})"
        )
        # Budget counts use ALL calibration observations, not the bounded fit sample.
        exact = []
        gap = cfg["review"]["incident_gap_minutes"]
        span = db.execute(
            "SELECT greatest(1,date_diff('day',min(time),max(time))+1) FROM scored WHERE split='calibration'"
        ).fetchone()[0]
        for row in table.itertuples():
            count = db.execute(f"""WITH flagged AS (SELECT host,time,lag(time) OVER(PARTITION BY host ORDER BY time) previous
                FROM scored WHERE split='calibration' AND score>{row.threshold})
                SELECT count(*) FROM flagged WHERE previous IS NULL OR epoch(time-previous)>{gap * 60}""").fetchone()[
                0
            ]
            exact.append(
                dict(
                    quantile=row.quantile,
                    threshold=row.threshold,
                    incidents=count,
                    incidents_per_day=count / span,
                )
            )
        exact = pd.DataFrame(exact)
        feasible = exact[exact.incidents_per_day <= cfg["review"]["incidents_per_day"]]
        selected = (feasible.iloc[0] if len(feasible) else exact.iloc[-1]).to_dict()
        threshold = float(selected["threshold"])
        exact.to_csv(folder / "threshold_sweep.csv", index=False)
        # Rewrite flags in SQL if full calibration changes the selected cutoff.
        export(
            db,
            f"SELECT * EXCLUDE(alert,threshold),score>{threshold} alert,{threshold} threshold FROM scored",
            folder / "calibrated.parquet",
        )
        db.close()
        (folder / "scores.parquet").unlink()
        (folder / "calibrated.parquet").rename(folder / "scores.parquet")
        db = duckdb.connect()
        # Flow plotting/evaluation uses minute maxima to avoid loading every raw flow.
        if level == "flow":
            report = db.execute(f"""SELECT host,time_bucket(INTERVAL '{cfg["features"]["window_minutes"]} minutes',time) AS time,split,
                max(score) score,max(percentile) percentile,bool_or(alert) alert,max(threshold) threshold
                FROM read_parquet({literal(folder / "scores.parquet")}) GROUP BY ALL""").df()
        else:
            report = db.execute(
                f"SELECT * FROM read_parquet({literal(folder / 'scores.parquet')})"
            ).df()
        db.close()
        report["time"] = pd.to_datetime(report.time, utc=True)
        inc = incidents(report[report.split == "test"], gap)
        inc.to_csv(folder / "incidents.csv", index=False)
        candidates = top[0]
        candidates["percentile"] = np.searchsorted(
            reference, candidates.score, side="right"
        ) / len(reference)
        candidates["alert"] = candidates.score > threshold
        z = transform.apply(candidates)
        context = detector.explain(z)
        candidates["explanation_kind"] = (
            "reconstruction residual"
            if name in {"pca", "robust_pca", "autoencoder"}
            else "standardized deviation; not attribution"
        )
        candidates["top_features"] = [
            ", ".join(np.array(transform.output_names)[np.argsort(row)[-3:][::-1]])
            for row in context
        ]
        candidates.to_csv(folder / "candidates.csv", index=False)
        pd.DataFrame(context, columns=transform.output_names).head(
            cfg["review"]["explained_candidates"]
        ).to_csv(folder / "candidate_feature_contributions.csv", index=False)
        metrics = population_stats(report[report.split == "test"], gap)
        metrics.update(
            model=name,
            level=level,
            features=len(names),
            threshold=threshold,
            calibration_incidents_per_day=selected["incidents_per_day"],
            budget_met=selected["incidents_per_day"]
            <= cfg["review"]["incidents_per_day"],
            report_unit="host-window"
            if level == "window"
            else "host-window with maximum flow score",
        )
        labels = cfg.get("labels", {})
        if labels.get("events_csv"):
            events = pd.read_csv(labels["events_csv"], dtype={"host": str})
            start, end = (
                ends[2],
                report.time.max()
                + pd.Timedelta(minutes=cfg["features"]["window_minutes"]),
            )
            events = events[
                (pd.to_datetime(events.end, utc=True) > start)
                & (pd.to_datetime(events.start, utc=True) < end)
            ]
            if len(events):
                scores, detail, _ = event_metrics(
                    report[report.split == "test"],
                    events,
                    cfg["features"]["window_minutes"],
                    labels.get("complete", False),
                )
                metrics.update(scores)
                detail.to_csv(folder / "event_detection.csv", index=False)
        if labels.get("reviews_csv"):
            reviews = pd.read_csv(labels["reviews_csv"])
            reviews = reviews[(reviews.model == name) & (reviews.level == level)].copy()
            reviews["time"] = pd.to_datetime(reviews.time, utc=True)
            reviewed = candidates.merge(
                reviews[["host", "time", "malicious"]], on=["host", "time"]
            )
            metrics["reviewed_candidates"] = len(reviewed)
            metrics["reviewed_precision"] = (
                float(reviewed.malicious.mean()) if len(reviewed) else None
            )
        if name in {"pca", "robust_pca"}:
            metrics["pca_rank"] = detector.model.n_components_
            metrics["explained_variance"] = float(
                detector.model.explained_variance_ratio_.sum()
            )
        if name == "robust_pca":
            metrics["rpca_training_relative_error"] = detector.rpca_relative_error
        if name == "autoencoder":
            metrics.update(
                epochs=detector.epochs, validation_loss=detector.validation_loss
            )
        if name == "gmm":
            metrics.update(
                components=detector.model.n_components,
                covariance=detector.model.covariance_type,
                converged=bool(detector.model.converged_),
            )
        # Seed stability is ranking agreement, not evidence of correctness.
        if name == "iforest":
            alternate = Detector(name, settings, settings["seed"] + 1).fit(x, v)
            a = np.argsort(detector.score(transform.apply(test)))[
                -min(100, len(test)) :
            ]
            b = np.argsort(alternate.score(transform.apply(test)))[
                -min(100, len(test)) :
            ]
            metrics["seed_top100_jaccard"] = len(set(a) & set(b)) / len(set(a) | set(b))
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        joblib.dump(
            dict(
                model=detector,
                transform=transform,
                threshold=threshold,
                calibration_scores=reference,
                window_minutes=cfg["features"]["window_minutes"],
                level=level,
            ),
            folder / "model.joblib",
        )
        if make_plots:
            window_sample = None
            if level == "window":
                # Load traces only for the peak test host, not all feature rows.
                peak = report[report.split == "test"].nlargest(1, "score").iloc[0]
                con = duckdb.connect()
                window_sample = con.execute(
                    f"SELECT *,window_start AS time FROM read_parquet({literal(path)}) WHERE host={literal(peak.host)}"
                ).df()
                con.close()
                window_sample["time"] = pd.to_datetime(window_sample.time, utc=True)
            plots(report, folder, cfg, window_sample)
        references[name] = set(zip(candidates.host, candidates.time))
        summaries.append(metrics)
    overlap = pd.DataFrame(
        {
            a: {
                b: len(references[a] & references[b])
                / max(1, len(references[a] | references[b]))
                for b in model_names
            }
            for a in model_names
        }
    )
    overlap.to_csv(out / "candidate_overlap.csv")
    pd.DataFrame(summaries).to_csv(out / "model_comparison.csv", index=False)
    return summaries


def run(cfg, feature_dir, out, ablations=False):
    ends = boundaries(cfg, feature_dir / "windows.parquet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            dict(config=cfg, boundaries=[str(e) for e in ends]), indent=2, default=str
        )
    )
    families = cfg["features"]["families"]
    names = [n for family in families for n in FAMILIES[family]]
    if cfg["features"].get("calendar_features"):
        names += ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    models = cfg["models"]["names"]
    if models == ["all"]:
        models = ALL_MODELS
    summaries = run_level(
        cfg, feature_dir, out / "window", "window", names, models, ends
    )
    if cfg.get("flow_models"):
        run_level(
            cfg,
            feature_dir,
            out / "flow",
            "flow",
            FLOW_FEATURES,
            cfg["flow_models"],
            ends,
        )
    if ablations:
        rows = []
        current = []
        for family in families:
            current += FAMILIES[family]
            result = run_level(
                cfg,
                feature_dir,
                out / "ablations" / family,
                "window",
                current,
                ["iforest"],
                ends,
                False,
            )
            rows.extend([dict(added_family=family, **r) for r in result])
        pd.DataFrame(rows).to_csv(out / "ablation_comparison.csv", index=False)
    return summaries
