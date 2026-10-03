from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .evaluation import incidents, psi


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plots(frame, out, cfg, feature_sample):
    out = Path(out)
    # Daily plot is additional context, never the only detection view.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for split, part in frame.groupby("split", observed=True):
        daily = part.set_index("time").score.resample("D").quantile(0.95)
        axes[0].plot(daily.index, daily, label=split)
        axes[1].hist(part.score, bins=60, alpha=0.4, density=True, label=split)
    axes[0].set(title="Daily p95: distribution drift context", ylabel="Raw score")
    axes[1].set(title="Score distributions", xlabel="Raw score")
    for ax in axes:
        ax.legend()
    finish(fig, out / "distributions.png")
    test = frame[frame.split == "test"].copy()
    if test.empty:
        return
    hourly = (
        test.set_index("time")
        .resample("h")
        .agg(flagged=("alert", "sum"), rows=("alert", "size"))
    )
    inc = incidents(test, cfg["review"]["incident_gap_minutes"])
    incident_counts = (
        inc.set_index("start").resample("h").size()
        if len(inc)
        else pd.Series(dtype=float)
    )
    hosts = test[test.alert].set_index("time").resample("h").host.nunique()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly.index, hourly.flagged, label="Flagged observations")
    ax.plot(incident_counts.index, incident_counts, label="New incidents")
    ax.plot(hosts.index, hosts, label="Affected hosts")
    ax.set(title="Hourly alert counts", ylabel="Count")
    ax.legend()
    context = cfg.get("context", {})
    if context.get("pentest_start") and context.get("pentest_end"):
        ax.axvspan(
            pd.to_datetime(context["pentest_start"], utc=True),
            pd.to_datetime(context["pentest_end"], utc=True),
            alpha=0.15,
            color="red",
            label="Context only",
        )
    finish(fig, out / "hourly_counts.png")
    hourly["rate"] = hourly.flagged / hourly.rows.replace(0, np.nan)
    hourly.to_csv(out / "hourly_stats.csv")
    weekly = (
        test.set_index("time")
        .resample("7D")
        .agg(
            rows=("score", "size"),
            flagged=("alert", "sum"),
            score_p99=("score", lambda x: x.quantile(0.99)),
        )
    )
    weekly["flagged_rate"] = weekly.flagged / weekly.rows
    weekly.to_csv(out / "weekly_stats.csv")
    # Plot only the most active alert hosts, binning to hour for display.
    selected = test.groupby("host").percentile.max().nlargest(20).index
    grid = (
        test[test.host.isin(selected)]
        .assign(hour=lambda x: x.time.dt.floor("h"))
        .pivot_table(index="host", columns="hour", values="percentile", aggfunc="max")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(grid.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_yticks(range(len(grid)), grid.index)
    ticks = np.unique(
        np.linspace(0, len(grid.columns) - 1, min(6, len(grid.columns))).astype(int)
    )
    ax.set_xticks(ticks, [str(grid.columns[i])[:16] for i in ticks], rotation=20)
    ax.set_title(
        "Top-host hourly maximum calibrated rank; missing cells are unobserved"
    )
    fig.colorbar(image, ax=ax, label="Calibration percentile")
    finish(fig, out / "host_heatmap.png")
    peak = test.loc[test.score.idxmax()]
    local = test[
        (test.host == peak.host) & (abs(test.time - peak.time) <= pd.Timedelta(hours=3))
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(local.time, local.score, marker=".", label="Observation scores")
    ax.axhline(
        float(peak.threshold), linestyle="--", color="red", label="Frozen cutoff"
    )
    ax.set_title(f"Local scores: {peak.host}")
    ax.legend()
    finish(fig, out / "local_scores.png")
    # Feature traces are context; deviations are not causal model attribution.
    if feature_sample is not None:
        local = feature_sample[
            (feature_sample.host == peak.host)
            & (abs(feature_sample.time - peak.time) <= pd.Timedelta(hours=3))
        ]
        names = [
            n
            for n in [
                "flows",
                "bytes",
                "unique_ips",
                "new_ip_rate",
                "bytes_robust_z",
                "interarrival_cv",
            ]
            if n in local
        ]
        fig, axes = plt.subplots(
            len(names), 1, figsize=(12, max(4, len(names) * 1.5)), squeeze=False
        )
        for ax, name in zip(axes.ravel(), names):
            ax.plot(local.time, local[name], marker=".")
            ax.set_ylabel(name)
        finish(fig, out / "feature_traces.png")


def diagnostics(train, test, names, out):
    rows = []
    for name in names:
        x = train[name]
        rows.append(
            dict(
                feature=name,
                missing_rate=float(x.isna().mean()),
                unique=int(x.nunique()),
                median=float(x.median()),
                p99=float(x.quantile(0.99)),
                test_psi=psi(x, test[name]),
            )
        )
    pd.DataFrame(rows).to_csv(out / "feature_diagnostics.csv", index=False)
    corr = train[names].corr(method="spearman")
    corr.to_csv(out / "spearman.csv")
    fig, ax = plt.subplots(
        figsize=(max(8, len(names) * 0.45), max(6, len(names) * 0.4))
    )
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax)
    finish(fig, out / "correlations.png")
    pairs = [
        dict(first=a, second=b, spearman=float(corr.loc[a, b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if abs(corr.loc[a, b]) >= 0.98
    ]
    pd.DataFrame(pairs, columns=["first", "second", "spearman"]).to_csv(
        out / "redundant_pairs.csv", index=False
    )
    # Plot signed-log distributions to make heavy-tailed raw features readable.
    fig, axes = plt.subplots(
        int(np.ceil(len(names) / 3)),
        3,
        figsize=(12, max(4, np.ceil(len(names) / 3) * 2.5)),
        squeeze=False,
    )
    for ax, name in zip(axes.ravel(), names):
        for label, data in [("train", train), ("test", test)]:
            values = data[name].dropna().to_numpy()
            if len(values):
                ax.hist(
                    np.sign(values) * np.log1p(np.abs(values)),
                    bins=30,
                    density=True,
                    alpha=0.4,
                    label=label,
                )
        ax.set_title(name, fontsize=9)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    axes.ravel()[0].legend()
    fig.suptitle("Feature distributions (signed log1p; bounded split samples)")
    finish(fig, out / "feature_distributions.png")
    drift = []
    for day, part in test.groupby(test.time.dt.floor("D")):
        for name in names:
            drift.append(
                dict(day=str(day), feature=name, psi=psi(train[name], part[name]))
            )
    pd.DataFrame(drift).to_csv(out / "daily_feature_psi.csv", index=False)
