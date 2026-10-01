from __future__ import annotations

from sklearn.ensemble import IsolationForest


def fit(x, settings: dict, seed: int):
    model = IsolationForest(n_estimators=int(settings["n_estimators"]),
                            max_samples=settings["max_samples"],
                            max_features=float(settings["max_features"]),
                            contamination="auto", n_jobs=int(settings["n_jobs"]),
                            random_state=int(seed), bootstrap=False)
    model.fit(x)
    return model


def score(model, x):
    # sklearn assigns lower scores to outliers; invert so high always means unusual.
    return -model.score_samples(x)
