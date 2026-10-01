from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit(x, settings: dict, seed: int):
    if x.shape[0] < 2:
        raise ValueError("PCA needs at least two training rows")
    n_components = max(1, min(int(settings["max_components"]), x.shape[1], x.shape[0] - 1))
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    model.fit(x)
    target = float(settings["explained_variance"])
    cumulative = np.cumsum(model.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, target, side="left") + 1)
    keep = max(1, min(keep, model.components_.shape[0]))
    model.components_ = model.components_[:keep]
    model.explained_variance_ = model.explained_variance_[:keep]
    model.explained_variance_ratio_ = model.explained_variance_ratio_[:keep]
    model.singular_values_ = model.singular_values_[:keep]
    model.n_components_ = keep
    return model


def score(model, x):
    reconstructed = model.inverse_transform(model.transform(x))
    return np.mean(np.square(x - reconstructed), axis=1)
