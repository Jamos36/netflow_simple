from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def fit(x, settings: dict, seed: int):
    if x.shape[0] < 2:
        raise ValueError("GMM needs at least two training rows")
    fit_rows = min(x.shape[0], int(settings.get("fit_rows", 10_000)))
    if fit_rows < x.shape[0]:
        rng = np.random.default_rng(seed)
        selected = rng.choice(x.shape[0], size=fit_rows, replace=False)
        x = x[selected]
    components = max(1, min(int(settings["components"]), x.shape[0] // 10, x.shape[1] * 2))
    model = GaussianMixture(n_components=components, covariance_type="diag", reg_covar=1e-4,
                            max_iter=int(settings["max_iter"]), random_state=int(seed),
                            init_params="k-means++", n_init=1, verbose=0)
    model.fit(x)
    return model


def score(model: GaussianMixture, x):
    return -model.score_samples(x)
