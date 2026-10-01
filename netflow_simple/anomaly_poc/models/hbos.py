from __future__ import annotations

import numpy as np


class HBOS:
    """Simple histogram-based outlier score with smoothed per-feature densities."""

    def __init__(self, bins: int = 20):
        self.bins = int(bins)
        self.edges: list[np.ndarray] = []
        self.probabilities: list[np.ndarray] = []

    def fit(self, x):
        self.edges.clear()
        self.probabilities.clear()
        for j in range(x.shape[1]):
            column = np.asarray(x[:, j], dtype=np.float64)
            values = column[np.isfinite(column)]
            if values.size == 0 or np.min(values) == np.max(values):
                self.edges.append(np.array([-np.inf, np.inf]))
                self.probabilities.append(np.array([1.0]))
                continue
            # Equal-width bins preserve density differences; quantile bins make most
            # training observations equally rare and can flatten the score.
            lo, hi = np.quantile(values, [0.005, 0.995])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(values)), float(np.max(values))
            edges = np.linspace(lo, hi, self.bins + 1)
            clipped = np.clip(values, edges[0], edges[-1])
            counts, _ = np.histogram(clipped, bins=edges)
            probabilities = (counts + 1.0) / (counts.sum() + len(counts))
            self.edges.append(edges)
            self.probabilities.append(probabilities)
        return self

    def score_samples(self, x):
        total = np.zeros(x.shape[0], dtype=np.float64)
        for j, (edges, probs) in enumerate(zip(self.edges, self.probabilities)):
            if len(probs) == 1:
                continue
            idx = np.searchsorted(edges, x[:, j], side="right") - 1
            idx = np.clip(idx, 0, len(probs) - 1)
            total += -np.log(probs[idx])
        return total


def fit(x, settings: dict, seed: int):
    return HBOS(int(settings["bins"])).fit(x)


def score(model: HBOS, x):
    return model.score_samples(x)
