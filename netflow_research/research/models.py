"""Common score convention: larger means more unusual. CPU only."""

import copy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor


class Transform:
    def fit(self, frame, names):
        self.names = names
        self.log = [
            n
            for n in names
            if n
            in {
                "flows",
                "bytes",
                "packets",
                "unique_ips",
                "unique_ports",
                "median_duration",
                "median_bytes",
                "duration",
                "bytes_per_packet",
            }
        ]
        x = self.raw(frame)
        self.median = np.nanmedian(x, axis=0)
        self.scale = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        self.scale = np.maximum(self.scale, 0.1)
        self.indicators = np.flatnonzero(np.isnan(x).any(axis=0))
        self.output_names = names + [names[i] + "__missing" for i in self.indicators]
        return self

    def raw(self, frame):
        x = frame[self.names].to_numpy(dtype=float).copy()
        for n in self.log:
            i = self.names.index(n)
            x[:, i] = np.log1p(np.maximum(x[:, i], 0))
        return x

    def apply(self, frame):
        x = self.raw(frame)
        missing = np.isnan(x)
        x = (np.where(missing, self.median, x) - self.median) / self.scale
        return np.column_stack([x, missing[:, self.indicators].astype(float)])


class Detector:
    def __init__(self, name, settings, seed=42):
        self.name, self.settings, self.seed = name, settings, seed

    def fit(self, x, validation=None):
        d = x.shape[1]
        self.center = x.mean(axis=0)
        if self.name == "iforest":
            self.model = IsolationForest(
                n_estimators=self.settings.get("trees", 200),
                max_samples=self.settings.get("max_samples", 512),
                n_jobs=4,
                random_state=self.seed,
            ).fit(x)
        elif self.name == "gmm":
            choices = []
            for k in self.settings.get("components", [1, 2, 4]):
                for covariance in self.settings.get("covariance", ["diag", "full"]):
                    if len(x) > k:
                        model = GaussianMixture(
                            k,
                            covariance_type=covariance,
                            reg_covar=1e-3,
                            n_init=2,
                            max_iter=100,
                            random_state=self.seed,
                        ).fit(x)
                        choices.append((model.bic(x), model))
            self.model = min(choices, key=lambda z: z[0])[1]
        elif self.name == "hbos":
            self.hist = []
            for j in range(d):
                if np.ptp(x[:, j]) < 1e-10:
                    self.hist.append(None)
                    continue
                count, edges = np.histogram(x[:, j], bins=self.settings.get("bins", 20))
                prob = (count + 1) / (len(x) + len(count) + 2)
                self.hist.append(
                    (
                        edges,
                        prob,
                        1 / (len(x) + len(count) + 2),
                        max(np.std(x[:, j]), 0.1),
                    )
                )
        elif self.name in {"pca", "robust_pca"}:
            clean = x
            if self.name == "robust_pca":
                # Principal Component Pursuit on TRAINING ONLY. New observations use
                # the frozen learned basis and robust sparse-residual projection.
                centered = x - self.center
                lam = 1 / np.sqrt(max(centered.shape))
                mu = 1.25 / max(np.linalg.norm(centered, 2), 1e-8)
                low, sparse, y = (
                    np.zeros_like(centered),
                    np.zeros_like(centered),
                    np.zeros_like(centered),
                )
                norm = max(np.linalg.norm(centered), 1e-8)
                for iteration in range(self.settings.get("rpca_iterations", 80)):
                    u, s, v = np.linalg.svd(
                        centered - sparse + y / mu, full_matrices=False
                    )
                    low = (u * np.maximum(s - 1 / mu, 0)) @ v
                    sparse = shrink(centered - low + y / mu, lam / mu)
                    residual = centered - low - sparse
                    y += mu * residual
                    if np.linalg.norm(residual) / norm < 1e-6:
                        break
                    mu *= 1.2
                clean = low + self.center
                self.rpca_relative_error = float(np.linalg.norm(residual) / norm)
            rank = max(
                1,
                min(
                    self.settings.get("components_pca", max(1, d // 2)),
                    d - 1,
                    len(x) - 1,
                ),
            )
            self.model = PCA(n_components=rank, random_state=self.seed).fit(clean)
            residual, distance = self.parts(x)
            self.part_scale = np.maximum(
                np.quantile(np.column_stack([residual, distance]), 0.95, axis=0), 1e-8
            )
        elif self.name == "autoencoder":
            bottleneck = max(1, min(self.settings.get("bottleneck", 4), d - 1))
            model = MLPRegressor(
                hidden_layer_sizes=(max(8, d), bottleneck, max(8, d)),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=min(256, len(x)),
                random_state=self.seed,
            )
            best, best_loss, stale = None, np.inf, 0
            for epoch in range(self.settings.get("epochs", 60)):
                model.partial_fit(x, x)
                loss = float(np.mean((validation - model.predict(validation)) ** 2))
                if loss < best_loss - 1e-5:
                    best, best_loss, stale = copy.deepcopy(model), loss, 0
                else:
                    stale += 1
                if stale >= self.settings.get("patience", 8):
                    break
            self.model, self.validation_loss, self.epochs = best, best_loss, epoch + 1
        else:
            raise ValueError(self.name)
        return self

    def reconstruct(self, x):
        if self.name == "autoencoder":
            return self.model.predict(x)
        if self.name == "robust_pca":
            sparse = np.zeros_like(x)
            for _ in range(8):
                projected = self.model.inverse_transform(
                    self.model.transform(x - sparse)
                )
                sparse = shrink(x - projected, self.settings.get("sparse_penalty", 0.5))
            return projected
        return self.model.inverse_transform(self.model.transform(x))

    def parts(self, x):
        residual = np.mean((x - self.reconstruct(x)) ** 2, axis=1)
        pc = self.model.transform(x)
        distance = np.mean(
            pc**2 / np.maximum(self.model.explained_variance_, 1e-8), axis=1
        )
        return residual, distance

    def score(self, x):
        if self.name in {"iforest", "gmm"}:
            return -self.model.score_samples(x)
        if self.name in {"pca", "robust_pca"}:
            return np.max(np.column_stack(self.parts(x)) / self.part_scale, axis=1)
        if self.name == "autoencoder":
            return np.mean((x - self.reconstruct(x)) ** 2, axis=1)
        total = np.zeros(len(x))
        for j, hist in enumerate(self.hist):
            if hist is None:
                total += np.abs(x[:, j] - self.center[j])
                continue
            edges, prob, tail, scale = hist
            idx = np.clip(
                np.searchsorted(edges, x[:, j], side="right") - 1, 0, len(prob) - 1
            )
            outside = (x[:, j] < edges[0]) | (x[:, j] > edges[-1])
            distance = np.maximum(edges[0] - x[:, j], 0) + np.maximum(
                x[:, j] - edges[-1], 0
            )
            total += np.where(
                outside, -np.log(tail) + distance / scale, -np.log(prob[idx])
            )
        return total

    def explain(self, x):
        # Reconstruction residuals for reconstruction models; robust standardized
        # deviation otherwise. The latter is context, NOT model attribution.
        return (
            (x - self.reconstruct(x)) ** 2
            if self.name in {"pca", "robust_pca", "autoencoder"}
            else np.abs(x)
        )


def shrink(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
