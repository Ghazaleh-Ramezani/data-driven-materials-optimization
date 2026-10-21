"""Model-based screening of new formulations inside the measured design region.

pH and acid concentration are coupled to the composition in the measured
films, so candidates are convex combinations of measured films made with the
same reducing agent. Every candidate therefore lies inside the convex hull of
real data, where the surrogates were fitted.
"""

from __future__ import annotations

import numpy as np

from .data import FEATURES

PAPER_WEIGHTS = (0.7, 0.3)  # F = 0.7 * conductivity + 0.3 * tensile strength


def hull_candidates(X, n=20000, k=3, seed=0) -> np.ndarray:
    """Random convex combinations of k measured films that share a reducing agent."""
    rng = np.random.default_rng(seed)
    agent = X[:, FEATURES.index("ascorbic")]
    out = np.empty((n, X.shape[1]))
    for i in range(n):
        a = rng.integers(0, 2)
        pool = np.flatnonzero(agent == a)
        idx = rng.choice(pool, size=k, replace=False)
        out[i] = rng.dirichlet(np.ones(k)) @ X[idx]
        out[i, FEATURES.index("ascorbic")] = float(a)  # exact, not 0.9999...
    return out


def score(Y, weights=PAPER_WEIGHTS) -> np.ndarray:
    return weights[0] * Y[:, 0] + weights[1] * Y[:, 1]


def screen(models: dict, X, Y, n=20000, seed=0) -> dict:
    """Rank hull candidates by the paper's objective under each fitted model."""
    cands = hull_candidates(X, n=n, seed=seed)
    preds = {name: m.predict(cands) for name, m in models.items()}
    names = list(models)
    primary = names[0]
    best = int(np.argmax(score(preds[primary])))
    observed = score(Y)
    return {
        "best_observed_index": int(np.argmax(observed)),
        "best_observed_F": float(observed.max()),
        "candidate": dict(zip(FEATURES, cands[best].round(4).tolist())),
        "predicted_F": {k: float(score(v[best : best + 1])[0]) for k, v in preds.items()},
        "predicted_targets": {k: v[best].round(4).tolist() for k, v in preds.items()},
        "n_candidates": n,
        "n_predicted_above_best_observed_by_all_models": int(
            np.all([score(v) > observed.max() for v in preds.values()], axis=0).sum()
        ),
    }
