"""Feature-selection stability and SHAP attributions."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def lasso_selection_frequency(X, Y, alpha, n_boot=200, seed=0) -> np.ndarray:
    """Fraction of bootstrap resamples in which each coefficient is non-zero.

    Inputs and targets are standardized on each resample, so one alpha applies
    to every target. Returns an array of shape (n_targets, n_features).
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((Y.shape[1], X.shape[1]))
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        Xs = StandardScaler().fit_transform(X[idx])
        Ys = StandardScaler().fit_transform(Y[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            coef = Lasso(alpha=alpha, max_iter=100_000).fit(Xs, Ys).coef_
        counts += np.abs(coef) > 1e-10
    return counts / n_boot


def standardized_lasso_coefficients(X, Y, alpha) -> np.ndarray:
    """LASSO coefficients on standardized inputs and targets, full dataset."""
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    return Lasso(alpha=alpha, max_iter=100_000).fit(Xs, Ys).coef_


FEATURE_GROUPS = {
    "composition (CNC, CNF, rGO)": ["CNC_wt_pct", "CNF_wt_pct", "rGO_wt_pct"],
    "reduction chemistry (acid, pH, agent)": ["acid_M", "pH", "ascorbic"],
}


def shap_values(model, X, n_background=20, seed=0) -> np.ndarray:
    """Kernel SHAP values of a fitted model, shape (n_targets, n_samples, n_features).

    With six features and nsamples >= 2**6, Kernel SHAP enumerates every
    coalition, so the values are exact for the chosen background.
    """
    import shap

    rng = np.random.default_rng(seed)
    background = X[rng.choice(len(X), size=min(n_background, len(X)), replace=False)]
    explainer = shap.KernelExplainer(model.predict, background)
    values = np.asarray(explainer.shap_values(X, nsamples=256, silent=True))
    if values.ndim == 3 and values.shape[0] == len(X):  # (samples, features, targets)
        values = np.moveaxis(values, -1, 0)
    return values


def mean_abs(values) -> np.ndarray:
    """Mean |SHAP| per target and feature, shape (n_targets, n_features)."""
    return np.abs(values).mean(axis=1)


def grouped_mean_abs(values, features, groups=FEATURE_GROUPS) -> dict:
    """Mean |sum of SHAP within a group| per target.

    SHAP values are additive, so summing within a group gives the group's
    attribution. For strongly correlated inputs the split inside a group is
    not identifiable, while the group total is.
    """
    out = {}
    for name, members in groups.items():
        idx = [features.index(m) for m in members]
        out[name] = np.abs(values[:, :, idx].sum(axis=2)).mean(axis=1)
    return out
