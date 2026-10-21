"""Model definitions with their inner-loop hyperparameter grids.

Every model is a scikit-learn pipeline, so feature scaling is fitted on the
training fold only. Targets are standardized inside TransformedTargetRegressor,
which lets one regularization strength serve all three targets even though
their scales differ by two orders of magnitude.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PAPER_LASSO_ALPHA = 0.2


def _scaled(est) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def _std_target(est) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def make_model(name: str, seed: int = 0):
    """Return (estimator, inner-CV parameter grid) for a named model."""
    if name == "mean":
        return DummyRegressor(), {}
    if name == "ols":
        return _std_target(_scaled(LinearRegression())), {}
    if name == "lasso":
        return (
            _std_target(_scaled(Lasso(max_iter=100_000))),
            {"regressor__model__alpha": np.logspace(-4, 0, 17).tolist()},
        )
    if name == "lasso_paper":
        # Fixed alpha on standardized inputs and raw targets, as in the paper.
        return _scaled(Lasso(alpha=PAPER_LASSO_ALPHA, max_iter=100_000)), {}
    if name == "mlp":
        return (
            _std_target(
                _scaled(MLPRegressor(solver="lbfgs", max_iter=5000, random_state=seed))
            ),
            {
                "regressor__model__hidden_layer_sizes": [(8,), (16,), (32, 16)],
                "regressor__model__alpha": [1e-3, 1e-2, 1e-1, 1.0],
            },
        )
    raise KeyError(f"unknown model {name!r}")


MODEL_NAMES = ["mean", "ols", "lasso", "lasso_paper", "mlp"]
MODEL_LABELS = {
    "mean": "Mean predictor",
    "ols": "Linear regression",
    "lasso": "LASSO, alpha tuned in inner CV",
    "lasso_paper": "LASSO, alpha = 0.2 (paper setting)",
    "mlp": "Multi-output neural network",
}
