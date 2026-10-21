"""Nested cross-validation, grouped cross-validation and split-to-split spread."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    train_test_split,
)

from .models import make_model


def _fit_predict(name, grid_seed, X_tr, Y_tr, X_te, seed):
    est, grid = make_model(name, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if grid:
            search = GridSearchCV(
                est, grid, cv=KFold(5, shuffle=True, random_state=grid_seed), scoring="r2"
            )
            search.fit(X_tr, Y_tr)
            return search.predict(X_te), search.best_params_
        model = clone(est).fit(X_tr, Y_tr)
        return model.predict(X_te), {}


def _scores(Y_true, Y_pred) -> dict:
    return {
        "r2": float(r2_score(Y_true, Y_pred)),
        "r2_per_target": r2_score(Y_true, Y_pred, multioutput="raw_values").tolist(),
        "rmse_per_target": np.sqrt(
            mean_squared_error(Y_true, Y_pred, multioutput="raw_values")
        ).tolist(),
    }


def nested_cv(name, X, Y, strata, n_splits=5, n_repeats=5, seed=0) -> dict:
    """Outer repeated stratified K-fold; hyperparameters chosen in an inner 5-fold.

    Returns fold scores plus the out-of-fold predictions of every repeat, so
    each film is predicted exactly once per repeat by a model that never saw it.
    """
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds, chosen = [], []
    oof = np.full((n_repeats,) + Y.shape, np.nan)
    for k, (tr, te) in enumerate(outer.split(X, strata)):
        pred, params = _fit_predict(name, seed + 1 + k, X[tr], Y[tr], X[te], seed)
        oof[k // n_splits, te] = pred
        folds.append(_scores(Y[te], pred))
        chosen.append(params)
    return {"folds": folds, "best_params": chosen, "oof": oof}


def grouped_cv(name, X, Y, groups, n_splits=5, seed=0) -> dict:
    """K-fold over composition groups: test compositions are absent from training."""
    folds = []
    oof = np.full(Y.shape, np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, Y, groups)):
        pred, _ = _fit_predict(name, seed + 101 + k, X[tr], Y[tr], X[te], seed)
        oof[te] = pred
        folds.append(_scores(Y[te], pred))
    return {"folds": folds, "oof": oof}


def split_spread(name, X, Y, strata, n_splits=200, test_size=0.2) -> np.ndarray:
    """R2 of a single stratified 80/20 split, repeated over many random seeds."""
    out = []
    for s in range(n_splits):
        tr, te = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=s, stratify=strata
        )
        pred, _ = _fit_predict(name, s, X[tr], Y[tr], X[te], 0)
        out.append([r2_score(Y[te], pred)] + r2_score(Y[te], pred, multioutput="raw_values").tolist())
    return np.asarray(out)


def summarize(folds: list[dict]) -> dict:
    r2 = np.array([f["r2"] for f in folds])
    per = np.array([f["r2_per_target"] for f in folds])
    rmse = np.array([f["rmse_per_target"] for f in folds])
    return {
        "r2_mean": float(r2.mean()),
        "r2_std": float(r2.std(ddof=1)),
        "r2_per_target_mean": per.mean(0).tolist(),
        "rmse_per_target_mean": rmse.mean(0).tolist(),
        "n_folds": int(len(folds)),
    }
