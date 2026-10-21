import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ddmo.data import composition_groups, design_matrix, load_films
from ddmo.evaluation import grouped_cv, nested_cv, split_spread, summarize
from ddmo.interpret import grouped_mean_abs, lasso_selection_frequency
from ddmo.models import MODEL_NAMES, make_model
from ddmo.optimize import hull_candidates


@pytest.fixture(scope="module")
def data():
    df = load_films()
    X, Y = design_matrix(df)
    return df, X, Y


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "mean"])
def test_scaling_lives_inside_the_pipeline(name):
    est, _ = make_model(name)
    pipe = est.regressor if isinstance(est, TransformedTargetRegressor) else est
    assert isinstance(pipe, Pipeline) and isinstance(pipe.steps[0][1], StandardScaler)


def test_outer_folds_partition_each_repeat(data):
    df, X, _ = data
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    seen = np.zeros((2, len(X)), int)
    for k, (tr, te) in enumerate(outer.split(X, df["agent"])):
        assert not set(tr) & set(te)
        seen[k // 5, te] += 1
    assert (seen == 1).all()


def test_nested_cv_predicts_every_film_once_per_repeat(data):
    df, X, Y = data
    res = nested_cv("lasso", X, Y, df["agent"].to_numpy(), n_repeats=1)
    assert np.isfinite(res["oof"]).all()
    s = summarize(res["folds"])
    assert s["n_folds"] == 5 and s["r2_mean"] > 0.9


def test_grouped_folds_never_split_a_composition(data):
    df, X, Y = data
    g = composition_groups(df)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, g):
        assert not set(g[tr]) & set(g[te])


def test_grouped_cv_runs(data):
    df, X, Y = data
    res = grouped_cv("ols", X, Y, composition_groups(df))
    assert np.isfinite(res["oof"]).all()


def test_paper_setting_reproduces_reported_r2(data):
    # Paper: alpha = 0.2 ("20%"), 80/20 split, overall R2 = 0.841
    df, X, Y = data
    spread = split_spread("lasso_paper", X, Y, df["agent"].to_numpy(), n_splits=50)
    assert 0.80 < np.median(spread[:, 0]) < 0.88


def test_mean_predictor_is_near_zero(data):
    df, X, Y = data
    s = summarize(nested_cv("mean", X, Y, df["agent"].to_numpy(), n_repeats=1)["folds"])
    assert s["r2_mean"] < 0.05


def test_selection_frequency_bounds(data):
    _, X, Y = data
    f = lasso_selection_frequency(X, Y, alpha=0.2, n_boot=10)
    assert f.shape == (3, X.shape[1]) and (f >= 0).all() and (f <= 1).all()


def test_grouped_shap_is_sum_within_group():
    values = np.zeros((1, 2, 6))
    values[0, :, 0], values[0, :, 2] = 1.0, -1.0  # cancel inside composition group
    values[0, :, 4] = 2.0
    feats = ["CNC_wt_pct", "CNF_wt_pct", "rGO_wt_pct", "acid_M", "pH", "ascorbic"]
    g = grouped_mean_abs(values, feats)
    assert g["composition (CNC, CNF, rGO)"][0] == 0.0
    assert g["reduction chemistry (acid, pH, agent)"][0] == 2.0


def test_hull_candidates_stay_inside_observed_ranges(data):
    _, X, _ = data
    c = hull_candidates(X, n=500)
    assert (c >= X.min(0) - 1e-12).all() and (c <= X.max(0) + 1e-12).all()
    assert set(np.unique(c[:, -1])) <= {0.0, 1.0}
