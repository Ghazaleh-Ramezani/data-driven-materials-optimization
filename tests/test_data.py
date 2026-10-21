import numpy as np

from ddmo.data import (
    DATA_SHA256,
    FEATURES,
    TARGETS,
    composition_groups,
    design_matrix,
    file_sha256,
    load_films,
    variance_inflation,
)


def test_dataset_matches_recorded_hash():
    assert file_sha256() == DATA_SHA256


def test_sixty_films_thirty_per_agent():
    df = load_films()
    assert len(df) == 60
    assert df["agent"].value_counts().to_dict() == {"citric": 30, "ascorbic": 30}
    assert sorted(df["sample_no"].unique()) == list(range(1, 31))


def test_published_values_spot_check():
    df = load_films().set_index(["agent", "sample_no"])
    # Table 2, row 28 and Table 3, row 28 of the paper
    assert df.loc[("citric", 28), ["conductivity_S_per_m", "tensile_strength_MPa", "thickness_um"]].tolist() == [1.62, 44, 152]
    assert df.loc[("ascorbic", 28), ["conductivity_S_per_m", "tensile_strength_MPa", "thickness_um"]].tolist() == [1.72, 46, 148]


def test_temperature_is_constant_and_excluded():
    df = load_films()
    assert df["temperature_C"].nunique() == 1
    assert "temperature_C" not in FEATURES


def test_design_matrix_shapes_and_agent_indicator():
    df = load_films()
    X, Y = design_matrix(df)
    assert X.shape == (60, len(FEATURES)) and Y.shape == (60, len(TARGETS))
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    agent = X[:, FEATURES.index("ascorbic")]
    assert set(agent) == {0.0, 1.0}
    assert agent.sum() == 30


def test_composition_groups_pair_the_two_agents():
    df = load_films()
    g = composition_groups(df)
    assert g.max() + 1 == 32
    # 28 compositions appear with both agents, 4 with one
    sizes = np.bincount(g)
    assert (sizes == 2).sum() == 28 and (sizes == 1).sum() == 4


def test_collinearity_is_severe():
    vif = variance_inflation(load_films())
    assert vif["pH"] > 10 and vif["CNC_wt_pct"] > 10
