"""Loading and encoding the CNC/CNF/rGO film dataset.

The dataset is Tables 2 and 3 of Ramezani et al., "Lasso Model-Based
Optimization of CNC/CNF/rGO Nanocomposites", Micromachines 16(4), 393 (2025),
doi:10.3390/mi16040393: 30 films reduced with citric acid and 30 with
L-ascorbic acid. See data/README.md for column definitions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cnc_cnf_rgo_films.csv"
DATA_SHA256 = "51ae1bd6f4bcb7865c28911fd9a34bee69612a50feb125da2972bd5cfe9252b3"

# Inputs that vary in the dataset. Temperature is 95 degC for every film, so it
# carries no information and is excluded from the design matrix.
NUMERIC_FEATURES = ["CNC_wt_pct", "CNF_wt_pct", "rGO_wt_pct", "acid_M", "pH"]
AGENT_FEATURE = "ascorbic"  # 1 = L-ascorbic acid, 0 = citric acid
FEATURES = NUMERIC_FEATURES + [AGENT_FEATURE]
TARGETS = ["conductivity_S_per_m", "tensile_strength_MPa", "thickness_um"]
TARGET_LABELS = {
    "conductivity_S_per_m": "Conductivity (S/m)",
    "tensile_strength_MPa": "Tensile strength (MPa)",
    "thickness_um": "Thickness (um)",
}


def file_sha256(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_films(path: Path = DATA_PATH, verify: bool = True) -> pd.DataFrame:
    """Return the 60-film table, optionally checking it against the recorded hash."""
    if verify and Path(path) == DATA_PATH and file_sha256(path) != DATA_SHA256:
        raise ValueError("data/cnc_cnf_rgo_films.csv differs from the recorded SHA-256")
    df = pd.read_csv(path)
    if set(df["agent"]) != {"citric", "ascorbic"}:
        raise ValueError(f"unexpected reducing agents: {sorted(set(df['agent']))}")
    return df


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode inputs (numeric features plus an agent indicator) and targets."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    X[AGENT_FEATURE] = (df["agent"] == "ascorbic").astype(float)
    return X[FEATURES].to_numpy(), df[TARGETS].astype(float).to_numpy()


def composition_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer id per (CNC, CNF, rGO) composition.

    28 compositions were made with both reducing agents, so a random split can
    put the citric and ascorbic films of one composition on opposite sides.
    Grouping by composition tests prediction for compositions never seen.
    """
    keys = df[["CNC_wt_pct", "CNF_wt_pct", "rGO_wt_pct"]].round(6).astype(str).agg("|".join, axis=1)
    return pd.factorize(keys)[0]


def variance_inflation(df: pd.DataFrame) -> dict[str, float]:
    """VIF of each input: the diagonal of the inverse correlation matrix."""
    X, _ = design_matrix(df)
    corr = np.corrcoef(X, rowvar=False)
    return dict(zip(FEATURES, np.diag(np.linalg.inv(corr)).round(2).tolist()))
