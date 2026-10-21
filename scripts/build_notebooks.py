"""Write and execute the three notebooks in notebooks/ from the data and results/.

    python scripts/run_all.py          # produces results/
    python scripts/build_notebooks.py  # writes executed notebooks
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

SETUP = """import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
from ddmo.data import FEATURES, TARGETS, design_matrix, load_films, variance_inflation
summary = json.loads((ROOT / "results" / "summary.json").read_text())
df = load_films()
X, Y = design_matrix(df)
pd.set_option("display.precision", 3)"""

NOTEBOOKS = {
    "01_eda_and_linear_regression.ipynb": [
        ("md", """# 1. Exploratory analysis and linear baselines

Sixty CNC/CNF/rGO films from Tables 2 and 3 of Ramezani et al., *Micromachines* 16(4), 393 (2025), doi:10.3390/mi16040393: 30 reduced with citric acid and 30 with L-ascorbic acid. Targets are conductivity, tensile strength and thickness."""),
        ("code", SETUP),
        ("code", """df.groupby("agent")[TARGETS].describe().T.round(2)"""),
        ("md", """Temperature is 95 °C for every film, so the design matrix uses the five varying inputs and an agent indicator."""),
        ("code", """corr = pd.DataFrame(np.corrcoef(np.column_stack([X, Y]), rowvar=False),
                    index=FEATURES + TARGETS, columns=FEATURES + TARGETS)
corr.round(2)"""),
        ("code", """pd.Series(variance_inflation(df), name="VIF")"""),
        ("md", """**Reading.** rGO loading correlates with conductivity at r = 0.98. Acid concentration, pH and the reducing agent are nearly interchangeable (VIF 19–50), and CNC and rGO move together (r = 0.92). Any per-feature attribution inside those groups is fragile, which shapes every later choice: regularization, stability analysis and grouped SHAP."""),
        ("code", """rows = []
for name in ["mean", "ols"]:
    s, g = summary["nested_cv"][name], summary["grouped_cv"][name]
    rows.append({"model": s["label"], "nested CV R2": s["r2_mean"], "+/-": s["r2_std"],
                 **{f"R2 {t}": v for t, v in s["r2_per_target"].items()},
                 "composition-grouped R2": g["r2_mean"]})
pd.DataFrame(rows).set_index("model")"""),
        ("md", """Linear regression already explains most of the variance out of sample. Thickness is the hardest target for a linear model, which the nonlinear model in notebook 3 addresses. The composition-grouped column holds out every film of a composition together, so it tests prediction for compositions never seen."""),
    ],
    "02_feature_selection_lasso.ipynb": [
        ("md", """# 2. LASSO: the paper's setting, tuned regularization and selection stability"""),
        ("code", SETUP),
        ("code", """rows = []
for name in ["lasso_paper", "lasso"]:
    s = summary["nested_cv"][name]
    rows.append({"model": s["label"], "nested CV R2": s["r2_mean"], "+/-": s["r2_std"],
                 **{f"R2 {t}": v for t, v in s["r2_per_target"].items()}})
pd.DataFrame(rows).set_index("model")"""),
        ("md", """The paper reports overall R² = 0.841 for LASSO with α = 0.2 on an 80/20 split. The same setting gives a median of 0.843 over 200 stratified 80/20 splits, so the published figure reproduces. One shared α on raw targets shrinks the small-scale target hardest: conductivity (range 0.28–1.72 S/m) reaches only R² ≈ 0.65 while tensile strength reaches 0.98. Standardizing the targets and choosing α in an inner loop lifts overall R² to 0.96."""),
        ("code", """spread = summary["single_split_spread"]
pd.DataFrame({k: v for k, v in spread.items() if k != "paper_reported_r2"}).T"""),
        ("md", """![R2 of single 80/20 splits](../results/figures/single_split_spread.svg)

Which 12 films land in the test set matters: across 200 stratified 80/20 splits the 5th–95th percentile range of R² is 0.75–0.87 for the paper setting and 0.90–0.98 with tuned α. Nested CV averages over 25 outer folds and keeps hyperparameter selection inside each training fold."""),
        ("code", """freq = pd.read_csv(ROOT / "results" / "lasso_selection_frequency.csv")
freq.set_index(["alpha", "target"]).round(2)"""),
        ("md", """**Selection stability** (200 bootstrap resamples, standardized inputs and targets). At α = 0.2, rGO is selected in every resample for all three targets and CNC in every resample for conductivity and tensile strength. Acid concentration, pH and the agent indicator trade places: they carry overlapping information, so LASSO keeps whichever one fits a given resample. The stable conclusion is about the group, not a single member."""),
    ],
    "03_nonlinear_models_and_optimization.ipynb": [
        ("md", """# 3. Multi-output neural network, SHAP and in-range screening"""),
        ("code", SETUP),
        ("code", """s = summary["nested_cv"]["mlp"]
print(s["label"], "| nested CV R2 =", s["r2_mean"], "+/-", s["r2_std"])
pd.Series(s["r2_per_target"], name="R2 per target")"""),
        ("md", """![Out-of-fold parity plots](../results/figures/parity_nested_cv.svg)

One network predicts all three properties from a shared hidden layer; hidden size and weight decay are chosen in the inner loop. It is most useful for thickness, where it raises out-of-sample R² from 0.89 (linear) to 0.97."""),
        ("code", """pd.DataFrame({m: pd.DataFrame(v).stack() for m, v in summary["shap_grouped_share"].items()}).round(3)"""),
        ("md", """![Share of mean |SHAP| per input, multi-output network](../results/figures/shap_importance_mlp.svg)

**SHAP.** Per feature, the network credits CNC more and LASSO credits rGO more, because CNC and rGO are correlated at 0.92 and the split of credit between them is not identifiable from these data. SHAP values are additive, so summing within groups gives an identifiable answer, and there the two models agree: composition carries about 83–91% of the attribution for every target and reduction chemistry the rest."""),
        ("code", """pd.Series(summary["screening"], dtype=object)"""),
        ("md", """**Screening inside the measured region.** Candidates are convex combinations of measured films with the same reducing agent, which keeps the coupled pH and acid concentration realistic. Under the paper's objective F = 0.7σ + 0.3Ts, no candidate is predicted to beat the best measured film (L-ascorbic acid #28, F = 15.0) by both models. The best film sits on the edge of the measured design, so further gains require experiments beyond it. That is the setting for sequential design with calibrated uncertainty, developed in the materials-mobo repository."""),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src) for kind, src in cells]
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    for cell in nb.cells:  # drop volatile execution timestamps
        cell.metadata.pop("execution", None)
    nbformat.write(nb, NB / name)
    print("wrote", NB / name)


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        build(name, cells)
