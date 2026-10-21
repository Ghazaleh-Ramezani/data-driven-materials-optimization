"""Regenerate every number and figure in results/ from data/cnc_cnf_rgo_films.csv.

    python scripts/run_all.py            # full run, about 5 minutes on 2 CPU cores
    python scripts/run_all.py --quick    # fewer repeats and resamples, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ddmo.data import (  # noqa: E402
    FEATURES,
    TARGET_LABELS,
    TARGETS,
    composition_groups,
    design_matrix,
    file_sha256,
    load_films,
    variance_inflation,
)
from ddmo.evaluation import grouped_cv, nested_cv, split_spread, summarize  # noqa: E402
from ddmo.interpret import (  # noqa: E402
    grouped_mean_abs,
    lasso_selection_frequency,
    mean_abs,
    shap_values,
    standardized_lasso_coefficients,
)
from ddmo.models import MODEL_LABELS, MODEL_NAMES, make_model  # noqa: E402
from ddmo.optimize import screen  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ddmo", "svg.fonttype": "none", "font.size": 9})
_NUM = re.compile(r"-?\d+\.\d{3,}")
_TAG = re.compile(r"<[^>]+>")


def save_svg(fig, path: Path) -> None:
    """Deterministic, compact SVG: no date, text kept as text, coordinates
    inside tags rounded to 0.01 px (visible labels untouched), no trailing
    whitespace."""
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    rounded = lambda m: f"{float(m.group()):.2f}"  # noqa: E731
    text = _TAG.sub(lambda t: _NUM.sub(rounded, t.group()), path.read_text())
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def fit_full(name, X, Y, seed=0):
    est, grid = make_model(name, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if grid:
            search = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2")
            search.fit(X, Y)
            return search.best_estimator_, search.best_params_
        return clone(est).fit(X, Y), {}


def r(x, nd=4):
    return [round(float(v), nd) for v in x] if np.ndim(x) else round(float(x), nd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    n_repeats, n_splits_spread, n_boot = (2, 30, 40) if args.quick else (5, 200, 200)

    df = load_films()
    X, Y = design_matrix(df)
    strata = df["agent"].to_numpy()
    groups = composition_groups(df)

    summary: dict = {
        "dataset": {
            "source": "Ramezani et al., Micromachines 16(4), 393 (2025), doi:10.3390/mi16040393, Tables 2 and 3",
            "sha256": file_sha256(),
            "n_films": int(len(df)),
            "n_per_agent": df["agent"].value_counts().sort_index().to_dict(),
            "n_compositions": int(groups.max() + 1),
            "features": FEATURES,
            "targets": TARGETS,
            "variance_inflation": variance_inflation(df),
        },
        "protocol": {
            "outer": f"RepeatedStratifiedKFold(5 folds x {n_repeats} repeats), stratified by reducing agent",
            "inner": "5-fold GridSearchCV on the outer training fold",
            "grouped": "GroupKFold(5) over (CNC, CNF, rGO) compositions",
            "split_spread": f"{n_splits_spread} stratified 80/20 splits",
        },
    }

    # 1. Nested CV and composition-grouped CV for every model
    rows, nested = [], {}
    for name in MODEL_NAMES:
        res = nested_cv(name, X, Y, strata, n_repeats=n_repeats, seed=0)
        nested[name] = res
        s = summarize(res["folds"])
        g = summarize(grouped_cv(name, X, Y, groups)["folds"])
        summary.setdefault("nested_cv", {})[name] = {
            "label": MODEL_LABELS[name],
            "r2_mean": r(s["r2_mean"]),
            "r2_std": r(s["r2_std"]),
            "r2_per_target": dict(zip(TARGETS, r(s["r2_per_target_mean"]))),
            "rmse_per_target": dict(zip(TARGETS, r(s["rmse_per_target_mean"]))),
            "n_folds": s["n_folds"],
        }
        summary.setdefault("grouped_cv", {})[name] = {
            "r2_mean": r(g["r2_mean"]),
            "r2_per_target": dict(zip(TARGETS, r(g["r2_per_target_mean"]))),
        }
        for k, f in enumerate(res["folds"]):
            rows.append({"model": name, "fold": k, "r2": f["r2"], **{f"r2_{t}": v for t, v in zip(TARGETS, f["r2_per_target"])}})
        print(f"{name:12s} nested R2 {s['r2_mean']:.3f} +/- {s['r2_std']:.3f}   grouped R2 {g['r2_mean']:.3f}")
    pd.DataFrame(rows).round(6).to_csv(out / "nested_cv_folds.csv", index=False)

    alphas = [p["regressor__model__alpha"] for p in nested["lasso"]["best_params"]]
    summary["nested_cv"]["lasso"]["inner_alpha_median"] = r(np.median(alphas), 6)
    summary["nested_cv"]["lasso"]["inner_alpha_range"] = [r(min(alphas), 6), r(max(alphas), 6)]

    # 2. Spread of a single 80/20 split: paper setting vs tuned alpha
    spread = {}
    for name in ["lasso_paper", "lasso"]:
        a = split_spread(name, X, Y, strata, n_splits=n_splits_spread)
        spread[name] = a
        summary.setdefault("single_split_spread", {})[name] = {
            "r2_median": r(np.median(a[:, 0])),
            "r2_p05": r(np.percentile(a[:, 0], 5)),
            "r2_p95": r(np.percentile(a[:, 0], 95)),
            "r2_per_target_median": dict(zip(TARGETS, r(np.median(a[:, 1:], axis=0)))),
        }
    summary["single_split_spread"]["paper_reported_r2"] = 0.841

    # 3. Feature-selection stability along a regularization path
    # (inputs and targets standardized, so alpha is comparable across targets)
    alpha_path = [0.01, 0.05, 0.1, 0.2]
    stab_rows, interp = [], {"n_bootstrap": n_boot, "alpha_path": alpha_path, "selection_frequency": {}, "standardized_coefficients": {}}
    for a in alpha_path:
        freq = lasso_selection_frequency(X, Y, alpha=a, n_boot=n_boot)
        coef = standardized_lasso_coefficients(X, Y, alpha=a)
        for i, t in enumerate(TARGETS):
            stab_rows.append({"alpha": a, "target": t, **dict(zip(FEATURES, freq[i]))})
        interp["selection_frequency"][str(a)] = {t: dict(zip(FEATURES, r(freq[i]))) for i, t in enumerate(TARGETS)}
        interp["standardized_coefficients"][str(a)] = {t: dict(zip(FEATURES, r(coef[i]))) for i, t in enumerate(TARGETS)}
    pd.DataFrame(stab_rows).round(4).to_csv(out / "lasso_selection_frequency.csv", index=False)
    summary["lasso_interpretation"] = interp

    # 4. Final models on all data, SHAP attributions and in-hull screening
    mlp, mlp_params = fit_full("mlp", X, Y)
    lasso, lasso_params = fit_full("lasso", X, Y)
    summary["final_models"] = {"mlp": {k: str(v) for k, v in mlp_params.items()}, "lasso": {k: r(v, 6) for k, v in lasso_params.items()}}
    shap_rows = []
    for name, model in [("mlp", mlp), ("lasso", lasso)]:
        vals = shap_values(model, X)
        imp = mean_abs(vals)
        grp = grouped_mean_abs(vals, FEATURES)
        for i, t in enumerate(TARGETS):
            shap_rows.append({"model": name, "target": t, **dict(zip(FEATURES, imp[i])), **{g: v[i] for g, v in grp.items()}})
        summary.setdefault("shap_mean_abs", {})[name] = {t: dict(zip(FEATURES, r(imp[i]))) for i, t in enumerate(TARGETS)}
        summary.setdefault("shap_grouped_share", {})[name] = {
            t: {g: r(v[i] / sum(w[i] for w in grp.values())) for g, v in grp.items()} for i, t in enumerate(TARGETS)
        }
    pd.DataFrame(shap_rows).round(6).to_csv(out / "shap_importance.csv", index=False)
    summary["screening"] = screen({"mlp": mlp, "lasso": lasso}, X, Y, n=4000 if args.quick else 20000)

    summary["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Figures
    oof_mlp, oof_lasso = nested["mlp"]["oof"][0], nested["lasso"]["oof"][0]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for row, (name, oof) in enumerate([("Multi-output NN", oof_mlp), ("LASSO", oof_lasso)]):
        for j, t in enumerate(TARGETS):
            ax = axes[row, j]
            for agent, mk in [("citric", "o"), ("ascorbic", "^")]:
                m = strata == agent
                ax.plot(Y[m, j], oof[m, j], mk, ms=3.5, label=agent, alpha=0.8)
            lo, hi = Y[:, j].min(), Y[:, j].max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(f"{name}: {TARGET_LABELS[t]}", fontsize=8)
            ax.set_xlabel("measured")
            ax.set_ylabel("out-of-fold prediction")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save_svg(fig, out / "figures" / "parity_nested_cv.svg")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.linspace(min(spread["lasso_paper"][:, 0].min(), spread["lasso"][:, 0].min()), 1.0, 30)
    ax.hist(spread["lasso_paper"][:, 0], bins=bins, alpha=0.7, label="alpha = 0.2 (paper setting)")
    ax.hist(spread["lasso"][:, 0], bins=bins, alpha=0.7, label="alpha tuned in inner CV")
    ax.axvline(0.841, color="k", ls="--", lw=0.8, label="paper: R2 = 0.841")
    ax.set_xlabel("R2 on one stratified 80/20 split")
    ax.set_ylabel("splits")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_svg(fig, out / "figures" / "single_split_spread.svg")

    imp = pd.DataFrame(shap_rows)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for j, t in enumerate(TARGETS):
        vals = imp[(imp.model == "mlp") & (imp.target == t)][FEATURES].iloc[0]
        vals = vals / vals.sum()
        axes[j].barh(FEATURES, vals.values)
        axes[j].set_title(TARGET_LABELS[t], fontsize=8)
        axes[j].set_xlabel("share of mean |SHAP|")
        axes[j].invert_yaxis()
    fig.tight_layout()
    save_svg(fig, out / "figures" / "shap_importance_mlp.svg")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
