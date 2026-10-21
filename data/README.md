# Data

`cnc_cnf_rgo_films.csv` holds the 60 measured films of

> G. Ramezani, I. Ocampo Silva, I. Stiharu, T. G. M. van de Ven, V. Nerguizian,
> "Lasso Model-Based Optimization of CNC/CNF/rGO Nanocomposites,"
> *Micromachines* **16**(4), 393 (2025). <https://doi.org/10.3390/mi16040393>

Rows 1–30 with `agent = citric` are Table 2 (citric acid reduction); rows with
`agent = ascorbic` are Table 3 (L-ascorbic acid reduction). Values are copied
as published. The article is open access under a Creative Commons Attribution
license; please cite it when reusing the data.

SHA-256: `51ae1bd6f4bcb7865c28911fd9a34bee69612a50feb125da2972bd5cfe9252b3`
(checked by `ddmo.data.load_films` and `tests/test_data.py`).

| Column | Meaning |
|---|---|
| `agent` | Reducing agent: `citric` or `ascorbic` (L-ascorbic acid) |
| `sample_no` | Row number within the published table |
| `CNC_wt_pct` | Cellulose nanocrystals, wt% |
| `CNF_wt_pct` | Cellulose nanofibrils, wt% |
| `rGO_wt_pct` | Reduced graphene oxide, wt% |
| `acid_M` | Reducing-acid concentration, M |
| `temperature_C` | Reduction temperature, °C (95 for every film) |
| `pH` | pH of the reaction mixture |
| `conductivity_S_per_m` | Electrical conductivity, S/m (four-point probe) |
| `tensile_strength_MPa` | Tensile strength, MPa |
| `thickness_um` | Film thickness, µm |

Temperature is constant, so the models use the five varying inputs plus an
agent indicator. Composition, acid concentration and pH are strongly
correlated (variance inflation factors up to about 50), which shapes every
modelling choice in this repository.
