# Global Sustainability Performance: reproducible analysis

This repository contains the data and original code release for Çelik et al. (2025), plus a clean runner for reproducing the main analysis and exporting its figures and tables.

Paper: Çelik, S., Öztürk, Ö. F., Akkucuk, U., & Şaşmaz, M. Ü. (2025). “Global sustainability performance and regional disparities: A machine learning approach based on the 2025 SDG Index.” *Sustainability*, 17(16), 7411. <https://doi.org/10.3390/su17167411>

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_replication.py
```

The script reads the bundled `SDG2025.csv`; it does not download data. Figures and tables are written to `replication_results/`. To choose another output folder or data file, run:

```bash
python run_replication.py --data path/to/SDG2025.csv --output-dir results
```

## What the runner does

The runner validates and reports the sample size, standardizes five numeric features, fits PCA, evaluates k=2–10, produces the k=5 cluster profiles, computes ANOVA/MANOVA tables, estimates Random Forest feature importance, and evaluates the six classifier families reported in the paper. Seeds are fixed at 42. K-Means `n_init` is an explicit option (`--n-init`, default 1).

Outputs include:

- `table0_baseline_summary.csv` and `table1_sample_counts.csv`
- `table2_k_sensitivity.csv` and `table2_k_sensitivity_summary.csv`
- computed ANOVA/MANOVA tables alongside separate comparisons with the paper's reported values
- cluster profiles, Random Forest importances, and classifier metrics alongside a separate Table 5 comparison
- correlation, cluster, sensitivity, feature-importance, confusion-matrix, and ROC figures

## Reproduction notes

- The CSV contains 167 countries. Twenty-four have missing progress values, leaving 143 complete cases for the five-feature analysis. The paper reports 166 countries.
- The executable baseline follows the released code's clustering implementation: five numeric features, complete-case rows, and `StandardScaler`. The paper's methods text describes min-max scaling and label encoding for region. This implementation difference is surfaced here rather than silently resolved.
- The released code does not set K-Means `n_init`. The default runner uses `n_init=1` because it most closely reproduces the paper's reported k=5 silhouette score in the tested environment. It also reproduces the paper's reported cluster sizes (56, 33, 35, 2, 17) and comes close to the reported classifier metrics. The sensitivity output compares `n_init=1, 5, 10`; the preferred k changes across these settings. Treat `n_init=1` as a documented reconstruction, not a confirmed setting from the authors.
- Recomputed ANOVA/MANOVA values are written separately from the paper's values. In particular, the computed tests do not reproduce the paper's reported test statistics, so the published values are retained only as comparison references.
- The ANOVA/MANOVA tables are computed from the current run. Because clusters and PCA coordinates are both derived from the same input features, these tests describe separation within this constructed dataset; they are not independent validation of real-world sustainability groups.
- Classifiers predict K-Means labels made from the same five features. Their scores measure how recoverable those labels are from the inputs, not external predictive validity.

`Codes.py` is retained as the original notebook export. It is not the supported entry point; use `run_replication.py` for a fresh, ordered run.
