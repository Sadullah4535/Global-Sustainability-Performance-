# Global Sustainability Performance

Replication materials for Çelik et al. (2025), “Global sustainability performance and regional disparities: A machine learning approach based on the 2025 SDG Index,” *Sustainability*, 17(16), 7411. <https://doi.org/10.3390/su17167411>

The dataset is included; no download is required.

## Run the analysis

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_replication.py
```

The runner reads `SDG2025.csv` and writes tables and figures to `replication_results/`. To specify a different data file or output directory:

```bash
python run_replication.py --data path/to/SDG2025.csv --output-dir results
```

## Analysis and outputs

The workflow reports sample counts and runs PCA, K-Means clustering and sensitivity analysis, cluster profiling, ANOVA/MANOVA, Random Forest feature importance, and the six classifier families. It exports the resulting tables and figures, including comparisons with values reported in the paper.

The clustering workflow uses the complete cases for `sdg_score`, `spillover_score`, `regional_score`, `population`, and `progress`. It standardizes these variables with `StandardScaler`, then fits PCA and K-Means. The baseline uses `k=5`, `random_state=42`, and `n_init=1`. Sensitivity tables evaluate `k=2–10` with `n_init` values of 1, 5, and 10. Classifier metrics use a stratified 70/30 train/test split with `random_state=42`.
