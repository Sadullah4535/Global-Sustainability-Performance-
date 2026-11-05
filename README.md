# Global Sustainability Performance

Replication materials for Çelik et al. (2025), “Global sustainability performance and regional disparities: A machine learning approach based on the 2025 SDG Index,” *Sustainability*, 17(16), 7411. <https://doi.org/10.3390/su17167411>

The repository contains the study dataset, the original code release (`Codes.py`), and a command-line analysis workflow (`run_replication.py`).

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

The analysis uses five numeric variables and complete cases. The K-Means baseline uses `k=5`, `random_state=42`, and `n_init=1`; the released code does not specify `n_init`. Sensitivity tables compare `n_init` values of 1, 5, and 10. The runner's command-line options and implementation details are documented in `run_replication.py`.
