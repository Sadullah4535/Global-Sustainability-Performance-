#!/usr/bin/env python3
"""Run the sustainability analysis using the bundled SDG2025.csv dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.multivariate.manova import MANOVA
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
FEATURES = ["sdg_score", "spillover_score", "regional_score", "population", "progress"]
RAW_COLUMNS = {
    "Country": "country",
    "2025 SDG Index Score": "sdg_score",
    "International Spillovers Score (0-100)": "spillover_score",
    "Regional Score (0-100)": "regional_score",
    "Population in 2024": "population",
    "Progress on Headline SDGi (p.p.)": "progress",
    "Regions used for the SDR": "region",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "SDG2025.csv",
        help="Path to the semicolon-separated SDG data CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "replication_results",
        help="Directory where figures and CSV tables will be written.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="K-Means initializations for the k=5 baseline (default: 1).",
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep=";", encoding="ISO-8859-1", engine="python")
    data = raw.rename(columns=RAW_COLUMNS)
    missing_columns = sorted(set(FEATURES) - set(data.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing expected feature columns: {missing_columns}")

    complete = data.dropna(subset=FEATURES).copy()
    print(f"Rows in CSV: {len(data)}")
    print(f"Complete cases used for clustering: {len(complete)}")
    print("Rows excluded for missing clustering features:", len(data) - len(complete))
    return data, complete


def save_cluster_count(data: pd.DataFrame, output_dir: Path) -> None:
    counts = pd.DataFrame(
        [
            {"sample": "CSV rows", "n": len(data)},
            {"sample": "complete cases", "n": int(data[FEATURES].dropna().shape[0])},
            {
                "sample": "rows with any missing clustering feature",
                "n": int(data[FEATURES].isna().any(axis=1).sum()),
            },
        ]
    )
    counts.to_csv(output_dir / "table1_sample_counts.csv", index=False)


def run_k_sensitivity(X_scaled: np.ndarray, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n_init in (1, 5, 10):
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=42, n_init=n_init)
            labels = model.fit_predict(X_scaled)
            rows.append(
                {
                    "n_init": n_init,
                    "k": k,
                    "wcss": model.inertia_,
                    "silhouette": silhouette_score(X_scaled, labels),
                }
            )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(output_dir / "table2_k_sensitivity.csv", index=False)

    summary = (
        sensitivity.loc[sensitivity.groupby("n_init")["silhouette"].idxmax()]
        [["n_init", "k", "silhouette"]]
        .rename(columns={"k": "best_k_by_silhouette"})
        .sort_values("n_init")
    )
    summary.to_csv(output_dir / "table2_k_sensitivity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n_init, group in sensitivity.groupby("n_init"):
        axes[0].plot(group["k"], group["wcss"], marker="o", label=f"n_init={n_init}")
        axes[1].plot(
            group["k"], group["silhouette"], marker="o", label=f"n_init={n_init}"
        )
    axes[0].set(title="Elbow curve", xlabel="Number of clusters (k)", ylabel="WCSS")
    axes[1].set(
        title="Silhouette by k and initialization",
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
    )
    axes[1].axhline(0, color="black", linewidth=0.6)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_k_sensitivity.png", dpi=160)
    plt.close(fig)
    return sensitivity


def calculate_validation(pca_scores: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    pca_frame = pd.DataFrame(pca_scores[:, :2], columns=["PCA1", "PCA2"])
    pca_frame["cluster"] = labels
    anova_rows = []
    for component in ("PCA1", "PCA2"):
        groups = [g[component].to_numpy() for _, g in pca_frame.groupby("cluster")]
        statistic, p_value = f_oneway(*groups)
        anova_rows.append(
            {"feature": component, "f_value": statistic, "p_value": p_value}
        )
    anova_frame = pd.DataFrame(anova_rows)
    anova_frame.to_csv(output_dir / "table3_anova_computed.csv", index=False)
    paper_anova = pd.DataFrame(
        [
            {"feature": "PCA1", "paper_f_value": 45.632, "paper_p_value": 1.23e-10},
            {"feature": "PCA2", "paper_f_value": 39.871, "paper_p_value": 3.45e-9},
        ]
    )
    anova_frame.merge(paper_anova, on="feature").assign(
        f_value_difference=lambda frame: frame["f_value"] - frame["paper_f_value"]
    ).to_csv(output_dir / "table3_anova_comparison.csv", index=False)

    result = MANOVA.from_formula("PCA1 + PCA2 ~ C(cluster)", data=pca_frame).mv_test()
    stats = result.results["C(cluster)"]["stat"]
    manova = stats.reset_index().rename(
        columns={"index": "test_statistic", "Value": "value", "F Value": "f_value", "Pr > F": "p_value"}
    )
    manova.to_csv(output_dir / "table4_manova_computed.csv", index=False)
    paper_manova = pd.DataFrame(
        [
            {"test_statistic": "Wilks' lambda", "paper_value": 0.243, "paper_f_value": 67.32},
            {"test_statistic": "Pillai's trace", "paper_value": 0.573, "paper_f_value": 70.89},
            {"test_statistic": "Hotelling-Lawley trace", "paper_value": 1.047, "paper_f_value": 72.10},
            {"test_statistic": "Roy's greatest root", "paper_value": 0.823, "paper_f_value": 69.55},
        ]
    )
    manova.merge(paper_manova, on="test_statistic").assign(
        value_difference=lambda frame: frame["value"] - frame["paper_value"],
        f_value_difference=lambda frame: frame["f_value"] - frame["paper_f_value"],
    ).to_csv(output_dir / "table4_manova_comparison.csv", index=False)


def fit_classifiers(
    X_scaled: np.ndarray, labels: np.ndarray, output_dir: Path
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )
    classes = np.sort(np.unique(labels))
    y_test_binary = label_binarize(y_test, classes=classes)
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=1
        ),
        "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    metric_rows = []
    roc_data = []
    for index, (name, estimator) in enumerate(models.items()):
        classifier = OneVsRestClassifier(estimator)
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        for class_index, class_label in enumerate(classes):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test_binary[:, class_index], probabilities[:, class_index]
            )
            roc_data.append(
                (index, name, class_label, false_positive_rate, true_positive_rate)
            )
        report = classification_report(
            y_test, predicted, output_dict=True, zero_division=0
        )
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "macro_roc_auc_ovr": roc_auc_score(
                    y_test_binary, probabilities, average="macro", multi_class="ovr"
                ),
            }
        )

        matrix = confusion_matrix(y_test, predicted, labels=classes)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=f"{name}: confusion matrix", xlabel="Predicted", ylabel="True")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "table5_classifier_metrics.csv", index=False)
    # Values transcribed from Table 5 of the cited paper; kept in a separate
    # comparison output so the reported and recomputed results stay distinct.
    reported = pd.DataFrame(
        [
            ["Random Forest", .977, .789, .800, .794, 1.000],
            ["SVM", .977, .982, .960, .968, 1.000],
            ["Decision Tree", .953, .948, .976, .960, .974],
            ["XGBoost", .907, .750, .708, .718, .835],
            ["ANN", .977, .982, .988, .984, 1.000],
            ["Logistic Regression", .953, .968, .968, .968, .994],
        ],
        columns=[
            "model", "paper_accuracy", "paper_macro_precision", "paper_macro_recall",
            "paper_macro_f1", "paper_macro_roc_auc_ovr",
        ],
    )
    comparison = metrics.merge(reported, on="model")
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc_ovr"):
        comparison[f"{metric}_difference"] = comparison[metric] - comparison[f"paper_{metric}"]
    comparison.to_csv(output_dir / "table5_classifier_comparison.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for index, (name, _) in enumerate(models.items()):
        ax = axes.flat[index]
        for _, model_name, class_label, fpr, tpr in roc_data:
            if model_name == name:
                ax.plot(fpr, tpr, label=f"Cluster {class_label}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(title=name, xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("One-vs-rest ROC curves for the six classifiers")
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_classifier_roc_curves.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, complete = load_data(args.data)
    save_cluster_count(data, args.output_dir)

    X = complete[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    pca_scores = pca.fit_transform(X_scaled)

    sensitivity = run_k_sensitivity(X_scaled, args.output_dir)
    baseline = KMeans(n_clusters=5, random_state=42, n_init=args.n_init)
    labels = baseline.fit_predict(X_scaled)
    complete["cluster"] = labels
    complete["PCA1"], complete["PCA2"], complete["PCA3"] = pca_scores.T
    summary = {
        "n_init": args.n_init,
        "k": 5,
        "silhouette": silhouette_score(X_scaled, labels),
        "cluster_sizes": str(complete["cluster"].value_counts().sort_index().to_dict()),
        "pca_variance_pc1": pca.explained_variance_ratio_[0],
        "pca_variance_pc2": pca.explained_variance_ratio_[1],
        "pca_variance_pc3": pca.explained_variance_ratio_[2],
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "table0_baseline_summary.csv", index=False)

    corr = complete[FEATURES].corr()
    corr.to_csv(args.output_dir / "table_correlation.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig2_correlation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        complete["PCA1"], complete["PCA2"], c=labels, cmap="tab10", alpha=0.8
    )
    ax.set(title="K-Means clusters in PCA space (k=5)", xlabel="PCA1", ylabel="PCA2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig3_pca_clusters.png", dpi=160)
    plt.close(fig)

    profiles = complete.groupby("cluster")[FEATURES].mean()
    profiles.to_csv(args.output_dir / "table_cluster_profiles.csv")
    scaled_profiles = pd.DataFrame(
        MinMaxScaler().fit_transform(profiles),
        columns=FEATURES,
        index=profiles.index,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(scaled_profiles, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Cluster feature means, scaled within each feature")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig4_cluster_profiles.png", dpi=160)
    plt.close(fig)

    calculate_validation(pca_scores, labels, args.output_dir)
    fit_classifiers(X_scaled, labels, args.output_dir)

    # This is the same feature set used to construct the cluster labels. The
    # score describes recoverability of those labels, not external validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )
    forest = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    importances = pd.DataFrame(
        {"feature": FEATURES, "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(args.output_dir / "table_rf_feature_importance.csv", index=False)
    pd.DataFrame(
        [{"train_accuracy": accuracy_score(y_train, forest.predict(X_train)),
          "test_accuracy": accuracy_score(y_test, forest.predict(X_test))}]
    ).to_csv(args.output_dir / "table_rf_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=importances, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig5_rf_importance.png", dpi=160)
    plt.close(fig)

    print(f"Baseline: k=5, n_init={args.n_init}, silhouette={summary['silhouette']:.4f}")
    print("Best k by silhouette for each n_init:")
    print(
        sensitivity.loc[sensitivity.groupby("n_init")["silhouette"].idxmax(),
                        ["n_init", "k", "silhouette"]].to_string(index=False)
    )
    print(f"Wrote figures and tables to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
