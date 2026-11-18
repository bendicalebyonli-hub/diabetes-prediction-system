"""
Exploratory Data Analysis (EDA) Module.
Investigates the Pima Indians Diabetes dataset, calculates statistical properties,
identifies invalid biological zero values, and exports visualizations.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ml.src.data_loader import load_raw_data

# Features where 0 indicates missing/unrecorded data, not a biological zero
INVALID_ZERO_FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
VALID_ZERO_FEATURES = ["Pregnancies"]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "eda_plots"


def analyze_missing_and_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes zero count, percentage, and clinical validity for each attribute.
    """
    records = []
    total_rows = len(df)

    for col in df.columns:
        if col == "Outcome":
            continue
        zero_count = int((df[col] == 0).sum())
        zero_pct = round((zero_count / total_rows) * 100, 2)
        is_invalid = col in INVALID_ZERO_FEATURES

        records.append({
            "Feature": col,
            "Total Records": total_rows,
            "Zero Count": zero_count,
            "Zero Percentage (%)": zero_pct,
            "Physiologically Valid 0?": "No (Missing Measurement)" if is_invalid else "Yes (Valid Count)",
            "Status": "REQUIRES_IMPUTATION" if is_invalid and zero_count > 0 else "VALID"
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def generate_eda_visualizations(df: pd.DataFrame, output_dir: Path = ARTIFACTS_DIR) -> dict:
    """
    Generates academic publication-quality EDA figures:
    1. Zero-value prevalence chart
    2. Distribution histograms with KDE
    3. Correlation matrix heatmap
    4. Target class balance breakdown
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. Missing Zero-Value Plot
    zero_df = analyze_missing_and_zero_values(df)
    plt.figure(figsize=(10, 5))
    colors = ["#dc2626" if status == "REQUIRES_IMPUTATION" else "#2563eb" for status in zero_df["Status"]]
    bars = plt.bar(zero_df["Feature"], zero_df["Zero Percentage (%)"], color=colors, edgecolor="black", alpha=0.85)
    plt.title("Prevalence of Zero Measurements (Red = Biologically Invalid / Missing)", fontsize=13, fontweight="bold")
    plt.xlabel("Clinical Feature", fontweight="bold")
    plt.ylabel("Percentage of Records with Zero (%)", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 55)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    zero_plot_path = output_dir / "missing_zeros_bar.png"
    plt.savefig(zero_plot_path, dpi=300)
    plt.close()

    # 2. Distribution Histograms with KDE (replacing invalid zeros with NaN for accurate visualization)
    df_imputed_view = df.copy()
    for col in INVALID_ZERO_FEATURES:
        df_imputed_view[col] = df_imputed_view[col].replace(0, np.nan)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    feature_cols = [c for c in df.columns if c != "Outcome"]

    for i, col in enumerate(feature_cols):
        ax = axes[i // 3, i % 3]
        sns.histplot(df_imputed_view[col].dropna(), kde=True, ax=ax, color="#0284c7", edgecolor="white")
        ax.set_title(f"Distribution: {col}", fontweight="bold", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Target outcome in the last quadrant
    ax_outcome = axes[2, 2]
    sns.countplot(x="Outcome", data=df, ax=ax_outcome, palette=["#10b981", "#ef4444"])
    ax_outcome.set_title("Target Distribution: Outcome", fontweight="bold", fontsize=11)
    ax_outcome.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
    ax_outcome.set_ylabel("Count")

    dist_plot_path = output_dir / "feature_distributions.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()

    # 3. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    corr = df_imputed_view.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix (With Invalid Zeros Excluded from Pairwise Stats)", fontsize=13, fontweight="bold")
    corr_plot_path = output_dir / "correlation_matrix.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()

    # 4. Target Class Balance
    plt.figure(figsize=(6, 5))
    outcome_counts = df["Outcome"].value_counts()
    outcome_pcts = df["Outcome"].value_counts(normalize=True) * 100
    labels = [f"0: Non-Diabetic\n(n={outcome_counts[0]}, {outcome_pcts[0]:.1f}%)",
              f"1: Diabetic\n(n={outcome_counts[1]}, {outcome_pcts[1]:.1f}%)"]
    plt.pie(outcome_counts, labels=labels, colors=["#10b981", "#ef4444"], autopct="", startangle=140,
            wedgeprops={"edgecolor": "black", "linewidth": 1})
    plt.title("Class Balance (Pima Indians Outcome)", fontweight="bold", fontsize=12)
    class_plot_path = output_dir / "class_balance.png"
    plt.savefig(class_plot_path, dpi=300)
    plt.close()

    # Summary metrics dictionary
    summary = {
        "total_records": int(len(df)),
        "class_distribution": {
            "non_diabetic_0": int(outcome_counts[0]),
            "diabetic_1": int(outcome_counts[1]),
            "prevalence_rate_pct": float(round(outcome_pcts[1], 2))
        },
        "zero_value_analysis": zero_df.to_dict(orient="records"),
        "top_outcome_correlations": corr["Outcome"].drop("Outcome").sort_values(ascending=False).to_dict()
    }

    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_eda():
    """Execution entrypoint for EDA"""
    df = load_raw_data()
    print("--- [EDA] Dataset Loaded Successfully ---")
    print(f"Total instances: {len(df)}")
    print("\n--- [EDA] Missing / Invalid Zero Analysis ---")
    zero_df = analyze_missing_and_zero_values(df)
    print(zero_df.to_string(index=False))

    print("\n--- [EDA] Generating Plots & Artifacts ---")
    summary = generate_eda_visualizations(df)
    print(f"EDA Artifacts successfully exported to: {ARTIFACTS_DIR}")
    return summary


if __name__ == "__main__":
    run_eda()
