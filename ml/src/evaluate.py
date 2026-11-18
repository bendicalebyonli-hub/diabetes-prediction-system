"""
Evaluation & Visualization Module for Diabetes Prediction Models.
Computes comprehensive biomedical classification metrics, Confusion Matrices,
and ROC-AUC curves.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation_plots"


def compute_comprehensive_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Calculates essential clinical machine learning metrics:
    Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def plot_roc_curves(model_predictions: dict, y_test, output_dir: Path = ARTIFACTS_DIR):
    """
    Plots multi-model ROC Curves side-by-side with AUC scores.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#2563eb", "Random Forest": "#059669", "Support Vector Machine": "#d97706"}

    for name, data in model_predictions.items():
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, "#4b5563"), lw=2.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    roc_path = output_dir / "roc_curves_comparison.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    return roc_path


def plot_confusion_matrices(model_predictions: dict, y_test, output_dir: Path = ARTIFACTS_DIR):
    """
    Plots individual confusion matrices for each evaluated model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = list(model_predictions.keys())

    for idx, name in enumerate(names):
        ax = axes[idx]
        y_pred = model_predictions[name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
            yticklabels=["Non-Diabetic (0)", "Diabetic (1)"]
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted Diagnosis", fontweight="bold")
        ax.set_ylabel("Actual Diagnosis", fontweight="bold")

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix_comparison.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    return cm_path


def plot_metrics_comparison_bar(comparison_df: pd.DataFrame, output_dir: Path = ARTIFACTS_DIR):
    """
    Generates a grouped bar chart comparing all models across 5 key metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "precision", "recall_sensitivity", "f1_score", "roc_auc"]
    labels_clean = ["Accuracy", "Precision", "Recall (Sens)", "F1-Score", "ROC-AUC"]

    plot_data = []
    for _, row in comparison_df.iterrows():
        for metric, label in zip(metrics_to_plot, labels_clean):
            plot_data.append({
                "Model": row["model_name"],
                "Metric": label,
                "Score": row[metric]
            })

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=["#2563eb", "#059669", "#d97706"])
    plt.title("Comparative Performance on Unseen Test Set (n=154)", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Evaluation Score (0.0 - 1.0)", fontweight="bold")
    plt.xlabel("Evaluation Metric", fontweight="bold")
    plt.legend(title="Algorithm", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    bar_path = output_dir / "model_metrics_comparison.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    return bar_path
