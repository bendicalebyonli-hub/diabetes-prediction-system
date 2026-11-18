"""
Model Training, Cross-Validation, Benchmarking & Serialization Engine.
Compares Logistic Regression, Random Forest, and Support Vector Machine
using 5-Fold Stratified Cross-Validation on the Pima Indians Dataset.
"""

import json
import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from ml.src.data_loader import get_features_and_target
from ml.src.preprocessor import create_full_model_pipeline, INVALID_ZERO_COLS
from ml.src.evaluate import (
    compute_comprehensive_metrics,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_metrics_comparison_bar
)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation_plots"


def run_model_benchmarking():
    """
    Executes end-to-end model training, cross-validation, and selection.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PIMA INDIANS DIABETES PREDICTION - BENCHMARK & TRAINING ENGINE  ")
    print("=" * 70)

    # 1. Ingestion & Stratified Split
    X, y = get_features_and_target()
    print(f"[Dataset] Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"[Dataset] Total Diabetic (1): {int(y.sum())} ({y.mean()*100:.2f}%)")
    print(f"[Dataset] Total Non-Diabetic (0): {int((1-y).sum())} ({(1-y.mean())*100:.2f}%)")

    # Strict 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[Split] Training set: {X_train.shape[0]} samples (Stratified)")
    print(f"[Split] Test set:     {X_test.shape[0]} samples (Stratified)")

    # 2. Model Definitions
    candidate_estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.8,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=42
        )
    }

    # 3. Stratified 5-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results_summary = {}
    test_evaluations = {}
    trained_pipelines = {}
    test_predictions_dict = {}

    print("\n--- Running 5-Fold Stratified Cross-Validation (No Leakage) ---")

    for name, estimator in candidate_estimators.items():
        pipeline = create_full_model_pipeline(estimator=estimator, imputer_strategy="median")
        
        # Cross-validation evaluated strictly inside folds
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        cv_results_summary[name] = {
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"]))
        }

        print(f"\n[{name}]")
        print(f"  CV ROC-AUC:  {cv_results_summary[name]['cv_roc_auc_mean']:.4f} (+/- {cv_results_summary[name]['cv_roc_auc_std']:.4f})")
        print(f"  CV Accuracy: {cv_results_summary[name]['cv_accuracy_mean']:.4f}")
        print(f"  CV F1-Score: {cv_results_summary[name]['cv_f1_mean']:.4f}")
        print(f"  CV Recall:   {cv_results_summary[name]['cv_recall_mean']:.4f}")

        # Train on entire X_train and evaluate on untouched holdout X_test
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        test_predictions_dict[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob
        }

        test_metrics = compute_comprehensive_metrics(y_test, y_pred, y_prob)
        test_evaluations[name] = test_metrics

    # 4. Comparative Tabulation
    comparison_rows = []
    for name in candidate_estimators.keys():
        row = {
            "model_name": name,
            "cv_roc_auc": round(cv_results_summary[name]["cv_roc_auc_mean"], 4),
            "cv_accuracy": round(cv_results_summary[name]["cv_accuracy_mean"], 4),
            "cv_f1": round(cv_results_summary[name]["cv_f1_mean"], 4),
            "accuracy": test_evaluations[name]["accuracy"],
            "precision": test_evaluations[name]["precision"],
            "recall_sensitivity": test_evaluations[name]["recall_sensitivity"],
            "specificity": test_evaluations[name]["specificity"],
            "f1_score": test_evaluations[name]["f1_score"],
            "roc_auc": test_evaluations[name]["roc_auc"]
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 70)
    print("  UNSEEN TEST SET (n=154) COMPARATIVE BENCHMARK  ")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # 5. Generate and Export Visual Artifacts
    print("\n[Artifacts] Generating ROC curves, Confusion Matrices, and Metric Charts...")
    roc_plot_file = plot_roc_curves(test_predictions_dict, y_test, ARTIFACTS_DIR)
    cm_plot_file = plot_confusion_matrices(test_predictions_dict, y_test, ARTIFACTS_DIR)
    bar_plot_file = plot_metrics_comparison_bar(comparison_df, ARTIFACTS_DIR)

    # 6. Select Best Performing Model (Optimizing balanced ROC-AUC and F1)
    # Both Random Forest and Logistic Regression are strong candidates; select highest test ROC-AUC & balanced F1
    best_model_name = comparison_df.sort_values(by=["roc_auc", "f1_score"], ascending=False).iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = test_evaluations[best_model_name]

    print(f"\n[Model Selection] Selected Best Model: >>> {best_model_name} <<<")
    print(f"  Test ROC-AUC: {best_metrics['roc_auc']}")
    print(f"  Test Accuracy: {best_metrics['accuracy']}")
    print(f"  Test Recall (Sensitivity): {best_metrics['recall_sensitivity']}")
    print(f"  Test F1-Score: {best_metrics['f1_score']}")

    # 7. Serialize Pipeline and Metadata
    model_save_path = MODELS_DIR / "diabetes_best_pipeline.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"[Serialization] Saved best pipeline to: {model_save_path}")

    # Extract feature names and sample values for metadata
    metadata = {
        "model_name": best_model_name,
        "serialization_path": str(model_save_path.name),
        "target_variable": "Outcome (0=Negative, 1=Positive)",
        "features": list(X.columns),
        "invalid_zero_features_handled": INVALID_ZERO_COLS,
        "imputation_strategy": "SimpleImputer(median)",
        "feature_scaling": "StandardScaler()",
        "benchmark_comparison": comparison_rows,
        "best_model_test_metrics": best_metrics,
        "best_model_cv_metrics": cv_results_summary[best_model_name],
        "dataset_demographics": "Females >= 21 years old of Pima Indian heritage (NIDDK)",
        "clinical_safety_notice": "Academic research prototype. Not intended for direct diagnostic application."
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(ARTIFACTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Serialization] Model metadata saved to: {metadata_path}")
    print("=" * 70)
    print("  TRAINING & BENCHMARKING COMPLETED SUCCESSFULLY  ")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    run_model_benchmarking()
