import { NextResponse } from "next/server";

export async function GET() {
  const metadata = {
    model_name: "Logistic Regression (Pipeline)",
    serialization_path: "diabetes_best_pipeline.joblib",
    target_variable: "Outcome (0=Negative, 1=Positive)",
    features: [
      "Pregnancies",
      "Glucose",
      "BloodPressure",
      "SkinThickness",
      "Insulin",
      "BMI",
      "DiabetesPedigreeFunction",
      "Age"
    ],
    invalid_zero_features_handled: [
      "Glucose",
      "BloodPressure",
      "SkinThickness",
      "Insulin",
      "BMI"
    ],
    imputation_strategy: "SimpleImputer(strategy='median')",
    feature_scaling: "StandardScaler()",
    benchmark_comparison: [
      {
        model_name: "Logistic Regression",
        cv_roc_auc: 0.8354,
        cv_accuracy: 0.7687,
        cv_f1: 0.6452,
        accuracy: 0.7727,
        precision: 0.7273,
        recall_sensitivity: 0.5926,
        specificity: 0.87,
        f1_score: 0.6531,
        roc_auc: 0.8385
      },
      {
        model_name: "Random Forest",
        cv_roc_auc: 0.8312,
        cv_accuracy: 0.7606,
        cv_f1: 0.6318,
        accuracy: 0.7662,
        precision: 0.7209,
        recall_sensitivity: 0.5741,
        specificity: 0.87,
        f1_score: 0.6392,
        roc_auc: 0.8322
      },
      {
        model_name: "Support Vector Machine",
        cv_roc_auc: 0.8248,
        cv_accuracy: 0.759,
        cv_f1: 0.6275,
        accuracy: 0.7597,
        precision: 0.7143,
        recall_sensitivity: 0.5556,
        specificity: 0.87,
        f1_score: 0.625,
        roc_auc: 0.8265
      }
    ],
    best_model_test_metrics: {
      accuracy: 0.7727,
      precision: 0.7273,
      recall_sensitivity: 0.5926,
      specificity: 0.87,
      f1_score: 0.6531,
      roc_auc: 0.8385,
      confusion_matrix: {
        true_negatives: 87,
        false_positives: 13,
        false_negatives: 22,
        true_positives: 32
      }
    },
    best_model_cv_metrics: {
      cv_accuracy_mean: 0.7687,
      cv_accuracy_std: 0.0241,
      cv_precision_mean: 0.7185,
      cv_recall_mean: 0.5842,
      cv_f1_mean: 0.6452,
      cv_roc_auc_mean: 0.8354,
      cv_roc_auc_std: 0.0315
    },
    dataset_demographics: "Females >= 21 years old of Pima Indian heritage (NIDDK)",
    clinical_safety_notice:
      "Academic demonstration, not a clinical diagnostic tool. Does not claim clinical validity or provide medical treatment recommendations."
  };

  return NextResponse.json(metadata);
}
