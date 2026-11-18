export interface DiabetesPredictionInput {
  pregnancies: number;
  glucose: number;
  blood_pressure: number;
  skin_thickness: number;
  insulin: number;
  bmi: number;
  diabetes_pedigree_function: number;
  age: number;
}

export interface ContributingFactor {
  feature: string;
  value: number;
  status: string;
  clinical_note: string;
}

export interface DiabetesPredictionOutput {
  prediction: number;
  class_label: string;
  probability: number;
  risk_category: "Low Risk" | "Moderate Risk" | "High Risk";
  confidence_score: number;
  contributing_factors: ContributingFactor[];
  model_version: string;
  clinical_safety_notice: string;
}

export interface ModelBenchmarkRow {
  model_name: string;
  cv_roc_auc: number;
  cv_accuracy: number;
  cv_f1: number;
  accuracy: number;
  precision: number;
  recall_sensitivity: number;
  specificity: number;
  f1_score: number;
  roc_auc: number;
}

export interface ModelInfoResponse {
  model_name: string;
  target_variable: string;
  features: string[];
  invalid_zero_features_handled: string[];
  imputation_strategy: string;
  feature_scaling: string;
  benchmark_comparison: ModelBenchmarkRow[];
  best_model_test_metrics: {
    accuracy: number;
    precision: number;
    recall_sensitivity: number;
    specificity: number;
    f1_score: number;
    roc_auc: number;
    confusion_matrix: {
      true_negatives: number;
      false_positives: number;
      false_negatives: number;
      true_positives: number;
    };
  };
  best_model_cv_metrics: {
    cv_accuracy_mean: number;
    cv_accuracy_std: number;
    cv_precision_mean: number;
    cv_recall_mean: number;
    cv_f1_mean: number;
    cv_roc_auc_mean: number;
    cv_roc_auc_std: number;
  };
  dataset_demographics: string;
  clinical_safety_notice: string;
}
