"""
Pydantic Schemas for FastAPI Diabetes Prediction API.
Includes strict clinical range validations and physiological constraints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DiabetesPredictionInput(BaseModel):
    """
    Input schema representing the 8 diagnostic features
    from the Pima Indians Diabetes Dataset.
    """
    pregnancies: int = Field(
        default=1,
        ge=0,
        le=20,
        description="Number of times pregnant (0 for nulliparous, up to 20)"
    )
    glucose: float = Field(
        default=120.0,
        ge=0.0,
        le=500.0,
        description="2-hour oral glucose tolerance test plasma glucose (mg/dL). Zero indicates unmeasured."
    )
    blood_pressure: float = Field(
        default=70.0,
        ge=0.0,
        le=220.0,
        description="Diastolic blood pressure (mm Hg). Zero indicates unmeasured."
    )
    skin_thickness: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Triceps skin fold thickness (mm). Zero indicates unmeasured."
    )
    insulin: float = Field(
        default=85.0,
        ge=0.0,
        le=900.0,
        description="2-Hour serum insulin (mu U/ml). Zero indicates unmeasured."
    )
    bmi: float = Field(
        default=28.5,
        ge=0.0,
        le=75.0,
        description="Body mass index (weight in kg / (height in m)^2). Zero indicates unmeasured."
    )
    diabetes_pedigree_function: float = Field(
        default=0.45,
        ge=0.0,
        le=3.0,
        description="Diabetes pedigree function scoring family history."
    )
    age: int = Field(
        default=35,
        ge=21,
        le=120,
        description="Patient age in years (cohort inclusion requires >= 21 years old)."
    )

    @field_validator("age")
    @classmethod
    def validate_minimum_cohort_age(cls, v: int) -> int:
        if v < 21:
            raise ValueError(
                "Age must be >= 21 years old. The underlying Pima Indians cohort was restricted to adult females aged 21 and above."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "pregnancies": 2,
                "glucose": 138.0,
                "blood_pressure": 72.0,
                "skin_thickness": 35.0,
                "insulin": 120.0,
                "bmi": 32.4,
                "diabetes_pedigree_function": 0.52,
                "age": 42
            }
        }
    }


class ContributingFactor(BaseModel):
    feature: str
    value: float
    status: str
    clinical_note: str


class DiabetesPredictionOutput(BaseModel):
    """
    Response schema delivering prediction results, probability,
    risk level, contributing clinical indicators, and mandatory disclaimers.
    """
    prediction: int = Field(description="Binary outcome: 0 (Non-Diabetic) or 1 (Diabetic)")
    class_label: str = Field(description="Human-readable diagnosis label")
    probability: float = Field(description="Calibrated probability of diabetes onset (0.0 to 1.0)")
    risk_category: str = Field(description="Clinical risk tier: Low Risk, Moderate Risk, or High Risk")
    confidence_score: float = Field(description="Confidence percentage (0% to 100%)")
    contributing_factors: List[ContributingFactor] = Field(description="Identified physiological risk indicators")
    model_version: str = Field(default="1.0.0")
    clinical_safety_notice: str = Field(
        default=(
            "ACADEMIC DEMONSTRATION ONLY: This tool is an educational machine learning demonstration "
            "trained on the historical Pima Indians Diabetes dataset (1990). It does not provide medical "
            "advice, clinical diagnoses, or treatment recommendations. Always consult a licensed physician."
        )
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    architecture: str


class ModelBenchmarkRow(BaseModel):
    model_name: str
    cv_roc_auc: float
    cv_accuracy: float
    cv_f1: float
    accuracy: float
    precision: float
    recall_sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: float


class ModelInfoResponse(BaseModel):
    model_name: str
    target_variable: str
    features: List[str]
    invalid_zero_features_handled: List[str]
    imputation_strategy: str
    feature_scaling: str
    benchmark_comparison: List[ModelBenchmarkRow]
    best_model_test_metrics: Dict[str, Any]
    best_model_cv_metrics: Dict[str, Any]
    dataset_demographics: str
    clinical_safety_notice: str
