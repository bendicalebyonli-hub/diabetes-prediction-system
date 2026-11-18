"""
Leakage-Free Preprocessing Module.
Custom scikit-learn transformers and pipeline builders for biomedical data.
Guarantees zero-leakage during cross-validation and testing.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Biologically invalid zero columns according to clinical literature
INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class BiologicalZeroToNan(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer that replaces zeros with NaN
    strictly on features where zero is biologically impossible/missing.
    
    Leaves non-biomedical zero variables (like Pregnancies = 0) untouched.
    """

    def __init__(self, target_columns: Optional[List[str]] = None):
        self.target_columns = target_columns or INVALID_ZERO_COLS
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in self.target_columns:
                if col in X_out.columns:
                    X_out[col] = X_out[col].replace(0, np.nan)
            return X_out.values
        else:
            # Numpy array
            X_out = np.copy(X).astype(float)
            if self.feature_names_in_:
                for col in self.target_columns:
                    if col in self.feature_names_in_:
                        idx = self.feature_names_in_.index(col)
                        X_out[:, idx] = np.where(X_out[:, idx] == 0, np.nan, X_out[:, idx])
            else:
                # Default index mapping based on standard dataset order
                # [Pregnancies(0), Glucose(1), BP(2), Skin(3), Insulin(4), BMI(5), DPF(6), Age(7)]
                zero_indices = [1, 2, 3, 4, 5]
                for idx in zero_indices:
                    if idx < X_out.shape[1]:
                        X_out[:, idx] = np.where(X_out[:, idx] == 0, np.nan, X_out[:, idx])
            return X_out


def build_preprocessing_pipeline(
    imputer_strategy: str = "median",
    scale_features: bool = True
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn preprocessing pipeline.
    
    Pipeline Steps:
    1. BiologicalZeroToNan: Converts physiologically invalid zeros to NaNs.
    2. SimpleImputer: Imputes NaNs using training-fold medians (robust against right-skew).
    3. StandardScaler: Normalizes all features to zero mean and unit variance.
    
    Returns:
        Pipeline: Preprocessing pipeline without estimator.
    """
    steps = [
        ("zero_replacer", BiologicalZeroToNan(target_columns=INVALID_ZERO_COLS)),
        ("imputer", SimpleImputer(strategy=imputer_strategy))
    ]

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def create_full_model_pipeline(
    estimator,
    imputer_strategy: str = "median",
    scale_features: bool = True
) -> Pipeline:
    """
    Wraps preprocessing and a machine learning classifier into a single,
    leakage-free, serializable pipeline.
    """
    preprocessor = build_preprocessing_pipeline(
        imputer_strategy=imputer_strategy,
        scale_features=scale_features
    )
    
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ]
    )
    return full_pipeline


if __name__ == "__main__":
    # Smoke test on mock biomedical data
    dummy_df = pd.DataFrame({
        "Pregnancies": [0, 2],
        "Glucose": [0, 120],       # 0 is invalid
        "BloodPressure": [70, 0],  # 0 is invalid
        "SkinThickness": [0, 25],  # 0 is invalid
        "Insulin": [0, 100],       # 0 is invalid
        "BMI": [0.0, 24.5],        # 0 is invalid
        "DiabetesPedigreeFunction": [0.5, 0.2],
        "Age": [25, 45]
    })
    pipe = build_preprocessing_pipeline()
    transformed = pipe.fit_transform(dummy_df)
    print("Preprocessing smoke test passed. Transformed shape:", transformed.shape)
