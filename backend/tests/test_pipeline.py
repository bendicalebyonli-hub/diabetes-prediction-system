"""
Unit tests for preprocessing pipeline and leakage prevention.
"""

import numpy as np
import pandas as pd
from ml.src.preprocessor import (
    BiologicalZeroToNan,
    build_preprocessing_pipeline,
    INVALID_ZERO_COLS
)


def test_zero_replacer_transformer():
    """Verify that zeros are converted to NaNs only on biological features"""
    df = pd.DataFrame({
        "Pregnancies": [0, 2],
        "Glucose": [0, 140],
        "BloodPressure": [80, 0],
        "SkinThickness": [0, 30],
        "Insulin": [0, 120],
        "BMI": [0.0, 28.0],
        "DiabetesPedigreeFunction": [0.5, 0.4],
        "Age": [22, 40]
    })
    transformer = BiologicalZeroToNan(target_columns=INVALID_ZERO_COLS)
    transformed = transformer.fit_transform(df)

    # Pregnancies (col index 0) should retain 0
    assert transformed[0, 0] == 0.0
    # Glucose (col index 1) was 0, must now be NaN
    assert np.isnan(transformed[0, 1])
    # BloodPressure (col index 2) was 0 on row 1, must be NaN
    assert np.isnan(transformed[1, 2])


def test_preprocessing_pipeline_no_nan_output():
    """Verify that full preprocessing pipeline imputes and standardizes cleanly"""
    df = pd.DataFrame({
        "Pregnancies": [0, 1, 3],
        "Glucose": [0, 120, 160],
        "BloodPressure": [70, 0, 84],
        "SkinThickness": [20, 0, 35],
        "Insulin": [80, 0, 200],
        "BMI": [22.0, 0.0, 34.0],
        "DiabetesPedigreeFunction": [0.2, 0.5, 0.8],
        "Age": [25, 35, 55]
    })
    pipe = build_preprocessing_pipeline(imputer_strategy="median", scale_features=True)
    out = pipe.fit_transform(df)

    # Should have no NaNs remaining
    assert not np.isnan(out).any()
    assert out.shape == (3, 8)
    # Check mean is approximately 0 (StandardScaler property)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-7)
