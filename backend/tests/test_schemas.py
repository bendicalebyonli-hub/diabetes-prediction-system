"""
Unit tests for Pydantic input schemas and clinical validation.
"""

import pytest
from pydantic import ValidationError
from backend.app.schemas import DiabetesPredictionInput


def test_valid_input_passes():
    """Test that standard physiological values pass validation"""
    valid_data = {
        "pregnancies": 3,
        "glucose": 130.0,
        "blood_pressure": 76.0,
        "skin_thickness": 28.0,
        "insulin": 95.0,
        "bmi": 29.4,
        "diabetes_pedigree_function": 0.45,
        "age": 38
    }
    obj = DiabetesPredictionInput(**valid_data)
    assert obj.age == 38
    assert obj.glucose == 130.0
    assert obj.pregnancies == 3


def test_age_below_cohort_minimum_fails():
    """Test that age < 21 raises a ValidationError reflecting dataset criteria"""
    invalid_data = {
        "pregnancies": 0,
        "glucose": 100.0,
        "blood_pressure": 70.0,
        "skin_thickness": 20.0,
        "insulin": 80.0,
        "bmi": 22.0,
        "diabetes_pedigree_function": 0.3,
        "age": 19  # Below 21
    }
    with pytest.raises(ValidationError) as exc_info:
        DiabetesPredictionInput(**invalid_data)
    assert "Age must be >= 21 years old" in str(exc_info.value)


def test_negative_values_fail():
    """Test that negative measurements for physical features raise ValidationError"""
    with pytest.raises(ValidationError):
        DiabetesPredictionInput(
            pregnancies=-1,
            glucose=100.0,
            blood_pressure=70.0,
            skin_thickness=20.0,
            insulin=80.0,
            bmi=24.0,
            diabetes_pedigree_function=0.5,
            age=30
        )


def test_zero_measurements_are_accepted_for_imputation():
    """Test that zero is accepted for glucose/bp/insulin so pipeline can impute them"""
    zero_data = {
        "pregnancies": 0,
        "glucose": 0.0,  # Zero indicating missing
        "blood_pressure": 0.0,
        "skin_thickness": 0.0,
        "insulin": 0.0,
        "bmi": 0.0,
        "diabetes_pedigree_function": 0.1,
        "age": 25
    }
    obj = DiabetesPredictionInput(**zero_data)
    assert obj.glucose == 0.0
    assert obj.blood_pressure == 0.0
