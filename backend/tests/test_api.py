"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds with status and safety notice"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "disclaimer" in data


def test_health_endpoint():
    """Verify health endpoint confirms service and model status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    """Verify model-info endpoint returns architecture and benchmark metrics"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "benchmark_comparison" in data
    assert len(data["features"]) == 8


def test_predict_endpoint_healthy_profile():
    """Verify prediction endpoint with a typical low-risk patient profile"""
    payload = {
        "pregnancies": 1,
        "glucose": 85.0,
        "blood_pressure": 66.0,
        "skin_thickness": 29.0,
        "insulin": 0.0,  # Unmeasured
        "bmi": 26.6,
        "diabetes_pedigree_function": 0.35,
        "age": 31
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_category" in data
    assert "clinical_safety_notice" in data
    assert data["prediction"] in (0, 1)


def test_predict_endpoint_high_risk_profile():
    """Verify prediction endpoint with a diabetic clinical profile"""
    payload = {
        "pregnancies": 6,
        "glucose": 180.0,
        "blood_pressure": 92.0,
        "skin_thickness": 38.0,
        "insulin": 220.0,
        "bmi": 38.5,
        "diabetes_pedigree_function": 1.2,
        "age": 52
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] >= 0.5
    assert data["risk_category"] == "High Risk"
    assert len(data["contributing_factors"]) > 0


def test_predict_endpoint_invalid_age():
    """Verify validation rejection on age below study cohort boundary (<21)"""
    payload = {
        "pregnancies": 0,
        "glucose": 110.0,
        "blood_pressure": 70.0,
        "skin_thickness": 20.0,
        "insulin": 75.0,
        "bmi": 23.0,
        "diabetes_pedigree_function": 0.3,
        "age": 18
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
