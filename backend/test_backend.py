import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "documentation" in data
    print("test_root: PASSED")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True
    print("test_health: PASSED")

def test_metadata():
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["num_features"] == 35
    assert "target_classes" in data
    assert "best_hyperparameters" in data
    print("test_metadata: PASSED")

def test_presets():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    presets = data["presets"]
    assert "Healthy Profile (Low Risk)" in presets
    assert "Pre-Diabetic Profile (Moderate Risk)" in presets
    assert "Diabetic Profile (High Risk)" in presets
    print("test_presets: PASSED")

def test_predictions():
    presets_resp = client.get("/api/presets")
    presets = presets_resp.json()["presets"]

    expected_outcomes = {
        "Healthy Profile (Low Risk)": {"code": 0, "label": "No Diabetes"},
        "Pre-Diabetic Profile (Moderate Risk)": {"code": 1, "label": "Pre-Diabetes"},
        "Diabetic Profile (High Risk)": {"code": 2, "label": "Diabetes"},
    }

    for preset_name, expected in expected_outcomes.items():
        patient_data = presets[preset_name]
        response = client.post("/api/predict", json=patient_data)
        assert response.status_code == 200, f"Failed on {preset_name}: {response.text}"
        result = response.json()

        assert result["status"] == "success"
        assert result["prediction_code"] == expected["code"], (
            f"Expected code {expected['code']} for {preset_name}, got {result['prediction_code']}"
        )
        assert result["prediction_label"] == expected["label"], (
            f"Expected label {expected['label']} for {preset_name}, got {result['prediction_label']}"
        )
        assert "confidence" in result
        assert "probabilities" in result
        assert len(result["probabilities"]) == 3
        assert "risk_category" in result
        assert "risk_color" in result
        assert len(result["recommendations"]) > 0

        print(
            f"test_prediction [{preset_name}]: PREDICTED '{result['prediction_label']}' "
            f"(Confidence: {result['confidence']*100:.1f}%, Risk: {result['risk_category']}) - PASSED"
        )

def test_validation_error():
    invalid_payload = {
        "age": -5,
        "gender": "UnknownGender",
        "glucose_fasting": 500,
    }
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 422
    print("test_validation_error: PASSED (Properly rejected with HTTP 422)")

if __name__ == "__main__":
    print("\n--- Starting FastAPI Backend Automated Tests ---")
    test_root()
    test_health()
    test_metadata()
    test_presets()
    test_predictions()
    test_validation_error()
    print("\n--- ALL BACKEND TESTS PASSED SUCCESSFULLY! ---\n")
