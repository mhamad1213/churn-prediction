import pytest

from churn.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(), reason="run python -m churn.tune first"
)

RISKY = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": 160.0,
}
SAFE = {**RISKY, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.0, "TotalCharges": 3600.0}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from churn.api import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_risky_customer_gets_contacted_with_reasons(client):
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] > 0.7
    assert body["contact_customer"] is True
    assert body["risk_level"] == "high"
    assert {x["feature"] for x in body["top_reasons"]} & {"tenure", "Contract"}


def test_loyal_customer_is_low_risk(client):
    body = client.post("/predict", json=SAFE).json()
    assert body["churn_probability"] < 0.2
    assert body["contact_customer"] is False


def test_invalid_input_is_rejected(client):
    r = client.post("/predict", json={**RISKY, "Contract": "Weekly"})
    assert r.status_code == 422  # 422 = "your data doesn't match the rules"


def test_home_page_is_served(client):
    assert "Churn Predictor" in client.get("/").text


def test_model_info_says_where_the_model_came_from(client):
    body = client.get("/model-info").json()
    assert body["source"] == "models/best_model.joblib"
    assert body["threshold"] == 0.43
