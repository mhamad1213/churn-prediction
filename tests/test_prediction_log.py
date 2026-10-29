import pytest
from test_api import RISKY, SAFE

from churn import prediction_log
from churn.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(), reason="run python -m churn.tune first"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh, empty log for each test."""
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path / 'predictions.db'}")
    from fastapi.testclient import TestClient

    from churn.api import app
    return TestClient(app)


def test_neon_url_uses_the_psycopg_driver(monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", "postgresql://u:p@host/db?sslmode=require")
    assert prediction_log.database_url() == "postgresql+psycopg://u:p@host/db?sslmode=require"


def test_every_prediction_is_logged(client):
    risky = client.post("/predict", json=RISKY).json()
    client.post("/predict", json=SAFE)

    df = prediction_log.load_predictions()
    assert len(df) == 2
    first = df.iloc[0]
    assert first["Contract"] == "Month-to-month"
    assert first["tenure"] == RISKY["tenure"]
    assert first["churn_probability"] == pytest.approx(risky["churn_probability"], abs=1e-3)
    assert bool(first["contact_customer"]) is True
    assert first["model_source"] == "models/best_model.joblib"


def test_invalid_requests_are_not_logged(client):
    client.post("/predict", json={**RISKY, "Contract": "Weekly"})
    assert prediction_log.summary()["predictions"] == 0


def test_monitoring_summary(client):
    assert client.get("/monitoring/summary").json()["predictions"] == 0
    for _ in range(3):
        client.post("/predict", json=RISKY)
    client.post("/predict", json=SAFE)

    body = client.get("/monitoring/summary").json()
    assert body["predictions"] == 4
    assert body["contact_rate"] == 0.75
    assert body["model"] == "models/best_model.joblib"


def test_a_broken_database_never_breaks_predictions(client, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", "postgresql://nobody:wrong@127.0.0.1:1/nothing")
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
