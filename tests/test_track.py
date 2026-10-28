import json

import joblib
import mlflow
import pytest

from churn import track
from churn.config import MODELS_DIR, REPORTS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(), reason="run tune + threshold first"
)


def test_promote_only_when_at_least_as_good():
    assert track.should_promote(0.66, None)
    assert track.should_promote(0.67, 0.66)
    assert not track.should_promote(0.65, 0.66)


def test_track_registers_a_loadable_champion(tmp_path, monkeypatch):
    # point MLflow at a throwaway folder so the test never touches your real mlflow.db
    monkeypatch.setattr(track, "ROOT", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    track.main()
    track.main()

    client = mlflow.MlflowClient()
    assert str(client.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION).version) == "2"

    from churn.data import load_data, split_xy

    X, _ = split_xy(load_data())
    registered = mlflow.sklearn.load_model(f"models:/{track.MODEL_NAME}@{track.CHAMPION}")
    original = joblib.load(MODELS_DIR / "best_model.joblib")
    assert (registered.predict_proba(X.head(50)) == original.predict_proba(X.head(50))).all()

    test_pr_auc = json.loads((REPORTS_DIR / "tuning_results.json").read_text())["test"]["pr_auc"]
    run_id = client.get_model_version(track.MODEL_NAME, "2").run_id
    assert client.get_run(run_id).data.metrics["test_pr_auc"] == pytest.approx(test_pr_auc)
