"""Track B1: record the trained model in MLflow and register it.

Reads what `churn.tune` and `churn.threshold` already produced (no retraining),
logs it as one MLflow run with a child run per candidate model, and registers
the final model as "churn-model". The new version becomes @champion only if
its test PR-AUC is at least as good as the current champion's.

Run:     python -m churn.track
Browse:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
"""

import json
import os
from datetime import datetime, timezone

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.model_selection import train_test_split

from churn.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, ROOT
from churn.data import load_data, split_xy

EXPERIMENT = "churn-prediction"
MODEL_NAME = "churn-model"
CHAMPION = "champion"

# MLflow saves sklearn models with skops, a safer format than pickle, which refuses
# to save code it hasn't been told to trust. Both of these are ours or sklearn's.
TRUSTED_TYPES = [
    "churn.features.add_features",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def tracking_uri() -> str:
    """Use MLFLOW_TRACKING_URI if set, otherwise a local SQLite file in the project."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def setup() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        # keep the saved models inside the project folder, next to mlflow.db
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def champion_pr_auc(client: MlflowClient) -> float | None:
    """Test PR-AUC of the current @champion, or None if there isn't one yet."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics["test_pr_auc"]


def should_promote(new_pr_auc: float, champion: float | None) -> bool:
    return champion is None or new_pr_auc >= champion


def main() -> None:
    setup()
    tuning = json.loads((REPORTS_DIR / "tuning_results.json").read_text())
    threshold = json.loads((MODELS_DIR / "threshold.json").read_text())
    strategies = json.loads((REPORTS_DIR / "threshold_test_results.json").read_text())
    best = next(s for s in strategies if s["strategy"].startswith("best"))
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    X, y = split_xy(load_data())
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    run_name = f"tune-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"
    with mlflow.start_run(run_name=run_name) as run:
        # 1. One child run per candidate, so the UI can compare them side by side
        candidates = {**tuning["step_a"], "gradient boosting (tuned)": tuning["tuned_cv"]}
        for name, metrics in candidates.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_metrics({f"cv_{k}": v for k, v in metrics.items()})

        # 2. The winner: settings, scores, business result and plots
        mlflow.set_tag("winner", tuning["winner"])
        mlflow.log_params(tuning["best_params"])
        mlflow.log_params({"threshold": threshold["threshold"], **threshold["assumptions"]})
        mlflow.log_metrics({f"cv_{k}": v for k, v in tuning["tuned_cv"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in tuning["test"].items()})
        mlflow.log_metrics({"test_profit": best["profit"], "test_recall_at_threshold": best["recall"]})
        mlflow.log_artifacts(str(REPORTS_DIR / "figures"), artifact_path="figures")

        # 3. Save the model itself and register it as a new version
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )

    # 4. Promote to @champion only if it's at least as good as the current one
    client = MlflowClient()
    version = info.registered_model_version
    current = champion_pr_auc(client)
    new = tuning["test"]["pr_auc"]
    if should_promote(new, current):
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        verdict = "promoted to @champion"
    else:
        verdict = f"kept the old champion (its PR-AUC {current:.3f} beats {new:.3f})"

    print(f"Logged run '{run_name}' ({run.info.run_id[:8]}) with {len(candidates)} candidate runs")
    print(f"Registered {MODEL_NAME} version {version}: test PR-AUC {new:.3f}, "
          f"profit £{best['profit']:,.0f} at threshold {threshold['threshold']}")
    print(f"Version {version} {verdict}")


if __name__ == "__main__":
    main()
