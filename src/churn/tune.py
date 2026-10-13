"""
1. Compare each model with and without class_weight="balanced".
2. Tune the gradient boosting model with RandomizedSearchCV.
3. Pick the best model by cross-validated PR-AUC, then score it ONCE on the
   held-out test set and save it.

Run: python -m churn.tune
"""

import json

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from churn.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy
from churn.evaluate import classification_report
from churn.train_models import make_pipeline

COLS = ["roc_auc", "pr_auc", "recall", "precision", "churners_caught_top20pct"]


def cv_report(pipe, X, y, cv) -> dict[str, float]:
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return classification_report(y, prob)


def main() -> None:
    X, y = split_xy(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Step A: class imbalance -------------------------------------------
    print("Step A: effect of class_weight='balanced'\n")
    models = {}
    for weight in [None, "balanced"]:
        tag = "balanced" if weight else "default"
        models[f"logreg ({tag})"] = make_pipeline(
            LogisticRegression(max_iter=1000, class_weight=weight), engineered=True
        )
        models[f"gradient boosting ({tag})"] = make_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, class_weight=weight, random_state=RANDOM_STATE
            ),
            engineered=True,
        )
    step_a = {name: cv_report(pipe, X_train, y_train, cv) for name, pipe in models.items()}
    print(pd.DataFrame(step_a).T[COLS].round(3).to_string(), "\n")

    # --- Step B: hyperparameter tuning --------------------------------------
    print("Step B: tuning gradient boosting (40 random combinations x 5 folds)...\n")
    search = RandomizedSearchCV(
        make_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            engineered=True,
        ),
        param_distributions={
            "clf__learning_rate": loguniform(0.01, 0.2),
            "clf__max_iter": randint(100, 500),
            "clf__max_depth": randint(2, 6),
            "clf__min_samples_leaf": randint(10, 100),
            "clf__l2_regularization": uniform(0, 2),
        },
        n_iter=40,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print("Best settings:", {k: round(float(v), 3) for k, v in best_params.items()})

    tuned = cv_report(search.best_estimator_, X_train, y_train, cv)
    final_candidates = {
        "logreg (balanced)": (models["logreg (balanced)"], step_a["logreg (balanced)"]),
        "gradient boosting (tuned)": (search.best_estimator_, tuned),
    }
    print(
        pd.DataFrame({k: v[1] for k, v in final_candidates.items()}).T[COLS].round(3).to_string(),
        "\n",
    )

    # --- Step C: pick winner on CV, score once on the test set ---------------
    winner = max(final_candidates, key=lambda k: final_candidates[k][1]["pr_auc"])
    model = final_candidates[winner][0].fit(X_train, y_train)
    test = classification_report(y_test, model.predict_proba(X_test)[:, 1])
    baseline = json.loads((REPORTS_DIR / "baseline_metrics.json").read_text())["test"]

    print(f"Winner (by CV PR-AUC): {winner}\n")
    print("Held-out test set, baseline vs winner:")
    print(pd.DataFrame({"baseline": baseline, winner: test}).T[COLS].round(3).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    (REPORTS_DIR / "tuning_results.json").write_text(
        json.dumps(
            {"step_a": step_a, "best_params": best_params, "tuned_cv": tuned,
             "winner": winner, "test": test},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
