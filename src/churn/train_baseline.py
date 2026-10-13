"""Milestone 2: logistic regression baseline.

Holds out a stratified 20% test set (kept untouched for the final model
comparison) and reports 5-fold CV metrics on the training portion.

Run: python -m churn.train_baseline
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy
from churn.evaluate import classification_report
from churn.features import build_preprocessor


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    X, y = split_xy(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_prob = cross_val_predict(build_model(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cv_metrics = classification_report(y_train, cv_prob)

    model = build_model().fit(X_train, y_train)
    test_metrics = classification_report(y_test, model.predict_proba(X_test)[:, 1])

    majority = classification_report(y_test, np.zeros(len(y_test)))["accuracy"]
    results = {
        "model": "logistic_regression_baseline",
        "cv_5fold": cv_metrics,
        "test": test_metrics,
        "always_no_churn_accuracy": majority,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline_logreg.joblib")
    (REPORTS_DIR / "baseline_metrics.json").write_text(json.dumps(results, indent=2))

    print(pd.DataFrame({"cv_5fold": cv_metrics, "test": test_metrics}).round(3))
    print(f"\nAccuracy of always predicting 'no churn': {majority:.3f}")


if __name__ == "__main__":
    main()
