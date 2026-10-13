import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn.config import RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy
from churn.evaluate import classification_report
from churn.features import add_features, build_preprocessor


def make_pipeline(model, engineered: bool) -> Pipeline:
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_features)))
    steps += [("prep", build_preprocessor(engineered)), ("clf", model)]
    return Pipeline(steps)


def candidates() -> dict[str, Pipeline]:
    return {
        "logreg (baseline)": make_pipeline(LogisticRegression(max_iter=1000), engineered=False),
        "logreg + features": make_pipeline(LogisticRegression(max_iter=1000), engineered=True),
        "random forest + features": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
            ),
            engineered=True,
        ),
        "gradient boosting + features": make_pipeline(
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=RANDOM_STATE),
            engineered=True,
        ),
    }


def main() -> None:
    X, y = split_xy(load_data())
    # Same split and seed as the baseline, so the test set is identical
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, pipe in candidates().items():
        print(f"Training {name} ...")
        prob = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        results[name] = classification_report(y_train, prob)

    table = pd.DataFrame(results).T.round(3)
    cols = ["roc_auc", "pr_auc", "recall", "precision", "churners_caught_top20pct"]
    print("\n5-fold CV on the training set:\n")
    print(table[cols].sort_values("roc_auc", ascending=False).to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
