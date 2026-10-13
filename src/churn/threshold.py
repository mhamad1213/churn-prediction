import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy

# Business assumptions (made up but realistic; change them and re-run)
OFFER_COST = 20  # £ cost of a retention offer (discount, call time) per contacted customer
CUSTOMER_VALUE = 300  # £ revenue lost when a customer churns
OFFER_SUCCESS_RATE = 0.30  # share of contacted churners the offer actually keeps


def profit_at(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Expected £ profit of contacting everyone with probability >= threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_prob) >= threshold
    caught = int((flagged & (y_true == 1)).sum())
    false_alarms = int((flagged & (y_true == 0)).sum())
    saved_revenue = caught * OFFER_SUCCESS_RATE * CUSTOMER_VALUE
    cost = flagged.sum() * OFFER_COST
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "churners_caught": caught,
        "false_alarms": false_alarms,
        "recall": caught / y_true.sum(),
        "precision": caught / flagged.sum() if flagged.sum() else 0.0,
        "profit": saved_revenue - cost,
    }


def profit_curve(y_true, y_prob) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    return pd.DataFrame([profit_at(y_true, y_prob, t) for t in thresholds])


def main() -> None:
    X, y = split_xy(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    # 1. Choose the threshold on out-of-fold training predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_prob = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    curve = profit_curve(y_train, train_prob)
    best = curve.loc[curve["profit"].idxmax()]
    threshold = float(best["threshold"])

    print(f"Assumptions: offer £{OFFER_COST}, customer worth £{CUSTOMER_VALUE}, "
          f"offer works {OFFER_SUCCESS_RATE:.0%} of the time\n")
    print("Training set (cross-validated), selected thresholds:")
    show = curve[curve["threshold"].isin([0.3, 0.4, 0.5, 0.6, 0.7, threshold])]
    print(show.round(3).to_string(index=False), "\n")
    print(f"Best threshold: {threshold:.2f}\n")

    # 2. Check it once on the untouched test set
    test_prob = model.predict_proba(X_test)[:, 1]
    test_rows = pd.DataFrame([
        {"strategy": "contact nobody", **profit_at(y_test, test_prob, 1.01)},
        {"strategy": "contact everyone", **profit_at(y_test, test_prob, 0.0)},
        {"strategy": "default 0.5", **profit_at(y_test, test_prob, 0.5)},
        {"strategy": f"best {threshold:.2f}", **profit_at(y_test, test_prob, threshold)},
    ]).set_index("strategy")
    print("Held-out test set:")
    print(test_rows.drop(columns="threshold").round(3).to_string())

    # 3. Save the threshold for the API, and the profit curve plot
    (MODELS_DIR / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "assumptions": {"offer_cost": OFFER_COST, "customer_value": CUSTOMER_VALUE,
                        "offer_success_rate": OFFER_SUCCESS_RATE},
    }, indent=2))
    (REPORTS_DIR / "threshold_test_results.json").write_text(
        test_rows.reset_index().to_json(orient="records", indent=2)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(curve["threshold"], curve["profit"], color="#2a6f97")
    ax1.axvline(threshold, ls="--", color="grey", label=f"best = {threshold:.2f}")
    ax1.axvline(0.5, ls=":", color="#c0392b", label="default 0.5")
    ax1.set(title="Expected profit (training CV)", xlabel="threshold", ylabel="£ profit")
    ax1.legend()
    ax2.plot(curve["threshold"], curve["recall"], label="recall")
    ax2.plot(curve["threshold"], curve["precision"], label="precision")
    ax2.axvline(threshold, ls="--", color="grey")
    ax2.set(title="Recall vs precision", xlabel="threshold")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "threshold_profit_curve.png", dpi=120)
    print("\nSaved models/threshold.json and reports/figures/threshold_profit_curve.png")


if __name__ == "__main__":
    main()
