import numpy as np
import pandas as pd
import pytest

from churn import drift

rng = np.random.default_rng(0)
CUSTOMERS = pd.DataFrame({
    "tenure": rng.integers(0, 72, 3000).astype(float),
    "MonthlyCharges": rng.normal(65, 30, 3000).clip(18, 120),
    "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 3000, p=[0.55, 0.21, 0.24]),
})


def test_same_data_has_no_drift():
    assert drift.psi_numeric(CUSTOMERS["MonthlyCharges"], CUSTOMERS["MonthlyCharges"]) == pytest.approx(0)
    assert drift.psi_categorical(CUSTOMERS["Contract"], CUSTOMERS["Contract"]) == pytest.approx(0)


def test_a_fresh_sample_from_the_same_customers_is_stable():
    sample = CUSTOMERS.sample(500, random_state=1)
    assert drift.psi_numeric(CUSTOMERS["tenure"], sample["tenure"]) < drift.WATCH
    assert drift.psi_categorical(CUSTOMERS["Contract"], sample["Contract"]) < drift.WATCH


def test_a_price_rise_is_major_drift():
    raised = CUSTOMERS["MonthlyCharges"] * 1.3
    assert drift.psi_numeric(CUSTOMERS["MonthlyCharges"], raised) > drift.MAJOR


def test_a_new_category_counts_as_drift():
    weekly = pd.Series(["Weekly"] * 300 + ["Month-to-month"] * 200)
    assert drift.psi_categorical(CUSTOMERS["Contract"], weekly) > drift.MAJOR


def test_status_bands():
    assert [drift.status(p) for p in (0.05, 0.1, 0.2, 0.25, 1.0)] == [
        "stable", "watch", "watch", "major drift", "major drift"]


def test_table_is_sorted_worst_first_and_says_what_changed():
    current = CUSTOMERS.sample(500, random_state=2).assign(
        MonthlyCharges=lambda d: d["MonthlyCharges"] * 1.3)
    table = drift.drift_table(CUSTOMERS, current)
    assert table.iloc[0]["feature"] == "MonthlyCharges"
    assert table.iloc[0]["status"] == "major drift"
    assert table["psi"].is_monotonic_decreasing
    assert table.iloc[0]["training data"].startswith("mean ")


def test_too_little_traffic_gives_no_verdict():
    report = drift.to_markdown(None, n_current=12, n_reference=5634, days=30)
    assert "Not enough traffic" in report and "last 30 days" in report
