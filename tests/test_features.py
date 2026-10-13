import pytest
from test_data import raw_row

from churn.data import clean
from churn.features import add_features


def test_new_customer_features():
    df = add_features(clean(raw_row(tenure=0, MonthlyCharges=50.0)))
    row = df.iloc[0]
    assert row["tenure_bucket"] == "0-12"
    assert row["num_addons"] == 1  # only OnlineBackup is "Yes" in raw_row
    assert row["avg_monthly_spend"] == 50.0  # tenure 0 falls back to current price
    assert row["charge_increase"] == 0.0
    assert row["new_monthly_customer"] == "Yes"


def test_long_term_customer_with_price_rise():
    df = add_features(
        clean(raw_row(tenure=24, MonthlyCharges=80.0, TotalCharges="1680", Contract="Two year"))
    )
    row = df.iloc[0]
    assert row["tenure_bucket"] == "13-24"
    assert row["avg_monthly_spend"] == pytest.approx(70.0)
    assert row["charge_increase"] == pytest.approx(10.0)
    assert row["new_monthly_customer"] == "No"


def test_original_dataframe_is_not_changed():
    original = clean(raw_row())
    add_features(original)
    assert "num_addons" not in original.columns
