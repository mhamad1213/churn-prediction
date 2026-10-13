import pandas as pd
import pandera.errors
import pytest

from churn.config import RAW_DATA
from churn.data import clean, load_data, schema, split_xy

needs_data = pytest.mark.skipif(not RAW_DATA.exists(), reason="run scripts/download_data.py")


def raw_row(**overrides):
    row = {
        "customerID": "0001-A", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": " ",
        "Churn": "No",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_clean_fills_blank_total_charges_for_new_customers():
    out = clean(raw_row())
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out.loc[0, "SeniorCitizen"] == "No"
    assert out.loc[0, "Churn"] == 0


def test_schema_rejects_unknown_contract():
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(clean(raw_row(Contract="Weekly", TotalCharges="10")))


@needs_data
def test_full_dataset_loads_and_validates():
    df = load_data()
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0
    X, y = split_xy(df)
    assert "customerID" not in X.columns
    assert 0.25 < y.mean() < 0.28
