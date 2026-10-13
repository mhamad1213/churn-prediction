"""Load, clean and validate the raw Telco churn data."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from churn.config import CATEGORICAL_COLS, ID_COL, NUMERIC_COLS, RAW_DATA, TARGET

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]

schema = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(str, unique=True),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": pa.Column(str, pa.Check.isin(YES_NO)),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO)),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO)),
        "tenure": pa.Column(int, pa.Check.in_range(0, 100)),
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(YES_NO + ["No phone service"])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "TechSupport": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "StreamingTV": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO)),
        "PaymentMethod": pa.Column(str),
        "MonthlyCharges": pa.Column(float, pa.Check.gt(0)),
        "TotalCharges": pa.Column(float, pa.Check.ge(0)),
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
)


input_schema = schema.remove_columns([ID_COL, TARGET])
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quirks of the raw file.

    - TotalCharges is stored as text and is blank for the 11 customers with
      tenure 0 (they haven't been billed yet), so blanks become 0.
    - SeniorCitizen is 0/1 while every other flag is Yes/No; make it Yes/No.
    - Churn becomes a 0/1 integer target.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_data(path: Path = RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV, clean it and validate it against the schema."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    df = pd.read_csv(path, dtype={"TotalCharges": str})
    return schema.validate(clean(df))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target, dropping the ID."""
    return df[NUMERIC_COLS + CATEGORICAL_COLS], df[TARGET]
