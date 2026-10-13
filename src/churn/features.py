"""Preprocessing shared by every model, so training and serving match."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS

ENGINEERED_NUMERIC = ["num_addons", "avg_monthly_spend", "charge_increase"]
ENGINEERED_CATEGORICAL = ["tenure_bucket", "new_monthly_customer"]


def build_preprocessor(engineered: bool = False) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones.

    With engineered=True it also expects the columns made by add_features().
    """
    numeric = NUMERIC_COLS + (ENGINEERED_NUMERIC if engineered else [])
    categorical = CATEGORICAL_COLS + (ENGINEERED_CATEGORICAL if engineered else [])
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
        ]
    )

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features based on the EDA findings."""
    df = df.copy()

    # 1. Tenure groups: churn drops sharply after the first year
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    # 2. How many add-on services the customer pays for
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)

    # 3. Average spend per month so far (tenure 0 = nobody billed yet, use current price)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"]
    )

    # 4. Paying more now than their historical average?
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # 5. The 51% churn group from the EDA heatmap
    is_new_monthly = (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)
    df["new_monthly_customer"] = np.where(is_new_monthly, "Yes", "No")

    return df
