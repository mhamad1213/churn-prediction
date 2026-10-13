"""Explain a single prediction with SHAP: which features pushed the risk up or down.

The model sees ~48 columns after one-hot encoding (e.g. Contract_One year,
Contract_Two year). For people, we add those back up into the original
column ("Contract"), so reasons read naturally.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS
from churn.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC

ALL_COLUMNS = NUMERIC_COLS + ENGINEERED_NUMERIC + CATEGORICAL_COLS + ENGINEERED_CATEGORICAL


def _original_column(encoded_name: str) -> str:
    """'num__tenure' -> 'tenure', 'cat__Contract_Two year' -> 'Contract'."""
    name = encoded_name.split("__", 1)[1]
    # longest match first, so "MultipleLines" isn't mistaken for a shorter name
    for col in sorted(ALL_COLUMNS, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


class Explainer:
    def __init__(self, model: Pipeline):
        self.model = model
        self.preprocess = model[:-1]  # every step except the classifier
        self.add_features = model[:-2]  # just the feature-engineering step
        self.tree_explainer = shap.TreeExplainer(model[-1])
        encoded = model.named_steps["prep"].get_feature_names_out()
        self.columns = [_original_column(n) for n in encoded]

    def top_reasons(self, customer: pd.DataFrame, n: int = 3) -> list[dict]:
        """The n features that pushed this customer's churn risk up the most."""
        encoded = self.preprocess.transform(customer)
        contributions = np.asarray(self.tree_explainer.shap_values(encoded))[0]
        per_column = pd.Series(contributions, index=self.columns).groupby(level=0).sum()
        with_features = self.add_features.transform(customer)  # to look up each value
        reasons = []
        for col, effect in per_column.sort_values(ascending=False).head(n).items():
            if effect <= 0:
                break  # only report features that actually increase risk
            reasons.append({
                "feature": col,
                "value": _plain(with_features[col].iloc[0]),
                "effect": round(float(effect), 3),
            })
        return reasons


def _plain(value):
    return value.item() if hasattr(value, "item") else value
