import pandas as pd
from test_data import raw_row

from churn.data import clean, split_xy
from churn.train_models import candidates


def test_every_candidate_trains_and_predicts_on_raw_columns():
    rows = [raw_row(customerID=f"id-{i}", tenure=i, TotalCharges=str(30.0 * i),
                    Churn="Yes" if i % 3 == 0 else "No") for i in range(1, 31)]
    X, y = split_xy(clean(pd.concat(rows, ignore_index=True)))
    for name, pipe in candidates().items():
        prob = pipe.fit(X, y).predict_proba(X)[:, 1]
        assert len(prob) == len(X), name
        assert ((prob >= 0) & (prob <= 1)).all(), name
