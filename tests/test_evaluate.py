import numpy as np

from churn.evaluate import classification_report, lift_at


def test_lift_perfect_ranking_catches_everyone_in_top_slice():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert lift_at(y, prob, 0.2) == 1.0


def test_report_has_expected_keys():
    rep = classification_report([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4])
    assert rep["roc_auc"] == 1.0
    assert {"recall", "pr_auc", "churners_caught_top20pct"} <= rep.keys()
