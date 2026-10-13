import numpy as np
import pytest

from churn.threshold import CUSTOMER_VALUE, OFFER_COST, OFFER_SUCCESS_RATE, profit_at


def test_profit_by_hand():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.2, 0.8, 0.1])
    r = profit_at(y, prob, 0.5)  # flags customers 0 and 2: one churner, one false alarm
    assert r["flagged"] == 3
    assert r["churners_caught"] == 1
    assert r["false_alarms"] == 1
    assert r["profit"] == pytest.approx(1 * OFFER_SUCCESS_RATE * CUSTOMER_VALUE - 2 * OFFER_COST)


def test_contacting_nobody_costs_nothing():
    r = profit_at(np.array([1, 0]), np.array([0.4, 0.3]), 0.99)
    assert r["flagged"] == 0
    assert r["profit"] == 0
