import numpy as np

from churn.neural_net import TorchMLPClassifier


def test_network_learns_a_simple_pattern():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(int)  # churn depends only on the first column
    clf = TorchMLPClassifier(max_epochs=100).fit(X, y)
    prob = clf.predict_proba(X)
    assert prob.shape == (400, 2)
    assert np.allclose(prob.sum(axis=1), 1)
    assert (clf.predict(X) == y).mean() > 0.9
