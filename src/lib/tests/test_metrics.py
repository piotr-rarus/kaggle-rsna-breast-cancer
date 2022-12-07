import numpy as np

from src.lib.metrics import pfbeta


def test_pfbeta() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    score = pfbeta(labels, predictions)
    proper_score = 0.9
    assert abs(score - proper_score) < 1e-9
