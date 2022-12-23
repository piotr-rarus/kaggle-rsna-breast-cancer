import numpy as np
import torch

from src.lib.metrics import pF1Beta


def test_pfbeta() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    pfbeta = pF1Beta()
    score = pfbeta(predictions, labels)
    proper_score = 0.9
    assert np.allclose(score, proper_score)
    score = pfbeta(torch.from_numpy(predictions), torch.from_numpy(labels))
    assert np.allclose(score, proper_score)
