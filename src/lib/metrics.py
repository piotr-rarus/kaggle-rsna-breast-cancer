import numpy as np
from numpy.typing import NDArray


def pfbeta(
    labels: NDArray[np.int_], preds: NDArray[np.float_], beta: float = 1.0
) -> float:
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result: float = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    return 0.0
