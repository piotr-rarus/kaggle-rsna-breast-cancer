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
    if (c_precision + c_recall) == 0:
        # by definition the return value here is 0.0
        # Yet returned in this way, it has the same partial derivatives
        # w.r.t precision and recall as the true pF1 (in the limit
        # as the precision goes to 0 and the recall goes to 0)
        zero: float = (1 + beta_squared) / beta_squared * c_recall
        return zero
    result: float = (
        (1 + beta_squared)
        * (c_precision * c_recall)
        / (beta_squared * c_precision + c_recall)
    )
    return result
