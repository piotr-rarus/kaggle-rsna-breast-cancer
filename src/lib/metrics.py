import numpy as np
import torch
from numpy.typing import NDArray


class pF1Beta(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def __call__(
        self,
        preds: NDArray[np.float_] | torch.Tensor,
        labels: NDArray[np.int_] | torch.Tensor,
    ) -> float | torch.Tensor:
        preds = preds.clip(0, 1)
        y_true_count = labels.sum()
        ctp = (preds * labels).sum()
        cfp = (preds * (1 - labels)).sum()
        beta_squared = self.beta**2
        c_precision: float | torch.Tensor = ctp / (ctp + cfp)
        c_recall: float | torch.Tensor = ctp / y_true_count
        if (c_precision + c_recall) == 0:
            # returns 0.0 as c_recall == 0
            return (1 + beta_squared) / beta_squared * c_recall
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
