import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 2
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x).relu()
        x = x.mean(dim=(-2, -1))
        return x
