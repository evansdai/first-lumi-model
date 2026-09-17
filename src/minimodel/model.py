"""A two-layer perceptron: the smallest model that can still fail for a GPU reason.

Nothing in this file knows about LUMI, ROCm or CUDA. It is the same model you
ran on your laptop, and that is the point (portability rule R1: the code stays
portable, the *runtime* differs per machine).
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Map a 2-D point to 2 class scores.

    Two hidden layers, ReLU, no dropout and no batch norm: the shortest model
    that still has weights worth training, and small enough to be obvious in a
    `torchinfo` summary.
    """

    def __init__(self, hidden: int = 32, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, 2)`` in, ``(batch, 2)`` class scores out.

        The output is raw scores, not probabilities: ``cross_entropy`` below
        applies log-softmax itself, and doing it here would apply it twice.
        """
        return self.net(x)
