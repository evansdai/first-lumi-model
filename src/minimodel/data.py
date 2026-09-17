"""Synthetic data: two overlapping blobs of points, two classes.

No download, no path, no dataset format. A training run that cannot see its data
is a different problem from a training run that cannot see the GPU, and this
example exists to isolate the second one.

Portability rule: the data is created on whichever device torch is already
using, and nothing here names a platform.
"""

from __future__ import annotations

import torch

# The two blob centres. The distance between them is 3.6 and the spread is 0.7,
# so the classes are 5.1 standard deviations apart along the axis that separates
# them: the best possible accuracy is ~99.5% (Phi(5.15/2)), and a small MLP gets
# close to that in a few hundred steps -- which is what makes a non-learning run
# visibly different from a working one.
CENTRES = ((-1.5, -1.0), (1.5, 1.0))
SPREAD = 0.7


def two_blobs(
    n: int = 4096,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x, y)``: ``x`` is float32 ``(n, 2)``, ``y`` is int64 ``(n,)``.

    The generator is fixed by ``seed`` and runs on the CPU, so the same seed
    reproduces the same data on a laptop and on a LUMI GCD. Points are moved to
    ``device`` only at the end -- generating noise on a GPU would make the
    dataset depend on the GPU's RNG implementation.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half = n // 2

    # Append the second blob's class label last, so the count is exactly n even
    # when it is odd (n - half != half in that case).
    parts = []
    for label, centre in enumerate(CENTRES):
        count = half if label == 0 else n - half
        offset = torch.tensor(centre, dtype=torch.float32)
        parts.append(torch.randn(count, 2, generator=gen) * SPREAD + offset)

    x = torch.cat(parts)
    y = torch.cat([torch.zeros(half), torch.ones(n - half)]).to(torch.int64)

    # Shuffle, so batches are not all one class in the first half of an epoch.
    order = torch.randperm(n, generator=gen)
    return x[order].to(device), y[order].to(device)


def split(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split into a training and a validation set, without shuffling again."""
    cut = int(len(x) * train_fraction)
    return x[:cut], y[:cut], x[cut:], y[cut:]
