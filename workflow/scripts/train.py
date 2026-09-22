#!/usr/bin/env python3
"""Train the tiny classifier and write down what happened.

The same file runs in three places, unchanged:

    your laptop            conda activate first-model
                           python workflow/scripts/train.py --out /tmp/run --epochs 30

    a LUMI login node      singularity exec ... python /workspace/workflow/scripts/train.py
    a LUMI-G GCD           the same, from inside a Slurm allocation

It is portable on purpose: no LUMI path, module name or vendor check anywhere
below. What changes per machine is the *runtime*, not the code.

Writes into --out:

    metrics.json    the numbers: loss and accuracy per epoch, wall time
    manifest.json   what produced them: device, torch, hip, host, job id, layer
    loss.png        the curves, if matplotlib can draw here
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

# matplotlib needs a writable directory for its font cache. Inside a LUMI
# container $HOME is deliberately not mounted (--no-home), so the default
# $HOME/.config/matplotlib is unwritable and plotting dies with a warning that
# looks like a plotting bug. Set the cache dir *before* matplotlib is imported:
# this line has to stay above the imports below.
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-{os.environ.get('USER', 'user')}")

import matplotlib

matplotlib.use("Agg")  # no display on a compute node; write files only

import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402  -- installed by layer-build, not in the image
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchinfo import summary  # noqa: E402  -- installed by layer-build

# Make `src/minimodel` importable however this file is invoked. One line, so a
# beginner does not have to know about PYTHONPATH or `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minimodel.data import split, two_blobs  # noqa: E402
from minimodel.model import MLP  # noqa: E402


def resolve_device(prefer: str = "auto") -> torch.device:
    """Pick the accelerator at runtime; never hard-code it in the code.

    ROCm (LUMI-G) reports itself as CUDA in PyTorch, so the first branch is the
    one LUMI takes. Apple's MPS is the laptop branch.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment(device: torch.device, args: argparse.Namespace) -> dict:
    """What produced the numbers. Read this before trusting a run.

    `hip` is the useful one on LUMI: non-null means the ROCm build of PyTorch is
    underneath. `hip: null` means a CPU build and a job that proved nothing.
    """
    return {
        "device": device.type,
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
        ),
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        # Set by the job script and by you; empty when run straight from a shell.
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        # Set by the job script (train.sbatch) and, for MODEL_LAYER, by env.sh. "unverified"
        # rather than "" when a field is absent: a hand-run's manifest should say what it does not
        # know, the same way step 2's optional image checksum does.
        "layer": os.environ.get("MODEL_LAYER", ""),
        "layer_sha256": os.environ.get("MODEL_LAYER_SHA256", "") or "unverified",
        "base_image": os.environ.get("BASE_IMAGE", ""),
        "base_image_sha256": os.environ.get("BASE_IMAGE_SHA256", "") or "unverified",
        "argv": sys.argv,
        "args": vars(args),
    }


def train(args: argparse.Namespace, device: torch.device, out: Path) -> dict:
    """Run the loop and return the metrics dict. No plotting, no file writes."""
    torch.manual_seed(args.seed)

    x, y = two_blobs(n=args.n_points, seed=args.seed, device=device)
    x_train, y_train, x_val, y_val = split(x, y)

    model = MLP(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Printed once, on purpose: it is the proof that the packages in the layer
    # were found, and it is the first thing to read in a job log.
    print(f"model summary ({sum(p.numel() for p in model.parameters())} parameters)")
    summary(model, input_size=(args.batch_size, 2), device=device.type)

    history: list[dict] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for start in range(0, len(x_train), args.batch_size):
            xb = x_train[start : start + args.batch_size]
            yb = y_train[start : start + args.batch_size]

            loss = F.cross_entropy(model(xb), yb)  # softmax is inside cross_entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():  # no gradients outside training: less memory, faster
            val_accuracy = (model(x_val).argmax(dim=1) == y_val).float().mean().item()

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / batches, 4),
            "val_accuracy": round(val_accuracy, 4),
        }
        history.append(row)
        print(f"epoch {epoch:3d}  train_loss {row['train_loss']:.4f}  "
              f"val_accuracy {row['val_accuracy']:.4f}")

    wall = time.time() - started
    return {
        "epochs": args.epochs,
        "wall_seconds": round(wall, 2),
        "seconds_per_epoch": round(wall / args.epochs, 3),
        "first_train_loss": history[0]["train_loss"],
        "last_train_loss": history[-1]["train_loss"],
        "last_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
    }


def plot(history: list[dict], out_path: Path) -> None:
    """Draw the two curves that answer 'did it learn?'.

    seaborn is one of the two packages layer-build installed, so this
    function is also the smoke test for the layer: if the imports at the top of
    this file failed, you would never get here.
    """
    frame = pd.DataFrame(history)
    sns.set_theme(context="talk", style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(data=frame, x="epoch", y="train_loss", ax=axes[0])
    axes[0].set_title("training loss")
    sns.lineplot(data=frame, x="epoch", y="val_accuracy", ax=axes[1])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("validation accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="run", help="output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    # Not a default of "cuda": on a laptop this script must still run, and on a
    # GPU job the *check* (`tools/check_platform.py cuda`) is what proves the
    # GCD, not this script quietly falling back to the CPU.
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device {device.type} | torch {torch.__version__} | "
          f"hip {getattr(torch.version, 'hip', None)}")

    metrics = train(args, device, out)
    metrics["manifest"] = environment(device, args)
    metrics["run_id"] = f"first-model-{args.epochs}ep-seed{args.seed}"

    plot(metrics["history"], out / "loss.png")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(metrics["manifest"], indent=2) + "\n")

    # The gate. A run whose loss never moved, or whose accuracy stayed at chance
    # (0.5 for two balanced classes), proves nothing about the hardware -- it
    # only proves the process started. Fail loudly instead.
    assert metrics["last_val_accuracy"] >= 0.85, (
        f"validation accuracy only {metrics['last_val_accuracy']} -- "
        "the model did not learn; read the log, not this assert"
    )

    print(f"OK {args.epochs} epochs in {metrics['wall_seconds']}s -- "
          f"val_accuracy {metrics['last_val_accuracy']}")
    print(f"wrote {out}/metrics.json, {out}/manifest.json, {out}/loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
