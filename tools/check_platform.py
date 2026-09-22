#!/usr/bin/env python3
"""Print what this machine can actually do. Run it first on every new platform.

Three targets, one script:

    Mac (MPS)            LUMI (ROCm)              NVIDIA HPC (CUDA)
    ./check_platform.py  srun ... python tools/check_platform.py  ./check_platform.py

The point is not the versions. It is (a) that a real tensor operation works on the
resolved device, and (b) which dtypes this backend can actually do — because the
portability contract this folder follows depends on both.

Exit code is non-zero if the device is unusable, so this works as a gate in a job script — but
only for the device it resolves. With no argument that is `cuda` if available, then `mps`, then
`cpu`, so on a GPU allocation whose GCD was never visible it can pass on the CPU and exit 0.
Pass a device (`cuda`, `mps`, `cpu`) when the job is supposed to prove that backend.
"""

from __future__ import annotations

import platform
import sys

import torch

# Report and probe; never let a probe exception escape as a traceback.
SEP = "-" * 58


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve the device at runtime; never name it in code.

    Note that ROCm masquerades as CUDA in PyTorch — torch.cuda.is_available() is True on
    LUMI-G — so two of the three targets share this branch.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def report_env() -> None:
    print(SEP)
    print(f"python       {sys.version.split()[0]}")
    print(f"platform     {platform.platform()}")
    print(f"torch        {torch.__version__}")
    # `hip` is None on a CUDA or CPU build; non-null means a ROCm build.
    print(f"hip          {getattr(torch.version, 'hip', None)}")
    print(f"cuda         {torch.version.cuda}")
    print(f"cuda avail   {torch.cuda.is_available()}")
    print(f"mps avail    {torch.backends.mps.is_available()}")


def probe_device(device: torch.device) -> bool:
    """A real operation, not just a capability flag."""
    print(SEP)
    print(f"device       {device.type}")

    if device.type == "cuda":
        print(f"device name  {torch.cuda.get_device_name(0)}")
        print(f"device count {torch.cuda.device_count()}")

    x = torch.ones((8, 8), device=device)
    y = (x @ x).sum().item()
    # ones(8,8) @ ones(8,8) is an 8x8 matrix of eights; the sum is 8*64 = 512.
    assert y == 512.0, f"matmul gave {y}, expected 512.0"
    print(f"matmul       ok ({y})")
    return True


def probe_dtypes(device: torch.device) -> None:
    """Which precisions this backend can actually do."""
    print(SEP)

    # R3: float64 is unsupported on MPS, and it raises rather than falling back.
    try:
        torch.ones(2, dtype=torch.float64, device=device)
        print("float64      yes")
    except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
        print(f"float64      NO ({type(exc).__name__})")

    # bf16 is uneven: check the capability rather than assuming it.
    if device.type == "cuda":
        print(f"bf16         {torch.cuda.is_bf16_supported()}")
    else:
        try:
            torch.ones(2, dtype=torch.bfloat16, device=device)
            print("bf16         tensor ok (support still uneven - test your real op)")
        except Exception as exc:  # noqa: BLE001
            print(f"bf16         NO ({type(exc).__name__})")


def main() -> int:
    report_env()

    prefer = sys.argv[1] if len(sys.argv) > 1 else "auto"
    device = resolve_device(prefer)

    try:
        probe_device(device)
    except Exception as exc:  # noqa: BLE001
        print(SEP)
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    probe_dtypes(device)

    print(SEP)
    print(f"OK - {device.type} is usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
