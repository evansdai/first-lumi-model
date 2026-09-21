# More than one GPU

[← back to the tutorial](../README.md)

The tutorial uses **one GCD** — one half of one MI250x card — because that is the cheapest shape that
can use a GPU, and because a bug is easier to find in one process than in eight.

## The rule

**One GCD is one process.** To use more, you ask Slurm for more and then start one process per GCD:

```bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
```

with either

- `srun python train.py` — one rank per GCD, or
- `torchrun --nproc-per-node=8 train.py` — the same thing through PyTorch's launcher.

**Never both.** `--ntasks-per-node=8` *and* `torchrun` starts 64 processes for 8 GCDs, and the
failure is not a clear one.

## The cost changes shape here

On `small-g` you are billed per GCD, so 8 GCDs for an hour is 4 GPU-hours — the same as a whole node
on `standard-g`. The difference is that `small-g` bills only what you ask for, while `standard-g`
bills the node whether you use one GCD or eight. See the tutorial's *What it cost* section.

## Where to read more

The [LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide) works both shapes out for
real models; start there rather than inventing one. Multi-GPU training has its own failure modes —
collective timeouts, one rank writing to a shared file, a rank that never starts — and they are much
easier to recognise once you have seen a working example.
