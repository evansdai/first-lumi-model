# Other ways to build the environment

[← back to the tutorial](../README.md)

The tutorial's route — borrow LUMI's AI image, add your packages as one SquashFS layer — is **one of
four**. They are not better and worse in general. They answer different questions, and the question
is always the same:

> **What does my task need that the AI image does not already have?**

## The four ways

| Way | What you produce | What it costs | Right when | It cannot do |
|---|---|---|---|---|
| **0. Use the image as it is** | nothing | nothing | the image already has every package you import | anything it lacks |
| **1. One SquashFS layer of extras** — the tutorial's route | one small file, mounted read-only | a few minutes, and a rebuild when the image changes | your extras are *additions*, and one environment serves your runs | replace the image's Python, torch/ROCm or NumPy; be chosen per step |
| **2. Extend the image into your own `.def`** | your own `.sif` | a container build — minutes, and it needs memory, so use an interactive job | you want your own image, and the additions are still compatible with the base | it still inherits the base's Python and ABI |
| **3. `cotainr` from a conda environment file** | your own `.sif`, built from your declaration | the longest — a full solve and build | you need a **different** Python, a different torch/ROCm, system packages, or exact historical pins | nothing about the environment — but it duplicates the base on disk, and it is slower to rebuild |

Two words in that table, in plain language:

- **`.def`** — a small text file that says how to build a container: which base to start from, and
  what to install into it. LUMI builds it into a `.sif`.
- **ABI** — the low-level contract between compiled pieces of software. Two libraries with different
  ABIs cannot be loaded into the same process, which is why "a different Python" is a hard boundary
  and not a preference.

## Why the order

Each step costs more than the one before and buys a different thing.

**Step 0 costs nothing and is right more often than people expect.** Check the image's own package
list before you assume you need a layer — the tutorial's step 2 shows how.

**Step 1 is cheap for a reason:** the environment sees the image's torch, so nothing large is
installed or duplicated. That is what `--system-site-packages` buys you, and it is also why the layer
is tied to one image.

**Steps 2 and 3 are what you reach for when the *image itself* is the problem.**

## The difference that decides between 1 and 3

**Do your extras *replace* something the image already defines, or add to it?**

- **Additions → a layer.** This is the common case, and the tutorial's route.
- **Replacements → your own image.** If your code pins its own Python, its own torch, a NumPy older
  than the image's, or needs a system package, a layer cannot express that. Trying produces a layer
  that *shadows* the image's proven ROCm build — which is how a working job breaks in a way that
  looks like a broken image.

## Two more differences worth knowing before you choose

- **A layer is bound globally; an image is chosen per step.** A layer is a runtime argument, so every
  step in a pipeline shares it. If two steps need environments that conflict, that is a two-image
  question, not a two-layer one.
- **A layer is tied to one image.** Its environment is built against that image's Python and ABI, so
  an image update means rebuilding *and* re-checking the layer. A `.sif` carries its own runtime and
  moves independently.

## Route 2 in full — extending the AI image

A five-line `.def`:

```
Bootstrap: docker
From: docker.io/lumiaifactory/lumi-multitorch:torch
%post
    pip install ...
```

built on LUMI with `module load CrayEnv PRoot; singularity build yours.sif yours.def`. The
[AI environment page](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/) has both
examples, and warns that the build needs memory — use an interactive job.

## Route 3 in full — `cotainr`

```
module load CrayEnv
module load cotainr
cotainr build my.sif --system=lumi-g --conda-env=environment.yml
```

Read [cotainr's LUMI ROCm example](https://github.com/DeiC-HPC/cotainr/tree/main/examples/LUMI/conda_pytorch_rocm)
before writing a `torch` line: the `pytorch` conda channel has no ROCm builds, and a container built
from it ends up CPU-only.

> **Neither route 2 nor route 3 is exercised in this tutorial.** They are described so you can tell
> *which question you are answering*. If you need one of them, work from LUMI's own examples rather
> than from this page.

## Building the layer for your own packages

Step 3 of the tutorial calls [`layer-build/`](../layer-build/MANUAL.md) — one Python file with no
dependencies, its manual, and its tests. It is the builder for this route, and it refuses the cases
that do not work rather than building a layer that breaks later:

- it reads a **conda environment file** — `environment.yml`, `conda env export --json`, or
  `conda list --json` — not a `requirements.txt`;
- it asks the image what it already has and installs only the difference;
- it **refuses to shadow the image's stack** — a different `numpy`, a second `torch` — and says which
  package, which version the image has, and why it matters;
- it exits `3` on a conflict, so a script or an agent can branch on the result, and `--json` gives the
  same answer as one object.

The manual is the contract: [`layer-build/MANUAL.md`](../layer-build/MANUAL.md). If your extras would
*replace* something the image already defines, no flag helps: that is route 2 or 3 above.

## The long form

This page is the short version, and it is the whole of what is published. The case it deliberately
skips is *several* environments in one workflow: when two steps need environments that conflict,
that is the two-image question above — not something one more layer can solve.
