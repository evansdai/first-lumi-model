#!/bin/bash
#SBATCH --job-name=first-model
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#
# One GCD for 20 minutes on the cheap partition. This is step 5 of README.md.
#
# Why these numbers: on `small-g` the bill is
#   max(ceil(cores/8), ceil(mem_GB/64), GCDs) * hours * 0.5
# GPU-hours, so 8 cores + 32 GB + 1 GCD is the cheapest shape that can use one
# GCD. 16 cores, or 128 GB, would cost exactly double for the same work, and
# `standard-g` would cost eight times more. `--time` is a maximum, not a
# promise: you are billed for the time the job actually runs.
#
# Submit it with your project id, from the directory the log should land in:
#   source /path/to/first-lumi-model/env.sh
#   export MODEL_LAYER="$LUMI_SOFTWARE/venvs/<your-layer>.sqsh"
#   cd "$LUMI_RUNS"
#   sbatch --account="$PROJECT_ID" \
#     "$EXAMPLE_DIR/workflow/profiles/lumi-g/train.sbatch"
#
# The project id is passed at submit time rather than written here, so this file
# stays the same for every project. (`#SBATCH` lines are read before the script
# runs, so they cannot read $PROJECT_ID themselves.)
set -euo pipefail

# Slurm copies the submitting environment into the job, so if you sourced env.sh
# in your shell, EXAMPLE_DIR and MODEL_LAYER are already set here.
: "${EXAMPLE_DIR:?source env.sh before submitting}"
: "${MODEL_LAYER:?export MODEL_LAYER=<the .sqsh from build-layer.sh>}"
. "$EXAMPLE_DIR/env.sh"

# /scratch/<project> is a symlink, and LUMI does not mount /scratch into a
# container: bind the resolved path, or the container never sees your files.
REPO="$(readlink -f "$EXAMPLE_DIR")"
SIF="$(cat "$LUMI_SOFTWARE/laifs/base.path")"
RUN="$LUMI_RUNS/$SLURM_JOB_ID"

# Check the layer you are about to mount. It is a few MB, so this is instant.
# The 14 GB base image is verified once, when the layer is built.
sha256sum -c "$MODEL_LAYER.sha256" >/dev/null

# Provenance: train.py copies these into manifest.json.
export BASE_IMAGE="$SIF"
# Step 2 of README.md makes the image checksum optional: with none recorded there
# is nothing to compare, and the manifest says so rather than staying empty. The
# layer's own checksum is always written by build-layer.sh, so it is always checked.
if [ -s "$LUMI_SOFTWARE/laifs/base.path.sha256" ]; then
  export BASE_IMAGE_SHA256="$(cut -d' ' -f1 "$LUMI_SOFTWARE/laifs/base.path.sha256")"
else
  export BASE_IMAGE_SHA256="unverified"
fi
export MODEL_LAYER_SHA256="$(head -1 "$MODEL_LAYER.sha256" | cut -d' ' -f1)"
mkdir -p "$RUN"

echo "job        $SLURM_JOB_ID on $(hostname)"
echo "image      $SIF"
echo "layer      $MODEL_LAYER"
echo "results    $RUN"
echo

# srun runs one step of the job. Everything inside it sees the allocation's GCD.

# The gate first: prove the GCD is visible before spending the allocation.
# `cuda` is passed explicitly -- with no argument the check falls back to the
# CPU and passes on a broken allocation (docs/workflow/70 §2).
srun singularity exec \
  --no-home \
  --bind "$REPO:/workspace:ro" \
  --pwd /workspace \
  "$SIF" \
  python /workspace/tools/check_platform.py cuda

# Then the training. `--device cuda` for the same reason: a silent CPU
# fallback would pass the accuracy assertion while proving nothing about
# the GPU.
srun singularity exec \
  --no-home \
  --bind "$REPO:/workspace:ro" \
  --bind "$MODEL_LAYER:/user-software:image-src=/" \
  --bind "$RUN:/run:rw" \
  --pwd /workspace \
  "$SIF" \
  /user-software/bin/python /workspace/workflow/scripts/train.py \
    --out /run --epochs 30 --device cuda
