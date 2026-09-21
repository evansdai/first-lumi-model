#!/bin/bash
# build-layer.sh -- turn "my extra packages" into ONE file LUMI can mount.
#
#   ./build-layer.sh <layer-id>
#   ./build-layer.sh first-model-20260917
#
# This is step 3 of README.md. It produces:
#
#   $LUMI_SOFTWARE/venvs/<layer-id>.sqsh            the layer: one file
#   $LUMI_SOFTWARE/venvs/<layer-id>.sqsh.freeze.txt what pip installed, exactly
#   $LUMI_SOFTWARE/venvs/<layer-id>.sqsh.manifest.txt  base image + checksums
#   $LUMI_SOFTWARE/venvs/<layer-id>.sqsh.sha256     the three files' hashes
#
# Why a layer and not a conda environment: an installed Python environment is
# tens of thousands of small files, and LUMI's shared filesystem (Lustre) is
# built for large files. LUMI refuses file-quota increases caused by Conda, and
# the documented answer is a container. A SquashFS layer is that answer at the
# smallest size: the AI image stays the runtime, and your packages arrive as one
# file that the container mounts read-only.
#
# WHAT THIS DOES NOT DECIDE: which packages belong in the extras list. That is
# workflow/envs/extra-requirements.txt, and its header explains the rule.
#
# RUN THIS IN A PLAIN LOGIN SHELL, not inside a container: every step is
# `singularity exec`, and containers do not nest on LUMI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/env.sh"

LAYER_ID="${1:-}"
if [ -z "$LAYER_ID" ]; then
  echo "usage: $(basename "$0") <layer-id>   e.g. first-model-20260917" >&2
  exit 2
fi
case "$LAYER_ID" in
*/*)
  echo "error: <layer-id> is a name, not a path: $LAYER_ID" >&2
  exit 2
  ;;
esac

EXTRAS="$HERE/workflow/envs/extra-requirements.txt"
BASE_PIN="$LUMI_SOFTWARE/laifs/base.path"
SQSH="$LUMI_SOFTWARE/venvs/$LAYER_ID.sqsh"
STAGE="${TMPDIR:-/tmp}/$USER/$LAYER_ID" # /tmp is node-local: build and
VENV="$STAGE/venv"                      # publish in the same session
RUN="$STAGE/run"

# ---- preflight: fail before anything expensive ------------------------------
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
  echo "error: this is running inside a container, which has no singularity." >&2
  echo "       Leave the container and run this in a plain login shell." >&2
  exit 1
fi
command -v singularity >/dev/null || {
  echo "error: no singularity on PATH" >&2
  exit 1
}
command -v mksquashfs >/dev/null || {
  echo "error: no mksquashfs on PATH" >&2
  exit 1
}

if [ ! -s "$BASE_PIN" ]; then
  echo "error: no AI image pinned at $BASE_PIN" >&2
  echo "       Do step 2 of README.md first." >&2
  exit 1
fi
BASE_SIF="$(cat "$BASE_PIN")"
[ -s "$BASE_SIF" ] || {
  echo "error: $BASE_PIN names a missing file" >&2
  exit 1
}

# Verify the image you are about to build against, when step 2 of README.md
# recorded a checksum beside it: a pin whose hash is never checked is not a pin.
# This reads the whole 14 GB (about 20 s on a login node), which is why jobs do
# not repeat it -- they check the small layer instead and record this image's
# hash in the manifest. With no checksum file, carry on and say so in the
# manifest, rather than failing a build that would have worked.
BASE_SHA="unverified"
if [ -s "$BASE_PIN.sha256" ]; then
  echo "verifying $(basename "$BASE_SIF") against $BASE_PIN.sha256"
  sha256sum -c "$BASE_PIN.sha256"
  BASE_SHA="$(cut -d' ' -f1 "$BASE_PIN.sha256")"
else
  echo "note: no checksum at $BASE_PIN.sha256 -- step 2 of README.md records one."
  echo "      carrying on; the manifest will say base_sha256=unverified."
fi
[ -s "$EXTRAS" ] || {
  echo "error: missing or empty extras list: $EXTRAS" >&2
  exit 1
}

# A published layer is immutable: a job may be reading it right now. A new
# package list is a new layer id, never an overwrite.
if [ -e "$SQSH" ] || [ -e "$SQSH.partial" ]; then
  echo "error: $SQSH already exists (or a '.partial' from an interrupted build)." >&2
  echo "       Pick a new <layer-id>, or move the old one aside first." >&2
  exit 1
fi
if [ -e "$VENV" ] || [ -e "$RUN" ]; then
  echo "error: staging directory left behind: $STAGE" >&2
  echo "       Move it aside, then rerun." >&2
  exit 1
fi

mkdir -p "$VENV" "$RUN" "$(dirname "$SQSH")"

echo "image   $BASE_SIF"
echo "extras  $EXTRAS ($(wc -l <"$EXTRAS") lines, comments included)"
echo "layer   $SQSH"
echo

# ---- build the venv INSIDE the image, at its final path ---------------------
# /user-software, not a scratch path: a venv bakes its own absolute path into
# every console script it installs, so a venv built at /somewhere/else and later
# mounted at /user-software produces executables that point at nothing.
#
# --system-site-packages is what makes this cheap: the venv sees the image's
# torch, numpy, pandas and matplotlib instead of installing its own copies. It
# is also what ties the layer to this exact image -- rebuild the layer when the
# image changes.
#
# Paths: LUMI does not mount /scratch into a container, so every path is bound
# explicitly, and the *full* project path is used because /scratch/<project> is
# a symlink.
echo "== building the venv =="
singularity exec \
  --no-home \
  --bind "$VENV:/user-software:rw" \
  --bind "$HERE:/workspace:ro" \
  --bind "$RUN:/run:rw" \
  --pwd /workspace \
  "$BASE_SIF" bash -lc '
    set -euo pipefail
    python -m venv /user-software --system-site-packages
    /user-software/bin/python -m pip install --no-cache-dir \
      --requirement /workspace/workflow/envs/extra-requirements.txt
    # No `pip check` here, deliberately -- see the dependency gate below the pack.
    /user-software/bin/python -m pip freeze --all > /run/freeze.txt
    echo
    echo "installed into the layer:"
    tail -n 5 /run/freeze.txt
  '

[ -s "$RUN/freeze.txt" ] || {
  echo "error: the build wrote no freeze.txt" >&2
  exit 1
}

# ---- pack the venv into one file --------------------------------------------
# -processors 1: a login node is shared with everyone else.
# -all-root: LUMI's umask makes files group-only; this makes the layer readable
#   inside the container, where you are a different user id.
echo
echo "== packing: thousands of files in, one file out =="
mksquashfs "$VENV" "$SQSH.partial" \
  -noappend -processors 1 -all-root -action "chmod(a+rX) @true"
[ -s "$SQSH.partial" ] || {
  echo "error: mksquashfs produced nothing" >&2
  exit 1
}

# ---- validate the PACKED layer, at the path jobs will mount it --------------
# Testing before packing would test a directory; this tests the artifact. The
# import check is the real one: it proves the extras can be imported *inside*
# the container, which is where your job runs.
echo
echo "== validating the packed layer at /user-software =="
# The dependency gate. NOT a bare `pip check`, and that is not a style choice.
#
# The AI image is inconsistent by `pip check`'s standard ON ITS OWN: it ships vllm 0.22.1, which
# requires compressed-tensors==0.15.0.1, and ships compressed-tensors 0.17.1 (read from the image's
# own published package list). A bare `pip check` therefore fails before your extras are involved,
# and `set -e` would stop the build here -- a working layer would look like a broken one.
#
# The question that matters is whether YOUR packages introduced a conflict, so ask it
# differentially: the image alone, then the image with your layer, and compare.
BASE_CHECK="$RUN/check-base.txt"
LAYER_CHECK="$RUN/check-layer.txt"
# The image's own python, by path: `python` would depend on the image's PATH, and this runs
# with --no-home and no profile.
singularity exec --no-home "$BASE_SIF" /opt/venv/bin/python -m pip check >"$BASE_CHECK" 2>&1 || true
singularity exec --no-home \
  -B "$SQSH.partial:/user-software:image-src=/" \
  "$BASE_SIF" /user-software/bin/python -m pip check >"$LAYER_CHECK" 2>&1 || true

echo "pre-existing conflicts in the image (expected, not your fault):"
sed 's/^/    /' "$BASE_CHECK"
if diff -q "$BASE_CHECK" "$LAYER_CHECK" >/dev/null; then
  echo "  your extras added none"
else
  echo "  YOUR EXTRAS CHANGED THIS -- fix them before publishing:"
  diff "$BASE_CHECK" "$LAYER_CHECK" | sed 's/^/    /'
  exit 1
fi

singularity exec \
  --no-home \
  -B "$SQSH.partial:/user-software:image-src=/" \
  "$BASE_SIF" \
  /user-software/bin/python -c '
import sys
print("interpreter:", sys.executable)
import seaborn, torchinfo            # noqa: F401 -- the two extras, imported
print("extras: seaborn", seaborn.__version__, "| torchinfo", torchinfo.__version__)
import torch
print("from the image: torch", torch.__version__, "| hip", torch.version.hip)
'

# ---- publish ----------------------------------------------------------------
echo
echo "== publishing =="
printf 'layer=%s\n' "$SQSH" >"$SQSH.manifest.txt.partial"
printf 'base_image=%s\n' "$BASE_SIF" >>"$SQSH.manifest.txt.partial"
printf 'base_sha256=%s\n' \
  "$BASE_SHA" >>"$SQSH.manifest.txt.partial"
printf 'requirements=%s\n' "$EXTRAS" >>"$SQSH.manifest.txt.partial"
printf 'created=%s\n' "$(date -u +%FT%TZ)" >>"$SQSH.manifest.txt.partial"

cp "$RUN/freeze.txt" "$SQSH.freeze.txt.partial"
mv "$SQSH.partial" "$SQSH"
mv "$SQSH.freeze.txt.partial" "$SQSH.freeze.txt"
mv "$SQSH.manifest.txt.partial" "$SQSH.manifest.txt"

# The manifest above is prose; sha256sum -c cannot read it. This list can:
#   (cd "$LUMI_SOFTWARE/venvs" && sha256sum -c first-model-*.sha256)
sha256sum "$SQSH" "$SQSH.freeze.txt" "$SQSH.manifest.txt" >"$SQSH.sha256"

echo
echo "published $SQSH"
echo
echo "Give every job this line (and put it in env.sh once you settle on a layer):"
echo "  export MODEL_LAYER=$SQSH"
