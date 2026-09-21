# env.sh -- the paths this example uses. Source it in every new shell:
#
#   . /path/to/first-lumi-model/env.sh
#
# The only line you must edit is PROJECT_ID. It is your LUMI project -- the one
# your allocation is billed to. `lumi-workspaces` lists the projects you belong
# to; the id looks like project_465003379.
#
# These are the same three names the author's `lumi-env` environment repository uses, so if you
# adopt that later nothing here changes meaning.

export PROJECT_ID=project_XXXXXXX # <-- EDIT THIS

export SCRATCH="/scratch/$PROJECT_ID" # the big, fast filesystem
export USER_SCRATCH="$SCRATCH/$USER"  # your own space inside it

export LUMI_CODE="$USER_SCRATCH/code"         # code checkouts
export LUMI_SOFTWARE="$USER_SCRATCH/software" # images and layers
export LUMI_RUNS="$USER_SCRATCH/runs"         # one directory per run

# Where this example lives. BASH_SOURCE is this file's own path, so this is
# right no matter which directory you source it from. (This file is bash, not
# sh: it uses an array. Source it with `.` from bash, never `sh env.sh`.)
export EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The layer built in step 3 of README.md. Uncomment and paste your own line
# after the first build, so every new shell knows it. Every job needs it.
# export MODEL_LAYER="$LUMI_SOFTWARE/venvs/first-model-YYYYMMDD.sqsh"
