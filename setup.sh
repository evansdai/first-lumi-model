#!/bin/bash
# setup.sh -- answer a few questions, and this writes env.sh and makes the directories for you.
#
#   ./setup.sh
#
# WHY THIS EXISTS. README.md step 0 asks you to edit env.sh by hand and create four directories. That
# is fine when you already know LUMI's layout, and tedious when you do not. This does the same thing
# while telling you what each answer is for. Every question has a recommended default: press Enter.
#
# It writes exactly one file -- env.sh, in this directory -- and creates the directories you name.
# Nothing is installed and nothing is submitted. If you would rather do it by hand, skip this and
# follow README.md step 0; the two produce the same result.
#
# Run it on a LUMI login node, in this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '%s\n' "$*"; }
ask()  { # ask <prompt> <default> -> echoes the answer
  local prompt="$1" default="$2" answer=""
  if [ -t 0 ]; then
    printf '  %s\n  [%s]: ' "$prompt" "$default" >&2
    read -r answer || true
  fi
  printf '%s' "${answer:-$default}"
}

say ""
say "Setting up the first-lumi-model example."
say ""
say "Three questions, all with defaults. Press Enter to accept one."
say "More detail on any of them: README.md step 0, and docs/10 in the environment manual."
say ""

if [ ! -t 0 ]; then
  say "  (not a terminal -- using the defaults for everything)"
  say ""
fi

# --- who is paying for this ---------------------------------------------------
say "1. Your LUMI project."
say "   This is the allocation your jobs are billed to. It is also part of every path below,"
say "   so getting it wrong shows up later as a permission error."
say "   If you do not know it, open another terminal and run:  lumi-workspaces"
# Deliberately NOT ${PROJECT_ID:-...}. `PROJECT_ID` is a common name and the author's own env.sh sets
# it, so inheriting it would silently write SOMEONE ELSE'S project into this file -- and every job
# would then bill them. The scriptable override is a name that cannot collide.
PROJECT_ID="$(ask "Project id" "${SETUP_PROJECT_ID:-project_XXXXXXX}")"

case "$PROJECT_ID" in
  *XXXXXXX*|"")
    say ""
    say "  That is still the placeholder. Run 'lumi-workspaces' on a login node to see the"
    say "  projects you belong to, then run this script again."
    exit 1
    ;;
esac

# --- where things go ----------------------------------------------------------
# The defaults follow docs/10: code, software, runs and data all live under your own space on
# /scratch, never under $HOME (a 100 000-file quota that cannot be raised) and never under /project
# (shared, and a 100 000-file quota of its own).
say ""
say "2. Where should the code live?"
say "   Recommended: your own space on /scratch. NOT \$HOME -- its file quota is 100 000 and"
say "   cannot be raised, and a code checkout is thousands of files."
CODE_DEFAULT="/scratch/$PROJECT_ID/$USER/code/first-lumi-model"
# Each answer can also be given as an environment variable, which makes this scriptable and is how
# it is tested: CODE_DIR=/somewhere ./setup.sh
CODE_DIR="$(ask "Code directory" "${CODE_DIR:-$CODE_DEFAULT}")"

say ""
say "3. Where should runs write?"
say "   Each run gets its own directory here: checkpoints, logs, figures. /scratch is right;"
say "   /project is shared and has a much smaller file quota."
RUNS_DIR="$(ask "Runs directory" "${RUNS_DIR:-/scratch/$PROJECT_ID/$USER/runs}")"

# The remaining two paths are derived rather than asked: they are almost never different, and every
# question is a chance to get one wrong.
SOFTWARE_DIR="/scratch/$PROJECT_ID/$USER/software"
DATA_DIR="/scratch/$PROJECT_ID/$USER/data"

say ""
say "Also using, derived from the above (edit env.sh later if you need to):"
say "  layers and images   $SOFTWARE_DIR"
say "  staged data         $DATA_DIR"
say ""

# --- write it -----------------------------------------------------------------
ENV="$HERE/env.sh"
cp "$ENV" "$ENV.before-setup" 2>/dev/null || true

cat > "$ENV" <<EOF
# env.sh -- written by setup.sh on $(date -u +%FT%TZ)
#
# The paths this example uses. Every script here reads this file; nothing else needs editing.
# The previous contents are in env.sh.before-setup if you want to compare.

export PROJECT_ID=$PROJECT_ID

export EXAMPLE_DIR=$CODE_DIR
export LUMI_RUNS=$RUNS_DIR
export LUMI_SOFTWARE=$SOFTWARE_DIR
export LUMI_DATA=$DATA_DIR
EOF

say "Wrote $ENV"
say ""

# --- make the directories -----------------------------------------------------
say "Creating directories:"
for d in "$CODE_DIR" "$RUNS_DIR" "$SOFTWARE_DIR" "$DATA_DIR"; do
  if mkdir -p "$d" 2>/dev/null; then
    say "  ok       $d"
  else
    say "  FAILED   $d"
    say "           Does '$PROJECT_ID' match a project you belong to? Check with lumi-workspaces."
    exit 1
  fi
done

say ""
say "Checking the paths are what you think they are:"
# shellcheck disable=SC1090
. "$ENV"
say "  PROJECT_ID      $PROJECT_ID"
say "  EXAMPLE_DIR     $EXAMPLE_DIR"
say "  LUMI_RUNS       $LUMI_RUNS"

# --- the shell line -----------------------------------------------------------
say ""
if grep -q 'first-lumi-model/env.sh' "$HOME/.bashrc" 2>/dev/null; then
  say "~/.bashrc already sources this env.sh -- nothing to do."
else
  say "Add this line to ~/.bashrc, so new shells know these paths?"
  say "  . $ENV"
  if [ -t 0 ]; then
    printf '  Add it now? [y/N]: '
    read -r yn || true
    case "${yn:-n}" in
      y|Y|yes|YES)
        printf '. %s\n' "$ENV" >> "$HOME/.bashrc"
        say "  Added. It takes effect in a NEW shell."
        ;;
      *) say "  Skipped. You can add it yourself later." ;;
    esac
  else
    say "  (not a terminal -- skipped)"
  fi
fi

# --- what next ----------------------------------------------------------------
say ""
say "Done. Where to go now:"
say ""
say "  README.md step 1   run the model on your laptop first -- it is the fastest place to"
say "                     find a bug, and nothing about LUMI makes one easier to find."
say "  README.md step 2   pin the AI image (one command, once)."
say "  README.md step 3   build the layer that carries your extra packages."
say ""
say "  If you want the reasoning behind the layout above -- why /scratch and not \$HOME, why"
say "  runs get their own directory -- README.md 'Files and commands on LUMI you will"
say "  actually use', and docs/10 in the environment manual."
say ""
