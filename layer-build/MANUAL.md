# layer-build

Put the packages a base container is **missing** into one SquashFS layer — and refuse, loudly,
anything that would shadow what the image already ships.

```bash
layer-build --base "$(cat "$LUMI_SOFTWARE/laifs/base.path")" \
            --env workflow/envs/extra-environment.yml \
            --id first-model-$(date -u +%Y%m%d) \
            --out "$LUMI_SOFTWARE/venvs" \
            --export-to env.sh
```

Run `layer-build --help` for the flags. This page is the contract: what it accepts, what it writes,
what it refuses, and what has actually been run.

## Where it runs

On a **LUMI login node, in a plain shell** — not inside a container. It drives `singularity`,
`mksquashfs` and `sha256sum`; containers do not nest on LUMI, so from inside one it stops with
exit `5` instead of failing obscurely. (That guard is not theoretical: the author's own agent
session runs inside the devtools container, and it fires there.)

It needs `singularity`, `mksquashfs` and `sha256sum` on `PATH`, and **the login node's own
`python3`** — no module to load, no container to start, no package to install. The tool therefore
avoids everything newer than Python 3.6 (`dataclasses`, PEP 585/604 annotations, `capture_output`).
That floor is measured, not guessed: the first run on a login node failed on `from __future__ import
annotations` (3.7) while its parser accepted the tool's f-strings (3.6), and after the change the tool
**ran on that same `python3`** — so the interpreter there is 3.6. `python3 -V` prints the number.
The *in-container* probe is a different interpreter — the image's python 3.12, which is what
`importlib.metadata` needs. Nothing is installed on the host: the packages go into a virtual
environment *inside* the base image, and the layer is one file.

## What it accepts

| Input | Example |
|---|---|
| `environment.yml` (a subset: `name`, `dependencies`, and the nested `pip:` list) | `layer-build --env environment.yml` |
| `conda env export --json` | `conda env export --json > env.json` |
| `conda list --json` (the resolved set, with exact versions) | `conda list --json > installed.json` |

The smallest useful file — this is what the tutorial's extras list looks like:

```yaml
name: first-model-extras
dependencies:
  - pip:
      - seaborn
      - torchinfo
```

A plain `requirements.txt` is **not** accepted; wrap it in the shape above. Sections other than
`dependencies` (`channels:`, `variables:`) are ignored, and a YAML shape the parser does not cover
(anchors, flow style, an inline mapping) is an error rather than a guess: a parser that
half-understands YAML builds an environment nobody asked for. A requirement carrying an environment
marker (`pkg; python_version<'3.9'`) is refused too — dropping the condition would install a package
on a platform the file excluded.

**A whole `conda env export` or `conda list` is the wrong input for this route**, even though the
tool reads both: a real export also contains the environment's own interpreter and its conda system
packages (`python`, `pip`, `_libgcc_mutex`, `libgcc-ng`, …). A layer cannot provide those — they are
part of the runtime it borrows — so the tool refuses them **by name** rather than skipping them
quietly. Pass the list of packages you *add* to the image, which is what the tutorial's extras file
is; if you need a different Python, that is route 2 or 3.

## What it writes

Beside the layer, at `--out`:

| File | What it is |
|---|---|
| `<id>.sqsh` | the layer: one file, mounted read-only at `/user-software` |
| `<id>.sqsh.freeze.txt` | exactly what the layer's interpreter sees (`pip freeze --all`) |
| `<id>.sqsh.manifest.txt` | layer id, tool version, base image **and its sha256** (or `unverified`), the environment file, timestamp, what was installed, and any `--allow-shadow` you used |
| `<id>.sqsh.sha256` | hashes of the three files above, so a job can check the layer it mounts (it names their final paths) |

Publication is **fail-closed, and the layer is the commit marker.** Every file is written under a
`.partial` name, then the records are renamed into place, then the layer. So a layer that exists
always has complete records beside it — and an interrupted publication can leave final records beside
a `.partial` layer. All of it blocks that layer id: an existing `<id>.sqsh`, any of its three
records, or any leftover `.partial` stops the next run with exit `2`. A new package set is a new
`--id`.

## What it refuses, and why

The rule the tool exists for: a virtual environment built with `--system-site-packages` can **add**
to the image, and can only **shadow** what the image has. Shadowing is how a working ROCm job
breaks — the image's torch is compiled against the image's numpy and its own ROCm runtime.

| Kind | When | What the tool says |
|---|---|---|
| `non-pip` | the environment asks for its own runtime or a conda system package: `python`, `pip`, `setuptools`, `wheel`, `conda`, `mamba`, `virtualenv`, or any name starting with `_` (which is why the check reads the raw name — normalization would turn `_libgcc_mutex` into `-libgcc-mutex`) | a layer cannot provide the runtime it borrows — that is a different container (route 2 or 3). Never overridable |
| `shadow-stack` | the image does not ship a package that belongs to the accelerator stack (`torch`, `torchvision`, `torchaudio`, `triton`, `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`, `scikit-learn`, `tensorboard`, `transformers`, `vllm`, `numba`, `llvmlite`, `pyarrow`) | installing it into the layer would shadow what the image's torch was built against |
| `version-drift` | the image has the package, at a version the environment does not accept | it names both versions and suggests `--allow-shadow` |
| `stack-drift` | the image **has** it and you asked for a different version, and you allowed it — then the *built* layer reports a different version for a stack package than the image does | the build fails rather than publishing a layer that moved the stack |
| `pip-check` | `pip check` inside the image **with the layer** reports a conflict the image alone does not have | the new lines are printed; the image's own pre-existing conflicts are not counted |

`--allow-shadow PKG` (repeatable) turns one `version-drift` or `shadow-stack` refusal into a
deliberate install, and records it in the manifest. It is the only way past those two; `non-pip`
cannot be allowed, because no flag makes a layer able to replace the interpreter.

## Exit codes

An agent — or a script — branches on these; they are part of the interface.

| Code | Meaning |
|---|---|
| `0` | the plan is clean, or the build succeeded |
| `2` | usage or input: bad flag, bad `--id`, unreadable environment file, an existing layer, a missing checksum file, or a base image that does not match `--base-sha256` |
| `3` | a conflict: something would shadow the image, or the layer added a `pip check` regression |
| `4` | the build or the verification failed (a package not importable, `mksquashfs` failed) |
| `5` | the environment is wrong: inside a container, or `singularity`/`mksquashfs`/`sha256sum` missing |

`--json` prints one object with `status`, `satisfied`, `missing`, `allowed_shadow`, `conflicts`
(each with `package`, `kind`, `requested`, `image_version`, `why`), `artifacts` and `export_to`.
Keys are sorted, so the output is stable and diffable. `--dry-run` does everything except build:
it prints the same plan and exits `0` or `3`.

## Flags

| Flag | Meaning |
|---|---|
| `--base PATH` | the pinned base image (`.sif`) — required |
| `--base-sha256 FILE` | a `sha256sum -c` file for `--base`. The tool reads the digest from it and compares it against its **own** hash of `--base` — it does not trust the filename inside the file — so a checksum for a different image is exit `2`, not a pass. Optional, because the tutorial's step 2 makes the image checksum optional; the manifest records `unverified` when it is not given |
| `--env PATH` | the environment file — required. All three shapes parse, but a whole `conda env export` / `conda list` is normally **refused** for this route, because it carries the runtime and conda system packages a layer cannot provide (see "What it accepts") |
| `--id NAME` | the layer id; default `<environment name>-<date>`. Letters, digits, `.`, `_`, `-` only, because it becomes a file name |
| `--out DIR` | where the four files go (default `.`) |
| `--python PATH` | the interpreter inside the image (default `python`) |
| `--allow-shadow PKG` | install PKG despite a version difference; repeatable; recorded in the manifest |
| `--dry-run` | plan and report; build nothing |
| `--json` | machine-readable report |
| `--export-to FILE` | after publishing, point `FILE`'s `MODEL_LAYER` line at the new layer |
| `--timeout SEC` | per command (default 3600) |
| `--keep-stage` | keep the staging directory for debugging, in `<out>/<id>.stage`. A second run refuses it if it already exists — move it aside first |

## `--export-to`, and what it will not do

Given `--export-to env.sh`, the tool rewrites that file's `export MODEL_LAYER=...` line (adding one
if there is none) so the next `srun`/`sbatch` finds the layer. Because that file is *sourced* by your
shells, it is executable text, and the rewrite is installed only after a subshell has sourced it and
printed the value back. So it refuses, rather than half-writes, when:

- the file is missing, not writable, or a symlink (a rewrite would replace the link, not its target);
- the file cannot be sourced cleanly by the shell that will read it: the check mirrors the job
(`. "$1"` under `set -euo pipefail`, the same options `train.sbatch` uses), so a file that exits
early, aborts part-way, or ends in a failing command with errexit disabled is refused;
- the path cannot be quoted into a form the shell reads back exactly.

The layer is already published when this runs, so a refusal here is a message, never a failed build.
The only things the tool deletes are its own: the temporary file this function writes beside your env
file, and the staging directory — and the staging directory only when the build **succeeds**. A
failed build leaves it behind (in `/tmp` on a compute node, which is RAM), so you can look at it;
`--keep-stage` keeps it deliberately and then refuses to reuse it.

## Limitations, stated rather than hidden

- **Version comparison is approximate, and refuses what it cannot read.** Release numbers compare
  numerically and trailing zeros are equal (`1.0` == `1.0.0`); `dev`/`a`/`b`/`rc` rank below a final
  release — with their serial, so `1.0rc1` != `1.0rc2` — and `.post` above it; a local part
  (`+rocm7.0`) is ignored, so the image's `2.10.0+rocm7.0` satisfies `==2.10.0`. It is not a full
  PEP 440 implementation, and the tool never uses it to decide whether two *installed* packages are
  equal — that comes from the image's own `pip list`. A specifier form it does not understand
  (`!=` against a range wildcard, for instance) is treated as unsatisfied, which errs toward a
  conflict rather than toward a silent pass.
- **A conda build string is dropped, not silently reinterpreted.** `=1.8.0=py312_0` becomes
  `==1.8.0`; a build *glob* (`=*=*pthreads*`) becomes "any version". pip has no way to express a build
  constraint, so the version is what gets compared — and that is the one place where a constraint is
  weakened rather than refused.
- **The YAML subset is a subset.** See above; unsupported shapes are errors.
- **The image inventory is `pip list --format=json`.** A package the image has only as a conda
  package, without pip metadata, will look missing.
- **`--allow-shadow` on a stack package is verified, not trusted**: if the built layer reports a
  different version for a stack package than the image does, the build fails (`stack-drift`).

## What has been run, and what has not

**Run, on this machine (Python 3.12.14):**

```bash
python3 layer-build/tests/test-layer-build.py     # 47 tests, OK
```

Covered by those tests, exactly: parsing all three accepted input shapes (including this
repository's own `environment.yml`, whose `libopenblas=*=*pthreads*` is a real build glob); the
refusals — an unsupported YAML shape, an environment marker, a JSON shape that is not a list, a plain
`requirements.txt`; version comparison (ordering, trailing zeros, pre-release serial, wildcards,
`~=`, local parts); the conda `=`/`>=`/build-string/build-glob conversions, with a multi-clause
constraint (`>=1,<2`) surviving intact; classification and every conflict kind decided *before* the
build (`non-pip` including `_`-prefixed names, `shadow-stack`, `version-drift`) plus `--allow-shadow`
overriding both overridable kinds; the `pip check` status policy (exit 1 is data, anything else is an
error); the differential regression gate; the generated import probe; `--json` output through
`print_report`; the checksum gate refusing a checksum that describes a different file; publication
itself, against real files on disk — all four records land, no `.partial` survives, `sha256sum -c`
verifies, and a publication whose hashing step fails leaves **no** final layer; and `record_export`
against a real file — round-trip, mode preserved, `exit 0` refused, `return 7` refused, a file that
runs `set +e` and then fails refused, safely-quoted paths accepted, a symlink refused.

Reachable CLI paths were exercised too: `--help`, `--version`, a missing environment file (exit 2), a

Both files also parse under the **3.6 grammar** (`ast.parse(..., feature_version=(3, 6))`), and contain none
of the 3.7+ runtime constructs named above — no `dataclasses`, no `from __future__ import annotations`,
no `capture_output`/`text=True`, no PEP 585/604 annotations. That is what makes the tool runnable on a
login node; it has still never been executed by a 3.6 interpreter, because this machine has 3.12.
malformed environment file (exit 2), a bad `--id` (exit 2), and the in-container guard (exit 5, fired
for real from inside the devtools container).

**Run on LUMI, 2026-09-22, on a login node (`uan01`) under the node's own `python3`:** the whole build
path. The dry run inventoried the pinned image (382 packages) and planned `seaborn, torchinfo` as the
delta; the real run created the venv inside the image, installed both, packed a 4,980,736-byte layer,
ran the stack-drift check, the import probe, `pip freeze` and the differential `pip check` against the
packed `.partial` (which also proves the `-B <layer>:/user-software:image-src=/` mount works),
published the four records, pointed `env.sh` at the layer, and `sha256sum -c` verified all three
hashes from the login directory. Job id: none — this is a login-node run, not a Slurm job.
**Run in a container on a GCD, 2026-09-22:** the layer was mounted at `/user-software` in an `srun
--pty` allocation on `dev-g` (job `22235946`) and the tutorial's own `train.py` imported the layer's
`torchinfo` and `seaborn`, trained three epochs on the MI250X (`hip 7.0.51831`, `OK 3 epochs ...`). That
is the mount, the importability and the GPU in one run. **Still unverified:** the batch path —
`sbatch` with `train.sbatch` (which is also what fills the run manifest's layer and image hashes),
`--keep-stage`, the human (non-JSON) report branch, `--allow-shadow`, and a conflict end to end (the
refusal path is tested, but has never been seen against a real image).
