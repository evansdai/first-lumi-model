# Run a Python training job on LUMI

From a conda environment on your laptop to a training job on one GPU.

This folder is a worked example of one route to that: it trains the small classifier in
`workflow/scripts/train.py` twice, first on your laptop under conda and then, unchanged, on one
LUMI-G GCD through Slurm. Each run writes a loss curve, `metrics.json` and `manifest.json`.

Steps 0-6 cover the whole route: run the script locally, pin the AI container image LUMI
publishes, build the packages that image is missing into one file, test the environment
interactively on a GPU compute node, and submit the same script to the queue.

```text
your laptop                 LUMI login node                    LUMI-G compute node
------------                ---------------                    -------------------
run train.py with conda     pin the image (step 2)
                            build the layer (step 3)
                            sbatch (step 5) ------------------>  the same train.py
                            read results (step 6) <-----------    on one GCD
```

The script does not change between the machines; only the runtime around it is LUMI-shaped.
Step 1 is the laptop run, steps 2, 3 and 6 happen on a login node, and steps 4 and 5 execute on a
compute node.

## What you need

- Your LUMI username and project id (`project_465003379`-style). `lumi-workspaces` on a login node
  lists them; the [web interface](https://www.lumi.csc.fi) shows them too.
- `ssh <user>@lumi.csc.fi` working. Login uses MyAccessID, not SSH keys — see
  [First steps](https://docs.lumi-supercomputer.eu/firststeps/accessLUMI/).
- Python and conda on your laptop, for step 1.
- Somewhere to put these files: `/scratch/<project>/<user>/code/`, created in step 0.

You do not need to know Slurm, Singularity or Lustre to follow this; each one is introduced where it
is first used. If you are new to HPC, read `Five facts about LUMI` and `Files and commands on LUMI
you will actually use` before step 0 — they are what every instruction below assumes. If you already
use Slurm and containers, go straight to step 0.

**Using VS Code rather than a terminal?** Edit `env.sh` in the editor, not with `$EDITOR` (if
`$EDITOR` is unset, a terminal command will not work at all — use `${EDITOR:-nano} env.sh`). When
you paste a command, check which machine the terminal is on: a Remote-SSH window and the web UI's
desktop are different places, and each step below says where it expects you to be. If you use a
coding assistant, tell it to read this folder's `AGENTS.md` and `README.md` first — do not assume it
reads them on its own.

## Five facts about LUMI

Five facts explain every instruction on this page.

1. **There are two kinds of machine, not one.** You `ssh` to a *login node*: small, shared, for
   editing and submitting. Your code runs on *compute nodes*, reached only through the scheduler
   (Slurm). You never `ssh` to a compute node.
2. **You are billed per GCD, and a GCD is half a card.** A LUMI-G node has eight GCDs; one GCD is
   one of the two Graphics Compute Dies in an MI250X module
   ([LUMI-G hardware](https://docs.lumi-supercomputer.eu/hardware/lumig/)). On the `small-g`
   partition, *1 GCD, 8 cores, 32 GB* costs 0.5 GPU-hours per hour, and that is the cheapest shape
   that can use a GPU at all ([billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/),
   checked 2026-09-21). The ceilings are what bite: 16 cores, or 128 GB, doubles the bill for the
   same work, because you are billed in slices of 8 cores and 64 GB per GCD. The arithmetic is in
   [What it cost](#what-it-cost).
3. **Never install a Python environment onto the shared filesystem.** An environment is tens of
   thousands of small files; LUMI's filesystems (Lustre) are built for large ones, and LUMI's own
   documentation says a file-quota increase caused by Conda is refused
   ([Python on LUMI](https://docs.lumi-supercomputer.eu/software/installing/python/)). Your
   environment arrives as **one file**.
4. **You do not build the GPU stack.** LUMI publishes ready-made AI containers with PyTorch and
   ROCm already in them — ROCm is AMD's GPU software stack, the counterpart to NVIDIA's CUDA
   ([AI environment](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)). Your job
   is to *add your packages* to one of those images, not to build a PyTorch.
5. **Debug interactively first, then submit a batch job.** `srun --pty` gives you a shell on a
   compute node with a GPU for half an hour. Only once it works there do you `sbatch` it.

## Files and commands on LUMI you will actually use

| Where | What it is |
|---|---|
| `/scratch/<project>/<user>/` | Your own space: large (50TB), fast, **not backed up**. Code, images and runs go here. |
| `/project/<project>/` | Shared with project members. Read-mostly, with a file quota of 50 GB only. |
| `$HOME` (`/users/<user>`) | 20 GB and 100 000 files, not expandable. Suitable for configuration and settings. |
| `/tmp` on a **compute node** | RAM, not disk, and it is charged against your job's memory. Fine for scratch, not for data. |

| Command | What it answers |
|---|---|
| `lumi-workspaces` | Which projects you are in, and your quotas. The first command to run. |
| `lumi-quota -v` | Disk usage and file counts. |
| `lumi-allocations` | GPU-hours and core-hours left. |
| `sinfo -s`, `squeue --me`, `sacct -X --me` | Partition state; what you have queued; what finished, and what it cost. |
| `www.lumi.csc.fi` | The web interface: file browser and upload, a shell, Jupyter, TensorBoard, a desktop. No tunnel needed. |

## What is in this folder

```text
first-lumi-model/
├── README.md                     you are here
├── AGENTS.md                     the same example, written for a terminal AI agent
├── env.sh                        your project id and paths -- the one file you edit
├── setup.sh                      writes env.sh and creates the directories, if you prefer
├── environment.yml               the conda environment from your laptop
├── build-layer.sh                step 3: your packages -> one .sqsh file
├── src/minimodel/                the model and its synthetic data (portable, no LUMI in it)
├── workflow/
│   ├── envs/extra-requirements.txt   what LUMI must ADD to the AI image
│   ├── profiles/lumi-g/train.sbatch  the LUMI-shaped part: resources, binds, paths
│   └── scripts/train.py              the training loop
└── tools/check_platform.py       "is there really a GPU in here?" -- run before you submit
```

The model and the training loop are portable Python: the same files run on your laptop and inside the
container on a GPU. The LUMI adaptation is the other four: `env.sh`, `build-layer.sh`, the extras
list (`workflow/envs/extra-requirements.txt`, which is written *against this image*) and
`workflow/profiles/lumi-g/train.sbatch`.

## Step 0 — put these files on LUMI (~5 minutes)

**Where you type.** From here on each step says *on your laptop* or *on a login node*. A login node
is the shell you get from `ssh <user>@lumi.csc.fi`; its prompt looks like `you@uanNN:~>`.

**On your laptop**, upload the folder. Either `scp`:

```bash
scp -r first-lumi-model <user>@lumi.csc.fi:/scratch/<project>/<user>/
```

or use the web interface (Files → navigate to `/scratch/<project>/<user>` → Upload).

**On a LUMI login node**, move it into place:

```bash
mkdir -p /scratch/<project>/<user>/code
mv /scratch/<project>/<user>/first-lumi-model /scratch/<project>/<user>/code/
cd /scratch/<project>/<user>/code/first-lumi-model
```

Now set your project id. Two ways, and either is fine.

**Edit `env.sh` by hand** — the only file you edit; it sets the paths and nothing else:

```bash
$EDITOR env.sh          # the only edit: PROJECT_ID, e.g. project_465003379
. ./env.sh              # so every command below sees $EXAMPLE_DIR, $LUMI_RUNS, ...

# Cross-check before going further: both must print something real, and the second must be a
# directory. A typo here surfaces much later as a confusing permission error.
echo "$PROJECT_ID" "/scratch/$PROJECT_ID/$USER"
test -d "/scratch/$PROJECT_ID/$USER" && echo "scratch path OK"

# the layer and runs go to
echo "$LUMI_SOFTWARE" "$LUMI_RUNS"
mkdir -p "$LUMI_SOFTWARE" "$LUMI_RUNS"
```

**Or run `./setup.sh`**, which asks three questions — your project id, where the code goes, where
runs go — writes `env.sh` for you, and creates the directories. It says why each answer matters as
it goes. Take the default for the second question: it is where this folder actually is. If you
answer with a different directory, the script stops and prints the commands that move the checkout
there, because `EXAMPLE_DIR` has to name a directory that exists. Then run `. ./env.sh`, so this
shell sees the paths it wrote.

Then add `env.sh` to `~/.bashrc` once, so new shells know it (replace the path with your own):

```bash
# In VS Code: open ~/.bashrc and add this line, replacing <project> with your own.
# From a terminal, the idempotent version -- running it twice does not add the line twice:
grep -q 'first-lumi-model/env.sh' ~/.bashrc || \
  printf '. /scratch/<project>/%s/code/first-lumi-model/env.sh\n' "$USER" >> ~/.bashrc
```

## Step 1 — run the model on your laptop first (seconds)

**On your laptop** — in the **original** folder you uploaded from, *not* in the LUMI terminal. A bug
is easier to find here than on a node. This example's model is small on purpose, but its shape is
the same as any training script: data, model, loop, metrics.

```bash
conda env create -f environment.yml
conda activate first-model
python workflow/scripts/train.py --out /tmp/first-run --epochs 30
```

You should see a `torchinfo` summary, thirty `epoch` lines with the loss falling, then
`OK 30 epochs ...`. Fix anything wrong here first: nothing about LUMI makes a bug easier to see.

## Step 2 — pin the AI image (once, ~1 minute)

**On a LUMI login node.**

LUMI's AI Factory publishes `lumi-multitorch-*` images at `/appl/local/laifs/containers/`
([AI environment](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)). The one you
want is a symlink, and a symlink is a moving target: the day LAIF publishes a new release,
`latest` points somewhere else and your job would run an image you never tested. So resolve it once
and record the exact path.

```bash
LAIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
mkdir -p "$LUMI_SOFTWARE/laifs"
readlink -f "$LAIF" > "$LUMI_SOFTWARE/laifs/base.path"
cat "$LUMI_SOFTWARE/laifs/base.path"
```

```bash
# Optional, and recommended: record the image's sha256 beside its path, so that you
# can prove later that a job ran the image you think it did. It reads the whole
# 14 GB image, so it is the slowest command in this step. Skip it and nothing
# else breaks: step 3 skips its image check, and each run's manifest.json records
# base_image_sha256: unverified instead of a hash.
sha256sum "$(cat "$LUMI_SOFTWARE/laifs/base.path")" \
  > "$LUMI_SOFTWARE/laifs/base.path.sha256"

# And prove the pin verifies: a recorded hash nobody checks is not a pin.
sha256sum -c "$LUMI_SOFTWARE/laifs/base.path.sha256"      # must print OK
```

That file is now *the* definition of your runtime, and every job reads it:

```bash
SIF="$(cat "$LUMI_SOFTWARE/laifs/base.path")"
singularity run "$SIF" pip list | head -30      # what is already inside
singularity run "$SIF" pip list | grep -i seaborn    # nothing: that is ours to add
```

Takes a few seconds; it starts the container without a GPU and prints its packages. **Do this before
writing your extras list** — most of your `environment.yml` is probably already in there.

## Step 3 — migrate the environment: build one layer file (~3 minutes)

**On a LUMI login node**, in a shell that is not inside a container: the build runs `singularity`
and `mksquashfs`, and containers do not nest on LUMI.

This is the step that replaces `conda env create`. LUMI's AI image is 14 GB, read-only and shared;
your layer is one file beside it, holding only what the image lacks:

```text
LUMI's AI image (14 GB, read-only, shared)     your layer (one small file)
┌──────────────────────────────────────────┐   ┌──────────────────────────┐
│ python 3.12   torch+rocm   numpy  pandas │ + │ seaborn, torchinfo,      │
│ matplotlib   tensorboard   h5py   ...    │   │ and every other package  │
│                                          │   │ YOUR code imports that   │
│ everything your code needs, except the   │   │ the image does not ship  │
│ packages you added yourself              │   └──────────────────────────┘
└──────────────────────────────────────────┘
```

At run time one container mounts both: the image as `/`, your layer at `/user-software`.

You write down the difference in `workflow/envs/extra-requirements.txt` — for this example that is
`seaborn` and `torchinfo`, and nothing else.

What makes this cheap is `python -m venv --system-site-packages`: it creates a normal virtual
environment, but one that can still **see the packages already installed in the image**. So the layer
holds only your additions, and the image's torch — the one built for this hardware — is what your
code imports. That is also why the layer is tied to one image, and why the image's version is
recorded beside it.

Then one command builds it, and it refuses to leave a half-built file behind:

```bash
LAYER_ID="first-model-$(date -u +%Y%m%d)"
./build-layer.sh "$LAYER_ID"

export MODEL_LAYER="$LUMI_SOFTWARE/venvs/$LAYER_ID.sqsh"
echo "$MODEL_LAYER"                # every job from now on needs this value
```

The script runs that `venv` *inside* the image, installs your extras, checks them by importing,
verifies the pinned image's checksum (when step 2 recorded one), packs the directory into one
SquashFS file (`.sqsh`), then validates the packed file at the path jobs will mount it. When LAIF
publishes a new base, rebuild the layer.

Then add the `export MODEL_LAYER=...` line to `env.sh`, so new shells have it too.

**What success looks like.** The script ends with `published <path>`, then prints the
`export MODEL_LAYER=...` line to copy. Four files must exist — the layer and three records beside it
— and a leftover `.partial` means the publication was interrupted, so treat that version as
unpublished:

```bash
ls -lh "$MODEL_LAYER"*
# .sqsh  .sqsh.freeze.txt  .sqsh.manifest.txt  .sqsh.sha256
```

<details>
<summary>What the layer is, and why not simply <code>conda create</code> on <code>/scratch</code></summary>

LUMI's [Python page](https://docs.lumi-supercomputer.eu/software/installing/python/) is blunt: a
Conda environment "tends to contain tens to hundreds of thousands of relatively small files", and
loading one "from multiple processes at the same time, puts a lot of strain on the Lustre file
system", so installing Python packages directly onto `/project`, `/scratch` or `/flash` is
"**strongly discouraged**". The same page gives the answer — a container — because it "solves the
'many small files' performance problem".

A SquashFS layer (`.sqsh`) is that answer at its smallest: thousands of files inside, **one** file
on Lustre, mounted read-only, reusable by every job. The disk cost is a few MB instead of a copy of
the whole 14 GB image.

</details>

## Step 4 — debug on a real GPU (`srun --pty`, up to 30 minutes)

**On a LUMI login node.**

This is why the example debugs interactively first: an environment problem is easier to read in a
live shell than in a job log. Ask
compute node that has it ([interactive jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/)):

```bash
srun --account="$PROJECT_ID" --partition=dev-g \
  --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash
```

`srun` waits for the resources, then your prompt is on a node (`nid0XXXXX`) with a GCD reserved for
you. `dev-g` is the debug partition: same cheap billing, shorter queue, meant for exactly this and
not for real training. Variables you exported on the login node travel into this shell with `srun`,
so `$EXAMPLE_DIR` and `$MODEL_LAYER` are already set. Now, *inside that shell*:

```bash
SIF="$(cat "$LUMI_SOFTWARE/laifs/base.path")"

# Check this before the binds below: if EXAMPLE_DIR is empty, the bind arrives as
# ":/workspace:ro", and Singularity reads `ro` as the destination -- the FATAL
# listed under "When something goes wrong" below.
echo "[$EXAMPLE_DIR]"
test -n "$EXAMPLE_DIR" || echo "EMPTY: source env.sh on the login node, rerun srun"

# (a) The gate: is there really a usable GPU in this allocation?
singularity exec --no-home --pwd /workspace \
  -B "$EXAMPLE_DIR:/workspace:ro" \
  "$SIF" python /workspace/tools/check_platform.py cuda

# (b) Train three epochs by hand, into the node's RAM.
mkdir -p /tmp/first-model
singularity exec --no-home --pwd /workspace \
  -B "$EXAMPLE_DIR:/workspace:ro" \
  -B "$MODEL_LAYER:/user-software:image-src=/" \
  -B /tmp/first-model:/run:rw \
  "$SIF" /user-software/bin/python /workspace/workflow/scripts/train.py \
    --out /run --epochs 3 --device cuda

# (c) Read what it wrote, then give the node back.
cat /tmp/first-model/metrics.json
exit
```

Two lines to look for in (a): `hip` non-null and `OK - cuda is usable`. **`hip None` means you are
inside a CPU build of PyTorch**, and any timing from there measures the CPU, not the GPU. In (b), the
seaborn/torchinfo summary appearing at all proves the layer is mounted and importable — that is the
migration working.

And why `cuda` and not `rocm`, on an **AMD** GPU: a ROCm build of PyTorch deliberately reuses the
CUDA interface, and its own note lists `rocm` and `hip` as *invalid* device strings — `cuda` is the
name for a GCD ([HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)). **HIP** is
AMD's CUDA-equivalent runtime and kernel language, the thing ROCm builds from
([What is HIP?](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)); the `hip` line
above is the version of it this torch was built against, so a CUDA or CPU build prints `None`.

Why the `-B` flags: LUMI does **not** mount `/scratch` or `/project` into a container, and
`/scratch/<project>` is a symlink, so you bind the *full* path
([container jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/)). Why
`/user-software/bin/python` and not `python`: that is the layer's own interpreter, the one that can
see your packages.

## Step 5 — submit the training job with `sbatch` (20 minutes, at most 0.17 GPU-hours)

**On a LUMI login node.**

A batch job is the same commands, written down, with a resource request on top, submitted to the
queue ([Slurm quickstart](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/)).
The log lands in the directory you submit from, so submit from the runs directory:

```bash
cd "$LUMI_RUNS"
sbatch --account="$PROJECT_ID" \
  "$EXAMPLE_DIR/workflow/profiles/lumi-g/train.sbatch"
squeue --me                       # PD = waiting, R = running
```

`squeue --me` will stop showing the job when it finishes. If it never starts, `squeue --me --start`
gives Slurm's own estimate, and `sinfo -s` shows partition state.

The job runs the same gate as step 4 before it trains — `check_platform.py cuda`, then training with
`--device cuda`. That is deliberate: a job whose GCD never appeared must **fail loudly** rather than
quietly fall back to the CPU and report a successful run that proved nothing about the GPU.

## Step 6 — read the results

**On a LUMI login node.**

```bash
cd "$LUMI_RUNS"
out=$(ls -1t first-model-*.out | head -1)   # newest log; -t sorts by time
job=${out#first-model-}; job=${job%.out}    # first-model-22123456.out -> 22123456

cat "$out"                       # the training log
cat "$job/metrics.json"          # loss and accuracy per epoch
cat "$job/manifest.json"         # device, torch, hip, host, job id, layer
```

Three things must be true, and they are the same three you checked in step 4: the log says
`device cuda` with a non-null `hip`, the loss falls, and the final line says `OK 30 epochs ...`.

`manifest.json` is the record of what produced the run: the device, the ROCm build, the resolved
image and — if step 2 recorded one — its sha256, the layer and its sha256, the host and the job id.
It does not record the source revision. For runs comparable by source as well as by runtime, add
`git -C "$EXAMPLE_DIR" rev-parse HEAD` and whether the worktree was clean. Provenance is not
reproducibility, though: that also needs the same seed, arguments and inputs.

To see the curve, copy it to your laptop:

```bash
# on your laptop. Replace the job id with the one in the filename above.
scp <user>@lumi.csc.fi:/scratch/<project>/<user>/runs/22123456/loss.png .
```

and to see the cost:

```bash
lumi-allocations                 # before and after; subtract
sacct -X --me --starttime today --format=JobID,Partition,Elapsed,State
```

## What it cost

| Step | Resources | GPU-hours, **if the job used its full time limit** |
|---|---|---|
| step 4, interactive debug | 1 GCD, 8 cores, 32 GB, `--time=00:30:00` | 0.25 |
| step 5, the batch job | 1 GCD, 8 cores, 32 GB, `--time=00:20:00` | 0.17 |

On `small-g` the bill is
`max(ceil(cores/8), ceil(memory_GB/64), GCDs) × hours × 0.5` GPU-hours, so each row above is one
slice of 8 cores and 64 GB for as long as the job ran.

Those are maxima, not the bill: `--time` is how long the job *may* run, and you are charged for the
time it actually ran. The real number is what `lumi-allocations` shows after minus before — so run it
once before you submit anything, and again afterwards.

The same 1-GCD job on `standard-g` would be billed as a whole node — 8× more for identical work
([billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/)).

## When something goes wrong

Each of these has a cause you can check. None of them means LUMI is broken.

| What you see | What it means | What to do |
|---|---|---|
| `FATAL: container creation failed: unable to add /workspace to mount list: destination must be an absolute path` | `$EXAMPLE_DIR` was **empty** in that shell — not a directory problem: `-B ":/workspace:ro"` is read as source `/workspace`, destination `ro` | on the compute node, `echo "[$EXAMPLE_DIR]"` — `[]` means the shell never sourced `env.sh`. Source it on the login node and rerun `srun --pty` |
| `ModuleNotFoundError: seaborn` | the layer is not mounted, or `python` is the image's own | add `-B "$MODEL_LAYER:/user-software:image-src=/"` and call `/user-software/bin/python` |
| `No such file` for a path you can `ls` on the login node | `/scratch` is not mounted in containers by default | bind the full path: `$LUMI_RUNS`, never `/scratch` |
| `hip None`, or training far slower than expected | CPU build, or no GPU in the allocation | check `--gpus-per-task=1`; run `check_platform.py cuda` |
| `FAILED: ... cuda` from `check_platform.py` | the allocation has no usable GCD | resubmit; check `sinfo -s` for node state |
| matplotlib complains about a config directory | `$HOME` is not mounted (`--no-home`) | already handled in `train.py`; do the same in your own code |
| `srun: error: ... Invalid generic resource (gres) specification` | attaching to a running job with a GPU count that does not fit | see [interactive jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/) → "GPUs resource specification" |
| `sbatch: error: invalid account` | `PROJECT_ID` in `env.sh` is wrong | `lumi-workspaces` prints the right one |
| job killed on a login node (`Killed`, exit 137) | heavy work on a login node: a 24 CPU-core-hour cap is enforced by killing | move it into `srun --pty` |
| quota warnings, `lumi-quota -v` file count exploding | packages were installed onto Lustre | that is exactly what the layer prevents; rebuild it instead |
| the job starts, then dies with `No module named torchinfo` | a package was added to the *image*'s Python instead of the layer | add it to `extra-requirements.txt` and rebuild with a new `LAYER_ID` |
| `error: this is running inside a container, which has no singularity` | you are in a container, and **containers do not nest on LUMI** — a `singularity` inside one cannot start another | find out for certain with `echo "$SINGULARITY_CONTAINER"`: a path means you are in one. Type `exit` to leave it and rerun the command in a plain login shell. On a login node you are normally *not* in one, so meeting this usually means something put you in a container deliberately |
| `OMP: Error #15 ... libomp.dylib already initialized`, **on your laptop, in step 1** | your process holds two OpenMP runtimes: conda-forge's default `numpy` comes with the OpenMP build of OpenBLAS, and pip's `torch` wheel ships its own `libomp.dylib` | re-solve step 1's environment: `conda env update -f environment.yml`, which pins `libopenblas=*=*pthreads*` — the build of OpenBLAS that links no OpenMP, so your process loads the runtime once. (`llvm-openmp` can still be *installed*; what matters is which runtime a process loads.) |

## Where to go next

Five short pointers. Each goes deeper only if you need it — nothing here is required for the
tutorial's steps.

| If you want to… | Read |
|---|---|
| **know where this route stops** — and what to do instead when it does | [`docs/other-ways-to-build-an-environment.md`](docs/other-ways-to-build-an-environment.md) — the same decision as a table: four ways to prepare an environment, and the one question that picks between them |
| **use more than one GCD** | [`docs/more-gpus.md`](docs/more-gpus.md) — one GCD is one process, and `--ntasks-per-node` plus `torchrun` is 64 processes for 8 GCDs |
| **watch a run, or use a notebook** | [`docs/observing-a-run.md`](docs/observing-a-run.md) — TensorBoard and Jupyter through LUMI's web interface |
| **let a coding agent do the boring parts** | [`docs/working-with-an-agent.md`](docs/working-with-an-agent.md) — what to tell it, and the five specific mistakes it will make here |
| **run several environments in one workflow** | the environment manual's §9, "The general pattern: adding a training task" — the long form of step 3, with sources. It lives in the author's `lumi-env` repository alongside this folder; ask for access if you want it |

**The short version of the boundary, in case you read nothing else:** this route is the cheap
default for one class of task — packages you *add* to the AI image. When your extras would
*replace* something the image already defines, you want your own container instead, and
[`docs/other-ways-to-build-an-environment.md`](docs/other-ways-to-build-an-environment.md) is where that
starts.

## Status: what has been run on LUMI and what has not

This example is written against LUMI's documentation and against the author's own run log, kept
separately in the `lumi-env` environment manual. Being precise about what has actually been executed
is a house rule there, and it is also how you should read anyone else's example.

| Claim | Status |
|---|---|
| The AI image path, its contents, `latest` being a symlink into a versioned directory | **verified** — read from the release directory and its own published package list, 2026-09-17 |
| A pinned `lumi-multitorch-full` image training a model on one GCD (`hip 7.0.51831`) | **verified** — job `22098853`, 2026-09-16 (a sibling toy model in the author's repository) |
| `srun --pty bash` giving a shell on a compute node, and jobs submitted from it | **verified** — jobs `22099993`, `22116162`, `22100268`, 2026-09-16/17 |
| `mksquashfs` available on a login node | **verified** — 2026-09-13 |
| **The layer route itself** — venv inside the image at `/user-software` → one `.sqsh` → validate at the final path → mount it in a job → train on one GCD | **verified**, on a real workflow, 2026-09-17/18: the layer was built, accepted under real Singularity and on a GCD, and the model trained (`hip 7.0.51831`, job `22149955`) |
| **`build-layer.sh` in this folder** — this tutorial's own copy of that build | **not yet run on LUMI.** It is step 3, and the first thing to check if something here fails. Its dependency gate was corrected on 2026-09-18 to match what the real image needs |
| This example end to end, as written | **not yet run.** Treat the first pass as the test |

If a step fails in a way this page does not explain, that is a defect in the page: write down the
command and its real output, then fix the page. Both, or the guide and reality diverge. Feedback and
corrections are welcome in the meantime.

## Links

- LUMI user guide: <https://docs.lumi-supercomputer.eu/>
- First steps and access: <https://docs.lumi-supercomputer.eu/firststeps/accessLUMI/>
- Slurm quickstart: <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/>
- Interactive jobs: <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/>
- Partitions and billing: <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/>,
  <https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/>
- Containers on LUMI: <https://docs.lumi-supercomputer.eu/software/containers/singularity/>,
  <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/>
- Installing Python, and why not with conda on Lustre:
  <https://docs.lumi-supercomputer.eu/software/installing/python/>
- AI environment (the images this example uses):
  <https://docs.lumi-supercomputer.eu/laif/software/ai-environment/>
- LUMI-G hardware, and what a GCD is: <https://docs.lumi-supercomputer.eu/hardware/lumig/>
- Web interface: <https://docs.lumi-supercomputer.eu/runjobs/webui/>
- AI agents on LUMI: <https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/>
- Worked examples from LUMI: <https://github.com/Lumi-supercomputer/LUMI-AI-Guide>
