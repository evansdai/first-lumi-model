# Your first model on LUMI — from a conda environment to a GPU job

This folder is a complete, small example of the thing you actually want to do: **take an
environment you already have, put it on LUMI in the form LUMI wants it, debug it interactively on a
GPU, then train.** It is aimed at readers in ralab. Please consider contributing by giving feedback
after you try it.

Six steps, about 45 minutes of **hands-on** time — plus however long your jobs queue, which is not
predictable. Well under 1 GPU-hour of the project's allocation.


```text
your laptop                       LUMI login node                    LUMI-G compute node
-----------                       ---------------                    -------------------
environment.yml   --scp/upload-->  first-lumi-model/
train.py  (CPU run, seconds)      |
                                  |  step 2: pin the AI image       step 4: srun --pty bash
                                  |  step 3: build one layer file     (debug on a GCD)
                                  |                                    |
                                  |  step 5: sbatch  ---------------->  train, write results
                                  |  step 6: read metrics.json  <------|
```

## The mental model (read this once, two minutes)

Five facts explain every LUMI instruction you will meet.

1. **There are two kinds of machine, not one.** You `ssh` to a *login node* — small, shared, for
   editing and submitting. Your code runs on *compute nodes*, reached only through the scheduler
   (Slurm). You never `ssh` to a compute node.
2. **The GPU is billed per GCD, per minute.** A GCD is half of one MI250x card; a LUMI-G node has
   eight. On the `small-g` partition the bill is

   `max(ceil(cores/8), ceil(memory_GB/64), GCDs) × hours × 0.5` GPU-hours

   — **so one GCD for one hour is 0.5 GPU-hours**, and *1 GCD, 8 cores, 32 GB* is the cheapest
   shape that can use a GPU. The ceilings are what bite: asking for 16 cores instead of 8 doubles
   the bill for the same work, because you are billed in slices of 8 cores and 64 GB per GCD.
   ([billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/), checked 2026-09-21)
3. **Never install a Python environment onto the shared filesystem.** An environment is tens of
   thousands of small files; LUMI's filesystems are built for large ones, and LUMI's own docs say a
   file-quota increase caused by Conda is refused. The environment arrives as **one file**.
4. **You do not build the GPU stack.** LUMI publishes ready-made AI containers with PyTorch and ROCm
   (AMD's GPU software stack — the equivalent of NVIDIA's CUDA, which you may have met instead)
   already in them. Your job is to *add your packages* to one of those, not to build a PyTorch.
5. **Debug interactively first, then submit a batch job.** `srun --pty` gives you a shell on a
   compute node with a GPU for half an hour. Only when the thing works there do you `sbatch` it.

## What you end up with

```text
first-lumi-model/
├── README.md                     you are here
├── AGENTS.md                     the same example, written for a terminal AI agent
├── env.sh                        your project id and paths -- the one file you edit
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
container on a GPU. The LUMI adaptation is the other four: `env.sh`, `build-layer.sh`, the extras list
(`workflow/envs/extra-requirements.txt`, which is written *against this image*) and
`workflow/profiles/lumi-g/train.sbatch`.
where you find your own bugs.

## Files and commands on LUMI you will actually use

| Where | What it is |
|---|---|
| `/scratch/<project>/<user>/` | Your own space: large, fast, **not backed up**. Put code, images and runs here. |
| `/project/<project>/` | Shared with your project. Read-mostly. |
| `$HOME` (`/users/<user>`) | 20 GB and 100 000 files, not expandable. Configuration only — never a code checkout. |
| `~/.bashrc` | The file that runs for every shell you open. This is where `env.sh` gets sourced. |
| `/appl/local/laifs/containers/` | LUMI's prebuilt AI images, read-only. `lumi-multitorch-latest.sif` is a symlink into a versioned directory. |
| `/tmp` on a **compute node** | RAM, not disk, and it is charged against your job's memory. Fine for scratch, not for data. |
| `lumi-workspaces` | Prints your projects, quotas and allocations. The first command to run. |
| `lumi-quota -v`, `lumi-allocations` | Disk usage and file counts; GPU-hours and core-hours left. |
| `sinfo -s`, `squeue --me`, `sacct -X --me` | Partitions; what you have queued; what finished and what it cost. |
| `www.lumi.csc.fi` | The web interface: file browser and upload, a shell, Jupyter, TensorBoard, a desktop. No tunnel needed. |

## Before you start

- Your LUMI username and project id (`project_465003379`-style). `lumi-workspaces` on a login node
  lists them; the [web interface](https://www.lumi.csc.fi) shows them too.
- `ssh <user>@lumi.csc.fi` works. Login uses MyAccessID, not SSH keys — see
  [First steps](https://docs.lumi-supercomputer.eu/firststeps/accessLUMI/).
- Somewhere to put these files: `/scratch/<project>/<user>/code/`, created below.

**If you work in VS Code rather than a terminal**, three things make this page easier:

1. **Edit files in the editor, not with `$EDITOR`.** Where a step says `$EDITOR env.sh`, open
   `env.sh` in VS Code and change the line. If `$EDITOR` is unset, a terminal command will not work
   at all — use `${EDITOR:-nano} env.sh` as the fallback.
2. **Paste commands into the integrated terminal, and check which machine it is on.** The editor
   window does not decide that: a Remote-SSH window and a web-UI Desktop terminal are different
   contexts. The "Where each command runs" note below is the thing to check.
3. **If you use an AI assistant, tell it to read the local files first** — this folder's
   `AGENTS.md` and `README.md`. Do not assume it reads them on its own; say so explicitly. That file
   lists the specific wrong turns an assistant tends to take here.

## Where each command runs

Two machines are involved and it matters which one you are typing on. 

- **On your laptop** — where you already have conda, and where this folder is a normal directory.
- **On a LUMI login node** — the prompt looks like `you@uanNN:~>` after `ssh <user>@lumi.csc.fi`.

## Step 0 — put these files on LUMI (~5 minutes)

**On your laptop**, upload the folder. Either `scp`:

```bash
scp -r first-lumi-model <user>@lumi.csc.fi:/scratch/<project>/<user>/
```

or use the web interface (Files → navigate to `/scratch/<project>/<user>` → Upload).

**Or skip the hand-editing: run `./setup.sh`.** It asks three questions — your project id, where
the code goes, where runs go — with a recommended default for each, writes `env.sh` for you, and
creates the directories. It tells you why each answer matters as it goes, and prints where to read
more. Everything below is what it does by hand, if you would rather see it.

**On a LUMI login node**, make the directories and set your project id:

```bash
mkdir -p /scratch/<project>/<user>/code
mv /scratch/<project>/<user>/first-lumi-model /scratch/<project>/<user>/code/
cd /scratch/<project>/<user>/code/first-lumi-model
$EDITOR env.sh          # the only edit: PROJECT_ID, e.g. project_465003379
. ./env.sh              # so every command below sees $EXAMPLE_DIR, $LUMI_RUNS, ...

# Cross-check before going further: both must print something real, and the second must be a
# directory. A typo here surfaces much later as a confusing permission error.
echo "$PROJECT_ID" "$USER_SCRATCH"
test -d "/scratch/$PROJECT_ID/$USER" && echo "scratch path OK"

mkdir -p "$LUMI_CODE" "$LUMI_SOFTWARE" "$LUMI_RUNS"
```

`env.sh` is the one file you edit. It sets the four paths this example uses and nothing else.
Add it to `~/.bashrc` once, so new shells know it (replace the path with your own):

```bash
# In VS Code: open ~/.bashrc and add this line, replacing <project> with your own.
# From a terminal, the idempotent version -- running it twice does not add the line twice:
grep -q 'first-lumi-model/env.sh' ~/.bashrc || \
  printf '. /scratch/<project>/%s/code/first-lumi-model/env.sh\n' "$USER" >> ~/.bashrc
```

## Step 1 — run the model on your laptop first (seconds)

**On your laptop** — in the **original** folder you uploaded from, *not* in the LUMI terminal.
The fastest place to find a bug in your own code is your laptop. This example's model is small on
purpose, but the shape is the same as any training script: data, model, loop, metrics.

```bash
conda env create -f environment.yml
conda activate first-model
python workflow/scripts/train.py --out /tmp/first-run --epochs 30
```

You should see a `torchinfo` summary, thirty `epoch` lines with the loss falling, then
`OK 30 epochs ...`. If not, fix it here — nothing about LUMI will be easier.

## Step 2 — pin the AI image (once, ~1 minute)

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
sha256sum "$(cat "$LUMI_SOFTWARE/laifs/base.path")" \
  > "$LUMI_SOFTWARE/laifs/base.path.sha256"

# And prove the pin verifies. A recorded hash nobody checks is not a pin.
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

This is the step that replaces `conda env create`. The picture:

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

The mechanism that makes this cheap is `python -m venv --system-site-packages`: it creates a normal
virtual environment, but one that can still **see the packages already installed in the image**. So
the layer holds only your additions, and the image's torch — the one built for this hardware — is
what your code imports. That is also why the layer is tied to one image, and why the image's version
is recorded beside it.

Then one command builds it, and it refuses to leave a half-built file behind:

```bash
LAYER_ID="first-model-$(date -u +%Y%m%d)"
./build-layer.sh "$LAYER_ID"

export MODEL_LAYER="$LUMI_SOFTWARE/venvs/$LAYER_ID.sqsh"
echo "$MODEL_LAYER"                # every job from now on needs this value
```

The script runs `python -m venv --system-site-packages` *inside* the image, installs your extras, checks
them by importing, verifies the pinned image's checksum, packs the directory into one SquashFS file,
then validates the packed file at the path jobs will mount it. `--system-site-packages` is what makes
this cheap and what ties the layer to this exact image: when LAIF publishes a new base, rebuild the
layer.

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

## Step 4 — poke at it on a real GPU (`srun --pty`, up to 30 minutes)

Debugging in the dark is what makes the first HPC week painful. Ask for one GCD and a shell on the
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
inside a CPU build of PyTorch**, and every timing you measure from there is a lie. In (b), the
seaborn/torchinfo summary appearing at all proves the layer is mounted and importable — that is the
migration working.

Why the `-B` flags: LUMI does **not** mount `/scratch` or `/project` into a container, and
`/scratch/<project>` is a symlink, so you bind the *full* path
([container jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/)). Why
`/user-software/bin/python` and not `python`: that is the layer's own interpreter, the one that can
see your packages.

## Step 5 — train for real with `sbatch` (20 minutes, up to 0.17 GPU-hours)

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

```bash
cd "$LUMI_RUNS"
out=$(ls -1t first-model-*.out | head -1)   # newest log; -t sorts by time
job=${out#first-model-}; job=${job%.out}    # first-model-22123456.out -> 22123456

cat "$out"                       # the training log
cat "$job/metrics.json"          # loss and accuracy per epoch
cat "$job/manifest.json"         # device, torch, hip, host, job id, layer
```

Three things must be true, and they are the same three you checked in step 4: the log says
`device cuda` with a non-null `hip`, the loss falls, and the final line says
`OK 30 epochs ...`. `manifest.json` is the record of what produced the run — the device, the ROCm
build, the resolved image and its sha256, the layer and its sha256, the host and the job id. It does
not name a source revision: this folder is not a git checkout, and if you make it one, add that field
yourself. Without it, two runs are comparable; with it, they are reproducible.

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

## When something goes wrong

Every one of these has an obvious cause; none of them means LUMI is broken.

| What you see | What it means | What to do |
|---|---|---|
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

## What it cost

| Step | Resources | GPU-hours, **if the job used its full time limit** |
|---|---|---|
| step 4, interactive debug | 1 GCD, 8 cores, 32 GB, `--time=00:30:00` | 0.25 |
| step 5, the batch job | 1 GCD, 8 cores, 32 GB, `--time=00:20:00` | 0.17 |

Those are maxima, not the bill: `--time` is how long the job *may* run, and you are charged for the
time it actually ran. The real number is what `lumi-allocations` shows after minus before — so run it
once before you submit anything, and again afterwards.

The same 1-GCD job on `standard-g` would be billed as a whole node — 8× more for identical work
([billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/)). Submitting small jobs
that finish is also what grows the project's allocation, so this is not the place to be shy.

## Where to go next

Five short pointers. Each goes deeper only if you need it — nothing here is required for the
tutorial's six steps.

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

This example is written against LUMI's documentation and against this repository's own run log
(the environment manual's `EVIDENCE.md`, in the author's `lumi-env` repository). Being precise about that is a house rule here,
and it is also how you should read anyone else's example.

| Claim | Status |
|---|---|
| The AI image path, its contents, `latest` being a symlink into a versioned directory | **verified** — read from the release directory and its own published package list, 2026-09-17 |
| A pinned `lumi-multitorch-full` image training a model on one GCD (`hip 7.0.51831`) | **verified** — job `22098853`, 2026-09-16 (a sibling toy model in this repository) |
| `srun --pty bash` giving a shell on a compute node, and jobs submitted from it | **verified** — jobs `22099993`, `22116162`, `22100268`, 2026-09-16/17 |
| `mksquashfs` available on a login node | **verified** — 2026-09-13 |
| **The layer route itself** — venv inside the image at `/user-software` → one `.sqsh` → validate at the final path → mount it in a job → train on one GCD | **verified**, on a real workflow, 2026-09-17/18: the layer was built, accepted under real Singularity and on a GCD, and the model trained (`hip 7.0.51831`, job `22149955`). The full log is `EVIDENCE.md` in the environment manual |
| **`build-layer.sh` in this folder** — this tutorial's own copy of that build | **not yet run on LUMI.** It is step 3, and the first thing to do if something here fails. Its dependency gate was corrected on 2026-09-18 to match what the real image needs |
| This example end to end, as written | **not yet run.** Treat the first pass as the test |

If a step fails in a way this page does not explain, that is a defect in the page: write down the
command and its real output, then fix the page. Both, or the guide and reality diverge.

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
- Web interface: <https://docs.lumi-supercomputer.eu/runjobs/webui/>
- AI agents on LUMI: <https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/>
- Worked examples from LUMI: <https://github.com/Lumi-supercomputer/LUMI-AI-Guide>
