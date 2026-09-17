# Your first model on LUMI — from a conda environment to a GPU job

You know Python and conda on your laptop. You have a LUMI account and you have never used an HPC
cluster. This folder is a complete, small example of the thing you actually want to do: **take an
environment you already have, put it on LUMI in the form LUMI wants it, debug it interactively on a
GPU, then train.**

Six steps, about 45 minutes of your time, well under 1 GPU-hour of the project's allocation.

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
   `max(ceil(cores/8), ceil(memory_GB/64), GCDs) × hours × 0.5` GPU-hours — the ceilings matter, so
   *1 GCD, 8 cores, 32 GB* is the cheapest shape that can use a GPU, and doubling the cores doubles
   the bill for the same work.
3. **Never install a Python environment onto the shared filesystem.** An environment is tens of
   thousands of small files; LUMI's filesystems are built for large ones, and LUMI's own docs say a
   file-quota increase caused by Conda is refused. The environment arrives as **one file**.
4. **You do not build the GPU stack.** LUMI publishes ready-made AI containers with ROCm and
   PyTorch already in them. Your job is to *add your packages* to one of those, not to build a
   PyTorch.
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

## Step 0 — put these files on LUMI (~5 minutes)

From your laptop, upload the folder. Either `scp`:

```bash
scp -r first-lumi-model <user>@lumi.csc.fi:/scratch/<project>/<user>/
```

or use the web interface (Files → navigate to `/scratch/<project>/<user>` → Upload).

*(If this repository is already deployed on your LUMI account as `~/lumi`, the same folder is at
`~/lumi/examples/first-lumi-model` — copy it to `/scratch` rather than editing it in place.)*

Then, on the login node, make the directories and set your project id:

```bash
mkdir -p /scratch/<project>/<user>/code
mv /scratch/<project>/<user>/first-lumi-model /scratch/<project>/<user>/code/
cd /scratch/<project>/<user>/code/first-lumi-model
$EDITOR env.sh          # the only edit: PROJECT_ID, e.g. project_465003379
. ./env.sh              # so every command below sees $EXAMPLE_DIR, $LUMI_RUNS, ...

mkdir -p "$LUMI_CODE" "$LUMI_SOFTWARE" "$LUMI_RUNS"
```

`env.sh` is the one file you edit. It sets the four paths this example uses and nothing else.
Add it to `~/.bashrc` once, so new shells know it (replace the path with your own):

```bash
printf '. /scratch/<project>/%s/code/first-lumi-model/env.sh\n' "$USER" >> ~/.bashrc
```

## Step 1 — run the model on your laptop first (seconds)

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
`seaborn` and `torchinfo`, and nothing else. Then one command builds it, and it refuses to leave a
half-built file behind:

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

**A bigger environment.** Two documented routes, in increasing cost:

- **Extend the AI image into your own container.** A five-line `.def` (`Bootstrap: docker`,
  `From: docker.io/lumiaifactory/lumi-multitorch:torch`, `%post pip install ...`) built on LUMI with
  `module load CrayEnv PRoot; singularity build yours.sif yours.def`; the
  [AI environment page](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/) has both
  examples and warns that the build needs memory — use an interactive job.
- **`cotainr`, from a conda environment file.** `module load CrayEnv; module load cotainr;
  cotainr build my.sif --system=lumi-g --conda-env=environment.yml`. Read
  [cotainr's LUMI ROCm example](https://github.com/DeiC-HPC/cotainr/tree/main/examples/LUMI/conda_pytorch_rocm)
  before writing a `torch` line: the `pytorch` conda channel has no ROCm builds, and a container
  built from it ends up CPU-only.

**More GPUs.** One GCD is one process. Eight GCDs on a node means `--gpus-per-node=8` and either
`--ntasks-per-node=8` (one rank per GCD) or `torchrun --nproc-per-node=8` — never both. The
[LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide) works both shapes out; start
there rather than inventing one.

**Curves while it trains.** The image already ships `tensorboard`. Write one event directory per run
from rank 0, then open the web interface at [www.lumi.csc.fi](https://www.lumi.csc.fi) → TensorBoard
([web UI apps](https://docs.lumi-supercomputer.eu/runjobs/webui/)).

**Jupyter on a compute node**, if you prefer notebooks to scripts: the same web interface, with
`lumi-multitorch` as the Python.

**Let an agent drive the boring parts.** [`AGENTS.md`](AGENTS.md) in this folder tells any terminal
coding agent what it may and may not do here. Read LUMI's own
[AI agent guide](https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/) too: you
are responsible for everything your agent does, and it should run in a container for that reason.

**The long form of step 3**, with sources and the alternatives, is this repository's
[`docs/50-workflow-envs-multiproject.md`](../../docs/50-workflow-envs-multiproject.md). It also covers
the case this example deliberately skips: several environments in one workflow.

## Status: what has been run on LUMI and what has not

This example is written against LUMI's documentation and against this repository's own run log
([`../../docs/EVIDENCE.md`](../../docs/EVIDENCE.md)). Being precise about that is a house rule here,
and it is also how you should read anyone else's example.

| Claim | Status |
|---|---|
| The AI image path, its contents, `latest` being a symlink into a versioned directory | **verified** — read from the release directory and its own published package list, 2026-09-17 |
| A pinned `lumi-multitorch-full` image training a model on one GCD (`hip 7.0.51831`) | **verified** — job `22098853`, 2026-09-16 (a sibling toy model in this repository) |
| `srun --pty bash` giving a shell on a compute node, and jobs submitted from it | **verified** — jobs `22099993`, `22116162`, `22100268`, 2026-09-16/17 |
| `mksquashfs` available on a login node | **verified** — 2026-09-13 |
| **The layer build in `build-layer.sh`** (venv inside the image → `.sqsh` → validate at `/user-software`), **and the `-B <layer>:/user-software:image-src=/` bind that jobs use** | **not yet run on LUMI.** It is step 3, and the first thing to do if something here fails |
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
