# AGENTS.md — the contract for a coding agent working in this folder

*If you are a human:* many terminal coding agents read a file called `AGENTS.md` on their own; that
convention is shared across several mainstream tools. If yours does not, start the session with
"read `AGENTS.md` and `README.md` first". Either way, this file is the agent's instructions, not a
substitute for `README.md` — read that one yourself. And do read LUMI's own
[AI agent guide](https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/): you are
responsible for everything your agent does, and it lists the ways this goes wrong.

---

You are working in a **teaching example**. Its reader is an experienced Python user who has never
used an HPC cluster, and it is one of their first LUMI jobs. Your job is to make it work and to keep
it honest.

## What is in here

| Path | It is |
|---|---|
| `README.md` | the human's six steps. The specification for what you are helping with. |
| `env.sh` | the paths. The user edits `PROJECT_ID`; you normally should not. |
| `build-layer.sh` | step 3: extras list → one `.sqsh` layer, inside the pinned AI image. |
| `workflow/envs/extra-requirements.txt` | what LUMI must add to the image. Read its header before changing it. |
| `workflow/scripts/train.py`, `src/minimodel/` | portable Python. No LUMI path belongs in either. |
| `workflow/profiles/lumi-g/train.sbatch` | the only other LUMI-shaped file: resources, binds, paths. |
| `tools/check_platform.py` | a verbatim copy of `~/lumi/templates/tools/check_platform.py`. Refresh it from there; do not edit the copy. |

The design is one sentence: **the AI image is fixed, your layer is the only thing you build, and the
model and training code stay portable.** Anything you add that blurs this is a regression, not a
feature.

## Hard boundaries

1. **You probably cannot run the interesting commands.** The usual LUMI agent session is inside a
   container with no `singularity` and no Slurm client. Check (`command -v singularity srun sbatch`)
   and say plainly what you could not run rather than describing what it would have printed.
2. **GPU-hours are the user's decision.** Do not submit, and do not suggest a submit command without
   saying what it will cost — one GCD, 8 cores, 32 GB is `0.5` GPU-hours per hour on `small-g`.
3. **Never install packages onto the shared filesystem.** No `conda create`, no `pip install
   --target` onto `/scratch`, `/project`, `/flash` or `$HOME`. Installation happens inside the build
   (staged in node-local `/tmp`) or not at all.
4. **Published artifacts are immutable.** Never overwrite an existing `.sqsh`, `.sif` or manifest. A
   new package set is a new `<layer-id>`.
5. **Never glob the LAIF directory and never write `lumi-multitorch-latest.sif` into a job.** The
   pinned path is `$LUMI_SOFTWARE/laifs/base.path`; read it, print it, use it.
6. **Never `rm`.** Move things aside with `trash` — recoverable, and it refuses when the quota is
   full.
7. **No secrets, and no real data.** No tokens, no credentials, no personal or confidential data in
   any file you write, in any layer, or in any log you paste.
8. **Never list the accelerator framework in an extras file.** `torch`, `numpy`, `pandas`, `scipy`,
   `matplotlib` and friends come from the image; a second copy shadows the ROCm build and is how a
   working job breaks. This is the one rule most likely to look "harmless" to you.
9. **Stay inside this folder.** Other repos and the user's other work are not yours to tidy.
10. **Do not put LUMI paths, module names or vendor checks into `src/` or `workflow/scripts/`.** They
    belong in `env.sh` and `workflow/profiles/lumi-*`, and nowhere else.

## What "done" means

Say a change is done only after the checks below, with the output shown, not summarized:

```bash
bash -n build-layer.sh workflow/profiles/lumi-g/train.sbatch
python3 -m py_compile workflow/scripts/train.py src/minimodel/*.py
```

Then, for claims about LUMI: name the source. Either a documentation URL, or an entry in this
repository's run log (`../../docs/EVIDENCE.md`), or the words **"unverified — and here is the command
that would settle it"**. Never a third option. A confident sentence with no source is the specific
defect this repository exists to correct, and it has been caught twice by review here already.

Also: any change to a script is not finished until the matching sentence in `README.md` (and in this
file, if it lists paths) still describes what the script does.

## How to be useful to a beginner

- **One step at a time.** Give the command, the expected output, and what to do if it differs.
- **Do not invent output.** If you have not seen it, say what you expect and why, and mark it.
- **Keep commands copy-safe**: under ~100 characters per line, `\` continuations, no `python -c` with
  indented bodies (a wrapped line pasted out of a terminal arrives with a real newline and a leading
  space — that has broken commands in this project before).
- **Explain the why in one sentence, not five.** The README is the long form; the chat is not.
- **Prefer the shortest correct change.** No new abstractions, no configuration for values that never
  change, no wrapper script around a script.
- **When you cannot proceed**, hand back the exact command for the human to run on a login or compute
  node, and what to look for in its output.

## A loop that works

1. Read `README.md`, this file, `env.sh` and the file you are about to change.
2. State a short plan and what it will cost (GPU-hours, quota). Wait for agreement.
3. Edit, then run the offline checks above.
4. Hand back the exact commands in order, and the lines that indicate success.
5. When the human pastes real output, read it, say what it means, and fix the *page* that was wrong
   as well as the script.

Prompts that work well in this folder, for the human's benefit:

- "read `AGENTS.md` and `README.md`, then tell me which lines in README step 4 are wrong."
- "here is the job log; why did it fail, and which line of which file should change?"
- "add package X the right way and give me the exact commands."
- "the layer build failed with this output; explain it in two sentences."

## Facts you should not have to re-derive

- **The image**: `/appl/local/laifs/containers/lumi-multitorch-latest.sif` is a symlink into a
  versioned release directory. The pinned copy is written to `$LUMI_SOFTWARE/laifs/base.path` with a
  sha256 beside it. The `full` tower ships python 3.12.3, torch 2.10.0+rocm7.0, numpy 2.3.5,
  pandas 2.3.3, matplotlib 3.11.1, scikit-learn 1.9.0, tensorboard 2.21.0, h5py 3.16.0 — read from
  the image's own published package list on 2026-09-17. To check a package yourself:
  `singularity run "$SIF" pip list | grep -i <name>`.
- **The billing**: on `small-g`, GPU-hours = `max(ceil(cores/8), ceil(mem_GB/64), GCDs) × hours × 0.5`.
  `standard-g` bills a whole node (4 GPU-hours per node-hour) whether you use it or not.
  ([billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/))
- **The container's filesystem**: `/scratch` and `/project` are not mounted into containers, and
  `/scratch/<project>` is a symlink — bind the resolved full path.
  ([container jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/))
- **Interactive**: `srun --pty bash` puts you on a compute node; `salloc` does not (its shell stays on
  the login node). ([interactive jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/))
- **Compute-node `/tmp` is RAM**, charged against the job's memory request.
- **`device cuda` + non-null `hip`** is what a working GPU job prints. `hip None` means a CPU build.
- **LUMI's own agent rules**, worth repeating to the human when relevant: run the agent in a
  container; do not give it credentials; expect it to be wrong about quotas and partitions; never
  share LUMI access with a third-party system.
  ([AI agent guide](https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/))
- **Documentation lookup**: LUMI publishes a public MCP server at
  `https://lumi-aif-agents.2.rahtiapp.fi/mcp` with a `retrieve_docs` tool over the LUMI documentation
  and the LUMI AI Guide. If your client supports MCP, prefer it over guessing; if it does not, fetch
  the page with `curl`.
  ([agent infrastructure](https://docs.lumi-supercomputer.eu/laif/software/agent-infrastructure/))

## Two things that look wrong and are not

- `torch.cuda.is_available()` is `True` and `device cuda` is printed on an **AMD** GPU: ROCm is
  CUDA-compatible at the API level in PyTorch. `torch.version.hip` is the field that tells you the
  ROCm version.
- Nothing in `src/` or `workflow/scripts/` mentions LUMI. That is the point, not an omission: the
  same file must run on the user's laptop and inside the container on a GCD.
