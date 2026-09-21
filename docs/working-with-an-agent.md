# Letting an agent drive the boring parts

[← back to the tutorial](../README.md)

If you use a coding agent — a terminal agent, or an assistant in your editor — it can do a lot of
this page for you: pin the image, write the extras list, run the build, submit the job, read the
result back.

**Three things to do first.**

1. **Tell it to read the local files.** This folder ships [`AGENTS.md`](../AGENTS.md), which states
   what an agent may and may not do here, and `README.md`, which is the specification. Do not assume
   your assistant reads them on its own — say so explicitly. Whether it picks them up automatically
   depends on the product, the mode and the configuration.

2. **Read [LUMI's own AI agent guide](https://docs.lumi-supercomputer.eu/development/ai-tools/ai-agent-guide/).**
   You are responsible for everything your agent does. It lists the ways this goes wrong.

3. **Know what it will get wrong here.** The mistakes below are specific and repeatable, which is why
   `AGENTS.md` names them:

   | The agent will want to… | Why that breaks |
   |---|---|
   | put `torch` or `numpy` in `extra-requirements.txt` | they are already in the image, and installing a second copy shadows the ROCm build that works |
   | run the build from inside a container | containers do not nest on LUMI; the script refuses, and the refusal is correct |
   | set `SINGULARITYENV_PYTHONPATH` to add a source path | it **replaces** the image's value, which is how the image exposes its whole Python stack — you lose numpy and torch |
   | hand-write a package's `.dist-info` to satisfy an import | it works once and is not a fix; install the distribution instead |
   | write its own `check_platform.py` | this folder already carries the maintained one, and a second copy drifts |

## The one rule that outranks the rest

**Every claim either has a source or is marked unverified.** If your agent tells you something about
LUMI that is not in this page and not on
[docs.lumi-supercomputer.eu](https://docs.lumi-supercomputer.eu/), ask it where that came from. A
plausible-sounding specific is the failure this folder is written to avoid — and it is the one an
agent produces most readily.
