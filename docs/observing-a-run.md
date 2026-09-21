# Watching a run: TensorBoard and Jupyter

[← back to the tutorial](../README.md)

Neither of these is needed for the tutorial. They are what you want once the thing trains and you
want to *see* it, or once you would rather poke at data in a notebook than re-run a script.

## TensorBoard

**The image already ships it** — nothing to install. What you have to do is write the event files
somewhere that outlives the job.

- Write **one event directory per run**, from rank 0 only. With several ranks writing to one
  directory you get interleaved, unreadable curves.
- Put it under your runs directory, so the job and the log travel together:

  ```python
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter(log_dir=f"{run_dir}/tb")
  writer.add_scalar("loss", loss, step)
  ```

- Read it through LUMI's web interface: [www.lumi.csc.fi](https://www.lumi.csc.fi) → TensorBoard
  ([web UI apps](https://docs.lumi-supercomputer.eu/runjobs/webui/)).

**There is no graphics protocol on a login node**, so do not expect to render plots over plain SSH.
Either use the web interface, or copy the results down and plot locally — the tutorial's step 6 shows
the `scp` line.

## Jupyter on a compute node

If you prefer notebooks to scripts, the same web interface gives you a Jupyter session on a compute
node with `lumi-multitorch` as the Python — the same image the tutorial uses.

Two things to know before you rely on it:

- **A notebook is a session, not a job.** It dies with the allocation, and nothing restarts it.
- **Billing is the same as any other allocation** — a GCD held open by an idle notebook costs the
  same as a GCD running training. Close it when you are done.

For anything you will run more than once, the tutorial's shape — a script, a bounded run, then
`sbatch` — is the one that survives.
