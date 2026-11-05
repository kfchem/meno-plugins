# Inputs of several jobs, for the readers' tests

Two inputs written after the programs' own manuals, each of two jobs: an
HF/STO-3G optimisation of water, then an HF/3-21G single point.

- `link1.gjf`: Gaussian, the second job after a `--Link1--` line, its
  geometry and guess read from the first's checkpoint (Gaussian's manual:
  the `--Link1--` separator, `%Chk`, and the `Geom` and `Guess`
  keywords).
- `multi.inp`: ORCA, the second job after a `$new_job` line (ORCA's
  manual on running several jobs from one input).

Their outputs are not kept here (docs/WORKSPACE.md, stage 3). To read
them in a reader's tests, run each in `calc-samples/jobs/`, where git
ignores what is written:

    g16 < link1.gjf > link1.log
    orca multi.inp > multi.out

The tests that need them are skipped where they are not there.
