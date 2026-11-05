"""A minimisation with restraints held, run as a job (docs/WORKFLOWS.md,
*Several at once*): Meno's executable runs it, apart from Meno, in the
job's folder, by the RDKit plugin's own Python, as the plugin's `prepare`
said.

    python -u minimise_job.py molecule.json

It reads the molecule - its atoms with their places, its bonds - its
restraints, where they were first held, and the step's options from the
file named; minimises it in the force field asked with its restraints held,
as the worker beside it does (`minimised`); and writes the places it
reached, its energy in hartrees, the force field used, whether it
converged and how far it missed what its restraints hold to minimised.json
for the plugin's `collect` to read back. What
it does it says as it goes, into the job's log.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import worker  # noqa: E402 - the worker beside it: RDKit, and how a molecule is minimised with restraints held
from rdkit import rdBase  # noqa: E402


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        given = json.load(f)
    rows = worker.restraint_rows(given.get("restraints"), len(given.get("atoms") or []))
    # (a coordination bond a distance restraint holds: held by it alone, no bond of the force field's)
    mol = worker.mol_of(given, worker.held_apart(given, rows))
    options = given.get("options") or {}
    field = options.get("field") or "MMFF94"
    iters = int(options.get("steps") or worker.STEPS)
    # (from where its restraints were first held, where the step asks so and they were: else from where its atoms are)
    start = given.get("start") if options.get("start") == "first" else None
    print(f"RDKit {rdBase.rdkitVersion}: {mol.GetNumAtoms()} atoms, {len(rows)} restraints, {field}, at most {iters} steps"
          f"{', from where its restraints were first held' if start is not None else ''}", flush=True)
    made = worker.minimised(mol, rows, field=field, iters=iters, start=start)
    print(f"{made['field']}: {'converged' if made['converged'] else f'not converged in {iters} steps'}", flush=True)
    with open(worker.MINIMISE_OUT, "w", encoding="utf-8") as f:
        json.dump({"version": rdBase.rdkitVersion, "atoms": [{"el": a.GetSymbol()} for a in mol.GetAtoms()], **made}, f)


if __name__ == "__main__":
    main()
