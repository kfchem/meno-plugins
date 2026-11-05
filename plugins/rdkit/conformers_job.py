"""RDKit's conformer search, run as a job (docs/WORKFLOWS.md, *Several at
once*): Meno's executable runs it, apart from Meno, in the job's folder, by
the RDKit plugin's own Python, as the plugin's `prepare` said.

    python -u conformers_job.py molecule.json

It reads the molecule - its atoms with their places, its bonds - and the
search's options from the file named, makes its conformers as the worker
beside it does for a structure drawn (ETKDG v3, then MMFF94, MMFF94s or
UFF), its stereo kept as its 3D structure has it, and writes them to
conformers.json for the plugin's `collect` to read back: its atoms, each
conformer's coordinates in the order RDKit embedded them, their energies in
hartrees, the force field used and how they were made. What it does it says
as it goes, into the job's log.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import worker  # noqa: E402 - the worker beside it: RDKit, and how conformers are made
from rdkit import rdBase  # noqa: E402


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        given = json.load(f)
    rows = worker.restraint_rows(given.get("restraints"), len(given.get("atoms") or []))
    mol = worker.mol_of(given, worker.held_apart(given, rows))
    # (how many to try as Meno sends it: the step's number, or as many as the molecule needs - said here, so that the log has it)
    how = worker.options_of({"options": given.get("options") or {}, "count": given.get("count")})
    print(f"RDKit {rdBase.rdkitVersion}: up to {how['count']} conformers of {mol.GetNumAtoms()} atoms, {how['field']}", flush=True)
    if rows:
        print(f"with {worker.restraints_words(rows)}", flush=True)
    made = worker.conformers(mol, **how, rows=rows)
    print(f"{len(made['frames'])} kept", flush=True)
    with open(worker.CONFORMERS_OUT, "w", encoding="utf-8") as f:
        json.dump({"version": rdBase.rdkitVersion, **made}, f)


if __name__ == "__main__":
    main()
