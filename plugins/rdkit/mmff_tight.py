"""Conformers optimised on in MMFF to tolerances tighter than RDKit's own
optimisation of many conformers at once stops at (worker `mmff_tight`): each
from where it stands, as far as it is asked - in processes of their own,
where there are many, as RDKit's ForceField.Minimize holds Python's lock and
threads would take turns.

A chunk is the molecule (RDKit's binary form, its one conformer any), how it
is optimised, and the places of the conformers to optimise; each comes back
as ForceField.Minimize answers - not converged (0 or 1) and its energy - with
the places it ended at. The same chunk gives the same answer here and in a
process of its own: nothing of the process is used but the molecule."""

import os
import threading
import time

from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Geometry import Point3D


def watch(parent):
    """As a process of the pool starts: ended as soon as `parent` - the worker that started it - is gone, however it
    went: the worker stopped by Meno, killed, has no time to end its processes itself. On Windows, by waiting on the
    parent itself; elsewhere, by its being taken over by another (its parent's id no longer the worker's)."""
    if os.name == "nt":
        import _winapi

        handle = _winapi.OpenProcess(0x00100000, False, parent)  # (SYNCHRONIZE: it may be waited on, and nothing else)

        def wait():
            _winapi.WaitForSingleObject(handle, _winapi.INFINITE)
            os._exit(0)
    else:

        def wait():
            while os.getppid() == parent:
                time.sleep(0.25)
            os._exit(0)

    threading.Thread(target=wait, daemon=True).start()


def tightened(chunk):
    """[(not converged, energy, places)] for each conformer of the chunk:
    (molecule, field, iters, force_tol, energy_tol, [places])."""
    binary, field, iters, force_tol, energy_tol, places = chunk
    mol = Chem.Mol(binary)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=field)
    conf = mol.GetConformer()
    out = []
    for xyz in places:
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]))
        # (the force field works on the conformer's own places: the molecule is kept while it does)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        more = ff.Minimize(maxIts=iters, forceTol=force_tol, energyTol=energy_tol)
        out.append((more, ff.CalcEnergy(), list(ff.Positions())))
    return out
