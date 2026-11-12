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

from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Geometry import Point3D


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
