"""The PySCF plugin's worker: a Molden file's orbitals and densities, on a grid.

A reader plugin's worker (docs/PLUGINS.md), run in a pixi environment of
its own, PySCF from conda-forge. One JSON object per line each way, as
every reader's:

    {"id": 7, "op": "read", "name": "job.molden", "text": "..."}
    {"id": 8, "op": "ask", "key": "orbital:0:34", "name": "job.molden", "text": "..."}
    {"id": 7, "ok": true, "result": {...}}

A plugin of its own, sharing no code with any other (the maintainer,
2026-10-06): it reads a Molden file with PySCF's own reader
(`pyscf.tools.molden.load`) and gives its molecule - its atoms and their
places, as PySCF reads them - and its orbitals and densities. The format
holds no charge or multiplicity, so it says neither. Its orbitals, in
Meno's general form (each its spin, energy in hartrees, occupation and
symmetry, as PySCF reads them, all of them - Meno lists them), carry each
orbital's surface, and a list of densities theirs, as promises (Meno's
`{"ask": key}`): asked for, the values are worked out on the grid Meno's
request gives (its `grid`), from the file sent again, nothing kept
between. Before any surface is promised,
the orbitals are checked as PySCF reads them - orthonormal in their basis -
so that none is drawn from a basis read wrong.

PySCF is used as a library; none of its code is Meno's.
"""

import base64
import json
import os
import re
import sys
import tempfile
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy  # noqa: E402
from pyscf.tools import molden  # noqa: E402

# ångströms in a bohr (CODATA 2018)
ANGSTROM_PER_BOHR = 0.529177210903
SCHEMA = 1
# how far orbitals may stray from orthonormal, read back, and still be drawn
ORTHONORMAL_MOST = 1e-3
# how many grid points are worked out at once
CHUNK = 20000
# how many points a grid it is given may have at most: as many as Meno lays (lib/calc/grids `GRID_LAID_MOST`), so that a request
# sets no more memory than that
POINTS_MOST = 90 ** 3


def is_molden(text):
    return re.match(r"\s*\[Molden Format\]", text, re.I) is not None


def _load_molden_text(text):
    fd, path = tempfile.mkstemp(suffix=".molden")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return molden.load(path)
    finally:
        os.remove(path)


class Orbitals:
    """A molecule's orbitals as PySCF reads them: its molecule, and, for each
    spin, the orbitals' energies (hartrees), coefficients, occupations and
    symmetries."""

    def __init__(self, loaded):
        mol, energy, coeff, occ, irrep, _spins = loaded
        self.mol = mol
        unrestricted = isinstance(coeff, tuple)
        self.energy = list(energy) if unrestricted else [energy]
        self.coeff = list(coeff) if unrestricted else [coeff]
        self.occ = list(occ) if unrestricted else [occ]
        if irrep is None:
            self.irrep = [None] * len(self.coeff)
        else:
            self.irrep = list(irrep) if unrestricted else [irrep]

    def straying(self):
        """How far the orbitals are from orthonormal, read back: none, read right."""
        s = self.mol.intor("int1e_ovlp")
        return max(float(numpy.abs(c.T @ s @ c - numpy.eye(c.shape[1])).max()) for c in self.coeff)


def load(m):
    """A request's Molden file, its orbitals as PySCF reads them."""
    text = m.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("nothing to read")
    if not is_molden(text):
        raise ValueError("not a Molden file")
    return Orbitals(_load_molden_text(text))


def orbitals_of(orbitals):
    """The orbitals in Meno's general form, as PySCF read them: each its spin
    - of both, where the file gives one set, else alpha then beta - its
    energy (hartrees), its occupation and its symmetry as the file names it,
    and its surface promised; all of them, in the file's order."""
    spins = len(orbitals.coeff)
    out = []
    for s in range(spins):
        sym = orbitals.irrep[s]
        sym = sym if sym is not None and len(sym) == len(orbitals.energy[s]) else None
        for i, (e, occ) in enumerate(zip(orbitals.energy[s], orbitals.occ[s])):
            out.append({
                "spin": "both" if spins == 1 else ("alpha", "beta")[s],
                "energy": float(e), "occupation": float(occ),
                **({"symmetry": str(sym[i])} if sym is not None and str(sym[i]).strip() else {}),
                "surface": {"ask": f"orbital:{s}:{i}"},
            })
    return out


def densities(orbitals, frame):
    """The electron density's list - and the spin density's, of an open shell - each its surface promised."""
    rows = [{"cells": ["Total"], "frame": frame, "surface": {"ask": "density:total"}}]
    if len(orbitals.coeff) == 2:
        rows.append({"cells": ["Spin"], "frame": frame, "surface": {"ask": "density:spin"}})
    return [{
        "id": "densities", "on": "list", "group": "Densities", "label": "Electron density",
        "columns": [{"label": "Density"}], "rows": rows,
    }]


def op_read(m):
    """A Molden file's molecule, as PySCF read it, and its orbitals: its
    charge and multiplicity, which the format does not hold, unsaid."""
    orbitals = load(m)
    mol = orbitals.mol
    atoms = [re.sub(r"[^A-Za-z]", "", mol.atom_pure_symbol(i)).capitalize() for i in range(mol.natm)]
    coords = mol.atom_coords(unit="Angstrom")
    result = {
        "schema": SCHEMA,
        "atoms": atoms,
        "frames": [[float(v) for xyz in coords for v in xyz]],
        "results": [],
    }
    if orbitals.straying() <= ORTHONORMAL_MOST:
        result["orbitals"] = orbitals_of(orbitals)
        result["results"] = densities(orbitals, 0)
    else:
        # (orbitals it could not read back exactly: none drawn wrong - the program's own Molden file reads right)
        result["results"] = [{
            "id": "surfaces", "on": "molecule", "group": "Orbitals", "label": "Surfaces",
            "value": "open a Molden file the program wrote",
        }]
    return result


def grid_given(asked):
    """The grid Meno's request gives (`grid`: its origin, its axes - each the
    step along it - and its points along each, in ångströms), in bohr: its
    origin, the step along each axis, and its points. Its axes run along x,
    y and z, as Meno lays a grid."""
    try:
        origin = [float(v) for v in asked["origin"]]
        axes = [[float(v) for v in a] for a in asked["axes"]]
        counts = [int(v) for v in asked["counts"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Meno's request gives no grid to work the values out on") from None
    if len(origin) != 3 or len(axes) != 3 or any(len(a) != 3 for a in axes) or len(counts) != 3:
        raise ValueError("Meno's request gives no grid to work the values out on")
    if any(axes[k][j] != 0 for k in range(3) for j in range(3) if j != k) or any(axes[k][k] <= 0 for k in range(3)) or any(c < 2 for c in counts):
        raise ValueError("Meno's request gives a grid whose axes do not run along x, y and z")
    if counts[0] * counts[1] * counts[2] > POINTS_MOST:
        raise ValueError(f"Meno's request gives a grid of more points than {POINTS_MOST}")
    return [v / ANGSTROM_PER_BOHR for v in origin], [axes[k][k] / ANGSTROM_PER_BOHR for k in range(3)], counts


def values_on(mol, origin, steps, counts, f):
    """`f` of the basis functions' values (points × functions) at every grid
    point, the last axis running fastest, worked out a chunk at a time."""
    axes = [origin[k] + steps[k] * numpy.arange(counts[k]) for k in range(3)]
    x, y, z = numpy.meshgrid(*axes, indexing="ij")
    points = numpy.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    out = numpy.empty(len(points))
    for start in range(0, len(points), CHUNK):
        ao = mol.eval_gto("GTOval", points[start:start + CHUNK])
        out[start:start + CHUNK] = f(ao)
    return out


def op_ask(m):
    key = str(m.get("key") or "")
    orbitals = load(m)
    if orbitals.straying() > ORTHONORMAL_MOST:
        raise ValueError("this file's orbitals cannot be read back exactly")
    mol = orbitals.mol
    origin, steps, counts = grid_given(m.get("grid") or {})
    hit = re.fullmatch(r"orbital:(\d+):(\d+)", key)
    if hit:
        s, i = int(hit.group(1)), int(hit.group(2))
        c = orbitals.coeff[s][:, i]
        values = values_on(mol, origin, steps, counts, lambda ao: ao @ c)
        signed, quantity = True, "orbital"
    elif key in ("density:total", "density:spin"):
        dms = [(c * occ) @ c.T for c, occ in zip(orbitals.coeff, orbitals.occ)]
        if len(dms) == 1 and key == "density:spin":
            raise ValueError("a closed shell has no spin density")
        dm = dms[0] if len(dms) == 1 else (dms[0] + dms[1] if key == "density:total" else dms[0] - dms[1])
        values = values_on(mol, origin, steps, counts, lambda ao: numpy.einsum("pi,pi->p", ao @ dm, ao))
        signed, quantity = key == "density:spin", "density"
    else:
        raise ValueError(f"no such promise: {key}")
    a = ANGSTROM_PER_BOHR
    return {
        "origin": [float(v) * a for v in origin],
        "axes": [[steps[0] * a, 0.0, 0.0], [0.0, steps[1] * a, 0.0], [0.0, 0.0, steps[2] * a]],
        "counts": counts,
        "values": base64.b64encode(values.astype("<f4").tobytes()).decode("ascii"),
        **({"signed": True} if signed else {}),
        "quantity": quantity,
    }


OPS = {"read": op_read, "ask": op_ask}


def answer(line):
    try:
        m = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "not JSON"}
    rid = m.get("id")
    op = OPS.get(m.get("op"))
    if op is None:
        return {"id": rid, "ok": False, "error": f"no such request: {m.get('op')}"}
    try:
        return {"id": rid, "ok": True, "result": op(m)}
    except Exception as e:  # an answer, never a crash
        return {"id": rid, "ok": False, "error": str(e) or type(e).__name__}


def main():
    import pyscf
    print(json.dumps({"event": "ready", "version": pyscf.__version__}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
