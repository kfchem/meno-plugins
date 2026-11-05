"""The NCIPLOT interface's worker: non-covalent interactions, worked out by NCIPLOT 4.

A plugin's worker (docs/PLUGINS.md, docs/NCI.md), run in an environment of
its own - Python alone, no package. NCIPLOT itself is a program installed
separately, under its own licence (the GPL): never fetched or shipped by
Meno, found where the system finds programs or where the chemist located
it (the manifest's `installed`), and run apart from Meno. One JSON object
per line each way, as every plugin's:

    {"id": 4, "op": "prepare", "step": "nci", "entries": [...], "options": {...}, "cores": 4}
    {"id": 4, "ok": true, "result": {"jobs": [{"entries": [0], "program": "nciplot", "args": ["input.nci"], "files": [...], "reads": []}]}}
    {"id": 5, "op": "collect", "step": "nci", "entries": [...], "options": {...}, "files": {}, "log": "...", "ended": "done"}
    {"id": 5, "ok": true, "result": {"surfaces": [{"name": "Non-covalent interactions", "rows": [...]}]}}

It fills the kind of step NCI, and runs nothing itself: `prepare` writes
each entry's geometry as an XYZ file - or, asked for interactions between
molecules alone, each of the entry's fragments (the molecules Meno says
its bonds make: docs/PLUGINS.md, *The entry a plugin is given*) in a file
of its own - and NCIPLOT's input; `collect` says which of the cube files
NCIPLOT wrote is the surface and which colours it, for Meno to read
(lib/calc/cube) - or, where the job failed, why. One job for each entry.

The input is written as NCIPLOT 4's own manual has it (NCIPLOT-4.2,
NCIPLOT_MANUAL.pdf, "The input" and "The output"), and from nothing else:

- the number of files to read, and each file's name, a line each: XYZ
  files, for the promolecular density NCIPLOT works with from a geometry;
- ONAME, the name its output files go by (in capitals): NCI;
- OUTPUT 2, its cube files alone: NCI-grad.cube, the reduced density
  gradient (RDG), and NCI-dens.cube, sign(λ2)ρ × 100;
- COARSE or FINE, its grids and their steps, or neither (0.1 Å);
- INTERMOLECULAR, where only interactions between the files' molecules
  are wanted;
- CUTPLOT, the density beyond which the cubes leave the RDG out - the
  option `density-cut`, 0.07 a.u. unless chosen, the manual's default in
  the promolecular case (NCIPLOT_MANUAL.pdf, 11 "CUTPLOT r1 r2") - and the
  RDG a surface is drawn at.

NCIPLOT is run as its README says: given its input, with NCIPLOT_HOME
naming its folder (the manifest's `env`) and OMP_NUM_THREADS its threads
(Meno sets them).

It keeps nothing between requests, and reaches no network.
"""

import json
import math
import re
import sys

VERSION = "0.3.0"
PROGRAM = "nciplot"
INPUT = "input.nci"
NAME = "NCI"
GRAD = f"{NAME}-grad.cube"
DENS = f"{NAME}-dens.cube"
GRIDS = {"coarse": "COARSE", "standard": "", "fine": "FINE"}
# the density beyond which the cubes leave the RDG out, unless chosen: the manual's default in the promolecular case
RHO_CUT = 0.07


class Refused(Exception):
    """What was asked cannot be done: said, not thrown."""


def number(v, what, least, most):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Refused(f"{what} is not a number")
    if not least <= v <= most:
        raise Refused(f"{what} must be from {least} to {most}")
    return float(v)


def atoms_of(entry):
    atoms = entry.get("atoms") or []
    if not atoms:
        raise Refused("an entry has no atoms")
    out = []
    for a in atoms:
        el = str(a.get("el", "")).strip()
        if not re.fullmatch(r"[A-Z][a-z]?", el):
            raise Refused(f"no element is called {el!r}")
        x, y, z = (a.get(k, 0.0) for k in ("x", "y", "z"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, z)):
            raise Refused("an atom has no place")
        out.append((el, float(x), float(y), float(z)))
    return out


def fragments_of(entry, n):
    """The entry's fragments, as Meno gives them: each atom in one, once."""
    given = entry.get("fragments")
    if not isinstance(given, list) or not all(isinstance(f, list) and f for f in given):
        raise Refused("the entry gives no fragments")
    seen = [i for f in given for i in f]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in seen) or sorted(seen) != list(range(n)):
        raise Refused("the fragments do not hold each atom once")
    return given


def xyz_text(atoms, which, title):
    lines = [str(len(which)), title]
    for i in which:
        el, x, y, z = atoms[i]
        lines.append(f"{el:<3}{x:>16.8f}{y:>16.8f}{z:>16.8f}")
    return "\n".join(lines) + "\n"


def files_of(entry, options):
    """An entry's job's files: its geometry - each of its fragments apart, where only interactions between them are wanted - and NCIPLOT's input."""
    atoms = atoms_of(entry)
    grid = options.get("grid", "standard")
    if grid not in GRIDS:
        raise Refused(f"no grid is called {grid!r}")
    iso = number(options.get("isovalue", 0.3), "The surface's RDG", 0.05, 1)
    cut = number(options.get("density-cut", RHO_CUT), "The density left out above", 0.01, 0.5)
    between = options.get("between", False)
    if not isinstance(between, bool):
        raise Refused("Whether only between molecules is neither on nor off")
    parts = fragments_of(entry, len(atoms)) if between else [list(range(len(atoms)))]
    if between and len(parts) < 2:
        raise Refused("only between molecules was asked for, but it is one molecule")
    if len(parts) > 100:
        raise Refused("NCIPLOT reads up to 100 files")
    names = ["molecule.xyz"] if len(parts) == 1 else [f"molecule{k + 1}.xyz" for k in range(len(parts))]
    files = [{"name": nm, "text": xyz_text(atoms, p, str(entry.get("name") or "molecule").replace("\n", " ")[:80])} for nm, p in zip(names, parts)]
    lines = [str(len(names)), *names, f"ONAME {NAME}", "OUTPUT 2"]
    if GRIDS[grid]:
        lines.append(GRIDS[grid])
    if between:
        lines.append("INTERMOLECULAR")
    lines.append(f"CUTPLOT {cut} {iso}")
    files.append({"name": INPUT, "text": "\n".join(lines) + "\n"})
    return files


def op_prepare(m):
    if m.get("step") != "nci":
        raise Refused(f"this plugin does no {m.get('step')!r}")
    options = m.get("options") or {}
    jobs = [{"entries": [i], "program": PROGRAM, "args": [INPUT], "files": files_of(e, options), "reads": []} for i, e in enumerate(m.get("entries") or [])]
    if not jobs:
        raise Refused("nothing came in")
    return {"jobs": jobs}


def why_of(log):
    """Why NCIPLOT stopped, as it says: its last line that says error, else its last."""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"\berror\b", line, re.I):
            return line
    return lines[-1] if lines else "NCIPLOT said nothing"


def op_collect(m):
    if m.get("step") != "nci":
        raise Refused(f"this plugin does no {m.get('step')!r}")
    if m.get("ended") != "done":
        return {"why": why_of(m.get("log") or "")}
    options = m.get("options") or {}
    iso = number(options.get("isovalue", 0.3), "The surface's RDG", 0.05, 1)
    reach = number(options.get("range", 0.07), "The colour scale", 0.01, 0.2)
    # (sign(λ2)ρ as NCIPLOT writes it: × 100)
    row = {
        "label": "Promolecular density",
        "file": GRAD,
        "iso": iso,
        "colour": {"file": DENS, "from": -100 * reach, "to": 100 * reach, "label": "sign(λ2)ρ × 100 (a.u.)"},
    }
    return {"surfaces": [{"name": "Non-covalent interactions", "rows": [row]} for _ in m.get("entries") or []]}


OPS = {"prepare": op_prepare, "collect": op_collect}


def answer(line):
    try:
        m = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "not JSON"}
    op = OPS.get(m.get("op"))
    try:
        if op is None:
            raise Refused(f"no such request: {m.get('op')!r}")
        return {"id": m.get("id"), "ok": True, "result": op(m)}
    except Refused as e:
        return {"id": m.get("id"), "ok": False, "error": f"NCIPLOT: {e}."}
    except Exception as e:  # noqa: BLE001 - said, not thrown: the worker keeps answering
        return {"id": m.get("id"), "ok": False, "error": f"NCIPLOT: {e}"}


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
