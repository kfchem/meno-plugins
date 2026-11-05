"""The cclib plugin's worker: a calculation program's output, read by cclib.

A reader plugin's worker (docs/PLUGINS.md), run in an environment of its
own. One JSON object per line on stdin, one per line back on stdout, as
every reader's:

    {"id": 7, "op": "read", "name": "job.out", "text": "..."}
    {"id": 7, "ok": true, "result": {...}}
    {"id": 7, "ok": false, "error": "..."}

It reads what it is sent and nothing else - no file of its own choosing,
no network - and says when cclib has been imported:

    {"event": "ready", "version": "1.9rc1"}

An adapter, and no more: the output is given to cclib as it is, and what
cclib returns is mapped into Meno's own plain data, whatever the program
(src/lib/calc/output.ts) - the atoms, each geometry, each one's energy in
hartrees, what the calculation was, each atom's mass; and the rest as
results in Meno's general form (src/lib/calc/results.ts), in Meno's units.
Nothing is read from the output's text but by cclib, and nothing cclib
returns is changed but in one way, as the ORCA and xTB plugins' readers do
it for their programs' outputs: the translations and rotations an output
lists among its vibrations, at a frequency of naught (under 1 cm-1 either
way), are no vibrations in Meno's form, so they are left out and the rest
numbered from one. What Meno does with the rest - a dipole taken about a
centre, a list kept about its frontier - is Meno's (docs/PLUGINS.md, *The
contract*). A program's own plugin reads its outputs where it is added;
this plugin reads them where it is chosen to. cclib is used as a library;
none of its code is Meno's.
"""

import json
import logging
import math
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import cclib  # noqa: E402
from cclib.parser.utils import PeriodicTable, convertor  # noqa: E402

# calories per mole in a hartree, for an entropy: the hartree (4.3597447222071e-18 J, CODATA 2018) times the
# Avogadro constant (6.02214076e23 per mole, exact, SI 2019), over the thermochemical calorie (4.184 J, exact)
CAL_PER_HARTREE = 627509.474
# debye in an elementary charge times an angstrom (e and the debye exact, SI 2019)
DEBYE_PER_E_ANGSTROM = 4.803204712570263

# the form of the answer, as Meno reads it
SCHEMA = 1

# partial charges' schemes, as chemists write them
SCHEMES = {
    "mulliken": "Mulliken",
    "lowdin": "Löwdin",
    "hirshfeld": "Hirshfeld",
    "chelpg": "CHELPG",
    "natural": "Natural (NPA)",
    "apt": "APT",
    "mbis": "MBIS",
    "cm5": "CM5",
}
# a frequency smaller than this, either way, is a translation or a rotation - which some outputs list
# among the vibrations, at naught - and no vibration (cm-1)
FREQUENCY_LEAST = 1.0

ELEMENTS = PeriodicTable().element


def has(data, name):
    return getattr(data, name, None) is not None


def frame_energies(data, frames):
    """Each frame's energy, in hartrees: the most correlated a step has,
    where the program gives one for every frame; or the last one, where it
    read a single geometry; or none."""
    candidates = []
    if has(data, "ccenergies"):
        candidates.append([float(e) for e in data.ccenergies])
    if has(data, "mpenergies"):
        candidates.append([float(row[-1]) for row in data.mpenergies])
    if has(data, "scfenergies"):
        candidates.append([float(e) for e in data.scfenergies])
    for found in candidates:
        if len(found) == frames:
            return [hartrees(e) for e in found]
    if frames == 1:
        for found in candidates:
            if found:
                return [hartrees(found[-1])]
    return None


def hartrees(ev):
    """An energy cclib gives in electronvolts, in hartrees again - by cclib's
    own factor, so that it is the output's value as printed."""
    return convertor(ev, "eV", "hartree")


def method_of(meta):
    """What the calculation was, as its program names it: the functional
    of a DFT calculation, or else the last method it ran."""
    if meta.get("functional"):
        return str(meta["functional"])
    methods = [m for m in meta.get("methods") or [] if m]
    return str(methods[-1]) if methods else None


def scheme_name(scheme):
    return SCHEMES.get(scheme, scheme.replace("_", " ").capitalize())


def per_atom(data, n):
    """Each atom's partial charges and spin populations, scheme by scheme."""
    out = []
    for attr, group, kind, quantity in (
        ("atomcharges", "Partial charges", "charges", "charge"),
        ("atomspins", "Spin populations", "spins", "spin"),
    ):
        if not has(data, attr):
            continue
        for scheme, values in getattr(data, attr).items():
            if scheme.endswith("_sum") or len(values) != n:
                continue
            out.append({
                "id": f"{kind}.{scheme}",
                "on": "atoms",
                "group": group,
                "label": scheme_name(scheme),
                "quantity": quantity,
                **({"digits": 3} if quantity == "number" else {}),
                "values": [float(q) for q in values],
            })
    return out


def per_frame(data, frames):
    """Each geometry's RMS gradient, where the output gives one for every geometry."""
    grads = getattr(data, "grads", None)
    if grads is None or len(grads) != frames:
        return []
    rms = [float((g ** 2).mean() ** 0.5) for g in grads]
    return [{
        "id": "rms-gradient",
        "on": "frames",
        "group": "Optimisation",
        "label": "RMS gradient",
        "unit": "Eh/bohr",
        "digits": 6,
        "values": rms,
    }]


def vibrations(data, n):
    """The vibrations, as a list: each one's frequency (an imaginary one negative), its
    symmetry, reduced mass, IR intensity and Raman activity where the output gives them;
    chosen, the molecule moves along its displacements.
    The translations and rotations an output lists with them, at a frequency of naught (under 1 cm-1 either
    way), are no vibrations in Meno's form: left out, and the rest numbered from one."""
    if not has(data, "vibfreqs"):
        return []
    freqs = [float(f) for f in data.vibfreqs]
    disps = getattr(data, "vibdisps", None)
    syms = getattr(data, "vibsyms", None)
    syms = syms if syms is not None and len(syms) == len(freqs) else None

    def each(name):
        """A value for each vibration, where the output gives one for each."""
        values = getattr(data, name, None)
        if values is None or len(values) != len(freqs):
            return None
        values = [float(v) for v in values]
        return values if all(math.isfinite(v) for v in values) else None

    masses, irs, ramans = each("vibrmasses"), each("vibirs"), each("vibramans")
    columns = [{"label": "Mode", "quantity": "number", "digits": 0}]
    if syms is not None:
        columns.append({"label": "Symmetry", "quantity": "symmetry"})
    columns.append({"label": "Frequency", "quantity": "wavenumber"})
    if masses is not None:
        columns.append({"label": "Reduced mass", "unit": "amu", "digits": 2})
    if irs is not None:
        columns.append({"label": "IR intensity", "unit": "km/mol", "digits": 1})
    if ramans is not None:
        columns.append({"label": "Raman activity", "unit": "Å⁴/amu", "digits": 2})
    rows = []
    for i, f in enumerate(freqs):
        if abs(f) < FREQUENCY_LEAST:
            continue
        cells = [len(rows) + 1] + ([str(syms[i])] if syms is not None else []) + [f]
        cells += [v[i] for v in (masses, irs, ramans) if v is not None]
        row = {"cells": cells}
        if disps is not None and i < len(disps) and len(disps[i]) == n:
            row["move"] = [float(v) for xyz in disps[i] for v in xyz]
        rows.append(row)
    if not rows:
        return []
    return [{"id": "vibrations", "on": "list", "group": "Vibrations", "label": "Vibrations", "columns": columns, "rows": rows}]


def finite(value):
    """`value` with each number that is not one - not a number, or infinite - as not known: JSON has neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def molecule_wide(data):
    """The molecule's thermochemistry, as cclib gives it: its free energy first in the chip's line."""
    out = []

    def add(id_, group, label, value, rank=None, **measure):
        if value is None:
            return
        out.append({
            "id": id_, "on": "molecule", "group": group, "label": label,
            "value": float(value), **measure, **({"rank": rank} if rank is not None else {}),
        })

    add("thermo.free-energy", "Thermochemistry", "Gibbs free energy", getattr(data, "freeenergy", None), rank=1, quantity="energy")
    add("thermo.enthalpy", "Thermochemistry", "Enthalpy", getattr(data, "enthalpy", None), quantity="energy")
    add("thermo.zpve", "Thermochemistry", "Zero-point energy", getattr(data, "zpve", None), quantity="energy")
    entropy = getattr(data, "entropy", None)
    add("thermo.entropy", "Thermochemistry", "Entropy",
        entropy * CAL_PER_HARTREE if entropy is not None else None, unit="cal/(mol·K)", digits=2)
    add("thermo.temperature", "Thermochemistry", "Temperature", getattr(data, "temperature", None), unit="K", digits=2)
    add("thermo.pressure", "Thermochemistry", "Pressure", getattr(data, "pressure", None), unit="atm", digits=2)
    return out


def dipole(data):
    """The dipole moment, as cclib gives it (`moments`: the point it was taken about, in angstroms, then the dipole,
    in debye): its size, its components and that point - Meno takes a charged molecule's about its centre of mass."""
    moments = getattr(data, "moments", None)
    if moments is None or len(moments) < 2 or len(moments[0]) != 3 or len(moments[1]) != 3:
        return []
    origin = [float(v) for v in moments[0]]
    vector = [float(v) for v in moments[1]]
    value = sum(v * v for v in vector) ** 0.5
    if not all(math.isfinite(v) for v in origin + vector) or not math.isfinite(value):
        return []
    return [{"id": "dipole", "on": "molecule", "group": "Properties", "label": "Dipole moment", "value": value,
             "quantity": "dipole", "rank": 2, "vector": vector, "origin": origin}]


def orbitals(data):
    """The orbitals cclib gives, in Meno's general form: each its spin (of both, where cclib gives one set; else alpha,
    then beta), its energy in hartrees, its occupation - that of an orbital at or below cclib's HOMO for its set (two,
    one set; one, each of two), none above - and its symmetry where cclib gives one; every one, in cclib's order. Meno
    lists them itself (src/lib/calc/orbitals.ts). None where cclib gives no HOMO for each set."""
    homos = getattr(data, "homos", None)
    energies = getattr(data, "moenergies", None)
    if homos is None or energies is None or len(homos) not in (1, 2) or len(energies) != len(homos):
        return None
    syms = getattr(data, "mosyms", None)
    spins = ["both"] if len(homos) == 1 else ["alpha", "beta"]
    full = 2.0 if len(homos) == 1 else 1.0
    out = []
    for s, spin in enumerate(spins):
        e = [float(v) for v in energies[s]]
        homo = int(homos[s])
        if not -1 <= homo < len(e):
            return None
        sym = syms[s] if syms is not None and s < len(syms) and len(syms[s]) == len(e) else None
        out += [
            {"spin": spin, "energy": hartrees(v), "occupation": full if i <= homo else 0.0, **({"symmetry": str(sym[i])} if sym is not None and str(sym[i]).strip() else {})}
            for i, v in enumerate(e)
        ]
    return out or None


def converged_of(data, frames):
    """Of an optimisation's or a relaxed scan's geometries, each one's being one cclib marks converged (`optstatus`,
    OPT_DONE); None where it marks none, or not one to each geometry."""
    status = getattr(data, "optstatus", None) if has(data, "optstatus") else None
    if status is None or len(status) != frames or frames < 1:
        return None
    done = getattr(data, "OPT_DONE", 4)
    return [bool(int(s) & done) for s in status]


def scan_said(data, frames):
    """What a run's geometries are, where cclib gives a scan's energies: a relaxed scan's points where there are
    more geometries than points (each point's optimisation steps among them), else a rigid one's."""
    if not has(data, "scanenergies") or len(data.scanenergies) < 2:
        return []
    points = len(data.scanenergies)
    label = "Relaxed scan" if frames > points else "Scan"
    return [{"id": "run.path", "on": "molecule", "group": "Calculation", "label": label, "value": f"{points} points", "rank": 0}]


def parsed(text, suffix):
    """What cclib reads of `text`, from a file of its own named with `suffix`, then let go."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return cclib.io.ccread(path)
    finally:
        os.remove(path)


def op_read(m):
    name = os.path.basename(str(m.get("name") or "output"))
    text = m.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("nothing to read")
    try:
        data = parsed(text, os.path.splitext(name)[1] or ".out")
    except Exception as e:  # noqa: BLE001 - said, as cclib said it
        raise ValueError(f"cclib could not read it ({str(e) or type(e).__name__})") from e
    if data is None or not has(data, "atomnos"):
        raise ValueError(f"No molecule was found in {name}.")
    meta = getattr(data, "metadata", {}) or {}
    atoms = [ELEMENTS[int(z)] for z in data.atomnos]
    frames = []
    if has(data, "atomcoords"):
        for step in data.atomcoords:
            if len(step) == len(atoms):
                frames.append([float(v) for xyz in step for v in xyz])
    masses = getattr(data, "atommasses", None)
    result = {
        "schema": SCHEMA,
        "program": meta.get("package"),
        "version": meta.get("package_version"),
        "method": method_of(meta),
        "basis": meta.get("basis_set"),
        "charge": int(data.charge) if has(data, "charge") else None,
        "multiplicity": int(data.mult) if has(data, "mult") else None,
        "atoms": atoms,
        **({"masses": [float(v) for v in masses]} if masses is not None and len(masses) == len(atoms) else {}),
        "frames": frames,
        "energies": frame_energies(data, len(frames)),
        "optimised": bool(data.optdone) if has(data, "optdone") else None,
    }
    converged = converged_of(data, len(frames))
    if converged is not None:
        result["converged"] = converged
    result["results"] = (
        scan_said(data, len(frames))
        + molecule_wide(data)
        + dipole(data)
        + per_frame(data, len(frames))
        + per_atom(data, len(atoms))
        + vibrations(data, len(atoms))
    )
    given = orbitals(data)
    if given:
        result["orbitals"] = given
    return result


OPS = {"read": op_read}


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
        # (a number that is not one - JSON has none - as not known: an answer Meno can read)
        return {"id": rid, "ok": True, "result": finite(op(m))}
    except Exception as e:  # an answer, never a crash
        return {"id": rid, "ok": False, "error": str(e) or type(e).__name__}


def main():
    print(json.dumps({"event": "ready", "version": cclib.__version__}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()


