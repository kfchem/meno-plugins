"""The xTB plugin's worker: steps of a workflow done with xtb, and xtb's
outputs read.

A plugin's worker (docs/PLUGINS.md, docs/WORKFLOWS.md), run in a pixi
environment of its own - xtb and Python from conda-forge. One JSON object
per line each way, as every plugin's:

    {"id": 4, "op": "prepare", "step": "optimise", "entries": [...], "options": {...}, "cores": 4}
    {"id": 4, "ok": true, "result": {"jobs": [{"entries": [0], "program": "xtb", "args": [...], "files": [...], "reads": [...]}]}}
    {"id": 5, "op": "collect", "step": "optimise", "entries": [...], "options": {...}, "files": {...}, "log": "...", "ended": "done"}
    {"id": 5, "ok": true, "result": {"outputs": [{...}]}}
    {"id": 6, "op": "read", "name": "run.out", "text": "..."}
    {"id": 6, "ok": true, "result": {...}}

It fills three kinds of step - Optimise, Energy, Frequencies - and runs
nothing itself: `prepare` says what each job is - its input, and the
command Meno runs, apart from Meno, in the job's folder - and `collect`
reads back what a job wrote, as Meno's own output form (lib/calc/output),
or says why it failed. One job for each entry. It reads xtb's outputs,
too - `read`, an output opened by hand - and a job's and an opened one's
are one reading (reader.py): `collect` reads a job's log with it, and adds
only what xtb wrote to files of its own and does not print - an
optimisation's path, and the geometry a run began with.

What xtb is asked is as xtb's own documentation has it
(xtb-docs.readthedocs.io: "Commandline Usage", "Geometry Optimization",
"Calculation of Vibrational Frequencies", "Implicit Solvation"):

- the input as Xmol (XYZ) coordinates in angstroms; `--chrg` the charge,
  `--uhf` the unpaired electrons; `--gfn 1`, `--gfn 2` or `--gfnff`;
  `--alpb` a solvent; `--opt` a level, or `--hess`; `-P` the cores;
- a lone atom - an ion of a salt, say - asked to be optimised is not:
  it has no geometry to optimise, and xtb stops with an error when asked
  to optimise one. Its energy is worked out where it is instead;
- an optimisation's path in `xtbopt.log`, Xmol frames whose comment line
  holds the total energy and the gradient norm, and its last geometry in
  `xtbopt.xyz`; whether it converged as xtb says it in its output (it
  writes `.xtboptok` whether it did or not);
- the solvation as its output says it, which Meno holds to what was
  asked, as of any program's step (the manifest's `solvation`).

It keeps nothing between requests, and reaches no network.
"""

import json
import math
import re
import sys

import reader

# the xtb its environment's lock pins
VERSION = "6.7.1"
PROGRAM = "xtb"

INPUT = "input.xyz"
PATH = "xtbopt.log"
LAST = "xtbopt.xyz"
# the frequencies in Gaussian's form, which xtb writes for MOLDEN ("Calculation of Vibrational Frequencies")
G98 = "g98.out"

METHODS = {
    "gfn2": (["--gfn", "2"], "GFN2-xTB"),
    "gfn1": (["--gfn", "1"], "GFN1-xTB"),
    "gfnff": (["--gfnff"], "GFN-FF"),
}
# the solvents ALPB is parametrised for, with GFN1-xTB, GFN2-xTB and GFN-FF alike
SOLVENTS = {
    "acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene", "ch2cl2", "chcl3", "cs2", "dioxane", "dmf", "dmso",
    "ether", "ethylacetate", "furane", "hexadecane", "hexane", "methanol", "nitromethane", "octanol", "woctanol",
    "phenol", "toluene", "thf", "water",
}
LEVELS = {"crude", "sloppy", "loose", "lax", "normal", "tight", "vtight", "extreme"}
STEPS = {"optimise", "energy", "frequencies"}


class Refused(Exception):
    """What was asked cannot be done: said, not thrown."""


def elements_of(entry):
    atoms = entry.get("atoms") or []
    if not atoms:
        raise Refused("an entry has no atoms")
    out = []
    for a in atoms:
        el = str(a.get("el", "")).strip()
        if not re.fullmatch(r"[A-Z][a-z]?", el):
            raise Refused(f"no element is called {el!r}")
        out.append(el)
    return out


def xyz_of(entry):
    """An entry as Xmol coordinates, in angstroms: the count, a comment line, an atom a line."""
    elements = elements_of(entry)
    lines = [str(len(elements)), str(entry.get("name") or "").replace("\n", " ")[:80]]
    for el, a in zip(elements, entry["atoms"]):
        x, y, z = (float(a.get(k, 0.0)) for k in ("x", "y", "z"))
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise Refused("an atom has no place")
        lines.append(f"{el:<3}{x:>18.10f}{y:>18.10f}{z:>18.10f}")
    return "\n".join(lines) + "\n"


def whole(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != int(v):
        raise Refused(f"{what} is not a whole number")
    return int(v)


def lone(entry):
    """Whether an entry is a single atom: nothing to optimise."""
    return len(elements_of(entry)) == 1


def refuse_isotopes(entry):
    """Frequencies of an entry with an isotope, refused, naming it: xtb (6.7.1)
    says it takes an atom's mass (xcontrol(7): $hess, "isotope", "element
    mass") and that it changed it, but its frequencies and thermochemistry
    come out as they would of the natural element - a deuterium as an H.
    Geometries and energies do not depend on the masses: those are run."""
    for a in entry.get("atoms") or []:
        iso = a.get("isotope")
        if iso:
            el = a.get("el")
            raise Refused(f"xTB's frequencies here take no isotope's mass: {iso}{el} would be given as {el} of natural mass")


def asked_of_all(step, options):
    """What a step's options ask of every entry, checked: a method, a solvent and a level it knows."""
    method = METHODS.get(options.get("method", "gfn2"))
    if method is None:
        raise Refused(f"no method is called {options.get('method')!r}")
    solvent = options.get("solvent", "none")
    if solvent != "none" and solvent not in SOLVENTS:
        raise Refused(f"no solvent is called {solvent!r}")
    if step == "optimise" and options.get("level", "normal") not in LEVELS:
        raise Refused(f"no level is called {options.get('level')!r}")
    return method, solvent


def args_for(step, options, entry, cores):
    method, solvent = asked_of_all(step, options)
    charge = whole(entry.get("charge", 0), "The charge")
    multiplicity = whole(entry.get("multiplicity", 1), "The multiplicity")
    if multiplicity < 1:
        raise Refused("the multiplicity is less than one")
    args = [INPUT]
    if step == "optimise":
        # (a lone atom: its energy where it is)
        if not lone(entry):
            args += ["--opt", options.get("level", "normal")]
    elif step == "frequencies":
        refuse_isotopes(entry)
        args += ["--hess"]
    args += method[0] + ["--chrg", str(charge), "--uhf", str(multiplicity - 1)]
    if solvent != "none":
        args += ["--alpb", solvent]
    if cores:
        args += ["-P", str(whole(cores, "The cores"))]
    return args


def op_prepare(m):
    """A job for each entry. One this plugin will not run - an isotope's
    frequencies, a charge that is no whole number - is refused alone, saying
    why (`refused`), and the others are run; where every one is, or what is
    asked of them all cannot be done - a method, a solvent or a level it does
    not know - the step is refused as a whole."""
    step = m.get("step")
    if step not in STEPS:
        raise Refused(f"this plugin does no {step!r}")
    options = m.get("options") or {}
    entries = m.get("entries") or []
    if not entries:
        raise Refused("nothing came in")
    asked_of_all(step, options)
    jobs = []
    refused = []
    first = None
    for i, entry in enumerate(entries):
        try:
            args = args_for(step, options, entry, m.get("cores"))
        except Refused as e:
            # (said as a refusal of the whole step is: "xTB: ...")
            refused.append({"entry": i, "why": f"xTB: {e}.", "kind": "refused"})
            first = first or e
            continue
        reads = [PATH, LAST] if step == "optimise" and not lone(entry) else [G98] if step == "frequencies" else []
        jobs.append({
            "entries": [i],
            "program": PROGRAM,
            "args": args,
            "files": [{"name": INPUT, "text": xyz_of(entry)}],
            "reads": reads,
        })
    if not jobs:
        raise first
    return {"jobs": jobs, **({"refused": refused} if refused else {})}


# --- what a job wrote ---------------------------------------------------------


def frames_of(text, count):
    """Xmol frames: each one's elements, its coordinates, as x, y, z of every atom, and its comment line."""
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        if not head.isdigit() or int(head) != count or i + 2 + count > len(lines):
            break
        comment = lines[i + 1]
        elements = []
        xyz = []
        for line in lines[i + 2: i + 2 + count]:
            parts = line.split()
            if len(parts) < 4:
                return frames
            elements.append(parts[0].capitalize())
            xyz += [reader.number(p) for p in parts[1:4]]
        if len(xyz) != 3 * count:
            break
        frames.append((elements, xyz, comment))
        i += 2 + count
    return frames


def energy_in(comment):
    """The total energy a frame's comment line holds: its first number ("energy: -5.170000000000 gnorm: 0.1 xtb: 6.7.1")."""
    m = re.search(reader.FLOAT, comment)
    return reader.number(m.group(0)) if m else None


def g98_intensities(text):
    """Each vibration's frequency (cm-1) and IR intensity (km/mol) as xtb's `g98.out` gives them, in its Gaussian
    form: "Frequencies --   612.3456   612.3456  1412.3456" over "IR Inten    --    67.8901    67.8900     0.0000"."""
    freqs, irs = [], []
    for line in (text or "").splitlines():
        head, _, values = line.partition("--")
        if head.strip() not in ("Frequencies", "IR Inten"):
            continue
        try:
            numbers = [float(v) for v in values.split()]
        except ValueError:
            return []
        (freqs if head.strip() == "Frequencies" else irs).extend(numbers)
    return list(zip(freqs, irs)) if len(freqs) == len(irs) else []


def overflowed_filled(results, g98):
    """The vibrations' IR intensities xtb's output printed too wide for their field - not known as read - from its
    `g98.out`, where it gives a vibration of the same frequency (to the hundredth printed): never another's."""
    vibrations = next((r for r in results if r["id"] == "vibrations"), None)
    if vibrations is None or [c["label"] for c in vibrations["columns"]][-1:] != ["IR intensity"]:
        return
    given = g98_intensities(g98)
    for row in vibrations["rows"]:
        if row["cells"][-1] is None:
            same = [ir for f, ir in given if abs(f - row["cells"][1]) <= 0.005 + 1e-9]
            if same:
                row["cells"][-1] = same[0]


def output_of(step, entry, options, log, files):
    """A job's output: its log, read as an output opened by hand is (reader.py) - and, from what xtb wrote to files
    of its own and does not print, the geometry it began with (the entry's) and an optimisation's path. Its charge
    and multiplicity are those it was asked to run at."""
    elements = elements_of(entry)
    n = len(elements)
    # (a run that stopped on an error is no output, though its process ended: why, as xtb says it)
    ended, why, _ = reader.run_of(log)
    if ended == "error":
        raise Refused(why)
    try:
        out = reader.read("xtb output", log)
    except ValueError as e:
        raise Refused(str(e).rstrip(".")) from e
    if out["atoms"] != elements:
        raise Refused("xtb wrote back another molecule")
    start = [float(entry["atoms"][i].get(k, 0.0)) for i in range(n) for k in ("x", "y", "z")]
    total = reader.total_energy(log)
    if step == "optimise" and n > 1:
        path = frames_of(files.get(PATH) or "", n)
        if not path:
            raise Refused("xtb wrote no optimisation")
        last = frames_of(files.get(LAST) or "", n)
        if any(els != elements for els, _, _ in path + last[:1]):
            raise Refused("xtb wrote back another molecule")
        frames = [xyz for _, xyz, _ in path]
        energies = [energy_in(c) for _, _, c in path]
        if last and last[0][1] != frames[-1]:
            frames.append(last[0][1])
            energies.append(energy_in(last[0][2]))
        if any(e is None for e in energies):
            energies = None
    elif total is None:
        raise Refused("xtb wrote no energy")
    else:
        frames, energies = [start], [total]
        # (a lone atom asked to be optimised: where it is, as optimised as it can be)
        if step == "optimise":
            out["optimised"] = True
    out.update({"frames": frames, "energies": energies, "charge": whole(entry.get("charge", 0), "The charge"),
                "multiplicity": whole(entry.get("multiplicity", 1), "The multiplicity")})
    # (where the log does not say the run came to its end, none of its geometries is one it found - as read by hand)
    if out.get("unmarked"):
        out["converged"] = [False] * len(frames)
    if out.get("version") is None:
        out["version"] = VERSION
    if out.get("method") is None:
        out["method"] = METHODS.get(options.get("method", "gfn2"), (None, None))[1]
    # (a frequency job's imaginary frequencies, as xtb judged them; another's, not judged)
    if step != "frequencies":
        out.pop("imaginary", None)
    else:
        overflowed_filled(out["results"], files.get(G98))
    return out


def op_collect(m):
    step = m.get("step")
    if step not in STEPS:
        raise Refused(f"this plugin does no {step!r}")
    files = m.get("files") or {}
    log = m.get("log") or ""
    entries = m.get("entries") or []
    if m.get("ended") != "done":
        return {"why": reader.why_of(log)}
    return {"outputs": [output_of(step, e, m.get("options") or {}, log, files) for e in entries]}


def op_read(m):
    """An xtb output opened by hand, read as a job's log is."""
    name = str(m.get("name") or "output").replace("\\", "/").rsplit("/", 1)[-1]
    try:
        return reader.read_jobs(name, m.get("text"), m.get("kind"))
    except ValueError as e:
        raise Refused(str(e).rstrip(".")) from e


def finite(value):
    """`value` with each number that is not one - not a number, or infinite - as not known: JSON has neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


OPS = {"prepare": op_prepare, "collect": op_collect, "read": op_read}


def answer(line):
    try:
        m = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "not JSON"}
    op = OPS.get(m.get("op"))
    try:
        if op is None:
            raise Refused(f"no such request: {m.get('op')!r}")
        return {"id": m.get("id"), "ok": True, "result": finite(op(m))}
    except Refused as e:
        return {"id": m.get("id"), "ok": False, "error": f"xTB: {e}."}
    except Exception as e:  # noqa: BLE001 - said, not thrown: the worker keeps answering
        return {"id": m.get("id"), "ok": False, "error": f"xTB: {e}"}


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
