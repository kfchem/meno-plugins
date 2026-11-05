"""The CREST plugin's worker: a workflow's conformer search done with CREST.

A plugin's worker (docs/PLUGINS.md, docs/WORKFLOWS.md), run in a pixi
environment of its own - CREST, the xtb it runs, and Python from
conda-forge, for macOS and Linux. One JSON object per line each way, as
every plugin's:

    {"id": 4, "op": "prepare", "step": "conformers", "entries": [...], "options": {...}, "cores": 4}
    {"id": 4, "ok": true, "result": {"jobs": [{"entries": [0], "program": "crest", "args": [...], "files": [...], "reads": [...]}]}}
    {"id": 5, "op": "collect", "step": "conformers", "entries": [...], "options": {...}, "files": {...}, "log": "...", "ended": "done"}
    {"id": 5, "ok": true, "result": {"outputs": [{...}]}}

It fills two kinds of step - Conformers and Optimise - and runs nothing
itself: `prepare` says what each job is - its input, and the command Meno
runs, apart from Meno, in the job's folder - and `collect` reads back what
a job wrote, as Meno's own output form (lib/calc/output), or says why it
failed. A conformer search is one job for each entry; an optimisation one
job for each molecule's entries together, CREST optimising them all from
one ensemble file.

What CREST is asked, and what is read of what it writes, are as CREST's
own documentation has them (crest-lab.github.io/crest-docs: "Command Line
Keywords", "Coordinate Files", "Conformational Sampling"), and from
nothing else:

- the input as Xmol (XYZ) coordinates in angstroms; `--chrg` the charge,
  `--uhf` the unpaired electrons; `--gfn1`, `--gfn2`, `--gfnff` or
  `--gfn2//gfnff`; `--alpb` a solvent; `--quick`, `--squick` or
  `--mquick` a quicker search; `--ewin` the energy window in kcal/mol;
  `--T` the cores;
- the conformers in `crest_conformers.xyz`, an ensemble file: Xmol
  frames, in angstroms, each one's comment line its energy in hartrees,
  lowest first;
- each conformer's population - its Boltzmann weight at 298.15 K, the
  rotamers that are degenerate forms of it counted in - and that
  degeneracy, from the table CREGEN prints ("Metadynamics-based
  Conformational Sampling": the columns Erel/kcal, Etot, weight/tot,
  conformer, set, degen, origin; the row that begins each conformer's set
  of rotamers gives the set's weight, its number and its degeneracy), the
  sets in the order of `crest_conformers.xyz`;
- an ensemble optimised with `--mdopt <file>` ("Ensemble Optimization"):
  each structure of the ensemble file given, optimised, written to
  `crest_ensemble.xyz` in the same order, each comment line its energy.

The solvation is as the run said it, never taken from the options: CREST
echoes each option it was given as it starts ("--alpb water : implicit
solvation"), and says no more of it - "ALPB(water)"; with none of the
options that ask for a model, "none". Whether that is what was asked is
Meno's to say, as of any program's step (the manifest's `solvation`).

A lone atom - an ion of a salt, say - has no conformers to search for and
nothing to optimise. The manifest says CREST is given none (`noLoneAtom`),
and Meno sends it none: a lone atom passes through Meno's step as it came
(docs/PLUGINS.md, `steps`).

It keeps nothing between requests, and reaches no network.
"""

import json
import math
import re
import sys

# the CREST its environment's lock pins
VERSION = "2.12"
PROGRAM = "crest"
SCHEMA = 1

INPUT = "input.xyz"
CONFORMERS = "crest_conformers.xyz"
ENSEMBLE_IN = "ensemble.xyz"
ENSEMBLE_OUT = "crest_ensemble.xyz"

METHODS = {
    "gfn2": (["--gfn2"], "GFN2-xTB"),
    "gfn1": (["--gfn1"], "GFN1-xTB"),
    "gfnff": (["--gfnff"], "GFN-FF"),
    "gfn2//gfnff": (["--gfn2//gfnff"], "GFN2-xTB//GFN-FF"),
}
# the solvents ALPB is parametrised for, with GFN1-xTB, GFN2-xTB and GFN-FF alike
SOLVENTS = {
    "acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene", "ch2cl2", "chcl3", "cs2", "dioxane", "dmf", "dmso",
    "ether", "ethylacetate", "furane", "hexadecane", "hexane", "methanol", "nitromethane", "octanol", "woctanol",
    "phenol", "toluene", "thf", "water",
}
SEARCHES = {"full": [], "quick": ["--quick"], "squick": ["--squick"], "mquick": ["--mquick"]}
# the energy window CREST's conformer search keeps by default (kcal/mol)
EWIN = 6.0
STEPS = {"conformers", "optimise"}
FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"


class Refused(Exception):
    """What was asked cannot be done: said, not thrown."""


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


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


def window_of(options):
    ewin = options.get("ewin", EWIN)
    if isinstance(ewin, bool) or not isinstance(ewin, (int, float)) or not math.isfinite(ewin) or ewin <= 0:
        raise Refused("the energy window is not a number above nought")
    return float(ewin)


def level_args(options, entry):
    """What both kinds of run are told: the method, the charge and unpaired electrons, the solvent."""
    method = METHODS.get(options.get("method", "gfn2"))
    if method is None:
        raise Refused(f"no method is called {options.get('method')!r}")
    solvent = options.get("solvent", "none")
    if solvent != "none" and solvent not in SOLVENTS:
        raise Refused(f"no solvent is called {solvent!r}")
    charge = whole(entry.get("charge", 0), "The charge")
    multiplicity = whole(entry.get("multiplicity", 1), "The multiplicity")
    if multiplicity < 1:
        raise Refused("the multiplicity is less than one")
    args = method[0] + ["--chrg", str(charge), "--uhf", str(multiplicity - 1)]
    if solvent != "none":
        args += ["--alpb", solvent]
    return args


def cores_args(cores):
    return ["--T", str(whole(cores, "The cores"))] if cores else []


def args_for(options, entry, cores):
    search = SEARCHES.get(options.get("search", "full"))
    if search is None:
        raise Refused(f"no search is called {options.get('search')!r}")
    return [INPUT] + level_args(options, entry) + search + ["--ewin", f"{window_of(options):g}"] + cores_args(cores)


def optimise_args(options, entry, cores):
    if options.get("method", "gfn2") == "gfn2//gfnff":
        raise Refused("an optimisation is done with one method")
    return ["--mdopt", ENSEMBLE_IN] + level_args(options, entry) + cores_args(cores)


def molecule_key(entry):
    """What an ensemble's structures must share: their atoms, in the same order, bonded the same way, and their charge
    and spin - one molecule's, as an ensemble file holds ("Coordinate Files"); two of another constitution, though
    their atoms come in the same order, each in an ensemble of its own. Which atoms are bonded, not the bonds' orders:
    one molecule's ring drawn as two Kekule structures is one molecule."""
    bonds = tuple(sorted((min(b["a1"], b["a2"]), max(b["a1"], b["a2"])) for b in entry.get("bonds") or []))
    return (tuple(elements_of(entry)), bonds, entry.get("charge", 0), entry.get("multiplicity", 1))


def op_prepare(m):
    step = m.get("step")
    if step not in STEPS:
        raise Refused(f"this plugin does no {step!r}")
    options = m.get("options") or {}
    entries = m.get("entries") or []
    if step == "optimise":
        # (the entries of each molecule together: one ensemble, one job)
        groups = {}
        for i, entry in enumerate(entries):
            groups.setdefault(molecule_key(entry), []).append(i)
        jobs = []
        for members in groups.values():
            first = entries[members[0]]
            args = optimise_args(options, first, m.get("cores"))
            jobs.append({
                "entries": members,
                "program": PROGRAM,
                "args": args,
                "files": [{"name": ENSEMBLE_IN, "text": "".join(xyz_of(entries[i]) for i in members)}],
                "reads": [ENSEMBLE_OUT],
            })
        if not jobs:
            raise Refused("nothing came in")
        return {"jobs": jobs}
    jobs = []
    for i, entry in enumerate(entries):
        args = args_for(options, entry, m.get("cores"))
        jobs.append({
            "entries": [i],
            "program": PROGRAM,
            "args": args,
            "files": [{"name": INPUT, "text": xyz_of(entry)}],
            "reads": [CONFORMERS],
        })
    if not jobs:
        raise Refused("nothing came in")
    return {"jobs": jobs}


# --- what a job wrote ---------------------------------------------------------


def frames_of(text, count):
    """Xmol frames: each one's elements, its coordinates as x, y, z of every atom, and its comment line."""
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
            xyz += [number(p) for p in parts[1:4]]
        frames.append((elements, xyz, comment))
        i += 2 + count
    return frames


def energy_in(comment):
    """The energy an ensemble frame's comment line holds, in hartrees: its first number."""
    m = re.search(FLOAT, comment)
    return number(m.group(0)) if m else None


def version_in(log):
    """The version CREST says it is, as it starts; where its log holds none - cut short before it - the one its
    environment's lock pins, the only CREST there is to have run (pixi.toml, `crest ==2.12`)."""
    m = re.search(r"\bVersion\s+(\d+(?:\.\d+)+)", log)
    return m.group(1) if m else VERSION


# the options that ask CREST for a solvation model ("Command Line Keywords")
SOLVATION_FLAGS = {"--alpb", "--gbsa"}


def solvation_of(log):
    """The solvation a run used, as it said it: CREST's echo of its options ("ALPB(water)"); "none" where the command
    it echoes asked for none; otherwise None."""
    echoed = re.search(r"^\s*--(alpb|gbsa)\s+(\S+)\s*:\s*implicit solvation", log, re.M)
    if echoed:
        return f"{echoed.group(1).upper()}({echoed.group(2)})"
    call = re.search(r"^\s*Command line input:\s*\n\s*>\s*(.*)$", log, re.M)
    if call and not SOLVATION_FLAGS.intersection(call.group(1).split()):
        return "none"
    return None


def solvated(out, log):
    """An output with its solvation as the run said it, where it said one."""
    solvation = solvation_of(log)
    return {**out, **({"solvation": solvation} if solvation is not None else {})}


def why_of(log):
    """Why CREST stopped, as it says: the first line it marks an error or a failure - not the Fortran
    runtime's own, as it stops, nor its backtrace; else the last warning it gave, with the line after
    it (a change in topology: what changed, not the remedies it lists after); else its last line."""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    said = [line for line in lines if not re.match(r"ERROR STOP|Error termination|#\d", line)]
    for line in said:
        if re.search(r"error|failed|abnormal", line, re.I):
            return line.strip("* ").strip() or line
    warned = [k for k, line in enumerate(said) if line.startswith("WARNING!")]
    if warned:
        k = warned[-1]
        return " ".join(said[k:k + 2])[len("WARNING!"):].strip()
    return said[-1] if said else lines[-1] if lines else "CREST said nothing"


def sets_in(log):
    """Each set of rotamers, as CREGEN's last table gives it: its weight and its degeneracy, in the order of the
    sets - which is crest_conformers.xyz's. None, where there is no such table."""
    lines = log.splitlines()
    heads = [i for i, line in enumerate(lines) if "Erel/kcal" in line and "weight/tot" in line and "degen" in line]
    if not heads:
        return None
    sets = {}
    for line in lines[heads[-1] + 1:]:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        # (the row that begins a set: rank, Erel, Etot, its own weight, the set's weight, the set, its degeneracy)
        if len(parts) >= 7 and parts[5].isdigit() and parts[6].isdigit():
            try:
                sets[int(parts[5])] = (number(parts[4]), int(parts[6]))
            except ValueError:
                return None
    if not sets or sorted(sets) != list(range(1, len(sets) + 1)) or any(d < 1 for _, d in sets.values()):
        return None
    return [sets[k] for k in range(1, len(sets) + 1)]


#: What CREGEN prints of how it took conformers as one, each as it prints it ("RMSD threshold : 0.1250"): the
#: result it is said as - its id, its label, its unit as CREST's documentation gives it ("Command Line Keywords":
#: --rthr in angstroms, --bthr a relative difference of rotational constants, --pthr a population, --ewin in
#: kcal/mol). What it does not print is not said: CREST 2.12 prints no pair energy threshold (--ethr).
CREGEN_SETTINGS = [
    ("RMSD threshold", "cregen.rmsd", "RMSD threshold", "Å"),
    ("Bconst threshold", "cregen.bconst", "Rotational-constant threshold (relative)", None),
    ("population threshold", "cregen.population", "Population threshold", None),
    ("conformer energy window  /kcal", "cregen.ewin", "Energy window", "kcal/mol"),
]
#: The group they are said under, on the molecule's card.
CREGEN_GROUP = "Same conformer (CREGEN)"


def cregen_settings_in(log):
    """How CREGEN took conformers as one, as the run's own output prints it - each setting in the last block of them
    it printed - as results of the molecule, each its value as printed and its unit; none of a setting not printed,
    and none at all where it printed none. Never as asked, never a default."""
    out = []
    for printed, key, label, unit in CREGEN_SETTINGS:
        found = re.findall(r"^\s*" + re.escape(printed) + r"\s*:\s*(\S+)\s*$", log, re.M)
        try:
            value = number(found[-1]) if found else None
        except ValueError:
            value = None
        if value is None or not math.isfinite(value):
            continue
        out.append({"id": key, "group": CREGEN_GROUP, "label": label, "on": "molecule", "value": value, **({"unit": unit} if unit else {})})
    return out


def populations_in(log):
    """Each conformer's population, as CREGEN's last table gives it: the weight of each set of rotamers. None, where there is no such table."""
    sets = sets_in(log)
    return [w for w, _ in sets] if sets else None


def degeneracies_in(log):
    """Each conformer's degeneracy - how many rotamers of it CREST found alike - as CREGEN's last table gives it. None, where there is no such table."""
    sets = sets_in(log)
    return [d for _, d in sets] if sets else None


def optimised_of(entries, options, files, log):
    """An ensemble's structures, optimised: each entry's geometry and energy, in the order they went."""
    if not entries:
        return []
    elements = elements_of(entries[0])
    found = frames_of(files.get(ENSEMBLE_OUT) or "", len(elements))
    if len(found) != len(entries):
        raise Refused("CREST wrote back another number of structures")
    if any(els != elements for els, _, _ in found):
        raise Refused("CREST wrote back another molecule")
    method = METHODS.get(options.get("method", "gfn2"), (None, None))[1]
    out = []
    for entry, (_, xyz, comment) in zip(entries, found):
        energy = energy_in(comment)
        out.append({
            "schema": SCHEMA,
            "program": "CREST",
            "version": version_in(log),
            "method": method,
            "charge": whole(entry.get("charge", 0), "The charge"),
            "multiplicity": whole(entry.get("multiplicity", 1), "The multiplicity"),
            "atoms": elements,
            "frames": [xyz],
            "energies": None if energy is None else [energy],
            "results": [],
        })
    return out


def output_of(entry, options, files, log):
    elements = elements_of(entry)
    n = len(elements)
    found = frames_of(files.get(CONFORMERS) or "", n)
    if not found:
        raise Refused("CREST wrote no conformers")
    if any(els != elements for els, _, _ in found):
        raise Refused("CREST wrote conformers of another molecule")
    energies = [energy_in(c) for _, _, c in found]
    method = METHODS.get(options.get("method", "gfn2"), (None, None))[1]
    populations = populations_in(log)
    degeneracies = degeneracies_in(log)
    return {
        "schema": SCHEMA,
        "program": "CREST",
        "version": version_in(log),
        "method": f"iMTD-GC, {method}" if method else None,
        "charge": whole(entry.get("charge", 0), "The charge"),
        "multiplicity": whole(entry.get("multiplicity", 1), "The multiplicity"),
        "atoms": elements,
        "frames": [xyz for _, xyz, _ in found],
        "energies": None if any(e is None for e in energies) else energies,
        "populations": populations if populations and len(populations) == len(found) else None,
        "degeneracies": degeneracies if degeneracies and len(degeneracies) == len(found) else None,
        # (how CREGEN took conformers as one - its thresholds as the run printed them: said, as Meno's own Duplicates
        #  says its own)
        "results": cregen_settings_in(log),
    }


def op_collect(m):
    step = m.get("step")
    if step not in STEPS:
        raise Refused(f"this plugin does no {step!r}")
    files = m.get("files") or {}
    log = m.get("log") or ""
    entries = m.get("entries") or []
    wrote = ENSEMBLE_OUT if step == "optimise" else CONFORMERS
    if m.get("ended") != "done" or wrote not in files:
        return {"why": why_of(log)}
    options = m.get("options") or {}
    if step == "optimise":
        outputs = optimised_of(entries, options, files, log)
    else:
        outputs = [output_of(e, options, files, log) for e in entries]
    return {"outputs": [solvated(o, log) for o in outputs]}


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
        return {"id": m.get("id"), "ok": False, "error": f"CREST: {e}."}
    except Exception as e:  # noqa: BLE001 - said, not thrown: the worker keeps answering
        return {"id": m.get("id"), "ok": False, "error": f"CREST: {e}"}


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
