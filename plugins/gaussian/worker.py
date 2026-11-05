"""The Gaussian plugin's worker: a molecule written as Gaussian 16 input, a
workflow's steps done with Gaussian 16, and Gaussian 16's output read.

A plugin's worker (docs/PLUGINS.md, docs/FILE-IO.md, docs/WORKFLOWS.md),
run in an environment of its own - Python alone, no package. Gaussian
itself is a program installed separately, under its own licence: never
fetched or shipped by Meno, found where the system finds programs or where
the chemist located it (the manifest's `installed`). One JSON object per
line on stdin, one per line back on stdout, as every plugin's:

    {"id": 3, "op": "write", "kind": "gaussian-input", "name": "water.gjf",
     "molecules": [{"name": "water", "atoms": [...], "bonds": [...]}],
     "options": {"job": "opt freq", "method": "B3LYP", ...}}
    {"id": 3, "ok": true, "result": {"text": "..."}}
    {"id": 4, "op": "prepare", "step": "optimise", "entries": [...], "options": {...}, "cores": 4}
    {"id": 4, "ok": true, "result": {"jobs": [{"entries": [0], "program": "g16", "args": [], "stdin": "input.gjf", ...}]}}
    {"id": 5, "op": "collect", "step": "optimise", "entries": [...], "files": {}, "log": "...", "ended": "done"}
    {"id": 5, "ok": true, "result": {"read": [{"kind": "gaussian", "log": true, "name": "water.log",
        "wavefunction": {"kind": "wfx", "file": "wavefunction.wfx", "name": "water.wfx"}}]}}
    {"id": 6, "op": "read", "name": "water.log", "text": "..."}
    {"id": 6, "ok": true, "result": {"schema": 1, "atoms": [...], "frames": [...], ...}}
    {"id": 7, "op": "ask", "kind": "gaussian-fchk", "key": "expansion", "name": "water.fchk", "text": "..."}
    {"id": 7, "ok": true, "result": {"expansion": {"nuclei": [...], "centres": [...], ...}}}
    {"id": 3, "ok": false, "error": "..."}

It writes what it is given and nothing else - no file of its own choosing,
no network - runs nothing itself, and says when it is ready:

    {"event": "ready", "version": "0.3.0"}

Its steps - Optimise, Energy, Frequencies - each make a job for an entry:
its input written as below, run as Gaussian's "Running Gaussian" page has
it - `g16 <input-file >output-file`: given no job name, Gaussian reads its
input from standard input and writes its output to standard output, which
is the job's log, followed by Meno as it runs; the scratch files where it
runs; its folder `g16root` the one above Gaussian's own and GAUSS_EXEDIR
Gaussian's own. `collect` reads nothing: it says that the log is
Gaussian's output, for Meno's readers to read as one opened is - and,
where Gaussian wrote one, that its wavefunction is kept with the molecule,
for a step that takes one (Multiwfn's NCI) - or, where the job failed,
why, as the output says, judged as `read` judges it (reader.run_of).

`read` reads a Gaussian output - a job's log, or a file opened by hand,
the one reading for both - in Meno's general form (reader.py). `ask`
gives what a formatted checkpoint's reading promised: its wavefunction,
whole (`expansion`), its basis and orbitals as Meno's general form has
them, from which Meno draws each orbital itself.

The input is laid out as Gaussian's own reference has it (gaussian.com,
"About Gaussian 16 Input", "Link 0 Commands", "Molecule Specifications"),
and from nothing else:

- Link 0 commands, each a line beginning with "%", not ended by a blank
  line: %Chk names the checkpoint file, %NProcShared the processors a run
  may use, %Mem its memory, a number of 8-byte words or with a unit.
- The route section, begun by "#", ended by a blank line: the method and
  basis set, empirical dispersion (the "EmpiricalDispersion" keyword: GD2,
  GD3 or GD3BJ), the job (SP, Opt, Freq) and any keywords besides. What
  is typed into it - the method, the basis set, the keywords - names no
  program and no file: a workspace's steps may come from someone else,
  and run when its Run is pressed. So it holds no External, the keyword
  that runs a program the input names (gaussian.com, "External":
  External="RunTink Amber", or in an ONIOM layer), however shortened; no
  "@", which places a file's contents in the input ("About Gaussian 16
  Input": @filename); no "%", which begins a Link 0 command, no "#",
  which begins a route line, no "!", which begins a comment, and no
  double quotes, in which a name is given.
- The title section, ended by a blank line: at most five lines, none of
  the characters the reference says to avoid (@ # ! an en dash _ \\ and
  control characters).
- The molecule specification, ended by a blank line: the charge and spin
  multiplicity on its first line, then each atom's element and its
  Cartesian coordinates, an isotope as "(Iso=n)" after the element.
- For a step's job keeping the wavefunction, "Output=WFX" in the route
  section, and the name of the file it writes, which Gaussian "read[s]
  from the input stream, on a separate line" (gaussian.com, "Output"),
  after the molecule specification, ended by a blank line.

A lone atom - an ion of a salt, say - asked to be optimised by a step is
run as SP: it has no geometry to optimise, and Gaussian, given Opt for
one atom, stops in its optimiser (l103). It comes back where it was, with
its energy.
"""

import json
import math
import os
import re
import sys

import reader

VERSION = "0.5.2"

# The elements by atomic number, from 1: what an atom's element may be.
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
NUMBER = {s: i + 1 for i, s in enumerate(SYMBOLS)}

# The job, as its keywords say it ("SP" is the default, said all the same).
JOBS = {"opt freq": "Opt Freq", "opt": "Opt", "freq": "Freq", "sp": "SP"}
DISPERSIONS = {"none": "", "GD2": "EmpiricalDispersion=GD2", "GD3": "EmpiricalDispersion=GD3", "GD3BJ": "EmpiricalDispersion=GD3BJ"}
# %Mem's value: a number, of 8-byte words, or with its unit after it, no space between.
MEMORY = re.compile(r"^\d+(KB|MB|GB|TB|KW|MW|GW|TW)?$", re.IGNORECASE)
# What the title section should not hold: these, control characters, and (the input being ASCII) anything else.
TITLE_AVOID = re.compile(r"[@#!–_\\\x00-\x1f\x7f]|[^\x00-\x7f]")
# What the route section's typed text may not hold: a file's contents placed (@), a Link 0 command (%), a route line (#), a comment (!), a name given ("").
ROUTE_AVOID = re.compile(r'[@%#!"]')


class Refused(Exception):
    """What it was given cannot be written: said to the chemist as it is."""


def one_line(value, what):
    """A text option as one line: a section ends at a line's end, so a line break in it is none of it."""
    if not isinstance(value, str):
        raise Refused(f"{what} is not text")
    return " ".join(value.split())


def route_words(value, what):
    """Text typed into the route section: one line, naming no program and no file - no External, however shortened, and none of ROUTE_AVOID."""
    text = one_line(value, what)
    if ROUTE_AVOID.search(text):
        raise Refused(f"{what} may hold no @, %, #, ! or double quotes: Gaussian reads them as a file, a Link 0 command, a route line, a comment or a name")
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        w = word.lower()
        if w.startswith("external") or (len(w) >= 3 and "external".startswith(w)):
            raise Refused(f'{what} may not hold "{word}": External runs a program the input names, and a step\'s input runs on whichever computer its workspace is opened on')
    return text


def whole(value, what, least=None):
    """A number option that must be a whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise Refused(f"{what} must be a whole number, not {value!r}")
    n = int(value)
    if least is not None and n < least:
        raise Refused(f"{what} must be at least {least}, not {n}")
    return n


def title_of(value, name):
    """The title section's line: the title given, else the file's name, without what the reference says to avoid."""
    t = " ".join(TITLE_AVOID.sub(" ", value or "").split())
    if not t:
        t = " ".join(TITLE_AVOID.sub(" ", name or "").split())
    return t or "Molecule"


def checkpoint_of(name):
    """The checkpoint file's name: the input's, without its extension, its spaces and what is not ASCII made "_"."""
    stem = re.sub(r"\.[^.]*$", "", (name or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    stem = re.sub(r"[\s\x00-\x1f\x7f]|[^\x00-\x7f]", "_", stem)
    return stem or "checkpoint"


def element_label(atom):
    """An atom's element label: its symbol, an isotope said as the reference has it - "C(Iso=13)"."""
    el = atom.get("el")
    if el not in NUMBER:
        raise Refused(f'"{el}" is no element: Gaussian is given atoms of elements only')
    iso = atom.get("isotope")
    if iso:
        return f"{el}(Iso={whole(iso, 'An isotope', 1)})"
    return el


def coordinate(atom, axis):
    v = atom.get(axis)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise Refused(f"an atom's {axis} is not a number")
    return float(v)


def gaussian_input(name, molecule, options):
    """The input file's text, as the reference lays it out."""
    atoms = molecule.get("atoms") or []
    if not atoms:
        raise Refused("there are no atoms to write")
    charge = whole(options.get("charge", 0), "The charge")
    multiplicity = whole(options.get("multiplicity", 1), "The spin multiplicity", 1)
    # (a multiplicity the molecule's electrons cannot have: Gaussian would stop at once)
    electrons = sum(NUMBER.get(a.get("el"), 0) for a in atoms) - charge
    if electrons < 0 or multiplicity - 1 > electrons or (electrons - (multiplicity - 1)) % 2:
        raise Refused(
            f"a charge of {str(charge).replace('-', '\u2212')} and a spin multiplicity of {multiplicity} cannot be: "
            f"the molecule would have {electrons} electrons"
        )

    lines = []
    # Link 0 commands
    if options.get("checkpoint", True):
        lines.append(f"%Chk={checkpoint_of(name)}")
    processors = whole(options.get("processors", 0), "Processors", 0)
    if processors:
        lines.append(f"%NProcShared={processors}")
    memory = one_line(options.get("memory", ""), "Memory").replace(" ", "")
    if memory:
        if not MEMORY.match(memory):
            raise Refused(f'memory must be a number with KB, MB, GB, TB, KW, MW, GW or TW after it, as 8GB - not "{memory}"')
        lines.append(f"%Mem={memory}")

    # the route section
    method = route_words(options.get("method", ""), "The method")
    basis = route_words(options.get("basis", ""), "The basis set")
    chemistry = "/".join(p for p in (method, basis) if p)
    job = JOBS.get(options.get("job", "sp"))
    if job is None:
        raise Refused(f"no job is called {options.get('job')!r}")
    dispersion = DISPERSIONS.get(options.get("dispersion", "none"))
    if dispersion is None:
        raise Refused(f"no empirical dispersion is called {options.get('dispersion')!r}")
    more = route_words(options.get("keywords", ""), "The keywords")
    lines.append(" ".join(p for p in ("#", chemistry, dispersion, job, more) if p))
    lines.append("")

    # the title section
    lines.append(title_of(options.get("title", ""), molecule.get("name") or name))
    lines.append("")

    # the molecule specification
    lines.append(f"{charge} {multiplicity}")
    for a in atoms:
        x, y, z = (coordinate(a, axis) for axis in ("x", "y", "z"))
        lines.append(f"{element_label(a):<10}{x:>16.8f}{y:>16.8f}{z:>16.8f}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --- a workflow's steps ---------------------------------------------------------

PROGRAM = "g16"
# the input file, which Gaussian reads as its standard input
INPUT = "input.gjf"
# what Gaussian's output is, as Meno's readers know it (their manifests' kinds)
KIND = "gaussian"
# the wavefunction a step's job keeps, and what kind of file it is: Meno's own kind of WFX file
WAVEFUNCTION = "wavefunction.wfx"
WAVEFUNCTION_KIND = "wfx"
# each kind of step, as its job's keyword says it
STEP_JOBS = {"optimise": "opt", "energy": "sp", "frequencies": "freq"}
# the steps whose stopped run's output is read for the geometry it reached (a run again's start; docs/WORKFLOWS.md, *Runs again*)
PARTIAL_STEPS = {"optimise"}
# what Gaussian prints at each step of an optimisation: a run that stopped before it computed nothing to start from
OPTIMISING = "Berny optimization"
SOLVATIONS = {"none", "PCM", "SMD"}
# the solvents offered, as Gaussian's SCRF keyword names them
SOLVENTS = {
    "Water", "Acetonitrile", "Acetone", "Benzene", "Chloroform", "Dichloromethane", "n,n-DiMethylFormamide",
    "DiMethylSulfoxide", "DiethylEther", "Ethanol", "EthylEthanoate", "n-Hexane", "Methanol", "TetraHydroFuran",
    "Toluene", "1,4-Dioxane", "CarbonTetraChloride", "Pyridine", "NitroMethane", "CycloHexane", "n-Octanol", "2-Propanol",
}


def step_input(step, entry, options, cores):
    """An entry's input for a step: the job its kind is, solvated as asked - SCRF=(PCM,Solvent=...) or SMD - the processors it may use from Meno's cores, no checkpoint."""
    job = STEP_JOBS.get(step)
    if job is None:
        raise Refused(f"this plugin does no {step!r}")
    # (a lone atom: its energy where it is)
    if step == "optimise" and len(entry.get("atoms") or []) == 1:
        job = STEP_JOBS["energy"]
    solvation = options.get("solvation", "none")
    if solvation not in SOLVATIONS:
        raise Refused(f"no solvation is called {solvation!r}")
    solvent = options.get("solvent", "Water")
    if solvation != "none" and solvent not in SOLVENTS:
        raise Refused(f"no solvent is called {solvent!r}")
    more = one_line(options.get("keywords", ""), "The keywords")
    scrf = f"SCRF=({solvation},Solvent={solvent})" if solvation != "none" else ""
    keep = keeps_wavefunction(options)
    given = {
        **options,
        "job": job,
        "keywords": " ".join(p for p in (scrf, more, "Output=WFX" if keep else "") if p),
        "charge": entry.get("charge", 0),
        "multiplicity": entry.get("multiplicity", 1),
        "title": entry.get("name") or "",
        "checkpoint": False,
        "processors": whole(cores, "The cores", 1) if cores else 0,
    }
    text = gaussian_input(INPUT, entry, given)
    # (the wavefunction's file, named on a line of its own after the molecule)
    return text + f"{WAVEFUNCTION}\n\n" if keep else text


def keeps_wavefunction(options):
    keep = options.get("wavefunction", True)
    if not isinstance(keep, bool):
        raise Refused("Whether to keep the wavefunction is neither on nor off")
    return keep


# MiB in each of %Mem's units: bytes by 1024s, words of 8 bytes; a bare number is words
MEMORY_MIB = {"": 8 / 1024 ** 2, "KB": 1 / 1024, "MB": 1, "GB": 1024, "TB": 1024 ** 2, "KW": 8 / 1024, "MW": 8, "GW": 8 * 1024, "TW": 8 * 1024 ** 2}


def memory_of(options):
    """The memory a job asks of Gaussian (%Mem), in MiB as the queue counts it, rounded up: none where it is not set
    (Gaussian's own default then)."""
    memory = one_line(options.get("memory", ""), "Memory").replace(" ", "")
    if not memory or not MEMORY.match(memory):
        return None
    number = re.match(r"\d+", memory).group(0)
    unit = memory[len(number):].upper()
    return max(1, math.ceil(int(number) * MEMORY_MIB[unit]))


def op_prepare(m):
    options = m.get("options") or {}
    memory = memory_of(options)
    jobs = []
    for i, entry in enumerate(m.get("entries") or []):
        jobs.append({
            "entries": [i],
            "program": PROGRAM,
            "args": [],
            "files": [{"name": INPUT, "text": step_input(m.get("step"), entry, options, m.get("cores"))}],
            "stdin": INPUT,
            "reads": [WAVEFUNCTION] if keeps_wavefunction(options) else [],
            **({"memory": memory} if memory else {}),
        })
    if not jobs:
        raise Refused("nothing came in")
    return {"jobs": jobs}


def output_name(entry):
    """What an entry's output is called: after the entry, as a file may be named."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(entry.get("name") or "").strip()).strip("._")
    return f"{stem or 'gaussian'}.log"


def why_of(output):
    """
    Why Gaussian stopped, as its output says (reader.run_of): the line before
    it says it ended in error, else that line; else - where it stopped with
    no word of an error, as a link that dies does - the last thing it printed.
    """
    _, why, _ = reader.run_of(output or "")
    if why:
        return why
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return f"Gaussian stopped without saying why, after: {lines[-1]}" if lines else "Gaussian said nothing"


def ended_well(output):
    """Whether Gaussian says it ended well, as `read` judges it (reader.run_of): its last lines say "Normal
    termination", and no "Error termination" is said. Its exit code alone is not taken for it: a run done is read only
    where its output says it is."""
    return reader.run_of(output or "")[0] == "normal"


def op_collect(m):
    if m.get("step") not in STEP_JOBS:
        raise Refused(f"this plugin does no {m.get('step')!r}")
    output = m.get("log") or ""
    if m.get("ended") != "done" or not ended_well(output):
        # (an optimisation that stopped: what it wrote read too, where it wrote anything - the geometry it reached, for a
        #  run again to start from - never as a result)
        partial = [{"kind": KIND, "log": True, "name": output_name(e)} for e in m.get("entries") or []] if m.get("step") in PARTIAL_STEPS and OPTIMISING in output else []
        return {"why": why_of(output), **({"partial": partial} if partial else {})}
    # (its wavefunction, where it wrote one: read back by Meno, kept with the molecule)
    wrote = WAVEFUNCTION in (m.get("files") or {})
    return {"read": [{"kind": KIND, "log": True, "name": output_name(e), **({"wavefunction": wavefunction_of(e)} if wrote else {})} for e in m.get("entries") or []]}


def wavefunction_of(entry):
    """The wavefunction an entry's job wrote, as Meno keeps it: its kind, its file, and the name it goes by, after the entry."""
    return {"kind": WAVEFUNCTION_KIND, "file": WAVEFUNCTION, "name": output_name(entry)[: -len(".log")] + ".wfx"}


def op_write(m):
    if m.get("kind") != "gaussian-input":
        raise Refused(f"this plugin writes no {m.get('kind')!r}")
    molecules = m.get("molecules") or []
    if len(molecules) != 1:
        raise Refused("a Gaussian input holds one molecule")
    return {"text": gaussian_input(m.get("name", ""), molecules[0], m.get("options") or {})}


def op_read(m):
    """A Gaussian output read (reader.py)."""
    name = os.path.basename(str(m.get("name") or "output"))
    text = m.get("text")
    if not isinstance(text, str) or not text:
        raise Refused("there is nothing to read")
    try:
        return reader.read_jobs(name, text)
    except ValueError as e:
        raise Refused(str(e)) from e


def op_ask(m):
    """A promise a reading made, kept: a formatted checkpoint's wavefunction, whole (`expansion`; reader.py
    `fchk_expansion`), for Meno to draw its orbitals from."""
    text = m.get("text")
    if m.get("key") != "expansion":
        raise Refused(f"it made no promise {m.get('key')!r}")
    if not isinstance(text, str) or not reader.is_fchk(text):
        raise Refused("only a formatted checkpoint holds its wavefunction")
    try:
        return {"expansion": reader.fchk_expansion(text)}
    except ValueError as e:
        raise Refused(str(e)) from e


def finite(value):
    """`value` with each number that is not one - not a number, or infinite - as not known: JSON has neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


OPS = {"write": op_write, "prepare": op_prepare, "collect": op_collect, "read": op_read, "ask": op_ask}


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
        return {"id": m.get("id"), "ok": False, "error": f"{what(m)}: {e}."}
    except Exception as e:  # noqa: BLE001 - said, not thrown: the worker keeps answering
        return {"id": m.get("id"), "ok": False, "error": f"{what(m)}: {e}"}


def what(m):
    """How a refusal begins: a file written, a file read, or a step."""
    if m.get("op") == "read":
        return "It could not be read as Gaussian's output"
    if m.get("op") == "ask":
        return "Its wavefunction could not be given"
    return "The Gaussian input could not be written" if m.get("op") == "write" else "Gaussian"


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
