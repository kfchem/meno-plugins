"""The ORCA plugin's worker: steps of a workflow done with ORCA.

A plugin's worker (docs/PLUGINS.md, docs/WORKFLOWS.md), run in an
environment of its own - Python alone, no package. ORCA itself is a
program installed separately, under its own licence: never fetched or
shipped by Meno, found where the system finds programs or where the
chemist located it (the manifest's `installed`). One JSON object per line
each way, as every plugin's:

    {"id": 4, "op": "prepare", "step": "optimise", "entries": [...], "options": {...}, "cores": 4}
    {"id": 4, "ok": true, "result": {"jobs": [{"entries": [0], "program": "orca", "args": ["input.inp"], "files": [...],
        "then": [{"program": "orca_2aim", "args": ["input"]}], "reads": ["input.wfx"]}]}}
    {"id": 5, "op": "collect", "step": "optimise", "entries": [...], "options": {...}, "files": {...}, "log": "...", "ended": "done"}
    {"id": 5, "ok": true, "result": {"read": [{"kind": "orca", "log": true, "name": "water.out",
        "wavefunction": {"kind": "wfx", "file": "input.wfx", "name": "water.wfx"}}]}}
    {"id": 6, "op": "read", "kind": "orca", "name": "water.out", "text": "..."}
    {"id": 6, "ok": true, "result": {"schema": 1, "program": "ORCA", "atoms": [...], "frames": [...], ...}}

It fills three kinds of step - Optimise, Energy, Frequencies - and runs
nothing itself: `prepare` writes each job's input and says the command
Meno runs, apart from Meno, in the job's folder; `collect` reads nothing
either, but says that what ORCA printed is ORCA's output, for Meno's
readers to read as one opened is - and, where the job kept it, that the
wavefunction is kept with the molecule, for a step that takes one
(Multiwfn's NCI) - or, where the job failed, why. One job for each entry.

The input is written as ORCA's own manual has it (ORCA 6.1 Manual,
faccts.de: "General Structure of the Input File", "Parallel and
Multi-Process Modules", "Implicit Solvation Models"), and from nothing
else:

- a simple input line, begun by "!": the method and basis set, a
  dispersion correction (D3BJ, D4), the run - SP, OPT or FREQ (NUMFREQ
  at ORCA's own xTB, which has no analytic Hessian: said on the step's
  line) - and, for implicit solvation, CPCM(solvent) or SMD(solvent). What is typed into
  it - the method, the basis set, the keywords - stays on that line, with
  no "#", which begins a comment, and no "%", which begins a block: the
  manual names a file or a program only in a block, in quotes (%MOInp
  "file", %method ProgExt "path"), so a step's text names neither - a
  workspace's steps may come from someone else, and run where it is
  opened;
- %pal nprocs, the processes a run uses, where it may use more than one;
  %maxcore, the memory for each, in MB;
- the coordinates, "* xyz charge multiplicity", an atom a line - its
  element and x, y, z in angstroms, and an isotope's mass after them,
  "M = 2.01410177812" ("Coordinate input": non-standard isotopes are
  "specified with the statements M = ... after the atomic coordinate
  definition"), the mass Meno gives with the entry (NIST's) - and "*" at
  their end.

A lone atom - an ion of a salt, say - asked to be optimised is run as SP:
it has no geometry to optimise, and ORCA, given OPT for one atom, warns
and stops with no energy. It comes back where it was, with its energy.

ORCA is run as the manual says: by its full path, given the input file,
printing its output. To keep the wavefunction, the job then runs
orca_2aim, installed beside ORCA, as the manual's "orca_2aim" has it
(9.2.9): `orca_2aim BaseName`, which reads ORCA's orbitals (BaseName.gbw)
and writes BaseName.wfx - the WFX format, for closed and open shells -
and BaseName.wfn. The job keeps the WFX.

It keeps nothing between requests, and reaches no network.
"""

import json
import math
import os
import re
import shutil
import sys

# (its reading of ORCA's output, beside it: a job's and a file's opened alike)
import reader

VERSION = "0.2.0"
PROGRAM = "orca"
INPUT = "input.inp"
# the wavefunction a job keeps: orca_2aim, beside ORCA, writing its orbitals as a WFX file
BASE = "input"
TO_WFX = {"program": "orca_2aim", "args": [BASE]}
WAVEFUNCTION = f"{BASE}.wfx"
WAVEFUNCTION_KIND = "wfx"
# what ORCA's output is, as Meno's readers know it (their manifests' kinds)
KIND = "orca"

RUNS = {"optimise": "OPT", "energy": "SP", "frequencies": "FREQ", "reaction-path": "IRC", "saddle-point": "OPTTS"}
#: A saddle point's first Hessian (its *First Hessian* option): computed at the start - the manual's 4.3.2.1, *Hessians
#: for Transition State Calculations*, "the most straightforward option" - or ORCA's own model one.
HESSIANS = {"computed": "%geom Calc_Hess true end", "model": ""}
#: The steps whose stopped run's output is read for the geometry it reached (a run again's start; docs/WORKFLOWS.md,
#: *Runs again*).
PARTIAL_STEPS = {"optimise"}
#: The heading of each step of an optimisation in ORCA's output: a run that stopped before it computed nothing to start from.
OPTIMISING = "GEOMETRY OPTIMIZATION CYCLE"
#: The words by which an optimisation that took the most cycles it allows without converging is known in the output
#: (they may run over two lines). The plugin reads such a run as not converged whatever follows - an error later in
#: other words, or a normal end: the geometry it reached is no converged one, and nothing of it is read as a result.
MAX_CYCLES = re.compile(r"did not converge but reached the maximum number of\s+optimization cycles")
MAX_CYCLES_SAID = "did not converge but reached the maximum number of optimization cycles"
#: The optimisations whose cycles ORCA counts so: an optimisation, a saddle point's.
OPTIMISATIONS = {"optimise", "saddle-point"}
# ORCA's own xTB (NATIVE-XTB2 and its kin): the plugin asks for its frequencies numerically - NUMFREQ, the numerical
# Hessian, by name - as no analytic Hessian is offered for it
NATIVE_XTB = re.compile(r"NATIVE-\S+", re.I)
NUMERICAL = "NUMFREQ"
# (said on the step's line, so that what was chosen for the chemist is never hidden)
NUMERICAL_SAID = "frequencies computed numerically"


def numerical(step, options):
    """Whether a step's frequencies are asked for numerically: a Frequencies step at ORCA's own xTB."""
    return step == "frequencies" and any(NATIVE_XTB.fullmatch(w) for w in str(options.get("method", "")).split())


#: Said on a saddle point's line where ORCA computes its first Hessian numerically: at ORCA's own xTB, which has no
#: analytic one.
NUMERICAL_HESSIAN_SAID = "its first Hessian computed numerically"


def numerical_hessian(step, options):
    """Whether a saddle point's first Hessian is computed numerically: computed, at ORCA's own xTB."""
    return step == "saddle-point" and options.get("hessian", "computed") == "computed" and any(NATIVE_XTB.fullmatch(w) for w in str(options.get("method", "")).split())
DISPERSIONS = {"none": "", "D3BJ": "D3BJ", "D4": "D4"}
# The methods ORCA has each correction's parameters for, by its own keyword names (ORCA 6.1 Manual, faccts.de,
# section 3.4 "Dispersion Corrections", 3.4.1.7 "Available Parameters": D3 with Becke-Johnson damping, table 3.16; D4,
# table 3.15). Any other has none of the program's own: the manual says the "-V" functionals (their own non-local
# dispersion) and post-HF methods take none, and gives the M06 family zero damping only for D3; the "-3c" composites and
# ORCA's own xTB are not in the tables.
D3BJ_PARAMETERS = {name.upper() for name in (
    "HF", "BP86", "BLYP", "REVPBE", "B97-D", "PBE", "RPBE", "RPW86PBE", "B3LYP", "BHANDHLYP", "TPSS", "TPSS0", "PBE0",
    "REVPBE38", "PW6B95", "B2PLYP", "mPWLYP", "OLYP", "BPBE", "OPBE", "B3PW91", "REVPBE0", "TPSSh", "CAM-B3LYP",
    "B2GP-PLYP", "PWPB95", "SCAN", "RSCAN", "R2SCAN", "R2SCANh", "R2SCAN0", "R2SCAN50", "WR2SCAN", "r2SCAN0-DH",
    "r2SCAN-CIDH", "r2SCAN-QIDH", "r2SCAN0-2", "Pr2SCAN50", "kPr2SCAN50", "wPr2SCAN50", "Pr2SCAN69", "revDSD-PBEP86/2021",
    "revDOD-PBEP86/2021", "DSD-BLYP", "DSD-BLYP/2013", "DSD-PBEB95", "DSD-PBEP86", "DSD-PBEP86/2013", "B97M-D3BJ",
    "wB97X-D3BJ", "wB97M-D3BJ", "wB97X-2", "PBE0DH", "PBE02", "PBE-QIDH",
)}
D4_PARAMETERS = {name.upper() for name in (
    "HF", "BLYP", "BPBE", "BP86", "BPW91", "GLYP", "LB94", "mPWLYP", "mPWPW", "OLYP", "OPBE", "PBE", "RPBE", "REVPBE",
    "PW86PBE", "RPW86PBE", "PW91", "PW91P86", "XLYP", "B97BECKE", "TPSS", "revTPSS", "SCAN", "B1LYP", "B3LYP", "BHANDHLYP",
    "B1P", "B3P86", "B1PW91", "B3PW91", "O3LYP", "REVPBE0", "REVPBE38", "PBE0", "PWP1", "PW1PW", "mPW1PW", "mPW1LYP",
    "PW6B95", "TPSSh", "TPSS0", "X3LYP", "M06L", "M06", "wB97", "wB97X", "B97M-D4", "CAM-B3LYP", "LC-BLYP", "B2PLYP",
    "B2GP-PLYP", "mPW2PLYP", "PWPB95", "B97-D", "RSCAN", "r2SCAN", "r2SCANh", "r2SCAN0", "r2SCAN50", "wB97X-D4rev",
    "wB97M-D4rev", "wR2SCAN", "r2SCAN0-DH", "r2SCAN-CIDH", "r2SCAN-QIDH", "r2SCAN0-2", "Pr2SCAN50", "kPr2SCAN50",
    "wPr2SCAN50", "Pr2SCAN69", "revDSD-PBEP86/2021", "revDOD-PBEP86/2021", "LRC-PBE",
)}
PARAMETERS = {"D3BJ": D3BJ_PARAMETERS, "D4": D4_PARAMETERS}
# The keywords that carry a correction by their own name (the manual's section 3.3, its keyword table): B97-D uses D2,
# the "-D3BJ" and "-D4"/"-D4rev" ones their own - given none besides by the method, so that none is asked twice.
OWN_CORRECTION = {name.upper() for name in ("B97-D", "wB97X-D3BJ", "wB97M-D3BJ", "B97M-D3BJ", "B97M-D4", "wB97X-D4rev", "wB97M-D4rev")}
# Given none by the method, though in the table: those above, and HF - a Hartree-Fock run is wanted plain, D3BJ on it
# only when chosen by name (ruled 2026-11-14, as the Gaussian plugin has it).
NONE_BY_METHOD = OWN_CORRECTION | {"HF"}
# The dispersion that follows the method (the card's default on Optimise and Frequencies): D3BJ where ORCA has its
# parameters for the method, none where it has not.
BY_METHOD = "by-method"


# A word that names a correlated wave-function method (post-HF: the manual's MP2, CCSD(T), CASSCF, NEVPT2 and their kin,
# with their prefixes): such a method takes no correction, whatever else the text names - "DLPNO-CCSD(T) HF" is no HF run.
POST_HF = re.compile(r"((DLPNO|LPNO|RI|SCS|SOS|OO|F12)-)*(MP[2-5]|CCSD|CC2|CC3|QCISD|CISD|CEPA|CPF|ACPF|CASSCF|CASPT2|NEVPT2|MRCI|MRCC)\b.*", re.I)


def method_of(method):
    """A method's text as the manual's tables name it: of its words (a simple input line's - "B3LYP RIJCOSX"), the first
    the tables name, upper case, a leading DLPNO- (the manual's prefix for a natively implemented double hybrid,
    DLPNO-B2PLYP its own example: section 3.3) taken off where what is left is named in them; else the text as it is,
    which no table names - a "-V" functional; and wherever a word names a post-HF method (`POST_HF`), so that the text
    is read as that method, not as a table's word beside it."""
    known = D3BJ_PARAMETERS | D4_PARAMETERS
    if any(POST_HF.fullmatch(word) for word in method.split()):
        return method.strip().upper()
    for word in method.upper().split():
        if word in known:
            return word
        if word.startswith("DLPNO-") and word[len("DLPNO-"):] in known:
            return word[len("DLPNO-"):]
    return method.strip().upper()


def dispersion_keyword(choice, method):
    """The simple input's dispersion for the card's choice and the method typed: by the method, D3BJ where ORCA has its
    parameters for it - and it is not HF, nor a keyword that carries its own - and none where not; D3BJ or D4 chosen by name,
    refused in words where ORCA has none of its own for the method."""
    name = method_of(method)
    if choice == BY_METHOD:
        return DISPERSIONS["D3BJ"] if name in D3BJ_PARAMETERS and name not in NONE_BY_METHOD else ""
    keyword = DISPERSIONS.get(choice)
    if keyword is None:
        raise Refused(f"no dispersion correction is called {choice!r}")
    if keyword and name not in PARAMETERS[choice]:
        raise Refused(f"ORCA has no {choice} parameters for {method}: choose None, or a method it has them for")
    return keyword
SOLVATIONS = {"none", "CPCM", "SMD"}
# the solvents offered, as ORCA names them - one word each, for a simple input line
SOLVENTS = {
    "water", "acetonitrile", "acetone", "benzene", "chloroform", "dichloromethane", "dimethylformamide",
    "dimethylsulfoxide", "diethylether", "ethanol", "ethylacetate", "hexane", "methanol", "tetrahydrofuran",
    "toluene", "1,4-dioxane", "ccl4", "pyridine", "nitromethane", "cyclohexane", "1-octanol", "2-propanol",
}
# a word of a simple input line: ASCII, nothing that starts a comment, a block or the coordinates
WORDS = re.compile(r"^[\x21-\x7e]+( [\x21-\x7e]+)*$")


class Refused(Exception):
    """What was asked cannot be done: said, not thrown."""


def words(value, what):
    """A text option as words of a simple input line: one line, no comment in it."""
    if not isinstance(value, str):
        raise Refused(f"{what} is not text")
    text = " ".join(value.split())
    if text and (not WORDS.match(text) or "#" in text or "%" in text):
        raise Refused(f"{what} may hold no #, no % and nothing but plain letters, digits and signs")
    return text


def whole(v, what, least=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != int(v):
        raise Refused(f"{what} is not a whole number")
    if least is not None and v < least:
        raise Refused(f"{what} must be at least {least}")
    return int(v)


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


def mass_of(el, atom):
    """An isotope's mass, as the manual has it written after the atom's
    coordinates ("M = ..."), from what Meno gives with the entry; nothing for
    an atom of no isotope. One Meno gives no mass for - none measured - is
    refused, naming it, rather than given as the element's natural mass."""
    iso = atom.get("isotope")
    if not iso:
        return ""
    mass = atom.get("mass")
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
        raise Refused(f"Meno knows no measured mass for {iso}{el}: ORCA would be given {el} of natural mass")
    return f"  M = {mass:.15g}"


#: How a frequency job's thermochemistry treats the low-frequency vibrations (its *Low-frequency vibrations* option):
#: ORCA's own, nothing asked ("none") - its default, read back from its output's "Quasi RRHO" line - or the harmonic
#: oscillator alone, the rigid-rotor harmonic oscillator a program with no low-mode treatment
#: gives: ORCA's %freq QuasiRRHO false. ("none" is the value a step's line leaves unsaid: its default.)
LOW_VIBRATIONS = {"none": "", "rrho": "%freq QuasiRRHO false end"}


#: The restraints an optimisation holds as ORCA constrains them (%geom Constraints; ORCA 6.1 manual, 4.1, *Running
#: Constrained Optimizations*): each exactly - a distance at its value, in angstroms ({B a b value C}), a dihedral at its
#: value, in degrees ({D a b c d value C}), atoms fixed where they stand ({C a C}) - atoms from 0, as ORCA numbers them.
#: The manual's constraints take no strength: a distance or a dihedral restrained (the entry record's `strength`) is
#: refused, never held exactly in its place. Of an optimisation to a saddle point, atoms fixed alone, as before: the manual gives its
#: constraints for an optimisation, and does not say that a distance or a dihedral is held under OptTS.
EXACT = {"optimise": ("distance", "dihedral"), "saddle-point": ()}
#: How many atoms each kind of restraint holds, and ORCA's letter for it.
SHAPES = {"distance": (2, "B"), "dihedral": (4, "D")}


def turned(degrees):
    """A dihedral's value as Meno writes one: in (-180, 180]."""
    d = math.fmod(degrees, 360.0)
    if d <= -180.0:
        d += 360.0
    elif d > 180.0:
        d -= 360.0
    return d


def held(step, options, entry):
    """What an optimisation - to a minimum, or to a saddle point - holds as ORCA constrains it, each in ORCA's words
    (`EXACT`): none unless Meno asks it to (`hold`: "fixed") and the entry has restraints; a distance or dihedral held
    exactly first, as they came, then the atoms fixed. A restraint it does not hold - one by a strength, of a kind it
    has not, or not as the entry record writes one - refused, saying which: never dropped unsaid, never held another
    way."""
    if step not in EXACT or options.get("hold", "no") != "fixed":
        return []
    n = len(entry.get("atoms") or [])
    exact, fixed = [], set()
    for r in entry.get("restraints") or []:
        kind, atoms = r.get("kind"), r.get("atoms") or []
        if any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < n for a in atoms):
            raise Refused("an atom its restraints hold is not one of its atoms")
        if kind == "fixed":
            if not atoms:
                raise Refused("a row of atoms fixed names none")
            fixed.update(atoms)
            continue
        if kind not in SHAPES:
            raise Refused(f"its restraints hold {kind!r}, which this plugin holds none of")
        what = "torsion" if kind == "dihedral" else kind
        said = f"{what} {'-'.join(str(a + 1) for a in atoms)}"
        if not r.get("exact") and r.get("strength") is not None:
            raise Refused(f"this plugin holds a {what} only exactly, not by a strength: {said} has a strength")
        if kind not in EXACT[step]:
            raise Refused(f"this plugin holds no {what} in an optimisation to a saddle point, only atoms fixed: {said}")
        count, letter = SHAPES[kind]
        value = r.get("value")
        if not r.get("exact") or r.get("strength") is not None or len(atoms) != count or len(set(atoms)) != count or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (kind == "distance" and not value > 0):
            raise Refused(f"its {said} is not a {what} held exactly at a value")
        value = turned(float(value)) if kind == "dihedral" else float(value)
        exact.append(f"{{{letter} {' '.join(str(a) for a in atoms)} {value:.6f} C}}")
    return exact + [f"{{C {a} C}}" for a in sorted(fixed)]


def orca_input(step, entry, options, cores):
    """An entry's input, as ORCA's manual lays it out."""
    run = RUNS.get(step)
    if run is None:
        raise Refused(f"this plugin does no {step!r}")
    # (ORCA's OptTS takes a guess alone: no sides beside it, as a three-structure search would)
    if entry.get("sides"):
        raise Refused("ORCA's OptTS takes no sides: wire neither, or have a program that does fill the step")
    hessian = HESSIANS.get(options.get("hessian", "computed")) if step == "saddle-point" else ""
    if hessian is None:
        raise Refused(f"no first Hessian is called {options.get('hessian')!r}")
    # (a lone atom: its energy where it is)
    if step == "optimise" and len(elements_of(entry)) == 1:
        run = RUNS["energy"]
    method = words(options.get("method", ""), "The method")
    if not method:
        raise Refused("there is no method")
    if numerical(step, options):
        run = NUMERICAL
    basis = words(options.get("basis", ""), "The basis set")
    dispersion = dispersion_keyword(options.get("dispersion", "none"), method)
    solvation = options.get("solvation", "none")
    if solvation not in SOLVATIONS:
        raise Refused(f"no solvation is called {solvation!r}")
    solvent = options.get("solvent", "water")
    if solvation != "none" and solvent not in SOLVENTS:
        raise Refused(f"no solvent is called {solvent!r}")
    more = words(options.get("keywords", ""), "The keywords")
    charge = whole(entry.get("charge", 0), "The charge")
    multiplicity = whole(entry.get("multiplicity", 1), "The multiplicity", 1)
    maxcore = whole(options.get("maxcore", 0), "The memory for each core", 0)
    vibrations = LOW_VIBRATIONS.get(options.get("vibrations", "none")) if step == "frequencies" else ""
    if vibrations is None:
        raise Refused(f"no treatment of low vibrations is called {options.get('vibrations')!r}")

    keywords = [method, basis, dispersion, run, f"{solvation}({solvent})" if solvation != "none" else "", more]
    lines = ["! " + " ".join(k for k in keywords if k)]
    if cores and whole(cores, "The cores", 1) > 1:
        lines.append(f"%pal nprocs {int(cores)} end")
    if maxcore:
        lines.append(f"%maxcore {maxcore}")
    if vibrations:
        lines.append(vibrations)
    # (a reaction path's steps each way, where said: ORCA's own else)
    steps = whole(options.get("maxiter", 0), "The steps each way", 0) if step == "reaction-path" else 0
    if steps:
        lines.append(f"%irc maxiter {steps} end")
    # (what it holds, where Meno asks it to: constraints in the %geom block (`EXACT`) - one block, beside a saddle point's
    #  first Hessian)
    constrained = held(step, options, entry)
    if constrained:
        constraints = "Constraints " + " ".join(constrained) + " end"
        lines.append(f"%geom {'Calc_Hess true ' if hessian else ''}{constraints} end")
    elif hessian:
        lines.append(hessian)
    lines.append(f"* xyz {charge} {multiplicity}")
    for el, a in zip(elements_of(entry), entry["atoms"]):
        x, y, z = (float(a.get(k, 0.0)) for k in ("x", "y", "z"))
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise Refused("an atom has no place")
        lines.append(f"{el:<3}{x:>18.10f}{y:>18.10f}{z:>18.10f}" + mass_of(el, a))
    lines.append("*")
    return "\n".join(lines) + "\n"


def keeps_wavefunction(options):
    keep = options.get("wavefunction", True)
    if not isinstance(keep, bool):
        raise Refused("Whether to keep the wavefunction is neither on nor off")
    return keep


def memory_of(options, cores):
    """The memory a job asks of ORCA, in MiB, as the queue counts it: %maxcore, the MB each process may use, times
    the processes - none where %maxcore is not set (ORCA's own default then, which it does not say)."""
    maxcore = whole(options.get("maxcore", 0), "The memory for each core", 0)
    processes = whole(cores, "The cores", 1) if cores else 1
    return maxcore * processes if maxcore else None


#: ORCA's keywords for the xTB methods it runs through the xtb program, not its own: the ORCA 6.1 manual, 3.5.2
#: Extended Tight-Binding: GFN0-xTB, GFN-xTB, GFN2-xTB, its simple input keywords
#: (https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/semiempirical.html#extended-tight-binding-gfn0-xtb-gfn-xtb-gfn2-xtb).
#: Each needs the xtb program, named in XTBEXE (the manual's 1.2.4 How do I install the xtb module?). Its native ones (3.5.3 Native GFN-xTB and
#: GFN2-xTB: NATIVE-XTB2, NATIVE-GFN2-XTB and the rest) run in ORCA itself, and need none.
XTB_KEYWORDS = {"XTB", "XTB0", "GFN0-XTB", "XTB1", "GFN1-XTB", "XTB2", "GFN2-XTB", "XTBFF", "GFN-FF"}


def needs_xtb(options):
    """Whether a job runs an xTB method through the xtb program: one of its keywords in the method or the keywords."""
    said = f"{options.get('method', '')} {options.get('keywords', '')}"
    return any(word.upper() in XTB_KEYWORDS for word in said.split())


#: A path search (ORCA's NEB-TS: the ORCA 6.1 manual, Nudged Elastic Band Method,
#: https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/neb.html): the path between an entry and the
#: end it is paired with, its highest point refined, and - where asked - that point's frequencies. Its files, for an
#: input named input.inp: the path as images, the point refined.
PATH_RUN = "NEB-TS"
#: The runs a path search offers (its *Path* option): a band, its highest point refined; or a band, a finer one about
#: its highest point, that refined (the manual's 4.5.12 *ZOOM-NEB* and 4.5.13 *NEB-TS*). Both write the same files and
#: end in one PATH SUMMARY FOR NEB-TS, read for both.
PATH_RUNS = ("NEB-TS", "ZOOM-NEB-TS")
END = "end.xyz"


def xyz_block(entry):
    """An entry's atoms as an XYZ file's lines: its elements and places."""
    lines = []
    for el, a in zip(elements_of(entry), entry["atoms"]):
        x, y, z = (float(a.get(k, 0.0)) for k in ("x", "y", "z"))
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise Refused("an atom has no place")
        lines.append(f"{el:<3}{x:>18.10f}{y:>18.10f}{z:>18.10f}")
    return lines


#: The tight-binding methods among ORCA's simple keywords, each by the name the xtb program gives it: those it runs through
#: the xtb program (`XTB_KEYWORDS`; the manual's 3.5.2, its table of simple input keywords), and its own (3.5.3) - one
#: method whichever runs it. A path search says its level by it, as its ends' are said (Meno compares them).
XTB_NAMES = {
    "XTB0": "GFN0-xTB", "GFN0-XTB": "GFN0-xTB",
    "XTB1": "GFN1-xTB", "GFN1-XTB": "GFN1-xTB", "GFN-XTB": "GFN1-xTB",
    "XTB": "GFN2-xTB", "XTB2": "GFN2-xTB", "GFN2-XTB": "GFN2-xTB",
    "XTBFF": "GFN-FF", "GFN-FF": "GFN-FF",
    "NATIVE-GFN-XTB": "GFN1-xTB", "NATIVE-GFN1-XTB": "GFN1-xTB", "NATIVE-XTB1": "GFN1-xTB",
    "NATIVE-XTB": "GFN2-xTB", "NATIVE-GFN2-XTB": "GFN2-xTB", "NATIVE-XTB2": "GFN2-xTB",
}


#: Words of a dispersion correction or a solvation among ORCA's simple keywords (the manual's table of simple input
#: keywords), each with or without its "(solvent)": a path search takes neither for now. Its ends' level is said by their
#: method, basis set and solvation alone, not their dispersion, so a path given one could be at a level its ends were not.
PATH_REFUSED_WORDS = {
    "D2", "D3", "D3BJ", "D3ZERO", "D3TZ", "D4", "NL", "SCNL", "ABC",
    "CPCM", "CPCMC", "CPCMX", "SMD", "ALPB", "GBSA", "DDCOSMO", "COSMO", "COSMORS", "DRACO",
}


def path_refusal(method, more):
    """Why a path search cannot be run as asked - or None. For now it runs at a tight-binding level alone (`XTB_NAMES`),
    with no dispersion correction or solvation added: a path's level must be its ends', and of their level Meno is told
    their method, basis set and solvation, not their dispersion."""
    if method.upper() not in XTB_NAMES:
        return f"a path search runs at a tight-binding level only for now, such as NATIVE-XTB2 or XTB2: {method} is not one"
    added = [w for w in f"{method} {more}".split() if w.split("(")[0].upper() in PATH_REFUSED_WORDS]
    if added:
        return f"a path search takes no dispersion correction or solvation for now, as its ends' level cannot say them: {', '.join(added)}"
    return None


def path_level(options):
    """A path search's method and basis, as its output says them: a tight-binding method by its name (`XTB_NAMES`), with
    no basis; else the method and basis as given."""
    method = " ".join(str(options.get("method", "")).split())
    basis = " ".join(str(options.get("basis", "")).split())
    named = XTB_NAMES.get(method.upper())
    return (named, None) if named else (method or None, basis or None)


def path_input(entry, options, cores):
    """A path search's input: from the entry to the end it is paired with (`end`), ORCA's NEB-TS alone - the path, and
    its highest point refined - with the method, basis set and keywords given. Its point's frequencies are a
    Frequencies step's, after it (one role to a step)."""
    end = entry.get("end")
    if not isinstance(end, dict):
        raise Refused("a path search needs an end for each entry")
    if elements_of(end) != elements_of(entry):
        raise Refused("the start and the end are not the same atoms in one numbering")
    method = words(options.get("method", ""), "The method")
    if not method:
        raise Refused("there is no method")
    basis = words(options.get("basis", ""), "The basis set")
    more = words(options.get("keywords", ""), "The keywords")
    why = path_refusal(method, more)
    if why:
        raise Refused(why)
    run = options.get("run", PATH_RUN)
    if run not in PATH_RUNS:
        raise Refused(f"no path is called {run!r}")
    # (the images between its ends - the manual's 4.5.16, NImages, 8 unless said - and the steps refining its highest
    #  point, the %geom block's (4.5.13), ORCA's own where none is said)
    images = whole(options.get("images", 8), "The images", 1)
    maxiter = whole(options.get("maxiter", 0), "The steps for refining its highest point", 0)
    charge = whole(entry.get("charge", 0), "The charge")
    multiplicity = whole(entry.get("multiplicity", 1), "The multiplicity", 1)
    maxcore = whole(options.get("maxcore", 0), "The memory for each core", 0)
    lines = ["! " + " ".join(k for k in [method, basis, run, more] if k)]
    if cores and whole(cores, "The cores", 1) > 1:
        lines.append(f"%pal nprocs {int(cores)} end")
    if maxcore:
        lines.append(f"%maxcore {maxcore}")
    if maxiter:
        lines.append(f"%geom maxiter {maxiter} end")
    lines += ["%neb", f'  neb_end_xyzfile "{END}"', f"  nimages {images}", "end", f"* xyz {charge} {multiplicity}", *xyz_block(entry), "*"]
    end_lines = xyz_block(end)
    return "\n".join(lines) + "\n", f"{len(end_lines)}\nend\n" + "\n".join(end_lines) + "\n"


#: A reaction path (ORCA's IRC: the ORCA 6.1 manual, 4.4 *Intrinsic Reaction Coordinate*), followed both ways from the
#: point given - the manual's default direction - in one job. Its file, for an input named input.inp: the whole path, one end to the other, each frame with its energy; its IRC PATH SUMMARY marks the point it
#: started from ("<= TS").
#: The program ORCA starts its processes with, where it runs more than one (the ORCA 6.1 manual, 2.5.2.1): Open MPI's
#: mpirun on macOS and Linux, Microsoft MPI's mpiexec on Windows - looked for where ORCA's jobs look for it, on the PATH.
LAUNCHER = "mpiexec" if sys.platform == "win32" else "mpirun"
#: Said once on the step's line where it is not there: each job then runs on one core, whatever the cores each job is
#: given - they apply to the other programs, which need no launcher - never refused, never failed at ORCA's start-up.
NO_LAUNCHER_SAID = "run on one core: no MPI on this computer"


#: How a program is looked for on a PATH: the standard library's (its tests give their own).
which = shutil.which


def job_path(programs):
    """The PATH ORCA's job is given (jobs.rs `installed_env`): ORCA's own folder, where it was found, before the PATH Meno
    was started with - this worker's, less its own environment's folders, which no job is given."""
    own = os.path.realpath(sys.prefix)
    rest = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and not os.path.realpath(p).startswith(own + os.sep)]
    found = (programs or {}).get(PROGRAM)
    return os.pathsep.join(([os.path.dirname(found)] if isinstance(found, str) and found else []) + rest)


def launcher_found(look=None, programs=None):
    """Whether the program ORCA starts its processes with is there (`LAUNCHER`), on the PATH its jobs are given."""
    return (look or which)(LAUNCHER, path=job_path(programs)) is not None


def op_prepare(m):
    step = m.get("step")
    options = m.get("options") or {}
    # (more than one core, and no MPI to run ORCA's processes with: one core each job, said - never a job ORCA ends at its
    #  start-up, as every one would, "mpirun: command not found")
    given = whole(m.get("cores"), "The cores", 1) if m.get("cores") else 1
    alone = given > 1 and not launcher_found(programs=m.get("programs"))
    if alone:
        m = {**m, "cores": 1}
    # (a reaction path's step offers no wavefunction kept: none written)
    # (a saddle point's step offers no wavefunction kept: none written)
    keep = keeps_wavefunction(options) and step != "reaction-path" and step != "saddle-point"
    memory = memory_of(options, m.get("cores"))
    jobs = []
    if step == "path-search":
        for i, entry in enumerate(m.get("entries") or []):
            text, end = path_input(entry, options, m.get("cores"))
            jobs.append({
                "entries": [i],
                "program": PROGRAM,
                "args": [INPUT],
                "files": [{"name": INPUT, "text": text}, {"name": END, "text": end}],
                "reads": [reader.MEP, reader.REFINED],
                **({"memory": memory} if memory else {}),
                **({"needs": ["xtb"]} if needs_xtb(options) else {}),
            })
        if not jobs:
            raise Refused("nothing came in")
        return {"jobs": [{**j, "cores": 1} for j in jobs] if alone else jobs, **({"note": NO_LAUNCHER_SAID} if alone else {})}
    # (an entry whose restraints ask what it does not hold refused alone, saying why - the others run; every one, and the
    #  step is refused as a whole)
    refused = []
    first = None
    for i, entry in enumerate(m.get("entries") or []):
        try:
            held(step, options, entry)
        except Refused as e:
            refused.append({"entry": i, "why": f"ORCA: {e}.", "kind": "refused"})
            first = first or e
            continue
        jobs.append({
            "entries": [i],
            "program": PROGRAM,
            "args": [INPUT],
            "files": [{"name": INPUT, "text": orca_input(step, entry, options, m.get("cores"))}],
            **({"then": [TO_WFX]} if keep else {}),
            "reads": [reader.IRC_FULL] if step == "reaction-path" else [WAVEFUNCTION] if keep else [],
            **({"memory": memory} if memory else {}),
            # (the xTB plugin's xtb, where an xTB method runs through it: Meno gives it as manifest.json's `needs` says)
            **({"needs": ["xtb"]} if needs_xtb(options) else {}),
        })
    if not jobs:
        raise first or Refused("nothing came in")
    said = NUMERICAL_SAID if numerical(step, options) else NUMERICAL_HESSIAN_SAID if numerical_hessian(step, options) else None
    note = "; ".join(n for n in (said, NO_LAUNCHER_SAID if alone else None) if n)
    return {"jobs": [{**j, "cores": 1} for j in jobs] if alone else jobs, **({"note": note} if note else {}), **({"refused": refused} if refused else {})}


def output_name(entry):
    """What an entry's output is called: after the entry, as a file may be named."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(entry.get("name") or "").strip()).strip("._")
    return f"{stem or 'orca'}.out"


def ended_well(log):
    """Whether ORCA says it ended well, as its output is read (reader `run_of`): "ORCA TERMINATED NORMALLY", and nowhere
    "error termination". Its exit code alone is not taken for it: a run that stopped at its start-up (a parallel run,
    its MPI not there) may still exit 0 - read as done, it would hand on as worked out what it never touched."""
    return reader.run_of(log)[0] == "normal"


#: How a parallel run whose mpirun is not there is known: the shell's words on mpirun in its output.
NO_MPI = re.compile(r"mpirun: (?:command )?not found", re.I)
#: That said plainly, and what to do: ORCA runs on more than one core with Open MPI on Linux and Mac alone, its mpirun on
#: the PATH (the ORCA 6.1 manual, 2.5.2.1, *Single Node Runs*).
NO_MPI_SAID = ("ORCA runs on more than one core only with Open MPI, whose mpirun this computer does not have "
               "(the ORCA manual, 2.5.2.1): set Cores for each job to 1 in Settings, or install Open MPI")


def not_ended_why(log, ended):
    """Why a job is not read: ORCA's own words where it said it ended by an error, or where its runner saw it fail; where
    it ended (exit 0) saying neither that nor that it terminated normally, that plainly, with the last thing it said. A
    parallel run that could not start for want of Open MPI is said plainly, with what to do."""
    if re.search(r"error termination in Startup", log, re.I) and NO_MPI.search(log):
        return NO_MPI_SAID
    if ended != "done" or re.search(r"error termination", log, re.I):
        return reader.why_of(log)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    return "ORCA's log does not say it ended normally" + (f"; it last said: {lines[-1]}" if lines else "")


#: The marker words by which a band that did not converge in the iterations it allows is known in ORCA's output.
NEB_NOT_CONVERGED = "NEB not converged"


def optimisation_not_converged(log):
    """Why an optimisation is not read, where ORCA took the most cycles it allows: plainly, with how many - and, where
    ORCA then ended by an error (in the properties it works out after, say), that too, in its words: the run is not
    converged, and it is said that it ended in error, not left to be taken for a run that ended well."""
    n = reader.cycles_of(log)
    went = f" in {n} cycles" if n is not None else ""
    return (f"The optimisation did not converge{went}, the most it was allowed (\"{MAX_CYCLES_SAID}\"): "
            "run it again to go on from the geometry it reached") + then_erred(log)


def then_erred(log):
    """Where ORCA, having taken the most cycles it allows, then ended by an error (in the properties it works out
    after, say): that too, in its words - never left to be taken for a run that ended well. Nothing where it did not."""
    return f"; ORCA then ended with an error: {reader.why_of(log)}" if re.search(r"error termination", log, re.I) else ""


def path_refining_not_converged(log):
    """Why a path search is not read, where its band converged but refining its highest point took the most steps ORCA
    allows: plainly, with how far each went, and what to change - and how ORCA ended after, where it ended by an error."""
    parts = dict(reader.path_iterations(log))
    neb, ts = parts.get("NEB"), parts.get("TS optimization")
    band = f"Its band converged in {neb} iterations" if neb is not None else "Its band converged"
    went = f" in {ts} steps" if ts is not None else ""
    return (f"{band}, but refining its highest point did not converge{went} (\"{MAX_CYCLES_SAID}\"): "
            "raise Steps for refining its highest point, or give ends nearer the path") + then_erred(log)


def path_not_converged(log):
    """Why a path search's band is not read, where ORCA says it did not converge: plainly, with how far it went."""
    parts = reader.path_iterations(log)
    neb = next((n for name, n in parts if name == "NEB"), None)
    went = f" in {neb} iterations" if neb is not None else ""
    return f"Its band did not converge{went}: ORCA stopped before refining a highest point (\"{NEB_NOT_CONVERGED}\")"


def op_collect(m):
    step = m.get("step")
    if step == "path-search":
        log = m.get("log") or ""
        # (a band that did not converge: said plainly - ORCA ends it by an error return, whose words say nothing of it)
        if NEB_NOT_CONVERGED in log:
            return {"why": path_not_converged(log), "kind": "not-converged"}
        # (its band converged, refining its highest point not, in the steps ORCA allows: said plainly - however ORCA ended
        #  after, by an error in its properties or normally - and nothing of it read: no point refined, no entry)
        if MAX_CYCLES.search(log):
            return {"why": path_refining_not_converged(log), "kind": "not-converged"}
        # (read only where ORCA says it ended well, as every step's job is: its exit code is no word on it)
        if m.get("ended") != "done" or not ended_well(log):
            return {"why": not_ended_why(log, m.get("ended"))}
        method, basis = path_level(m.get("options") or {})
        return {"outputs": [reader.path_output(elements_of(e), m.get("files") or {}, log, method, basis) for e in m.get("entries") or []]}
    if step not in RUNS:
        raise Refused(f"this plugin does no {step!r}")
    log = m.get("log") or ""
    # (an optimisation - a saddle point's too - that took the most cycles ORCA allows without converging: said plainly,
    #  however ORCA ended after - never read as a result; the geometry it reached read for a run again to start from, out
    #  of Failed, as one that stopped is)
    if step in OPTIMISATIONS and MAX_CYCLES.search(log):
        partial = [{"kind": KIND, "log": True, "name": output_name(e)} for e in m.get("entries") or []]
        return {"why": optimisation_not_converged(log), "kind": "not-converged", "partial": partial}
    if m.get("ended") != "done" or not ended_well(log):
        # (an optimisation that stopped: what it wrote read too, where it wrote anything - the geometry it reached, for a
        #  run again to start from - never as a result)
        partial = [{"kind": KIND, "log": True, "name": output_name(e)} for e in m.get("entries") or []] if step in PARTIAL_STEPS and OPTIMISING in log else []
        return {"why": not_ended_why(log, m.get("ended")), **({"partial": partial} if partial else {})}
    if step == "reaction-path":
        method, basis = path_level(m.get("options") or {})
        return {"outputs": [reader.irc_output(elements_of(e), m.get("files") or {}, log, method, basis) for e in m.get("entries") or []]}
    # (its wavefunction, where orca_2aim wrote one: read back by Meno, kept with the molecule)
    wrote = WAVEFUNCTION in (m.get("files") or {})
    return {"read": [{"kind": KIND, "log": True, "name": output_name(e), **({"wavefunction": wavefunction_of(e)} if wrote else {})} for e in m.get("entries") or []]}


def wavefunction_of(entry):
    """The wavefunction an entry's job wrote, as Meno keeps it: its kind, its file, and the name it goes by, after the entry."""
    return {"kind": WAVEFUNCTION_KIND, "file": WAVEFUNCTION, "name": output_name(entry)[: -len(".out")] + ".wfx"}


def op_read(m):
    """An ORCA output, read into Meno's form (reader `read_jobs`): a job's own, which `collect` hands Meno to read, or a
    file opened by hand - one reading for both; every number one, a value not printed not given."""
    name = os.path.basename(str(m.get("name") or "output"))
    return reader.finite(reader.read_jobs(name, m.get("text")))


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
        return {"id": m.get("id"), "ok": True, "result": op(m)}
    except Refused as e:
        return {"id": m.get("id"), "ok": False, "error": f"ORCA: {e}."}
    except Exception as e:  # noqa: BLE001 - said, not thrown: the worker keeps answering
        return {"id": m.get("id"), "ok": False, "error": f"ORCA: {e}"}


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
