"""xtb's output, read: what the xtb program printed of one run, in Meno's
general form (src/lib/calc/output.ts, results.ts) - the one reading of it,
for an output opened by hand and for a job's alike. The worker's `read`
answers with it, and its `collect` reads a job's log with it and adds only
what xtb wrote to files of its own: an optimisation's path, and the
geometry a run began with, which xtb does not print.

Read from xtb's own documentation (xtb-docs.readthedocs.io: "Commandline
Usage", "Geometry Optimization", "Calculation of Vibrational Frequencies",
"Implicit Solvation", "Properties") and from what xtb 6.7.1 prints, and
from nothing else. Each pattern below quotes the printed form it reads,
its numbers made up. Python's standard library alone, and no file read: the
text is given.

What xtb prints, and where:
- its version in its banner; the method in its setup ("Hamiltonian
  GFN2-xTB"), or GFN-FF's own heading; the charge and the unpaired
  electrons in its setup, or - GFN-FF, which prints no such setup - the
  total charge of its summary;
- the atoms in the table of elements it prints as it begins, by their
  places ("1, 2", "5-7"); no coordinates, but for an optimisation's final
  structure, which it prints as it was given them (Xmol, or a Turbomole
  `$coord` in bohr);
- the total energy in the closing summary, the HOMO-LUMO gap in its
  SUMMARY, the orbitals it prints about the frontier, each atom's charges
  (GFN2-xTB's atomic partial charges, the q of its property printout;
  GFN1-xTB's Mulliken and CM5 charges) and the dipole moment (GFN2-xTB's "full" row,
  GFN1-xTB's "dipole moment from electron density", GFN-FF's from its
  charges) - each of the geometry it ended on, the last it printed;
- the dipole is taken about the origin of the coordinates xtb was given -
  so its real outputs have it: the q-only row is the charges' moment about
  that origin, of an optimised ion's final structure too - and is given
  with it (`origin`): where Meno takes it about is Meno's;
- the frequencies, reduced masses and IR intensities of its Frequency
  Printout; the thermochemistry of its Thermodynamic Functions and its
  THERMODYNAMIC summary, and how it was worked out;
- the solvation as its setup says it; an optimisation's outcome as it says
  it ("GEOMETRY OPTIMIZATION CONVERGED", "FAILED TO CONVERGE"); the
  imaginary frequencies it judged significant; its warnings; and how the
  run ended - "normal termination of xtb" and "abnormal termination of
  xtb" it writes to its other stream, so that an output of its own stream
  alone says how it ended by its own last lines ("* finished run on", or
  "[ERROR] Program stopped due to fatal error" and its numbered messages).
"""

import math
import re

SCHEMA = 1
PROGRAM = "xtb"

# a number as xtb prints one: "-5.012345678901", "0.1000000E-05", "0.159E+27"
FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"
# debye in an atomic unit of dipole (e bohr)
DEBYE_PER_AU = 2.541746473
# the bohr, in angstroms (CODATA 2018)
BOHR = 0.529177210903
# a frequency smaller than this, either way, is a translation or a rotation xtb lists at naught (cm-1)
FREQUENCY_LEAST = 1.0
# the elements xtb has parameters for, H to Rn, by their atomic numbers
ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def last(pattern, text, flags=re.M):
    """The last match of `pattern` in `text`: of several prints, the last is of the geometry the run ended on."""
    found = list(pattern.finditer(text) if isinstance(pattern, re.Pattern) else re.finditer(pattern, text, flags))
    return found[-1] if found else None


# --- what the run was ----------------------------------------------------------

# "   * xtb version 6.7.1 (abc1234) compiled by 'someone' on 2026-01-01"
VERSION = re.compile(r"^\s*\*\s*xtb version\s+(\S+)", re.M)
# the setup box: ":  Hamiltonian                  GFN2-xTB          :"
HAMILTONIAN = re.compile(r"^\s*:\s*Hamiltonian\s+(\S+)\s*:", re.M)
# GFN-FF's heading, as it begins: "|                   G F N - F F                   |"
GFNFF_HEADING = re.compile(r"^\s*\|\s*G F N - F F\s*\|", re.M)
# the setup box: ":  net charge                         -1          :", ":  unpaired electrons                  1          :"
NET_CHARGE = re.compile(r"^\s*:\s*net charge\s+(-?\d+)\s*:", re.M)
UNPAIRED = re.compile(r"^\s*:\s*unpaired electrons\s+(\d+)\s*:", re.M)
# the SUMMARY: ":: total charge               0.000000000000 e     ::"
TOTAL_CHARGE = re.compile(r"::\s*total charge\s+(" + FLOAT + r")\s+e\s*::")


def version_of(text):
    m = VERSION.search(text)
    return m.group(1) if m else None


def method_of(text):
    """The method, as xtb names it: "GFN2-xTB", "GFN1-xTB" - or "GFN-FF", by its heading."""
    m = last(HAMILTONIAN, text)
    if m:
        return m.group(1)
    return "GFN-FF" if GFNFF_HEADING.search(text) else None


def charge_of(text):
    """The charge: as the setup says it, or the summary's total charge where it is a whole number."""
    m = last(NET_CHARGE, text)
    if m:
        return int(m.group(1))
    m = last(TOTAL_CHARGE, text)
    if m:
        q = number(m.group(1))
        if math.isfinite(q) and abs(q - round(q)) < 1e-6:
            return int(round(q))
    return None


def multiplicity_of(text):
    """The multiplicity: one more than the unpaired electrons the setup says; none where it says none (GFN-FF)."""
    m = last(UNPAIRED, text)
    return int(m.group(1)) + 1 if m else None


# --- the atoms and the geometry ------------------------------------------------

# the table of elements a run begins with: "   ID    Z sym.   atoms" and rows "    3    1 H      5-7"
ATOM_TABLE = re.compile(r"^\s*ID\s+Z\s+sym\.\s+atoms\s*$", re.M)
ATOM_ROW = re.compile(r"^\s*\d+\s+(\d+)\s+([A-Za-z]{1,2})\s+([\d,\s-]+?)\s*$")


def atoms_of(text):
    """The atoms' elements, in their order, from the table xtb prints as it begins: each element, and the places of
    its atoms ("1, 2", "5-7"). None where there is no such table, or it does not give every place once."""
    m = ATOM_TABLE.search(text)
    if not m:
        return None
    placed = {}
    for line in text[m.end():].splitlines()[1:]:
        row = ATOM_ROW.match(line)
        if not row:
            break
        z = int(row.group(1))
        el = ELEMENTS[z] if 0 < z < len(ELEMENTS) else row.group(2).capitalize()
        for part in row.group(3).replace(" ", "").split(","):
            if not part:
                continue
            a, _, b = part.partition("-")
            for k in range(int(a), int(b or a) + 1):
                if k in placed:
                    return None
                placed[k] = el
    if not placed or sorted(placed) != list(range(1, len(placed) + 1)):
        return None
    return [placed[k] for k in range(1, len(placed) + 1)]


# an optimisation's last geometry: "final structure:" over a rule, then as xtb was given it
FINAL = re.compile(r"^\s*final structure:\s*\n\s*=+\s*\n", re.M)


def final_structure(text, elements):
    """An optimisation's final structure, as x, y, z of every atom in angstroms: printed as xtb was given the
    geometry - Xmol in angstroms, or a Turbomole `$coord` in bohr. None where it printed none, or another form."""
    m = last(FINAL, text)
    if not m or not elements:
        return None
    lines = text[m.end():].splitlines()
    n = len(elements)
    rows = None
    scale = 1.0
    if lines and lines[0].strip() == str(n):
        rows = lines[2:2 + n]
        cols = (0, 1, 2, 3)
    elif lines and lines[0].strip().startswith("$coord"):
        rows = lines[1:1 + n]
        cols = (3, 0, 1, 2)
        scale = BOHR
    if not rows or len(rows) != n:
        return None
    xyz = []
    for el, line in zip(elements, rows):
        parts = line.split()
        if len(parts) < 4 or parts[cols[0]].capitalize() != el:
            return None
        try:
            xyz += [number(parts[k]) * scale for k in cols[1:]]
        except ValueError:
            return None
    return xyz if all(math.isfinite(v) for v in xyz) else None


# --- energies ------------------------------------------------------------------

# the closing summary: "| TOTAL ENERGY               -5.012345678901 Eh   |"; its SUMMARY's ":: total energy  -5.07 Eh ::"
TOTAL = re.compile(r"\|\s*TOTAL ENERGY\s+(" + FLOAT + r")\s+Eh\s*\|")
SUMMARY_TOTAL = re.compile(r"::\s*total energy\s+(" + FLOAT + r")\s+Eh\s*::")


def total_energy(text):
    """The total energy, in hartrees: the closing summary's, else the last SUMMARY's."""
    m = last(TOTAL, text) or last(SUMMARY_TOTAL, text)
    if not m:
        return None
    e = number(m.group(1))
    return e if math.isfinite(e) else None


# --- how the run went ----------------------------------------------------------

# "-1- xtb_geoopt: Geometry optimization did not converge": a numbered message, as xtb gives its errors and warnings
NUMBERED = re.compile(r"^\s*-\d+-\s+(.*\S)\s*$", re.M)
# how an optimisation ended: "*** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***",
# "*** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 2 ITERATIONS ***"
OPT_CONVERGED = re.compile(r"\*\*\*\s*GEOMETRY OPTIMIZATION CONVERGED")
OPT_FAILED = re.compile(r"\*\*\*\s*FAILED TO CONVERGE GEOMETRY OPTIMIZATION")


def run_of(text):
    """How the run ended, as xtb says it: "error" - and why, in its words - where it stopped on one ("[ERROR]
    Program stopped due to fatal error", or "abnormal termination of xtb" on its other stream); "normal" where
    it came to its end ("normal termination of xtb" on its other stream, or its own stream's last "* finished
    run on"); "cut" where it says neither. No SCF is said to have failed: xtb stops where one does."""
    if "[ERROR] Program stopped" in text or "abnormal termination of xtb" in text:
        return "error", why_of(text), []
    if "normal termination of xtb" in text or re.search(r"^\s*\*\s*finished run on", text, re.M):
        return "normal", None, []
    return "cut", None, []


def why_of(text):
    """Why xtb stopped, as it says: its first numbered message, and the last, where there are two - the error
    block's, after "[ERROR]", where there is one."""
    at = text.rfind("[ERROR]")
    said = [m.group(1).strip() for m in NUMBERED.finditer(text[at:] if at >= 0 else text)]
    said = [s[len("Error:"):].strip() if s.startswith("Error:") else s for s in said]
    if not said:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        return lines[-1] if lines else "xtb said nothing"
    return said[0] if len(said) == 1 else f"{said[0]}: {said[-1]}"


def optimised_of(text):
    """Whether an optimisation came to an end, as xtb says it: None where it optimised nothing."""
    ok, failed = last(OPT_CONVERGED, text), last(OPT_FAILED, text)
    if ok is None and failed is None:
        return None
    return failed is None or (ok is not None and ok.start() > failed.start())


# (xtb's own words, as it warns of frequencies at a geometry whose gradient is not near zero: Meno's warning kind
#  hessian-not-stationary, as docs/PLUGINS.md states it, its words beside it)
HESSIAN_NOT_STATIONARY = "Hessian on incompletely optimized geometry"


def warnings_of(text):
    """What xtb warned of as it ran on: the numbered messages of each block it heads "[WARNING]", each once."""
    out = []
    for block in re.finditer(r"^\[WARNING\][^\n]*\n((?:[ \t]*-\d+-[^\n]*\n?)+)", text, re.M):
        for m in NUMBERED.finditer(block.group(1)):
            said = m.group(1)
            if all(w["text"] != said for w in out):
                out.append({"kind": "hessian-not-stationary" if HESSIAN_NOT_STATIONARY in said else "other", "text": said})
    return out


# --- the solvation -------------------------------------------------------------

# the setup ("Implicit Solvation"): "      * Solvation model:               ALPB", its solvent and reference state a few lines on
SOLVATION_MODEL = re.compile(r"^\s*\*\s*Solvation model:\s*(\S.*?)\s*$", re.M)
SOLVENT_LINE = re.compile(r"^\s*Solvent\s+(\S.*?)\s*$", re.M)
REFERENCE_LINE = re.compile(r"^\s*Reference state\s+(\S.*?)\s*$", re.M)
# the setup box, where none is used: ":  GBSA solvation                  false          :"
NO_SOLVATION = re.compile(r"^\s*:\s*GBSA solvation\s+false\s*:", re.M)
# the options that ask xtb for a solvation model ("Commandline Usage"), as the program call it prints says them
SOLVATION_FLAGS = {"--alpb", "--gbsa", "-g", "--cosmo", "--cpcmx"}


def solvation_of(text):
    """The solvation a run used, as xtb printed it in its setup: "ALPB(water), reference state gsolv [1 M
    gas/solution]"; "none" where its setup says none is used - or, where it prints no such setup (GFN-FF), where
    its program call asked for none and it printed no model; otherwise None - not said."""
    m = SOLVATION_MODEL.search(text)
    if m:
        block = "\n".join(text[m.end():].splitlines()[1:12])
        solvent = SOLVENT_LINE.search(block)
        reference = REFERENCE_LINE.search(block)
        if not solvent:
            return None
        return f"{m.group(1)}({solvent.group(1)})" + (f", reference state {reference.group(1)}" if reference else "")
    if NO_SOLVATION.search(text):
        return "none"
    call = re.search(r"program call\s*:\s*(.*)$", text, re.M)
    if call and not SOLVATION_FLAGS.intersection(call.group(1).split()):
        return "none"
    return None


# --- results -------------------------------------------------------------------


def orbitals_of(text):
    """The orbitals xtb printed, in Meno's general form (src/lib/calc/orbitals.ts): each of both spins, its energy in
    hartrees, its occupation - none printed, none in it - and its number as xtb counts them, in xtb's order, from the
    last "Orbital Energies and Occupations" table: "         4        2.0000           -0.4412345             -12.0066
    (HOMO)". xtb prints those about the frontier and leaves the rest out between them ("...") - those it printed alone,
    by their numbers, none made up. Meno names them and works out the gap; none where it printed none."""
    at = text.rfind("Orbital Energies and Occupations")
    if at < 0:
        return []
    out = []
    for line in text[at:].splitlines()[3:]:
        if set(line.strip()) <= {"-"} and out:
            break
        # (the orbitals it leaves out, far from the frontier: "       ...           ...                  ...")
        if line.strip() and set(line.split()) == {"..."}:
            continue
        m = re.match(r"^\s*(\d+)\s+(?:(" + FLOAT + r")\s+)?(" + FLOAT + r")\s+(" + FLOAT + r")(?:\s+\((HOMO|LUMO)\))?\s*$", line)
        if not m:
            if out:
                break
            continue
        out.append({"spin": "both", "energy": number(m.group(3)), "occupation": number(m.group(2)) if m.group(2) else 0.0, "number": int(m.group(1))})
    return out


# GFN2-xTB's property printout: "     #   Z          covCN         q      C6AA      α(0)" over rows "     1   8 O        1.612    -0.534    23.456     6.789"
GFN2_CHARGES = re.compile(r"^\s*#\s+Z\s+covCN\s+q\s+C6AA", re.M)
# GFN1-xTB's: "  Mulliken/CM5 charges         n(s)   n(p)   n(d)" over rows "     1O    -0.65432 -0.98765   1.678  4.987  0.000"
GFN1_CHARGES = re.compile(r"^\s*Mulliken/CM5 charges", re.M)


def charges_of(text, n):
    """Each atom's partial charges, as the last property printout gives them ("Properties"): GFN2-xTB's atomic
    partial charges, its q; GFN1-xTB's Mulliken and CM5 charges. None where they are not one to each atom."""
    out = []

    def add(id_, label, values):
        if len(values) == n and all(math.isfinite(v) for v in values):
            out.append({"id": id_, "on": "atoms", "group": "Partial charges", "label": label, "quantity": "charge", "values": values})

    m = last(GFN2_CHARGES, text)
    if m:
        rows = text[m.end():].splitlines()[1:1 + n]
        try:
            # (its "atomic partial charge q", of no scheme the documentation names: xtb's own)
            add("charges", "xTB", [number(r.split()[4]) for r in rows])
        except (IndexError, ValueError):
            pass
    m = last(GFN1_CHARGES, text)
    if m:
        rows = text[m.end():].splitlines()[1:1 + n]
        try:
            # (the atom's place and element run together: "1O", "12Cl")
            cells = [re.match(r"^\s*\d+[A-Za-z]{1,2}\s+(" + FLOAT + r")\s+(" + FLOAT + r")", r).groups() for r in rows]
            add("charges.mulliken", "Mulliken", [number(a) for a, _ in cells])
            add("charges.cm5", "CM5", [number(b) for _, b in cells])
        except (AttributeError, ValueError):
            pass
    return out


# GFN2-xTB's: "molecular dipole:" over a heading, its "q only:" row, then "   full:       -0.000       0.000      -0.902       2.292"
GFN2_DIPOLE = re.compile(
    r"molecular dipole:[^\n]*\n[^\n]*\n\s*q only:[^\n]*\n\s*full:\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")"
)
# GFN-FF's, of its charges alone: "molecular dipole:" over a heading and " q only:        0.000       0.000      -0.695       1.767"
GFNFF_DIPOLE = re.compile(
    r"molecular dipole:[^\n]*\n[^\n]*\n\s*q only:\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\n(?!\s*full:)"
)
# GFN1-xTB's: "dipole moment from electron density (au)" over "     X       Y       Z   " and "   0.0000   0.0000  -1.1234  total (Debye):    2.855"
GFN1_DIPOLE = re.compile(
    r"dipole moment from electron density \(au\)[^\n]*\n[^\n]*\n\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+total \(Debye\):\s+(" + FLOAT + r")"
)


# the Geometry Summary xtb prints after an optimisation: "   center of mass at/Å    :        0.3912345       0.0012345       0.0000000"
CENTRE = re.compile(r"^\s*center of mass at/Å\s*:\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*$", re.M)


def dipole_of(text):
    """The dipole moment as xtb printed it last - of the geometry it ended on: its total in debye, as printed; its
    x, y and z, printed in atomic units, in debye; the point it is taken about, the origin of the coordinates xtb
    was given (`origin`, angstroms) - for a charged molecule's dipole depends on it; and, where xtb printed it after,
    that geometry's centre of mass (`centre`, angstroms, in the same coordinates)."""
    found = [m for p in (GFN2_DIPOLE, GFNFF_DIPOLE, GFN1_DIPOLE) for m in p.finditer(text)]
    if not found:
        return []
    m = max(found, key=lambda m: m.start())
    vector = [number(m.group(k)) * DEBYE_PER_AU for k in (1, 2, 3)]
    total = number(m.group(4))
    if not all(math.isfinite(v) for v in vector + [total]):
        return []
    # (where it says where the centre of mass of the same geometry is - its Geometry Summary, printed after an
    #  optimisation's property printout - that too, in the same coordinates)
    centre = last(CENTRE, text)
    centre = [number(centre.group(k)) for k in (1, 2, 3)] if centre and centre.start() > m.start() else None
    return [{
        "id": "dipole", "on": "molecule", "group": "Properties", "label": "Dipole moment",
        "value": total, "quantity": "dipole", "rank": 2, "vector": vector, "origin": [0.0, 0.0, 0.0],
        **({"centre": centre} if centre and all(math.isfinite(v) for v in centre) else {}),
    }]


def pairs_after(heading, text):
    """The numbered values under a heading of the Frequency Printout, in order: "   1:  6.12   2: 11.34" (a wide
    value may run into its number: "10:134.56"); one too wide for its field, which xtb prints as asterisks
    ("9:******"), not known (None)."""
    at = text.rfind(heading)
    if at < 0:
        return None
    values = {}
    for line in text[at:].splitlines()[1:]:
        found = re.findall(r"(\d+):\s*(" + FLOAT + r"|\*+)", line)
        if not found:
            break
        for k, v in found:
            values[int(k)] = None if v.startswith("*") else number(v)
    return [values[k] for k in sorted(values)] if values else None


# the Frequency Printout's heading: " projected vibrational frequencies (cm⁻¹)" - of a linear molecule,
# " vibrational frequencies (cm⁻¹)"
FREQUENCIES = re.compile(r"^[ \t]*(?:projected )?vibrational frequencies \(cm", re.M)


def frequencies_of(text):
    """The frequencies of the last Frequency Printout (cm-1, an imaginary one negative), as its "eigval :" rows give
    them; none where it printed none."""
    m = last(FREQUENCIES, text)
    if not m:
        return None
    freqs = []
    for line in text[m.start():].splitlines()[1:]:
        row = re.match(r"^\s*eigval\s*:\s*(.*)$", line)
        if not row:
            break
        freqs += [number(v) for v in row.group(1).split()]
    return freqs or None


def vibrations_of(text):
    """The vibrations, as a list: each one's frequency, reduced mass and IR intensity, where the Frequency Printout
    gives one for each - the translations and rotations it lists at naught left out."""
    freqs = frequencies_of(text)
    if not freqs:
        return []
    masses = pairs_after(" reduced masses (amu)", text)
    irs = pairs_after(" IR intensities (km", text)
    masses = masses if masses and len(masses) == len(freqs) else None
    irs = irs if irs and len(irs) == len(freqs) else None
    columns = [{"label": "Mode", "quantity": "number", "digits": 0}, {"label": "Frequency", "quantity": "wavenumber"}]
    if masses is not None:
        columns.append({"label": "Reduced mass", "unit": "amu", "digits": 2})
    if irs is not None:
        columns.append({"label": "IR intensity", "unit": "km/mol", "digits": 1})
    rows = []
    for i, f in enumerate(freqs):
        if abs(f) < FREQUENCY_LEAST:
            continue
        rows.append({"cells": [len(rows) + 1, f] + ([masses[i]] if masses is not None else []) + ([irs[i]] if irs is not None else [])})
    return [{"id": "vibrations", "on": "list", "group": "Vibrations", "label": "Vibrations", "columns": columns, "rows": rows}]


def imaginary_of(text, frequencies):
    """The imaginary frequencies xtb judged significant, in cm-1, negative: as many as it says it found ("found 1
    significant imaginary frequency"), the most negative of `frequencies`; none where it says nothing of them."""
    found = last(r"found\s+(\d+)\s+significant imaginary frequenc", text)
    n = int(found.group(1)) if found else 0
    return sorted(f for f in frequencies if math.isfinite(f) and f < 0)[:n]


def treatment_of(text):
    """How the thermochemistry was worked out, as the setup printed before it says: its rotor cutoff, its imaginary
    cutoff and its scaling factor; None where it printed no setup."""
    at = text.rfind("Thermodynamic Functions")
    if at < 0:
        return None
    said = {}
    for key in ("scaling factor", "rotor cutoff", "imag. cutoff"):
        m = re.search(r":\s*" + re.escape(key) + r"\s+(" + FLOAT + r")", text[at:])
        if not m:
            return None
        said[key] = f"{number(m.group(1)):g}"
    return f"rotor cutoff {said['rotor cutoff']} cm⁻¹, imag. cutoff {said['imag. cutoff']} cm⁻¹, scaling factor {said['scaling factor']}"


# a linear molecule's point group, as xtb's box spells it, in Schoenflies' notation: "din" of a molecule whose symmetry
# elements it lists as "(i) (Cinf) (C2) 2*(sigma)" ("It seems to be the Dinfh point group"), "cin" of one with "(Cinf)
# (sigma)" - the rest as printed
LINEAR_GROUPS = {"din": "D\u221eh", "cin": "C\u221ev"}


def thermo_of(text):
    """The thermochemistry a frequency job printed: its THERMODYNAMIC summary's free energy, zero-point energy and
    G(RRHO) contribution; the closing summary's enthalpy; the entropy of its "TOT" row; and the table's H(T) and T*S
    at the temperature used - of several prints, each the last, so that all are of one."""
    out = []

    def add(id_, label, value, rank=None, **measure):
        out.append({"id": id_, "on": "molecule", "group": "Thermochemistry", "label": label, "value": value, **measure, **({"rank": rank} if rank else {})})

    def summary(label):
        # ":: total free energy          -7.123456789012 Eh   ::"
        m = last(r"::\s*" + re.escape(label) + r"\s+(" + FLOAT + r")\s+Eh", text)
        return number(m.group(1)) if m else None

    free = summary("total free energy")
    if free is not None:
        add("thermo.free-energy", "Gibbs free energy", free, rank=1, quantity="energy")
    zpe = summary("zero point energy")
    if zpe is not None:
        add("thermo.zpve", "Zero-point energy", zpe, quantity="energy")
    contrib = summary("G(RRHO) contrib.")
    if contrib is not None:
        add("thermo.g-correction", "Correction to Gibbs free energy", contrib, quantity="energy")
    # (the closing summary's: "| TOTAL ENTHALPY             -7.101234567890 Eh   |")
    enthalpy = last(r"\|\s*TOTAL ENTHALPY\s+(" + FLOAT + r")\s+Eh", text)
    if enthalpy:
        add("thermo.enthalpy", "Enthalpy", number(enthalpy.group(1)), quantity="energy")
    # (the entropy of the whole, in cal/(mol K), its "TOT" row: enthalpy, heat capacity, entropy, entropy in J/(mol K) -
    #  "         TOT                        2345.6789     8.1234    51.2345   214.3651")
    tot = last(r"^\s*TOT\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*$", text)
    if tot:
        add("thermo.entropy", "Entropy", number(tot.group(3)), unit="cal/(mol·K)", digits=2)
    # the table under its last heading: the row of the temperature used - marked "(used)", or the last
    # ("    298.15    0.371234E-02    0.281234E-01    0.241234E-01    0.421234E-02")
    at = text.rfind("H(0)-H(T)+PV")
    row = re.compile(r"^\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*(\(used\))?\s*$")
    rows = []
    for line in (text[at:].splitlines()[1:] if at >= 0 else []):
        if set(line.strip()) <= {"-"}:
            if rows:
                break
            continue
        # (what xtb wrote to its other stream may fall between the rows, in a log of both: passed over)
        m = row.match(line)
        if m:
            rows.append(m.groups())
    if rows:
        used = [r for r in rows if r[5]] or rows[-1:]
        t, _, h, ts, _, _ = used[-1]
        add("thermo.h-correction", "Correction to enthalpy", number(h), quantity="energy")
        add("thermo.ts", "T·S", number(ts), quantity="energy")
        add("thermo.temperature", "Temperature", number(t), unit="K", digits=2)
    treatment = treatment_of(text)
    if treatment and out:
        add("thermo.treatment", "Treatment", treatment)
    # (the symmetry its thermochemistry took, as its box prints it: ":  symmetry   c2v  :", ":  rotational number   2  :")
    group = last(r":\s*symmetry\s+(\S+)\s*:", text)
    rotational = last(r":\s*rotational number\s+(\d+)\s*:", text)
    if out and group:
        add("thermo.point-group", "Point group", LINEAR_GROUPS.get(group.group(1), group.group(1)))
    if out and rotational:
        add("thermo.symmetry-number", "Symmetry number", int(rotational.group(1)), unit="none", digits=0)
    return out


def run_results(ended, why):
    """How the run went, where it did not go well: in a few words, first in the chip's line - and, where it stopped
    on an error, why, in xtb's words."""
    if ended == "error":
        out = [{"id": "run", "on": "molecule", "group": "Calculation", "label": "Run", "value": "ended with an error", "rank": 0}]
        if why:
            out.append({"id": "run.why", "on": "molecule", "group": "Calculation", "label": "Error", "value": " ".join(why.split())})
        return out
    if ended == "cut":
        return [{"id": "run", "on": "molecule", "group": "Calculation", "label": "Run", "value": "did not come to its end", "rank": 0}]
    return []


# --- the whole -----------------------------------------------------------------


def read(name, text, followed=False):
    """One run's output, in Meno's form: what xtb printed of it. Raises ValueError where it names no atoms."""
    if not isinstance(text, str) or not text:
        raise ValueError("nothing to read")
    atoms = atoms_of(text)
    if not atoms:
        ended, why, _ = run_of(text)
        raise ValueError(f"No molecule was found in {name}." + (f" xtb stopped: {why}." if ended == "error" and why else ""))
    final = final_structure(text, atoms)
    frames = [final] if final else []
    total = total_energy(text)
    ended, why, _ = run_of(text)
    optimised = optimised_of(text)
    frequencies = frequencies_of(text)
    out = {
        "schema": SCHEMA,
        "program": PROGRAM,
        "version": version_of(text),
        "method": method_of(text),
        "basis": None,
        "charge": charge_of(text),
        "multiplicity": multiplicity_of(text),
        "atoms": atoms,
        "frames": frames,
        "energies": [total] if frames and total is not None else None,
        "optimised": optimised,
    }
    solvation = solvation_of(text)
    if solvation is not None:
        out["solvation"] = solvation
    # (where the run did not end as it should, its geometry is not one it found)
    if frames and ended in ("error", "cut"):
        out["converged"] = [False] * len(frames)
        out["unmarked"] = ended
    out["results"] = (
        run_results(ended, why)
        + thermo_of(text)
        + dipole_of(text)
        + charges_of(text, len(atoms))
        + vibrations_of(text)
    )
    orbitals = orbitals_of(text)
    if orbitals:
        out["orbitals"] = orbitals
    if frequencies is not None:
        out["imaginary"] = imaginary_of(text, frequencies)
    warnings = warnings_of(text)
    if warnings:
        out["warnings"] = warnings
    return out


# --- the frequencies xtb writes in Gaussian's form, for MOLDEN ------------------

# its first lines ("Calculation of Vibrational Frequencies": g98.out, "in GAUSSIAN-format, which can be opened with the
# popular MOLDEN program"): " Entering Gaussian System", " Gaussian 98:", " frequency output generated by the xtb code"
G98_MARK = "frequency output generated by the xtb code"
# "  Center     Atomic     Atomic              Coordinates (Angstroms)" over rows "    1          6             0       -0.000002   -0.000001   -0.530000"
G98_ATOM = re.compile(r"^\s*\d+\s+(\d+)\s+\d+\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*$")


def read_g98(name, text):
    """xtb's frequencies in Gaussian's form, as it writes them for MOLDEN (`g98.out`): its geometry (the "Standard
    orientation"), and its vibrations - each one's frequency, reduced mass and IR intensity, and its motion, the
    normal coordinates under it. Nothing else is in it: no energy, no method."""
    if not isinstance(text, str) or G98_MARK not in text[:2000]:
        raise ValueError(f"No molecule was found in {name}.")
    at = text.find("Standard orientation:")
    atoms, xyz = [], []
    for line in text[at:].splitlines()[5:] if at >= 0 else []:
        m = G98_ATOM.match(line)
        if not m:
            break
        z = int(m.group(1))
        if not 0 < z < len(ELEMENTS):
            raise ValueError(f"No molecule was found in {name}.")
        atoms.append(ELEMENTS[z])
        xyz += [number(m.group(k)) for k in (2, 3, 4)]
    if not atoms:
        raise ValueError(f"No molecule was found in {name}.")
    n = len(atoms)
    freqs, masses, irs, moves = [], [], [], []
    lines = text.splitlines()
    k = 0
    for i, line in enumerate(lines):
        head, _, values = line.partition("--")
        key = head.strip()
        if key in ("Frequencies", "Red. masses", "IR Inten"):
            try:
                numbers = [number(v) for v in values.split()]
            except ValueError:
                raise ValueError(f"{name}: its {key} do not read") from None
            {"Frequencies": freqs, "Red. masses": masses, "IR Inten": irs}[key].extend(numbers)
            # (as many modes in the block as its frequencies: its heading below names three, however many there are)
            if key == "Frequencies":
                k = len(numbers)
        # (the normal coordinates: "Atom AN      X      Y      Z        X ...", then a row an atom, three to each mode)
        elif line.split()[:2] == ["Atom", "AN"]:
            block = [[] for _ in range(k)]
            for row in lines[i + 1:i + 1 + n]:
                cells = row.split()[2:]
                if len(cells) != 3 * k:
                    raise ValueError(f"{name}: its normal coordinates do not read")
                for j in range(k):
                    block[j] += [number(v) for v in cells[3 * j:3 * j + 3]]
            moves += block
    columns = [{"label": "Mode", "quantity": "number", "digits": 0}, {"label": "Frequency", "quantity": "wavenumber"}]
    each = [(label, unit, digits, v) for label, unit, digits, v in (("Reduced mass", "amu", 2, masses), ("IR intensity", "km/mol", 1, irs)) if len(v) == len(freqs)]
    columns += [{"label": label, "unit": unit, "digits": digits} for label, unit, digits, _ in each]
    rows = []
    for i, f in enumerate(freqs):
        if abs(f) < FREQUENCY_LEAST:
            continue
        row = {"cells": [len(rows) + 1, f] + [v[i] for _, _, _, v in each]}
        if len(moves) == len(freqs) and len(moves[i]) == 3 * n:
            row["move"] = moves[i]
        rows.append(row)
    return {
        "schema": SCHEMA,
        "program": PROGRAM,
        "atoms": atoms,
        "frames": [xyz],
        "results": [{"id": "vibrations", "on": "list", "group": "Vibrations", "label": "Vibrations", "columns": columns, "rows": rows}] if rows else [],
    }


def jobs(text):
    """The runs a file holds: xtb writes one run an output."""
    return [text]


def read_jobs(name, text, kind=None):
    """What a file holds: its one run, read - or, of xtb's frequencies in Gaussian's form (`g98.out`), those."""
    if kind == "xtb-g98" or (isinstance(text, str) and G98_MARK in text[:2000]):
        return read_g98(name, text)
    return read(name, text)
