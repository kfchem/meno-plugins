"""ORCA's output, read into Meno's form (docs/PLUGINS.md, *The contract*; src/lib/calc/output.ts).

The ORCA plugin's own reading of what ORCA wrote: a job's output and a file opened by hand alike, one reading for
both. Written from the ORCA 6.1 manual (faccts.de) and the program's printed forms, and from nothing else. Python alone, no package:
each function is given the text and gives back plain data; nothing here reads a file of its own choosing.

What it gives, in Meno's units: the atoms and each geometry ORCA wrote, in angstroms; each geometry's energy as
ORCA states it, in hartrees; what the calculation was - the program and its version, the method and basis set,
the solvation, the charge and multiplicity, each atom's mass; how the run went; and everything else as results in
the general form (src/lib/calc/results.ts). A value ORCA did not print is not read: none is worked out in its place -
but the entropy, its printed T S term over the temperature it printed.
"""

import math
import re

#: The form of the answer, as Meno reads it.
SCHEMA = 1
#: The bohr, in angstroms (CODATA 2018).
BOHR = 0.529177210903
#: Calories per mole in a hartree, for an entropy: the hartree (4.3597447222071e-18 J, CODATA 2018) times the Avogadro
#: constant (6.02214076e23 per mole, exact, SI 2019), over the thermochemical calorie (4.184 J, exact).
CAL_PER_HARTREE = 627509.474

# --- the run as a whole ---------------------------------------------------------------------------------------------

#: The version in ORCA's banner: "Program Version 9.8.7  -  RELEASE   -".
VERSION = re.compile(r"^\s*Program Version (\S+)", re.M)
#: What ORCA prints where it came to its end, and where it stopped on an error ("ORCA finished by error termination in
#: <its part>"), and its error lines ("[file ...]: Error (...): <its message>").
NORMAL = "****ORCA TERMINATED NORMALLY****"
ERROR_TERMINATION = re.compile(r"error termination", re.I)
ERROR_LINE = re.compile(r"^(?:\[file [^\]]*\]:\s*)?(Error\b.*\S)\s*$", re.M)
#: Each SCF's end, converged or not ("*           SCF CONVERGED AFTER   9 CYCLES          *", "SCF NOT CONVERGED AFTER   2
#: CYCLES").
SCF_END = re.compile(r"SCF (NOT )?CONVERGED AFTER\s+\d+\s+CYCLES")


def run_of(text, followed=False):
    """How the run went, as ORCA says in its output: "normal" where it came to its end; "error" where it stopped on one
    - and why, in its words - or "cut" where it says neither, cut short or still running; and whether each SCF it ran
    converged, in order. `followed`: a job of a file of several that another follows ($new_job), which ORCA says came
    to its end only once, at the file's end - so that one it went on from did, unless it says it stopped."""
    text = text or ""
    why = why_of(text) if ERROR_TERMINATION.search(text) else None
    ended = "error" if why else "normal" if NORMAL in text or followed else "cut"
    scf = [not m.group(1) for m in SCF_END.finditer(text)]
    return ended, why, scf


def why_of(text):
    """Why ORCA stopped, in its words: its error's own message (the words after "Error (...):"), else the line it says it ended by an error on, else the first that begins
    "ERROR" or "FATAL ERROR", else the first of its other error words ("INPUT ERROR", "... aborting the run"); else - where
    it stopped with no word of an error, as a run brought down does - the last thing it said. Not every line with
    "error" in it: its SCF settings table has a "Startup error" row."""
    found = ERROR_LINE.search(text or "")
    if found:
        said = re.sub(r"^Error\s*(?:\([^)]*\))?\s*:\s*", "", found.group(1)).strip()
        if said:
            return said
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for pick in (r"(?i:error termination)", r"^(?:FATAL )?ERROR\b", r"(?i:\berror\s*[(:]|\binput error\b|error return|\baborting\b)"):
        for line in lines:
            if re.search(pick, line):
                return line
    return f"ORCA stopped without saying why, after: {lines[-1]}" if lines else "ORCA said nothing"

# --- the input, as the output echoes it -----------------------------------------------------------------------------

#: A line of the input as the output echoes it: "|  1> ! B3LYP D3BJ def2-SVP".
ECHO = re.compile(r"^\|\s*\d+>\s?(.*?)\s*$", re.M)
#: The end of the echoed input.
INPUT_END = "****END OF INPUT****"


def echoed(text):
    """ORCA's input as its output echoes it, each line's text, up to its end."""
    out = []
    for m in ECHO.finditer(text or ""):
        line = m.group(1)
        if INPUT_END in line:
            break
        out.append(line)
    return out


def input_whole(text):
    """Whether the output echoes its input whole: to its end mark."""
    return INPUT_END in (text or "")


def keywords(text):
    """The simple keywords of the input, as the output echoes it: each word after a "!", its comments (from # to the
    line's end, or to the next #: the manual's "Comments") left out."""
    words = []
    for line in echoed(text):
        if line.startswith("!"):
            words += re.sub(r"#[^#]*#?", " ", line[1:]).split()
    return words


def blocks(text):
    """The input's blocks, as the output echoes them: each "%name ... end" block's name, lower case, with its text after
    the name - the lines up to its "end", comments left out."""
    out, name, body = [], None, []
    for line in echoed(text):
        bare = re.sub(r"#[^#]*#?", " ", line).strip()
        if name is None:
            m = re.match(r"%(\w+)\s*(.*)$", bare)
            if not m:
                continue
            name, rest = m.group(1).lower(), m.group(2)
            # (a block on one line: "%freq QuasiRRHO false end")
            if re.search(r"(^|\s)end\s*$", rest, re.I):
                out.append((name, re.sub(r"(^|\s)end\s*$", " ", rest, flags=re.I).strip()))
                name = None
            else:
                body = [rest]
            continue
        if re.fullmatch(r"end", bare, re.I):
            out.append((name, " ".join(body).strip()))
            name, body = None, []
        else:
            body.append(bare)
    return out


# --- the level: method, basis set, solvation ------------------------------------------------------------------------

#: What ORCA says it took a keyword as: "The B3LYP functional is recognized", "The B97-3c composite method is
#: recognized", "The omegaB97X-D3 functional is recognized. Fit by Chai et al." - the line may go on after it.
RECOGNISED = re.compile(r"^The (\S+) (?:functional|composite method) is recognized(?:\s*$|\.)", re.M)
#: The orbital basis set it says it used, under that heading - not an auxiliary one, under its own.
BASIS = re.compile(r"^----- Orbital basis set information -----\s*\nYour calculation utilizes the basis: (\S+)\s*$", re.M)
#: The correlated methods' simple keywords, which the final energy is then of: the MP2 variants (the ORCA 6.1 manual,
#: Table 3.34) and the MDCI module's (Table 3.43), with the integral handling and local prefixes it names
#: (AOX-, AO-, MO-, RI-, RI34-, DLPNO-), F12 and spin scaling.
CORRELATED = re.compile(
    r"(?:(?:AOX|AO|MO|RI34|OO)-)?(?:F12/?D?-)?(?:(?:SCS|SOS)-)?(?:RI-|DLPNO-)?(?:(?:SCS|SOS)-)?"
    r"(?:MP2|RIMP2|MP2RI|MP3|CCSD|CCSD\(T1?\)|QCISD|QCISD\(T\)|CEPA/1|NCEPA/1|NCPF/1)(?:-F12D?(?:/RI)?)?",
    re.I,
)
#: The tight-binding methods' simple keywords, each to the method it runs - by the xtb program (the manual, Table 3.25)
#: or by ORCA itself (Table 3.27) - named as the xtb program names it, and who ran it.
XTB = {
    "XTB0": ("GFN0-xTB", "xtb"), "GFN0-XTB": ("GFN0-xTB", "xtb"),
    "XTB1": ("GFN1-xTB", "xtb"), "GFN1-XTB": ("GFN1-xTB", "xtb"), "GFN-XTB": ("GFN1-xTB", "xtb"),
    "XTB": ("GFN2-xTB", "xtb"), "XTB2": ("GFN2-xTB", "xtb"), "GFN2-XTB": ("GFN2-xTB", "xtb"),
    "XTBFF": ("GFN-FF", "xtb"), "GFN-FF": ("GFN-FF", "xtb"),
    "NATIVE-GFN-XTB": ("GFN1-xTB", "orca"), "NATIVE-GFN1-XTB": ("GFN1-xTB", "orca"), "NATIVE-XTB1": ("GFN1-xTB", "orca"),
    "NATIVE-XTB": ("GFN2-xTB", "orca"), "NATIVE-GFN2-XTB": ("GFN2-xTB", "orca"), "NATIVE-XTB2": ("GFN2-xTB", "orca"),
}
#: The density functionals' simple keywords, as the ORCA 6.1 manual lists them (3.3, "Density Functional Theory"):
#: Table 3.1, the local functionals; 3.2, the (meta-)GGAs; 3.4, the global hybrids; 3.6, the range-separated hybrids;
#: 3.8 and 3.9, the global and range-separated double hybrids. Matched whole, in any case, as ORCA takes them - a
#: functional ORCA says it recognised named first, so these are for the rest (B3LYP with no dispersion correction says
#: none). The composite methods among them (3.2, 3.4, 3.6) have their own basis set.
FUNCTIONALS = {w.upper() for w in """
    HFS LDA LSD VWN VWN5 VWN3 PWLDA
    BP86 BLYP OLYP GLYP XLYP PW91 MPWPW MPWLYP PBE RPBE REVPBE RPW86PBE PWP
    B97M-V B97M-D3BJ B97M-D4 SCANFUNC RSCAN R2SCAN M06L TPSS REVTPSS
    B1LYP B3LYP B3LYP/G O3LYP X3LYP B1P86 B3P86 B3PW91 PW1PW MPW1PW MPW1LYP PBE0 REVPBE0 REVPBE38 BHANDHLYP
    M06 M062X PW6B95 TPSSH TPSS0 R2SCANH R2SCAN0 R2SCAN50
    WB97 WB97X WB97X-V WB97X-D3 WB97X-D3BJ WB97X-D4 WB97X-D4REV CAM-B3LYP LC-BLYP LC-PBE
    WB97M-V WB97M-D3BJ WB97M-D4 WB97M-D4REV WR2SCAN
    DSD-BLYP DSD-BLYP/2013 DSD-PBEP86 DSD-PBEP86/2013 DSD-PBEB95 B2PLYP MPW2PLYP B2GP-PLYP B2K-PLYP B2T-PLYP B2NC-PLYP
    PWPB95 PBE-QIDH PBE0-DH REVDSD-PBEP86/2021 REVDSD-PBEP86-D4/2021 REVDOD-PBEP86/2021 REVDOD-PBEP86-D4/2021
    R2SCAN-CIDH R2SCAN-QIDH R2SCAN0-2 R2SCAN0-DH PR2SCAN50 PR2SCAN69 KPR2SCAN50
    SCS/SOS-B2PLYP21 SCS-PBE-QIDH SOS-PBE-QIDH SCS-B2GP-PLYP21 SOS-B2GP-PLYP21
    WB97M(2) WB97X-2 WPR2SCAN50 RSX-QIDH RSX-0DH WB2PLYP WB2GP-PLYP WB88PP86 WPBEPP86
    SCS/SOS-WB2PLYP SCS-WB2GP-PLYP SOS-WB2GP-PLYP SCS-RSX-QIDH SOS-RSX-QIDH SCS-WB88PP86 SOS-WB88PP86 SCS-WPBEPP86 SOS-WPBEPP86
""".split()}
COMPOSITES = {"B97-3C", "R2SCAN-3C", "PBEH-3C", "B3LYP-3C", "WB97X-3C"}
#: Who ran a tight-binding method, said beside it.
XTB_BY = {"xtb": "the xtb program, through ORCA", "orca": "ORCA itself (its native implementation)"}
#: Hartree-Fock, as the SCF settings say the Hamiltonian is: "Ab initio Hamiltonian  Method .... Hartree-Fock(GTOs)" -
#: and a density functional's, "Density Functional     Method          .... DFT(GTOs)".
HF_HAMILTONIAN = re.compile(r"^\s*Ab initio Hamiltonian\s+Method\s+\.+\s+Hartree-Fock", re.M)
DFT_HAMILTONIAN = re.compile(r"^\s*Density Functional\s+Method\s+\.+\s+DFT", re.M)
#: The Hartree-Fock keywords (the manual, Table 2.4): "HF", and "RHF", "UHF" and "ROHF", which say its kind too.
HF_WORDS = {"HF", "RHF", "UHF", "ROHF"}


def xtb_of(text):
    """The tight-binding method the input names among its simple keywords, and what ran it - the one, where it names
    one alone; else none."""
    found = {XTB[w.upper()] for w in keywords(text) if w.upper() in XTB}
    return found.pop() if len(found) == 1 else None


def level_of(text):
    """The method and basis set, each as the input names it: the functional or composite method ORCA says it
    recognised, else one the manual lists that the input names (`FUNCTIONALS`, `COMPOSITES`); a correlated method the
    input names (its final energy is that method's); Hartree-Fock, where ORCA says it ran it and the input names it, or
    names no method at all; a tight-binding method, as the xtb program names it.
    The basis set ORCA says it used, where the input names it - none for a composite or tight-binding method, whose own
    it is. Otherwise none: never a kind of method in a method's place."""
    words = keywords(text)

    def key(word):
        # (ORCA names a range-separated functional "omega..." where its input says "w...": omegaB97X-D3 for wB97X-D3)
        word = word.lower()
        return re.sub(r"[^0-9a-z]", "", "w" + word[5:] if word.startswith("omega") else word)

    correlated = [w for w in words if CORRELATED.fullmatch(w)]
    recognised = {key(x) for x in RECOGNISED.findall(text or "")}
    named = [w for w in words if key(w) in recognised]
    # (a functional or composite method the manual lists, where ORCA said it recognised none - and only where it ran DFT)
    listed = [w for w in words if w.upper() in FUNCTIONALS | COMPOSITES] if not named and DFT_HAMILTONIAN.search(text or "") else []
    named = named or listed
    xtb = xtb_of(text)
    if correlated:
        method = correlated[0] if len({w.upper() for w in correlated}) == 1 else None
    elif named:
        method = named[0] if len({key(w) for w in named}) == 1 else None
    elif xtb:
        return xtb[0], None
    elif HF_HAMILTONIAN.search(text or "") and not DFT_HAMILTONIAN.search(text or ""):
        said = [w for w in words if w.upper() in HF_WORDS]
        method = said[0] if said else "HF" if not words else None
    else:
        method = None
    composite = bool(named) and not correlated and (re.search(r"composite method is recognized", text or "") is not None or any(w.upper() in COMPOSITES for w in named))
    basis = BASIS.search(text or "")
    basis = basis.group(1) if basis and not composite and any(w.lower() == basis.group(1).lower() for w in words) else None
    return method, basis


#: The solvation simple keywords (the manual, "Implicit Solvation Models"): "CPCM(water)", "SMD(chloroform)", "ALPB(water)"
#: (with a tight-binding method), and "CPCM" alone - water, its default.
SOLVATION_WORD = re.compile(r"(CPCM|CPCMC|SMD|ALPB|DDCOSMO)(?:\((.+)\))?", re.I)


def solvation_of(text):
    """The solvation the run was worked out in, as its input says it - a simple keyword ("CPCM(water)") or the %cpcm
    block (its "smd true", its "SMDsolvent" or "solvent", or its dielectric constant and refractive index) - written
    model(solvent): "CPCM(water)", "SMD(chloroform)", "CPCM(epsilon 80.4, refrac 1.33)"; "none" where the input, echoed
    whole, asks for none - the gas phase; unsaid (None) where the output does not echo its input whole."""
    if not input_whole(text):
        return None
    model, solvent = None, None
    for w in keywords(text):
        m = SOLVATION_WORD.fullmatch(w)
        if m:
            model = m.group(1).upper()
            solvent = m.group(2) or (solvent if model != "CPCM" or solvent else None)
    found = [body for name, body in blocks(text) if name == "cpcm"]
    if found:
        body = " ".join(found)
        model = model or "CPCM"
        if re.search(r"\bsmd\s+true\b", body, re.I):
            model = "SMD"
        named = re.search(r"\b(?:SMDsolvent|solvent)\s+\"([^\"]+)\"", body, re.I)
        if named:
            solvent = named.group(1)
        epsilon = re.search(r"\bepsilon\s+(\S+)", body, re.I)
        refrac = re.search(r"\brefrac\s+(\S+)", body, re.I)
        if epsilon and not named:
            solvent = ", ".join([f"epsilon {epsilon.group(1)}"] + ([f"refrac {refrac.group(1)}"] if refrac else []))
    if not model:
        return "none"
    # (CPCM alone is water, ORCA's default solvent - the manual: "If no solvent is specified, water is used")
    return f"{model}({solvent or 'water'})"


# --- the charge, the multiplicity and the geometries ----------------------------------------------------------------

#: The SCF settings' charge and multiplicity: "Total Charge           Charge          ....    0".
CHARGE = re.compile(r"^\s*Total Charge\s+Charge\s+\.+\s+(-?\d+)", re.M)
MULT = re.compile(r"^\s*Multiplicity\s+Mult\s+\.+\s+(\d+)", re.M)
#: The input's coordinates line, as echoed: "* xyz 0 1", "* xyzfile -1 2 start.xyz".
STAR = re.compile(r"^\*\s*(?:xyz|xyzfile|int|internal|gzmt|pdbfile)\s+(-?\d+)\s+(\d+)", re.I)
#: Where ORCA writes each geometry it works at, in angstroms and in atomic units: its element and x, y, z; and, in
#: atomic units, each one's nuclear charge and mass ("   0 O     8.0000    0    15.999    0.000000 ...").
GEOMETRY = "CARTESIAN COORDINATES (ANGSTROEM)"
GEOMETRY_AU = "CARTESIAN COORDINATES (A.U.)"
AU_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
#: A number Fortran could not fit in its field: written as asterisks.
OVERFLOW = re.compile(r"^\*{2,}$")


def charge_of(text):
    """The charge and multiplicity, as the SCF settings print them - the last - else as the input's coordinates line
    says them; each none where neither does."""
    c, m = CHARGE.findall(text or ""), MULT.findall(text or "")
    if c and m:
        return int(c[-1]), int(m[-1])
    for line in echoed(text):
        s = STAR.match(line)
        if s:
            return int(s.group(1)), int(s.group(2))
    return (int(c[-1]) if c else None), (int(m[-1]) if m else None)


def number(token):
    """A number as ORCA prints it; one too wide for its field (asterisks), or that is none, not known (None)."""
    if OVERFLOW.match(token):
        return None
    try:
        v = float(token)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def table_after(lines, k):
    """The rows of a block from line `k`, its title: after the rule under the title, and any blank line after it, each
    line up to a blank one."""
    i = k + 1
    while i < len(lines) and (re.fullmatch(r"\s*-+\s*", lines[i]) or (i <= k + 2 and not lines[i].strip())):
        i += 1
    rows = []
    while i < len(lines) and lines[i].strip():
        rows.append(lines[i])
        i += 1
    return rows, i


def geometries(text):
    """Each geometry ORCA wrote, in order: where it stands in the text (its line), its atoms' elements and places in
    angstroms - point charges ("Q", which the manual's "Coordinate input" gives as charges, not atoms) left out - and,
    from the block in atomic units after it, which atoms are ghosts (no nuclear charge, no mass: the manual's "Ghost
    atoms") and each atom's mass. A row that does not read leaves its geometry out."""
    lines = (text or "").split("\n")
    out = []
    for k, line in enumerate(lines):
        if not line.startswith(GEOMETRY):
            continue
        rows, end = table_after(lines, k)
        atoms, ok = [], True
        for row in rows:
            parts = row.split()
            if len(parts) != 4:
                ok = False
                break
            if parts[0].upper() == "Q":
                continue
            xyz = [number(t) for t in parts[1:]]
            if any(v is None for v in xyz):
                ok = False
                break
            atoms.append((element(parts[0]), xyz))
        if not ok or not atoms:
            continue
        ghosts, masses = set(), None
        au = next((j for j in range(end, min(end + 4, len(lines))) if lines[j].startswith(GEOMETRY_AU)), None)
        if au is not None:
            found = [AU_ROW.match(r) for r in table_after(lines, au)[0]]
            # (point charges there too, by their charge: left out as above)
            found = [m for m in found if m and m.group(2).upper() != "Q"]
            if len(found) == len(atoms):
                ghosts = {i for i, m in enumerate(found) if float(m.group(3)) == 0.0 and float(m.group(5)) == 0.0}
                masses = [float(m.group(5)) for m in found]
        out.append({"line": k, "atoms": atoms, "ghosts": ghosts, "masses": masses})
    return out


def element(symbol):
    """An element's symbol as Meno writes it: "O", "Cl" - ORCA prints it so, an isotope's or a ghost's as its element's."""
    s = re.sub(r"[^A-Za-z]", "", symbol)
    return s[:1].upper() + s[1:].lower()


#: The input's one block of coordinates, as echoed, where ORCA wrote none of its own (`input_geometry`).
MOVES = re.compile(r".*OPT.*|.*NEB.*|IRC|MD|.*SCAN.*", re.I)
MOVES_BLOCKS = re.compile(r"^(%geom|%irc|%neb|%md|%compound|%coords|\$new_job)\b", re.I)
UNITS = {"angs": 1.0, "bohrs": BOHR}


def input_geometry(text):
    """The input's geometry, where ORCA wrote none of its own - as it writes none for frequencies the xtb program works
    out through it - and it is surely the one the run worked at: the input's one "* xyz" block, its atoms each an element
    and three numbers, its units angstroms unless a simple keyword says Bohrs (the manual's default, and its "! Bohrs");
    and the run one of a single geometry, asking for no optimisation, path, scan or dynamics, and no block that may move
    it. Its atoms and places, or none and why - nothing to say where no input was echoed."""
    lines = echoed(text)
    if not lines:
        return None, None
    words = [w.lower() for w in keywords(text)]
    if any(MOVES.fullmatch(w) for w in words) or any(MOVES_BLOCKS.match(line) for line in lines):
        return None, None
    units = [UNITS[w] for w in words if w in UNITS]
    if len(set(units)) > 1:
        return None, "its input's units are not clear"
    scale = units[0] if units else 1.0
    starts = [i for i, line in enumerate(lines) if line.startswith("*")]
    if len(starts) != 2 or not re.fullmatch(r"\*\s*xyz\s+-?\d+\s+\d+", lines[starts[0]], re.I) or lines[starts[1]] != "*":
        return None, "its input gives no one block of coordinates"
    atoms = []
    for line in lines[starts[0] + 1 : starts[1]]:
        parts = line.split()
        xyz = [number(t) for t in parts[1:]] if len(parts) == 4 else [None]
        if len(parts) != 4 or not re.fullmatch(r"[A-Za-z]{1,2}", parts[0]) or any(v is None for v in xyz):
            return None, f"its input's atom {line!r} is not an element and its place"
        atoms.append((element(parts[0]), [v * scale for v in xyz]))
    return ({"line": 0, "atoms": atoms, "ghosts": set(), "masses": None} if atoms else None), (None if atoms else "its input gives no atoms")


# --- energies, and which geometries are what ------------------------------------------------------------------------

#: ORCA's own statement of a geometry's whole electronic energy, after it: the SCF's and every term the method adds -
#: dispersion, gCP, a composite method's others, a correlated method's correlation ("FINAL SINGLE POINT ENERGY
#: -93.312345678901").
FINAL = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(\S+)\s*$", re.M)
#: What the energies are, said with them.
FINAL_SAID = "ORCA final single-point energy (includes dispersion and gCP where the method has them)"
#: A nudged elastic band among the simple keywords: NEB, NEB-CI, NEB-TS, ZOOM-NEB and the like.
NEB = re.compile(r"(ZOOM-)?NEB(-[A-Z]+)?", re.I)
#: Where an optimisation says it converged, and where ORCA works the energy out once more at the point it converged on:
#: that geometry is the one found.
OPT_CYCLE = "GEOMETRY OPTIMIZATION CYCLE"
OPT_CONVERGED = re.compile(r"\*\*\*\s+THE (?:TS )?OPTIMIZATION HAS CONVERGED\s+\*\*\*")
STATIONARY = "FINAL ENERGY EVALUATION AT THE STATIONARY POINT"
#: A relaxed scan's steps ("RELAXED SURFACE SCAN STEP   1"); a reaction path's summary; a band's.
SCAN_STEP = re.compile(r"RELAXED SURFACE SCAN STEP\s+\d+")
IRC_SUMMARY = "IRC PATH SUMMARY"
NEB_SUMMARY = "PATH SUMMARY"
#: An excited state followed through an optimisation: its root ("%tddft ... iroot 1 end") with Opt.
IROOT = re.compile(r"\biroot\s+\d+", re.I)


def energies_of(text, geos):
    """Each geometry's energy as ORCA states it (`FINAL`), one to each geometry in their order - the first stated after
    it, before the next; one stated again unchanged is the same; with no reason. Or none, and why: a compound job's, a
    band's, one stated where no geometry of its own stands, a geometry with none, one that is no number."""
    if any(line.lower().startswith("%compound") for line in echoed(text)):
        return None, "a compound job's single points are not one to each geometry"
    if any(NEB.fullmatch(w) for w in keywords(text)):
        return None, "a nudged elastic band's images are not one to each geometry"
    if not geos:
        return None, None
    starts = [g["line"] for g in geos]
    text_lines = (text or "").split("\n")
    finals = [(k, m.group(1)) for k, line in enumerate(text_lines) for m in [FINAL.match(line)] if m]
    values = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text_lines)
        mine = [v for k, v in finals if start < k < end]
        if not mine:
            # (a run cut short after its last geometry, before its energy: that one has none, and is not kept either)
            if i == len(starts) - 1:
                break
            return None, "a geometry has no final single-point energy of its own"
        if len(set(mine)) != 1:
            return None, "a final single-point energy stands at no geometry of its own"
        v = number(mine[0])
        if v is None:
            return None, "a final single-point energy is not a number"
        values.append(v)
    before = [v for k, v in finals if k < starts[0]]
    if before and geos[0]["line"] > 0:
        return None, "a final single-point energy stands at no geometry of its own"
    return values, None


def marks_of(text, geos, ended):
    """Of an optimisation's or a relaxed scan's geometries, each one's being one the run found converged: each the
    geometry ORCA worked its energy out at once more, at the point it converged on. None where it optimised nothing.
    And why none of its geometries is one it found, where none is marked: "error" or "cut" where the run did not end
    as it should; "path" for a reaction path's or a band's points read; "unstated" for more geometries than one, none
    of them marked; none for one geometry of a run that ended as it should."""
    lines = (text or "").split("\n")
    # (a band's: the geometry its output gives is the one it started from, its path and point refined in files of their own)
    if geos and any(NEB.fullmatch(w) for w in keywords(text)):
        return [False] * len(geos), ended if ended in ("error", "cut") else "path"
    stationary = [k for k, line in enumerate(lines) if STATIONARY in line]
    optimised = OPT_CYCLE in (text or "") or bool(stationary)
    marks = None
    if optimised and geos:
        starts = [g["line"] for g in geos]
        marks = [False] * len(starts)
        # (the geometry written next after each mark of the point converged on)
        for s in stationary:
            after = next((i for i, g in enumerate(starts) if g > s), None)
            if after is not None:
                marks[after] = True
    if marks is not None and any(marks):
        return marks, None
    n = len(geos)
    if n < 1:
        return marks, None
    if ended in ("error", "cut"):
        return marks, ended
    # (an optimisation that came to its end without converging: each geometry marked not converged - none found)
    if marks is not None:
        return marks, None
    return None, ("unstated" if n > 1 else None)


def optimised_of(text):
    """Whether an optimisation came to an end converged: none where the run optimised nothing - or where its geometries
    are a relaxed scan's points or a band's start, no optimisation's end."""
    if OPT_CYCLE not in (text or "") or SCAN_STEP.search(text or "") or any(NEB.fullmatch(w) for w in keywords(text)):
        return None
    return bool(OPT_CONVERGED.search(text or ""))


# --- results: how the run went, and what the geometries are ---------------------------------------------------------

def run_results(ended, why, scf):
    """How the run went, where it did not go well: in a few words, first in the chip's line - and, where it stopped on an
    error, why, in ORCA's words (its runs of spaces closed up), among its results."""
    if ended == "error":
        label, said = "Run", "ended with an error"
    elif ended == "cut":
        label, said = "Run", "did not come to its end"
    elif scf and False in scf:
        label, said = "SCF", "not converged"
    else:
        return []
    out = [{"id": "run", "on": "molecule", "group": "Calculation", "label": label, "value": said, "rank": 0}]
    if ended == "error" and why:
        out.append({"id": "run.why", "on": "molecule", "group": "Calculation", "label": "Error", "value": " ".join(why.split())})
    return out


def path_said(text, frames):
    """What a run's geometries are where they are no optimisation's path - a relaxed scan's points; a reaction path's
    or a band's, which ORCA writes to files of their own, not among the geometries in its output - and that an excited
    state's energies are not read: first in the chip's line."""
    out = []
    steps = len(SCAN_STEP.findall(text or ""))
    words = [w.upper() for w in keywords(text)]
    if steps:
        out.append({"id": "run.path", "on": "molecule", "group": "Calculation", "label": "Relaxed scan", "value": f"{steps} points", "rank": 0})
    elif IRC_SUMMARY in (text or "") or "IRC" in words:
        out.append({"id": "run.path", "on": "molecule", "group": "Calculation", "label": "Reaction path", "value": "not read", "rank": 0})
    elif any(NEB.fullmatch(w) for w in words):
        out.append({"id": "run.path", "on": "molecule", "group": "Calculation", "label": "Path", "value": "not read", "rank": 0})
    if excited_path(text, frames):
        out.append({"id": "run.energies", "on": "molecule", "group": "Calculation", "label": "Excited state", "value": "energies not read", "rank": 0})
    return out


def excited_path(text, frames):
    """Whether a run's geometries are an excited state's: a root followed (%tddft's iroot) over more than one geometry."""
    return frames > 1 and any(name == "tddft" and IROOT.search(body) for name, body in blocks(text))


def energies_said(energies, unread):
    """What the energies are, where they were read - and, where they could not be, that and why, first in the chip's line."""
    if energies:
        return [{"id": "energy.what", "on": "molecule", "group": "Calculation", "label": "Energy", "value": FINAL_SAID}]
    if unread:
        return [{"id": "run.energies", "on": "molecule", "group": "Calculation", "label": "Energies", "value": f"not read: {unread}", "rank": 0}]
    return []


def xtb_said(text):
    """Of a run at a tight-binding method, what ran it: the xtb program, or ORCA itself."""
    xtb = xtb_of(text)
    return [{"id": "method.by", "on": "molecule", "group": "Calculation", "label": f"{xtb[0]} run by", "value": XTB_BY[xtb[1]]}] if xtb else []


# --- thermochemistry ------------------------------------------------------------------------------------------------

#: The last thermochemistry ORCA printed, each value as it prints it ("Temperature         ...   298.15 K", "Zero point
#: energy                ...      0.02134567 Eh", "Total Enthalpy                    ...    -76.31234567 Eh", "Final
#: entropy term                ...      0.02187654 Eh", "Final Gibbs free energy         ...    -76.33456789 Eh",
#: "G-E(el)                           ...      0.00412345 Eh"), the symmetry it took ("Point Group:  C2v, Symmetry
#: Number:   2"), and how it treated the low vibrations ("Quasi RRHO          ...    False"; "Vibrational entropy computed
#: according to the QRRHO of S. Grimme ... using a reference frequency of 100.0 cm-1").
THERMO_START = re.compile(r"^THERMOCHEMISTRY AT", re.M)
THERMO = {
    "temperature": re.compile(r"^Temperature\s+\.\.\.\s+(\S+) K", re.M),
    "pressure": re.compile(r"^Pressure\s+\.\.\.\s+(\S+) atm", re.M),
    "zpve": re.compile(r"^Zero point energy\s+\.\.\.\s+(\S+) Eh", re.M),
    "enthalpy": re.compile(r"^Total Enthalpy\s+\.\.\.\s+(\S+) Eh", re.M),
    "ts": re.compile(r"^Final entropy term\s+\.\.\.\s+(\S+) Eh", re.M),
    "g": re.compile(r"^Final Gibbs free energy\s+\.\.\.\s+(\S+) Eh", re.M),
    "g-e": re.compile(r"^G-E\(el\)\s+\.\.\.\s+(\S+) Eh", re.M),
}
SYMMETRY = re.compile(r"Point Group:\s+(\S+),\s+Symmetry Number:\s+(\d+)")
QRRHO = re.compile(r"QRRHO of S\. Grimme.*?reference frequency of\s+([0-9.]+)\s*cm-1", re.S)
QRRHO_SAID = re.compile(r"Quasi RRHO\s+\.\.\.\s+(True|False)")


def thermo_of(text):
    """The last thermochemistry the run printed, as results: the free energy first in the chip's line, the enthalpy, the
    zero-point energy, the entropy (in cal/(mol K): its T S over T), the correction to G (G less the electronic energy, as
    ORCA prints it), the temperature and pressure; the point group and the rotational symmetry number it took; and how it
    treated the low vibrations - quasi-RRHO (Grimme) at its reference frequency, or RRHO where it says it did not."""
    text = text or ""
    starts = [m.start() for m in THERMO_START.finditer(text)]
    if not starts:
        return []
    part = text[starts[-1]:]
    got = {}
    for name, rx in THERMO.items():
        m = rx.search(part)
        v = number(m.group(1)) if m else None
        if v is not None:
            got[name] = v
    out = []

    def add(id_, label, value, rank=None, **measure):
        out.append({"id": id_, "on": "molecule", "group": "Thermochemistry", "label": label, "value": value, **measure, **({"rank": rank} if rank is not None else {})})

    if "g" in got:
        add("thermo.free-energy", "Gibbs free energy", got["g"], rank=1, quantity="energy")
    if "enthalpy" in got:
        add("thermo.enthalpy", "Enthalpy", got["enthalpy"], quantity="energy")
    if "zpve" in got:
        add("thermo.zpve", "Zero-point energy", got["zpve"], quantity="energy")
    if "ts" in got and got.get("temperature"):
        add("thermo.entropy", "Entropy", got["ts"] / got["temperature"] * CAL_PER_HARTREE, unit="cal/(mol·K)", digits=2)
    if "g-e" in got:
        add("thermo.g-correction", "Correction to Gibbs free energy", got["g-e"], quantity="energy")
    if "temperature" in got:
        add("thermo.temperature", "Temperature", got["temperature"], unit="K", digits=2)
    if "pressure" in got:
        add("thermo.pressure", "Pressure", got["pressure"], unit="atm", digits=2)
    sym = list(SYMMETRY.finditer(part))
    if sym:
        # (in Meno's form, standard notation: ORCA's "C(inf)v", "D(inf)h" a linear molecule's C∞v, D∞h)
        add("thermo.point-group", "Point group", re.sub(r"\(?inf\)?", "\u221e", sym[-1].group(1), flags=re.I), quantity="symmetry")
        add("thermo.symmetry-number", "Symmetry number", int(sym[-1].group(2)), unit="none", digits=0)
    said = list(QRRHO_SAID.finditer(part))
    q = list(QRRHO.finditer(part))
    if said and said[-1].group(1) == "False":
        add("thermo.treatment", "Treatment", "RRHO")
    elif q:
        add("thermo.treatment", "Treatment", f"quasi-RRHO (Grimme), {float(q[-1].group(1)):g} cm⁻¹")
    return out


# --- vibrations -----------------------------------------------------------------------------------------------------

#: The last list of frequencies ("     6:    1712.34 cm**-1", "     6:    -812.34 cm**-1  ***imaginary mode***" - and, where
#: the run used its symmetry, each mode's irreducible representation, numbered: "     6:      41.23 cm**-1    2-Au"), and the
#: normal modes after it; the IR spectrum ("  6:   1712.34   0.012345   62.34  0.002345  (...)", its Int in km/mol) and
#: the Raman spectrum ("   6:      1612.34      5.123456      0.412345", its Activity).
FREQ_START = "VIBRATIONAL FREQUENCIES"
FREQ_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1(\s+\*\*\*imaginary mode\*\*\*)?(?:\s+\d+-(\S+))?\s*$")
IR_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(\S+)\s+(\S+)\s")
RAMAN_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(\S+)\s+(\S+)\s*$")


def frequency_list(text):
    """The last list of frequencies ORCA printed: each mode's number, its frequency in cm-1 (an imaginary one negative,
    as it prints it), whether it marked it imaginary, and its symmetry where it says; none where it printed none."""
    text = text or ""
    at = text.rfind(FREQ_START)
    if at < 0:
        return None
    out = []
    for line in text[at:].split("\n")[1:]:
        if "NORMAL MODES" in line:
            break
        m = FREQ_ROW.match(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2)), bool(m.group(3)), m.group(4)))
    return out


def imaginary_of(text):
    """The frequencies ORCA marked imaginary in its last list of them, each negative as it prints it; none where it
    worked out no frequencies."""
    found = frequency_list(text)
    return None if found is None else [f for _, f, imaginary, _ in found if imaginary]


def normal_modes(text, n):
    """The last normal modes ORCA printed, each mode's 3n displacements by its number - as columns of six, each row an
    atom's x, y or z; none where they do not read."""
    text = text or ""
    at = text.rfind("NORMAL MODES")
    if at < 0 or at < text.rfind(FREQ_START):
        return None
    lines = text[at:].split("\n")
    modes, head = {}, None
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            if modes and head is None:
                break
            continue
        # (a block's columns, by mode number - six or ten to a block - and, where the run used its symmetry, each mode's
        #  irreducible representation under it: "1-Au")
        if all(p.isdigit() for p in parts):
            head = [int(p) for p in parts]
            continue
        if head and all(re.fullmatch(r"\d+-\S+", p) for p in parts):
            continue
        if head and parts[0].isdigit() and len(parts) == len(head) + 1:
            row = int(parts[0])
            for j, col in enumerate(head):
                v = number(parts[j + 1])
                modes.setdefault(col, {})[row] = v
            continue
        if modes:
            break
    out = {}
    for col, rows in modes.items():
        values = [rows.get(i) for i in range(3 * n)]
        if len(rows) == 3 * n and all(v is not None for v in values):
            out[col] = values
    return out


def spectrum(text, title, row):
    """A spectrum's values by mode number, from its last table under `title`: the third column of each row."""
    text = text or ""
    at = text.rfind(title)
    if at < 0:
        return {}
    out = {}
    for line in text[at:].split("\n")[1:]:
        m = row.match(line)
        if m:
            v = number(m.group(3) if title.startswith("RAMAN") else m.group(4))
            if v is not None:
                out[int(m.group(1))] = v
        elif out and not line.strip():
            break
    return out


def vibrations_of(text, n, ghosts):
    """The vibrations, as a list: each one's frequency (an imaginary one negative), its IR intensity and Raman activity
    where ORCA printed them; chosen, the molecule moves along its normal mode. Its translations and rotations, at a
    frequency of naught, left out; a ghost atom's displacements too."""
    found = frequency_list(text)
    if not found:
        return []
    modes = normal_modes(text, n)
    ir = spectrum(text, "IR SPECTRUM", IR_ROW)
    raman = spectrum(text, "RAMAN SPECTRUM", RAMAN_ROW)
    vibrating = [(k, f) for k, f, _, _ in found if f != 0.0]
    irreps = {k: sym for k, _, _, sym in found}
    has_sym = all(irreps.get(k) for k, _ in vibrating)
    if not vibrating:
        return []
    # (a column where the spectrum gives any of them - a mode it leaves out, an imaginary one, not known there)
    has_ir = any(k in ir for k, _ in vibrating)
    has_raman = any(k in raman for k, _ in vibrating)
    columns = [{"label": "Mode", "quantity": "number", "digits": 0}] + ([{"label": "Symmetry", "quantity": "symmetry"}] if has_sym else []) + [{"label": "Frequency", "quantity": "wavenumber"}]
    if has_ir:
        columns.append({"label": "IR intensity", "unit": "km/mol", "digits": 1})
    if has_raman:
        columns.append({"label": "Raman activity", "unit": "Å⁴/amu", "digits": 2})
    keep = [i for i in range(n) if i not in ghosts]
    rows = []
    for k, f in vibrating:
        cells = [len(rows) + 1] + ([irreps[k]] if has_sym else []) + [f] + ([ir.get(k)] if has_ir else []) + ([raman.get(k)] if has_raman else [])
        row = {"cells": cells}
        if modes and k in modes:
            row["move"] = [modes[k][3 * i + c] for i in keep for c in range(3)]
        rows.append(row)
    return [{"id": "vibrations", "on": "list", "group": "Vibrations", "label": "Vibrations", "columns": columns, "rows": rows}]


# --- the dipole, partial charges, orbitals --------------------------------------------------------------------------

#: The dipole moment block: the method and state it is of, its total in atomic units ("Total Dipole Moment    :
#: -0.000000000      -0.000000000      -0.812345678"), and its magnitude in both ("Magnitude (a.u.)       :      0.812345678",
#: "Magnitude (Debye)      :      2.064791234"); and the point it was taken about, said before it ("Position of electric
#: origin ...  0.000000 -0.000000 -0.000000", or "... (  0.0000,  0.0000,  0.1086)"), in bohr.
DIPOLE_BLOCK = re.compile(r"^DIPOLE MOMENT\s*\n-+\s*\n(.*?)^Magnitude \(Debye\)[^\n]*", re.M | re.S)
DIPOLE_TOTAL = re.compile(r"^Total Dipole Moment\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)
MAGNITUDE_AU = re.compile(r"^Magnitude \(a\.u\.\)\s*:\s*(\S+)", re.M)
MAGNITUDE_D = re.compile(r"^Magnitude \(Debye\)\s*:\s*(\S+)", re.M)
ORIGIN = re.compile(r"^Position of electric origin\s+\.\.\.\s+\(?\s*(-?\d+\.\d+),?\s+(-?\d+\.\d+),?\s+(-?\d+\.\d+)\s*\)?", re.M)
STATE = re.compile(r"^State\s*:\s*(\d+)", re.M)
METHOD_SCF = re.compile(r"^Method\s*:\s*SCF\b", re.M)
#: The relaxed density's population analysis of an excited state: "RELAXED CIS/TDA DENSITY POPULATION ANALYSIS" over "IROOT 1".
ROOT_POPULATION = re.compile(r"DENSITY POPULATION ANALYSIS\s*\n\s*IROOT\s+(\d+)")
#: The excited states a TD-DFT or CIS job worked out ("STATE  1:  E=   0.151234 au ...").
EXCITED = re.compile(r"^STATE\s+\d+:\s+E=", re.M)


def states_of(text):
    """Of a job with excited states, which state its dipole and its charges are of, as the output says: "ground state",
    "excited state 1" - or "" where it does not say, so that they are left out; the orbitals are always the ground
    state's. For a job with none: None each, said as ever."""
    text = text or ""
    if not EXCITED.search(text):
        return None, None
    blocks_ = [m.group(1) for m in DIPOLE_BLOCK.finditer(text)]
    root = STATE.search(blocks_[-1]) if blocks_ else None
    dipole = "" if not blocks_ else f"excited state {root.group(1)}" if root else "ground state" if METHOD_SCF.search(blocks_[-1]) else ""
    head = text[: text.rfind("MULLIKEN ATOMIC CHARGES")].rfind("ORCA POPULATION ANALYSIS")
    root = ROOT_POPULATION.search(text[max(0, head - 400):head]) if head >= 0 else None
    charges = "" if head < 0 else f"excited state {root.group(1)}" if root else "ground state"
    return dipole, charges


def of_state(results, state):
    """Results said to be of the state they are: their labels so, or left out where the output does not say which; as
    they are where the job has no excited states."""
    if state is None:
        return results
    return [{**r, "label": f"{r['label']}, {state}"} for r in results] if state else []


def dipole_of(text):
    """The last dipole moment ORCA printed, as it printed it: its magnitude in debye, its components in debye (its total
    in atomic units, by ORCA's own ratio of the magnitudes it printed in each), and the point it was taken about, in
    angstroms - the last ORCA said before it. None where it printed none, or no point."""
    text = text or ""
    found = list(DIPOLE_BLOCK.finditer(text))
    if not found:
        return []
    block = found[-1].group(0)
    total, au, debye = DIPOLE_TOTAL.search(block), MAGNITUDE_AU.search(block), MAGNITUDE_D.search(block)
    value = number(debye.group(1)) if debye else None
    if value is None:
        return []
    vector = [number(v) for v in total.groups()] if total else None
    a = number(au.group(1)) if au else None
    origins = [m for m in ORIGIN.finditer(text[: found[-1].start()])]
    out = {"id": "dipole", "on": "molecule", "group": "Properties", "label": "Dipole moment", "value": value, "quantity": "dipole", "rank": 2}
    if vector and all(v is not None for v in vector) and a:
        out["vector"] = [v * value / a for v in vector]
    elif vector and all(v == 0 for v in vector):
        out["vector"] = [0.0, 0.0, 0.0]
    if origins and "vector" in out:
        out["origin"] = [float(v) * BOHR for v in origins[-1].groups()]
    return [out]


#: Each scheme's partial charges as ORCA prints them, the last of each: "MULLIKEN ATOMIC CHARGES" (and spin populations,
#: "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS", a column each), "LOEWDIN ...", rows "   0 O :   -0.812345"; the
#: Hirshfeld and MBIS analyses' tables ("  ATOM     CHARGE      SPIN", rows "   0 O   -0.323456    0.000000"); CHELPG's
#: ("CHELPG Charges", rows "  0   O   :      -0.734567").
SCHEMES = [
    ("mulliken", "Mulliken", re.compile(r"^MULLIKEN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?\s*$", re.M), "colon"),
    ("lowdin", "Löwdin", re.compile(r"^LOEWDIN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?\s*$", re.M), "colon"),
    ("hirshfeld", "Hirshfeld", re.compile(r"^HIRSHFELD ANALYSIS\s*$", re.M), "table"),
    ("mbis", "MBIS", re.compile(r"^MBIS ANALYSIS\s*$", re.M), "table"),
    ("chelpg", "CHELPG", re.compile(r"^CHELPG Charges\s*$", re.M), "colon"),
]
COLON_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\S*\s*:\s*(\S+)(?:\s+(\S+))?\s*$")
TABLE_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\S*\s+(\S+)(?:\s+(\S+))*\s*$")


def charges_of(text, n, ghosts):
    """Each scheme's partial charges, and spin populations where it gives them - the last of each, one to each atom, a
    ghost atom's left out."""
    text = text or ""
    out = []
    keep = [i for i in range(n) if i not in ghosts]
    for scheme, label, title, form in SCHEMES:
        found = list(title.finditer(text))
        if not found:
            continue
        lines = text[found[-1].start():].split("\n")
        charges, spins = {}, {}
        started = False
        for line in lines[1:80]:
            if form == "colon":
                m = COLON_ROW.match(line)
                # (a point charge, listed with the atoms: no atom's)
                if m and m.group(2).upper() == "Q":
                    continue
                if m:
                    started = True
                    charges[int(m.group(1))] = number(m.group(3))
                    if m.group(4):
                        spins[int(m.group(1))] = number(m.group(4))
                elif started:
                    break
            else:
                parts = line.split()
                if parts[:1] == ["ATOM"]:
                    head = parts
                    started = True
                    continue
                if started and parts and parts[0].isdigit() and len(parts) >= 3 and parts[1].upper() == "Q":
                    continue
                if started and parts and parts[0].isdigit() and len(parts) >= 3:
                    charges[int(parts[0])] = number(parts[2])
                    if "SPIN" in head:
                        k = head.index("SPIN")
                        spins[int(parts[0])] = number(parts[k + 1]) if k + 1 < len(parts) else None
                elif started and charges:
                    break
        for kind, values, group, quantity in (("charges", charges, "Partial charges", "charge"), ("spins", spins, "Spin populations", "spin")):
            if len(values) == n and all(values.get(i) is not None for i in range(n)):
                if kind == "spins" and all(values[i] == 0 for i in range(n)):
                    continue
                out.append({"id": f"{kind}.{scheme}", "on": "atoms", "group": group, "label": label, "quantity": quantity, "values": [values[i] for i in keep]})
    return out


#: The orbital energies: under "ORBITAL ENERGIES", each orbital's number, occupation and energy in hartrees and in eV
#: ("   4   2.0000      -0.412345       -11.2205") - and, where the run used its symmetry, its irreducible representation,
#: numbered ("   0   2.0000     -10.123456      -275.4730    1-Bu"); an open shell's under "SPIN UP ORBITALS" and "SPIN DOWN
#: ORBITALS".
ORBITAL_ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\S+)(?:\s+\d+-(\S+))?\s*$")


def orbital_rows(lines):
    """The rows of one list of orbitals, from its first: each orbital's occupation, energy in hartrees and symmetry (none
    where ORCA printed none), up to the first line that is none."""
    rows, started = [], False
    for line in lines:
        m = ORBITAL_ROW.match(line)
        if m:
            started = True
            rows.append((float(m.group(2)), number(m.group(3)), m.group(5)))
        elif started:
            break
    return rows


def orbitals_of(text):
    """The orbitals ORCA printed last, in Meno's general form: each its spin (of both, for a restricted calculation; else
    alpha, then beta), its energy in hartrees, its occupation and its symmetry where ORCA printed one - every one, in
    ORCA's order. Meno lists them itself (src/lib/calc/orbitals.ts); none where ORCA printed none, or not as a list it
    reads whole. Of a job with excited states, they are the SCF's, the ground state's."""
    text = text or ""
    at = text.rfind("ORBITAL ENERGIES")
    if at < 0:
        return None
    lines = text[at:].split("\n")[1:]
    up = next((k for k, line in enumerate(lines[:6]) if "SPIN UP ORBITALS" in line), None)
    if up is None:
        spins = [("both", orbital_rows(lines))]
    else:
        down = next((k for k, line in enumerate(lines) if "SPIN DOWN ORBITALS" in line), None)
        spins = [("alpha", orbital_rows(lines[up + 1:]))] + ([("beta", orbital_rows(lines[down + 1:]))] if down is not None else [])
    if not all(rows for _, rows in spins) or len(spins) != (1 if up is None else 2):
        return None
    if any(e is None for _, rows in spins for _, e, _ in rows):
        return None
    return [
        {"spin": spin, "energy": e, "occupation": occ, **({"symmetry": irrep} if irrep else {})}
        for spin, rows in spins
        for occ, e, irrep in rows
    ]


#: Each optimisation cycle's gradient, its components by atom ("CARTESIAN GRADIENT", rows "   1   O   :   -0.000000000    0.000000000    0.012345678").
GRADIENT_ROW = re.compile(r"^\s*\d+\s+[A-Za-z]{1,2}\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s*$")


def gradients_of(text, geos):
    """Each geometry's RMS gradient, in Eh/bohr - where ORCA printed one after every geometry, before the next."""
    lines = (text or "").split("\n")
    starts = [g["line"] for g in geos]
    values = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        at = next((k for k in range(start, end) if lines[k].startswith("CARTESIAN GRADIENT")), None)
        if at is None:
            return []
        comps = []
        for line in table_after(lines, at)[0]:
            m = GRADIENT_ROW.match(line)
            if m:
                comps += [number(v) for v in m.groups()]
        if not comps or any(v is None for v in comps):
            return []
        values.append(math.sqrt(sum(v * v for v in comps) / len(comps)))
    if not values:
        return []
    return [{"id": "rms-gradient", "on": "frames", "group": "Optimisation", "label": "RMS gradient", "unit": "Eh/bohr", "digits": 6, "values": values}]


# --- one job ---------------------------------------------------------------------------------------------------------


def without(atoms_of, gone):
    return [a for i, a in enumerate(atoms_of) if i not in gone]


def read(name, text, followed=False):
    """One ORCA job's output, in Meno's form. Raises ValueError where it holds no molecule."""
    if not isinstance(text, str) or not text:
        raise ValueError("nothing to read")
    geos = geometries(text)
    given = None
    if not geos:
        one, given = input_geometry(text)
        geos = [one] if one else []
    if not geos:
        raise ValueError(f"No molecule was found in {name}." + (f" ORCA wrote no geometry, and {given}." if given else ""))
    # (the atoms are those of the first geometry; one of other atoms is no step of this molecule's)
    first = [el for el, _ in geos[0]["atoms"]]
    geos = [g for g in geos if [el for el, _ in g["atoms"]] == first]
    # (a band's output: the geometry it started from alone - its images and the point it refined are in files of their
    #  own, and the steps of that refinement written here are no path's points; none read as a result)
    if any(NEB.fullmatch(w) for w in keywords(text)):
        geos = geos[:1]
    ghosts = geos[0]["ghosts"]
    keep = [i for i in range(len(first)) if i not in ghosts]
    atoms = [first[i] for i in keep]
    ended, why, scf = run_of(text, followed)
    energies, unread = energies_of(text, geos)
    # (a run that stopped: its geometries after the last it worked an energy out for left out, as it did not finish them)
    if ended in ("error", "cut") and energies is not None and 0 < len(energies) < len(geos):
        geos = geos[: len(energies)]
    if energies is not None and len(energies) != len(geos):
        energies, unread = None, unread or f"{len(energies)} final single-point energies for {len(geos)} geometries"
    # (an SCF that did not converge, the run carrying on: its energy not to be taken)
    if energies is not None and scf and False in scf:
        energies, unread = None, "an SCF did not converge"
    frames = [[v for i in keep for v in geo["atoms"][i][1]] for geo in geos]
    if excited_path(text, len(frames)):
        energies, unread = None, None
    method, basis = level_of(text)
    charge, multiplicity = charge_of(text)
    masses = geos[-1]["masses"]
    version = VERSION.search(text)
    marks, unmarked = marks_of(text, geos, ended)
    dipole_state, charges_state = states_of(text)
    n = len(first)
    result = {
        "schema": SCHEMA,
        "program": "ORCA",
        "version": version.group(1) if version else None,
        "method": method,
        "basis": basis,
        "solvation": solvation_of(text),
        "charge": charge,
        "multiplicity": multiplicity,
        "atoms": atoms,
        **({"masses": [masses[i] for i in keep]} if masses and len(masses) == n else {}),
        "frames": frames,
        "energies": energies,
        "optimised": optimised_of(text),
        "results": (
            run_results(ended, why, scf)
            + path_said(text, len(frames))
            + energies_said(energies, unread)
            + xtb_said(text)
            + thermo_of(text)
            + of_state(dipole_of(text), dipole_state)
            + gradients_of(text, geos)
            + of_state(charges_of(text, n, ghosts), charges_state)
            + vibrations_of(text, n, ghosts)
        ),
    }
    orbitals = orbitals_of(text)
    if orbitals:
        result["orbitals"] = orbitals
    if marks is not None:
        result["converged"] = [False] * len(frames) if unmarked else marks[: len(frames)]
    if unmarked:
        result["converged"] = [False] * len(frames)
        result["unmarked"] = unmarked
    imaginary = imaginary_of(text)
    if imaginary is not None:
        result["imaginary"] = imaginary
    return result


# --- a file of several jobs -----------------------------------------------------------------------------------------

#: Each job's start in a file of several ($new_job, the manual's "Multiple Jobs in a Single Input"): "$$$$$$$$$$$$$$$$
#: JOB NUMBER  2 $$$$$$$$$$$$$$".
JOB_MARK = re.compile(r"\$+\s+JOB NUMBER\s+\d+\s+\$+")


def jobs(text):
    """The jobs a file holds one after another, each a text read as an output of its own: each begun by its JOB NUMBER,
    after what the output says before the first - its input, as echoed, cut down to that job's own, and what ORCA says
    of each job's basis set before it. The file itself, where it holds one."""
    lines = text.split("\n")
    marks = [k for k, line in enumerate(lines) if JOB_MARK.search(line)]
    if len(marks) < 2:
        return [text]
    head = lines[: marks[0]]
    echo = [k for k, line in enumerate(head) if re.match(r"\|\s*\d+>", line)]
    news = [k for k in echo if re.match(r"\|\s*\d+>\s*\$new_job\b", head[k], re.I)]
    parts = [k for k, line in enumerate(head) if line.startswith("----- Orbital basis set information")]
    title = next((k for k, line in enumerate(head) if line.strip() == "INPUT FILE"), len(head))
    title -= 1 if title and head[title - 1].startswith("=") else 0
    part_of = {}
    if len(parts) == len(marks):
        for i, (p, q) in enumerate(zip(parts, parts[1:] + [title])):
            part_of.update({k: i for k in range(p, q)})
    echoed_ = set(echo)
    out = []
    for j, (a, b) in enumerate(zip(marks, marks[1:] + [len(lines)])):
        mine = [
            line for k, line in enumerate(head)
            if part_of.get(k, j) == j and (k not in echoed_ or (sum(n <= k for n in news) == j and k not in news))
        ]
        out.append("\n".join(mine + lines[a:b]))
    return out


def job_kind(out):
    """What a job did, in words, by what was read of it."""
    results = out.get("results") or []
    vibrations = any(r.get("id") == "vibrations" for r in results)
    path = next((r["label"] for r in results if r.get("id") == "run.path" and r.get("value") != "not read"), None)
    if path:
        return path
    if out.get("optimised") is not None:
        return "Optimisation and frequencies" if vibrations else "Optimisation"
    return "Frequencies" if vibrations else "Single point"


def read_jobs(name, text):
    """What a file holds: as `read` reads it - or, where it holds several jobs, each job read as an output of its own,
    said by its place among them and what it did (`job`), all of them in order (`jobs`), the answer's own fields its last
    job's. A job nothing is read of is left out; its place is kept."""
    texts = jobs(text) if isinstance(text, str) and text else [text]
    if len(texts) < 2:
        return read(name, text)
    out = []
    for k, t in enumerate(texts):
        try:
            job = read(name, t, followed=k < len(texts) - 1)
        except ValueError:
            continue
        job["job"] = {"number": k + 1, "of": len(texts), "kind": job_kind(job)}
        out.append(job)
    if not out:
        raise ValueError(f"No molecule was found in {name}.")
    return {**out[-1], "jobs": out}


def finite(value):
    """`value` with each number that is not one - not a number, or infinite - as not known: JSON has neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


# --- a path's own files: a band's images and the point it refined, a reaction path's points ------------------------

#: The files ORCA writes a path's points to, after its input's base name (the job's "input.inp"): a reaction path's, end
#: to end; a band's images, and the point it refined.
BASE = "input"
IRC_FULL = f"{BASE}_IRC_Full_trj.xyz"
MEP = f"{BASE}_MEP_trj.xyz"
REFINED = f"{BASE}_NEB-TS_converged.xyz"
#: What each of a path's frames is, in Meno's words for a path (docs/PLUGINS.md, *Path search* and *Follow the
#: reaction path*): the frames result's id, and the words for the frame a step takes as the entry - the point a band
#: refined, the point a reaction path started from. The other frames are said in this reader's own words.
PATH_FRAME = "path.frame"
REFINED_SAID = "highest point refined"
STARTED_SAID = "the point it started from"
#: What ORCA says where a side stopped at its iteration cap, before its end.
IRC_CAPPED = "MAXIMUM NUMBER OF ITERATIONS REACHED"


def capped_said(capped):
    """Which sides of a reaction path stopped, in words: "The forward side", "The forward and backward sides" - or "A
    side" where ORCA said one did and which cannot be told."""
    if capped == ["one"]:
        return "A side"
    return f"The {' and '.join(capped)} {'side' if len(capped) == 1 else 'sides'}"


def irc_output(atoms, files, log, method=None, basis=None):
    """A reaction path's output, in Meno's form: its frames one end to the other, as ORCA wrote them, each with its
    energy and what it is, the point it started from marked as the run's converged geometry; its level, as given; and,
    where a side stopped at ORCA's iteration cap before its end, that said, the output then no converged one. `atoms`,
    the elements the run was given, in order."""
    n = len(atoms)
    frames = xyz_frames(files.get(IRC_FULL, ""), n)
    at = log.rfind("IRC PATH SUMMARY")
    rows = re.findall(r"^\s*(\d+)\s+-?\d+\.\d+.*?(<= TS)?\s*$", log[at:], re.M) if at >= 0 else []
    starts = [int(r) - 1 for r, ts in rows if ts]
    if not frames or len(rows) != len(frames) or len(starts) != 1:
        return {"why": "ORCA wrote no reaction path that can be read", "kind": "no-geometry"}
    k = starts[0]
    said = [f"backward, step {k - i} of {k}" if i < k else STARTED_SAID if i == k else f"forward, step {i - k} of {len(frames) - 1 - k}" for i in range(len(frames))]
    # (each side as ORCA ran it - the forward first, then the backward - and whether it stopped at its cap)
    fwd, back = log.find("FORWARD IRC"), log.find("BACKWARD IRC")
    sides = {"forward": log[fwd:back] if 0 <= fwd < back else "", "backward": log[back:at] if 0 <= back < at else ""}
    capped = [side for side, text in sides.items() if IRC_CAPPED in text]
    # (ORCA said a side stopped, but which cannot be told from where it said it: a side, never read as complete)
    if not capped and IRC_CAPPED in log:
        capped = ["one"]
    return {
        "schema": 1,
        "program": "ORCA",
        **({"method": method} if method else {}),
        **({"basis": basis} if basis else {}),
        "atoms": list(atoms),
        "frames": [places for places, _ in frames],
        "energies": [energy for _, energy in frames],
        "converged": [i == k for i in range(len(frames))],
        **({"optimised": False, "warnings": [{"kind": "other", "text": f"{capped_said(capped)} stopped at ORCA's iteration cap before reaching an end (\"{IRC_CAPPED}\"): {'its end is' if len(capped) == 1 else 'their ends are'} no minimum"}]} if capped else {}),
        "results": [{"id": PATH_FRAME, "on": "frames", "group": "Path", "label": "What it is", "values": said}],
    }



def xyz_frames(text, n):
    """An XYZ file's frames, each its places and the energy its comment line gives ("... E -4.412345678901"), as ORCA
    writes a path's images and the point it refines; none where it does not read as frames of `n` atoms."""
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].strip() != str(n) or i + 2 + n > len(lines):
            return None
        said = re.search(r"\bE\s+(-?\d+\.\d+)", lines[i + 1])
        places = []
        for line in lines[i + 2:i + 2 + n]:
            parts = line.split()
            if len(parts) < 4:
                return None
            places += [float(v) for v in parts[1:4]]
        if len(places) != 3 * n or not said:
            return None
        frames.append((places, float(said.group(1))))
        i += 2 + n
    return frames or None


def summary_order(log):
    """The rows of ORCA's PATH SUMMARY FOR NEB-TS, in its order: each image's number, and "TS" for the point it
    refined; none where the log has none."""
    at = log.rfind("PATH SUMMARY FOR NEB-TS")
    if at < 0:
        return None
    rows = []
    for line in log[at:].splitlines()[1:]:
        m = re.match(r"\s*(\d+|TS)\s+-?\d+\.\d+\s", line)
        if m:
            rows.append(m.group(1))
        elif rows and not line.strip():
            break
        elif rows and line.strip().startswith("-"):
            break
    return rows if "TS" in rows else None


def frequencies_of(log):
    """The vibrations ORCA listed last, in cm-1 as it prints them - an imaginary one negative - its zeros (the molecule's
    translations and rotations) left out; none where it worked out no frequencies (Freq off)."""
    at = log.rfind("VIBRATIONAL FREQUENCIES")
    if at < 0:
        return None
    found = []
    for line in log[at:].splitlines()[1:]:
        if "NORMAL MODES" in line:
            break
        m = re.match(r"\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
        if m and float(m.group(1)) != 0.0:
            found.append(float(m.group(1)))
    return found


#: The marker words of an image's SCF that did not converge ("... SCF has not converged ..."), after which ORCA carries
#: on with the band.
IMAGE_SCF_FAILED = re.compile(r"^.*\bSCF has not converged\b.*$", re.M)


def path_iterations(log):
    """How many iterations each part of a NEB-TS run took, as ORCA's STATISTICS counts them ("NEB ... 5", "TS
    optimization ... 4"), in its order; none where it gives none."""
    at = log.rfind("Number of iterations:")
    if at < 0:
        return []
    parts = []
    for line in log[at:].splitlines()[1:]:
        m = re.match(r"\s+(\S.*?)\s+\.\.\.\s+(\d+)\b", line)
        if not m:
            break
        parts.append((m.group(1), int(m.group(2))))
    return parts


def iterations_said(parts):
    """A run's iterations in words: "NEB 5 · TS optimization 4"."""
    return " \u00b7 ".join(f"{name} {n}" for name, n in parts)


def cycles_of(log):
    """How many cycles an optimisation took, as ORCA numbers them ("GEOMETRY OPTIMIZATION CYCLE   12"): the last; none
    where it numbers none."""
    found = re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", log)
    return int(found[-1]) if found else None


def path_output(atoms, files, log, method=None, basis=None):
    """A path search's output, in Meno's form: its frames in the order of ORCA's PATH SUMMARY - the images, and the
    point it refined where the summary has it - each with its energy and what it is, the refined one marked as the
    run's converged geometry; its level, as given; and that point's imaginary frequencies, where ORCA worked out
    frequencies (asked for among the keywords). `atoms`, the elements the run was given, in order."""
    n = len(atoms)
    images = xyz_frames(files.get(MEP, ""), n)
    refined = xyz_frames(files.get(REFINED, ""), n)
    order = summary_order(log)
    if not images or not refined or not order:
        return {"why": "ORCA wrote no path and point refined that can be read", "kind": "no-geometry"}
    numbered = [r for r in order if r != "TS"]
    if numbered != [str(k) for k in range(len(images))]:
        return {"why": "ORCA's path summary does not list the images it wrote", "kind": "failed"}
    frames, energies, said, converged = [], [], [], []
    for r in order:
        places, energy = refined[-1] if r == "TS" else images[int(r)]
        frames.append(places)
        energies.append(energy)
        said.append(REFINED_SAID if r == "TS" else f"NEB image {int(r) + 1} of {len(images)}")
        converged.append(r == "TS")
    imaginary = imaginary_of(log)
    parts = path_iterations(log)
    failed = IMAGE_SCF_FAILED.findall(log)
    vibrations = frequencies_of(log)
    # (the point refined's vibrations, where they were worked out: what Meno counts its imaginary ones from - none among them is none)
    listed = [{"id": "vibrations", "on": "list", "group": "Vibrations", "label": "Vibrations", "columns": [{"label": "Frequency", "quantity": "wavenumber"}],
               "rows": [{"cells": [f]} for f in vibrations]}] if vibrations is not None else []
    return {
        "schema": 1,
        "program": "ORCA",
        **({"method": method} if method else {}),
        **({"basis": basis} if basis else {}),
        "atoms": list(atoms),
        "frames": frames,
        "energies": energies,
        "converged": converged,
        **({"imaginary": imaginary} if imaginary is not None else {}),
        "results": [
            {"id": PATH_FRAME, "on": "frames", "group": "Path", "label": "What it is", "values": said},
            # (how far each part went, as ORCA counts it: its band converged, and the point refined - else it is not read)
            *([{"id": "path.iterations", "on": "molecule", "group": "Path", "label": "Converged, iterations", "value": iterations_said(parts)}] if parts else []),
            *listed,
        ],
        # (an image whose SCF did not converge, ORCA carrying on with the band: said, in its words - it moves nothing)
        **({"warnings": [{"kind": "other", "text": f"{len(failed)} of the band's SCF calculations did not converge, ORCA carrying on: {failed[0].strip()}"}]} if failed else {}),
    }


