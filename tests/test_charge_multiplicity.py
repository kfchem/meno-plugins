"""Each entry's charge and spin multiplicity through the step plugins' workers:
a property, over seeded random sets of small molecules (docs/WORKFLOWS.md,
*What changes in the contract*).

For any set of entries - mixed charges and spins - each worker's `prepare`
writes every entry's charge and multiplicity into its program's input, in
that program's own syntax as the worker implements it, and never another
entry's; where its `collect` builds output entries, each keeps its own. A
combination its electrons cannot have is refused, or written as it was
given - never changed.

The xTB, CREST, ORCA and Gaussian workers need Python alone, so these run
anywhere:

    python3 -m unittest tests/test_charge_multiplicity.py

RDKit's needs RDKit: its tests are skipped without it, and run in the
plugin's own environment (its venv's python, as above).

What each program is asked is checked against the syntax its worker's
docstring cites: xtb's and CREST's `--chrg` and `--uhf` (unpaired
electrons, the multiplicity less one), ORCA's `* xyz charge multiplicity`,
Gaussian's charge and multiplicity on the molecule specification's first
line. The programs' outputs below are short, written as those workers'
own tests write them; no output of a real run is kept.
"""

import importlib.util
import json
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"


def load(plugin):
    # (a worker imports the modules beside it - its reader - from its own folder, as Python started on it has that folder on its
    #  path; each plugin's own, so that one's is never another's taken from those already imported)
    folder = str(PLUGINS / plugin)
    sys.path.insert(0, folder)
    sys.modules.pop("reader", None)
    try:
        spec = importlib.util.spec_from_file_location(f"plugin_{plugin}", PLUGINS / plugin / "worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        sys.modules.pop("reader", None)
    return module


xtb = load("xtb")
crest = load("crest")
orca = load("orca")
gaussian = load("gaussian")
try:
    rdkit = load("rdkit")
except ImportError:
    rdkit = None


def defaults(plugin, kind):
    manifest = json.loads((PLUGINS / plugin / "manifest.json").read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if kind in s["kinds"])
    return {o["id"]: o["default"] for o in step.get("options", [])}


def ask(worker, op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    if not said["ok"]:
        raise AssertionError(said["error"])
    return said["result"]


def H(x, y, z):
    return {"el": "H", "x": x, "y": y, "z": z}


# Small molecules, as Meno gives an entry to a plugin: its atoms - each its
# element, place in angstroms, charge and radical - its bonds, and its charge
# and multiplicity as Export reads them. Some are of the same atoms in
# another charge or spin (methyl's cation, radical and anion; methylene's
# singlet and triplet; dioxygen and superoxide).
SPECIES = [
    ("water", 0, 1, [{"el": "O", "x": 0, "y": 0, "z": 0.119}, H(0, 0.763, -0.477), H(0, -0.763, -0.477)]),
    ("methanol", 0, 1, [{"el": "C", "x": -0.047, "y": 0.662, "z": 0}, {"el": "O", "x": -0.047, "y": -0.757, "z": 0}, H(-1.086, 0.975, 0), H(0.438, 1.083, 0.888), H(0.438, 1.083, -0.888), H(0.869, -1.054, 0)]),
    ("ammonium", 1, 1, [{"el": "N", "x": 0, "y": 0, "z": 0, "charge": 1}, H(0.59, 0.59, 0.59), H(-0.59, -0.59, 0.59), H(-0.59, 0.59, -0.59), H(0.59, -0.59, -0.59)]),
    ("hydroxide", -1, 1, [{"el": "O", "x": 0, "y": 0, "z": 0, "charge": -1}, H(0, 0, 0.97)]),
    ("methyl", 0, 2, [{"el": "C", "x": 0, "y": 0, "z": 0, "radical": "doublet"}, H(1.08, 0, 0), H(-0.54, 0.935, 0), H(-0.54, -0.935, 0)]),
    ("methyl cation", 1, 1, [{"el": "C", "x": 0, "y": 0, "z": 0, "charge": 1}, H(1.09, 0, 0), H(-0.545, 0.944, 0), H(-0.545, -0.944, 0)]),
    ("methyl anion", -1, 1, [{"el": "C", "x": 0, "y": 0, "z": 0, "charge": -1}, H(1.03, 0, -0.36), H(-0.515, 0.892, -0.36), H(-0.515, -0.892, -0.36)]),
    ("dioxygen", 0, 3, [{"el": "O", "x": 0, "y": 0, "z": 0, "radical": "doublet"}, {"el": "O", "x": 0, "y": 0, "z": 1.21, "radical": "doublet"}]),
    ("superoxide", -1, 2, [{"el": "O", "x": 0, "y": 0, "z": 0, "charge": -1}, {"el": "O", "x": 0, "y": 0, "z": 1.33, "radical": "doublet"}]),
    ("hydrazinium", 2, 1, [{"el": "N", "x": 0, "y": 0, "z": 0, "charge": 1}, {"el": "N", "x": 0, "y": 0, "z": 1.45, "charge": 1}, H(0.97, 0, -0.34), H(-0.48, 0.84, -0.34), H(-0.48, -0.84, -0.34), H(-0.97, 0, 1.79), H(0.48, 0.84, 1.79), H(0.48, -0.84, 1.79)]),
    ("singlet methylene", 0, 1, [{"el": "C", "x": 0, "y": 0, "z": 0, "radical": "singlet"}, H(0.86, 0, 0.62), H(-0.86, 0, 0.62)]),
    ("triplet methylene", 0, 3, [{"el": "C", "x": 0, "y": 0, "z": 0, "radical": "triplet"}, H(1.02, 0, 0.28), H(-1.02, 0, 0.28)]),
    ("aminium", 1, 2, [{"el": "N", "x": 0, "y": 0, "z": 0, "charge": 1, "radical": "doublet"}, H(1.02, 0, 0), H(-0.51, 0.883, 0), H(-0.51, -0.883, 0)]),
]
# (the second atom bonded to the first, and so on - but methanol's hydroxyl H and hydrazinium's second NH3 on their own atoms)
BONDED_TO = {"methanol": [None, 0, 0, 0, 0, 1], "hydrazinium": [None, 0, 0, 0, 0, 1, 1, 1]}


def entry(rng, k, name, charge, multiplicity, atoms):
    """An entry of a species, laid 4 angstroms along x for each before it and shaken a little: its place tells it apart."""
    bonded = BONDED_TO.get(name, [None] + [0] * (len(atoms) - 1))
    return {
        "name": f"{name} {k + 1}",
        "atoms": [{**a, "x": a["x"] + 4 * k + 0.3 * (rng.random() - 0.5), "y": a["y"] + 0.3 * (rng.random() - 0.5), "z": a["z"] + 0.3 * (rng.random() - 0.5)} for a in atoms],
        "bonds": [{"a1": b, "a2": i, "order": 1} for i, b in enumerate(bonded) if b is not None],
        "charge": charge,
        "multiplicity": multiplicity,
    }


def entries_of(seed, most=5):
    """One to `most` entries of species picked at random: mixed charges and spins."""
    rng = random.Random(seed)
    return [entry(rng, k, *rng.choice(SPECIES)) for k in range(rng.randint(1, most))]


# (the sets that stress it most: the same atoms in another charge or spin, side by side)
HARD = [["singlet methylene", "triplet methylene"], ["methyl", "methyl cation", "methyl anion"], ["dioxygen", "superoxide"], ["ammonium", "aminium", "hydrazinium", "hydroxide"]]


def sets():
    """The sets every worker is given: seeded random ones, and the hard ones."""
    out = [entries_of(seed) for seed in range(1, 41)]
    for names in HARD:
        rng = random.Random(len(names))
        out.append([entry(rng, k, *next(s for s in SPECIES if s[0] == n)) for k, n in enumerate(names)])
    return out


def state(e):
    return (e["charge"], e["multiplicity"])


def xmol(e, comment):
    """An entry's geometry as an Xmol frame, written as the workers' tests write one."""
    return "\n".join([str(len(e["atoms"])), comment] + [f"{a['el']}  {a['x']:.6f} {a['y']:.6f} {a['z']:.6f}" for a in e["atoms"]]) + "\n"


def xtb_log(e):
    """An xtb run's log of an entry, as xtb 6.7.1 prints its parts the worker reads: the table of its atoms (each
    on a row of its own), its total energy, and its end - the charge and unpaired electrons of its setup left out,
    so that what is read back is the entry's."""
    rows = "".join(f"    {k + 1}    {xtb.reader.ELEMENTS.index(a['el'])} {a['el']:<2}     {k + 1}\n" for k, a in enumerate(e["atoms"]))
    return f"   ID    Z sym.   atoms\n{rows}\n          | TOTAL ENERGY               -5.070000000000 Eh   |\n * finished run on 2026/01/01 at 12:00:01.000\n"


def flag(args, name):
    """The value given after a flag on a command line, where it is there once."""
    at = [i for i, a in enumerate(args) if a == name]
    if len(at) != 1 or at[0] + 1 >= len(args):
        raise AssertionError(f"{name} is not given once in {args}")
    return args[at[0] + 1]


def written(job):
    """What a job's files say of its charge and spin besides its command line: xtb's and CREST's .CHRG and .UHF, where written."""
    files = {f["name"]: f["text"] for f in job["files"]}
    return {k: files[k].strip() for k in (".CHRG", ".UHF") if k in files}


class Xtb(unittest.TestCase):
    """`--chrg` the charge, `--uhf` the unpaired electrons (xtb-docs, "Commandline Usage"); one job for each entry."""

    def test_each_job_is_told_its_own_entry_s_charge_and_unpaired_electrons(self):
        for kind in ("optimise", "energy", "frequencies"):
            for given in sets():
                jobs = ask(xtb, "prepare", step=kind, entries=given, options=defaults("xtb", kind), cores=2)["jobs"]
                self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
                for job in jobs:
                    (i,) = job["entries"]
                    charge, multiplicity = state(given[i])
                    self.assertEqual((flag(job["args"], "--chrg"), flag(job["args"], "--uhf")), (str(charge), str(multiplicity - 1)), given[i]["name"])
                    self.assertEqual(written(job), {}, "a .CHRG or .UHF file besides the command line")

    def test_what_it_reads_back_keeps_each_entry_s_own(self):
        for given in sets():
            for kind in ("optimise", "energy", "frequencies"):
                for e in given:
                    files = {"xtbopt.log": xmol(e, " energy: -5.06 gnorm: 0.1") + xmol(e, " energy: -5.07 gnorm: 0.0001")} if kind == "optimise" else {}
                    out = ask(xtb, "collect", step=kind, entries=[e], options=defaults("xtb", kind), files=files, log=xtb_log(e), ended="done")["outputs"]
                    self.assertEqual([state(o) for o in out], [state(e)], e["name"])


class Crest(unittest.TestCase):
    """`--chrg` and `--uhf`, as xtb's (crest-docs, "Command Line Keywords"): a search one job for each entry; an
    ensemble optimised one job for each molecule - the same atoms, charge and multiplicity."""

    def test_a_search_is_told_its_own_entry_s_charge_and_unpaired_electrons(self):
        for given in sets():
            jobs = ask(crest, "prepare", step="conformers", entries=given, options=defaults("crest", "conformers"), cores=2)["jobs"]
            self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
            for job in jobs:
                (i,) = job["entries"]
                charge, multiplicity = state(given[i])
                self.assertEqual((flag(job["args"], "--chrg"), flag(job["args"], "--uhf")), (str(charge), str(multiplicity - 1)), given[i]["name"])
                self.assertEqual(written(job), {})

    def test_an_ensemble_holds_entries_of_one_charge_and_multiplicity_and_is_told_them(self):
        for given in sets():
            jobs = ask(crest, "prepare", step="optimise", entries=given, options=defaults("crest", "optimise"), cores=2)["jobs"]
            self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
            for job in jobs:
                states = {state(given[i]) for i in job["entries"]}
                self.assertEqual(len(states), 1, [given[i]["name"] for i in job["entries"]])
                ((charge, multiplicity),) = states
                self.assertEqual((flag(job["args"], "--chrg"), flag(job["args"], "--uhf")), (str(charge), str(multiplicity - 1)))
                self.assertEqual(written(job), {})

    def test_what_it_reads_back_keeps_each_entry_s_own(self):
        for given in sets():
            for e in given:
                conformers = xmol(e, "  -11.39270131") + xmol(e, "  -11.39100000")
                (out,) = ask(crest, "collect", step="conformers", entries=[e], options={}, files={"crest_conformers.xyz": conformers}, log="", ended="done")["outputs"]
                self.assertEqual(state(out), state(e), e["name"])
            for job in ask(crest, "prepare", step="optimise", entries=given, options=defaults("crest", "optimise"))["jobs"]:
                mine = [given[i] for i in job["entries"]]
                ensemble = "".join(xmol(e, f"  {-11.39 - k / 1000:.8f}") for k, e in enumerate(mine))
                out = ask(crest, "collect", step="optimise", entries=mine, options={}, files={"crest_ensemble.xyz": ensemble}, log="", ended="done")["outputs"]
                self.assertEqual([state(o) for o in out], [state(e) for e in mine])


class Orca(unittest.TestCase):
    """"* xyz charge multiplicity" before the coordinates (ORCA 6.1 Manual, "General Structure of the Input File"); one job for each entry."""

    def test_each_input_holds_its_own_entry_s_charge_and_multiplicity(self):
        for kind in ("optimise", "energy", "frequencies"):
            for given in sets():
                jobs = ask(orca, "prepare", step=kind, entries=given, options=defaults("orca", kind), cores=1)["jobs"]
                self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
                for job in jobs:
                    (i,) = job["entries"]
                    lines = job["files"][0]["text"].splitlines()
                    starts = [line for line in lines if line.startswith("*") and line != "*"]
                    self.assertEqual(starts, ["* xyz {} {}".format(*state(given[i]))], given[i]["name"])

    def test_what_it_reads_back_is_for_meno_s_readers_and_says_none_of_its_own(self):
        # (ORCA prints the charge and multiplicity it was given; Meno's readers read them from its output)
        for given in sets():
            read = ask(orca, "collect", step="energy", entries=given, options={}, files={}, log="ORCA output\n****ORCA TERMINATED NORMALLY****\n", ended="done")["read"]
            self.assertEqual(len(read), len(given))
            self.assertTrue(all("charge" not in r and "multiplicity" not in r for r in read))


class Gaussian(unittest.TestCase):
    """The charge and spin multiplicity on the molecule specification's first line, after the route and title
    sections, each ended by a blank line (gaussian.com, "Molecule Specifications"); one job for each entry."""

    def test_each_input_holds_its_own_entry_s_charge_and_multiplicity(self):
        for kind in ("optimise", "energy", "frequencies"):
            for given in sets():
                jobs = ask(gaussian, "prepare", step=kind, entries=given, options=defaults("gaussian", kind), cores=1)["jobs"]
                self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
                for job in jobs:
                    (i,) = job["entries"]
                    sections = job["files"][0]["text"].split("\n\n")
                    # (Link 0 and the route; the title; the molecule)
                    self.assertTrue(sections[0].splitlines()[-1].startswith("#"))
                    self.assertEqual(sections[2].splitlines()[0], "{} {}".format(*state(given[i])), given[i]["name"])
                    self.assertEqual(len(sections[2].splitlines()), 1 + len(given[i]["atoms"]))

    def test_what_it_reads_back_is_for_meno_s_readers_and_says_none_of_its_own(self):
        for given in sets():
            read = ask(gaussian, "collect", step="energy", entries=given, options={}, files={}, log=" Normal termination of Gaussian 16", ended="done")["read"]
            self.assertEqual(len(read), len(given))
            self.assertTrue(all("charge" not in r and "multiplicity" not in r for r in read))


# A charge and multiplicity the entry's electrons cannot have: CH3's 9 electrons a singlet, water's 10 a
# doublet, one H's single electron a quartet (a triplet radical drawn on it, as Meno reads one), H2 with
# fewer electrons than none, and a multiplicity below one.
METHYL = next(s for s in SPECIES if s[0] == "methyl")[3]
WATER = SPECIES[0][3]
IMPOSSIBLE = [
    ("methyl, a singlet", METHYL, 0, 1),
    ("water, a doublet", WATER, 0, 2),
    ("a hydrogen atom, a quartet", [H(0, 0, 0)], 0, 4),
    ("H2 with a charge of 3", [H(0, 0, 0), H(0, 0, 0.74)], 3, 2),
    ("water, a multiplicity of nought", WATER, 0, 0),
]


class Impossible(unittest.TestCase):
    """What cannot be is refused, or passed on as it was given: never changed."""

    def told(self, worker, job):
        if worker in (xtb, crest):
            return (int(flag(job["args"], "--chrg")), int(flag(job["args"], "--uhf")) + 1)
        text = job["files"][0]["text"]
        if worker is orca:
            line = next(line for line in text.splitlines() if line.startswith("* xyz "))
            return tuple(int(v) for v in line.split()[2:4])
        return tuple(int(v) for v in text.split("\n\n")[2].splitlines()[0].split())

    def test_refused_or_written_as_given(self):
        for worker, plugin, kind in ((xtb, "xtb", "energy"), (crest, "crest", "conformers"), (crest, "crest", "optimise"), (orca, "orca", "energy"), (gaussian, "gaussian", "energy")):
            for what, atoms, charge, multiplicity in IMPOSSIBLE:
                e = {"name": what, "atoms": atoms, "bonds": [], "charge": charge, "multiplicity": multiplicity}
                said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": kind, "entries": [e], "options": defaults(plugin, kind)}))
                if said["ok"]:
                    (job,) = said["result"]["jobs"]
                    self.assertEqual(self.told(worker, job), (charge, multiplicity), f"{plugin} {kind}: {what}")


@unittest.skipIf(rdkit is None, "RDKit is not here: run in the RDKit plugin's environment")
class Rdkit(unittest.TestCase):
    """A conformer search's job is given each atom's charge and radical as they came (molecule.json), and read back
    gives no charge or multiplicity of its own: Meno keeps the entry's."""

    def test_each_job_is_given_its_own_entry_s_charges_and_radicals(self):
        for given in sets():
            jobs = ask(rdkit, "prepare", step="conformers", entries=given, options=defaults("rdkit", "conformers"))["jobs"]
            self.assertEqual(sorted(i for j in jobs for i in j["entries"]), list(range(len(given))))
            for job in jobs:
                (i,) = job["entries"]
                molecule = json.loads(job["files"][0]["text"])
                marks = lambda atoms: [(a["el"], a.get("charge") or 0, a.get("radical")) for a in atoms]  # noqa: E731
                self.assertEqual(marks(molecule["atoms"]), marks(given[i]["atoms"]), given[i]["name"])
                # (and the molecule RDKit makes of them has the entry's charge)
                self.assertEqual(rdkit.Chem.GetFormalCharge(rdkit.mol_of(molecule)), given[i]["charge"], given[i]["name"])

    def test_what_it_reads_back_says_no_charge_or_multiplicity(self):
        e = entries_of(7, 1)[0]
        made = json.dumps({"version": "test", "atoms": [{"el": a["el"]} for a in e["atoms"]], "frames": [[v for a in e["atoms"] for v in (a["x"], a["y"], a["z"])]], "energies": [-0.01], "field": "MMFF94"})
        (out,) = ask(rdkit, "collect", step="conformers", entries=[e], options={}, files={"conformers.json": made}, log="", ended="done")["outputs"]
        self.assertNotIn("charge", out)
        self.assertNotIn("multiplicity", out)


if __name__ == "__main__":
    unittest.main()
