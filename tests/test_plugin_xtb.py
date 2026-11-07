"""Tests for the xTB plugin's worker (plugins/xtb/worker.py).

The worker needs Python alone - xtb is run by Meno, not by it - so these run
anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_xtb.py

What xtb is asked, and what is read of what it writes, are checked against
xtb's own documentation (xtb-docs.readthedocs.io): the texts below are
short, written after its examples and xtb 6.7.1's printed forms, and no
output of a real run is kept. How an output is read is the reader's
(tests/test_reader_xtb.py); here, that a job is read as an output
opened by hand is, and what it adds of the files xtb wrote.
"""

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/xtb/worker.py"
MANIFEST = ROOT / "plugins/xtb/manifest.json"
# (the worker imports its reader from its own folder, as Python started on it has that folder on its path)
sys.path.insert(0, str(WORKER.parent))
spec = importlib.util.spec_from_file_location("plugin_xtb", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

ETHYNE = {
    "name": "ethyne",
    "atoms": [
        {"el": "H", "x": 0.0, "y": 0.0, "z": 0.0},
        {"el": "C", "x": 0.0, "y": 0.0, "z": 1.0},
        {"el": "C", "x": 0.0, "y": 0.0, "z": 2.0},
        {"el": "H", "x": 0.0, "y": 0.0, "z": 3.0},
    ],
    "bonds": [{"a1": 0, "a2": 1, "order": 1}, {"a1": 1, "a2": 2, "order": 3}, {"a1": 2, "a2": 3, "order": 1}],
    "charge": 0,
    "multiplicity": 1,
}

# a lone atom, as the ion of a salt is
SODIUM = {"name": "sodium", "atoms": [{"el": "Na", "x": 1.0, "y": 2.0, "z": 3.0}], "bonds": [], "charge": 1, "multiplicity": 1}

# (after "Geometry Optimization", Example 1: two frames, each comment line holding the energy and the gradient norm)
PATH = """4
 energy: -5.170000000000 gnorm: 0.100000000000 xtb: 6.7.1
H  0.00 0.00 0.00
C  0.00 0.00 1.00
C  0.00 0.00 2.00
H  0.00 0.00 3.00
4
 SCF done         -5.206771946579          0.000476954973
H           0.00000000000000   -0.00000000000000   -0.14662251809779
C          -0.00000000000000    0.00000000000000    0.90317992211836
C          -0.00000000000000    0.00000000000000    2.09682010367354
H          -0.00000000000000    0.00000000000000    3.14662249230588
"""

# (a run's log, as xtb 6.7.1 prints one, both its streams: its table of elements, its setup, its summary and its end)
def log_of(atoms="    1    1 H      1, 4\n    2    6 C      2, 3\n", body="", call="xtb input.xyz --gfn 2 -P 4", setup=True):
    box = (
        "          :  # electrons                        10          :\n"
        "          :  Hamiltonian                  GFN2-xTB          :\n"
        "          :  GBSA solvation                  false          :\n"
        "          :  net charge                          0          :\n"
        "          :  unpaired electrons                  0          :\n"
    ) if setup else ""
    return (
        "   * xtb version 6.7.1 (abc1234) compiled by 'someone' on 2026-01-01\n"
        f"   ID    Z sym.   atoms\n{atoms}\n"
        f"          program call               : {call}\n{box}{body}"
        "         :: total energy              -5.200000000000 Eh    ::\n"
        "         :: HOMO-LUMO gap              9.500000000000 eV    ::\n"
        "          | TOTAL ENERGY               -5.206771946579 Eh   |\n"
        " * finished run on 2026/01/01 at 12:00:01.000\n"
        "normal termination of xtb\n"
    )


CHARGES = """
     #   Z          covCN         q      C6AA      α(0)
     1   1 H        0.900     0.100     3.000     2.000
     2   6 C        1.900    -0.100    20.000     7.000
     3   6 C        1.900    -0.100    20.000     7.000
     4   1 H        0.900     0.100     3.000     2.000
"""

# (after "Calculation of Vibrational Frequencies": the Frequency Printout, the setup printed before the thermochemistry,
# the table - a line of xtb's other stream fallen between its rows, as a log of both streams has it - the THERMODYNAMIC
# summary, and the summary the run ends with: its total enthalpy, the total energy and H(T) of the row used)
FREQUENCIES = """
 projected vibrational frequencies (cm⁻¹)
eigval :       -0.00    -0.00    -0.00     0.00     0.00     0.00
eigval :     -120.50   829.07   900.00   950.00  1000.00  2000.00
 reduced masses (amu)
   1:  1.00   2:  1.00   3:  1.00   4:  1.00   5:  1.00   6:  1.00   7:  2.00   8:  3.00
   9:  1.00  10:  1.00  11:  1.00  12:  1.00
 IR intensities (km·mol⁻¹)
   1:  0.00   2:  0.00   3:  0.00   4:  0.00   5:  0.00   6:  0.00   7:  1.50   8:  2.50
   9:  0.00  10:  0.00  11:  0.00  12:  0.00
          :  symmetry                              c2v      :
          :  rotational number                       2      :
          |             Thermodynamic Functions             |
          :  scaling factor                  1.0000000      :
          :  rotor cutoff                   50.0000000 cm⁻¹ :
          :  imag. cutoff                  -20.0000000 cm⁻¹ :
    T/K    H(0)-H(T)+PV         H(T)/Eh          T*S/Eh         G(T)/Eh
------------------------------------------------------------------------
 150.00    0.251234E-02    0.551234E-01    0.131234E-01    0.421234E-01
Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG
 298.15    0.612345E-02    0.581234E-01    0.312345E-01    0.268889E-01 (used)
------------------------------------------------------------------------
        :: total free energy          -8.612345678901 Eh   ::
        :: zero point energy           0.051234567891 Eh   ::
        :: G(RRHO) contrib.           -0.025123456789 Eh   ::
         | TOTAL ENTHALPY             -8.581234567890 Eh   |
"""

SOLVATED = """
      * Solvation model:               ALPB
        Solvent                        thf
        Parameter file                 internal GFN2-xTB/ALPB
        Reference state                gsolv [1 M gas/solution]
"""


def ask(op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    if not said["ok"]:
        raise AssertionError(said["error"])
    return said["result"]


def defaults(kind):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if kind in s["kinds"])
    return {o["id"]: o["default"] for o in step["options"]}


def collect(step, log, files=None, entry=ETHYNE, options=None, ended="done"):
    return ask("collect", step=step, entries=[entry], options=options if options is not None else defaults(step), files=files or {}, log=log, ended=ended)


class Prepare(unittest.TestCase):
    def test_one_job_for_each_entry_with_its_input_and_command(self):
        jobs = ask("prepare", step="optimise", entries=[ETHYNE, {**ETHYNE, "charge": -1, "multiplicity": 2}], options=defaults("optimise"), cores=4)["jobs"]
        self.assertEqual(len(jobs), 2)
        first, second = jobs
        self.assertEqual(first["entries"], [0])
        self.assertEqual(first["program"], "xtb")
        self.assertEqual(first["args"], ["input.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "0", "--uhf", "0", "-P", "4"])
        self.assertEqual(second["args"][5:9], ["--chrg", "-1", "--uhf", "1"])
        files = {f["name"]: f["text"] for f in first["files"]}
        # Xmol: the count, a comment line, then an element and x, y, z a line - and nothing else: what is read is what xtb prints
        self.assertEqual(list(files), ["input.xyz"])
        lines = files["input.xyz"].splitlines()
        self.assertEqual(lines[0], "4")
        self.assertEqual(lines[2].split(), ["H", "0.0000000000", "0.0000000000", "0.0000000000"])
        self.assertEqual(first["reads"], ["xtbopt.log", "xtbopt.xyz"])

    def test_an_optimisation_holds_the_atoms_fixed_meno_asks_it_to_from_1_as_xcontrol_numbers_them(self):
        fixed = {**ETHYNE, "restraints": [{"kind": "fixed", "atoms": [3, 0]}, {"kind": "fixed", "atoms": [0]}]}
        [job] = ask("prepare", step="optimise", entries=[fixed], options={**defaults("optimise"), "hold": "fixed"})["jobs"]
        self.assertEqual(job["args"][-2:], ["--input", "fix.inp"])
        files = {f["name"]: f["text"] for f in job["files"]}
        self.assertEqual(files["fix.inp"], "$fix\n   atoms: 1,4\n$end\n$opt\n   engine=lbfgs\n$end\n")
        # (not asked to - Meno's default - or nothing fixed: as ever, the restraints riding along unheld)
        for options, entry in [(defaults("optimise"), fixed), ({**defaults("optimise"), "hold": "no"}, fixed), ({**defaults("optimise"), "hold": "fixed"}, ETHYNE)]:
            [job] = ask("prepare", step="optimise", entries=[entry], options=options)["jobs"]
            self.assertNotIn("--input", job["args"])
            self.assertEqual([f["name"] for f in job["files"]], ["input.xyz"])
        # (another step holds none)
        [energy] = ask("prepare", step="energy", entries=[fixed], options={"method": "gfn2", "solvent": "none", "hold": "fixed"})["jobs"]
        self.assertNotIn("--input", energy["args"])

    def test_an_optimisation_asked_to_hold_a_distance_refuses_that_entry_naming_it(self):
        both = {**ETHYNE, "restraints": [{"kind": "fixed", "atoms": [0]}, {"kind": "distance", "atoms": [1, 2], "value": 1.2, "strength": 500}]}
        out = ask("prepare", step="optimise", entries=[both, ETHYNE], options={**defaults("optimise"), "hold": "fixed"})
        self.assertEqual([j["entries"] for j in out["jobs"]], [[1]])
        self.assertEqual(out["refused"], [{"entry": 0, "why": "xTB: its restraints hold distance, which it holds none of: only atoms fixed.", "kind": "refused"}])
        wrong = {**ETHYNE, "restraints": [{"kind": "fixed", "atoms": [9]}]}
        with self.assertRaises(Exception):
            ask("prepare", step="optimise", entries=[wrong], options={**defaults("optimise"), "hold": "fixed"})

    def test_the_method_the_solvent_and_the_kind_of_run(self):
        args = ask("prepare", step="frequencies", entries=[ETHYNE], options={"method": "gfnff", "solvent": "water"})["jobs"][0]["args"]
        self.assertIn("--hess", args)
        self.assertIn("--gfnff", args)
        self.assertEqual(args[args.index("--alpb") + 1], "water")
        energy = ask("prepare", step="energy", entries=[ETHYNE], options={"method": "gfn1", "solvent": "none"})["jobs"][0]
        self.assertNotIn("--opt", energy["args"])
        self.assertNotIn("--hess", energy["args"])
        self.assertNotIn("--alpb", energy["args"])
        self.assertEqual(energy["args"][energy["args"].index("--gfn") + 1], "1")
        self.assertEqual(energy["reads"], [])

    def test_frequencies_of_an_isotope_refused_naming_it_other_runs_as_before(self):
        # (xtb 6.7.1 says it changed the mass, but its frequencies are the natural element's: refused, said in words)
        labelled = {**ETHYNE, "atoms": ETHYNE["atoms"][:3] + [{**ETHYNE["atoms"][3], "isotope": 2, "mass": 2.01410177812}]}
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "frequencies", "entries": [labelled], "options": defaults("frequencies")}))
        self.assertFalse(said["ok"])
        self.assertIn("2H would be given as H", said["error"])
        # (a geometry or an energy does not depend on the masses: run as before)
        for step in ("optimise", "energy"):
            self.assertEqual(len(ask("prepare", step=step, entries=[labelled], options=defaults(step))["jobs"]), 1)

    def test_refuses_an_entry_alone_and_prepares_the_others(self):
        # (the second labelled: its frequencies refused, saying why, as a refusal of the step is said; the first and third run)
        labelled = {**ETHYNE, "atoms": ETHYNE["atoms"][:3] + [{**ETHYNE["atoms"][3], "isotope": 2, "mass": 2.01410177812}]}
        half = {**ETHYNE, "charge": 0.5}
        said = ask("prepare", step="frequencies", entries=[ETHYNE, labelled, ETHYNE, half], options=defaults("frequencies"))
        self.assertEqual([j["entries"] for j in said["jobs"]], [[0], [2]])
        self.assertEqual([(r["entry"], r["kind"]) for r in said["refused"]], [(1, "refused"), (3, "refused")])
        self.assertTrue(said["refused"][0]["why"].startswith("xTB: xTB's frequencies here take no isotope's mass: 2H"))
        self.assertEqual(said["refused"][1]["why"], "xTB: The charge is not a whole number.")
        # (none refused: none said)
        self.assertNotIn("refused", ask("prepare", step="frequencies", entries=[ETHYNE], options=defaults("frequencies")))
        # (what is asked of them all, not known: the step refused as a whole, whatever the entries)
        whole = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "optimise", "entries": [ETHYNE, labelled], "options": {"level": "perfect"}}))
        self.assertFalse(whole["ok"])
        self.assertIn("no level is called 'perfect'", whole["error"])

    def test_refuses_what_it_does_not_know(self):
        for options in ({"method": "dft"}, {"solvent": "lava"}, {"level": "perfect"}):
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "optimise", "entries": [ETHYNE], "options": options}))
            self.assertFalse(said["ok"])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "conformers", "entries": [ETHYNE], "options": {}}))
        self.assertFalse(said["ok"])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [{**ETHYNE, "charge": 0.5}], "options": {}}))
        self.assertFalse(said["ok"])

    def test_a_lone_atom_is_not_optimised_but_has_its_energy_worked_out(self):
        job = ask("prepare", step="optimise", entries=[SODIUM], options=defaults("optimise"))["jobs"][0]
        self.assertNotIn("--opt", job["args"])
        self.assertEqual(job["args"][1:7], ["--gfn", "2", "--chrg", "1", "--uhf", "0"])
        self.assertEqual(job["reads"], [])

    def test_every_solvent_offered_is_one_it_takes(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for step in manifest["steps"]:
            solvents = next(o for o in step["options"] if o["id"] == "solvent")["choices"]
            self.assertEqual({c["value"] for c in solvents} - {"none"}, worker.SOLVENTS)


class Collect(unittest.TestCase):
    def test_a_job_is_read_as_an_output_opened_by_hand_is(self):
        # (one reading: what an opened output says of the run, a job's says the same - with the geometry the job began with)
        log = log_of(body=CHARGES)
        job = collect("energy", log)["outputs"][0]
        opened = ask("read", name="run.out", text=log)
        for key in ("program", "version", "method", "solvation", "atoms", "charge", "multiplicity"):
            self.assertEqual(job[key], opened[key], key)
        self.assertEqual(job["results"], opened["results"])
        self.assertEqual(opened["frames"], [])
        self.assertEqual(job["frames"], [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]])
        self.assertEqual(job["energies"], [-5.206771946579])

    def test_an_optimisation_s_path_and_whether_it_converged_as_xtb_says(self):
        files = {"xtbopt.log": PATH, ".xtboptok": ""}
        out = collect("optimise", log_of(body="   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***\n" + CHARGES), files)["outputs"][0]
        self.assertEqual(out["atoms"], ["H", "C", "C", "H"])
        self.assertEqual(len(out["frames"]), 2)
        self.assertAlmostEqual(out["frames"][1][2], -0.14662251809779)
        self.assertEqual(out["energies"], [-5.17, -5.206771946579])
        self.assertTrue(out["optimised"])
        self.assertEqual((out["program"], out["version"], out["method"]), ("xtb", "6.7.1", "GFN2-xTB"))
        # (its orbitals, where it printed them, as data Meno lists - and works the gap out of: none printed here)
        self.assertEqual({r["id"] for r in out["results"]}, {"charges"})
        # (xtb writes .xtboptok where it did not converge too: what it says decides)
        failed = collect("optimise", log_of(body="   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 2 ITERATIONS ***\n"), files)["outputs"][0]
        self.assertFalse(failed["optimised"])

    def test_a_lone_atom_optimised_is_where_it_was_with_its_energy(self):
        # (no electrons xtb counts: no HOMO, and the gap it gives - the LUMO's energy - left out)
        log = log_of(atoms="    1   11 Na     1\n").replace("# electrons                        10", "# electrons                         0")
        out = collect("optimise", log, entry=SODIUM)["outputs"][0]
        self.assertEqual(out["atoms"], ["Na"])
        self.assertEqual(out["frames"], [[1.0, 2.0, 3.0]])
        self.assertEqual(out["energies"], [-5.206771946579])
        self.assertTrue(out["optimised"])
        self.assertEqual(out["charge"], 1)
        self.assertNotIn("homo-lumo-gap", {r["id"] for r in out["results"]})

    def test_frequencies_and_thermochemistry(self):
        out = collect("frequencies", log_of(body=SOLVATED + FREQUENCIES, call="xtb input.xyz --hess --gfn 2 --alpb thf -P 4"), options={"method": "gfn2", "solvent": "thf"})["outputs"][0]
        self.assertEqual(out["energies"], [-5.206771946579])
        self.assertEqual(out["solvation"], "ALPB(thf), reference state gsolv [1 M gas/solution]")
        vibrations = next(r for r in out["results"] if r["id"] == "vibrations")
        # (the translations and rotations left out; an imaginary one kept, negative)
        self.assertEqual([row["cells"] for row in vibrations["rows"]][:2], [[1, -120.5, 2.0, 1.5], [2, 829.07, 3.0, 2.5]])
        thermo = {r["id"]: r["value"] for r in out["results"] if r["on"] == "molecule"}
        self.assertAlmostEqual(thermo["thermo.free-energy"], -8.612345678901)
        self.assertAlmostEqual(thermo["thermo.zpve"], 0.051234567891)
        self.assertAlmostEqual(thermo["thermo.g-correction"], -0.025123456789)
        self.assertAlmostEqual(thermo["thermo.temperature"], 298.15)
        self.assertAlmostEqual(thermo["thermo.h-correction"], 0.0581234)
        self.assertAlmostEqual(thermo["thermo.ts"], 0.0312345)
        self.assertAlmostEqual(thermo["thermo.enthalpy"], -8.581234567890)
        self.assertEqual(thermo["thermo.treatment"], "rotor cutoff 50 cm⁻¹, imag. cutoff -20 cm⁻¹, scaling factor 1")
        self.assertEqual((thermo["thermo.point-group"], thermo["thermo.symmetry-number"]), ("c2v", 2))
        # (none said found significant: none, judged so)
        self.assertEqual(out["imaginary"], [])

    def test_gfn_ff_s_frequencies_as_it_prints_them(self):
        # (GFN-FF writes no dump: its frequencies are its Frequency Printout's)
        log = log_of(body="          |                   G F N - F F                   |\n" + FREQUENCIES, call="xtb input.xyz --hess --gfnff -P 4", setup=False)
        out = collect("frequencies", log, options={"method": "gfnff", "solvent": "none"})["outputs"][0]
        self.assertEqual(out["method"], "GFN-FF")
        self.assertEqual(out["solvation"], "none")
        self.assertEqual(len(next(r for r in out["results"] if r["id"] == "vibrations")["rows"]), 6)
        # (its multiplicity, which it does not print, the one it was asked)
        self.assertEqual(out["multiplicity"], 1)

    def test_an_intensity_too_wide_for_its_field_is_the_g98_out_s(self):
        # (xtb's output prints an IR intensity of 1000 km/mol or more as asterisks; the g98.out it writes for MOLDEN, in
        # Gaussian's form, gives it: taken from there for the vibration of the same frequency, and from nowhere else)
        wide = FREQUENCIES.replace("   9:  0.00  10:  0.00", "   9:******  10:  0.00")
        g98 = (
            " Frequencies --  -120.5000               829.0700               900.0000\n"
            " IR Inten    --     1.5000                 2.5000              1012.3456\n"
        )
        rows = lambda out: next(r for r in out["results"] if r["id"] == "vibrations")["rows"]
        self.assertEqual(rows(collect("frequencies", log_of(body=wide), {"g98.out": g98})["outputs"][0])[2]["cells"], [3, 900.0, 1.0, 1012.3456])
        self.assertIsNone(rows(collect("frequencies", log_of(body=wide))["outputs"][0])[2]["cells"][-1])
        self.assertIn("g98.out", ask("prepare", step="frequencies", entries=[ETHYNE], options=defaults("frequencies"))["jobs"][0]["reads"])

    def test_the_imaginary_frequencies_xtb_judged_significant(self):
        found = FREQUENCIES + " found            1  significant imaginary frequency\n"
        self.assertEqual(collect("frequencies", log_of(body=found))["outputs"][0]["imaginary"], [-120.5])
        # (not a frequency job: not judged)
        self.assertNotIn("imaginary", collect("energy", log_of(body=found))["outputs"][0])

    def test_warns_in_xtb_s_words_never_fails(self):
        warned = "[WARNING] Runtime exception occurred\n-1- hessian_numhess: Hessian on incompletely optimized geometry!\n"
        out = collect("frequencies", log_of(body=FREQUENCIES + warned))["outputs"][0]
        self.assertEqual(out["warnings"], [{"kind": "hessian-not-stationary", "text": "hessian_numhess: Hessian on incompletely optimized geometry!"}])
        self.assertNotIn("warnings", collect("frequencies", log_of(body=FREQUENCIES))["outputs"][0])

    def test_says_the_solvation_it_ran_in_whatever_was_asked(self):
        # (whether that is as asked is Meno's to say: the manifest's solvation)
        for asked, body, solvation in (("none", SOLVATED, "ALPB(thf), reference state gsolv [1 M gas/solution]"), ("water", SOLVATED, "ALPB(thf), reference state gsolv [1 M gas/solution]"), ("thf", "", "none")):
            said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "energy", "entries": [ETHYNE], "options": {"solvent": asked}, "files": {}, "log": log_of(body=body), "ended": "done"}))
            self.assertTrue(said["ok"], asked)
            self.assertEqual(said["result"]["outputs"][0]["solvation"], solvation)

    def test_the_dipole_as_xtb_printed_it_with_the_point_it_took_it_about(self):
        # (where a charged molecule's is taken about is Meno's, not the reader's: as printed, with its origin)
        printout = "molecular dipole:\n                 x           y           z       tot (Debye)\n q only:        0.000      -0.000      -0.607\n   full:       -0.000      -0.000      -0.872       2.216\n"
        out = collect("energy", log_of(body=printout), entry={**ETHYNE, "charge": 1})["outputs"][0]
        dipole = next(r for r in out["results"] if r["id"] == "dipole")
        self.assertEqual((dipole["value"], dipole["label"], dipole["origin"]), (2.216, "Dipole moment", [0.0, 0.0, 0.0]))

    def test_another_molecule_written_back_is_refused(self):
        said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "energy", "entries": [ETHYNE], "options": {}, "files": {}, "log": log_of(atoms="    1    8 O      1-4\n"), "ended": "done"}))
        self.assertFalse(said["ok"])
        self.assertIn("another molecule", said["error"])
        for name in ("xtbopt.log", "xtbopt.xyz"):
            files = {"xtbopt.log": PATH, "xtbopt.xyz": PATH.split("4\n SCF")[0]}
            files[name] = files[name].replace("C  0.00 0.00 2.00", "N  0.00 0.00 2.00", 1)
            said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "optimise", "entries": [ETHYNE], "options": {}, "files": files, "log": log_of(), "ended": "done"}))
            self.assertFalse(said["ok"], name)
            self.assertIn("another molecule", said["error"])

    def test_says_why_a_job_failed(self):
        log = "\n########\n[ERROR] Program stopped due to fatal error\n-2- Some atoms in the start geometry are *very* close\n-1- Found *very* short distance of  0.000E+00 for O1-H2\n########\n"
        said = collect("optimise", log, ended="failed")
        self.assertEqual(said["why"], "Some atoms in the start geometry are *very* close: Found *very* short distance of  0.000E+00 for O1-H2")
        self.assertEqual(collect("energy", "", ended="stopped")["why"], "xtb said nothing")
        # (a process that ended though xtb says it stopped on an error: no output, and why)
        stopped = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "energy", "entries": [ETHYNE], "options": {}, "files": {}, "log": log_of() + log, "ended": "done"}))
        self.assertFalse(stopped["ok"])
        self.assertIn("Some atoms in the start geometry are *very* close", stopped["error"])

    def test_an_energy_xtb_did_not_print_is_none(self):
        said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "energy", "entries": [ETHYNE], "options": {}, "files": {}, "log": "", "ended": "done"}))
        self.assertFalse(said["ok"])


class Read(unittest.TestCase):
    def test_an_output_opened_by_hand(self):
        out = ask("read", name="C:\\runs\\run.out", text=log_of(body=CHARGES))
        self.assertEqual((out["program"], out["method"], out["atoms"]), ("xtb", "GFN2-xTB", ["H", "C", "C", "H"]))
        said = worker.answer(json.dumps({"id": 1, "op": "read", "name": "run.out", "text": "nothing of xtb's"}))
        self.assertFalse(said["ok"])
        self.assertIn("No molecule was found in run.out", said["error"])

    def test_the_manifest_declares_the_kind_it_reads_written_by_its_own_program(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        kind = next(k for k in manifest["kinds"] if k["id"] == "xtb")
        self.assertTrue(all(k["writtenBy"] == "xtb" for k in manifest["kinds"]))
        self.assertEqual(manifest["reads"], ["xtb", "xtb-g98"])
        self.assertIn(kind["writtenBy"], {p for s in manifest["steps"] for p in s["programs"]})
        self.assertTrue((MANIFEST.parent / kind["grammar"]["file"]).exists())


if __name__ == "__main__":
    unittest.main()
