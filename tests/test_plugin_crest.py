"""Tests for the CREST plugin's worker (plugins/crest/worker.py).

The worker needs Python alone - CREST is run by Meno, not by it - so these
run anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_crest.py

What CREST is asked, and what is read of what it writes, are checked
against CREST's own documentation (crest-lab.github.io/crest-docs): the
texts below are short, written after it, and no output of a real run is
kept.
"""

import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/crest/worker.py"
MANIFEST = ROOT / "plugins/crest/manifest.json"
spec = importlib.util.spec_from_file_location("plugin_crest", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

METHANOL = {
    "name": "methanol",
    "atoms": [
        {"el": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"el": "O", "x": 1.4, "y": 0.0, "z": 0.0},
        {"el": "H", "x": 1.7, "y": 0.9, "z": 0.0},
        {"el": "H", "x": -0.4, "y": 1.0, "z": 0.0},
        {"el": "H", "x": -0.4, "y": -0.5, "z": 0.9},
        {"el": "H", "x": -0.4, "y": -0.5, "z": -0.9},
    ],
    "bonds": [{"a1": 0, "a2": 1, "order": 1}, {"a1": 1, "a2": 2, "order": 1}, {"a1": 0, "a2": 3, "order": 1}, {"a1": 0, "a2": 4, "order": 1}, {"a1": 0, "a2": 5, "order": 1}],
    "charge": 0,
    "multiplicity": 1,
}

# (after "Coordinate Files", ensemble files: Xmol frames in angstroms, each comment line an energy in hartrees, lowest first)
# (after "Command Line Keywords": CREST echoing the options it was given as it starts)
ECHO_THF = """ Command line input:
 > crest input.xyz --gfn2 --chrg 0 --uhf 0 --alpb thf --ewin 6 --T 4

  -gfn2 : Use of GFN2-xTB requested.
  --alpb thf : implicit solvation
  --ewin 6
"""

ENSEMBLE = """6
       -11.39270131
C         -0.0100000000        0.0000000000        0.0000000000
O          1.4000000000        0.0000000000        0.0000000000
H          1.7000000000        0.9000000000        0.0000000000
H         -0.4000000000        1.0000000000        0.0000000000
H         -0.4000000000       -0.5000000000        0.9000000000
H         -0.4000000000       -0.5000000000       -0.9000000000
6
       -11.39100000
C          0.0000000000        0.0000000000        0.0000000000
O          1.4000000000        0.0000000000        0.0000000000
H          1.7000000000       -0.9000000000        0.0000000000
H         -0.4000000000        1.0000000000        0.0000000000
H         -0.4000000000       -0.5000000000        0.9000000000
H         -0.4000000000       -0.5000000000       -0.9000000000
"""


def ask(op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    if not said["ok"]:
        raise AssertionError(said["error"])
    return said["result"]


def defaults():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if "conformers" in s["kinds"])
    return {o["id"]: o["default"] for o in step["options"]}


class Prepare(unittest.TestCase):
    def test_one_job_for_each_entry_with_its_input_and_command(self):
        jobs = ask("prepare", step="conformers", entries=[METHANOL, {**METHANOL, "charge": 1, "multiplicity": 2}], options=defaults(), cores=4)["jobs"]
        self.assertEqual(len(jobs), 2)
        first, second = jobs
        self.assertEqual(first["entries"], [0])
        self.assertEqual(first["program"], "crest")
        self.assertEqual(first["args"], ["input.xyz", "--gfn2", "--chrg", "0", "--uhf", "0", "--ewin", "6", "--T", "4"])
        self.assertEqual(second["args"][2:6], ["--chrg", "1", "--uhf", "1"])
        files = {f["name"]: f["text"] for f in first["files"]}
        # Xmol: the count, a comment line, then an element and x, y, z a line
        lines = files["input.xyz"].splitlines()
        self.assertEqual(lines[0], "6")
        self.assertEqual(lines[1], "methanol")
        self.assertEqual(lines[3].split(), ["O", "1.4000000000", "0.0000000000", "0.0000000000"])
        self.assertEqual(first["reads"], ["crest_conformers.xyz"])

    def test_the_method_the_solvent_the_search_and_the_window(self):
        args = ask("prepare", step="conformers", entries=[METHANOL], options={"method": "gfn2//gfnff", "solvent": "water", "search": "squick", "ewin": 3.5})["jobs"][0]["args"]
        self.assertIn("--gfn2//gfnff", args)
        self.assertEqual(args[args.index("--alpb") + 1], "water")
        self.assertIn("--squick", args)
        self.assertEqual(args[args.index("--ewin") + 1], "3.5")
        self.assertNotIn("--T", args)

    def test_refuses_what_it_does_not_know(self):
        for options in ({"method": "dft"}, {"solvent": "lava"}, {"search": "slow"}, {"ewin": 0}, {"ewin": "6"}):
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "conformers", "entries": [METHANOL], "options": options}))
            self.assertFalse(said["ok"], options)
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "frequencies", "entries": [METHANOL], "options": {}}))
        self.assertFalse(said["ok"])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "optimise", "entries": [METHANOL], "options": {"method": "gfn2//gfnff"}}))
        self.assertFalse(said["ok"])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "conformers", "entries": [{**METHANOL, "charge": 0.5}], "options": {}}))
        self.assertFalse(said["ok"])

    def test_an_ensemble_optimised_one_job_for_each_molecule(self):
        # ("Ensemble Optimization": crest --mdopt <file>, each structure optimised, written to crest_ensemble.xyz in the same order)
        water = {"name": "water", "atoms": [{"el": "O", "x": 0, "y": 0, "z": 0}, {"el": "H", "x": 0.96, "y": 0, "z": 0}, {"el": "H", "x": -0.24, "y": 0.93, "z": 0}], "bonds": [], "charge": 0, "multiplicity": 1}
        jobs = ask("prepare", step="optimise", entries=[METHANOL, water, METHANOL], options={"method": "gfn1", "solvent": "water"}, cores=2)["jobs"]
        self.assertEqual([j["entries"] for j in jobs], [[0, 2], [1]])
        first = jobs[0]
        self.assertEqual(first["args"], ["--mdopt", "ensemble.xyz", "--gfn1", "--chrg", "0", "--uhf", "0", "--alpb", "water", "--T", "2"])
        self.assertEqual(first["reads"], ["crest_ensemble.xyz"])
        ensemble = {f["name"]: f["text"] for f in first["files"]}["ensemble.xyz"].splitlines()
        # (two Xmol frames, one after the other)
        self.assertEqual([ensemble[0], ensemble[8]], ["6", "6"])

    def test_an_ensemble_holds_one_constitution_though_another_has_its_atoms_in_the_same_order(self):
        # (the same atoms in the same order, one H bonded to O in one and to C in the other: two molecules, two ensembles)
        other = {**METHANOL, "bonds": [b if b["a2"] != 5 else {**b, "a1": 1} for b in METHANOL["bonds"]]}
        self.assertNotEqual(other["bonds"], METHANOL["bonds"])
        jobs = ask("prepare", step="optimise", entries=[METHANOL, other, METHANOL], options={"method": "gfn2"})["jobs"]
        self.assertEqual([j["entries"] for j in jobs], [[0, 2], [1]])
        # (the bonds of one molecule, written in another order or either way round, or with other orders - a ring drawn
        #  as another Kekule structure - are the same molecule's)
        turned = {**METHANOL, "bonds": [{**b, "a1": b["a2"], "a2": b["a1"]} for b in reversed(METHANOL["bonds"])]}
        kekule = {**METHANOL, "bonds": [{**b, "order": 2} if i == 0 else b for i, b in enumerate(METHANOL["bonds"])]}
        jobs = ask("prepare", step="optimise", entries=[METHANOL, turned, kekule], options={"method": "gfn2"})["jobs"]
        self.assertEqual([j["entries"] for j in jobs], [[0, 1, 2]])

    def test_any_energy_window_above_nought(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ewin = next(o for s in manifest["steps"] for o in s["options"] if o["id"] == "ewin")
        self.assertNotIn("max", ewin)
        self.assertIn("--ewin", ask("prepare", step="conformers", entries=[METHANOL], options={**defaults(), "ewin": 150})["jobs"][0]["args"])

    def test_the_programs_its_jobs_run_are_those_its_manifest_names(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for step in manifest["steps"]:
            # (CREST alone: the xtb in its environment is CREST's, run by CREST)
            self.assertEqual(step["programs"], ["crest"])

    def test_it_says_its_program_takes_no_lone_atom_and_runs_none(self):
        # (CREST stops on one atom: Meno passes one through as it came, never sending it - nothing here runs another program for it)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual([s.get("noLoneAtom") for s in manifest["steps"]], [True, True])
        for name in ("lone", "lone_job", "lone_of", "XTB"):
            self.assertFalse(hasattr(worker, name), name)

    def test_every_option_offered_is_one_it_takes(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        step = next(s for s in manifest["steps"] if "conformers" in s["kinds"])
        offered = {o["id"]: {c["value"] for c in o.get("choices", [])} for o in step["options"]}
        self.assertEqual(offered["solvent"] - {"none"}, worker.SOLVENTS)
        self.assertEqual(offered["method"], set(worker.METHODS))
        self.assertEqual(offered["search"], set(worker.SEARCHES))
        # (a search starts from 3D; an optimisation is done with one method)
        self.assertEqual(step["takes"], ["molecules", "conformers"])
        optimise = next(s for s in manifest["steps"] if "optimise" in s["kinds"])
        offered = {o["id"]: {c["value"] for c in o.get("choices", [])} for o in optimise["options"]}
        self.assertEqual(offered["solvent"] - {"none"}, worker.SOLVENTS)
        self.assertEqual(offered["method"], set(worker.METHODS) - {"gfn2//gfnff"})
        self.assertEqual(manifest["systems"], ["macos", "linux"])
        # (CREST's own version is what the plugin says it uses; the manifest's `version` is the plugin's)
        self.assertEqual({u["name"]: u["version"] for u in manifest["uses"]}["CREST"], worker.VERSION)


class Collect(unittest.TestCase):
    def test_the_conformers_lowest_first_with_their_energies(self):
        out = ask("collect", step="conformers", entries=[METHANOL], options={"method": "gfn2", "solvent": "thf"},
                  files={"crest_conformers.xyz": ENSEMBLE}, log=" Version 2.12,   Thu 19. Mai\n" + ECHO_THF, ended="done")["outputs"][0]
        self.assertEqual(out["atoms"], ["C", "O", "H", "H", "H", "H"])
        self.assertEqual(len(out["frames"]), 2)
        self.assertAlmostEqual(out["frames"][0][0], -0.01)
        self.assertAlmostEqual(out["frames"][1][7], -0.9)
        self.assertEqual(out["energies"], [-11.39270131, -11.391])
        self.assertEqual((out["program"], out["version"], out["method"]), ("CREST", "2.12", "iMTD-GC, GFN2-xTB"))
        # (the solvation as CREST echoed its options: no more than it says)
        self.assertEqual(out["solvation"], "ALPB(thf)")

    def test_the_solvation_as_the_run_said_it(self):
        self.assertEqual(worker.solvation_of(ECHO_THF), "ALPB(thf)")
        self.assertEqual(worker.solvation_of(ECHO_THF.replace(" --alpb thf", "").replace("  --alpb thf : implicit solvation\n", "")), "none")
        self.assertIsNone(worker.solvation_of(""))
        # (CREST's own words alone: another program's setup is not read)
        self.assertIsNone(worker.solvation_of("      * Solvation model:               ALPB\n        Solvent                        water\n"))
        # (said as it ran, whatever was asked: whether that is as asked is Meno's to say - the manifest's solvation)
        for asked in ("none", "water"):
            said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "conformers", "entries": [METHANOL], "options": {"solvent": asked}, "files": {"crest_conformers.xyz": ENSEMBLE}, "log": ECHO_THF, "ended": "done"}))
            self.assertTrue(said["ok"], asked)
            self.assertEqual({o["solvation"] for o in said["result"]["outputs"]}, {"ALPB(thf)"})
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual([s.get("solvation") for s in manifest["steps"]], [{"says": "ALPB({solvent})", "none": {"id": "solvent", "is": "none"}}] * 2)

    def test_each_conformers_population_as_cregen_gives_it(self):
        # (after "Metadynamics-based Conformational Sampling": CREGEN's table, the row that begins each set of rotamers giving the set's weight)
        log = """ Version 2.12
       Erel/kcal        Etot weight/tot  conformer     set   degen     origin
       1   0.000   -11.39270    0.40000    0.70000       1       2     mtd4
       2   0.000   -11.39270    0.30000                                mtd1
       3   1.069   -11.39100    0.30000    0.30000       2       1     mtd3
T /K                                  :   298.15
"""
        self.assertEqual(worker.populations_in(log), [0.7, 0.3])
        # (and each one's degeneracy: the first found twice alike, the second once)
        self.assertEqual(worker.degeneracies_in(log), [2, 1])
        out = ask("collect", step="conformers", entries=[METHANOL], options={}, files={"crest_conformers.xyz": ENSEMBLE}, log=log, ended="done")["outputs"][0]
        self.assertEqual((out["populations"], out["degeneracies"]), ([0.7, 0.3], [2, 1]))
        # (no table, or not one for each conformer: none)
        self.assertIsNone(worker.populations_in(" Version 2.12\n"))
        short = ask("collect", step="conformers", entries=[METHANOL], options={}, files={"crest_conformers.xyz": ENSEMBLE}, log=log.replace("3   1.069   -11.39100    0.30000    0.30000       2       1     mtd3", ""), ended="done")
        self.assertIsNone(short["outputs"][0]["populations"])

    def test_how_cregen_took_conformers_as_one_as_the_run_printed_it(self):
        # (after "Conformational Sampling": CREGEN's settings as it prints them before sorting - a first block with
        #  other values, and the last, which counts; written here as CREST 2.12 sets them out)
        block = """ CREGEN> running
 RMSD threshold                 :   {r}
 Bconst threshold               :   0.0100
 population threshold           :   0.0500
 conformer energy window  /kcal :   {w}
 # fragment in coord            :     1
"""
        log = " Version 2.12\n" + block.format(r="0.2500", w="12.0000") + block.format(r="0.1250", w="6.0000")
        said = {r["label"]: (r["value"], r.get("unit")) for r in worker.cregen_settings_in(log)}
        self.assertEqual(said, {
            "RMSD threshold": (0.125, "Å"),
            "Rotational-constant threshold (relative)": (0.01, None),
            "Population threshold": (0.05, None),
            "Energy window": (6.0, "kcal/mol"),
        })
        out = ask("collect", step="conformers", entries=[METHANOL], options={}, files={"crest_conformers.xyz": ENSEMBLE}, log=log, ended="done")["outputs"][0]
        self.assertEqual({r["group"] for r in out["results"]}, {"Same conformer (CREGEN)"})
        self.assertTrue(all(r["on"] == "molecule" for r in out["results"]))
        # (none printed: none said - never a default, nor what was asked)
        self.assertEqual(worker.cregen_settings_in(" Version 2.12\n"), [])
        bare = ask("collect", step="conformers", entries=[METHANOL], options={"ewin": 6}, files={"crest_conformers.xyz": ENSEMBLE}, log=" Version 2.12\n", ended="done")["outputs"][0]
        self.assertEqual(bare["results"], [])

    def test_an_ensemble_optimised_read_back_in_its_order(self):
        out = ask("collect", step="optimise", entries=[METHANOL, METHANOL], options={"method": "gfn2"}, files={"crest_ensemble.xyz": ENSEMBLE}, log="", ended="done")["outputs"]
        self.assertEqual([o["energies"] for o in out], [[-11.39270131], [-11.391]])
        self.assertEqual([len(o["frames"]) for o in out], [1, 1])
        self.assertEqual(out[1]["method"], "GFN2-xTB")
        said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "optimise", "entries": [METHANOL], "options": {}, "files": {"crest_ensemble.xyz": ENSEMBLE}, "log": "", "ended": "done"}))
        self.assertFalse(said["ok"])

    def test_conformers_of_another_molecule_are_not_taken(self):
        said = worker.answer(json.dumps({"id": 1, "op": "collect", "step": "conformers", "entries": [METHANOL], "options": {},
                                         "files": {"crest_conformers.xyz": ENSEMBLE.replace("O    ", "N    ")}, "log": "", "ended": "done"}))
        self.assertFalse(said["ok"])

    def test_says_why_a_job_failed(self):
        log = "\n CREST\nERROR STOP \n\nError termination. Backtrace:\n#0  0x100e04403\n\n  Initial geometry optimization failed!\n  Please check your input.\n"
        said = ask("collect", step="conformers", entries=[METHANOL], options={}, files={}, log=log, ended="failed")
        self.assertEqual(said["why"], "Initial geometry optimization failed!")
        self.assertEqual(ask("collect", step="conformers", entries=[METHANOL], options={}, files={}, log="", ended="stopped")["why"], "CREST said nothing")
        # (a change in topology, the remedies listed after it: what changed)
        topology = (" xTB Geometry Optimization\nERROR STOP abnormal termination of crest\n\nError termination. Backtrace:\n#0  0x104fa4403\n"
                    " WARNING! Change in topology detected!\n The topology change was seen in the initial geometry optimization.\n"
                    " Try either of these options:\n    C) Fix the initial input geometry by introducing bond length constraints\n"
                    "       or by using a method with fixed topology (GFN-FF).\n")
        self.assertEqual(ask("collect", step="conformers", entries=[METHANOL], options={}, files={}, log=topology, ended="failed")["why"],
                         "Change in topology detected! The topology change was seen in the initial geometry optimization.")
        # (ended, but with no conformers written: why, as it said)
        self.assertEqual(ask("collect", step="conformers", entries=[METHANOL], options={}, files={}, log="a\nlast words\n", ended="done")["why"], "last words")


if __name__ == "__main__":
    unittest.main()
