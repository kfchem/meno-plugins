"""Tests for the NCIPLOT and Multiwfn interfaces' workers (plugins/nciplot, multiwfn).

The workers need Python alone - NCIPLOT and Multiwfn are run by Meno, not
by them - so these run anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_nci.py

What the programs are given is checked against their own manuals
(NCIPLOT-4.2's NCIPLOT_MANUAL.pdf; the Multiwfn 2026.10.1 manual). Where
a program is installed and named - NCIPLOT_BIN with NCIPLOT_HOME, or
MULTIWFN_BIN, with MULTIWFN_WFN a wavefunction to read for the run from
one - a job is run too, and its cube files looked for; elsewhere those
tests are skipped. No output
of a real run is kept.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"plugin_{name}", ROOT / f"plugins/{name}/worker.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


nciplot = load("nciplot")
multiwfn = load("multiwfn")

# the water dimer, two molecules bonded within themselves
DIMER = {
    "name": "water dimer",
    "atoms": [
        {"el": "O", "x": -1.551007, "y": -0.114520, "z": 0.0},
        {"el": "H", "x": -1.934259, "y": 0.762503, "z": 0.0},
        {"el": "H", "x": -0.599677, "y": 0.040712, "z": 0.0},
        {"el": "O", "x": 1.350625, "y": 0.111469, "z": 0.0},
        {"el": "H", "x": 1.680398, "y": -0.373741, "z": -0.758561},
        {"el": "H", "x": 1.680398, "y": -0.373741, "z": 0.758561},
    ],
    "bonds": [{"a1": 0, "a2": 1, "order": 1}, {"a1": 0, "a2": 2, "order": 1}, {"a1": 3, "a2": 4, "order": 1}, {"a1": 3, "a2": 5, "order": 1}],
    "fragments": [[0, 1, 2], [3, 4, 5]],
    "charge": 0,
    "multiplicity": 1,
}


def ask(worker, op, **m):
    return worker.answer(json.dumps({"id": 1, "op": op, "step": "nci", **m}))


def run(job, program, env):
    with tempfile.TemporaryDirectory() as d:
        for f in job["files"]:
            pathlib.Path(d, f["name"]).write_text(f["text"], encoding="utf-8")
        stdin = open(pathlib.Path(d, job["stdin"]), encoding="utf-8") if job.get("stdin") else subprocess.DEVNULL
        done = subprocess.run([program, *job["args"]], cwd=d, env={**os.environ, **env}, stdin=stdin, capture_output=True, text=True, timeout=900)
        return done, sorted(os.listdir(d))


class Nciplot(unittest.TestCase):
    def test_writes_the_geometry_and_its_input_as_the_manual_has_them(self):
        said = ask(nciplot, "prepare", entries=[DIMER], options={})
        self.assertTrue(said["ok"], said)
        [job] = said["result"]["jobs"]
        self.assertEqual((job["program"], job["args"], job["reads"]), ("nciplot", ["input.nci"], []))
        files = {f["name"]: f["text"] for f in job["files"]}
        # (the number of files, each's name; its outputs' name; the cubes alone; the density cut and the surface's RDG)
        self.assertEqual(files["input.nci"].splitlines(), ["1", "molecule.xyz", "ONAME NCI", "OUTPUT 2", "CUTPLOT 0.07 0.3"])
        xyz = files["molecule.xyz"].splitlines()
        self.assertEqual(xyz[0], "6")
        self.assertEqual(xyz[2].split()[0], "O")

    def test_puts_each_molecule_in_a_file_of_its_own_for_interactions_between_them_alone(self):
        said = ask(nciplot, "prepare", entries=[DIMER], options={"between": True, "grid": "fine", "isovalue": 0.4})
        files = {f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}
        self.assertEqual(files["input.nci"].splitlines(), ["2", "molecule1.xyz", "molecule2.xyz", "ONAME NCI", "OUTPUT 2", "FINE", "INTERMOLECULAR", "CUTPLOT 0.07 0.4"])
        self.assertEqual(files["molecule1.xyz"].splitlines()[0], "3")
        # (the density left out above, where chosen: CUTPLOT's first value - 0.07 unless chosen, the manual's default)
        said = ask(nciplot, "prepare", entries=[DIMER], options={"density-cut": 0.05})
        files = {f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}
        self.assertEqual(files["input.nci"].splitlines()[-1], "CUTPLOT 0.05 0.3")
        # (one molecule has none to be between)
        one = {**DIMER, "atoms": DIMER["atoms"][:3], "bonds": DIMER["bonds"][:2], "fragments": [[0, 1, 2]]}
        self.assertFalse(ask(nciplot, "prepare", entries=[one], options={"between": True})["ok"])

    def test_writes_the_fragments_it_is_given_and_works_out_none(self):
        # (its fragments as given, not as its bonds would make them: each fragment's atoms, in its file, in order)
        given = {**DIMER, "fragments": [[0, 2], [1, 3, 4, 5]]}
        files = {f["name"]: f["text"] for f in ask(nciplot, "prepare", entries=[given], options={"between": True})["result"]["jobs"][0]["files"]}
        self.assertEqual([files[n].splitlines()[0] for n in ("molecule1.xyz", "molecule2.xyz")], ["2", "4"])
        self.assertEqual([line.split()[0] for line in files["molecule1.xyz"].splitlines()[2:]], ["O", "H"])
        # (none given, or not each atom once: refused in words)
        for fragments in [None, [], [[0, 1, 2]], [[0, 1, 2], [2, 3, 4, 5]], [[0, 1, 2], [3, 4, 5, 6]], [[0, 1, 2], []], [[0, 1, 2], [3, 4, True]]]:
            entry = {k: v for k, v in DIMER.items() if k != "fragments"} if fragments is None else {**DIMER, "fragments": fragments}
            said = ask(nciplot, "prepare", entries=[entry], options={"between": True})
            self.assertFalse(said["ok"], fragments)
            self.assertTrue(said["error"].startswith("NCIPLOT: "), said)
        # (not asked for between: one file, the fragments not looked at)
        self.assertTrue(ask(nciplot, "prepare", entries=[{**DIMER, "fragments": None}], options={})["ok"])

    def test_says_which_cube_is_the_surface_and_which_colours_it(self):
        said = ask(nciplot, "collect", entries=[DIMER], options={"range": 0.05}, files={}, log="", ended="done")
        [s] = said["result"]["surfaces"]
        [row] = s["rows"]
        self.assertEqual((row["file"], row["iso"], row["colour"]["file"]), ("NCI-grad.cube", 0.3, "NCI-dens.cube"))
        # (sign(λ2)ρ as NCIPLOT writes it, × 100)
        self.assertAlmostEqual(row["colour"]["from"], -5)
        self.assertAlmostEqual(row["colour"]["to"], 5)
        failed = ask(nciplot, "collect", entries=[DIMER], options={}, files={}, log="reading\n ERROR: no atoms\n", ended="failed")
        self.assertEqual(failed["result"], {"why": "ERROR: no atoms"})

    def test_refuses_what_it_cannot_do(self):
        for options in [{"grid": "huge"}, {"isovalue": 0}, {"between": "yes"}, {"density-cut": 0}, {"density-cut": 0.6}, {"density-cut": "0.07"}]:
            self.assertFalse(ask(nciplot, "prepare", entries=[DIMER], options=options)["ok"], options)
        self.assertFalse(ask(nciplot, "prepare", entries=[{**DIMER, "atoms": [{"el": "x1", "x": 0, "y": 0, "z": 0}]}], options={})["ok"])
        self.assertFalse(nciplot.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [DIMER]}))["ok"])

    @unittest.skipUnless(os.environ.get("NCIPLOT_BIN") and os.environ.get("NCIPLOT_HOME"), "NCIPLOT is not named here")
    def test_nciplot_writes_the_cubes_it_says(self):
        job = ask(nciplot, "prepare", entries=[DIMER], options={"between": True})["result"]["jobs"][0]
        done, files = run(job, os.environ["NCIPLOT_BIN"], {"OMP_NUM_THREADS": "2"})
        self.assertEqual(done.returncode, 0, done.stdout[-2000:])
        self.assertIn("NCI-grad.cube", files)
        self.assertIn("NCI-dens.cube", files)


class Multiwfn(unittest.TestCase):
    WFN = {**DIMER, "files": [{"kind": "molden", "name": "water.molden", "text": "[Molden Format]\n"}]}

    def test_writes_the_wavefunction_and_what_multiwfn_is_told_as_its_manual_has_them(self):
        said = ask(multiwfn, "prepare", entries=[self.WFN], options={"grid": "high"}, cores=4)
        self.assertTrue(said["ok"], said)
        [job] = said["result"]["jobs"]
        self.assertEqual((job["program"], job["args"], job["stdin"]), ("multiwfn", ["wavefunction.molden", "-nt", "4", "-silent"], "nci.txt"))
        files = {f["name"]: f["text"] for f in job["files"]}
        self.assertEqual(files["wavefunction.molden"], "[Molden Format]\n")
        # (weak interaction, NCI, the grid, the cubes, back, back, quit - read back, for collect to know which NCI it was)
        self.assertEqual(files["nci.txt"].splitlines(), ["20", "1", "3", "3", "0", "0", "q"])
        self.assertEqual(job["reads"], ["nci.txt"])
        # (each kind under the extension Multiwfn tells it by)
        for kind, ext in [("gaussian-fchk", ".fch"), ("wfx", ".wfx")]:
            entry = {**DIMER, "files": [{"kind": kind, "name": "x", "text": "x"}]}
            self.assertEqual(ask(multiwfn, "prepare", entries=[entry], options={})["result"]["jobs"][0]["args"][0], "wavefunction" + ext)
        # (a kind it does not name - no plugin declares one - is no wavefunction of its: from the geometry alone, where
        #  the fallback is chosen)
        entry = {**DIMER, "files": [{"kind": "wfn", "name": "x", "text": "x"}]}
        self.assertEqual(ask(multiwfn, "prepare", entries=[entry], options={"fallback": True})["result"]["jobs"][0]["args"][0], "molecule.xyz")

    def test_works_from_the_geometry_alone_where_promolecular_is_asked_for(self):
        orca = {**DIMER, "files": [{"kind": "orca", "name": "x.out", "text": "x"}]}
        for entry, options in [(DIMER, {"fallback": True}), (orca, {"fallback": True}), (self.WFN, {"density": "promolecular"}), (DIMER, {"density": "promolecular"})]:
            said = ask(multiwfn, "prepare", entries=[entry], options={**options, "grid": "low"}, cores=2)
            self.assertTrue(said["ok"], said)
            [job] = said["result"]["jobs"]
            self.assertEqual(job["args"], ["molecule.xyz", "-nt", "2", "-silent"])
            files = {f["name"]: f["text"] for f in job["files"]}
            self.assertNotIn("wavefunction.molden", files)
            # (NCI based on promolecular density: the same steps after it, 3.23.2)
            self.assertEqual(files["nci.txt"].splitlines(), ["20", "2", "1", "3", "0", "0", "q"])
            lines = files["molecule.xyz"].splitlines()
            self.assertEqual((lines[0], lines[1]), ("6", "water dimer"))
            self.assertEqual(lines[2].split(), ["O", "-1.55100700", "-0.11452000", "0.00000000"])
        # (the fallback chosen, a step's entries each as they can be: one with a wavefunction from it, one from its geometry)
        jobs = ask(multiwfn, "prepare", entries=[self.WFN, DIMER], options={"fallback": True})["result"]["jobs"]
        self.assertEqual([j["args"][0] for j in jobs], ["wavefunction.molden", "molecule.xyz"])

    def test_refuses_a_molecule_with_no_wavefunction_unless_the_fallback_is_chosen(self):
        # (off unless chosen: the one with no wavefunction refused in words, alone - out of the Failed exit - the
        #  other run from its wavefunction; no step's surfaces of two densities)
        said = ask(multiwfn, "prepare", entries=[self.WFN, DIMER], options={})
        self.assertTrue(said["ok"], said)
        self.assertEqual([(j["entries"], j["args"][0]) for j in said["result"]["jobs"]], [([0], "wavefunction.molden")])
        [refused] = said["result"]["refused"]
        self.assertEqual((refused["entry"], refused["kind"]), (1, "refused"))
        self.assertIn("read from no wavefunction", refused["why"])
        self.assertIn("Promolecular where a molecule has no wavefunction", refused["why"])
        # (none with one: the step refused as a whole, saying why)
        said = ask(multiwfn, "prepare", entries=[DIMER], options={})
        self.assertFalse(said["ok"])
        self.assertIn("no molecule was read from a wavefunction", said["error"])
        # (all with one: nothing refused)
        self.assertNotIn("refused", ask(multiwfn, "prepare", entries=[self.WFN], options={})["result"])

    def test_refuses_what_it_cannot_do(self):
        self.assertFalse(ask(multiwfn, "prepare", entries=[DIMER], options={"density": "exact"})["ok"])
        self.assertFalse(ask(multiwfn, "prepare", entries=[DIMER], options={"fallback": "yes"})["ok"])
        self.assertFalse(ask(multiwfn, "prepare", entries=[{**DIMER, "atoms": []}], options={"fallback": True})["ok"])
        self.assertFalse(ask(multiwfn, "prepare", entries=[{**DIMER, "atoms": [{"el": "c 1", "x": 0, "y": 0, "z": 0}]}], options={"fallback": True})["ok"])
        self.assertFalse(ask(multiwfn, "prepare", entries=[], options={})["ok"])

    def test_says_which_cube_is_the_surface_and_which_colours_it(self):
        said = ask(multiwfn, "collect", entries=[self.WFN], options={}, files={"nci.txt": "20\n1\n2\n3\n0\n0\nq\n"}, log="", ended="done")
        [row] = said["result"]["surfaces"][0]["rows"]
        self.assertEqual((row["label"], row["file"], row["iso"], row["colour"]["file"], row["colour"]["from"], row["colour"]["to"]), ("Wavefunction's density", "func2.cub", 0.5, "func1.cub", -0.035, 0.02))
        # (of the promolecular density, as Multiwfn was told: its own surface's RDG)
        said = ask(multiwfn, "collect", entries=[DIMER], options={}, files={"nci.txt": "20\n2\n2\n3\n0\n0\nq\n"}, log="", ended="done")
        [row] = said["result"]["surfaces"][0]["rows"]
        self.assertEqual((row["label"], row["iso"]), ("Promolecular density", 0.3))
        said = ask(multiwfn, "collect", entries=[DIMER], options={"isovalue-promolecular": 0.25}, files={"nci.txt": "20\n2\n2\n3\n0\n0\nq\n"}, log="", ended="done")
        self.assertEqual(said["result"]["surfaces"][0]["rows"][0]["iso"], 0.25)
        self.assertIn("why", ask(multiwfn, "collect", entries=[self.WFN], options={}, files={}, log=" Error: cannot read\n", ended="failed")["result"])
        self.assertFalse(ask(multiwfn, "collect", entries=[self.WFN], options={"from": 0.01, "to": 0.0}, files={}, log="", ended="done")["ok"])

    @unittest.skipUnless(os.environ.get("MULTIWFN_BIN") and os.environ.get("MULTIWFN_WFX"), "Multiwfn is not named here")
    def test_multiwfn_writes_the_cubes_it_says(self):
        text = pathlib.Path(os.environ["MULTIWFN_WFX"]).read_text(encoding="utf-8")
        entry = {**DIMER, "files": [{"kind": "wfx", "name": "x.wfx", "text": text}]}
        job = ask(multiwfn, "prepare", entries=[entry], options={"grid": "low"}, cores=2)["result"]["jobs"][0]
        done, files = run(job, os.environ["MULTIWFN_BIN"], {"Multiwfnpath": str(pathlib.Path(os.environ["MULTIWFN_BIN"]).parent)})
        self.assertEqual(done.returncode, 0, done.stdout[-2000:])
        self.assertIn("func1.cub", files)
        self.assertIn("func2.cub", files)

    @unittest.skipUnless(os.environ.get("MULTIWFN_BIN"), "Multiwfn is not named here")
    def test_multiwfn_writes_the_cubes_it_says_from_the_geometry_alone(self):
        job = ask(multiwfn, "prepare", entries=[DIMER], options={"grid": "low", "density": "promolecular"}, cores=2)["result"]["jobs"][0]
        done, files = run(job, os.environ["MULTIWFN_BIN"], {"Multiwfnpath": str(pathlib.Path(os.environ["MULTIWFN_BIN"]).parent)})
        self.assertEqual(done.returncode, 0, done.stdout[-2000:])
        self.assertIn("has been exported to func2.cub", done.stdout)
        self.assertIn("func1.cub", files)
        self.assertIn("func2.cub", files)


class Manifests(unittest.TestCase):
    def test_declare_what_the_workers_read(self):
        for name, ids in [("nciplot", {"grid", "between", "isovalue", "density-cut", "range"}), ("multiwfn", {"density", "fallback", "grid", "isovalue", "isovalue-promolecular", "from", "to"})]:
            m = json.loads((ROOT / f"plugins/{name}/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({o["id"] for o in m["steps"][0]["options"]}, ids, name)
            self.assertEqual(m["steps"][0]["kinds"], ["nci"])

    def test_the_density_cut_the_card_offers_is_the_one_the_worker_takes(self):
        # (its default the worker's own, its ends taken and a step past each refused: a card offers nothing it refuses)
        m = json.loads((ROOT / "plugins/nciplot/manifest.json").read_text(encoding="utf-8"))
        cut = next(o for o in m["steps"][0]["options"] if o["id"] == "density-cut")
        self.assertEqual(cut["default"], nciplot.RHO_CUT)
        for v in (cut["min"], cut["max"]):
            said = ask(nciplot, "prepare", entries=[DIMER], options={"density-cut": v})
            self.assertTrue(said["ok"], said)
            self.assertEqual({f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}["input.nci"].splitlines()[-1], f"CUTPLOT {float(v)} 0.3")
        for v in (round(cut["min"] - cut["step"], 10), round(cut["max"] + cut["step"], 10)):
            self.assertFalse(ask(nciplot, "prepare", entries=[DIMER], options={"density-cut": v})["ok"], v)


if __name__ == "__main__":
    unittest.main()
