"""Tests for the CReM plugin's worker (plugins/crem/worker.py).

The worker needs CReM and RDKit: these run in the plugin's own environment,
installed from its lock (CI's job "plugins" does so on every push):

    <the CReM plugin's Python> -m unittest tests/test_plugin_crem.py

Those that make molecules need its fragment database too, unpacked as Meno
unpacks it, in the folder the app keeps the plugin's files in (or the one
MENO_CREM_FILES names); elsewhere they are skipped - it is 2 GB unpacked,
too much for every push. The app's environment for it, under its data
folder at uv/plugin-crem/venv, will do. Nothing of the database is kept here.
"""

import importlib.util
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/crem/worker.py"
#: Where the app keeps the plugin's files on a Mac, as Meno sets it up (src-tauri/src/plugfiles.rs).
APP_FILES = pathlib.Path.home() / "Library/Application Support/Meno/uv/plugin-crem/files"
FILES = pathlib.Path(os.environ.get("MENO_CREM_FILES") or APP_FILES)

try:
    importlib.import_module("crem")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    THERE = True
    HERE = (FILES / "chembl33_sa2_f5.db" / "chembl33_sa2_f5.db").exists()
except ImportError:  # (CI's Python, or another plugin's)
    THERE = HERE = False


def load(files=FILES):
    """The worker, as Meno starts it: the files' folder its one argument."""
    sys.argv = [str(WORKER), str(files)]
    spec = importlib.util.spec_from_file_location("plugin_crem", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def entry(smiles):
    """A structure drawn as Meno gives one: a MOL block for each of its parts, laid out in 2D, wedged as the SMILES says."""
    blocks = []
    for part in Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True):
        AllChem.Compute2DCoords(part)
        blocks.append(Chem.MolToMolBlock(part))
    return {"molblocks": blocks}


@unittest.skipUnless(THERE, "CReM is not here")
class Seeds(unittest.TestCase):
    """What the worker reads of a seed, before any fragment is looked for: no database needed."""

    @classmethod
    def setUpClass(cls):
        cls.worker = load()

    def test_reads_a_seed_as_drawn_its_stereo_as_its_blocks_say(self):
        for smiles in ["C[C@@H](N)c1ccccc1", "C[C@H](N)c1ccccc1", "C/C=C/C(=O)O", "C/C=C\\C(=O)O", "CC(N)c1ccccc1"]:
            got = self.worker.seed_of(entry(smiles))
            self.assertEqual(Chem.MolToSmiles(got), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), smiles)

    def test_takes_a_seed_as_it_is_never_refusing_it(self):
        # (its parts, charges and isotopes as drawn: what it is, the general steps say)
        for smiles in ["C[NH3+].[Cl-]", "C[N+](C)(C)C", "[13CH3]O", "C[CH2]"]:
            self.assertEqual(Chem.MolToSmiles(self.worker.seed_of(entry(smiles))), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), smiles)
        self.assertEqual(self.worker.seed_of({"molblocks": []}), "it is a label of no element")
        self.assertEqual(self.worker.seed_of({"molblocks": ["not a block"]}), "a part RDKit cannot read")

    def test_says_where_its_database_is_not(self):
        bare = load(pathlib.Path("/nonexistent-meno-files"))
        with self.assertRaisesRegex(ValueError, "fragment database is not here"):
            bare.op_run({"step": "generate", "entries": [entry("CO")], "options": {}})
        with self.assertRaisesRegex(ValueError, "no step 'optimise'"):
            bare.op_run({"step": "optimise", "entries": [], "options": {}})


@unittest.skipUnless(HERE, "CReM and its fragment database are not here")
class Generate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load()
        cls.seed = entry("CC(=O)Nc1ccc(O)cc1")

    def made(self, operation, entries=None, **more):
        return self.worker.op_run({"step": "generate", "entries": entries or [self.seed], "options": {"operation": operation, "count": 8, "seed": 7, **more}})

    def test_gives_the_same_structures_for_the_same_seed(self):
        # (the known answers, measured here twice: CReM 0.3.2, RDKit 2026.03.6, chembl33_sa2_f5 of Zenodo record
        #  16909329, count 8, seed 7 - each SMILES as CReM gave it)
        grow = self.made("grow")
        self.assertEqual(grow["said"], "8 made from 1 seed (hydrogens replaced, radius 3, seed 7)")
        self.assertEqual([o["smiles"] for o in grow["outputs"]], GROWN)
        mutate = self.made("mutate")
        self.assertEqual([o["smiles"] for o in mutate["outputs"]], MUTATED)
        self.assertEqual(self.made("grow"), grow)

    def test_gives_each_as_crem_gave_it_its_values_named_and_its_seed(self):
        got = self.made("grow")
        out = got["outputs"][0]
        # (nothing made of it here: no atoms, no geometry, no stereoisomers - its SMILES, its values and its seed)
        self.assertEqual(set(out), {"smiles", "properties", "seed"})
        self.assertEqual(out["seed"], 0)
        self.assertEqual(out["properties"]["change"], {"value": "[H][*:1]>>C[*:1]", "unit": "none"})
        # (only what CReM gives: no value of the seed's made here)
        self.assertEqual(set(out["properties"]), {"change", "frequency"})
        self.assertEqual(got["seeds"], [{"place": 0, "made": 8}])
        self.assertIsInstance(out["properties"]["frequency"]["value"], int)
        self.assertEqual(got["by"]["program"], "CReM")
        self.assertEqual(got["by"]["version"], "0.3.2")
        self.assertTrue(got["by"]["model"].startswith("chembl33_sa2_f5 (CReM fragment databases, Zenodo 10.5281/zenodo.16909329"), got["by"]["model"])
        self.assertEqual(got["by"]["method"], "hydrogens replaced, radius 3, at most 2 heavy atoms added, seed 7")

    def test_tells_each_seed_by_its_place_made_some_none_or_not_read(self):
        # (a lone chloride ion: read, but nothing in the database grows on it)
        got = self.made("grow", [self.seed, {"molblocks": ["not a block"]}, entry("Oc1ccccc1"), entry("[Cl-]")], count=3)
        self.assertEqual(sorted({o["seed"] for o in got["outputs"]}), [0, 2])
        self.assertEqual(got["seeds"], [{"place": 0, "made": 3}, {"place": 1, "why": "a part RDKit cannot read"}, {"place": 2, "made": 3}, {"place": 3, "made": 0}])
        self.assertEqual(got["said"], "6 made from 3 seeds (hydrogens replaced, radius 3, seed 7)")
        # (none read: nothing made, and each seed's why in the tally, for Meno to name)
        none = self.made("grow", [{"molblocks": ["not a block"]}, {"molblocks": []}])
        self.assertEqual(none["outputs"], [])
        self.assertEqual(none["said"], "No seed could be read")
        self.assertEqual(none["seeds"], [{"place": 0, "why": "a part RDKit cannot read"}, {"place": 1, "why": "it is a label of no element"}])

    def test_refuses_an_option_outside_what_it_may_be_never_running_another(self):
        for options, said in [
            ({"count": 150}, "How many from each seed is 150: it takes a whole number from 1 to 100"),
            ({"radius": 0}, "Surroundings matched is 0: it takes a whole number from 1 to 5"),
            ({"adds": 2.5}, "Heavy atoms added is 2.5: it takes a whole number from 1 to 10"),
            ({"seed": -1}, "Random seed is -1: it takes a whole number from 0 to 2147483647"),
            ({"operation": "link"}, "Change is 'link': it takes 'mutate' or 'grow'"),
        ]:
            with self.assertRaises(ValueError) as e:
                self.worker.op_run({"step": "generate", "entries": [self.seed], "options": options})
            self.assertEqual(str(e.exception), said)

    def test_cuts_nothing_and_refuses_no_run_for_its_size(self):
        # (Meno takes no more of a run than it takes of any - workflow/programs: none of it, said - so the plugin keeps no cap of its own)
        self.assertFalse(hasattr(self.worker, "MOST"))
        got = self.made("grow", [self.seed] * 3, count=8)
        self.assertEqual(len(got["outputs"]), sum(s["made"] for s in got["seeds"]))


#: What CReM made of the seed (count 8, seed 7), as it gave each - measured on its own environment and database.
GROWN = ["COc1ccc(NC(C)=O)cc1", "O=C(CO)Nc1ccc(O)cc1", "CC(=O)N(C)c1ccc(O)cc1", "CC(=O)Nc1ccc(O)cc1F", "COc1cc(NC(C)=O)ccc1O", "CC(=O)Nc1ccc(O)c(C)c1", "CCN(C(C)=O)c1ccc(O)cc1", "CCCC(=O)Nc1ccc(O)cc1"]
MUTATED = ["CN(C)C(=O)c1ccc(O)cc1", "Cc1nc2ccccc2c(=O)[nH]1", "OCCN1CCOCC1", "CCc1ccccc1NC(C)=O", "CC(=O)Nc1nc2ccccc2[nH]1", "N#CCC(=O)Nc1ccc(O)cc1", "CCCCc1ccc(C(=O)O)nc1", "CC(=O)Nc1cccc(O)c1"]


if __name__ == "__main__":
    unittest.main()
