"""Tests for the REINVENT 4 plugin's worker (plugins/reinvent/worker.py).

The worker needs REINVENT 4 and its prior: these run in the plugin's own
environment, installed from its lock, with the prior in the folder the app
keeps the plugin's files in (or the one MENO_REINVENT_FILES names):

    <the REINVENT 4 plugin's Python> -m unittest tests/test_plugin_reinvent.py

(The app's environment for it, under its data folder at
uv/plugin-reinvent/venv, will do.) Elsewhere they are skipped: the
environment is about a gigabyte, too much for every push.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/reinvent/worker.py"
#: Where the app keeps the plugin's files on a Mac, as Meno sets it up (src-tauri/src/plugfiles.rs).
APP_FILES = pathlib.Path.home() / "Library/Application Support/Meno/uv/plugin-reinvent/files"
FILES = pathlib.Path(os.environ.get("MENO_REINVENT_FILES") or APP_FILES)

try:
    importlib.import_module("reinvent")
    HERE = (FILES / "reinvent_pubchem.prior").exists()
except ImportError:  # (CI's Python, or another plugin's)
    HERE = False


def load():
    """The worker, as Meno starts it: the files' folder its one argument."""
    sys.argv = [str(WORKER), str(FILES)]
    spec = importlib.util.spec_from_file_location("plugin_reinvent", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


@unittest.skipUnless(HERE, "REINVENT 4 and its prior are not here")
class Generate(unittest.TestCase):
    """What the worker hands on: each SMILES as REINVENT 4 sampled it, with the model's own values named - nothing
    made of it in 3D, nothing enumerated, nothing refused (docs/PLUGINS.md, *Generate*)."""

    @classmethod
    def setUpClass(cls):
        cls.worker = load()
        cls.made = cls.worker.op_run({"step": "generate", "entries": [], "options": {"count": 20, "seed": 7}})

    def test_gives_the_same_smiles_for_the_same_seed(self):
        # (the known answer, measured here twice, in two environments installed apart: REINVENT 4.8.24, the prior
        #  of Zenodo record 20701824, seed 7, on the CPU - its first SMILES and NLL the same as before Meno stopped
        #  making them 3D, as sampling never depended on it)
        self.assertEqual(self.made["said"], "20 sampled from reinvent_pubchem.prior, seed 7, CPU: 1 invalid, 19 valid")
        first = self.made["outputs"][0]
        self.assertEqual(first["smiles"], "O=C(c1ccc(CN2C(=O)SC(=Cc3ccccc3)C2=O)cc1)N1CCN(CCOc2ccccc2)CC1")
        self.assertEqual(first["properties"]["nll"], {"value": 36.9, "unit": "none"})
        again = self.worker.op_run({"step": "generate", "entries": [], "options": {"count": 20, "seed": 7}})
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.made, sort_keys=True))

    def test_hands_on_each_smiles_as_sampled_and_nothing_else(self):
        outputs = self.made["outputs"]
        # (every one sampled, duplicates and those RDKit would not read among them: Meno reads and counts them)
        self.assertEqual(len(outputs), 20)
        for o in outputs:
            self.assertEqual(set(o), {"smiles", "properties"})
            self.assertIsInstance(o["smiles"], str)
            self.assertEqual(set(o["properties"]) - {"smiles_state"}, {"nll"})
            self.assertEqual(o["properties"]["nll"]["unit"], "none")
            # (REINVENT 4's SMILES_state, as it writes it: 0 invalid, 1 valid, 2 a duplicate)
            self.assertIn(o["properties"]["smiles_state"]["value"], (0, 1, 2))
        # (as sampled, never written again: what is read is what the model said)
        with tempfile.TemporaryDirectory() as folder:
            rows = self.worker.sampled(20, 7, folder)
        self.assertEqual([o["smiles"] for o in outputs], [r[0] for r in rows])

    def test_says_how_many_of_each_smiles_state(self):
        # (REINVENT 4's own states, by its names, as many as it gave of each)
        states = [o["properties"]["smiles_state"]["value"] for o in self.made["outputs"]]
        tally = ", ".join(f"{states.count(k)} {name}" for k, name in self.worker.STATES.items() if states.count(k))
        self.assertEqual(self.made["said"], f"20 sampled from {self.worker.PRIOR}, seed 7, CPU: {tally}")

    def test_writes_an_nll_that_is_no_number_as_none_and_counts_it(self):
        real = self.worker.sampled
        self.worker.sampled = lambda count, seed, folder: [("CCO", 1, 12.5), ("CCN", 1, float("nan")), ("C=C", 2, float("inf"))]
        try:
            made = self.worker.op_run({"step": "generate", "entries": [], "options": {"count": 3, "seed": 7}})
        finally:
            self.worker.sampled = real
        # (what JSON can hold: never NaN, which would fail the whole answer)
        json.dumps(made, allow_nan=False)
        self.assertEqual([o["properties"]["nll"]["value"] for o in made["outputs"]], [12.5, None, None])
        self.assertTrue(made["said"].endswith(": 2 valid, 1 duplicate, 2 with no NLL (not a number)"), made["said"])

    def test_says_what_made_them(self):
        self.assertEqual(
            self.made["by"],
            {
                "program": "REINVENT 4",
                "version": self.worker.VERSION,
                "model": self.worker.PRIOR_SAID,
                "method": "sampled, seed 7, on the CPU",
            },
        )

    def test_refuses_an_option_outside_what_it_may_be_never_running_another(self):
        # (a seed of 0 among them: REINVENT 4 sets none for it, so the run would not sample the same way again)
        for options, said in [
            ({"count": 0, "seed": 7}, "How many is 0: it takes a whole number from 1 to 100"),
            ({"count": 500}, "How many is 500: it takes a whole number from 1 to 100"),
            ({"count": 2.5}, "How many is 2.5: it takes a whole number from 1 to 100"),
            ({"seed": 0}, "Random seed is 0: it takes a whole number from 1 to 2147483647"),
            ({"seed": "7"}, "Random seed is 7: it takes a whole number from 1 to 2147483647"),
        ]:
            with self.assertRaises(ValueError) as e:
                self.worker.op_run({"step": "generate", "entries": [], "options": options})
            self.assertEqual(str(e.exception), said)
        with self.assertRaises(ValueError):
            self.worker.op_run({"step": "optimise", "entries": [], "options": {}})


class Options(unittest.TestCase):
    """What the step offers, as the manifest says it and the worker takes it - needing nothing of REINVENT 4."""

    def test_the_seed_starts_at_one_and_every_one_is_kept(self):
        manifest = json.loads((ROOT / "plugins/reinvent/manifest.json").read_text(encoding="utf-8"))
        seed = next(o for o in manifest["steps"][0]["options"] if o["id"] == "seed")
        self.assertEqual((seed["min"], seed["default"]), (1, 1))
        text = WORKER.read_text(encoding="utf-8")
        self.assertIn("unique_molecules = false", text)
        self.assertNotIn("randomize_smiles", text)


if __name__ == "__main__":
    unittest.main()
