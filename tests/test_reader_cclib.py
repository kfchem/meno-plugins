"""Tests for the cclib plugin's worker (plugins/cclib/worker.py).

They need cclib, so they run in an environment built from its lock - as
CI's job "Plugins" builds it - or, by hand:

    uv venv .venv-cclib --python 3.12
    uv pip install --python .venv-cclib/bin/python --require-hashes --no-deps \
        -r plugins/cclib/requirements.lock
    .venv-cclib/bin/python -m unittest tests/test_reader_cclib.py

The programs' outputs they read are not in the repository
(docs/WORKSPACE.md, stage 3): put small ones - cclib's own samples, from
its repository's data folder - in calc-samples/, as
calc-samples/XTB/dvb_sp.out and so on; tests/samples.json says
where each of one per program comes from. Those tests are skipped where
the files are not there.
"""

import importlib.util
import json
import pathlib
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/cclib/worker.py"
SAMPLES = ROOT / "calc-samples"
spec = importlib.util.spec_from_file_location("reader_cclib", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def read(path):
    """The worker's answer to reading a sample, as the app asks for it (the
    app reads a file as UTF-8, whatever the system's own encoding)."""
    line = json.dumps({"id": 1, "op": "read", "name": path.name, "text": path.read_text(encoding="utf-8")})
    return worker.answer(line)


def result(r, id_):
    """One of what it found, by its id, in Meno's general form."""
    return next((x for x in r["results"] if x["id"] == id_), None)


def sample(rel):
    path = SAMPLES / rel
    if not path.exists():
        raise unittest.SkipTest(f"no {rel} in calc-samples/")
    return path


class Protocol(unittest.TestCase):
    def test_answers_only_what_it_does(self):
        # (it says its version in its ready line, not when asked)
        a = worker.answer(json.dumps({"id": 3, "op": "ping"}))
        self.assertEqual((a["id"], a["ok"]), (3, False))

    def test_answers_a_failure_rather_than_crashing(self):
        a = worker.answer(json.dumps({"id": 4, "op": "read", "name": "x.out", "text": "nothing a program wrote\n"}))
        self.assertEqual((a["id"], a["ok"]), (4, False))
        self.assertTrue(a["error"])
        self.assertFalse(worker.answer(json.dumps({"id": 5, "op": "read", "name": "x.out", "text": ""}))["ok"])
        self.assertFalse(worker.answer(json.dumps({"id": 6, "op": "run", "code": "1"}))["ok"])


class Energies(unittest.TestCase):
    def test_an_energy_is_the_outputs_value_as_printed(self):
        # (an energy in electronvolts, back in hartrees by the same factor: as it was, to the last digit)
        from cclib.parser.utils import convertor

        for printed in (-76.0123456789, -1.1167593073, -2573.987654321098):
            self.assertAlmostEqual(worker.hartrees(convertor(printed, "hartree", "eV")), printed, delta=1e-10)


class Orbitals(unittest.TestCase):
    def test_an_open_shell_s_orbitals_alpha_then_beta_each_occupied_by_one(self):
        data = types.SimpleNamespace(homos=[1, 0], moenergies=[[-10.0, -5.0, 2.0], [-9.0, 1.0, 3.0]], mosyms=None)
        o = worker.orbitals(data)
        self.assertEqual([(x["spin"], x["occupation"]) for x in o], [("alpha", 1.0), ("alpha", 1.0), ("alpha", 0.0), ("beta", 1.0), ("beta", 0.0), ("beta", 0.0)])
        self.assertAlmostEqual(o[0]["energy"], worker.hartrees(-10.0))
        self.assertIsNone(worker.orbitals(types.SimpleNamespace(homos=None, moenergies=None)))


class XTB(unittest.TestCase):
    def test_an_optimisation_gives_its_final_geometry(self):
        r = read(sample("XTB/dvb_opt.out"))["result"]
        self.assertEqual((r["program"], r["method"]), ("xTB", "GFN2-xTB"))
        self.assertEqual(len(r["frames"]), 1)
        self.assertEqual(len(r["energies"]), 1)

    def test_a_single_point_has_no_geometry_to_give(self):
        r = read(sample("XTB/dvb_sp.out"))["result"]
        self.assertEqual(r["frames"], [])
        self.assertIsNone(r["energies"])


class Vibrations(unittest.TestCase):
    def test_translations_and_rotations_listed_at_naught_are_no_vibrations(self):
        # (as an xTB output lists them: six at naught before the molecule's own, with intensities of their own; an imaginary one kept, negative)
        data = types.SimpleNamespace(
            vibfreqs=[-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -120.5, 1539.1, 3642.25],
            vibirs=[38.18, 58.16, 32.15, 167.64, 70.04, 108.65, 4.0, 133.26, 6.78],
            vibdisps=[[[0.0, 0.0, float(k)]] * 3 for k in range(9)],
            vibsyms=None,
        )
        v = worker.vibrations(data, 3)[0]
        self.assertEqual([row["cells"] for row in v["rows"]], [[1, -120.5, 4.0], [2, 1539.1, 133.26], [3, 3642.25, 6.78]])
        # (each its own displacements)
        self.assertEqual([row["move"][2] for row in v["rows"]], [6.0, 7.0, 8.0])
        self.assertEqual(worker.vibrations(types.SimpleNamespace(vibfreqs=[0.0] * 6), 3), [])

    def test_reduced_masses_and_raman_activities_where_the_output_gives_them(self):
        data = types.SimpleNamespace(
            vibfreqs=[1600.0, 3650.0, 3750.0], vibsyms=["A1", "A1", "B2"], vibirs=[11.6, 1.2, 4.8],
            vibrmasses=[1.08, 1.05, 1.08], vibramans=[8.1, 43.6, 17.3],
        )
        v = worker.vibrations(data, 3)[0]
        self.assertEqual([c["label"] for c in v["columns"]], ["Mode", "Symmetry", "Frequency", "Reduced mass", "IR intensity", "Raman activity"])
        self.assertEqual(v["columns"][-1]["unit"], "Å⁴/amu")
        self.assertEqual(v["rows"][1]["cells"], [2, "A1", 3650.0, 1.05, 1.2, 43.6])
        # (where it gives none - or not one for each - no column)
        data.vibramans, data.vibrmasses = None, [1.0]
        self.assertEqual([c["label"] for c in worker.vibrations(data, 3)[0]["columns"]], ["Mode", "Symmetry", "Frequency", "IR intensity"])

    def test_an_xtb_output_s_vibrations_are_the_molecule_s_own(self):
        r = read(sample("XTB/dvb_ir.out"))["result"]
        rows = result(r, "vibrations")["rows"]
        # (divinylbenzene: 20 atoms, 3 x 20 - 6 vibrations, numbered from one)
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(abs(row["cells"][1]) >= 1 for row in rows))
        self.assertEqual([row["cells"][0] for row in rows], list(range(1, 55)))


class EveryProgram(unittest.TestCase):
    """Each program whose kind cclib's manifest brings, from a real output of
    it (tests/samples.json says which, and where each comes from)."""

    def test_each_gives_its_molecule_named_by_its_program(self):
        table = json.loads((ROOT / "tests/samples.json").read_text(encoding="utf-8"))
        manifest = json.loads((WORKER.parent / "manifest.json").read_text(encoding="utf-8"))
        read_by_cclib = set(manifest["reads"])
        for s in table["samples"]:
            with self.subTest(s["file"]):
                self.assertIn(s["kind"], read_by_cclib)
                a = read(sample(s["file"]))
                self.assertTrue(a["ok"], a.get("error"))
                r = a["result"]
                self.assertTrue(r["atoms"])
                self.assertTrue(r["program"])
                if s.get("geometry", True):
                    self.assertTrue(r["frames"])
                    self.assertTrue(all(len(f) == 3 * len(r["atoms"]) for f in r["frames"]))


if __name__ == "__main__":
    unittest.main()
