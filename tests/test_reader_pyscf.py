"""Tests for the PySCF plugin's worker (plugins/pyscf/worker.py).

They need PySCF, so they run in an environment pixi makes from the
plugin's lock:

    cp plugins/pyscf/pixi.* /tmp/pyscf-env/
    pixi install --frozen --manifest-path /tmp/pyscf-env/pixi.toml
    /tmp/pyscf-env/.pixi/envs/default/bin/python -m unittest tests/test_reader_pyscf.py

The file they read is not in the repository (docs/WORKSPACE.md, stage 3):
water.molden in calc-samples/cube/, written by PySCF itself. Those tests
are skipped where it is not there.
"""

import base64
import importlib.util
import json
import pathlib
import unittest

import numpy

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/pyscf/worker.py"
SAMPLES = ROOT / "calc-samples"
spec = importlib.util.spec_from_file_location("reader_pyscf", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
ANGSTROM_PER_BOHR = 0.529177210903


# a grid as Meno's request gives one (lib/calc/grids `gridAbout`): about water's atoms, 4 Å past them, 0.2 Å apart (ångströms)
GRID = {"origin": [-4.0, -4.76, -3.88], "axes": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], "counts": [41, 49, 44]}


def ask(op, path, **more):
    line = json.dumps({"id": 1, "op": op, "name": path.name, "text": path.read_text(encoding="utf-8"), **more})
    a = worker.answer(line)
    if not a["ok"]:
        raise AssertionError(a["error"])
    return a["result"]


def sample(rel):
    path = SAMPLES / rel
    if not path.exists():
        raise unittest.SkipTest(f"no {rel} in calc-samples/")
    return path


def result(r, id_):
    return next((x for x in r["results"] if x["id"] == id_), None)


def integral(grid, f):
    """The sum of `f` of a grid's values times each point's volume (bohr³)."""
    values = numpy.frombuffer(base64.b64decode(grid["values"]), dtype="<f4")
    step = [a[k] / ANGSTROM_PER_BOHR for k, a in enumerate(grid["axes"])]
    assert len(values) == grid["counts"][0] * grid["counts"][1] * grid["counts"][2]
    return float(f(values.astype(float)).sum() * step[0] * step[1] * step[2])


class Protocol(unittest.TestCase):
    def test_says_what_it_is_and_answers_a_failure_rather_than_crashing(self):
        # (it says its version in its ready line, not when asked)
        a = worker.answer(json.dumps({"id": 3, "op": "ping"}))
        self.assertEqual((a["id"], a["ok"]), (3, False))
        self.assertFalse(worker.answer(json.dumps({"id": 4, "op": "read", "name": "x.out", "text": "nothing\n"}))["ok"])
        self.assertFalse(worker.answer(json.dumps({"id": 5, "op": "run", "code": "1"}))["ok"])

    def test_shares_no_code_with_another_reader(self):
        self.assertNotIn("import reader_", WORKER.read_text(encoding="utf-8"))


class Molden(unittest.TestCase):
    def test_a_molden_file_its_molecule_its_orbitals_and_their_densities_each_a_promise(self):
        r = ask("read", sample("cube/water.molden"))
        self.assertEqual(r["atoms"], ["O", "H", "H"])
        self.assertEqual(len(r["frames"]), 1)
        # (every orbital, in Meno's general form - Meno names and lists them)
        orbitals = r["orbitals"]
        self.assertIsNone(result(r, "orbitals"))
        self.assertTrue(all(o["spin"] == "both" for o in orbitals))
        self.assertEqual([o["occupation"] for o in orbitals[:6]], [2, 2, 2, 2, 2, 0])
        self.assertLess(orbitals[4]["energy"], orbitals[5]["energy"])
        self.assertEqual([o["surface"]["ask"] for o in orbitals[:2]], ["orbital:0:0", "orbital:0:1"])
        self.assertEqual(result(r, "densities")["rows"][0]["surface"], {"ask": "density:total"})

    def test_an_orbital_asked_for_is_on_a_grid_its_whole_square_one(self):
        r = ask("read", sample("cube/water.molden"))
        homo = r["orbitals"][4]
        g = ask("ask", sample("cube/water.molden"), key=homo["surface"]["ask"], grid=GRID)
        self.assertTrue(g["signed"])
        self.assertEqual(g["quantity"], "orbital")
        self.assertNotIn("iso", g)
        # (on the grid Meno gave, as it gave it)
        self.assertEqual(g["counts"], GRID["counts"])
        self.assertEqual([round(v, 9) for v in g["origin"]], GRID["origin"])
        self.assertEqual([round(a[k], 9) for k, a in enumerate(g["axes"])], [0.2, 0.2, 0.2])
        self.assertAlmostEqual(integral(g, lambda v: v * v), 1.0, delta=0.02)

    def test_says_where_meno_gives_no_grid_it_can_work_on(self):
        a = worker.answer(json.dumps({"id": 2, "op": "ask", "key": "orbital:0:0", "name": "x.molden", "text": sample("cube/water.molden").read_text(encoding="utf-8")}))
        self.assertEqual((a["ok"], a["error"]), (False, "Meno's request gives no grid to work the values out on"))
        skew = {**GRID, "axes": [[0.2, 0.1, 0], [0, 0.2, 0], [0, 0, 0.2]]}
        b = worker.answer(json.dumps({"id": 3, "op": "ask", "key": "orbital:0:0", "name": "x.molden", "text": sample("cube/water.molden").read_text(encoding="utf-8"), "grid": skew}))
        self.assertEqual((b["ok"], b["error"]), (False, "Meno's request gives a grid whose axes do not run along x, y and z"))
        big = {**GRID, "counts": [91, 90, 90]}
        c = worker.answer(json.dumps({"id": 4, "op": "ask", "key": "orbital:0:0", "name": "x.molden", "text": sample("cube/water.molden").read_text(encoding="utf-8"), "grid": big}))
        self.assertEqual((c["ok"], c["error"]), (False, f"Meno's request gives a grid of more points than {90 ** 3}"))

    def test_the_density_asked_for_is_its_occupied_orbitals_squared_point_by_point(self):
        path = sample("cube/water.molden")
        g = ask("ask", path, key="density:total", grid=GRID)
        self.assertNotIn("signed", g)
        self.assertEqual(g["quantity"], "density")
        density = numpy.frombuffer(base64.b64decode(g["values"]), dtype="<f4")
        # (water: five orbitals, two electrons each - on the same grid)
        squares = sum(
            2 * numpy.frombuffer(base64.b64decode(ask("ask", path, key=f"orbital:0:{i}", grid=GRID)["values"]), dtype="<f4").astype(float) ** 2
            for i in range(5)
        )
        self.assertLess(float(numpy.abs(density - squares).max()), 1e-3 * float(density.max()))


class Species(unittest.TestCase):
    def test_a_molden_file_says_no_charge_or_multiplicity_the_format_holds_neither(self):
        r = ask("read", sample("cube/water.molden"))
        self.assertNotIn("charge", r)
        self.assertNotIn("multiplicity", r)


class OnlyMolden(unittest.TestCase):
    def test_a_file_that_is_no_molden_file_is_not_read(self):
        # (a Molden file's sections, but not its first line; and text of no file of its kind)
        for text in ("[Atoms] AU\nO 1 8 0.0 0.0 0.0\n", "nothing\n"):
            a = worker.answer(json.dumps({"id": 6, "op": "read", "name": "job.out", "text": text}))
            self.assertEqual((a["ok"], a["error"]), (False, "not a Molden file"))


if __name__ == "__main__":
    unittest.main()
