"""Tests for the workers of the plugins that read structures from pictures
(plugins/molscribe/worker.py, .../molnextr/worker.py).

Each stands alone - no code shared - and both are tested alike here. What
needs Python alone runs anywhere, and CI runs it:

    python3 -m unittest tests/test_plugin_pictures.py

A picture's pixels need numpy and Pillow - the plugins' environments have
them - and are skipped where they are not installed. The models themselves
are not run here: their trained models are a gigabyte each, downloaded only
when a plugin is added. Where a reading's atoms lie in the picture, which
end of a wedge it draws narrow and what is unsure are Meno's
(src/lib/figure/inkFit.ts), not the workers'.
"""

import base64
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"


def load(plugin):
    spec = importlib.util.spec_from_file_location(f"plugin_{plugin}", PLUGINS / plugin / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


WORKERS = {p: load(p) for p in ("molscribe", "molnextr")}

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401

    PIXELS = True
except ImportError:
    PIXELS = False


class Atoms(unittest.TestCase):
    def test_an_element_with_its_charge_or_a_label_as_drawn(self):
        for name, w in WORKERS.items():
            with self.subTest(name):
                self.assertEqual(w.atom_of("C"), {"el": "C"})
                self.assertEqual(w.atom_of("Cl"), {"el": "Cl"})
                # (stereo and hydrogens in brackets are the model's notation, not a label drawn)
                self.assertEqual(w.atom_of("[C@@H]"), {"el": "C"})
                self.assertEqual(w.atom_of("[C@]"), {"el": "C"})
                self.assertEqual(w.atom_of("[Si]"), {"el": "Si"})
                self.assertEqual(w.atom_of("[H]"), {"el": "H"})
                self.assertEqual(w.atom_of("[N+]"), {"el": "N", "charge": 1})
                self.assertEqual(w.atom_of("[O-]"), {"el": "O", "charge": -1})
                self.assertEqual(w.atom_of("[Fe+3]"), {"el": "Fe", "charge": 3})
                self.assertEqual(w.atom_of("[O--]"), {"el": "O", "charge": -2})
                # (aromatic, as SMILES writes it)
                self.assertEqual(w.atom_of("n"), {"el": "N"})
                self.assertEqual(w.atom_of("[nH]"), {"el": "N"})
                # (an isotope, with its mass number)
                self.assertEqual(w.atom_of("[2H]"), {"el": "H", "isotope": 2})
                self.assertEqual(w.atom_of("[13CH3]"), {"el": "C", "isotope": 13})
                # (a group's name, as drawn: Meno reads it as it reads a label typed)
                self.assertEqual(w.atom_of("[OTBS]"), {"label": "OTBS"})
                self.assertEqual(w.atom_of("CO2Me"), {"label": "CO2Me"})
                self.assertEqual(w.atom_of("Ph"), {"label": "Ph"})
                # (no symbol: nothing said of it)
                self.assertEqual(w.atom_of(""), {})

    def test_a_wedge_is_said_by_which_end_lies_nearer(self):
        for name, w in WORKERS.items():
            with self.subTest(name):
                self.assertEqual(w.BONDS["single"], (1, None))
                self.assertEqual(w.BONDS["double"], (2, None))
                self.assertEqual(w.BONDS["triple"], (3, None))
                # (the model's 5 at [i][j]: a wedge begun at i, j nearer; 6: j farther)
                self.assertEqual(w.BONDS["solid wedge"], (1, "up"))
                self.assertEqual(w.BONDS["dashed wedge"], (1, "down"))


class Answers(unittest.TestCase):
    def test_answers_only_what_it_reads_and_never_crashes(self):
        for name, w in WORKERS.items():
            with self.subTest(name):
                self.assertEqual(w.answer("not json"), {"ok": False, "error": "not JSON"})
                said = w.answer(json.dumps({"id": 4, "op": "run", "code": "print(1)"}))
                self.assertEqual((said["id"], said["ok"]), (4, False))
                self.assertIn("no such request", said["error"])

    def test_says_where_its_trained_model_is_missing(self):
        for name, w in WORKERS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as empty:
                w.FILES = empty
                w._model.clear()
                with self.assertRaises(RuntimeError) as e:
                    w.model()
                self.assertIn("trained model is not here", str(e.exception))

    def test_leaves_the_models_graphs_unwritten(self):
        # (the model's atoms and bonds are what Meno takes; writing its graph as SMILES - which can run without end or
        # crash the process - is left out, as the model is loaded)
        import sys
        import types

        libraries = {"molscribe": ("molscribe.interface", "MolScribe"), "molnextr": ("MolNexTR.model", "molnextr")}
        for name, w in WORKERS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as files:
                module, make = libraries[name]
                lib = types.ModuleType(module)
                lib.convert_graph_to_smiles = lambda *a, **k: self.fail("the graph was written")
                setattr(lib, make, lambda *a, **k: "the model")
                torch = types.ModuleType("torch")
                torch.device = lambda d: d
                package = types.ModuleType(module.split(".")[0])
                setattr(package, module.split(".")[1], lib)
                fakes = {"torch": torch, module.split(".")[0]: package, module: lib}
                saved = {k: sys.modules.get(k) for k in fakes}
                sys.modules.update(fakes)
                try:
                    (pathlib.Path(files) / w.WEIGHTS).write_bytes(b"")
                    w.FILES = files
                    w._model.clear()
                    self.assertEqual(w.model(), "the model")
                    self.assertIs(lib.convert_graph_to_smiles, w.graphs_unwritten)
                finally:
                    for k, v in saved.items():
                        if v is None:
                            sys.modules.pop(k, None)
                        else:
                            sys.modules[k] = v
                    w._model.clear()
                # (one empty SMILES and molfile a picture, as the model zips them with its predictions)
                smiles, molfiles, _ = w.graphs_unwritten([[(0, 0)], [(1, 1)]], [["C"], ["N"]], [[[0]], [[0]]], images=[None, None])
                self.assertEqual((smiles, molfiles), (["", ""], ["", ""]))

    @unittest.skipUnless(PIXELS, "needs numpy and Pillow")
    def test_a_see_through_picture_is_read_on_white(self):
        from PIL import Image
        import io

        for name, w in WORKERS.items():
            with self.subTest(name):
                im = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
                im.putpixel((1, 1), (0, 0, 0, 255))
                buf = io.BytesIO()
                im.save(buf, "PNG")
                rgb = w.picture_of(base64.b64encode(buf.getvalue()).decode())
                self.assertEqual(rgb.shape, (3, 4, 3))
                self.assertEqual(tuple(rgb[0, 0]), (255, 255, 255))
                self.assertEqual(tuple(rgb[1, 1]), (0, 0, 0))


class Reading(unittest.TestCase):
    """What a worker answers for a picture: its model's reading as the model gives it."""

    PREDICTED = {
        # (MolScribe's keys, and MolNexTR's: the same reading)
        "molscribe": {
            "atoms": [
                {"atom_symbol": "C", "x": 0.1, "y": 0.2, "confidence": 0.98},
                {"atom_symbol": "[2H]", "x": 0.5, "y": 0.9, "confidence": 0.4},
                {"atom_symbol": "[OTBS]", "x": 0.9, "y": 0.1},
            ],
            "bonds": [
                {"bond_type": "solid wedge", "endpoint_atoms": [0, 1], "confidence": 0.99},
                {"bond_type": "aromatic", "endpoint_atoms": [0, 2], "confidence": 0.7},
                {"bond_type": "dative", "endpoint_atoms": [1, 2]},
            ],
        },
        "molnextr": {
            "atom_sets": [
                {"atom_symbol": "C", "coords": [0.1, 0.2], "confidence": 0.98},
                {"atom_symbol": "[2H]", "coords": [0.5, 0.9], "confidence": 0.4},
                {"atom_symbol": "[OTBS]", "coords": [0.9, 0.1]},
            ],
            "bond_sets": [
                {"bond_type": "solid wedge", "endpoints": [0, 1], "confidence": 0.99},
                {"bond_type": "aromatic", "endpoints": [0, 2], "confidence": 0.7},
                {"bond_type": "dative", "endpoints": [1, 2]},
            ],
        },
    }

    def test_gives_the_models_places_symbols_bonds_and_confidence_as_it_gives_them(self):
        class Model:
            def __init__(self, p):
                self.p = p

            def predict_image(self, *a, **k):
                return self.p

        for name, w in WORKERS.items():
            with self.subTest(name):
                w._model["it"] = Model(self.PREDICTED[name])
                picture_of, w.picture_of = w.picture_of, lambda picture: None
                try:
                    got = w.op_read_picture({"picture": ""})
                finally:
                    w._model.clear()
                    w.picture_of = picture_of
                self.assertEqual(
                    got,
                    {
                        "atoms": [
                            {"x": 0.1, "y": 0.2, "el": "C", "confidence": 0.98},
                            {"x": 0.5, "y": 0.9, "el": "H", "isotope": 2, "confidence": 0.4},
                            {"x": 0.9, "y": 0.1, "label": "OTBS"},
                        ],
                        # (a bond of a type it does not know, left out)
                        "bonds": [
                            {"a": 0, "b": 1, "order": 1, "stereo": "up", "confidence": 0.99},
                            {"a": 0, "b": 2, "order": 1, "aromatic": True, "confidence": 0.7},
                        ],
                        "pad": w.PAD,
                    },
                )


if __name__ == "__main__":
    unittest.main()
