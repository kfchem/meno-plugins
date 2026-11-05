"""Tests for the Film looks and motions plugin (plugins/blender-presets), run by the Blender plugin
through the interface `preset` (docs/plugins/blender-presets.md).

    python3 -m unittest tests/test_plugin_blender_presets.py

The manifest is checked everywhere. Where Blender is installed - BLENDER names it, or it is where macOS usually has it - a
look of each is rendered on a tagged scene, its files put in the job as Meno puts them; elsewhere those tests are skipped.
"""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RENDERER = ROOT / "plugins/blender"
FOLDER = ROOT / "plugins/blender-presets"
spec = importlib.util.spec_from_file_location("plugin_blender_for_presets", RENDERER / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
MANIFEST = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
PRESETS = MANIFEST["presets"]


def blender():
    named = os.environ.get("BLENDER")
    for p in [named, "/Applications/Blender.app/Contents/MacOS/Blender", os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"), shutil.which("blender")]:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def declared_files():
    """Every file the presets section names, as Meno copies them into a job."""
    items = [x["file"] for kind in ("looks", "studios", "motions") for x in PRESETS.get(kind, [])]
    return sorted(set(items + PRESETS.get("files", [])))


class Manifest(unittest.TestCase):
    def test_it_uses_the_renderer_s_interface_in_the_versions_the_renderer_has(self):
        renderer = json.loads((RENDERER / "manifest.json").read_text(encoding="utf-8"))
        [dep] = MANIFEST["depends"]
        self.assertEqual(dep["plugin"], renderer["id"])
        self.assertEqual(renderer["interfaces"], [{"id": "preset", "version": "1.0.0"}])
        self.assertEqual(dep["interfaces"], [{"id": "preset", "versions": ">=1.0.0 <2.0.0"}])
        self.assertEqual(dep["versions"], ">=0.7.0 <0.8.0")
        self.assertTrue(renderer["version"].startswith("0.7."))

    def test_every_file_it_names_is_in_its_folder(self):
        for f in declared_files():
            self.assertTrue((FOLDER / f).is_file(), f)

    def test_the_renderer_declares_no_look_or_motion_of_its_own(self):
        renderer = json.loads((RENDERER / "manifest.json").read_text(encoding="utf-8"))
        for role in renderer["roles"]:
            self.assertNotIn("looks", role)
            self.assertNotIn("motions", role)

    def test_says_its_licence_and_carries_its_text(self):
        self.assertEqual(MANIFEST["licence"], "GPL-3.0-or-later")
        self.assertIn("GNU GENERAL PUBLIC LICENSE", (FOLDER / "LICENSE").read_text(encoding="utf-8"))
        for f in declared_files():
            if (FOLDER / f).is_file():
                self.assertTrue((FOLDER / f).read_text(encoding="utf-8").startswith("# SPDX-License-Identifier: GPL-3.0-or-later"), f)

    def test_a_look_s_own_choices_are_shown_with_it(self):
        looks = {x["id"] for x in PRESETS["looks"]}
        self.assertTrue(PRESETS["options"])
        for o in PRESETS["options"]:
            self.assertEqual(o["section"], "look", o["id"])
            self.assertEqual(o["when"]["id"], "look", o["id"])
            self.assertTrue(set(o["when"]["in"]) <= looks, o["id"])

    def test_each_studio_is_its_own_background_said_so(self):
        renderer = json.loads((RENDERER / "manifest.json").read_text(encoding="utf-8"))
        background = next(o for o in renderer["roles"][0]["options"] if o["id"] == "background")
        for st in PRESETS["studios"]:
            [fixed] = st["fixed"]
            self.assertEqual(fixed["id"], "background", st["id"])
            self.assertIn(fixed["value"], [c["value"] for c in background["choices"]], st["id"])
            self.assertTrue(fixed["why"], st["id"])

    def test_each_motion_says_its_length_and_its_loop(self):
        for m in PRESETS["motions"]:
            text = (FOLDER / m["file"]).read_text(encoding="utf-8")
            self.assertRegex(text, r"(?m)^NOMINAL = ", m["id"])
            self.assertRegex(text, r'(?m)^(LOOPS = "(cycle|seamless)"|def LOOPS\(ctx\):)', m["id"])
            # (a change of molecules keys a piece, A becoming B; a motion of one molecule keys it)
            self.assertIn("def key_change(ctx, a, b):" if "change" in m.get("shortest", {}) else "def key(ctx):", text, m["id"])


class Withdrawn(unittest.TestCase):
    """The liquid change and the ice look, withdrawn for now (docs/RENDER.md, *Decided*): not offered."""

    def test_neither_is_offered(self):
        self.assertNotIn("ice", [x["id"] for x in PRESETS["looks"]])
        self.assertNotIn("change-liquid", [x["id"] for x in PRESETS["motions"]])
        self.assertFalse((FOLDER / "motion_change_liquid.py").exists())


# one ball, tagged as Meno tags an atom, in a molecule's node, before an orthographic camera lit by a sun: glTF 2.0
def tagged_ball():
    import base64
    import math
    import struct
    rings, around = 8, 16
    pos, idx = [], []
    for r in range(rings + 1):
        t = math.pi * r / rings
        for a in range(around + 1):
            p = 2 * math.pi * a / around
            pos += [math.sin(t) * math.cos(p), math.cos(t), -math.sin(t) * math.sin(p)]
    for r in range(rings):
        for a in range(around):
            i = r * (around + 1) + a
            idx += [i, i + around + 1, i + 1, i + 1, i + around + 1, i + around + 2]
    data = struct.pack(f"<{len(pos)}f", *pos) + struct.pack(f"<{len(idx)}H", *idx)
    data += b"\0" * (-len(data) % 4)
    n = len(pos) // 3
    return {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_lights_punctual"],
        "scene": 0,
        "scenes": [{"nodes": [0, 2, 3], "extras": {"meno": {"ambientLight": 0.9}}}],
        "nodes": [
            {"name": "Molecule", "children": [1], "extras": {"meno": {"molecule": 0, "bonds": []}}},
            {"mesh": 0, "scale": [0.8, 0.8, 0.8], "extras": {"meno": {"atom": 0, "element": "C"}}},
            {"camera": 0, "translation": [0, 0, 5]},
            {"extensions": {"KHR_lights_punctual": {"light": 0}}, "rotation": [-0.3647052, 0.2798481, 0.1159169, 0.8804762]},
        ],
        "cameras": [{"type": "orthographic", "orthographic": {"xmag": 2, "ymag": 1.5, "znear": 0.01, "zfar": 10}}],
        "extensions": {"KHR_lights_punctual": {"lights": [{"type": "directional", "intensity": 1}]}},
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.2, 0.2, 1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "meshes": [{"name": "ball", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": [-1, -1, -1], "max": [1, 1, 1]},
            {"bufferView": 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos) * 4},
            {"buffer": 0, "byteOffset": len(pos) * 4, "byteLength": len(idx) * 2},
        ],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
    }


def run_job(d, entry, options, scene=None):
    """The renderer's job for one entry, its preset's files put where Meno puts them, run by Blender: its output."""
    (pathlib.Path(d) / "scene.gltf").write_text(json.dumps(scene or tagged_ball()), encoding="utf-8")
    said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "render", "entries": [entry], "options": options}))
    job = json.loads(said)["result"]["jobs"][0] if isinstance(said, str) else said["result"]["jobs"][0]
    for f in job["files"]:
        (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
    # (those it names that are there: Manifest says where one is not)
    for f in [f for f in declared_files() if (FOLDER / f).is_file()]:
        to = pathlib.Path(d, "presets", MANIFEST["id"], f)
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FOLDER / f, to)
    return subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=900)


SCENE = {"scene": "scene.gltf", "film": False, "width": 96, "height": 72, "frames": 1, "fps": 1}


@unittest.skipUnless(blender(), "Blender is not installed here")
class LooksRender(unittest.TestCase):
    def test_every_look_renders_meno_s_ball(self):
        for look in PRESETS["looks"]:
            with self.subTest(look=look["id"]), tempfile.TemporaryDirectory() as d:
                entry = {**SCENE, "presets": [{"interface": "preset", "from": MANIFEST["id"], "look": {"id": look["id"], "file": look["file"]}, "values": {}}]}
                run = run_job(d, entry, {"samples": 2, "background": "white", "device": "cpu", "denoise": False})
                self.assertEqual(run.returncode, 0, run.stdout[-2500:] + run.stderr[-2500:])
                self.assertTrue((pathlib.Path(d) / "picture.png").is_file())


def opaque_share(png):
    """The share of a picture's pixels that are opaque, as Blender reads it."""
    code = "import bpy,sys; p=list(bpy.data.images.load(sys.argv[-1]).pixels); a=p[3::4]; print('SHARE', sum(1 for x in a if x > 0.99) / len(a))"
    run = subprocess.run([blender(), "-b", "--factory-startup", "--python-expr", code, "--", str(png)], capture_output=True, text=True, timeout=300)
    return float(next(l.split()[1] for l in run.stdout.splitlines() if l.startswith("SHARE")))


@unittest.skipUnless(blender(), "Blender is not installed here")
class StudiosRender(unittest.TestCase):
    def test_every_studio_renders_round_meno_s_ball_on_each_background(self):
        for studio in PRESETS["studios"]:
            for bg in ("transparent", "white", "black", "grey"):
                with self.subTest(studio=studio["id"], background=bg), tempfile.TemporaryDirectory() as d:
                    entry = {**SCENE, "presets": [{"interface": "preset", "from": MANIFEST["id"], "studio": {"id": studio["id"], "file": studio["file"]}, "values": {}}]}
                    run = run_job(d, entry, {"samples": 2, "background": bg, "device": "cpu", "denoise": False})
                    self.assertEqual(run.returncode, 0, run.stdout[-2500:] + run.stderr[-2500:])
                    self.assertTrue((pathlib.Path(d) / "picture.png").is_file())
                    # (the studio is the backdrop chosen: the picture is its backdrop all over, on a transparent background too)
                    self.assertGreater(opaque_share(pathlib.Path(d) / "picture.png"), 0.99)
                    # (and Blender's log says nothing is to go: no deprecated part of its API is used)
                    self.assertNotIn("DeprecationWarning", run.stdout + run.stderr)


def two_balls():
    """Two molecules of one ball each, tagged as Meno tags them: a change's first and next."""
    g = tagged_ball()
    g["nodes"].append({"name": "Molecule 2", "translation": [3, 0, 0], "children": [5], "extras": {"meno": {"molecule": 1, "bonds": []}}})
    g["nodes"].append({"mesh": 0, "scale": [0.5, 0.5, 0.5], "extras": {"meno": {"atom": 0, "element": "N"}}})
    g["scenes"][0]["nodes"].append(4)
    return g


@unittest.skipUnless(blender(), "Blender is not installed here")
class MotionsRender(unittest.TestCase):
    def test_every_motion_makes_a_short_film_once_and_as_a_loop(self):
        for m in PRESETS["motions"]:
            change = "change" in m.get("shortest", {})
            for mode in ("once", "loop"):
                with self.subTest(motion=m["id"], mode=mode), tempfile.TemporaryDirectory() as d:
                    entry = {**SCENE, "film": True, "width": 64, "height": 48, "frames": 6, "fps": 6, "mode": mode,
                             "presets": [{"interface": "preset", "from": MANIFEST["id"], "motion": {"id": m["id"], "file": m["file"]}, "values": {}}]}
                    run = run_job(d, entry, {"samples": 1, "background": "black", "device": "cpu", "denoise": False}, scene=two_balls() if change else None)
                    self.assertEqual(run.returncode, 0, run.stdout[-2500:] + run.stderr[-2500:])
                    self.assertTrue((pathlib.Path(d) / "film.mp4").is_file())
                    self.assertNotIn("DeprecationWarning", run.stdout + run.stderr)
