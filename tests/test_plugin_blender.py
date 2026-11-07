"""Tests for the Blender plugin's worker (plugins/blender/worker.py).

The worker needs Python alone - Blender is run by Meno, not by it - so
these run anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_blender.py

What Blender is given is checked against Blender's own manual
(docs.blender.org, Blender 5.2, "Command Line Arguments"). Where Blender is
installed - BLENDER names it, or it is where macOS usually has it - the
render script is run too, on a scene of one ball, and what it made looked
at; elsewhere that test is skipped.
"""

import importlib.util
import json
import math
import re
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDER = ROOT / "plugins/blender"
spec = importlib.util.spec_from_file_location("plugin_blender", FOLDER / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

SCENE = {"scene": "scene.gltf", "film": False, "width": 320, "height": 240, "frames": 1, "fps": 1}


def ask(op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    return said


class Prepare(unittest.TestCase):
    def test_runs_blender_in_the_background_with_its_script_and_settings(self):
        said = ask("prepare", step="render", entries=[SCENE], options={}, cores=4)
        self.assertTrue(said["ok"], said)
        [job] = said["result"]["jobs"]
        self.assertEqual(job["program"], "blender")
        self.assertEqual(job["entries"], [0])
        args = job["args"]
        # (as Blender's manual has them: in the background, its factory settings, no sound, no network, the threads given, the script, then its settings after --)
        self.assertEqual(args[:4], ["-b", "--factory-startup", "-noaudio", "--offline-mode"])
        self.assertEqual(args[args.index("-t") + 1], "4")
        self.assertEqual(args[args.index("--python-exit-code") + 1], "1")
        self.assertLess(args.index("-P"), args.index("--"))
        self.assertEqual(args[args.index("-P") + 1], "render.py")
        self.assertEqual(args[-2:], ["--", "render.json"])
        files = {f["name"]: f["text"] for f in job["files"]}
        self.assertEqual(files["render.py"], (FOLDER / "render.py").read_text(encoding="utf-8"))
        settings = json.loads(files["render.json"])
        self.assertEqual(settings["out"], "picture.png")
        self.assertEqual((settings["width"], settings["height"], settings["engine"], settings["samples"]), (320, 240, "cycles", 128))
        self.assertEqual(job["reads"], [])

    def test_the_looks_shading_and_outlines_are_written_as_settings_from_the_entry(self):
        entry = {**SCENE, "shading": "toon", "band": 0.3, "outlines": "mixed", "outline": 4, "outline_colour": "white"}
        said = ask("prepare", step="render", entries=[entry], options={})
        settings = json.loads({f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}["render.json"])
        self.assertEqual((settings["shading"], settings["band"], settings["outlines"], settings["outline"], settings["outline_colour"]), ("toon", 0.3, "mixed", 4, "white"))
        # (lit and not outlined, as before, where the entry says nothing)
        plain = ask("prepare", step="render", entries=[SCENE], options={})
        plain = json.loads({f["name"]: f["text"] for f in plain["result"]["jobs"][0]["files"]}["render.json"])
        self.assertEqual((plain["shading"], plain["outlines"]), ("lit", "none"))

    def test_a_film_is_written_as_one(self):
        said = ask("prepare", step="render", entries=[{**SCENE, "film": True, "frames": 240, "fps": 30}], options={"quality": "lossless", "background": "black"})
        settings = json.loads({f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}["render.json"])
        self.assertEqual((settings["out"], settings["frames"], settings["fps"], settings["quality"], settings["background"]), ("film.mp4", 240, 30, "lossless", "black"))

    def test_the_data_folder_the_worker_runs_from_goes_to_the_job_for_what_is_kept(self):
        import types
        fake = types.SimpleNamespace(prefix="/somewhere/uv/plugin-blender/venv")
        real = worker.sys
        try:
            worker.sys = fake
            home = worker.plugin_home()
        finally:
            worker.sys = real
        # (only where that folder exists: a path that is not there gives none)
        self.assertIsNone(home)
        with tempfile.TemporaryDirectory() as d:
            venv = os.path.join(d, "plugin-blender", "venv")
            os.makedirs(venv)
            try:
                worker.sys = types.SimpleNamespace(prefix=venv)
                self.assertEqual(worker.plugin_home(), os.path.join(d, "plugin-blender"))
                said = ask("prepare", step="render", entries=[SCENE], options={})
            finally:
                worker.sys = real
            settings = json.loads({f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}["render.json"])
            self.assertEqual(settings["probe_dir"], os.path.join(d, "plugin-blender"))

    def test_the_device_is_a_three_way_choice_automatic_by_default(self):
        m = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        device = next(o for o in m["roles"][0]["options"] if o["id"] == "device")
        self.assertEqual([c["value"] for c in device["choices"]], ["auto", "gpu", "cpu"])
        self.assertEqual(device["default"], "auto")
        for chosen in ("auto", "gpu", "cpu"):
            said = ask("prepare", step="render", entries=[SCENE], options={"device": chosen})
            settings = json.loads({f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}["render.json"])
            self.assertEqual(settings["device"], chosen)
        none = ask("prepare", step="render", entries=[SCENE], options={})
        self.assertEqual(json.loads({f["name"]: f["text"] for f in none["result"]["jobs"][0]["files"]}["render.json"])["device"], "auto")
        self.assertFalse(ask("prepare", step="render", entries=[SCENE], options={"device": "tpu"})["ok"])

    def test_refuses_what_it_cannot_do(self):
        for entry, options in [
            ({**SCENE, "scene": "../scene.gltf"}, {}),
            ({**SCENE, "scene": "scene.blend"}, {}),
            ({**SCENE, "width": 0}, {}),
            ({**SCENE, "width": 100000}, {}),
            ({**SCENE, "frames": 2.5}, {}),
            (SCENE, {"engine": "workbench"}),
            (SCENE, {"samples": 0}),
            (SCENE, {"background": "#ff0000"}),
            (SCENE, {"denoise": "yes"}),
            (SCENE, {"key": float("nan")}),
            (SCENE, {"softness": 120}),
            ({**SCENE, "shading": "sketch"}, {}),
            ({**SCENE, "outlines": "dotted"}, {}),
            ({**SCENE, "outline": 0}, {}),
            ({**SCENE, "outline": 50}, {}),
            ({**SCENE, "band": 0}, {}),
            ({**SCENE, "outline_colour": "#000000"}, {}),
        ]:
            said = ask("prepare", step="render", entries=[entry], options=options)
            self.assertFalse(said["ok"], (entry, options))
            self.assertTrue(said["error"].startswith("Blender: "))
        self.assertFalse(ask("prepare", step="optimise", entries=[SCENE], options={})["ok"])
        self.assertFalse(ask("prepare", step="render", entries=[], options={})["ok"])


class Preset(unittest.TestCase):
    """A preset plugin's look, studio or motion (docs/blender-presets.md): passed to render.py as Meno hands it, its
    shape checked; the renderer declares no look or motion of its own."""

    def entry(self, **preset):
        return {**SCENE, "presets": [{"interface": "preset", "from": "some-presets", **preset}]}

    def test_is_passed_to_render_py_with_its_options(self):
        said = ask("prepare", step="render", entries=[self.entry(look={"id": "gold", "file": "looks/gold.py"}, values={"wood": "oak"})], options={})
        files = {f["name"]: f["text"] for f in said["result"]["jobs"][0]["files"]}
        settings = json.loads(files["render.json"])
        self.assertEqual(settings["presets"], [{"interface": "preset", "from": "some-presets", "look": {"id": "gold", "file": "looks/gold.py"}, "values": {"wood": "oak"}}])
        self.assertIn("presets.py", files)

    def test_one_of_another_shape_is_refused(self):
        for bad in [{"interface": "other", "from": "x"}, {"interface": "preset", "from": "Not An Id"},
                    {"interface": "preset", "from": "x", "look": {"file": "../elsewhere.py"}},
                    {"interface": "preset", "from": "x", "motion": {"file": "motions/m.sh"}},
                    {"interface": "preset", "from": "x", "values": {"a": [1]}}]:
            said = ask("prepare", step="render", entries=[{**SCENE, "presets": [bad]}], options={})
            self.assertFalse(said["ok"], bad)
        # (one look among them at most, and each plugin once)
        two = [{"interface": "preset", "from": "a", "look": {"file": "a.py"}}, {"interface": "preset", "from": "b", "look": {"file": "b.py"}}]
        self.assertFalse(ask("prepare", step="render", entries=[{**SCENE, "presets": two}], options={})["ok"])
        again = [{"interface": "preset", "from": "a", "look": {"file": "a.py"}}, {"interface": "preset", "from": "a", "motion": {"file": "m.py"}}]
        self.assertFalse(ask("prepare", step="render", entries=[{**SCENE, "presets": again}], options={})["ok"])

    def test_the_renderer_declares_no_look_or_motion(self):
        m = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["interfaces"], [{"id": "preset", "version": "1.0.0"}])
        for role in m["roles"]:
            self.assertNotIn("looks", role)
            self.assertNotIn("motions", role)


class Collect(unittest.TestCase):
    def test_says_what_it_made(self):
        said = ask("collect", step="render", entries=[SCENE, {**SCENE, "film": True}], options={}, files={}, log="", ended="done")
        self.assertEqual(said["result"], {"made": [
            {"entry": 0, "file": "picture.png", "media": "image/png"},
            {"entry": 1, "file": "film.mp4", "media": "video/mp4"},
        ]})

    def test_says_it_rendered_on_the_processor_only_where_the_graphics_processor_was_asked_for_and_not_found(self):
        fell = "Rendering on the processor (no graphics processor Blender can use)\nprogress 100/100\n"
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=fell, ended="done")
        self.assertEqual(said["result"]["note"], "Rendered on the processor")
        for log in ("Rendering on the graphics processor\n", "Rendering on the processor\n", ""):
            said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=log, ended="done")
            self.assertNotIn("note", said["result"])

    def test_says_why_it_failed(self):
        log = "Blender 5.2.2\nRead blend\nError: Python: Traceback ...\nrender.py: the scene has no camera\nBlender quit\n"
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=log, ended="failed")
        self.assertEqual(said["result"], {"why": "The scene has no camera"})

    def test_says_what_python_raised_not_blenders_own_last_words(self):
        # (as Blender 5.2.2 ends a script that raised, and - after a large render - what it did not free)
        tail = "Error: script failed, file: 'render.py', exiting.\nBlender 5.2.2 LTS\n\nBlender quit\n"
        leak = "Error: Not freed memory blocks: 21897, total unfreed memory 8815.848637 MB\n"
        raised = "Traceback (most recent call last):\n  File \"motion_assemble.py\", line 12, in <module>\nTypeError: 'NoneType' object is not subscriptable\n"
        for log in (raised + tail, raised + tail + leak):
            said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=log, ended="failed")
            self.assertEqual(said["result"], {"why": "TypeError: 'NoneType' object is not subscriptable"})
        # (a message of several lines, all of it, as one; an exception raised while another was handled, the last)
        many = "Traceback (most recent call last):\nKeyError: 'col'\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\nValueError: the scene's molecule has no atoms\nsee its entry: entries[0]\n"
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=many + tail + leak, ended="failed")
        self.assertEqual(said["result"], {"why": "ValueError: the scene's molecule has no atoms see its entry: entries[0]"})
        # (the desk's probe: Blender's progress after what was raised is not part of it)
        probe = ("Traceback (most recent call last):\nRuntimeError: a stand-in\n"
                 "Fra:1 Mem:120.5M (Peak 140.2M) | Time:00:01.20 | Rendering 1 / 128 samples\n"
                 "Saved: '/Users/someone/Meno/jobs/x/work/frames/frame_0001.png'\n"
                 "Time: 00:01.84 (Saving: 00:00.03)\n"
                 "03:30.912  render           | Saved: 'frame_0002.png'\n"
                 "Info: Saved \"frame_0002.png\"\nWarning: a stand-in\nprogress 4/100\nRendering on the processor\n")
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=probe + tail + leak, ended="failed")
        self.assertEqual(said["result"], {"why": "RuntimeError: a stand-in"})
        # (Blender's own error, where Python raised none: said; its last words alone, said as they are)
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log="Error: Cannot read file 'scene.gltf'\n" + tail + leak, ended="failed")
        self.assertEqual(said["result"], {"why": "Error: Cannot read file 'scene.gltf'"})
        said = ask("collect", step="render", entries=[SCENE], options={}, files={}, log=tail.split("\n")[0], ended="failed")
        self.assertEqual(said["result"], {"why": "Error: script failed, file: 'render.py', exiting."})


class RenderScriptNames(unittest.TestCase):
    """render.py runs only in Blender, so a name it never defines fails only there, at the line that uses it - on a computer
    whose path through it no test here takes (a motion's film where no graphics processor can be used, say)."""

    def test_render_py_uses_no_name_it_never_defines(self):
        import ast
        import builtins
        tree = ast.parse((FOLDER / "render.py").read_text(encoding="utf-8"))
        defined = set(dir(builtins)) | {"__file__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        self.assertEqual(sorted(used - defined), [])

    def test_its_fallback_lines_are_those_the_worker_reads(self):
        src = (FOLDER / "render.py").read_text(encoding="utf-8")
        self.assertIn(f"FELL_BACK = {worker.FELL_BACK!r}".replace("'", '"'), src)
        self.assertIn(f"FASTER = {worker.FASTER!r}".replace("'", '"'), src)


class FilmStrips(unittest.TestCase):
    """How the frames a film plays become the sequencer's image strips (render.py `strip_runs`): a strip reads every file from
    its first file's folder, so each run from one folder is a strip of its own."""

    @classmethod
    def setUpClass(cls):
        import ast
        tree = ast.parse((FOLDER / "render.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "strip_runs")
        ns = {"os": os}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "render.py", "exec"), ns)
        cls.runs = staticmethod(ns["strip_runs"])

    def test_every_film_frame_is_its_own_file_across_folders_of_the_same_names(self):
        # (a film of pieces from three folders holding the very same names, one of them played twice and reversed: piece 1
        # forward, piece 2, piece 3's backward, its forward from its second, piece 4, piece 5's backward)
        d = lambda name: os.path.join("job", "frames", name)
        f = lambda folder, n: os.path.join(d(folder), f"f_{n:04d}.png")
        frames = ([f("melt-01", n) for n in (1, 2, 3)] + [f("reform-01", n) for n in (1, 2)] + [f("melt-02", n) for n in (3, 2, 1)]
                  + [f("melt-02", n) for n in (2, 3)] + [f("reform-02", n) for n in (1, 2)] + [f("melt-03", n) for n in (3, 2, 1)])
        runs = self.runs(frames)
        # (each film frame, as its strip plays it: its strip's folder, the name at its place in the strip)
        played = []
        for folder, names, start in runs:
            self.assertEqual(start, len(played) + 1)
            played += [os.path.join(folder, name) for name in names]
        self.assertEqual(played, frames)
        # (one run for melt 2 played backward then forward: one folder, its names in the order played)
        self.assertEqual([(os.path.basename(r[0]), r[1]) for r in runs][2], ("melt-02", ["f_0003.png", "f_0002.png", "f_0001.png", "f_0002.png", "f_0003.png"]))
        self.assertEqual(len(runs), 5)

    def test_a_film_of_one_folder_is_one_strip(self):
        frames = [os.path.join("job", "frames", f"f_{n:04d}.png") for n in (1, 2, 3, 2)]
        self.assertEqual(self.runs(frames), [(os.path.join("job", "frames"), ["f_0001.png", "f_0002.png", "f_0003.png", "f_0002.png"], 1)])


class ProgressStatistics(unittest.TestCase):
    """Blender's own time taken and time remaining, read off a statistics line (render.py `Progress.clocks`): the bar's smoothing."""

    @classmethod
    def setUpClass(cls):
        import re as _re
        src = (FOLDER / "render.py").read_text(encoding="utf-8")
        start = src.index("CLOCK = ")
        end = src.index("class Progress:")
        ns = {"re": _re}
        exec(src[start:end], ns)  # the two patterns, as written
        cls.clock = ns["CLOCK"]

    def test_reads_the_time_taken_and_remaining(self):
        text = "Time: 00:00.80 | Remaining: 00:35.42 | Mem: 20.1M | Sample 1/128"
        found = {m.group(1): (int(m.group(2) or 0) * 60 + int(m.group(3))) * 60 + float(m.group(4)) for m in self.clock.finditer(text)}
        self.assertAlmostEqual(found["Time"], 0.8)
        self.assertAlmostEqual(found["Remaining"], 35.42)

    def test_a_line_with_remaining_alone_gives_no_taken_and_the_class_measures_it_itself(self):
        import re as _re
        import time as _time
        import threading as _threading
        src = (FOLDER / "render.py").read_text(encoding="utf-8")
        start = src.index("CLOCK = ")
        end = src.index("def render(scene, s, folder):")
        said = []
        ns = {"re": _re, "time": _time, "threading": _threading, "say": said.append}
        exec(src[start:end], ns)
        progress = ns["Progress"](1)
        # (statistics mid-render, as Blender 5.2.2 prints them on a graphics processor: Remaining, never Time)
        self.assertEqual(progress.clocks("Remaining: 00:41.13 | Mem: 254M | Sample 1/128"), (None, 41.13))
        progress.begin_frame()
        _time.sleep(0.3)
        progress.stats("Remaining: 00:41.13 | Mem: 254M | Sample 1/128")
        self.assertIsNotNone(progress.anchor)
        taken = progress.anchor[1]
        self.assertGreater(taken, 0.25)
        self.assertAlmostEqual(progress.anchor[2], taken + 41.13, places=1)
        # (and the ticker, between the lines, goes on by the clock, never back, never to the end)
        _time.sleep(1.2)
        progress.end()
        marks = [int(m.group(1)) for line in said for m in [_re.match(r"progress (\d+)/100", line)] if m]
        self.assertEqual(marks, sorted(marks))
        self.assertLess(max(marks or [0]), 100)

    def test_reads_hours(self):
        m = [m for m in self.clock.finditer("Remaining: 1:02:03.5")][0]
        self.assertEqual((m.group(2), m.group(3), m.group(4)), ("1", "02", "03.5"))


class Manifest(unittest.TestCase):
    def test_says_its_licence_and_carries_its_text(self):
        m = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["licence"], "GPL-3.0-or-later")
        self.assertIn("GNU GENERAL PUBLIC LICENSE", (FOLDER / "LICENSE").read_text(encoding="utf-8"))
        self.assertIn("Version 3", (FOLDER / "LICENSE").read_text(encoding="utf-8"))
        for f in ("worker.py", "render.py", "presets.py"):
            self.assertTrue((FOLDER / f).read_text(encoding="utf-8").startswith("# SPDX-License-Identifier: GPL-3.0-or-later"), f)
        # (the options it declares: each id the worker reads)
        ids = {o["id"] for o in m["roles"][0]["options"]}
        self.assertEqual(ids, {"engine", "samples", "denoise", "device", "background", "colours", "key", "ambient", "softness", "quality"})
        # (each in the part of Render's card it belongs to: the background and the lights with the scene, the rest with how it
        # is rendered)
        sections = {o["id"]: o.get("section") for o in m["roles"][0]["options"]}
        self.assertEqual({k for k, v in sections.items() if v == "scene"}, {"background", "key", "ambient", "softness"})
        self.assertEqual({k for k, v in sections.items() if v == "render"}, {"engine", "samples", "denoise", "device", "colours", "quality"})



def blender():
    named = os.environ.get("BLENDER")
    for p in [named, "/Applications/Blender.app/Contents/MacOS/Blender", os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"), shutil.which("blender")]:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# one white ball before an orthographic camera, lit by a sun: glTF 2.0, written by hand
BALL = {
    "asset": {"version": "2.0"},
    "extensionsUsed": ["KHR_lights_punctual"],
    "scene": 0,
    "scenes": [{"nodes": [0, 1, 2], "extras": {"meno": {"ambientLight": 0.9}}}],
    "nodes": [
        {"mesh": 0},
        {"camera": 0, "translation": [0, 0, 5]},
        {"extensions": {"KHR_lights_punctual": {"light": 0}}, "rotation": [-0.3647052, 0.2798481, 0.1159169, 0.8804762]},
    ],
    "cameras": [{"type": "orthographic", "orthographic": {"xmag": 2, "ymag": 1.5, "znear": 0.01, "zfar": 10}}],
    "extensions": {"KHR_lights_punctual": {"lights": [{"type": "directional", "intensity": 1}]}},
    "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 1}}],
}


def ball_mesh():
    """A coarse sphere of radius 1, its buffer embedded."""
    import base64
    import math
    import struct
    rings, around = 8, 16
    pos = []
    for r in range(rings + 1):
        t = math.pi * r / rings
        for a in range(around + 1):
            p = 2 * math.pi * a / around
            pos += [math.sin(t) * math.cos(p), math.cos(t), -math.sin(t) * math.sin(p)]
    idx = []
    row = around + 1
    for r in range(rings):
        for a in range(around):
            q = r * row + a
            idx += [q, q + row, q + 1, q + 1, q + row, q + row + 1]
    data = struct.pack(f"<{len(pos)}f", *pos) + struct.pack(f"<{len(idx)}H", *idx)
    n = len(pos) // 3
    return {
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 0}, "indices": 1, "material": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": [-1, -1, -1], "max": [1, 1, 1]},
            {"bufferView": 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4 * len(pos)}, {"buffer": 0, "byteOffset": 4 * len(pos), "byteLength": 2 * len(idx)}],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
    }


# a small molecule for the motions: methylamine (C, N and five H), its atoms where Meno writes them, in angstroms
METHYLAMINE = {"atoms": [{"el": "C", "p": [-0.7, 0.0, 0.0]}, {"el": "N", "p": [0.75, 0.0, 0.0]}, {"el": "H", "p": [-1.1, 1.0, 0.0]},
                         {"el": "H", "p": [-1.1, -0.5, 0.87]}, {"el": "H", "p": [-1.1, -0.5, -0.87]}, {"el": "H", "p": [1.1, 0.95, 0.1]}, {"el": "H", "p": [1.1, -0.5, 0.8]}],
               "bonds": [{"a": 0, "b": 1, "order": 1}, {"a": 0, "b": 2, "order": 1}, {"a": 0, "b": 3, "order": 1}, {"a": 0, "b": 4, "order": 1}, {"a": 1, "b": 5, "order": 1}, {"a": 1, "b": 6, "order": 1}]}


# and another, for a change of molecules: methanol (C, O and four H)
METHANOL = {"atoms": [{"el": "C", "p": [-0.7, 0.0, 0.0]}, {"el": "O", "p": [0.7, 0.0, 0.0]}, {"el": "H", "p": [-1.1, 1.0, 0.0]},
                      {"el": "H", "p": [-1.1, -0.5, 0.87]}, {"el": "H", "p": [-1.1, -0.5, -0.87]}, {"el": "H", "p": [1.05, 0.9, 0.0]}],
            "bonds": [{"a": 0, "b": 1, "order": 1}, {"a": 0, "b": 2, "order": 1}, {"a": 0, "b": 3, "order": 1}, {"a": 0, "b": 4, "order": 1}, {"a": 1, "b": 5, "order": 1}]}



@unittest.skipUnless(blender(), "Blender is not installed here")
class Renders(unittest.TestCase):
    def test_renders_a_picture_on_white(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "scene.gltf").write_text(json.dumps({**BALL, **ball_mesh()}), encoding="utf-8")
            said = ask("prepare", step="render", entries=[SCENE], options={"samples": 4, "background": "white", "device": "cpu"})
            job = said["result"]["jobs"][0]
            for f in job["files"]:
                (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
            run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            self.assertIn("progress 100/100", run.stdout)
            # (never back: each line at least the one before)
            marks = [int(m) for m in re.findall(r"^progress (\d+)/100$", run.stdout, re.M)]
            self.assertEqual(marks, sorted(marks))
            self.assertGreater(len(marks), 2, "the samples moved it")
            png = (pathlib.Path(d) / "picture.png").read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            # (its size: the PNG's header, width and height)
            self.assertEqual((int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")), (320, 240))


    def test_the_cartoon_look_outlines_a_ball_and_lights_it_in_one_band(self):
        # (white background; a ball of the base colour red, lit by Meno's key light: pixels of its outline are black, its lit part flat red, the rest of the ball darker)
        with tempfile.TemporaryDirectory() as d:
            scene = {**BALL, **ball_mesh()}
            scene["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1], "metallicFactor": 0, "roughnessFactor": 1}}]
            (pathlib.Path(d) / "scene.gltf").write_text(json.dumps(scene), encoding="utf-8")
            entry = {**SCENE, "shading": "toon", "outlines": "mixed", "outline": 4}
            said = ask("prepare", step="render", entries=[entry], options={"samples": 4, "background": "white", "device": "cpu"})
            job = said["result"]["jobs"][0]
            for f in job["files"]:
                (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
            run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            png = (pathlib.Path(d) / "picture.png").read_bytes()
            width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
            row = [png_pixel(png, x, height // 2) for x in range(width)]
            black = [p for p in row if max(p[:3]) < 40]
            self.assertTrue(black, "no outline pixel on the middle row")
            # (a ball of one colour in few bands: three colours - the white behind, the outline's black, the red's one band - make up
            # nearly the whole row, but for the edges' anti-aliasing)
            from collections import Counter
            top = sum(n for _, n in Counter(tuple(p[:3]) for p in row).most_common(3))
            self.assertGreater(top, 0.8 * width)

    def run_job(self, d, options, probe_file=None):
        (pathlib.Path(d) / "scene.gltf").write_text(json.dumps({**BALL, **ball_mesh()}), encoding="utf-8")
        said = ask("prepare", step="render", entries=[{**SCENE, "width": 160, "height": 120}], options={"samples": 4, "background": "white", **options})
        job = said["result"]["jobs"][0]
        files = {f["name"]: f["text"] for f in job["files"]}
        settings = json.loads(files["render.json"])
        if probe_file:
            settings["probe_file"] = probe_file
        files["render.json"] = json.dumps(settings)
        for name, text in files.items():
            (pathlib.Path(d) / name).write_text(text, encoding="utf-8")
        run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=900)
        self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
        return run.stdout

    def test_the_automatic_choice_times_the_two_once_and_keeps_the_faster(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as k:
            cache = os.path.join(k, "device-probe.json")
            first = self.run_job(d, {"device": "auto"}, cache)
            if "no graphics processor Blender can use" in first:
                self.skipTest("no graphics processor here to time against the processor")
            self.assertIn("Checking which is faster...", first)
            kept = json.load(open(cache))
            self.assertEqual(len(kept), 1)
            entry = next(iter(kept.values()))
            self.assertIn(entry["device"], ("gpu", "cpu"))
            self.assertGreater(entry["cpu"], 0)
            self.assertGreater(entry["gpu"], 0)
            # (the second render takes what was kept, and does not time again)
            second = self.run_job(d, {"device": "auto"}, cache)
            self.assertNotIn("Checking which is faster...", second)
            self.assertIn("as timed before", second)
            expected = "Rendering on the graphics processor" if entry["device"] == "gpu" else "Rendering on the processor (the faster one here)"
            self.assertIn(expected, second)

    def motion_film(self, motion, molecules, device, probe_file=None):
        """A Loop of a motion, tiny, as Meno runs it: its log."""
        with tempfile.TemporaryDirectory() as m:
            scene = {**BALL, **ball_mesh(), "scenes": [{"nodes": [0, 1, 2], "extras": {"meno": {"ambientLight": 0.9, "molecules": molecules}}}]}
            (pathlib.Path(m) / "scene.gltf").write_text(json.dumps(scene), encoding="utf-8")
            entry = {"scene": "scene.gltf", "film": True, "width": 64, "height": 36, "frames": 6, "fps": 1, "motion": motion, "mode": "loop"}
            said = ask("prepare", step="render", entries=[entry], options={"samples": 4, "background": "white", "device": device})
            job = said["result"]["jobs"][0]
            files = {f["name"]: f["text"] for f in job["files"]}
            if probe_file:
                files["render.json"] = json.dumps({**json.loads(files["render.json"]), "probe_file": probe_file})
            for name, text in files.items():
                (pathlib.Path(m) / name).write_text(text, encoding="utf-8")
            run = subprocess.run([blender(), *job["args"]], cwd=m, capture_output=True, text=True, timeout=900)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            return run.stdout

    def test_the_processor_choice_never_times_and_says_so(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as k:
            cache = os.path.join(k, "device-probe.json")
            out = self.run_job(d, {"device": "cpu"}, cache)
            self.assertNotIn("Checking which is faster", out)
            self.assertIn("Rendering on the processor", out)
            self.assertFalse(os.path.exists(cache))

    def made(self, d, entry):
        """The renderer's job for one entry on the ball's scene, run by Blender in `d`."""
        (pathlib.Path(d) / "scene.gltf").write_text(json.dumps({**BALL, **ball_mesh()}), encoding="utf-8")
        said = ask("prepare", step="render", entries=[entry], options={"samples": 2, "background": "white", "device": "cpu"})
        job = said["result"]["jobs"][0]
        for f in job["files"]:
            (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
        run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])

    def read_back(self, code, path):
        out = subprocess.run([blender(), "-b", "--factory-startup", "--python-expr", code, "--", str(path)], capture_output=True, text=True, timeout=300)
        return next(l for l in out.stdout.splitlines() if l.startswith("READ")).split()[1:]

    def test_a_film_of_an_odd_side_is_a_video_a_pixel_more_its_frames_as_asked(self):
        size = "import bpy,sys; c=bpy.data.movieclips.load(sys.argv[-1]); print('READ', c.size[0], c.size[1])"
        for (w, h), video in (((65, 37), ["66", "38"]), ((64, 36), ["64", "36"])):
            with self.subTest(size=(w, h)), tempfile.TemporaryDirectory() as d:
                self.made(d, {**SCENE, "film": True, "frames": 3, "fps": 3, "width": w, "height": h})
                # (the video H.264 can carry: even sides - an even side as it was - read back by Blender)
                self.assertEqual(self.read_back(size, pathlib.Path(d) / "film.mp4"), video)

    def test_a_padded_films_frames_lie_unscaled_in_its_corner(self):
        # (the video's first frame against the rendered frame over white: closest where neither is moved - not a pixel
        # across, nor up)
        code = """
import bpy, sys, glob
d = sys.argv[-1]
im = bpy.data.images.load(sorted(glob.glob(d + '/frames/*.png'))[0]); w, h = im.size; f = list(im.pixels)
mv = bpy.data.images.load(d + '/film.mp4'); W, H = mv.size; v = list(mv.pixels)
def off(dx, dy):
    t = n = 0
    for y in range(h):
        for x in range(w):
            if x + dx < W and y + dy < H:
                i = 4 * (y * w + x); a = f[i + 3]
                t += abs(v[4 * ((y + dy) * W + x + dx)] - (f[i] * a + 1 - a)); n += 1
    return t / n
print('READ', off(0, 0), off(1, 0), off(0, 1))
"""
        with tempfile.TemporaryDirectory() as d:
            self.made(d, {**SCENE, "film": True, "frames": 2, "fps": 2, "width": 129, "height": 73})
            here, across, up = map(float, self.read_back(code, d))
            # (by far: a frame drawn between pixels - moved half a pixel - is nearly as far from both)
            self.assertLess(here, across / 2)
            self.assertLess(here, up / 2)

    def test_a_picture_keeps_its_own_size_odd_or_not(self):
        size = "import bpy,sys; i=bpy.data.images.load(sys.argv[-1]); print('READ', i.size[0], i.size[1])"
        with tempfile.TemporaryDirectory() as d:
            self.made(d, {**SCENE, "width": 65, "height": 37})
            self.assertEqual(self.read_back(size, pathlib.Path(d) / "picture.png"), ["65", "37"])

    def test_a_film_moves_its_progress_by_its_frames_samples_and_never_back(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "scene.gltf").write_text(json.dumps({**BALL, **ball_mesh()}), encoding="utf-8")
            said = ask("prepare", step="render", entries=[{**SCENE, "film": True, "frames": 3, "fps": 3, "width": 64, "height": 48}], options={"samples": 8, "background": "white", "device": "cpu"})
            job = said["result"]["jobs"][0]
            for f in job["files"]:
                (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
            run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            marks = [int(m) for m in re.findall(r"^progress (\d+)/100$", run.stdout, re.M)]
            self.assertEqual(marks, sorted(marks))
            # (a third done after the first frame, two thirds after the second: not only at the frames' ends)
            self.assertTrue(any(30 <= m <= 40 for m in marks), marks)
            self.assertTrue(any(60 <= m <= 70 for m in marks), marks)
            self.assertEqual(marks[-1], 100)
            self.assertIn("Rendering on the processor", run.stdout)

    def test_colours_a_surface_by_its_points_colours(self):
        # (a ball each of whose points is red, over a white material: COLOR_0 multiplies the base colour - glTF 3.9.2)
        import base64
        import struct
        mesh = ball_mesh()
        n = mesh["accessors"][0]["count"]
        red = struct.pack(f"<{4 * n}f", *([1.0, 0.0, 0.0, 1.0] * n))
        data = base64.b64decode(mesh["buffers"][0]["uri"].split(",", 1)[1]) + red
        mesh["buffers"][0] = {"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}
        mesh["bufferViews"].append({"buffer": 0, "byteOffset": len(data) - len(red), "byteLength": len(red)})
        mesh["accessors"].append({"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC4"})
        mesh["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = 2
        scene = {**BALL, **mesh, "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1}}]}
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "scene.gltf").write_text(json.dumps(scene), encoding="utf-8")
            said = ask("prepare", step="render", entries=[SCENE], options={"samples": 4, "background": "white", "device": "cpu", "engine": "eevee"})
            job = said["result"]["jobs"][0]
            for f in job["files"]:
                (pathlib.Path(d) / f["name"]).write_text(f["text"], encoding="utf-8")
            run = subprocess.run([blender(), *job["args"]], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            r, g, b = png_pixel((pathlib.Path(d) / "picture.png").read_bytes(), 160, 120)
            self.assertGreater(r, 2 * max(g, b), (r, g, b))


def png_pixel(png, x, y):
    """A pixel of an 8-bit RGB or RGBA PNG, as its specification lays one out (W3C PNG, 2nd ed.: IHDR, IDAT, filters 0-4)."""
    import zlib
    at, idat, width, height, kind = 8, b"", 0, 0, 0
    while at < len(png):
        length = int.from_bytes(png[at : at + 4], "big")
        tag, body = png[at + 4 : at + 8], png[at + 8 : at + 8 + length]
        if tag == b"IHDR":
            width, height, depth, kind = int.from_bytes(body[0:4], "big"), int.from_bytes(body[4:8], "big"), body[8], body[9]
            assert depth == 8 and kind in (2, 6), (depth, kind)
        elif tag == b"IDAT":
            idat += body
        at += 12 + length
    raw = zlib.decompress(idat)
    step = 3 if kind == 2 else 4
    stride = width * step
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(height):
        f, line = raw[i], bytearray(raw[i + 1 : i + 1 + stride])
        i += 1 + stride
        for k in range(stride):
            a = line[k - step] if k >= step else 0
            b, c = prev[k], (prev[k - step] if k >= step else 0)
            if f == 1:
                line[k] = (line[k] + a) & 255
            elif f == 2:
                line[k] = (line[k] + b) & 255
            elif f == 3:
                line[k] = (line[k] + (a + b) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[k] = (line[k] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    p = rows[y][x * step : x * step + 3]
    return p[0], p[1], p[2]


if __name__ == "__main__":
    unittest.main()


spec_p = importlib.util.spec_from_file_location("plugin_blender_presets", FOLDER / "presets.py")
presets = importlib.util.module_from_spec(spec_p)
spec_p.loader.exec_module(presets)


class Thing:
    """An object of the scene as Blender's importer leaves it: its name, its parent, Meno's tag as its own property, its scale."""

    def __init__(self, name, meno=None, parent=None, scale=(1.0, 1.0, 1.0)):
        self.name, self.parent, self.scale, self.props = name, parent, scale, ({"meno": meno} if meno else {})

    def get(self, key):
        return self.props.get(key)


class PresetMolecules(unittest.TestCase):
    """The molecules a preset is handed (docs/blender-presets.md, *ctx*): Meno's, found by the tags it wrote."""

    def scene(self):
        a = Thing("Ethene", {"molecule": 1, "bonds": [{"a": 0, "b": 1, "order": 2}, {"a": 0, "b": 2, "order": 1}]})
        b = Thing("Water", {"molecule": 0, "bonds": []})
        return [
            a, b, Thing("Camera"),
            Thing("ball.001", {"atom": 1, "element": "C"}, a, (0.3, 0.3, 0.3)),
            Thing("ball", {"atom": 0, "element": "C"}, a, (0.3, 0.3, 0.3)),
            Thing("ball.002", {"atom": 2, "element": "H"}, a, (0.2, 0.2, 0.2)),
            Thing("stick", {"bond": 0}, a), Thing("stick.001", {"bond": 0}, a), Thing("stick.002", {"bond": 1}, a),
            Thing("ball.003", {"atom": 0, "element": "O"}, b),
        ]

    def test_in_the_order_meno_wrote_them_with_their_atoms_and_bonds(self):
        mols = presets.molecules_of(self.scene())
        self.assertEqual([m["object"].name for m in mols], ["Water", "Ethene"])
        ethene = mols[1]
        self.assertEqual([(a["index"], a["element"]) for a in ethene["atoms"]], [(0, "C"), (1, "C"), (2, "H")])
        self.assertAlmostEqual(ethene["atoms"][2]["radius"], 0.2)
        # (a double bond's two lines, both its sticks)
        self.assertEqual([o.name for o in ethene["bonds"][0]["objects"]], ["stick", "stick.001"])
        self.assertEqual(ethene["bonds"][0]["order"], 2)
        self.assertEqual(ethene["bonds"][1]["atoms"], (0, 2))
        self.assertEqual([a["element"] for a in mols[0]["atoms"]], ["O"])

    def test_what_is_untagged_is_none_of_them(self):
        self.assertEqual(presets.molecules_of([Thing("Light"), Thing("Camera")]), [])


class PresetFiles(unittest.TestCase):
    """A preset's module is run only from the files Meno put in the job for it (`preset/<plugin id>/`)."""

    def entry(self, file):
        return {"presets": [{"interface": "preset", "from": "some-presets", "look": {"id": "x", "file": file}}]}

    def test_the_chosen_module_is_run_its_shared_files_beside_it(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d, "presets", "some-presets")
            (base / "looks").mkdir(parents=True)
            (base / "shared.py").write_text("WORD = 'shared'\n")
            (base / "looks" / "x.py").write_text("import shared\nWORD = shared.WORD\n")
            mod = presets.load(d, self.entry("looks/x.py"), "look")
            self.assertEqual(mod.WORD, "shared")
            self.assertIsNone(presets.load(d, self.entry("looks/x.py"), "motion"))

    def test_none_outside_its_folder(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "presets", "some-presets").mkdir(parents=True)
            pathlib.Path(d, "elsewhere.py").write_text("X = 1\n")
            with self.assertRaises(SystemExit):
                presets.load(d, self.entry("../../elsewhere.py"), "look")

    def test_nothing_where_no_preset_was_chosen(self):
        self.assertIsNone(presets.load("/nowhere", {}, "look"))

    def test_the_easing_curves_run_from_nought_to_one(self):
        for f in (presets.ease, presets.ease_in, presets.ease_out):
            self.assertEqual((f(0.0), f(1.0)), (0.0, 1.0))
            self.assertEqual((f(-1.0), f(2.0)), (0.0, 1.0))
