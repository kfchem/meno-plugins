"""The catalogue as tools/build_catalogue.py makes it from the plugins' archives (tools/pack.py): the plugin a plugin
brings presets to (`presetsFor`), and the plugins no longer offered (`retired`, tools/retired.json), none of them also on
offer. Python alone."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ISSUED = "2026-11-06T00:00:00Z"


def build(folder: pathlib.Path, retired: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", ""), **({"CATALOGUE_RETIRED": str(retired)} if retired else {})}
    return subprocess.run([sys.executable, "-I", str(ROOT / "tools/build_catalogue.py"), str(folder), "https://example.org/releases", ISSUED], capture_output=True, text=True, env=env)


class Catalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        for plugin in sorted(p for p in (ROOT / "plugins").iterdir() if (p / "manifest.json").is_file()):
            subprocess.run([sys.executable, "-I", str(ROOT / "tools/pack.py"), str(plugin), str(cls.folder)], check=True, capture_output=True)
        done = build(cls.folder)
        assert done.returncode == 0, done.stderr
        cls.catalogue = json.loads(done.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_preset_plugin_says_the_plugin_it_brings_presets_to_and_no_other_does(self):
        said = {e["id"]: e["presetsFor"] for e in self.catalogue["plugins"] if "presetsFor" in e}
        self.assertEqual(said, {"blender-presets": "blender"})

    def test_the_plugins_no_longer_offered_each_say_why_and_none_is_on_offer(self):
        retired = self.catalogue["retired"]
        self.assertEqual([r["id"] for r in retired], ["wfx", "getting-started"])
        for r in retired:
            self.assertEqual(set(r), {"id", "name", "why"})
        self.assertFalse({r["id"] for r in retired} & {e["id"] for e in self.catalogue["plugins"]})

    def test_a_plugin_both_offered_and_retired_or_said_wrong_makes_no_catalogue(self):
        for said, why in [([{"id": "rdkit", "name": "RDKit", "why": "A made-up reason"}], "offered and retired at once"),
                          ([{"id": "old", "name": "Old"}], "not an id, a name and why"),
                          ([{"id": "old", "name": "Old", "why": "x"}, {"id": "old", "name": "Old", "why": "y"}], "an id twice")]:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump(said, f)
            try:
                done = build(self.folder, pathlib.Path(f.name))
                self.assertNotEqual(done.returncode, 0, why)
                self.assertIn(why, done.stderr)
            finally:
                os.unlink(f.name)

if __name__ == "__main__":
    unittest.main()
