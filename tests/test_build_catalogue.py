"""The catalogue as tools/build_catalogue.py makes it from the plugins' archives (tools/pack.py): the plugin a plugin
brings presets to (`presetsFor`), and the plugins no longer offered (`retired`, tools/retired.json), none of them also on
offer. Python alone."""

import json
import os
import pathlib
import io
import subprocess
import sys
import tarfile
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

def archive_of(folder: pathlib.Path, manifest: dict) -> None:
    """A made-up plugin's archive, as tools/pack.py lays one out: its manifest at the root, and nothing else."""
    raw = json.dumps(manifest).encode("utf-8")
    with tarfile.open(folder / f"{manifest['id']}-{manifest['version']}.tar.gz", "w:gz") as t:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))


def presets(id: str, depends: list) -> dict:
    """A made-up preset plugin, its presets brought through the interface `look`, using `depends`."""
    return {"id": id, "name": "Made-up looks", "version": "1.0.0", "depends": depends, "presets": {"interface": "look", "looks": []}}


LOOK = {"id": "look", "versions": ">=1.0.0 <2.0.0"}


class MadeUp(unittest.TestCase):
    """The catalogue made from made-up archives: what it refuses to make, and what it writes only where there is some."""

    def build(self, manifests: list, retired: list | None = None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            for m in manifests:
                archive_of(folder, m)
            said = folder / "retired.json"
            said.write_text(json.dumps(retired if retired is not None else []), encoding="utf-8")
            return build(folder, said)

    def test_a_preset_plugin_whose_interface_is_used_through_one_plugin_says_it(self):
        done = self.build([presets("looks", [{"plugin": "renderer", "versions": ">=1.0.0 <2.0.0", "interfaces": [LOOK]}])])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["plugins"][0]["presetsFor"], "renderer")

    def test_a_preset_plugin_whose_interface_is_used_through_none_or_two_makes_no_catalogue(self):
        for depends, count in [([{"plugin": "renderer", "versions": ">=1.0.0 <2.0.0"}], 0),
                               ([{"plugin": "one", "versions": ">=1.0.0 <2.0.0", "interfaces": [LOOK]}, {"plugin": "two", "versions": ">=1.0.0 <2.0.0", "interfaces": [LOOK]}], 2)]:
            done = self.build([presets("looks", depends)])
            self.assertNotEqual(done.returncode, 0, count)
            self.assertIn(f"used through {count} plugins, not one", done.stderr)

    def test_retired_is_written_only_where_there_is_one(self):
        done = self.build([presets("looks", [{"plugin": "renderer", "versions": ">=1.0.0 <2.0.0", "interfaces": [LOOK]}])], retired=[])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("retired", json.loads(done.stdout))

    def test_a_retired_id_name_and_why_within_meno_s_bounds_and_none_past_them(self):
        good = {"id": "old-1", "name": "N" * 60, "why": "W" * 160}
        done = self.build([], retired=[good])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["retired"], [good])
        for bad in [{**good, "id": "Old"}, {**good, "id": "-old"}, {**good, "id": "o" * 41}, {**good, "name": "N" * 61}, {**good, "why": "W" * 161}, {**good, "why": "  "}]:
            done = self.build([], retired=[bad])
            self.assertNotEqual(done.returncode, 0, bad)
            self.assertIn("not an id, a name and why", done.stderr)

if __name__ == "__main__":
    unittest.main()
