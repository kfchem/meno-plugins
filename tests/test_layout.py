"""Each plugin's folder as Meno takes it: its manifest names it as its folder is, with a version of three numbers;
nothing in it is a link or a cache; the GPL plugins carry their licence; each lock pins its packages by hash; and its archive is the same bytes each time it
is made (tools/pack.py). Python alone."""

import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = sorted(p for p in (ROOT / "plugins").iterdir() if p.is_dir())


class Layout(unittest.TestCase):
    def test_each_folder_is_its_manifest_s_plugin(self):
        self.assertTrue(PLUGINS)
        for folder in PLUGINS:
            with self.subTest(folder.name):
                m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(m["id"], folder.name)
                self.assertRegex(m["version"], r"^\d{1,6}\.\d{1,6}\.\d{1,6}$")
                self.assertIn(m.get("contract", 1), (1,))
                for p in folder.rglob("*"):
                    self.assertFalse(p.is_symlink(), p)
                    self.assertNotIn("__pycache__", p.parts, p)
                if m.get("licence", "").startswith("GPL-"):
                    self.assertTrue((folder / "LICENSE").is_file(), folder.name)

    def test_each_preset_file_is_in_its_folder(self):
        # (every file a preset plugin names - each look's, studio's and motion's, and those they share - is in its folder)
        for folder in PLUGINS:
            presets = json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("presets")
            if not presets:
                continue
            names = [x["file"] for k in ("looks", "studios", "motions") for x in presets.get(k, [])] + presets.get("files", [])
            for name in names:
                with self.subTest(f"{folder.name}: {name}"):
                    self.assertTrue((folder / name).is_file())

    def test_each_lock_pins_its_packages_by_hash(self):
        # (as Meno sets up only from a lock pinned so - one that names no package, a plugin needing Python alone, has none to pin)
        package = re.compile(r"^[A-Za-z0-9_.-]+(==| @ )")
        seen = 0
        for lock in sorted(ROOT.glob("plugins/*/requirements.lock")):
            lines = lock.read_text(encoding="utf-8").splitlines()
            if any(package.match(l) for l in lines):
                with self.subTest(lock.parent.name):
                    self.assertTrue(any(l.lstrip().startswith("--hash=") for l in lines))
                seen += 1
        self.assertGreater(seen, 0)

    def test_an_archive_is_the_same_bytes_each_time(self):
        folder = next(p for p in PLUGINS if (p / "worker.py").exists())
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            run = lambda out: subprocess.run([sys.executable, "-I", str(ROOT / "tools/pack.py"), str(folder), out], check=True, capture_output=True, text=True).stdout.split()
            first, second = run(a), run(b)
        self.assertEqual(first[1:], second[1:])
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", first[1]))


if __name__ == "__main__":
    unittest.main()
