"""A plugin's environment is set up from the one lock it carries (requirements.lock), on whichever system it is added: so a
plugin said to be added on more than one system - or on any, saying none (manifest.json, `systems`) - carries a lock made
for every system, its packages for each by their markers, never one resolved for a single platform. Python alone."""

import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"


class SystemLocks(unittest.TestCase):
    def test_a_lock_is_made_for_every_system_its_plugin_may_be_added_on(self):
        seen = 0
        for folder in sorted(p for p in PLUGINS.iterdir() if (p / "requirements.lock").is_file()):
            command = next((l for l in (folder / "requirements.lock").read_text(encoding="utf-8").splitlines() if "uv pip compile" in l), None)
            if command is None:
                continue
            systems = json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("systems", [])
            # (one platform's lock would do for a plugin added on one system alone)
            if len(systems) == 1:
                continue
            self.assertIn("--universal", command, folder.name)
            self.assertNotIn("--python-platform", command, folder.name)
            seen += 1
        self.assertGreater(seen, 0)

    def test_reinvent_4_is_for_macos_and_windows_with_the_index_torch_and_no_linux_gpu_library(self):
        manifest = json.loads((PLUGINS / "reinvent/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["systems"], ["macos", "windows"])
        lock = (PLUGINS / "reinvent/requirements.lock").read_text(encoding="utf-8")
        self.assertRegex(lock, re.compile(r"^torch==\S+ \\$", re.M))
        self.assertIn("--no-sources", lock)
        self.assertNotRegex(lock, re.compile(r"^(nvidia-|triton|cuda-)", re.M))
        # (Windows' own, by its marker)
        self.assertRegex(lock, re.compile(r"^colorama==\S+ ; sys_platform == 'win32'", re.M))


if __name__ == "__main__":
    unittest.main()
