"""A plugin's version and what it uses, kept apart: `version` is the plugin's own, semver; `uses` names the programs and
libraries it wraps with the version each says - and each is held to the lock or pin the plugin carries, so that the two
cannot drift: a pip lock's `name==version`, a pixi lock's `name-version-` package, the commit a git pin names, or a
download's own file name. Python alone."""

import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = sorted(p for p in (ROOT / "plugins").iterdir() if (p / "manifest.json").is_file())


def plain(v: str) -> str:
    """A version as a lock writes it: leading zeros of a part dropped (2026.03.6 is 2026.3.6)."""
    return "".join(str(int(p)) if p.isdigit() else p for p in re.split(r"([.+])", v))


def pinned(folder: pathlib.Path, name: str, version: str) -> bool:
    lower = name.lower()
    v, _, sha = version.partition("+")
    pip = folder / "requirements.lock"
    if pip.is_file():
        text = pip.read_text(encoding="utf-8")
        # (a pip lock: name==version, or a git archive pinned by its commit)
        if re.search(rf"^{re.escape(lower)}=={re.escape(plain(v))}\b", text, re.M | re.I):
            return True
        if sha and re.search(rf"^{re.escape(lower)} @ .*{re.escape(sha)}", text, re.M | re.I):
            return True
    pixi = folder / "pixi.lock"
    if pixi.is_file():
        text = pixi.read_text(encoding="utf-8")
        if f"/{lower}-{v}-" in text or f"/{lower}-{plain(v)}-" in text:
            return True
    # (a program fetched whole: its own file name says its version)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for d in m.get("downloads") or []:
        if f"{lower}-{v}" in str(d.get("file", "")).lower() or f"{lower}-{v}" in str(d.get("url", "")).lower():
            return True
    return False


class Versions(unittest.TestCase):
    def test_a_plugin_s_version_is_its_own_semver(self):
        self.assertTrue(PLUGINS)
        for folder in PLUGINS:
            m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertRegex(str(m["version"]), r"^\d+\.\d+\.\d+$", folder.name)

    def test_uses_names_each_with_the_version_its_lock_or_pin_fixes(self):
        for folder in PLUGINS:
            m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            for u in m.get("uses") or []:
                with self.subTest(f"{folder.name}: {u['name']} {u['version']}"):
                    self.assertTrue(pinned(folder, u["name"], u["version"]), f"{folder.name} says it uses {u['name']} {u['version']}, which no lock or pin of its own fixes")


if __name__ == "__main__":
    unittest.main()
