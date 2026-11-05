"""The marks a plugin's kinds are told by (a manifest's `kinds[].marks`): each the shortest fragment that tells its
format - a few words, no slogan or tagline. How Meno tells a file by them is Meno's, tested there. Python alone."""

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOST = 40


def kinds():
    for mf in sorted((ROOT / "plugins").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        for k in m.get("kinds", []):
            yield m["id"], k


class Marks(unittest.TestCase):
    def test_each_mark_is_a_short_fragment(self):
        for plugin, k in kinds():
            for mark in k.get("marks", []):
                with self.subTest(f"{plugin} {k['id']}"):
                    self.assertTrue(mark["text"].strip())
                    self.assertLessEqual(len(mark["text"]), MOST, mark["text"])


if __name__ == "__main__":
    unittest.main()
