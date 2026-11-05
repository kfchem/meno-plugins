"""The marks a plugin's kinds are told by (a manifest's `kinds[].marks`): each the shortest fragment that tells its
format - a few words, no slogan or tagline - and told in a line written by hand, made up round it, where its place
(`at`) says. Python alone."""

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

    def test_each_mark_tells_its_kind_in_a_made_up_line(self):
        for plugin, k in kinds():
            for mark in k.get("marks", []):
                text = mark["text"]
                line = text + " made-up words" if mark.get("at") == "line-start" else "  made-up words " + text + " more\n"
                with self.subTest(f"{plugin} {k['id']}"):
                    if mark.get("at") == "line-start":
                        self.assertTrue(line.lstrip().startswith(text.lstrip()))
                    else:
                        self.assertIn(text, line)


if __name__ == "__main__":
    unittest.main()
