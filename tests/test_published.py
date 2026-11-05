"""The release the catalogue takes of each plugin (tools/published.py): its newest published one - a draft never, nor
a prerelease, nor another plugin's whose id begins the same. Python alone."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from published import newest  # noqa: E402


class Published(unittest.TestCase):
    def test_the_newest_published_release_of_that_plugin_alone(self):
        releases = [
            {"tagName": "alpha-v0.9.0", "isDraft": False, "isPrerelease": False},
            {"tagName": "alpha-v0.10.0", "isDraft": False, "isPrerelease": False},
            {"tagName": "alpha-v0.11.0", "isDraft": True, "isPrerelease": False},
            {"tagName": "alpha-v0.12.0", "isDraft": False, "isPrerelease": True},
            {"tagName": "alpha-looks-v2.0.0", "isDraft": False, "isPrerelease": False},
            {"tagName": "catalogue", "isDraft": False, "isPrerelease": False},
        ]
        # (by version, not as listed: 0.10.0 after 0.9.0; the draft and the prerelease passed over)
        self.assertEqual(newest(releases, "alpha"), "alpha-v0.10.0")
        self.assertEqual(newest(releases, "alpha-looks"), "alpha-looks-v2.0.0")
        # (none published: nothing)
        self.assertEqual(newest(releases[2:4], "alpha"), "")
        self.assertEqual(newest(releases, "beta"), "")


if __name__ == "__main__":
    unittest.main()
