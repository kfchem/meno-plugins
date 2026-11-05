"""A plugin's newest PUBLISHED release, as the catalogue is made from (release.yml): of the releases listed
(`gh release list --json tagName,isDraft,isPrerelease`), its tags `<id>-v<major>.<minor>.<patch>` - that id exactly -
neither a draft nor a prerelease, the highest version. Its tag, or nothing where it has none published.
Usage: python3 -I tools/published.py <releases.json> <id>"""

import json
import re
import sys


def newest(releases, plugin):
    found = []
    for r in releases:
        m = re.fullmatch(re.escape(plugin) + r"-v(\d+)\.(\d+)\.(\d+)", r.get("tagName", ""))
        if m and not r.get("isDraft") and not r.get("isPrerelease"):
            found.append((tuple(int(x) for x in m.groups()), r["tagName"]))
    return max(found)[1] if found else ""


if __name__ == "__main__":
    print(newest(json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2]))
