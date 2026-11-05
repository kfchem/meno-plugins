"""The catalogue Meno reads (`catalogue.json`): each plugin's latest release - its id, version, what it is, under which
licence, the plugin contract it is written to - no fact of its environment, which is known once it is downloaded, the systems it runs on, what it does (roles, kinds of step, kinds it
reads), the kinds of file it brings that are told by their marks (`fileKinds`: so that Meno can say, of a file nothing
added reads, which plugin would read it), the tables of atoms its program prints as papers quote them (`quotedTables`, for
the same), and its archive (url, SHA-256, size) and its manifest's SHA-256 - and when it was issued, which is what
Meno orders catalogues by: UTC, to the second, as `2026-11-05T13:04:09Z` and in no other form.
Usage: python3 -I tools/build_catalogue.py <archives folder> <base url> [<issued>] > catalogue.json, the folder
holding each plugin's latest `<id>-<version>.tar.gz` and nothing else; `issued` the time given (the release job's, later
than the catalogue published), or now."""

import datetime
import hashlib
import io
import json
import pathlib
import re
import sys
import tarfile

folder, base = pathlib.Path(sys.argv[1]), sys.argv[2].rstrip("/")
ISSUED = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
issued = sys.argv[3] if len(sys.argv) > 3 else datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
if not ISSUED.fullmatch(issued):
    sys.exit(f"issued must be UTC to the second, as 2026-11-05T13:04:09Z: {issued}")
datetime.datetime.strptime(issued, "%Y-%m-%dT%H:%M:%SZ")
entries = []
for archive in sorted(folder.glob("*.tar.gz")):
    data = archive.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        member = t.getmember("manifest.json")
        raw = t.extractfile(member).read()
    m = json.loads(raw)
    if archive.name != f"{m['id']}-{m['version']}.tar.gz":
        sys.exit(f"{archive.name}: not its manifest's id and version")
    steps = sorted({k for s in m.get("steps", []) for k in s.get("kinds", [])})
    entry = {
        "id": m["id"],
        "version": m["version"],
        "name": m["name"],
        "description": m.get("description", ""),
        "licence": m.get("licence", ""),
        "contract": m.get("contract", 1),
        "systems": m.get("systems", []),
        "roles": [r["role"] for r in m.get("roles", [])],
        "kinds": steps,
        "reads": m.get("reads", []),
        "source": {"url": f"{base}/{m['id']}-v{m['version']}/{archive.name}", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
        "manifestSha256": hashlib.sha256(raw).hexdigest(),
    }
    # (each kind of file it brings that is told by its marks, as Meno tells it: its id, name, extensions and marks, and
    #  whether the plugin's own program writes it)
    #  - only those it reads; a mark's `at` and `anyCase` written only where they are set, never as false
    def mark(k):
        return {"text": k["text"], **({"at": "line-start"} if k.get("at") == "line-start" else {}), **({"anyCase": True} if k.get("anyCase") is True else {})}

    told = [
        {"id": k["id"], "name": k["name"], "extensions": k.get("extensions", []), "marks": [mark(x) for x in k["marks"]], **({"writtenBy": k["writtenBy"]} if k.get("writtenBy") else {})}
        for k in m.get("kinds", [])
        if k.get("marks") and k["id"] in m.get("reads", [])
    ]
    if told:
        entry["fileKinds"] = told
    # (each table of atoms its program prints, as papers quote it - so that Meno can say, of such a table pasted where
    #  no plugin added reads it, whose it is to read: as the manifest declares it)
    #  - one Meno's own check of a manifest would drop is left out (lib/plugins/manifest `acceptQuotedTables`)
    def table_ok(t):
        lines = lambda v, least: isinstance(v, list) and least <= len(v) <= 8 and all(isinstance(x, str) and 1 <= len(x.strip()) <= 80 for x in v)
        cols = t.get("columns")
        return (isinstance(t.get("id"), str) and re.fullmatch(r"[a-z][a-z0-9-]{0,39}", t["id"]) is not None and lines(t.get("headings"), 1) and lines(t.get("heads", []), 0)
                and isinstance(cols, list) and len(set(cols)) == len(cols) and set(cols) <= {"centre", "atomic-number", "element", "type", "x", "y", "z"}
                and {"x", "y", "z"} <= set(cols) and ("atomic-number" in cols) != ("element" in cols))

    tables, ids = [], set()
    for t in m.get("quotedTables", []):
        if table_ok(t) and t["id"] not in ids and len(tables) < 4:
            ids.add(t["id"])
            # (as the manifest's check keeps it: each line trimmed, nothing else of it)
            tables.append({"id": t["id"], "headings": [x.strip() for x in t["headings"]], "heads": [x.strip() for x in t.get("heads", [])], "columns": t["columns"]})
    if tables:
        entry["quotedTables"] = tables
    for extra in ("cite", "terms"):
        if m.get(extra):
            entry[extra] = m[extra]
    entries.append(entry)
print(json.dumps({"schema": 1, "issued": issued, "plugins": entries}, indent=2, ensure_ascii=False))
