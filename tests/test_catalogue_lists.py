"""The two lists tools/build_catalogue.py makes, one issued (tools/readers.json): catalogue.json, only the entries the
oldest released Meno still in use reads whole - an entry naming a role, a system or a kind it does not know would refuse
it the whole list - and catalogue-2.json, every entry. Made-up archives. Python alone."""

import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
READERS = json.loads((ROOT / "tools/readers.json").read_text(encoding="utf-8"))


def archive_of(folder: pathlib.Path, manifest: dict) -> None:
    raw = json.dumps(manifest).encode("utf-8")
    with tarfile.open(folder / f"{manifest['id']}-{manifest['version']}.tar.gz", "w:gz") as t:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))


def plugin(id: str, **more) -> dict:
    return {"id": id, "name": id.title(), "version": "1.0.0", "licence": "Apache-2.0", **more}


class Lists(unittest.TestCase):
    def build(self, manifests: list, retired: list | None = None) -> tuple:
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            for m in manifests:
                archive_of(folder, m)
            full = folder / "catalogue-2.json"
            # (the plugins no longer offered, made up here where given: tools/retired.json else)
            env = None
            if retired is not None:
                said = folder.parent / f"{folder.name}-retired.json"
                said.write_text(json.dumps(retired), encoding="utf-8")
                self.addCleanup(said.unlink)
                env = {**os.environ, "CATALOGUE_RETIRED": str(said)}
            done = subprocess.run([sys.executable, "-I", str(ROOT / "tools/build_catalogue.py"), str(folder), "https://example.org/releases", "2026-11-06T00:00:00Z", "--full", str(full)],
                                  capture_output=True, text=True, env=env)
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout), json.loads(full.read_text(encoding="utf-8"))

    def test_an_entry_the_oldest_meno_reads_is_in_both_lists_one_it_does_not_in_the_full_list_alone(self):
        known = READERS["roles"][0]
        oldest, full = self.build([
            plugin("known", roles=[{"role": known}]),
            plugin("newrole", roles=[{"role": "a-role-no-meno-knows-yet"}]),
            plugin("newsystem", systems=["a-system-no-meno-knows"]),
            plugin("newkind", steps=[{"kinds": ["Not-A-Kind"]}]),
            # (one known role beside an unknown one: still one the oldest Meno does not read whole)
            plugin("mixedroles", roles=[{"role": known}, {"role": "language-model"}]),
            # (a kind it reads outside the pattern)
            plugin("newread", reads=["Bad Kind"]),
        ])
        self.assertEqual([e["id"] for e in full["plugins"]], ["known", "mixedroles", "newkind", "newread", "newrole", "newsystem"])
        self.assertEqual([e["id"] for e in oldest["plugins"]], ["known"])
        self.assertEqual((oldest["issued"], oldest["schema"]), (full["issued"], full["schema"]))

    def test_both_lists_say_what_is_retired_and_neither_does_where_nothing_is(self):
        gone = [{"id": "old-one", "name": "Old one", "why": "Its work is done by another plugin now."}]
        oldest, full = self.build([plugin("known", roles=[{"role": READERS["roles"][0]}])], retired=gone)
        self.assertEqual((oldest["retired"], full["retired"]), (gone, gone))
        oldest, full = self.build([plugin("known", roles=[{"role": READERS["roles"][0]}])], retired=[])
        self.assertNotIn("retired", oldest)
        self.assertNotIn("retired", full)

    def test_the_vocabulary_names_the_oldest_released_meno_still_in_use(self):
        tags = [l.strip() for l in (ROOT / "tools/released.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")] if (ROOT / "tools/released.txt").is_file() else [READERS["meno"]]
        self.assertIn(READERS["meno"], tags)


if __name__ == "__main__":
    unittest.main()
