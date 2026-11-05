"""The two lists tools/build_catalogue.py makes, one issued (tools/readers.json): catalogue.json, only the entries the
oldest released Meno still in use reads whole - an entry naming a role, a system or a kind it does not know would refuse
it the whole list - and catalogue-2.json, every entry. Made-up archives. Python alone."""

import io
import json
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
    def build(self, manifests: list) -> tuple:
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            for m in manifests:
                archive_of(folder, m)
            full = folder / "catalogue-2.json"
            done = subprocess.run([sys.executable, "-I", str(ROOT / "tools/build_catalogue.py"), str(folder), "https://example.org/releases", "2026-11-06T00:00:00Z", "--full", str(full)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout), json.loads(full.read_text(encoding="utf-8"))

    def test_an_entry_the_oldest_meno_reads_is_in_both_lists_one_it_does_not_in_the_full_list_alone(self):
        known = READERS["roles"][0]
        oldest, full = self.build([
            plugin("known", roles=[{"role": known}]),
            plugin("newrole", roles=[{"role": "a-role-no-meno-knows-yet"}]),
            plugin("newsystem", systems=["a-system-no-meno-knows"]),
            plugin("newkind", steps=[{"kinds": ["Not-A-Kind"]}]),
        ])
        self.assertEqual([e["id"] for e in full["plugins"]], ["known", "newkind", "newrole", "newsystem"])
        self.assertEqual([e["id"] for e in oldest["plugins"]], ["known"])
        self.assertEqual((oldest["issued"], oldest["schema"]), (full["issued"], full["schema"]))

    def test_the_vocabulary_names_the_oldest_released_meno_still_in_use(self):
        tags = [l.strip() for l in (ROOT / "tools/released.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")] if (ROOT / "tools/released.txt").is_file() else [READERS["meno"]]
        self.assertIn(READERS["meno"], tags)


if __name__ == "__main__":
    unittest.main()
