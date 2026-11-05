"""A plugin's archive, as the catalogue gives it (`<id>-<version>.tar.gz`): its folder's contents at the archive's root -
regular files and folders alone, in order, with no owner, no time and fixed modes - the same bytes each time it is
made from the same files. Usage: python3 -I tools/pack.py <plugins/id> <out folder>. Prints the archive's path, its
SHA-256 and its size, a line each."""

import gzip
import hashlib
import io
import json
import pathlib
import sys
import tarfile

folder, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
if manifest["id"] != folder.name:
    sys.exit(f"{folder}: its manifest's id is {manifest['id']}")
files = sorted(p for p in folder.rglob("*") if "__pycache__" not in p.parts)
for p in files:
    if p.is_symlink() or not (p.is_file() or p.is_dir()):
        sys.exit(f"{p}: neither a file nor a folder")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
    for p in files:
        info = tarfile.TarInfo(p.relative_to(folder).as_posix())
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
        if p.is_dir():
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            t.addfile(info)
        else:
            data = p.read_bytes()
            info.size, info.mode = len(data), 0o644
            t.addfile(info, io.BytesIO(data))
packed = io.BytesIO()
with gzip.GzipFile(fileobj=packed, mode="wb", mtime=0, filename="") as z:
    z.write(buf.getvalue())
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"{manifest['id']}-{manifest['version']}.tar.gz"
out.write_bytes(packed.getvalue())
print(out)
print(hashlib.sha256(packed.getvalue()).hexdigest())
print(len(packed.getvalue()))
