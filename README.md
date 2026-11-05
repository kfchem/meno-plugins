# Meno's plugins

The plugins [Meno](https://github.com/kfchem/meno) installs, each a folder of its own in `plugins/<id>/`: its
manifest (`manifest.json`, Meno's plugin contract), its worker, its lock and whatever else it is. Meno knows none of
them by name; it reads each as data, from a signed catalogue this repository publishes.

- `plugins/<id>/` - a plugin, as it is installed. Its own licence where it is not this repository's (Apache-2.0):
  the Blender plugins' are GPL-3.0-or-later, in their folders.
- `tests/` - each plugin's tests, written by hand: no program's output is part of this repository.
- `docs/` - what a plugin needs said beyond its manifest.
- `tools/` - its archive (`pack.py`), the catalogue (`build_catalogue.py`) and the releases it is made from
  (`published.py`), and the public key the catalogue is signed with (`catalogue.pub`), the one Meno checks it by.

A plugin is released in three steps:

1. **Tag:** `<id>-v<version>` on main, its version the manifest's, cut on the maintainer's word. Its folder is checked
   and packed as `<id>-<version>.tar.gz` (its contents at the archive's root), and its release made as a **draft**.
2. **Publish:** the maintainer publishes that draft. A draft is never in the catalogue.
3. **Catalogue:** publishing it makes the catalogue again, from every plugin's newest published release, signed and
   checked against `tools/catalogue.pub` (`catalogue.json`, `catalogue.json.minisig`, on the release `catalogue`, which
   needs no Publish of its own).

Whether the signing secrets are the key and password of `tools/catalogue.pub` is checked by hand, uploading nothing
(`check-key.yml`).

The programs the plugins run are installed separately, under their own licences, or set up by Meno where a plugin's
manifest says how - never shipped here.
