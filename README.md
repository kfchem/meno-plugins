# Meno's plugins

The plugins [Meno](https://github.com/kfchem/meno) installs, each a folder of its own in `plugins/<id>/`: its
manifest (`manifest.json`, Meno's plugin contract), its worker, its lock and whatever else it is. Meno knows none of
them by name; it reads each as data, from a signed catalogue this repository publishes.

- `plugins/<id>/` - a plugin, as it is installed. Its own licence where it is not this repository's (Apache-2.0):
  the Blender plugins' are GPL-3.0-or-later, in their folders.
- `tests/` - each plugin's tests, written by hand: no program's output is part of this repository.
- `docs/` - what a plugin needs said beyond its manifest.
- `tools/` - its archive (`pack.py`), the catalogue (`build_catalogue.py`), and the public key the catalogue is
  signed with (`catalogue.pub`), the one Meno checks it by.

A plugin is released by a tag `<id>-v<version>`, its version the manifest's: its tests run, its folder is packed as
`<id>-<version>.tar.gz` (its contents at the archive's root), and the catalogue is made again from every plugin's
latest release and signed (`catalogue.json`, `catalogue.json.minisig`, on the release `catalogue`).

The programs the plugins run are installed separately, under their own licences, or set up by Meno where a plugin's
manifest says how - never shipped here.
