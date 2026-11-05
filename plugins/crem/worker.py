"""The CReM plugin's worker: new structures made from seeds, for Meno's step
*Generate* (its manifest; docs/WORKFLOWS.md, *Kinds of step*), with CReM
(BSD-3-Clause; Polishchuk, J. Cheminform. 2020, 12, 28): each seed's
fragments replaced, or its hydrogens, by fragments found in the same
surroundings in a database of them, built from ChEMBL 33.

One JSON object per line on stdin, one per line back on stdout:

    {"id": 3, "op": "run", "step": "generate", "entries": [{"molblocks": [...]}], "options": {"operation": "mutate", "count": 20}}
    {"id": 3, "ok": true, "result": {"outputs": [...], "said": "...", "by": {...}}}
    {"id": 3, "ok": false, "error": "..."}

It runs no code it is sent - only the operation below - and reaches no
network: the fragment database is downloaded by Meno, with the user's
leave, into the folder given as this script's one argument, and unpacked
there (manifest.json, `downloads`).

    {"event": "ready", "version": "0.3.2"}

Each seed is a structure drawn, given as its MOL blocks, read as RDKit
reads them. CReM replaces a fragment of it (*Replace fragments*) - or a
hydrogen (*Add to it*) - by a fragment the database holds in the same
surroundings, `radius` bonds out: the larger the radius, the closer the
surroundings must match, and the fewer and more conservative the
structures. Where more are found than asked for, CReM picks those it makes
with the random seed given, so that the same asking gives the same ones.

Each goes back as CReM gives it, and nothing more is made of it here
(docs/PLUGINS.md, *Generate*): its SMILES; its values, named - the change
made (`change`, a reaction SMARTS as CReM writes it) and how many times the
fragment put in occurs in the database (`frequency`); which seed it was
made from, by its place among those given (`seed` beside them); and, for
each seed by its place, how many it made or why it was not read (`seeds`),
from which Meno names them. An option outside what it may be is refused,
never run as another. Nothing is made 3D, no stereo is enumerated, nothing
is refused or cut: the general steps after it do that, and Meno takes no
more from one run than it takes of any (docs/PLUGINS.md, *Generate*). A
seed RDKit cannot read is counted, with why.
"""

import importlib.metadata
import json
import os
import sys

from crem.crem import grow_mol, mutate_mol
from rdkit import Chem, RDLogger

# RDKit's own complaints go nowhere: a seed that could not be read is counted, with why.
RDLogger.DisableLog("rdApp.*")

# The folder Meno keeps the plugin's files in (the manifest's `downloads`).
FILES = sys.argv[1] if len(sys.argv) > 1 else None
#: The fragment database, as the manifest downloads it and Meno unpacks it - into a folder of its name - and what
#: it is, as its record says.
DB = "chembl33_sa2_f5.db"
DB_SAID = "chembl33_sa2_f5 (CReM fragment databases, Zenodo 10.5281/zenodo.16909329; from ChEMBL 33, fragments of at most 15 heavy atoms)"
#: The most structures one seed gives in one run - a number of structures, never a time.
MOST_EACH = 100
#: The random seed where none is asked for.
SEED = 1
#: The surroundings a fragment is matched in, in bonds out from it, where none is asked for: CReM's own default.
RADIUS = 3
#: The most heavy atoms a change adds, where none is asked for: CReM's own default for both operations.
ADDS = 2
#: What each operation is called, as the step's options and its outputs say it.
OPERATIONS = {"mutate": "fragments replaced", "grow": "hydrogens replaced"}

VERSION = importlib.metadata.version("crem")


def seed_of(entry):
    """A seed as RDKit reads it, from a structure drawn as Meno gives one: its MOL blocks, its parts joined - as
    drawn, its stereo as its blocks say it. Else why not: a string."""
    blocks = entry.get("molblocks") or []
    # (Meno gives every part of one drawn, a lone ion's too: none, only where it is a label of no element alone)
    if not blocks:
        return "it is a label of no element"
    mol = None
    for block in blocks:
        part = Chem.MolFromMolBlock(block)
        if part is None:
            return "a part RDKit cannot read"
        mol = part if mol is None else Chem.CombineMols(mol, part)
    return mol


#: Each number option as the step's card names it (the manifest's labels).
LABELS = {"count": "How many from each seed", "radius": "Surroundings matched", "adds": "Heavy atoms added", "seed": "Random seed"}


def number(options, key, default, low, high):
    """An option as a whole number, as it was set - or its default, where it was not. Refused, naming it and what it
    may be, where it is not a whole number from `low` to `high`: never run with another than the one set."""
    v = options.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or not low <= v <= high:
        raise ValueError(f"{LABELS[key]} is {v}: it takes a whole number from {low} to {high}")
    return int(v)


def changed(mol, db, operation, radius, adds, count, seed):
    """What CReM makes of one seed: up to `count` new molecules, [(smiles, change, frequency)], in the order it
    gives them - picked with the random seed `seed` where more are found."""
    # (*Add to it* replaces a hydrogen - grow_mol adds the seed's hydrogens itself, its documentation's way; *Replace
    #  fragments* a fragment of one heavy atom or more, `min_size=1`: a hydrogen replaced is the other operation's)
    if operation == "grow":
        rows = grow_mol(mol, db, radius=radius, min_atoms=1, max_atoms=adds, max_replacements=count, seed=seed, return_rxn=True, return_rxn_freq=True, ncores=1)
    else:
        rows = mutate_mol(mol, db, radius=radius, min_size=1, max_inc=adds, max_replacements=count, seed=seed, return_rxn=True, return_rxn_freq=True, ncores=1)
    return [(r[0], r[1], r[2]) for r in rows][:count]


def op_run(m):
    """*Generate* from seeds: each structure CReM made, as it gave it, with which seed it was made from; what made
    them (`by`); and the run in a line (`said`)."""
    if m.get("step") != "generate":
        raise ValueError(f"There is no step {m.get('step')!r} to run at once.")
    db = os.path.join(FILES or "", DB, DB)
    if not FILES or not os.path.isfile(db):
        raise ValueError("CReM's fragment database is not here: add the plugin again in Settings, Plugins")
    entries = m.get("entries") or []
    if not entries:
        raise ValueError("No seed came in")
    o = m.get("options") or {}
    operation = o.get("operation", "mutate")
    if operation not in OPERATIONS:
        raise ValueError(f"Change is {operation!r}: it takes {' or '.join(repr(k) for k in OPERATIONS)}")
    count = number(o, "count", 20, 1, MOST_EACH)
    radius = number(o, "radius", RADIUS, 1, 5)
    adds = number(o, "adds", ADDS, 1, 10)
    seed = number(o, "seed", SEED, 0, 2**31 - 1)
    how = f"{OPERATIONS[operation]}, radius {radius}, at most {adds} heavy atoms added, seed {seed}"
    # (each seed's own tally - how many it made, or why it was not read - by its place: Meno names each from it)
    outputs, seeds = [], []
    for place, entry in enumerate(entries):
        mol = seed_of(entry)
        if not isinstance(mol, str):
            try:
                Chem.SanitizeMol(mol)
            except Exception:  # RDKit's own refusals, of every kind: a seed it cannot take is one it cannot read
                mol = "not a structure RDKit can take"
        if isinstance(mol, str):
            seeds.append({"place": place, "why": mol})
            continue
        made = changed(mol, db, operation, radius, adds, count, seed)
        seeds.append({"place": place, "made": len(made)})
        for smiles, change, frequency in made:
            outputs.append({
                "smiles": smiles,
                "properties": {
                    "change": {"value": change, "unit": "none"},
                    "frequency": {"value": frequency, "unit": "none"},
                },
                "seed": place,
            })
    taken = sum(1 for s in seeds if "made" in s)
    # (none read: nothing made, each seed's why in the tally - Meno names each from it)
    said = "No seed could be read" if not taken else f"{len(outputs)} made from {taken} seed{'' if taken == 1 else 's'} ({OPERATIONS[operation]}, radius {radius}, seed {seed})"
    return {"outputs": outputs, "seeds": seeds, "said": said, "by": {"program": "CReM", "version": VERSION, "model": DB_SAID, "method": how}}


OPS = {"run": op_run}


def answer(line):
    try:
        m = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "not JSON"}
    rid = m.get("id")
    op = OPS.get(m.get("op"))
    if op is None:
        return {"id": rid, "ok": False, "error": f"no such request: {m.get('op')}"}
    try:
        return {"id": rid, "ok": True, "result": op(m)}
    except Exception as e:  # an answer, never a crash
        return {"id": rid, "ok": False, "error": str(e) or type(e).__name__}


def main():
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
