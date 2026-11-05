"""The MolScribe plugin's worker: structures read from pictures, for Meno's
role *Structures read from pictures* (its manifest; docs/PLUGINS.md,
docs/PDF.md *Stage 5*), with MolScribe (MIT; Qian et al., J. Chem. Inf.
Model. 2023, 63, 1925).

One JSON object per line on stdin, one per line back on stdout:

    {"id": 3, "op": "read_picture", "picture": "<a PNG or a JPEG, base64>"}
    {"id": 3, "ok": true, "result": {"atoms": [...], "bonds": [...]}}
    {"id": 3, "ok": false, "error": "..."}

It runs no code it is sent - only the operation below - and reaches no
network: MolScribe's trained model is downloaded by Meno, with the user's
leave, into the folder given as this script's one argument. It says it is
ready at once; MolScribe and its model are loaded for the first picture -
a minute, the first time a computer loads them, as it looks them over:

    {"event": "ready", "version": "1.1.1+7296a30"}

A structure read is its atoms - each where MolScribe places it, as it
gives it: across the picture as it cropped it, with `pad` pixels of white
round what is drawn, or across the structure itself - and an element with
its charge and isotope, or a label as drawn (a group's: OTBS); and its
bonds by index: single, double or triple, or in a ring MolScribe reads as
aromatic; a wedge or a hash as which end lies nearer (`up`: `b` nearer
than `a`), which is all MolScribe tells; and MolScribe's confidence in
each. Meno lays it on the picture and says what is unsure (docs/PDF.md, *A
picture read*).
"""

import base64
import contextlib
import io
import json
import os
import re
import sys

# The trained model, in the folder Meno gives (the manifest's `downloads`).
WEIGHTS = "swin_base_char_aux_1m680k.pth"
# The white MolScribe keeps round what is drawn, in pixels, as it crops a picture (its CropWhite).
PAD = 5
VERSION = "1.1.1+7296a30"

FILES = sys.argv[1] if len(sys.argv) > 1 else None
_model = {}


def graphs_unwritten(coords, symbols, edges, images=None, num_workers=1):
    """The model's graphs left unwritten as SMILES and molfiles: Meno takes its atoms and bonds as predicted, and writing
    a graph as predicted - in RDKit - can run without end, or crash the process, on a picture that is no structure."""
    return [""] * len(coords), [""] * len(coords), 0.0


def model():
    """MolScribe, loaded once from its trained model, on the CPU: the same on every computer."""
    if "it" not in _model:
        if not FILES or not os.path.isfile(os.path.join(FILES, WEIGHTS)):
            raise RuntimeError("MolScribe's trained model is not here: add MolScribe again in Settings, Plugins.")
        import torch
        import molscribe.interface as mi

        # (its atoms and bonds read; the graph not written as SMILES, which Meno does not take)
        mi.convert_graph_to_smiles = graphs_unwritten
        with contextlib.redirect_stdout(sys.stderr):
            _model["it"] = mi.MolScribe(os.path.join(FILES, WEIGHTS), device=torch.device("cpu"))
    return _model["it"]


def picture_of(data):
    """A picture as MolScribe takes it: RGB pixels, white where it is see-through."""
    import numpy as np
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(data)))
    im.load()
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, im)
    return np.array(im.convert("RGB"))


# A bracketed atom as MolScribe writes one: [C@@H], [N+], [O-], [2H], [Si]
BRACKET = re.compile(r"^\[(\d*)([A-Z][a-z]?|[a-z]{1,2})(@{0,2})(H\d*)?([+-]\d*|[+-]+)?\]$")
ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh "
    "Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
)


def atom_of(symbol):
    """An atom's symbol as MolScribe gives it, as Meno takes it: an element with its charge, and its isotope's mass number
    where it names one; or a label as drawn (Ph, OTBS); or nothing, where it gives no symbol."""
    s = symbol.strip()
    m = BRACKET.match(s)
    if m:
        el = m.group(2)
        el = el[0].upper() + el[1:] if el.islower() else el
        if el in ELEMENTS:
            sign = m.group(5) or ""
            charge = 0
            if sign:
                digits = sign.lstrip("+-")
                charge = (int(digits) if digits else len(sign)) * (1 if sign[0] == "+" else -1)
            out = {"el": el}
            if charge:
                out["charge"] = charge
            if m.group(1):
                out["isotope"] = int(m.group(1))
            return out
    plain = s[0].upper() + s[1:] if s.islower() and len(s) <= 2 else s
    if plain in ELEMENTS:
        return {"el": plain}
    label = s[1:-1] if s.startswith("[") and s.endswith("]") else s
    return {"label": label} if label else {}


# MolScribe's bond types: order, and for a wedge which end lies nearer - its edges[i][j] 5 a wedge begun at i (j nearer),
# 6 a hash begun at i (j farther); the two are one bond seen from each end.
BONDS = {"single": (1, None), "double": (2, None), "triple": (3, None), "aromatic": (1, None), "solid wedge": (1, "up"), "dashed wedge": (1, "down")}


def op_read_picture(m):
    rgb = picture_of(m["picture"])
    with contextlib.redirect_stdout(sys.stderr):
        p = model().predict_image(rgb, return_atoms_bonds=True, return_confidence=True)
    atoms = p.get("atoms") or []
    bonds = [b for b in p.get("bonds") or [] if b.get("bond_type") in BONDS]
    if not atoms or not bonds:
        return {"atoms": [], "bonds": []}
    out_atoms = []
    for a in atoms:
        atom = {"x": float(a["x"]), "y": float(a["y"]), **atom_of(a["atom_symbol"])}
        if a.get("confidence") is not None:
            atom["confidence"] = float(a["confidence"])
        out_atoms.append(atom)
    out_bonds = []
    for b in bonds:
        order, stereo = BONDS[b["bond_type"]]
        i, j = b["endpoint_atoms"]
        bond = {"a": int(i), "b": int(j), "order": order}
        if b["bond_type"] == "aromatic":
            bond["aromatic"] = True
        if stereo:
            bond["stereo"] = stereo
        if b.get("confidence") is not None:
            bond["confidence"] = float(b["confidence"])
        out_bonds.append(bond)
    return {"atoms": out_atoms, "bonds": out_bonds, "pad": PAD}


OPS = {"read_picture": op_read_picture}


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
