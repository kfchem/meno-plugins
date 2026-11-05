# SPDX-License-Identifier: GPL-3.0-or-later
#
# The Blender plugin's side of the interface `preset` (docs/plugins/blender-presets.md): the `ctx` a preset module is run
# against, built from the scene Meno sent, and the running of a look's, a studio's or a motion's module.
#
# Copyright (C) 2026 Meno contributors
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. It is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License, in this folder's LICENSE, for more details.
"""The interface `preset`, version 1, as this renderer implements it.

Imported by render.py, inside Blender. What does not need Blender - finding Meno's molecules by their tags, the easing
curves, where a preset's files are - is written so that it runs outside it too (scripts/calc/test_plugin_blender.py).
"""
import math
import os
import sys
import types

VERSION = 1

# (what Meno asked that a preset is told - the rest of the entry is the renderer's)
ASKED = ("width", "height", "fps", "frames", "seconds", "mode", "background", "key", "ambient", "softness",
         "molecules", "surfaces", "shading", "band", "outlines", "outline", "outline_colour")


def tag(obj):
    """What Meno tagged an object with in the scene (its node's extras, `meno`, which Blender's importer keeps as the object's
    own property), as a plain dict; none where it has none."""
    try:
        t = obj.get("meno")
    except AttributeError:
        return None
    if t is None:
        return None
    return t.to_dict() if hasattr(t, "to_dict") else dict(t)


def molecules_of(objects):
    """Meno's molecules in the scene, by their tags, in the order Meno wrote them: each its node, its atoms (ball, index,
    element, radius) and its bonds (sticks, order, atoms at its ends, a ring centre's atoms)."""
    roots, atoms, sticks = {}, {}, {}
    for o in objects:
        t = tag(o)
        if not t:
            continue
        if "molecule" in t:
            roots[o.name] = (int(t["molecule"]), o, t)
        elif "atom" in t:
            atoms.setdefault(top_of(o), []).append((o, t))
        elif "bond" in t:
            sticks.setdefault(top_of(o), []).append((o, t))
    out = []
    for name, (index, root, t) in sorted(roots.items(), key=lambda kv: kv[1][0]):
        mine = [(o, a) for o, a in atoms.get(name, [])]
        mine.sort(key=lambda oa: int(oa[1]["atom"]))
        bonds = []
        for j, b in enumerate(t.get("bonds") or []):
            bonds.append({
                "objects": [o for o, s in sticks.get(name, []) if int(s["bond"]) == j],
                "order": float(b.get("order", 1)),
                "atoms": (int(b["a"]), int(b["b"])),
                "centre": [int(c) for c in b["centre"]] if b.get("centre") else None,
            })
        out.append({
            "index": index,
            "object": root,
            "atoms": [{"object": o, "index": int(a["atom"]), "element": str(a.get("element", "")),
                       "radius": radius_of(o), "colour": colour_of(o)} for o, a in mine],
            "bonds": bonds,
        })
    return out


def top_of(obj):
    """The name of the molecule node an object is under: its topmost tagged parent."""
    found = None
    o = getattr(obj, "parent", None)
    while o is not None:
        t = tag(o)
        if t and "molecule" in t:
            found = o.name
        o = getattr(o, "parent", None)
    return found


def radius_of(obj):
    """A ball's radius in the scene: Meno's unit ball, scaled."""
    scale = getattr(obj, "scale", None)
    return float(scale[0]) if scale is not None else 0.0


def colour_of(obj):
    """The colour Meno drew a ball or a stick in, linear: its material's base colour."""
    try:
        mat = obj.active_material
        node = mat.node_tree.nodes.get("Principled BSDF")
        c = node.inputs["Base Color"].default_value
        return (float(c[0]), float(c[1]), float(c[2]))
    except (AttributeError, KeyError, TypeError):
        return (0.8, 0.8, 0.8)


# --- the helpers version 1 guarantees -------------------------------------------------------------------------------

def ease(t):
    """Easing in and out on 0..1."""
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_in(t):
    t = min(1.0, max(0.0, t))
    return t * t * t


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def key(obj, path, frame, value):
    """`obj`'s `path` set to `value` and keyed at `frame`: an object's own property, or a socket's `default_value` (obj
    then being the socket)."""
    if path == "default_value":
        obj.default_value = value
        obj.keyframe_insert("default_value", frame=frame)
        return
    setattr(obj, path, value)
    obj.keyframe_insert(path, frame=frame)


class Ctx:
    """What a preset module is run against (docs/plugins/blender-presets.md, *ctx*)."""

    version = VERSION

    def __init__(self, scene, entry, folder, molecules, camera, say):
        self.scene = scene
        self.request = {k: entry[k] for k in ASKED if k in entry}
        # (the options of the preset plugins chosen from, together: their ids are each plugin's own)
        self.request["values"] = {k: v for p in entry.get("presets") or [] for k, v in (p.get("values") or {}).items()}
        self.molecules = molecules
        self.camera = camera
        self.folder = folder
        self.say = say
        self.ease, self.ease_in, self.ease_out, self.key, self.colour_of = ease, ease_in, ease_out, key, colour_of

    def camera_at(self, yaw, pitch, margin=0.9):
        """Meno's camera round the molecules, looking at their middle, far enough back that all fit in `margin` of the frame."""
        from mathutils import Vector
        balls = [(a["object"].matrix_world.translation, a["radius"]) for m in self.molecules for a in m["atoms"]]
        if not balls:
            return
        mid = sum((p for p, _ in balls), Vector()) / len(balls)
        reach = max((p - mid).length + r for p, r in balls)
        cam = self.camera
        y, p = math.radians(yaw), math.radians(pitch)
        towards = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)))
        if cam.data.type == "ORTHO":
            cam.data.ortho_scale = 2 * reach / margin
            dist = max(cam.data.clip_end / 2, 4 * reach)
        else:
            half = math.atan(cam.data.sensor_width / 2 / cam.data.lens)
            dist = reach / (math.sin(half) * margin)
        cam.location = mid + towards * dist
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = (mid - cam.location).to_track_quat("-Z", "Y")

    def material(self, name):
        """A new node material, its Principled BSDF linked to its output, to build on."""
        import bpy
        m = bpy.data.materials.new(name)
        return m


# --- the preset's files ---------------------------------------------------------------------------------------------

def item_of(entry, kind):
    """The preset plugin's choices (`presets`, one for each plugin) that chose a look, studio or motion of `kind`; none where
    none did."""
    return next((p for p in entry.get("presets") or [] if p.get("interface") == "preset" and p.get(kind)), None)


def preset_folder(folder, item):
    """Where a preset plugin's files were put in the job (`presets/<plugin id>/`), by Meno."""
    return os.path.join(folder, "presets", item["from"])


def load(folder, entry, kind):
    """The chosen look's, studio's or motion's module, run from the preset's folder - its shared files importable beside it;
    none where none of that kind was chosen."""
    item = item_of(entry, kind)
    if item is None:
        return None
    base = preset_folder(folder, item)
    name = item[kind].get("file")
    path = os.path.normpath(os.path.join(base, name))
    if not path.startswith(os.path.normpath(base) + os.sep) or not os.path.isfile(path):
        raise SystemExit(f"render.py: the preset's {kind} is not among its files")
    if base not in sys.path:
        sys.path.insert(0, base)
    mod = types.ModuleType(f"preset_{kind}")
    mod.__file__ = path
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), mod.__dict__)
    return mod
