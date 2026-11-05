# SPDX-License-Identifier: GPL-3.0-or-later
#
# What the film motions of this plugin share (motion_*.py): Meno's molecule staged for a motion, the easing curves, the
# node helpers, the motions' finish, their light rig, the camera and the finishing compositor.
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
"""The kit a film motion is keyed with (docs/plugins/blender-presets.md, *A motion*).

A motion's `key(ctx)` calls `run(ctx, module, nominal)`, which executes this kit in a namespace holding `CFG` - the motion's
settings, from `ctx.request` - and `CTX`, and hands the names it defines to the motion's module, whose `keyed()` then keys
Meno's molecule: ATOMS and BONDS (each atom's element, place and radius, each bond's atoms and order), ATOM_OBJ and
BOND_OBJ (Meno's own balls and sticks), the easing curves, studio(), camera_at() and finish().

The molecule is Meno's, as the renderer took it in: its balls and sticks, their elements, places, radii and colours. It is
staged here at 4 cm an ångström - the size of a product on a table - its middle at the origin, and the scene's other objects
with it: the canvas's up is the stage's up, and what faces the canvas's viewer faces the camera at its start.

CFG: seconds (the film's), nominal (the motion's own length: its moments are timed in it, and played over the film's
length), frames, fps, mode ("once" or "loop"), bg ("dark" or "white"), and a change's piece's own keys.
"""
import math
import os
import random

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

random.seed(CFG.get("seed", 3))
K = 0.04                     # metres per angstrom: the molecule is the size of a product on a table
FPS = CFG.get("fps", 30)
N = int(CFG["frames"])
LOOP = CFG.get("mode", "once") == "loop"
# (a motion's moments are timed in its own length, `nominal`, and played over the film's: a film shorter or longer than the
# motion plays it faster or slower, all of it - a loop is timed in the film's own length)
SCALE = (CFG["nominal"] / CFG["seconds"]) if CFG.get("nominal") and not LOOP else 1.0
CFG["seconds"] = CFG["seconds"] * SCALE
CTX = CFG["ctx"]
sc = CTX.scene

# --- easing ------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x

def lin(x, a, b):
    return clamp((x - a) / (b - a)) if b != a else float(x >= a)

def smooth(x):
    x = clamp(x); return x * x * (3 - 2 * x)

def smoother(x):
    x = clamp(x); return x * x * x * (x * (x * 6 - 15) + 10)

def in_out(x, p=3.0):
    x = clamp(x)
    return 0.5 * (2 * x) ** p if x < 0.5 else 1 - 0.5 * (2 - 2 * x) ** p

def out_c(x, p=3.0):
    x = clamp(x); return 1 - (1 - x) ** p

def in_c(x, p=3.0):
    x = clamp(x); return x ** p

def out_back(x, s=1.4):
    x = clamp(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2

def spring(x, freq=2.2, damp=5.0):
    """0 -> 1 with a damped overshoot; x in seconds after release."""
    if x <= 0: return 0.0
    return 1 - math.exp(-damp * x) * math.cos(2 * math.pi * freq * x)

def bump(x, a, b):
    """0 -> 1 -> 0 over [a, b], smooth."""
    if x <= a or x >= b: return 0.0
    u = (x - a) / (b - a); return math.sin(math.pi * u) ** 2

def tsec(f):
    """A frame's moment in the motion's own time, in seconds."""
    return (f - 1) / FPS * SCALE

# --- Meno's molecule, staged -------------------------------------------------------

# (Meno's lights go: the motion lights the stage with its own rig, studio())
for o_ in [o for o in sc.objects if o.type == "LIGHT"]:
    bpy.data.objects.remove(o_)
bpy.context.view_layer.update()
# (a change's piece is one molecule becoming the next: the stage is that molecule's - `molecule` - the others hidden, the next
# given as its atoms and bonds alone - NEXT_ATOMS, NEXT_BONDS - for the change to draw)
ONE = CFG.get("molecule")
SHOWN = [m for m in CTX.molecules if ONE is None or m["index"] == ONE]

def hide(o):
    # (out of the render alone: hidden from the viewport too, it would be left out of the scene's evaluation, and the next
    # molecule's places, read from it, would stay where Meno put them, unstaged)
    o.hide_render = True
    for c in o.children: hide(c)

for m_ in CTX.molecules:
    if m_ not in SHOWN: hide(m_["object"])
balls_ = [(m, a) for m in SHOWN for a in m["atoms"]]
if not balls_:
    raise SystemExit("motionkit.py: the scene holds no atoms to move")
heavy_ = [a["object"].matrix_world.translation.copy() for _, a in balls_ if a["element"] != "H"] or [a["object"].matrix_world.translation.copy() for _, a in balls_]
centre = sum(heavy_, Vector()) / len(heavy_)
# (the stage: everything Meno sent - its molecules, a studio's backdrop - at 4 cm an ångström, its molecules' middle at the
# origin; the camera apart, which the motion places)
STAGE = bpy.data.objects.new("stage", None); sc.collection.objects.link(STAGE)
STAGE.scale = (K, K, K); STAGE.location = -centre * K
for o_ in [o for o in sc.objects if o.parent is None and o is not STAGE and o is not CTX.camera]:
    o_.parent = STAGE
bpy.context.view_layer.update()
# (Meno's balls and sticks themselves on the motion's root, where it stands, each where Meno put it: their places in metres)
COL = bpy.data.collections.new("molecule"); sc.collection.children.link(COL)
ROOT = bpy.data.objects.new("root", None); COL.objects.link(ROOT)
ROOT.rotation_mode = "XYZ"

def onto_root(ob):
    mw = ob.matrix_world.copy()
    ob.parent = ROOT
    ob.rotation_mode = "QUATERNION"
    ob.matrix_world = mw
    if ob.name not in COL.objects: COL.objects.link(ob)

ATOMS, ATOM_OBJ, at_ = [], [], {}
for m_, a_ in balls_:
    ob = a_["object"]
    onto_root(ob)
    r = ob.scale.x
    at_[(m_["index"], a_["index"])] = len(ATOMS)
    ob["col"] = a_["colour"]; ob["colB"] = a_["colour"]; ob["stick"] = 0.0
    ob["rnd"] = random.random(); ob["idx"] = len(ATOMS); ob["r"] = r
    ATOMS.append({"el": a_["element"], "p": ob.location.copy(), "r": r, "col": tuple(a_["colour"])})
    ATOM_OBJ.append(ob)

BONDS, BOND_OBJ, half_ = [], [], {}

def half_of(me):
    """How far a stick's mesh reaches along its length (Z) from its middle."""
    if me.name not in half_:
        half_[me.name] = max((abs(v.co.z) for v in me.vertices), default=1.0) or 1.0
    return half_[me.name]

for m_ in SHOWN:
    for b_ in m_["bonds"]:
        a, c = at_.get((m_["index"], b_["atoms"][0])), at_.get((m_["index"], b_["atoms"][1]))
        # (a stick to a ring's centre, or to an atom not drawn, stays where Meno put it)
        if a is None or c is None or b_["centre"]:
            continue
        bi = len(BONDS)
        o_ = b_["order"]
        BONDS.append({"a": a, "b": c, "order": o_ if o_ in (1, 2, 3) else 1})
        pa, pb = ATOMS[a]["p"], ATOMS[c]["p"]
        d = pb - pa
        for ob in b_["objects"]:
            onto_root(ob)
            h = half_of(ob.data)
            axis = ob.rotation_quaternion @ Vector((0, 0, 1)) * (h * ob.scale.z)
            e0, e1 = ob.location - axis, ob.location + axis
            t0, t1 = sorted(((e0 - pa).dot(d) / max(d.length_squared, 1e-12), (e1 - pa).dot(d) / max(d.length_squared, 1e-12)))
            off = ob.location - (pa + d * ((t0 + t1) / 2))
            ob["col"] = CTX.colour_of(ob); ob["colB"] = ob["col"]; ob["stick"] = 1.0
            ob["rnd"] = random.random(); ob["bond"] = bi; ob["rad"] = ob.scale.x; ob["t0"] = t0; ob["t1"] = t1; ob["half"] = h
            BOND_OBJ.append((ob, bi, off))

if CFG.get("stand"):
    # (stood on the floor, where asked: its lowest atom just above z = 0 - so that every molecule of a change of molecules
    # stands on the same floor, and one piece's last frame is the next one's first)
    zlow_ = min(a["p"].z - a["r"] for a in ATOMS) - CFG.get("float", 0.004)
    for ob in ATOM_OBJ + [b[0] for b in BOND_OBJ]: ob.location.z -= zlow_
    for a in ATOMS: a["p"] = a["p"] - Vector((0, 0, zlow_))
NEXT_ATOMS, NEXT_BONDS = [], []
if CFG.get("next") is not None:
    nx_ = next(m for m in CTX.molecules if m["index"] == CFG["next"])
    bpy.context.view_layer.update()
    pts_ = [(a, a["object"].matrix_world.translation.copy(), a["object"].matrix_world.to_scale().x) for a in nx_["atoms"]]
    hv_ = [p for a, p, _ in pts_ if a["element"] != "H"] or [p for _, p, _ in pts_]
    c_ = sum(hv_, Vector()) / len(hv_)
    at2_ = {}
    for a, p, r in pts_:
        at2_[a["index"]] = len(NEXT_ATOMS)
        NEXT_ATOMS.append({"el": a["element"], "p": p - c_, "r": r, "col": tuple(a["colour"])})
    # (stood on the same floor as the stage's molecule)
    low_ = min(a["p"].z - a["r"] for a in NEXT_ATOMS) - CFG.get("float", 0.004)
    for a in NEXT_ATOMS: a["p"] = a["p"] - Vector((0, 0, low_))
    for b_ in nx_["bonds"]:
        a, c = at2_.get(b_["atoms"][0]), at2_.get(b_["atoms"][1])
        if a is None or c is None or b_["centre"]:
            continue
        o_ = b_["order"]
        NEXT_BONDS.append({"a": a, "b": c, "order": o_ if o_ in (1, 2, 3) else 1})
# (a stick's radius as Meno draws one, for the sticks a change draws of its own)
STICK_R = sorted(b[0]["rad"] for b in BOND_OBJ)[len(BOND_OBJ) // 2] if BOND_OBJ else 0.105 * K
NB = [[] for _ in ATOMS]
for i, b in enumerate(BONDS):
    NB[b["a"]].append(b["b"]); NB[b["b"]].append(b["a"])

def graph_dist(sources):
    d = [1e9] * len(ATOMS); q = list(sources)
    for s in sources: d[s] = 0
    while q:
        n = []
        for i in q:
            for j in NB[i]:
                if d[j] > d[i] + 1:
                    d[j] = d[i] + 1; n.append(j)
        q = n
    return d

# (its reach from its middle: where it stands on the floor, from the middle of its height)
MID = Vector((0, 0, 0.5 * (min(a["p"].z for a in ATOMS) + max(a["p"].z for a in ATOMS)))) if CFG.get("stand") else Vector()
RADIUS = max((a["p"] - MID).length for a in ATOMS) + 0.35 * K

# --- scene ------------------------------------------------------------------------

sc.render.fps = FPS; sc.render.fps_base = 1
sc.frame_start, sc.frame_end = 1, N
sc.render.engine = "CYCLES"
cy = sc.cycles
cy.use_adaptive_sampling = True; cy.adaptive_threshold = CFG.get("noise", 0.008)
cy.max_bounces = 8; cy.diffuse_bounces = 3; cy.glossy_bounces = 4; cy.transmission_bounces = 8; cy.transparent_max_bounces = 16
cy.caustics_reflective = False; cy.caustics_refractive = False
cy.blur_glossy = 1.0
sc.render.use_motion_blur = CFG.get("blur", True)
sc.render.motion_blur_shutter = CFG.get("shutter", 0.5)
# (the colours are the renderer's, as asked; the light's level is the motion's own, its rig drawn for it - brought down so
# that Meno's molecule at rest reads as bright as Meno lights it with no studio)
sc.view_settings.exposure = CFG.get("exposure", -2.4 if CFG.get("bg") == "white" else -3.2)
sc.render.use_persistent_data = True

# --- node helpers -----------------------------------------------------------------

def nnew(nt, kind, **kw):
    n = nt.nodes.new(kind)
    for k, v in kw.items():
        if k in n.inputs:
            n.inputs[k].default_value = v
        else:
            setattr(n, k, v)
    return n

def L(nt, a, sa, b, sb):
    nt.links.new(a.outputs[sa] if not isinstance(sa, str) or sa in a.outputs else a.outputs[sa], b.inputs[sb])

def math_(nt, op, a, b=None, c=None, clamp_=False):
    n = nnew(nt, "ShaderNodeMath", operation=op, use_clamp=clamp_)
    for i, x in enumerate((a, b, c)):
        if x is None: continue
        if isinstance(x, (int, float)): n.inputs[i].default_value = x
        else: nt.links.new(x, n.inputs[i])
    return n.outputs[0]

def attr(nt, name, kind="OBJECT"):
    return nnew(nt, "ShaderNodeAttribute", attribute_type=kind, attribute_name=name)

def along(nt, z):
    """Where along its bond a point of a stick is - 0 at its first atom, 1 at its second - from its object's Z: one of Meno's
    sticks may be a stretch of its bond (a half in its atom's colour)."""
    u0 = attr(nt, "t0").outputs["Fac"]; u1 = attr(nt, "t1").outputs["Fac"]; h = attr(nt, "half").outputs["Fac"]
    s = math_(nt, "MULTIPLY_ADD", math_(nt, "DIVIDE", z, h), 0.5, 0.5)
    return math_(nt, "MULTIPLY_ADD", s, math_(nt, "SUBTRACT", u1, u0), u0)

def time_value(nt):
    """Seconds since the film's start, as a node output (a driver on frame)."""
    v = nnew(nt, "ShaderNodeValue")
    fc = v.outputs[0].driver_add("default_value")
    fc.driver.expression = f"(frame - 1) / {FPS} * {SCALE}"
    return v.outputs[0]

def new_material(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL": nt.nodes.remove(n)
    return m, nt, next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")

def hero_bsdf(nt, colour_socket):
    """The hero finish: a lacquered, slightly translucent coat over the element colour."""
    pr = nnew(nt, "ShaderNodeBsdfPrincipled", Roughness=0.32, **{"Coat Weight": 1.0, "Coat Roughness": 0.035, "Specular IOR Level": 0.5})
    nt.links.new(colour_socket, pr.inputs["Base Color"])
    return pr


# --- the molecule's objects ------------------------------------------------------

# (meshes for what a motion adds of its own - sparks, shards, a ring's balls)
def sphere_mesh(name, seg=96, rings=48):
    import bmesh
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    return me

def cylinder_mesh(name, seg=48):
    import bmesh
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=1.0, radius2=1.0, depth=2.0)
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = abs(p.normal.z) < 0.5
    return me

SPHERE = sphere_mesh("sphere"); CYL = cylinder_mesh("cyl")

def bond_frame(pa, pb):
    d = pb - pa
    q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    return (pa + pb) / 2, q, d.length

def place_bond(ob, bi, off, pa, pb, f=None, grow=None):
    """Put one of a bond's sticks between two atom positions - its own stretch of the bond, beside it as Meno drew it - and key
    it on frame f."""
    d = pb - pa
    m, q, ln = bond_frame(pa + d * ob["t0"] + off, pa + d * ob["t1"] + off)
    ob.location = m; ob.rotation_quaternion = q
    ob.scale = (ob["rad"], ob["rad"], max(ln / 2 / ob["half"], 1e-5))
    if f is not None:
        ob.keyframe_insert("location", frame=f); ob.keyframe_insert("rotation_quaternion", frame=f); ob.keyframe_insert("scale", frame=f)

def hero_material(name="hero", extra=None):
    """The element colour per object (two-tone sticks split at their middle)."""
    m, nt, out = new_material(name)
    ca = attr(nt, "col"); cb = attr(nt, "colB")
    gen = nnew(nt, "ShaderNodeTexCoord")
    sep = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(gen.outputs["Object"], sep.inputs[0])
    is_b = math_(nt, "GREATER_THAN", sep.outputs["Z"], 0.0)
    isstick = attr(nt, "stick").outputs["Fac"]
    fac = math_(nt, "MULTIPLY", is_b, isstick)
    mix = nnew(nt, "ShaderNodeMix", data_type="RGBA")
    nt.links.new(fac, mix.inputs[0]); nt.links.new(ca.outputs["Color"], mix.inputs[6]); nt.links.new(cb.outputs["Color"], mix.inputs[7])
    pr = hero_bsdf(nt, mix.outputs[2])
    sh = pr.outputs[0]
    if extra:
        sh = extra(nt, sh, mix.outputs[2], pr)
    nt.links.new(sh, out.inputs["Surface"])
    return m

# (each stick laid from its bond's first atom to its second, as the motions key it: its Z running along the bond)
for ob_, bi_, off_ in BOND_OBJ:
    place_bond(ob_, bi_, off_, ATOMS[BONDS[bi_]["a"]]["p"], ATOMS[BONDS[bi_]["b"]]["p"])

MOL_OBJ = ATOM_OBJ + [b[0] for b in BOND_OBJ]

def assign(m, objs=None):
    """A material on Meno's balls and sticks, or on given objects alone (linked to the object)."""
    for ob in (MOL_OBJ if objs is None else objs):
        if not ob.material_slots:
            ob.data.materials.append(m)
        if objs is None:
            ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = m


# --- the motion's light rig ---------------------------------------------------------

# --- studio ------------------------------------------------------------------------

def area(name, size, power, colour, loc, target=Vector(), shape="RECTANGLE", size_y=None, spread=180):
    ld = bpy.data.lights.new(name, "AREA"); ld.shape = shape
    ld.size = size; ld.size_y = size_y if size_y else size; ld.energy = power; ld.color = colour
    ld.spread = math.radians(spread)
    ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (target - Vector(loc)).to_track_quat("-Z", "Y")
    ob.visible_camera = False
    return ob


WHITE = CFG.get("bg") == "white"
# (a studio's backdrop, where one was chosen - the studio plugin's own, staged with the rest: the paper sweep, or the dark
# wall; none where the scene is as asked, its background put under it by the renderer)
BACKDROP = next((o for o in sc.objects if o.type == "MESH" and o.name.split(".")[0] in ("paper", "wall")), None)
SWEEP = "visible" if BACKDROP is not None else "catcher"
sc.render.film_transparent = BACKDROP is None

def studio(dark=True, key=1.0, rim=1.0):
    """The motion's lights - a key, a fill, a top light and two rims - and the backdrop, where a studio was chosen: (lights,
    backdrop or None)."""
    for o in [o for o in sc.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)
    w = bpy.data.worlds.new("w"); sc.world = w
    bg = w.node_tree.nodes["Background"]
    if WHITE:
        bg.inputs["Color"].default_value = (0.55, 0.57, 0.6, 1); bg.inputs["Strength"].default_value = CFG.get("ambient", 0.12)
        lights = {
            "key": area("key", 1.4, 700 * key, (1.0, 0.97, 0.94), Vector((-1.3, -1.4, 1.7)), size_y=0.9),
            "fill": area("fill", 2.0, 110 * key, (0.94, 0.97, 1.0), Vector((1.8, -1.2, -0.1)), size_y=1.2),
            "top": area("top", 2.4, 240 * key, (1, 1, 1), Vector((0.0, -0.3, 1.9)), size_y=0.18),
            "rimR": area("rimR", 0.14, 500 * rim, (0.85, 0.92, 1.0), Vector((1.3, 1.2, 0.5)), size_y=1.4),
            "rimL": area("rimL", 0.14, 350 * rim, (1.0, 0.9, 0.8), Vector((-1.3, 1.1, 0.3)), size_y=1.4),
        }
        if BACKDROP is not None:
            # (a soft source overhead pools light on the paper round the molecule, lighting the paper alone - Blender's light
            # linking - and the molecule's lights the molecule alone: the shadow under it is the sky's)
            zf = min(a["p"].z - a["r"] for a in ATOMS) - CFG.get("float", 0.004)
            sky = area("sky", CFG.get("sky_size", 0.8), CFG.get("sky", 330.0), (0.985, 1.0, 0.99), Vector((0.0, 0.15, zf + 1.6)), target=Vector((0.0, 0.35, zf)), size_y=1.4)
            sky.visible_glossy = False
            paper = bpy.data.collections.new("paper"); paper.objects.link(BACKDROP)
            sky.light_linking.receiver_collection = paper
            if CFG.get("subject_only", True):
                for ob in lights.values(): ob.light_linking.receiver_collection = COL
            lights["sky"] = sky
        return lights, BACKDROP
    bg.inputs["Color"].default_value = (0.0035, 0.0042, 0.006, 1) if dark else (0.6, 0.62, 0.66, 1)
    bg.inputs["Strength"].default_value = 1.0
    lights = {
        "key": area("key", 1.4, 700 * key, (1.0, 0.95, 0.9), Vector((-1.4, -1.6, 1.3)), size_y=0.9),
        "fill": area("fill", 2.0, 120 * key, (0.85, 0.92, 1.0), Vector((1.8, -1.2, -0.3)), size_y=1.2),
        "top": area("top", 2.4, 260 * key, (1, 1, 1), Vector((0.0, -0.3, 1.9)), size_y=0.18),
        "rimR": area("rimR", 0.14, 1400 * rim, (0.45, 0.75, 1.0), Vector((1.3, 1.5, 0.5)), size_y=1.6),
        "rimL": area("rimL", 0.14, 1000 * rim, (1.0, 0.62, 0.35), Vector((-1.4, 1.4, 0.2)), size_y=1.6),
    }
    return lights, BACKDROP

def pool_of(bd):
    """A dark backdrop's own light - the strength its pool glows with - to key; none where there is none."""
    if bd is None:
        return None
    e = next((n for m in bd.data.materials for n in m.node_tree.nodes if n.type == "EMISSION"), None)
    return e.inputs["Strength"] if e is not None else None

# --- camera ----------------------------------------------------------------------------

# (Meno's camera, made the motion's: a lens, the sensor across the frame's width whatever its shape - Blender's own Auto
# puts it across the longer side, the height of a portrait film, which would cut a tall molecule's ends)
CAMOB = CTX.camera; CAM = CAMOB.data
CAM.type = "PERSP"; CAM.lens = CFG.get("lens", 85); CAM.sensor_width = 36
CAM.sensor_fit = "HORIZONTAL"
CAM.clip_start = 0.05; CAM.clip_end = 50
CAMOB.animation_data_clear()
sc.camera = CAMOB
FOCUS = bpy.data.objects.new("focus", None); sc.collection.objects.link(FOCUS)
CAM.dof.use_dof = CFG.get("dof", True); CAM.dof.focus_object = FOCUS; CAM.dof.aperture_fstop = CFG.get("fstop", 4.0)
CAM.dof.aperture_blades = 7; CAM.dof.aperture_rotation = math.radians(12)

def fit_distance(margin=1.0):
    """The distance at which the molecule's widest extent fills `margin` of the frame's width (heights kept in too)."""
    hw = CAM.sensor_width / 2 / CAM.lens
    hh = hw * sc.render.resolution_y / sc.render.resolution_x
    w = max(abs(a["p"].x) + a["r"] for a in ATOMS); h = max(abs(a["p"].z) + a["r"] for a in ATOMS)
    return max(w / (hw * margin), h / (hh * margin))

def camera_at(f, yaw, pitch, dist, look=Vector(), roll=0.0, lens=None, fstop=None, focus=None):
    """Orbit camera: yaw about the vertical, pitch up, at a distance, looking at `look` (keyed on frame f)."""
    pitch = pitch + CFG.get("pitch_add", 18.0 if WHITE else 0.0)
    y, p = math.radians(yaw), math.radians(pitch)
    pos = look + Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p))) * dist
    CAMOB.location = pos
    q = (look - pos).to_track_quat("-Z", "Y")
    if roll: q = q @ Quaternion((0, 0, 1), math.radians(roll))
    CAMOB.rotation_mode = "QUATERNION"; CAMOB.rotation_quaternion = q
    CAMOB.keyframe_insert("location", frame=f); CAMOB.keyframe_insert("rotation_quaternion", frame=f)
    if lens is not None:
        CAM.lens = lens; CAM.keyframe_insert("lens", frame=f)
    if fstop is not None:
        CAM.dof.aperture_fstop = fstop; CAM.dof.keyframe_insert("aperture_fstop", frame=f)
    FOCUS.location = focus if focus is not None else look
    FOCUS.keyframe_insert("location", frame=f)

def linear_keys():
    for a in bpy.data.actions:
        for fc in all_fcurves(a):
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"

def all_fcurves(action):
    try:
        return list(action.fcurves)
    except AttributeError:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out += list(bag.fcurves)
        return out

# --- compositing -----------------------------------------------------------------------

def finish(bloom=0.12, streaks=0.0, dispersion=0.012, vignette=0.35, grain=0.012, threshold=5.0, glow_size=0.5):
    if WHITE:
        # high key: no bloom to speak of (glow washes out on white), no lens tint, a faint vignette
        bloom, streaks, dispersion, vignette, grain, threshold = CFG.get("white_bloom", 0.05), 0.0, 0.0, 0.07, 0.004, 9.0
    g = bpy.data.node_groups.new("finish", "CompositorNodeTree")
    g.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    n, l = g.nodes, g.links
    rl = n.new("CompositorNodeRLayers")
    img = rl.outputs["Image"]
    if bloom:
        gl = n.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"; gl.inputs["Quality"].default_value = "High"
        gl.inputs["Threshold"].default_value = threshold; gl.inputs["Strength"].default_value = bloom
        gl.inputs["Size"].default_value = glow_size; gl.inputs["Smoothness"].default_value = 0.4
        l.new(img, gl.inputs["Image"]); img = gl.outputs["Image"]
    if streaks:
        st = n.new("CompositorNodeGlare")
        st.inputs["Type"].default_value = "Streaks"; st.inputs["Quality"].default_value = "High"
        st.inputs["Threshold"].default_value = threshold * 2.5; st.inputs["Strength"].default_value = streaks
        st.inputs["Streaks"].default_value = 2; st.inputs["Streaks Angle"].default_value = 0.0; st.inputs["Fade"].default_value = 0.93
        l.new(img, st.inputs["Image"]); img = st.outputs["Image"]
    if dispersion:
        ld = n.new("CompositorNodeLensdist")
        ld.inputs["Dispersion"].default_value = dispersion; ld.inputs["Distortion"].default_value = 0.0
        l.new(img, ld.inputs["Image"]); img = ld.outputs["Image"]
    if vignette:
        el = n.new("CompositorNodeEllipseMask")
        el.inputs["Size"].default_value = (1.25, 1.25)
        bl = n.new("CompositorNodeBlur"); bl.inputs["Size"].default_value = (500, 500) if "Size" in bl.inputs else None
        l.new(el.outputs["Mask"], bl.inputs["Image"])
        mr = n.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 1 - vignette
        l.new(bl.outputs["Image"], mr.inputs["Value"])
        mx = n.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs[0].default_value = 1.0
        l.new(img, mx.inputs[6]); l.new(mr.outputs["Result"], mx.inputs[7]); img = mx.outputs[2]
    if grain:
        nz = n.new("ShaderNodeTexNoise"); nz.noise_dimensions = "3D"
        nz.inputs["Scale"].default_value = 900.0; nz.inputs["Detail"].default_value = 0.0
        tv = n.new("CompositorNodeTime") if False else None
        mx = n.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "OVERLAY"; mx.inputs[0].default_value = grain * 8
        l.new(img, mx.inputs[6]); l.new(nz.outputs["Color"], mx.inputs[7]); img = mx.outputs[2]
    out = n.new("NodeGroupOutput"); l.new(img, out.inputs["Image"])
    sc.compositing_node_group = g; sc.render.use_compositing = True
    return g


# --- render --------------------------------------------------------------------------

def render():
    """The renderer renders the frames the motion keyed (docs/plugins/blender-presets.md): nothing here."""
    return None
