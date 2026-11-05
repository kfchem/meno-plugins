# SPDX-License-Identifier: GPL-3.0-or-later
#
# The part Film looks and motions' changes of molecules share: the molecule a piece ends as, who goes where, and the
# camera that runs on through every piece of the film.
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
"""A change of molecules (docs/plugins/blender-presets.md, *A motion*): a film in pieces, one for each molecule that becomes
the next - A, the piece's molecule, staged by the kit (motionkit.py), into B, the next. Executed after the kit, in its
namespace, by motion_kit.py `change`, before the change's own `keyed()`.

CFG adds: next (B's index among Meno's molecules: the kit gives it as NEXT_ATOMS and NEXT_BONDS), t_offset (where the piece
starts in the film, in seconds), t_total (the film's length) and loop (whether the film comes back to its first molecule).
Every molecule stands on the same floor (the kit's `stand`), turned as the canvas shows it. Who goes where in a change
(`hungarian`) is the change's choreography: which grain flies to which place, never a mapping of the molecules.
"""

# --- B, built as studio.py builds A: the scene's up the studio's up, its heavy atoms' middle on the vertical, on the floor
# (B: Meno's next molecule, as the kit gives it - its atoms' elements, places, radii and colours, stood on the floor)
BA, BB = NEXT_ATOMS, NEXT_BONDS

def middle(atoms):
    """The middle of a molecule's height, where the camera looks."""
    return 0.5 * (min(a["p"].z for a in atoms) + max(a["p"].z for a in atoms))

# --- who goes where
def hungarian(cost):
    """Least-cost assignment, rows to columns, of a square matrix: for each row its column (Kuhn-Munkres, O(n^3))."""
    n = len(cost); INF = float("inf")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = [INF] * (n + 1); used = [False] * (n + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    row = [0] * n
    for j in range(1, n + 1):
        if p[j]: row[p[j] - 1] = j - 1
    return row

# --- B's bonds, as the rig draws a molecule: a double bond two sticks side by side, a triple three - so that B, made, is
# the very molecule the next piece starts from
B_NB = [[] for _ in BA]
for b_ in BB:
    B_NB[b_["a"]].append(b_["b"]); B_NB[b_["b"]].append(b_["a"])

def b_side(b):
    a_, c_ = b["a"], b["b"]; pa_, pc_ = BA[a_]["p"], BA[c_]["p"]; d_ = (pc_ - pa_).normalized()
    for n_ in B_NB[a_] + B_NB[c_]:
        if n_ not in (a_, c_):
            v_ = BA[n_]["p"] - pa_; s_ = v_ - d_ * v_.dot(d_)
            if s_.length > 1e-6: return s_.normalized()
    s_ = d_.cross(Vector((0, 0, 1)))
    return s_.normalized() if s_.length > 1e-6 else d_.cross(Vector((1, 0, 0))).normalized()

def b_sticks():
    """B's sticks, each (object, bond, its two ends, its radius): placed whole, for the change to draw as it will."""
    out = []
    for bi, b in enumerate(BB):
        order = b["order"]
        sv = b_side(b) if order > 1 else Vector()
        offs = [Vector()] if order == 1 else ([sv * 0.13 * K, -sv * 0.13 * K] if order == 2 else [Vector(), sv * 0.2 * K, -sv * 0.2 * K])
        rad = STICK_R * (1.0 if order == 1 else 0.62 if order == 2 else 0.5)
        for k_, off in enumerate(offs):
            pa, pb = BA[b["a"]]["p"] + off, BA[b["b"]]["p"] + off
            ob = bpy.data.objects.new(f"Bbond{bi:03d}-{k_}", CYL); COL.objects.link(ob); ob.rotation_mode = "QUATERNION"
            mm, q, ln = bond_frame(pa, pb)
            ob.location = mm; ob.rotation_quaternion = q; ob.scale = (rad, rad, ln / 2)
            ob["col"] = BA[b["a"]]["col"]; ob["colB"] = BA[b["b"]]["col"]; ob["stick"] = 1.0; ob["rad"] = rad; ob["t0"] = 0.0; ob["t1"] = 1.0; ob["half"] = 1.0
            out.append((ob, b, pa, pb, rad))
    return out

def draw_stick(ob, pa, pb, rad, g, f):
    """A stick drawn out from its middle to `g` of its length (thin as well as short until it is drawn), keyed on frame f."""
    mid = pa.lerp(pb, 0.5)
    mm, q, ln = bond_frame(mid.lerp(pa, g), mid.lerp(pb, g))
    th = rad * smooth(lin(g, 0.0, 0.15))
    ob.location = mm; ob.rotation_quaternion = q; ob.scale = (max(th, 1e-6), max(th, 1e-6), max(ln / 2, 1e-5))
    for path in ("location", "rotation_quaternion", "scale"): ob.keyframe_insert(path, frame=f)

# --- the camera: one move through the whole film - turning slowly the one way (Once) or swinging and back (Loop) - each
# piece framing A, then B as it is made; where one piece ends the next begins, just there
T_OFF, T_TOTAL = float(CFG.get("t_offset", 0.0)), float(CFG.get("t_total", CFG["seconds"]))
# (frames to a second of the change's own time - its moments are timed in it, `tsec`, and played over the piece's length)
FPSM = FPS / SCALE

def film_time(f):
    """A frame's moment in the whole film, in seconds (the film's, not the change's own)."""
    return T_OFF + (f - 1) / FPS

def fit(atoms, look_z, margin=None):
    """The distance at which a molecule's widest extent fills `margin` of the frame, looked at from the middle of its height."""
    margin = margin or CFG.get("margin", 0.62)
    hw = CAM.sensor_width / 2 / CAM.lens; hh = hw * sc.render.resolution_y / sc.render.resolution_x
    w = max(abs(a["p"].x) + a["r"] for a in atoms); h = max(abs(a["p"].z - look_z) + a["r"] for a in atoms)
    return max(w / (hw * margin), h / (hh * margin))

Z_A, Z_B = middle(ATOMS), middle(BA)
D_A, D_B = fit(ATOMS, Z_A), fit(BA, Z_B)

def change_camera(f, refit, far=None, far_z=None, out=0.0):
    """The camera at frame f: framing A until the change refits it to B (`refit`, 0 to 1); `out` of the way, where asked, to
    framing what the change spreads out - at least `far` away, looking at the height `far_z`."""
    u = film_time(f) / max(T_TOTAL, 1e-6)
    if CFG.get("loop"):
        yaw = -14.0 + 16.0 * math.sin(2 * math.pi * u); pitch = 7.0 + 2.0 * math.sin(2 * math.pi * u + 0.8)
    else:
        yaw = -24.0 + 30.0 * in_out(u, 2.2); pitch = 6.0 + 4.0 * in_out(u, 2.2)
    d = (D_A + (D_B - D_A) * refit) * CFG.get("frame_room", 1.08)
    if far is not None: d = d + (max(d, far) - d) * out
    z = Z_A + (Z_B - Z_A) * refit
    if far_z is not None: z = z + (far_z - z) * out
    look = Vector((0, 0, z))
    camera_at(f, yaw, pitch, d, look=look, fstop=CFG.get("fstop", 5.0))
