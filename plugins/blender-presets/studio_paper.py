# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Meno contributors. Under the GPL, in this plugin's LICENSE.
"""The studio "paper" (manifest `presets.studios`): a seamless sweep - a floor curving up into a wall behind - in the
background's colour, big soft lights, and a soft source overhead pooling light on the floor round the molecules. On a
transparent background - the studio being the backdrop chosen - the paper is white."""
import math

import bmesh
import bpy
from mathutils import Vector

import studio_common as common

S = common.SIZE
# (its lights as bright as drawn for, at its own exposure)
GAIN = 2 ** -4.2


def setup(ctx):
    mid, low, reach = common.stage(ctx)
    bg = ctx.request.get("background", "transparent")
    paper = common.COLOURS.get(bg, common.COLOURS["white"])
    common.put_out(ctx)
    common.world(ctx, (0.55, 0.57, 0.6), 0.12 * GAIN)
    # (the floor, a little below the molecules - and below what Meno's camera frames, where it looks level: seen edge on, the
    # floor's line and the sweep's curve above it would cross the picture)
    zf = low - 0.1
    cam = ctx.camera
    if cam.data.type == "ORTHO" and abs((cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).z) < 0.2:
        aspect = ctx.request.get("height", 1) / max(ctx.request.get("width", 1), 1)
        zf = min(zf, cam.matrix_world.translation.z - cam.data.ortho_scale * aspect / 2 - 0.5)
    yw, r = mid.y + max(0.75 * S, 2.5 * reach), 0.45 * S    # the wall behind, and the sweep's radius
    prof = [(mid.y - 9.0 * S, zf), (yw - r, zf)]
    for k in range(1, 24):
        a = math.pi / 2 * k / 24
        prof.append((yw - r + r * math.sin(a), zf + r - r * math.cos(a)))
    prof += [(yw, zf + r), (yw, zf + 9.0 * S)]
    me = bpy.data.meshes.new("paper")
    bm = bmesh.new()
    left = [bm.verts.new((mid.x - 9.0 * S, y, z)) for y, z in prof]
    right = [bm.verts.new((mid.x + 9.0 * S, y, z)) for y, z in prof]
    for i in range(len(prof) - 1):
        bm.faces.new((left[i], right[i], right[i + 1], left[i + 1]))
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    sweep = bpy.data.objects.new("paper", me)
    ctx.scene.collection.objects.link(sweep)
    me.materials.append(common.surface("paper", paper))
    at = lambda x, y, z: mid + Vector((x, y, z)) * S
    common.area(ctx, "key", 1.4, 700, (1.0, 0.97, 0.94), at(-1.3, -1.4, 1.7), mid, 0.9, gain=GAIN)
    common.area(ctx, "fill", 2.0, 110, (0.94, 0.97, 1.0), at(1.8, -1.2, -0.1), mid, 1.2, gain=GAIN)
    common.area(ctx, "top", 2.4, 240, (1, 1, 1), at(0.0, -0.3, 1.9), mid, 0.18, gain=GAIN)
    common.area(ctx, "rimR", 0.14, 500, (0.85, 0.92, 1.0), at(1.3, 1.2, 0.5), mid, 1.4, gain=GAIN)
    common.area(ctx, "rimL", 0.14, 350, (1.0, 0.9, 0.8), at(-1.3, 1.1, 0.3), mid, 1.4, gain=GAIN)
    sky = common.area(ctx, "sky", 0.8, 330, (0.985, 1.0, 0.99), Vector((mid.x, mid.y + 0.15 * S, zf + 1.6 * S)), Vector((mid.x, mid.y + 0.35 * S, zf)), 1.4, gain=GAIN)
    sky.visible_glossy = False
    common.seen_to(ctx, Vector((mid.x, yw, zf + 9.0 * S)))
    common.soft(ctx)
