# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Meno contributors. Under the GPL, in this plugin's LICENSE.
"""The studio "cyclorama" (manifest `presets.studios`): a wall far behind, lit as a soft pool by its own light and seen by
the camera alone, in the background's colour, with a key, a fill, a top light and two coloured rims behind the molecules.
On a transparent background - the studio being the backdrop chosen - the wall is dark."""
import math

import bmesh
import bpy
from mathutils import Vector

import studio_common as common

S = common.SIZE
# (its lights as bright as drawn for, at its own exposure)
GAIN = 2 ** -2.9


def setup(ctx):
    mid, low, reach = common.stage(ctx)
    bg = ctx.request.get("background", "transparent")
    common.put_out(ctx)
    common.world(ctx, (0.0035, 0.0042, 0.006), GAIN)
    at = lambda x, y, z: mid + Vector((x, y, z)) * S
    common.area(ctx, "key", 1.4, 700, (1.0, 0.95, 0.9), at(-1.4, -1.6, 1.3), mid, 0.9, gain=GAIN)
    common.area(ctx, "fill", 2.0, 120, (0.85, 0.92, 1.0), at(1.8, -1.2, -0.3), mid, 1.2, gain=GAIN)
    common.area(ctx, "top", 2.4, 260, (1, 1, 1), at(0.0, -0.3, 1.9), mid, 0.18, gain=GAIN)
    common.area(ctx, "rimR", 0.14, 1400, (0.45, 0.75, 1.0), at(1.3, 1.5, 0.5), mid, 1.6, gain=GAIN)
    common.area(ctx, "rimL", 0.14, 1000, (1.0, 0.62, 0.35), at(-1.4, 1.4, 0.2), mid, 1.6, gain=GAIN)
    common.soft(ctx)
    pool = common.COLOURS.get(bg, common.COLOURS["black"])
    me = bpy.data.meshes.new("wall")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=6.0 * S)
    bm.to_mesh(me)
    bm.free()
    wall = bpy.data.objects.new("wall", me)
    ctx.scene.collection.objects.link(wall)
    wall.location = (mid.x, mid.y + max(3.2 * S, 4 * reach), mid.z)
    wall.rotation_euler = (math.radians(90), 0, 0)
    common.seen_to(ctx, wall.location)
    m = bpy.data.materials.new("wall")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.42 / S, 0.62 / S, 1)
    mp.inputs["Location"].default_value = (0.0, 0.25, 0)
    gr = nt.nodes.new("ShaderNodeTexGradient")
    gr.gradient_type = "SPHERICAL"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].color = (0.0, 0.0, 0.0, 1)
    e[1].position = 1.0
    e[1].color = (*[max(0.045, c) for c in pool], 1)
    ramp.color_ramp.interpolation = "EASE"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    nt.links.new(mp.outputs[0], gr.inputs[0])
    nt.links.new(gr.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], em.inputs[0])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    wall.visible_glossy = False
    wall.visible_diffuse = False
    wall.visible_transmission = False
    wall.visible_shadow = False
