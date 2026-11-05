# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Meno contributors. Under the GPL, in this plugin's LICENSE.
"""What the two studios share (docs/plugins/blender-presets.md, *A studio*): where Meno's molecules stand, and lights and
surfaces sized to them.

A studio is drawn here in metres for a molecule at 4 cm an ångström - the size of a product on a table - and put round
Meno's molecules at their own size: the scene's unit is Meno's ångström, so every length is 25 times, and every light's
power 625 times (as bright on a molecule 25 times as far), times the key light asked for.
"""
import math

import bpy
from mathutils import Vector

SIZE = 25.0          # the scene's units (Meno's ångströms) to a metre of the studio
COLOURS = {"white": (0.8, 0.8, 0.81), "black": (0.004, 0.005, 0.007), "grey": (0.22, 0.22, 0.23)}


def stage(ctx):
    """Where Meno's molecules are: their middle, their lowest point, their reach from the middle - in the scene's units."""
    balls = [(a["object"].matrix_world.translation.copy(), a["radius"]) for m in ctx.molecules for a in m["atoms"]]
    if not balls:
        return Vector(), 0.0, 1.0
    mid = sum((p for p, _ in balls), Vector()) / len(balls)
    low = min(p.z - r for p, r in balls)
    reach = max((p - mid).length + r for p, r in balls)
    return mid, low, reach


def area(ctx, name, size, power, colour, at, target, size_y=None, gain=1.0):
    """A soft area light of the studio, never seen by the camera, at `at` looking at `target` (metres of the studio); `gain`
    the studio's own exposure, its lights drawn for one."""
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE"
    ld.size = size * SIZE
    ld.size_y = (size_y or size) * SIZE
    ld.energy = power * gain * SIZE * SIZE * float(ctx.request.get("key", 1.0))
    ld.color = colour
    ob = bpy.data.objects.new(name, ld)
    ctx.scene.collection.objects.link(ob)
    ob.location = at
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (target - at).to_track_quat("-Z", "Y")
    ob.visible_camera = False
    return ob


def put_out(ctx):
    """Meno's own lights taken away: the studio lights the molecules, as strong as the key asked for."""
    for o in [o for o in ctx.scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)


def world(ctx, colour, strength):
    w = bpy.data.worlds.new("studio")
    ctx.scene.world = w
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*colour, 1)
    bg.inputs["Strength"].default_value = strength * float(ctx.request.get("ambient", 1.0))


def surface(name, colour, roughness=0.85):
    m = bpy.data.materials.new(name)
    pr = m.node_tree.nodes["Principled BSDF"]
    pr.inputs["Base Color"].default_value = (*colour, 1)
    pr.inputs["Roughness"].default_value = roughness
    return m


def soft(ctx):
    """Shadows as soft as asked: the lights' spread narrowed for a harder shadow (softness in degrees, 10 Meno's own)."""
    spread = max(1.0, min(180.0, float(ctx.request.get("softness", 10)) * 18.0))
    for o in ctx.scene.objects:
        if o.type == "LIGHT" and o.data.type == "AREA":
            o.data.spread = math.radians(spread)


def seen_to(ctx, far):
    """Meno's camera seeing as far as the studio's backdrop (`far`, a point of it): Meno's sees only as far as its molecules."""
    cam = ctx.camera
    d = (far - cam.matrix_world.translation).length * 1.2
    if cam.data.clip_end < d:
        cam.data.clip_end = d
