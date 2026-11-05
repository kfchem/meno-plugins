# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "reveal" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Rim-light reveal: out of darkness a rim of light traces the molecule, then the key light comes up; on white, a pale ghost and its shadow develop into colour behind a fine cool line.

Keyed with the kit (motionkit.py) on Meno's molecule: docs/plugins/blender-presets.md, *A motion*.
"""
import motion_kit as kit

NOMINAL = 7.0
# (played as a cycle only until it has settled - not its Once's long last hold - so that it turns back without a pause)
CYCLE = 5.4


def LOOPS(ctx):
    """Its rims circle of themselves in the dark; on white it cycles."""
    return "cycle" if ctx.request.get("background", "transparent") in ("white", "transparent") else "seamless"


def key(ctx):
    kit.run(ctx, globals(), CYCLE if ctx.request.get("cycle") else NOMINAL)


def keyed():
    def develop(nt, sh, col, pr):
        """On white: the molecule starts as white matte (only its form and shadow), its colour developing behind a front that sweeps left to right."""
        Tm = time_value(nt)
        geo = nnew(nt, "ShaderNodeNewGeometry"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Position"], sp.inputs[0])
        x0, x1 = min(a["p"].x for a in ATOMS) - 0.05, max(a["p"].x for a in ATOMS) + 0.05
        t0, t1 = 0.8, 3.6
        front = math_(nt, "ADD", math_(nt, "MULTIPLY", math_(nt, "DIVIDE", math_(nt, "SUBTRACT", Tm, t0), t1 - t0, None, True), x1 - x0), x0)
        d = math_(nt, "SUBTRACT", front, sp.outputs["X"])           # > 0: developed
        fac = math_(nt, "DIVIDE", math_(nt, "ADD", d, 0.0), 0.06, None, True)
        ghost = nnew(nt, "ShaderNodeBsdfPrincipled", Roughness=0.4, **{"Coat Weight": 0.6, "Coat Roughness": 0.08}); ghost.inputs["Base Color"].default_value = CFG.get("ghost", (0.42, 0.43, 0.45, 1))
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(fac, mx.inputs[0]); nt.links.new(ghost.outputs[0], mx.inputs[1]); nt.links.new(sh, mx.inputs[2])
        # a fine cool line of light at the front
        band = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "POWER", math_(nt, "DIVIDE", d, 0.012), 2.0), -1.0))
        on = math_(nt, "MULTIPLY", math_(nt, "GREATER_THAN", Tm, t0), math_(nt, "LESS_THAN", Tm, t1 + 0.2))
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = (0.35, 0.65, 1.0, 1)
        nt.links.new(math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", band, on), 2.5), em.inputs["Strength"])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(mx.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        return ad.outputs[0]

    assign(hero_material(extra=develop if WHITE else None))
    lights, bd = studio()
    D = fit_distance(CFG.get("margin", 0.72))
    base = {k: v.data.energy for k, v in lights.items()}
    pool = None if WHITE else pool_of(bd)

    def put_light(ob, ang, h, r, f):
        a = math.radians(ang)
        ob.location = Vector((math.sin(a) * r, math.cos(a) * r, h))
        ob.rotation_quaternion = (Vector((0, 0, 0)) - ob.location).to_track_quat("-Z", "Y")
        ob.keyframe_insert("location", frame=f); ob.keyframe_insert("rotation_quaternion", frame=f)

    def power(name, v, f):
        lights[name].data.energy = base[name] * v; lights[name].data.keyframe_insert("energy", frame=f)

    for k in ('rimR', 'rimL'): lights[k].visible_volume_scatter = False
    near = min(ATOMS, key=lambda a: a["p"].y)["p"]
    T = CFG["seconds"]
    for f in range(1, N + 1):
        t = tsec(f)
        if WHITE and not LOOP:
            # on white: the lights stay; the camera pushes in, and focus racks onto the whole as the colour arrives
            c = in_out(lin(t, 0.0, T - 0.8), 3)
            fo = smoother(lin(t, 2.6, 4.6))
            camera_at(f, -18 + 26 * c, 3 + 6 * c, D * (1.16 - 0.16 * c), fstop=2.0 + 3.6 * fo, focus=near.lerp(Vector(), fo))
            continue
        if not LOOP:
            # the cool rim travels behind the subject, left to right, and settles
            u = in_out(lin(t, 0.5, 2.8), 3)
            put_light(lights["rimR"], -72 + 128 * u, 0.30 + 0.2 * u, 1.7, f)
            power("rimR", smooth(lin(t, 0.4, 0.9)) * (1.0 + 1.0 * bump(t, 0.5, 3.0)), f)
            if pool is not None:
                pool.default_value = 0.15 + 0.85 * smoother(lin(t, 2.8, 5.4)); pool.keyframe_insert("default_value", frame=f)
            put_light(lights["rimL"], -48, 0.2, 1.7, f)
            power("rimL", smoother(lin(t, 2.0, 3.4)), f)
            k = smoother(lin(t, 3.0, 5.2))
            power("key", k, f); power("top", smoother(lin(t, 3.4, 5.4)), f); power("fill", smoother(lin(t, 3.8, 5.6)), f)
            c = in_out(lin(t, 0.0, T - 0.8), 3)
            fo = smoother(lin(t, 3.4, 5.6))
            camera_at(f, -18 + 26 * c, 3 + 6 * c, D * (1.16 - 0.16 * c), fstop=2.0 + 3.6 * fo, focus=near.lerp(Vector(), fo))
        else:
            u = t / T
            put_light(lights["rimR"], 360 * u, 0.4, 1.7, f); put_light(lights["rimL"], 360 * u + 180, 0.25, 1.7, f)
            power("rimR", 1.0, f); power("rimL", 1.0, f); power("key", 0.18, f); power("top", 0.3, f); power("fill", 0.1, f)
            camera_at(f, 0, 6, D, fstop=4.0)
    linear_keys()
    finish(bloom=0.18, threshold=4.0)
    render()
