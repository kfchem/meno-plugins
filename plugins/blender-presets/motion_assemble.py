# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "assemble" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Assemble: the atoms fly in on curves and land with a squash, in waves outward from the middle, and each bond is drawn by a bright front.

Keyed with the kit (motionkit.py) on Meno's molecule: docs/plugins/blender-presets.md, *A motion*.
"""
import motion_kit as kit

NOMINAL = 8.0
LOOPS = "cycle"
# (played as a cycle only until it has settled, so that it turns back without a pause)
CYCLE = 6.0


def key(ctx):
    kit.run(ctx, globals(), CYCLE if ctx.request.get("cycle") else NOMINAL)


def keyed():
    FLASH = CFG.get("flash", (0.25, 0.65, 1.0, 1))
    heavy = [i for i, a in enumerate(ATOMS) if a["el"] != "H"] or list(range(len(ATOMS)))
    core = min(heavy, key=lambda i: ATOMS[i]["p"].length)
    DIST = graph_dist([core])
    maxd = max(1, max(DIST[i] for i in heavy if DIST[i] < 1e8))
    T_START, T_SPAN = 0.4, 2.9
    F_HEAVY, F_H = 1.25, 0.8
    arrive = [0.0] * len(ATOMS)
    for i, a in enumerate(ATOMS):
        if a["el"] != "H":
            arrive[i] = T_START + T_SPAN * min(DIST[i], maxd) / maxd + F_HEAVY + 0.12 * random.random()
    for i, a in enumerate(ATOMS):
        if a["el"] == "H":
            arrive[i] = (arrive[NB[i][0]] if NB[i] else T_START + T_SPAN + F_HEAVY) + 0.35 + 0.3 * random.random()
    up = Vector((0, 0, 1))
    paths = []
    for i, a in enumerate(ATOMS):
        P = a["p"]; d = P.normalized() if P.length > 1e-6 else Vector((1, 0, 0))
        d = (d + Vector((random.uniform(-.5, .5), random.uniform(-.6, .2), random.uniform(-.5, .5)))).normalized()
        S0 = P + d * RADIUS * random.uniform(2.6, 3.6)
        tan = d.cross(up).normalized() if d.cross(up).length > 1e-3 else Vector((1, 0, 0))
        C = S0.lerp(P, 0.55) + tan * RADIUS * random.uniform(0.8, 1.4)          # all swirl the same way
        paths.append((S0, C, P))

    def bez(S0, C, P, u):
        return S0 * (1 - u) ** 2 + C * 2 * u * (1 - u) + P * u * u

    for ob in ATOM_OBJ: ob["ta"] = arrive[ob["idx"]]; ob["fl"] = 0.35 if ATOMS[ob["idx"]]["el"] == "H" else 1.0
    for ob, bi, off in BOND_OBJ:
        a, b = BONDS[bi]["a"], BONDS[bi]["b"]
        ob["tb"] = max(arrive[a], arrive[b]) + 0.06
        ob["flip"] = 1.0 if arrive[a] > arrive[b] else 0.0      # grows from the atom that landed first

    def assemble_shader(nt, sh, col, pr):
        T = time_value(nt)
        stick = attr(nt, "stick").outputs["Fac"]
        co = nnew(nt, "ShaderNodeTexCoord"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(co.outputs["Object"], sp.inputs[0])
        u = along(nt, sp.outputs["Z"])
        flip = attr(nt, "flip").outputs["Fac"]
        ud = math_(nt, "ABSOLUTE", math_(nt, "SUBTRACT", u, flip))        # 0 at the growing end
        since = math_(nt, "SUBTRACT", T, attr(nt, "tb").outputs["Fac"])
        front = math_(nt, "DIVIDE", since, 0.24)
        shown = math_(nt, "LESS_THAN", ud, front)
        vis = math_(nt, "MAXIMUM", shown, math_(nt, "SUBTRACT", 1.0, stick))
        band = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "POWER", math_(nt, "DIVIDE", math_(nt, "SUBTRACT", front, ud), 0.12), 2.0), -1.0))
        band = math_(nt, "MULTIPLY", band, math_(nt, "LESS_THAN", front, 1.15))
        after = math_(nt, "SUBTRACT", since, 0.24)
        flash = math_(nt, "MULTIPLY", math_(nt, "GREATER_THAN", after, 0.0), math_(nt, "EXPONENT", math_(nt, "DIVIDE", after, -0.32)))
        s_stick = math_(nt, "ADD", math_(nt, "MULTIPLY", band, 40.0), math_(nt, "MULTIPLY", flash, 6.0))
        sa = math_(nt, "SUBTRACT", T, attr(nt, "ta").outputs["Fac"])
        lw = nnew(nt, "ShaderNodeLayerWeight", Blend=0.35)
        rimf = math_(nt, "POWER", math_(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 1.5)
        aflash = math_(nt, "MULTIPLY", math_(nt, "GREATER_THAN", sa, 0.0), math_(nt, "EXPONENT", math_(nt, "DIVIDE", sa, -0.28)))
        s_atom = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", aflash, attr(nt, "fl").outputs["Fac"]), math_(nt, "MULTIPLY_ADD", rimf, 14.0, 1.5))
        strength = math_(nt, "ADD", math_(nt, "MULTIPLY", s_stick, stick), math_(nt, "MULTIPLY", s_atom, math_(nt, "SUBTRACT", 1.0, stick)))
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = FLASH; nt.links.new(strength, em.inputs["Strength"])
        add = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
        tr = nnew(nt, "ShaderNodeBsdfTransparent")
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(vis, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(add.outputs[0], mx.inputs[2])
        return mx.outputs[0]

    assign(hero_material(extra=assemble_shader))
    lights, bd = studio()
    for k in ("rimR", "rimL"): lights[k].visible_volume_scatter = False
    D = fit_distance(CFG.get("margin", 0.72))
    T = CFG["seconds"]
    for f in range(1, N + 1):
        t = tsec(f)
        for ob in ATOM_OBJ:
            i = ob["idx"]; F = F_H if ATOMS[i]["el"] == "H" else F_HEAVY
            u = (t - (arrive[i] - F)) / F
            if u <= 0:
                ob.location = paths[i][0]; s = 0.0
            else:
                ob.location = bez(*paths[i], out_back(u, 1.25) if u < 1 else 1.0)
                s = 0.35 + 0.65 * smooth(u / 0.4)
                dt = t - arrive[i]
                if dt > 0: s *= 1 + 0.13 * math.exp(-5.5 * dt) * math.sin(2 * math.pi * 2.6 * dt + math.pi)
            r = ob["r"] * s; ob.scale = (r, r, r)
            ob.keyframe_insert("location", frame=f); ob.keyframe_insert("scale", frame=f)
        c = in_out(lin(t, 0.0, T - 0.8), 3)
        camera_at(f, -32 + 40 * c, 2 + 7 * c, D * (1.28 - 0.28 * c), fstop=4.5)
    linear_keys()
    finish(bloom=0.6, threshold=2.0, glow_size=0.6)
    render()
