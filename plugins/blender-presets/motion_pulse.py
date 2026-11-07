# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "pulse" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Pulse: a wave of energy runs out from the nitrogens (else the middle) along the bonds; on white, a cobalt wave that throws a pool of blue light onto the paper.

Keyed with the kit (motionkit.py) on Meno's molecule: docs/blender-presets.md, *A motion*.
"""
import motion_kit as kit

NOMINAL = 6.0
LOOPS = "seamless"


def key(ctx):
    kit.run(ctx, globals(), NOMINAL)


def keyed():
    # (from the nitrogens; a molecule with none, from the heavy atom nearest its middle)
    SRC = [i for i, a in enumerate(ATOMS) if a["el"] == "N"] or [min((i for i, a in enumerate(ATOMS) if a["el"] != "H"), key=lambda i: ATOMS[i]["p"].length, default=0)]
    DIST = graph_dist(SRC)
    for ob in ATOM_OBJ: ob["d"] = float(DIST[ob["idx"]])
    for ob, bi, off in BOND_OBJ:
        ob["dA"] = float(DIST[BONDS[bi]["a"]]); ob["dB"] = float(DIST[BONDS[bi]["b"]])
    V = CFG.get("speed", 9.0)          # bonds a second (faster: the maintainer, 2026-10-16)
    T0 = 0.0 if LOOP else 1.5          # release
    EVERY = CFG["seconds"] / CFG.get("pulses", 3) if LOOP else 1e6
    W, TAIL = 0.16, CFG.get("tail", 0.35 if WHITE else 0.8)
    ENERGY = CFG.get("energy", (0.0, 0.18, 1.0, 1) if WHITE else (0.02, 0.35, 1.0, 1))
    ECHO = 0.0 if LOOP else 0.55

    def pulse_shader(nt, sh, col, pr):
        T = time_value(nt)
        co = nnew(nt, "ShaderNodeTexCoord"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(co.outputs["Object"], sp.inputs[0])
        u = along(nt, sp.outputs["Z"])
        dA = attr(nt, "dA").outputs["Fac"]; dB = attr(nt, "dB").outputs["Fac"]; d = attr(nt, "d").outputs["Fac"]
        stick = attr(nt, "stick").outputs["Fac"]
        ds = math_(nt, "MINIMUM", math_(nt, "ADD", dA, u), math_(nt, "ADD", dB, math_(nt, "SUBTRACT", 1.0, u)))
        deff = nnew(nt, "ShaderNodeMix", data_type="FLOAT"); nt.links.new(stick, deff.inputs[0]); nt.links.new(d, deff.inputs[2]); nt.links.new(ds, deff.inputs[3])
        ph = math_(nt, "SUBTRACT", T, T0)
        if LOOP: ph = math_(nt, "FLOORED_MODULO", ph, EVERY)
        def glow(shift, k):
            x = math_(nt, "SUBTRACT", math_(nt, "MULTIPLY", math_(nt, "SUBTRACT", ph, shift), V), deff.outputs[0])
            xw = math_(nt, "DIVIDE", x, W)
            head = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", xw, xw), -1.0))
            xa = math_(nt, "DIVIDE", x, 0.6)
            heada = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", xa, xa), -1.0))
            tail = math_(nt, "MULTIPLY", math_(nt, "GREATER_THAN", x, 0.0), math_(nt, "EXPONENT", math_(nt, "DIVIDE", x, -TAIL)))
            return math_(nt, "MULTIPLY", head, k), math_(nt, "MULTIPLY", tail, k), math_(nt, "MULTIPLY", heada, k)
        h1, t1, a1 = glow(0.0, 1.0)
        head, tail, heada = h1, t1, a1
        if ECHO:
            h2, t2, a2 = glow(ECHO, 0.35)
            head = math_(nt, "ADD", h1, h2); tail = math_(nt, "ADD", t1, t2); heada = math_(nt, "ADD", a1, a2)
        # charge gathering at the sources just before release
        pre = math_(nt, "POWER", math_(nt, "MAP_RANGE", 0, 0, 0) if False else math_(nt, "DIVIDE", math_(nt, "ADD", ph, 0.5) if not LOOP else math_(nt, "SUBTRACT", ph, EVERY - 0.5), 0.5, None, True), 2.0)
        issrc = math_(nt, "LESS_THAN", deff.outputs[0], 0.01)
        pre = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", pre, issrc), math_(nt, "LESS_THAN", ph, 0.0) if not LOOP else math_(nt, "GREATER_THAN", ph, EVERY - 0.5))
        lw = nnew(nt, "ShaderNodeLayerWeight", Blend=0.35)
        rimf = math_(nt, "SUBTRACT", 1.0, lw.outputs["Facing"])          # 0 facing, 1 at the edge
        rimf = math_(nt, "POWER", rimf, 1.6)
        atom = math_(nt, "SUBTRACT", 1.0, stick)
        # sticks: a hot narrow band; atoms: a glowing shell (edge) and a smaller core flash
        s_stick = math_(nt, "ADD", math_(nt, "MULTIPLY", head, 28.0), math_(nt, "MULTIPLY", tail, 3.0))
        s_atom = math_(nt, "ADD", math_(nt, "MULTIPLY", math_(nt, "ADD", math_(nt, "MULTIPLY", heada, 30.0), math_(nt, "MULTIPLY", tail, 2.5)), rimf),
                       math_(nt, "MULTIPLY", heada, 16.0))
        s_atom = math_(nt, "ADD", s_atom, math_(nt, "MULTIPLY", math_(nt, "ADD", rimf, 0.3), math_(nt, "MULTIPLY", pre, 14.0)))
        strength = math_(nt, "ADD", math_(nt, "MULTIPLY", s_stick, stick), math_(nt, "MULTIPLY", s_atom, atom))
        if WHITE:
            # on white: the wave is colour - the surface turns a saturated blue where it passes - with only a faint glow
            tint = math_(nt, "MINIMUM", math_(nt, "ADD", math_(nt, "MULTIPLY", heada, math_(nt, "SUBTRACT", 1.0, stick)), math_(nt, "ADD", math_(nt, "MULTIPLY", head, stick), math_(nt, "MULTIPLY", tail, 0.75))), 1.0)
            tint = math_(nt, "ADD", tint, math_(nt, "MULTIPLY", pre, math_(nt, "SUBTRACT", 1.0, stick)))
            blue = nnew(nt, "ShaderNodeBsdfPrincipled", Roughness=0.3, **{"Coat Weight": 1.0, "Coat Roughness": 0.04})
            # (the head a clear electric cyan, the tail settling to ultramarine)
            hue = nnew(nt, "ShaderNodeMix", data_type="RGBA"); nt.links.new(math_(nt, "MINIMUM", math_(nt, "ADD", math_(nt, "MULTIPLY", heada, math_(nt, "SUBTRACT", 1.0, stick)), math_(nt, "MULTIPLY", head, stick)), 1.0), hue.inputs[0])
            hue.inputs[6].default_value = CFG.get("tail_rgb", (0.012, 0.03, 0.55, 1)); hue.inputs[7].default_value = CFG.get("head_rgb", (0.0, 0.32, 0.95, 1))
            nt.links.new(hue.outputs[2], blue.inputs["Base Color"])
            mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(math_(nt, "MINIMUM", tint, 1.0), mx.inputs[0]); nt.links.new(sh, mx.inputs[1]); nt.links.new(blue.outputs[0], mx.inputs[2])
            # the front throws a pool of blue light onto the paper as it passes, but the camera sees no glow on the molecule itself -
            # only the deep lacquered colour (light added to a lit white surface goes pastel) and a small spark at the head
            lp = nnew(nt, "ShaderNodeLightPath")
            unseen = math_(nt, "SUBTRACT", 1.0, lp.outputs["Is Camera Ray"])
            cast = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", strength, unseen), CFG.get("white_glow", 5.0))
            spark = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", math_(nt, "ADD", math_(nt, "MULTIPLY", heada, math_(nt, "SUBTRACT", 1.0, stick)), math_(nt, "MULTIPLY", head, stick)), lp.outputs["Is Camera Ray"]), CFG.get("spark", 1.0))
            em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = CFG.get("glow_rgb", (0.0, 0.22, 1.0, 1)); nt.links.new(math_(nt, "ADD", cast, spark), em.inputs["Strength"])
            add = nnew(nt, "ShaderNodeAddShader"); nt.links.new(mx.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
            return add.outputs[0]
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = ENERGY; nt.links.new(strength, em.inputs["Strength"])
        add = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1])
        return add.outputs[0]

    assign(hero_material(extra=pulse_shader))
    lights, bd = studio(key=CFG.get("key", 0.55), rim=0.8)
    for k in ("rimR", "rimL"): lights[k].visible_volume_scatter = False
    if WHITE and "paper" in bpy.data.collections:
        # (the molecule's own light lands on the paper alone: lit by it, neighbouring atoms would wash each other pastel)
        for ob in COL.objects:
            if ob.type == "MESH": ob.light_linking.receiver_collection = bpy.data.collections["paper"]
    D = fit_distance(CFG.get("margin", 0.72))

    def head_at(t, d):
        ph = t - T0
        if LOOP: ph = ph % EVERY
        s = 0.0
        for shift, k in ((0.0, 1.0), (ECHO, 0.35)) if ECHO else ((0.0, 1.0),):
            x = (ph - shift) * V - d
            s += k * math.exp(-(x / 0.8) ** 2)
        return s

    for f in range(1, N + 1):
        t = tsec(f)
        for ob in ATOM_OBJ:
            r = ob["r"] * (1 + 0.10 * head_at(t - 0.04, ob["d"]))
            ob.scale = (r, r, r); ob.keyframe_insert("scale", frame=f)
        if LOOP:
            ROOT.rotation_euler = (0, 0, 2 * math.pi * (f - 1) / N); ROOT.keyframe_insert("rotation_euler", frame=f)
            camera_at(f, 0, 7, D, fstop=4.5)
        else:
            c = in_out(lin(t, 0.0, CFG["seconds"] - 0.7), 3)
            camera_at(f, -10 + 16 * c, 5 + 3 * c, D * (1.08 - 0.10 * c), fstop=4.0)
    linear_keys()
    finish(bloom=0.9, threshold=1.6, glow_size=0.62)
    render()
