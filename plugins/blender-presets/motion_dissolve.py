# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "dissolve" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Dissolve: the molecule burns away from one side into embers (on white, amber grains cooling to graphite dust) that drift off.

Keyed with the kit (motionkit.py) on Meno's molecule: docs/blender-presets.md, *A motion*.
"""
import motion_kit as kit

NOMINAL = 7.0
LOOPS = "cycle"
# (played as a cycle only until it has settled, so that it turns back without a pause)
CYCLE = 5.6


def key(ctx):
    kit.run(ctx, globals(), CYCLE if ctx.request.get("cycle") else NOMINAL)


def keyed():
    T = CFG["seconds"]
    T1, T2 = 1.2, 5.0                       # the front's start and end
    xs = [a["p"].x for a in ATOMS]
    XMIN, XMAX = min(xs) - 0.5 * K, max(xs) + 0.5 * K
    NSCALE, NAMP = 9.0, 0.35
    SPAN = (XMAX - XMIN)

    def front_shader(nt, s_out):
        """s: where a surface point is in the front's order, 0 first .. ~1 last."""
        geo = nnew(nt, "ShaderNodeNewGeometry"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Position"], sp.inputs[0])
        nz = nnew(nt, "ShaderNodeTexNoise", Scale=NSCALE, Detail=2.0, Roughness=0.5); nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
        s = math_(nt, "ADD", math_(nt, "DIVIDE", math_(nt, "SUBTRACT", sp.outputs["X"], XMIN), SPAN), math_(nt, "MULTIPLY", math_(nt, "SUBTRACT", nz.outputs[0], 0.5), NAMP * 2))
        return s

    def dissolve_shader(nt, sh, col, pr):
        Tm = time_value(nt)
        s = front_shader(nt, None)
        fr = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", Tm, T1), T2 - T1)
        fr = math_(nt, "SUBTRACT", math_(nt, "MULTIPLY", fr, 1.0 + 2 * NAMP), NAMP)      # covers the noise's reach
        gone = math_(nt, "LESS_THAN", s, fr)
        edge = math_(nt, "SUBTRACT", s, fr)                 # > 0 still there; small: about to go
        heat = math_(nt, "EXPONENT", math_(nt, "DIVIDE", edge, -0.025))
        heat = math_(nt, "MULTIPLY", heat, math_(nt, "SUBTRACT", 1.0, gone))
        bb = nnew(nt, "ShaderNodeBlackbody"); nt.links.new(math_(nt, "MULTIPLY_ADD", heat, 1900.0, 1000.0), bb.inputs[0])
        em = nnew(nt, "ShaderNodeEmission"); nt.links.new(bb.outputs[0], em.inputs["Color"]); nt.links.new(math_(nt, "MULTIPLY", math_(nt, "POWER", heat, 1.5), 5.0 if WHITE else 14.0), em.inputs["Strength"])
        # the surface darkens (chars) just before it goes
        char = math_(nt, "EXPONENT", math_(nt, "DIVIDE", edge, -0.09))
        dark = nnew(nt, "ShaderNodeBsdfDiffuse"); dark.inputs["Color"].default_value = (0.004, 0.003, 0.003, 1)
        m1 = nnew(nt, "ShaderNodeMixShader"); nt.links.new(math_(nt, "MULTIPLY", char, 0.97), m1.inputs[0]); nt.links.new(sh, m1.inputs[1]); nt.links.new(dark.outputs[0], m1.inputs[2])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(m1.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        tr = nnew(nt, "ShaderNodeBsdfTransparent")
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(gone, mx.inputs[0]); nt.links.new(ad.outputs[0], mx.inputs[1]); nt.links.new(tr.outputs[0], mx.inputs[2])
        return mx.outputs[0]

    assign(hero_material(extra=dissolve_shader))
    lights, bd = studio()
    for k in ("rimR", "rimL"): lights[k].visible_volume_scatter = False

    # --- the embers: Geometry Nodes on a points object -----------------------------------
    def gn_embers():
        g = bpy.data.node_groups.new("embers", "GeometryNodeTree")
        g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n, l = g.nodes, g.links
        def N_(kind, **kw):
            x = n.new(kind)
            for k, v in kw.items():
                if k in x.inputs: x.inputs[k].default_value = v
                else: setattr(x, k, v)
            return x
        def M(op, a, b=None, c=None):
            x = N_("ShaderNodeMath", operation=op)
            for i, v in enumerate((a, b, c)):
                if v is None: continue
                if isinstance(v, (int, float)): x.inputs[i].default_value = v
                else: l.new(v, x.inputs[i])
            return x.outputs[0]
        def V(op, a, b=None, scale=None):
            x = N_("ShaderNodeVectorMath", operation=op)
            for i, v in enumerate((a, b)):
                if v is None: continue
                if isinstance(v, (tuple, list)): x.inputs[i].default_value = v
                else: l.new(v, x.inputs[i])
            if scale is not None:
                if isinstance(scale, (int, float)): x.inputs["Scale"].default_value = scale
                else: l.new(scale, x.inputs["Scale"])
            return x.outputs[0]
        ci = N_("GeometryNodeCollectionInfo"); ci.inputs["Collection"].default_value = COL; ci.transform_space = "ORIGINAL"
        ri = N_("GeometryNodeRealizeInstances"); l.new(ci.outputs[0], ri.inputs[0])
        dp = N_("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM"); dp.inputs["Density"].default_value = CFG.get("density", 0.55e6)
        l.new(ri.outputs[0], dp.inputs["Mesh"])
        pos = N_("GeometryNodeInputPosition").outputs[0]
        # store where each grain started, so later evaluations use the same front and drift
        st = N_("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="POINT"); st.inputs["Name"].default_value = "p0"
        l.new(dp.outputs["Points"], st.inputs["Geometry"]); l.new(pos, st.inputs["Value"])
        p0 = N_("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR"); p0.inputs["Name"].default_value = "p0"; p0 = p0.outputs[0]
        sx = N_("ShaderNodeSeparateXYZ"); l.new(p0, sx.inputs[0])
        nz = N_("ShaderNodeTexNoise", noise_dimensions="3D"); nz.inputs["Scale"].default_value = NSCALE; nz.inputs["Detail"].default_value = 2.0; nz.inputs["Roughness"].default_value = 0.5
        l.new(p0, nz.inputs["Vector"])
        s = M("ADD", M("DIVIDE", M("SUBTRACT", sx.outputs["X"], XMIN), SPAN), M("MULTIPLY", M("SUBTRACT", nz.outputs[0], 0.5), NAMP * 2))
        # release time: when the front reaches s
        te = M("ADD", M("MULTIPLY", M("DIVIDE", M("ADD", s, NAMP), 1.0 + 2 * NAMP), T2 - T1), T1)
        # (the scene's time, in the motion's own: played over the film's length as the rest is - studio.py's SCALE)
        tm = M("MULTIPLY", N_("GeometryNodeInputSceneTime").outputs["Seconds"], SCALE)
        age = M("SUBTRACT", tm, te)
        rnd = N_("FunctionNodeRandomValue", data_type="FLOAT"); 
        for x in rnd.inputs:
            if x.type == "VALUE" and x.name in ("Min", "Max"): x.default_value = 0.0 if x.name == "Min" else 1.0
        rv = N_("FunctionNodeRandomValue", data_type="FLOAT_VECTOR"); 
        for x in rv.inputs:
            if x.type == "VECTOR" and x.name in ("Min", "Max"): x.default_value = (-1, -1, -1) if x.name == "Min" else (1, 1, 1)
        rv.inputs["Seed"].default_value = 7
        rndo = next(o for o in rnd.outputs if o.type == 'VALUE')
        rvo = next(o for o in rv.outputs if o.type == 'VECTOR')
        life = M("MULTIPLY_ADD", rndo, 1.6, 1.6)
        a = M("MAXIMUM", age, 0.0)
        # rising and drifting right: speeds grow (heat), with a per-grain scatter and a turbulent drift
        U = RADIUS
        base_v = V("SCALE", N_("FunctionNodeInputVector", vector=(0.35, 0.05, 0.75)).outputs[0], None, U)
        scat = V("SCALE", rvo, None, 0.55 * U)
        vel = V("ADD", base_v, scat)
        disp = V("SCALE", vel, None, M("MULTIPLY_ADD", a, M("MULTIPLY", a, 0.6), 0.0) if False else M("ADD", M("MULTIPLY", a, 0.55), M("MULTIPLY", M("MULTIPLY", a, a), 0.28)))
        tv = N_("ShaderNodeTexNoise", noise_dimensions="4D"); tv.inputs["Scale"].default_value = 3.0; tv.inputs["Detail"].default_value = 1.0
        l.new(p0, tv.inputs["Vector"]); l.new(M("MULTIPLY", a, 0.35), tv.inputs["W"])
        turb = V("SCALE", V("SUBTRACT", tv.outputs["Color"], (0.5, 0.5, 0.5)), None, M("MULTIPLY", M("MULTIPLY", a, a), 0.9 * U))
        newp = V("ADD", V("ADD", p0, disp), turb)
        sp = N_("GeometryNodeSetPosition"); l.new(st.outputs[0], sp.inputs["Geometry"]); l.new(newp, sp.inputs["Position"])
        an = M("DIVIDE", a, life)
        alive = M("MULTIPLY", M("GREATER_THAN", age, 0.0), M("LESS_THAN", an, 1.0))
        rad = M("MULTIPLY", M("MULTIPLY", alive, M("POWER", M("SUBTRACT", 1.0, M("MINIMUM", an, 1.0)), 0.6)), CFG.get("grain", 0.0011 if WHITE else 0.0007) * (0.6 + 0.0))
        rad = M("MULTIPLY", rad, M("MULTIPLY_ADD", rndo, 0.9, 0.6))
        sr = N_("GeometryNodeSetPointRadius"); l.new(sp.outputs[0], sr.inputs["Points"]); l.new(rad, sr.inputs["Radius"])
        sa = N_("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT"); sa.inputs["Name"].default_value = "an"
        l.new(sr.outputs[0], sa.inputs["Geometry"]); l.new(an, sa.inputs["Value"])
        sm = N_("GeometryNodeSetMaterial"); l.new(sa.outputs[0], sm.inputs["Geometry"]); sm.inputs["Material"].default_value = ember_mat
        out = N_("NodeGroupOutput"); l.new(sm.outputs[0], out.inputs[0])
        return g

    ember_mat, nt, outn = new_material("ember")
    an = attr(nt, "an", "GEOMETRY").outputs["Fac"]
    bb = nnew(nt, "ShaderNodeBlackbody"); nt.links.new(math_(nt, "MULTIPLY_ADD", math_(nt, "EXPONENT", math_(nt, "MULTIPLY", an, -3.0)), 2300.0, 900.0), bb.inputs[0])
    em = nnew(nt, "ShaderNodeEmission"); nt.links.new(bb.outputs[0], em.inputs["Color"])
    nt.links.new(math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", math_(nt, "EXPONENT", math_(nt, "MULTIPLY", an, -4.0)), math_(nt, "SUBTRACT", 1.0, an, None, True)), CFG.get("ember", 14.0)), em.inputs["Strength"])
    if WHITE:
        # on white: the grains are matter, not light - hot orange cooling to deep red and charcoal, shadowed, a faint glow only while young
        ramp = nnew(nt, "ShaderNodeValToRGB"); e = ramp.color_ramp.elements
        # (young grains amber-gold, cooling fast through umber to graphite: a gold edge on a dark dust, not a red cloud)
        e[0].position, e[0].color = 0.0, CFG.get("young", (0.95, 0.48, 0.07, 1)); e[1].position, e[1].color = 1.0, (0.018, 0.018, 0.02, 1)
        e.new(CFG.get("cool_at", 0.12)).color = (0.09, 0.04, 0.025, 1)
        nt.links.new(an, ramp.inputs[0])
        gr = nnew(nt, "ShaderNodeBsdfPrincipled", Roughness=0.5); nt.links.new(ramp.outputs[0], gr.inputs["Base Color"])
        em.inputs["Strength"].default_value = 0.0
        for l_ in list(em.inputs["Strength"].links): nt.links.remove(l_)
        nt.links.new(math_(nt, "MULTIPLY", math_(nt, "EXPONENT", math_(nt, "MULTIPLY", an, -8.0)), 1.0), em.inputs["Strength"])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(gr.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        nt.links.new(ad.outputs[0], outn.inputs["Surface"])
    else:
        nt.links.new(em.outputs[0], outn.inputs["Surface"])

    me = bpy.data.meshes.new("embers"); emb = bpy.data.objects.new("embers", me); sc.collection.objects.link(emb)
    mod = emb.modifiers.new("embers", "NODES"); mod.node_group = gn_embers()
    emb.visible_shadow = WHITE

    D = fit_distance(CFG.get("margin", 0.72))
    for f in range(1, N + 1):
        t = tsec(f)
        c = in_out(lin(t, 0.0, T - 0.5), 2.5)
        camera_at(f, -6 + 12 * c, 5 + 2 * c, D * (1.06 - 0.08 * c), fstop=4.5)
    linear_keys()
    finish(bloom=0.5, threshold=2.5, glow_size=0.55)
    render()
