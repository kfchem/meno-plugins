# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "change-scan" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Scan: a sheet of light passes down through A and leaves B. Below the sheet A stands as it was, its cut edge lit; just
above it B is a hologram - a rim of light, seen through - that turns solid a little higher; the sheet goes on to the
paper and fades. A and B stand in the same place.

Keyed with the kit (motionkit.py) and the change module (change.py), on Meno's molecules: docs/blender-presets.md,
*A motion*.
"""
import motion_kit as kit

NOMINAL = 5.0
LOOPS = "seamless"


def key_change(ctx, a, b):
    kit.change(ctx, globals(), NOMINAL, a, b)


def keyed():
    T = CFG["seconds"]
    TS, TE = CFG.get("scan_at", 0.9), CFG.get("scan_to", 3.2)     # the sheet's passage: from, to
    GLOW = CFG.get("glow_rgb", (0.08, 0.3, 1.0, 1))

    # --- the sheet's height: from above both to the paper, eased
    ZTOP = max(max(a["p"].z + a["r"] for a in ATOMS), max(a["p"].z + a["r"] for a in BA)) + 0.5 * K
    ZBOT = min(min(a["p"].z - a["r"] for a in ATOMS), min(a["p"].z - a["r"] for a in BA)) - 0.3 * K
    BAND = CFG.get("band", 0.16) * (ZTOP - ZBOT) + 0.25 * K     # how far above the sheet B is a hologram
    EDGE = CFG.get("edge", 0.035) * (ZTOP - ZBOT) + 0.05 * K    # how far below it A's cut is lit
    def height(t): return ZTOP - (ZTOP - ZBOT) * in_out(lin(t, TS, TE), 2.0)

    def height_node(nt):
        """The sheet's height at the frame's time, as the shader has it: the same easing as `height` (in_out, power 2)."""
        Tm = time_value(nt)
        x = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", Tm, TS), TE - TS, clamp_=True)
        lo = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", x, x), 2.0)                                  # 2x^2, x < 0.5
        one_x = math_(nt, "SUBTRACT", 1.0, x)
        hi = math_(nt, "SUBTRACT", 1.0, math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", one_x, one_x), 2.0))   # 1 - 2(1-x)^2
        e = nnew(nt, "ShaderNodeMix", data_type="FLOAT")
        nt.links.new(math_(nt, "GREATER_THAN", x, 0.5), e.inputs[0]); nt.links.new(lo, e.inputs[2]); nt.links.new(hi, e.inputs[3])
        return math_(nt, "SUBTRACT", ZTOP, math_(nt, "MULTIPLY", e.outputs[0], ZTOP - ZBOT))

    def world_z(nt):
        g = nnew(nt, "ShaderNodeNewGeometry"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(g.outputs["Position"], sp.inputs[0])
        return sp.outputs["Z"]

    def a_cut(nt, sh, col, pr):
        """A: there below the sheet, gone above it; its cut edge lit."""
        H = height_node(nt); z = world_z(nt)
        below = math_(nt, "LESS_THAN", z, H)
        d = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", H, z), EDGE)
        edge = math_(nt, "MULTIPLY", math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", d, d), -1.0)), below)
        live = math_(nt, "LESS_THAN", H, ZTOP - 1e-4)         # (no glow before the sheet comes)
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = GLOW
        nt.links.new(math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", edge, live), CFG.get("edge_glow_white", 3.0) if WHITE else CFG.get("edge_glow", 12.0)), em.inputs["Strength"])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        tr = nnew(nt, "ShaderNodeBsdfTransparent")
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(below, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(ad.outputs[0], mx.inputs[2])
        return mx.outputs[0]

    def b_rise(nt, sh, col, pr):
        """B: gone below the sheet; a hologram just above it - a rim of light, seen through; solid above that."""
        H = height_node(nt); z = world_z(nt)
        above = math_(nt, "GREATER_THAN", z, H)
        # (solid above the band - and all of it once the sheet has reached the paper: the band closes behind it)
        after = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", time_value(nt), TE), 0.35, clamp_=True)
        solid = smooth_node(nt, math_(nt, "ADD", math_(nt, "DIVIDE", math_(nt, "SUBTRACT", z, H), BAND), after))
        lw = nnew(nt, "ShaderNodeLayerWeight", Blend=0.4)
        rim = math_(nt, "POWER", math_(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 1.6)
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = GLOW
        nt.links.new(math_(nt, "MULTIPLY", math_(nt, "MULTIPLY_ADD", rim, 6.0, 0.6), CFG.get("holo_white", 2.0) if WHITE else CFG.get("holo", 4.0)), em.inputs["Strength"])
        tr = nnew(nt, "ShaderNodeBsdfTransparent")
        holo = nnew(nt, "ShaderNodeAddShader"); nt.links.new(em.outputs[0], holo.inputs[0]); nt.links.new(tr.outputs[0], holo.inputs[1])
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(solid, mx.inputs[0]); nt.links.new(holo.outputs[0], mx.inputs[1]); nt.links.new(sh, mx.inputs[2])
        tr2 = nnew(nt, "ShaderNodeBsdfTransparent")
        out = nnew(nt, "ShaderNodeMixShader"); nt.links.new(above, out.inputs[0]); nt.links.new(tr2.outputs[0], out.inputs[1]); nt.links.new(mx.outputs[0], out.inputs[2])
        return out.outputs[0]

    def smooth_node(nt, x):
        x = math_(nt, "MINIMUM", math_(nt, "MAXIMUM", x, 0.0), 1.0)
        return math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", x, x), math_(nt, "SUBTRACT", 3.0, math_(nt, "MULTIPLY", x, 2.0)))

    mA = hero_material("heroA", extra=a_cut)
    assign(mA)
    mB = hero_material("heroB", extra=b_rise)
    # (their glow lights nothing round them: not sampled as lights, so that a molecule at rest is rendered as the rig renders
    # it - the piece's last frame the next one's first)
    for m_ in (mA, mB):
        m_.cycles.emission_sampling = "NONE"

    # --- B's objects, whole from the start (as the rig draws a molecule), seen only above the sheet
    for ob in COL.objects:
        if ob.type == "MESH": ob.parent = None
    for ob in ATOM_OBJ: ob.location = ATOMS[ob["idx"]]["p"]
    BOBJ = []
    for j, a in enumerate(BA):
        ob = bpy.data.objects.new(f"Batom{j:03d}", SPHERE); COL.objects.link(ob)
        ob.location = a["p"]; ob.scale = (a["r"],) * 3
        ob["col"] = a["col"]; ob["colB"] = a["col"]; ob["stick"] = 0.0
        BOBJ.append(ob)
    BOBJ += [x[0] for x in b_sticks()]
    assign(mB, BOBJ)
    # (each molecule seen only while some of it is: A hidden once the sheet is below it, B before the sheet reaches it - so
    # that neither costs a frame where it is all see-through)
    for ob in ATOM_OBJ + [b[0] for b in BOND_OBJ]:
        ob.hide_render = False; ob.keyframe_insert("hide_render", frame=1)
        ob.hide_render = True; ob.keyframe_insert("hide_render", frame=int(TE * FPSM) + 3)
    for ob in BOBJ:
        ob.hide_render = True; ob.keyframe_insert("hide_render", frame=1)
        ob.hide_render = False; ob.keyframe_insert("hide_render", frame=max(2, int(TS * FPSM)))

    # --- the sheet: a disc of light, brightest at its rim and where it is seen edge on, fading out at its edge
    import bmesh
    SR = CFG.get("sheet_scale", 1.25) * max(max(Vector((a["p"].x, a["p"].y)).length for a in ATOMS + BA), 0.1) + 0.6 * K
    me = bpy.data.meshes.new("sheet"); bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, segments=128, radius=SR); bm.to_mesh(me); bm.free()
    SHEET = bpy.data.objects.new("sheet", me); COL.objects.link(SHEET)
    ms, nts, outs = new_material("sheet")
    co = nnew(nts, "ShaderNodeTexCoord"); gr = nnew(nts, "ShaderNodeTexGradient", gradient_type="SPHERICAL")
    mp = nnew(nts, "ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / SR, 1 / SR, 1 / SR)
    nts.links.new(co.outputs["Object"], mp.inputs["Vector"]); nts.links.new(mp.outputs["Vector"], gr.inputs["Vector"])
    fall = math_(nts, "POWER", gr.outputs["Fac"], 0.6)                         # (1 at the middle, 0 at the rim: soft)
    lw = nnew(nts, "ShaderNodeLayerWeight", Blend=0.15)
    graze = math_(nts, "MULTIPLY_ADD", lw.outputs["Facing"], 2.5, 0.4)
    fade = nnew(nts, "ShaderNodeValue"); fade.label = "fade"
    em = nnew(nts, "ShaderNodeEmission"); em.inputs["Color"].default_value = GLOW
    nts.links.new(math_(nts, "MULTIPLY", math_(nts, "MULTIPLY", math_(nts, "MULTIPLY", fall, graze), fade.outputs[0]), CFG.get("sheet_white", 0.5) if WHITE else CFG.get("sheet", 3.0)), em.inputs["Strength"])
    tr = nnew(nts, "ShaderNodeBsdfTransparent")
    ad = nnew(nts, "ShaderNodeAddShader"); nts.links.new(em.outputs[0], ad.inputs[0]); nts.links.new(tr.outputs[0], ad.inputs[1])
    nts.links.new(ad.outputs[0], outs.inputs["Surface"])
    me.materials.append(ms)
    SHEET.visible_shadow = False
    # (its rim: a hoop of cobalt, seen on white as a line of colour, not only as light)
    cu = bpy.data.curves.new("hoopc", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = CFG.get("hoop_r", 0.012) * SR; cu.bevel_resolution = 6
    sp_ = cu.splines.new("NURBS"); n_ = 64; sp_.points.add(n_ - 1); sp_.use_cyclic_u = True
    for k in range(n_):
        a_ = 2 * math.pi * k / n_; sp_.points[k].co = (SR * math.cos(a_), SR * math.sin(a_), 0.0, 1.0)
    HOOP = bpy.data.objects.new("hoop", cu); COL.objects.link(HOOP)
    mh, nth, outh = new_material("hoop")
    ph = nnew(nth, "ShaderNodeBsdfPrincipled", Roughness=0.25); ph.inputs["Base Color"].default_value = GLOW
    ph.inputs["Emission Color"].default_value = GLOW
    hfade = nnew(nth, "ShaderNodeValue")
    nth.links.new(math_(nth, "MULTIPLY", hfade.outputs[0], CFG.get("hoop_glow_white", 1.0) if WHITE else CFG.get("hoop_glow", 10.0)), ph.inputs["Emission Strength"])
    trh = nnew(nth, "ShaderNodeBsdfTransparent")
    mxh = nnew(nth, "ShaderNodeMixShader"); nth.links.new(hfade.outputs[0], mxh.inputs[0]); nth.links.new(trh.outputs[0], mxh.inputs[1]); nth.links.new(ph.outputs[0], mxh.inputs[2])
    nth.links.new(mxh.outputs[0], outh.inputs["Surface"])
    cu.materials.append(mh); HOOP.visible_shadow = False
    for f in range(1, N + 1):
        t = tsec(f)
        SHEET.location = (0, 0, height(t)); SHEET.keyframe_insert("location", frame=f)
        HOOP.location = (0, 0, height(t)); HOOP.keyframe_insert("location", frame=f)
        fade.outputs[0].default_value = smooth(lin(t, TS - 0.35, TS)) * (1 - smooth(lin(t, TE, TE + 0.45)))
        fade.outputs[0].keyframe_insert("default_value", frame=f)
        hfade.outputs[0].default_value = fade.outputs[0].default_value; hfade.outputs[0].keyframe_insert("default_value", frame=f)
    SHEET.hide_render = True; SHEET.keyframe_insert("hide_render", frame=1)
    SHEET.hide_render = False; SHEET.keyframe_insert("hide_render", frame=max(2, int((TS - 0.4) * FPSM)))
    SHEET.hide_render = True; SHEET.keyframe_insert("hide_render", frame=int((TE + 0.5) * FPSM) + 1)
    # (a light that goes with the sheet, down on to the paper, where asked: the scan seen on the floor - none by default, as
    # it shows the molecule's shadow as blotches)
    hidden_with_sheet = [HOOP]
    if CFG.get("scan_light", 0.0) > 0:
        glow = area("scanlight", 2 * SR, 0.0, GLOW[:3], Vector((0, 0, ZTOP)), target=Vector((0, 0, -1.0)), shape="DISK")
        for f in range(1, N + 1):
            t = tsec(f)
            glow.location = (0, 0, height(t) + 0.002); glow.keyframe_insert("location", frame=f)
            glow.data.energy = CFG["scan_light"] * smooth(lin(t, TS - 0.35, TS)) * (1 - smooth(lin(t, TE, TE + 0.45)))
            glow.data.keyframe_insert("energy", frame=f)
        hidden_with_sheet.append(glow)
    # (the hoop and any light there only while the sheet is: a light of no strength left in would still change how the others
    # are sampled, and the piece's last frame would not be the next one's first)
    for ob_ in hidden_with_sheet:
        ob_.hide_render = True; ob_.keyframe_insert("hide_render", frame=1)
        ob_.hide_render = False; ob_.keyframe_insert("hide_render", frame=max(2, int((TS - 0.4) * FPSM)))
        ob_.hide_render = True; ob_.keyframe_insert("hide_render", frame=int((TE + 0.5) * FPSM) + 1)

    # --- the camera: the film's one move, refitting from A to B as the sheet passes
    lights, cyc = studio()
    for f in range(1, N + 1):
        change_camera(f, in_out(lin(tsec(f), TS, TE + 0.4), 2.4))
    for a in bpy.data.actions:
        for fc in all_fcurves(a):
            if fc.data_path == "hide_render": continue
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    print(f"SCAN from {ZTOP / K:.1f} A to {ZBOT / K:.1f} A, {TS:.2f}-{TE:.2f}s, sheet {SR / K:.1f} A", flush=True)
    finish(bloom=0.5, threshold=2.5, glow_size=0.55)
    render()
