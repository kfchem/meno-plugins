# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "scan" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Scan: a sheet of light passes down the molecule, a hologram above it and the solid below.

Keyed with the kit (motionkit.py) on Meno's molecule: docs/plugins/blender-presets.md, *A motion*.
"""
import motion_kit as kit

NOMINAL = 7.0
LOOPS = "seamless"


def key(ctx):
    kit.run(ctx, globals(), NOMINAL)


def keyed():
    HOLO = CFG.get("holo", (0.05, 0.55, 1.0, 1))
    zs = [a["p"].z for a in ATOMS]
    ZTOP, ZBOT = max(zs) + 0.5 * K, min(zs) - 0.5 * K
    T = CFG["seconds"]

    def hz(t):
        if not LOOP:
            return ZTOP + (ZBOT - ZTOP) * in_out(lin(t, 1.7, 4.7), 2.2)
        u = (t / T) % 1.0
        if u < 0.5: return ZTOP + (ZBOT - ZTOP) * in_out(lin(u, 0.05, 0.42), 2.2)
        return ZBOT + (ZTOP - ZBOT) * in_out(lin(u, 0.55, 0.92), 2.2)

    def flicker(t):
        if LOOP: return 1.0
        steps = [(0.15, 0.35), (0.3, 0.0), (0.42, 0.6), (0.55, 0.15), (0.75, 1.0)]
        v = 0.0
        for a, b in steps:
            if t >= a: v = b
        return v

    # the plane's height and the hologram's brightness as keyed values the shaders read
    ctrl = bpy.data.objects.new("scan", None); sc.collection.objects.link(ctrl)
    vals = bpy.data.objects.new("scan values", None); sc.collection.objects.link(vals)

    def drv(nt, path, ob=None):
        ob = ob or vals
        v = nnew(nt, "ShaderNodeValue"); fc = v.outputs[0].driver_add("default_value")
        var = fc.driver.variables.new(); var.type = "TRANSFORMS"
        var.targets[0].id = ob; var.targets[0].transform_type = path; var.targets[0].transform_space = "WORLD_SPACE"
        fc.driver.expression = "var"
        return v.outputs[0]

    def scan_shader(nt, sh, col, pr):
        Tm = time_value(nt)
        H = drv(nt, "LOC_Z", ctrl); FL = drv(nt, "LOC_X"); RF = drv(nt, "LOC_Z")
        geo = nnew(nt, "ShaderNodeNewGeometry"); sp = nnew(nt, "ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Position"], sp.inputs[0])
        z = sp.outputs["Z"]
        rel = math_(nt, "SUBTRACT", z, H)                       # > 0: passed (solid)
        solid = math_(nt, "GREATER_THAN", rel, 0.0)
        lw = nnew(nt, "ShaderNodeLayerWeight", Blend=0.3)
        rim = math_(nt, "POWER", math_(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 2.0)
        lines = math_(nt, "MULTIPLY_ADD", math_(nt, "SINE", math_(nt, "ADD", math_(nt, "MULTIPLY", z, 2 * math.pi / (0.0045)), math_(nt, "MULTIPLY", Tm, -9.0))), 0.35, 0.65)
        nz = nnew(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=1.0); nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
        w = nnew(nt, "ShaderNodeValue"); nt.links.new(Tm, nz.inputs["W"]) if "W" in nz.inputs and nz.noise_dimensions == "4D" else None
        hol = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY_ADD", rim, 5.0, 0.35), lines)
        hol = math_(nt, "MULTIPLY", hol, FL)
        em_h = nnew(nt, "ShaderNodeEmission"); em_h.inputs["Color"].default_value = HOLO; nt.links.new(hol, em_h.inputs["Strength"])
        tr = nnew(nt, "ShaderNodeBsdfTransparent")
        holo = nnew(nt, "ShaderNodeAddShader"); nt.links.new(tr.outputs[0], holo.inputs[0]); nt.links.new(em_h.outputs[0], holo.inputs[1])
        # the hot band at the plane, on the surfaces
        bw = 0.006
        band = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "POWER", math_(nt, "DIVIDE", rel, bw), 2.0), -1.0))
        em_b = nnew(nt, "ShaderNodeEmission"); em_b.inputs["Color"].default_value = (0.75, 0.92, 1.0, 1); nt.links.new(math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", band, 45.0 / 0.7), drv(nt, "LOC_Y")), em_b.inputs["Strength"])
        # the solid keeps a cyan rim for a while after the plane (fading with distance above it)
        after = math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", math_(nt, "EXPONENT", math_(nt, "DIVIDE", rel, -0.045)), rim), RF)
        em_r = nnew(nt, "ShaderNodeEmission"); em_r.inputs["Color"].default_value = HOLO; nt.links.new(math_(nt, "MULTIPLY", after, 3.0), em_r.inputs["Strength"])
        so = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, so.inputs[0]); nt.links.new(em_r.outputs[0], so.inputs[1])
        mx = nnew(nt, "ShaderNodeMixShader"); nt.links.new(solid, mx.inputs[0]); nt.links.new(holo.outputs[0], mx.inputs[1]); nt.links.new(so.outputs[0], mx.inputs[2])
        out = nnew(nt, "ShaderNodeAddShader"); nt.links.new(mx.outputs[0], out.inputs[0]); nt.links.new(em_b.outputs[0], out.inputs[1])
        return out.outputs[0]

    assign(hero_material(extra=scan_shader))
    lights, bd = studio()
    for k in ("rimR", "rimL"): lights[k].visible_volume_scatter = False
    # the scanning sheet: a faint horizontal plane of light at the plane's height, seen by the camera only
    import bmesh
    me = bpy.data.meshes.new("sheet"); bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=RADIUS * 2.4); bm.to_mesh(me); bm.free()
    sheet = bpy.data.objects.new("sheet", me); sc.collection.objects.link(sheet)
    sheet.parent = ctrl; sheet.location = (0, 0, 0)
    m, nt, outn = new_material("sheet")
    tc = nnew(nt, "ShaderNodeTexCoord"); gr = nnew(nt, "ShaderNodeTexGradient", gradient_type="SPHERICAL")
    vm = nnew(nt, "ShaderNodeVectorMath", operation="SCALE"); vm.inputs["Scale"].default_value = 1 / (RADIUS * 1.5)
    nt.links.new(tc.outputs["Object"], vm.inputs[0]); nt.links.new(vm.outputs[0], gr.inputs[0])
    mr = nnew(nt, "ShaderNodeMath", operation="POWER"); nt.links.new(gr.outputs["Fac"], mr.inputs[0]); mr.inputs[1].default_value = 2.5
    em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = HOLO
    sheetk = nnew(nt, "ShaderNodeMath", operation="MULTIPLY"); nt.links.new(mr.outputs[0], sheetk.inputs[0]); sheetk.inputs[1].default_value = 0.0
    fcv = sheetk.inputs[1].driver_add("default_value"); var = fcv.driver.variables.new(); var.type = "TRANSFORMS"
    var.targets[0].id = vals; var.targets[0].transform_type = "LOC_Y"; fcv.driver.expression = "var"
    nt.links.new(sheetk.outputs[0], em.inputs["Strength"])
    tr = nnew(nt, "ShaderNodeBsdfTransparent"); ad = nnew(nt, "ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1]); nt.links.new(ad.outputs[0], outn.inputs["Surface"])
    me.materials.append(m)
    sheet.visible_glossy = False; sheet.visible_diffuse = False; sheet.visible_shadow = False; sheet.visible_transmission = False
    # the sheet's brightness rides in the control's Y: on while it moves
    D = fit_distance(CFG.get("margin", 0.72))
    for f in range(1, N + 1):
        t = tsec(f)
        if LOOP:
            u = (t / T) % 1.0
            on = bump(u, 0.02, 0.45) + bump(u, 0.52, 0.95)
            camera_at(f, 0, 7, D, fstop=4.5)
            ROOT.rotation_euler = (0, 0, 2 * math.pi * (f - 1) / N); ROOT.keyframe_insert("rotation_euler", frame=f)
        else:
            on = smooth(lin(t, 1.4, 1.9)) * (1 - smooth(lin(t, 4.5, 5.0)))
            c = in_out(lin(t, 0.0, T - 0.8), 3)
            camera_at(f, -12 + 22 * c, 4 + 5 * c, D * (1.1 - 0.1 * c), fstop=4.5)
        ctrl.location = (0, 0, hz(t)); ctrl.keyframe_insert("location", frame=f)
        rf = 1.0 if LOOP else 1 - smoother(lin(t, 4.6, 6.0))
        vals.location = (flicker(t), 0.7 * on, rf); vals.keyframe_insert("location", frame=f)
    linear_keys()
    finish(bloom=0.6, threshold=2.0, glow_size=0.6)
    render()
