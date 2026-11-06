# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "change-shatter" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Shatter: A breaks into shards of its own spheres and sticks (a Voronoi fracture made here, Bullet rigid bodies, a kick
and the floor), and the shards gather into B's atoms, which fuse in a wave up from the paper; B's bonds are drawn after.
The rigid bodies are baked, turned to keys, then blended into the flights.

Keyed with the kit (motionkit.py) and the change module (change.py), on Meno's molecules: docs/blender-presets.md,
*A motion*.
"""
import motion_kit as kit

NOMINAL = 5.0
LOOPS = "seamless"


def key_change(ctx, a, b):
    kit.change(ctx, globals(), NOMINAL, a, b)


def keyed():
    import time

    import bmesh
    # --- the fracture: each of the molecule's spheres and sticks cut by the bisecting planes between random seeds inside it,
    # each cell closed over its cut (no add-on)

    def _cell(make, seeds, i, inner_mat=1):
        bm = bmesh.new(); make(bm)
        for f in bm.faces: f.material_index = 0
        si = seeds[i]
        for j, sj in enumerate(seeds):
            if j == i: continue
            n = (sj - si).normalized(); co = (si + sj) / 2
            res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co, plane_no=n, clear_outer=True)
            cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
            if cut:
                made = bmesh.ops.edgeloop_fill(bm, edges=cut)
                for f in made.get("faces", []): f.material_index = inner_mat
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        return bm

    def _seeds(centre, spread, k, axis=None, length=0.0):
        out = []
        for _ in range(k):
            if axis is None:
                while True:
                    v = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
                    if v.length <= 1: break
                out.append(centre + v * spread * 0.8)
            else:
                out.append(centre + axis * random.uniform(-0.5, 0.5) * length + Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))) * spread * 0.4)
        return out

    def fracture(skin, inner, per_atom=4, per_stick=2, parent=None):
        """The shards of the molecule: objects with their origins at their own centres, each remembering the atom (or bond) it came
        from and its colour. `skin` and `inner` are the outer and the cut faces' materials."""
        shards = []
        def add(bm, name, col, colB, home, kind):
            me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
            if not me.vertices: return
            c = sum((v.co for v in me.vertices), Vector()) / len(me.vertices)
            for v in me.vertices: v.co -= c
            for p in me.polygons: p.use_smooth = p.material_index == 0
            me.materials.append(skin); me.materials.append(inner)
            ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob.location = c
            ob["col"] = col; ob["colB"] = colB; ob["stick"] = 0.0 if kind == "atom" else 1.0; ob["home"] = home; ob["kind"] = kind
            shards.append(ob)
            return ob
        for i, a in enumerate(ATOMS):
            r, p = a["r"], a["p"]
            k = per_atom if a["el"] != "H" else max(2, per_atom // 2)
            seeds = _seeds(p, r, k)
            for s in range(k):
                bm = _cell(lambda b: bmesh.ops.create_icosphere(b, subdivisions=3, radius=r, matrix=Matrix.Translation(p)), seeds, s)
                add(bm, f"shard-a{i}-{s}", a["col"], a["col"], i, "atom")
        for ob, bi, off in BOND_OBJ:
            rad, ln = ob["rad"], ob.scale.z * 2 * ob["half"]
            axis = (ob.rotation_quaternion @ Vector((0, 0, 1))).normalized()
            mtx = Matrix.Translation(ob.location) @ ob.rotation_quaternion.to_matrix().to_4x4()
            seeds = _seeds(ob.location, rad, per_stick, axis, ln)
            for s in range(per_stick):
                bm = _cell(lambda b: bmesh.ops.create_cone(b, cap_ends=True, segments=16, radius1=rad, radius2=rad, depth=ln, matrix=mtx), seeds, s)
                sh = add(bm, f"shard-b{bi}-{s}", ob["col"], ob["colB"], bi, "bond")
                if sh is not None:
                    # (a stick's two colours, split at its middle: a shard takes the colour of the half it lies in)
                    c = ob["colB"] if (sh.location - ob.location).dot(axis) > 0 else ob["col"]
                    sh["col"] = c; sh["colB"] = c; sh["stick"] = 0.0
        return shards



    # (the settings of the film the maintainer approved, 2026-10-21: a gentle kick, the shards kept near, a slow gather)
    for k_, v_ in {"kick": 0.8, "turb": 1.0, "land_at": 3.7, "wave": 0.8, "flight": 1.0, "flash": 4.0}.items():
        CFG.setdefault(k_, v_)
    T = CFG["seconds"]
    TB = CFG.get("break_at", 1.0)          # A breaks
    TG = CFG.get("gather_at", 2.1)         # the shards lift off towards B
    TL = CFG.get("land_at", 3.3)           # the last shard lands
    FB, FG, FL = (int(round(t * FPSM)) + 1 for t in (TB, TG, TL))

    # --- B's objects: its atoms, to fuse as their shards land, and its sticks, to be drawn after
    BCOL = bpy.data.collections.new("B"); sc.collection.children.link(BCOL)
    b_atoms = []
    for i, a in enumerate(BA):
        ob = bpy.data.objects.new(f"Batom{i:03d}", SPHERE); BCOL.objects.link(ob)
        ob.location = a["p"]; ob["col"] = a["col"]; ob["colB"] = a["col"]; ob["stick"] = 0.0; ob["r"] = a["r"]
        b_atoms.append(ob)
    b_bonds = []
    for ob, b, pa, pb, rad in b_sticks():
        COL.objects.unlink(ob); BCOL.objects.link(ob)
        b_bonds.append((ob, b, pa, pb))

    # --- materials: the molecules' own; a shard's cut face hot on dark, bare porcelain on white --------------------------------
    def fuse_flash(nt, sh, col, pr):
        """B's atoms flash as they fuse (their `fuse` time, in seconds); every other part has none."""
        Tm = time_value(nt); fu = attr(nt, "fuse").outputs["Fac"]
        x = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", Tm, fu), 0.07)
        fl = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", x, x), -1.0))
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = CFG.get("flash_rgb", (0.35, 0.7, 1.0, 1))
        fl = math_(nt, "MULTIPLY", fl, math_(nt, "GREATER_THAN", fu, 0.01))      # (only parts that fuse: the others have none)
        nt.links.new(math_(nt, "MULTIPLY", fl, (CFG.get("flash_white", 0.0) if WHITE else CFG.get("flash", 10.0))), em.inputs["Strength"])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        return ad.outputs[0]
    hero = hero_material(extra=fuse_flash)
    assign(hero)
    for ob in b_atoms + [x[0] for x in b_bonds]:
        ob.data = ob.data  # shared meshes carry the hero material
    inner, nti, outi = new_material("inner")
    if WHITE:
        pi = nnew(nti, "ShaderNodeBsdfPrincipled", Roughness=0.55); pi.inputs["Base Color"].default_value = (0.86, 0.85, 0.83, 1)
        nti.links.new(pi.outputs[0], outi.inputs["Surface"])
    else:
        Tm = time_value(nti)
        cool = math_(nti, "EXPONENT", math_(nti, "MULTIPLY", math_(nti, "MAXIMUM", math_(nti, "SUBTRACT", Tm, TB), 0.0), -2.2))
        pi = nnew(nti, "ShaderNodeBsdfPrincipled", Roughness=0.4); pi.inputs["Base Color"].default_value = (0.05, 0.03, 0.02, 1)
        bb = nnew(nti, "ShaderNodeBlackbody"); bb.inputs["Temperature"].default_value = 1900
        em = nnew(nti, "ShaderNodeEmission"); nti.links.new(bb.outputs[0], em.inputs["Color"]); nti.links.new(math_(nti, "MULTIPLY", cool, CFG.get("hot", 12.0)), em.inputs["Strength"])
        ad = nnew(nti, "ShaderNodeAddShader"); nti.links.new(pi.outputs[0], ad.inputs[0]); nti.links.new(em.outputs[0], ad.inputs[1])
        nti.links.new(ad.outputs[0], outi.inputs["Surface"])

    lights, cyc = studio()

    # --- the shards -------------------------------------------------------------------------------------------------------------
    for ob in COL.objects:
        if ob.type == "MESH": ob.parent = None
    for ob in ATOM_OBJ: ob.location = ATOMS[ob["idx"]]["p"]
    shards = fracture(hero, inner, CFG.get("per_atom", 4), CFG.get("per_stick", 2))
    SCOL = bpy.data.collections.new("shards"); sc.collection.children.link(SCOL)
    for s in shards:
        for c in s.users_collection: c.objects.unlink(s)
        SCOL.objects.link(s)
    if WHITE and CFG.get("subject_only", True):
        for o in list(SCOL.objects) + list(BCOL.objects): COL.objects.link(o)

    # A is whole until it breaks, the shards there from then on
    def shown(ob, f_on, f_off=None):
        ob.hide_render = f_on > 1; ob.keyframe_insert("hide_render", frame=1)
        ob.hide_render = False; ob.keyframe_insert("hide_render", frame=f_on)
        if f_off is not None:
            ob.hide_render = True; ob.keyframe_insert("hide_render", frame=f_off)
    for ob in ATOM_OBJ + [b[0] for b in BOND_OBJ]:
        ob.hide_render = False; ob.keyframe_insert("hide_render", frame=1)
        ob.hide_render = True; ob.keyframe_insert("hide_render", frame=FB)

    # --- rigid bodies: the shards held (animated) until the break, then free; the paper passive; one outward kick
    if sc.rigidbody_world is None:
        with bpy.context.temp_override(scene=sc): bpy.ops.rigidbody.world_add()
    rw = sc.rigidbody_world; rw.substeps_per_frame = 20; rw.solver_iterations = 20
    rw.point_cache.frame_start, rw.point_cache.frame_end = 1, FG + 2
    def rb(ob, kind):
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob], scene=sc):
            bpy.ops.rigidbody.object_add(type=kind)
    for s in shards:
        rb(s, "ACTIVE")
        r = s.rigid_body; r.collision_shape = "CONVEX_HULL"; r.mass = 0.02; r.friction = 0.6; r.restitution = CFG.get("bounce", 0.35)
        r.collision_margin = 0.0005; r.linear_damping = 0.15; r.angular_damping = 0.1
        r.kinematic = True; r.keyframe_insert("kinematic", frame=1); r.keyframe_insert("kinematic", frame=FB - 1)
        r.kinematic = False; r.keyframe_insert("kinematic", frame=FB)
    if not WHITE or cyc is None:
        # (the dark studio has no floor to see, nor a scene as asked: the shards land on one the camera does not)
        import bmesh
        fm_ = bpy.data.meshes.new("floor"); bm_ = bmesh.new(); bmesh.ops.create_grid(bm_, x_segments=1, y_segments=1, size=4.0); bm_.to_mesh(fm_); bm_.free()
        cyc = bpy.data.objects.new("floor", fm_); sc.collection.objects.link(cyc)
        cyc.location = (0, 0, min(a["p"].z - a["r"] for a in ATOMS) - 0.004); cyc.hide_render = True
    rb(cyc, "PASSIVE"); cyc.rigid_body.collision_shape = "MESH"; cyc.rigid_body.friction = 0.7
    bpy.ops.object.effector_add(type="FORCE", location=(0, 0, min(a["p"].z for a in ATOMS) - 0.02))
    kick = bpy.context.active_object; kick.field.falloff_type = "SPHERE"; kick.field.use_max_distance = False
    for f, s_ in ((1, 0.0), (FB - 1, 0.0), (FB, CFG.get("kick", 2.5)), (FB + 3, CFG.get("kick", 2.5)), (FB + 4, 0.0)):
        kick.field.strength = s_; kick.field.keyframe_insert("strength", frame=f)
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0, 0, 0))
    tb_ = bpy.context.active_object; tb_.field.strength = CFG.get("turb", 0.6); tb_.field.size = 0.08

    t0 = time.time()
    with bpy.context.temp_override(scene=sc):
        bpy.ops.ptcache.bake_all(bake=True)
    print(f"BAKED rigid {time.time() - t0:.0f}s", flush=True)
    # the simulated flight as keys, then blended into the gathering
    sim = {}
    for f in range(FB, FG + 1):
        sc.frame_set(f)
        for s in shards: sim[(s.name, f)] = s.matrix_world.copy()
    for s in shards:
        with bpy.context.temp_override(object=s, active_object=s, selected_objects=[s], scene=sc):
            bpy.ops.rigidbody.object_remove()
    with bpy.context.temp_override(object=cyc, active_object=cyc, selected_objects=[cyc], scene=sc):
        bpy.ops.rigidbody.object_remove()

    # --- who goes where: every B atom gets the shard nearest it first, then each shard the B atom nearest it
    targets = [a["p"] for a in BA]
    rest = {s.name: sim[(s.name, FG)].to_translation() for s in shards}
    free_ = set(s.name for s in shards); goal = {}
    for j, p in sorted(enumerate(targets), key=lambda jp: jp[1].z):
        best = min(free_, key=lambda n: (rest[n] - p).length) if free_ else None
        if best: goal[best] = j; free_.discard(best)
    for n in free_: goal[n] = min(range(len(targets)), key=lambda j: (rest[n] - targets[j]).length)
    # B builds up from the paper: its atoms fuse in a wave by height, and the shards bound for each atom arrive as it fuses
    WAVE = CFG.get("wave", 0.7)
    order = sorted(range(len(targets)), key=lambda j: targets[j].z)
    fuse_f = {j: FL - int(round(WAVE * FPSM)) + int(round(WAVE * FPSM * k / max(1, len(order) - 1))) for k, j in enumerate(order)}
    FLIGHT = CFG.get("flight", 1.0)
    arrive = {}
    for s in shards:
        s.rotation_mode = "QUATERNION"
        n = s.name; j = goal[n]
        f1 = fuse_f[j] + random.randint(-1, 1)
        f0 = max(FG, f1 - int(round(FLIGHT * FPSM * random.uniform(0.85, 1.15))))
        arrive[n] = f1
        for f in range(1, FL + 8):
            if f < FB:
                m = sim[(n, FB)]
            elif f <= FG:
                m = sim[(n, f)]
            else:
                m = sim[(n, FG)]
            loc, rot, scl = m.decompose()
            if f > f0:
                e = in_out(lin(f, f0, f1), 2.6)
                # (a lift, an arc up over the others, and a spin that unwinds as it lands)
                arc = Vector((0, 0, 0.10 * math.sin(math.pi * e)))
                loc = loc.lerp(targets[j], e) + arc
                rot = rot.slerp(Quaternion(), e)
                scl = scl * (1 - 0.7 * smooth(lin(f, f1 - 2, f1 + 1)))
            s.location, s.rotation_quaternion, s.scale = loc, rot, scl
            for path in ("location", "rotation_quaternion", "scale"): s.keyframe_insert(path, frame=f)
        s.hide_render = True; s.keyframe_insert("hide_render", frame=1)
        s.hide_render = False; s.keyframe_insert("hide_render", frame=FB)
        s.hide_render = True; s.keyframe_insert("hide_render", frame=f1 + 2)
    # B's atoms fuse as their first shard lands (an overshoot and settle), its bonds drawn from each end after
    first = dict(fuse_f)
    for j, ob in enumerate(b_atoms):
        fj = first.get(j, FL)
        r = ob["r"]; ob["fuse"] = (fj - 1) / FPSM
        for f in range(1, N + 1):
            x = (f - fj) / FPSM
            k = 0.0 if x < -0.12 else spring(x + 0.12, 2.4, 6.5)
            ob.scale = (r * k,) * 3; ob.keyframe_insert("scale", frame=f)
    for ob, b, pa, pb in b_bonds:
        fa, fb_ = first.get(b["a"], FL), first.get(b["b"], FL)
        f0 = max(fa, fb_) + 2
        for f in range(1, N + 1):
            g = out_c(lin(f, f0, f0 + 0.35 * FPSM), 3)
            mid = pa.lerp(pb, 0.5)
            a_ = mid.lerp(pa, g); b_ = mid.lerp(pb, g)
            mm, q, ln = bond_frame(a_, b_)
            # (thin as well as short until it is drawn: a stick of no length but full radius would show as a disc)
            th = ob["rad"] * smooth(lin(g, 0.0, 0.15))
            ob.location = mm; ob.rotation_quaternion = q; ob.scale = (th, th, max(ln / 2, 1e-5))
            for path in ("location", "rotation_quaternion", "scale"): ob.keyframe_insert(path, frame=f)

    # --- the camera: the film's one move, refitting from A to B as it breaks and gathers
    for f in range(1, N + 1):
        change_camera(f, in_out(lin(tsec(f), TB, TL + 0.6), 2.4))
    for a in bpy.data.actions:
        for fc in all_fcurves(a):
            if fc.data_path in ("hide_render", "kinematic"): continue
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    finish(bloom=0.5, threshold=2.5, glow_size=0.55)
    render()

