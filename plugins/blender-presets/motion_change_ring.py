# SPDX-License-Identifier: GPL-3.0-or-later
#
# The film motion "change-ring" of Film looks and motions. Copyright (C) 2026 Meno contributors.
# Under the GNU General Public License, version 3 or later: see this plugin's LICENSE.
"""Ring: A's atoms gather into a ring and land as B. A's bonds let go as the atoms lift, in a wave round the molecule; the
atoms rise into one ring about it, which turns, slows, and lets them fall into B's places - each keeping its own colour and
size until it lands, where it becomes B's element with a squash and a glint. Where B has more atoms, one buds off on the
ring beside its source; where fewer, the spare ones fall into a neighbour and are gone. B's bonds are drawn as their two
atoms have landed.

Keyed with the kit (motionkit.py) and the change module (change.py), on Meno's molecules: docs/plugins/blender-presets.md,
*A motion*.
"""
import motion_kit as kit

NOMINAL = 5.0
LOOPS = "seamless"


def key_change(ctx, a, b):
    kit.change(ctx, globals(), NOMINAL, a, b)


def keyed():
    T = CFG["seconds"]
    TU = CFG.get("lift_at", 0.7)           # the first atom lifts off
    LIFT_SPREAD = CFG.get("lift_spread", 0.55)   # the lift-offs, round the molecule
    LIFT = CFG.get("lift", 0.7)            # one atom's rise into the ring
    TURN_AT, TURN_FOR = CFG.get("turn_at", 0.9), CFG.get("turn_for", 2.0)    # the ring's turn: from, and for how long
    DROP_AT = CFG.get("drop_at", 2.35)     # the first atom falls from the ring
    DROP_SPREAD = CFG.get("drop_spread", 0.5)
    DROP = CFG.get("drop", 0.62)           # one atom's fall
    WAVE = CFG.get("wave", 0.06)           # the ring's ripple, against its radius

    # --- the ring: round both molecules, at their middle height
    def flat(p): return Vector((p.x, p.y))
    both = [a["p"] for a in ATOMS] + [a["p"] for a in BA]
    RING_R = CFG.get("ring_scale", 1.12) * max(flat(p).length for p in both) + 0.4 * K
    RING_Z = sum(p.z for p in both) / len(both) + CFG.get("ring_rise", 0.25) * RADIUS
    def ang(p): return math.atan2(p.y, p.x)
    def turned(t): return CFG.get("turn", 1.5 * math.pi) * in_out(lin(t, TURN_AT, TURN_AT + TURN_FOR), 2.2)
    TURN = turned(1e9)
    def slot(theta, t, inner=False):
        """A place on the ring at angle theta (as it started), at time t: turned with the ring, rippling."""
        a_ = theta + turned(t); rr = RING_R * (0.9 if inner else 1.0)
        return Vector((rr * math.cos(a_), rr * math.sin(a_), RING_Z + WAVE * RING_R * math.sin(2 * a_ - 3.0 * t)))

    # --- who lands where: on the ring each source has its angle; an atom falls to the B atom nearest where the ring has
    # turned it, heavy to heavy and hydrogen to hydrogen, the same element preferred (Hungarian, least in all). Where B has
    # more, a bud takes the ring beside a source; where fewer, the spare ones fall into the B atom nearest them
    EPEN = (0.9 * K) ** 2
    FLY = []      # each: source A atom, its angle on the ring, the B atom it lands as (or falls into), kind
    for heavy in (True, False):
        As = [i for i, a in enumerate(ATOMS) if (a["el"] != "H") == heavy]
        Bs = [j for j, a in enumerate(BA) if (a["el"] != "H") == heavy]
        if not As or not Bs:
            continue
        srcs = [(i, ang(ATOMS[i]["p"]), "move") for i in As]
        nbud = max(0, len(Bs) - len(As))
        if nbud:
            # (buds spread round the ring: each beside its source, half a place on)
            step = 2 * math.pi / (len(As) + nbud)
            order = sorted(As, key=lambda i: ang(ATOMS[i]["p"]))
            for k in range(nbud):
                i = order[(k * len(order)) // nbud]
                srcs.append((i, ang(ATOMS[i]["p"]) + 0.5 * step * (1 if k % 2 else -1), "bud"))
        # (even places on the ring, in the order the sources stand round the molecule, turned as little as may be from where
        # they stand: a necklace, not clumps)
        srcs.sort(key=lambda s_: s_[1] % (2 * math.pi))
        m_ = len(srcs); even = [2 * math.pi * k / m_ for k in range(m_)]
        off = math.atan2(sum(math.sin(s_[1] - e) for s_, e in zip(srcs, even)), sum(math.cos(s_[1] - e) for s_, e in zip(srcs, even)))
        srcs = [(s_[0], e + off, s_[2]) for s_, e in zip(srcs, even)]
        n = max(len(srcs), len(Bs))
        def at_drop(s): return slot(s[1], DROP_AT, inner=not heavy)
        cost = [[((at_drop(srcs[r]) - BA[Bs[c]]["p"]).length_squared + (0 if ATOMS[srcs[r][0]]["el"] == BA[Bs[c]]["el"] else EPEN))
                 if r < len(srcs) and c < len(Bs) else 0.0 for c in range(n)] for r in range(n)]
        got = hungarian(cost)
        for r, c in enumerate(got):
            if r >= len(srcs): continue
            i, th, kind = srcs[r]
            if c < len(Bs):
                FLY.append({"i": i, "th": th, "j": Bs[c], "kind": kind, "inner": not heavy})
            else:
                j = min(Bs, key=lambda j: (at_drop(srcs[r]) - BA[j]["p"]).length)
                FLY.append({"i": i, "th": th, "j": j, "kind": "merge", "inner": not heavy})
    print("RING " + ", ".join(f"{sum(F['kind'] == k for F in FLY)} {k}" for k in ("move", "bud", "merge")) + f", ring {RING_R / K:.1f} A", flush=True)

    # --- when: lift-offs in a wave round the molecule, by their places on the ring; falls in a wave round it too
    a0 = min(F["th"] for F in FLY)
    for F in FLY:
        u_ = ((F["th"] - a0) % (2 * math.pi)) / (2 * math.pi)
        F["t0"] = TU + LIFT_SPREAD * u_ + 0.05 * random.random() + (0.12 if F["kind"] == "bud" else 0.0)
        F["t1"] = F["t0"] + LIFT * random.uniform(0.9, 1.1)
    # (the falls sweep round the ring as the lift-offs did, from the place nearest the camera's side on)
    for F in FLY:
        u_ = ((F["th"] + turned(DROP_AT) - a0) % (2 * math.pi)) / (2 * math.pi)
        F["t2"] = DROP_AT + DROP_SPREAD * u_ + 0.03 * random.random()
        F["t3"] = F["t2"] + DROP * random.uniform(0.92, 1.08)
    landed = {}
    for F in FLY:
        if F["kind"] != "merge": landed[F["j"]] = min(landed.get(F["j"], 1e9), F["t3"])

    def place(F, t):
        """Where a flyer is at time t: from its A place up into its ring place, round with the ring, down to its B place."""
        S0 = ATOMS[F["i"]]["p"]; P = BA[F["j"]]["p"]
        if t < F["t2"]:
            e = in_out(lin(t, F["t0"], F["t1"]), 2.6)
            up = Vector((0, 0, 0.35 * RADIUS * math.sin(math.pi * e)))          # (a lift, up over the others)
            return S0.lerp(slot(F["th"], t, F["inner"]), e) + up
        e = in_out(lin(t, F["t2"], F["t3"]), 2.2)
        return slot(F["th"], t, F["inner"]).lerp(P, e) + Vector((0, 0, 0.18 * RADIUS * math.sin(math.pi * e)))

    # --- the flyers: A's own atoms, and a sphere for each bud
    for ob in COL.objects:
        if ob.type == "MESH": ob.parent = None
    for ob in ATOM_OBJ: ob.location = ATOMS[ob["idx"]]["p"]
    for F in FLY:
        if F["kind"] == "bud":
            F["ob"] = bpy.data.objects.new(f"bud{F['j']:03d}", SPHERE); COL.objects.link(F["ob"])
            F["r0"] = 0.0
        else:
            F["ob"] = ATOM_OBJ[F["i"]]; F["r0"] = ATOMS[F["i"]]["r"]
        F["r1"] = BA[F["j"]]["r"]
        ob = F["ob"]
        ob["col"] = ATOMS[F["i"]]["col"]; ob["col2"] = BA[F["j"]]["col"]; ob["stick"] = 0.0
        ob["land"] = F["t3"] if F["kind"] != "merge" else -1.0

    # --- materials: a flyer keeps its own colour until it lands, and turns to B's element in the glint of landing
    def landing(nt, sh, col, pr):
        Tm = time_value(nt); ld = attr(nt, "land").outputs["Fac"]
        x = math_(nt, "DIVIDE", math_(nt, "SUBTRACT", Tm, ld), CFG.get("glint_for", 0.045))
        fl = math_(nt, "EXPONENT", math_(nt, "MULTIPLY", math_(nt, "MULTIPLY", x, x), -1.0))
        fl = math_(nt, "MULTIPLY", fl, math_(nt, "GREATER_THAN", ld, 0.0))
        em = nnew(nt, "ShaderNodeEmission"); em.inputs["Color"].default_value = CFG.get("flash_rgb", (0.62, 0.8, 1.0, 1))
        nt.links.new(math_(nt, "MULTIPLY", fl, CFG.get("flash_white", 0.5) if WHITE else CFG.get("flash", 8.0)), em.inputs["Strength"])
        ad = nnew(nt, "ShaderNodeAddShader"); nt.links.new(sh, ad.inputs[0]); nt.links.new(em.outputs[0], ad.inputs[1])
        return ad.outputs[0]
    hero = hero_material()
    assign(hero)
    fm, ntf, outf = new_material("flyer")
    ca, cb, mxa = attr(ntf, "col"), attr(ntf, "col2"), attr(ntf, "mx")
    mix = nnew(ntf, "ShaderNodeMix", data_type="RGBA")
    ntf.links.new(mxa.outputs["Fac"], mix.inputs[0]); ntf.links.new(ca.outputs["Color"], mix.inputs[6]); ntf.links.new(cb.outputs["Color"], mix.inputs[7])
    prf = hero_bsdf(ntf, mix.outputs[2])
    ntf.links.new(landing(ntf, prf.outputs[0], mix.outputs[2], prf), outf.inputs["Surface"])
    assign(fm, [F["ob"] for F in FLY])

    for F in FLY:
        ob = F["ob"]
        for f in range(1, N + 1):
            t = tsec(f)
            ob.location = place(F, t)
            ob["mx"] = smooth(lin(t, F["t3"] - 0.05, F["t3"] + 0.03))
            if F["kind"] == "bud":
                r = F["r0"] + (F["r1"] - F["r0"]) * out_c(lin(t, F["t0"], F["t1"]), 2.0)
                r = min(r, ATOMS[F["i"]]["r"]) if t < F["t3"] else r
                r += (F["r1"] - r) * smooth(lin(t, F["t3"] - 0.08, F["t3"]))
            elif F["kind"] == "merge":
                r = F["r0"] * (1 - smooth(lin(t, F["t3"] - 0.12, F["t3"])))
            else:
                r = F["r0"] + (F["r1"] - F["r0"]) * smooth(lin(t, F["t3"] - 0.08, F["t3"]))
            dt = t - F["t3"]
            if F["kind"] != "merge" and dt > 0:
                r *= 1 + 0.17 * math.exp(-6.5 * dt) * math.sin(2 * math.pi * 2.8 * dt + math.pi)
            ob.scale = (max(r, 1e-6),) * 3
            ob.keyframe_insert("location", frame=f); ob.keyframe_insert("scale", frame=f); ob.keyframe_insert('["mx"]', frame=f)
        if F["kind"] == "merge":
            ob.hide_render = False; ob.keyframe_insert("hide_render", frame=1)
            ob.hide_render = True; ob.keyframe_insert("hide_render", frame=int(F["t3"] * FPSM) + 2)
        elif F["kind"] == "bud":
            ob.hide_render = True; ob.keyframe_insert("hide_render", frame=1)
            ob.hide_render = False; ob.keyframe_insert("hide_render", frame=max(1, int(F["t0"] * FPSM)))

    # --- A's bonds let go: each follows its atoms as the first of them lifts, and shrinks into its middle
    own = {F["i"]: F for F in FLY if F["kind"] != "bud"}
    for ob, bi, off in BOND_OBJ:
        b = BONDS[bi]; a_, c_ = b["a"], b["b"]
        tg = min(own[a_]["t0"], own[c_]["t0"]) - 0.02
        for f in range(1, N + 1):
            t = tsec(f)
            g = 1 - out_c(lin(t, tg, tg + 0.24), 2.5)
            # (its own stretch of the bond, beside it as Meno drew it)
            qa, qb = place(own[a_], t), place(own[c_], t)
            pa, pb = qa + (qb - qa) * ob["t0"] + off, qa + (qb - qa) * ob["t1"] + off
            m_ = pa.lerp(pb, 0.5)
            mm, q, ln = bond_frame(m_.lerp(pa, g), m_.lerp(pb, g))
            th = ob["rad"] * smooth(lin(g, 0.0, 0.25))
            ob.location = mm; ob.rotation_quaternion = q; ob.scale = (max(th, 1e-6), max(th, 1e-6), max(ln / 2 / ob["half"], 1e-5))
            for path in ("location", "rotation_quaternion", "scale"): ob.keyframe_insert(path, frame=f)

    # --- B's bonds, drawn out from their middles once both their atoms have landed
    for ob, b, pa, pb, rad in b_sticks():
        t0 = max(landed.get(b["a"], T), landed.get(b["b"], T)) + 0.03
        for f in range(1, N + 1):
            draw_stick(ob, pa, pb, rad, out_c(lin(tsec(f), t0, t0 + 0.3), 3), f)

    # --- the camera: the film's one move, drawn back to hold the ring while it is out
    lights, cyc = studio()
    hw = CAM.sensor_width / 2 / CAM.lens
    D_R = (RING_R + 0.4 * K) / (hw * CFG.get("ring_margin", 0.66))
    TL = max(landed.values())
    for f in range(1, N + 1):
        t = tsec(f)
        wide = in_out(lin(t, TU, TU + LIFT + LIFT_SPREAD), 2.4) * (1 - in_out(lin(t, DROP_AT, TL), 2.4))
        change_camera(f, in_out(lin(t, DROP_AT, TL + 0.4), 2.4), far=D_R, far_z=RING_Z, out=wide)
    for a in bpy.data.actions:
        for fc in all_fcurves(a):
            if fc.data_path == "hide_render": continue
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    print(f"RING lands by {TL:.2f}s of {T:.2f}s", flush=True)
    finish(bloom=0.5, threshold=2.5, glow_size=0.55)
    render()
