# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Meno contributors. Under the GPL, in this plugin's LICENSE.
"""How a motion's module is keyed with the kit (motionkit.py): docs/plugins/blender-presets.md, *A motion*."""
import os


def settings(ctx, nominal, **more):
    """The motion's settings, from the request: its length and the film's, its frames, Once or Loop, the background."""
    r = ctx.request
    bg = "white" if r.get("background", "transparent") in ("white", "transparent") else "dark"
    cfg = {"ctx": ctx, "seconds": r["seconds"], "nominal": None if r.get("mode") == "loop" else nominal,
           "frames": r["frames"], "fps": r["fps"], "mode": r.get("mode", "once"), "bg": bg,
           "pulses": max(1, int(round(r["seconds"] / 3.0)))}
    cfg.update(more)
    return cfg


def change(ctx, motion, nominal, a, b):
    """A change's piece - molecule `a` becoming `b` - keyed with the kit and the change module: each molecule on the same
    floor, the piece's place in the whole film (`t_offset`, `t_total`) and whether the film comes back to its first."""
    r = ctx.request
    run(ctx, motion, nominal, shared=("change.py",), molecule=a["index"], next=b["index"], stand=True,
        t_offset=r.get("start", 0.0), t_total=r.get("whole", r["seconds"]), loop=r.get("loop", False),
        piece=r.get("piece", 1), pieces=r.get("pieces", 1))


def run(ctx, motion, nominal, shared=(), **more):
    """The kit executed for this motion - and the files it shares with others (`shared`: the change module, say) after it -
    their names handed to the motion's module (`motion`, its globals), and the module's `keyed()` called."""
    space = {"CFG": settings(ctx, nominal, **more), "__name__": "motionkit"}
    home = os.path.dirname(os.path.abspath(motion["__file__"]))
    for name in ("motionkit.py", *shared):
        with open(os.path.join(home, name), encoding="utf-8") as f:
            exec(compile(f.read(), name, "exec"), space)
    motion.update({k: v for k, v in space.items() if not k.startswith("__")})
    motion["keyed"]()
