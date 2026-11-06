# SPDX-License-Identifier: GPL-3.0-or-later
#
# The Blender plugin's render script: a picture or a film of the scene
# Meno wrote, rendered by Blender.
#
# Copyright (C) 2026 Meno contributors
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. It is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License, in this folder's LICENSE, for more details.
"""Renders the scene Meno wrote - glTF 2.0 - as a picture or a film.

Run by Blender, in its own Python, as the job the plugin's worker prepared
(worker.py): Blender in the background, with its factory settings, this
script, and after "--" the job's settings, a JSON file in the job's folder:

    blender -b --factory-startup -noaudio --offline-mode --python-exit-code 1 -P render.py -- render.json

It is under the GPL, as a script using Blender's Python API is: a program
of its own, apart from Meno, which it speaks to only through the files in
the job's folder and its arguments (docs/PLUGINS.md, *Licences*).

What it does, by Blender's own manual and API reference (docs.blender.org,
Blender 5.2): it empties Blender's scene, takes in the glTF scene with
Blender's glTF importer (its camera, its key light, its materials - their
extensions too - and, for a film, its animation), lights the world evenly
as Meno's light all round is, sets the render engine, its samples and
noise removal, the colour transform and the background, and renders the
picture, or each frame of the film into a video - saying how far it is, a
line at a time: "progress <done>/<all>" - as hundredths of the whole, the samples of a
picture, or of each frame of a film, moving it, from Blender's own render statistics.

The settings (all checked by the worker that wrote them):

    scene       the glTF file's name, in the job's folder
    out         the name to write: a picture's (PNG) or a film's (MP4)
    film        a film, not a picture
    width, height, frames, fps
    engine      "cycles" or "eevee"
    samples, denoise
    device      "auto" - the faster of the graphics processor and the processor, found once by a timing and kept -, "gpu" or "cpu"
    background  "transparent", "white", "black" or "grey"
    colours     "neutral", "agx" or "standard"
    key, ambient   the key light's and the light all round's strengths, as factors of Meno's
    softness    how soft the shadows are: the key light's width, in degrees
    quality     a film's: "high", "medium" or "lossless"
    presets     where a preset plugin's look, studio or motion is chosen (docs/blender-presets.md): for each such
                plugin, which (`from`), the chosen ones' files (`look`, `studio`, `motion`) and its own options (`values`); its files
                are in the job's folder, under presets/<plugin id>/, put there by Meno
    mode        a film's: "once" or "loop"
    shading     "lit", as a renderer lights it, or "toon": each material a Toon BSDF in the colour it had,
                lit in one band - `band` its size - which Cycles draws as a band, without noise removal
    outlines    "none", "thin" (Freestyle's lines, one pixel), "outer" (the picture's silhouette grown
                by `outline` pixels) or "mixed" (both); `outline_colour`: "black", "white" or "grey"
"""

import glob
import json
import math
import os
import re
import sys
import threading
import time

import bpy

sys.path.insert(0, os.getcwd())
import presets  # noqa: E402 - beside this script, in the job's folder

# What a factor of 1 of Meno's lights is, in Blender's units: Meno's key
# light, a sun of this many watts a square metre; its light all round, a
# world this bright.
SUN_STRENGTH = 3.0
WORLD_STRENGTH = 0.6

OUTLINES = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0), "grey": (0.5, 0.5, 0.5)}
BACKGROUNDS = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "grey": (0.5, 0.5, 0.5)}
TRANSFORMS = {"neutral": "Khronos PBR Neutral", "agx": "AgX", "standard": "Standard"}
QUALITIES = {"high": "HIGH", "medium": "MEDIUM", "lossless": "PERC_LOSSLESS"}
GPUS = ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI")


def say(line):
    print(line, flush=True)


def settings_of():
    argv = sys.argv
    if "--" not in argv or argv.index("--") + 1 >= len(argv):
        raise SystemExit("render.py: no settings given")
    with open(argv[argv.index("--") + 1], encoding="utf-8") as f:
        return json.load(f)


def empty_scene():
    """Blender's factory scene, emptied: nothing of its own in what is rendered."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def take_in(scene, path, fps):
    """The glTF scene, taken in by Blender's importer - at the film's rate, so that its keys' seconds fall on its frames."""
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=path)


def camera_of(scene, gltf):
    """The camera Meno placed, orthographic, seeing what Meno framed: as wide as its frame (twice glTF's xmag), the picture its shape."""
    cams = [o for o in scene.objects if o.type == "CAMERA"]
    if not cams:
        raise SystemExit("render.py: the scene has no camera")
    cam = cams[0]
    ortho = gltf["cameras"][0]["orthographic"]
    cam.data.type = "ORTHO"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.ortho_scale = 2 * ortho["xmag"]
    cam.data.clip_start = ortho["znear"]
    cam.data.clip_end = ortho["zfar"]
    scene.camera = cam


def light(scene, s, ambient):
    """The key light as Meno's, as strong as asked and as soft; the world lit evenly, as Meno's light all round. What is behind is left transparent: the background is put under it afterwards (`assemble`), as it is asked for."""
    for o in scene.objects:
        if o.type == "LIGHT" and o.data.type == "SUN":
            o.data.energy = SUN_STRENGTH * s["key"]
            o.data.angle = math.radians(s["softness"])
    world = bpy.data.worlds.new("Meno")
    scene.world = world
    nodes = world.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    lit = nodes.new("ShaderNodeBackground")
    lit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    lit.inputs["Strength"].default_value = WORLD_STRENGTH * ambient * s["ambient"]
    world.node_tree.links.new(lit.outputs["Background"], out.inputs["Surface"])
    scene.render.film_transparent = True


def toon(scene, s):
    """Toon shading: each material's surface a Toon BSDF in the colour the material had - its colour attribute's, where it took one - lit in one band, `band` its size; a surface that let light through still does, by its alpha (Blender's manual, Shader Nodes > Toon BSDF)."""
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nt = mat.node_tree
        principled = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if principled is None or out is None:
            continue
        toon_node = nt.nodes.new("ShaderNodeBsdfToon")
        toon_node.component = "DIFFUSE"
        toon_node.inputs["Size"].default_value = s["band"]
        toon_node.inputs["Smooth"].default_value = 0.0
        colour = principled.inputs["Base Color"]
        if colour.links:
            nt.links.new(colour.links[0].from_socket, toon_node.inputs["Color"])
        else:
            toon_node.inputs["Color"].default_value = colour.default_value
        surface = toon_node.outputs["BSDF"]
        alpha = principled.inputs["Alpha"]
        if alpha.links or alpha.default_value < 1.0:
            clear = nt.nodes.new("ShaderNodeBsdfTransparent")
            mix = nt.nodes.new("ShaderNodeMixShader")
            if alpha.links:
                nt.links.new(alpha.links[0].from_socket, mix.inputs["Fac"])
            else:
                mix.inputs["Fac"].default_value = alpha.default_value
            nt.links.new(clear.outputs["BSDF"], mix.inputs[1])
            nt.links.new(toon_node.outputs["BSDF"], mix.inputs[2])
            surface = mix.outputs["Shader"]
        nt.links.new(surface, out.inputs["Surface"])


def lines(scene, s):
    """Outlines. Freestyle (Blender's manual, Render > Freestyle) draws the shapes' silhouettes, borders and creases: every line at one pixel - "thin" - or, where the outer edge is thick, the inner lines at one pixel. The outer edge - "outer", "mixed" - is the picture's own silhouette grown by the width, in the outline's colour, under the picture: one edge round the molecule, however its shapes overlap, which Freestyle's own external contour is not (it is each shape's)."""
    colour = OUTLINES[s["outline_colour"]]
    if s["outlines"] in ("thin", "mixed"):
        scene.render.use_freestyle = True
        scene.render.line_thickness_mode = "ABSOLUTE"
        scene.render.line_thickness = 1.0
        layer = scene.view_layers[0]
        layer.use_freestyle = True
        sets = layer.freestyle_settings.linesets
        lineset = sets[0] if len(sets) else sets.new("Meno outline")
        lineset.linestyle = bpy.data.linestyles.new("Meno outline")
        lineset.select_silhouette = True
        lineset.select_border = True
        lineset.select_crease = True
        lineset.linestyle.color = colour
        lineset.linestyle.thickness = 1.0
    if s["outlines"] in ("outer", "mixed"):
        group = bpy.data.node_groups.new("Meno outline", "CompositorNodeTree")
        group.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        nodes, links = group.nodes, group.links
        seen = nodes.new("CompositorNodeRLayers")
        grown = nodes.new("CompositorNodeDilateErode")
        grown.inputs["Size"].default_value = int(round(s["outline"]))
        shape = nodes.new("CompositorNodeSetAlpha")
        shape.inputs["Image"].default_value = (colour[0], colour[1], colour[2], 1.0)
        over = nodes.new("CompositorNodeAlphaOver")
        out = nodes.new("NodeGroupOutput")
        links.new(seen.outputs["Alpha"], grown.inputs["Mask"])
        links.new(grown.outputs["Mask"], shape.inputs["Alpha"])
        links.new(shape.outputs["Image"], over.inputs["Background"])
        links.new(seen.outputs["Image"], over.inputs["Foreground"])
        links.new(over.outputs["Image"], out.inputs["Image"])
        scene.compositing_node_group = group
        scene.render.use_compositing = True


def gpu_ready():
    """Whether Blender has a graphics processor to render on, as its Cycles preferences say: the first kind it has, all of its devices on."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in GPUS:
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        found = [d for d in prefs.devices if d.type == kind]
        if found:
            for d in prefs.devices:
                d.use = d.type == kind
            return True
    return False


def data_folder(s):
    """Where what is learned of this computer is kept (docs/RENDER.md): `device-probe.json` in the plugin's data folder, which the worker passes (`probe_dir`); else, found from Blender's own path, the folder above its `files`. A settings' `probe_file` names the file itself, as a test does."""
    if s.get("probe_file"):
        return s["probe_file"]
    # (the plugin's data folder, as its worker passes it: whatever Blender's own path looks like)
    if s.get("probe_dir"):
        return os.path.join(s["probe_dir"], "device-probe.json")
    path = os.path.abspath(bpy.app.binary_path)
    while True:
        parent = os.path.dirname(path)
        if parent == path:
            return None
        if os.path.basename(path) == "files":
            return os.path.join(parent, "device-probe.json")
        path = parent


def probe_key():
    """What a timing is true of: the graphics processors Blender sees, by name, and its version."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    names = sorted(d.name for d in prefs.devices if d.use and d.type != "CPU")
    return f"{bpy.app.version_string}|{prefs.compute_device_type}|{'+'.join(names)}"


def timed(scene, device):
    """One render of the scene as it is set, not written: how long it took, in seconds."""
    scene.cycles.device = device
    started = time.monotonic()
    bpy.ops.render.render()
    return time.monotonic() - started


def faster_is_gpu(scene, s):
    """Whether the graphics processor is the faster here: found by a timing once - 16 samples at an eighth of the picture's width and height, the processor and the graphics processor, the latter warmed up first, its kernels prepared - and kept, by the processors' names and Blender's version, in the plugin's data folder."""
    key = probe_key()
    path = data_folder(s)
    known = {}
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                known = json.load(f)
        except (OSError, ValueError):
            known = {}
    if isinstance(known.get(key), dict) and known[key].get("device") in ("gpu", "cpu"):
        say(f"Rendering on the {'graphics processor' if known[key]['device'] == 'gpu' else 'processor'}, as timed before")
        return known[key]["device"] == "gpu"
    say("Checking which is faster...")
    r = scene.render
    was = (r.resolution_x, r.resolution_y, r.resolution_percentage, scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.device, r.use_compositing, r.use_freestyle)
    try:
        r.resolution_x, r.resolution_y, r.resolution_percentage = max(16, s["width"] // 8), max(16, s["height"] // 8), 100
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        r.use_compositing = False
        r.use_freestyle = False
        # (the graphics processor's first render prepares its kernels: that is not what is being timed)
        scene.cycles.samples = 1
        timed(scene, "GPU")
        scene.cycles.samples = 16
        cpu = timed(scene, "CPU")
        gpu = timed(scene, "GPU")
    finally:
        r.resolution_x, r.resolution_y, r.resolution_percentage, scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.device, r.use_compositing, r.use_freestyle = was
    chosen = "gpu" if gpu < cpu else "cpu"
    say(f"Processor {cpu:.1f} s, graphics processor {gpu:.1f} s: the {'graphics processor' if chosen == 'gpu' else 'processor'} is faster")
    if path:
        # (the times kept to a tenth of a millisecond: a small picture's can take less than a hundredth of a second)
        known[key] = {"device": chosen, "cpu": round(cpu, 4), "gpu": round(gpu, 4)}
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(known, f, indent=1)
        except OSError:
            pass
    return chosen == "gpu"


_DECIDED = []


def use_gpu(scene, s):
    """Whether this render is on the graphics processor, and the log saying which: asked for it, and Blender has one it can use - or, where it is left to decide, the one that is faster here. Decided once for the render."""
    if not _DECIDED:
        _DECIDED.append(decide_gpu(scene, s))
    return _DECIDED[0]


# (what the log says where the graphics processor was asked for, or left to choose, and Blender has none it can use - or the
# processor timed faster: the worker reads these lines to say so on the card, worker.py's FELL_BACK and FASTER)
FELL_BACK = "Rendering on the processor (no graphics processor Blender can use)"
FASTER = "Rendering on the processor (the faster one here)"


def decide_gpu(scene, s):
    device = s["device"]
    if device == "cpu":
        say("Rendering on the processor")
        return False
    if not gpu_ready():
        say(FELL_BACK)
        return False
    if device == "auto":
        scene.render.engine = "CYCLES"
        if not faster_is_gpu(scene, s):
            say(FASTER)
            return False
    say("Rendering on the graphics processor")
    return True


def engine(scene, s):
    # (the cartoon look's band is the Toon BSDF's: Cycles draws it as one, and its noise is not removed, which would soften the band's edge)
    if s["shading"] == "toon":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = s["samples"]
        scene.cycles.use_denoising = False
        scene.cycles.device = "GPU" if use_gpu(scene, s) else "CPU"
        return
    if s["engine"] == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = s["samples"]
        return
    scene.render.engine = "CYCLES"
    scene.cycles.samples = s["samples"]
    scene.cycles.use_denoising = s["denoise"]
    # (said, in the log, which it is: a graphics processor asked for and none found falls back to the processor)
    scene.cycles.device = "GPU" if use_gpu(scene, s) else "CPU"


def output(scene, s):
    """What is rendered: its size, its colours as asked, each frame a PNG with its transparency."""
    scene.render.resolution_x = s["width"]
    scene.render.resolution_y = s["height"]
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = TRANSFORMS[s["colours"]]
    scene.view_settings.look = "None"
    image = scene.render.image_settings
    image.media_type = "IMAGE"
    image.file_format = "PNG"
    image.color_mode = "RGBA"
    image.color_depth = "8"


def frames_of(scene, count):
    """The film's frames: from where its keys begin, as many as Meno made."""
    first = None
    for o in scene.objects:
        ad = o.animation_data
        if ad and ad.action:
            start = ad.action.frame_range[0]
            first = start if first is None else min(first, start)
    first = int(round(first if first is not None else scene.frame_start))
    scene.frame_start = first
    scene.frame_end = first + count - 1


# How far a frame's samples are, as Blender says it in its render statistics
# (the string `bpy.app.handlers.render_stats` is given): Cycles - "... | Sample 12/128" -
# and EEVEE - "Rendering 12 / 128 samples"
# Blender's render statistics carry the time taken and its own estimate of what remains: "Time: 00:04.50 | Remaining: 00:35.42 | Sample 1/128"
CLOCK = re.compile(r"\b(Time|Remaining):\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
SAMPLES = re.compile(r"(?:\bSample\s+(\d+)\s*/\s*(\d+)|Rendering\s+(\d+)\s*/\s*(\d+)\s+samples)")


class Progress:
    """The picture's or the film's progress, said as "progress <done>/<all>": each frame as many parts as it has samples, those it has done and the frames before - never back, and not a line for each sample, but for each hundredth of the whole (a bar moves as it is told, and is not jumped)."""

    PARTS = 100

    def __init__(self, frames):
        self.frames = frames
        self.frame = 0
        self.said = -1
        # (Blender's estimate of the frame, as of the moment it was told - the bar goes on by the clock from there, up to nearly the end)
        self.anchor = None
        # (when this frame began, by the script's own clock: Blender's statistics mid-render carry "Remaining" but not "Time")
        self.began = None
        self.lock = threading.RLock()
        self.ended = threading.Event()
        self.ticker = None

    def tick(self):
        """Between the statistics Blender reports - seconds apart on a graphics processor, a sample each - the bar goes on by the clock, along its own estimate; it never says the end, and never goes back."""
        while not self.ended.wait(0.25):
            with self.lock:
                a = self.anchor
                if a is None or a[3] != self.frame:
                    continue
                wall, taken, whole_time, _ = a
                share = min(0.97, (taken + (time.monotonic() - wall)) / whole_time)
                whole = (self.frame + share) / self.frames
                self.say_at(int(whole * self.PARTS * 0.999), self.PARTS)

    def end(self):
        self.ended.set()

    def say_at(self, done, total):
        # (a hundredth of the whole at least, or the last; the bar's own clock and Blender's statistics both say, one line at a time)
        with self.lock:
            mark = (done * self.PARTS) // total
            if mark > self.said or done >= total:
                if done >= total and self.said >= self.PARTS:
                    return
                self.said = max(self.said, mark)
                say(f"progress {self.said}/{self.PARTS}")

    @staticmethod
    def clocks(text):
        """The time taken and the time remaining that a statistics line says, in seconds: each, or None."""
        found = {}
        for m in CLOCK.finditer(text or ""):
            found[m.group(1)] = (int(m.group(2) or 0) * 60 + int(m.group(3))) * 60 + float(m.group(4))
        return found.get("Time"), found.get("Remaining")

    def begin_frame(self, *_):
        with self.lock:
            self.began = time.monotonic()

    def stats(self, text, *_):
        taken, left = self.clocks(text)
        if taken is None and left is not None:
            # (only "Remaining" said: the time taken is measured from the frame's beginning here - or from the first line, if that was missed)
            with self.lock:
                if self.began is None:
                    self.began = time.monotonic()
                taken = time.monotonic() - self.began
        if taken is not None and left is not None and taken + left > 0:
            # (Blender's own estimate: the share of this frame's time that has passed - moves the bar between the samples it reports,
            # where one sample takes seconds; the sample's own share is the floor, and the bar never goes back)
            whole = (self.frame + min(1.0, taken / (taken + left))) / self.frames
            with self.lock:
                self.say_at(int(whole * self.PARTS * 0.999), self.PARTS)
                self.anchor = (time.monotonic(), taken, taken + left, self.frame)
            if self.ticker is None:
                self.ticker = threading.Thread(target=self.tick, daemon=True)
                self.ticker.start()
        m = SAMPLES.search(text or "")
        if not m:
            return
        at, of = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(4)))
        if of <= 0:
            return
        # (the frame's samples as a share of the film; the last sample's frame is told done when it is written)
        whole = (self.frame + min(at, of) / of) / self.frames
        self.say_at(int(whole * self.PARTS * 0.999), self.PARTS)

    def frame_done(self):
        with self.lock:
            self.frame += 1
            self.anchor = None
            self.began = None
            self.say_at(int((self.frame / self.frames) * self.PARTS), self.PARTS)


def render(scene, s, folder):
    """Each frame rendered, saying how far it is: the frames' files, in order. A picture's samples move the bar; a film's, its frames' samples and its frames."""
    progress = Progress(1 if not s["film"] else s["frames"])
    bpy.app.handlers.render_pre.append(progress.begin_frame)
    bpy.app.handlers.render_stats.append(progress.stats)
    bpy.app.handlers.render_post.append(lambda *_: progress.frame_done())
    say("progress 0/100")
    if not s["film"]:
        scene.render.filepath = os.path.join(folder, "frame.png")
        bpy.ops.render.render(write_still=True)
        progress.end()
        say("progress 100/100")
        return [os.path.join(folder, "frame.png")]
    frames_of(scene, s["frames"])
    scene.render.filepath = os.path.join(folder, "frames", "####")
    bpy.ops.render.render(animation=True)
    progress.end()
    made = sorted(glob.glob(os.path.join(folder, "frames", "*.png")))
    if len(made) != s["frames"]:
        raise SystemExit(f"render.py: Blender rendered {len(made)} of {s['frames']} frames")
    return made


def strip_runs(frames):
    """The frames a film plays, as the sequencer's image strips take them: a strip reads all its files from one folder - its
    first file's - so each run of frames from one folder is a strip of its own, (folder, the files' names, the film frame it
    starts at). A change's pieces each have their own folder, their files the same names (f_0001.png...), and a cycle's
    Loop plays its frames forward, then back: never one strip for them all, which would play the first folder's
    throughout, and the names kept in the order they are played."""
    runs = []
    for k, f in enumerate(frames):
        d, name = os.path.split(f)
        if runs and runs[-1][0] == d:
            runs[-1][1].append(name)
        else:
            runs.append((d, [name], k + 1))
    return runs


def even_up(n):
    """`n`, or the even number just above it."""
    return n + n % 2


def assemble(frames, s, folder):
    """
    The frames rendered, put over the background asked for - in the
    colours they were rendered in, so that white is white - and written:
    a picture, a PNG; a film, an MP4 (H.264), a video having no
    transparency (a film asked for on a transparent background is put on
    white). Done by Blender's sequencer, in a scene of its own.
    """
    out = os.path.join(folder, s["out"])
    background = s["background"]
    if s["film"] and background == "transparent":
        background = "white"
    if background == "transparent":
        os.replace(frames[0], out)
        return
    made = bpy.data.scenes.new("Assembled")
    # (a film's video, H.264 in 4:2:0, keeps colour for each 2x2 pixels, and its encoder refuses an odd side: where a side
    # Meno asked for is odd, the video is a pixel wider or taller, the frames at their own size in its corner and the
    # background asked for in that pixel. A picture keeps its own size)
    made.render.resolution_x = even_up(s["width"]) if s["film"] else s["width"]
    made.render.resolution_y = even_up(s["height"]) if s["film"] else s["height"]
    made.render.resolution_percentage = 100
    made.render.fps = s["fps"]
    made.render.fps_base = 1.0
    made.frame_start = 1
    made.frame_end = len(frames)
    # (the frames' colours as they are: no transform over them again)
    made.view_settings.view_transform = "Standard"
    made.view_settings.look = "None"
    made.render.use_sequencer = True
    made.render.use_compositing = False
    strips = made.sequence_editor_create().strips
    under = strips.new_effect(name="Background", type="COLOR", channel=1, frame_start=1, length=len(frames))
    under.color = BACKGROUNDS[background]
    for k, (d, names, start) in enumerate(strip_runs(frames)):
        over = strips.new_image(name=f"Rendered {k + 1}", filepath=os.path.join(d, names[0]), channel=2, frame_start=start)
        for name in names[1:]:
            over.elements.append(name)
        over.blend_type = "ALPHA_OVER"
        # (at their own size, never scaled to a padded video's, nor moved: Blender sets a strip a pixel smaller than its canvas
        # in the canvas's middle on its pixels - its corner - each pixel the frame's own, none drawn between two)
        over.transform.filter = "NEAREST"
        over.transform.scale_x = over.transform.scale_y = 1.0
    image = made.render.image_settings
    if s["film"]:
        image.media_type = "VIDEO"
        image.file_format = "FFMPEG"
        image.color_mode = "RGB"
        ff = made.render.ffmpeg
        ff.format = "MPEG4"
        ff.codec = "H264"
        ff.constant_rate_factor = QUALITIES[s["quality"]]
        ff.ffmpeg_preset = "GOOD"
        ff.audio_codec = "NONE"
        stem = os.path.join(folder, "film-")
        made.render.filepath = stem
        say("Making the film")
        bpy.ops.render.render(animation=True, scene=made.name)
        # (Blender names a video after its frames: it is given the name Meno asked for)
        videos = sorted(glob.glob(stem + "*.mp4"))
        if not videos:
            raise SystemExit("render.py: Blender wrote no film")
        os.replace(videos[-1], out)
    else:
        image.media_type = "IMAGE"
        image.file_format = "PNG"
        image.color_mode = "RGB"
        image.color_depth = "8"
        made.render.filepath = out
        bpy.ops.render.render(write_still=True, scene=made.name)


# A look, a studio or a motion of a preset plugin (docs/blender-presets.md): its module, from the files Meno put in the
# job, run against `ctx` - the request and Meno's molecules as this scene has them - and nothing else.

def built(s, folder, scene_path, gltf, meno):
    """Meno's scene taken in, lit, its engine set, its look's parts made, a preset's look and studio added where chosen, and
    its output set: the scene, and the `ctx` a preset is run against."""
    scene = empty_scene()
    take_in(scene, scene_path, s["fps"])
    scene = bpy.context.scene
    camera_of(scene, gltf)
    light(scene, s, float(meno.get("ambientLight", 1.0)))
    engine(scene, s)
    ctx = presets.Ctx(scene, s, folder, presets.molecules_of(scene.objects), scene.camera, say)
    look = presets.load(folder, s, "look")
    if look is not None:
        dress(ctx, look)
    if s["shading"] == "toon":
        toon(scene, s)
    lines(scene, s)
    studio = presets.load(folder, s, "studio")
    if studio is not None:
        studio.setup(ctx)
    output(scene, s)
    return scene, ctx


def dress(ctx, look):
    """A preset look's materials on Meno's balls and sticks, each asked for by what it is: a ball or a stick, its colour as Meno
    drew it, and its element - a stick's, where it is coloured as one of its atoms is."""
    if hasattr(look, "setup"):
        look.setup(ctx)
    made = {}

    def put(obj, part):
        k = (part["kind"], part["element"], tuple(round(c, 4) for c in part["colour"]))
        if k not in made:
            made[k] = look.material(ctx, part)
        obj.data.materials.clear()
        obj.data.materials.append(made[k])

    for m in ctx.molecules:
        by_index = {a["index"]: a for a in m["atoms"]}
        for a in m["atoms"]:
            put(a["object"], {"kind": "ball", "colour": a["colour"], "element": a["element"]})
        for b in m["bonds"]:
            ends = [by_index[i] for i in b["atoms"] if i in by_index]
            for o in b["objects"]:
                c = presets.colour_of(o)
                el = next((e["element"] for e in ends if all(abs(x - y) < 1e-3 for x, y in zip(e["colour"], c))), "")
                put(o, {"kind": "stick", "colour": c, "element": el})
    if hasattr(look, "after"):
        look.after(ctx)


def keyed_frames(scene, folder, sub, count, progress):
    """The scene's frames 1 to `count` rendered into `sub`: their files, in order."""
    scene.frame_start, scene.frame_end = 1, count
    out = os.path.join(folder, "frames", sub)
    scene.render.filepath = os.path.join(out, "####")
    bpy.app.handlers.render_pre.append(progress.begin_frame)
    bpy.app.handlers.render_stats.append(progress.stats)
    bpy.app.handlers.render_post.append(lambda *_: progress.frame_done())
    bpy.ops.render.render(animation=True)
    made = sorted(glob.glob(os.path.join(out, "*.png")))
    if len(made) != count:
        raise SystemExit(f"render.py: Blender rendered {len(made)} of {count} frames")
    return made


def motion_film(s, folder, scene_path, gltf, meno, motion):
    """A film of a preset's motion of one molecule: keyed by the motion (`key`), over as many frames as the film has - or, where
    its Loop is a cycle (`LOOPS`, or `LOOPS(ctx)`), half of them and one, played forward and then back. The frames, in order."""
    scene, ctx = built(s, folder, scene_path, gltf, meno)
    kind = motion.LOOPS(ctx) if callable(getattr(motion, "LOOPS", None)) else getattr(motion, "LOOPS", "cycle")
    loop = s["mode"] == "loop"
    cycle = loop and kind != "seamless"
    n = s["frames"] // 2 + 1 if cycle else s["frames"]
    ctx.request.update(frames=n, seconds=n / s["fps"], mode="loop" if loop and not cycle else "once", cycle=cycle)
    motion.key(ctx)
    progress = Progress(n)
    say("progress 0/100")
    made = keyed_frames(scene, folder, "motion", n, progress)
    progress.end()
    return made + made[-2:0:-1] if cycle else made


def share(total, parts):
    """`total` frames shared among `parts`, the first ones a frame more where they do not divide."""
    return [total // parts + (1 if k < total % parts else 0) for k in range(parts)]


def change_film(s, folder, scene_path, gltf, meno, motion):
    """A film of a preset's motion of one molecule becoming the next, in pieces - Once to the last, Loop back to the first -
    each keyed by the motion (`key_change`) in a scene of its own, its frames a share of the film's. The frames, in order."""
    loop = s["mode"] == "loop"
    # (the molecules, as Meno tagged their nodes)
    mols = sum(1 for n in gltf.get("nodes", []) if "molecule" in ((n.get("extras") or {}).get("meno") or {}))
    if mols < 2:
        raise SystemExit("render.py: a change of molecules needs two molecules or more, rendered together")
    pairs = [(i, (i + 1) % mols) for i in range(mols if loop else mols - 1)]
    total = s["frames"]
    if total < 2 * len(pairs):
        raise SystemExit(f"render.py: {total} frames are too few for {len(pairs)} changes")
    progress = Progress(total)
    say("progress 0/100")
    made, start = [], 0
    for k, ((i, j), n) in enumerate(zip(pairs, share(total, len(pairs)))):
        scene, ctx = built(s, folder, scene_path, gltf, meno)
        ctx.request.update(frames=n, seconds=n / s["fps"], mode="once", piece=k + 1, pieces=len(pairs),
                           start=start / s["fps"], whole=total / s["fps"], loop=loop)
        say(f"Change {k + 1} of {len(pairs)}")
        motion.key_change(ctx, ctx.molecules[i], ctx.molecules[j])
        made += keyed_frames(scene, folder, f"piece-{k + 1:02d}", n, progress)
        start += n
    progress.end()
    return made


def main():
    s = settings_of()
    folder = os.getcwd()
    scene_path = os.path.join(folder, s["scene"])
    with open(scene_path, encoding="utf-8") as f:
        gltf = json.load(f)
    meno = (gltf.get("scenes") or [{}])[0].get("extras", {}).get("meno", {})
    motion = presets.load(folder, s, "motion")
    if motion is not None:
        film = change_film if hasattr(motion, "key_change") else motion_film
        assemble(film(s, folder, scene_path, gltf, meno, motion), s, folder)
    else:
        scene, _ = built(s, folder, scene_path, gltf, meno)
        assemble(render(scene, s, folder), s, folder)
    say(f"Rendered {s['out']}")


main()
