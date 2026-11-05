# Blender presets: the interface `preset`, version 1

The Blender plugin is a renderer: it renders the scene Meno sends, as
Meno asks, and adds nothing of its own (docs/RENDER.md). Looks, studios
and motions made with Blender's own materials, light and keys come from
**preset plugins**, which use the renderer through the interface it
declares:

```json
"interfaces": [{"id": "preset", "version": "1.0.0"}]
```

A preset plugin names the renderer in its `depends`, with the versions of
it and of the interface it was written for (docs/PLUGINS.md, *Depends*):

```json
"depends": [{"plugin": "blender", "versions": ">=0.7.0 <0.8.0",
             "interfaces": [{"id": "preset", "versions": ">=1.0.0 <2.0.0"}]}]
```

and declares its presets (docs/PLUGINS.md, *Presets*): its looks and
motions, each shaped as a render role's (`id`, `label`, `description`,
and for a look `seeThrough`, for a motion `shortest` and `fixed`), its
studios (`id`, `label`, `description`), each with the `file` it is made
by; the options that adjust them (`options`, each shown `when` its look,
studio or motion is chosen); and the files those share (`files`).

A preset plugin has no worker. Meno copies the files its presets name
into the render job (`presets/<plugin id>/`) - which names those files may
have is Meno's to say (docs/PLUGINS.md, *Presets*), the renderer checking
the entry's shape alone - and the renderer runs the chosen ones inside its
Blender, after it has built Meno's scene, against
one object, `ctx`. A preset runs against `ctx` alone: it reaches nothing
of Meno's, of another plugin's or of the network beyond what the
renderer's own job has. That is what using the renderer means.

## `ctx`

| Field | What it is |
|---|---|
| `version` | `1`. |
| `scene` | Blender's scene, as the renderer built it from Meno's. |
| `request` | What Meno asked: `width`, `height`, `fps`, `frames`, `seconds`, `mode` (`once` or `loop`), `background` (`transparent`, `white`, `black` or `grey`), `key`, `ambient` and `softness` (the light against the canvas's), the look's parts as asked, and `values`: the options of the preset plugins chosen from, together - so one preset plugin's option ids are never another's. |
| `molecules` | Meno's molecules, in the order Meno wrote them. Each has `object` (its node), `atoms` and `bonds`. An atom: `object` (its ball), `index` (its atom in the molecule), `element`, `radius` (the ball's, in the scene's units) and `colour` (linear RGB, as Meno drew it). A bond: `objects` (its sticks: a double bond's lines are two, a line coloured by its atoms is two halves), `order`, `atoms` (the indices at its ends) and `centre` (a stick to a ring's centre: that ring's atoms). |
| `camera` | Meno's camera. A motion may key it. |
| `folder` | Where the job's files are, the preset's beside them. |
| `say(line)` | Writes a line to the job's log, which Meno shows. |
| `colour_of(obj)` | The colour Meno drew a ball or a stick in, linear. |

And the helpers version 1 guarantees:

| Helper | What it does |
|---|---|
| `ease(t)`, `ease_in(t)`, `ease_out(t)` | Easing curves on 0..1. |
| `key(obj, path, frame, value)` | Sets `obj`'s `path` (`location`, `rotation_euler`, `scale`, or a node input's `default_value`) to `value` and keys it at `frame`. |
| `camera_at(yaw, pitch, margin)` | Places Meno's camera round the molecules, looking at their middle, far enough back that all of them fit with `margin` to spare. |
| `material(name)` | A new node material, its Principled BSDF and output linked, to build on. |

## What a preset module defines

**A look** (`looks[].file`):

- `material(ctx, part) -> bpy.types.Material`, called for each ball and
  stick: `part` is `{"kind": "ball" | "stick", "colour": (r, g, b),
  "element": "C"}`; a stick coloured by its atom has that atom's element,
  one in the bond's colour none.
- optionally `setup(ctx)`, before its materials, for what the look needs
  round the molecules (a panel lighting its metal, say), and the render
  settings its materials need. It never replaces the background.
- optionally `after(ctx)`, once its materials are on, for what it adds
  to the balls and sticks themselves (hair, say).

**A studio** (`studios[].file`): `setup(ctx)`, which builds the backdrop
and its lights from `ctx.request`, the lights as strong as asked. A studio
is the backdrop chosen, so it draws its backdrop whatever the background
asked: the renderer has no rule against a studio and a transparent
background, nor does Meno, which knows no plugin's studios. Where the
studio is itself the background, its manifest says so (`fixed` on the
renderer's `background`, as a motion fixes its renderer), and the card
shows the background as the studio sets it, said and not to be changed,
so that what the card says is what the picture is.

**A motion** (`motions[].file`):

- `NOMINAL`: the length in seconds it is timed in; a film of another
  length plays it faster or slower.
- `LOOPS`: `"seamless"` where its Loop joins of itself, `"cycle"` where a
  Loop plays it forward and then back - or a function of `ctx` saying
  which, where that depends on what was asked. A cycle's half is keyed
  with `ctx.request["cycle"]` true, its Once over half the frames.
- a motion of one molecule: `key(ctx)`, which keys Meno's atoms, bonds and
  camera over `ctx.request["frames"]`;
- a motion of one molecule becoming the next (its `shortest` says
  `change`): `key_change(ctx, a, b)`, which keys one piece, molecule `a`
  becoming molecule `b`, both Meno's as drawn. The renderer makes the
  pieces, one for each change, the camera running on through them, and
  joins them.

The renderer renders the frames and makes the film.

## What a preset may change

A preset changes only what it is for. A look dresses Meno's atoms and
bonds; a motion moves them, its camera and its own effects; a studio puts
a backdrop and lights round them. None redraws the molecules from a
palette or sizes of its own: their elements, places, radii and colours
are Meno's.

A change of molecules may choose which of A's atoms flies to which of B's
places (Film looks and motions' Ring takes the least travel in all,
the same element preferred). That choice is the motion's choreography,
made inside its own file for the film alone: never a mapping of one
molecule onto the other, never said as one, and nothing of it reaches
Meno.

## Versions

The interface's version is three numbers. A new field or helper is a new
minor version (1.1.0); anything taken away, or changed in what it means,
a new major one (2.0.0). The renderer says the version it implements, and
a preset plugin the versions it was written for; Meno offers its presets
only while both its renderer's version and the interface's are within
what it says, and stops it otherwise, saying why.
