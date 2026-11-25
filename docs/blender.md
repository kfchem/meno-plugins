# The Blender plugin: a film on a transparent background

A film rendered on Transparent (the option `film-background`, `transparent`) is two files from one render, both said made
for its entry by `collect`:

```json
{"made": [
  {"entry": 0, "file": "film.mp4", "media": "video/mp4", "matte": "below"},
  {"entry": 0, "file": "film.mov", "media": "video/quicktime"}
]}
```

- **`film.mp4`, the film for the page**: H.264 in an MP4, as a film on a colour is, each frame of which holds the picture
  and its alpha side by side. One half holds the picture, its colours multiplied by their alpha (on black); the other
  holds the alpha as grey (red, green and blue each the alpha). `matte` says where the alpha lies: `"below"` the picture,
  or to its `"right"`. Each half is the size asked for, made even, the frame in its bottom-left corner as a padded film's
  is, the pad clear. The halves lie one above the other where the whole fits within 4096 x 2304 pixels (what H.264 at
  level 5.1 holds, ITU-T H.264 Table A-1), else side by side where that fits; a film too large for either is refused
  before it is rendered, with a sentence saying how large it may be.
- **`film.mov`, the film for slide programs**: QuickTime, ProRes 4444, with an alpha channel, from the same frames, at
  the size asked for made even. Not every program reads this file.

`matte` is said only on a `video/mp4` item; an item of `video/quicktime` only beside one that says it. A film on White,
Black or Grey is one MP4, as before.

How it is made (`render.py`): the frames are rendered with their transparency, as for a picture. The QuickTime file is
written from them by Blender's sequencer with no background under them; each frame is joined with its alpha in Blender's
own Python (`joined`, with numpy), and the joined frames written as the MP4 by the same sequencer, over black.
