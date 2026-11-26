"""Tests for how the Blender plugin joins a film's frames with their alpha (plugins/blender/render.py `joined`).

render.py runs in Blender's own Python, which has numpy; `joined` is read out of it alone and run here on a made-up frame,
in an environment that has numpy too; where there is none, it is skipped:

    python -m unittest tests/test_plugin_blender_film.py
"""

import ast
import pathlib
import unittest

try:
    import numpy as np
except ImportError:  # (Python alone: nothing to join with)
    np = None

FOLDER = pathlib.Path(__file__).resolve().parents[1] / "plugins/blender"


def functions(*names):
    tree = ast.parse((FOLDER / "render.py").read_text(encoding="utf-8"))
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), "render.py", "exec"), ns)
    return ns


joined = functions("joined", "even_up")["joined"] if np is not None else None

RED = (230, 51, 26)


def scene(w=64, h=64):
    """A made-up frame, its rows from the top: a red disc of alpha 255 inside a ring of alpha 128, on alpha 0 (clear, black)."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - 16, yy - h // 2)
    a = np.where(d < 8, 255, np.where(d < 14, 128, 0)).astype(np.uint8)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[a > 0, :3] = RED
    rgba[..., 3] = a
    return rgba


CORE, RING, OUTSIDE = (32, 16), (32, 27), (32, 50)  # (row, column)


@unittest.skipIf(np is None, "numpy is not installed here")
class Joined(unittest.TestCase):
    def test_below_its_picture_premultiplied_on_black_and_its_alpha_as_grey_under_it(self):
        out = joined(np, scene(), "below")
        self.assertEqual(out.shape, (128, 64, 3))
        picture, alpha = out[:64], out[64:]
        self.assertEqual([tuple(alpha[p]) for p in (CORE, RING, OUTSIDE)], [(255, 255, 255), (128, 128, 128), (0, 0, 0)])
        # (the ring's red multiplied by 128/255, to the nearest)
        self.assertEqual([tuple(picture[p]) for p in (CORE, RING, OUTSIDE)], [RED, (115, 26, 13), (0, 0, 0)])

    def test_to_the_right_of_its_picture(self):
        out = joined(np, scene(), "right")
        self.assertEqual(out.shape, (64, 128, 3))
        picture, alpha = out[:, :64], out[:, 64:]
        self.assertEqual([tuple(alpha[p]) for p in (CORE, RING, OUTSIDE)], [(255, 255, 255), (128, 128, 128), (0, 0, 0)])
        self.assertEqual([tuple(picture[p]) for p in (CORE, RING, OUTSIDE)], [RED, (115, 26, 13), (0, 0, 0)])

    def test_an_odd_frame_lies_in_the_bottom_left_corner_of_each_even_half_its_pad_clear(self):
        out = joined(np, scene(65, 37), "below")
        # (each half 66 x 38: the frame's 37 rows the half's last, its 65 columns the first; the pad, a row above and a column right, clear)
        self.assertEqual(out.shape, (76, 66, 3))
        picture, alpha = out[:38], out[38:]
        self.assertEqual(tuple(alpha[1 + 37 // 2, 16]), (255, 255, 255))
        self.assertEqual(tuple(picture[1 + 37 // 2, 16]), RED)
        self.assertFalse(alpha[0].any())
        self.assertFalse(alpha[:, 65].any())
        self.assertFalse(picture[0].any())


if __name__ == "__main__":
    unittest.main()
