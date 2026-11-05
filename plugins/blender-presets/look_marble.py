# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Meno contributors. Under the GPL, in this plugin's LICENSE.
"""The look "marble" (manifest `presets.looks`), made by looks_shared.py: docs/plugins/blender-presets.md."""
import looks_shared as shared

LOOK = "marble"


def setup(ctx):
    shared.setup(ctx, LOOK)


def material(ctx, part):
    return shared.material(ctx, LOOK, part)


def after(ctx):
    shared.after(ctx, LOOK)
