#!/usr/bin/env python3
"""
Interactive Cat GUI — explores the 5-level compositional hierarchy.

Uses matplotlib sliders to control each parameter grouped by Lie-group level.
Directly renders via the v5 JointedCat renderer.

Usage:
    python cat_gui.py                   # default 256px
    python cat_gui.py --img_size 512    # larger
    python cat_gui.py --condition Everything  # start with random params for condition

Author: G. Ruffini / Technical Note companion code
"""

import argparse
import numpy as np
import matplotlib
import sys

# Use TkAgg if available (better for interactive), fallback to default
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from compositional_cat_v2 import (
    JointedCat, LEVEL_PARAMS, CONDITIONS, sample_params
)


# ── Slider layout ──────────────────────────────────────────

LEVEL_NAMES = {
    1: "L1: Pose  SO(1)¹⁶",
    2: "L2: Appearance  R⁶",
    3: "L3: Placement  R²×SO(3)",
    4: "L4: Camera  SE(2)×R⁺",
    5: "L5: Background  R¹",
}

# Friendly names for parameters
PARAM_LABELS = {
    'cam_angle': 'Camera rotation',
    'cam_tx': 'Camera X',
    'cam_ty': 'Camera Y',
    'cam_scale': 'Camera scale',
    'root_x': 'Body X',
    'root_y': 'Body Y',
    'root_angle': 'Yaw (in-plane)',
    'root_elevation': 'Elevation (pitch)',
    'root_roll': 'Roll (tilt)',
    'spine_0': 'Spine joint 1',
    'spine_1': 'Spine joint 2',
    'spine_2': 'Spine joint 3',
    'leg_bl_upper': 'Back-L upper',
    'leg_bl_lower': 'Back-L lower',
    'leg_br_upper': 'Back-R upper',
    'leg_br_lower': 'Back-R lower',
    'leg_fl_upper': 'Front-L upper',
    'leg_fl_lower': 'Front-L lower',
    'leg_fr_upper': 'Front-R upper',
    'leg_fr_lower': 'Front-R lower',
    'head_pan': 'Head pan',
    'head_tilt': 'Head tilt',
    'head_roll': 'Head roll',
    'tail_0': 'Tail joint 1',
    'tail_1': 'Tail joint 2',
    'body_hue': 'Body hue',
    'body_sat': 'Body saturation',
    'body_val': 'Body brightness',
    'limb_thickness': 'Limb thickness',
    'eye_size': 'Eye size',
    'stripe_intensity': 'Stripes',
    'bg_grey': 'BG grey level',
}


def build_gui(img_size: int = 256, initial_condition: str = 'Static'):
    """Build the interactive matplotlib GUI."""

    cat = JointedCat()

    # Optionally randomize initial params
    if initial_condition != 'Static':
        rng = np.random.RandomState(42)
        cat.params = sample_params(CONDITIONS.get(initial_condition, []), rng)

    # Collect all slider specs grouped by level
    slider_specs = []  # (param_name, lo, hi, level)
    for level in sorted(LEVEL_PARAMS.keys()):
        for pname, (lo, hi) in LEVEL_PARAMS[level].items():
            slider_specs.append((pname, lo, hi, level))

    n_sliders = len(slider_specs)

    # ── Figure layout ──
    # Left panel: sliders, Right panel: cat image
    # Taller figure to fit 32 sliders without cramping
    fig = plt.figure(figsize=(14, 11))
    fig.canvas.manager.set_window_title('Compositional Cat — Hierarchy Explorer')
    fig.patch.set_facecolor('#f5f5f0')

    # Image axis (right side)
    ax_img = fig.add_axes([0.48, 0.06, 0.50, 0.88])
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Initial render
    img = cat.render(img_size=img_size)
    img_handle = ax_img.imshow(np.array(img))
    ax_img.set_title(f'v6 Renderer — {img_size}×{img_size}', fontsize=11)

    # ── Slider axes (left side) ──
    # Compact sizing for 32 params + 5 level headers
    slider_height = 0.016
    slider_gap = 0.003
    slider_left = 0.13
    slider_width = 0.29
    top = 0.97

    sliders = {}
    current_y = top

    header_colors = {1: '#c44', 2: '#a85', 3: '#2a7', 4: '#47c', 5: '#666'}
    level_colors = {1: '#e88', 2: '#ca8', 3: '#8c8', 4: '#88c', 5: '#aaa'}

    prev_level = None
    for pname, lo, hi, level in slider_specs:
        # Level header — thin colored separator with label
        if level != prev_level:
            if prev_level is not None:
                current_y -= 0.005  # gap between groups
            # Draw a thin colored line as separator
            hc = header_colors.get(level, '#333')
            fig.add_axes([slider_left, current_y - 0.001, slider_width, 0.001],
                         facecolor=hc, xticks=[], yticks=[])
            # Level label to the LEFT of sliders (no overlap)
            fig.text(0.01, current_y - 0.003, LEVEL_NAMES[level],
                     fontsize=6.5, fontweight='bold', color=hc,
                     verticalalignment='top', fontstyle='italic')
            current_y -= 0.012
            prev_level = level

        ax_s = fig.add_axes([slider_left, current_y, slider_width, slider_height])
        label = PARAM_LABELS.get(pname, pname)
        s = Slider(ax_s, label, lo, hi,
                   valinit=cat.params[pname],
                   valstep=(hi - lo) / 200,
                   color=level_colors.get(level, '#ddd'))
        ax_s.tick_params(labelsize=4)
        for text in ax_s.texts:
            text.set_fontsize(5.5)

        sliders[pname] = s
        current_y -= (slider_height + slider_gap)

    # ── Update function ──
    def update(val=None):
        for pname, s in sliders.items():
            cat.params[pname] = s.val
        img = cat.render(img_size=img_size)
        img_handle.set_data(np.array(img))
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    # ── Buttons ──
    # Reset button
    ax_reset = fig.add_axes([0.02, 0.02, 0.08, 0.03])
    btn_reset = Button(ax_reset, 'Reset', color='#eee', hovercolor='#ddd')
    btn_reset.label.set_fontsize(8)

    def reset(event):
        cat.set_defaults()
        for pname, s in sliders.items():
            s.set_val(cat.params[pname])
        update()
    btn_reset.on_clicked(reset)

    # Random buttons per condition
    ax_rand = fig.add_axes([0.12, 0.02, 0.10, 0.03])
    btn_rand = Button(ax_rand, 'Random All', color='#e8e8ff', hovercolor='#ccf')
    btn_rand.label.set_fontsize(8)

    def randomize(event):
        rng = np.random.RandomState()
        cat.params = sample_params(CONDITIONS['Everything'], rng)
        for pname, s in sliders.items():
            s.set_val(cat.params[pname])
        update()
    btn_rand.on_clicked(randomize)

    # Random per level buttons
    button_x = 0.24
    level_buttons = {}
    for level in sorted(LEVEL_PARAMS.keys()):
        ax_b = fig.add_axes([button_x, 0.02, 0.06, 0.03])
        btn = Button(ax_b, f'Rand L{level}', color='#f0f0e0', hovercolor='#dde')
        btn.label.set_fontsize(6)

        def make_rand_level(lv):
            def rand_level(event):
                rng = np.random.RandomState()
                for pn, (lo, hi) in LEVEL_PARAMS[lv].items():
                    val = rng.uniform(lo, hi)
                    cat.params[pn] = val
                    if pn in sliders:
                        sliders[pn].set_val(val)
                update()
            return rand_level

        btn.on_clicked(make_rand_level(level))
        level_buttons[level] = btn
        button_x += 0.065

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Cat GUI')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--condition', type=str, default='Static',
                        choices=list(CONDITIONS.keys()),
                        help='Initial condition for parameter randomization')
    args = parser.parse_args()

    build_gui(img_size=args.img_size, initial_condition=args.condition)
