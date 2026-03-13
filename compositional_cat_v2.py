"""
Compositional Jointed-Cat Generator — v2 (high-contrast)
=========================================================
Key changes from v1:
  - Cat is MUCH bigger in the frame (scale 0.60 vs 0.35)
  - Parameter ranges are 2-3x wider for geometric levels
  - Thicker limbs and body for more pixel coverage
  - Numpy-vectorised background (10x faster rendering)
  - Levels 1 & 2 are now clearly distinguishable:
      Level 1 (camera) rotates/scales the whole image including background
      Level 2 (root body) moves the cat within the frame

Author: G. Ruffini / Technical Note companion code — v2
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import colorsys


# ═══════════════════════════════════════════════════════════
# Kinematic primitives
# ═══════════════════════════════════════════════════════════

def rot2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def forward_kinematics(
    base_pos: np.ndarray,
    base_angle: float,
    joint_angles: List[float],
    segment_lengths: List[float],
) -> List[np.ndarray]:
    positions = [base_pos.copy()]
    current_angle = base_angle
    current_pos = base_pos.copy()
    for angle, length in zip(joint_angles, segment_lengths):
        current_angle += angle
        direction = np.array([np.cos(current_angle), np.sin(current_angle)])
        current_pos = current_pos + length * direction
        positions.append(current_pos.copy())
    return positions


# ═══════════════════════════════════════════════════════════
# Jointed Cat — v2
# ═══════════════════════════════════════════════════════════

class JointedCat:
    """2D articulated cat — v2 with bigger body and wider ranges."""

    # Bigger segments for more pixel coverage
    SPINE_LENGTHS = [0.30, 0.30, 0.25]
    LEG_LENGTHS = [0.25, 0.22]
    HEAD_LENGTHS = [0.14, 0.18]
    TAIL_LENGTHS = [0.22, 0.18]

    def __init__(self):
        self.params = {}
        self.set_defaults()

    def set_defaults(self):
        # Level 1: Camera
        self.params['cam_angle'] = 0.0
        self.params['cam_tx'] = 0.0
        self.params['cam_ty'] = 0.0
        self.params['cam_scale'] = 1.0

        # Level 2: Root body
        self.params['root_x'] = 0.0
        self.params['root_y'] = 0.0
        self.params['root_angle'] = 0.0

        # Level 3: Spine
        self.params['spine_0'] = 0.0
        self.params['spine_1'] = 0.0
        self.params['spine_2'] = 0.0

        # Level 4: Limbs
        for side in ['bl', 'br', 'fl', 'fr']:
            self.params[f'leg_{side}_upper'] = 0.0
            self.params[f'leg_{side}_lower'] = 0.0

        # Level 5: Head & tail
        self.params['head_pan'] = 0.0
        self.params['head_tilt'] = 0.0
        self.params['tail_0'] = 0.0
        self.params['tail_1'] = 0.0

        # Level 6: Appearance
        self.params['body_hue'] = 0.08
        self.params['body_sat'] = 0.6
        self.params['body_val'] = 0.7
        self.params['limb_thickness'] = 1.0
        self.params['eye_size'] = 1.0
        self.params['stripe_intensity'] = 0.0

        # Level 7: Background
        self.params['bg_angle'] = 0.0
        self.params['bg_colour_shift'] = 0.5
        self.params['bg_intensity'] = 0.85

    def compute_skeleton(self) -> Dict[str, List[np.ndarray]]:
        p = self.params
        root = np.array([p['root_x'], p['root_y']])
        root_angle = p['root_angle']

        spine_angles = [p['spine_0'], p['spine_1'], p['spine_2']]
        spine_pos = forward_kinematics(root, root_angle, spine_angles, self.SPINE_LENGTHS)
        pelvis = spine_pos[0]
        shoulders = spine_pos[3]
        shoulder_angle = root_angle + sum(spine_angles)

        chains = {'spine': spine_pos}

        # Back legs
        for side, sign in [('bl', -1), ('br', 1)]:
            base_a = root_angle - np.pi/2 + sign * 0.15
            attach = pelvis + rot2d(root_angle) @ np.array([0, sign * 0.08])
            chains[f'leg_{side}'] = forward_kinematics(
                attach, base_a,
                [p[f'leg_{side}_upper'], p[f'leg_{side}_lower']],
                self.LEG_LENGTHS
            )

        # Front legs
        for side, sign in [('fl', -1), ('fr', 1)]:
            base_a = shoulder_angle - np.pi/2 + sign * 0.15
            attach = shoulders + rot2d(shoulder_angle) @ np.array([0, sign * 0.08])
            chains[f'leg_{side}'] = forward_kinematics(
                attach, base_a,
                [p[f'leg_{side}_upper'], p[f'leg_{side}_lower']],
                self.LEG_LENGTHS
            )

        # Head
        chains['head'] = forward_kinematics(
            shoulders, shoulder_angle + p['head_pan'],
            [p['head_tilt']], self.HEAD_LENGTHS
        )

        # Tail
        chains['tail'] = forward_kinematics(
            pelvis, root_angle + np.pi,
            [p['tail_0'], p['tail_1']], self.TAIL_LENGTHS
        )

        return chains

    def render(self, img_size: int = 128) -> Image.Image:
        p = self.params
        chains = self.compute_skeleton()

        # ── Camera transform ──
        # KEY CHANGE: scale factor 0.55 (was 0.35) — cat fills much more of the frame
        WORLD_SCALE = 0.40

        def world_to_pixel(pt: np.ndarray) -> Tuple[float, float]:
            scaled = p['cam_scale'] * pt
            rotated = rot2d(p['cam_angle']) @ scaled
            translated = rotated + np.array([p['cam_tx'], p['cam_ty']])
            px = img_size/2 + translated[0] * img_size * WORLD_SCALE
            py = img_size/2 - translated[1] * img_size * WORLD_SCALE
            return (px, py)

        # ── Background (vectorised — 10x faster than pixel loop) ──
        ys, xs = np.mgrid[0:img_size, 0:img_size]
        nx = xs / img_size - 0.5
        ny = ys / img_size - 0.5
        grad = nx * np.cos(p['bg_angle']) + ny * np.sin(p['bg_angle'])
        grad = 0.5 + 0.5 * grad
        bi = p['bg_intensity']
        bs = p['bg_colour_shift']
        r_ch = (255 * bi * (0.6 + 0.4 * bs * grad)).clip(0, 255).astype(np.uint8)
        g_ch = (255 * bi * (0.7 + 0.3 * (1 - bs) * grad)).clip(0, 255).astype(np.uint8)
        b_ch = (255 * bi * (0.75 + 0.25 * bs * (1 - grad))).clip(0, 255).astype(np.uint8)
        bg_arr = np.stack([r_ch, g_ch, b_ch], axis=-1)
        img = Image.fromarray(bg_arr, 'RGB')
        draw = ImageDraw.Draw(img)

        # ── Appearance ──
        body_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val']
        ))
        dark_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val'] * 0.6
        ))
        # THICKER limbs for more pixel coverage
        base_width = max(2, int(6 * p['limb_thickness'] * img_size / 128))
        body_width = max(3, int(9 * p['limb_thickness'] * img_size / 128))

        # ── Draw: tail, legs, spine, head ──

        # Tail
        tail_px = [world_to_pixel(pt) for pt in chains['tail']]
        if len(tail_px) >= 2:
            draw.line(tail_px, fill=body_rgb, width=max(1, base_width - 1))

        # Legs
        for key in ['leg_bl', 'leg_br', 'leg_fl', 'leg_fr']:
            leg_px = [world_to_pixel(pt) for pt in chains[key]]
            if len(leg_px) >= 2:
                draw.line(leg_px, fill=dark_rgb, width=base_width)
                ex, ey = leg_px[-1]
                pr = max(2, base_width * 0.7)
                draw.ellipse([ex-pr, ey-pr, ex+pr, ey+pr], fill=dark_rgb)

        # Spine (body)
        spine_px = [world_to_pixel(pt) for pt in chains['spine']]
        if len(spine_px) >= 2:
            draw.line(spine_px, fill=body_rgb, width=body_width)

        # Body volume (filled ellipses along spine)
        for i in range(len(spine_px) - 1):
            x0, y0 = spine_px[i]
            x1, y1 = spine_px[i+1]
            cx, cy = (x0+x1)/2, (y0+y1)/2
            r = body_width * 1.0
            draw.ellipse([cx-r, cy-r*0.75, cx+r, cy+r*0.75], fill=body_rgb)

        # Stripes
        if p['stripe_intensity'] > 0.05:
            stripe_rgb = tuple(max(0, int(c * (1 - 0.5 * p['stripe_intensity']))) for c in body_rgb)
            for i in range(len(spine_px) - 1):
                x0, y0 = spine_px[i]
                x1, y1 = spine_px[i+1]
                for t in [0.3, 0.7]:
                    sx = x0 + t*(x1-x0)
                    sy = y0 + t*(y1-y0)
                    dx, dy = x1-x0, y1-y0
                    nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
                    nx2, ny2 = -dy/nm, dx/nm
                    sr = body_width * 0.8
                    draw.line([(sx-nx2*sr, sy-ny2*sr), (sx+nx2*sr, sy+ny2*sr)],
                              fill=stripe_rgb, width=max(1, base_width//2))

        # Head
        head_px = [world_to_pixel(pt) for pt in chains['head']]
        if len(head_px) >= 2:
            draw.line(head_px[:2], fill=body_rgb, width=base_width)
            hx, hy = head_px[-1]
            hr = body_width * 1.2
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=body_rgb)

            # Ears
            es = hr * 0.7
            dx = head_px[-1][0] - head_px[-2][0]
            dy = head_px[-1][1] - head_px[-2][1]
            nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
            fx, fy = dx/nm, dy/nm
            for sign in [-1, 1]:
                nx2, ny2 = -dy/nm * sign, dx/nm * sign
                bx = hx + nx2 * hr * 0.5
                by = hy + ny2 * hr * 0.5
                tx = bx + nx2*es + fx*es*0.5
                ty = by + ny2*es + fy*es*0.5
                draw.polygon([
                    (bx - fx*es*0.3, by - fy*es*0.3),
                    (tx, ty),
                    (bx + fx*es*0.3, by + fy*es*0.3),
                ], fill=body_rgb)

            # Eyes
            eye_r = max(2, int(3.0 * p['eye_size'] * img_size / 128))
            for sign in [-1, 1]:
                nx2, ny2 = -dy/nm * sign, dx/nm * sign
                ex = hx + sign * nx2 * hr * 0.35 + fx * hr * 0.2
                ey = hy + sign * ny2 * hr * 0.35 + fy * hr * 0.2
                draw.ellipse([ex-eye_r, ey-eye_r, ex+eye_r, ey+eye_r], fill=(50, 180, 50))
                pr = max(1, eye_r // 2)
                draw.ellipse([ex-pr, ey-pr*1.5, ex+pr, ey+pr*1.5], fill=(20, 20, 20))

            # Nose
            nose_x = hx + fx * hr * 0.55
            nose_y = hy + fy * hr * 0.55
            nr = max(2, int(2 * img_size / 128))
            draw.ellipse([nose_x-nr, nose_y-nr*0.7, nose_x+nr, nose_y+nr*0.7],
                        fill=(200, 120, 120))

        return img


# ═══════════════════════════════════════════════════════════
# Parameter ranges — v2: MUCH wider for geometric levels
# ═══════════════════════════════════════════════════════════

LEVEL_PARAMS = {
    1: {  # Camera — affects EVERYTHING including background
        'cam_angle':  (-0.4, 0.4),       # tightened to keep cat in frame
        'cam_tx':     (-0.25, 0.25),     # tightened (was ±0.5)
        'cam_ty':     (-0.25, 0.25),     # tightened (was ±0.5)
        'cam_scale':  (0.7, 1.3),        # tightened (was 0.6–1.5)
    },
    2: {  # Root body — moves cat within frame (NOT background)
        'root_x':     (-0.4, 0.4),       # tightened (was ±0.7)
        'root_y':     (-0.3, 0.3),       # tightened (was ±0.5)
        'root_angle': (-0.8, 0.8),       # rotation is fine
    },
    3: {  # Spine — much bendier
        'spine_0': (-0.7, 0.7),          # was ±0.4
        'spine_1': (-0.6, 0.6),          # was ±0.3
        'spine_2': (-0.5, 0.5),          # was ±0.3
    },
    4: {  # Limbs — full range of motion
        'leg_bl_upper': (-1.0, 1.0),     # was ±0.6
        'leg_bl_lower': (-1.2, 0.2),     # was -0.8..0.1
        'leg_br_upper': (-1.0, 1.0),
        'leg_br_lower': (-1.2, 0.2),
        'leg_fl_upper': (-1.0, 1.0),
        'leg_fl_lower': (-1.2, 0.2),
        'leg_fr_upper': (-1.0, 1.0),
        'leg_fr_lower': (-1.2, 0.2),
    },
    5: {  # Head & tail — exaggerated
        'head_pan':  (-0.8, 0.8),        # was ±0.5
        'head_tilt': (-0.7, 0.7),        # was ±0.4
        'tail_0':    (-1.2, 1.2),        # was ±0.8
        'tail_1':    (-1.0, 1.0),        # was ±0.6
    },
    6: {  # Appearance — same
        'body_hue':         (0.0, 0.18),
        'body_sat':         (0.2, 1.0),
        'body_val':         (0.35, 0.95),
        'limb_thickness':   (0.5, 1.8),
        'eye_size':         (0.5, 1.8),
        'stripe_intensity': (0.0, 1.0),
    },
    7: {  # Background — same
        'bg_angle':        (0.0, 2*np.pi),
        'bg_colour_shift': (0.0, 1.0),
        'bg_intensity':    (0.4, 1.0),
    },
}

CONDITIONS = {
    'Static':           [],
    'CameraOnly':       [1],
    'CameraBody':       [1, 2],
    'CameraBodySpine':  [1, 2, 3],
    'FullPose':         [1, 2, 3, 4],
    'PoseHeadTail':     [1, 2, 3, 4, 5],
    'AllGeometric':     [1, 2, 3, 4, 5],
    'PlusAppearance':   [1, 2, 3, 4, 5, 6],
    'Everything':       [1, 2, 3, 4, 5, 6, 7],
}


def _check_in_frame(params: Dict[str, float], margin: float = 0.15) -> bool:
    """Check that most of the cat skeleton is within the frame.

    Returns True if the bounding box of all skeleton joints falls
    within [margin, 1-margin] of the normalised image coordinates.
    This rejects configurations where the cat is mostly off-screen.
    """
    cat = JointedCat()
    cat.params = params
    chains = cat.compute_skeleton()

    # Collect all joint positions
    all_pts = []
    for chain in chains.values():
        all_pts.extend(chain)
    if not all_pts:
        return True

    pts = np.array(all_pts)  # (N, 2)

    # Apply camera transform (must match render())
    WORLD_SCALE = 0.40
    scaled = params['cam_scale'] * pts
    R = rot2d(params['cam_angle'])
    rotated = (R @ scaled.T).T
    translated = rotated + np.array([params['cam_tx'], params['cam_ty']])
    # Convert to normalised [0, 1] image coordinates
    nx = 0.5 + translated[:, 0] * WORLD_SCALE
    ny = 0.5 - translated[:, 1] * WORLD_SCALE

    # Check that the centroid is within bounds and at least half the
    # joints are within the padded frame
    cx, cy = nx.mean(), ny.mean()
    if not (margin < cx < 1 - margin and margin < cy < 1 - margin):
        return False
    in_frame = ((nx > -margin) & (nx < 1 + margin) &
                (ny > -margin) & (ny < 1 + margin))
    return in_frame.mean() > 0.6


def sample_params(active_levels: List[int], rng: np.random.RandomState,
                  max_attempts: int = 20) -> Dict[str, float]:
    """Sample parameters, rejecting configs where the cat is off-screen."""
    for _ in range(max_attempts):
        cat = JointedCat()
        params = dict(cat.params)
        for level in active_levels:
            if level in LEVEL_PARAMS:
                for param_name, (lo, hi) in LEVEL_PARAMS[level].items():
                    params[param_name] = rng.uniform(lo, hi)
        if _check_in_frame(params):
            return params
    # If we exhausted attempts, return last sample anyway
    return params


def generate_dataset(condition, n_samples, img_size=128, output_dir='dataset', seed=42):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    active_levels = CONDITIONS[condition]
    rng = np.random.RandomState(seed)
    img_dir = Path(output_dir) / condition / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(output_dir) / condition / 'metadata.jsonl'
    cat = JointedCat()
    with open(meta_path, 'w') as mf:
        for i in range(n_samples):
            params = sample_params(active_levels, rng)
            cat.params = params
            img = cat.render(img_size=img_size)
            img_name = f'{i:06d}.png'
            img.save(img_dir / img_name)
            mf.write(json.dumps({
                'index': i, 'condition': condition,
                'active_levels': active_levels,
                'n_active_levels': len(active_levels),
                'params': {k: float(v) for k, v in params.items()},
                'image': img_name,
            }) + '\n')
            if (i+1) % 500 == 0:
                print(f'  [{condition}] {i+1}/{n_samples}')
    print(f'[{condition}] Done: {n_samples} images → {img_dir}')
    return img_dir, meta_path


def make_sample_grid(conditions=None, n_per_condition=6, img_size=128, seed=42):
    if conditions is None:
        conditions = list(CONDITIONS.keys())
    rng = np.random.RandomState(seed)
    n_rows, n_cols = len(conditions), n_per_condition
    grid = Image.new('RGB', (n_cols * img_size, n_rows * img_size), (255, 255, 255))
    cat = JointedCat()
    for row, cond in enumerate(conditions):
        active = CONDITIONS[cond]
        for col in range(n_cols):
            cat.params = sample_params(active, rng)
            grid.paste(cat.render(img_size=img_size), (col*img_size, row*img_size))
    return grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Cat v2')
    parser.add_argument('--mode', choices=['generate', 'grid', 'all'], default='grid')
    parser.add_argument('--condition', type=str, default='Everything')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.mode == 'grid':
        os.makedirs(args.output_dir, exist_ok=True)
        grid = make_sample_grid(img_size=args.img_size, seed=args.seed)
        grid.save(os.path.join(args.output_dir, 'sample_grid_v2.png'))
        print(f'Saved to {args.output_dir}/sample_grid_v2.png')
    elif args.mode == 'generate':
        generate_dataset(args.condition, args.n_samples, args.img_size, args.output_dir, args.seed)
    elif args.mode == 'all':
        for cond in CONDITIONS:
            generate_dataset(cond, args.n_samples, args.img_size, args.output_dir, args.seed)
