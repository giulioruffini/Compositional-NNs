"""
Compositional Jointed-Cat Generator
====================================
A 2D articulated cat model with a fully controlled Lie-group
compositional hierarchy, for testing dynamic-depth predictions.

Hierarchy (flag of sub-groups):
  Level 1: Camera       — SE(2) x R+  (rotation, translation, scale)
  Level 2: Root body    — SE(2)        (body position & orientation)
  Level 3: Spine        — SO(2)^3      (3 spine joints)
  Level 4: Limbs        — SO(2)^8      (2 joints x 4 legs)
  Level 5: Head & tail  — SO(2)^4      (head pan/tilt, 2 tail joints)
  Level 6: Appearance   — R^6          (body colour HSV, limb thickness, eye size, stripe intensity)
  Level 7: Background   — R^3          (gradient angle, colour shift, intensity)

Each image comes with full metadata: which levels were active,
all parameter values, enabling exact verification of gate predictions.

Author: G. Ruffini / Technical Note companion code
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
# Kinematic primitives (Product of Exponentials)
# ═══════════════════════════════════════════════════════════

def rot2d(angle: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def apply_se2(point: np.ndarray, angle: float, tx: float, ty: float) -> np.ndarray:
    """Apply SE(2) transformation: rotate then translate."""
    return rot2d(angle) @ point + np.array([tx, ty])


def forward_kinematics(
    base_pos: np.ndarray,
    base_angle: float,
    joint_angles: List[float],
    segment_lengths: List[float],
) -> List[np.ndarray]:
    """Forward kinematics for a planar kinematic chain (product of exponentials in SO(2)).

    Args:
        base_pos: (2,) world position of the chain root.
        base_angle: Initial angle in radians.
        joint_angles: Per-joint rotation (added sequentially).
        segment_lengths: Length of each segment.

    Returns:
        List of (2,) positions: [base_pos, joint_1_pos, ..., end_pos].
    """
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
# The Jointed Cat model
# ═══════════════════════════════════════════════════════════

class JointedCat:
    """2D articulated cat with a 7-level Lie-group compositional hierarchy.

    The cat is built from kinematic chains (product of exponentials in SO(2)):
    - Spine: 3-segment chain from pelvis to shoulders
    - 4 legs: each a 2-segment chain from hip/shoulder
    - Head: 2-segment chain (neck + head) from shoulder tip
    - Tail: 2-segment chain from pelvis

    Attributes:
        params: Dict of parameter name -> float. Keys come from LEVEL_PARAMS;
            set via set_defaults() then override with sample_params() for a condition.
    """

    # Default segment lengths (in normalised coordinates, body ~ 1.0)
    SPINE_LENGTHS = [0.25, 0.25, 0.20]     # pelvis -> mid -> shoulders
    LEG_LENGTHS = [0.20, 0.18]              # upper -> lower
    HEAD_LENGTHS = [0.12, 0.15]             # neck -> head
    TAIL_LENGTHS = [0.18, 0.15]             # tail base -> tail tip

    def __init__(self):
        self.params = {}
        self.set_defaults()

    def set_defaults(self):
        """Set all parameters to rest pose."""
        # Level 1: Camera
        self.params['cam_angle'] = 0.0
        self.params['cam_tx'] = 0.0
        self.params['cam_ty'] = 0.0
        self.params['cam_scale'] = 1.0

        # Level 2: Root body SE(2)
        self.params['root_x'] = 0.0
        self.params['root_y'] = 0.0
        self.params['root_angle'] = 0.0

        # Level 3: Spine SO(2)^3
        self.params['spine_0'] = 0.0
        self.params['spine_1'] = 0.0
        self.params['spine_2'] = 0.0

        # Level 4: Limbs SO(2)^8
        # Back-left, back-right, front-left, front-right
        # Each has hip/shoulder angle + knee/elbow angle
        for side in ['bl', 'br', 'fl', 'fr']:
            self.params[f'leg_{side}_upper'] = 0.0
            self.params[f'leg_{side}_lower'] = 0.0

        # Level 5: Head & tail
        self.params['head_pan'] = 0.0
        self.params['head_tilt'] = 0.0
        self.params['tail_0'] = 0.0
        self.params['tail_1'] = 0.0

        # Level 6: Appearance
        self.params['body_hue'] = 0.08        # orange-ish
        self.params['body_sat'] = 0.6
        self.params['body_val'] = 0.7
        self.params['limb_thickness'] = 1.0    # multiplier
        self.params['eye_size'] = 1.0          # multiplier
        self.params['stripe_intensity'] = 0.0  # 0 = no stripes

        # Level 7: Background
        self.params['bg_angle'] = 0.0
        self.params['bg_colour_shift'] = 0.5
        self.params['bg_intensity'] = 0.85

    def compute_skeleton(self) -> Dict[str, List[np.ndarray]]:
        """
        Compute all joint positions using forward kinematics.
        Returns dict of named chains, each a list of 2D points.
        """
        p = self.params

        # ── Level 2: Root position ──
        root = np.array([p['root_x'], p['root_y']])
        root_angle = p['root_angle']

        # ── Level 3: Spine chain ──
        # Spine goes from pelvis (root) toward head
        spine_angles = [p['spine_0'], p['spine_1'], p['spine_2']]
        spine_pos = forward_kinematics(
            root, root_angle, spine_angles, self.SPINE_LENGTHS
        )
        pelvis = spine_pos[0]
        mid_spine = spine_pos[1]
        upper_spine = spine_pos[2]
        shoulders = spine_pos[3]

        # Current angle at shoulders (cumulative)
        shoulder_angle = root_angle + sum(spine_angles)

        # ── Level 4: Legs ──
        chains = {'spine': spine_pos}

        # Back legs attach at pelvis
        # Rest angles: legs point downward (roughly -pi/2 from body)
        for i, (side, sign) in enumerate([('bl', -1), ('br', 1)]):
            base_leg_angle = root_angle - np.pi/2 + sign * 0.15  # slight splay
            upper_a = p[f'leg_{side}_upper']
            lower_a = p[f'leg_{side}_lower']
            # Offset attachment point slightly to sides
            attach = pelvis + rot2d(root_angle) @ np.array([0, sign * 0.06])
            leg_pos = forward_kinematics(
                attach, base_leg_angle, [upper_a, lower_a], self.LEG_LENGTHS
            )
            chains[f'leg_{side}'] = leg_pos

        # Front legs attach at shoulders
        for i, (side, sign) in enumerate([('fl', -1), ('fr', 1)]):
            base_leg_angle = shoulder_angle - np.pi/2 + sign * 0.15
            upper_a = p[f'leg_{side}_upper']
            lower_a = p[f'leg_{side}_lower']
            attach = shoulders + rot2d(shoulder_angle) @ np.array([0, sign * 0.06])
            leg_pos = forward_kinematics(
                attach, base_leg_angle, [upper_a, lower_a], self.LEG_LENGTHS
            )
            chains[f'leg_{side}'] = leg_pos

        # ── Level 5: Head ──
        head_pos = forward_kinematics(
            shoulders, shoulder_angle + p['head_pan'],
            [p['head_tilt']], self.HEAD_LENGTHS
        )
        chains['head'] = head_pos

        # Tail attaches at pelvis, going backward
        tail_base_angle = root_angle + np.pi  # opposite to body direction
        tail_pos = forward_kinematics(
            pelvis, tail_base_angle,
            [p['tail_0'], p['tail_1']], self.TAIL_LENGTHS
        )
        chains['tail'] = tail_pos

        return chains

    def render(self, img_size: int = 128) -> Image.Image:
        """Render the cat to a PIL RGB image using current self.params.

        Applies camera (Level 1), skeleton from pose params (2–5), appearance (Level 6),
        and background (Level 7).

        Args:
            img_size: Output image width and height in pixels.

        Returns:
            PIL.Image in mode 'RGB', size (img_size, img_size).
        """
        p = self.params

        # Compute skeleton in world coordinates
        chains = self.compute_skeleton()

        # ── Level 1: Camera transform ──
        # Map world coords to pixel coords
        def world_to_pixel(pt: np.ndarray) -> Tuple[float, float]:
            # Apply camera SE(2) + scale
            scaled = p['cam_scale'] * pt
            rotated = rot2d(p['cam_angle']) @ scaled
            translated = rotated + np.array([p['cam_tx'], p['cam_ty']])
            # Map to pixel: centre of image, y-flip
            px = img_size/2 + translated[0] * img_size * 0.35
            py = img_size/2 - translated[1] * img_size * 0.35
            return (px, py)

        # ── Level 7: Background ──
        img = Image.new('RGB', (img_size, img_size))
        bg_pixels = img.load()
        bg_angle = p['bg_angle']
        bg_shift = p['bg_colour_shift']
        bg_int = p['bg_intensity']

        for y in range(img_size):
            for x in range(img_size):
                # Gradient along bg_angle direction
                nx = (x / img_size - 0.5)
                ny = (y / img_size - 0.5)
                grad_val = nx * np.cos(bg_angle) + ny * np.sin(bg_angle)
                grad_val = 0.5 + 0.5 * grad_val  # [0, 1]
                # Background colour
                r = int(255 * bg_int * (0.6 + 0.4 * bg_shift * grad_val))
                g = int(255 * bg_int * (0.7 + 0.3 * (1-bg_shift) * grad_val))
                b = int(255 * bg_int * (0.75 + 0.25 * bg_shift * (1-grad_val)))
                bg_pixels[x, y] = (
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b))
                )

        draw = ImageDraw.Draw(img)

        # ── Level 6: Appearance parameters ──
        body_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val']
        ))
        dark_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val'] * 0.6
        ))
        base_width = max(1, int(4 * p['limb_thickness'] * img_size / 128))
        body_width = max(1, int(6 * p['limb_thickness'] * img_size / 128))

        # Draw order: tail, legs, spine, head (back to front)

        # Tail
        tail_pixels = [world_to_pixel(pt) for pt in chains['tail']]
        if len(tail_pixels) >= 2:
            draw.line(tail_pixels, fill=body_rgb,
                     width=max(1, base_width - 1))

        # Legs
        for key in ['leg_bl', 'leg_br', 'leg_fl', 'leg_fr']:
            leg_pixels = [world_to_pixel(pt) for pt in chains[key]]
            if len(leg_pixels) >= 2:
                draw.line(leg_pixels, fill=dark_rgb, width=base_width)
                # Paw (small circle at end)
                ex, ey = leg_pixels[-1]
                paw_r = max(1, base_width * 0.6)
                draw.ellipse([ex-paw_r, ey-paw_r, ex+paw_r, ey+paw_r],
                           fill=dark_rgb)

        # Spine (body)
        spine_pixels = [world_to_pixel(pt) for pt in chains['spine']]
        if len(spine_pixels) >= 2:
            draw.line(spine_pixels, fill=body_rgb, width=body_width)

        # Body ellipses along spine for volume
        for i in range(len(spine_pixels) - 1):
            x0, y0 = spine_pixels[i]
            x1, y1 = spine_pixels[i+1]
            cx, cy = (x0+x1)/2, (y0+y1)/2
            r = body_width * 0.8
            draw.ellipse([cx-r, cy-r*0.7, cx+r, cy+r*0.7], fill=body_rgb)

        # Stripes (Level 6)
        if p['stripe_intensity'] > 0.05:
            stripe_rgb = tuple(
                max(0, int(c * (1 - 0.4 * p['stripe_intensity'])))
                for c in body_rgb
            )
            for i in range(len(spine_pixels) - 1):
                x0, y0 = spine_pixels[i]
                x1, y1 = spine_pixels[i+1]
                # Draw 2 stripes per segment
                for t in [0.3, 0.7]:
                    sx = x0 + t*(x1-x0)
                    sy = y0 + t*(y1-y0)
                    # Perpendicular stripe
                    dx, dy = x1-x0, y1-y0
                    norm = max(1e-6, np.sqrt(dx*dx + dy*dy))
                    nx, ny = -dy/norm, dx/norm
                    sr = body_width * 0.6
                    draw.line(
                        [(sx - nx*sr, sy - ny*sr), (sx + nx*sr, sy + ny*sr)],
                        fill=stripe_rgb,
                        width=max(1, base_width // 2)
                    )

        # Head
        head_pixels = [world_to_pixel(pt) for pt in chains['head']]
        if len(head_pixels) >= 2:
            # Neck
            draw.line(head_pixels[:2], fill=body_rgb, width=base_width)
            # Head circle
            hx, hy = head_pixels[-1]
            head_r = body_width * 1.0
            draw.ellipse(
                [hx-head_r, hy-head_r, hx+head_r, hy+head_r],
                fill=body_rgb
            )

            # Ears (triangles)
            ear_size = head_r * 0.7
            for sign in [-1, 1]:
                # Compute head direction for ear placement
                dx = head_pixels[-1][0] - head_pixels[-2][0]
                dy = head_pixels[-1][1] - head_pixels[-2][1]
                norm = max(1e-6, np.sqrt(dx*dx + dy*dy))
                # Perpendicular
                nx, ny = -dy/norm * sign, dx/norm * sign
                # Forward
                fx, fy = dx/norm, dy/norm
                ear_base_x = hx + nx * head_r * 0.5
                ear_base_y = hy + ny * head_r * 0.5
                ear_tip_x = ear_base_x + nx * ear_size + fx * ear_size * 0.5
                ear_tip_y = ear_base_y + ny * ear_size + fy * ear_size * 0.5
                ear_pts = [
                    (ear_base_x - fx*ear_size*0.3, ear_base_y - fy*ear_size*0.3),
                    (ear_tip_x, ear_tip_y),
                    (ear_base_x + fx*ear_size*0.3, ear_base_y + fy*ear_size*0.3),
                ]
                draw.polygon(ear_pts, fill=body_rgb)

            # Eyes
            eye_r = max(1, int(2.5 * p['eye_size'] * img_size / 128))
            dx = head_pixels[-1][0] - head_pixels[-2][0]
            dy = head_pixels[-1][1] - head_pixels[-2][1]
            norm = max(1e-6, np.sqrt(dx*dx + dy*dy))
            nx, ny = -dy/norm, dx/norm
            fx, fy = dx/norm, dy/norm
            for sign in [-1, 1]:
                ex = hx + sign * nx * head_r * 0.35 + fx * head_r * 0.2
                ey = hy + sign * ny * head_r * 0.35 + fy * head_r * 0.2
                draw.ellipse(
                    [ex-eye_r, ey-eye_r, ex+eye_r, ey+eye_r],
                    fill=(50, 180, 50)
                )
                # Pupil
                pr = max(1, eye_r // 2)
                draw.ellipse(
                    [ex-pr, ey-pr*1.5, ex+pr, ey+pr*1.5],
                    fill=(20, 20, 20)
                )

            # Nose
            nose_x = hx + fx * head_r * 0.55
            nose_y = hy + fy * head_r * 0.55
            nr = max(1, int(1.5 * img_size / 128))
            draw.ellipse(
                [nose_x-nr, nose_y-nr*0.7, nose_x+nr, nose_y+nr*0.7],
                fill=(200, 120, 120)
            )

        return img


# ═══════════════════════════════════════════════════════════
# Dataset generator with controlled conditions
# ═══════════════════════════════════════════════════════════
#
# Hierarchy (7 levels): each level corresponds to a subgroup of the generative
# parameters. LEVEL_PARAMS defines (param_name, (lo, hi)) per level; CONDITIONS
# select which levels are active for a given dataset (e.g. FullPose = levels 1–4).

# Parameter ranges for each level (level_id -> {param_name: (low, high)})
LEVEL_PARAMS = {
    1: {  # Camera SE(2) x R+
        'cam_angle':  (-0.3, 0.3),
        'cam_tx':     (-0.4, 0.4),
        'cam_ty':     (-0.4, 0.4),
        'cam_scale':  (0.7, 1.4),
    },
    2: {  # Root body SE(2)
        'root_x':     (-0.5, 0.5),
        'root_y':     (-0.3, 0.3),
        'root_angle': (-0.5, 0.5),
    },
    3: {  # Spine SO(2)^3
        'spine_0': (-0.4, 0.4),
        'spine_1': (-0.3, 0.3),
        'spine_2': (-0.3, 0.3),
    },
    4: {  # Limbs SO(2)^8
        'leg_bl_upper': (-0.6, 0.6),
        'leg_bl_lower': (-0.8, 0.1),
        'leg_br_upper': (-0.6, 0.6),
        'leg_br_lower': (-0.8, 0.1),
        'leg_fl_upper': (-0.6, 0.6),
        'leg_fl_lower': (-0.8, 0.1),
        'leg_fr_upper': (-0.6, 0.6),
        'leg_fr_lower': (-0.8, 0.1),
    },
    5: {  # Head & tail
        'head_pan':  (-0.5, 0.5),
        'head_tilt': (-0.4, 0.4),
        'tail_0':    (-0.8, 0.8),
        'tail_1':    (-0.6, 0.6),
    },
    6: {  # Appearance
        'body_hue':         (0.02, 0.15),
        'body_sat':         (0.3, 0.9),
        'body_val':         (0.4, 0.9),
        'limb_thickness':   (0.6, 1.5),
        'eye_size':         (0.6, 1.5),
        'stripe_intensity': (0.0, 1.0),
    },
    7: {  # Background
        'bg_angle':        (0.0, 2*np.pi),
        'bg_colour_shift': (0.0, 1.0),
        'bg_intensity':    (0.5, 1.0),
    },
}

# Named conditions: each key maps to the list of active level ids (1–7)
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


def sample_params(
    active_levels: List[int],
    rng: np.random.RandomState,
) -> Dict[str, float]:
    """Sample parameters for the cat: active levels get random values, rest use defaults.

    Args:
        active_levels: List of level ids (1–7) to randomise; others stay at JointedCat defaults.
        rng: Random state for reproducibility.

    Returns:
        Dict of param_name -> float, suitable for JointedCat.params.
    """
    cat = JointedCat()
    params = dict(cat.params)  # start from defaults

    for level in active_levels:
        if level in LEVEL_PARAMS:
            for param_name, (lo, hi) in LEVEL_PARAMS[level].items():
                params[param_name] = rng.uniform(lo, hi)

    return params


def generate_dataset(
    condition: str,
    n_samples: int,
    img_size: int = 128,
    output_dir: str = 'dataset',
    seed: int = 42,
) -> None:
    """Generate a dataset for a given condition and save to disk.

    Args:
        condition: Key from CONDITIONS (e.g. 'FullPose', 'Everything').
        n_samples: Number of images to generate.
        img_size: Output image size (H=W).
        output_dir: Root directory; creates {output_dir}/{condition}/images/ and metadata.
        seed: Random seed for sampling parameters.

    Saves:
        - {output_dir}/{condition}/images/*.png — PNG images
        - {output_dir}/{condition}/metadata.jsonl — one JSON line per sample (params, etc.)

    Raises:
        ValueError: If condition is not in CONDITIONS.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}. "
                        f"Choose from: {list(CONDITIONS.keys())}")

    active_levels = CONDITIONS[condition]
    rng = np.random.RandomState(seed)

    img_dir = Path(output_dir) / condition / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(output_dir) / condition / 'metadata.jsonl'

    cat = JointedCat()

    with open(meta_path, 'w') as meta_file:
        for i in range(n_samples):
            # Sample parameters
            params = sample_params(active_levels, rng)

            # Set and render
            cat.params = params
            img = cat.render(img_size=img_size)

            # Save image
            img_name = f'{i:06d}.png'
            img.save(img_dir / img_name)

            # Save metadata
            meta = {
                'index': i,
                'condition': condition,
                'active_levels': active_levels,
                'n_active_levels': len(active_levels),
                'params': {k: float(v) for k, v in params.items()},
                'image': img_name,
            }
            meta_file.write(json.dumps(meta) + '\n')

            if (i + 1) % 500 == 0:
                print(f'  [{condition}] {i+1}/{n_samples} generated')

    print(f'[{condition}] Done: {n_samples} images saved to {img_dir}')
    return img_dir, meta_path


# ═══════════════════════════════════════════════════════════
# Quick visualisation: sample grid
# ═══════════════════════════════════════════════════════════

def make_sample_grid(
    conditions: Optional[List[str]] = None,
    n_per_condition: int = 5,
    img_size: int = 128,
    seed: int = 42,
) -> Image.Image:
    """
    Create a grid of sample images: rows = conditions, columns = samples.
    Useful for visual sanity check.
    """
    if conditions is None:
        conditions = list(CONDITIONS.keys())

    rng = np.random.RandomState(seed)
    n_rows = len(conditions)
    n_cols = n_per_condition

    grid = Image.new('RGB', (n_cols * img_size, n_rows * img_size), (255, 255, 255))
    cat = JointedCat()

    for row, cond in enumerate(conditions):
        active = CONDITIONS[cond]
        for col in range(n_cols):
            params = sample_params(active, rng)
            cat.params = params
            img = cat.render(img_size=img_size)
            grid.paste(img, (col * img_size, row * img_size))

    return grid


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compositional Jointed-Cat Dataset Generator'
    )
    parser.add_argument('--mode', choices=['generate', 'grid', 'all'],
                       default='grid',
                       help='Mode: generate a dataset, make a sample grid, '
                            'or generate all conditions')
    parser.add_argument('--condition', type=str, default='Everything',
                       help=f'Condition name. Options: {list(CONDITIONS.keys())}')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples per condition')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (square)')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.mode == 'grid':
        print('Generating sample grid...')
        grid = make_sample_grid(img_size=args.img_size, seed=args.seed)
        grid.save(os.path.join(args.output_dir, 'sample_grid.png'))
        print(f'Saved sample grid to {args.output_dir}/sample_grid.png')

    elif args.mode == 'generate':
        generate_dataset(
            condition=args.condition,
            n_samples=args.n_samples,
            img_size=args.img_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    elif args.mode == 'all':
        for cond in CONDITIONS:
            generate_dataset(
                condition=cond,
                n_samples=args.n_samples,
                img_size=args.img_size,
                output_dir=args.output_dir,
                seed=args.seed,
            )
