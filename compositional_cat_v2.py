"""
Compositional Jointed-Cat Generator — v3 (improved realism)
=============================================================
Hierarchical generative model for 2D articulated cats.

Each level of the hierarchy corresponds to a Lie (pseudo)group action
on the image-generating process, forming a compositional flag:
  G = H_0  ⊃  H_1  ⊃  ...  ⊃  H_5

Generative story: "pose the cat, dress it up, place it, photograph it"

  Level 0 (Static)     : Identity — one fixed cat image
  Level 1 (Pose)       : SO(1)^15    — spine, limbs, head, tail joints
  Level 2 (Appearance) : R^6         — color, thickness, stripes, eyes
  Level 3 (Placement)  : SE(2)       — cat position & rotation in scene
  Level 4 (Camera)     : SE(2) × R+  — observer zoom, pan, rotation
  Level 5 (Background) : R^3         — gradient direction/color/intensity

v3 improvements:
  - Eyes: pupils are vertical slits relative to HEAD direction (not image)
  - Body: smooth polygon hull along spine (not stacked circles)
  - Ears: pointier triangles with inner ear color
  - Whiskers on the face
  - Tapered tail (thick→thin)
  - Rounder paws with visible toes
  - Better proportions (shoulders wider, hips narrower)

Author: G. Ruffini / Technical Note companion code — v3
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
    """2D articulated cat — v3 with improved proportions."""

    SPINE_LENGTHS = [0.30, 0.30, 0.25]
    LEG_LENGTHS = [0.25, 0.22]
    HEAD_LENGTHS = [0.20, 0.20]  # longer neck so head pan/tilt is visible
    TAIL_LENGTHS = [0.22, 0.18]

    def __init__(self):
        self.params = {}
        self.set_defaults()

    def set_defaults(self):
        # Level 1: Pose — spine, limbs, head, tail (intrinsic articulation)
        self.params['spine_0'] = 0.0
        self.params['spine_1'] = 0.0
        self.params['spine_2'] = 0.0
        for side in ['bl', 'br', 'fl', 'fr']:
            self.params[f'leg_{side}_upper'] = 0.0
            self.params[f'leg_{side}_lower'] = 0.0
        self.params['head_pan'] = 0.0
        self.params['head_tilt'] = 0.0
        self.params['tail_0'] = 0.0
        self.params['tail_1'] = 0.0

        # Level 2: Appearance — surface properties
        self.params['body_hue'] = 0.08
        self.params['body_sat'] = 0.6
        self.params['body_val'] = 0.7
        self.params['limb_thickness'] = 1.0
        self.params['eye_size'] = 1.0
        self.params['stripe_intensity'] = 0.0

        # Level 3: Placement — cat position & rotation in scene (SE(2))
        self.params['root_x'] = 0.0
        self.params['root_y'] = 0.0
        self.params['root_angle'] = 0.0

        # Level 4: Camera — observation transform (SE(2) × R+)
        self.params['cam_angle'] = 0.0
        self.params['cam_tx'] = 0.0
        self.params['cam_ty'] = 0.0
        self.params['cam_scale'] = 1.0

        # Level 5: Background — environment
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

    def _make_body_hull(self, spine_px, widths):
        """Create smooth body outline polygon from spine points and per-segment widths."""
        if len(spine_px) < 2:
            return []

        # Build left and right contour along spine
        left_pts, right_pts = [], []
        for i in range(len(spine_px)):
            if i < len(spine_px) - 1:
                dx = spine_px[i+1][0] - spine_px[i][0]
                dy = spine_px[i+1][1] - spine_px[i][1]
            else:
                dx = spine_px[i][0] - spine_px[i-1][0]
                dy = spine_px[i][1] - spine_px[i-1][1]
            nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
            # Normal perpendicular to spine direction
            nx, ny = -dy/nm, dx/nm
            w = widths[min(i, len(widths)-1)]
            left_pts.append((spine_px[i][0] + nx*w, spine_px[i][1] + ny*w))
            right_pts.append((spine_px[i][0] - nx*w, spine_px[i][1] - ny*w))

        # Close the hull: left side forward, right side backward
        hull = left_pts + list(reversed(right_pts))
        return hull

    def render(self, img_size: int = 128) -> Image.Image:
        p = self.params
        chains = self.compute_skeleton()

        WORLD_SCALE = 0.40

        def world_to_pixel(pt: np.ndarray) -> Tuple[float, float]:
            scaled = p['cam_scale'] * pt
            rotated = rot2d(p['cam_angle']) @ scaled
            translated = rotated + np.array([p['cam_tx'], p['cam_ty']])
            px = img_size/2 + translated[0] * img_size * WORLD_SCALE
            py = img_size/2 - translated[1] * img_size * WORLD_SCALE
            return (px, py)

        # ── Background (vectorised) ──
        ys, xs = np.mgrid[0:img_size, 0:img_size]
        ux = xs / img_size - 0.5
        uy = ys / img_size - 0.5
        grad = ux * np.cos(p['bg_angle']) + uy * np.sin(p['bg_angle'])
        grad = 0.5 + 0.5 * grad
        bi = p['bg_intensity']
        bs = p['bg_colour_shift']
        r_ch = (255 * bi * (0.6 + 0.4 * bs * grad)).clip(0, 255).astype(np.uint8)
        g_ch = (255 * bi * (0.7 + 0.3 * (1 - bs) * grad)).clip(0, 255).astype(np.uint8)
        b_ch = (255 * bi * (0.75 + 0.25 * bs * (1 - grad))).clip(0, 255).astype(np.uint8)
        bg_arr = np.stack([r_ch, g_ch, b_ch], axis=-1)
        img = Image.fromarray(bg_arr, 'RGB')
        draw = ImageDraw.Draw(img)

        # ── Colours ──
        body_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val']
        ))
        dark_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'], p['body_val'] * 0.55
        ))
        belly_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            p['body_hue'], p['body_sat'] * 0.4, min(1.0, p['body_val'] * 1.2)
        ))
        inner_ear_rgb = tuple(int(255*c) for c in colorsys.hsv_to_rgb(
            0.0, 0.35, min(1.0, p['body_val'] * 1.1)
        ))

        # ── Widths (scale with image and thickness param) ──
        sc = img_size / 128.0
        thick = p['limb_thickness']
        leg_w = max(2, int(5.5 * thick * sc))
        # Body widths along spine: hips → mid-body → shoulders (shoulders wider)
        body_widths = [
            max(3, int(7.5 * thick * sc)),   # pelvis
            max(4, int(10.0 * thick * sc)),   # mid
            max(4, int(10.5 * thick * sc)),   # mid-front
            max(4, int(9.0 * thick * sc)),    # shoulders
        ]

        # ── 1. Tapered tail ──
        tail_px = [world_to_pixel(pt) for pt in chains['tail']]
        if len(tail_px) >= 2:
            for i in range(len(tail_px) - 1):
                t = i / max(1, len(tail_px) - 1)
                w = max(1, int(leg_w * (1.0 - 0.6 * t)))  # thick→thin
                draw.line([tail_px[i], tail_px[i+1]], fill=body_rgb, width=w)
            # Tail tip
            tx, ty = tail_px[-1]
            tr = max(1, leg_w * 0.3)
            draw.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=body_rgb)

        # ── 2. Back legs (behind body) ──
        for key in ['leg_bl', 'leg_br']:
            leg_px = [world_to_pixel(pt) for pt in chains[key]]
            if len(leg_px) >= 2:
                # Upper leg
                draw.line([leg_px[0], leg_px[1]], fill=dark_rgb, width=leg_w)
                if len(leg_px) >= 3:
                    # Lower leg
                    draw.line([leg_px[1], leg_px[2]], fill=dark_rgb, width=max(2, leg_w - 1))
                    # Paw (oval)
                    fx2, fy2 = leg_px[-1]
                    pw = max(2, leg_w * 0.8)
                    ph = max(2, leg_w * 0.5)
                    draw.ellipse([fx2-pw, fy2-ph, fx2+pw, fy2+ph], fill=dark_rgb)

        # ── 3. Body hull (smooth polygon) ──
        spine_px = [world_to_pixel(pt) for pt in chains['spine']]
        if len(spine_px) >= 2:
            hull = self._make_body_hull(spine_px, body_widths)
            if len(hull) >= 3:
                draw.polygon(hull, fill=body_rgb)

            # Belly highlight (lighter strip along bottom of body)
            belly_widths = [w * 0.5 for w in body_widths]
            belly_hull = self._make_body_hull(spine_px, belly_widths)
            if len(belly_hull) >= 3:
                # Offset downward slightly
                offset_hull = [(x, y + body_widths[0] * 0.25) for x, y in belly_hull]
                draw.polygon(offset_hull, fill=belly_rgb)

        # ── 4. Stripes on body ──
        if p['stripe_intensity'] > 0.05:
            stripe_rgb = tuple(max(0, int(c * (1 - 0.5 * p['stripe_intensity']))) for c in body_rgb)
            for i in range(len(spine_px) - 1):
                x0, y0 = spine_px[i]
                x1, y1 = spine_px[i+1]
                dx, dy = x1 - x0, y1 - y0
                nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
                nx2, ny2 = -dy/nm, dx/nm
                w = body_widths[min(i, len(body_widths)-1)]
                for t in [0.25, 0.5, 0.75]:
                    sx = x0 + t * (x1 - x0)
                    sy = y0 + t * (y1 - y0)
                    sw = max(1, int(leg_w * 0.4))
                    draw.line([(sx - nx2*w*0.9, sy - ny2*w*0.9),
                               (sx + nx2*w*0.9, sy + ny2*w*0.9)],
                              fill=stripe_rgb, width=sw)

        # ── 5. Front legs (over body) ──
        for key in ['leg_fl', 'leg_fr']:
            leg_px = [world_to_pixel(pt) for pt in chains[key]]
            if len(leg_px) >= 2:
                draw.line([leg_px[0], leg_px[1]], fill=dark_rgb, width=leg_w)
                if len(leg_px) >= 3:
                    draw.line([leg_px[1], leg_px[2]], fill=dark_rgb, width=max(2, leg_w - 1))
                    fx2, fy2 = leg_px[-1]
                    pw = max(2, leg_w * 0.8)
                    ph = max(2, leg_w * 0.5)
                    draw.ellipse([fx2-pw, fy2-ph, fx2+pw, fy2+ph], fill=dark_rgb)

        # ── 6. Head ──
        head_px = [world_to_pixel(pt) for pt in chains['head']]
        if len(head_px) >= 2:
            # Neck
            neck_w = max(2, int(6 * thick * sc))
            draw.line(head_px[:2], fill=body_rgb, width=neck_w)

            hx, hy = head_px[-1]
            hr = max(4, int(10 * thick * sc))

            # Head direction vector
            dx = head_px[-1][0] - head_px[-2][0]
            dy = head_px[-1][1] - head_px[-2][1]
            nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
            fx, fy = dx/nm, dy/nm       # forward direction
            rx, ry = -fy, fx             # right direction (perpendicular)

            # ── Ears (pointy triangles with inner ear) ──
            es = hr * 0.85
            for sign in [-1, 1]:
                # Ear base on side of head
                bx = hx + sign * rx * hr * 0.55
                by = hy + sign * ry * hr * 0.55
                # Ear tip: outward + forward
                tx = bx + sign * rx * es * 0.7 + fx * es * 0.6
                ty = by + sign * ry * es * 0.7 + fy * es * 0.6
                # Outer ear
                draw.polygon([
                    (bx - fx * es * 0.35, by - fy * es * 0.35),
                    (tx, ty),
                    (bx + fx * es * 0.25, by + fy * es * 0.25),
                ], fill=body_rgb)
                # Inner ear (smaller, pink)
                ibx = bx + sign * rx * es * 0.08
                iby = by + sign * ry * es * 0.08
                itx = ibx + sign * rx * es * 0.45 + fx * es * 0.38
                ity = iby + sign * ry * es * 0.45 + fy * es * 0.38
                draw.polygon([
                    (ibx - fx * es * 0.2, iby - fy * es * 0.2),
                    (itx, ity),
                    (ibx + fx * es * 0.12, iby + fy * es * 0.12),
                ], fill=inner_ear_rgb)

            # Head circle (drawn after ears so it covers ear bases)
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=body_rgb)

            # ── Eyes (pupils oriented relative to head, not image) ──
            eye_r = max(2, int(3.2 * p['eye_size'] * sc))
            for sign in [-1, 1]:
                # Eye position: offset sideways + slightly forward
                ex = hx + sign * rx * hr * 0.38 + fx * hr * 0.22
                ey = hy + sign * ry * hr * 0.38 + fy * hr * 0.22
                # White of eye (slightly larger)
                wr = eye_r * 1.15
                draw.ellipse([ex - wr, ey - wr, ex + wr, ey + wr], fill=(240, 240, 235))
                # Iris (green/amber)
                draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r],
                             fill=(80, 180, 60))
                # Pupil: vertical slit RELATIVE TO HEAD DIRECTION
                # The "up" axis of the head is the rx,ry direction
                # Pupil is elongated along this axis
                pr = max(1, eye_r * 0.35)
                ph = max(1, eye_r * 0.85)
                # Draw as a polygon (4 points of a narrow diamond along head's lateral axis)
                pupil_pts = [
                    (ex + rx * ph, ey + ry * ph),  # top of slit (head-relative)
                    (ex + fx * pr, ey + fy * pr),   # right of slit
                    (ex - rx * ph, ey - ry * ph),  # bottom of slit
                    (ex - fx * pr, ey - fy * pr),   # left of slit
                ]
                draw.polygon(pupil_pts, fill=(15, 15, 15))
                # Specular highlight
                spr = max(1, eye_r * 0.25)
                sx = ex + fx * eye_r * 0.2 + rx * eye_r * 0.15
                sy = ey + fy * eye_r * 0.2 + ry * eye_r * 0.15
                draw.ellipse([sx - spr, sy - spr, sx + spr, sy + spr],
                             fill=(255, 255, 255))

            # ── Nose (triangular, pink) ──
            nose_x = hx + fx * hr * 0.6
            nose_y = hy + fy * hr * 0.6
            nr = max(2, int(2.5 * sc))
            draw.polygon([
                (nose_x + fx * nr * 0.3, nose_y + fy * nr * 0.3),
                (nose_x + rx * nr, nose_y + ry * nr),
                (nose_x - rx * nr, nose_y - ry * nr),
            ], fill=(200, 130, 130))

            # ── Mouth (small lines below nose) ──
            mw = max(1, int(1 * sc))
            mx = nose_x + fx * nr * 0.8
            my = nose_y + fy * nr * 0.8
            draw.line([(mx, my), (mx + rx * nr * 1.2 + fx * nr * 0.5,
                                  my + ry * nr * 1.2 + fy * nr * 0.5)],
                      fill=(160, 100, 100), width=mw)
            draw.line([(mx, my), (mx - rx * nr * 1.2 + fx * nr * 0.5,
                                  my - ry * nr * 1.2 + fy * nr * 0.5)],
                      fill=(160, 100, 100), width=mw)

            # ── Whiskers ──
            ww = max(1, int(1 * sc))
            whisker_base_r = hr * 0.5
            whisker_len = hr * 1.5
            for sign in [-1, 1]:
                # 3 whiskers per side
                for angle_off in [-0.2, 0.0, 0.2]:
                    wb_x = hx + fx * whisker_base_r * 0.8 + sign * rx * whisker_base_r * 0.5
                    wb_y = hy + fy * whisker_base_r * 0.8 + sign * ry * whisker_base_r * 0.5
                    # Whisker direction: mostly sideways + slight forward/angle offset
                    wd_x = sign * rx + fx * angle_off
                    wd_y = sign * ry + fy * angle_off
                    wnm = max(1e-6, np.sqrt(wd_x**2 + wd_y**2))
                    wd_x, wd_y = wd_x / wnm, wd_y / wnm
                    we_x = wb_x + wd_x * whisker_len
                    we_y = wb_y + wd_y * whisker_len
                    draw.line([(wb_x, wb_y), (we_x, we_y)],
                              fill=(180, 180, 180), width=ww)

        return img


# ═══════════════════════════════════════════════════════════
# Hierarchy: build the cat → place it → observe it
#
#   Level 1: POSE        — intrinsic articulation (spine + limbs + head/tail)
#   Level 2: APPEARANCE  — surface properties (color, stripes, proportions)
#   Level 3: PLACEMENT   — cat position & rotation in the scene (SE(2))
#   Level 4: CAMERA      — observer transform: zoom, pan, rotation (SE(2)×R+)
#   Level 5: BACKGROUND  — environment (gradient, colour, intensity)
#
# Generative story: "pose the cat, dress it up, put it somewhere,
#                     point the camera, choose the backdrop"
# ═══════════════════════════════════════════════════════════

LEVEL_PARAMS = {
    1: {  # POSE — all joint angles (15 params)
        'spine_0':       (-0.7, 0.7),
        'spine_1':       (-0.6, 0.6),
        'spine_2':       (-0.5, 0.5),
        'leg_bl_upper':  (-1.0, 1.0),
        'leg_bl_lower':  (-1.2, 0.2),
        'leg_br_upper':  (-1.0, 1.0),
        'leg_br_lower':  (-1.2, 0.2),
        'leg_fl_upper':  (-1.0, 1.0),
        'leg_fl_lower':  (-1.2, 0.2),
        'leg_fr_upper':  (-1.0, 1.0),
        'leg_fr_lower':  (-1.2, 0.2),
        'head_pan':      (-0.8, 0.8),
        'head_tilt':     (-0.7, 0.7),
        'tail_0':        (-1.2, 1.2),
        'tail_1':        (-1.0, 1.0),
    },
    2: {  # APPEARANCE — surface properties (6 params)
        'body_hue':         (0.0, 0.18),
        'body_sat':         (0.2, 1.0),
        'body_val':         (0.35, 0.95),
        'limb_thickness':   (0.5, 1.8),
        'eye_size':         (0.5, 1.8),
        'stripe_intensity': (0.0, 1.0),
    },
    3: {  # PLACEMENT — cat in scene: SE(2) (3 params)
        'root_x':     (-0.4, 0.4),
        'root_y':     (-0.3, 0.3),
        'root_angle': (-0.8, 0.8),
    },
    4: {  # CAMERA — observation: SE(2)×R+ (4 params)
        'cam_angle':  (-0.4, 0.4),
        'cam_tx':     (-0.25, 0.25),
        'cam_ty':     (-0.25, 0.25),
        'cam_scale':  (0.7, 1.3),
    },
    5: {  # BACKGROUND — environment (3 params)
        'bg_angle':        (0.0, 2*np.pi),
        'bg_colour_shift': (0.0, 1.0),
        'bg_intensity':    (0.4, 1.0),
    },
}

CONDITIONS = {
    'Static':           [],           # L0: one fixed cat
    'PoseOnly':         [1],          # L1: vary articulation
    'PoseAppearance':   [1, 2],       # L1+2: + colour/stripes
    'PosAppPlace':      [1, 2, 3],    # L1-3: + cat position/rotation
    'PosAppPlaceCam':   [1, 2, 3, 4], # L1-4: + camera transform
    'Everything':       [1, 2, 3, 4, 5],  # L1-5: + background
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
