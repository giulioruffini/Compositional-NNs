"""
Compositional Jointed-Cat Generator — v6 (enhanced realism)
=============================================================
Hierarchical generative model for 2D articulated cats.

Each level of the hierarchy corresponds to a Lie (pseudo)group action
on the image-generating process, forming a compositional flag:
  G = H_0  ⊃  H_1  ⊃  ...  ⊃  H_5

Generative story: "pose the cat, dress it up, place it, photograph it"

  Level 0 (Static)     : Identity — one fixed cat image
  Level 1 (Pose)       : SO(1)^16    — spine, limbs, head (pan/tilt/roll), tail
  Level 2 (Appearance) : R^6         — color, thickness, stripes, eyes
  Level 3 (Placement)  : R^2 × SO(3) — position + full 3D rotation (5 params)
  Level 4 (Camera)     : SE(2) × R+  — observer zoom, pan, rotation
  Level 5 (Background) : R^1         — uniform greyscale intensity

v6 improvements (over v5):
  - 2× supersampling anti-aliasing (render at 2× then LANCZOS downsample)
  - Drop shadow under paws (grounding the cat in the scene)
  - Thin body outline for definition against similar backgrounds
  - Smooth Bézier-interpolated tail (cubic spline through FK joints)
  - Paw toe details (small dark lines)
  - Depth-ordered limb rendering (near/far legs from root_angle)
  - 3D sphere-projected eyes with per-eye visibility and foreshortening

Author: G. Ruffini / Technical Note companion code — v6
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

def _cubic_interpolate(points, n_interp=12):
    """Cubic spline interpolation through a list of 2D points.

    Returns a denser list of (x, y) tuples for smooth curves.
    Falls back to linear if scipy unavailable or < 3 points.
    """
    if len(points) < 3:
        return points
    try:
        from scipy.interpolate import CubicSpline
        pts = np.array(points)
        t = np.linspace(0, 1, len(pts))
        t_fine = np.linspace(0, 1, n_interp)
        cs_x = CubicSpline(t, pts[:, 0])
        cs_y = CubicSpline(t, pts[:, 1])
        return [(float(cs_x(ti)), float(cs_y(ti))) for ti in t_fine]
    except ImportError:
        # Fallback: linear interpolation with more points
        result = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            steps = max(2, n_interp // (len(points) - 1))
            for j in range(steps):
                frac = j / steps
                result.append((x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)))
        result.append(points[-1])
        return result


class JointedCat:
    """2D articulated cat — v6 with anti-aliasing, shadows, outlines."""

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
        self.params['head_roll'] = 0.0   # tilts face around neck axis
        self.params['tail_0'] = 0.0
        self.params['tail_1'] = 0.0

        # Level 2: Appearance — surface properties
        self.params['body_hue'] = 0.08
        self.params['body_sat'] = 0.6
        self.params['body_val'] = 0.7
        self.params['limb_thickness'] = 1.0
        self.params['eye_size'] = 1.0
        self.params['stripe_intensity'] = 0.0

        # Level 3: Placement — position + SO(3) rotation
        self.params['root_x'] = 0.0
        self.params['root_y'] = 0.0
        self.params['root_angle'] = 0.0       # yaw (in-plane)
        self.params['root_elevation'] = 0.0   # pitch (view from above/below)
        self.params['root_roll'] = 0.0        # roll (cat tilts sideways)

        # Level 4: Camera — observation transform (SE(2) × R+)
        self.params['cam_angle'] = 0.0
        self.params['cam_tx'] = 0.0
        self.params['cam_ty'] = 0.0
        self.params['cam_scale'] = 1.0

        # Level 5: Background — uniform greyscale
        self.params['bg_grey'] = 0.75

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

    @staticmethod
    def _oriented_ellipse_pts(cx, cy, ux, uy, a, b, n_pts=24):
        """Return polygon points for an ellipse with oriented axes.

        (ux, uy) = unit vector for the semi-axis of length `a`.
        Perpendicular direction gets semi-axis length `b`.
        """
        vx, vy = -uy, ux  # perpendicular
        pts = []
        for i in range(n_pts):
            t = 2 * np.pi * i / n_pts
            ct, st = np.cos(t), np.sin(t)
            px = cx + a * ct * ux + b * st * vx
            py = cy + a * ct * uy + b * st * vy
            pts.append((px, py))
        return pts

    def render(self, img_size: int = 128, _aa_scale: int = 2) -> Image.Image:
        """Render the cat. Uses 2× supersampling AA by default."""
        if _aa_scale > 1:
            hi_res = self._render_internal(img_size * _aa_scale)
            return hi_res.resize((img_size, img_size), Image.LANCZOS)
        return self._render_internal(img_size)

    def _render_internal(self, img_size: int) -> Image.Image:
        p = self.params
        chains = self.compute_skeleton()

        WORLD_SCALE = 0.40

        # ── Pseudo-3D rotation (SO(3) foreshortening) ──
        # Embed 2D skeleton in 3D, apply elevation + roll rotations
        # around the root position, then orthographic-project back to 2D.
        root_2d = np.array([p['root_x'], p['root_y']])
        elev = p.get('root_elevation', 0.0)
        roll = p.get('root_roll', 0.0)
        cos_e, sin_e = np.cos(elev), np.sin(elev)
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        # Foreshortening scale for rendered sizes (geometric mean)
        foreshorten = np.sqrt(abs(cos_e * cos_r))

        def world_to_pixel(pt: np.ndarray) -> Tuple[float, float]:
            # 1. Translate to root-relative coords
            rel = pt - root_2d
            x, y = rel[0], rel[1]
            # 2. Embed in 3D: (x, y, 0)
            # 3. Apply Rx(elev): (x, y*cos_e, y*sin_e)
            # 4. Apply Ry(roll): (x*cos_r + y*sin_e*sin_r, y*cos_e, ...)
            # 5. Orthographic project: take (x', y')
            x2 = x * cos_r + y * sin_e * sin_r   # cross-term is key
            y2 = y * cos_e
            # Translate back
            world_pt = root_2d + np.array([x2, y2])
            # Camera transform
            scaled = p['cam_scale'] * world_pt
            rotated = rot2d(p['cam_angle']) @ scaled
            translated = rotated + np.array([p['cam_tx'], p['cam_ty']])
            px = img_size/2 + translated[0] * img_size * WORLD_SCALE
            py = img_size/2 - translated[1] * img_size * WORLD_SCALE
            return (px, py)

        # ── Background (uniform greyscale) ──
        grey_val = int(np.clip(p['bg_grey'] * 255, 0, 255))
        bg_arr = np.full((img_size, img_size, 3), grey_val, dtype=np.uint8)
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
        # Outline colour: slightly darker than body
        outline_rgb = tuple(max(0, int(c * 0.45)) for c in body_rgb)
        # Shadow colour: semi-transparent dark
        shadow_rgb = tuple(max(0, int(grey_val * 0.55)) for _ in range(3))

        # ── Widths (scale with image, thickness, and foreshortening) ──
        sc = img_size / 128.0
        thick = p['limb_thickness']
        fs = foreshorten  # sizes shrink with 3D rotation
        leg_w = max(2, int(5.5 * thick * sc * fs))
        # Body widths along spine: hips → mid-body → shoulders (shoulders wider)
        body_widths = [
            max(3, int(7.5 * thick * sc * fs)),   # pelvis
            max(4, int(10.0 * thick * sc * fs)),   # mid
            max(4, int(10.5 * thick * sc * fs)),   # mid-front
            max(4, int(9.0 * thick * sc * fs)),    # shoulders
        ]

        # ── 0. Drop shadow ──
        # Collect all paw (foot) positions and draw an elliptical shadow
        paw_positions = []
        for key in ['leg_bl', 'leg_br', 'leg_fl', 'leg_fr']:
            if key in chains and len(chains[key]) >= 3:
                paw_positions.append(world_to_pixel(chains[key][-1]))
        if paw_positions:
            paw_xs = [pp[0] for pp in paw_positions]
            paw_ys = [pp[1] for pp in paw_positions]
            # Shadow center: mean of paw x, bottom of paw y (+ offset)
            scx = sum(paw_xs) / len(paw_xs)
            scy = max(paw_ys) + leg_w * 0.4
            # Shadow width spans paws, height is thin
            sw = (max(paw_xs) - min(paw_xs)) * 0.45 + leg_w * 2.5
            sh = max(2, leg_w * 0.7)
            shadow_pts = self._oriented_ellipse_pts(scx, scy, 1, 0, sw, sh, n_pts=24)
            draw.polygon(shadow_pts, fill=shadow_rgb)

        # ── 1. Tapered tail (smooth Bézier curve) ──
        tail_px = [world_to_pixel(pt) for pt in chains['tail']]
        if len(tail_px) >= 2:
            # Interpolate through FK joints for a smooth curve
            smooth_tail = _cubic_interpolate(tail_px, n_interp=16)
            for i in range(len(smooth_tail) - 1):
                t = i / max(1, len(smooth_tail) - 1)
                w = max(1, int(leg_w * (1.0 - 0.7 * t)))  # thick→thin
                draw.line([smooth_tail[i], smooth_tail[i+1]], fill=body_rgb, width=w)
            # Tail tip (round cap)
            tx, ty = smooth_tail[-1]
            tr = max(1, leg_w * 0.25)
            draw.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=body_rgb)

        # ── Depth-ordered limb rendering ──
        # Determine which side is "far" (behind body) vs "near" (in front)
        # based on root_angle: when facing right (angle≈0), 'l' legs are far
        # Compute the perpendicular direction to root_angle
        root_angle = p['root_angle']
        # Perpendicular dot product: positive → 'l' legs are far side
        facing_sign = np.sin(root_angle + sum([p['spine_0'], p['spine_1'], p['spine_2']]) * 0.5)
        if facing_sign >= 0:
            far_legs = ['leg_bl', 'leg_br', 'leg_fl', 'leg_fr']
            near_legs = ['leg_br', 'leg_bl', 'leg_fr', 'leg_fl']
        else:
            far_legs = ['leg_br', 'leg_bl', 'leg_fr', 'leg_fl']
            near_legs = ['leg_bl', 'leg_br', 'leg_fl', 'leg_fr']
        # Far back/front legs drawn first, then body, then near legs
        back_far = [k for k in far_legs if k.startswith('leg_b')][:1]
        front_far = [k for k in far_legs if k.startswith('leg_f')][:1]
        back_near = [k for k in near_legs if k.startswith('leg_b')][:1]
        front_near = [k for k in near_legs if k.startswith('leg_f')][:1]

        def _draw_leg(key, color):
            leg_px = [world_to_pixel(pt) for pt in chains[key]]
            if len(leg_px) < 2:
                return
            # Upper leg
            draw.line([leg_px[0], leg_px[1]], fill=color, width=leg_w)
            if len(leg_px) >= 3:
                # Lower leg (slightly thinner)
                draw.line([leg_px[1], leg_px[2]], fill=color, width=max(2, leg_w - 1))
                # Knee joint circle
                kx, ky = leg_px[1]
                kr = max(1, leg_w * 0.45)
                draw.ellipse([kx-kr, ky-kr, kx+kr, ky+kr], fill=color)
                # Paw (oval)
                pawx, pawy = leg_px[-1]
                pw = max(2, leg_w * 0.8)
                ph_paw = max(2, leg_w * 0.5)
                draw.ellipse([pawx-pw, pawy-ph_paw, pawx+pw, pawy+ph_paw], fill=color)
                # Toe lines
                # Direction from knee to paw (for toe orientation)
                tdx = pawx - leg_px[1][0]
                tdy = pawy - leg_px[1][1]
                tnm = max(1e-6, np.sqrt(tdx**2 + tdy**2))
                tdx, tdy = tdx / tnm, tdy / tnm
                tnx, tny = -tdy, tdx   # perpendicular
                toe_w = max(1, int(sc * 0.8))
                toe_len = pw * 0.65
                for ts in [-0.4, 0.0, 0.4]:
                    tx = pawx + tnx * ts * pw
                    ty = pawy + tny * ts * pw
                    draw.line([(tx, ty), (tx + tdx * toe_len, ty + tdy * toe_len)],
                              fill=outline_rgb, width=toe_w)

        # ── 2. Far-side legs (behind body) ──
        for key in back_far:
            _draw_leg(key, dark_rgb)
        for key in front_far:
            _draw_leg(key, dark_rgb)

        # ── 3. Body hull (smooth polygon with outline) ──
        spine_px = [world_to_pixel(pt) for pt in chains['spine']]
        if len(spine_px) >= 2:
            hull = self._make_body_hull(spine_px, body_widths)
            if len(hull) >= 3:
                # Body outline (drawn slightly larger)
                outline_w = max(1, int(1.5 * sc))
                draw.polygon(hull, fill=body_rgb, outline=outline_rgb)

            # Belly highlight (lighter strip along bottom of body)
            belly_widths = [w * 0.5 for w in body_widths]
            belly_hull = self._make_body_hull(spine_px, belly_widths)
            if len(belly_hull) >= 3:
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

        # ── 5. Near-side legs (in front of body) ──
        # Slightly darker than body so they stand out, but lighter than far legs
        near_rgb = tuple(int(b * 0.78 + d * 0.22) for b, d in zip(body_rgb, dark_rgb))
        for key in back_near:
            _draw_leg(key, near_rgb)
        for key in front_near:
            _draw_leg(key, near_rgb)

        # ── 6. Head (ball-on-rod model) ──
        # The neck is a rod from shoulders to head center.
        # head_pan / head_tilt move the rod (FK chain).
        # head_roll rotates the ball around the rod axis:
        #   → lateral (ear-to-ear) offsets foreshorten by cos(head_roll)
        #   → forward (snout) offsets unchanged
        head_px = [world_to_pixel(pt) for pt in chains['head']]
        if len(head_px) >= 2:
            # Neck
            neck_w = max(2, int(6 * thick * sc))
            draw.line(head_px[:2], fill=body_rgb, width=neck_w)

            hx, hy = head_px[-1]
            hr = max(4, int(10 * thick * sc * fs))

            # Rod direction (from FK chain)
            dx = head_px[-1][0] - head_px[-2][0]
            dy = head_px[-1][1] - head_px[-2][1]
            nm = max(1e-6, np.sqrt(dx*dx + dy*dy))
            fx, fy = dx/nm, dy/nm       # rod/forward direction (snout)
            rx, ry = -fy, fx             # lateral direction (ear-to-ear)

            # Head roll: rotation around rod axis foreshortens lateral extent
            hroll = p.get('head_roll', 0.0)
            lat_scale = np.cos(hroll)  # <1 when rolled → face foreshortens

            # Helper: place a point in head-local coords (fwd, lat)
            # fwd = offset along rod, lat = offset perpendicular to rod
            def head_pt(fwd, lat):
                return (hx + fx * fwd + rx * lat * lat_scale,
                        hy + fy * fwd + ry * lat * lat_scale)

            # Head ellipse: wider ear-to-ear, narrower front-to-back
            head_a = hr * 1.15 * abs(lat_scale)  # lateral semi-axis foreshortens
            head_b = hr * 0.88                     # forward semi-axis stays

            # ── Ears (pointy triangles with inner ear) ──
            es = hr * 0.85
            for sign in [-1, 1]:
                bx, by = head_pt(0, sign * hr * 0.55)
                # Ear tip: outward + forward
                tx = bx + sign * rx * es * 0.7 * lat_scale + fx * es * 0.6
                ty = by + sign * ry * es * 0.7 * lat_scale + fy * es * 0.6
                draw.polygon([
                    (bx - fx * es * 0.35, by - fy * es * 0.35),
                    (tx, ty),
                    (bx + fx * es * 0.25, by + fy * es * 0.25),
                ], fill=body_rgb)
                # Inner ear
                ibx, iby = head_pt(0, sign * (hr * 0.55 + es * 0.08))
                itx = ibx + sign * rx * es * 0.45 * lat_scale + fx * es * 0.38
                ity = iby + sign * ry * es * 0.45 * lat_scale + fy * es * 0.38
                draw.polygon([
                    (ibx - fx * es * 0.2, iby - fy * es * 0.2),
                    (itx, ity),
                    (ibx + fx * es * 0.12, iby + fy * es * 0.12),
                ], fill=inner_ear_rgb)

            # Head ellipse (oriented — covers ear bases, with outline)
            head_pts = self._oriented_ellipse_pts(
                hx, hy, rx, ry, head_a, head_b, n_pts=32)
            draw.polygon(head_pts, fill=body_rgb, outline=outline_rgb)

            # ── Eyes (3D sphere projection) ──
            # The head is a 3D sphere. Eyes are placed at angular positions
            # on the surface. FACE_TILT angles the face toward the viewer
            # (z-axis = toward camera). Each eye's visibility = how much its
            # surface normal points toward the camera.
            #
            # 3D basis:  fwd=(fx,fy,0)  lat=(rx,ry,0)  dors=(0,0,1)
            # Face tilt rotates fwd toward dors by FACE_TILT radians.
            # head_roll rotates lat/dors around fwd.

            FACE_TILT = 0.35   # ~20° intrinsic tilt toward viewer
            ct_ft, st_ft = np.cos(FACE_TILT), np.sin(FACE_TILT)

            # Face basis after tilt (fwd tilts toward camera)
            # face_fwd = ct*fwd + st*dors
            # face_dors = -st*fwd + ct*dors
            # After head_roll ψ around fwd:
            #   face_lat  = cos(ψ)*lat + sin(ψ)*face_dors
            #   face_dors_r = -sin(ψ)*lat + cos(ψ)*face_dors
            cr_h, sr_h = np.cos(hroll), np.sin(hroll)

            # z-components of each basis vector (only z matters for visibility)
            face_fwd_z = st_ft                       # sin(FACE_TILT)
            face_lat_z = sr_h * (-st_ft * 0 + ct_ft) # sr_h * ct_ft (from face_dors_z)
            # Actually let me compute properly:
            # face_dors = (-st_ft*fx, -st_ft*fy, ct_ft)
            # face_lat = cr_h*(rx,ry,0) + sr_h*(-st_ft*fx, -st_ft*fy, ct_ft)
            # face_lat_z = sr_h * ct_ft
            face_lat_z = sr_h * ct_ft
            # face_dors_r = -sr_h*(rx,ry,0) + cr_h*(-st_ft*fx,-st_ft*fy,ct_ft)
            # face_dors_r_z = cr_h * ct_ft
            face_dors_r_z = cr_h * ct_ft

            # 2D components of face basis (for projecting positions)
            face_fwd_x = ct_ft * fx
            face_fwd_y = ct_ft * fy
            face_lat_x = cr_h * rx + sr_h * (-st_ft * fx)
            face_lat_y = cr_h * ry + sr_h * (-st_ft * fy)
            face_dors_x = -sr_h * rx + cr_h * (-st_ft * fx)
            face_dors_y = -sr_h * ry + cr_h * (-st_ft * fy)

            EYE_ALPHA = 0.40   # angular distance from face center
            EYE_BETA = 0.50    # lateral offset angle

            eye_r = max(2, int(3.2 * p['eye_size'] * sc))
            ca, sa = np.cos(EYE_ALPHA), np.sin(EYE_ALPHA)

            for sign in [-1, 1]:
                beta = sign * EYE_BETA
                sb, cb = np.sin(beta), np.cos(beta)

                # Direction vector of eye on sphere (unit)
                # d = ca*face_fwd + sa*sb*face_lat + sa*cb*face_dors_r
                # Visibility = d_z component (toward camera)
                vis = ca * face_fwd_z + sa * sb * face_lat_z + sa * cb * face_dors_r_z
                if vis < 0.05:
                    continue  # eye facing away from camera

                # 2D position (project: ignore z, keep x,y)
                ex = hx + hr * (ca * face_fwd_x + sa * sb * face_lat_x + sa * cb * face_dors_x)
                ey = hy + hr * (ca * face_fwd_y + sa * sb * face_lat_y + sa * cb * face_dors_y)

                # Eye size scales with visibility
                eff = eye_r * vis

                # Eye foreshortening: the eye is an ellipse on the sphere surface.
                # Along the face_fwd direction it's foreshortened more than laterally.
                # Compute how much the eye is foreshortened in each face-local axis.
                # The lateral unit vector in 2D (for orientation)
                elat_x, elat_y = face_lat_x, face_lat_y
                efwd_x, efwd_y = face_fwd_x, face_fwd_y
                elnm = max(1e-6, np.sqrt(elat_x**2 + elat_y**2))
                elat_x, elat_y = elat_x / elnm, elat_y / elnm
                efnm = max(1e-6, np.sqrt(efwd_x**2 + efwd_y**2))
                efwd_x, efwd_y = efwd_x / efnm, efwd_y / efnm

                # Sclera (oriented ellipse: wider laterally, narrower in fwd direction)
                wr_lat = eff * 1.15           # semi-axis along lateral
                wr_fwd = eff * 1.15 * vis     # foreshorten along forward by vis
                wr_fwd = max(wr_fwd, eff * 0.35)  # clamp so it doesn't vanish
                sclera_pts = self._oriented_ellipse_pts(
                    ex, ey, elat_x, elat_y, wr_lat, wr_fwd, n_pts=20)
                draw.polygon(sclera_pts, fill=(240, 240, 235))

                # Iris (slightly smaller oriented ellipse)
                ir_lat = eff * 0.95
                ir_fwd = eff * 0.95 * vis
                ir_fwd = max(ir_fwd, eff * 0.28)
                iris_pts = self._oriented_ellipse_pts(
                    ex, ey, elat_x, elat_y, ir_lat, ir_fwd, n_pts=20)
                draw.polygon(iris_pts, fill=(80, 180, 60))

                # Pupil slit (oriented along face lateral direction)
                pr = max(1, eff * 0.3)   # width along forward
                ph = max(1, eff * 0.8)   # height along lateral
                pupil_pts = [
                    (ex + elat_x * ph, ey + elat_y * ph),
                    (ex + efwd_x * pr, ey + efwd_y * pr),
                    (ex - elat_x * ph, ey - elat_y * ph),
                    (ex - efwd_x * pr, ey - efwd_y * pr),
                ]
                draw.polygon(pupil_pts, fill=(15, 15, 15))

                # Specular highlight (small circle offset toward top-right)
                spr = max(1, eff * 0.25)
                hlt_x = ex + elat_x * eff * 0.2 - efwd_x * eff * 0.15
                hlt_y = ey + elat_y * eff * 0.2 - efwd_y * eff * 0.15
                draw.ellipse([hlt_x - spr, hlt_y - spr,
                              hlt_x + spr, hlt_y + spr],
                             fill=(255, 255, 255))

            # ── Nose (on rod axis — unaffected by roll) ──
            nose_x = hx + fx * hr * 0.6
            nose_y = hy + fy * hr * 0.6
            nr = max(2, int(2.5 * sc))
            draw.polygon([
                (nose_x + fx * nr * 0.3, nose_y + fy * nr * 0.3),
                head_pt(hr * 0.6, nr),
                head_pt(hr * 0.6, -nr),
            ], fill=(200, 130, 130))

            # ── Mouth ──
            mw = max(1, int(1 * sc))
            mx = nose_x + fx * nr * 0.8
            my = nose_y + fy * nr * 0.8
            m1x, m1y = head_pt(hr * 0.6 + nr * 1.3, nr * 1.2)
            m2x, m2y = head_pt(hr * 0.6 + nr * 1.3, -nr * 1.2)
            draw.line([(mx, my), (m1x, m1y)], fill=(160, 100, 100), width=mw)
            draw.line([(mx, my), (m2x, m2y)], fill=(160, 100, 100), width=mw)

            # ── Whiskers (lateral extent foreshortens) ──
            ww = max(1, int(1 * sc))
            whisker_base_r = hr * 0.5
            whisker_len = hr * 1.5
            for sign in [-1, 1]:
                for angle_off in [-0.2, 0.0, 0.2]:
                    wb_x, wb_y = head_pt(whisker_base_r * 0.8,
                                         sign * whisker_base_r * 0.5)
                    wd_x = sign * rx * lat_scale + fx * angle_off
                    wd_y = sign * ry * lat_scale + fy * angle_off
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
#   Level 1: POSE        — intrinsic articulation (spine + limbs + head/tail + head roll)
#   Level 2: APPEARANCE  — surface properties (color, stripes, proportions)
#   Level 3: PLACEMENT   — position R² + full SO(3) rotation (yaw, elevation, roll)
#   Level 4: CAMERA      — observer transform: zoom, pan, rotation (SE(2)×R+)
#   Level 5: BACKGROUND  — uniform greyscale intensity
#
# Generative story: "pose the cat, dress it up, put it somewhere,
#                     point the camera, choose the backdrop"
# ═══════════════════════════════════════════════════════════

LEVEL_PARAMS = {
    1: {  # POSE — all joint angles + head roll (16 params)
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
        'head_roll':     (-0.5, 0.5),   # tilt face around neck axis
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
    3: {  # PLACEMENT — R² + SO(3): position + full 3D rotation (5 params)
        'root_x':         (-0.4, 0.4),
        'root_y':         (-0.3, 0.3),
        'root_angle':     (-np.pi, np.pi),     # yaw: in-plane rotation
        'root_elevation': (-0.8, 0.8),          # pitch: view from above/below
        'root_roll':      (-0.8, 0.8),          # roll: cat tilts sideways
    },
    4: {  # CAMERA — observation: SE(2)×R+ (4 params)
        'cam_angle':  (-0.4, 0.4),
        'cam_tx':     (-0.25, 0.25),
        'cam_ty':     (-0.25, 0.25),
        'cam_scale':  (0.7, 1.3),
    },
    5: {  # BACKGROUND — uniform greyscale (1 param)
        'bg_grey':  (0.3, 0.95),
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

    # Apply 3D rotation foreshortening (must match render(): Rx(e)·Ry(r))
    root_2d = np.array([params['root_x'], params['root_y']])
    elev = params.get('root_elevation', 0.0)
    roll_3d = params.get('root_roll', 0.0)
    ce, se = np.cos(elev), np.sin(elev)
    cr, sr = np.cos(roll_3d), np.sin(roll_3d)
    rel = pts - root_2d
    x_new = rel[:, 0] * cr + rel[:, 1] * se * sr
    y_new = rel[:, 1] * ce
    pts = root_2d + np.column_stack([x_new, y_new])

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
