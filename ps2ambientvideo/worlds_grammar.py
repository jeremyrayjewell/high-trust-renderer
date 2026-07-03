from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import ps2fx
from .palettes import Palette


@dataclass(frozen=True)
class WorldMaterial:
    fill: tuple[int, int, int]
    shadow: tuple[int, int, int]
    highlight: tuple[int, int, int]
    edge: tuple[int, int, int]
    alpha: float
    specular: tuple[int, int, int]
    reflection: tuple[int, int, int]
    seam: tuple[int, int, int]


def _mat(
    fill: tuple[int, int, int],
    *,
    edge: tuple[int, int, int] | None = None,
    shade: float = 0.72,
    highlight: float = 1.14,
) -> dict[str, tuple[int, int, int]]:
    edge_color = edge or ps2fx.shade_color(fill, 0.6)
    return {
        "fill": fill,
        "edge": edge_color,
        "shade": ps2fx.shade_color(fill, shade),
        "highlight": ps2fx.shade_color(fill, highlight),
        "shadow": ps2fx.shade_color(fill, 0.48),
    }


def _material(
    kind: str,
    *,
    tint: tuple[int, int, int] | None = None,
    edge: tuple[int, int, int] | None = None,
) -> WorldMaterial:
    presets: dict[str, WorldMaterial] = {
        "translucent_plastic": WorldMaterial((112, 178, 220), (54, 94, 138), (206, 234, 248), (66, 110, 150), 0.9, (244, 248, 255), (190, 222, 240), (128, 188, 220)),
        "reflective_glass": WorldMaterial((86, 168, 208), (36, 82, 118), (222, 242, 250), (58, 114, 146), 0.3, (244, 250, 255), (188, 226, 244), (132, 190, 214)),
        "polished_water": WorldMaterial((22, 106, 198), (8, 66, 148), (218, 242, 255), (56, 130, 188), 0.88, (244, 250, 255), (176, 220, 246), (106, 176, 222)),
        "glossy_floor": WorldMaterial((96, 154, 196), (34, 72, 110), (216, 234, 244), (60, 108, 138), 1.0, (244, 248, 255), (184, 214, 234), (126, 166, 196)),
        "cloud_volume": WorldMaterial((242, 247, 255), (166, 194, 220), (252, 252, 255), (182, 210, 232), 0.74, (255, 255, 255), (232, 242, 250), (208, 224, 238)),
        "chrome_edge": WorldMaterial((196, 214, 228), (86, 104, 122), (248, 250, 255), (110, 132, 150), 1.0, (255, 255, 255), (226, 236, 246), (172, 190, 208)),
        "matte_grass": WorldMaterial((58, 170, 74), (20, 90, 34), (130, 214, 122), (32, 100, 48), 1.0, (202, 236, 192), (108, 182, 110), (82, 144, 86)),
        "dark_space_metal": WorldMaterial((52, 82, 146), (12, 24, 54), (154, 184, 226), (76, 108, 154), 1.0, (220, 232, 250), (120, 154, 196), (84, 104, 142)),
    }
    base = presets[kind]
    if tint is None and edge is None:
        return base
    fill = tint or base.fill
    return WorldMaterial(
        fill=fill,
        shadow=ps2fx.lerp_color(base.shadow, ps2fx.shade_color(fill, 0.56), 0.6),
        highlight=ps2fx.lerp_color(base.highlight, ps2fx.shade_color(fill, 1.24), 0.28),
        edge=edge or ps2fx.lerp_color(base.edge, ps2fx.shade_color(fill, 0.66), 0.52),
        alpha=base.alpha,
        specular=ps2fx.lerp_color(base.specular, (255, 255, 255), 0.24),
        reflection=ps2fx.lerp_color(base.reflection, ps2fx.shade_color(fill, 1.08), 0.34),
        seam=ps2fx.lerp_color(base.seam, ps2fx.shade_color(fill, 0.92), 0.46),
    )


def render_world_scene(mode, frame, t: float, features: dict[str, float], palette: Palette) -> None:
    if mode.name == "memory_beach":
        mode._render_plaza(frame, t, features, palette)
        return
    if mode.name == "blue_sky_aero_terminal":
        _render_sky_world(mode, frame, t, features, palette, mode._recipe_tags())
        return

    tags = mode._recipe_tags()
    family = mode.config.family or getattr(mode.meta, "family", "")
    water_tags = {"water", "fountain", "pool", "ocean", "waterfall", "splash", "jet", "terrace"}
    grass_tags = {"grass", "garden", "park", "lawn", "botanical", "field"}
    glass_tags = {"glass", "atrium", "terminal", "elevator", "greenhouse"}

    if {"people", "dancing", "crowd", "commuters", "figures", "celebration", "walking"} & tags:
        _render_people_world(mode, frame, t, features, palette, tags)
    elif {"space", "orbital", "planet", "moon", "asteroid", "solar", "stars", "observation"} & tags or family == "space_cosmic":
        _render_space_world(mode, frame, t, features, palette, tags)
    elif (water_tags & tags) and not (glass_tags & tags and not ({"fountain", "pool", "waterfall", "splash", "jet", "ocean", "terrace"} & tags)):
        _render_water_world(mode, frame, t, features, palette, tags)
    elif (grass_tags & tags) or family == "nature_weather":
        _render_grass_world(mode, frame, t, features, palette, tags)
    elif (glass_tags & tags) or family == "architecture_interior":
        _render_glass_world(mode, frame, t, features, palette, tags)
    elif {"sky", "cloud", "sunlit"} & tags:
        _render_sky_world(mode, frame, t, features, palette, tags)
    elif family == "water_glass":
        _render_water_world(mode, frame, t, features, palette, tags)
    elif family in {"cyber_urban", "transit", "ui_computer"}:
        _render_cyber_city_world(mode, frame, t, features, palette, tags)
    else:
        _render_sky_world(mode, frame, t, features, palette, tags)


def _world_fog(base: tuple[int, int, int], palette: Palette, amount: float = 0.08) -> tuple[int, int, int]:
    return ps2fx.lerp_color(base, palette.glow, amount)


def _platform_quad(z0: float, z1: float, width0: float, width1: float, y: float = -1.08) -> list[tuple[float, float, float]]:
    return [
        (-width0, y, z0),
        (width0, y, z0),
        (width1, y, z1),
        (-width1, y, z1),
    ]


def _draw_small_clouds(frame, t: float, *, count: int, top_band: float, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    layer = frame.copy()
    max_y = int(h * top_band)
    for idx in range(count):
        cx = int(w * (0.08 + idx / max(1, count - 1) * 0.84) + math.sin(t * 0.04 + idx * 0.9) * 16)
        cy = int(h * (0.08 + (idx % 3) * 0.05))
        cy = min(max_y, cy)
        rx = 30 + (idx % 4) * 12
        ry = 10 + (idx % 3) * 4
        for part in range(3):
            ox = int((part - 1) * rx * 0.55)
            oy = int(math.sin(idx + part) * 2)
            cv2.ellipse(layer, (cx + ox, cy + oy), (max(10, rx - part * 5), max(6, ry - part * 2)), 0, 0, 360, ps2fx.bgr(color), -1, lineType=cv2.LINE_AA)
    frame[:] = ps2fx.alpha_blend(frame, layer, 0.75)


def _draw_cloud_bank(
    frame,
    t: float,
    *,
    y_ratio: float,
    scale: float,
    color: tuple[int, int, int],
    shadow: tuple[int, int, int],
    alpha: float,
    drift: float = 18.0,
) -> None:
    h, w = frame.shape[:2]
    layer = np.zeros_like(frame)
    cy = int(h * y_ratio)
    for idx in range(5):
        cx = int(w * (0.12 + idx * 0.2) + math.sin(t * 0.05 + idx * 0.9) * drift)
        rx = int((68 + idx * 14) * scale)
        ry = int((18 + (idx % 3) * 7) * scale)
        shadow_y = cy + int((12 + idx) * scale)
        cv2.ellipse(layer, (cx, shadow_y), (rx, ry), 0, 0, 360, ps2fx.bgr(shadow), -1, lineType=cv2.LINE_AA)
        for part, offset, lift, part_scale in (
            (0, 0.0, -0.02, 1.0),
            (1, -0.34, -0.18, 0.8),
            (2, 0.32, -0.1, 0.74),
            (3, -0.08, -0.26, 0.56),
            (4, 0.14, 0.08, 0.46),
        ):
            px = int(cx + rx * offset)
            py = cy + int(ry * lift)
            prx = max(12, int(rx * part_scale))
            pry = max(8, int(ry * (0.82 if part < 4 else 0.62)))
            cv2.ellipse(layer, (px, py), (prx, pry), 0, 0, 360, ps2fx.bgr(color), -1, lineType=cv2.LINE_AA)
            cv2.ellipse(
                layer,
                (px - prx // 5, py - pry // 3),
                (max(8, int(prx * 0.44)), max(6, int(pry * 0.34))),
                0,
                0,
                360,
                ps2fx.bgr(ps2fx.shade_color(color, 1.1)),
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                layer,
                (px + prx // 5, py + pry // 4),
                (max(8, int(prx * 0.56)), max(6, int(pry * 0.42))),
                0,
                0,
                360,
                ps2fx.bgr(ps2fx.shade_color(shadow, 0.94)),
                -1,
                lineType=cv2.LINE_AA,
            )
    frame[:] = ps2fx.alpha_blend(frame, layer, alpha)


def _draw_sun(frame, center: tuple[int, int], radius: int, glow: tuple[int, int, int], core: tuple[int, int, int]) -> None:
    ps2fx.draw_glow_circle(frame, center, radius, glow, 0.42)
    cv2.circle(frame, center, max(6, radius // 2), ps2fx.bgr(core), -1, lineType=cv2.LINE_AA)


def _draw_floating_island(frame, camera, fog: tuple[int, int, int], center: tuple[float, float, float], scale: float) -> None:
    x, y, z = center
    stone = _mat((118, 152, 168), edge=(76, 112, 132))
    grass = _mat((92, 198, 116), edge=(52, 118, 72))
    trunk = _mat((108, 82, 62), edge=(72, 54, 42))
    leaf = _mat((78, 192, 108), edge=(46, 114, 70))
    ps2fx.draw_cuboid_3d(frame, camera, (x, y - 0.2, z), (1.8 * scale, 0.6 * scale, 1.4 * scale), stone["fill"], fog, edge=stone["edge"])
    ps2fx.draw_quad_3d(
        frame,
        camera,
        [
            (x - 0.95 * scale, y + 0.08, z - 0.68 * scale),
            (x + 0.95 * scale, y + 0.08, z - 0.68 * scale),
            (x + 0.78 * scale, y + 0.18, z + 0.72 * scale),
            (x - 0.78 * scale, y + 0.18, z + 0.72 * scale),
        ],
        grass["fill"],
        grass["edge"],
        fog,
        1.0,
    )
    for tx in (-0.42, 0.0, 0.42):
        ps2fx.draw_cuboid_3d(frame, camera, (x + tx * scale, y + 0.44, z + tx * 0.16), (0.08 * scale, 0.4 * scale, 0.08 * scale), trunk["fill"], fog, edge=trunk["edge"])
        ps2fx.draw_cuboid_3d(frame, camera, (x + tx * scale, y + 0.76, z + tx * 0.16), (0.4 * scale, 0.34 * scale, 0.4 * scale), leaf["fill"], fog, edge=leaf["edge"])


def _draw_walkway_rails(frame, camera, fog: tuple[int, int, int], near_z: float, far_z: float, width: float, color: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    ps2fx.draw_quad_3d(frame, camera, [(-width, -0.82, near_z), (-width + 0.18, -0.82, near_z), (-width * 0.64, -0.08, far_z), (-width * 0.74, -0.08, far_z)], color, glow, fog, 1.0)
    ps2fx.draw_quad_3d(frame, camera, [(width - 0.18, -0.82, near_z), (width, -0.82, near_z), (width * 0.74, -0.08, far_z), (width * 0.64, -0.08, far_z)], color, glow, fog, 1.0)


def _draw_glass_billboards(frame, camera, fog: tuple[int, int, int], panels: list[tuple[float, float, float, float, float, tuple[int, int, int], tuple[int, int, int], float]]) -> None:
    for x, y, z, width, height, fill, edge, alpha in panels:
        ps2fx.draw_billboard_3d(frame, camera, (x, y, z), (width, height), fill, edge, fog, alpha)


def _draw_foreground_stems(frame, camera, fog: tuple[int, int, int], stems: list[tuple[float, float, float, float, tuple[int, int, int], tuple[int, int, int]]]) -> None:
    for x, y, z, height, stem, leaf in stems:
        ps2fx.draw_cuboid_3d(frame, camera, (x, y, z), (0.08, height, 0.08), stem, fog, edge=ps2fx.shade_color(stem, 0.62))
        ps2fx.draw_cuboid_3d(frame, camera, (x + 0.08, y + height * 0.48, z + 0.04), (0.38, 0.22, 0.16), leaf, fog, edge=ps2fx.shade_color(leaf, 0.58))
        ps2fx.draw_cuboid_3d(frame, camera, (x - 0.08, y + height * 0.72, z - 0.04), (0.32, 0.18, 0.14), leaf, fog, edge=ps2fx.shade_color(leaf, 0.52))


def _draw_side_fountain_arcs(frame, camera, fog: tuple[int, int, int], t: float, f: dict[str, float], center_z: float, width: float) -> None:
    highlight = (246, 250, 255)
    aqua = (138, 222, 246)
    edge = (56, 132, 194)
    for side in (-1, 1):
        base_x = side * width
        for arc_idx in range(4):
            arc_z = center_z + 0.5 + arc_idx * 0.55
            arc_height = 0.9 + arc_idx * 0.16 + max(0.0, math.sin(t * 1.4 + arc_idx * 0.7)) * 0.44 + f["bass"] * 0.16
            arc_x = base_x + side * (0.16 + arc_idx * 0.08)
            top_x = arc_x - side * (0.24 + arc_idx * 0.1)
            top_y = 0.12 + arc_height
            ps2fx.draw_quad_3d(
                frame,
                camera,
                [
                    (arc_x - 0.05, -0.78, arc_z),
                    (arc_x + 0.05, -0.78, arc_z),
                    (top_x + 0.03, top_y, arc_z + 0.18),
                    (top_x - 0.03, top_y, arc_z + 0.18),
                ],
                aqua,
                edge,
                fog,
                0.92,
            )
            ps2fx.draw_billboard_3d(frame, camera, (top_x, top_y + 0.08, arc_z + 0.2), (0.1, 0.1), highlight, aqua, fog, 0.98)
            ps2fx.draw_billboard_3d(frame, camera, (top_x - side * 0.12, top_y - 0.06, arc_z + 0.26), (0.08, 0.08), highlight, aqua, fog, 0.86)


def _draw_near_pass_panels(frame, camera, fog: tuple[int, int, int], t: float, panels: list[tuple[float, float, float, float, float, tuple[int, int, int], tuple[int, int, int], float]]) -> None:
    drift = math.sin(t * 0.26) * 0.12
    shifted = []
    for x, y, z, width, height, fill, edge, alpha in panels:
        shifted.append((x + drift, y, z, width, height, fill, edge, alpha))
    _draw_glass_billboards(frame, camera, fog, shifted)


def _draw_floor_stripes(
    frame,
    camera,
    fog: tuple[int, int, int],
    *,
    near_z: float,
    far_z: float,
    width_near: float,
    width_far: float,
    y: float,
    count: int,
    color: tuple[int, int, int],
    edge: tuple[int, int, int],
    alpha: float,
    thickness: float = 0.18,
) -> None:
    for idx in range(count):
        mix = idx / max(1, count - 1)
        z0 = near_z + (far_z - near_z) * mix
        z1 = min(far_z, z0 + thickness + mix * 0.16)
        w0 = width_near + (width_far - width_near) * mix
        mix_1 = min(1.0, (z1 - near_z) / max(0.001, far_z - near_z))
        w1 = width_near + (width_far - width_near) * mix_1
        inset = 0.4 + mix * 0.24
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [(-w0 + inset, y, z0), (w0 - inset, y, z0), (w1 - inset * 0.85, y, z1), (-w1 + inset * 0.85, y, z1)],
            color,
            edge,
            fog,
            alpha,
        )


def _lerp3(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    )


def _draw_panel_rows_3d(
    frame,
    camera,
    fog: tuple[int, int, int],
    *,
    near_z: float,
    far_z: float,
    width_near: float,
    width_far: float,
    y: float,
    count: int,
    fill_a: tuple[int, int, int],
    fill_b: tuple[int, int, int],
    edge: tuple[int, int, int],
    alpha: float,
    thickness: float = 0.48,
) -> None:
    for idx in range(count):
        mix = idx / max(1, count - 1)
        z0 = near_z + (far_z - near_z) * mix
        z1 = min(far_z, z0 + thickness + mix * 0.12)
        w0 = width_near + (width_far - width_near) * mix
        mix_1 = min(1.0, (z1 - near_z) / max(0.001, far_z - near_z))
        w1 = width_near + (width_far - width_near) * mix_1
        inset = 0.26 + mix * 0.18
        fill = fill_a if idx % 2 == 0 else fill_b
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [(-w0 + inset, y, z0), (w0 - inset, y, z0), (w1 - inset * 0.92, y, z1), (-w1 + inset * 0.92, y, z1)],
            fill,
            edge,
            fog,
            alpha,
        )


def _draw_quad_seams_3d(
    frame,
    camera,
    fog: tuple[int, int, int],
    quad: list[tuple[float, float, float]],
    seam: tuple[int, int, int],
    edge: tuple[int, int, int],
    *,
    count: int,
    alpha: float,
) -> None:
    if len(quad) != 4 or count <= 0:
        return
    for idx in range(count):
        t = (idx + 1) / (count + 1)
        t1 = min(1.0, t + 0.028)
        left0 = _lerp3(quad[0], quad[3], t)
        right0 = _lerp3(quad[1], quad[2], t)
        right1 = _lerp3(quad[1], quad[2], t1)
        left1 = _lerp3(quad[0], quad[3], t1)
        ps2fx.draw_quad_3d(frame, camera, [left0, right0, right1, left1], seam, edge, fog, alpha)


def _draw_material_billboard(
    frame,
    camera,
    fog: tuple[int, int, int],
    center: tuple[float, float, float],
    size: tuple[float, float],
    material: WorldMaterial,
    *,
    alpha_scale: float = 1.0,
    seam_count: int = 0,
) -> None:
    cx, cy, cz = center
    half_w, half_h = size[0] * 0.5, size[1] * 0.5
    quad = [
        (cx - half_w, cy - half_h, cz),
        (cx + half_w, cy - half_h, cz),
        (cx + half_w, cy + half_h, cz),
        (cx - half_w, cy + half_h, cz),
    ]
    ps2fx.draw_quad_3d(frame, camera, quad, material.fill, material.edge, fog, min(1.0, material.alpha * alpha_scale))
    _draw_quad_seams_3d(frame, camera, fog, quad, material.seam, material.edge, count=seam_count, alpha=0.18 * alpha_scale)
    ps2fx.draw_quad_3d(
        frame,
        camera,
        [
            (cx - half_w * 0.12, cy - half_h * 0.9, cz),
            (cx + half_w * 0.18, cy - half_h * 0.9, cz),
            (cx + half_w * 0.08, cy + half_h * 0.88, cz),
            (cx - half_w * 0.22, cy + half_h * 0.88, cz),
        ],
        material.specular,
        material.reflection,
        fog,
        0.12 * alpha_scale,
    )


def _draw_water_surface_bands(
    frame,
    camera,
    fog: tuple[int, int, int],
    t: float,
    *,
    center_z: float,
    width: float,
    depth: float,
    y: float,
) -> None:
    for idx in range(7):
        mix = idx / 6.0
        z = center_z + 0.34 + mix * depth
        wobble = math.sin(t * 0.7 + idx * 0.9) * 0.12
        band_width = width * (1.72 - mix * 0.46)
        ps2fx.draw_billboard_3d(
            frame,
            camera,
            (wobble * 0.4, y + math.sin(t * 0.55 + idx) * 0.015, z),
            (band_width, 0.08 + (idx % 2) * 0.02),
            (188, 224, 244),
            (64, 132, 188),
            fog,
            0.34 if idx % 2 == 0 else 0.22,
        )
        ps2fx.draw_billboard_3d(
            frame,
            camera,
            (-wobble * 0.24, y - 0.03, z + 0.08),
            (band_width * 0.92, 0.04),
            (24, 82, 164),
            (20, 74, 146),
            fog,
            0.2,
        )


def _draw_glass_reflection_bands(
    frame,
    camera,
    fog: tuple[int, int, int],
    bands: list[tuple[float, float, float, float, float]],
    color: tuple[int, int, int],
    edge: tuple[int, int, int],
    alpha: float,
) -> None:
    for x, y, z, width, height in bands:
        ps2fx.draw_billboard_3d(frame, camera, (x, y, z), (width, height), color, edge, fog, alpha)


def _draw_panel_seams_2d(
    frame: np.ndarray,
    *,
    horizontal: list[int] = (),
    vertical: list[int] = (),
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if not horizontal and not vertical:
        return
    layer = frame.copy()
    h, w = frame.shape[:2]
    col = ps2fx.bgr(color)
    for y in horizontal:
        if 0 <= y < h:
            cv2.line(layer, (0, y), (w, y), col, 1, lineType=cv2.LINE_AA)
    for x in vertical:
        if 0 <= x < w:
            cv2.line(layer, (x, 0), (x, h), col, 1, lineType=cv2.LINE_AA)
    frame[:] = ps2fx.alpha_blend(frame, layer, alpha)


def _draw_tree_cluster(frame, camera, fog: tuple[int, int, int], t: float, positions: list[tuple[float, float, float]], *, bright: bool = False) -> None:
    leaf_a = (56, 172, 80) if bright else (52, 146, 72)
    leaf_b = (124, 224, 122) if bright else (82, 190, 96)
    for idx, (x, y, z) in enumerate(positions):
        sway = math.sin(t * 0.18 + idx * 0.9) * 0.05
        ps2fx.draw_cuboid_3d(frame, camera, (x, y - 0.1, z), (0.12, 0.7, 0.12), (98, 72, 52), fog, edge=(66, 50, 38))
        ps2fx.draw_cuboid_3d(frame, camera, (x + sway, y + 0.44, z), (0.82, 0.72, 0.82), leaf_a, fog, edge=(42, 96, 54))
        ps2fx.draw_cuboid_3d(frame, camera, (x - sway * 0.4, y + 0.76, z + 0.04), (0.44, 0.36, 0.44), leaf_b, fog, edge=(54, 118, 66))


def _draw_flower_beds(frame, camera, fog: tuple[int, int, int], z_values: tuple[float, ...]) -> None:
    flower_colors = ((240, 96, 188), (248, 214, 86), (86, 210, 244))
    for row, z in enumerate(z_values):
        for idx, x in enumerate((-3.8, -2.6, -1.2, 1.0, 2.4, 3.8)):
            color = flower_colors[(row + idx) % len(flower_colors)]
            ps2fx.draw_billboard_3d(frame, camera, (x, -0.54, z + idx * 0.05), (0.22, 0.18), color, ps2fx.shade_color(color, 0.56), fog, 1.0)


def _draw_ripple_pool(frame, camera, fog: tuple[int, int, int], center_z: float, width: float, depth: float, *, fill: tuple[int, int, int], edge: tuple[int, int, int]) -> None:
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(center_z, center_z + depth, width, depth * 0.85), fill, edge, fog, 0.98)
    for ring_idx in range(5):
        z = center_z + 0.62 + ring_idx * 0.32
        band_w = width * (1.86 - ring_idx * 0.24)
        ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.94, z), (band_w, 0.08), ps2fx.shade_color(fill, 1.16), edge, fog, 0.38)
        ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.97, z + 0.04), (band_w * 0.92, 0.04), ps2fx.shade_color(fill, 0.86), ps2fx.shade_color(edge, 0.8), fog, 0.26)


def _draw_splash_fountain(frame, camera, fog: tuple[int, int, int], t: float, f: dict[str, float], *, center_z: float, width: float, pool_depth: float, waterfall: bool = False) -> None:
    water = _material("polished_water")
    pool_fill = water.fill
    pool_edge = water.edge
    pool_high = water.specular
    jet_fill = ps2fx.lerp_color(water.fill, water.highlight, 0.38)
    drip = (196, 242, 255)
    _draw_ripple_pool(frame, camera, fog, center_z, width, pool_depth, fill=pool_fill, edge=pool_edge)
    _draw_water_surface_bands(frame, camera, fog, t, center_z=center_z, width=width, depth=pool_depth, y=-0.92)
    jet_positions = [-1.3, -0.75, -0.25, 0.25, 0.75, 1.3]
    for jet_idx, base_x in enumerate(jet_positions):
        x = base_x * (width / 2.3)
        pulse = 0.48 + max(0.0, math.sin(t * 1.6 + jet_idx * 0.55)) * 0.64 + f["beat"] * 0.3
        jet_h = 1.28 + pulse * 1.44
        jet_z = center_z + 0.4 + jet_idx * 0.11
        ps2fx.draw_billboard_3d(frame, camera, (x, -0.84, jet_z + 0.1), (0.6, 0.18), (18, 74, 136), (20, 78, 142), fog, 0.54)
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [
                (x - 0.1, -0.84, jet_z),
                (x + 0.1, -0.84, jet_z),
                (x + 0.03, -0.06 + jet_h, jet_z + 0.06),
                (x - 0.03, -0.06 + jet_h, jet_z + 0.06),
            ],
            jet_fill,
            water.edge,
            fog,
            0.88,
        )
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [
                (x - 0.03, -0.66, jet_z + 0.02),
                (x + 0.03, -0.66, jet_z + 0.02),
                (x + 0.015, -0.18 + jet_h * 0.92, jet_z + 0.08),
                (x - 0.015, -0.18 + jet_h * 0.92, jet_z + 0.08),
            ],
            pool_high,
            water.reflection,
            fog,
            0.72,
        )
        splash_sign = -1.0 if jet_idx % 2 == 0 else 1.0
        sheet_tip = (x + splash_sign * (0.46 + f["bass"] * 0.22), 0.3 + jet_h * 0.54, jet_z + 0.42)
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [
                (x - 0.04, -0.08 + jet_h * 0.82, jet_z + 0.04),
                (x + 0.04, -0.08 + jet_h * 0.82, jet_z + 0.04),
                (sheet_tip[0] + 0.12 * splash_sign, sheet_tip[1] - 0.2, sheet_tip[2] + 0.08),
                (sheet_tip[0] - 0.08 * splash_sign, sheet_tip[1], sheet_tip[2]),
            ],
            ps2fx.shade_color(jet_fill, 1.12),
            water.reflection,
            fog,
            0.52,
        )
        for droplet_idx in range(8):
            arc_dir = -1.0 if droplet_idx % 2 == 0 else 1.0
            arc_x = x + arc_dir * (0.16 + droplet_idx * 0.14) * (1.0 + f["bass"] * 0.28)
            arc_y = -0.1 + jet_h * (0.54 + droplet_idx * 0.09) - droplet_idx * droplet_idx * 0.018
            arc_z = jet_z + droplet_idx * 0.08
            ps2fx.draw_billboard_3d(frame, camera, (arc_x, arc_y, arc_z), (0.13, 0.13), drip, jet_fill, fog, 1.0)
            if droplet_idx < 4:
                ps2fx.draw_billboard_3d(frame, camera, (arc_x + arc_dir * 0.08, arc_y - 0.12, arc_z + 0.04), (0.08, 0.08), pool_high, water.reflection, fog, 0.78)
        ps2fx.draw_billboard_3d(frame, camera, (x, -0.84, jet_z + 0.12), (0.58, 0.12), pool_high, water.reflection, fog, 0.72)
        ps2fx.draw_billboard_3d(frame, camera, (x, -0.86, jet_z + 0.06), (0.82, 0.16), (246, 250, 255), (144, 214, 238), fog, 0.34)
        ps2fx.draw_billboard_3d(frame, camera, (x, -0.78, jet_z + 0.18), (0.16, 0.16), (252, 252, 255), water.highlight, fog, 0.26)
        ps2fx.draw_billboard_3d(frame, camera, (x + splash_sign * 0.22, -0.92, jet_z + 0.42), (0.86, 0.08), (248, 252, 255), (150, 214, 236), fog, 0.3)
    if waterfall:
        for side in (-1, 1):
            for idx in range(7):
                z = center_z + 0.8 + idx * 0.34
                ps2fx.draw_billboard_3d(frame, camera, (side * (width + 0.52), 0.42, z), (0.22, 2.8), (126, 212, 244), (72, 142, 196), fog, 0.76)
                ps2fx.draw_billboard_3d(frame, camera, (side * (width + 0.46), 0.42, z + 0.02), (0.08, 2.8), pool_high, (92, 172, 214), fog, 0.42)
                ps2fx.draw_billboard_3d(frame, camera, (side * (width + 0.58), 0.42, z + 0.08), (0.08, 2.8), (184, 230, 248), (88, 160, 204), fog, 0.26)


def _segment_quad(a: tuple[int, int], b: tuple[int, int], width_a: float, width_b: float) -> list[tuple[int, int]]:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    length = max(1.0, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    return [
        (int(a[0] + nx * width_a), int(a[1] + ny * width_a)),
        (int(a[0] - nx * width_a), int(a[1] - ny * width_a)),
        (int(b[0] - nx * width_b), int(b[1] - ny * width_b)),
        (int(b[0] + nx * width_b), int(b[1] + ny * width_b)),
    ]


def _person_depth(camera, center: tuple[float, float, float], scale: float) -> float:
    probe, depth = ps2fx.project_point(camera, (center[0], center[1] - 0.12 * scale, center[2] - 0.02 * scale), 640, 360)
    return depth if probe is not None else -9999.0


def _draw_colored_person(frame, camera, fog: tuple[int, int, int], t: float, f: dict[str, float], center: tuple[float, float, float], scale: float, idx: int, walking: bool = False) -> bool:
    h, w = frame.shape[:2]
    cx, cy, cz = center
    beat_lift = f["beat"] * 0.28 + f["onset"] * 0.1
    dance_speed = 1.2 if walking else 2.0
    sway = math.sin(t * dance_speed + idx * 0.7) * 0.16 * scale
    bounce = max(0.0, math.sin(t * 1.8 + idx * 0.85)) * 0.28 * scale + beat_lift * 0.22 * scale
    step = math.sin(t * (1.3 if walking else 2.4) + idx * 0.6) * 0.34 * scale
    lean = math.sin(t * 0.9 + idx * 0.4) * 0.16 * scale
    arm_raise = (0.14 if walking else 0.44) * scale + max(0.0, math.sin(t * 2.1 + idx)) * 0.34 * scale
    palette_cycle = [
        ((74, 192, 206), (232, 242, 248)),
        ((214, 122, 176), (246, 226, 238)),
        ((216, 194, 104), (246, 238, 184)),
        ((112, 166, 226), (228, 236, 248)),
        ((98, 188, 128), (228, 246, 228)),
    ]
    body_fill, edge = palette_cycle[idx % len(palette_cycle)]
    plastic = _material("translucent_plastic", tint=body_fill, edge=ps2fx.lerp_color(edge, ps2fx.shade_color(body_fill, 0.62), 0.54))
    joints = {
        "hip": (cx + sway, cy - 0.04 + bounce, cz),
        "chest": (cx + sway + lean * 0.5, cy + 0.54 * scale + bounce, cz),
        "head": (cx + sway + lean * 0.75, cy + 1.02 * scale + bounce, cz),
        "ls": (cx - 0.24 * scale + sway, cy + 0.52 * scale + bounce + lean, cz),
        "rs": (cx + 0.24 * scale + sway, cy + 0.52 * scale + bounce - lean, cz),
        "lh": (cx - 0.14 * scale + sway, cy - 0.04 * scale + bounce, cz),
        "rh": (cx + 0.14 * scale + sway, cy - 0.04 * scale + bounce, cz),
        "la": (cx - 0.52 * scale + sway, cy + 0.32 * scale + bounce + arm_raise, cz + 0.02),
        "ra": (cx + 0.52 * scale + sway, cy + 0.24 * scale + bounce + arm_raise * 0.82, cz + 0.02),
        "lk": (cx - 0.14 * scale + sway + step * 0.46, cy - 0.48 * scale + bounce, cz + 0.02),
        "rk": (cx + 0.14 * scale + sway - step * 0.46, cy - 0.48 * scale + bounce, cz - 0.02),
        "lf": (cx - 0.16 * scale + sway + step, cy - 0.98 * scale + bounce, cz + 0.04),
        "rf": (cx + 0.16 * scale + sway - step, cy - 0.98 * scale + bounce, cz - 0.04),
    }
    screen_points: dict[str, tuple[int, int]] = {}
    depths: list[float] = []
    for name, point in joints.items():
        screen, depth = ps2fx.project_point(camera, point, w, h)
        if screen is None:
            return False
        screen_points[name] = screen
        depths.append(depth)
    depth = float(sum(depths) / len(depths))
    fill = ps2fx.fog_color(plastic.fill, fog, depth, density=0.055)
    edge_col = ps2fx.fog_color(plastic.edge, fog, depth * 0.82, density=0.05)
    pelvis_fill = ps2fx.fog_color(plastic.shadow, fog, depth, density=0.055)
    torso = [screen_points["ls"], screen_points["rs"], screen_points["rh"], screen_points["lh"]]
    torso_fill = ps2fx.fog_color(ps2fx.lerp_color(plastic.fill, plastic.highlight, 0.12), fog, depth, density=0.05)
    torso_high = ps2fx.fog_color(plastic.highlight, fog, depth * 0.66, density=0.05)
    torso_shadow = ps2fx.fog_color(plastic.shadow, fog, depth, density=0.055)
    shell_alpha = min(0.96, 0.88 + scale * 0.02)
    core_alpha = 0.28 if scale > 2.1 else 0.22
    ps2fx.draw_shaded_polygon(frame, torso, torso_fill, edge_col, shell_alpha)
    ps2fx.draw_shaded_polygon(frame, [screen_points["ls"], screen_points["chest"], screen_points["hip"]], torso_high, None, 0.46)
    ps2fx.draw_shaded_polygon(frame, [screen_points["rs"], screen_points["rh"], screen_points["hip"]], torso_shadow, None, 0.42)
    ps2fx.draw_shaded_polygon(frame, [screen_points["ls"], screen_points["rs"], screen_points["chest"]], ps2fx.fog_color(ps2fx.shade_color(fill, 1.2), fog, depth * 0.62, density=0.05), None, 0.24)
    ps2fx.draw_shaded_polygon(frame, [screen_points["lh"], screen_points["hip"], screen_points["rh"], screen_points["chest"]], ps2fx.fog_color(ps2fx.shade_color(fill, 0.66), fog, depth * 0.82, density=0.05), None, core_alpha)
    pelvis = [screen_points["lh"], screen_points["rh"], screen_points["rk"], screen_points["lk"]]
    ps2fx.draw_shaded_polygon(frame, pelvis, pelvis_fill, edge_col, shell_alpha)
    ps2fx.draw_shaded_polygon(frame, [screen_points["lh"], screen_points["rh"], screen_points["hip"]], ps2fx.fog_color(ps2fx.shade_color(fill, 1.08), fog, depth * 0.7, density=0.05), None, 0.18)
    for a, b, wa, wb in (
        ("ls", "la", 10.0, 5.0),
        ("rs", "ra", 10.0, 5.0),
        ("lh", "lk", 9.0, 6.0),
        ("rh", "rk", 9.0, 6.0),
        ("lk", "lf", 6.0, 4.0),
        ("rk", "rf", 6.0, 4.0),
        ("chest", "head", 7.0, 5.0),
    ):
        poly = _segment_quad(screen_points[a], screen_points[b], wa, wb)
        limb_fill = ps2fx.lerp_color(fill, plastic.shadow if a in {"rs", "rh", "rk"} else plastic.highlight, 0.1)
        ps2fx.draw_shaded_polygon(frame, poly, limb_fill, edge_col, shell_alpha - 0.03)
        if len(poly) >= 4:
            ps2fx.draw_shaded_polygon(frame, [poly[0], poly[1], poly[2]], ps2fx.shade_color(limb_fill, 1.08), None, 0.32)
            ps2fx.draw_shaded_polygon(frame, [poly[0], poly[3], poly[2]], ps2fx.fog_color(ps2fx.shade_color(fill, 1.18), fog, depth * 0.62, density=0.05), None, 0.18)
    for foot_name, direction in (("lf", -1), ("rf", 1)):
        foot = screen_points[foot_name]
        ankle = screen_points["lk" if foot_name == "lf" else "rk"]
        foot_poly = _segment_quad(ankle, (foot[0] + direction * 10, foot[1] + 3), 4.0, 6.0)
        ps2fx.draw_shaded_polygon(frame, foot_poly, ps2fx.shade_color(fill, 0.58), edge_col, 0.9)
    foot_left = screen_points["lf"]
    foot_right = screen_points["rf"]
    shadow_center = ((foot_left[0] + foot_right[0]) // 2, max(foot_left[1], foot_right[1]) + 8)
    shadow_axes = (
        max(12, abs(foot_right[0] - foot_left[0]) // 2 + 12),
        max(4, int(16.0 / max(1.2, depth))),
    )
    cv2.ellipse(
        frame,
        shadow_center,
        shadow_axes,
        0,
        0,
        360,
        ps2fx.bgr(ps2fx.fog_color((26, 48, 74), fog, depth * 0.46)),
        -1,
        lineType=cv2.LINE_AA,
    )
    reflection_color = ps2fx.fog_color(plastic.reflection, fog, depth * 0.72, density=0.05)
    reflection_layer = frame.copy()
    for name_a, name_b, shrink in (("lh", "lk", 0.75), ("rh", "rk", 0.75), ("lk", "lf", 0.7), ("rk", "rf", 0.7)):
        a = screen_points[name_a]
        b = screen_points[name_b]
        reflected_a = (a[0], shadow_center[1] + max(4, shadow_center[1] - a[1]))
        reflected_b = (b[0], shadow_center[1] + max(4, shadow_center[1] - b[1]))
        poly = _segment_quad(reflected_a, reflected_b, 6.0 * shrink, 3.0 * shrink)
        ps2fx.draw_shaded_polygon(reflection_layer, poly, reflection_color, None, 0.12 if scale > 2.2 else 0.08)
    frame[:] = ps2fx.alpha_blend(frame, reflection_layer, 0.72)
    head = screen_points["head"]
    head_radius = max(6, int(22.0 / max(1.2, depth)))
    head_poly = [
        (head[0], head[1] - head_radius),
        (head[0] + head_radius, head[1]),
        (head[0], head[1] + head_radius),
        (head[0] - head_radius, head[1]),
    ]
    head_fill = ps2fx.fog_color(ps2fx.lerp_color(plastic.fill, (244, 248, 255), 0.54), fog, depth, density=0.05)
    ps2fx.draw_shaded_polygon(frame, head_poly, head_fill, edge_col, 0.92)
    cv2.line(
        frame,
        (screen_points["ls"][0] + 2, screen_points["ls"][1] - 6),
        (screen_points["rh"][0] - 2, screen_points["rh"][1] + 4),
        ps2fx.bgr(ps2fx.fog_color(plastic.specular, fog, depth * 0.56, density=0.05)),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (head[0] - head_radius // 2, head[1] - head_radius // 2),
        (head[0] + head_radius // 3, head[1] - head_radius // 3),
        ps2fx.bgr(ps2fx.fog_color((246, 250, 255), fog, depth * 0.5)),
        1,
        lineType=cv2.LINE_AA,
    )
    return True


def _draw_people_cluster(frame, camera, fog: tuple[int, int, int], t: float, f: dict[str, float], *, center_z: float, walking: bool) -> None:
    ps2fx.debug_begin_object("polygon_people_world")
    rows = (
        (center_z - 0.1, 4, 1.95, 2.9, -0.08),
        (center_z + 1.9, 3, 1.85, 2.15, -0.02),
        (center_z + 3.8, 3, 1.55, 1.55, 0.04),
    )
    figures: list[tuple[float, tuple[float, float, float], float, int]] = []
    for row_idx, (z, count, spacing, scale, y) in enumerate(rows):
        offset = -((count - 1) * spacing * 0.5)
        for col in range(count):
            x = offset + col * spacing + math.sin(t * 0.24 + row_idx + col * 0.7) * 0.12
            center = (x, y, z + col * 0.03)
            figures.append((_person_depth(camera, center, scale), center, scale, row_idx * 8 + col))
    for _, center, scale, idx in sorted(figures, key=lambda item: (-item[0], item[2], item[1][0])):
        _draw_colored_person(frame, camera, fog, t, f, center, scale, idx, walking=walking)
    ps2fx.debug_end_object()


def _draw_supporting_people(
    frame,
    camera,
    fog: tuple[int, int, int],
    t: float,
    f: dict[str, float],
    *,
    center_z: float,
    walking: bool,
    scale_bias: float = 1.0,
    lateral_bias: float = 0.0,
) -> None:
    ps2fx.debug_begin_object("supporting_people_world")
    rows = (
        (center_z, 3, 1.55, 1.5 * scale_bias, -0.12),
        (center_z + 1.65, 2, 1.85, 1.18 * scale_bias, -0.08),
    )
    figures: list[tuple[float, tuple[float, float, float], float, int]] = []
    for row_idx, (z, count, spacing, scale, y) in enumerate(rows):
        offset = -((count - 1) * spacing * 0.5) + lateral_bias + row_idx * 0.16
        for col in range(count):
            x = offset + col * spacing + math.sin(t * 0.22 + row_idx * 0.8 + col * 0.6) * 0.08
            center = (x, y, z + col * 0.04)
            figures.append((_person_depth(camera, center, scale), center, scale, 40 + row_idx * 6 + col))
    for _, center, scale, idx in sorted(figures, key=lambda item: (-item[0], item[2], item[1][0])):
        _draw_colored_person(frame, camera, fog, t, f, center, scale, idx, walking=walking)
    ps2fx.debug_end_object()


def _draw_city_traffic(frame, camera, fog: tuple[int, int, int], t: float, z_values: tuple[float, ...]) -> None:
    for idx, z in enumerate(z_values):
        lane_shift = ((t * 0.95 + idx * 0.8) % 1.0) * 3.4 - 1.7
        ps2fx.draw_cuboid_3d(frame, camera, (lane_shift, -0.82, z), (0.76, 0.34, 1.26), (224, 236, 244), fog, edge=(244, 248, 255))
        ps2fx.draw_billboard_3d(frame, camera, (lane_shift - 0.22, -0.74, z - 0.44), (0.12, 0.1), (250, 102, 198), (248, 246, 255), fog, 1.0)
        ps2fx.draw_billboard_3d(frame, camera, (lane_shift + 0.22, -0.74, z - 0.44), (0.12, 0.1), (114, 234, 250), (248, 246, 255), fog, 1.0)


def _draw_space_backdrop(frame, t: float) -> None:
    ps2fx.gradient_background(frame, (2, 6, 18), (10, 20, 54))
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, int(h * 0.58)), (w, h), ps2fx.bgr((4, 8, 18)), -1)
    planet_center = (int(w * 0.77), int(h * 0.28))
    planet_r = int(h * 0.19)
    cv2.circle(frame, planet_center, planet_r, ps2fx.bgr((144, 196, 255)), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, planet_center, planet_r, ps2fx.bgr((236, 246, 255)), 2, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, planet_center, (planet_r + 28, max(10, planet_r // 2)), 22, 0, 360, ps2fx.bgr((212, 232, 255)), 2, lineType=cv2.LINE_AA)
    moon_center = (int(w * 0.16 + math.sin(t * 0.05) * 8), int(h * 0.16))
    cv2.circle(frame, moon_center, int(h * 0.04), ps2fx.bgr((246, 250, 255)), -1, lineType=cv2.LINE_AA)
    for idx in range(120):
        x = int((idx * 71) % w)
        y = int((idx * 43 + idx * idx * 3) % int(h * 0.72))
        radius = 1 if idx % 4 else 2
        cv2.circle(frame, (x, y), radius, ps2fx.bgr((236, 244, 255)), -1, lineType=cv2.LINE_AA)


def _render_sky_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    h, w = frame.shape[:2]
    camera = mode._camera(t * 0.34, dolly=t * (0.09 + f["bass"] * 0.04), bob=0.014, orbit=0.34 + f["section"] * 0.2, f=f)
    fog = _world_fog((170, 214, 238), palette, 0.012)
    terminal_mode = mode.name == "blue_sky_aero_terminal"
    if terminal_mode:
        fog = _world_fog((134, 188, 214), palette, 0.004)
        ps2fx.gradient_background(frame, (86, 184, 248), (208, 240, 248))
        horizon_y = int(h * 0.62)
        cv2.rectangle(frame, (0, horizon_y), (w, h), ps2fx.bgr((92, 164, 206)), -1)
        _draw_panel_seams_2d(frame, horizontal=[int(h * 0.58), int(h * 0.66)], color=(154, 204, 224), alpha=0.1)
        _draw_small_clouds(frame, t, count=5, top_band=0.12, color=(246, 250, 255))
        _draw_small_clouds(frame, t + 8.0, count=3, top_band=0.22, color=(226, 240, 248))
        _draw_cloud_bank(frame, t, y_ratio=0.19, scale=0.92, color=(246, 250, 255), shadow=(170, 198, 220), alpha=0.42)
        _draw_cloud_bank(frame, t + 11.0, y_ratio=0.29, scale=0.74, color=(236, 244, 252), shadow=(152, 184, 210), alpha=0.28, drift=12.0)
    else:
        ps2fx.gradient_background(frame, (92, 184, 248), (162, 228, 244))
        _draw_small_clouds(frame, t, count=6, top_band=0.15, color=(244, 248, 252))
        _draw_small_clouds(frame, t + 8.0, count=4, top_band=0.28, color=(220, 238, 248))
        _draw_cloud_bank(frame, t, y_ratio=0.2, scale=1.0, color=(244, 248, 255), shadow=(176, 206, 228), alpha=0.32)
    _draw_sun(frame, (int(w * 0.82), int(h * 0.17)), 36, (246, 252, 255), (255, 242, 206))
    for ray_idx in range(5):
        y = int(h * (0.18 + ray_idx * 0.045))
        cv2.line(frame, (int(w * 0.6), y), (w, y + 18 + ray_idx * 8), ps2fx.bgr((222, 242, 248)), 1, lineType=cv2.LINE_AA)
    ps2fx.debug_begin_object("sky_world")
    if terminal_mode:
        deck_material = _material("glossy_floor", tint=(74, 138, 188))
        roof_material = _material("chrome_edge", tint=(232, 246, 252))
        glass_material = _material("reflective_glass", tint=(74, 170, 216))
        shadow_mat = _mat((20, 68, 110), edge=(18, 48, 78))
        terminal_mat = _mat((214, 236, 244), edge=(88, 132, 156))
        deck_quad = _platform_quad(camera.z + 1.2, camera.z + 14.8, 7.4, 5.8)
        ps2fx.draw_quad_3d(frame, camera, deck_quad, deck_material.fill, deck_material.edge, fog, 1.0)
        _draw_quad_seams_3d(frame, camera, fog, deck_quad, deck_material.seam, deck_material.edge, count=8, alpha=0.18)
        ps2fx.draw_quad_3d(frame, camera, [(-7.2, -1.18, camera.z + 1.4), (7.2, -1.18, camera.z + 1.4), (5.8, -1.46, camera.z + 12.2), (-5.8, -1.46, camera.z + 12.2)], deck_material.shadow, (30, 62, 90), fog, 0.95)
        ps2fx.draw_quad_3d(frame, camera, [(-6.6, -0.96, camera.z + 1.5), (6.6, -0.96, camera.z + 1.5), (5.2, -0.96, camera.z + 11.6), (-5.2, -0.96, camera.z + 11.6)], shadow_mat["fill"], shadow_mat["edge"], fog, 0.92)
        ps2fx.draw_quad_3d(frame, camera, [(-6.8, -0.82, camera.z + 2.4), (-2.0, -0.82, camera.z + 2.4), (-1.0, -0.46, camera.z + 8.2), (-5.1, -0.46, camera.z + 6.1)], deck_material.highlight, deck_material.edge, fog, 0.98)
        ps2fx.draw_quad_3d(frame, camera, [(2.0, -0.82, camera.z + 2.4), (6.8, -0.82, camera.z + 2.4), (5.1, -0.46, camera.z + 6.1), (1.0, -0.46, camera.z + 8.2)], deck_material.highlight, deck_material.edge, fog, 0.98)
        ps2fx.draw_quad_3d(frame, camera, [(-1.2, -0.82, camera.z + 2.0), (1.2, -0.82, camera.z + 2.0), (0.8, -0.66, camera.z + 10.2), (-0.8, -0.66, camera.z + 10.2)], (58, 134, 190), (42, 96, 140), fog, 0.94)
        left_roof = [(-7.1, 1.08, camera.z + 1.8), (-0.6, 1.42, camera.z + 2.6), (-0.2, 2.72, camera.z + 8.8), (-5.8, 2.42, camera.z + 6.4)]
        right_roof = [(0.6, 1.42, camera.z + 2.6), (7.1, 1.08, camera.z + 1.8), (5.8, 2.42, camera.z + 6.4), (0.2, 2.72, camera.z + 8.8)]
        ps2fx.draw_quad_3d(frame, camera, left_roof, roof_material.fill, roof_material.edge, fog, 0.6)
        ps2fx.draw_quad_3d(frame, camera, right_roof, roof_material.fill, roof_material.edge, fog, 0.6)
        _draw_quad_seams_3d(frame, camera, fog, left_roof, roof_material.seam, roof_material.edge, count=5, alpha=0.12)
        _draw_quad_seams_3d(frame, camera, fog, right_roof, roof_material.seam, roof_material.edge, count=5, alpha=0.12)
        left_glass = [(-6.9, -0.1, camera.z + 1.9), (-1.2, 0.22, camera.z + 2.8), (-0.7, 2.2, camera.z + 8.4), (-5.8, 1.92, camera.z + 6.2)]
        right_glass = [(1.2, 0.22, camera.z + 2.8), (6.9, -0.1, camera.z + 1.9), (5.8, 1.92, camera.z + 6.2), (0.7, 2.2, camera.z + 8.4)]
        ps2fx.draw_quad_3d(frame, camera, left_glass, glass_material.fill, glass_material.edge, fog, 0.28)
        ps2fx.draw_quad_3d(frame, camera, right_glass, glass_material.fill, glass_material.edge, fog, 0.28)
        _draw_quad_seams_3d(frame, camera, fog, left_glass, glass_material.seam, glass_material.edge, count=4, alpha=0.12)
        _draw_quad_seams_3d(frame, camera, fog, right_glass, glass_material.seam, glass_material.edge, count=4, alpha=0.12)
        ps2fx.draw_quad_3d(frame, camera, [(-6.8, 0.72, camera.z + 2.2), (-0.8, 0.98, camera.z + 2.9), (-0.6, 1.28, camera.z + 7.8), (-6.0, 1.08, camera.z + 5.8)], roof_material.shadow, roof_material.edge, fog, 0.92)
        ps2fx.draw_quad_3d(frame, camera, [(0.8, 0.98, camera.z + 2.9), (6.8, 0.72, camera.z + 2.2), (6.0, 1.08, camera.z + 5.8), (0.6, 1.28, camera.z + 7.8)], roof_material.shadow, roof_material.edge, fog, 0.92)
        ps2fx.draw_quad_3d(frame, camera, [(-7.0, 0.86, camera.z + 1.9), (-0.8, 1.12, camera.z + 2.7), (-0.5, 1.46, camera.z + 8.4), (-6.0, 1.24, camera.z + 6.0)], (122, 156, 176), (72, 98, 118), fog, 0.38)
        ps2fx.draw_quad_3d(frame, camera, [(0.8, 1.12, camera.z + 2.7), (7.0, 0.86, camera.z + 1.9), (6.0, 1.24, camera.z + 6.0), (0.5, 1.46, camera.z + 8.4)], (122, 156, 176), (72, 98, 118), fog, 0.38)
        for idx, x in enumerate((-5.9, -3.1, -0.2, 2.7, 5.5)):
            z = camera.z + 2.4 + idx * 1.95
            ps2fx.draw_cuboid_3d(frame, camera, (x, 0.34, z), (0.34, 2.9, 0.34), terminal_mat["fill"], fog, edge=terminal_mat["edge"])
        for idx, z in enumerate((camera.z + 4.2, camera.z + 7.4, camera.z + 10.8)):
            px = -4.6 + idx * 4.6 + math.sin(t * 0.42 + idx) * 0.26
            ps2fx.draw_cuboid_3d(frame, camera, (px, -0.64, z), (1.12, 0.42, 0.62), (68, 148, 210), fog, edge=(44, 102, 144))
            ps2fx.draw_billboard_3d(frame, camera, (px, 0.02, z - 0.14), (1.12, 0.32), (196, 236, 248), (84, 148, 176), fog, 0.72)
        for idx, z in enumerate((camera.z + 3.6, camera.z + 6.8, camera.z + 10.2)):
            ps2fx.draw_quad_3d(frame, camera, [(-6.2, -0.82, z), (-4.3, -0.82, z), (-4.0, 0.42, z + 1.9), (-5.7, 0.42, z + 1.8)], (206, 226, 232), (102, 138, 154), fog, 0.94)
            ps2fx.draw_quad_3d(frame, camera, [(-6.2, -1.02, z), (-4.3, -1.02, z), (-4.1, -1.26, z + 1.9), (-5.9, -1.22, z + 1.8)], (70, 108, 138), (42, 72, 98), fog, 0.92)
            ps2fx.draw_billboard_3d(frame, camera, (-5.0, 0.02, z + 0.8), (0.88, 0.3), (108, 198, 226), (64, 120, 146), fog, 0.9)
        _draw_walkway_rails(frame, camera, fog, camera.z + 1.5, camera.z + 13.8, 6.2, (206, 234, 244), (96, 154, 180))
        _draw_floor_stripes(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.1,
            far_z=camera.z + 13.0,
            width_near=6.2,
            width_far=4.8,
            y=-0.95,
            count=6,
            color=(148, 196, 220),
            edge=(92, 132, 156),
            alpha=0.18,
            thickness=0.16,
        )
        _draw_panel_rows_3d(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.0,
            far_z=camera.z + 13.2,
            width_near=6.4,
            width_far=5.0,
            y=-0.93,
            count=7,
            fill_a=(118, 180, 214),
            fill_b=(86, 148, 188),
            edge=(66, 112, 140),
            alpha=0.22,
            thickness=0.42,
        )
        _draw_glass_reflection_bands(
            frame,
            camera,
            fog,
            [
                (-3.8, 0.98, camera.z + 4.8, 0.18, 2.2),
                (3.6, 0.84, camera.z + 6.7, 0.14, 1.9),
                (-2.2, 0.62, camera.z + 8.8, 0.12, 1.5),
            ],
            (234, 244, 248),
            (122, 176, 196),
            0.18,
        )
        _draw_near_pass_panels(
            frame,
            camera,
            fog,
            t,
            [
                (-6.1, 1.04, camera.z + 2.0, 0.82, 2.54, (126, 208, 230), (72, 138, 164), 0.16),
                (5.8, 1.12, camera.z + 2.48, 0.7, 2.02, (150, 220, 236), (88, 148, 172), 0.14),
            ],
        )
        _draw_material_billboard(frame, camera, fog, (-5.6, 0.92, camera.z + 2.2), (0.72, 2.4), glass_material, alpha_scale=0.8, seam_count=3)
        _draw_material_billboard(frame, camera, fog, (5.2, 1.04, camera.z + 2.64), (0.62, 1.92), glass_material, alpha_scale=0.72, seam_count=2)
        ps2fx.draw_billboard_3d(frame, camera, (-5.8, -0.72, camera.z + 3.6), (1.42, 0.1), (226, 238, 244), (130, 182, 204), fog, 0.56)
        ps2fx.draw_billboard_3d(frame, camera, (4.9, -0.68, camera.z + 6.8), (1.26, 0.1), (226, 238, 244), (130, 182, 204), fog, 0.48)
        ps2fx.debug_end_object()
        return
    platform_mat = _mat((92, 176, 210), edge=(62, 126, 156))
    beam_mat = _mat((136, 196, 220), edge=(84, 144, 170))
    rail_mat = _mat((158, 210, 228), edge=(88, 148, 174))
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.6, camera.z + 13.4, 5.9, 4.8), platform_mat["fill"], platform_mat["edge"], fog, 0.96)
    ps2fx.draw_billboard_3d(frame, camera, (-4.6, 0.08, camera.z + 1.85), (0.8, 1.9), beam_mat["fill"], beam_mat["edge"], fog, 0.48)
    _draw_walkway_rails(frame, camera, fog, camera.z + 1.8, camera.z + 12.0, 5.1, rail_mat["fill"], rail_mat["edge"])
    for idx, center in enumerate(((-4.6, 0.34, camera.z + 3.8), (-0.3, 0.84, camera.z + 6.6), (3.6, 0.52, camera.z + 9.8))):
        _draw_floating_island(frame, camera, fog, center, 1.0 - idx * 0.12)
    ps2fx.draw_billboard_3d(frame, camera, (-2.8, 0.82, camera.z + 6.4), (1.32, 0.34), (214, 244, 252), (118, 176, 196), fog, 0.86)
    ps2fx.draw_billboard_3d(frame, camera, (2.2, 0.66, camera.z + 10.2), (1.72, 0.4), (214, 244, 252), (118, 176, 196), fog, 0.76)
    ps2fx.draw_quad_3d(frame, camera, [(-3.4, 0.44, camera.z + 6.2), (-2.1, 0.44, camera.z + 6.2), (-1.8, 1.14, camera.z + 7.0), (-3.2, 1.14, camera.z + 7.0)], (84, 178, 210), (56, 112, 140), fog, 0.44)
    ps2fx.draw_quad_3d(frame, camera, [(1.2, 0.3, camera.z + 9.8), (3.1, 0.3, camera.z + 9.8), (2.7, 1.2, camera.z + 10.8), (0.9, 1.2, camera.z + 10.8)], (94, 188, 218), (62, 122, 150), fog, 0.4)
    for idx, x in enumerate((-4.2, -1.3, 1.6, 4.4)):
        ps2fx.draw_cuboid_3d(frame, camera, (x, 0.04, camera.z + 3.3 + idx * 1.9), (0.3, 2.0, 0.3), (132, 188, 208), fog, edge=(82, 132, 154))
    _draw_near_pass_panels(
        frame,
        camera,
        fog,
        t,
        [
            (-5.25, 0.7, camera.z + 2.05, 0.64, 1.6, (110, 192, 226), (72, 138, 166), 0.28),
            (4.8, 1.02, camera.z + 2.8, 0.54, 1.26, (146, 210, 230), (86, 148, 170), 0.24),
        ],
    )
    for idx in range(5):
        px = int(w * (0.18 + idx * 0.12) + math.sin(t * 0.3 + idx) * 18)
        py = int(h * (0.18 + (idx % 2) * 0.05))
        cv2.line(frame, (px, py), (px + 8, py - 4), ps2fx.bgr((244, 248, 252)), 1, lineType=cv2.LINE_AA)
    ps2fx.debug_end_object()


def _render_grass_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    camera = mode._camera(t * 0.3, dolly=t * (0.07 + f["bass"] * 0.03), bob=0.014, orbit=0.28 + f["section"] * 0.18, f=f)
    fog = _world_fog((122, 184, 120), palette, 0.014)
    ps2fx.gradient_background(frame, (112, 194, 238), (172, 230, 232))
    ps2fx.debug_begin_object("grass_world")
    grass_mat = _mat((40, 174, 54), edge=(24, 98, 36))
    path_mat = _mat((174, 150, 90), edge=(112, 92, 52))
    bench_mat = _mat((110, 146, 124), edge=(62, 96, 74))
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.2, camera.z + 15.0, 7.2, 6.4), grass_mat["fill"], grass_mat["edge"], fog, 1.0)
    _draw_floor_stripes(
        frame,
        camera,
        fog,
        near_z=camera.z + 2.1,
        far_z=camera.z + 14.2,
        width_near=6.8,
        width_far=5.7,
        y=-0.97,
        count=5,
        color=(62, 156, 74),
        edge=(42, 118, 54),
        alpha=0.16,
        thickness=0.22,
    )
    for idx, z in enumerate((camera.z + 7.0, camera.z + 10.8, camera.z + 14.6)):
        width = 8.2 - idx * 1.0
        ridge_fill = (12 + idx * 8, 132 + idx * 18, 24 + idx * 10)
        ridge_edge = (18, 94 + idx * 10, 34 + idx * 4)
        ps2fx.draw_quad_3d(frame, camera, [(-width, -0.98, z), (width, -0.98, z), (width * 0.78, -0.48 + idx * 0.08, z + 2.0), (-width * 0.78, -0.48 + idx * 0.08, z + 2.0)], ridge_fill, ridge_edge, fog, 0.98)
    ps2fx.draw_quad_3d(frame, camera, [(-1.1, -0.98, camera.z + 1.5), (1.1, -0.98, camera.z + 1.5), (0.58, -0.98, camera.z + 14.0), (-0.58, -0.98, camera.z + 14.0)], path_mat["fill"], path_mat["edge"], fog, 0.98)
    ps2fx.draw_cuboid_3d(frame, camera, (-5.3, -0.64, camera.z + 1.7), (0.2, 1.55, 0.2), (124, 98, 70), fog, edge=(78, 60, 42))
    ps2fx.draw_billboard_3d(frame, camera, (-4.8, 0.18, camera.z + 2.1), (0.78, 0.48), (200, 220, 164), (110, 132, 76), fog, 0.92)
    _draw_foreground_stems(
        frame,
        camera,
        fog,
        [
            (4.95, -0.76, camera.z + 1.85, 1.5, (86, 120, 62), (84, 202, 98)),
            (5.35, -0.76, camera.z + 2.1, 1.22, (86, 120, 62), (108, 220, 116)),
            (-4.6, -0.76, camera.z + 2.4, 1.16, (86, 120, 62), (96, 214, 108)),
        ],
    )
    _draw_tree_cluster(frame, camera, fog, t, [(-4.5, -0.48, camera.z + 3.0), (-2.4, -0.48, camera.z + 5.1), (0.2, -0.48, camera.z + 7.0), (2.8, -0.48, camera.z + 8.8), (4.8, -0.48, camera.z + 10.6)], bright=True)
    _draw_flower_beds(frame, camera, fog, (camera.z + 3.8, camera.z + 6.2, camera.z + 8.7))
    ps2fx.draw_cuboid_3d(frame, camera, (3.8, -0.7, camera.z + 4.6), (0.84, 0.34, 0.48), bench_mat["fill"], fog, edge=bench_mat["edge"])
    ps2fx.draw_cuboid_3d(frame, camera, (-3.4, -0.54, camera.z + 6.4), (1.12, 0.82, 0.66), (96, 160, 138), fog, edge=(52, 98, 78))
    _draw_supporting_people(frame, camera, fog, t, f, center_z=camera.z + 4.2, walking=True, scale_bias=0.88, lateral_bias=1.2)
    if {"greenhouse", "glass"} & tags:
        for z in (camera.z + 4.0, camera.z + 7.0, camera.z + 10.0):
            ps2fx.draw_quad_3d(frame, camera, [(-1.9, 0.26, z), (1.9, 0.26, z), (1.6, 1.44, z + 0.8), (-1.6, 1.44, z + 0.8)], (128, 214, 214), (70, 126, 122), fog, 0.18)
            ps2fx.draw_billboard_3d(frame, camera, (0.0, 1.12, z + 0.42), (2.9, 0.14), (224, 244, 248), (104, 156, 162), fog, 0.42)
        for side in (-1, 1):
            ps2fx.draw_quad_3d(frame, camera, [(side * 1.4, -0.9, camera.z + 2.2), (side * 4.7, -0.9, camera.z + 5.0), (side * 4.0, 1.42, camera.z + 10.0), (side * 1.0, 1.24, camera.z + 6.2)], (78, 170, 168), (54, 116, 112), fog, 0.18)
    ps2fx.debug_end_object()


def _render_water_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    h, w = frame.shape[:2]
    camera = mode._camera(t * 0.34, dolly=t * (0.09 + f["bass"] * 0.04), bob=0.014, orbit=0.36 + f["section"] * 0.22, f=f)
    fog = _world_fog((144, 202, 228), palette, 0.012)
    ps2fx.gradient_background(frame, (88, 184, 232), (156, 218, 240))
    horizon = int(h * 0.42)
    cv2.rectangle(frame, (0, horizon), (w, h), ps2fx.bgr((8, 88, 172)), -1)
    _draw_sun(frame, (int(w * 0.8), int(h * 0.2)), 30, (230, 246, 255), (252, 238, 204))
    ps2fx.debug_begin_object("water_world")
    water_mat = _mat((18, 102, 198), edge=(52, 126, 188))
    shadow_mat = _mat((8, 78, 154), edge=(26, 92, 146))
    terrace_mode = mode.name == "ocean_glass_terrace"
    if terrace_mode:
        fog = _world_fog((126, 188, 214), palette, 0.004)
        cv2.rectangle(frame, (0, int(h * 0.44)), (w, h), ps2fx.bgr((10, 86, 174)), -1)
        cv2.circle(frame, (int(w * 0.8), int(h * 0.18)), int(h * 0.1), ps2fx.bgr((250, 238, 204)), -1, lineType=cv2.LINE_AA)
        for idx, y in enumerate((0.48, 0.56, 0.64)):
            cv2.line(frame, (0, int(h * y)), (w, int(h * (y + 0.01 * math.sin(t * 0.3 + idx)))), ps2fx.bgr((182, 220, 244) if idx == 0 else (84, 144, 210)), 2 if idx == 0 else 1, lineType=cv2.LINE_AA)
        water_material = _material("polished_water", tint=(18, 116, 204))
        terrace_material = _material("glossy_floor", tint=(172, 224, 236))
        glass_material = _material("reflective_glass", tint=(104, 192, 224))
        ocean_quad = _platform_quad(camera.z + 1.0, camera.z + 17.4, 7.4, 6.5)
        ps2fx.draw_quad_3d(frame, camera, ocean_quad, water_material.fill, water_material.edge, fog, 1.0)
        _draw_quad_seams_3d(frame, camera, fog, ocean_quad, water_material.seam, water_material.edge, count=8, alpha=0.14)
        ps2fx.draw_quad_3d(frame, camera, [(-7.2, -1.02, camera.z + 3.3), (7.2, -1.02, camera.z + 3.3), (9.4, -1.02, camera.z + 20.2), (-9.4, -1.02, camera.z + 20.2)], (8, 84, 176), (38, 106, 164), fog, 1.0)
        ps2fx.draw_quad_3d(frame, camera, [(-7.2, -1.18, camera.z + 1.6), (7.2, -1.18, camera.z + 1.6), (9.1, -1.46, camera.z + 18.6), (-9.1, -1.46, camera.z + 18.6)], (40, 88, 136), (30, 70, 108), fog, 0.95)
        for band_idx, z in enumerate((camera.z + 5.6, camera.z + 8.8, camera.z + 12.6, camera.z + 16.0)):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.84, z), (8.4 - band_idx * 0.9, 0.12), (196, 228, 246), (66, 138, 196), fog, 0.42)
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.9, z + 0.12), (8.0 - band_idx * 0.92, 0.06), (234, 244, 250), (110, 174, 210), fog, 0.24)
        shadow_fill = (44, 92, 128)
        left_terrace = [(-6.6, -0.9, camera.z + 1.6), (-1.2, -0.9, camera.z + 1.6), (-0.2, -0.46, camera.z + 7.8), (-5.1, -0.46, camera.z + 5.1)]
        right_terrace = [(1.2, -0.9, camera.z + 1.6), (6.6, -0.9, camera.z + 1.6), (5.1, -0.46, camera.z + 5.1), (0.2, -0.46, camera.z + 7.8)]
        ps2fx.draw_quad_3d(frame, camera, left_terrace, terrace_material.fill, terrace_material.edge, fog, 0.98)
        ps2fx.draw_quad_3d(frame, camera, right_terrace, terrace_material.fill, terrace_material.edge, fog, 0.98)
        _draw_quad_seams_3d(frame, camera, fog, left_terrace, terrace_material.seam, terrace_material.edge, count=4, alpha=0.16)
        _draw_quad_seams_3d(frame, camera, fog, right_terrace, terrace_material.seam, terrace_material.edge, count=4, alpha=0.16)
        ps2fx.draw_quad_3d(frame, camera, [(-6.6, -1.02, camera.z + 1.7), (-1.2, -1.02, camera.z + 1.7), (-0.3, -1.26, camera.z + 7.7), (-5.4, -1.18, camera.z + 5.3)], shadow_fill, (34, 82, 114), fog, 0.94)
        ps2fx.draw_quad_3d(frame, camera, [(1.2, -1.02, camera.z + 1.7), (6.6, -1.02, camera.z + 1.7), (5.4, -1.18, camera.z + 5.3), (0.3, -1.26, camera.z + 7.7)], shadow_fill, (34, 82, 114), fog, 0.94)
        for side in (-1, 1):
            rail_quad = [(side * 1.8, -0.42, camera.z + 2.2), (side * 6.3, -0.42, camera.z + 1.9), (side * 5.8, 0.44, camera.z + 11.0), (side * 1.5, 0.44, camera.z + 8.1)]
            ps2fx.draw_quad_3d(frame, camera, rail_quad, glass_material.fill, glass_material.edge, fog, 0.24)
            _draw_quad_seams_3d(frame, camera, fog, rail_quad, glass_material.seam, glass_material.edge, count=4, alpha=0.12)
            ps2fx.draw_quad_3d(frame, camera, [(side * 1.8, -0.56, camera.z + 2.2), (side * 6.3, -0.56, camera.z + 1.9), (side * 5.9, -0.92, camera.z + 11.0), (side * 1.7, -0.92, camera.z + 8.1)], (52, 98, 132), (34, 68, 96), fog, 0.92)
            for idx in range(5):
                z = camera.z + 2.7 + idx * 2.25
                ps2fx.draw_cuboid_3d(frame, camera, (side * 5.9, 0.18, z), (0.16, 1.06, 0.16), (212, 236, 244), fog, edge=(82, 142, 170))
        for z in (camera.z + 3.2, camera.z + 6.6, camera.z + 10.4):
            ps2fx.draw_billboard_3d(frame, camera, (-6.0, -0.84, z), (1.18, 0.12), (230, 242, 248), (126, 182, 208), fog, 0.84)
            ps2fx.draw_billboard_3d(frame, camera, (6.0, -0.84, z + 0.36), (1.18, 0.12), (230, 242, 248), (126, 182, 208), fog, 0.84)
        ps2fx.draw_billboard_3d(frame, camera, (-6.2, 0.82, camera.z + 2.0), (0.7, 2.3), (106, 194, 224), (64, 126, 154), fog, 0.18)
        ps2fx.draw_billboard_3d(frame, camera, (6.0, 0.98, camera.z + 2.6), (0.56, 1.84), (134, 208, 230), (76, 138, 164), fog, 0.14)
        for z in (camera.z + 6.0, camera.z + 10.2, camera.z + 14.4):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.72, z), (1.4, 0.1), (226, 242, 248), (120, 184, 214), fog, 0.8)
        _draw_floor_stripes(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.4,
            far_z=camera.z + 16.2,
            width_near=6.9,
            width_far=6.0,
            y=-0.99,
            count=6,
            color=(104, 176, 206),
            edge=(60, 118, 146),
            alpha=0.16,
            thickness=0.18,
        )
        _draw_panel_rows_3d(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.3,
            far_z=camera.z + 15.8,
            width_near=6.8,
            width_far=5.9,
            y=-0.96,
            count=6,
            fill_a=(142, 198, 220),
            fill_b=(102, 160, 194),
            edge=(70, 120, 148),
            alpha=0.2,
            thickness=0.44,
        )
        for z in (camera.z + 4.8, camera.z + 9.4, camera.z + 14.2):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.96, z), (7.4, 0.06), (220, 236, 246), (98, 162, 198), fog, 0.26)
        ps2fx.debug_end_object()
        return
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.1, camera.z + 15.6, 6.6, 5.6), water_mat["fill"], water_mat["edge"], fog, 0.98)
    ps2fx.draw_quad_3d(frame, camera, [(-6.6, -1.02, camera.z + 1.2), (-2.8, -1.02, camera.z + 1.2), (-1.8, -0.92, camera.z + 4.2), (-5.4, -0.92, camera.z + 4.2)], shadow_mat["fill"], shadow_mat["edge"], fog, 0.92)
    ps2fx.draw_billboard_3d(frame, camera, (4.8, -0.52, camera.z + 2.0), (0.32, 1.76), (78, 176, 226), (44, 118, 176), fog, 0.82)
    if {"ocean", "terrace"} & tags:
        ps2fx.draw_quad_3d(frame, camera, [(-8.0, -1.02, camera.z + 4.4), (8.0, -1.02, camera.z + 4.4), (10.2, -1.02, camera.z + 20.0), (-10.2, -1.02, camera.z + 20.0)], (12, 98, 196), (46, 118, 182), fog, 1.0)
        for band_idx, z in enumerate((camera.z + 6.0, camera.z + 10.2, camera.z + 14.4)):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.86, z), (7.2 - band_idx * 1.0, 0.12), (214, 236, 248), (82, 162, 210), fog, 0.52)
    _draw_water_surface_bands(frame, camera, fog, t, center_z=camera.z + 5.6, width=3.8 if {"ocean", "terrace"} & tags else 3.0, depth=3.4 if {"ocean", "terrace"} & tags else 2.5, y=-0.9)
    _draw_splash_fountain(frame, camera, fog, t, f, center_z=camera.z + 5.6, width=3.9 if {"ocean", "terrace"} & tags else 3.1, pool_depth=3.4 if {"ocean", "terrace"} & tags else 2.5, waterfall=bool({"waterfall", "atrium"} & tags))
    _draw_side_fountain_arcs(frame, camera, fog, t, f, camera.z + 5.4, 2.6 if {"ocean", "terrace"} & tags else 2.1)
    for z in (camera.z + 3.0, camera.z + 5.4, camera.z + 8.2):
        ps2fx.draw_billboard_3d(frame, camera, (-3.8, -0.82, z), (1.4, 0.12), (204, 228, 244), (78, 154, 196), fog, 0.56)
    if not ({"ocean", "terrace"} & tags):
        _draw_supporting_people(frame, camera, fog, t, f, center_z=camera.z + 4.7, walking=False, scale_bias=0.8, lateral_bias=-2.55)
    if {"glass", "atrium"} & tags:
        for z in (camera.z + 3.2, camera.z + 6.4, camera.z + 9.6):
            ps2fx.draw_quad_3d(frame, camera, [(-4.6, -0.88, z), (-1.2, -0.12, z), (-0.9, 1.96, z + 1.16), (-4.3, 1.88, z + 1.16)], (68, 142, 184), (48, 112, 144), fog, 0.16)
            ps2fx.draw_quad_3d(frame, camera, [(1.2, -0.12, z), (4.6, -0.88, z), (4.3, 1.88, z + 1.16), (0.9, 1.96, z + 1.16)], (76, 154, 194), (52, 118, 150), fog, 0.16)
    ps2fx.debug_end_object()


def _render_space_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    camera = mode._camera(t * 0.3, dolly=t * (0.06 + f["bass"] * 0.02), bob=0.01, orbit=0.48 + f["section"] * 0.24, f=f)
    fog = _world_fog((86, 108, 162), palette, 0.01)
    _draw_space_backdrop(frame, t)
    ps2fx.debug_begin_object("space_world")
    station_mat = _mat((52, 82, 146), edge=(98, 128, 182))
    panel_mat = _mat((126, 154, 194), edge=(72, 96, 138))
    rail_mat = _mat((136, 170, 214), edge=(74, 104, 150))
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.5, camera.z + 12.0, 5.2, 3.8), station_mat["fill"], station_mat["edge"], fog, 0.94)
    _draw_floor_stripes(
        frame,
        camera,
        fog,
        near_z=camera.z + 2.2,
        far_z=camera.z + 11.1,
        width_near=4.9,
        width_far=3.5,
        y=-1.02,
        count=4,
        color=(84, 108, 162),
        edge=(56, 74, 114),
        alpha=0.18,
        thickness=0.22,
    )
    ps2fx.draw_billboard_3d(frame, camera, (-4.6, 0.42, camera.z + 1.8), (0.42, 2.0), panel_mat["fill"], panel_mat["edge"], fog, 0.78)
    _draw_walkway_rails(frame, camera, fog, camera.z + 1.8, camera.z + 10.8, 4.2, rail_mat["fill"], rail_mat["edge"])
    for idx, z in enumerate((camera.z + 3.6, camera.z + 6.8, camera.z + 10.2)):
        ps2fx.draw_quad_3d(frame, camera, [(-1.8 + idx * 0.64, 0.18, z), (0.1 + idx * 0.64, 0.18, z), (0.3 + idx * 0.64, 1.0, z + 0.64), (-1.6 + idx * 0.64, 1.0, z + 0.64)], (110, 150, 196), (72, 98, 142), fog, 0.34)
        ps2fx.draw_billboard_3d(frame, camera, (-0.8 + idx * 0.9, 0.78, z + 0.36), (1.8 + idx * 0.22, 0.12), (226, 242, 252), (134, 178, 206), fog, 0.56)
    for side in (-1, 1):
        ps2fx.draw_quad_3d(frame, camera, [(side * 0.4, 0.3, camera.z + 3.2), (side * 2.8, 1.24, camera.z + 5.6), (side * 2.2, 1.82, camera.z + 9.2), (side * 0.26, 0.46, camera.z + 6.0)], (126, 154, 196), (76, 100, 142), fog, 0.52)
        ps2fx.draw_cuboid_3d(frame, camera, (side * 4.8, 0.32, camera.z + 2.2), (0.34, 2.4, 0.34), (98, 120, 172), fog, edge=(62, 84, 126))
    for idx, z in enumerate((camera.z + 5.0, camera.z + 8.4, camera.z + 11.0)):
        x = -3.8 + idx * 3.8
        ps2fx.draw_cuboid_3d(frame, camera, (x, 0.96 + math.sin(t * 0.18 + idx) * 0.12, z), (0.7, 0.7, 0.7), (112, 128, 176), fog, edge=(74, 94, 136))
    if {"garden"} & tags:
        _draw_tree_cluster(frame, camera, fog, t, [(-1.7, -0.48, camera.z + 3.8), (1.6, -0.48, camera.z + 4.6), (-0.3, -0.48, camera.z + 6.0)], bright=True)
    ps2fx.debug_end_object()


def _render_people_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    h, w = frame.shape[:2]
    camera = mode._camera(t * 0.24, dolly=t * (0.05 + f["bass"] * 0.02), bob=0.01, orbit=0.26 + f["section"] * 0.14, f=f)
    fog = _world_fog((154, 196, 214), palette, 0.012)
    ps2fx.gradient_background(frame, (100, 190, 234), (160, 220, 238))
    _draw_sun(frame, (int(w * 0.84), int(h * 0.18)), 26, (238, 246, 255), (255, 238, 198))
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.0, camera.z + 15.0, 7.0, 5.4), (38, 88, 156), (72, 120, 184), fog, 1.0)
    _draw_floor_stripes(
        frame,
        camera,
        fog,
        near_z=camera.z + 2.0,
        far_z=camera.z + 12.5,
        width_near=5.8,
        width_far=4.1,
        y=-0.98,
        count=5,
        color=(86, 114, 176),
        edge=(64, 92, 144),
        alpha=0.18,
        thickness=0.18,
    )
    ps2fx.draw_quad_3d(frame, camera, [(-4.4, -1.0, camera.z + 3.8), (4.4, -1.0, camera.z + 3.8), (3.3, -1.0, camera.z + 9.4), (-3.3, -1.0, camera.z + 9.4)], (212, 92, 184), (118, 68, 126), fog, 0.28)
    ps2fx.draw_billboard_3d(frame, camera, (-5.0, 0.18, camera.z + 2.0), (0.54, 1.42), (72, 172, 224), (46, 112, 156), fog, 0.2)
    ps2fx.draw_billboard_3d(frame, camera, (4.8, 0.4, camera.z + 2.8), (0.44, 1.18), (64, 160, 216), (42, 102, 148), fog, 0.16)
    for z in (camera.z + 5.0, camera.z + 8.8):
        ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.58, z), (2.9, 0.16), (232, 242, 248), (108, 182, 208), fog, 0.48)
        ps2fx.draw_quad_3d(frame, camera, [(-1.6, 0.14, z), (1.6, 0.14, z), (1.2, 0.66, z + 0.54), (-1.2, 0.66, z + 0.54)], (90, 170, 202), (60, 118, 146), fog, 0.26)
    for idx, x in enumerate((-4.8, -2.5, 2.5, 4.8)):
        ps2fx.draw_cuboid_3d(frame, camera, (x, 0.02, camera.z + 5.0 + idx * 1.0), (0.18, 1.28, 0.18), (88, 198, 236), fog, edge=(60, 132, 172))
    for z, fill in ((camera.z + 3.2, (246, 86, 206)), (camera.z + 6.8, (86, 220, 252))):
        ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.9, z), (6.2, 0.16), fill, (64, 98, 150), fog, 0.48)
    role = mode._shot_role()
    if role == "pullback_people_skyline":
        ps2fx.draw_quad_3d(frame, camera, [(-5.2, -0.98, camera.z + 4.1), (5.2, -0.98, camera.z + 4.1), (4.2, -0.98, camera.z + 10.4), (-4.2, -0.98, camera.z + 10.4)], (54, 96, 164), (78, 118, 176), fog, 0.42)
        ps2fx.draw_billboard_3d(frame, camera, (-3.2, 0.34, camera.z + 5.2), (1.24, 0.76), (102, 214, 242), (62, 148, 184), fog, 0.74)
        ps2fx.draw_billboard_3d(frame, camera, (3.1, 0.42, camera.z + 6.8), (1.12, 0.68), (248, 128, 208), (132, 76, 126), fog, 0.7)
        _draw_supporting_people(frame, camera, fog, t, f, center_z=camera.z + 3.1, walking=bool({"walking", "commuters"} & tags), scale_bias=1.18, lateral_bias=0.0)
    else:
        _draw_people_cluster(frame, camera, fog, t, f, center_z=camera.z + 1.95, walking=bool({"walking", "commuters"} & tags))
    for idx in range(6):
        px = int(w * (0.12 + idx * 0.14))
        py = int(h * (0.78 + math.sin(t * 0.16 + idx) * 0.02))
        cv2.circle(frame, (px, py), 3 + idx % 2, ps2fx.bgr((220, 228, 244)), -1, lineType=cv2.LINE_AA)


def _render_cyber_city_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    camera = mode._camera(t * 0.38, dolly=t * (0.1 + f["bass"] * 0.04), bob=0.014, orbit=0.3 + f["section"] * 0.18, f=f)
    fog = _world_fog((104, 146, 186), palette, 0.016)
    skybridge_mode = mode.name == "white_civic_skybridge"
    if skybridge_mode:
        ps2fx.gradient_background(frame, (104, 196, 238), (188, 230, 242))
    else:
        ps2fx.gradient_background(frame, (38, 88, 152), (92, 162, 222))
    _draw_small_clouds(frame, t, count=2, top_band=0.12, color=(170, 208, 236))
    ps2fx.debug_begin_object("cyber_city_world")
    if skybridge_mode:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, int(h * 0.7)), (w, h), ps2fx.bgr((78, 170, 118)), -1)
        for idx, x in enumerate((0.08, 0.22, 0.76, 0.9)):
            cv2.rectangle(frame, (int(w * x), int(h * (0.54 - idx * 0.02))), (int(w * (x + 0.1)), int(h * 0.7)), ps2fx.bgr((88 + idx * 14, 156 + idx * 10, 202 + idx * 8)), -1)
        bridge_fill = (214, 232, 240)
        bridge_edge = (112, 144, 164)
        shadow_fill = (84, 108, 132)
        rail_fill = (114, 192, 220)
        deck = [(-7.2, -0.66, camera.z + 1.8), (7.2, -0.66, camera.z + 1.8), (5.1, -0.28, camera.z + 10.8), (-5.1, -0.28, camera.z + 10.8)]
        ps2fx.draw_quad_3d(frame, camera, deck, bridge_fill, bridge_edge, fog, 1.0)
        ps2fx.draw_quad_3d(frame, camera, [(-7.2, -0.96, camera.z + 1.8), (7.2, -0.96, camera.z + 1.8), (5.5, -1.26, camera.z + 10.7), (-5.5, -1.26, camera.z + 10.7)], shadow_fill, (56, 76, 98), fog, 0.98)
        ps2fx.draw_quad_3d(frame, camera, [(-7.4, -0.66, camera.z + 3.0), (-4.8, -0.66, camera.z + 3.0), (-4.3, -0.02, camera.z + 7.4), (-6.5, -0.02, camera.z + 6.2)], (174, 214, 228), (100, 136, 154), fog, 0.96)
        _draw_walkway_rails(frame, camera, fog, camera.z + 1.8, camera.z + 11.0, 6.2, rail_fill, (74, 126, 154))
        for idx, x in enumerate((-6.1, -3.2, 0.0, 3.2, 6.0)):
            z = camera.z + 2.4 + idx * 1.72
            ps2fx.draw_cuboid_3d(frame, camera, (x, -0.02, z), (0.3, 1.78, 0.3), (192, 220, 230), fog, edge=(98, 136, 154))
        for idx, z in enumerate((camera.z + 3.0, camera.z + 6.0, camera.z + 9.0)):
            ps2fx.draw_cuboid_3d(frame, camera, (-7.0, -0.82, z), (1.54, 0.44, 1.0), (88, 184, 126), fog, edge=(52, 110, 78))
            ps2fx.draw_cuboid_3d(frame, camera, (7.0, -0.82, z + 0.5), (1.78, 0.52, 1.12), (120, 194, 148), fog, edge=(62, 120, 88))
        ps2fx.draw_billboard_3d(frame, camera, (-6.6, 0.68, camera.z + 2.0), (0.62, 2.1), (136, 206, 232), (82, 134, 158), fog, 0.18)
        ps2fx.debug_end_object()
        return
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.2, camera.z + 16.0, 6.8, 5.0), (22, 58, 112), (72, 106, 154), fog, 0.98)
    ps2fx.draw_quad_3d(frame, camera, [(-1.3, -1.06, camera.z + 1.3), (1.3, -1.06, camera.z + 1.3), (0.9, -1.06, camera.z + 16.0), (-0.9, -1.06, camera.z + 16.0)], (10, 40, 88), (34, 78, 132), fog, 0.96)
    _draw_floor_stripes(
        frame,
        camera,
        fog,
        near_z=camera.z + 2.0,
        far_z=camera.z + 14.2,
        width_near=5.0,
        width_far=3.8,
        y=-1.01,
        count=5,
        color=(64, 104, 164),
        edge=(42, 74, 118),
        alpha=0.16,
        thickness=0.2,
    )
    _draw_walkway_rails(frame, camera, fog, camera.z + 1.4, camera.z + 14.0, 4.8, (118, 164, 210), (66, 102, 148))
    for z in (camera.z + 3.0, camera.z + 6.0, camera.z + 9.2, camera.z + 12.8):
        ps2fx.draw_quad_3d(frame, camera, [(-5.6, -0.16, z), (5.6, -0.16, z), (4.6, 0.28, z + 1.8), (-4.6, 0.28, z + 1.8)], (114, 208, 236), (74, 132, 166), fog, 0.84)
    for side in (-1, 1):
        for idx, z in enumerate((camera.z + 3.0, camera.z + 5.8, camera.z + 8.8, camera.z + 12.0)):
            ps2fx.draw_cuboid_3d(frame, camera, (side * (3.2 + idx * 0.18), 0.18 + idx * 0.24, z), (1.6 + idx * 0.22, 3.0 + idx * 0.4, 1.6), (40 + idx * 12, 92 + idx * 12, 148 + idx * 12), fog, edge=(70, 112, 160))
            if idx < 3:
                ps2fx.draw_billboard_3d(frame, camera, (side * (2.3 + idx * 0.26), 0.82 + idx * 0.12, z - 0.8), (1.3, 0.74), (246, 96, 202) if idx % 2 == 0 else (96, 224, 252), (92, 66, 116) if idx % 2 == 0 else (62, 132, 168), fog, 0.96)
    if {"market", "grid", "arcade"} & tags:
        for z in (camera.z + 4.0, camera.z + 7.0, camera.z + 10.0):
            for side in (-1, 1):
                ps2fx.draw_cuboid_3d(frame, camera, (side * 2.4, -0.64, z), (1.2, 0.88, 1.0), (58, 74, 94), fog, edge=(34, 46, 66))
                ps2fx.draw_billboard_3d(frame, camera, (side * 1.86, 0.44, z - 0.12), (1.42, 0.86), (246, 92, 198), (108, 52, 106), fog, 0.82)
    _draw_city_traffic(frame, camera, fog, t, (camera.z + 4.2, camera.z + 7.8, camera.z + 11.2))
    ps2fx.debug_end_object()


def _render_glass_world(mode, frame, t: float, f: dict[str, float], palette: Palette, tags: set[str]) -> None:
    camera = mode._camera(t * 0.28, dolly=t * (0.08 + f["bass"] * 0.03), bob=0.013, orbit=0.32 + f["section"] * 0.18, f=f)
    fog = _world_fog((154, 198, 218), palette, 0.01)
    greenhouse_mode = mode.name == "glass_greenhouse"
    elevator_mode = mode.name == "glass_elevator_shaft"
    if greenhouse_mode:
        ps2fx.gradient_background(frame, (96, 192, 236), (26, 96, 160))
    elif elevator_mode:
        ps2fx.gradient_background(frame, (92, 182, 226), (34, 106, 162))
    else:
        ps2fx.gradient_background(frame, (88, 184, 228), (34, 112, 182))
    ps2fx.debug_begin_object("glass_world")
    if greenhouse_mode:
        fog = _world_fog((126, 186, 208), palette, 0.004)
        floor_material = _material("glossy_floor", tint=(26, 102, 144))
        glass_material = _material("reflective_glass", tint=(76, 166, 208))
        roof_material = _material("chrome_edge", tint=(148, 214, 232))
        plant_stem = (76, 128, 86)
        plant_leaf = (44, 176, 86)
        base_quad = _platform_quad(camera.z + 1.0, camera.z + 15.2, 6.8, 5.6)
        ps2fx.draw_quad_3d(frame, camera, base_quad, floor_material.fill, floor_material.edge, fog, 1.0)
        _draw_quad_seams_3d(frame, camera, fog, base_quad, floor_material.seam, floor_material.edge, count=8, alpha=0.18)
        ps2fx.draw_quad_3d(frame, camera, [(-6.3, -1.0, camera.z + 1.8), (6.3, -1.0, camera.z + 1.8), (5.2, -1.0, camera.z + 13.6), (-5.2, -1.0, camera.z + 13.6)], (8, 86, 132), (30, 76, 118), fog, 0.62)
        ps2fx.draw_quad_3d(frame, camera, [(-5.8, -0.96, camera.z + 3.2), (5.8, -0.96, camera.z + 3.2), (4.4, 0.18, camera.z + 11.4), (-4.4, 0.18, camera.z + 11.4)], (22, 118, 148), (34, 88, 116), fog, 0.84)
        ps2fx.draw_quad_3d(frame, camera, [(-5.8, -1.18, camera.z + 3.2), (5.8, -1.18, camera.z + 3.2), (4.6, -1.46, camera.z + 11.4), (-4.6, -1.46, camera.z + 11.4)], (24, 74, 102), (22, 58, 82), fog, 0.94)
        for z in (camera.z + 3.0, camera.z + 6.0, camera.z + 9.0):
            left_glass = [(-5.8, -0.94, z), (-1.4, -0.04, z), (-0.9, 2.28, z + 1.34), (-5.2, 2.08, z + 1.34)]
            right_glass = [(1.4, -0.04, z), (5.8, -0.94, z), (5.2, 2.08, z + 1.34), (0.9, 2.28, z + 1.34)]
            roof_quad = [(-5.6, 1.9, z), (5.6, 1.9, z), (4.6, 2.56, z + 1.7), (-4.6, 2.56, z + 1.7)]
            ps2fx.draw_quad_3d(frame, camera, left_glass, glass_material.fill, glass_material.edge, fog, 0.28)
            ps2fx.draw_quad_3d(frame, camera, right_glass, glass_material.fill, glass_material.edge, fog, 0.28)
            ps2fx.draw_quad_3d(frame, camera, roof_quad, roof_material.fill, roof_material.edge, fog, 0.3)
            _draw_quad_seams_3d(frame, camera, fog, left_glass, glass_material.seam, glass_material.edge, count=4, alpha=0.12)
            _draw_quad_seams_3d(frame, camera, fog, right_glass, glass_material.seam, glass_material.edge, count=4, alpha=0.12)
            _draw_quad_seams_3d(frame, camera, fog, roof_quad, roof_material.seam, roof_material.edge, count=4, alpha=0.1)
            ps2fx.draw_quad_3d(frame, camera, [(-5.4, 1.72, z + 0.18), (5.4, 1.72, z + 0.18), (4.5, 1.98, z + 1.74), (-4.5, 1.98, z + 1.74)], (94, 128, 146), (60, 88, 106), fog, 0.4)
        for x in (-4.8, -2.4, 0.0, 2.4, 4.8):
            ps2fx.draw_cuboid_3d(frame, camera, (x, 0.1, camera.z + 3.9 + abs(x) * 0.34), (0.28, 2.7, 0.28), (192, 226, 236), fog, edge=(92, 138, 156))
        for idx, x in enumerate((-3.8, -2.0, -0.2, 1.8, 3.6)):
            z = camera.z + 4.0 + idx * 1.48
            ps2fx.draw_cuboid_3d(frame, camera, (x, -0.1, z), (0.26, 1.9, 0.26), plant_stem, fog, edge=(48, 86, 58))
            ps2fx.draw_cuboid_3d(frame, camera, (x + 0.02, 0.98, z + 0.06), (1.12, 1.26, 1.12), plant_leaf, fog, edge=(30, 98, 50))
            ps2fx.draw_cuboid_3d(frame, camera, (x - 0.08, 1.54, z), (0.62, 0.72, 0.62), (92, 212, 120), fog, edge=(42, 112, 64))
        for side in (-1, 1):
            ps2fx.draw_quad_3d(frame, camera, [(side * 2.1, -0.58, camera.z + 2.8), (side * 5.5, -0.58, camera.z + 2.3), (side * 5.0, 0.2, camera.z + 8.7), (side * 1.9, 0.2, camera.z + 6.8)], (62, 146, 176), (44, 96, 120), fog, 0.24)
        ps2fx.draw_cuboid_3d(frame, camera, (-5.8, -0.06, camera.z + 2.2), (0.9, 2.2, 0.9), (30, 158, 82), fog, edge=(26, 92, 54))
        ps2fx.draw_cuboid_3d(frame, camera, (5.6, 0.08, camera.z + 2.8), (0.84, 2.1, 0.84), (42, 164, 88), fog, edge=(28, 98, 56))
        _draw_floor_stripes(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.2,
            far_z=camera.z + 13.8,
            width_near=6.0,
            width_far=4.8,
            y=-0.99,
            count=6,
            color=(48, 126, 156),
            edge=(34, 88, 116),
            alpha=0.18,
            thickness=0.18,
        )
        _draw_panel_rows_3d(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.2,
            far_z=camera.z + 13.6,
            width_near=5.9,
            width_far=4.9,
            y=-0.95,
            count=6,
            fill_a=(62, 144, 170),
            fill_b=(32, 102, 132),
            edge=(34, 88, 112),
            alpha=0.24,
            thickness=0.4,
        )
        _draw_glass_reflection_bands(
            frame,
            camera,
            fog,
            [
                (-3.8, 1.02, camera.z + 4.1, 0.12, 2.3),
                (3.6, 0.94, camera.z + 6.5, 0.1, 2.0),
                (-1.2, 0.76, camera.z + 8.6, 0.08, 1.6),
            ],
            (230, 242, 248),
            (118, 170, 188),
            0.16,
        )
        for z in (camera.z + 4.6, camera.z + 7.8, camera.z + 11.2):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.9, z), (5.2, 0.08), (216, 232, 240), (96, 148, 172), fog, 0.3)
        _draw_near_pass_panels(
            frame,
            camera,
            fog,
            t,
            [
                (-5.8, 0.82, camera.z + 1.9, 0.82, 2.46, (92, 184, 216), (58, 120, 152), 0.14),
                (5.4, 1.08, camera.z + 2.5, 0.64, 1.96, (116, 196, 224), (66, 128, 156), 0.12),
            ],
        )
        _draw_material_billboard(frame, camera, fog, (-5.7, 0.92, camera.z + 2.0), (0.76, 2.34), glass_material, alpha_scale=0.76, seam_count=3)
        _draw_material_billboard(frame, camera, fog, (5.1, 1.0, camera.z + 2.56), (0.58, 1.86), glass_material, alpha_scale=0.68, seam_count=2)
        ps2fx.debug_end_object()
        return
    if elevator_mode:
        fog = _world_fog((132, 190, 212), palette, 0.004)
        shaft_material = _material("dark_space_metal", tint=(28, 106, 152))
        floor_material = _material("glossy_floor", tint=(56, 132, 174))
        rail_material = _material("chrome_edge", tint=(202, 232, 240))
        glass_material = _material("reflective_glass", tint=(86, 172, 208))
        cab_fill = (170, 222, 234)
        cab_shadow = (70, 120, 150)
        shaft_quad = _platform_quad(camera.z + 1.1, camera.z + 15.0, 6.4, 5.1)
        ps2fx.draw_quad_3d(frame, camera, shaft_quad, shaft_material.fill, shaft_material.edge, fog, 1.0)
        _draw_quad_seams_3d(frame, camera, fog, shaft_quad, shaft_material.seam, shaft_material.edge, count=7, alpha=0.16)
        ps2fx.draw_quad_3d(frame, camera, [(-6.2, -1.0, camera.z + 1.8), (6.2, -1.0, camera.z + 1.8), (4.9, -1.0, camera.z + 13.8), (-4.9, -1.0, camera.z + 13.8)], floor_material.fill, floor_material.edge, fog, 0.9)
        ps2fx.draw_quad_3d(frame, camera, [(-6.2, -1.18, camera.z + 1.8), (6.2, -1.18, camera.z + 1.8), (5.0, -1.46, camera.z + 13.8), (-5.0, -1.46, camera.z + 13.8)], (32, 82, 114), (24, 58, 82), fog, 0.94)
        for side in (-1, 1):
            glass_quad = [(side * 1.0, -0.88, camera.z + 2.0), (side * 5.4, -0.88, camera.z + 4.4), (side * 4.6, 2.3, camera.z + 10.8), (side * 0.8, 2.3, camera.z + 6.4)]
            ps2fx.draw_quad_3d(frame, camera, glass_quad, glass_material.fill, glass_material.edge, fog, 0.22)
            _draw_quad_seams_3d(frame, camera, fog, glass_quad, glass_material.seam, glass_material.edge, count=5, alpha=0.12)
            ps2fx.draw_quad_3d(frame, camera, [(side * 1.0, -0.94, camera.z + 2.2), (side * 5.0, -0.94, camera.z + 4.3), (side * 4.3, 0.16, camera.z + 10.3), (side * 0.9, 0.16, camera.z + 6.2)], (54, 126, 162), (38, 84, 110), fog, 0.24)
            _draw_material_billboard(frame, camera, fog, (side * 5.7, 0.9, camera.z + 2.0), (0.56, 2.52), glass_material, alpha_scale=0.66, seam_count=3)
            _draw_material_billboard(frame, camera, fog, (side * 3.1, 1.02, camera.z + 5.1), (0.28, 3.02), glass_material, alpha_scale=0.6, seam_count=3)
        ps2fx.draw_quad_3d(frame, camera, [(-2.8, -0.92, camera.z + 2.4), (2.8, -0.92, camera.z + 2.4), (1.2, -0.16, camera.z + 9.2), (-1.2, -0.16, camera.z + 9.2)], rail_material.fill, rail_material.edge, fog, 0.96)
        ps2fx.draw_quad_3d(frame, camera, [(-2.8, -1.08, camera.z + 2.4), (2.8, -1.08, camera.z + 2.4), (1.3, -1.28, camera.z + 9.2), (-1.3, -1.28, camera.z + 9.2)], (66, 104, 132), (44, 72, 98), fog, 0.92)
        _draw_floor_stripes(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.4,
            far_z=camera.z + 13.8,
            width_near=5.4,
            width_far=4.0,
            y=-0.99,
            count=5,
            color=(122, 174, 198),
            edge=(74, 120, 146),
            alpha=0.16,
            thickness=0.16,
        )
        _draw_panel_rows_3d(
            frame,
            camera,
            fog,
            near_z=camera.z + 2.3,
            far_z=camera.z + 13.6,
            width_near=5.3,
            width_far=4.1,
            y=-0.96,
            count=6,
            fill_a=(142, 194, 214),
            fill_b=(88, 144, 174),
            edge=(68, 116, 140),
            alpha=0.18,
            thickness=0.38,
        )
        _draw_glass_reflection_bands(
            frame,
            camera,
            fog,
            [
                (-4.8, 1.12, camera.z + 3.6, 0.1, 2.3),
                (4.6, 1.0, camera.z + 5.7, 0.08, 2.0),
                (0.0, 0.88, camera.z + 8.2, 0.06, 1.8),
            ],
            (236, 244, 248),
            (126, 170, 188),
            0.16,
        )
        for z in (camera.z + 4.2, camera.z + 7.6, camera.z + 10.8):
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.96, z), (4.2, 0.08), (222, 236, 244), (102, 156, 184), fog, 0.28)
        ps2fx.draw_quad_3d(frame, camera, [(-2.3, -0.82, camera.z + 4.0), (-0.5, -0.82, camera.z + 4.0), (-0.2, 0.78, camera.z + 6.2), (-1.9, 0.78, camera.z + 6.2)], cab_fill, (104, 152, 176), fog, 0.92)
        ps2fx.draw_quad_3d(frame, camera, [(-2.3, -0.98, camera.z + 4.0), (-0.5, -0.98, camera.z + 4.0), (-0.2, -1.12, camera.z + 6.2), (-1.9, -1.12, camera.z + 6.2)], cab_shadow, (54, 82, 108), fog, 0.96)
        ps2fx.draw_quad_3d(frame, camera, [(0.9, -0.72, camera.z + 6.8), (2.5, -0.72, camera.z + 6.8), (1.8, 0.6, camera.z + 8.6), (0.4, 0.6, camera.z + 8.6)], cab_fill, (104, 152, 176), fog, 0.9)
        _draw_near_pass_panels(
            frame,
            camera,
            fog,
            t,
            [
                (-5.4, 0.86, camera.z + 2.0, 0.76, 2.34, (98, 184, 214), (58, 118, 146), 0.12),
                (5.1, 1.08, camera.z + 2.52, 0.6, 1.82, (126, 198, 222), (70, 126, 152), 0.1),
            ],
        )
        ps2fx.debug_end_object()
        return
    ps2fx.draw_quad_3d(frame, camera, _platform_quad(camera.z + 1.2, camera.z + 14.0, 6.3, 5.0), (18, 84, 126), (46, 110, 146), fog, 0.98)
    ps2fx.draw_quad_3d(frame, camera, [(-6.0, -1.02, camera.z + 2.1), (6.0, -1.02, camera.z + 2.1), (4.8, -1.02, camera.z + 13.8), (-4.8, -1.02, camera.z + 13.8)], (14, 102, 176), (34, 94, 146), fog, 0.34)
    _draw_near_pass_panels(
        frame,
        camera,
        fog,
        t,
        [
            (-5.2, 0.62, camera.z + 1.95, 0.76, 2.2, (78, 166, 206), (46, 112, 148), 0.2),
            (4.9, 0.92, camera.z + 2.45, 0.52, 1.62, (112, 190, 222), (58, 122, 154), 0.16),
        ],
    )
    for z in (camera.z + 3.0, camera.z + 6.0, camera.z + 9.0):
        ps2fx.draw_quad_3d(frame, camera, [(-5.0, -0.9, z), (-1.3, -0.12, z), (-1.0, 1.96, z + 1.1), (-4.6, 1.9, z + 1.1)], (36, 118, 166), (50, 102, 130), fog, 0.18)
        ps2fx.draw_quad_3d(frame, camera, [(1.3, -0.12, z), (5.0, -0.9, z), (4.6, 1.9, z + 1.1), (1.0, 1.96, z + 1.1)], (42, 132, 178), (56, 110, 138), fog, 0.18)
        ps2fx.draw_quad_3d(frame, camera, [(-5.0, 1.72, z), (5.0, 1.72, z), (4.4, 2.1, z + 1.4), (-4.4, 2.1, z + 1.4)], (126, 188, 212), (78, 126, 152), fog, 0.18)
    for x in (-3.7, -1.25, 1.25, 3.7):
        ps2fx.draw_cuboid_3d(frame, camera, (x, 0.0, camera.z + 4.0 + (x * x) * 0.06), (0.3, 2.3, 0.3), (122, 166, 182), fog, edge=(72, 110, 126))
    ps2fx.draw_quad_3d(frame, camera, [(-3.4, -0.92, camera.z + 2.4), (-0.7, -0.56, camera.z + 5.9), (-0.1, -0.22, camera.z + 8.4), (-2.8, -0.6, camera.z + 4.9)], (114, 180, 208), (68, 120, 150), fog, 0.64)
    ps2fx.draw_quad_3d(frame, camera, [(0.7, -0.56, camera.z + 5.9), (3.4, -0.92, camera.z + 2.4), (2.8, -0.6, camera.z + 4.9), (0.1, -0.22, camera.z + 8.4)], (114, 180, 208), (68, 120, 150), fog, 0.64)
    if {"greenhouse"} & tags:
        for z in (camera.z + 3.8, camera.z + 6.8, camera.z + 9.8):
            roof_quad = [(-2.6, 1.44, z), (2.6, 1.44, z), (2.1, 2.18, z + 0.96), (-2.1, 2.18, z + 0.96)]
            ps2fx.draw_quad_3d(frame, camera, roof_quad, (126, 198, 214), (74, 126, 144), fog, 0.22)
            _draw_quad_seams_3d(frame, camera, fog, roof_quad, (218, 236, 242), (74, 126, 144), count=4, alpha=0.1)
        for x in (-2.6, -0.9, 0.9, 2.6):
            ps2fx.draw_cuboid_3d(frame, camera, (x, -0.06, camera.z + 5.4 + abs(x) * 0.35), (0.22, 1.9, 0.22), (118, 174, 136), fog, edge=(66, 112, 82))
            ps2fx.draw_cuboid_3d(frame, camera, (x, 0.84, camera.z + 5.5 + abs(x) * 0.35), (0.72, 1.0, 0.72), (58, 162, 90), fog, edge=(34, 96, 60))
        ps2fx.draw_cuboid_3d(frame, camera, (0.0, 0.18, camera.z + 7.6), (0.86, 2.8, 0.86), (82, 154, 194), fog, edge=(56, 110, 146))
        ps2fx.draw_billboard_3d(frame, camera, (0.0, 1.38, camera.z + 7.4), (1.24, 1.84), (88, 178, 204), (62, 120, 146), fog, 0.28)
        _draw_supporting_people(frame, camera, fog, t, f, center_z=camera.z + 4.9, walking=True, scale_bias=0.76, lateral_bias=-1.5)
    elif {"water", "atrium"} & tags:
        _draw_splash_fountain(frame, camera, fog, t, f, center_z=camera.z + 6.0, width=2.4, pool_depth=1.9, waterfall=bool({"waterfall"} & tags))
    else:
        ps2fx.draw_billboard_3d(frame, camera, (-2.4, 1.0, camera.z + 4.2), (1.3, 0.78), (76, 176, 214), (54, 116, 150), fog, 0.72)
        ps2fx.draw_billboard_3d(frame, camera, (2.5, 0.92, camera.z + 7.2), (1.3, 0.74), (146, 196, 210), (76, 116, 132), fog, 0.68)
    ps2fx.debug_end_object()
