from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


_DEBUG_STATE: dict[str, object] = {
    "enabled": False,
    "mode": "",
    "camera_z": 0.0,
    "current_object": None,
    "objects": {},
    "polygons": 0,
}

_RENDER_STYLE: dict[str, float] = {
    "line_scale": 1.0,
}


def bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def set_render_style(*, line_scale: float = 1.0) -> None:
    _RENDER_STYLE["line_scale"] = float(np.clip(line_scale, 0.45, 2.0))


def scaled_line_thickness(base: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(float(base) * float(_RENDER_STYLE["line_scale"]))))


def gradient_background(frame: np.ndarray, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_arr = np.array(bgr(top), dtype=np.float32)
    bottom_arr = np.array(bgr(bottom), dtype=np.float32)
    grad = top_arr * (1.0 - ramp) + bottom_arr * ramp
    frame[:] = np.repeat(grad[:, None, :], width, axis=1).astype(np.uint8)


def additive_blend(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    mixed = base.astype(np.float32) + layer.astype(np.float32) * float(alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def alpha_blend(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0.0)


def vignette(frame: np.ndarray, strength: float = 0.35) -> None:
    h, w = frame.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x + y * y)
    mask = 1.0 - np.clip(radius, 0.0, 1.2) * strength
    frame[:] = np.clip(frame.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def scanlines(frame: np.ndarray, intensity: float = 0.12) -> None:
    frame[::2] = np.clip(frame[::2].astype(np.float32) * (1.0 - intensity), 0, 255).astype(np.uint8)


def fog_overlay(frame: np.ndarray, amount: float, color: tuple[int, int, int]) -> None:
    if amount <= 0.0:
        return
    h, w = frame.shape[:2]
    noise = cv2.GaussianBlur(np.random.randint(0, 255, (h, w), dtype=np.uint8), (0, 0), 15)
    fog = np.zeros_like(frame)
    fog[:] = bgr(color)
    fog = cv2.addWeighted(fog, 0.65, cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR), 0.35, 0.0)
    frame[:] = alpha_blend(frame, fog, min(0.45, amount))


def apply_bloom(frame: np.ndarray, strength: float, threshold: float = 132.0) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 2.0))
    if strength <= 0.001:
        return frame
    glow_source = np.clip(frame.astype(np.float32) - float(threshold), 0.0, 255.0).astype(np.uint8)
    bloom = cv2.GaussianBlur(glow_source, (0, 0), 1.2 + strength * 1.8)
    mixed = cv2.addWeighted(frame, 1.0, bloom, 0.06 + strength * 0.12, 0.0)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def apply_exposure(frame: np.ndarray, exposure: float) -> np.ndarray:
    adjusted = np.clip(frame.astype(np.float32) * float(exposure), 0, 255)
    return adjusted.astype(np.uint8)


def apply_tone_curve(frame: np.ndarray, gamma: float = 0.94, shoulder: float = 1.7) -> np.ndarray:
    data = frame.astype(np.float32) / 255.0
    data = np.power(np.clip(data, 0.0, 1.0), float(gamma))
    data = data / (1.0 + np.maximum(0.0, data - 0.72) * float(shoulder))
    return np.clip(data * 255.0, 0, 255).astype(np.uint8)


def add_dither(frame: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.001:
        return frame
    pattern = np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=np.float32,
    )
    tiled = np.tile(pattern, (frame.shape[0] // 4 + 1, frame.shape[1] // 4 + 1))[: frame.shape[0], : frame.shape[1]]
    noise = (tiled - 7.5) * (1.8 + amount * 2.2)
    dithered = frame.astype(np.float32) + noise[..., None]
    return np.clip(dithered, 0, 255).astype(np.uint8)


def apply_jitter(frame: np.ndarray, t: float, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 3.0))
    if amount <= 0.001:
        return frame
    dx = int(round(math.sin(t * 11.0) * amount))
    dy = int(round(math.cos(t * 7.0) * amount))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def draw_glow_circle(canvas: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: float) -> None:
    if radius <= 1:
        return
    for scale, weight in ((2.4, 0.08), (1.8, 0.16), (1.2, 0.34), (1.0, 0.58)):
        layer = np.zeros_like(canvas)
        cv2.circle(layer, center, max(1, int(radius * scale)), bgr(color), -1, lineType=cv2.LINE_AA)
        canvas[:] = additive_blend(canvas, layer, alpha * weight)


def draw_ring_outline(canvas: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], thickness: int = 2, glow: float = 0.2) -> None:
    if radius <= 1:
        return
    if glow > 0.001:
        layer = np.zeros_like(canvas)
        cv2.circle(layer, center, radius, bgr(color), max(1, scaled_line_thickness(thickness + 1)), lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.4)
        canvas[:] = additive_blend(canvas, layer, glow)
    cv2.circle(canvas, center, radius, bgr(color), max(1, scaled_line_thickness(thickness)), lineType=cv2.LINE_AA)


def draw_perspective_rails(canvas: np.ndarray, color: tuple[int, int, int], t: float, speed: float, lanes: int = 3) -> None:
    h, w = canvas.shape[:2]
    horizon = int(h * 0.58)
    offsets = np.linspace(-0.32, 0.32, lanes)
    for offset in offsets:
        x_bottom = int(w * (0.5 + offset))
        x_top = int(w * (0.5 + offset * 0.18))
        cv2.line(canvas, (x_bottom, h), (x_top, horizon), bgr(color), scaled_line_thickness(1.6), lineType=cv2.LINE_AA)
    for i in range(12):
        prog = ((i / 12.0) + t * speed) % 1.0
        y = int(horizon + prog * prog * (h - horizon))
        cv2.line(canvas, (int(w * 0.18), y), (int(w * 0.82), y), bgr(color), 1, lineType=cv2.LINE_AA)


def draw_ocean_bands(canvas: np.ndarray, sky: tuple[int, int, int], sea: tuple[int, int, int], foam: tuple[int, int, int], t: float) -> None:
    h, w = canvas.shape[:2]
    horizon = int(h * 0.46)
    gradient_background(canvas, sky, sea)
    cv2.rectangle(canvas, (0, horizon), (w, h), bgr(sea), -1)
    for i in range(10):
        y = int(horizon + i * (h - horizon) / 10)
        wobble = int(math.sin(t * 0.2 + i * 0.6) * 6)
        cv2.line(canvas, (0, y + wobble), (w, y - wobble), bgr(foam), 1, lineType=cv2.LINE_AA)


def draw_speed_lines(canvas: np.ndarray, color: tuple[int, int, int], t: float, strength: float) -> None:
    h, w = canvas.shape[:2]
    layer = np.zeros_like(canvas)
    count = max(8, int(18 + strength * 16))
    for i in range(count):
        x = int(((i * 37 + int(t * 40)) % (w + 80)) - 40)
        y = int((i / count) * h)
        cv2.line(layer, (x, y), (x + int(16 + strength * 30), y), bgr(color), 1, lineType=cv2.LINE_AA)
    canvas[:] = additive_blend(canvas, layer, 0.32)


def draw_horizon_grid(canvas: np.ndarray, color: tuple[int, int, int], beat: float, offset: float) -> None:
    h, w = canvas.shape[:2]
    horizon = int(h * 0.58)
    spacing = max(28, int(54 - beat * 18))
    line_color = bgr(color)
    for i in range(22):
        y = horizon + int(((i * spacing + offset * spacing) ** 1.03) * 0.55)
        if y >= h:
            break
        cv2.line(canvas, (0, y), (w, y), line_color, 1, lineType=cv2.LINE_AA)
    for x in np.linspace(-0.2, 1.2, 18):
        p0 = (int(w * x), h)
        p1 = (w // 2 + int((x - 0.5) * w * 0.08), horizon)
        cv2.line(canvas, p0, p1, line_color, 1, lineType=cv2.LINE_AA)


def draw_reflection_streaks(canvas: np.ndarray, color: tuple[int, int, int], strength: float, t: float) -> None:
    h, w = canvas.shape[:2]
    layer = np.zeros_like(canvas)
    count = 12
    for i in range(count):
        x = int((i / count) * w + math.sin(t * 0.35 + i) * w * 0.06)
        y0 = int(h * 0.62 + (i % 3) * 14)
        y1 = min(h - 1, y0 + int(h * (0.18 + 0.18 * strength)))
        cv2.line(layer, (x, y0), (x + int(strength * 18), y1), bgr(color), scaled_line_thickness(1.5), lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 7)
    canvas[:] = additive_blend(canvas, layer, min(0.75, 0.22 + strength * 0.4))


def draw_ui_panels(canvas: np.ndarray, color: tuple[int, int, int], t: float, energy: float) -> None:
    h, w = canvas.shape[:2]
    layer = np.zeros_like(canvas)
    boxes = [
        (int(w * 0.06), int(h * 0.08), int(w * 0.22), int(h * 0.12)),
        (int(w * 0.72), int(h * 0.1), int(w * 0.2), int(h * 0.14)),
        (int(w * 0.12), int(h * 0.78), int(w * 0.24), int(h * 0.1)),
    ]
    for idx, (x, y, bw, bh) in enumerate(boxes):
        wobble = int(math.sin(t * 0.6 + idx) * 6)
        cv2.rectangle(layer, (x, y + wobble), (x + bw, y + bh + wobble), bgr(color), scaled_line_thickness(1.0), lineType=cv2.LINE_AA)
        for row in range(4):
            yline = y + wobble + 10 + row * 16
            cv2.line(layer, (x + 12, yline), (x + bw - 12, yline), bgr(color), 1, lineType=cv2.LINE_AA)
    canvas[:] = additive_blend(canvas, cv2.GaussianBlur(layer, (0, 0), 1.2), 0.24 + energy * 0.2)


def draw_particles(canvas: np.ndarray, color: tuple[int, int, int], t: float, density: float, vertical_bias: float = -1.0) -> None:
    h, w = canvas.shape[:2]
    count = max(10, int(120 * density))
    layer = np.zeros_like(canvas)
    for i in range(count):
        px = int((math.sin(i * 91.73 + t * 0.2) * 0.5 + 0.5) * w)
        py = int(((i * 43) % h + t * vertical_bias * (10 + i % 7)) % h)
        radius = 1 + (i % 3)
        cv2.circle(layer, (px, py), radius, bgr(color), -1, lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.5)
    canvas[:] = additive_blend(canvas, layer, 0.35)


def debug_reset(mode_name: str = "", camera_z: float = 0.0, enabled: bool = False) -> None:
    _DEBUG_STATE["enabled"] = enabled
    _DEBUG_STATE["mode"] = mode_name
    _DEBUG_STATE["camera_z"] = camera_z
    _DEBUG_STATE["current_object"] = None
    _DEBUG_STATE["objects"] = {}
    _DEBUG_STATE["polygons"] = 0


def debug_set_camera(camera_z: float) -> None:
    _DEBUG_STATE["camera_z"] = camera_z


def debug_begin_object(label: str) -> None:
    if _DEBUG_STATE["enabled"]:
        _DEBUG_STATE["current_object"] = label


def debug_end_object() -> None:
    if _DEBUG_STATE["enabled"]:
        _DEBUG_STATE["current_object"] = None


def debug_snapshot() -> dict[str, object]:
    objects = _DEBUG_STATE["objects"]
    visible = sum(1 for bbox in objects.values() if bbox is not None)
    return {
        "mode": _DEBUG_STATE["mode"],
        "camera_z": float(_DEBUG_STATE["camera_z"]),
        "visible_polygons": int(_DEBUG_STATE["polygons"]),
        "visible_objects": int(visible),
        "boxes": {label: bbox for label, bbox in objects.items() if bbox is not None},
    }


def _debug_record_polygon(points: list[tuple[int, int]]) -> None:
    if not _DEBUG_STATE["enabled"] or not points:
        return
    _DEBUG_STATE["polygons"] = int(_DEBUG_STATE["polygons"]) + 1
    label = _DEBUG_STATE["current_object"]
    if not label:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    objects: dict[str, tuple[int, int, int, int] | None] = _DEBUG_STATE["objects"]  # type: ignore[assignment]
    prev = objects.get(label)
    if prev is None:
        objects[label] = bbox
    else:
        objects[label] = (
            min(prev[0], bbox[0]),
            min(prev[1], bbox[1]),
            max(prev[2], bbox[2]),
            max(prev[3], bbox[3]),
        )


@dataclass(frozen=True)
class Camera3D:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    fov: float = 1.05


def camera_motion(t: float, *, dolly: float = 0.0, bob: float = 0.0, orbit: float = 0.0) -> Camera3D:
    return Camera3D(
        x=math.sin(t * 0.18 + orbit) * 0.12,
        y=math.sin(t * 0.7 + orbit * 0.3) * bob,
        z=dolly,
        yaw=math.sin(t * 0.16 + orbit) * 0.08,
        pitch=math.cos(t * 0.11 + orbit * 0.5) * 0.035,
    )


def shade_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    factor = float(np.clip(factor, 0.0, 2.0))
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def fog_color(color: tuple[int, int, int], fog: tuple[int, int, int], depth: float, density: float = 0.08) -> tuple[int, int, int]:
    blend = 1.0 - math.exp(-max(0.0, depth) * density)
    return lerp_color(color, fog, blend)


def _camera_space_point(camera: Camera3D, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x = point[0] - camera.x
    y = point[1] - camera.y
    z = point[2] - camera.z

    cy = math.cos(camera.yaw)
    sy = math.sin(camera.yaw)
    x, z = x * cy - z * sy, x * sy + z * cy

    cp = math.cos(camera.pitch)
    sp = math.sin(camera.pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    return x, y, z


def _project_camera_point(point: tuple[float, float, float], width: int, height: int, fov: float) -> tuple[int, int]:
    scale = min(width, height) * fov / point[2]
    sx = int(width * 0.5 + point[0] * scale)
    sy = int(height * 0.5 - point[1] * scale)
    return sx, sy


def _clip_polygon_to_near_plane(points: list[tuple[float, float, float]], near_z: float = 0.05) -> list[tuple[float, float, float]]:
    if not points:
        return []
    result: list[tuple[float, float, float]] = []
    for current, previous in zip(points, [points[-1], *points[:-1]]):
        prev_inside = previous[2] > near_z
        curr_inside = current[2] > near_z
        if curr_inside != prev_inside:
            dz = current[2] - previous[2]
            if abs(dz) > 1e-6:
                t = (near_z - previous[2]) / dz
                result.append(
                    (
                        previous[0] + (current[0] - previous[0]) * t,
                        previous[1] + (current[1] - previous[1]) * t,
                        near_z,
                    )
                )
        if curr_inside:
            result.append(current)
    return result


def _clip_line_to_near_plane(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    near_z: float = 0.05,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    a = start
    b = end
    if a[2] <= near_z and b[2] <= near_z:
        return None
    if a[2] > near_z and b[2] > near_z:
        return a, b
    dz = b[2] - a[2]
    if abs(dz) <= 1e-6:
        return None
    t = (near_z - a[2]) / dz
    mid = (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        near_z,
    )
    if a[2] <= near_z:
        return mid, b
    return a, mid


def project_point(camera: Camera3D, point: tuple[float, float, float], width: int, height: int) -> tuple[tuple[int, int] | None, float]:
    x, y, z = _camera_space_point(camera, point)
    if z <= 0.05:
        return None, z

    return _project_camera_point((x, y, z), width, height, camera.fov), z


def project_points(camera: Camera3D, points: list[tuple[float, float, float]], width: int, height: int) -> tuple[list[tuple[int, int]] | None, float]:
    projected: list[tuple[int, int]] = []
    depths: list[float] = []
    for point in points:
        screen, depth = project_point(camera, point, width, height)
        if screen is None:
            return None, depth
        projected.append(screen)
        depths.append(depth)
    return projected, float(sum(depths) / max(1, len(depths)))


def draw_shaded_polygon(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int],
    edge: tuple[int, int, int] | None = None,
    alpha: float = 1.0,
) -> None:
    polygon = np.array(points, dtype=np.int32)
    if alpha >= 0.999:
        cv2.fillConvexPoly(canvas, polygon, bgr(fill), lineType=cv2.LINE_AA)
    else:
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
        layer = np.zeros_like(canvas)
        cv2.fillConvexPoly(layer, polygon, bgr(fill), lineType=cv2.LINE_AA)
        mask_f = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
        canvas[:] = np.clip(canvas.astype(np.float32) * (1.0 - mask_f) + layer.astype(np.float32) * mask_f, 0, 255).astype(np.uint8)
    _debug_record_polygon(points)
    if edge is not None:
        edge_color = lerp_color(edge, fill, 0.28)
        cv2.polylines(canvas, [polygon], True, bgr(edge_color), scaled_line_thickness(1.35), lineType=cv2.LINE_AA)


def draw_quad_3d(
    canvas: np.ndarray,
    camera: Camera3D,
    points_3d: list[tuple[float, float, float]],
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    fog: tuple[int, int, int],
    alpha: float = 1.0,
) -> float | None:
    camera_points = [_camera_space_point(camera, point) for point in points_3d]
    clipped = _clip_polygon_to_near_plane(camera_points)
    if len(clipped) < 3:
        return None
    pts2d = [_project_camera_point(point, canvas.shape[1], canvas.shape[0], camera.fov) for point in clipped]
    depth = float(sum(point[2] for point in clipped) / len(clipped))
    draw_shaded_polygon(canvas, pts2d, fog_color(fill, fog, depth), fog_color(edge, fog, depth * 0.8), alpha)
    return depth


def draw_billboard_3d(
    canvas: np.ndarray,
    camera: Camera3D,
    center: tuple[float, float, float],
    size: tuple[float, float],
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    fog: tuple[int, int, int],
    alpha: float = 1.0,
) -> float | None:
    cx, cy, cz = center
    half_w, half_h = size[0] * 0.5, size[1] * 0.5
    points = [
        (cx - half_w, cy - half_h, cz),
        (cx + half_w, cy - half_h, cz),
        (cx + half_w, cy + half_h, cz),
        (cx - half_w, cy + half_h, cz),
    ]
    return draw_quad_3d(canvas, camera, points, fill, edge, fog, alpha)


def draw_cuboid_3d(
    canvas: np.ndarray,
    camera: Camera3D,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[int, int, int],
    fog: tuple[int, int, int],
    edge: tuple[int, int, int] | None = None,
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    corners = {
        "nbl": (cx - sx, cy - sy, cz - sz),
        "nbr": (cx + sx, cy - sy, cz - sz),
        "ntl": (cx - sx, cy + sy, cz - sz),
        "ntr": (cx + sx, cy + sy, cz - sz),
        "fbl": (cx - sx, cy - sy, cz + sz),
        "fbr": (cx + sx, cy - sy, cz + sz),
        "ftl": (cx - sx, cy + sy, cz + sz),
        "ftr": (cx + sx, cy + sy, cz + sz),
    }
    faces = [
        (["nbl", "nbr", "ntr", "ntl"], shade_color(color, 0.94)),
        (["nbl", "nbr", "fbr", "fbl"], shade_color(color, 0.72)),
        (["nbr", "ntr", "ftr", "fbr"], shade_color(color, 0.58)),
        (["fbl", "fbr", "ftr", "ftl"], shade_color(color, 0.48)),
        (["ntl", "ntr", "ftr", "ftl"], shade_color(color, 1.08)),
        (["nbl", "ntl", "ftl", "fbl"], shade_color(color, 0.82)),
    ]
    draw_faces: list[tuple[float, list[tuple[int, int]], tuple[int, int, int], tuple[int, int, int] | None]] = []
    for keys, face_color in faces:
        camera_points = [_camera_space_point(camera, corners[key]) for key in keys]
        clipped = _clip_polygon_to_near_plane(camera_points)
        if len(clipped) >= 3:
            pts2d = [_project_camera_point(point, canvas.shape[1], canvas.shape[0], camera.fov) for point in clipped]
            depth = float(sum(point[2] for point in clipped) / len(clipped))
            draw_faces.append((depth, pts2d, face_color, fog_color(edge or face_color, fog, depth * 0.9) if edge else None))
    for depth, pts2d, face_color, edge_color in sorted(draw_faces, key=lambda item: item[0], reverse=True):
        draw_shaded_polygon(canvas, pts2d, fog_color(face_color, fog, depth), edge_color, 1.0)


def draw_floor_grid_3d(
    canvas: np.ndarray,
    camera: Camera3D,
    color: tuple[int, int, int],
    fog: tuple[int, int, int],
    x_range: tuple[float, float] = (-5.0, 5.0),
    z_range: tuple[float, float] = (2.0, 22.0),
    spacing: float = 1.0,
    y: float = -1.2,
) -> None:
    x = x_range[0]
    while x <= x_range[1] + 1e-6:
        clipped = _clip_line_to_near_plane(
            _camera_space_point(camera, (x, y, z_range[0])),
            _camera_space_point(camera, (x, y, z_range[1])),
        )
        if clipped is not None:
            start = _project_camera_point(clipped[0], canvas.shape[1], canvas.shape[0], camera.fov)
            end = _project_camera_point(clipped[1], canvas.shape[1], canvas.shape[0], camera.fov)
            depth = (clipped[0][2] + clipped[1][2]) * 0.5
            cv2.line(canvas, start, end, bgr(fog_color(color, fog, depth)), scaled_line_thickness(1.0), lineType=cv2.LINE_AA)
        x += spacing
    z = z_range[0]
    while z <= z_range[1] + 1e-6:
        clipped = _clip_line_to_near_plane(
            _camera_space_point(camera, (x_range[0], y, z)),
            _camera_space_point(camera, (x_range[1], y, z)),
        )
        if clipped is not None:
            start = _project_camera_point(clipped[0], canvas.shape[1], canvas.shape[0], camera.fov)
            end = _project_camera_point(clipped[1], canvas.shape[1], canvas.shape[0], camera.fov)
            depth = (clipped[0][2] + clipped[1][2]) * 0.5
            cv2.line(canvas, start, end, bgr(fog_color(color, fog, depth)), scaled_line_thickness(1.0), lineType=cv2.LINE_AA)
        z += spacing


def draw_corridor_planes(
    canvas: np.ndarray,
    camera: Camera3D,
    wall_color: tuple[int, int, int],
    floor_color: tuple[int, int, int],
    ceil_color: tuple[int, int, int],
    fog: tuple[int, int, int],
    width: float = 4.2,
    height: float = 2.4,
    near_z: float = 1.2,
    far_z: float = 18.0,
) -> None:
    draw_quad_3d(canvas, camera, [(-width, -height, near_z), (width, -height, near_z), (width, -height, far_z), (-width, -height, far_z)], floor_color, shade_color(floor_color, 1.1), fog)
    draw_quad_3d(canvas, camera, [(-width, height, near_z), (width, height, near_z), (width, height, far_z), (-width, height, far_z)], ceil_color, shade_color(ceil_color, 1.05), fog, 0.95)
    draw_quad_3d(canvas, camera, [(-width, -height, near_z), (-width, height, near_z), (-width, height, far_z), (-width, -height, far_z)], wall_color, shade_color(wall_color, 1.1), fog)
    draw_quad_3d(canvas, camera, [(width, -height, near_z), (width, height, near_z), (width, height, far_z), (width, -height, far_z)], shade_color(wall_color, 0.82), shade_color(wall_color, 1.0), fog)


def draw_portal_ring_3d(
    canvas: np.ndarray,
    camera: Camera3D,
    center: tuple[float, float, float],
    radius: float,
    color: tuple[int, int, int],
    fog: tuple[int, int, int],
    segments: int = 32,
    y_scale: float = 1.0,
    thickness: int = 2,
) -> float | None:
    pts3d = []
    for i in range(segments):
        ang = i / segments * math.tau
        pts3d.append((center[0] + math.cos(ang) * radius, center[1] + math.sin(ang) * radius * y_scale, center[2]))
    pts2d, depth = project_points(camera, pts3d, canvas.shape[1], canvas.shape[0])
    if pts2d is None:
        return None
    cv2.polylines(canvas, [np.array(pts2d, dtype=np.int32)], True, bgr(fog_color(color, fog, depth)), max(1, scaled_line_thickness(thickness)), lineType=cv2.LINE_AA)
    _debug_record_polygon(pts2d)
    return depth
