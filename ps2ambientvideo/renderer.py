from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import ps2fx
from .audio import analyze_audio
from .palettes import PALETTES
from .ps2fx import (
    Camera3D,
    add_dither,
    additive_blend,
    apply_bloom,
    apply_exposure,
    apply_jitter,
    apply_tone_curve,
    debug_begin_object,
    debug_end_object,
    debug_reset,
    debug_set_camera,
    debug_snapshot,
    draw_billboard_3d,
    draw_corridor_planes,
    draw_cuboid_3d,
    draw_particles,
    draw_floor_grid_3d,
    draw_portal_ring_3d,
    draw_reflection_streaks,
    draw_ui_panels,
    fog_overlay,
    scanlines,
    set_render_style,
    vignette,
)
from .timeline import active_modes, build_modes, build_timeline, debug_targets_for_timeline, normalize_preset_name, palette_for
from .blender_backend import render_video_blender


def _assert_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required on PATH to encode MP4 output.")
    return ffmpeg


def _global_layers(
    width: int,
    height: int,
    t: float,
    features: dict[str, float],
    palette_name: str,
    fog_strength: float,
    particle_scale: float,
    ui_scale: float,
    reflection_scale: float,
    fog_tint: tuple[int, int, int] | None = None,
    scan_fog_bias: float = 1.0,
    transition_mix: float = 0.0,
    scene_grammar: str = "worlds",
    aesthetic: str = "frutiger_cyber",
) -> np.ndarray:
    from .palettes import PALETTES  # local import keeps public API small

    palette = PALETTES[palette_name]
    layer = np.zeros((height, width, 3), dtype=np.uint8)
    worlds_clean = scene_grammar == "worlds" and aesthetic == "frutiger_cyber"
    particle_amount = particle_scale * (0.06 + features["highs"] * 0.08) if worlds_clean else particle_scale * (0.1 + features["highs"] * 0.18)
    ui_amount = ui_scale * features["energy"] * (0.05 if worlds_clean else 0.12)
    fog_amount = (0.004 + features["bass"] * 0.015) if worlds_clean else (0.02 + features["bass"] * 0.05)
    draw_particles(layer, palette.glow, t, particle_amount, vertical_bias=-0.18)
    draw_ui_panels(layer, palette.ui, t * 0.8, ui_amount)
    if fog_strength > 0.001 and fog_amount > 0.001:
        fog_overlay(layer, fog_amount * fog_strength * scan_fog_bias, fog_tint or palette.glow)
    if reflection_scale > 0.001:
        streak_scale = reflection_scale * ((0.03 + features["highs"] * 0.06) if worlds_clean else (0.08 + features["highs"] * 0.12))
        draw_reflection_streaks(layer, palette.ui, streak_scale, t * 0.72)
    if transition_mix > 0.02 and scene_grammar != "worlds":
        draw_reflection_streaks(layer, palette.accent, reflection_scale * (0.12 + transition_mix * 0.2 + features["bass"] * 0.08), t)
        draw_portal_ring_3d(layer, Camera3D(0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 3.6 + transition_mix * 0.5), 1.05 + transition_mix * 0.35, palette.glow, palette.mid, y_scale=0.72, thickness=2)
    return layer


def _resolve_render_look(
    preset: str,
    render_profile: str,
    aesthetic: str,
    bloom_strength: float,
    fog_strength: float,
    exposure: float,
    line_scale: float | None,
) -> dict[str, float | tuple[int, int, int]]:
    is_qa = render_profile == "qa"
    is_depth_showcase = preset == "depth_showcase"
    is_geometry_calibration = preset == "geometry_calibration"
    if aesthetic == "crt_dark":
        look = {
            "particle_scale": 0.75 if is_qa else 1.0,
            "ui_scale": 0.8 if is_qa else 1.0,
            "persistent_alpha": 0.22 if is_qa else 0.28,
            "local_fog_strength": fog_strength * (0.86 if is_qa else 1.0),
            "local_bloom_strength": bloom_strength * (0.9 if is_qa else 1.0),
            "local_exposure": exposure + (0.06 if is_qa else 0.0),
            "scanline_intensity": 0.1 if is_qa else 0.12,
            "vignette_strength": 0.24 if is_qa else 0.28,
            "dither_strength": 0.38 if is_depth_showcase else 0.5,
            "jitter_strength": 0.4,
            "reflection_scale": 1.0,
            "scan_fog_bias": 0.8 if is_qa else 1.0,
            "brightness_lift": 16.0 if is_qa else 12.0 if (is_depth_showcase or is_geometry_calibration) else 0.0,
            "fog_tint": (160, 182, 200),
        }
    elif aesthetic == "lowpoly":
        look = {
            "particle_scale": 0.1 if is_qa else 0.16,
            "ui_scale": 0.24 if is_qa else 0.36,
            "persistent_alpha": 0.08 if is_qa else 0.12,
            "local_fog_strength": fog_strength * (0.24 if is_qa else 0.34),
            "local_bloom_strength": bloom_strength * (0.58 if is_qa else 0.68),
            "local_exposure": exposure + (0.18 if is_qa else 0.12),
            "scanline_intensity": 0.02 if is_qa else 0.03,
            "vignette_strength": 0.1 if is_qa else 0.12,
            "dither_strength": 0.12 if is_depth_showcase else 0.16,
            "jitter_strength": 0.28,
            "reflection_scale": 0.45,
            "scan_fog_bias": 0.72,
            "brightness_lift": 18.0 if (is_depth_showcase or is_geometry_calibration) else 10.0,
            "fog_tint": (196, 220, 236),
        }
    else:
        look = {
            "particle_scale": 0.015 if is_qa else 0.025,
            "ui_scale": 0.08 if is_qa else 0.12,
            "persistent_alpha": 0.015 if is_qa else 0.025,
            "local_fog_strength": fog_strength * (0.025 if is_qa else 0.035),
            "local_bloom_strength": bloom_strength * (0.08 if is_qa else 0.12),
            "local_exposure": exposure + (0.08 if is_qa else 0.04),
            "scanline_intensity": 0.0,
            "vignette_strength": 0.0,
            "dither_strength": 0.0,
            "jitter_strength": 0.08,
            "reflection_scale": 0.2,
            "scan_fog_bias": 0.18,
            "brightness_lift": 10.0 if (is_depth_showcase or is_geometry_calibration) else 6.0,
            "fog_tint": (236, 244, 248),
            "bloom_threshold": 198.0,
            "tone_gamma": 0.82,
            "tone_shoulder": 1.04,
            "line_scale": 0.48 if line_scale is None else line_scale,
        }
    if "bloom_threshold" not in look:
        look["bloom_threshold"] = 132.0
    if "tone_gamma" not in look:
        look["tone_gamma"] = 0.94
    if "tone_shoulder" not in look:
        look["tone_shoulder"] = 1.7
    if "line_scale" not in look:
        look["line_scale"] = 1.0 if line_scale is None else line_scale
    if is_depth_showcase or is_geometry_calibration:
        look["particle_scale"] = float(look["particle_scale"]) * 0.7
        look["persistent_alpha"] = float(look["persistent_alpha"]) * 0.9
        look["local_fog_strength"] = float(look["local_fog_strength"]) * 0.85
        look["ui_scale"] = float(look["ui_scale"]) * 0.85
    return look


def _draw_debug_labels(
    frame: np.ndarray,
    t: float,
    active: list[tuple[object, float]],
    debug_info: dict[str, object] | None = None,
    *,
    include_counts: bool = True,
    include_boxes: bool = True,
) -> None:
    lines = [f"t {t:05.2f}s"]
    for segment, weight in active[:3]:
        lines.append(f"{segment.name} {weight:.2f}")
    if debug_info is not None and include_counts:
        lines.append(f"cam z {float(debug_info.get('camera_z', 0.0)):05.2f}")
        lines.append(f"obj {int(debug_info.get('visible_objects', 0))} poly {int(debug_info.get('visible_polygons', 0))}")
    box_w = min(frame.shape[1] - 10, 156)
    box_h = 8 + len(lines) * 10
    box_x = 6
    box_y = 6
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (10, 14, 18), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (168, 186, 194), 1)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (box_x + 4, box_y + 12 + idx * 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (225, 234, 238), 1, cv2.LINE_AA)
    if debug_info is not None and include_boxes:
        for idx, (label, bbox) in enumerate(list(dict(debug_info.get("boxes", {})).items())[:8]):
            x0, y0, x1, y1 = bbox
            color = (80 + (idx * 37) % 160, 220 - (idx * 19) % 120, 240)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
            cv2.putText(frame, label[:14], (x0 + 2, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)


def _render_geometry_calibration(
    frame: np.ndarray,
    t: float,
    features: dict[str, float],
    palette_name: str,
) -> None:
    palette = PALETTES[palette_name]
    ps2fx.gradient_background(frame, ps2fx.lerp_color(palette.bg, (22, 30, 42), 0.55), ps2fx.lerp_color(palette.mid, (54, 72, 98), 0.4))
    camera = Camera3D(x=math.sin(t * 0.2) * 0.05, y=0.02, z=t * 0.12, yaw=math.sin(t * 0.12) * 0.02, pitch=-0.015, fov=1.0)
    debug_set_camera(camera.z)
    fog = (132, 152, 176)
    debug_begin_object("corridor")
    draw_corridor_planes(frame, camera, (72, 92, 116), (104, 120, 138), (42, 52, 66), fog, width=2.8, height=1.8, near_z=camera.z + 2.0, far_z=camera.z + 10.0)
    debug_end_object()
    debug_begin_object("floor_grid")
    draw_floor_grid_3d(frame, camera, palette.ui, fog, x_range=(-2.8, 2.8), z_range=(camera.z + 2.0, camera.z + 10.0), spacing=0.8, y=-0.95)
    debug_end_object()
    for label, center, size, color in (
        ("cube_near_z3.0", (-1.35, -0.08, camera.z + 3.0), (1.1, 1.4, 1.1), palette.accent),
        ("cube_mid_z5.4", (0.0, 0.0, camera.z + 5.4), (1.4, 1.8, 1.4), palette.ui),
        ("cube_far_z7.8", (1.45, 0.15, camera.z + 7.8), (1.6, 2.2, 1.6), palette.glow),
    ):
        debug_begin_object(label)
        draw_cuboid_3d(frame, camera, center, size, color, fog, edge=palette.glow)
        debug_end_object()
    debug_begin_object("billboard_z4.4")
    draw_billboard_3d(frame, camera, (1.95, 0.55, camera.z + 4.4), (1.15, 0.75), palette.accent, palette.glow, fog, 1.0)
    debug_end_object()
    debug_begin_object("ring_z6.4")
    draw_portal_ring_3d(frame, camera, (-1.7, 0.1, camera.z + 6.4), 0.9 + features["beat"] * 0.1, palette.glow, fog, y_scale=0.8, thickness=3)
    debug_end_object()


def _nearest_frame_index(times: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(times - target)))


def _save_debug_frame(
    output_dir: Path,
    frame: np.ndarray,
    t: float,
    active: list[tuple[object, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_label = "none"
    if active:
        mode_label = "+".join(segment.name for segment, _ in active[:2])
    filename = f"t_{t:05.2f}s__{mode_label}.png".replace(":", "_")
    cv2.imwrite(str(output_dir / filename), frame)


def render_video(
    input_path: Path,
    output_path: Path,
    duration: float | None,
    width: int,
    height: int,
    fps: int,
    seed: int,
    preset: str,
    crf: int,
    audio_bitrate: str,
    analysis_backend: str,
    render_scale: float,
    bloom_strength: float,
    fog_strength: float,
    exposure: float,
    ps2_jitter: float,
    line_scale: float | None = None,
    render_profile: str | None = None,
    aesthetic: str = "frutiger_cyber",
    scene_grammar: str = "worlds",
    render_engine: str = "opencv",
    debug_frames_dir: Path | None = None,
    debug_labels: bool = False,
    debug_raw_frames: bool = False,
    blender_proof_stills: bool = False,
    blender_smoke_scene: str | None = None,
    blender_quality: str = "proof",
    blender_diagnostic_engine: str | None = None,
) -> None:
    preset = normalize_preset_name(preset)
    if aesthetic in {"retro_clean", "ps2_clean"}:
        aesthetic = "lowpoly"
    if render_engine == "blender":
        render_video_blender(
            input_path=input_path,
            output_path=output_path,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
            preset=preset,
            crf=crf,
            audio_bitrate=audio_bitrate,
            analysis_backend=analysis_backend,
            render_profile=render_profile,
            aesthetic=aesthetic,
            scene_grammar=scene_grammar,
            debug_frames_dir=debug_frames_dir,
            debug_labels=debug_labels,
            blender_proof_stills=blender_proof_stills,
            blender_smoke_scene=blender_smoke_scene,
            blender_quality=blender_quality,
            blender_diagnostic_engine=blender_diagnostic_engine,
        )
        return
    ffmpeg = _assert_ffmpeg()
    analysis = analyze_audio(input_path, fps=fps, duration_limit=duration, backend=analysis_backend)
    timeline = build_timeline(analysis.duration, preset=preset, scene_grammar=scene_grammar)
    modes = {mode.name: mode for mode in build_modes(scene_grammar=scene_grammar)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_scale = float(np.clip(render_scale, 0.2, 1.0))
    internal_width = max(160, int(width * render_scale))
    internal_height = max(90, int(height * render_scale))
    debug_targets = debug_targets_for_timeline(analysis.duration, preset, timeline)
    debug_indices = {_nearest_frame_index(analysis.frame_times, ts): ts for ts in debug_targets} if debug_targets else {}
    is_depth_showcase = preset == "depth_showcase"
    is_geometry_calibration = preset == "geometry_calibration"
    resolved_profile = render_profile or ("qa" if (debug_frames_dir is not None or debug_labels or debug_raw_frames) else "final")
    is_qa = resolved_profile == "qa"
    debug_raw_dir = debug_frames_dir.parent / f"{debug_frames_dir.name}_raw" if debug_frames_dir is not None and debug_raw_frames else None
    look = _resolve_render_look(preset, resolved_profile, aesthetic, bloom_strength, fog_strength, exposure, line_scale)
    set_render_style(line_scale=float(look["line_scale"]))

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-stream_loop",
        "-1",
        "-i",
        str(input_path),
        "-t",
        str(analysis.duration),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    rng = np.random.default_rng(seed)
    try:
        assert process.stdin is not None
        for index, t in enumerate(analysis.frame_times):
            features = analysis.features_at(index)
            if index > 0:
                prev_features = analysis.features_at(index - 1)
                features["section_pulse"] = float(abs(features["section"] - prev_features["section"]) > 0.08)
            else:
                features["section_pulse"] = 0.0
            frame = np.zeros((internal_height, internal_width, 3), dtype=np.uint8)

            active = active_modes(float(t), timeline)
            if not active and timeline:
                fallback_segment = timeline[min(len(timeline) - 1, index % len(timeline))]
                active = [(fallback_segment, 1.0)]

            debug_overlay_enabled = is_qa and debug_labels
            debug_reset(active[0][0].name if active else "", enabled=debug_overlay_enabled)
            for segment, weight in active:
                palette = palette_for(segment.palette_name)
                if segment.name == "geometry_calibration":
                    _render_geometry_calibration(frame, float(t), features, segment.palette_name)
                else:
                    mode = modes[segment.name]
                    mode.set_context(segment)
                    modes[segment.name].render(frame, float(t), features, float(weight), palette, rng)

            if debug_raw_dir is not None and index in debug_indices:
                if debug_overlay_enabled:
                    raw_labeled = frame.copy()
                    _draw_debug_labels(raw_labeled, float(t), active, debug_snapshot(), include_counts=True, include_boxes=True)
                    raw_frame = cv2.resize(raw_labeled, (width, height), interpolation=cv2.INTER_LINEAR if resolved_profile == "final" else cv2.INTER_NEAREST)
                else:
                    raw_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR if resolved_profile == "final" else cv2.INTER_NEAREST)
                _save_debug_frame(debug_raw_dir, raw_frame, debug_indices[index], active)

            persistent = np.zeros_like(frame)
            base_palette = active[0][0].palette_name if active else next(iter(PALETTES))
            transition_mix = float(active[1][1]) if len(active) > 1 else 0.0
            persistent[:] = _global_layers(
                internal_width,
                internal_height,
                float(t),
                features,
                base_palette,
                float(look["local_fog_strength"]),
                float(look["particle_scale"]),
                float(look["ui_scale"]),
                float(look["reflection_scale"]),
                tuple(look["fog_tint"]),
                float(look["scan_fog_bias"]),
                transition_mix,
                scene_grammar=scene_grammar,
                aesthetic=aesthetic,
            )
            transition_alpha = 0.0 if scene_grammar == "worlds" else transition_mix * (0.05 if is_qa else 0.08)
            frame[:] = additive_blend(frame, persistent, float(look["persistent_alpha"]) + transition_alpha)
            scanline_intensity = float(look["scanline_intensity"])
            if scanline_intensity > 0.001:
                scanlines(frame, scanline_intensity + features["onset"] * 0.02)
            vignette_strength = float(look["vignette_strength"])
            if vignette_strength > 0.001:
                vignette(frame, vignette_strength)
            frame = apply_tone_curve(frame, float(look["tone_gamma"]), float(look["tone_shoulder"]))
            frame = apply_bloom(frame, float(look["local_bloom_strength"]) * (0.82 + features["energy"] * 0.06 + features["section_pulse"] * 0.06), float(look["bloom_threshold"]))
            dither_strength = float(look["dither_strength"])
            if dither_strength > 0.001:
                frame = add_dither(frame, dither_strength)
            frame = apply_exposure(frame, float(look["local_exposure"]))
            brightness_lift = float(look["brightness_lift"])
            if brightness_lift > 0.001:
                frame = np.clip(frame.astype(np.float32) + brightness_lift, 0, 255).astype(np.uint8)
            frame = apply_jitter(frame, float(t), ps2_jitter * float(look["jitter_strength"]))
            if debug_overlay_enabled:
                labeled_frame = frame.copy()
                _draw_debug_labels(labeled_frame, float(t), active, debug_snapshot(), include_counts=True, include_boxes=True)
                output_frame = cv2.resize(labeled_frame, (width, height), interpolation=cv2.INTER_LINEAR if resolved_profile == "final" else cv2.INTER_NEAREST)
            else:
                output_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR if resolved_profile == "final" else cv2.INTER_NEAREST)
            if debug_frames_dir is not None and index in debug_indices:
                _save_debug_frame(debug_frames_dir, output_frame, debug_indices[index], active)
            process.stdin.write(output_frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")
