from __future__ import annotations

import shutil
from pathlib import Path

from .audio import analyze_audio


SOFTBODY_PRESETS = (
    "soft",
    "medium",
    "firm",
    "stable_soft",
    "stable_medium",
    "stable_firm",
)

SOFTBODY_VISUALIZATIONS = ("shaded", "wireframe", "displacement", "nodes")

SOFTBODIES_SCENES = ("floating", "obstacle_course", "translucent")


def _import_softbodies_runtime():
    try:
        from .softbodies_engine.render_video import encode_video, render_frames
        from .softbodies_engine.scene import SceneConfig
    except Exception as exc:  # pragma: no cover - depends on optional runtime packages
        raise RuntimeError(
            "Softbodies backend requires optional dependencies and runtime support. "
            "Install Panda3D/imageio extras, for example: "
            "python -m pip install panda3d imageio imageio-ffmpeg librosa soundfile"
        ) from exc
    return render_frames, encode_video, SceneConfig


def _resolve_softbodies_scene(preset: str, requested_scene: str) -> str:
    if requested_scene:
        return requested_scene
    if preset == "worlds_material_proof":
        return "translucent"
    return "floating"


def _softbodies_debug_targets(duration: float) -> list[float]:
    if duration <= 0:
        return []
    fractions = [0.15, 0.35, 0.55, 0.75, 0.92]
    return sorted({round(duration * value, 2) for value in fractions})


def _copy_debug_frames(frames_dir: Path, debug_frames_dir: Path, fps: int, duration: float, scene_name: str) -> None:
    debug_frames_dir.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration * fps)))
    for target in _softbodies_debug_targets(duration):
        frame_index = min(total_frames - 1, max(0, int(round(target * fps))))
        source = frames_dir / f"frame_{frame_index:06d}.png"
        if not source.exists():
            continue
        filename = f"t_{target:05.2f}s__softbodies_{scene_name}.png".replace(":", "_")
        shutil.copy2(source, debug_frames_dir / filename)


def render_video_softbodies(
    input_path: Path,
    output_path: Path,
    duration: float | None,
    width: int,
    height: int,
    fps: int,
    seed: int,
    preset: str,
    audio_bitrate: str,
    analysis_backend: str,
    debug_frames_dir: Path | None = None,
    softbodies_scene: str = "floating",
    softbody_preset: str = "stable_medium",
    softbody_visualization: str = "shaded",
) -> None:
    del audio_bitrate
    render_frames, encode_video, SceneConfig = _import_softbodies_runtime()

    analysis = analyze_audio(input_path, fps=fps, duration_limit=duration, backend=analysis_backend)
    render_duration = float(analysis.duration)
    scene_name = _resolve_softbodies_scene(preset, softbodies_scene)
    # The vendored Panda3D path performs its own librosa-heavy analysis step
    # before frame rendering. For the integrated backend we currently use our
    # local analysis only to resolve duration and then drive the runtime with
    # its synthetic feature path so short validation renders remain practical.
    runtime_audio_path = None
    encoded_audio_path = str(input_path) if input_path.exists() else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if debug_frames_dir is not None:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = output_path.parent / f"{output_path.stem}_softbodies_metrics.csv"
    layout_output = output_path.parent / f"{output_path.stem}_softbodies_layout.json"
    body_summary_output = output_path.parent / f"{output_path.stem}_softbodies_body_summary.json"

    scene_config = SceneConfig(
        width=width,
        height=height,
        duration=render_duration,
        seed=seed,
        floating_softbody_scene=scene_name == "floating",
        softbody_obstacle_course=scene_name == "obstacle_course",
        true_softbody_translucent=scene_name == "translucent",
        softbody_preset=softbody_preset,
        softbody_visualization=softbody_visualization,
        # Favor the cheapest integrated floating-scene preset by default so
        # backend validation does not inherit the upstream medium-cost path.
        floating_performance_intensity="low",
        metrics_output=str(metrics_output),
        course_layout_output=str(layout_output) if scene_name == "obstacle_course" else "",
        body_summary_output=str(body_summary_output) if scene_name == "obstacle_course" else "",
    )

    frames_dir = output_path.parent / f".tmp_{output_path.stem}_softbodies_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        render_frames(
            audio_path=runtime_audio_path,
            duration=render_duration,
            fps=fps,
            width=width,
            height=height,
            frames_dir=frames_dir,
            seed=seed,
            scene_config=scene_config,
        )
        if debug_frames_dir is not None:
            _copy_debug_frames(frames_dir, debug_frames_dir, fps, render_duration, scene_name)
        encode_video(
            frames_dir=frames_dir,
            fps=fps,
            output_path=output_path,
            audio_path=encoded_audio_path,
            duration=render_duration,
            width=width,
            height=height,
        )
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
