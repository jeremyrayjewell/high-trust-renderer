from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import imageio.v3 as iio
from panda3d.core import Filename, PNMImage, loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "sync-video 0")
loadPrcFileData("", "show-frame-rate-meter 0")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase

import imageio_ffmpeg

from .audio_analysis import analyze_audio
from .scene import SceneConfig
from .scene_factory import build_scene


class OfflineRenderApp(ShowBase):
    def __init__(self, width: int, height: int) -> None:
        loadPrcFileData("", f"win-size {width} {height}")
        super().__init__()
        self.disableMouse()
        self.camLens.setAspectRatio(width / max(height, 1))


def _texture_dimensions(texture) -> tuple[int, int]:
    if texture is None:
        return (0, 0)
    return (int(texture.getXSize()), int(texture.getYSize()))


def inspect_runtime_dimensions(app: ShowBase, scene, width: int, height: int) -> dict[str, object]:
    expected = (int(width), int(height))
    runtime = {
        "window": (int(app.win.getXSize()), int(app.win.getYSize())),
        "lens_aspect": float(app.camLens.getAspectRatio()),
    }
    if runtime["window"] != expected:
        raise RuntimeError(f"Offscreen window size mismatch: expected {expected}, got {runtime['window']}")
    expected_aspect = width / max(height, 1)
    if abs(runtime["lens_aspect"] - expected_aspect) > 1e-6:
        raise RuntimeError(f"Camera aspect ratio mismatch: expected {expected_aspect:0.6f}, got {runtime['lens_aspect']:0.6f}")
    if scene is not None and getattr(scene, "pipeline", None) is not None:
        pipeline = scene.pipeline
        runtime["pipeline"] = {
            "scene_color": {
                "buffer": (int(pipeline.scene_color_buffer.getXSize()), int(pipeline.scene_color_buffer.getYSize())),
                "texture": _texture_dimensions(pipeline.scene_color_tex),
            },
            "front_depth": {
                "buffer": (int(pipeline.front_depth_buffer.getXSize()), int(pipeline.front_depth_buffer.getYSize())),
                "texture": _texture_dimensions(pipeline.front_depth_tex),
            },
            "back_depth": {
                "buffer": (int(pipeline.back_depth_buffer.getXSize()), int(pipeline.back_depth_buffer.getYSize())),
                "texture": _texture_dimensions(pipeline.back_depth_tex),
            },
            "front_normal": {
                "buffer": (int(pipeline.front_normal_buffer.getXSize()), int(pipeline.front_normal_buffer.getYSize())),
                "texture": _texture_dimensions(pipeline.front_normal_tex),
            },
            "front_local": {
                "buffer": (int(pipeline.front_local_buffer.getXSize()), int(pipeline.front_local_buffer.getYSize())),
                "texture": _texture_dimensions(pipeline.front_local_tex),
            },
        }
        for name, entry in runtime["pipeline"].items():
            buffer_dims = entry["buffer"]
            texture_dims = entry["texture"]
            if buffer_dims != expected:
                raise RuntimeError(f"Compositor buffer size mismatch for {name}: expected {expected}, got {buffer_dims}")
            if texture_dims[0] < width or texture_dims[1] < height:
                raise RuntimeError(
                    f"Compositor texture undersized for {name}: expected at least {expected}, got {texture_dims}"
                )
    print(f"[validate] window={runtime['window']} lens_aspect={runtime['lens_aspect']:.6f}")
    for name, entry in runtime.get("pipeline", {}).items():
        print(f"[validate] {name}.buffer={entry['buffer']} {name}.texture={entry['texture']}")
    return runtime


def inspect_png_dimensions(frame_path: Path) -> tuple[int, int]:
    image = iio.imread(frame_path)
    return (int(image.shape[1]), int(image.shape[0]))


def inspect_video_dimensions(video_path: Path) -> tuple[int, int]:
    reader = imageio_ffmpeg.read_frames(str(video_path))
    try:
        metadata = next(reader)
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    size = metadata.get("size")
    if not size:
        raise RuntimeError(f"Unable to read encoded video dimensions for {video_path}")
    return (int(size[0]), int(size[1]))


def render_frames(
    audio_path: str | None,
    duration: float,
    fps: int,
    width: int,
    height: int,
    frames_dir: Path,
    seed: int,
    scene_config: SceneConfig | None = None,
) -> tuple[Path, int]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(duration * fps))
    dt = 1.0 / fps

    app = OfflineRenderApp(width=width, height=height)
    features = analyze_audio(audio_path, duration=duration, seed=seed)
    config = scene_config or SceneConfig(width=width, height=height, seed=seed)
    scene = build_scene(app, features, config)

    app.graphicsEngine.renderFrame()
    app.graphicsEngine.renderFrame()
    inspect_runtime_dimensions(app, scene, width, height)

    for frame_index in range(total_frames):
        if frame_index > 0:
            current_time = (frame_index - 1) / fps
            sample = features.sample(current_time)
            scene.step(dt, sample)
        app.graphicsEngine.renderFrame()
        pnm = PNMImage()
        if not app.win.getScreenshot(pnm):
            raise RuntimeError("Failed to capture offscreen frame from Panda3D window.")
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        if not pnm.write(Filename.fromOsSpecific(str(frame_path))):
            raise RuntimeError(f"Failed to write frame to {frame_path}")
        if pnm.getXSize() != width or pnm.getYSize() != height:
            raise RuntimeError(
                f"Captured frame size mismatch at frame {frame_index}: expected {(width, height)}, got {(pnm.getXSize(), pnm.getYSize())}"
            )
        if frame_index == 0:
            png_dims = inspect_png_dimensions(frame_path)
            if png_dims != (width, height):
                raise RuntimeError(f"Saved PNG size mismatch: expected {(width, height)}, got {png_dims}")
            print(f"[validate] first_png={png_dims}")
        if frame_index % max(1, fps // 2) == 0 or frame_index == total_frames - 1:
            print(f"[render] frame {frame_index + 1}/{total_frames}")

    if hasattr(scene, "finalize"):
        scene.finalize()
    app.destroy()
    return frames_dir, total_frames


def encode_video(
    frames_dir: Path,
    fps: int,
    output_path: Path,
    audio_path: str | None,
    duration: float,
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
    ]
    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", str(audio_path), "-t", f"{duration:.3f}", "-shortest"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
        ]
    )
    if audio_path and Path(audio_path).exists():
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(str(output_path))

    print("[encode] " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    video_dims = inspect_video_dimensions(output_path)
    if video_dims != (width, height):
        raise RuntimeError(f"Encoded MP4 size mismatch: expected {(width, height)}, got {video_dims}")
    print(f"[validate] encoded_mp4={video_dims}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic offline renderer for the gummy video MVP.")
    parser.add_argument("--audio", type=str, default=str(Path("assets") / "track.wav"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=str, default=str(Path("output") / "gummy_test.mp4"))
    parser.add_argument("--frames-dir", type=str, default=str(Path("frames")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gelatin-opacity", type=float, default=0.90)
    parser.add_argument("--gelatin-cloudiness", type=float, default=0.92)
    parser.add_argument("--transmission-strength", type=float, default=0.30)
    parser.add_argument("--fresnel-strength", type=float, default=0.08)
    parser.add_argument("--specular-strength", type=float, default=0.36)
    parser.add_argument("--wobble-strength", type=float, default=1.0)
    parser.add_argument("--deformation-strength", type=float, default=1.0)
    parser.add_argument("--backlight-strength", type=float, default=1.60)
    parser.add_argument("--rear-face-strength", type=float, default=0.01)
    parser.add_argument("--thickness-absorption", type=float, default=0.96)
    parser.add_argument("--minimum-body-light", type=float, default=0.50)
    parser.add_argument("--exposure", type=float, default=1.30)
    parser.add_argument("--inner-layer-opacity", type=float, default=0.0)
    parser.add_argument("--physics-mode", type=str, choices=("hybrid", "soft"), default="hybrid")
    parser.add_argument("--absorption-color", type=float, nargs=3, default=(0.22, 0.08, 0.04))
    parser.add_argument("--absorption-density", type=float, default=1.75)
    parser.add_argument("--scattering-strength", type=float, default=0.46)
    parser.add_argument("--translucency-cloudiness", type=float, default=0.40)
    parser.add_argument("--refraction-strength", type=float, default=0.035)
    parser.add_argument("--ior", type=float, default=1.10)
    parser.add_argument("--surface-opacity", type=float, default=0.98)
    parser.add_argument("--thickness-scale", type=float, default=0.80)
    parser.add_argument("--material-debug", action="store_true")
    parser.add_argument("--geometry-debug", action="store_true")
    parser.add_argument("--deformation-debug", action="store_true")
    parser.add_argument("--surface-calibration-debug", action="store_true")
    parser.add_argument("--surface-rest-debug", action="store_true")
    parser.add_argument("--surface-floating-rest-debug", action="store_true")
    parser.add_argument("--floating-softbody-scene", action="store_true")
    parser.add_argument("--softbody-obstacle-course", action="store_true")
    parser.add_argument("--course-seed", type=int, default=None)
    parser.add_argument("--floating-debug-overlay", action="store_true")
    parser.add_argument("--framing-debug-overlay", action="store_true")
    parser.add_argument("--floating-performance-intensity", type=str, choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--true-softbody-debug", action="store_true")
    parser.add_argument("--true-softbody-stress-debug", action="store_true")
    parser.add_argument("--tetra-softbody-debug", action="store_true")
    parser.add_argument("--translucency-debug", action="store_true")
    parser.add_argument("--true-softbody-translucent", action="store_true")
    parser.add_argument("--true-softbody-translucent-stress", action="store_true")
    parser.add_argument("--softbody-preset", type=str, choices=("soft", "medium", "firm", "stable_soft", "stable_medium", "stable_firm"), default="stable_medium")
    parser.add_argument("--softbody-visualization", type=str, choices=("shaded", "wireframe", "displacement", "nodes"), default="shaded")
    parser.add_argument("--softbody-profile", dest="softbody_profiles", action="append", choices=("very_soft", "soft", "medium", "springy", "firm_bouncy"), default=[])
    parser.add_argument("--translucency-view", type=str, choices=("composite", "front-depth", "back-depth", "thickness", "normals", "refraction-offset", "transmittance"), default="composite")
    parser.add_argument("--translucency-preset", type=str, choices=("subtle", "balanced", "exaggerated"), default="balanced")
    parser.add_argument("--transmission-gain", type=float, default=1.15)
    parser.add_argument("--surface-reflection-strength", type=float, default=0.20)
    parser.add_argument("--metrics-output", type=str, default="")
    parser.add_argument("--course-layout-output", type=str, default="")
    parser.add_argument("--plate-speed", type=float, default=0.22)
    parser.add_argument("--plate-travel", type=float, default=0.36)
    parser.add_argument("--plate-hold", type=float, default=0.90)
    parser.add_argument("--plate-clearance", type=float, default=0.70)
    parser.add_argument("--minimum-safe-volume-ratio", type=float, default=0.68)
    parser.add_argument("--plate-calibration-output", type=str, default="output/surface_plate_calibration.csv")
    parser.add_argument("--impact-calibration-output", type=str, default="output/surface_impact_calibration.csv")
    parser.add_argument("--support-ablation-output", type=str, default="output/surface_support_ablation.csv")
    parser.add_argument("--contact-comparison-output", type=str, default="output/surface_contact_comparison.csv")
    parser.add_argument("--gravity-contact-sweep-output", type=str, default="output/surface_gravity_contact_sweep.csv")
    parser.add_argument("--rest-sweep-output", type=str, default="output/surface_rest_sweep.csv")
    parser.add_argument("--inversion-events-output", type=str, default="output/surface_inversion_events.csv")
    parser.add_argument("--surface-contact-mode", type=str, choices=("bullet_only", "penalty_old", "projection", "velocity_projection", "support_spheres", "soft_contact_grid"), default="support_spheres")
    parser.add_argument("--show-rest-ghost", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    default_audio = str(Path("assets") / "track.wav")
    requested_audio_path = Path(args.audio)
    if args.softbody_obstacle_course and args.audio != default_audio and not requested_audio_path.exists():
        raise FileNotFoundError(f"Requested obstacle-course audio file does not exist: {requested_audio_path}")
    audio = args.audio if requested_audio_path.exists() else None
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)
    if args.metrics_output:
        metrics_output = args.metrics_output
    elif args.softbody_obstacle_course:
        metrics_output = str(output_path.parent / f"{output_path.stem}_metrics.csv")
    elif args.floating_softbody_scene:
        metrics_output = str(output_path.parent / f"{output_path.stem}_metrics.csv")
    else:
        metrics_output = str(output_path.parent / "softbody_stress_metrics.csv")
    if args.course_layout_output:
        course_layout_output = args.course_layout_output
    elif args.softbody_obstacle_course:
        course_layout_output = str(output_path.parent / f"{output_path.stem}_layout.json")
    else:
        course_layout_output = ""
    if args.softbody_obstacle_course:
        body_summary_output = str(output_path.parent / f"{output_path.stem}_body_summary.json")
    else:
        body_summary_output = ""
    scene_config = SceneConfig(
        width=args.width,
        height=args.height,
        duration=args.duration,
        seed=args.seed,
        gelatin_opacity=args.gelatin_opacity,
        gelatin_cloudiness=args.gelatin_cloudiness,
        transmission_strength=args.transmission_strength,
        fresnel_strength=args.fresnel_strength,
        specular_strength=args.specular_strength,
        wobble_strength=args.wobble_strength,
        deformation_strength=args.deformation_strength,
        backlight_strength=args.backlight_strength,
        rear_face_strength=args.rear_face_strength,
        thickness_absorption=args.thickness_absorption,
        minimum_body_light=args.minimum_body_light,
        exposure=args.exposure,
        inner_layer_opacity=args.inner_layer_opacity,
        physics_mode=args.physics_mode,
        material_debug=args.material_debug,
        geometry_debug=args.geometry_debug,
        deformation_debug=args.deformation_debug,
        surface_calibration_debug=args.surface_calibration_debug,
        surface_rest_debug=args.surface_rest_debug,
        surface_floating_rest_debug=args.surface_floating_rest_debug,
        floating_softbody_scene=args.floating_softbody_scene,
        softbody_obstacle_course=args.softbody_obstacle_course,
        course_seed=args.course_seed if args.course_seed is not None else args.seed,
        floating_debug_overlay=args.floating_debug_overlay,
        framing_debug_overlay=args.framing_debug_overlay,
        floating_performance_intensity=args.floating_performance_intensity,
        true_softbody_debug=args.true_softbody_debug,
        true_softbody_stress_debug=args.true_softbody_stress_debug,
        tetra_softbody_debug=args.tetra_softbody_debug,
        translucency_debug=args.translucency_debug,
        true_softbody_translucent=args.true_softbody_translucent,
        true_softbody_translucent_stress=args.true_softbody_translucent_stress,
        softbody_preset=args.softbody_preset,
        softbody_visualization=args.softbody_visualization,
        softbody_profiles=tuple(args.softbody_profiles),
        translucency_view=args.translucency_view,
        translucency_preset=args.translucency_preset,
        absorption_color=tuple(args.absorption_color),
        absorption_density=args.absorption_density,
        transmission_gain=args.transmission_gain,
        scattering_strength=args.scattering_strength,
        translucency_cloudiness=args.translucency_cloudiness,
        refraction_strength=args.refraction_strength,
        ior=args.ior,
        surface_opacity=args.surface_opacity,
        surface_reflection_strength=args.surface_reflection_strength,
        thickness_scale=args.thickness_scale,
        metrics_output=metrics_output,
        course_layout_output=course_layout_output,
        body_summary_output=body_summary_output,
        plate_speed=args.plate_speed,
        plate_travel=args.plate_travel,
        plate_hold=args.plate_hold,
        plate_clearance=args.plate_clearance,
        minimum_safe_volume_ratio=args.minimum_safe_volume_ratio,
        plate_calibration_output=args.plate_calibration_output,
        impact_calibration_output=args.impact_calibration_output,
        support_ablation_output=args.support_ablation_output,
        contact_comparison_output=args.contact_comparison_output,
        gravity_contact_sweep_output=args.gravity_contact_sweep_output,
        rest_sweep_output=args.rest_sweep_output,
        inversion_events_output=args.inversion_events_output,
        surface_contact_mode=args.surface_contact_mode,
        show_rest_ghost=args.show_rest_ghost,
    )

    print(
        f"[start] duration={args.duration}s fps={args.fps} size={args.width}x{args.height} "
        f"seed={args.seed} audio={'yes' if audio else 'no'}"
    )
    render_frames(
        audio_path=audio,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        frames_dir=frames_dir,
        seed=args.seed,
        scene_config=scene_config,
    )
    encode_video(
        frames_dir=frames_dir,
        fps=args.fps,
        output_path=output_path,
        audio_path=audio,
        duration=args.duration,
        width=args.width,
        height=args.height,
    )
    print(f"[done] video written to {output_path}")

    if not args.keep_frames and frames_dir.exists():
        shutil.rmtree(frames_dir)
        print(f"[cleanup] removed {frames_dir}")


if __name__ == "__main__":
    main()
