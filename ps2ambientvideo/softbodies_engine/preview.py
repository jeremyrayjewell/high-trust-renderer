from __future__ import annotations

import argparse
from pathlib import Path

from panda3d.core import KeyboardButton, WindowProperties, loadPrcFileData

loadPrcFileData("", "window-title Gummy Video Preview")
loadPrcFileData("", "sync-video 0")
loadPrcFileData("", "show-frame-rate-meter 1")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from .audio_analysis import analyze_audio
from .scene import SceneConfig
from .scene_factory import build_scene


class PreviewApp(ShowBase):
    def __init__(
        self,
        audio: str | None,
        duration: float,
        seed: int,
        width: int,
        height: int,
        smoke_test_seconds: float = 0.0,
        scene_config: SceneConfig | None = None,
    ) -> None:
        super().__init__()
        self.disableMouse()
        self.camLens.setAspectRatio(width / max(height, 1))
        self.accept("escape", self.userExit)
        self.accept("r", self.reset_simulation)

        self.duration = duration
        self.seed = seed
        self.smoke_test_seconds = smoke_test_seconds
        self.elapsed = 0.0
        self.accumulator = 0.0
        self.fixed_dt = 1.0 / 60.0
        self.audio_path = audio
        self.features = analyze_audio(audio, duration=duration, seed=seed)
        self.scene_config = scene_config or SceneConfig(width=width, height=height, seed=seed)
        self.scene = build_scene(self, self.features, self.scene_config)

        if self.win is not None:
            props = WindowProperties()
            props.setSize(width, height)
            self.win.requestProperties(props)

        self.taskMgr.add(self._update_task, "preview-update")

    def reset_simulation(self) -> None:
        self.elapsed = 0.0
        self.accumulator = 0.0
        self.scene.reset()

    def _update_task(self, task: Task) -> Task:
        dt = min(globalClock.getDt(), 1.0 / 20.0)
        self.accumulator += dt

        while self.accumulator >= self.fixed_dt:
            sample = self.features.sample(self.elapsed)
            self.scene.step(self.fixed_dt, sample)
            self.elapsed += self.fixed_dt
            self.accumulator -= self.fixed_dt
            if self.elapsed >= self.duration:
                self.elapsed = 0.0
                self.scene.reset()

        if self.mouseWatcherNode.isButtonDown(KeyboardButton.asciiKey(b"q")):
            self.userExit()

        if self.smoke_test_seconds > 0.0 and task.time >= self.smoke_test_seconds:
            self.userExit()
            return Task.done

        return Task.cont


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive preview for the gummy video scene.")
    parser.add_argument("--audio", type=str, default=str(Path("assets") / "track.wav"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
    parser.add_argument("--smoke-test-seconds", type=float, default=0.0, help="Automatically exit after N seconds for validation.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    default_audio = str(Path("assets") / "track.wav")
    requested_audio_path = Path(args.audio)
    if args.softbody_obstacle_course and args.audio != default_audio and not requested_audio_path.exists():
        raise FileNotFoundError(f"Requested obstacle-course audio file does not exist: {requested_audio_path}")
    audio = args.audio if requested_audio_path.exists() else None
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
        metrics_output=args.metrics_output,
        course_layout_output=args.course_layout_output,
        body_summary_output=args.course_layout_output.replace("_layout.json", "_body_summary.json") if args.course_layout_output else "",
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
    app = PreviewApp(
        audio=audio,
        duration=args.duration,
        seed=args.seed,
        width=args.width,
        height=args.height,
        smoke_test_seconds=args.smoke_test_seconds,
        scene_config=scene_config,
    )
    app.run()


if __name__ == "__main__":
    main()
