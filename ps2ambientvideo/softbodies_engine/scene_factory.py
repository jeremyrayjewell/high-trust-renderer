from __future__ import annotations

from .experimental_scene import ExperimentalConfig, ExperimentalScene
from .scene import GummyScene, SceneConfig
from .surface_calibration_scene import SurfaceCalibrationConfig, SurfaceCalibrationScene


def build_scene(base, audio_features, config: SceneConfig):
    if config.surface_calibration_debug or config.surface_rest_debug or config.surface_floating_rest_debug:
        return SurfaceCalibrationScene(
            base,
            audio_features,
            SurfaceCalibrationConfig(
                width=config.width,
                height=config.height,
                seed=config.seed,
                softbody_preset=config.softbody_preset,
                softbody_visualization=config.softbody_visualization,
                metrics_output=config.metrics_output,
                plate_speed=config.plate_speed,
                plate_travel=config.plate_travel,
                plate_hold=config.plate_hold,
                plate_clearance=config.plate_clearance,
                minimum_safe_volume_ratio=config.minimum_safe_volume_ratio,
                plate_calibration_output=config.plate_calibration_output,
                impact_calibration_output=config.impact_calibration_output,
                support_ablation_output=config.support_ablation_output,
                contact_comparison_output=config.contact_comparison_output,
                gravity_contact_sweep_output=config.gravity_contact_sweep_output,
                rest_sweep_output=config.rest_sweep_output,
                inversion_events_output=config.inversion_events_output,
                surface_rest_debug=config.surface_rest_debug,
                surface_floating_rest_debug=config.surface_floating_rest_debug,
                surface_contact_mode=config.surface_contact_mode,
                show_rest_ghost=config.show_rest_ghost,
            ),
        )
    if (
        config.physics_mode == "soft"
        or config.floating_softbody_scene
        or config.softbody_obstacle_course
        or config.true_softbody_debug
        or config.true_softbody_stress_debug
        or config.tetra_softbody_debug
        or config.translucency_debug
        or config.true_softbody_translucent
        or config.true_softbody_translucent_stress
    ):
        experimental_config = ExperimentalConfig(
            width=config.width,
            height=config.height,
            duration=config.duration,
            seed=config.seed,
            physics_mode=config.physics_mode,
            floating_softbody_scene=config.floating_softbody_scene,
            softbody_obstacle_course=config.softbody_obstacle_course,
            course_seed=config.course_seed,
            floating_debug_overlay=config.floating_debug_overlay,
            framing_debug_overlay=config.framing_debug_overlay,
            floating_performance_intensity=config.floating_performance_intensity,
            true_softbody_debug=config.true_softbody_debug,
            true_softbody_stress_debug=config.true_softbody_stress_debug,
            tetra_softbody_debug=config.tetra_softbody_debug,
            translucency_debug=config.translucency_debug,
            true_softbody_translucent=config.true_softbody_translucent,
            true_softbody_translucent_stress=config.true_softbody_translucent_stress,
            softbody_preset=config.softbody_preset,
            softbody_visualization=config.softbody_visualization,
            softbody_profiles=config.softbody_profiles,
            translucency_view=config.translucency_view,
            translucency_preset=config.translucency_preset,
            absorption_color=config.absorption_color,
            absorption_density=config.absorption_density,
            transmission_gain=config.transmission_gain,
            scattering_strength=config.scattering_strength,
            cloudiness=config.translucency_cloudiness,
            refraction_strength=config.refraction_strength,
            ior=config.ior,
            surface_opacity=config.surface_opacity,
            specular_strength=config.specular_strength,
            fresnel_strength=config.fresnel_strength,
            surface_reflection_strength=config.surface_reflection_strength,
            thickness_scale=config.thickness_scale,
            metrics_output=config.metrics_output,
            course_layout_output=config.course_layout_output,
            body_summary_output=config.body_summary_output,
        )
        return ExperimentalScene(base, audio_features, experimental_config)
    return GummyScene(base, audio_features, config)
