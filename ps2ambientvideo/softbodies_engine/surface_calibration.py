from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SURFACE_CONTACT_MODES = (
    "bullet_only",
    "penalty_old",
    "projection",
    "velocity_projection",
    "support_spheres",
    "soft_contact_grid",
)


@dataclass(frozen=True)
class PlateMotionConfig:
    rest_duration: float = 0.8
    impact_time: float = 1.1
    pre_plate_duration: float = 2.5
    plate_speed: float = 0.22
    plate_travel: float = 0.36
    plate_hold: float = 0.8
    plate_clearance: float = 0.70


@dataclass(frozen=True)
class SafetyThresholds:
    minimum_safe_volume_ratio: float = 0.68
    floor_tolerance: float = -0.02
    min_triangle_area_proxy: float = -0.08
    max_edge_stretch_ratio: float = 1.85
    min_edge_compression_ratio: float = 0.42
    camera_safe_radius: float = 2.9


@dataclass(frozen=True)
class PlateCalibrationResult:
    plate_travel: float
    minimum_volume_ratio: float
    recovery_volume_ratio: float
    permanent_deformation: float
    max_displacement: float
    max_node_velocity: float
    minimum_node_height: float
    nodes_below_floor: int
    body_left_safe_region: bool
    has_non_finite: bool
    passed: bool


@dataclass(frozen=True)
class ImpactCalibrationResult:
    impact_speed: float
    minimum_volume_ratio: float
    recovery_volume_ratio: float
    permanent_deformation: float
    max_displacement: float
    max_node_velocity: float
    minimum_node_height: float
    nodes_below_floor: int
    body_left_safe_region: bool
    has_non_finite: bool
    passed: bool


def ease_in_out(value: float) -> float:
    clamped = min(max(value, 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def plate_height_for_time(
    time_seconds: float,
    start_height: float,
    config: PlateMotionConfig,
) -> tuple[str, float]:
    descent_duration = config.plate_travel / max(config.plate_speed, 1e-6)
    release_duration = descent_duration
    impact_end = config.impact_time + config.pre_plate_duration
    compress_start = impact_end
    hold_start = compress_start + descent_duration
    release_start = hold_start + config.plate_hold
    recovery_start = release_start + release_duration

    if time_seconds < config.rest_duration:
        return "rest", start_height
    if time_seconds < config.impact_time:
        return "pre_impact", start_height
    if time_seconds < impact_end:
        return "impact_recovery", start_height
    if time_seconds < hold_start:
        progress = ease_in_out((time_seconds - compress_start) / max(descent_duration, 1e-6))
        return "plate_descent", start_height - config.plate_travel * progress
    if time_seconds < release_start:
        return "plate_hold", start_height - config.plate_travel
    if time_seconds < recovery_start:
        progress = ease_in_out((time_seconds - release_start) / max(release_duration, 1e-6))
        return "plate_release", (start_height - config.plate_travel) + config.plate_travel * progress
    return "final_recovery", start_height


def should_trigger_safe_release(
    volume_ratio: float,
    minimum_node_height: float,
    min_triangle_area_proxy: float,
    max_edge_stretch_ratio: float,
    min_edge_compression_ratio: float,
    has_non_finite: bool,
    thresholds: SafetyThresholds,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if volume_ratio < thresholds.minimum_safe_volume_ratio:
        reasons.append("volume")
    if minimum_node_height < thresholds.floor_tolerance:
        reasons.append("floor")
    if min_triangle_area_proxy < thresholds.min_triangle_area_proxy:
        reasons.append("triangle")
    if max_edge_stretch_ratio > thresholds.max_edge_stretch_ratio:
        reasons.append("stretch")
    if min_edge_compression_ratio < thresholds.min_edge_compression_ratio:
        reasons.append("compression")
    if has_non_finite:
        reasons.append("non_finite")
    return (len(reasons) > 0, reasons)


def select_plate_candidate(results: list[PlateCalibrationResult]) -> PlateCalibrationResult | None:
    passing = [result for result in results if result.passed]
    if not passing:
        return None
    return sorted(passing, key=lambda item: (item.plate_travel, -item.permanent_deformation))[-1]


def select_impact_candidate(results: list[ImpactCalibrationResult]) -> ImpactCalibrationResult | None:
    passing = [result for result in results if result.passed]
    if not passing:
        return None
    return sorted(passing, key=lambda item: (item.impact_speed, -item.permanent_deformation))[-1]


def surface_calibration_metric_fields() -> list[str]:
    return [
        "time",
        "phase",
        "preset",
        "variant",
        "volume_ratio",
        "current_volume",
        "rest_volume",
        "center_of_mass_x",
        "center_of_mass_y",
        "center_of_mass_z",
        "center_of_mass_drift",
        "raw_mean_displacement",
        "raw_max_displacement",
        "aligned_rms_deformation",
        "aligned_max_deformation",
        "max_velocity",
        "mean_velocity",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "minimum_node_height",
        "nodes_below_floor",
        "supported_node_count",
        "mean_support_force",
        "max_support_force",
        "total_support_force",
        "corrected_node_count",
        "damped_node_count",
        "total_correction_magnitude",
        "mean_vertical_velocity_before",
        "mean_vertical_velocity_after",
        "min_triangle_area_proxy",
        "max_edge_stretch_ratio",
        "min_edge_compression_ratio",
        "top_region_deformation",
        "bottom_region_deformation",
        "struck_corner_deformation",
        "opposite_corner_deformation",
        "kinetic_energy_proxy",
        "node_count",
        "solver_substeps",
        "physics_substep_dt",
        "simulation_step_duration_ms",
        "plate_position",
        "plate_travel",
        "plate_hold",
        "impact_speed",
        "safety_release_triggered",
        "safety_reasons",
        "dt",
        "raw_solver_volume",
        "post_support_volume",
        "support_mode",
    ]


def project_penetrating_nodes(
    positions: np.ndarray,
    *,
    floor_height: float,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected = positions.copy()
    target_height = floor_height + epsilon
    indices = np.where(positions[:, 2] < target_height)[0].astype(np.int32)
    if indices.size == 0:
        return indices, corrected, np.zeros((0,), dtype=np.float32)
    corrections = target_height - positions[indices, 2]
    corrected[indices, 2] = target_height
    return indices, corrected, corrections.astype(np.float32)


def velocity_project_penetrating_nodes(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    floor_height: float,
    epsilon: float = 0.0,
    restitution: float = 0.0,
    tangential_friction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_height = floor_height + epsilon
    corrected = velocities.copy()
    indices = np.where(positions[:, 2] < target_height)[0].astype(np.int32)
    damped = np.zeros((velocities.shape[0],), dtype=bool)
    if indices.size == 0:
        return indices, corrected, np.zeros((0,), dtype=np.float32), damped
    corrections = target_height - positions[indices, 2]
    penetrating_velocities = corrected[indices]
    downward = penetrating_velocities[:, 2] < 0.0
    if np.any(downward):
        penetrating_velocities[downward, 2] = -penetrating_velocities[downward, 2] * restitution
        damped[indices[downward]] = True
    if tangential_friction > 0.0:
        friction_scale = np.clip(1.0 - tangential_friction, 0.0, 1.0)
        penetrating_velocities[:, 0] *= friction_scale
        penetrating_velocities[:, 1] *= friction_scale
    corrected[indices] = penetrating_velocities
    return indices, corrected, corrections.astype(np.float32), damped
