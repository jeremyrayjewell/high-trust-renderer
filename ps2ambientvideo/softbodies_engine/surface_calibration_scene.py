from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletBoxShape, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape, BulletWorld
from panda3d.core import AmbientLight, CardMaker, DirectionalLight, Filename, Material, NodePath, PointLight, Shader, TextNode, Vec3, Vec4

from .audio_analysis import AudioFeatures, AudioSample
from .gummy_object import generate_shape_mesh_data
from .soft_body_object import SOFTBODY_PRESETS, SoftBodyMetrics, SoftBodyPreset, TrueSoftBodyObject, compute_signed_surface_volume, compute_surface_triangle_area_proxies, create_debug_plate, create_render_geom_node
from .surface_calibration import (
    ImpactCalibrationResult,
    PlateCalibrationResult,
    PlateMotionConfig,
    SafetyThresholds,
    SURFACE_CONTACT_MODES,
    plate_height_for_time,
    project_penetrating_nodes,
    select_impact_candidate,
    select_plate_candidate,
    should_trigger_safe_release,
    surface_calibration_metric_fields,
    velocity_project_penetrating_nodes,
)


@dataclass
class SurfaceCalibrationConfig:
    width: int
    height: int
    seed: int
    softbody_preset: str = "stable_medium"
    softbody_visualization: str = "shaded"
    duration: float = 10.5
    plate_speed: float = 0.16
    plate_travel: float = 0.30
    plate_hold: float = 1.1
    plate_clearance: float = 0.82
    minimum_safe_volume_ratio: float = 0.68
    metrics_output: str = "output/surface_calibration_metrics.csv"
    plate_calibration_output: str = "output/surface_plate_calibration.csv"
    impact_calibration_output: str = "output/surface_impact_calibration.csv"
    support_ablation_output: str = "output/surface_support_ablation.csv"
    contact_comparison_output: str = "output/surface_contact_comparison.csv"
    gravity_contact_sweep_output: str = "output/surface_gravity_contact_sweep.csv"
    rest_sweep_output: str = "output/surface_rest_sweep.csv"
    inversion_events_output: str = "output/surface_inversion_events.csv"
    physics_substeps: int = 12
    physics_substep_dt: float = 1.0 / 360.0
    surface_rest_debug: bool = False
    surface_floating_rest_debug: bool = False
    surface_contact_mode: str = "support_spheres"
    show_rest_ghost: bool = False


@dataclass
class SupportStats:
    supported_node_count: int = 0
    total_support_force: float = 0.0
    mean_support_force: float = 0.0
    max_support_force: float = 0.0
    floor_supported_node_count: int = 0
    plate_supported_node_count: int = 0
    corrected_node_count: int = 0
    damped_node_count: int = 0
    total_correction_magnitude: float = 0.0
    mean_vertical_velocity_before: float = 0.0
    mean_vertical_velocity_after: float = 0.0


@dataclass(frozen=True)
class SupportMode:
    name: str
    floor_contact_mode: str = "penalty_old"
    gravity_enabled: bool = True
    bullet_floor_enabled: bool = True
    custom_floor_enabled: bool = True
    custom_plate_enabled: bool = True
    plate_enabled: bool = True
    freeze_after_seconds: float | None = None
    support_geometry: str | None = None


class SurfaceCalibrationScene:
    def __init__(self, base: ShowBase, audio_features: AudioFeatures, config: SurfaceCalibrationConfig) -> None:
        self.base = base
        self.audio_features = audio_features
        self.config = config
        self.sim_time = 0.0
        self.phase_name = "rest"
        self.metrics_output = Path(config.metrics_output)
        self.plate_calibration_output = Path(config.plate_calibration_output)
        self.impact_calibration_output = Path(config.impact_calibration_output)
        self.support_ablation_output = Path(config.support_ablation_output)
        self.contact_comparison_output = Path(config.contact_comparison_output)
        self.gravity_contact_sweep_output = Path(config.gravity_contact_sweep_output)
        self.rest_sweep_output = Path(config.rest_sweep_output)
        self.inversion_events_output = Path(config.inversion_events_output)
        self.metrics_rows: list[dict[str, object]] = []
        self.safety_events: list[str] = []
        self.inversion_events: list[dict[str, object]] = []
        self.rng = np.random.default_rng(config.seed)
        self.is_rest_debug = config.surface_rest_debug
        self.is_floating_rest_debug = config.surface_floating_rest_debug
        self.surface_contact_mode = config.surface_contact_mode if config.surface_contact_mode in SURFACE_CONTACT_MODES else "support_spheres"
        if self.is_floating_rest_debug:
            self.support_mode = SupportMode(
                name="floating_rest",
                floor_contact_mode="bullet_only",
                gravity_enabled=False,
                bullet_floor_enabled=False,
                custom_floor_enabled=False,
                custom_plate_enabled=False,
                plate_enabled=False,
            )
        else:
            self.support_mode = self._make_support_mode(
                self.surface_contact_mode,
                name="rest_only" if self.is_rest_debug else "calibration",
                plate_enabled=not self.is_rest_debug,
            )
        self.current_support_stats = SupportStats()
        self.latest_raw_solver_volume = 0.0
        self.latest_post_support_volume = 0.0
        self.support_bodies: list[BulletRigidBodyNode] = []

        self.root = self.base.render.attachNewNode("surface-calibration-scene")
        self.base.setBackgroundColor(0.02, 0.02, 0.024, 1.0)
        self.base.camLens.setNearFar(0.1, 200.0)
        self.base.camLens.setFov(34.0)
        self.base.camera.setPos(4.0, -12.2, 4.3)
        self.base.camera.lookAt(0.0, 0.0, 1.45)

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.surface_shader = Shader.load(
            Shader.SL_GLSL,
            Filename.fromOsSpecific(str(shader_dir / "soft_debug.vert")),
            Filename.fromOsSpecific(str(shader_dir / "soft_debug.frag")),
        )

        self.thresholds = SafetyThresholds(minimum_safe_volume_ratio=config.minimum_safe_volume_ratio)
        self.motion_config = PlateMotionConfig(
            plate_speed=config.plate_speed,
            plate_travel=config.plate_travel,
            plate_hold=config.plate_hold,
            plate_clearance=config.plate_clearance,
        )
        self.selected_impact_speed = 6.4
        self.selected_plate_travel = config.plate_travel
        self.selected_preset = config.softbody_preset if config.softbody_preset in SOFTBODY_PRESETS else "stable_medium"
        self.selected_preset_override: SoftBodyPreset | None = None
        self.surface_mass = 5.4
        self.gravity_scale = 1.0
        self.floor_contact_stiffness = 4200.0
        self.floor_contact_damping = 260.0
        self.floor_contact_clearance = -0.008
        self.floor_velocity_recovery = 0.0
        self.plate_contact_stiffness = 1120.0
        self.plate_contact_damping = 96.0
        self.plate_contact_clearance = 0.028
        self.wall_contact_stiffness = 120.0
        self.wall_contact_damping = 18.0

        self._run_support_ablation()
        self._run_contact_comparison()
        self._run_gravity_contact_sweep()
        self._run_rest_sweep()
        if not self.is_rest_debug and not self.is_floating_rest_debug:
            self._run_calibration_sweeps()

        gravity = Vec3(0.0, 0.0, (-9.81 * self.gravity_scale) if self.support_mode.gravity_enabled else 0.0)
        self.world = BulletWorld()
        self.world.setGravity(gravity)
        self.world_info = self.world.getWorldInfo()
        self.world_info.setGravity(gravity)

        self._build_environment()
        self._build_floor()
        self._build_floor_support_geometry(self.root, self.world, self.support_mode)
        self._build_lights()
        self._build_containment_walls()

        self.soft_body = self._create_soft_body(self.selected_preset_override or self.selected_preset, self.config.softbody_visualization)
        self.plate_np: NodePath | None = None
        self.plate_body: BulletRigidBodyNode | None = None
        self.plate_visual: NodePath | None = None
        if self.support_mode.plate_enabled:
            self.plate_np, self.plate_body = create_debug_plate(
                self.root,
                half_extents=Vec3(1.95, 1.95, 0.20),
                position=Vec3(0.0, 0.0, self._plate_start_height()),
                kinematic=True,
            )
            self.world.attachRigidBody(self.plate_body)
            self.plate_visual = self._create_visual_geom(
                "surface-calibration-plate",
                "rectangular_cuboid",
                radius=1.0,
                color=(0.84, 0.86, 0.90, 1.0),
                parent=self.plate_np,
            )
            self.plate_visual.setScale(1.95, 1.95, 0.16)

        self.projectile_np: NodePath | None = None
        self.projectile_body: BulletRigidBodyNode | None = None
        self.projectile_visual: NodePath | None = None
        self.impact_spawned = False
        self.impact_contact_applied = False
        self.safety_release_triggered = False
        self.release_started_at: float | None = None
        self.release_origin_height: float = self._plate_start_height()
        self.last_plate_height = self._plate_start_height()
        self.plate_phase_started = False
        self.plate_baseline_min_height = 0.0
        self.plate_baseline_triangle = 0.0
        self.rest_ghost_np: NodePath | None = None
        if config.show_rest_ghost:
            self.rest_ghost_np = self._create_rest_ghost()
        self.overlay = self._create_overlay()

    def _create_overlay(self) -> OnscreenText:
        return OnscreenText(
            text="surface calibration",
            parent=self.base.a2dTopLeft,
            pos=(0.04, -0.08),
            scale=0.045,
            fg=(0.92, 0.96, 1.0, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )

    def _plate_start_height(self) -> float:
        return float(self._soft_body_start_height() + 1.55 + self.motion_config.plate_clearance + 0.28)

    def _soft_body_start_height(self) -> float:
        return 1.70

    def _make_support_mode(self, contact_mode: str, *, name: str, plate_enabled: bool) -> SupportMode:
        if contact_mode == "bullet_only":
            return SupportMode(
                name=name,
                floor_contact_mode=contact_mode,
                gravity_enabled=True,
                bullet_floor_enabled=True,
                custom_floor_enabled=False,
                custom_plate_enabled=plate_enabled,
                plate_enabled=plate_enabled,
            )
        if contact_mode == "penalty_old":
            return SupportMode(
                name=name,
                floor_contact_mode=contact_mode,
                gravity_enabled=True,
                bullet_floor_enabled=False,
                custom_floor_enabled=True,
                custom_plate_enabled=plate_enabled,
                plate_enabled=plate_enabled,
            )
        if contact_mode in {"projection", "velocity_projection"}:
            return SupportMode(
                name=name,
                floor_contact_mode=contact_mode,
                gravity_enabled=True,
                bullet_floor_enabled=True,
                custom_floor_enabled=True,
                custom_plate_enabled=plate_enabled,
                plate_enabled=plate_enabled,
            )
        if contact_mode == "support_spheres":
            return SupportMode(
                name=name,
                floor_contact_mode=contact_mode,
                gravity_enabled=True,
                bullet_floor_enabled=False,
                custom_floor_enabled=False,
                custom_plate_enabled=plate_enabled,
                plate_enabled=plate_enabled,
                support_geometry="support_spheres",
            )
        if contact_mode == "soft_contact_grid":
            return SupportMode(
                name=name,
                floor_contact_mode=contact_mode,
                gravity_enabled=True,
                bullet_floor_enabled=False,
                custom_floor_enabled=False,
                custom_plate_enabled=plate_enabled,
                plate_enabled=plate_enabled,
                support_geometry="soft_contact_grid",
            )
        return self._make_support_mode("support_spheres", name=name, plate_enabled=plate_enabled)

    def _build_environment(self) -> None:
        backdrop = CardMaker("surface-calibration-backdrop")
        backdrop.setFrame(-18, 18, -10, 14)
        backdrop_np = self.root.attachNewNode(backdrop.generate())
        backdrop_np.setP(90)
        backdrop_np.setPos(0.0, 18.0, 5.8)
        backdrop_np.setColor(Vec4(0.045, 0.048, 0.058, 1.0))

    def _build_floor(self) -> None:
        card = CardMaker("surface-floor-card")
        card.setFrame(-18, 18, -18, 18)
        floor_parent = self.root
        if self.support_mode.bullet_floor_enabled:
            floor_shape = BulletPlaneShape(Vec3(0, 0, 1), 0.0)
            floor_node = BulletRigidBodyNode("surface-floor")
            floor_node.addShape(floor_shape)
            floor_node.setFriction(0.96)
            floor_parent = self.root.attachNewNode(floor_node)
            self.world.attachRigidBody(floor_node)
        visual = floor_parent.attachNewNode(card.generate())
        visual.setP(-90)
        visual.setColor(0.12, 0.12, 0.13, 1.0)
        material = Material()
        material.setAmbient(Vec4(0.08, 0.08, 0.09, 1.0))
        material.setDiffuse(Vec4(0.14, 0.14, 0.15, 1.0))
        material.setSpecular(Vec4(0.32, 0.32, 0.34, 1.0))
        material.setShininess(18.0)
        visual.setMaterial(material, 1)

    def _build_floor_support_geometry(self, parent: NodePath, world: BulletWorld, support_mode: SupportMode) -> list[BulletRigidBodyNode]:
        bodies: list[BulletRigidBodyNode] = []
        if support_mode.support_geometry is None:
            return bodies
        if support_mode.support_geometry == "support_spheres":
            radius = 0.28
            spacing = 0.54
            extent = range(-4, 5)
            z = -0.16
        else:
            radius = 0.20
            spacing = 0.40
            extent = range(-5, 6)
            z = -0.10
        for ix in extent:
            for iy in extent:
                body = BulletRigidBodyNode(f"{support_mode.support_geometry}-{ix}-{iy}")
                body.addShape(BulletSphereShape(radius))
                body.setFriction(0.94)
                body.setRestitution(0.0)
                body_np = parent.attachNewNode(body)
                body_np.setPos(ix * spacing, iy * spacing, z)
                world.attachRigidBody(body)
                bodies.append(body)
        return bodies

    def _build_lights(self) -> None:
        ambient = AmbientLight("surface-ambient")
        ambient.setColor(Vec4(0.34, 0.34, 0.36, 1.0))
        ambient_np = self.root.attachNewNode(ambient)
        self.base.render.setLight(ambient_np)

        key = DirectionalLight("surface-key")
        key.setColor(Vec4(1.42, 1.38, 1.28, 1.0))
        key_np = self.root.attachNewNode(key)
        key_np.setPos(-7.2, -8.0, 11.0)
        key_np.lookAt(0.0, 0.0, 1.8)
        self.base.render.setLight(key_np)

        back = PointLight("surface-back")
        back.setColor(Vec4(1.56, 1.38, 1.08, 1.0))
        back_np = self.root.attachNewNode(back)
        back_np.setPos(0.0, 8.2, 6.0)
        self.base.render.setLight(back_np)

        fill = PointLight("surface-fill")
        fill.setColor(Vec4(0.72, 0.78, 0.86, 1.0))
        fill_np = self.root.attachNewNode(fill)
        fill_np.setPos(-5.6, -5.5, 4.8)
        self.base.render.setLight(fill_np)

    def _attach_wall(self, parent: NodePath, world: BulletWorld, position: Vec3, half_extents: Vec3) -> None:
        wall = BulletRigidBodyNode("surface-wall")
        wall.addShape(BulletBoxShape(half_extents))
        wall.setFriction(0.88)
        wall_np = parent.attachNewNode(wall)
        wall_np.setPos(position)
        world.attachRigidBody(wall)

    def _build_containment_walls(self) -> None:
        self._attach_wall(self.root, self.world, Vec3(4.8, 0.0, 2.5), Vec3(0.22, 5.0, 3.2))
        self._attach_wall(self.root, self.world, Vec3(-4.8, 0.0, 2.5), Vec3(0.22, 5.0, 3.2))
        self._attach_wall(self.root, self.world, Vec3(0.0, 4.8, 2.5), Vec3(5.0, 0.22, 3.2))
        self._attach_wall(self.root, self.world, Vec3(0.0, -5.2, 2.5), Vec3(5.0, 0.22, 3.2))

    def _create_visual_geom(self, name: str, shape_kind: str, radius: float, color: tuple[float, float, float, float], parent: NodePath) -> NodePath:
        name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name)) % 997
        mesh = generate_shape_mesh_data(shape_kind, seed=self.config.seed + name_seed, radius=radius)
        geom = create_render_geom_node(name, mesh.vertices, mesh.normals, mesh.texcoords, mesh.indices)
        node = parent.attachNewNode(geom)
        node.setShader(self.surface_shader)
        node.setShaderInput("base_color", Vec4(*color))
        return node

    def _create_soft_body(self, preset_name: str | SoftBodyPreset, visualization: str) -> TrueSoftBodyObject:
        soft_body = TrueSoftBodyObject(
            self.root,
            self.world,
            self.world_info,
            variant="surface",
            radius=1.55,
            mass=self.surface_mass,
            seed=self.config.seed,
            name="surface-calibration",
            initial_position=Vec3(0.0, 0.0, self._soft_body_start_height()),
            preset_name=preset_name,
            visualization=visualization,
            max_displacement_visual=0.12,
        )
        soft_body.render_node.setShader(self.surface_shader)
        soft_body.render_node.setShaderInput("base_color", Vec4(0.24, 0.88, 1.0, 1.0))
        soft_body.wireframe_np.setShader(self.surface_shader)
        soft_body.wireframe_np.setShaderInput("base_color", Vec4(0.96, 0.96, 1.0, 1.0))
        return soft_body

    def _create_rest_ghost(self) -> NodePath:
        ghost_geom = create_render_geom_node(
            "surface-rest-ghost",
            self.soft_body.render_mesh.vertices + self.soft_body.initial_position[None, :],
            self.soft_body.render_mesh.normals,
            self.soft_body.render_mesh.texcoords,
            self.soft_body.render_mesh.indices,
        )
        ghost = self.root.attachNewNode(ghost_geom)
        ghost.setRenderModeWireframe()
        ghost.setRenderModeThickness(1.2)
        ghost.setShader(self.surface_shader)
        ghost.setShaderInput("base_color", Vec4(1.0, 1.0, 1.0, 0.20))
        ghost.setColorScale(Vec4(1.0, 1.0, 1.0, 0.18))
        ghost.setDepthWrite(False)
        ghost.setBin("transparent", 10)
        return ghost

    def _spawn_projectile(self, world: BulletWorld, parent: NodePath, impact_speed: float) -> tuple[NodePath, BulletRigidBodyNode, NodePath]:
        body = BulletRigidBodyNode("surface-impact-projectile")
        body.addShape(BulletSphereShape(0.16))
        body.setMass(0.72)
        body.setFriction(0.72)
        body.setRestitution(0.10)
        body_np = parent.attachNewNode(body)
        target = np.asarray([1.00, -1.00, self._soft_body_start_height() + 1.02], dtype=np.float32)
        start = target + np.asarray([2.0, -1.9, 0.22], dtype=np.float32)
        body_np.setPos(float(start[0]), float(start[1]), float(start[2]))
        world.attachRigidBody(body)
        velocity = target - start
        velocity = velocity / max(float(np.linalg.norm(velocity)), 1e-6) * impact_speed
        body.setLinearVelocity(Vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])))
        body.setAngularVelocity(Vec3(2.2, -1.4, 1.1))
        visual = self._create_visual_geom(
            "surface-impact-visual",
            "rounded_octahedron",
            radius=0.20,
            color=(1.0, 0.76, 0.24, 1.0),
            parent=body_np,
        )
        return body_np, body, visual

    def _current_plate_height(self, time_seconds: float) -> tuple[str, float]:
        start_height = self._plate_start_height()
        if self.safety_release_triggered and self.release_started_at is not None:
            release_elapsed = max(time_seconds - self.release_started_at, 0.0)
            duration = self.selected_plate_travel / max(self.motion_config.plate_speed, 1e-6)
            progress = min(max(release_elapsed / max(duration, 1e-6), 0.0), 1.0)
            eased = progress * progress * (3.0 - 2.0 * progress)
            height = self.release_origin_height + (start_height - self.release_origin_height) * eased
            if progress >= 1.0:
                return "final_recovery", start_height
            return "plate_release", height
        return plate_height_for_time(
            time_seconds,
            start_height,
            PlateMotionConfig(
                rest_duration=self.motion_config.rest_duration,
                impact_time=self.motion_config.impact_time,
                pre_plate_duration=self.motion_config.pre_plate_duration,
                plate_speed=self.motion_config.plate_speed,
                plate_travel=self.selected_plate_travel,
                plate_hold=self.motion_config.plate_hold,
                plate_clearance=self.motion_config.plate_clearance,
            ),
        )

    def _build_metrics_row(self, dt: float, physics_seconds: float, metrics: SoftBodyMetrics) -> dict[str, object]:
        return {
            "time": round(self.sim_time, 6),
            "phase": self.phase_name,
            "preset": self.selected_preset,
            "variant": "surface",
            "volume_ratio": metrics.volume_ratio,
            "current_volume": metrics.current_volume,
            "rest_volume": metrics.rest_volume,
            "center_of_mass_x": float(metrics.center_of_mass[0]),
            "center_of_mass_y": float(metrics.center_of_mass[1]),
            "center_of_mass_z": float(metrics.center_of_mass[2]),
            "center_of_mass_drift": metrics.center_of_mass_drift,
            "raw_mean_displacement": metrics.raw_mean_displacement,
            "raw_max_displacement": metrics.raw_max_displacement,
            "aligned_rms_deformation": metrics.aligned_rms_deformation,
            "aligned_max_deformation": metrics.aligned_max_deformation,
            "max_velocity": metrics.max_velocity,
            "mean_velocity": metrics.mean_velocity,
            "bbox_x": float(metrics.bounding_box_dimensions[0]),
            "bbox_y": float(metrics.bounding_box_dimensions[1]),
            "bbox_z": float(metrics.bounding_box_dimensions[2]),
            "minimum_node_height": metrics.minimum_node_height,
            "nodes_below_floor": metrics.nodes_below_floor,
            "supported_node_count": self.current_support_stats.supported_node_count,
            "mean_support_force": self.current_support_stats.mean_support_force,
            "max_support_force": self.current_support_stats.max_support_force,
            "total_support_force": self.current_support_stats.total_support_force,
            "corrected_node_count": self.current_support_stats.corrected_node_count,
            "damped_node_count": self.current_support_stats.damped_node_count,
            "total_correction_magnitude": self.current_support_stats.total_correction_magnitude,
            "mean_vertical_velocity_before": self.current_support_stats.mean_vertical_velocity_before,
            "mean_vertical_velocity_after": self.current_support_stats.mean_vertical_velocity_after,
            "min_triangle_area_proxy": metrics.min_triangle_area_proxy,
            "max_edge_stretch_ratio": metrics.max_edge_stretch_ratio,
            "min_edge_compression_ratio": metrics.min_edge_compression_ratio,
            "top_region_deformation": metrics.top_region_deformation,
            "bottom_region_deformation": metrics.bottom_region_deformation,
            "struck_corner_deformation": metrics.struck_corner_deformation,
            "opposite_corner_deformation": metrics.opposite_corner_deformation,
            "kinetic_energy_proxy": metrics.kinetic_energy_proxy,
            "node_count": metrics.node_count,
            "solver_substeps": self.config.physics_substeps,
            "physics_substep_dt": self.config.physics_substep_dt,
            "simulation_step_duration_ms": physics_seconds * 1000.0,
            "plate_position": self.last_plate_height,
            "plate_travel": self.selected_plate_travel,
            "plate_hold": self.motion_config.plate_hold,
            "impact_speed": self.selected_impact_speed,
            "safety_release_triggered": self.safety_release_triggered,
            "safety_reasons": "|".join(self.safety_events[-1:]) if self.safety_events else "",
            "dt": dt,
            "raw_solver_volume": self.latest_raw_solver_volume,
            "post_support_volume": self.latest_post_support_volume,
            "support_mode": self.support_mode.name,
        }

    def _update_overlay(self, metrics: SoftBodyMetrics) -> None:
        self.overlay.setText(
            "\n".join(
                [
                    f"phase: {self.phase_name}",
                    f"contact: {self.support_mode.floor_contact_mode}",
                    f"time: {self.sim_time:05.2f}s",
                    f"volume ratio: {metrics.volume_ratio:0.3f}",
                    f"aligned rms: {metrics.aligned_rms_deformation:0.3f}",
                    f"aligned max: {metrics.aligned_max_deformation:0.3f}",
                    f"max vel: {metrics.max_velocity:0.3f}",
                    f"min node z: {metrics.minimum_node_height:0.3f}",
                    f"support nodes: {self.current_support_stats.supported_node_count}",
                    f"support total: {self.current_support_stats.total_support_force:0.1f}",
                    f"corrected: {self.current_support_stats.corrected_node_count}",
                    f"damped: {self.current_support_stats.damped_node_count}",
                    f"plate z: {self.last_plate_height:0.3f}",
                    f"substeps: {self.config.physics_substeps}",
                ]
            )
        )

    def _simulate_candidate(
        self,
        preset_name: str | SoftBodyPreset,
        plate_travel: float,
        impact_speed: float,
        include_plate: bool,
        total_duration: float,
    ) -> dict[str, object]:
        preset_label = preset_name.name if isinstance(preset_name, SoftBodyPreset) else preset_name
        parent = NodePath("surface-calibration-temp")
        world = BulletWorld()
        world.setGravity(Vec3(0.0, 0.0, -9.81))
        world_info = world.getWorldInfo()
        world_info.setGravity(Vec3(0.0, 0.0, -9.81))

        floor_node: BulletRigidBodyNode | None = None
        if self.support_mode.bullet_floor_enabled:
            floor_shape = BulletPlaneShape(Vec3(0, 0, 1), 0.0)
            floor_node = BulletRigidBodyNode("temp-floor")
            floor_node.addShape(floor_shape)
            parent.attachNewNode(floor_node)
            world.attachRigidBody(floor_node)
        support_bodies = self._build_floor_support_geometry(parent, world, self.support_mode)
        self._attach_wall(parent, world, Vec3(4.8, 0.0, 2.5), Vec3(0.22, 5.0, 3.2))
        self._attach_wall(parent, world, Vec3(-4.8, 0.0, 2.5), Vec3(0.22, 5.0, 3.2))
        self._attach_wall(parent, world, Vec3(0.0, 4.8, 2.5), Vec3(5.0, 0.22, 3.2))
        self._attach_wall(parent, world, Vec3(0.0, -5.2, 2.5), Vec3(5.0, 0.22, 3.2))

        soft_body = TrueSoftBodyObject(
            parent,
            world,
            world_info,
            variant="surface",
            radius=1.55,
            mass=self.surface_mass,
            seed=self.config.seed,
            name=f"temp-{preset_label}",
            initial_position=Vec3(0.0, 0.0, self._soft_body_start_height()),
            preset_name=preset_name,
            visualization="shaded",
            max_displacement_visual=0.12,
        )
        soft_body.render_node.hide()
        soft_body.wireframe_np.hide()
        soft_body.node_points_np.hide()

        plate_np, plate_body = create_debug_plate(
            parent,
            half_extents=Vec3(1.95, 1.95, 0.20),
            position=Vec3(0.0, 0.0, self._plate_start_height()),
            kinematic=True,
        )
        world.attachRigidBody(plate_body)

        projectile_np: NodePath | None = None
        projectile_body: BulletRigidBodyNode | None = None
        impact_spawned = False
        impact_contact_applied = False
        safety_release_triggered = False
        release_started_at: float | None = None
        release_origin_height = self._plate_start_height()
        plate_phase_started = False
        plate_baseline_min_height = 0.0
        plate_baseline_triangle = 0.0
        minimum_volume_ratio = math.inf
        recovery_volume_ratio = 0.0
        max_aligned = 0.0
        max_velocity = 0.0
        min_height = math.inf
        has_non_finite = False
        left_safe_region = False
        final_rms = 0.0

        frame_dt = 1.0 / 30.0
        motion = PlateMotionConfig(
            rest_duration=self.motion_config.rest_duration,
            impact_time=self.motion_config.impact_time,
            pre_plate_duration=self.motion_config.pre_plate_duration,
            plate_speed=self.motion_config.plate_speed,
            plate_travel=plate_travel,
            plate_hold=self.motion_config.plate_hold,
            plate_clearance=self.motion_config.plate_clearance,
        )

        for frame_index in range(int(round(total_duration / frame_dt))):
            sim_time = frame_index * frame_dt
            if not impact_spawned and sim_time >= motion.impact_time:
                projectile_np, projectile_body, _ = self._spawn_projectile(world, parent, impact_speed)
                impact_spawned = True

            if include_plate:
                if safety_release_triggered and release_started_at is not None:
                    release_elapsed = max(sim_time - release_started_at, 0.0)
                    duration = plate_travel / max(motion.plate_speed, 1e-6)
                    progress = min(max(release_elapsed / max(duration, 1e-6), 0.0), 1.0)
                    eased = progress * progress * (3.0 - 2.0 * progress)
                    plate_height = release_origin_height + (self._plate_start_height() - release_origin_height) * eased
                    phase_name = "plate_release"
                else:
                    phase_name, plate_height = plate_height_for_time(sim_time, self._plate_start_height(), motion)
                plate_np.setZ(plate_height)
                if phase_name == "plate_descent" and not plate_phase_started:
                    baseline_metrics = soft_body.get_metrics()
                    plate_phase_started = True
                    plate_baseline_min_height = baseline_metrics.minimum_node_height
                    plate_baseline_triangle = baseline_metrics.min_triangle_area_proxy
            else:
                phase_name = "impact_recovery"
                plate_height = self._plate_start_height()

            if projectile_np is not None and projectile_body is not None and not impact_contact_applied:
                center = np.asarray([projectile_np.getX(), projectile_np.getY(), projectile_np.getZ()], dtype=np.float32)
                positions = soft_body.get_node_positions()
                distances = np.linalg.norm(positions - center[None, :], axis=1)
                indices = np.where(distances <= 0.30)[0].astype(np.int32)
                if indices.size > 0:
                    velocity = projectile_body.getLinearVelocity()
                    impulse = np.asarray([velocity.x, velocity.y, velocity.z], dtype=np.float32) * 0.28
                    soft_body.apply_velocity_to_indices(indices, impulse)
                    projectile_body.setLinearVelocity(projectile_body.getLinearVelocity() * 0.15)
                    impact_contact_applied = True

            self._apply_custom_contacts(soft_body, plate_height, phase_name)

            world.doPhysics(frame_dt, self.config.physics_substeps, self.config.physics_substep_dt)
            self._apply_post_step_contacts(soft_body, plate_height, phase_name)
            metrics = soft_body.get_metrics()
            minimum_volume_ratio = min(minimum_volume_ratio, metrics.volume_ratio)
            recovery_volume_ratio = metrics.volume_ratio
            max_aligned = max(max_aligned, metrics.aligned_max_deformation)
            max_velocity = max(max_velocity, metrics.max_velocity)
            min_height = min(min_height, metrics.minimum_node_height)
            has_non_finite = has_non_finite or metrics.has_non_finite
            left_safe_region = left_safe_region or abs(float(metrics.center_of_mass[0])) > self.thresholds.camera_safe_radius or abs(float(metrics.center_of_mass[1])) > self.thresholds.camera_safe_radius
            final_rms = metrics.aligned_rms_deformation

            if include_plate:
                if phase_name in {"plate_descent", "plate_hold"}:
                    dynamic_min_height = max(
                        metrics.minimum_node_height,
                        min(self.thresholds.floor_tolerance, plate_baseline_min_height - 0.035),
                    )
                    dynamic_triangle = max(
                        metrics.min_triangle_area_proxy,
                        min(self.thresholds.min_triangle_area_proxy, plate_baseline_triangle - 0.04),
                    )
                    trigger, _ = should_trigger_safe_release(
                        metrics.volume_ratio,
                        dynamic_min_height,
                        dynamic_triangle,
                        metrics.max_edge_stretch_ratio,
                        metrics.min_edge_compression_ratio,
                        metrics.has_non_finite,
                        self.thresholds,
                    )
                    if trigger and not safety_release_triggered:
                        safety_release_triggered = True
                        release_started_at = sim_time
                        release_origin_height = plate_np.getZ()

        result = {
            "minimum_volume_ratio": minimum_volume_ratio,
            "recovery_volume_ratio": recovery_volume_ratio,
            "permanent_deformation": final_rms,
            "max_displacement": max_aligned,
            "max_node_velocity": max_velocity,
            "minimum_node_height": min_height,
            "nodes_below_floor": metrics.nodes_below_floor,
            "body_left_safe_region": left_safe_region,
            "has_non_finite": has_non_finite,
        }
        soft_body.destroy()
        if projectile_body is not None:
            world.removeRigidBody(projectile_body)
        world.removeRigidBody(plate_body)
        if floor_node is not None:
            world.removeRigidBody(floor_node)
        for body in support_bodies:
            world.removeRigidBody(body)
        parent.removeNode()
        return result

    def _simulate_rest_case(
        self,
        *,
        case_name: str,
        support_mode: SupportMode,
        duration: float,
        preset: SoftBodyPreset | str | None = None,
        mass: float = 5.4,
        gravity_scale: float = 1.0,
        floor_contact_stiffness: float | None = None,
        floor_contact_damping: float | None = None,
    ) -> dict[str, object]:
        parent = NodePath(f"rest-{case_name}")
        gravity_value = -9.81 * gravity_scale if support_mode.gravity_enabled else 0.0
        world = BulletWorld()
        world.setGravity(Vec3(0.0, 0.0, gravity_value))
        world_info = world.getWorldInfo()
        world_info.setGravity(Vec3(0.0, 0.0, gravity_value))

        floor_node: BulletRigidBodyNode | None = None
        if support_mode.bullet_floor_enabled:
            floor_shape = BulletPlaneShape(Vec3(0, 0, 1), 0.0)
            floor_node = BulletRigidBodyNode(f"{case_name}-floor")
            floor_node.addShape(floor_shape)
            floor_np = parent.attachNewNode(floor_node)
            world.attachRigidBody(floor_node)
        support_bodies = self._build_floor_support_geometry(parent, world, support_mode)

        soft_body = TrueSoftBodyObject(
            parent,
            world,
            world_info,
            variant="surface",
            radius=1.55,
            mass=mass,
            seed=self.config.seed,
            name=f"rest-{case_name}",
            initial_position=Vec3(0.0, 0.0, self._soft_body_start_height()),
            preset_name=preset or self.selected_preset,
            visualization="shaded",
            max_displacement_visual=0.12,
        )
        soft_body.render_node.hide()
        soft_body.wireframe_np.hide()
        soft_body.node_points_np.hide()

        previous_support_mode = self.support_mode
        previous_stats = self.current_support_stats
        previous_floor_stiffness = self.floor_contact_stiffness
        previous_floor_damping = self.floor_contact_damping
        self.support_mode = support_mode
        self.current_support_stats = SupportStats()
        if floor_contact_stiffness is not None:
            self.floor_contact_stiffness = floor_contact_stiffness
        if floor_contact_damping is not None:
            self.floor_contact_damping = floor_contact_damping

        frame_dt = 1.0 / 30.0
        min_volume_ratio = math.inf
        max_volume_ratio = 0.0
        final_metrics: SoftBodyMetrics | None = None
        total_support_force = 0.0
        support_samples = 0

        for frame_index in range(int(round(duration / frame_dt))):
            sim_time = frame_index * frame_dt
            if support_mode.freeze_after_seconds is not None and sim_time >= support_mode.freeze_after_seconds:
                world.setGravity(Vec3(0.0, 0.0, 0.0))
                world_info.setGravity(Vec3(0.0, 0.0, 0.0))

            self._reset_support_stats()
            self._apply_custom_contacts(soft_body, self._plate_start_height(), "rest_settle")
            world.doPhysics(frame_dt, self.config.physics_substeps, self.config.physics_substep_dt)
            self._apply_post_step_contacts(soft_body, self._plate_start_height(), "rest_settle")
            metrics = soft_body.get_metrics()
            min_volume_ratio = min(min_volume_ratio, metrics.volume_ratio)
            max_volume_ratio = max(max_volume_ratio, metrics.volume_ratio)
            total_support_force += self.current_support_stats.total_support_force
            support_samples += 1
            final_metrics = metrics

        assert final_metrics is not None
        result = {
            "case": case_name,
            "support_mode": support_mode.name,
            "contact_mode": support_mode.floor_contact_mode,
            "final_volume_ratio": final_metrics.volume_ratio,
            "final_aligned_rms_deformation": final_metrics.aligned_rms_deformation,
            "final_aligned_max_deformation": final_metrics.aligned_max_deformation,
            "min_volume_ratio": min_volume_ratio,
            "max_volume_ratio": max_volume_ratio,
            "min_node_height": final_metrics.minimum_node_height,
            "nodes_below_floor": final_metrics.nodes_below_floor,
            "total_support_force": total_support_force,
            "mean_support_force": total_support_force / max(support_samples, 1),
            "corrected_node_count": self.current_support_stats.corrected_node_count,
            "damped_node_count": self.current_support_stats.damped_node_count,
            "total_correction_magnitude": self.current_support_stats.total_correction_magnitude,
            "mean_vertical_velocity_before": self.current_support_stats.mean_vertical_velocity_before,
            "mean_vertical_velocity_after": self.current_support_stats.mean_vertical_velocity_after,
            "drifts_or_falls": bool(abs(float(final_metrics.center_of_mass[0])) > 0.5 or abs(float(final_metrics.center_of_mass[1])) > 0.5 or float(final_metrics.center_of_mass[2]) < 0.5),
            "penetrates_floor": bool(final_metrics.nodes_below_floor > 0 or final_metrics.minimum_node_height < self.thresholds.floor_tolerance),
            "floor_contact_active": support_mode.bullet_floor_enabled or support_mode.custom_floor_enabled or support_mode.support_geometry is not None,
            "raw_solver_volume": final_metrics.current_volume,
            "rest_volume": final_metrics.rest_volume,
            "mean_node_velocity": final_metrics.mean_velocity,
            "max_node_velocity": final_metrics.max_velocity,
            "triangle_inversion": bool(final_metrics.min_triangle_area_proxy < -1e-4),
            "compressed_equilibrium": bool(final_metrics.volume_ratio < 0.97),
            "jitter": bool(final_metrics.mean_velocity > 0.22 or final_metrics.max_velocity > 1.4),
        }

        self.support_mode = previous_support_mode
        self.current_support_stats = previous_stats
        self.floor_contact_stiffness = previous_floor_stiffness
        self.floor_contact_damping = previous_floor_damping
        soft_body.destroy()
        if floor_node is not None:
            world.removeRigidBody(floor_node)
        for body in support_bodies:
            world.removeRigidBody(body)
        parent.removeNode()
        return result

    def _run_support_ablation(self) -> None:
        if self.support_ablation_output.exists():
            return
        cases = [
            ("bullet_only", SupportMode(name="bullet_only", gravity_enabled=True, bullet_floor_enabled=True, custom_floor_enabled=False, custom_plate_enabled=False, plate_enabled=False)),
            ("custom_only", self._make_support_mode("penalty_old", name="custom_only", plate_enabled=False)),
            ("bullet_and_custom", SupportMode(name="bullet_and_custom", floor_contact_mode="penalty_old", gravity_enabled=True, bullet_floor_enabled=True, custom_floor_enabled=True, custom_plate_enabled=False, plate_enabled=False)),
            ("floating_rest", SupportMode(name="floating_rest", gravity_enabled=False, bullet_floor_enabled=False, custom_floor_enabled=False, custom_plate_enabled=False, plate_enabled=False)),
            ("fall_then_freeze", SupportMode(name="fall_then_freeze", gravity_enabled=True, bullet_floor_enabled=False, custom_floor_enabled=False, custom_plate_enabled=False, plate_enabled=False, freeze_after_seconds=0.8)),
        ]
        rows = [self._simulate_rest_case(case_name=name, support_mode=mode, duration=3.0 if name != "floating_rest" else 2.0) for name, mode in cases]
        self.support_ablation_output.parent.mkdir(parents=True, exist_ok=True)
        with self.support_ablation_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _run_contact_comparison(self) -> None:
        if self.contact_comparison_output.exists():
            return
        rows = [
            self._simulate_rest_case(case_name=mode, support_mode=self._make_support_mode(mode, name=mode, plate_enabled=False), duration=4.0 if mode != "bullet_only" else 3.0)
            for mode in SURFACE_CONTACT_MODES
        ]
        self.contact_comparison_output.parent.mkdir(parents=True, exist_ok=True)
        with self.contact_comparison_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _run_gravity_contact_sweep(self) -> None:
        if self.gravity_contact_sweep_output.exists():
            return
        rows: list[dict[str, object]] = []
        for mode in ("projection", "velocity_projection", "support_spheres", "soft_contact_grid"):
            support_mode = self._make_support_mode(mode, name=f"{mode}_gravity", plate_enabled=False)
            for gravity_scale in (0.0, 0.25, 0.50, 0.75, 1.00):
                result = self._simulate_rest_case(
                    case_name=f"{mode}_g{gravity_scale:.2f}",
                    support_mode=support_mode,
                    duration=3.0,
                    gravity_scale=gravity_scale,
                )
                result["gravity_scale"] = gravity_scale
                rows.append(result)
        self.gravity_contact_sweep_output.parent.mkdir(parents=True, exist_ok=True)
        with self.gravity_contact_sweep_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _run_rest_sweep(self) -> None:
        if self._load_existing_rest_sweep():
            return
        base = SOFTBODY_PRESETS[self.selected_preset]
        presets = [
            ("rest_light", replace(base, name="rest_light", pressure=base.pressure + 0.6, volume_conservation=base.volume_conservation + 0.4, damping=base.damping + 0.02)),
            ("rest_balanced", replace(base, name="rest_balanced", pressure=base.pressure + 1.0, volume_conservation=base.volume_conservation + 0.8, damping=base.damping + 0.03)),
            ("rest_support", replace(base, name="rest_support", pressure=base.pressure + 0.8, volume_conservation=base.volume_conservation + 0.6, damping=base.damping + 0.04)),
            ("rest_hold", replace(base, name="rest_hold", linear_stiffness=min(base.linear_stiffness + 0.04, 0.68), angular_stiffness=min(base.angular_stiffness + 0.06, 0.48), volume_preservation=min(base.volume_preservation + 0.10, 0.90), pose_matching=min(base.pose_matching + 0.02, 0.12), pressure=base.pressure + 2.2, volume_conservation=base.volume_conservation + 1.8, damping=base.damping + 0.05)),
            ("rest_pressurized", replace(base, name="rest_pressurized", linear_stiffness=min(base.linear_stiffness + 0.06, 0.72), angular_stiffness=min(base.angular_stiffness + 0.08, 0.52), volume_preservation=min(base.volume_preservation + 0.14, 0.94), pose_matching=min(base.pose_matching + 0.03, 0.14), pressure=base.pressure + 3.0, volume_conservation=base.volume_conservation + 2.4, damping=base.damping + 0.06)),
        ]
        preset_lookup = {preset.name: preset for _, preset in presets}
        parameter_rows: list[dict[str, object]] = []
        for label, preset in presets:
            for gravity_scale in (0.85, 1.0):
                for mass, stiffness, damping in ((4.2, 2200.0, 160.0), (4.6, 2600.0, 190.0), (5.0, 3200.0, 220.0)):
                    result = self._simulate_rest_case(
                        case_name=f"{label}_{mass:.1f}_g{gravity_scale:.2f}",
                        support_mode=self._make_support_mode(self.surface_contact_mode, name="rest_sweep", plate_enabled=False),
                        duration=4.0,
                        preset=preset,
                        mass=mass,
                        gravity_scale=gravity_scale,
                        floor_contact_stiffness=stiffness,
                        floor_contact_damping=damping,
                    )
                    result.update(
                        {
                            "preset_name": preset.name,
                            "linear_stiffness": preset.linear_stiffness,
                            "angular_stiffness": preset.angular_stiffness,
                            "volume_preservation": preset.volume_preservation,
                            "pose_matching": preset.pose_matching,
                            "pressure": preset.pressure,
                            "volume_conservation": preset.volume_conservation,
                            "damping": preset.damping,
                            "mass": mass,
                            "gravity_scale": gravity_scale,
                            "support_stiffness": stiffness,
                            "support_damping": damping,
                        }
                    )
                    parameter_rows.append(result)
        ranked_rows = sorted(
            parameter_rows,
            key=lambda row: abs(float(row["final_volume_ratio"]) - 1.0) + float(row["final_aligned_rms_deformation"]) + max(0.0, -float(row["min_node_height"])) * 4.0,
        )
        for row in ranked_rows[:6]:
            verify = self._simulate_rest_case(
                case_name=f"verify_{row['preset_name']}_{row['mass']}_g{row['gravity_scale']}",
                support_mode=self._make_support_mode(self.surface_contact_mode, name="rest_verify", plate_enabled=False),
                duration=8.0,
                preset=preset_lookup[str(row["preset_name"])],
                mass=float(row["mass"]),
                gravity_scale=float(row["gravity_scale"]),
                floor_contact_stiffness=float(row["support_stiffness"]),
                floor_contact_damping=float(row["support_damping"]),
            )
            row["verified_final_volume_ratio"] = verify["final_volume_ratio"]
            row["verified_min_volume_ratio"] = verify["min_volume_ratio"]
            row["verified_min_node_height"] = verify["min_node_height"]
            row["verified_penetrates_floor"] = verify["penetrates_floor"]
            row["verified_final_aligned_rms"] = verify["final_aligned_rms_deformation"]
        for row in ranked_rows[6:]:
            row["verified_final_volume_ratio"] = ""
            row["verified_min_volume_ratio"] = ""
            row["verified_min_node_height"] = ""
            row["verified_penetrates_floor"] = ""
            row["verified_final_aligned_rms"] = ""
        valid_rows = [
            row
            for row in parameter_rows
            if row["verified_final_volume_ratio"] != "" and 0.97 <= float(row["verified_final_volume_ratio"]) <= 1.03 and not bool(row["verified_penetrates_floor"])
        ]
        fallback_rows = [
            row
            for row in parameter_rows
            if row["verified_final_volume_ratio"] != "" and float(row["verified_min_node_height"]) >= -0.03 and float(row["verified_final_volume_ratio"]) >= 0.94
        ]
        chosen_rows = valid_rows or fallback_rows
        if chosen_rows:
            best_row = min(
                chosen_rows,
                key=lambda row: abs(float(row["verified_final_volume_ratio"]) - 1.0) + float(row["verified_final_aligned_rms"]) + max(0.0, -float(row["verified_min_node_height"])) * 4.0,
            )
            self.surface_mass = float(best_row["mass"])
            self.gravity_scale = float(best_row["gravity_scale"])
            self.floor_contact_stiffness = float(best_row["support_stiffness"])
            self.floor_contact_damping = float(best_row["support_damping"])
            match = next((preset for _, preset in presets if preset.name == best_row["preset_name"]), None)
            if match is not None:
                self.selected_preset_override = match
                self.selected_preset = match.name
        self.rest_sweep_output.parent.mkdir(parents=True, exist_ok=True)
        with self.rest_sweep_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0].keys()))
            writer.writeheader()
            writer.writerows(parameter_rows)

    def _load_existing_rest_sweep(self) -> bool:
        if not self.rest_sweep_output.exists():
            return False
        try:
            with self.rest_sweep_output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows = [row for row in rows if row.get("verified_final_volume_ratio", "")]
            if not rows:
                return False
            best_row = min(
                rows,
                key=lambda row: abs(float(row["verified_final_volume_ratio"]) - 1.0) + float(row["verified_final_aligned_rms"]) + max(0.0, -float(row["verified_min_node_height"])) * 4.0,
            )
            self.surface_mass = float(best_row["mass"])
            self.gravity_scale = float(best_row["gravity_scale"])
            self.floor_contact_stiffness = float(best_row["support_stiffness"])
            self.floor_contact_damping = float(best_row["support_damping"])
            base = SOFTBODY_PRESETS[self.selected_preset]
            presets = {
                "rest_light": replace(base, name="rest_light", pressure=base.pressure + 0.6, volume_conservation=base.volume_conservation + 0.4, damping=base.damping + 0.02),
                "rest_balanced": replace(base, name="rest_balanced", pressure=base.pressure + 1.0, volume_conservation=base.volume_conservation + 0.8, damping=base.damping + 0.03),
                "rest_support": replace(base, name="rest_support", pressure=base.pressure + 0.8, volume_conservation=base.volume_conservation + 0.6, damping=base.damping + 0.04),
                "rest_hold": replace(base, name="rest_hold", linear_stiffness=min(base.linear_stiffness + 0.04, 0.68), angular_stiffness=min(base.angular_stiffness + 0.06, 0.48), volume_preservation=min(base.volume_preservation + 0.10, 0.90), pose_matching=min(base.pose_matching + 0.02, 0.12), pressure=base.pressure + 2.2, volume_conservation=base.volume_conservation + 1.8, damping=base.damping + 0.05),
                "rest_pressurized": replace(base, name="rest_pressurized", linear_stiffness=min(base.linear_stiffness + 0.06, 0.72), angular_stiffness=min(base.angular_stiffness + 0.08, 0.52), volume_preservation=min(base.volume_preservation + 0.14, 0.94), pose_matching=min(base.pose_matching + 0.03, 0.14), pressure=base.pressure + 3.0, volume_conservation=base.volume_conservation + 2.4, damping=base.damping + 0.06),
            }
            preset_name = str(best_row["preset_name"])
            if preset_name in presets:
                self.selected_preset_override = presets[preset_name]
                self.selected_preset = preset_name
            return True
        except Exception:
            return False

    def _run_calibration_sweeps(self) -> None:
        if self._load_existing_calibration():
            return
        sweep_preset: str | SoftBodyPreset = self.selected_preset_override or "stable_medium"
        impact_candidates: list[ImpactCalibrationResult] = []
        for impact_speed in (5.2, 5.8, 6.4, 7.0):
            result = self._simulate_candidate(sweep_preset, self.motion_config.plate_travel, impact_speed, include_plate=False, total_duration=4.0)
            passed = (
                result["minimum_volume_ratio"] >= 0.88
                and not result["body_left_safe_region"]
                and result["nodes_below_floor"] == 0
                and not result["has_non_finite"]
            )
            impact_candidates.append(
                ImpactCalibrationResult(
                    impact_speed=impact_speed,
                    minimum_volume_ratio=result["minimum_volume_ratio"],
                    recovery_volume_ratio=result["recovery_volume_ratio"],
                    permanent_deformation=result["permanent_deformation"],
                    max_displacement=result["max_displacement"],
                    max_node_velocity=result["max_node_velocity"],
                    minimum_node_height=result["minimum_node_height"],
                    nodes_below_floor=result["nodes_below_floor"],
                    body_left_safe_region=result["body_left_safe_region"],
                    has_non_finite=result["has_non_finite"],
                    passed=passed,
                )
            )
        impact_choice = select_impact_candidate(impact_candidates)
        if impact_choice is not None:
            self.selected_impact_speed = impact_choice.impact_speed
        self._write_impact_calibration_csv(impact_candidates)

        plate_candidates: list[PlateCalibrationResult] = []
        for plate_travel in (0.18, 0.24, 0.30, 0.36, 0.42):
            result = self._simulate_candidate(sweep_preset, plate_travel, self.selected_impact_speed, include_plate=True, total_duration=8.5)
            passed = (
                result["minimum_volume_ratio"] >= 0.72
                and 0.95 <= result["recovery_volume_ratio"] <= 1.05
                and not result["body_left_safe_region"]
                and result["nodes_below_floor"] == 0
                and not result["has_non_finite"]
            )
            plate_candidates.append(
                PlateCalibrationResult(
                    plate_travel=plate_travel,
                    minimum_volume_ratio=result["minimum_volume_ratio"],
                    recovery_volume_ratio=result["recovery_volume_ratio"],
                    permanent_deformation=result["permanent_deformation"],
                    max_displacement=result["max_displacement"],
                    max_node_velocity=result["max_node_velocity"],
                    minimum_node_height=result["minimum_node_height"],
                    nodes_below_floor=result["nodes_below_floor"],
                    body_left_safe_region=result["body_left_safe_region"],
                    has_non_finite=result["has_non_finite"],
                    passed=passed,
                )
            )
        plate_choice = select_plate_candidate(plate_candidates)
        if plate_choice is not None:
            self.selected_plate_travel = plate_choice.plate_travel
        self._write_plate_calibration_csv(plate_candidates)

    def _load_existing_calibration(self) -> bool:
        if not self.impact_calibration_output.exists() or not self.plate_calibration_output.exists():
            return False
        try:
            with self.impact_calibration_output.open(newline="", encoding="utf-8") as handle:
                impact_rows = list(csv.DictReader(handle))
            with self.plate_calibration_output.open(newline="", encoding="utf-8") as handle:
                plate_rows = list(csv.DictReader(handle))
            passing_impacts = [row for row in impact_rows if row.get("passed", "").lower() == "true"]
            passing_plates = [row for row in plate_rows if row.get("passed", "").lower() == "true"]
            if not passing_impacts or not passing_plates:
                return False
            self.selected_impact_speed = float(passing_impacts[-1]["impact_speed"])
            self.selected_plate_travel = float(passing_plates[-1]["plate_travel"])
            return True
        except Exception:
            return False

    def _write_plate_calibration_csv(self, rows: list[PlateCalibrationResult]) -> None:
        self.plate_calibration_output.parent.mkdir(parents=True, exist_ok=True)
        with self.plate_calibration_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dict__.keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)

    def _write_impact_calibration_csv(self, rows: list[ImpactCalibrationResult]) -> None:
        self.impact_calibration_output.parent.mkdir(parents=True, exist_ok=True)
        with self.impact_calibration_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dict__.keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)

    def _spawn_live_projectile(self) -> None:
        if self.impact_spawned:
            return
        self.projectile_np, self.projectile_body, self.projectile_visual = self._spawn_projectile(self.world, self.root, self.selected_impact_speed)
        self.impact_spawned = True
        self.impact_contact_applied = False

    def _reset_support_stats(self) -> None:
        self.current_support_stats = SupportStats()

    def _record_support_forces(self, indices: np.ndarray, forces: np.ndarray, source: str) -> None:
        if indices.size == 0:
            return
        magnitudes = np.linalg.norm(forces, axis=1)
        if magnitudes.size == 0:
            return
        total_force = float(magnitudes.sum())
        previous_total = self.current_support_stats.total_support_force
        previous_count = self.current_support_stats.supported_node_count
        new_count = previous_count + int(indices.size)
        new_total = previous_total + total_force
        self.current_support_stats.total_support_force = new_total
        self.current_support_stats.supported_node_count = new_count
        self.current_support_stats.max_support_force = max(self.current_support_stats.max_support_force, float(magnitudes.max()))
        self.current_support_stats.mean_support_force = new_total / max(new_count, 1)
        if source == "floor":
            self.current_support_stats.floor_supported_node_count += int(indices.size)
        elif source == "plate":
            self.current_support_stats.plate_supported_node_count += int(indices.size)

    def _record_velocity_corrections(
        self,
        indices: np.ndarray,
        delta_velocities: np.ndarray,
        before_vertical: np.ndarray,
        after_vertical: np.ndarray,
        *,
        damped_count: int,
        correction_magnitude: np.ndarray,
    ) -> None:
        if indices.size == 0:
            return
        magnitudes = np.linalg.norm(delta_velocities, axis=1)
        self.current_support_stats.corrected_node_count += int(indices.size)
        self.current_support_stats.damped_node_count += int(damped_count)
        self.current_support_stats.total_correction_magnitude += float(correction_magnitude.sum())
        self.current_support_stats.mean_vertical_velocity_before = float(before_vertical.mean())
        self.current_support_stats.mean_vertical_velocity_after = float(after_vertical.mean())
        if magnitudes.size:
            self.current_support_stats.max_support_force = max(self.current_support_stats.max_support_force, float(magnitudes.max()))

    def _apply_floor_penalty(self, soft_body: TrueSoftBodyObject) -> None:
        if not self.support_mode.custom_floor_enabled or self.support_mode.floor_contact_mode != "penalty_old":
            return
        positions = soft_body.get_node_positions()
        velocities = soft_body.get_node_velocities()
        indices = np.where(positions[:, 2] < self.floor_contact_clearance)[0].astype(np.int32)
        if indices.size == 0:
            return
        penetration = self.floor_contact_clearance - positions[indices, 2]
        downward_velocity = np.maximum(-velocities[indices, 2], 0.0)
        upward = penetration * self.floor_contact_stiffness + downward_velocity * self.floor_contact_damping
        forces = np.zeros((indices.size, 3), dtype=np.float32)
        forces[:, 2] = upward.astype(np.float32)
        soft_body.apply_forces(indices, forces)
        self._record_support_forces(indices, forces, "floor")

    def _apply_floor_projection(self, soft_body: TrueSoftBodyObject) -> None:
        if not self.support_mode.custom_floor_enabled or self.support_mode.floor_contact_mode not in {"projection", "velocity_projection"}:
            return
        positions = soft_body.get_node_positions()
        velocities = soft_body.get_node_velocities()
        if self.support_mode.floor_contact_mode == "projection":
            indices, _, corrections = project_penetrating_nodes(
                positions,
                floor_height=0.0,
                epsilon=0.035,
            )
            if indices.size == 0:
                return
            before_vertical = velocities[indices, 2].copy()
            target_velocity = np.maximum((corrections / max(self.config.physics_substep_dt, 1e-6)) * 1.25, 0.0)
            delta = np.zeros((indices.size, 3), dtype=np.float32)
            delta[:, 2] = np.maximum(target_velocity - velocities[indices, 2], 0.0).astype(np.float32)
            soft_body.apply_velocities(indices, delta)
            after_vertical = velocities[indices, 2] + delta[:, 2]
            self._record_velocity_corrections(
                indices,
                delta,
                before_vertical,
                after_vertical,
                damped_count=0,
                correction_magnitude=corrections,
            )
            return
        indices, corrected_velocities, corrections, damped = velocity_project_penetrating_nodes(
            positions,
            velocities,
            floor_height=0.0,
            epsilon=0.035,
            restitution=0.03,
            tangential_friction=0.12,
        )
        if indices.size == 0:
            return
        before = velocities[indices]
        after = corrected_velocities[indices]
        delta = after - before
        delta[:, 2] += np.maximum((corrections / max(self.config.physics_substep_dt, 1e-6)) * 1.15, 0.0).astype(np.float32)
        soft_body.apply_velocities(indices, delta)
        self._record_velocity_corrections(
            indices,
            delta,
            before[:, 2],
            before[:, 2] + delta[:, 2],
            damped_count=int(np.count_nonzero(damped[indices])),
            correction_magnitude=corrections,
        )

    def _apply_plate_penalty(self, soft_body: TrueSoftBodyObject, plate_height: float) -> None:
        if not self.support_mode.custom_plate_enabled:
            return
        plate_bottom = plate_height - 0.20 - self.plate_contact_clearance
        positions = soft_body.get_node_positions()
        velocities = soft_body.get_node_velocities()
        indices = np.where(positions[:, 2] > plate_bottom)[0].astype(np.int32)
        if indices.size == 0:
            return
        penetration = positions[indices, 2] - plate_bottom
        upward_velocity = np.maximum(velocities[indices, 2], 0.0)
        downward = penetration * self.plate_contact_stiffness + upward_velocity * self.plate_contact_damping
        forces = np.zeros((indices.size, 3), dtype=np.float32)
        forces[:, 2] = -downward.astype(np.float32)
        soft_body.apply_forces(indices, forces)
        self._record_support_forces(indices, np.abs(forces), "plate")

    def _apply_wall_penalties(self, soft_body: TrueSoftBodyObject) -> None:
        positions = soft_body.get_node_positions()
        velocities = soft_body.get_node_velocities()
        forces = np.zeros_like(positions, dtype=np.float32)
        active = np.zeros((positions.shape[0],), dtype=bool)
        for axis, limit in ((0, 4.5), (1, 4.8)):
            positive = positions[:, axis] > limit
            if np.any(positive):
                penetration = positions[positive, axis] - limit
                response = penetration * self.wall_contact_stiffness + np.maximum(velocities[positive, axis], 0.0) * self.wall_contact_damping
                forces[positive, axis] -= response.astype(np.float32)
                active |= positive
            negative = positions[:, axis] < -limit
            if np.any(negative):
                penetration = -limit - positions[negative, axis]
                response = penetration * self.wall_contact_stiffness - np.minimum(velocities[negative, axis], 0.0) * self.wall_contact_damping
                forces[negative, axis] += response.astype(np.float32)
                active |= negative
        indices = np.where(active)[0].astype(np.int32)
        if indices.size:
            soft_body.apply_forces(indices, forces[indices])

    def _apply_projectile_overlap(self, soft_body: TrueSoftBodyObject, projectile_np: NodePath | None, projectile_body: BulletRigidBodyNode | None) -> None:
        if projectile_np is None or projectile_body is None or self.impact_contact_applied:
            return
        center = np.asarray([projectile_np.getX(), projectile_np.getY(), projectile_np.getZ()], dtype=np.float32)
        positions = soft_body.get_node_positions()
        distances = np.linalg.norm(positions - center[None, :], axis=1)
        indices = np.where(distances <= 0.30)[0].astype(np.int32)
        if indices.size == 0:
            return
        velocity = projectile_body.getLinearVelocity()
        impulse = np.asarray([velocity.x, velocity.y, velocity.z], dtype=np.float32) * 0.28
        soft_body.apply_velocity_to_indices(indices, impulse)
        projectile_body.setLinearVelocity(projectile_body.getLinearVelocity() * 0.15)
        self.impact_contact_applied = True

    def _apply_custom_contacts(self, soft_body: TrueSoftBodyObject, plate_height: float, phase_name: str) -> None:
        self._apply_floor_penalty(soft_body)
        self._apply_wall_penalties(soft_body)
        if phase_name in {"plate_descent", "plate_hold", "plate_release"}:
            self._apply_plate_penalty(soft_body, plate_height)

    def _apply_post_step_contacts(self, soft_body: TrueSoftBodyObject, plate_height: float, phase_name: str) -> None:
        del plate_height, phase_name
        self._apply_floor_projection(soft_body)

    def _sync_projectile(self) -> None:
        if self.projectile_np is None or self.projectile_visual is None:
            return
        self.projectile_visual.setH(self.projectile_visual.getH() + 4.0)
        if self.projectile_np.getZ() < -3.0 or abs(self.projectile_np.getX()) > 18.0 or abs(self.projectile_np.getY()) > 18.0:
            self.projectile_np.hide()

    def _update_plate_live(self, metrics: SoftBodyMetrics) -> None:
        if not self.support_mode.plate_enabled or self.plate_np is None:
            if self.is_floating_rest_debug:
                self.phase_name = "floating_rest"
            else:
                self.phase_name = "rest_settle" if self.is_rest_debug else "impact_recovery"
            self.last_plate_height = self._plate_start_height()
            return
        phase_name, plate_height = self._current_plate_height(self.sim_time)
        self.phase_name = phase_name
        self.last_plate_height = plate_height
        self.plate_np.setZ(plate_height)
        if phase_name == "plate_descent" and not self.plate_phase_started:
            self.plate_phase_started = True
            self.plate_baseline_min_height = metrics.minimum_node_height
            self.plate_baseline_triangle = metrics.min_triangle_area_proxy
        dynamic_min_height = metrics.minimum_node_height
        dynamic_triangle = metrics.min_triangle_area_proxy
        if self.plate_phase_started:
            dynamic_min_height = max(
                metrics.minimum_node_height,
                min(self.thresholds.floor_tolerance, self.plate_baseline_min_height - 0.035),
            )
            dynamic_triangle = max(
                metrics.min_triangle_area_proxy,
                min(self.thresholds.min_triangle_area_proxy, self.plate_baseline_triangle - 0.04),
            )
        trigger, reasons = should_trigger_safe_release(
            metrics.volume_ratio,
            dynamic_min_height,
            dynamic_triangle,
            metrics.max_edge_stretch_ratio,
            metrics.min_edge_compression_ratio,
            metrics.has_non_finite,
            self.thresholds,
        )
        if trigger and not self.safety_release_triggered and phase_name in {"plate_descent", "plate_hold"}:
            self.safety_release_triggered = True
            self.release_started_at = self.sim_time
            self.release_origin_height = plate_height
            reason = "|".join(reasons)
            self.safety_events.append(f"{self.sim_time:0.3f}:{reason}")
            print(f"[surface-safety] triggered at t={self.sim_time:0.3f}s reasons={reason}")

    def _log_inversion_events(self, metrics: SoftBodyMetrics) -> None:
        positions = self.soft_body.get_node_positions()
        proxies = compute_surface_triangle_area_proxies(
            positions,
            self.soft_body.sim_mesh.surface_triangles,
            self.soft_body.rest_triangle_normals,
        )
        negative = np.where(proxies < -1e-4)[0].astype(np.int32)
        if negative.size == 0:
            return
        worst_index = int(negative[np.argmin(proxies[negative])])
        self.inversion_events.append(
            {
                "time": round(self.sim_time, 6),
                "phase": self.phase_name,
                "triangle_index": worst_index,
                "negative_triangle_count": int(negative.size),
                "min_triangle_area_proxy": float(proxies[worst_index]),
                "volume_ratio": metrics.volume_ratio,
                "minimum_node_height": metrics.minimum_node_height,
            }
        )

    def reset(self) -> None:
        self.root.removeNode()
        self.overlay.destroy()
        self.__init__(self.base, self.audio_features, self.config)

    def step(self, dt: float, sample: AudioSample) -> None:
        del sample
        self.sim_time += dt

        if not self.is_rest_debug and not self.is_floating_rest_debug and not self.impact_spawned and self.sim_time >= self.motion_config.impact_time:
            self._spawn_live_projectile()

        preview_metrics = self.soft_body.get_metrics()
        self._update_plate_live(preview_metrics)
        self._reset_support_stats()
        self._apply_projectile_overlap(self.soft_body, self.projectile_np, self.projectile_body)
        self._apply_custom_contacts(self.soft_body, self.last_plate_height, self.phase_name)

        start = time.perf_counter()
        self.world.doPhysics(dt, self.config.physics_substeps, self.config.physics_substep_dt)
        physics_seconds = time.perf_counter() - start
        self._apply_post_step_contacts(self.soft_body, self.last_plate_height, self.phase_name)

        self.soft_body.sync_render_mesh()
        self._sync_projectile()
        metrics = self.soft_body.get_metrics()
        self.latest_raw_solver_volume = metrics.current_volume
        self.latest_post_support_volume = metrics.current_volume
        self._log_inversion_events(metrics)
        self._update_overlay(metrics)
        self.metrics_rows.append(self._build_metrics_row(dt, physics_seconds, metrics))

    def get_metrics(self) -> SoftBodyMetrics:
        return self.soft_body.get_metrics()

    def finalize(self) -> None:
        if self.metrics_rows:
            self.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=surface_calibration_metric_fields())
                writer.writeheader()
                writer.writerows(self.metrics_rows)
            print(f"[metrics] wrote {self.metrics_output}")
        self.inversion_events_output.parent.mkdir(parents=True, exist_ok=True)
        with self.inversion_events_output.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "time",
                "phase",
                "triangle_index",
                "negative_triangle_count",
                "min_triangle_area_proxy",
                "volume_ratio",
                "minimum_node_height",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.inversion_events)
        print(f"[metrics] wrote {self.inversion_events_output}")
