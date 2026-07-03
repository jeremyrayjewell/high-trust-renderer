from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletBoxShape, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape, BulletWorld
from panda3d.core import AmbientLight, CardMaker, DirectionalLight, Filename, Material, NodePath, Point2, PointLight, Shader, TextNode, Vec3, Vec4

from .audio_analysis import AudioFeatures, AudioSample
from .gummy_object import generate_shape_mesh_data
from .procedural_course import CourseLayout, CourseObstacle, CourseSection, build_procedural_course_layout, sample_route_position
from .soft_body_object import SoftBodyMetrics, SoftBodyPreset, TrueSoftBodyObject, create_debug_plate, create_render_geom_node
from .thickness_renderer import ThicknessCompositePipeline


@dataclass
class ExperimentalConfig:
    width: int
    height: int
    duration: float
    seed: int
    physics_mode: str = "soft"
    true_softbody_debug: bool = False
    true_softbody_stress_debug: bool = False
    tetra_softbody_debug: bool = False
    floating_softbody_scene: bool = False
    softbody_obstacle_course: bool = False
    course_seed: int = 42
    floating_debug_overlay: bool = False
    framing_debug_overlay: bool = False
    floating_performance_intensity: str = "medium"
    translucency_debug: bool = False
    true_softbody_translucent: bool = False
    true_softbody_translucent_stress: bool = False
    softbody_preset: str = "soft"
    softbody_visualization: str = "shaded"
    softbody_profiles: tuple[str, ...] = ()
    translucency_view: str = "composite"
    translucency_preset: str = "balanced"
    absorption_color: tuple[float, float, float] = (0.06, 0.42, 0.52)
    absorption_density: float = 0.9
    transmission_gain: float = 1.15
    scattering_strength: float = 0.56
    cloudiness: float = 0.30
    refraction_strength: float = 0.08
    ior: float = 1.15
    surface_opacity: float = 0.92
    specular_strength: float = 0.85
    fresnel_strength: float = 0.12
    surface_reflection_strength: float = 0.20
    thickness_scale: float = 1.0
    metrics_output: str = "output/softbody_stress_metrics.csv"
    course_layout_output: str = ""
    body_summary_output: str = ""


@dataclass
class FloatingSoftActor:
    name: str
    soft_body: TrueSoftBodyObject
    base_color: tuple[float, float, float, float]
    role: str
    shape_kind: str
    delay_seconds: float
    center_target: np.ndarray
    bass_state: float = 0.0
    next_beat_time: float = 0.0
    next_onset_time: float = 0.0
    next_twist_time: float = 0.0
    next_high_time: float = 0.0
    beat_counter: int = 0
    active_event: str = "idle"
    active_impulses: int = 0
    selectors: dict[str, np.ndarray] = field(default_factory=dict)
    base_center_target: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    orbit_phase: float = 0.0
    orbit_radius: float = 0.0
    orbit_rate: float = 0.0
    response_scale: float = 1.0
    motion_scale: float = 1.0
    envelopes: list["FloatingEnvelope"] = field(default_factory=list)
    profile_name: str = "medium"
    release_time: float = 0.0
    release_trigger: str = "start"
    released: bool = True
    contact_count: int = 0
    next_contact_time: float = 0.0
    event_log: str = "idle"
    lane_bias: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    unstuck_events: int = 0
    last_progress_time: float = 0.0
    last_progress_height: float = 0.0
    meaningful_impacts: int = 0
    visible_frames: int = 0
    peak_deformation: float = 0.0
    peak_velocity: float = 0.0


@dataclass
class FloatingEnvelope:
    name: str
    action: str
    start_time: float
    attack: float
    hold: float
    decay: float
    magnitude: float
    selector_key: str | None = None
    vector: np.ndarray | None = None
    axis: np.ndarray | None = None


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    kind: str
    position: Vec3
    size: Vec3
    hpr: tuple[float, float, float]
    color: tuple[float, float, float, float]
    influence_radius: float


FLOATING_INTENSITY_PRESETS: dict[str, dict[str, float]] = {
    "low": {
        "gravity": -0.18,
        "centering": 0.65,
        "beat_force": 4.6,
        "beat_counter_force": 0.40,
        "onset_force": 1.7,
        "twist_force": 0.72,
        "high_force": 0.24,
        "breath_force": 2.2,
        "initial_velocity": 0.28,
        "separation_force": 0.65,
        "orbital_force": 0.32,
        "camera_orbit": 0.28,
        "event_scale": 0.82,
    },
    "medium": {
        "gravity": -0.25,
        "centering": 0.90,
        "beat_force": 8.6,
        "beat_counter_force": 0.78,
        "onset_force": 3.6,
        "twist_force": 1.72,
        "high_force": 0.44,
        "breath_force": 4.4,
        "initial_velocity": 0.52,
        "separation_force": 1.05,
        "orbital_force": 0.52,
        "camera_orbit": 0.35,
        "event_scale": 1.00,
    },
    "high": {
        "gravity": -0.28,
        "centering": 1.18,
        "beat_force": 12.2,
        "beat_counter_force": 1.05,
        "onset_force": 5.2,
        "twist_force": 2.45,
        "high_force": 0.72,
        "breath_force": 6.4,
        "initial_velocity": 0.72,
        "separation_force": 1.62,
        "orbital_force": 0.94,
        "camera_orbit": 0.42,
        "event_scale": 1.62,
    },
}


COURSE_BODY_PROFILES: dict[str, dict[str, object]] = {
    "very_soft": {
        "preset": SoftBodyPreset(
            name="profile_very_soft",
            linear_stiffness=0.26,
            angular_stiffness=0.10,
            volume_preservation=0.46,
            pose_matching=0.0,
            pressure=3.6,
            volume_conservation=1.1,
            damping=0.06,
            position_iterations=15,
            velocity_iterations=7,
            drift_iterations=4,
            cluster_iterations=4,
            cluster_count=8,
        ),
        "response_scale": 2.35,
        "motion_scale": 0.98,
        "breath_scale": 1.44,
        "recovery_scale": 0.80,
        "contact_dent_scale": 1.95,
        "rebound_scale": 1.10,
        "twist_scale": 1.18,
    },
    "soft": {
        "preset": SoftBodyPreset(
            name="profile_soft",
            linear_stiffness=0.34,
            angular_stiffness=0.15,
            volume_preservation=0.52,
            pose_matching=0.01,
            pressure=4.0,
            volume_conservation=1.3,
            damping=0.07,
            position_iterations=16,
            velocity_iterations=7,
            drift_iterations=4,
            cluster_iterations=4,
            cluster_count=8,
        ),
        "response_scale": 2.05,
        "motion_scale": 0.96,
        "breath_scale": 1.18,
        "recovery_scale": 0.92,
        "contact_dent_scale": 1.68,
        "rebound_scale": 1.06,
        "twist_scale": 1.06,
    },
    "medium": {
        "preset": SoftBodyPreset(
            name="profile_medium",
            linear_stiffness=0.42,
            angular_stiffness=0.22,
            volume_preservation=0.60,
            pose_matching=0.03,
            pressure=4.6,
            volume_conservation=1.8,
            damping=0.08,
            position_iterations=18,
            velocity_iterations=8,
            drift_iterations=4,
            cluster_iterations=5,
            cluster_count=9,
        ),
        "response_scale": 1.58,
        "motion_scale": 1.16,
        "breath_scale": 0.88,
        "recovery_scale": 1.02,
        "contact_dent_scale": 1.36,
        "rebound_scale": 1.14,
        "twist_scale": 1.20,
    },
    "springy": {
        "preset": SoftBodyPreset(
            name="profile_springy",
            linear_stiffness=0.50,
            angular_stiffness=0.24,
            volume_preservation=0.64,
            pose_matching=0.04,
            pressure=5.1,
            volume_conservation=2.0,
            damping=0.06,
            position_iterations=18,
            velocity_iterations=9,
            drift_iterations=5,
            cluster_iterations=5,
            cluster_count=10,
        ),
        "response_scale": 1.26,
        "motion_scale": 1.58,
        "breath_scale": 0.52,
        "recovery_scale": 1.28,
        "contact_dent_scale": 1.28,
        "rebound_scale": 1.42,
        "twist_scale": 1.38,
    },
    "firm_bouncy": {
        "preset": SoftBodyPreset(
            name="profile_firm_bouncy",
            linear_stiffness=0.56,
            angular_stiffness=0.30,
            volume_preservation=0.70,
            pose_matching=0.05,
            pressure=5.4,
            volume_conservation=2.2,
            damping=0.06,
            position_iterations=20,
            velocity_iterations=9,
            drift_iterations=5,
            cluster_iterations=5,
            cluster_count=10,
        ),
        "response_scale": 1.06,
        "motion_scale": 1.74,
        "breath_scale": 0.34,
        "recovery_scale": 1.42,
        "contact_dent_scale": 1.08,
        "rebound_scale": 1.58,
        "twist_scale": 1.48,
    },
}

OBSTACLE_BODY_PROFILES = COURSE_BODY_PROFILES


TRANSLUCENCY_PRESETS: dict[str, dict[str, object]] = {
    "subtle": {
        "absorption_color": (0.05, 0.28, 0.40),
        "absorption_density": 0.55,
        "transmission_gain": 1.05,
        "scattering_strength": 0.36,
        "cloudiness": 0.22,
        "refraction_strength": 0.045,
        "ior": 1.08,
        "surface_opacity": 0.86,
        "specular_strength": 0.72,
        "fresnel_strength": 0.08,
        "surface_reflection_strength": 0.16,
        "thickness_scale": 0.82,
    },
    "balanced": {
        "absorption_color": (0.24, 0.92, 1.00),
        "absorption_density": 0.66,
        "transmission_gain": 1.34,
        "scattering_strength": 0.62,
        "cloudiness": 0.34,
        "refraction_strength": 0.095,
        "ior": 1.14,
        "surface_opacity": 0.94,
        "specular_strength": 0.90,
        "fresnel_strength": 0.11,
        "surface_reflection_strength": 0.24,
        "thickness_scale": 1.08,
    },
    "exaggerated": {
        "absorption_color": (0.18, 0.96, 1.00),
        "absorption_density": 0.54,
        "transmission_gain": 1.56,
        "scattering_strength": 0.74,
        "cloudiness": 0.42,
        "refraction_strength": 0.170,
        "ior": 1.24,
        "surface_opacity": 0.93,
        "specular_strength": 1.02,
        "fresnel_strength": 0.16,
        "surface_reflection_strength": 0.30,
        "thickness_scale": 1.28,
    },
}


def safe_frame_bounds(left_pct: float = 0.08, right_pct: float = 0.92, bottom_pct: float = 0.10, top_pct: float = 0.90) -> tuple[float, float, float, float]:
    return (
        -1.0 + left_pct * 2.0,
        -1.0 + right_pct * 2.0,
        -1.0 + bottom_pct * 2.0,
        -1.0 + top_pct * 2.0,
    )


def edge_margin_from_bounds(bounds: tuple[float, float, float, float], safe_bounds: tuple[float, float, float, float]) -> float:
    min_x, max_x, min_y, max_y = bounds
    safe_min_x, safe_max_x, safe_min_y, safe_max_y = safe_bounds
    return min(
        min_x - safe_min_x,
        safe_max_x - max_x,
        min_y - safe_min_y,
        safe_max_y - max_y,
    )


class ExperimentalScene:
    def __init__(self, base: ShowBase, audio_features: AudioFeatures, config: ExperimentalConfig) -> None:
        self.base = base
        self.audio_features = audio_features
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.sim_time = 0.0
        self.phase_name = "rest"
        self.aspect_ratio = config.width / max(config.height, 1)
        self.safe_bounds = safe_frame_bounds()
        self.safe_frame_violation_total = 0
        self.current_safe_frame_violation_count = 0
        self.current_edge_margin_min = 1.0
        self.current_screen_stats: dict[str, dict[str, float]] = {}
        self.current_camera_distance = 0.0
        self.floating_stage_half_width = max(8.8, 5.6 * self.aspect_ratio + 1.4)
        self.floating_stage_height = 5.8
        self.floating_intensity = FLOATING_INTENSITY_PRESETS.get(config.floating_performance_intensity, FLOATING_INTENSITY_PRESETS["medium"])
        self.root = self.base.render.attachNewNode("experimental-scene")
        self.base.setBackgroundColor(0.02, 0.02, 0.024, 1.0)
        self.env_nodes: list[NodePath] = []
        self.gel_nodes: list[NodePath] = []
        self.floating_panels: list[NodePath] = []
        self.floating_grid_lines: list[NodePath] = []
        self.projectiles: list[tuple[NodePath, BulletRigidBodyNode, NodePath]] = []
        self.metrics_rows: list[dict[str, object]] = []
        self.metrics_output = Path(config.metrics_output)
        self.course_layout_output = Path(config.course_layout_output) if config.course_layout_output else None
        self.body_summary_output = Path(config.body_summary_output) if config.body_summary_output else None
        self.pipeline: ThicknessCompositePipeline | None = None
        self.overlay: OnscreenText | None = None
        self.floating_actors: list[FloatingSoftActor] = []
        self.floating_bumpers: list[tuple[NodePath, BulletRigidBodyNode]] = []
        self.course_obstacles: list[tuple[CourseObstacle, NodePath, BulletRigidBodyNode]] = []
        self.course_layout: CourseLayout | None = None
        self.course_sections: tuple[CourseSection, ...] = ()
        self.course_metrics_cache: dict[str, SoftBodyMetrics] = {}
        self.containment_projection_events = 0
        self.last_active_event = "idle"
        self.active_impulse_count = 0
        self.floating_cues_fired: set[str] = set()
        self.course_release_events: set[str] = set()
        self.course_obstacle_count = 0
        self.course_camera_route_time = 0.0

        self.is_surface_stress = config.true_softbody_debug or config.true_softbody_stress_debug
        self.is_tetra_stress = config.tetra_softbody_debug
        self.is_floating_scene = config.floating_softbody_scene
        self.is_obstacle_course = config.softbody_obstacle_course
        self.is_soft_stress = self.is_surface_stress or self.is_tetra_stress or config.true_softbody_translucent_stress
        self.is_translucent = (
            config.translucency_debug
            or config.true_softbody_translucent
            or config.true_softbody_translucent_stress
            or self.is_floating_scene
            or self.is_obstacle_course
        )

        self.base.camLens.setNearFar(0.1, 200.0)
        self.base.camLens.setAspectRatio(self.aspect_ratio)
        if self.is_obstacle_course:
            self.base.camLens.setFov(33.0)
            self.base.camera.setPos(0.2, -12.8, 5.8)
            self.base.camera.lookAt(0.1, 0.4, 3.2)
        elif self.is_floating_scene:
            self.base.camLens.setFov(34.0)
            self.base.camera.setPos(0.0, -11.2, 3.35)
            self.base.camera.lookAt(0.0, 0.0, 1.55)
        elif self.is_soft_stress:
            self.base.camLens.setFov(36.0)
            self.base.camera.setPos(3.8, -13.4, 4.0)
            self.base.camera.lookAt(0.0, 0.0, 1.25)
        elif self.is_translucent:
            self.base.camLens.setFov(34.0)
            self.base.camera.setPos(0.0, -10.4, 3.9)
            self.base.camera.lookAt(0.0, 0.0, 1.85)
        else:
            self.base.camLens.setFov(34.0)
            self.base.camera.setPos(0.0, -13.5, 4.9)
            self.base.camera.lookAt(0.0, 0.0, 1.7)

        self.world = BulletWorld()
        gravity = Vec3(0.0, 0.0, -9.81 if (self.is_soft_stress or config.true_softbody_translucent or config.true_softbody_translucent_stress or self.is_surface_stress or self.is_tetra_stress) else 0.0)
        if self.is_obstacle_course:
            gravity = Vec3(0.0, 0.0, -2.65 - 0.45 * float(self.floating_intensity["event_scale"]))
        elif self.is_floating_scene:
            gravity = Vec3(0.0, 0.0, float(self.floating_intensity["gravity"]))
        if config.translucency_debug and not config.true_softbody_translucent_stress:
            gravity = Vec3(0.0, 0.0, 0.0)
        self.world.setGravity(gravity)
        self.world_info = self.world.getWorldInfo()
        self.world_info.setGravity(gravity)
        self.gravity = gravity

        self.soft_body: TrueSoftBodyObject | None = None
        self.thin_cube_np: NodePath | None = None
        self.thick_cube_np: NodePath | None = None
        self.torus_np: NodePath | None = None
        self.floor_np: NodePath | None = None
        self.plate_np: NodePath | None = None
        self.plate_body: BulletRigidBodyNode | None = None
        self.plate_visual: NodePath | None = None
        self.bottom_anchor_indices = np.asarray([], dtype=np.int32)
        self.bottom_hold_indices = np.asarray([], dtype=np.int32)
        self.top_indices = np.asarray([], dtype=np.int32)
        self.left_indices = np.asarray([], dtype=np.int32)
        self.right_indices = np.asarray([], dtype=np.int32)
        self.struck_indices = np.asarray([], dtype=np.int32)
        self.opposite_indices = np.asarray([], dtype=np.int32)
        self.center_indices = np.asarray([], dtype=np.int32)
        self.anchor_targets = np.zeros((0, 3), dtype=np.float32)
        self.bottom_hold_targets = np.zeros((0, 3), dtype=np.float32)
        self.center_planar_targets = np.zeros((0, 2), dtype=np.float32)
        self.projectile_spawned = False
        self.second_projectile_spawned = False
        self.camera_focus = np.asarray([0.0, 0.0, 1.55], dtype=np.float32)

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.surface_shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(str(shader_dir / "soft_debug.vert")), Filename.fromOsSpecific(str(shader_dir / "soft_debug.frag")))

        self._build_environment()
        if not self.is_floating_scene and not self.is_obstacle_course:
            self._build_floor()
        self._build_lights()
        self._spawn_mode()
        if self.config.floating_debug_overlay or self.config.framing_debug_overlay:
            self.overlay = self._create_overlay()

        if self.is_translucent:
            self.pipeline = ThicknessCompositePipeline(self.base, config.width, config.height)
            for node in self.env_nodes:
                self.pipeline.attach_environment(node)
            for node in self.gel_nodes:
                self.pipeline.attach_gelatin(node)
            self.pipeline.set_material_inputs(**self._translucency_inputs())
            self.pipeline.set_view_mode(config.translucency_view)
            self.pipeline.update_camera(self.base.cam)

    def _translucency_inputs(self) -> dict[str, object]:
        preset = TRANSLUCENCY_PRESETS[self.config.translucency_preset].copy()
        if tuple(self.config.absorption_color) != (0.22, 0.08, 0.04):
            preset["absorption_color"] = self.config.absorption_color
        preset["absorption_density"] = self.config.absorption_density if self.config.absorption_density != 1.75 else preset["absorption_density"]
        preset["scattering_strength"] = self.config.scattering_strength if self.config.scattering_strength != 0.46 else preset["scattering_strength"]
        preset["cloudiness"] = self.config.cloudiness if self.config.cloudiness != 0.40 else preset["cloudiness"]
        preset["refraction_strength"] = self.config.refraction_strength if self.config.refraction_strength != 0.035 else preset["refraction_strength"]
        preset["ior"] = self.config.ior if self.config.ior != 1.10 else preset["ior"]
        preset["surface_opacity"] = self.config.surface_opacity if self.config.surface_opacity != 0.98 else preset["surface_opacity"]
        preset["specular_strength"] = self.config.specular_strength if self.config.specular_strength != 0.72 else preset["specular_strength"]
        preset["fresnel_strength"] = self.config.fresnel_strength if self.config.fresnel_strength != 0.08 else preset["fresnel_strength"]
        preset["thickness_scale"] = self.config.thickness_scale if self.config.thickness_scale != 0.80 else preset["thickness_scale"]
        preset["transmission_gain"] = self.config.transmission_gain
        preset["surface_reflection_strength"] = self.config.surface_reflection_strength
        return preset

    def _create_overlay(self) -> OnscreenText:
        return OnscreenText(
            text="floating softbody",
            parent=self.base.a2dTopLeft,
            pos=(0.04, -0.08),
            scale=0.042,
            fg=(0.94, 0.97, 1.0, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )

    def _build_environment(self) -> None:
        if self.is_obstacle_course:
            self._build_obstacle_course_background()
        elif self.is_floating_scene:
            self._build_floating_background()
        elif self.is_translucent:
            self._build_translucency_background()
        else:
            backdrop = CardMaker("stress-backdrop")
            backdrop.setFrame(-18, 18, -10, 14)
            backdrop_np = self.root.attachNewNode(backdrop.generate())
            backdrop_np.setP(90)
            backdrop_np.setPos(0.0, 18.0, 5.6)
            backdrop_np.setColor(Vec4(0.045, 0.048, 0.058, 1.0))
            self.env_nodes.append(backdrop_np)

    def _build_floating_background(self) -> None:
        wall_half_width = self.floating_stage_half_width + 1.8
        wall_height = self.floating_stage_height
        grid_half_width = self.floating_stage_half_width + 0.9
        wall = CardMaker("floating-back-wall")
        wall.setFrame(-wall_half_width, wall_half_width, -0.6, wall_height)
        wall_np = self.root.attachNewNode(wall.generate())
        wall_np.setP(90)
        wall_np.setPos(0.0, 8.8, 2.6)
        wall_np.setColor(Vec4(0.10, 0.14, 0.22, 1.0))
        self.env_nodes.append(wall_np)

        for x_pos in np.linspace(-grid_half_width, grid_half_width, 11):
            line = CardMaker(f"floating-vline-{x_pos:.2f}")
            line.setFrame(-0.03, 0.03, -0.55, wall_height)
            line_np = self.root.attachNewNode(line.generate())
            line_np.setPos(float(x_pos), 8.7, 0.0)
            line_np.setColor(Vec4(0.12, 0.24, 0.42, 1.0))
            self.env_nodes.append(line_np)
            self.floating_grid_lines.append(line_np)

        for z_pos in np.linspace(0.15, wall_height - 0.25, 7):
            line = CardMaker(f"floating-hline-{z_pos:.2f}")
            line.setFrame(-grid_half_width, grid_half_width, -0.03, 0.03)
            line_np = self.root.attachNewNode(line.generate())
            line_np.setPos(0.0, 8.7, float(z_pos))
            line_np.setColor(Vec4(0.10, 0.22, 0.36, 1.0))
            self.env_nodes.append(line_np)
            self.floating_grid_lines.append(line_np)

        panel_colors = (
            Vec4(0.18, 0.84, 1.0, 1.0),
            Vec4(1.0, 0.54, 0.28, 1.0),
            Vec4(0.92, 0.32, 0.68, 1.0),
        )
        panel_positions = np.linspace(-self.floating_stage_half_width * 0.74, self.floating_stage_half_width * 0.74, 5)
        for index, x_pos in enumerate(panel_positions):
            panel = CardMaker(f"floating-panel-{index}")
            panel.setFrame(-1.45, 1.45, -0.35, wall_height - 0.35)
            panel_np = self.root.attachNewNode(panel.generate())
            panel_np.setPos(float(x_pos), 8.55, 0.4)
            panel_np.setColor(panel_colors[index % len(panel_colors)])
            panel_np.setPythonTag("base_pos", (float(x_pos), 8.55, 0.4))
            self.env_nodes.append(panel_np)
            self.floating_panels.append(panel_np)

        for index, offset in enumerate((-self.floating_stage_half_width * 0.62, self.floating_stage_half_width * 0.52)):
            glow = CardMaker(f"floating-glow-{index}")
            glow.setFrame(-2.6, 2.6, -0.32, 0.68)
            glow_np = self.root.attachNewNode(glow.generate())
            glow_np.setH(-26.0 if index == 0 else 24.0)
            glow_np.setPos(float(offset), 7.9, 3.9 - 0.8 * index)
            glow_np.setColor(Vec4(0.18 + 0.20 * index, 0.22 + 0.14 * index, 0.42 + 0.12 * index, 1.0))
            glow_np.setPythonTag("base_pos", (float(offset), 7.9, 3.9 - 0.8 * index))
            self.env_nodes.append(glow_np)
            self.floating_panels.append(glow_np)

        halo = CardMaker("floating-halo")
        halo.setFrame(-(wall_half_width + 1.0), wall_half_width + 1.0, -1.0, wall_height + 0.8)
        halo_np = self.root.attachNewNode(halo.generate())
        halo_np.setP(90)
        halo_np.setPos(0.0, 9.2, 2.5)
        halo_np.setColor(Vec4(0.03, 0.06, 0.12, 1.0))
        self.env_nodes.append(halo_np)

    def _build_obstacle_course_background(self) -> None:
        wall_half_width = self.floating_stage_half_width + 2.1
        wall_height = 11.8
        wall = CardMaker("course-back-wall")
        wall.setFrame(-wall_half_width, wall_half_width, -0.5, wall_height)
        wall_np = self.root.attachNewNode(wall.generate())
        wall_np.setP(90)
        wall_np.setPos(0.0, 9.8, 4.7)
        wall_np.setColor(Vec4(0.05, 0.08, 0.14, 1.0))
        self.env_nodes.append(wall_np)

        for x_pos in np.linspace(-wall_half_width * 0.82, wall_half_width * 0.82, 4):
            panel = CardMaker(f"course-panel-{x_pos:.2f}")
            panel.setFrame(-1.2, 1.2, -0.2, wall_height - 0.9)
            panel_np = self.root.attachNewNode(panel.generate())
            panel_np.setPos(float(x_pos), 9.55, 0.25)
            tint = 0.10 + 0.06 * np.sin(x_pos * 0.55)
            panel_np.setColor(Vec4(0.05 + tint * 0.55, 0.10 + tint * 0.22, 0.20 + tint * 0.46, 1.0))
            panel_np.setPythonTag("base_pos", (float(x_pos), 9.55, 0.25))
            self.env_nodes.append(panel_np)
            self.floating_panels.append(panel_np)

        for z_pos in np.linspace(1.1, wall_height - 1.2, 5):
            line = CardMaker(f"course-hline-{z_pos:.2f}")
            line.setFrame(-wall_half_width * 0.76, wall_half_width * 0.76, -0.02, 0.02)
            line_np = self.root.attachNewNode(line.generate())
            line_np.setPos(0.0, 9.45, float(z_pos))
            line_np.setColor(Vec4(0.08, 0.16, 0.28, 1.0))
            self.env_nodes.append(line_np)
            self.floating_grid_lines.append(line_np)

        for index, x_pos in enumerate(np.linspace(-wall_half_width * 0.72, wall_half_width * 0.72, 5)):
            glow = CardMaker(f"course-diag-{index}")
            glow.setFrame(-2.5, 2.5, -0.08, 0.16)
            glow_np = self.root.attachNewNode(glow.generate())
            glow_np.setH(-28.0 if index % 2 == 0 else 24.0)
            glow_np.setPos(float(x_pos), 8.95, 9.6 - index * 1.85)
            glow_np.setColor(Vec4(0.08 + 0.04 * index, 0.10 + 0.03 * index, 0.24 + 0.06 * index, 1.0))
            glow_np.setPythonTag("base_pos", (float(x_pos), 8.95, 9.6 - index * 1.85))
            self.env_nodes.append(glow_np)
            self.floating_panels.append(glow_np)

    def _build_translucency_background(self) -> None:
        wall = CardMaker("trans-wall")
        wall.setFrame(-7.6, 7.6, 0.0, 5.2)
        wall_np = self.root.attachNewNode(wall.generate())
        wall_np.setP(90)
        wall_np.setPos(0.0, 7.5, 2.6)
        wall_np.setColor(Vec4(0.92, 0.92, 0.96, 1.0))
        self.env_nodes.append(wall_np)

        for x_pos in np.linspace(-6.8, 6.8, 15):
            line = CardMaker(f"vline-{x_pos:.2f}")
            line.setFrame(-0.02, 0.02, 0.0, 5.2)
            line_np = self.root.attachNewNode(line.generate())
            line_np.setPos(float(x_pos), 7.45, 0.0)
            line_np.setColor(Vec4(0.08, 0.12, 0.18, 1.0))
            self.env_nodes.append(line_np)

        for z_pos in np.linspace(0.3, 5.0, 10):
            line = CardMaker(f"hline-{z_pos:.2f}")
            line.setFrame(-7.6, 7.6, -0.02, 0.02)
            line_np = self.root.attachNewNode(line.generate())
            line_np.setPos(0.0, 7.45, float(z_pos))
            line_np.setColor(Vec4(0.10, 0.14, 0.20, 1.0))
            self.env_nodes.append(line_np)

        for index, x_pos in enumerate(np.linspace(-5.6, 5.6, 7)):
            bar = CardMaker(f"bar-{index}")
            bar.setFrame(-0.24, 0.24, 0.0, 5.2)
            bar_np = self.root.attachNewNode(bar.generate())
            bar_np.setPos(float(x_pos), 7.35, 0.0)
            bar_np.setColor(Vec4(0.94 - index * 0.08, 0.22 + index * 0.08, 0.28 + (index % 2) * 0.32, 1.0))
            self.env_nodes.append(bar_np)

        for x_pos in np.linspace(-5.5, 5.5, 6):
            marker = CardMaker(f"floor-marker-{x_pos:.2f}")
            marker.setFrame(-0.06, 0.06, -5.5, 5.5)
            marker_np = self.root.attachNewNode(marker.generate())
            marker_np.setP(-90)
            marker_np.setPos(float(x_pos), 0.0, 0.02)
            marker_np.setColor(Vec4(0.86, 0.88, 0.92, 1.0))
            self.env_nodes.append(marker_np)

    def _build_floor(self) -> None:
        floor_shape = BulletPlaneShape(Vec3(0, 0, 1), 0.0)
        floor_node = BulletRigidBodyNode("exp-floor")
        floor_node.addShape(floor_shape)
        floor_node.setFriction(0.96)
        floor_np = self.root.attachNewNode(floor_node)
        self.world.attachRigidBody(floor_node)

        card = CardMaker("exp-floor-card")
        card.setFrame(-18, 18, -18, 18)
        visual = floor_np.attachNewNode(card.generate())
        visual.setP(-90)
        visual.setColor(0.12, 0.12, 0.13, 1.0)
        material = Material()
        material.setAmbient(Vec4(0.08, 0.08, 0.09, 1.0))
        material.setDiffuse(Vec4(0.14, 0.14, 0.15, 1.0))
        material.setSpecular(Vec4(0.32, 0.32, 0.34, 1.0))
        material.setShininess(18.0)
        visual.setMaterial(material, 1)
        self.floor_np = floor_np
        self.env_nodes.append(visual)

    def _create_bumper(self, name: str, position: Vec3, radius: float, color: tuple[float, float, float, float]) -> tuple[NodePath, BulletRigidBodyNode]:
        body = BulletRigidBodyNode(name)
        body.addShape(BulletSphereShape(radius))
        body.setFriction(0.92)
        body.setRestitution(0.08)
        body_np = self.root.attachNewNode(body)
        body_np.setPos(position)
        self.world.attachRigidBody(body)
        visual = self._create_visual_geom(f"{name}-visual", "rounded_octahedron", radius=radius * 1.15, color=color, parent=body_np)
        visual.setScale(1.0, 1.0, 1.0)
        visual.setColorScale(Vec4(color[0], color[1], color[2], 0.18))
        self.env_nodes.append(visual)
        return body_np, body

    def _build_floating_containment(self) -> None:
        x_limit = 3.55 + 0.86 * self.aspect_ratio
        y_limit = 4.35
        z_center = 1.95
        z_half = 2.9
        wall_specs = [
            (Vec3(x_limit + 0.38, 0.0, z_center), Vec3(0.24, y_limit, z_half)),
            (Vec3(-(x_limit + 0.38), 0.0, z_center), Vec3(0.24, y_limit, z_half)),
            (Vec3(0.0, y_limit + 0.45, z_center), Vec3(x_limit + 0.25, 0.24, z_half)),
            (Vec3(0.0, -(y_limit + 0.92), z_center), Vec3(x_limit + 0.25, 0.24, z_half)),
            (Vec3(0.0, 0.0, z_center + z_half + 0.58), Vec3(x_limit + 0.25, y_limit + 0.12, 0.24)),
            (Vec3(0.0, 0.0, z_center - z_half - 0.62), Vec3(x_limit + 0.25, y_limit + 0.12, 0.24)),
        ]
        for index, (pos, half_extents) in enumerate(wall_specs):
            node = BulletRigidBodyNode(f"floating-wall-{index}")
            node.addShape(BulletBoxShape(half_extents))
            wall_np = self.root.attachNewNode(node)
            wall_np.setPos(pos)
            self.world.attachRigidBody(node)

        bumper_specs = [
            (Vec3(-1.9, -0.4, 1.35), 0.46, (0.26, 0.92, 1.0, 1.0)),
            (Vec3(1.8, 0.7, 2.1), 0.54, (1.0, 0.44, 0.62, 1.0)),
            (Vec3(0.1, 1.8, 1.55), 0.42, (0.98, 0.86, 0.24, 1.0)),
        ]
        for name_index, (position, radius, color) in enumerate(bumper_specs):
            self.floating_bumpers.append(self._create_bumper(f"floating-bumper-{name_index}", position, radius, color))

    def _build_lights(self) -> None:
        ambient = AmbientLight("exp-ambient")
        if self.is_floating_scene or self.is_obstacle_course:
            ambient.setColor(Vec4(0.48, 0.50, 0.54, 1.0))
        else:
            ambient.setColor(Vec4(0.42, 0.42, 0.45, 1.0) if self.is_translucent else Vec4(0.34, 0.34, 0.36, 1.0))
        ambient_np = self.root.attachNewNode(ambient)
        self.base.render.setLight(ambient_np)

        key = DirectionalLight("exp-key")
        key.setColor(Vec4(1.56, 1.48, 1.34, 1.0) if (self.is_floating_scene or self.is_obstacle_course) else Vec4(1.42, 1.38, 1.28, 1.0))
        key_np = self.root.attachNewNode(key)
        key_np.setPos(-6.0, -6.4, 10.5 if (self.is_floating_scene or self.is_obstacle_course) else 11.0)
        key_np.lookAt(0.0, 0.0, 3.6 if self.is_obstacle_course else (1.6 if self.is_floating_scene else 1.6))
        self.base.render.setLight(key_np)

        back = PointLight("exp-back")
        back.setColor(Vec4(1.92, 1.76, 1.30, 1.0) if (self.is_floating_scene or self.is_obstacle_course) else Vec4(1.86, 1.68, 1.22, 1.0))
        back_np = self.root.attachNewNode(back)
        back_np.setPos(0.0, 8.8 if self.is_obstacle_course else (7.8 if self.is_floating_scene else 8.2), 8.2 if self.is_obstacle_course else 6.0)
        self.base.render.setLight(back_np)

        fill = PointLight("exp-fill")
        fill.setColor(Vec4(0.82, 0.88, 0.96, 1.0) if (self.is_floating_scene or self.is_obstacle_course) else Vec4(0.76, 0.82, 0.90, 1.0))
        fill_np = self.root.attachNewNode(fill)
        fill_np.setPos(-5.8, -5.4, 5.4 if self.is_obstacle_course else (4.6 if self.is_floating_scene else 4.8))
        self.base.render.setLight(fill_np)

    def _spawn_mode(self) -> None:
        if self.is_obstacle_course:
            self._spawn_obstacle_course_scene()
        elif self.is_floating_scene:
            self._spawn_floating_scene()
        elif self.is_soft_stress:
            variant = "tetra" if self.is_tetra_stress else "surface"
            self.soft_body = self._create_soft_body(variant, Vec3(0.0, 0.0, 2.60), visualization=self.config.softbody_visualization)
            self.plate_np, self.plate_body = create_debug_plate(self.root, half_extents=Vec3(1.85, 1.85, 0.22), position=Vec3(0.0, 0.0, 5.40), kinematic=True)
            self.world.attachRigidBody(self.plate_body)
            self.plate_visual = self._create_visual_geom("plate-visual", "rectangular_cuboid", radius=1.0, color=(0.82, 0.84, 0.88, 1.0), parent=self.plate_np)
            self.plate_visual.setScale(1.85, 1.85, 0.16)
            self._prepare_softbody_regions()
        elif self.config.true_softbody_translucent:
            self.soft_body = self._create_soft_body("surface", Vec3(0.0, 0.0, 2.0), visualization="shaded")
        elif self.config.translucency_debug:
            self._create_translucency_objects()
        else:
            self.soft_body = self._create_soft_body("surface", Vec3(0.0, 0.0, 2.0), visualization="shaded")

    def _spawn_floating_scene(self) -> None:
        self._build_floating_containment()
        actors = [
            ("floating-hero", "rectangular_cuboid", "hero", Vec3(-2.05, -1.05, 2.00), 1.24, 4.1, (0.24, 0.92, 1.0, 1.0), np.asarray([-1.60, -0.78, 1.96], dtype=np.float32), 1.34, 0.90),
            ("floating-secondary", "rounded_cube", "secondary", Vec3(2.55, 1.45, 1.42), 0.70, 2.25, (1.0, 0.44, 0.60, 1.0), np.asarray([2.25, 1.28, 1.48], dtype=np.float32), 0.62, 1.58),
        ]
        for index, (name, shape_kind, role, position, radius, mass, color, target_center, response_scale, motion_scale) in enumerate(actors):
            soft_body = TrueSoftBodyObject(
                self.root,
                self.world,
                self.world_info,
                variant="surface",
                radius=radius,
                mass=mass,
                seed=self.config.seed + index * 17,
                name=name,
                initial_position=position,
                preset_name=self.config.softbody_preset,
                render_shape_kind=shape_kind,
                visualization=self.config.softbody_visualization,
                max_displacement_visual=0.11,
            )
            soft_body.set_phase_name("floating")
            soft_body.set_soft_soft_collision(True)
            soft_body.render_node.setShader(self.surface_shader)
            soft_body.render_node.setShaderInput("base_color", Vec4(*color))
            soft_body.wireframe_np.setShader(self.surface_shader)
            soft_body.wireframe_np.setShaderInput("base_color", Vec4(0.98, 0.98, 1.0, 1.0))
            self.gel_nodes.append(soft_body.render_node)
            actor = FloatingSoftActor(
                name=name,
                soft_body=soft_body,
                base_color=color,
                role=role,
                shape_kind=shape_kind,
                delay_seconds=0.04 * index,
                center_target=target_center,
                base_center_target=target_center.copy(),
                orbit_phase=float(index) * 1.7,
                orbit_radius=1.35 - 0.24 * index,
                orbit_rate=0.18 + 0.08 * index,
                response_scale=response_scale,
                motion_scale=motion_scale,
            )
            self._prepare_floating_regions(actor)
            self.floating_actors.append(actor)

        velocity_scale = float(self.floating_intensity["initial_velocity"])
        initial_velocities = (
            np.asarray([velocity_scale * 0.76, 0.34, 0.18], dtype=np.float32),
            np.asarray([-velocity_scale * 1.42, -0.48, 0.22], dtype=np.float32),
        )
        for actor, velocity in zip(self.floating_actors, initial_velocities, strict=True):
            all_indices = np.arange(actor.soft_body.get_node_positions().shape[0], dtype=np.int32)
            actor.soft_body.apply_velocity_to_indices(all_indices, velocity)
            axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32) if actor.role == "hero" else np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            actor.soft_body.apply_opposing_twist(axis, float(self.floating_intensity["twist_force"]) * 0.42 * actor.response_scale)

    def _selected_obstacle_profiles(self) -> tuple[str, str, str, str, str]:
        defaults = ("very_soft", "soft", "medium", "springy", "firm_bouncy")
        if not self.config.softbody_profiles:
            return defaults
        chosen = list(self.config.softbody_profiles[:5])
        while len(chosen) < 5:
            chosen.append(defaults[len(chosen)])
        return tuple(chosen[:5])  # type: ignore[return-value]

    def _course_actor_specs(self) -> list[dict[str, object]]:
        profile_names = self._selected_obstacle_profiles()
        return [
            {
                "name": "hero-soft",
                "shape_kind": "rectangular_cuboid",
                "role": "hero",
                "profile_name": profile_names[0],
                "radius": 1.62,
                "mass": 6.2,
                "color": (0.18, 0.94, 1.0, 1.0),
                "lane_bias": np.asarray([0.00, 0.00, 0.0], dtype=np.float32),
            },
            {
                "name": "coral-cube",
                "shape_kind": "triangular_prism",
                "role": "coral",
                "profile_name": profile_names[1],
                "radius": 1.22,
                "mass": 4.1,
                "color": (1.0, 0.42, 0.54, 1.0),
                "lane_bias": np.asarray([1.15, -0.62, 0.0], dtype=np.float32),
            },
            {
                "name": "violet-spring",
                "shape_kind": "rounded_cube",
                "role": "spring",
                "profile_name": profile_names[3],
                "radius": 0.96,
                "mass": 2.8,
                "color": (0.78, 0.56, 1.0, 1.0),
                "lane_bias": np.asarray([-1.35, 0.84, 0.0], dtype=np.float32),
            },
            {
                "name": "amber-prism",
                "shape_kind": "hexagonal_prism",
                "role": "prism",
                "profile_name": profile_names[2],
                "radius": 1.10,
                "mass": 3.5,
                "color": (1.0, 0.76, 0.18, 1.0),
                "lane_bias": np.asarray([1.72, 0.94, 0.0], dtype=np.float32),
            },
            {
                "name": "teal-octa",
                "shape_kind": "rounded_octahedron",
                "role": "octa",
                "profile_name": profile_names[4],
                "radius": 1.04,
                "mass": 3.0,
                "color": (0.36, 1.0, 0.76, 1.0),
                "lane_bias": np.asarray([-1.86, -0.96, 0.0], dtype=np.float32),
            },
        ]

    def _spawn_obstacle_course_scene(self) -> None:
        actor_specs = self._course_actor_specs()
        body_names = tuple(str(spec["name"]) for spec in actor_specs)
        body_metadata = {
            str(spec["name"]): {
                "shape_kind": str(spec["shape_kind"]),
                "profile_name": str(spec["profile_name"]),
                "color": tuple(spec["color"]),
            }
            for spec in actor_specs
        }
        self.course_layout = build_procedural_course_layout(
            self.audio_features,
            duration=min(self.audio_features.duration, self.config.duration),
            seed=self.config.course_seed,
            body_names=body_names,
            body_metadata=body_metadata,
        )
        self.course_layout = replace(
            self.course_layout,
            releases=tuple(
                replace(
                    release,
                    trigger_feature=release.trigger,
                    shape_kind=str(body_metadata.get(release.body_name, {}).get("shape_kind", "")),
                    profile_name=str(body_metadata.get(release.body_name, {}).get("profile_name", "")),
                    color=tuple(body_metadata.get(release.body_name, {}).get("color", (1.0, 1.0, 1.0, 1.0))),
                )
                for release in self.course_layout.releases
            ),
        )
        self.course_sections = self.course_layout.sections
        self._build_obstacle_course_obstacles()

        release_lookup = {release.body_name: release for release in self.course_layout.releases}
        intensity_scale = float(self.floating_intensity["event_scale"])
        for index, spec in enumerate(actor_specs):
            profile_name = str(spec["profile_name"])
            profile = COURSE_BODY_PROFILES[profile_name]
            release = release_lookup[str(spec["name"])]
            spawn_x, spawn_y, spawn_z = sample_route_position(self.course_layout, release.time_seconds)
            lane_bias = np.asarray(spec["lane_bias"], dtype=np.float32)
            spawn_pos = Vec3(
                float(spawn_x + lane_bias[0]),
                float(spawn_y + lane_bias[1]),
                float(spawn_z + 2.2 + 0.72 * index),
            )
            soft_body = TrueSoftBodyObject(
                self.root,
                self.world,
                self.world_info,
                variant="surface",
                radius=float(spec["radius"]),
                mass=float(spec["mass"]),
                seed=self.config.seed + index * 31,
                name=str(spec["name"]),
                initial_position=spawn_pos,
                preset_name=profile["preset"],
                render_shape_kind=str(spec["shape_kind"]),
                visualization=self.config.softbody_visualization,
                max_displacement_visual=0.36 if str(spec["role"]) == "hero" else 0.30,
                simulation_detail_multiplier=0.66 if str(spec["role"]) == "hero" else 0.58,
                render_detail_multiplier=1.16 if str(spec["role"]) == "hero" else 1.02,
                surface_color=spec["color"],
            )
            soft_body.set_phase_name("course_wait")
            soft_body.set_soft_soft_collision(True)
            soft_body.render_node.setShader(self.surface_shader)
            soft_body.render_node.setShaderInput("base_color", Vec4(*spec["color"]))
            self.gel_nodes.append(soft_body.render_node)
            actor = FloatingSoftActor(
                name=str(spec["name"]),
                soft_body=soft_body,
                base_color=spec["color"],
                role=str(spec["role"]),
                shape_kind=str(spec["shape_kind"]),
                delay_seconds=0.0,
                center_target=np.asarray([spawn_pos.x, spawn_pos.y, spawn_pos.z], dtype=np.float32),
                base_center_target=np.asarray([spawn_pos.x, spawn_pos.y, spawn_pos.z], dtype=np.float32),
                orbit_phase=0.0,
                orbit_radius=0.0,
                orbit_rate=0.0,
                response_scale=float(profile["response_scale"]) * intensity_scale,
                motion_scale=float(profile["motion_scale"]) * (0.92 + 0.08 * intensity_scale),
                profile_name=profile_name,
                release_time=float(release.time_seconds),
                release_trigger=release.trigger,
                released=index == 0,
                lane_bias=lane_bias.copy(),
                last_progress_time=0.0,
                last_progress_height=float(spawn_pos.z),
            )
            self._prepare_floating_regions(actor)
            self.floating_actors.append(actor)

        initial_twist_axes = (
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        )
        initial_velocities = (
            np.asarray([0.40, 0.22, -0.18], dtype=np.float32),
            np.asarray([-0.28, -0.18, -0.10], dtype=np.float32),
            np.asarray([0.34, -0.32, -0.06], dtype=np.float32),
            np.asarray([-0.22, 0.30, -0.08], dtype=np.float32),
            np.asarray([0.18, 0.34, -0.10], dtype=np.float32),
        )
        for actor, velocity, axis in zip(self.floating_actors, initial_velocities, initial_twist_axes, strict=True):
            all_indices = np.arange(actor.soft_body.get_node_positions().shape[0], dtype=np.int32)
            actor.soft_body.apply_velocity_to_indices(all_indices, velocity * np.float32(actor.motion_scale))
            actor.soft_body.apply_opposing_twist(axis / max(np.linalg.norm(axis), 1e-6), float(self.floating_intensity["twist_force"]) * 0.42 * actor.response_scale)

    def _build_obstacle_course_obstacles(self) -> None:
        x_limit = 5.2 + 0.90 * self.aspect_ratio
        y_limit = 6.4
        wall_specs = [
            (Vec3(x_limit + 0.58, 0.0, -4.0), Vec3(0.24, y_limit, 24.0)),
            (Vec3(-(x_limit + 0.58), 0.0, -4.0), Vec3(0.24, y_limit, 24.0)),
            (Vec3(0.0, y_limit + 0.52, -4.0), Vec3(x_limit + 0.40, 0.24, 24.0)),
            (Vec3(0.0, -(y_limit + 0.52), -4.0), Vec3(x_limit + 0.40, 0.24, 24.0)),
        ]
        for index, (pos, half_extents) in enumerate(wall_specs):
            node = BulletRigidBodyNode(f"course-wall-{index}")
            node.addShape(BulletBoxShape(half_extents))
            wall_np = self.root.attachNewNode(node)
            wall_np.setPos(pos)
            self.world.attachRigidBody(node)

        if self.course_layout is None:
            return
        for obstacle in self.course_layout.obstacles:
            body = BulletRigidBodyNode(obstacle.name)
            if obstacle.kind == "sphere":
                body.addShape(BulletSphereShape(float(obstacle.size[0])))
                visual_kind = "rounded_octahedron"
            else:
                body.addShape(BulletBoxShape(Vec3(*obstacle.size)))
                visual_kind = "rectangular_cuboid"
            body.setFriction(0.90)
            body.setRestitution(0.18 if obstacle.kind == "sphere" else 0.10)
            body_np = self.root.attachNewNode(body)
            body_np.setPos(*obstacle.position)
            body_np.setHpr(*obstacle.rotation)
            body_np.setPythonTag("base_pos", obstacle.position)
            body_np.setPythonTag("base_hpr", obstacle.rotation)
            body_np.setPythonTag("feature_source", obstacle.feature_source)
            body_np.setPythonTag("animated", obstacle.animated)
            self.world.attachRigidBody(body)
            visual = self._create_visual_geom(
                f"{obstacle.name}-visual",
                visual_kind,
                radius=max(float(obstacle.size[0]), 0.24),
                color=obstacle.color,
                parent=body_np,
            )
            visual.setScale(*obstacle.size)
            obstacle_tint = Vec4(
                obstacle.color[0] * 0.52,
                obstacle.color[1] * 0.52,
                obstacle.color[2] * 0.56,
                0.18 if obstacle.kind != "rail" else 0.13,
            )
            visual.setColorScale(obstacle_tint)
            visual.setPythonTag("feature_source", obstacle.feature_source)
            self.env_nodes.append(visual)
            self.course_obstacles.append((obstacle, body_np, body))
        self.course_obstacle_count = len(self.course_obstacles)

    def _section_for_time(self, time_seconds: float) -> CourseSection | None:
        for section in self.course_sections:
            if section.start_time <= time_seconds < section.end_time:
                return section
        return self.course_sections[-1] if self.course_sections else None

    def _course_phase(self, t: float) -> str:
        if self.course_layout is None:
            return "course"
        progress = t / max(self.course_layout.duration, 1e-6)
        if progress < 0.12:
            return "intro"
        if progress < 0.34:
            return "cascade_a"
        if progress < 0.58:
            return "cascade_b"
        if progress < 0.82:
            return "compression_run"
        return "final_drop"

    def _course_time_for_actor(self, actor: FloatingSoftActor, look_ahead: float = 0.0) -> float:
        if self.course_layout is None:
            return self.sim_time
        return float(np.clip(max(self.sim_time, actor.release_time) + look_ahead, 0.0, self.course_layout.duration))

    def _course_waypoint_for_actor(self, actor: FloatingSoftActor) -> np.ndarray:
        if self.course_layout is None:
            return actor.base_center_target.copy()
        route_x, route_y, route_z = sample_route_position(self.course_layout, self._course_time_for_actor(actor, look_ahead=0.95))
        target = np.asarray([route_x, route_y, route_z], dtype=np.float32) + actor.lane_bias
        target[2] += 0.28
        return target

    def _course_release_and_hold(self, actor: FloatingSoftActor) -> None:
        all_indices = np.arange(actor.soft_body.get_node_positions().shape[0], dtype=np.int32)
        if not actor.released:
            actor.soft_body.apply_spring_anchor(all_indices, actor.soft_body.rest_positions, stiffness=16.0, damping=4.0)
            if self.sim_time >= actor.release_time:
                actor.released = True
                release_push = np.asarray(
                    [
                        0.22 * np.sign(actor.lane_bias[0] if abs(actor.lane_bias[0]) > 1e-5 else 1.0),
                        0.16 * np.sign(actor.lane_bias[1] if abs(actor.lane_bias[1]) > 1e-5 else -1.0),
                        -0.26,
                    ],
                    dtype=np.float32,
                )
                actor.soft_body.apply_velocity_to_indices(all_indices, release_push * np.float32(actor.motion_scale))
                actor.soft_body.apply_opposing_twist(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), 0.56 * actor.response_scale)
                actor.event_log = f"release:{actor.release_trigger}"
                actor.active_event = actor.event_log
                self.last_active_event = actor.event_log
                self.course_release_events.add(actor.name)
                actor.last_progress_time = self.sim_time
                actor.last_progress_height = float(actor.soft_body.get_node_positions().mean(axis=0)[2])
        actor.soft_body.set_phase_name("course_hold" if not actor.released else self.phase_name)

    def _apply_course_boundary_safety(self, actor: FloatingSoftActor) -> None:
        positions = actor.soft_body.get_node_positions()
        velocities = actor.soft_body.get_node_velocities()
        corrections = np.zeros_like(positions, dtype=np.float32)
        active = np.zeros((positions.shape[0],), dtype=bool)
        horizontal_limit = 4.8 + 0.95 * self.aspect_ratio
        upper = np.asarray([horizontal_limit, 5.8, 15.8], dtype=np.float32)
        lower = np.asarray([-horizontal_limit, -5.8, -24.5], dtype=np.float32)
        for axis in range(3):
            high_mask = positions[:, axis] > upper[axis]
            if np.any(high_mask):
                corrections[high_mask, axis] -= (positions[high_mask, axis] - upper[axis]) * 8.8
                corrections[high_mask, axis] -= np.maximum(velocities[high_mask, axis], 0.0) * 1.4
                active |= high_mask
            low_mask = positions[:, axis] < lower[axis]
            if np.any(low_mask):
                corrections[low_mask, axis] += (lower[axis] - positions[low_mask, axis]) * 8.8
                corrections[low_mask, axis] += np.maximum(-velocities[low_mask, axis], 0.0) * 1.4
                active |= low_mask
        indices = np.where(active)[0].astype(np.int32)
        if indices.size:
            actor.soft_body.apply_velocities(indices, corrections[indices] * np.float32(0.05))
            self.containment_projection_events += int(indices.size)

    def _apply_course_guidance(self, actor: FloatingSoftActor) -> None:
        if not actor.released:
            return
        waypoint = self._course_waypoint_for_actor(actor)
        center = actor.soft_body.get_node_positions().mean(axis=0)
        error = waypoint - center
        horizontal_force = np.clip(error[:2] * np.asarray([0.22, 0.18], dtype=np.float32) * actor.motion_scale, -0.68, 0.68)
        depth_force = np.clip(error[2] * 0.08 * actor.motion_scale, -0.26, 0.14)
        force = np.asarray([horizontal_force[0], horizontal_force[1], depth_force], dtype=np.float32)
        actor.soft_body.apply_global_force(force)
        actor.center_target = waypoint

    def _apply_course_unstuck(self, actor: FloatingSoftActor) -> None:
        if not actor.released:
            return
        positions = actor.soft_body.get_node_positions()
        velocities = actor.soft_body.get_node_velocities()
        center = positions.mean(axis=0)
        mean_velocity = float(np.linalg.norm(velocities, axis=1).mean())
        if center[2] < actor.last_progress_height - 0.65:
            actor.last_progress_time = self.sim_time
            actor.last_progress_height = float(center[2])
            return
        if self.sim_time - actor.last_progress_time < 1.5:
            return
        if mean_velocity > 0.26:
            return
        nudge = np.asarray(
            [
                0.42 * np.sign(actor.lane_bias[0] if abs(actor.lane_bias[0]) > 1e-5 else 1.0),
                -0.32 * np.sign(actor.lane_bias[1] if abs(actor.lane_bias[1]) > 1e-5 else -1.0),
                -0.42,
            ],
            dtype=np.float32,
        )
        all_indices = np.arange(actor.soft_body.get_node_positions().shape[0], dtype=np.int32)
        actor.soft_body.apply_velocity_to_indices(all_indices, nudge)
        actor.unstuck_events += 1
        actor.last_progress_time = self.sim_time
        actor.event_log = "unstuck"
        actor.active_event = "unstuck"

    def _dominant_selector_for_normal(self, actor: FloatingSoftActor, normal: np.ndarray) -> str:
        axis = int(np.argmax(np.abs(normal)))
        if axis == 0:
            return "face_x_pos" if normal[0] >= 0.0 else "face_x_neg"
        if axis == 1:
            return "face_y_pos" if normal[1] >= 0.0 else "face_y_neg"
        return "face_z_pos" if normal[2] >= 0.0 else "face_z_neg"

    def _apply_course_contacts(self, actor: FloatingSoftActor) -> None:
        if not actor.released:
            return
        center = actor.soft_body.get_node_positions().mean(axis=0)
        for obstacle, obstacle_np, _ in self.course_obstacles:
            obstacle_center = np.asarray([obstacle_np.getX(), obstacle_np.getY(), obstacle_np.getZ()], dtype=np.float32)
            delta = center - obstacle_center
            distance = float(np.linalg.norm(delta))
            trigger_distance = obstacle.influence_radius + actor.soft_body.radius * 0.92
            if distance >= trigger_distance or self.sim_time < actor.next_contact_time:
                continue
            normal = delta / max(distance, 1e-5)
            profile = COURSE_BODY_PROFILES[actor.profile_name]
            selector_key = self._dominant_selector_for_normal(actor, normal)
            contact_strength = (trigger_distance - distance) * (10.8 + actor.response_scale * 4.6)
            actor.soft_body.apply_region_force(actor.selectors[selector_key], normal * np.float32(contact_strength * 0.62))
            tangent = np.cross(normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
            if np.linalg.norm(tangent) < 1e-5:
                tangent = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-5)
            actor.soft_body.apply_global_force(tangent * np.float32(1.10 * actor.motion_scale * float(profile.get("rebound_scale", 1.0))))
            actor.soft_body.apply_opposing_twist(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                0.56 * actor.response_scale * float(profile.get("twist_scale", 1.0)),
            )
            self._apply_contact_dent(actor, obstacle, normal, contact_strength)
            actor.contact_count += 1
            actor.next_contact_time = self.sim_time + (0.14 if actor.profile_name in {"springy", "firm_bouncy"} else 0.22)
            actor.active_impulses += 3
            actor.active_event = f"contact:{obstacle.name}"
            actor.event_log = actor.active_event
            self.last_active_event = actor.active_event
            break

    def _apply_course_audio(self, actor: FloatingSoftActor, sample: AudioSample) -> None:
        if not actor.released:
            actor.active_event = "waiting"
            return
        actor.active_event = actor.event_log if actor.event_log != "idle" else "descent"
        breath_scale = float(COURSE_BODY_PROFILES[actor.profile_name]["breath_scale"])
        bass_target = np.clip((sample.bass - 0.18) * 0.72, -0.06, 0.18)
        breath_delta = (bass_target - actor.bass_state) * 0.18
        actor.bass_state += breath_delta
        if abs(breath_delta) > 0.004:
            self._add_force_envelope(
                actor,
                name="bass-breath",
                action="breath",
                magnitude=breath_delta * float(self.floating_intensity["breath_force"]) * breath_scale,
                attack=0.05,
                hold=0.10,
                decay=0.24,
            )
            actor.active_event = "bass_breath"
            actor.event_log = actor.active_event

        if sample.is_beat and self.sim_time >= actor.next_beat_time:
            beat_selector_map = {
                "hero": "face_z_pos",
                "coral": "face_x_neg",
                "spring": "corner_a",
                "prism": "face_y_pos",
                "octa": "corner_c",
            }
            beat_recover_map = {
                "hero": "face_z_neg",
                "coral": "face_x_pos",
                "spring": "corner_b",
                "prism": "face_y_neg",
                "octa": "corner_b",
            }
            beat_vector = np.asarray(
                [
                    0.84 * np.sign(actor.lane_bias[0] if abs(actor.lane_bias[0]) > 1e-5 else 1.0),
                    -0.26 * np.sign(actor.lane_bias[1] if abs(actor.lane_bias[1]) > 1e-5 else -1.0),
                    -0.18,
                ],
                dtype=np.float32,
            )
            beat_vector *= np.float32((0.18 + sample.bass * 0.14) * actor.motion_scale * float(self.floating_intensity["beat_force"]))
            self._add_force_envelope(
                actor,
                name="beat-drive",
                action="global_velocity",
                magnitude=1.0,
                attack=0.02,
                hold=0.05,
                decay=0.24,
                vector=beat_vector,
            )
            self._add_force_envelope(
                actor,
                name="beat-dent",
                action="region_velocity",
                magnitude=1.0,
                attack=0.02,
                hold=0.07,
                decay=0.22,
                selector_key=beat_selector_map.get(actor.role, "face_z_pos"),
                vector=np.asarray([0.0, 0.0, -0.98 * actor.response_scale], dtype=np.float32) if actor.role == "hero" else np.asarray([0.62, 0.24, -0.52], dtype=np.float32) * np.float32(actor.response_scale),
            )
            self._add_force_envelope(
                actor,
                name="beat-recover",
                action="region_velocity",
                magnitude=1.0,
                attack=0.04,
                hold=0.06,
                decay=0.28,
                selector_key=beat_recover_map.get(actor.role, "face_z_neg"),
                vector=np.asarray([0.0, 0.0, 0.56 * actor.response_scale], dtype=np.float32) if actor.role == "hero" else np.asarray([-0.34, -0.14, 0.32], dtype=np.float32) * np.float32(actor.response_scale),
            )
            actor.next_beat_time = self.sim_time + 0.34
            actor.active_event = "beat_drive"
            actor.event_log = actor.active_event

        if sample.onset >= 0.52 and self.sim_time >= actor.next_onset_time:
            corner_cycle = {
                "hero": "corner_b",
                "coral": "corner_c",
                "spring": "corner_a",
                "prism": "corner_b",
                "octa": "corner_c",
            }
            corner_key = corner_cycle.get(actor.role, "corner_a")
            self._add_force_envelope(
                actor,
                name="onset-strike",
                action="region_velocity",
                magnitude=1.0,
                attack=0.01,
                hold=0.04,
                decay=0.30,
                selector_key=corner_key,
                vector=np.asarray([0.86, -0.58, 0.28], dtype=np.float32) * np.float32(float(self.floating_intensity["onset_force"]) * 0.26 * actor.response_scale),
            )
            actor.next_onset_time = self.sim_time + 0.32
            actor.active_event = f"onset:{corner_key}"
            actor.event_log = actor.active_event

        if sample.mid >= 0.40 and self.sim_time >= actor.next_twist_time:
            axis_map = {
                "hero": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                "coral": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                "spring": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "prism": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
                "octa": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            }
            self._add_force_envelope(
                actor,
                name="mid-twist",
                action="twist",
                magnitude=(0.48 + sample.mid * 0.72) * actor.response_scale,
                attack=0.04,
                hold=0.08,
                decay=0.34,
                axis=axis_map.get(actor.role, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
            )
            actor.next_twist_time = self.sim_time + 0.38
            actor.active_event = "mid_twist"
            actor.event_log = actor.active_event

        if sample.high >= 0.52 and self.sim_time >= actor.next_high_time:
            selector_key = "face_y_pos" if actor.role in {"hero", "octa"} else "face_x_neg"
            self._add_force_envelope(
                actor,
                name="high-ripple",
                action="region_velocity",
                magnitude=1.0,
                attack=0.01,
                hold=0.02,
                decay=0.12,
                selector_key=selector_key,
                vector=np.asarray([0.11, 0.08, 0.06], dtype=np.float32) * np.float32(float(self.floating_intensity["high_force"]) * actor.response_scale),
            )
            actor.next_high_time = self.sim_time + 0.16

    def _update_course_background_animation(self, sample: AudioSample) -> None:
        for index, panel in enumerate(self.floating_panels):
            base_x, base_y, base_z = panel.getPythonTag("base_pos")
            sweep = 0.22 * np.sin(self.sim_time * 0.12 + index * 0.5)
            panel.setPos(base_x + sweep, base_y, base_z + 0.18 * np.cos(self.sim_time * 0.10 + index * 0.4))
            panel.setColorScale(1.0 + sample.rms * 0.16, 1.0 + sample.high * 0.12, 1.0 + sample.mid * 0.10, 1.0)
        for index, line in enumerate(self.floating_grid_lines):
            pulse = 0.92 + 0.10 * np.sin(self.sim_time * 0.40 + index * 0.18)
            line.setColorScale(pulse, pulse + sample.high * 0.12, 1.0 + sample.rms * 0.08, 1.0)
        for obstacle, obstacle_np, _ in self.course_obstacles:
            if not obstacle.animated:
                continue
            base_pos = obstacle_np.getPythonTag("base_pos")
            base_hpr = obstacle_np.getPythonTag("base_hpr")
            if obstacle.feature_source == "beat":
                obstacle_np.setHpr(base_hpr[0], base_hpr[1], base_hpr[2] + 8.0 * np.sin(self.sim_time * 1.8 + obstacle.section_id * 0.3) * sample.rms)
            elif obstacle.feature_source == "onset":
                obstacle_np.setPos(base_pos[0], base_pos[1] + 0.18 * np.sin(self.sim_time * 1.4 + obstacle.section_id), base_pos[2])
            elif obstacle.feature_source == "bass":
                obstacle_np.setScale(1.0 + sample.bass * 0.08, 1.0 + sample.bass * 0.08, 1.0 + sample.bass * 0.08)
            elif obstacle.feature_source == "high":
                obstacle_np.setPos(base_pos[0], base_pos[1], base_pos[2] + 0.16 * np.sin(self.sim_time * 2.2 + obstacle.section_id))

    def _update_course_camera(self, sample: AudioSample) -> None:
        released = [actor for actor in self.floating_actors if actor.released]
        active = released if released else self.floating_actors
        if not active:
            return
        centers = np.asarray([self.course_metrics_cache[actor.name].center_of_mass for actor in active], dtype=np.float32)
        combined = np.vstack([actor.soft_body.get_node_positions() for actor in active])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        span = np.maximum(maxs - mins, np.asarray([0.1, 0.1, 0.1], dtype=np.float32))
        hero_center = self.course_metrics_cache[self.floating_actors[0].name].center_of_mass
        group_center = centers.mean(axis=0)
        lead_body = min(active, key=lambda actor: self.course_metrics_cache[actor.name].center_of_mass[2])
        lead_center = self.course_metrics_cache[lead_body.name].center_of_mass
        focus = hero_center * 0.40 + group_center * 0.28 + lead_center * 0.32
        focus[2] += 0.65
        self.camera_focus += (focus - self.camera_focus) * 0.06
        self.phase_name = self._course_phase(self.sim_time)
        vfov = np.deg2rad(33.0)
        hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * self.aspect_ratio)
        width_requirement = (span[0] * 0.96) / max(np.tan(hfov * 0.5) * 0.82, 1e-3)
        height_requirement = (span[2] * 0.90) / max(np.tan(vfov * 0.5) * 0.78, 1e-3)
        base_distance = 7.1 + sample.rms * 0.7
        distance = max(base_distance, width_requirement, height_requirement) + span[1] * 0.45
        route_look_x, route_look_y, route_look_z = sample_route_position(self.course_layout, self._course_time_for_actor(lead_body, look_ahead=1.25)) if self.course_layout else (0.0, 0.0, 0.0)
        self.course_camera_route_time = self._course_time_for_actor(lead_body, look_ahead=1.25)
        x_offset = 0.82 * np.sin(self.sim_time * 0.16)
        y_offset = 0.44 * np.cos(self.sim_time * 0.12)
        z_offset = 3.25 + 0.24 * np.cos(self.sim_time * 0.18)
        self.base.camera.setPos(
            self.camera_focus[0] + x_offset,
            self.camera_focus[1] - distance + y_offset,
            self.camera_focus[2] + z_offset,
        )
        self.base.camera.lookAt(
            float(self.camera_focus[0] * 0.56 + route_look_x * 0.44),
            float(self.camera_focus[1] * 0.64 + route_look_y * 0.36),
            float(self.camera_focus[2] * 0.52 + route_look_z * 0.48 + 0.86),
        )
        self._update_safe_frame_stats()
        if self.current_edge_margin_min < 0.05:
            correction = (0.05 - self.current_edge_margin_min) * 3.2
            self.base.camera.setY(self.base.render, self.base.camera.getY(self.base.render) - correction)
            self._update_safe_frame_stats()
        elif self.current_edge_margin_min > 0.18:
            tighten = min((self.current_edge_margin_min - 0.18) * 2.2, 0.48)
            self.base.camera.setY(self.base.render, self.base.camera.getY(self.base.render) + tighten)
            self._update_safe_frame_stats()
        self.current_camera_distance = float(np.linalg.norm(np.asarray(self.base.camera.getPos(self.base.render)) - self.camera_focus))

    def _update_course_overlay(self, sample: AudioSample) -> None:
        if self.overlay is None:
            return
        current_section = self._section_for_time(self.sim_time)
        lines = [
            "softbody obstacle course",
            f"time: {self.sim_time:05.2f}s section: {current_section.section_id if current_section else '-'}",
            f"phase: {self.phase_name}",
            f"event: {self.last_active_event}",
            f"beat/onset: {int(sample.is_beat)} / {sample.onset:0.2f}",
            f"obstacles: {self.course_obstacle_count} route_t: {self.course_camera_route_time:0.2f}",
            f"camera distance: {self.current_camera_distance:0.2f} safe: {self.current_safe_frame_violation_count}",
        ]
        for actor in self.floating_actors:
            metrics = self.course_metrics_cache.get(actor.name) or actor.soft_body.get_metrics()
            release_state = "released" if actor.released else f"{actor.release_trigger}@{actor.release_time:0.1f}"
            active_envelopes = [envelope.name for envelope in actor.envelopes if self._envelope_value(envelope, self.sim_time) > 1e-4]
            lines.append(
                f"{actor.name}:{actor.profile_name} {release_state} vol={metrics.volume_ratio:0.3f} deform={metrics.aligned_max_deformation:0.3f} contacts={actor.contact_count} unstuck={actor.unstuck_events} env={','.join(active_envelopes[:2]) if active_envelopes else 'none'}"
            )
        if self.config.framing_debug_overlay:
            lines.append(f"frame: {self.config.width}x{self.config.height} aspect={self.aspect_ratio:0.3f}")
            lines.append(f"edge margin min: {self.current_edge_margin_min:0.3f}")
        self.overlay.setText("\n".join(lines))

    def _record_course_metrics(self, dt: float, physics_seconds: float, sample: AudioSample) -> None:
        current_section = self._section_for_time(self.sim_time)
        for actor in self.floating_actors:
            metrics = self.course_metrics_cache.get(actor.name) or actor.soft_body.get_metrics()
            screen_stats = self.current_screen_stats.get(actor.role, {})
            if (
                screen_stats.get("radius_y", 0.0) > 0.015
                and -1.0 <= screen_stats.get("center_x", -2.0) <= 1.0
                and -1.0 <= screen_stats.get("center_y", -2.0) <= 1.0
            ):
                actor.visible_frames += 1
            actor.peak_deformation = max(actor.peak_deformation, float(metrics.aligned_max_deformation))
            actor.peak_velocity = max(actor.peak_velocity, float(metrics.max_velocity))
            active_envelopes = [envelope.name for envelope in actor.envelopes if self._envelope_value(envelope, self.sim_time) > 1e-4]
            self.metrics_rows.append(
                {
                    "audio_time": round(self.sim_time, 6),
                    "phase": self.phase_name,
                    "section_id": current_section.section_id if current_section else -1,
                    "section_theme": current_section.theme if current_section else "",
                    "generated_obstacle_count": self.course_obstacle_count,
                    "body_id": actor.name,
                    "body_shape": actor.shape_kind,
                    "body_color": "|".join(f"{component:0.3f}" for component in actor.base_color[:3]),
                    "body_profile": actor.profile_name,
                    "release_state": "released" if actor.released else "holding",
                    "release_trigger": actor.release_trigger,
                    "position_x": float(metrics.center_of_mass[0]),
                    "position_y": float(metrics.center_of_mass[1]),
                    "position_z": float(metrics.center_of_mass[2]),
                    "velocity": metrics.mean_velocity,
                    "max_velocity": metrics.max_velocity,
                    "angular_velocity_proxy": metrics.twist_angle_proxy,
                    "volume_ratio": metrics.volume_ratio,
                    "aligned_deformation": metrics.aligned_max_deformation,
                    "aligned_rms_deformation": metrics.aligned_rms_deformation,
                    "active_event": actor.active_event,
                    "active_envelopes": "|".join(active_envelopes),
                    "contact_count": actor.contact_count,
                    "meaningful_impact_count": actor.meaningful_impacts,
                    "unstuck_events": actor.unstuck_events,
                    "screen_x": screen_stats.get("center_x", 0.0),
                    "screen_y": screen_stats.get("center_y", 0.0),
                    "screen_radius_x": screen_stats.get("radius_x", 0.0),
                    "screen_radius_y": screen_stats.get("radius_y", 0.0),
                    "safe_frame_violations": self.current_safe_frame_violation_count,
                    "containment_corrections": self.containment_projection_events,
                    "camera_distance": float(np.linalg.norm(np.asarray(self.base.camera.getPos(self.base.render)) - metrics.center_of_mass)),
                    "audio_beat": int(sample.is_beat),
                    "audio_onset": sample.onset,
                    "audio_rms": sample.rms,
                    "audio_bass": sample.bass,
                    "audio_mid": sample.mid,
                    "audio_high": sample.high,
                    "simulation_step_duration_ms": physics_seconds * 1000.0,
                    "dt": dt,
                }
            )

    def _create_soft_body(self, variant: str, position: Vec3, visualization: str) -> TrueSoftBodyObject:
        soft_body = TrueSoftBodyObject(
            self.root,
            self.world,
            self.world_info,
            variant=variant,
            radius=1.55 if self.is_soft_stress else 1.25,
            mass=5.2 if variant == "surface" else 6.8,
            seed=self.config.seed,
            name=f"soft-{variant}",
            initial_position=position,
            preset_name=self.config.softbody_preset,
            visualization=visualization,
            max_displacement_visual=0.05 if variant == "surface" else 0.10,
        )
        soft_body.set_phase_name(self.phase_name)
        soft_body.render_node.setShader(self.surface_shader)
        soft_body.render_node.setShaderInput("base_color", Vec4(0.24, 0.88, 1.0, 1.0))
        soft_body.wireframe_np.setShader(self.surface_shader)
        soft_body.wireframe_np.setShaderInput("base_color", Vec4(0.96, 0.96, 1.0, 1.0))
        if self.is_translucent:
            self.gel_nodes.append(soft_body.render_node)
            self.gel_nodes.append(soft_body.wireframe_np)
            self.gel_nodes.append(soft_body.node_points_np)
        else:
            self.gel_nodes.append(soft_body.render_node)
        return soft_body

    def _create_visual_geom(self, name: str, shape_kind: str, radius: float, color: tuple[float, float, float, float], parent: NodePath) -> NodePath:
        name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name)) % 997
        mesh = generate_shape_mesh_data(shape_kind, seed=self.config.seed + name_seed, radius=radius)
        geom = create_render_geom_node(name, mesh.vertices, mesh.normals, mesh.texcoords, mesh.indices)
        node = parent.attachNewNode(geom)
        node.setShader(self.surface_shader)
        node.setShaderInput("base_color", Vec4(*color))
        return node

    def _create_projectile(self, name: str, start_pos: Vec3, velocity: Vec3) -> tuple[NodePath, BulletRigidBodyNode, NodePath]:
        body = BulletRigidBodyNode(name)
        body.addShape(BulletSphereShape(0.18))
        body.setMass(1.45)
        body.setFriction(0.86)
        body.setRestitution(0.12)
        body_np = self.root.attachNewNode(body)
        body_np.setPos(start_pos)
        self.world.attachRigidBody(body)
        body.setLinearVelocity(velocity)
        body.setAngularVelocity(Vec3(3.2, -2.1, 1.4))
        visual = self._create_visual_geom(f"{name}-visual", "rounded_octahedron", radius=0.22, color=(1.0, 0.76, 0.24, 1.0), parent=body_np)
        return body_np, body, visual

    def _create_translucency_objects(self) -> None:
        thin_cube_root = self.root.attachNewNode("thin-cube-root")
        thin_cube_root.setPos(-4.2, 0.0, 1.85)
        thin_cube_root.setScale(1.0, 0.55, 1.0)
        thin_cube = self._create_visual_geom("thin-cube", "rounded_cube", radius=1.15, color=(0.18, 0.98, 1.0, 1.0), parent=thin_cube_root)
        self.gel_nodes.append(thin_cube_root)
        self.thin_cube_np = thin_cube_root

        thick_cube_root = self.root.attachNewNode("thick-cube-root")
        thick_cube_root.setPos(0.0, 0.0, 1.95)
        thick_cube_root.setScale(1.0, 1.35, 1.0)
        thick_cube = self._create_visual_geom("thick-cube", "rounded_cube", radius=1.25, color=(0.95, 0.42, 0.56, 1.0), parent=thick_cube_root)
        self.gel_nodes.append(thick_cube_root)
        self.thick_cube_np = thick_cube_root

        torus_root = self.root.attachNewNode("translucent-torus-root")
        torus_root.setPos(4.3, 0.0, 1.85)
        torus = self._create_visual_geom("trans-torus", "torus_ring", radius=1.20, color=(1.0, 0.84, 0.14, 1.0), parent=torus_root)
        self.gel_nodes.append(torus_root)
        self.torus_np = torus_root

    def _prepare_softbody_regions(self) -> None:
        if self.soft_body is None:
            return
        self.bottom_anchor_indices = self.soft_body.select_bottom_nodes(0.62, label="stress-bottom-anchor")
        bottom_center = np.asarray([0.0, 0.0, -self.soft_body.radius], dtype=np.float32) + self.soft_body.initial_position
        center_point = self.soft_body.initial_position.copy()
        self.bottom_hold_indices = self.soft_body.select_nodes_near_point(bottom_center, 0.72, label="stress-bottom-hold")
        self.center_indices = self.soft_body.select_nodes_near_point(center_point, 1.15, label="stress-center")
        if self.center_indices.size == 0:
            distances = np.linalg.norm(self.soft_body.rest_positions - center_point[None, :], axis=1)
            self.center_indices = np.argsort(distances)[:32].astype(np.int32)
        self.top_indices = self.soft_body.select_top_nodes(0.42, label="stress-top")
        self.left_indices = self.soft_body.select_nodes_on_face(0, -1, 0.34, label="stress-left")
        self.right_indices = self.soft_body.select_nodes_on_face(0, 1, 0.34, label="stress-right")
        self.struck_indices = self.soft_body.select_corner_nodes(1, -1, 1, 0.42, label="stress-struck-corner")
        self.opposite_indices = self.soft_body.select_corner_nodes(-1, 1, -1, 0.42, label="stress-opposite-corner")
        self.anchor_targets = self.soft_body.rest_positions[self.bottom_anchor_indices].copy()
        self.bottom_hold_targets = self.soft_body.rest_positions[self.bottom_hold_indices].copy()
        self.center_planar_targets = self.soft_body.rest_positions[self.center_indices][:, :2].copy()

    def _prepare_floating_regions(self, actor: FloatingSoftActor) -> None:
        body = actor.soft_body
        face_thickness = 0.30 if actor.role == "hero" else 0.16
        z_threshold = 0.48 if actor.role == "hero" else 0.34
        corner_radius = 0.46 if actor.role == "hero" else 0.34
        actor.selectors["face_x_neg"] = body.select_nodes_on_face(0, -1, face_thickness, label=f"{actor.name}-x-neg")
        actor.selectors["face_x_pos"] = body.select_nodes_on_face(0, 1, face_thickness, label=f"{actor.name}-x-pos")
        actor.selectors["face_y_neg"] = body.select_nodes_on_face(1, -1, face_thickness, label=f"{actor.name}-y-neg")
        actor.selectors["face_y_pos"] = body.select_nodes_on_face(1, 1, face_thickness, label=f"{actor.name}-y-pos")
        actor.selectors["face_z_pos"] = body.select_top_nodes(z_threshold, label=f"{actor.name}-z-pos")
        actor.selectors["face_z_neg"] = body.select_bottom_nodes(z_threshold, label=f"{actor.name}-z-neg")
        actor.selectors["corner_a"] = body.select_corner_nodes(1, 1, 1, corner_radius, label=f"{actor.name}-corner-a")
        actor.selectors["corner_b"] = body.select_corner_nodes(-1, -1, -1, corner_radius, label=f"{actor.name}-corner-b")
        actor.selectors["corner_c"] = body.select_corner_nodes(-1, 1, 1, corner_radius, label=f"{actor.name}-corner-c")
        center_radius = 0.72 if actor.role == "hero" else 0.44
        actor.selectors["center"] = body.select_nodes_near_rest_point(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), center_radius, label=f"{actor.name}-center")
        if actor.selectors["center"].size == 0:
            distances = np.linalg.norm(body.local_rest_positions, axis=1)
            actor.selectors["center"] = np.argsort(distances)[: max(6, distances.shape[0] // 24)].astype(np.int32)

    def _apply_planar_centering(self, stiffness: float, damping: float) -> None:
        if self.soft_body is None or self.center_indices.size == 0:
            return
        self.soft_body.apply_planar_anchor(self.center_indices, self.center_planar_targets, stiffness=stiffness, damping=damping)

    def _apply_floating_boundary_safety(self, actor: FloatingSoftActor) -> None:
        positions = actor.soft_body.get_node_positions()
        velocities = actor.soft_body.get_node_velocities()
        corrections = np.zeros_like(positions, dtype=np.float32)
        active = np.zeros((positions.shape[0],), dtype=bool)
        horizontal_limit = 3.45 + 0.86 * self.aspect_ratio
        limits = np.asarray([horizontal_limit, 4.05, 4.85], dtype=np.float32)
        lower = np.asarray([-horizontal_limit, -4.55, -0.35], dtype=np.float32)
        for axis in range(3):
            high_mask = positions[:, axis] > limits[axis]
            if np.any(high_mask):
                corrections[high_mask, axis] -= (positions[high_mask, axis] - limits[axis]) * 12.0
                corrections[high_mask, axis] -= np.maximum(velocities[high_mask, axis], 0.0) * 1.8
                active |= high_mask
            low_mask = positions[:, axis] < lower[axis]
            if np.any(low_mask):
                corrections[low_mask, axis] += (lower[axis] - positions[low_mask, axis]) * 12.0
                corrections[low_mask, axis] += np.maximum(-velocities[low_mask, axis], 0.0) * 1.8
                active |= low_mask
        indices = np.where(active)[0].astype(np.int32)
        if indices.size:
            actor.soft_body.apply_velocities(indices, corrections[indices] * np.float32(0.05))
            self.containment_projection_events += int(indices.size)

    def _apply_floating_centering(self, actor: FloatingSoftActor) -> None:
        metrics = actor.soft_body.get_metrics()
        error = actor.center_target - metrics.center_of_mass
        centering = float(self.floating_intensity["centering"])
        force = np.clip(error * np.asarray([1.0, 0.9, 0.8], dtype=np.float32) * centering, -0.52, 0.52)
        actor.soft_body.apply_global_force(force)

    def _course_phase(self, t: float) -> str:
        if t < 2.0:
            return "spawn"
        if t < 5.0:
            return "upper_course"
        if t < 9.0:
            return "mid_course"
        if t < 12.0:
            return "lower_rebound"
        return "exit"

    def _course_waypoint_for_actor(self, actor: FloatingSoftActor) -> np.ndarray:
        waypoints = [
            np.asarray([-2.4, 0.3, 8.4], dtype=np.float32),
            np.asarray([1.8, -0.2, 7.0], dtype=np.float32),
            np.asarray([-1.8, 0.8, 5.8], dtype=np.float32),
            np.asarray([0.4, -0.5, 4.4], dtype=np.float32),
            np.asarray([2.1, 0.2, 2.9], dtype=np.float32),
            np.asarray([-0.8, 0.3, 1.7], dtype=np.float32),
            np.asarray([1.3, -0.2, 0.6], dtype=np.float32),
        ]
        if actor.role == "spring":
            offsets = np.asarray([0.6, 0.8, 0.0], dtype=np.float32)
        elif actor.role == "support":
            offsets = np.asarray([0.3, -0.5, 0.0], dtype=np.float32)
        else:
            offsets = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        center = actor.soft_body.get_metrics().center_of_mass
        for waypoint in waypoints:
            if center[2] >= waypoint[2]:
                return waypoint + offsets
        return waypoints[-1] + offsets

    def _floating_phase(self, t: float) -> str:
        if t < 2.0:
            return "reveal"
        if t < 4.0:
            return "first_strike"
        if t < 6.0:
            return "close_pass"
        if t < 8.0:
            return "twist_arc"
        if t < 10.0:
            return "separation"
        return "finale"

    def _envelope_value(self, envelope: FloatingEnvelope, t: float) -> float:
        elapsed = t - envelope.start_time
        if elapsed < 0.0:
            return 0.0
        attack_end = envelope.attack
        hold_end = attack_end + envelope.hold
        decay_end = hold_end + envelope.decay
        if elapsed <= attack_end:
            return envelope.magnitude * (elapsed / max(envelope.attack, 1e-6))
        if elapsed <= hold_end:
            return envelope.magnitude
        if elapsed <= decay_end:
            return envelope.magnitude * max(0.0, 1.0 - ((elapsed - hold_end) / max(envelope.decay, 1e-6)))
        return 0.0

    def _prune_actor_envelopes(self, actor: FloatingSoftActor) -> None:
        actor.envelopes = [
            envelope
            for envelope in actor.envelopes
            if (self.sim_time - envelope.start_time) <= (envelope.attack + envelope.hold + envelope.decay + 1e-6)
        ]

    def _add_force_envelope(
        self,
        actor: FloatingSoftActor,
        name: str,
        action: str,
        magnitude: float,
        attack: float,
        hold: float,
        decay: float,
        selector_key: str | None = None,
        vector: np.ndarray | None = None,
        axis: np.ndarray | None = None,
    ) -> None:
        actor.envelopes.append(
            FloatingEnvelope(
                name=name,
                action=action,
                start_time=self.sim_time,
                attack=attack,
                hold=hold,
                decay=decay,
                magnitude=magnitude,
                selector_key=selector_key,
                vector=None if vector is None else np.asarray(vector, dtype=np.float32),
                axis=None if axis is None else np.asarray(axis, dtype=np.float32),
            )
        )

    def _apply_actor_envelopes(self, actor: FloatingSoftActor) -> None:
        self._prune_actor_envelopes(actor)
        for envelope in actor.envelopes:
            value = self._envelope_value(envelope, self.sim_time)
            if value <= 1e-5:
                continue
            if envelope.action == "region_force" and envelope.selector_key is not None and envelope.vector is not None:
                actor.soft_body.apply_region_force(actor.selectors[envelope.selector_key], envelope.vector * np.float32(value))
            elif envelope.action == "region_velocity" and envelope.selector_key is not None and envelope.vector is not None:
                actor.soft_body.apply_velocity_to_indices(actor.selectors[envelope.selector_key], envelope.vector * np.float32(value))
            elif envelope.action == "twist" and envelope.axis is not None:
                actor.soft_body.apply_opposing_twist(envelope.axis, value)
            elif envelope.action == "breath":
                actor.soft_body.apply_bass_breath(value)
            elif envelope.action == "global_force" and envelope.vector is not None:
                actor.soft_body.apply_global_force(envelope.vector * np.float32(value))
            elif envelope.action == "global_velocity" and envelope.vector is not None:
                actor.soft_body.apply_velocity_to_indices(
                    np.arange(actor.soft_body.get_node_positions().shape[0], dtype=np.int32),
                    envelope.vector * np.float32(value),
                )
            actor.active_impulses += 1

    def _contact_corner_selector(self, actor: FloatingSoftActor, normal: np.ndarray) -> str:
        return (
            f"corner_"
            f"{'a' if normal[0] >= 0.0 and normal[1] < 0.0 else 'b' if normal[0] < 0.0 and normal[2] >= 0.0 else 'c'}"
        )

    def _apply_contact_dent(
        self,
        actor: FloatingSoftActor,
        obstacle: CourseObstacle,
        normal: np.ndarray,
        strength: float,
    ) -> None:
        selector_key = self._dominant_selector_for_normal(actor, normal)
        opposite_key = self._dominant_selector_for_normal(actor, -normal)
        tangent = np.cross(normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(tangent) < 1e-5:
            tangent = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-5)
        profile = COURSE_BODY_PROFILES[actor.profile_name]
        dent_scale = float(profile.get("contact_dent_scale", 1.0))
        rebound_scale = float(profile.get("rebound_scale", 1.0))
        twist_scale = float(profile.get("twist_scale", 1.0))
        assist_scale = dent_scale * (1.0 + 0.26 * float(self.floating_intensity["event_scale"]))
        inward_velocity = (-normal * np.float32(strength * 0.18 * assist_scale)).astype(np.float32)
        counter_velocity = (normal * np.float32(strength * 0.08 * rebound_scale)).astype(np.float32)
        shear_velocity = (tangent * np.float32(strength * 0.06 * twist_scale)).astype(np.float32)
        self._add_force_envelope(actor, f"{obstacle.name}-dent", "region_velocity", 1.0, 0.01, 0.05, 0.18, selector_key=selector_key, vector=inward_velocity)
        self._add_force_envelope(actor, f"{obstacle.name}-counter", "region_velocity", 1.0, 0.03, 0.05, 0.28, selector_key=opposite_key, vector=counter_velocity)
        self._add_force_envelope(actor, f"{obstacle.name}-shear", "region_velocity", 1.0, 0.02, 0.04, 0.22, selector_key=self._contact_corner_selector(actor, normal), vector=shear_velocity)
        self._add_force_envelope(actor, f"{obstacle.name}-rebound", "global_velocity", 1.0, 0.03, 0.06, 0.32, vector=(normal * np.float32(strength * 0.035 * rebound_scale)))
        self._add_force_envelope(actor, f"{obstacle.name}-wobble", "twist", strength * 0.080 * twist_scale, 0.03, 0.06, 0.36, axis=np.asarray([normal[1], -normal[0], 0.55], dtype=np.float32))
        actor.meaningful_impacts += 1

    def _queue_choreography_cues(self) -> None:
        cues = (
            ("hero_strike", 2.05),
            ("secondary_pass", 4.08),
            ("hero_twist", 6.02),
            ("rebound_split", 8.10),
            ("finale_converge", 10.08),
        )
        for cue_name, cue_time in cues:
            if cue_name not in self.floating_cues_fired and self.sim_time >= cue_time:
                self.floating_cues_fired.add(cue_name)
                self._trigger_choreography_cue(cue_name)

    def _trigger_choreography_cue(self, cue_name: str) -> None:
        if len(self.floating_actors) < 2:
            return
        self.last_active_event = cue_name
        event_scale = float(self.floating_intensity["event_scale"])
        hero = self.floating_actors[0]
        secondary = self.floating_actors[1]
        if cue_name == "hero_strike":
            self._add_force_envelope(hero, "hero_strike_corner", "region_force", 4.6 * hero.response_scale * event_scale, 0.05, 0.10, 0.56, selector_key="corner_b", vector=np.asarray([-1.25, 0.38, 0.52], dtype=np.float32))
            self._add_force_envelope(hero, "hero_counter", "region_force", 2.2 * hero.response_scale * event_scale, 0.04, 0.08, 0.40, selector_key="face_x_pos", vector=np.asarray([0.72, -0.12, 0.10], dtype=np.float32))
            hero.soft_body.apply_velocity_to_indices(np.arange(hero.soft_body.get_node_positions().shape[0], dtype=np.int32), np.asarray([0.18, 0.44, 0.12], dtype=np.float32) * np.float32(event_scale))
        elif cue_name == "secondary_pass":
            self._add_force_envelope(secondary, "secondary_dive", "global_force", 3.0 * secondary.motion_scale * event_scale, 0.08, 0.22, 0.92, vector=np.asarray([-0.18, -1.22, 0.28], dtype=np.float32))
            self._add_force_envelope(secondary, "secondary_corner", "region_force", 1.7 * secondary.response_scale * event_scale, 0.05, 0.08, 0.34, selector_key="corner_a", vector=np.asarray([0.82, -0.46, 0.18], dtype=np.float32))
        elif cue_name == "hero_twist":
            self._add_force_envelope(hero, "hero_twist", "twist", 2.8 * hero.response_scale * event_scale, 0.12, 0.34, 0.98, axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
            self._add_force_envelope(hero, "hero_breath", "breath", 1.10 * hero.response_scale * event_scale, 0.16, 0.34, 0.94)
        elif cue_name == "rebound_split":
            self._add_force_envelope(hero, "hero_rebound", "global_force", 1.4 * hero.motion_scale * event_scale, 0.04, 0.10, 0.55, vector=np.asarray([-0.75, 0.52, 0.12], dtype=np.float32))
            self._add_force_envelope(secondary, "secondary_rebound", "global_force", 2.7 * secondary.motion_scale * event_scale, 0.04, 0.10, 0.55, vector=np.asarray([1.10, -0.68, 0.16], dtype=np.float32))
        elif cue_name == "finale_converge":
            self._add_force_envelope(hero, "hero_finale", "region_force", 4.0 * hero.response_scale * event_scale, 0.06, 0.16, 0.86, selector_key="face_z_neg", vector=np.asarray([0.0, 0.0, 0.95], dtype=np.float32))
            self._add_force_envelope(secondary, "secondary_finale_twist", "twist", 1.8 * secondary.response_scale * event_scale, 0.05, 0.16, 0.72, axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float32))

    def _update_floating_targets(self, sample: AudioSample) -> None:
        if not self.floating_actors:
            return
        phase = self._floating_phase(self.sim_time)
        performance_scale = 1.0 + sample.rms * 0.35
        stage_x = min(self.floating_stage_half_width * 0.44, 4.35)
        phase_targets = {
            "reveal": (
                np.asarray([-stage_x * 0.58, -1.35, 2.00], dtype=np.float32),
                np.asarray([stage_x * 0.72, 1.72, 1.42], dtype=np.float32),
            ),
            "first_strike": (
                np.asarray([-stage_x * 0.34, -0.48, 2.15], dtype=np.float32),
                np.asarray([stage_x * 0.54, 0.92, 1.58], dtype=np.float32),
            ),
            "close_pass": (
                np.asarray([-stage_x * 0.30, 1.18, 2.02], dtype=np.float32),
                np.asarray([0.78, -2.55, 1.22], dtype=np.float32),
            ),
            "twist_arc": (
                np.asarray([0.12, -0.05, 2.20], dtype=np.float32),
                np.asarray([stage_x * 0.66, 1.54, 1.16], dtype=np.float32),
            ),
            "separation": (
                np.asarray([-stage_x * 0.80, -1.24, 1.88], dtype=np.float32),
                np.asarray([stage_x * 0.86, 1.60, 1.62], dtype=np.float32),
            ),
            "finale": (
                np.asarray([-stage_x * 0.24, -0.70, 1.96], dtype=np.float32),
                np.asarray([stage_x * 0.34, 1.05, 1.52], dtype=np.float32),
            ),
        }
        targets = phase_targets[phase]
        for index, actor in enumerate(self.floating_actors):
            orbit_angle = self.sim_time * actor.orbit_rate + actor.orbit_phase
            target = targets[index].copy()
            target[0] += np.cos(orbit_angle) * actor.orbit_radius * performance_scale * 0.65
            target[1] += np.sin(orbit_angle * 0.8) * actor.orbit_radius * 0.42
            target[2] += np.sin(orbit_angle * 1.1) * (0.16 if actor.role == "hero" else 0.26)
            actor.center_target = target
        self.phase_name = phase

    def _apply_floating_group_dynamics(self, sample: AudioSample) -> None:
        if len(self.floating_actors) < 2:
            return
        orbital_force = float(self.floating_intensity["orbital_force"]) * (0.9 + sample.rms * 0.5)
        separation_force = float(self.floating_intensity["separation_force"])
        for actor in self.floating_actors:
            metrics = actor.soft_body.get_metrics()
            radial = metrics.center_of_mass - self.camera_focus
            radial[2] = 0.0
            tangent = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), radial)
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                tangent /= tangent_norm
                actor.soft_body.apply_global_force(tangent * orbital_force)
        for index, actor_a in enumerate(self.floating_actors):
            metrics_a = actor_a.soft_body.get_metrics()
            for actor_b in self.floating_actors[index + 1 :]:
                metrics_b = actor_b.soft_body.get_metrics()
                delta = metrics_a.center_of_mass - metrics_b.center_of_mass
                distance = float(np.linalg.norm(delta))
                if distance < 1e-5:
                    continue
                direction = delta / distance
                if distance < 3.35:
                    strength = (3.35 - distance) * separation_force
                    actor_a.soft_body.apply_global_force(direction * strength)
                    actor_b.soft_body.apply_global_force(-direction * strength)
                elif distance > 5.15:
                    drift = (distance - 5.15) * 0.14
                    actor_a.soft_body.apply_global_force(-direction * drift)
                    actor_b.soft_body.apply_global_force(direction * drift)

    def _screen_space_separation(self) -> float:
        if len(self.floating_actors) < 2:
            return 0.0
        centers = [actor.soft_body.get_metrics().center_of_mass for actor in self.floating_actors[:2]]
        projected: list[tuple[float, float]] = []
        for center in centers:
            camera_point = self.base.cam.getRelativePoint(self.base.render, Vec3(float(center[0]), float(center[1]), float(center[2])))
            p2 = Point2()
            if self.base.camLens.project(camera_point, p2):
                projected.append((float(p2[0]), float(p2[1])))
        if len(projected) != 2:
            return 0.0
        ax, ay = projected[0]
        bx, by = projected[1]
        return float(np.hypot(ax - bx, ay - by))

    def _project_point_to_ndc(self, point: np.ndarray) -> tuple[float, float] | None:
        camera_point = self.base.cam.getRelativePoint(self.base.render, Vec3(float(point[0]), float(point[1]), float(point[2])))
        p2 = Point2()
        if self.base.camLens.project(camera_point, p2):
            return (float(p2[0]), float(p2[1]))
        return None

    def _compute_actor_screen_stats(self, actor: FloatingSoftActor) -> dict[str, float]:
        positions = actor.soft_body.get_node_positions()
        if positions.shape[0] > 160:
            stride = max(1, positions.shape[0] // 160)
            sampled = positions[::stride]
        else:
            sampled = positions
        projected = [self._project_point_to_ndc(position) for position in sampled]
        projected_xy = np.asarray([point for point in projected if point is not None], dtype=np.float32)
        center_point = self._project_point_to_ndc(actor.soft_body.get_metrics().center_of_mass)
        if projected_xy.size == 0:
            x_center = 0.0 if center_point is None else center_point[0]
            y_center = 0.0 if center_point is None else center_point[1]
            return {
                "center_x": float(x_center),
                "center_y": float(y_center),
                "radius_x": 0.0,
                "radius_y": 0.0,
                "min_x": float(x_center),
                "max_x": float(x_center),
                "min_y": float(y_center),
                "max_y": float(y_center),
            }
        min_x = float(np.min(projected_xy[:, 0]))
        max_x = float(np.max(projected_xy[:, 0]))
        min_y = float(np.min(projected_xy[:, 1]))
        max_y = float(np.max(projected_xy[:, 1]))
        center_x = float(center_point[0]) if center_point is not None else float((min_x + max_x) * 0.5)
        center_y = float(center_point[1]) if center_point is not None else float((min_y + max_y) * 0.5)
        return {
            "center_x": center_x,
            "center_y": center_y,
            "radius_x": float((max_x - min_x) * 0.5),
            "radius_y": float((max_y - min_y) * 0.5),
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }

    def _update_safe_frame_stats(self) -> None:
        if not self.floating_actors:
            self.current_screen_stats = {}
            self.current_safe_frame_violation_count = 0
            self.current_edge_margin_min = 1.0
            return
        stats: dict[str, dict[str, float]] = {}
        safe_min_x, safe_max_x, safe_min_y, safe_max_y = self.safe_bounds
        violation_count = 0
        edge_margin_min = 1.0
        for actor in self.floating_actors:
            actor_stats = self._compute_actor_screen_stats(actor)
            stats[actor.role] = actor_stats
            bounds = (
                actor_stats["min_x"],
                actor_stats["max_x"],
                actor_stats["min_y"],
                actor_stats["max_y"],
            )
            margin = edge_margin_from_bounds(bounds, self.safe_bounds)
            edge_margin_min = min(edge_margin_min, margin)
            if (
                actor_stats["min_x"] < safe_min_x
                or actor_stats["max_x"] > safe_max_x
                or actor_stats["min_y"] < safe_min_y
                or actor_stats["max_y"] > safe_max_y
            ):
                violation_count += 1
        self.current_screen_stats = stats
        self.current_safe_frame_violation_count = violation_count
        self.safe_frame_violation_total += violation_count
        self.current_edge_margin_min = edge_margin_min

    def _apply_floating_audio(self, actor: FloatingSoftActor, sample: AudioSample) -> None:
        actor.active_event = "idle"
        actor.active_impulses = 0
        self._apply_floating_centering(actor)
        self._apply_floating_boundary_safety(actor)
        self._apply_actor_envelopes(actor)

        breath_target = np.clip((sample.bass - 0.26) * 0.68, -0.12, 0.22)
        breath_delta = (breath_target - actor.bass_state) * (0.24 if actor.role == "hero" else 0.20)
        actor.bass_state += breath_delta
        breath_scale = 1.0 if actor.role == "hero" else 0.55
        actor.soft_body.apply_bass_breath(breath_delta * float(self.floating_intensity["breath_force"]) * actor.response_scale * breath_scale)
        if abs(breath_delta) > 1e-4:
            actor.active_impulses += 1

        if sample.is_beat and self.sim_time >= actor.next_beat_time:
            face_order = ("face_x_neg", "face_y_pos", "face_x_pos", "face_z_neg")
            face_key = face_order[actor.beat_counter % len(face_order)]
            opposite_key = {
                "face_x_neg": "face_x_pos",
                "face_y_pos": "face_y_neg",
                "face_x_pos": "face_x_neg",
                "face_z_neg": "face_z_pos",
            }[face_key]
            direction_map = {
                "face_x_neg": np.asarray([0.95, 0.15, 0.10], dtype=np.float32),
                "face_y_pos": np.asarray([0.0, -1.0, 0.12], dtype=np.float32),
                "face_x_pos": np.asarray([-0.95, -0.18, 0.10], dtype=np.float32),
                "face_z_neg": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            }
            direction = direction_map[face_key] / max(float(np.linalg.norm(direction_map[face_key])), 1e-6)
            beat_strength = np.float32((float(self.floating_intensity["beat_force"]) + sample.bass * 4.2) * actor.response_scale)
            actor.soft_body.apply_region_force(actor.selectors[face_key], direction * beat_strength)
            if actor.role == "hero":
                actor.soft_body.apply_region_force(actor.selectors[opposite_key], -direction * beat_strength * np.float32(float(self.floating_intensity["beat_counter_force"])))
            else:
                actor.soft_body.apply_global_force(-direction * np.float32(beat_strength * 0.22))
            actor.next_beat_time = self.sim_time + 0.30
            actor.beat_counter += 1
            actor.active_event = f"beat:{face_key}"
            actor.active_impulses += 2

        if sample.onset >= 0.56 and self.sim_time >= actor.next_onset_time:
            corner_key = ("corner_a", "corner_c", "corner_b")[actor.beat_counter % 3]
            impulse = np.asarray([1.85, -1.34, 0.92], dtype=np.float32)
            impulse *= np.float32(float(self.floating_intensity["onset_force"]) * (0.62 + sample.onset * 0.42) * actor.response_scale)
            actor.soft_body.apply_region_force(actor.selectors[corner_key], impulse)
            center_counter = np.float32(0.24 if actor.role == "hero" else 0.10)
            actor.soft_body.apply_region_force(actor.selectors["center"], -impulse * center_counter)
            actor.next_onset_time = self.sim_time + 0.26
            actor.active_event = f"onset:{corner_key}"
            actor.active_impulses += 2

        if sample.mid >= 0.42 and self.sim_time >= actor.next_twist_time:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32) if actor.name.endswith("cube") else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            twist_strength = (float(self.floating_intensity["twist_force"]) + sample.mid * 0.66) * actor.response_scale
            actor.soft_body.apply_opposing_twist(axis, twist_strength)
            actor.next_twist_time = self.sim_time + 0.34
            actor.active_event = "mid_twist"
            actor.active_impulses += 1

        if sample.high >= 0.50 and self.sim_time >= actor.next_high_time:
            ripple_key = ("corner_b", "face_z_pos", "corner_c")[actor.beat_counter % 3]
            actor.soft_body.apply_region_force(actor.selectors[ripple_key], np.asarray([0.34, 0.20, 0.28], dtype=np.float32) * np.float32((float(self.floating_intensity["high_force"]) + sample.high * 0.40) * actor.response_scale))
            actor.next_high_time = self.sim_time + 0.12
            if actor.active_event == "idle":
                actor.active_event = f"high:{ripple_key}"
            actor.active_impulses += 1

        if actor.active_event == "idle":
            active_envelopes = [envelope.name for envelope in actor.envelopes if self._envelope_value(envelope, self.sim_time) > 1e-4]
            if active_envelopes:
                actor.active_event = active_envelopes[0]

        self.last_active_event = actor.active_event if actor.active_event != "idle" else self.last_active_event

    def _update_floating_camera(self, sample: AudioSample) -> None:
        if not self.floating_actors:
            return
        centers = np.asarray([actor.soft_body.get_metrics().center_of_mass for actor in self.floating_actors], dtype=np.float32)
        positions = [actor.soft_body.get_node_positions() for actor in self.floating_actors]
        combined = np.vstack(positions)
        hero_center = centers[0]
        group_center = centers.mean(axis=0)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        span = np.maximum(maxs - mins, np.asarray([0.1, 0.1, 0.1], dtype=np.float32))
        target = hero_center * 0.40 + group_center * 0.60
        target[2] = float(np.clip(target[2] + 0.08, 1.20, 2.80))
        self.camera_focus += (target - self.camera_focus) * 0.08
        phase = self._floating_phase(self.sim_time)
        base_distance = {
            "reveal": 8.0,
            "first_strike": 7.3,
            "close_pass": 6.6,
            "twist_arc": 6.9,
            "separation": 7.8,
            "finale": 6.8,
        }[phase]
        vfov = np.deg2rad(34.0)
        hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * self.aspect_ratio)
        width_requirement = (span[0] * 0.78) / max(np.tan(hfov * 0.5) * 0.84, 1e-3)
        height_requirement = (span[2] * 0.86) / max(np.tan(vfov * 0.5) * 0.80, 1e-3)
        depth_requirement = 0.95 * span[1]
        orbit_radius = max(base_distance, width_requirement, height_requirement) + depth_requirement
        orbit_radius += sample.rms * 0.75 + 0.32 * np.sin(self.sim_time * 0.24)
        camera_orbit = float(self.floating_intensity["camera_orbit"])
        x_offset = np.sin(self.sim_time * 0.22) * camera_orbit * (1.05 + span[0] * 0.05)
        y_offset = np.cos(self.sim_time * 0.18) * 0.46
        z_offset = 1.05 + span[2] * 0.24 + np.cos(self.sim_time * 0.24) * 0.20
        self.base.camera.setPos(
            self.camera_focus[0] + x_offset,
            self.camera_focus[1] - orbit_radius + y_offset,
            self.camera_focus[2] + z_offset,
        )
        self.base.camera.lookAt(
            float(self.camera_focus[0]),
            float(self.camera_focus[1] + 0.10 * np.sin(self.sim_time * 0.16)),
            float(self.camera_focus[2] + 0.04 * np.sin(self.sim_time * 0.31)),
        )
        self._update_safe_frame_stats()
        if self.current_edge_margin_min < 0.06:
            correction = (0.06 - self.current_edge_margin_min) * 2.8
            self.base.camera.setY(self.base.render, self.base.camera.getY(self.base.render) - correction)
            self._update_safe_frame_stats()
        elif self.current_edge_margin_min > 0.18:
            tighten = min((self.current_edge_margin_min - 0.18) * 2.1, 0.55)
            self.base.camera.setY(self.base.render, self.base.camera.getY(self.base.render) + tighten)
            self._update_safe_frame_stats()
        self.current_camera_distance = float(np.linalg.norm(np.asarray(self.base.camera.getPos(self.base.render)) - self.camera_focus))

    def _update_floating_background_animation(self, sample: AudioSample) -> None:
        for index, panel in enumerate(self.floating_panels):
            base_x, base_y, base_z = panel.getPythonTag("base_pos")
            offset = np.sin(self.sim_time * (0.18 + index * 0.03) + index) * 0.16
            panel.setPos(base_x + offset, base_y, base_z + 0.08 * np.cos(self.sim_time * 0.22 + index))
            panel.setScale(1.0 + 0.02 * np.sin(self.sim_time * 0.42 + index), 1.0, 1.0 + 0.04 * np.cos(self.sim_time * 0.35 + index))
            panel.setColorScale(1.0 + sample.rms * 0.12, 1.0 + sample.high * 0.10, 1.0 + sample.mid * 0.08, 1.0)
        for index, line in enumerate(self.floating_grid_lines):
            line.setColorScale(1.0, 1.0 + sample.high * 0.08, 1.0 + sample.rms * 0.08, 1.0)

    def _update_floating_overlay(self, sample: AudioSample) -> None:
        if self.overlay is None or not self.floating_actors:
            return
        metrics = self.floating_actors[0].soft_body.get_metrics()
        active_envelopes = [envelope.name for envelope in self.floating_actors[0].envelopes if self._envelope_value(envelope, self.sim_time) > 1e-4]
        lines = [
            "floating softbody",
            f"time: {self.sim_time:05.2f}s",
            f"phase: {self.phase_name}",
            f"event: {self.last_active_event}",
            f"impulses: {self.active_impulse_count}",
            f"hero envelopes: {', '.join(active_envelopes[:3]) if active_envelopes else 'none'}",
            f"rms/bass/mid/high: {sample.rms:0.2f} / {sample.bass:0.2f} / {sample.mid:0.2f} / {sample.high:0.2f}",
            f"volume ratio: {metrics.volume_ratio:0.3f}",
            f"aligned max: {metrics.aligned_max_deformation:0.3f}",
            f"node count: {metrics.node_count}",
            f"preset/intensity: {self.config.softbody_preset} / {self.config.floating_performance_intensity}",
            f"containment corrections: {self.containment_projection_events}",
        ]
        if self.config.framing_debug_overlay:
            hero_stats = self.current_screen_stats.get("hero", {})
            secondary_stats = self.current_screen_stats.get("secondary", {})
            lines.extend(
                [
                    f"frame: {self.config.width}x{self.config.height} aspect={self.aspect_ratio:0.3f}",
                    f"camera distance: {self.current_camera_distance:0.2f}",
                    f"hero screen: {hero_stats.get('center_x', 0.0):0.2f}, {hero_stats.get('center_y', 0.0):0.2f}",
                    f"secondary screen: {secondary_stats.get('center_x', 0.0):0.2f}, {secondary_stats.get('center_y', 0.0):0.2f}",
                    f"safe violations: {self.current_safe_frame_violation_count}",
                    f"edge margin min: {self.current_edge_margin_min:0.3f}",
                ]
            )
        self.overlay.setText("\n".join(lines))

    def _record_floating_metrics(self, dt: float, physics_seconds: float, sample: AudioSample) -> None:
        screen_separation = self._screen_space_separation()
        for actor in self.floating_actors:
            metrics = actor.soft_body.get_metrics()
            active_envelopes = [envelope.name for envelope in actor.envelopes if self._envelope_value(envelope, self.sim_time) > 1e-4]
            envelope_strength = float(sum(self._envelope_value(envelope, self.sim_time) for envelope in actor.envelopes))
            hero_stats = self.current_screen_stats.get("hero", {})
            secondary_stats = self.current_screen_stats.get("secondary", {})
            self.metrics_rows.append(
                {
                    "time": round(self.sim_time, 6),
                    "phase": self.phase_name,
                    "actor": actor.name,
                    "role": actor.role,
                    "preset": self.config.softbody_preset,
                    "intensity": self.config.floating_performance_intensity,
                    "volume_ratio": metrics.volume_ratio,
                    "center_of_mass_x": float(metrics.center_of_mass[0]),
                    "center_of_mass_y": float(metrics.center_of_mass[1]),
                    "center_of_mass_z": float(metrics.center_of_mass[2]),
                    "max_aligned_deformation": metrics.aligned_max_deformation,
                    "rms_aligned_deformation": metrics.aligned_rms_deformation,
                    "max_velocity": metrics.max_velocity,
                    "mean_velocity": metrics.mean_velocity,
                    "twist_angle_proxy": metrics.twist_angle_proxy,
                    "top_region_deformation": metrics.top_region_deformation,
                    "bottom_region_deformation": metrics.bottom_region_deformation,
                    "kinetic_energy_proxy": metrics.kinetic_energy_proxy,
                    "node_count": metrics.node_count,
                    "active_event": actor.active_event,
                    "active_impulses": actor.active_impulses,
                    "active_envelopes": "|".join(active_envelopes),
                    "active_envelope_strength": envelope_strength,
                    "audio_rms": sample.rms,
                    "audio_bass": sample.bass,
                    "audio_mid": sample.mid,
                    "audio_high": sample.high,
                    "containment_projection_events": self.containment_projection_events,
                    "camera_distance": float(np.linalg.norm(np.asarray(self.base.camera.getPos(self.base.render)) - metrics.center_of_mass)),
                    "screen_space_separation": screen_separation,
                    "hero_screen_x": hero_stats.get("center_x", 0.0),
                    "hero_screen_y": hero_stats.get("center_y", 0.0),
                    "hero_screen_radius_x": hero_stats.get("radius_x", 0.0),
                    "hero_screen_radius_y": hero_stats.get("radius_y", 0.0),
                    "secondary_screen_x": secondary_stats.get("center_x", 0.0),
                    "secondary_screen_y": secondary_stats.get("center_y", 0.0),
                    "secondary_screen_radius_x": secondary_stats.get("radius_x", 0.0),
                    "secondary_screen_radius_y": secondary_stats.get("radius_y", 0.0),
                    "safe_frame_violation_count": self.current_safe_frame_violation_count,
                    "safe_frame_violation_total": self.safe_frame_violation_total,
                    "edge_margin_min": self.current_edge_margin_min,
                    "render_width": self.config.width,
                    "render_height": self.config.height,
                    "render_aspect": self.aspect_ratio,
                    "simulation_step_duration_ms": physics_seconds * 1000.0,
                    "dt": dt,
                }
            )

    def reset(self) -> None:
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None
        self.root.removeNode()
        self.__init__(self.base, self.audio_features, self.config)

    def _update_translucent_objects(self) -> None:
        if self.thin_cube_np is not None:
            self.thin_cube_np.setHpr(self.sim_time * 10.0, 8.0 * np.sin(self.sim_time * 0.6), 0.0)
        if self.thick_cube_np is not None:
            self.thick_cube_np.setHpr(self.sim_time * 7.0, 11.0 * np.sin(self.sim_time * 0.5 + 0.4), 0.0)
        if self.torus_np is not None:
            self.torus_np.setHpr(self.sim_time * 13.0, 16.0, 0.0)

    def _stress_phase(self, t: float) -> str:
        if t < 0.8:
            return "rest"
        if t < 1.3:
            return "corner_strike"
        if t < 2.2:
            return "propagation"
        if t < 3.8:
            return "plate_compression"
        if t < 4.8:
            return "release"
        if t < 6.2:
            return "shear"
        if t < 7.6:
            return "twist"
        if t < 8.4:
            return "second_impact"
        return "recovery"

    def _update_plate(self, t: float) -> None:
        if self.plate_np is None:
            return
        if t < 2.2:
            self.plate_np.setZ(5.20)
        elif t < 3.8:
            compress = np.interp(t, [2.2, 3.8], [5.40, 2.00])
            self.plate_np.setZ(compress)
        elif t < 4.8:
            release = np.interp(t, [3.8, 4.8], [2.00, 5.40])
            self.plate_np.setZ(release)
        else:
            self.plate_np.setZ(5.40)

    def _apply_softbody_stress(self, dt: float) -> None:
        if self.soft_body is None:
            return
        t = self.sim_time
        self.phase_name = self._stress_phase(t)
        self.soft_body.set_phase_name(self.phase_name)
        self._update_plate(t)
        self._apply_planar_centering(stiffness=24.0, damping=4.0)

        if self.phase_name == "corner_strike" and not self.projectile_spawned:
            self.projectiles.append(self._create_projectile("corner-projectile", Vec3(3.0, -3.8, 4.0), Vec3(-16.0, 12.0, -2.5)))
            self.soft_body.apply_velocity_to_indices(self.struck_indices, np.asarray([-5.0, 2.8, -2.8], dtype=np.float32))
            self.projectile_spawned = True

        if self.phase_name in {"corner_strike", "propagation"}:
            self.soft_body.apply_spring_anchor(self.bottom_hold_indices, self.bottom_hold_targets, stiffness=12.0, damping=1.8)

        if self.phase_name == "plate_compression":
            self.soft_body.apply_spring_anchor(self.bottom_hold_indices, self.bottom_hold_targets, stiffness=18.0, damping=2.2)
            self.soft_body.apply_force_to_indices(self.top_indices, np.asarray([0.0, 0.0, -28.0], dtype=np.float32))

        if self.phase_name == "shear":
            top_target = self.soft_body.rest_positions[self.top_indices] + np.asarray([1.25, 0.0, 0.0], dtype=np.float32)
            self.soft_body.apply_spring_anchor(self.bottom_hold_indices, self.bottom_hold_targets, stiffness=28.0, damping=3.0)
            self.soft_body.apply_spring_anchor(self.top_indices, top_target, stiffness=13.0, damping=1.8)
            self.soft_body.apply_force_to_indices(self.top_indices, np.asarray([34.0, 0.0, 0.0], dtype=np.float32))

        if self.phase_name == "twist":
            self.soft_body.apply_spring_anchor(self.bottom_hold_indices, self.bottom_hold_targets, stiffness=26.0, damping=3.0)
            top_right = np.intersect1d(self.top_indices, self.right_indices, assume_unique=False)
            top_left = np.intersect1d(self.top_indices, self.left_indices, assume_unique=False)
            self.soft_body.apply_force_to_indices(top_right, np.asarray([0.0, 34.0, 0.0], dtype=np.float32))
            self.soft_body.apply_force_to_indices(top_left, np.asarray([0.0, -34.0, 0.0], dtype=np.float32))

        if self.phase_name == "second_impact" and not self.second_projectile_spawned:
            self.projectiles.append(self._create_projectile("second-projectile", Vec3(-3.0, -3.8, 2.9), Vec3(13.0, 9.5, 1.8)))
            self.soft_body.apply_velocity_to_indices(self.opposite_indices, np.asarray([3.2, 1.9, 1.4], dtype=np.float32))
            self.second_projectile_spawned = True

        if self.phase_name in {"propagation", "release", "recovery"}:
            self.soft_body.apply_force_to_indices(self.opposite_indices, np.asarray([0.0, 0.0, 0.0], dtype=np.float32))

    def _sync_projectiles(self) -> None:
        for body_np, _, visual in self.projectiles:
            visual.setH(visual.getH() + 4.0)
            if body_np.getZ() < -3.0 or abs(body_np.getX()) > 18.0 or abs(body_np.getY()) > 18.0:
                body_np.hide()

    def _record_metrics(self, dt: float, physics_seconds: float) -> None:
        if self.soft_body is None:
            return
        metrics = self.soft_body.get_metrics()
        row = {
            "time": round(self.sim_time, 6),
            "phase": self.phase_name,
            "preset": self.config.softbody_preset,
            "variant": self.soft_body.variant,
            "volume_ratio": metrics.volume_ratio,
            "center_of_mass_x": float(metrics.center_of_mass[0]),
            "center_of_mass_y": float(metrics.center_of_mass[1]),
            "center_of_mass_z": float(metrics.center_of_mass[2]),
            "max_displacement": metrics.aligned_max_deformation,
            "mean_displacement": metrics.aligned_rms_deformation,
            "max_velocity": metrics.max_velocity,
            "mean_velocity": metrics.mean_velocity,
            "bbox_x": float(metrics.bounding_box_dimensions[0]),
            "bbox_y": float(metrics.bounding_box_dimensions[1]),
            "bbox_z": float(metrics.bounding_box_dimensions[2]),
            "top_region_displacement": metrics.top_region_deformation,
            "bottom_region_displacement": metrics.bottom_region_deformation,
            "struck_corner_displacement": metrics.struck_corner_deformation,
            "opposite_corner_displacement": metrics.opposite_corner_deformation,
            "twist_angle_proxy": metrics.twist_angle_proxy,
            "kinetic_energy_proxy": metrics.kinetic_energy_proxy,
            "node_count": metrics.node_count,
            "solver_substeps": 12,
            "simulation_step_duration_ms": physics_seconds * 1000.0,
            "dt": dt,
        }
        self.metrics_rows.append(row)

    def step(self, dt: float, sample: AudioSample) -> None:
        self.sim_time += dt

        if self.is_obstacle_course:
            start = time.perf_counter()
            self.active_impulse_count = 0
            self.last_active_event = "descent"
            self.phase_name = self._course_phase(self.sim_time)
            for actor in self.floating_actors:
                actor.active_impulses = 0
                self._course_release_and_hold(actor)
                self._apply_course_boundary_safety(actor)
                self._apply_course_guidance(actor)
                self._apply_course_audio(actor, sample)
                self._apply_actor_envelopes(actor)
                self._apply_course_contacts(actor)
                self._apply_course_unstuck(actor)
                self.active_impulse_count += actor.active_impulses
            self.world.doPhysics(dt, 12, dt / 12.0)
            physics_seconds = time.perf_counter() - start
            self.course_metrics_cache = {}
            for actor in self.floating_actors:
                actor.soft_body.sync_render_mesh()
                self.course_metrics_cache[actor.name] = actor.soft_body.get_metrics()
            self._update_course_background_animation(sample)
            self._update_course_camera(sample)
            self._update_course_overlay(sample)
            self._record_course_metrics(dt, physics_seconds, sample)
        elif self.is_floating_scene:
            start = time.perf_counter()
            self.last_active_event = "idle"
            self.active_impulse_count = 0
            self._update_floating_targets(sample)
            self._queue_choreography_cues()
            for actor in self.floating_actors:
                delayed_time = max(self.sim_time - actor.delay_seconds, 0.0)
                delayed_sample = self.audio_features.sample(delayed_time)
                self._apply_floating_audio(actor, delayed_sample)
                self.active_impulse_count += actor.active_impulses
            self._apply_floating_group_dynamics(sample)
            self.world.doPhysics(dt, 12, dt / 12.0)
            physics_seconds = time.perf_counter() - start
            for actor in self.floating_actors:
                actor.soft_body.sync_render_mesh()
            self._update_floating_background_animation(sample)
            self._update_floating_camera(sample)
            self._update_floating_overlay(sample)
            self._record_floating_metrics(dt, physics_seconds, sample)
        elif self.is_soft_stress:
            self._apply_softbody_stress(dt)
            start = time.perf_counter()
            self.world.doPhysics(dt, 12, dt / 12.0)
            physics_seconds = time.perf_counter() - start
            self.soft_body.sync_render_mesh()
            self._sync_projectiles()
            self._record_metrics(dt, physics_seconds)
        elif self.config.true_softbody_translucent:
            start = time.perf_counter()
            self.world.doPhysics(dt, 10, dt / 12.0)
            physics_seconds = time.perf_counter() - start
            if self.soft_body is not None:
                self.soft_body.sync_render_mesh()
                self.phase_name = "settle"
                self._record_metrics(dt, physics_seconds)
        else:
            self._update_translucent_objects()

        if self.pipeline is not None:
            self.pipeline.update_camera(self.base.cam)
            self.pipeline.set_time_inputs(self.sim_time)

    def get_metrics(self) -> SoftBodyMetrics | None:
        if self.soft_body is None:
            return None
        return self.soft_body.get_metrics()

    def _course_deformation_targets(self) -> dict[str, float]:
        return {
            "hero-soft": 0.18,
            "coral-cube": 0.12,
            "violet-spring": 0.08,
            "amber-prism": 0.08,
            "teal-octa": 0.06,
        }

    def _build_course_body_summary(self) -> dict[str, object]:
        targets = self._course_deformation_targets()
        bodies: list[dict[str, object]] = []
        for actor in self.floating_actors:
            rows = [row for row in self.metrics_rows if row.get("body_id") == actor.name]
            if not rows:
                continue
            volume_values = [float(row["volume_ratio"]) for row in rows if row.get("volume_ratio") is not None]
            contact_count = max(int(float(row.get("contact_count") or 0)) for row in rows)
            impact_count = max(int(float(row.get("meaningful_impact_count") or 0)) for row in rows)
            unstuck_count = max(int(float(row.get("unstuck_events") or 0)) for row in rows)
            safe_violations = max(int(float(row.get("safe_frame_violations") or 0)) for row in rows)
            bodies.append(
                {
                    "body_id": actor.name,
                    "shape": actor.shape_kind,
                    "color": [float(component) for component in actor.base_color[:3]],
                    "profile": actor.profile_name,
                    "release_time": actor.release_time,
                    "release_trigger": actor.release_trigger,
                    "contact_count": contact_count,
                    "meaningful_impact_count": impact_count,
                    "peak_deformation": actor.peak_deformation,
                    "peak_velocity": actor.peak_velocity,
                    "volume_range": [min(volume_values), max(volume_values)] if volume_values else [None, None],
                    "unstuck_count": unstuck_count,
                    "stayed_visible": actor.visible_frames > max(24, int(self.config.duration * 30 * 0.18)),
                    "visible_frames": actor.visible_frames,
                    "deformation_target": targets.get(actor.name, 0.0),
                    "passed_deformation_target": actor.peak_deformation >= targets.get(actor.name, 0.0),
                    "safe_frame_violations": safe_violations,
                }
            )
        return {
            "audio_path": self.audio_features.source,
            "course_seed": self.config.course_seed,
            "duration": self.config.duration,
            "body_count": len(bodies),
            "bodies": bodies,
        }

    def finalize(self) -> None:
        if self.course_layout_output is not None and self.course_layout is not None:
            self.course_layout_output.parent.mkdir(parents=True, exist_ok=True)
            self.course_layout_output.write_text(json.dumps(self.course_layout.to_dict(), indent=2), encoding="utf-8")
            print(f"[layout] wrote {self.course_layout_output}")
        if self.body_summary_output is not None and self.floating_actors and self.config.softbody_obstacle_course:
            self.body_summary_output.parent.mkdir(parents=True, exist_ok=True)
            self.body_summary_output.write_text(json.dumps(self._build_course_body_summary(), indent=2), encoding="utf-8")
            print(f"[body-summary] wrote {self.body_summary_output}")
        if not self.metrics_rows:
            return
        self.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.metrics_rows[0].keys())
        with self.metrics_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_rows)
        print(f"[metrics] wrote {self.metrics_output}")
