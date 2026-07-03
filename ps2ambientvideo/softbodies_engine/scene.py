from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletBoxShape, BulletDebugNode, BulletPlaneShape, BulletRigidBodyNode, BulletWorld
from panda3d.core import AmbientLight, CardMaker, DirectionalLight, Fog, Material, PointLight, Vec3, Vec4

from .audio_analysis import AudioFeatures, AudioSample
from .gummy_object import GelatinControls, GummyConfig, GummyObject, format_mesh_validation_report, generate_shape_validation_reports


@dataclass
class SceneConfig:
    width: int = 1280
    height: int = 720
    duration: float = 10.0
    seed: int = 42
    show_debug: bool = False
    gelatin_opacity: float = 0.90
    gelatin_cloudiness: float = 0.92
    transmission_strength: float = 0.30
    fresnel_strength: float = 0.08
    specular_strength: float = 0.36
    wobble_strength: float = 1.0
    deformation_strength: float = 1.0
    backlight_strength: float = 1.60
    rear_face_strength: float = 0.01
    thickness_absorption: float = 0.96
    minimum_body_light: float = 0.50
    exposure: float = 1.30
    inner_layer_opacity: float = 0.0
    physics_mode: str = "hybrid"
    material_debug: bool = False
    geometry_debug: bool = False
    deformation_debug: bool = False
    surface_calibration_debug: bool = False
    surface_rest_debug: bool = False
    surface_floating_rest_debug: bool = False
    floating_softbody_scene: bool = False
    softbody_obstacle_course: bool = False
    course_seed: int = 42
    floating_debug_overlay: bool = False
    framing_debug_overlay: bool = False
    floating_performance_intensity: str = "medium"
    true_softbody_debug: bool = False
    true_softbody_stress_debug: bool = False
    tetra_softbody_debug: bool = False
    translucency_debug: bool = False
    true_softbody_translucent: bool = False
    true_softbody_translucent_stress: bool = False
    softbody_preset: str = "stable_medium"
    softbody_visualization: str = "shaded"
    softbody_profiles: tuple[str, ...] = ()
    translucency_view: str = "composite"
    translucency_preset: str = "balanced"
    absorption_color: tuple[float, float, float] = (0.22, 0.08, 0.04)
    absorption_density: float = 1.75
    transmission_gain: float = 1.15
    scattering_strength: float = 0.46
    translucency_cloudiness: float = 0.40
    refraction_strength: float = 0.035
    ior: float = 1.10
    surface_opacity: float = 0.98
    surface_reflection_strength: float = 0.20
    thickness_scale: float = 0.80
    metrics_output: str = ""
    course_layout_output: str = ""
    body_summary_output: str = ""
    plate_speed: float = 0.22
    plate_travel: float = 0.36
    plate_hold: float = 0.90
    plate_clearance: float = 0.70
    minimum_safe_volume_ratio: float = 0.68
    plate_calibration_output: str = "output/surface_plate_calibration.csv"
    impact_calibration_output: str = "output/surface_impact_calibration.csv"
    support_ablation_output: str = "output/surface_support_ablation.csv"
    contact_comparison_output: str = "output/surface_contact_comparison.csv"
    gravity_contact_sweep_output: str = "output/surface_gravity_contact_sweep.csv"
    rest_sweep_output: str = "output/surface_rest_sweep.csv"
    inversion_events_output: str = "output/surface_inversion_events.csv"
    surface_contact_mode: str = "support_spheres"
    show_rest_ghost: bool = False


class GummyScene:
    def __init__(self, base: ShowBase, audio_features: AudioFeatures, config: SceneConfig) -> None:
        self.base = base
        self.audio_features = audio_features
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.sim_time = 0.0
        self.group_focus = np.array([0.0, 0.0, 1.9], dtype=np.float32)
        self.objects: list[GummyObject] = []
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0.0, 0.0, -9.81 if not (config.geometry_debug or config.material_debug or config.deformation_debug) else 0.0))
        self.controls = GelatinControls(
            gelatin_opacity=config.gelatin_opacity,
            gelatin_cloudiness=config.gelatin_cloudiness,
            transmission_strength=config.transmission_strength,
            fresnel_strength=config.fresnel_strength,
            specular_strength=config.specular_strength,
            wobble_strength=config.wobble_strength,
            deformation_strength=config.deformation_strength,
            backlight_strength=config.backlight_strength,
            rear_face_strength=config.rear_face_strength,
            thickness_absorption=config.thickness_absorption,
            minimum_body_light=config.minimum_body_light,
            exposure=config.exposure,
            inner_layer_opacity=config.inner_layer_opacity,
        )

        self.root = self.base.render.attachNewNode("gummy-scene")
        self.base.setBackgroundColor(0.025, 0.024, 0.034, 1.0)
        if config.material_debug:
            self.base.camLens.setFov(40.0)
            self.base.camera.setPos(0.0, -20.5, 5.2)
            self.base.camera.lookAt(0.0, 0.0, 1.9)
        elif config.deformation_debug:
            self.base.camLens.setFov(34.0)
            self.base.camera.setPos(0.0, -13.5, 4.8)
            self.base.camera.lookAt(0.0, 0.0, 1.8)
        elif config.geometry_debug:
            self.base.camLens.setFov(44.0)
            self.base.camera.setPos(0.0, -28.0, 5.8)
            self.base.camera.lookAt(0.0, 0.0, 1.8)
        else:
            self.base.camLens.setFov(36.0)
            self.base.camera.setPos(0.0, -14.6, 5.4)
            self.base.camera.lookAt(0.0, 0.0, 1.8)
        self.base.camLens.setNearFar(0.1, 200.0)

        self._build_background()
        self._build_lights()
        self._build_floor()
        if not (config.geometry_debug or config.material_debug or config.deformation_debug):
            self._build_containment_walls()
        self._spawn_gummies()
        if config.geometry_debug or config.material_debug or config.deformation_debug:
            for report in generate_shape_validation_reports(radius=1.0, seed=config.seed):
                print("[mesh] " + format_mesh_validation_report(report))
        initial_sample = self.audio_features.sample(0.0)
        self._update_transparency_sort()
        self._update_shader_inputs(initial_sample)
        camera_position = self.base.camera.getPos(self.base.render)
        for gummy in self.objects:
            gummy.update_visual(0.0, 0.0, initial_sample, camera_position)

        if config.show_debug:
            debug = BulletDebugNode("bullet-debug")
            debug.showWireframe(True)
            debug.showConstraints(True)
            debug.showBoundingBoxes(False)
            debug.showNormals(False)
            debug_np = self.root.attachNewNode(debug)
            debug_np.show()
            self.world.setDebugNode(debug)

    def _build_background(self) -> None:
        fog = Fog("scene-fog")
        fog.setColor(0.035, 0.036, 0.046)
        fog.setExpDensity(0.006)
        self.base.render.setFog(fog)

        card = CardMaker("backdrop")
        card.setFrame(-28, 28, -12, 20)
        backdrop = self.root.attachNewNode(card.generate())
        backdrop.setP(90)
        backdrop.setPos(0.0, 32.0, 8.5)
        backdrop.setColor(Vec4(0.065, 0.070, 0.092, 1.0))

        side_left = self.root.attachNewNode(card.generate())
        side_left.setP(90)
        side_left.setR(18)
        side_left.setPos(-16.0, 16.0, 7.0)
        side_left.setScale(0.65, 1.0, 1.0)
        side_left.setColor(Vec4(0.040, 0.043, 0.055, 1.0))

        side_right = self.root.attachNewNode(card.generate())
        side_right.setP(90)
        side_right.setR(-14)
        side_right.setPos(14.0, 15.0, 6.5)
        side_right.setScale(0.55, 1.0, 1.0)
        side_right.setColor(Vec4(0.038, 0.041, 0.052, 1.0))

    def _build_lights(self) -> None:
        self.base.render.setShaderAuto()

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.44, 0.44, 0.46, 1.0))
        ambient_np = self.root.attachNewNode(ambient)
        self.base.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor(Vec4(1.44, 1.38, 1.30, 1.0))
        key.setShadowCaster(True, 2048, 2048)
        key.getLens().setFilmSize(30, 24)
        key.getLens().setNearFar(1.0, 60.0)
        key_np = self.root.attachNewNode(key)
        key_np.setPos(-7.0, -7.2, 12.5)
        key_np.lookAt(0.0, 0.0, 1.4)
        self.base.render.setLight(key_np)

        fill = PointLight("fill")
        fill.setColor(Vec4(0.72, 0.66, 0.62, 1.0))
        fill.setAttenuation(Vec3(1.0, 0.0, 0.010))
        fill_np = self.root.attachNewNode(fill)
        fill_np.setPos(-4.5, -4.6, 4.6)
        self.base.render.setLight(fill_np)

        back = PointLight("back")
        back.setColor(Vec4(1.56, 1.28, 1.04, 1.0))
        back.setAttenuation(Vec3(1.0, 0.0, 0.007))
        back_np = self.root.attachNewNode(back)
        back_np.setPos(2.4, 8.6, 6.3)
        self.base.render.setLight(back_np)

        rim = PointLight("rim")
        rim.setColor(Vec4(0.96, 1.12, 1.24, 1.0))
        rim.setAttenuation(Vec3(1.0, 0.0, 0.012))
        rim_np = self.root.attachNewNode(rim)
        rim_np.setPos(5.8, -2.4, 6.4)
        self.base.render.setLight(rim_np)

        self.key_light = key
        self.fill_light = fill
        self.back_light = back
        self.rim_light = rim
        self.key_np = key_np
        self.fill_np = fill_np
        self.back_np = back_np
        self.rim_np = rim_np

    def _build_floor(self) -> None:
        floor_shape = BulletPlaneShape(Vec3(0, 0, 1), 0.0)
        floor_node = BulletRigidBodyNode("floor")
        floor_node.addShape(floor_shape)
        floor_node.setFriction(0.94)
        floor_node.setRestitution(0.08)
        floor_np = self.root.attachNewNode(floor_node)
        floor_np.setPos(0, 0, 0)
        self.world.attachRigidBody(floor_node)

        card = CardMaker("floor-visual")
        card.setFrame(-20, 20, -20, 20)
        visual = floor_np.attachNewNode(card.generate())
        visual.setP(-90)
        visual.setPos(0, 0, 0)
        visual.setColor(0.11, 0.11, 0.12, 1.0)
        visual.setShaderAuto()

        material = Material()
        material.setAmbient(Vec4(0.09, 0.09, 0.10, 1.0))
        material.setDiffuse(Vec4(0.15, 0.15, 0.16, 1.0))
        material.setSpecular(Vec4(0.34, 0.34, 0.36, 1.0))
        material.setShininess(18.0)
        visual.setMaterial(material, 1)

    def _build_containment_walls(self) -> None:
        walls = [
            (Vec3(4.9, 0.0, 2.0), Vec3(0.18, 4.2, 2.8)),
            (Vec3(-4.9, 0.0, 2.0), Vec3(0.18, 4.2, 2.8)),
            (Vec3(0.0, 4.3, 2.0), Vec3(4.9, 0.18, 2.8)),
            (Vec3(0.0, -4.5, 2.0), Vec3(4.9, 0.18, 2.8)),
        ]
        for index, (pos, half_extents) in enumerate(walls):
            node = BulletRigidBodyNode(f"wall-{index}")
            node.addShape(BulletBoxShape(half_extents))
            wall_np = self.root.attachNewNode(node)
            wall_np.setPos(pos)
            self.world.attachRigidBody(node)

    def _spawn_gummies(self) -> None:
        bright_palette = {
            "rounded_cube": (0.18, 0.84, 1.0, 1.0),
            "rectangular_cuboid": (1.0, 0.34, 0.56, 1.0),
            "triangular_prism": (1.0, 0.58, 0.10, 1.0),
            "rounded_tetrahedron": (0.42, 1.0, 0.48, 1.0),
            "rounded_octahedron": (0.58, 0.36, 1.0, 1.0),
            "hexagonal_prism": (1.0, 0.18, 0.68, 1.0),
            "torus_ring": (1.0, 0.84, 0.12, 1.0),
        }

        shape_specs = [
            "rounded_cube",
            "rectangular_cuboid",
            "triangular_prism",
            "rounded_octahedron",
            "torus_ring",
        ]
        if self.config.geometry_debug:
            shape_specs = [
                "rounded_cube",
                "rectangular_cuboid",
                "triangular_prism",
                "rounded_tetrahedron",
                "rounded_octahedron",
                "hexagonal_prism",
                "torus_ring",
            ]
        if self.config.material_debug:
            shape_specs = ["rounded_cube", "rectangular_cuboid", "torus_ring"]
        if self.config.deformation_debug:
            shape_specs = ["rounded_cube"]

        if self.config.geometry_debug:
            positions = [
                (-7.2, 1.6, 1.9),
                (-2.4, 1.6, 1.9),
                (2.4, 1.6, 1.9),
                (7.2, 1.6, 1.9),
                (-4.8, -1.6, 1.9),
                (0.0, -1.6, 1.9),
                (4.8, -1.6, 1.9),
            ]
            radii = [1.02, 1.00, 1.00, 1.02, 1.02, 0.98, 1.04]
        elif self.config.material_debug:
            positions = [(-5.8, 0.0, 1.9), (0.0, 0.0, 1.9), (5.9, 0.0, 1.9)]
            radii = [1.28, 1.34, 1.28]
        elif self.config.deformation_debug:
            positions = [(0.0, 0.0, 1.8)]
            radii = [1.45]
        else:
            positions = [
                (-2.4, -0.2, 2.2),
                (-0.8, 0.5, 2.9),
                (0.9, -0.1, 3.4),
                (2.3, 0.4, 2.6),
                (0.2, -0.9, 3.9),
            ]
            radii = [1.12, 1.14, 1.08, 1.10, 1.06]

        for index, shape_kind in enumerate(shape_specs):
            mass = 0.0 if (self.config.geometry_debug or self.config.material_debug or self.config.deformation_debug) else 1.30 + 0.26 * float(self.rng.random())
            self.objects.append(
                GummyObject(
                    self.root,
                    self.world,
                    self.rng,
                    GummyConfig(
                        name=f"gummy-{index}",
                        radius=radii[index],
                        color=bright_palette[shape_kind],
                        position=positions[index],
                        mass=mass,
                        beat_cooldown=0.28 + 0.02 * (index % 3),
                        onset_cooldown=0.20 + 0.02 * (index % 4),
                        beat_gain=0.80 + 0.12 * float(self.rng.random()),
                        onset_gain=0.88 + 0.18 * float(self.rng.random()),
                        bass_sensitivity=0.24 + 0.06 * float(self.rng.random()),
                        wobble_gain=0.22 + 0.08 * float(self.rng.random()),
                        audio_delay=0.010 * index,
                        shape_kind=shape_kind,
                        shape_seed=int(self.config.seed * 100 + index * 31 + 11),
                    ),
                    self.controls,
                )
            )

    def reset(self) -> None:
        for gummy in self.objects:
            gummy.destroy()
        self.objects.clear()
        self.root.removeNode()
        self.__init__(self.base, self.audio_features, self.config)

    def _update_transparency_sort(self) -> None:
        camera_pos = self.base.camera.getPos(self.base.render)
        distances = []
        for gummy in self.objects:
            distances.append((float((camera_pos - gummy.body_np.getPos(self.base.render)).length()), gummy))
        for distance, gummy in sorted(distances, key=lambda item: item[0], reverse=True):
            gummy.update_sort(distance)

    def _update_shader_inputs(self, sample: AudioSample) -> None:
        camera_position = self.base.camera.getPos(self.base.render)
        key_position = self.key_np.getPos(self.base.render)
        fill_position = self.fill_np.getPos(self.base.render)
        back_position = self.back_np.getPos(self.base.render)

        key_color = Vec3(1.08 + sample.rms * 0.12, 1.04 + sample.rms * 0.10, 1.00 + sample.rms * 0.08)
        fill_color = Vec3(0.42 + sample.mid * 0.08, 0.38 + sample.mid * 0.06, 0.36 + sample.high * 0.04)
        back_strength = self.config.backlight_strength * (1.0 + sample.high * 0.12 + sample.rms * 0.08)
        back_color = Vec3(1.06 * back_strength, 0.88 * back_strength, 0.72 * back_strength)

        for gummy in self.objects:
            for layer in (gummy.outer_layer.node, gummy.inner_layer.node):
                layer.setShaderInput("camera_world_position", camera_position)
                layer.setShaderInput("key_light_position", key_position)
                layer.setShaderInput("fill_light_position", fill_position)
                layer.setShaderInput("back_light_position", back_position)
                layer.setShaderInput("key_light_color", key_color)
                layer.setShaderInput("fill_light_color", fill_color)
                layer.setShaderInput("back_light_color", back_color)

    def _step_debug_showcase(self, dt: float, sample: AudioSample) -> None:
        self.sim_time += dt
        self._update_transparency_sort()
        self._update_shader_inputs(sample)
        camera_position = self.base.camera.getPos(self.base.render)
        for index, gummy in enumerate(self.objects):
            if self.config.deformation_debug:
                gummy.body_np.setPos(0.0, 0.0, 1.85)
                gummy.body_np.setHpr(self.sim_time * 8.0, 7.0 * np.sin(self.sim_time * 0.9), 3.0 * np.cos(self.sim_time * 0.7))
                pulse = 0.5 + 0.5 * np.sin(self.sim_time * 2.4)
                impulse = np.array([0.2, -0.12, 1.0], dtype=np.float32)
                impulse /= np.linalg.norm(impulse)
                gummy.last_impulse_world = Vec3(float(impulse[0]), float(impulse[1]), float(impulse[2]))
                gummy.squash_state = 0.14 * pulse
                gummy.inflate_state = 0.08 * (1.0 - pulse)
                gummy.wobble_energy = 0.18 + 0.12 * pulse
                gummy.ripple_energy = 0.02 + 0.03 * pulse
                gummy.bend_state = np.array([0.06 * np.sin(self.sim_time * 1.8), 0.035 * np.cos(self.sim_time * 1.5)], dtype=np.float32)
                gummy.lag_state = np.array([0.11 * np.sin(self.sim_time * 2.2), -0.05 * np.cos(self.sim_time * 1.7), 0.03 * np.sin(self.sim_time * 2.0)], dtype=np.float32)
            elif self.config.geometry_debug:
                base_positions = [
                    (-7.2, 1.6, 1.9),
                    (-2.4, 1.6, 1.9),
                    (2.4, 1.6, 1.9),
                    (7.2, 1.6, 1.9),
                    (-4.8, -1.6, 1.9),
                    (0.0, -1.6, 1.9),
                    (4.8, -1.6, 1.9),
                ]
                bx, by, bz = base_positions[index]
                gummy.body_np.setPos(bx, by, bz + 0.06 * np.sin(self.sim_time * 0.7 + index * 0.4))
                gummy.body_np.setHpr(self.sim_time * (10.0 + index * 1.2), 8.0 + np.sin(self.sim_time * 0.5 + index) * 4.0, np.cos(self.sim_time * 0.45 + index) * 2.0)
            else:
                base_positions = [(-5.8, 0.0, 1.9), (0.0, 0.0, 1.9), (5.9, 0.0, 1.9)]
                bx, by, bz = base_positions[index]
                gummy.body_np.setPos(bx, by, bz + 0.06 * np.sin(self.sim_time * 0.65 + index * 0.4))
                gummy.body_np.setHpr(self.sim_time * (9.0 + index * 1.8), 7.0 + np.sin(self.sim_time * 0.5 + index) * 3.0, np.cos(self.sim_time * 0.4 + index) * 1.6)
            gummy.wobble_energy = 0.05 + sample.bass * 0.05
            gummy.ripple_energy = 0.01 + sample.high * 0.01
            gummy.inflate_state = 0.03 + sample.bass * 0.02
            if self.config.deformation_debug:
                gummy.inflate_state = max(gummy.inflate_state, 0.05)
            gummy.update_visual(dt, self.sim_time, sample, camera_position)

    def _update_camera_focus(self, sample: AudioSample) -> None:
        positions = np.asarray([[g.body_np.getX(), g.body_np.getY(), g.body_np.getZ()] for g in self.objects], dtype=np.float32)
        center = positions.mean(axis=0)
        target = np.array([center[0] * 0.18, center[1] * 0.04, np.clip(center[2] - 1.7, 1.8, 2.7)], dtype=np.float32)
        self.group_focus += (target - self.group_focus) * 0.04
        drift = 0.08 + sample.rms * 0.05
        self.base.camera.setPos(
            self.group_focus[0] + np.sin(self.sim_time * 0.18) * drift,
            -14.6 + self.group_focus[1] * 0.05 + np.cos(self.sim_time * 0.14) * 0.10,
            5.4 + np.cos(self.sim_time * 0.22) * 0.08,
        )
        self.base.camera.lookAt(self.group_focus[0], self.group_focus[1], self.group_focus[2])

    def step(self, dt: float, sample: AudioSample) -> None:
        if self.config.geometry_debug or self.config.material_debug or self.config.deformation_debug:
            self._step_debug_showcase(dt, sample)
            return

        self.sim_time += dt
        delayed_samples = [self.audio_features.sample(max(self.sim_time - gummy.config.audio_delay, 0.0)) for gummy in self.objects]
        for gummy, delayed_sample in zip(self.objects, delayed_samples, strict=True):
            gummy.apply_music(self.sim_time, delayed_sample, dt)
            position = gummy.body_np.getPos()
            height_bias = max(0.0, 2.4 - position.z) * 0.012
            gummy.body.applyCentralImpulse(Vec3(-position.x * 0.0035, -position.y * 0.0025, height_bias))

        self.world.doPhysics(dt, 1, dt)
        self._update_transparency_sort()
        self._update_shader_inputs(sample)
        camera_position = self.base.camera.getPos(self.base.render)
        for gummy, delayed_sample in zip(self.objects, delayed_samples, strict=True):
            gummy.update_visual(dt, self.sim_time, delayed_sample, camera_position)

        light_boost = 1.12 + sample.rms * 0.14
        self.key_light.setColor(Vec4(1.22 * light_boost, 1.16 * light_boost, 1.08 * light_boost, 1.0))
        self.fill_light.setColor(Vec4(0.58 + sample.mid * 0.06, 0.52 + sample.mid * 0.05, 0.48 + sample.high * 0.04, 1.0))
        back_boost = self.config.backlight_strength * (1.08 + sample.high * 0.14 + sample.rms * 0.08)
        self.back_light.setColor(Vec4(1.06 * back_boost, 0.90 * back_boost, 0.74 * back_boost, 1.0))
        self.rim_light.setColor(Vec4(0.96 + sample.high * 0.06, 1.08 + sample.high * 0.07, 1.22 + sample.high * 0.07, 1.0))
        self._update_camera_focus(sample)
