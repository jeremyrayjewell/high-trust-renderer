from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .. import ps2fx
from ..mode_manifest import get_mode_meta
from ..palettes import Palette
from ..worlds_grammar import render_world_scene


@dataclass(frozen=True)
class ModeConfig:
    name: str
    scene: str
    palette_hint: str
    speed: float = 1.0
    density: float = 1.0
    structure: float = 1.0
    atmosphere: float = 1.0
    family: str = ""
    scene_recipe: str = ""
    dominant_geometry: tuple[str, ...] = ()
    movement_type: tuple[str, ...] = ()
    camera_roles: tuple[str, ...] = ()
    suitable_presets: tuple[str, ...] = ()


class BaseMode:
    config: ModeConfig

    def __init__(self, config: ModeConfig) -> None:
        self.config = config
        self._current_segment = None
        self._scene_grammar = "legacy_plaza"

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def meta(self):
        return get_mode_meta(self.name)

    def set_context(self, segment) -> None:
        self._current_segment = segment

    def set_scene_grammar(self, scene_grammar: str) -> None:
        self._scene_grammar = scene_grammar

    def render(
        self,
        frame: np.ndarray,
        t: float,
        audio_features: dict[str, float],
        weight: float,
        palette: Palette,
        rng: np.random.Generator,
    ) -> None:
        if weight <= 0.001:
            return
        layer = np.zeros_like(frame)
        if self._scene_grammar == "worlds":
            render_world_scene(self, layer, t, audio_features, palette)
            frame[:] = ps2fx.alpha_blend(frame, layer, weight)
            return
        scene = self.config.scene
        if scene == "highway":
            self._render_highway(layer, t, audio_features, palette)
        elif scene == "atrium":
            self._render_atrium(layer, t, audio_features, palette)
        elif scene == "submerged":
            self._render_submerged(layer, t, audio_features, palette)
        elif scene == "transit":
            self._render_transit(layer, t, audio_features, palette)
        elif scene == "corridor":
            self._render_corridor(layer, t, audio_features, palette)
        elif scene == "aquarium":
            self._render_aquarium(layer, t, audio_features, palette)
        elif scene == "shrine":
            self._render_shrine(layer, t, audio_features, palette)
        elif scene == "city":
            self._render_city(layer, t, audio_features, palette)
        elif scene == "chrome":
            self._render_chrome(layer, t, audio_features, palette)
        elif scene == "space":
            self._render_space(layer, t, audio_features, palette)
        elif scene == "plaza":
            self._render_plaza(layer, t, audio_features, palette)
        elif scene == "map":
            self._render_map(layer, t, audio_features, palette)
        elif scene == "garden":
            self._render_garden(layer, t, audio_features, palette)
        else:
            self._render_geometry(layer, t, audio_features, palette)
        self._render_variant_scene_additions(layer, t, audio_features, palette)
        frame[:] = ps2fx.alpha_blend(frame, layer, weight)

    def _horizon(self, frame: np.ndarray, bias: float = 0.0) -> int:
        return int(frame.shape[0] * (0.52 + 0.08 * (1.0 - self.config.structure) + bias))

    def _is(self, *names: str) -> bool:
        return self.name in names

    def _shot_role(self) -> str:
        return getattr(self._current_segment, "shot_role", "forward_travel")

    def _segment_progress(self, t: float) -> float:
        segment = self._current_segment
        if segment is None:
            return 0.0
        span = max(0.001, float(segment.end - segment.start))
        return float(np.clip((t - float(segment.start)) / span, 0.0, 1.0))

    def _camera(self, t: float, *, dolly: float = 0.0, bob: float = 0.05, orbit: float = 0.0, f: dict[str, float] | None = None):
        features = f or {}
        role = self._shot_role()
        progress = self._segment_progress(t)
        bass_push = features.get("bass", 0.0) * 0.18 + features.get("beat", 0.0) * 0.06
        mids_sway = features.get("mids", 0.0) * 0.08
        camera = ps2fx.camera_motion(t, dolly=dolly, bob=bob, orbit=orbit)
        x, y, z, yaw, pitch, fov = camera.x, camera.y, camera.z, camera.yaw, camera.pitch, camera.fov
        if role == "wide_establishing":
            x *= 0.72
            y += 0.12
            z -= bass_push * 0.1
            x += (progress - 0.5) * 0.16
            yaw *= 1.2
            pitch -= 0.045
            fov = 1.12
        elif role == "low_angle_corridor":
            y -= 0.2
            z += bass_push * 0.2
            x += math.sin(progress * math.pi) * 0.06
            yaw *= 0.8
            pitch += 0.03
            fov = 1.1
        elif role == "side_parallax":
            x += math.sin(t * 0.42) * (0.22 + mids_sway) + (progress - 0.5) * 0.22
            z *= 0.82
            yaw += math.sin(t * 0.22) * 0.05 + mids_sway * 0.3
            pitch -= 0.018
            fov = 1.08
        elif role == "close_foreground_pass":
            x += math.sin(t * 0.36) * 0.18 + (progress - 0.5) * 0.28
            y -= 0.06
            z += bass_push * 0.28
            yaw += 0.03 + math.sin(t * 0.24) * 0.04
            pitch += 0.012
            fov = 1.16
        elif role == "calm_horizon_hold":
            x *= 0.5
            y += 0.05
            z *= 0.45
            yaw *= 0.55
            yaw += math.sin(progress * math.pi) * 0.02
            pitch -= 0.03
            fov = 1.02
        elif role == "wide_crane_rise":
            x *= 0.6
            y += 0.22 + progress * 0.22
            z -= bass_push * 0.08
            yaw += (progress - 0.5) * 0.08
            pitch -= 0.08
            fov = 1.18
        elif role == "slow_orbit_around_object":
            x += math.sin(t * 0.18 + progress * math.tau) * (0.32 + mids_sway)
            z *= 0.28
            yaw += 0.12 + math.sin(t * 0.18 + progress * math.tau) * 0.18
            pitch -= 0.02
            fov = 1.1
        elif role == "side_tracking_shot":
            x += (progress - 0.5) * 0.44 + math.sin(t * 0.2) * 0.08
            z *= 0.78
            yaw += 0.08
            pitch -= 0.015
            fov = 1.08
        elif role == "diagonal_dolly":
            x += (progress - 0.5) * 0.24
            y += (progress - 0.5) * 0.08
            z += bass_push * 0.24
            yaw += 0.06
            pitch -= 0.012
            fov = 1.12
        elif role == "reverse_pullback":
            x *= 0.72
            z -= progress * 0.55
            yaw -= 0.05 + math.sin(t * 0.16) * 0.03
            pitch -= 0.02
            fov = 1.16
        elif role == "low_flythrough":
            y -= 0.26
            z += bass_push * 0.32
            x += math.sin(t * 0.28) * 0.06
            pitch += 0.04
            fov = 1.15
        elif role == "high_overlook_drift":
            x += math.sin(t * 0.18) * 0.12
            y += 0.28
            z *= 0.34
            yaw += (progress - 0.5) * 0.1
            pitch -= 0.12
            fov = 1.2
        elif role == "spiral_approach":
            x += math.sin(progress * math.tau) * 0.22
            y += math.cos(progress * math.tau) * 0.08
            z += bass_push * 0.18
            yaw += 0.16 + progress * 0.12
            pitch -= 0.01
            fov = 1.12
        elif role == "long_horizon_glide":
            x += (progress - 0.5) * 0.36
            y += 0.08
            z *= 0.18
            yaw += math.sin(t * 0.12) * 0.03
            pitch -= 0.05
            fov = 1.04
        elif role == "vertical_elevator_rise":
            y += progress * 0.4 - 0.12
            z *= 0.22
            yaw += math.sin(t * 0.1) * 0.025
            pitch -= 0.08
            fov = 1.1
        elif role == "crossing_parallax_pan":
            x += (progress - 0.5) * 0.52
            z *= 0.62
            yaw += 0.1 + math.sin(t * 0.22) * 0.04
            pitch -= 0.025
            fov = 1.1
        elif role == "sweeping_crane_rise":
            x += math.sin(t * 0.22) * 0.12 + (progress - 0.5) * 0.18
            y += 0.3 + progress * 0.4
            z -= progress * 0.12
            yaw += 0.08 + (progress - 0.5) * 0.12
            pitch -= 0.12
            fov = 1.22
        elif role == "wide_orbit_core":
            x += math.sin(t * 0.16 + progress * math.tau) * (0.48 + mids_sway)
            z *= 0.22
            yaw += 0.16 + math.sin(t * 0.16 + progress * math.tau) * 0.24
            pitch -= 0.04
            fov = 1.16
        elif role == "fast_forward_flythrough":
            y -= 0.14
            z += 0.18 + bass_push * 0.48 + progress * 0.12
            x += math.sin(t * 0.34) * 0.08
            yaw += 0.03
            pitch += 0.018
            fov = 1.18
        elif role == "low_reflective_glide":
            y -= 0.22
            x += (progress - 0.5) * 0.16
            z += bass_push * 0.24
            yaw += math.sin(t * 0.18) * 0.03
            pitch += 0.028
            fov = 1.12
        elif role == "side_tracking_towers":
            x += (progress - 0.5) * 0.68 + math.sin(t * 0.16) * 0.08
            y += 0.06
            z *= 0.56
            yaw += 0.12
            pitch -= 0.04
            fov = 1.16
        elif role == "diagonal_glass_dolly":
            x += (progress - 0.5) * 0.34
            y += 0.08 + (progress - 0.5) * 0.12
            z += bass_push * 0.3
            yaw += 0.08
            pitch -= 0.04
            fov = 1.14
        elif role == "pullback_to_city":
            x *= 0.55
            y += 0.16
            z -= progress * 0.7
            yaw -= 0.08 + math.sin(t * 0.14) * 0.02
            pitch -= 0.06
            fov = 1.2
        elif role == "horizon_water_drift":
            x += (progress - 0.5) * 0.48
            y += 0.1
            z *= 0.14
            yaw += math.sin(t * 0.1) * 0.04
            pitch -= 0.06
            fov = 1.05
        elif role == "dive_to_fountain":
            x += math.sin(t * 0.2) * 0.1
            y += 0.34 - progress * 0.44
            z += 0.14 + bass_push * 0.26 + progress * 0.1
            yaw += (progress - 0.5) * 0.08
            pitch += 0.08 - progress * 0.14
            fov = 1.2
        elif role == "pullback_people_skyline":
            x *= 0.42
            y += 0.12
            z -= progress * 0.82
            yaw -= 0.04
            pitch -= 0.08
            fov = 1.18
        elif role == "sweeping_panorama_deck":
            x += (progress - 0.5) * 0.7 + math.sin(t * 0.16) * 0.1
            y += 0.18 + math.sin(progress * math.pi) * 0.08
            z *= 0.22
            yaw += 0.16 + (progress - 0.5) * 0.16
            pitch -= 0.12
            fov = 1.24
        elif role == "orbit_subject":
            x += math.sin(progress * math.tau + t * 0.18) * (0.56 + mids_sway * 1.2)
            y += 0.08 + math.sin(progress * math.pi) * 0.06
            z *= 0.18
            yaw += 0.18 + math.cos(progress * math.tau + t * 0.18) * 0.26
            pitch -= 0.04
            fov = 1.18
        elif role == "sweeping_arc":
            x += (progress - 0.5) * 0.82 + math.sin(t * 0.2) * 0.14
            y += 0.16 + math.sin(progress * math.pi) * 0.12
            z -= progress * 0.12
            yaw += 0.14 + (progress - 0.5) * 0.22
            pitch -= 0.09
            fov = 1.22
        elif role == "low_orbit_pass":
            x += math.sin(progress * math.tau) * 0.42 + (progress - 0.5) * 0.18
            y -= 0.22
            z += bass_push * 0.28
            yaw += 0.14 + math.cos(progress * math.tau) * 0.2
            pitch += 0.03
            fov = 1.16
        elif role == "rising_spiral":
            x += math.sin(progress * math.tau * 1.2) * 0.36
            y += progress * 0.36 - 0.04
            z *= 0.16
            yaw += 0.12 + progress * 0.28
            pitch -= 0.1 + progress * 0.02
            fov = 1.18
        elif role == "side_orbit_reveal":
            x += (progress - 0.5) * 0.58 + math.sin(progress * math.tau) * 0.24
            y += 0.06
            z *= 0.34
            yaw += 0.18 + math.sin(progress * math.tau) * 0.18
            pitch -= 0.05
            fov = 1.17
        elif role == "water_surface_flyover":
            x += math.sin(t * 0.16) * 0.08 + (progress - 0.5) * 0.22
            y -= 0.18
            z += 0.12 + bass_push * 0.22
            yaw += math.sin(progress * math.pi) * 0.08
            pitch += 0.015
            fov = 1.14
        else:
            z += bass_push * 0.18
            x += (progress - 0.5) * 0.1
            yaw += math.sin(t * 0.2) * mids_sway * 0.2
        camera = ps2fx.Camera3D(x=x, y=y, z=z, yaw=yaw, pitch=pitch, fov=fov)
        ps2fx.debug_set_camera(camera.z)
        return camera

    def _draw_tower_band(self, frame: np.ndarray, palette: Palette, t: float, levels: tuple[float, ...], signs: bool = True) -> None:
        h, w = frame.shape[:2]
        for layer_idx, scale in enumerate(levels):
            base_y = int(h * (0.36 + layer_idx * 0.14))
            color = ps2fx.lerp_color(palette.bg, palette.ui, 0.16 + layer_idx * 0.18)
            for i in range(12):
                x = int((i / 12.0) * w + math.sin(t * 0.18 * (layer_idx + 1) + i * 0.6) * 12)
                bw = int(18 + scale * 42 + (i % 4) * 10)
                bh = int(h * (0.12 + scale * 0.2 + ((i + layer_idx) % 3) * 0.05))
                cv2.rectangle(frame, (x, base_y - bh), (x + bw, base_y), ps2fx.bgr(color), -1)
                for row in range(8, bh, 16):
                    cv2.line(frame, (x + 4, base_y - row), (x + bw - 4, base_y - row), ps2fx.bgr(palette.accent), 1)
                if signs and i % 3 == 0:
                    sy = max(8, base_y - bh + 10 + (i % 2) * 12)
                    cv2.rectangle(frame, (x + 3, sy), (x + min(bw - 3, 22), sy + 8), ps2fx.bgr(palette.glow), -1)

    def _draw_columns(self, frame: np.ndarray, palette: Palette, x_positions: list[int], top: int, bottom: int, width: int = 8) -> None:
        for x in x_positions:
            cv2.rectangle(frame, (x - width // 2, top), (x + width // 2, bottom), ps2fx.bgr(palette.ui), -1)
            cv2.line(frame, (x - width // 2, top), (x + width // 2, top), ps2fx.bgr(palette.glow), 1)

    def _draw_rain(self, frame: np.ndarray, palette: Palette, t: float, amount: float) -> None:
        h, w = frame.shape[:2]
        layer = np.zeros_like(frame)
        count = max(18, int(80 * amount))
        for i in range(count):
            x = int((i * 23 + t * 40) % (w + 40)) - 20
            y = int((i * 17 + t * 70) % (h + 60)) - 30
            cv2.line(layer, (x, y), (x - 8, y + 18), ps2fx.bgr(palette.ui), 1, lineType=cv2.LINE_AA)
        frame[:] = ps2fx.additive_blend(frame, layer, 0.26)

    def _render_variant_scene_additions(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        if not self.config.scene_recipe:
            return
        if self.name == self.config.scene_recipe and self.name in {
            "endless_highway",
            "glass_mall_atrium",
            "submerged_city",
            "memory_train",
            "ring_corridor",
            "digital_aquarium",
            "cyber_shrine",
            "airport_at_night",
            "skybridge_city",
            "data_center_dream",
            "liquid_chrome_room",
            "bios_temple",
            "solar_sail_space",
            "neon_rain_alley",
            "dream_office_lobby",
            "arcology_exterior",
            "soft_geometry_field",
            "vapor_plaza",
            "crt_navigation_map",
            "moon_pool",
            "polygon_garden",
            "virtual_hotel",
            "dream_metro",
            "hologram_market",
            "blue_os_desktop_world",
            "weather_simulation_room",
            "crystal_server_lake",
            "tunnel_of_screens",
            "rooftop_antenna_field",
            "fractal_parking_garage",
            "ocean_interface",
            "green_wireframe_valley",
            "neon_monorail",
            "dream_arcade_floor",
            "satellite_weather_map",
            "corporate_fountain_core",
            "memory_beach",
        }:
            return

        family = self.config.family or self.meta.family
        camera = self._variant_camera(t, f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.12)
        if family == "transit":
            self._render_transit_variant(frame, camera, t, f, palette, fog)
        elif family == "cyber_urban":
            self._render_city_variant(frame, camera, t, f, palette, fog)
        elif family == "water_glass":
            self._render_water_variant(frame, camera, t, f, palette, fog)
        elif family == "ui_computer":
            self._render_ui_variant(frame, camera, t, f, palette, fog)
        elif family in {"nature_weather", "space_cosmic"}:
            self._render_nature_variant(frame, camera, t, f, palette, fog)
        elif family == "architecture_interior":
            self._render_architecture_variant(frame, camera, t, f, palette, fog)
        elif family == "abstract_geometry":
            self._render_abstract_variant(frame, camera, t, f, palette, fog)

    def _variant_camera(self, t: float, f: dict[str, float]):
        scene = self.config.scene
        if scene in {"highway", "transit", "corridor"}:
            return self._camera(t * self.config.speed, dolly=t * (0.42 + f["bass"] * 0.22), bob=0.025, orbit=0.24 + f["section"] * 0.18, f=f)
        if scene in {"city", "plaza"}:
            return self._camera(t * 0.5 * self.config.speed, dolly=t * (0.12 + f["bass"] * 0.06), bob=0.018, orbit=0.42 + f["section"] * 0.22, f=f)
        if scene in {"atrium", "aquarium", "submerged", "chrome"}:
            return self._camera(t * 0.42 * self.config.speed, dolly=t * (0.08 + f["bass"] * 0.04), bob=0.018, orbit=0.56 + f["section"] * 0.28, f=f)
        return self._camera(t * 0.46 * self.config.speed, dolly=t * (0.1 + f["bass"] * 0.05), bob=0.018, orbit=0.38 + f["section"] * 0.24, f=f)

    def _recipe_tags(self) -> set[str]:
        return set(self.config.scene_recipe.split("_"))

    def _motion_offset(self, motion: tuple[str, ...], index: int, t: float) -> tuple[float, float, float]:
        x = y = z = 0.0
        if "lateral_drift" in motion or "parallax" in motion:
            x += math.sin(t * 0.18 + index * 0.6) * 0.22
        if "long_glide" in motion:
            x += math.sin(t * 0.08 + index * 0.4) * 0.3
        if "rise_fall" in motion:
            y += math.sin(t * 0.22 + index * 0.5) * 0.14
        if "depth_pass" in motion or "slide_past" in motion:
            z -= (t * 0.55 + index * 0.72) % 2.4
        if "undulate" in motion:
            y += math.sin(t * 0.28 + index * 0.9) * 0.08
        if "sway" in motion:
            x += math.cos(t * 0.24 + index * 0.7) * 0.1
        if "spiral" in motion:
            x += math.sin(t * 0.3 + index) * 0.14
            y += math.cos(t * 0.3 + index) * 0.08
        if "orbit" in motion:
            x += math.sin(t * 0.16 + index * 0.8) * 0.18
        if "jet_pulse" in motion:
            y += max(0.0, math.sin(t * 0.9 + index * 0.4)) * 0.22
        return x, y, z

    def _draw_open_sky(
        self,
        frame: np.ndarray,
        palette: Palette,
        t: float,
        *,
        warm: float = 0.0,
        sun_strength: float = 0.8,
        cloud_density: float = 1.0,
    ) -> None:
        h, w = frame.shape[:2]
        horizon = int(h * 0.58)
        top = ps2fx.lerp_color((106, 176, 236), palette.glow, 0.42)
        bottom = ps2fx.lerp_color((94, 198, 208), palette.mid, 0.34)
        if warm > 0.001:
            top = ps2fx.lerp_color(top, (255, 222, 184), warm * 0.72)
            bottom = ps2fx.lerp_color(bottom, (214, 218, 174), warm * 0.42)
        cv2.rectangle(frame, (0, 0), (w, horizon), ps2fx.bgr(top), -1)
        for y in range(horizon):
            blend = y / max(1, horizon - 1)
            color = ps2fx.lerp_color(top, bottom, blend)
            cv2.line(frame, (0, y), (w, y), ps2fx.bgr(color), 1)
        sun_center = (int(w * (0.78 + math.sin(t * 0.02) * 0.03)), int(h * 0.18))
        ps2fx.draw_glow_circle(frame, sun_center, int(42 + sun_strength * 14), palette.glow, 0.3 + sun_strength * 0.12)
        cv2.circle(frame, sun_center, int(18 + sun_strength * 6), ps2fx.bgr(ps2fx.lerp_color((255, 240, 198), palette.glow, 0.12)), -1, lineType=cv2.LINE_AA)
        cloud_count = max(4, int(7 * cloud_density))
        for idx in range(cloud_count):
            cx = int(w * (0.1 + idx * 0.13) + math.sin(t * 0.06 + idx) * 22)
            cy = int(h * (0.14 + (idx % 3) * 0.08) + math.cos(t * 0.05 + idx * 1.2) * 8)
            rx = int(36 + (idx % 4) * 14)
            ry = int(14 + (idx % 3) * 6)
            color = ps2fx.lerp_color((214, 236, 244), palette.glow, 0.12)
            for part in range(3):
                ox = int((part - 1) * rx * 0.55)
                oy = int(math.sin(idx + part) * 3)
                cv2.ellipse(frame, (cx + ox, cy + oy), (max(10, rx - part * 6), max(6, ry - part * 2)), 0, 0, 360, ps2fx.bgr(color), -1, lineType=cv2.LINE_AA)

    def _draw_grass_plane(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        *,
        path: bool = True,
        near_z: float | None = None,
        far_z: float | None = None,
    ) -> None:
        base_near = camera.z + 1.0 if near_z is None else near_z
        base_far = camera.z + 15.5 if far_z is None else far_z
        grass = ps2fx.lerp_color((72, 148, 86), palette.mid, 0.12)
        grass_edge = ps2fx.lerp_color((126, 208, 132), palette.glow, 0.08)
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [(-6.4, -1.18, base_near), (6.4, -1.18, base_near), (8.0, -1.18, base_far), (-8.0, -1.18, base_far)],
            grass,
            grass_edge,
            fog,
            1.0,
        )
        for x in np.linspace(-5.8, 5.8, 11):
            blade_color = ps2fx.lerp_color(grass_edge, palette.glow, 0.12)
            ps2fx.draw_billboard_3d(frame, camera, (x, -0.62 + math.sin(t * 0.2 + x) * 0.02, camera.z + 2.2 + (x % 2.0) * 0.25), (0.45, 0.35), blade_color, grass_edge, fog, 0.35)
        if path:
            ps2fx.draw_quad_3d(
                frame,
                camera,
                [(-0.95, -1.14, base_near + 0.2), (0.95, -1.14, base_near + 0.2), (2.2, -1.14, base_far), (-2.2, -1.14, base_far)],
                (186, 198, 182),
                (214, 226, 202),
                fog,
                0.92,
            )

    def _draw_tree_cluster(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        positions: list[tuple[float, float, float]],
    ) -> None:
        for idx, (x, y, z) in enumerate(positions):
            sway = math.sin(t * 0.22 + idx * 0.7) * 0.06
            ps2fx.draw_cuboid_3d(frame, camera, (x, y - 0.1, z), (0.12, 0.65, 0.12), (116, 92, 74), fog, edge=palette.ui)
            ps2fx.draw_cuboid_3d(frame, camera, (x + sway, y + 0.42, z + 0.05), (0.72, 0.7, 0.72), ps2fx.lerp_color((84, 174, 108), palette.mid, 0.22), fog, edge=palette.glow)
            ps2fx.draw_cuboid_3d(frame, camera, (x - sway * 0.5, y + 0.68, z + 0.08), (0.4, 0.34, 0.4), ps2fx.lerp_color((126, 214, 146), palette.glow, 0.12), fog, edge=palette.ui)

    def _draw_water_feature(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        f: dict[str, float],
        *,
        center_z: float,
        width: float = 2.6,
        depth: float = 2.2,
        splash_scale: float = 1.0,
    ) -> None:
        pool_fill = ps2fx.lerp_color((82, 164, 208), palette.mid, 0.16)
        pool_edge = ps2fx.lerp_color((188, 232, 248), palette.glow, 0.08)
        ps2fx.draw_quad_3d(
            frame,
            camera,
            [(-width, -1.04, center_z), (width, -1.04, center_z), (depth, -1.04, center_z + depth), (-depth, -1.04, center_z + depth)],
            pool_fill,
            pool_edge,
            fog,
            0.9,
        )
        for ring_idx in range(3):
            radius = 0.8 + ring_idx * 0.38 + f["bass"] * 0.12
            ps2fx.draw_portal_ring_3d(frame, camera, (0.0, -0.95, center_z + 0.82 + ring_idx * 0.34), radius, palette.glow, fog, y_scale=0.16, thickness=2)
        jet_count = 5
        for jet_idx in range(jet_count):
            x = (-0.9 + jet_idx * 0.45) * splash_scale
            pulse = 0.55 + max(0.0, math.sin(t * 1.3 + jet_idx * 0.6)) * 0.55 + f["beat"] * 0.28
            jet_height = 0.9 + pulse * 0.95
            ps2fx.draw_billboard_3d(frame, camera, (x, -0.35 + jet_height * 0.22, center_z + 0.7 + jet_idx * 0.08), (0.1, jet_height), palette.glow, palette.ui, fog, 0.95)
            tip_z = center_z + 0.7 + jet_idx * 0.08
            for droplet_idx in range(3):
                droplet_y = -0.1 + jet_height * (0.46 + droplet_idx * 0.16)
                droplet_x = x + math.sin(t * 0.8 + jet_idx + droplet_idx) * 0.12
                ps2fx.draw_billboard_3d(frame, camera, (droplet_x, droplet_y, tip_z + droplet_idx * 0.05), (0.1, 0.1), palette.accent, palette.glow, fog, 0.9)

    @staticmethod
    def _segment_quad_2d(a: tuple[int, int], b: tuple[int, int], width_a: float, width_b: float) -> list[tuple[int, int]]:
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        length = max(1.0, math.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length
        return [
            (int(a[0] + nx * width_a), int(a[1] + ny * width_a)),
            (int(a[0] - nx * width_a), int(a[1] - ny * width_a)),
            (int(b[0] - nx * width_b), int(b[1] - ny * width_b)),
            (int(b[0] + nx * width_b), int(b[1] + ny * width_b)),
        ]

    def _draw_polygon_person(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        f: dict[str, float],
        center: tuple[float, float, float],
        scale: float,
        idx: int,
        *,
        walking: bool = False,
    ) -> bool:
        h, w = frame.shape[:2]
        cx, cy, cz = center
        beat_lift = f["beat"] * 0.22 + f["onset"] * 0.08
        sway = math.sin(t * (1.0 if walking else 1.8) + idx * 0.7) * 0.12 * scale
        bounce = max(0.0, math.sin(t * 1.4 + idx * 0.9)) * 0.18 * scale + beat_lift * 0.14 * scale
        step = math.sin(t * (1.3 if walking else 2.2) + idx * 0.8) * 0.18 * scale
        arm_raise = (0.08 if walking else 0.26) * scale + max(0.0, math.sin(t * 2.0 + idx)) * 0.24 * scale
        shoulder_tilt = math.sin(t * 0.9 + idx * 0.5) * 0.06 * scale
        joints = {
            "hip": (cx + sway, cy - 0.05 + bounce, cz),
            "chest": (cx + sway * 0.86, cy + 0.46 * scale + bounce, cz),
            "head": (cx + sway * 0.95, cy + 0.84 * scale + bounce, cz),
            "ls": (cx - 0.18 * scale + sway, cy + 0.44 * scale + bounce + shoulder_tilt, cz),
            "rs": (cx + 0.18 * scale + sway, cy + 0.44 * scale + bounce - shoulder_tilt, cz),
            "lh": (cx - 0.12 * scale + sway, cy - 0.04 * scale + bounce, cz),
            "rh": (cx + 0.12 * scale + sway, cy - 0.04 * scale + bounce, cz),
            "la": (cx - 0.38 * scale + sway, cy + 0.3 * scale + bounce + arm_raise, cz + 0.02),
            "ra": (cx + 0.38 * scale + sway, cy + 0.26 * scale + bounce + arm_raise * 0.75, cz + 0.02),
            "lf": (cx - 0.12 * scale + sway + step, cy - 0.72 * scale + bounce, cz + 0.04),
            "rf": (cx + 0.12 * scale + sway - step, cy - 0.72 * scale + bounce, cz - 0.04),
        }
        screen_points: dict[str, tuple[int, int]] = {}
        depths: list[float] = []
        for name, point in joints.items():
            screen, depth = ps2fx.project_point(camera, point, w, h)
            if screen is None:
                return False
            screen_points[name] = screen
            depths.append(depth)
        depth = float(sum(depths) / len(depths))
        body_fill = ps2fx.fog_color(ps2fx.lerp_color(palette.ui, palette.accent, 0.18 + (idx % 3) * 0.12), fog, depth)
        body_edge = ps2fx.fog_color(palette.glow, fog, depth * 0.84)
        torso = [
            screen_points["ls"],
            screen_points["rs"],
            screen_points["rh"],
            screen_points["lh"],
        ]
        ps2fx.draw_shaded_polygon(frame, torso, body_fill, body_edge, 1.0)
        for a, b, wa, wb in (
            ("ls", "la", 5.0, 3.0),
            ("rs", "ra", 5.0, 3.0),
            ("lh", "lf", 5.0, 3.0),
            ("rh", "rf", 5.0, 3.0),
            ("chest", "head", 4.0, 3.0),
        ):
            poly = self._segment_quad_2d(screen_points[a], screen_points[b], wa, wb)
            ps2fx.draw_shaded_polygon(frame, poly, body_fill, body_edge, 1.0)
        head_center = screen_points["head"]
        head_radius = max(4, int(14.0 / max(1.2, depth)))
        head_poly = [
            (head_center[0], head_center[1] - head_radius),
            (head_center[0] + head_radius, head_center[1]),
            (head_center[0], head_center[1] + head_radius),
            (head_center[0] - head_radius, head_center[1]),
        ]
        ps2fx.draw_shaded_polygon(frame, head_poly, ps2fx.fog_color(ps2fx.lerp_color((248, 236, 214), palette.glow, 0.24), fog, depth), body_edge, 1.0)
        return True

    def _draw_people_group(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        f: dict[str, float],
        *,
        crowd_center_z: float,
        rows: int = 3,
        walking: bool = False,
    ) -> None:
        ps2fx.debug_begin_object("polygon_people")
        for row in range(rows):
            z = crowd_center_z + row * 1.9
            spread = 1.45 + row * 0.42
            y = -0.22 + row * 0.02
            for col in range(-2, 3):
                x = col * spread * 0.42 + math.sin(t * 0.12 + row + col) * 0.06
                scale = 0.86 - row * 0.12
                self._draw_polygon_person(frame, camera, palette, fog, t, f, (x, y, z), scale, row * 5 + col + 3, walking=walking)
        ps2fx.debug_end_object()

    def _draw_space_habitat(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        *,
        garden: bool = False,
    ) -> None:
        h, w = frame.shape[:2]
        planet_center = (int(w * 0.78), int(h * 0.24))
        planet_r = int(h * 0.16)
        cv2.circle(frame, planet_center, planet_r, ps2fx.bgr(ps2fx.lerp_color((176, 206, 255), palette.glow, 0.2)), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, planet_center, planet_r, ps2fx.bgr(palette.ui), 2, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, planet_center, (planet_r + 24, max(8, planet_r // 2)), 16, 0, 360, ps2fx.bgr(palette.accent), 2, lineType=cv2.LINE_AA)
        ps2fx.draw_quad_3d(frame, camera, [(-4.6, -1.06, camera.z + 1.3), (4.6, -1.06, camera.z + 1.3), (3.5, -1.06, camera.z + 10.8), (-3.5, -1.06, camera.z + 10.8)], ps2fx.lerp_color((168, 212, 230), palette.mid, 0.2), palette.glow, fog, 0.96)
        for z in (camera.z + 3.4, camera.z + 6.4, camera.z + 9.2):
            ps2fx.draw_portal_ring_3d(frame, camera, (0.0, 0.25 + math.sin(t * 0.16 + z) * 0.04, z), 1.55, palette.accent, fog, y_scale=0.56, thickness=3)
        for side in (-1, 1):
            ps2fx.draw_quad_3d(frame, camera, [(side * 0.6, 0.2, camera.z + 3.2), (side * 3.0, 1.15, camera.z + 5.1), (side * 2.7, 1.45, camera.z + 7.2), (side * 0.5, 0.46, camera.z + 5.8)], ps2fx.lerp_color((204, 246, 255), palette.glow, 0.12), palette.ui, fog, 0.62)
        if garden:
            self._draw_tree_cluster(frame, camera, palette, fog, t, [(-1.8, -0.48, camera.z + 4.0), (1.6, -0.48, camera.z + 4.8), (-0.5, -0.48, camera.z + 6.5)])

    def _draw_geometry_tokens(
        self,
        frame: np.ndarray,
        camera,
        palette: Palette,
        fog: tuple[int, int, int],
        t: float,
        motion: tuple[str, ...],
        near_z: float,
        far_z: float,
    ) -> None:
        geom = self.config.dominant_geometry
        if not geom:
            return
        if "rails" in geom:
            ps2fx.draw_quad_3d(frame, camera, [(-1.8, -1.12, near_z), (-1.45, -1.12, near_z), (-0.5, -1.12, far_z), (-0.72, -1.12, far_z)], palette.ui, palette.glow, fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(1.45, -1.12, near_z), (1.8, -1.12, near_z), (0.72, -1.12, far_z), (0.5, -1.12, far_z)], palette.ui, palette.glow, fog, 1.0)
        if "water_slabs" in geom:
            for i, z in enumerate(np.linspace(near_z + 0.6, far_z - 1.0, 4)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                ps2fx.draw_quad_3d(frame, camera, [(-3.8 + xo, -1.0 + yo, z + zo), (3.8 + xo, -1.0 + yo, z + zo), (3.0 + xo, -1.0 + yo, z + 1.8 + zo), (-3.0 + xo, -1.0 + yo, z + 1.8 + zo)], ps2fx.lerp_color(palette.mid, palette.glow, 0.22), palette.ui, fog, 0.42)
        if "columns" in geom:
            for i, z in enumerate(np.linspace(near_z + 1.0, far_z, 5)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                for side in (-1, 1):
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 3.3 + xo, -0.05 + yo, z + zo), (0.34, 2.1, 0.34), palette.ui, fog)
        if "towers" in geom:
            for i, z in enumerate(np.linspace(near_z + 2.2, far_z + 3.0, 6)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                ps2fx.draw_cuboid_3d(frame, camera, (-4.9 + xo, -0.5 + yo, z + zo), (1.1, 3.3, 1.1), palette.mid, fog)
                ps2fx.draw_cuboid_3d(frame, camera, (4.6 + xo * 0.8, -0.42 + yo, z + 0.8 + zo), (1.3, 3.9, 1.3), palette.ui, fog)
        if "signage_blocks" in geom or "floating_panels" in geom:
            for i, z in enumerate(np.linspace(near_z + 1.6, far_z, 6)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                side = -1 if i % 2 == 0 else 1
                ps2fx.draw_billboard_3d(frame, camera, (side * (2.4 + 0.28 * i) + xo, 0.55 + yo, z + zo), (1.25, 0.52), palette.accent, palette.glow, fog, 0.88)
        if "arches" in geom:
            for i, z in enumerate(np.linspace(near_z + 1.4, far_z, 4)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                ps2fx.draw_quad_3d(frame, camera, [(-2.6 + xo, 1.15 + yo, z + zo), (2.6 + xo, 1.15 + yo, z + zo), (2.1 + xo, 0.55 + yo, z + 0.8 + zo), (-2.1 + xo, 0.55 + yo, z + 0.8 + zo)], ps2fx.lerp_color(palette.ui, palette.glow, 0.18), palette.glow, fog, 1.0)
        if "ring_gates" in geom:
            for i, z in enumerate(np.linspace(near_z + 1.8, far_z, 5)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                radius = 1.0 + 0.16 * math.sin(t * 0.25 + i)
                ps2fx.draw_portal_ring_3d(frame, camera, (xo, yo + 0.08, z + zo), radius, palette.accent, fog, y_scale=0.76, thickness=3)
        if "ramps" in geom:
            ps2fx.draw_quad_3d(frame, camera, [(-1.6, -1.08, near_z + 0.8), (0.2, -0.5, near_z + 0.8), (1.8, 0.1, far_z - 1.2), (0.1, -0.48, far_z - 1.2)], ps2fx.lerp_color(palette.mid, palette.ui, 0.22), palette.glow, fog, 0.92)
        if "shards" in geom or "pyramids" in geom or "prisms" in geom:
            for i, z in enumerate(np.linspace(near_z + 2.0, far_z, 6)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                spin = t * 0.24 + i * 0.5
                width = 0.55 + (i % 3) * 0.12
                px = math.sin(spin) * 0.36 + (i - 2.5) * 0.7 + xo
                apex = (px, 1.0 + yo, z + zo)
                base_a = (px - width, -0.28 + yo, z + 0.35 + zo)
                base_b = (px + width, -0.28 + yo, z + 0.35 + zo)
                ps2fx.draw_quad_3d(frame, camera, [apex, base_a, base_b], ps2fx.lerp_color(palette.ui, palette.glow, 0.15 + (i % 2) * 0.18), palette.glow, fog, 0.96)
        if "pods" in geom:
            for i, z in enumerate(np.linspace(near_z + 1.2, far_z, 5)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                ps2fx.draw_cuboid_3d(frame, camera, (math.sin(i * 0.6) * 1.8 + xo, 0.48 + yo, z + zo), (1.2, 0.46, 0.5), palette.accent, fog, edge=palette.glow)
        if "antenna_arrays" in geom:
            for i, z in enumerate(np.linspace(near_z + 2.0, far_z + 1.2, 5)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                base = (math.sin(i) * 3.4 + xo, 0.2 + yo, z + zo)
                ps2fx.draw_cuboid_3d(frame, camera, base, (0.16, 2.8, 0.16), palette.ui, fog)
                ps2fx.draw_quad_3d(frame, camera, [(base[0] - 0.7, 1.1 + yo, z + zo), (base[0] + 0.7, 1.1 + yo, z + zo), (base[0], 1.45 + yo, z + 0.2 + zo)], palette.glow, palette.accent, fog, 0.85)
        if "cable_arcs" in geom:
            for i, z in enumerate(np.linspace(near_z + 2.2, far_z, 4)):
                points = []
                for j in range(6):
                    px = -2.2 + j * 0.85
                    py = 0.9 + math.sin(j / 5 * math.pi) * (0.8 + 0.12 * math.sin(t * 0.3 + i))
                    pt, depth = ps2fx.project_point(camera, (px, py, z + i * 0.5), frame.shape[1], frame.shape[0])
                    if pt is not None and depth > 0.05:
                        points.append(pt)
                if len(points) >= 2:
                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, ps2fx.bgr(palette.glow), 1, lineType=cv2.LINE_AA)
        if "low_poly_plants" in geom:
            for i, z in enumerate(np.linspace(near_z + 2.0, far_z, 5)):
                xo, yo, zo = self._motion_offset(motion, i, t)
                px = -3.8 + i * 1.9 + xo
                ps2fx.draw_cuboid_3d(frame, camera, (px, -0.7 + yo, z + zo), (0.12, 0.7, 0.12), (72, 124, 86), fog)
                ps2fx.draw_quad_3d(frame, camera, [(px, 0.1 + yo, z + zo), (px - 0.42, -0.18 + yo, z + 0.18 + zo), (px + 0.42, -0.18 + yo, z + 0.18 + zo)], (86, 176, 118), palette.glow, fog, 0.88)

    def _render_frutiger_world_variant(
        self,
        frame: np.ndarray,
        camera,
        t: float,
        f: dict[str, float],
        palette: Palette,
        fog: tuple[int, int, int],
        tags: set[str],
    ) -> None:
        open_sky_tags = {"sky", "cloud", "sunlit", "white", "floating", "park", "garden", "lawn", "plaza", "terminal", "bridge", "promenade", "glass", "greenhouse"}
        warm = 0.22 if {"sun", "sunlit", "sunset", "ocean"} & tags else 0.0
        if tags & open_sky_tags:
            self._draw_open_sky(frame, palette, t, warm=warm, sun_strength=0.85, cloud_density=1.25 if "cloud" in tags else 1.0)
        if {"grass", "park", "garden", "lawn", "botanical", "greenhouse", "promenade", "field"} & tags:
            self._draw_grass_plane(frame, camera, palette, fog, t, path="field" not in tags)
            tree_positions = [(-3.6, -0.48, camera.z + 3.2), (3.2, -0.48, camera.z + 4.6), (-1.0, -0.48, camera.z + 7.6), (2.1, -0.48, camera.z + 9.0)]
            self._draw_tree_cluster(frame, camera, palette, fog, t, tree_positions)
        if {"fountain", "splash", "water", "jet", "pool", "ocean", "waterfall", "reflective"} & tags:
            center_z = camera.z + (5.4 if {"plaza", "court"} & tags else 6.6)
            self._draw_water_feature(frame, camera, palette, fog, t, f, center_z=center_z, width=2.9 if "ocean" not in tags else 3.6, depth=2.4 if "ocean" not in tags else 3.2, splash_scale=1.15 if {"splash", "jet"} & tags else 0.82)
            if {"ocean", "terrace"} & tags:
                ps2fx.draw_quad_3d(frame, camera, [(-6.4, -1.04, camera.z + 7.5), (6.4, -1.04, camera.z + 7.5), (8.4, -1.04, camera.z + 17.5), (-8.4, -1.04, camera.z + 17.5)], ps2fx.lerp_color((104, 176, 214), palette.mid, 0.18), palette.glow, fog, 0.88)
        if {"orbital", "planet", "asteroid", "solar", "moon", "stars", "observation"} & tags:
            self._draw_space_habitat(frame, camera, palette, fog, t, garden=bool({"garden", "plaza"} & tags))
        if {"dancing", "crowd", "walking", "commuters", "celebration", "figures", "people"} & tags:
            self._draw_people_group(frame, camera, palette, fog, t, f, crowd_center_z=camera.z + 4.8, rows=3, walking=bool({"walking", "commuters"} & tags))
        if {"plaza", "civic", "court", "terminal", "skybridge", "deck"} & tags:
            ps2fx.debug_begin_object("world_deck")
            ps2fx.draw_quad_3d(frame, camera, [(-5.3, -1.12, camera.z + 1.4), (5.3, -1.12, camera.z + 1.4), (4.2, -1.12, camera.z + 13.0), (-4.2, -1.12, camera.z + 13.0)], ps2fx.lerp_color((168, 214, 218), palette.mid, 0.18), palette.ui, fog, 0.72)
            for x in (-3.8, -1.2, 1.2, 3.8):
                ps2fx.draw_cuboid_3d(frame, camera, (x, -0.02, camera.z + 4.2 + (x % 2.0) * 0.4), (0.28, 1.8, 0.28), ps2fx.lerp_color((164, 214, 222), palette.ui, 0.14), fog, edge=palette.glow)
            ps2fx.debug_end_object()

    def _variant_background_shapes(self, frame: np.ndarray, camera, palette: Palette, fog: tuple[int, int, int], tags: set[str], t: float) -> None:
        if {"mountain", "hillside", "cliffs", "valley"} & tags:
            for i, z in enumerate((camera.z + 10.0, camera.z + 14.0, camera.z + 18.0)):
                span = 6.2 - i * 0.8
                ps2fx.draw_quad_3d(frame, camera, [(-span, -0.9, z), (span, -0.9, z), (span * 0.6, 1.9 - i * 0.15, z + 0.8), (-span * 0.7, 1.5 - i * 0.1, z + 0.8)], ps2fx.lerp_color(palette.bg, palette.mid, 0.18 + i * 0.08), palette.ui, fog, 0.7)
        if {"ferry", "coastal", "river", "canal", "reservoir", "tidal", "waterfront", "moonlit"} & tags:
            for i in range(4):
                y = -1.1 + math.sin(t * 0.22 + i) * 0.03
                ps2fx.draw_quad_3d(frame, camera, [(-6.0, y, camera.z + 4.0 + i * 2.6), (6.0, y, camera.z + 4.0 + i * 2.6), (5.4, y, camera.z + 6.0 + i * 2.6), (-5.4, y, camera.z + 6.0 + i * 2.6)], ps2fx.lerp_color(palette.mid, palette.glow, 0.12), palette.ui, fog, 0.34)
        if {"cloud", "storm", "typhoon", "rain"} & tags:
            for i in range(5):
                px = -4.8 + i * 2.3
                pz = camera.z + 7.0 + i * 1.5
                ps2fx.draw_billboard_3d(frame, camera, (px, 2.0 + math.sin(t * 0.12 + i) * 0.08, pz), (2.2, 0.8), ps2fx.lerp_color(palette.mid, palette.glow, 0.18), palette.glow, fog, 0.18)

    def _render_transit_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        tags = self._recipe_tags()
        near_z = camera.z + 1.0
        far_z = camera.z + 12.5
        self._render_frutiger_world_variant(frame, camera, t, f, palette, fog, tags)
        self._variant_background_shapes(frame, camera, palette, fog, tags, t)
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_city_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        tags = self._recipe_tags()
        near_z = camera.z + 1.2
        far_z = camera.z + 14.5
        self._render_frutiger_world_variant(frame, camera, t, f, palette, fog, tags)
        self._variant_background_shapes(frame, camera, palette, fog, tags, t)
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_water_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        tags = self._recipe_tags()
        near_z = camera.z + 1.1
        far_z = camera.z + 10.5
        self._render_frutiger_world_variant(frame, camera, t, f, palette, fog, tags)
        self._variant_background_shapes(frame, camera, palette, fog, tags, t)
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_ui_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        near_z = camera.z + 1.0
        far_z = camera.z + 11.8
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)
        if "route" in self._recipe_tags() or "map" in self._recipe_tags() or "radar" in self._recipe_tags():
            for i in range(4):
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, 0.0, camera.z + 3.0 + i * 2.1), 0.6 + i * 0.22, palette.glow, fog, y_scale=0.58, thickness=2)

    def _render_nature_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        tags = self._recipe_tags()
        near_z = camera.z + 1.4
        far_z = camera.z + 13.0
        self._render_frutiger_world_variant(frame, camera, t, f, palette, fog, tags)
        self._variant_background_shapes(frame, camera, palette, fog, tags, t)
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_architecture_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        near_z = camera.z + 1.0
        far_z = camera.z + 12.0
        self._render_frutiger_world_variant(frame, camera, t, f, palette, fog, self._recipe_tags())
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_abstract_variant(self, frame: np.ndarray, camera, t: float, f: dict[str, float], palette: Palette, fog: tuple[int, int, int]) -> None:
        near_z = camera.z + 1.0
        far_z = camera.z + 10.8
        self._draw_geometry_tokens(frame, camera, palette, fog, t, self.config.movement_type, near_z, far_z)

    def _render_highway(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        sky = ps2fx.lerp_color(palette.bg, palette.mid, 0.25)
        ps2fx.gradient_background(frame, sky, ps2fx.lerp_color(palette.mid, palette.glow, 0.08))
        camera = self._camera(t * self.config.speed, dolly=t * (0.55 + f["bass"] * 0.35), bob=0.03, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.08)
        near_z = camera.z + 0.4
        far_z = camera.z + 20.0
        ps2fx.debug_begin_object("road_shell")
        ps2fx.draw_corridor_planes(frame, camera, (34, 40, 52), (52, 58, 70), (22, 26, 34), fog, width=7.2, height=3.2, near_z=near_z, far_z=far_z)
        ps2fx.draw_quad_3d(frame, camera, [(-3.6, -1.28, near_z), (3.6, -1.28, near_z), (5.6, -1.28, far_z), (-5.6, -1.28, far_z)], (58, 60, 68), palette.glow, fog, 1.0)
        ps2fx.draw_quad_3d(frame, camera, [(-1.95, -1.26, near_z + 0.1), (1.95, -1.26, near_z + 0.1), (2.85, -1.26, far_z), (-2.85, -1.26, far_z)], (72, 74, 82), palette.ui, fog, 0.95)
        ps2fx.draw_floor_grid_3d(frame, camera, palette.ui, fog, x_range=(-4.8, 4.8), z_range=(near_z, far_z), spacing=0.95, y=-1.28)
        ps2fx.debug_end_object()
        ps2fx.debug_begin_object("roadside_signs")
        for z in np.linspace(camera.z + 2.4, camera.z + 22.0, 13):
            sign_z = z - (camera.z % 3.2)
            for side in (-1, 1):
                px = side * (3.55 + (z % 2.0) * 0.2)
                ps2fx.draw_cuboid_3d(frame, camera, (px, -0.05, sign_z), (0.16, 1.9, 0.16), palette.ui, fog)
                ps2fx.draw_billboard_3d(frame, camera, (px + side * 0.72, 0.52, sign_z + 0.28), (1.45, 0.8), palette.accent, palette.glow, fog, 1.0)
                ps2fx.draw_billboard_3d(frame, camera, (side * 1.52, -0.9, sign_z + 0.02), (0.26, 1.1), palette.glow if side < 0 else palette.accent, palette.ui, fog, 1.0)
        ps2fx.draw_billboard_3d(frame, camera, (-3.25, 0.35, 2.2), (1.9, 1.1), palette.accent, palette.glow, fog, 1.0)
        ps2fx.draw_billboard_3d(frame, camera, (3.2, 0.48, 3.0), (1.45, 0.84), palette.glow, palette.ui, fog, 1.0)
        ps2fx.debug_end_object()
        ps2fx.debug_begin_object("tunnel_ribs")
        for z in np.linspace(camera.z + 3.0, camera.z + 24.0, 11):
            rib_z = z - (camera.z % 2.8)
            ps2fx.draw_quad_3d(frame, camera, [(-5.8, 2.25, rib_z), (5.8, 2.25, rib_z), (4.5, 0.85, rib_z), (-4.5, 0.85, rib_z)], ps2fx.lerp_color(palette.mid, palette.ui, 0.36), palette.glow, fog, 1.0)
        ps2fx.debug_end_object()
        ps2fx.debug_begin_object("lane_markers")
        for lane_x in (-1.2, 0.0, 1.2):
            for z in np.linspace(camera.z + 1.8, camera.z + 20.0, 15):
                dash_z = z - (camera.z * 1.2 % 1.8)
                ps2fx.draw_billboard_3d(frame, camera, (lane_x, -1.18, dash_z), (0.22, 1.0), palette.glow if lane_x == 0 else palette.accent, palette.ui, fog, 1.0)
        ps2fx.debug_end_object()
        ps2fx.draw_reflection_streaks(frame, palette.accent, 0.12 + f["bass"] * 0.24, t)

    def _render_atrium(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        top = ps2fx.lerp_color(palette.mid, palette.glow, 0.46 + f["energy"] * 0.08)
        bottom = ps2fx.lerp_color(palette.bg, palette.mid, 0.52)
        ps2fx.gradient_background(frame, top, bottom)
        camera = self._camera(t * 0.45, dolly=t * (0.08 + f["beat"] * 0.05), bob=0.02, orbit=0.35 + f["section"] * 0.4, f=f)
        fog = ps2fx.lerp_color(palette.ui, palette.glow, 0.2)
        ps2fx.debug_begin_object("atrium_shell")
        ps2fx.draw_corridor_planes(frame, camera, ps2fx.lerp_color(palette.mid, palette.ui, 0.34), ps2fx.lerp_color(palette.mid, palette.glow, 0.2), ps2fx.lerp_color(palette.bg, palette.ui, 0.2), fog, width=6.4, height=3.2, near_z=1.0, far_z=14.0)
        ps2fx.draw_quad_3d(frame, camera, [(-5.7, -1.24, camera.z + 1.0), (5.7, -1.24, camera.z + 1.0), (4.8, -1.24, camera.z + 14.0), (-4.8, -1.24, camera.z + 14.0)], (142, 172, 184), palette.ui, fog, 0.94)
        ps2fx.draw_quad_3d(frame, camera, [(-5.4, -1.1, camera.z + 1.2), (5.4, -1.1, camera.z + 1.2), (4.3, -1.1, camera.z + 14.0), (-4.3, -1.1, camera.z + 14.0)], (92, 150, 166), palette.glow, fog, 0.34)
        ps2fx.draw_floor_grid_3d(frame, camera, palette.ui, fog, x_range=(-5.2, 5.2), z_range=(camera.z + 1.4, camera.z + 14.0), spacing=0.9, y=-1.22)
        ps2fx.debug_end_object()
        for x in (-3.4, -1.4, 1.4, 3.4):
            for z in (3.0, 5.5, 8.5, 11.5):
                ps2fx.draw_cuboid_3d(frame, camera, (x, 0.05, camera.z + z), (0.34, 2.6, 0.34), ps2fx.lerp_color(palette.mid, palette.ui, 0.45), fog, edge=palette.glow)
        if self._is("glass_mall_atrium"):
            ps2fx.debug_begin_object("atrium_glass")
            for z in np.linspace(camera.z + 2.2, camera.z + 10.4, 5):
                left = [(-4.9, -1.0, z), (-1.3, -0.28, z), (-1.15, 1.2, z + 1.4), (-4.6, 1.55, z + 1.4)]
                right = [(1.3, -0.28, z), (4.9, -1.0, z), (4.6, 1.55, z + 1.4), (1.15, 1.2, z + 1.4)]
                ps2fx.draw_quad_3d(frame, camera, left, (124, 198, 210), palette.ui, fog, 0.22)
                ps2fx.draw_quad_3d(frame, camera, right, (114, 188, 202), palette.ui, fog, 0.22)
            ps2fx.draw_quad_3d(frame, camera, [(-1.25, -1.12, camera.z + 1.8), (0.2, -0.28, camera.z + 1.8), (1.28, 0.36, camera.z + 7.0), (-0.2, -0.46, camera.z + 7.0)], (138, 196, 206), palette.glow, fog, 0.58)
            ps2fx.draw_quad_3d(frame, camera, [(0.4, -0.48, camera.z + 2.1), (1.9, 0.34, camera.z + 2.1), (2.95, 0.96, camera.z + 7.5), (1.38, 0.16, camera.z + 7.5)], (126, 184, 196), palette.glow, fog, 0.54)
            for z in (camera.z + 3.2, camera.z + 6.4, camera.z + 9.6):
                ps2fx.draw_cuboid_3d(frame, camera, (-4.2, -0.55, z), (0.75, 0.95, 0.75), (82, 156, 112), fog, edge=palette.glow)
                ps2fx.draw_cuboid_3d(frame, camera, (4.0, -0.55, z + 0.5), (0.75, 0.95, 0.75), (82, 156, 112), fog, edge=palette.glow)
            for z in (camera.z + 2.8, camera.z + 6.2, camera.z + 9.4):
                ps2fx.draw_billboard_3d(frame, camera, (-2.2 + math.sin(t * 0.22 + z) * 0.1, 0.8, z), (1.45, 0.88), palette.ui, palette.glow, fog, 0.28)
                ps2fx.draw_billboard_3d(frame, camera, (2.35 + math.cos(t * 0.18 + z) * 0.1, 0.9, z + 0.2), (1.2, 0.74), palette.accent, palette.glow, fog, 0.22)
            ps2fx.debug_end_object()
            for z in (camera.z + 3.5, camera.z + 6.0, camera.z + 8.5, camera.z + 11.0):
                ps2fx.draw_quad_3d(frame, camera, [(-4.8, 1.0, z), (4.8, 1.0, z), (4.8, 2.0, z + 0.2), (-4.8, 2.0, z + 0.2)], (150, 214, 224), palette.ui, fog, 0.18)
        elif self._is("dream_office_lobby"):
            ps2fx.draw_cuboid_3d(frame, camera, (-3.1, -0.3, camera.z + 7.0), (1.0, 1.5, 0.2), palette.bg, fog)
            ps2fx.draw_cuboid_3d(frame, camera, (3.1, -0.3, camera.z + 7.8), (1.0, 1.7, 0.2), palette.bg, fog)
            for z in (4.5, 8.5, 12.5):
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, -0.55, camera.z + z), 0.65 + f["beat"] * 0.18, palette.accent, fog, y_scale=0.35, thickness=2)
        else:
            for z in (4.0, 9.0, 14.0):
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, -0.55, camera.z + z), 0.72 + f["beat"] * 0.15, palette.accent, fog, y_scale=0.28, thickness=2)
        ps2fx.draw_ui_panels(frame, palette.ui, t, f["energy"] * 0.1)
        ps2fx.draw_reflection_streaks(frame, palette.glow, 0.12 + f["highs"] * 0.1, t)

    def _render_submerged(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, ps2fx.lerp_color(palette.glow, palette.mid, 0.26), palette.bg)
        waterline = int(h * 0.24)
        cv2.rectangle(frame, (0, 0), (w, waterline), ps2fx.bgr(ps2fx.lerp_color(palette.glow, palette.ui, 0.28)), -1)
        for i in range(12):
            x = int((i + 0.5) / 12 * w)
            sway = int(math.sin(t * (0.35 + i * 0.02) + i) * (6 + f["mids"] * 18))
            building = np.array([[x - 18, h], [x + 18, h], [x + 16 + sway, int(h * 0.44)], [x - 14 + sway, int(h * 0.47)]], dtype=np.int32)
            cv2.fillConvexPoly(frame, building, ps2fx.bgr(ps2fx.lerp_color(palette.bg, palette.mid, 0.24)))
        for i in range(16):
            prog = ((i / 16.0) + t * 0.16) % 1.0
            radius = int(4 + prog * 10 + f["bass"] * 10)
            cx = int((0.08 + i * 0.055) % 0.94 * w)
            cy = int(h - prog * h * 0.78)
            cv2.circle(frame, (cx, cy), radius, ps2fx.bgr(palette.ui), 1, lineType=cv2.LINE_AA)
        if self._is("crystal_server_lake"):
            for i in range(7):
                cx = int(w * (0.14 + i * 0.11))
                pts = np.array([[cx, int(h * 0.4)], [cx + 10, int(h * 0.54)], [cx, int(h * 0.7)], [cx - 10, int(h * 0.54)]], dtype=np.int32)
                cv2.polylines(frame, [pts], True, ps2fx.bgr(palette.glow), 2)
                cv2.line(frame, (cx, int(h * 0.4)), (cx, int(h * 0.16)), ps2fx.bgr(palette.ui), 1)
        elif self._is("ocean_interface"):
            for i in range(4):
                px = int(w * (0.12 + i * 0.2))
                cv2.rectangle(frame, (px, int(h * 0.18)), (px + 34, int(h * 0.3)), ps2fx.bgr(palette.ui), 1)
                cv2.line(frame, (px + 6, int(h * 0.24)), (px + 28, int(h * 0.24)), ps2fx.bgr(palette.accent), 1)
            for i in range(8):
                y = int(h * (0.52 + i * 0.04))
                cv2.line(frame, (0, y), (w, y + int(math.sin(t * 0.28 + i) * 4)), ps2fx.bgr(palette.glow), 1)
        else:
            for i in range(5):
                panel_x = int(w * (0.1 + i * 0.16))
                cv2.rectangle(frame, (panel_x, int(h * 0.22)), (panel_x + 18, int(h * 0.68)), ps2fx.bgr(palette.ui), 1)
        ps2fx.fog_overlay(frame, 0.04 + f["bass"] * 0.08, palette.glow)

    def _render_transit(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, palette.bg, palette.mid)
        speed = self.config.speed * (0.7 + f["bass"] * 0.5)
        camera = self._camera(t * self.config.speed, dolly=t * speed * (0.56 + f["beat"] * 0.1), bob=0.025 if not self._is("airport_at_night") else 0.015, orbit=0.15 + f["section"] * 0.25, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.1)
        if self._is("airport_at_night"):
            ps2fx.draw_corridor_planes(frame, camera, (78, 98, 116), (118, 154, 172), (38, 54, 72), fog, width=6.6, height=2.9, near_z=1.0, far_z=18.0)
            ps2fx.draw_floor_grid_3d(frame, camera, palette.ui, fog, x_range=(-3.4, 3.4), z_range=(1.2, 18.0), spacing=1.1, y=-1.24)
            ps2fx.draw_quad_3d(frame, camera, [(-6.2, -1.22, camera.z + 1.0), (6.2, -1.22, camera.z + 1.0), (4.9, -1.22, camera.z + 18.0), (-4.9, -1.22, camera.z + 18.0)], (138, 170, 182), palette.ui, fog, 0.92)
            for z in np.linspace(3.0, 22.0, 14):
                local_z = z - (camera.z % 4.0)
                for side in (-1, 1):
                    ps2fx.draw_billboard_3d(frame, camera, (side * (1.8 + z * 0.06), -1.15, local_z), (0.12, 0.22), palette.accent, palette.ui, fog, 0.95)
            for z in (6.0, 10.0, 14.0):
                ps2fx.draw_quad_3d(frame, camera, [(-4.8, 0.4, z), (-1.8, 0.4, z), (-1.4, 1.9, z + 0.24), (-4.6, 1.9, z + 0.24)], (172, 228, 236), palette.glow, fog, 0.22)
                ps2fx.draw_quad_3d(frame, camera, [(1.8, 0.4, z + 0.4), (4.8, 0.4, z + 0.4), (4.6, 1.9, z + 0.58), (1.5, 1.9, z + 0.58)], (156, 216, 226), palette.glow, fog, 0.22)
            ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.12, camera.z + 10.5), (2.8, 1.0), ps2fx.lerp_color(palette.glow, palette.ui, 0.34), palette.accent, fog, 0.2)
            for z in (camera.z + 4.5, camera.z + 9.5, camera.z + 14.5):
                ps2fx.draw_billboard_3d(frame, camera, (-3.2, 0.95, z), (1.45, 0.54), palette.ui, palette.glow, fog, 0.26)
        else:
            ps2fx.draw_corridor_planes(frame, camera, (54, 62, 76), (82, 96, 116), (20, 28, 42), fog, width=6.0, height=2.8, near_z=0.9, far_z=16.0)
            ps2fx.draw_floor_grid_3d(frame, camera, palette.ui, fog, x_range=(-4.2, 4.2), z_range=(1.0, 16.0), spacing=0.95, y=-1.2)
        if self._is("memory_train"):
            ps2fx.debug_begin_object("train_frame")
            cv2.rectangle(frame, (int(w * 0.04), int(h * 0.04)), (int(w * 0.96), int(h * 0.95)), ps2fx.bgr(ps2fx.lerp_color(palette.bg, palette.ui, 0.32)), 10)
            cv2.rectangle(frame, (0, int(h * 0.68)), (int(w * 0.28), h), ps2fx.bgr(ps2fx.lerp_color(palette.bg, palette.mid, 0.42)), -1)
            cv2.rectangle(frame, (0, int(h * 0.12)), (int(w * 0.12), h), ps2fx.bgr(ps2fx.lerp_color(palette.mid, palette.ui, 0.34)), -1)
            cv2.rectangle(frame, (int(w * 0.04), int(h * 0.82)), (int(w * 0.28), h), ps2fx.bgr((52, 60, 74)), -1)
            cv2.line(frame, (int(w * 0.52), int(h * 0.04)), (int(w * 0.52), int(h * 0.95)), ps2fx.bgr(palette.ui), 4)
            cv2.line(frame, (int(w * 0.16), int(h * 0.72)), (int(w * 0.28), int(h * 0.72)), ps2fx.bgr(palette.glow), 2)
            ps2fx.debug_end_object()
            ps2fx.debug_begin_object("platform")
            ps2fx.draw_quad_3d(frame, camera, [(-4.8, -0.75, camera.z + 1.8), (4.8, -0.75, camera.z + 1.8), (4.8, -0.75, camera.z + 12.8), (-4.8, -0.75, camera.z + 12.8)], (78, 78, 82), palette.ui, fog, 0.9)
            for z in np.linspace(camera.z + 2.0, camera.z + 12.5, 8):
                local_z = z - (camera.z * 1.15 % 2.0)
                ps2fx.draw_cuboid_3d(frame, camera, (-4.8, -0.1, local_z), (1.8, 2.35, 1.8), ps2fx.lerp_color(palette.bg, palette.ui, 0.24), fog, edge=palette.glow)
                ps2fx.draw_cuboid_3d(frame, camera, (4.7, -0.05, local_z + 0.6), (1.8, 2.1, 1.7), ps2fx.lerp_color(palette.mid, palette.ui, 0.28), fog, edge=palette.glow)
                ps2fx.draw_cuboid_3d(frame, camera, (0.0, -0.15, local_z + 0.2), (3.6, 0.38, 1.7), (92, 90, 94), fog)
                ps2fx.draw_cuboid_3d(frame, camera, (-2.7, 0.4, local_z + 0.15), (0.22, 1.65, 0.22), palette.ui, fog)
                ps2fx.draw_billboard_3d(frame, camera, (2.8, 0.2, local_z + 0.15), (1.55, 0.62), palette.accent, palette.glow, fog, 1.0)
                ps2fx.draw_billboard_3d(frame, camera, (-2.1, 0.26, local_z + 0.24), (0.9, 0.3), palette.glow, palette.ui, fog, 0.95)
            ps2fx.debug_end_object()
        elif self._is("dream_metro"):
            ps2fx.debug_begin_object("rails")
            ps2fx.draw_quad_3d(frame, camera, [(-0.95, -1.18, camera.z + 1.0), (-0.45, -1.18, camera.z + 1.0), (-0.15, -1.18, camera.z + 14.0), (-0.45, -1.18, camera.z + 14.0)], palette.ui, palette.glow, fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(0.45, -1.18, camera.z + 1.0), (0.95, -1.18, camera.z + 1.0), (0.45, -1.18, camera.z + 14.0), (0.15, -1.18, camera.z + 14.0)], palette.ui, palette.glow, fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(-4.6, -0.78, camera.z + 2.0), (-1.45, -0.78, camera.z + 2.0), (-1.45, -0.78, camera.z + 14.0), (-4.6, -0.78, camera.z + 14.0)], (84, 82, 88), palette.ui, fog, 0.95)
            ps2fx.debug_end_object()
            ps2fx.debug_begin_object("tunnel_columns")
            for z in np.linspace(camera.z + 2.0, camera.z + 13.0, 8):
                local_z = z - (camera.z * 1.05 % 2.2)
                for side in (-1, 1):
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 3.4, -0.1, local_z), (0.34, 2.2, 0.34), palette.ui, fog)
                    ps2fx.draw_billboard_3d(frame, camera, (side * 2.65, 0.52, local_z + 0.18), (0.95, 0.34), palette.accent, palette.glow, fog, 1.0)
                ps2fx.draw_quad_3d(frame, camera, [(-4.8, 1.7, local_z), (4.8, 1.7, local_z), (4.0, 1.25, local_z + 0.3), (-4.0, 1.25, local_z + 0.3)], ps2fx.lerp_color(palette.mid, palette.ui, 0.26), palette.glow, fog, 0.88)
            ps2fx.debug_end_object()
            ps2fx.draw_speed_lines(frame, palette.glow, t, 0.14 + f["beat"] * 0.08)
        elif self._is("neon_monorail"):
            ps2fx.debug_begin_object("monorail_track")
            ps2fx.draw_quad_3d(frame, camera, [(-2.8, 0.24, camera.z + 0.8), (2.8, 0.24, camera.z + 0.8), (2.0, 0.12, camera.z + 14.0), (-2.0, 0.12, camera.z + 14.0)], (178, 212, 224), palette.glow, fog, 1.0)
            ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.78, camera.z + 2.1), (2.3, 0.72), palette.accent, palette.glow, fog, 0.78)
            ps2fx.debug_end_object()
            ps2fx.debug_begin_object("gates_towers")
            for z in np.linspace(camera.z + 2.2, camera.z + 12.0, 8):
                local_z = z - (camera.z * 0.95 % 2.4)
                ps2fx.draw_quad_3d(frame, camera, [(-2.2, 0.52, local_z), (2.2, 0.52, local_z), (1.55, 0.22, local_z + 2.0), (-1.55, 0.22, local_z + 2.0)], palette.ui, palette.glow, fog, 1.0)
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, 0.34, local_z + 1.15), 1.45 + f["beat"] * 0.16, palette.accent, fog, y_scale=0.72, thickness=3)
            for z in np.linspace(camera.z + 2.8, camera.z + 12.5, 7):
                local_z = z - (camera.z % 2.0)
                ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.7, local_z), (1.7, 0.5), palette.accent, palette.glow, fog, 1.0)
                ps2fx.draw_cuboid_3d(frame, camera, (-4.5, -0.95, local_z), (1.3, 2.8, 1.3), ps2fx.lerp_color(palette.mid, palette.ui, 0.25), fog, edge=palette.glow)
                ps2fx.draw_cuboid_3d(frame, camera, (4.5, -0.9, local_z + 0.6), (1.3, 3.0, 1.3), ps2fx.lerp_color(palette.bg, palette.ui, 0.32), fog, edge=palette.glow)
            ps2fx.debug_end_object()
        else:
            for i in range(16):
                prog = (i / 16.0 + t * self.config.speed * (0.4 + f["beat"])) % 1.0
                x = int(w * 0.12 + prog * (w * 0.76))
                ps2fx.draw_glow_circle(frame, (x, int(h * 0.55)), int(7 + 14 * f["beat"]), palette.accent, 0.2)
        ps2fx.draw_reflection_streaks(frame, palette.glow, 0.08 + f["highs"] * 0.1, t)

    def _render_corridor(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, palette.bg, ps2fx.lerp_color(palette.mid, palette.bg, 0.2))
        camera = self._camera(t * 0.75, dolly=t * (0.32 + f["bass"] * 0.18), bob=0.02, orbit=0.3 + f["section"] * 0.28, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.1)
        if self._is("ring_corridor"):
            ps2fx.draw_corridor_planes(frame, camera, (26, 30, 38), (20, 24, 32), (10, 14, 20), fog, width=4.8, height=2.7, near_z=1.0, far_z=16.0)
            for z in np.linspace(2.8, 14.0, 11):
                local_z = z - (camera.z % 2.2)
                ps2fx.draw_portal_ring_3d(frame, camera, (math.sin(t * 0.4 + z * 0.1) * 0.18, math.cos(t * 0.35 + z * 0.16) * 0.12, local_z), 1.45 + math.sin(t * 0.5 + z) * 0.12, ps2fx.lerp_color(palette.ui, palette.accent, (z % 6.0) / 6.0), fog, y_scale=0.8, thickness=3)
        elif self._is("tunnel_of_screens"):
            ps2fx.debug_begin_object("screen_cases")
            ps2fx.draw_corridor_planes(frame, camera, (42, 48, 58), (26, 30, 36), (16, 18, 24), fog, width=4.8, height=2.5, near_z=camera.z + 1.0, far_z=camera.z + 13.5)
            ps2fx.draw_quad_3d(frame, camera, [(-3.8, -1.12, camera.z + 1.0), (3.8, -1.12, camera.z + 1.0), (3.0, -1.12, camera.z + 12.5), (-3.0, -1.12, camera.z + 12.5)], (46, 54, 64), palette.ui, fog, 0.95)
            for z in np.linspace(camera.z + 1.9, camera.z + 11.8, 7):
                local_z = z - (camera.z % 1.9)
                for side in (-1, 1):
                    x = side * 2.55
                    ps2fx.draw_cuboid_3d(frame, camera, (x, 0.18, local_z), (1.3, 1.7, 0.46), (34, 40, 48), fog, edge=palette.ui)
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 3.5, 0.52, local_z + 0.32), (0.3, 2.0, 0.3), (28, 34, 40), fog)
            ps2fx.draw_cuboid_3d(frame, camera, (-2.9, 0.32, camera.z + 1.55), (1.7, 2.0, 0.58), (38, 44, 52), fog, edge=palette.glow)
            ps2fx.draw_cuboid_3d(frame, camera, (2.95, 0.38, camera.z + 2.15), (1.7, 2.0, 0.58), (38, 44, 52), fog, edge=palette.glow)
            ps2fx.debug_end_object()
            ps2fx.debug_begin_object("screen_faces")
            for z in np.linspace(camera.z + 1.9, camera.z + 11.8, 7):
                local_z = z - (camera.z % 1.9)
                for side in (-1, 1):
                    x = side * 2.55
                    ps2fx.draw_billboard_3d(frame, camera, (x - side * 0.02, 0.32, local_z - 0.26), (1.02, 0.74), palette.ui, palette.glow, fog, 0.82)
                    ps2fx.draw_billboard_3d(frame, camera, (x - side * 0.02, 0.64, local_z - 0.3), (1.02, 0.14), palette.accent, palette.accent, fog, 1.0)
            ps2fx.draw_billboard_3d(frame, camera, (-2.9, 0.42, camera.z + 1.42), (1.32, 0.95), palette.ui, palette.glow, fog, 0.86)
            ps2fx.draw_billboard_3d(frame, camera, (2.95, 0.48, camera.z + 2.02), (1.32, 0.95), palette.ui, palette.glow, fog, 0.86)
            ps2fx.debug_end_object()
        elif self._is("virtual_hotel"):
            ps2fx.debug_begin_object("hotel_corridor")
            ps2fx.draw_corridor_planes(frame, camera, ps2fx.lerp_color(palette.bg, (116, 88, 74), 0.4), ps2fx.lerp_color(palette.bg, (128, 100, 84), 0.34), ps2fx.lerp_color(palette.bg, palette.ui, 0.18), fog, width=4.4, height=2.4, near_z=camera.z + 1.0, far_z=camera.z + 13.0)
            ps2fx.draw_quad_3d(frame, camera, [(-4.2, -1.12, camera.z + 1.0), (4.2, -1.12, camera.z + 1.0), (3.4, -1.12, camera.z + 12.0), (-3.4, -1.12, camera.z + 12.0)], (120, 94, 82), palette.ui, fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(-4.0, 1.5, camera.z + 1.0), (4.0, 1.5, camera.z + 1.0), (3.1, 0.96, camera.z + 12.0), (-3.1, 0.96, camera.z + 12.0)], (90, 70, 62), palette.glow, fog, 0.95)
            for z in np.linspace(camera.z + 2.2, camera.z + 11.5, 7):
                local_z = z - (camera.z % 1.9)
                ps2fx.draw_billboard_3d(frame, camera, (-3.35, -0.08, local_z), (1.35, 1.62), palette.accent, palette.ui, fog, 0.55)
                ps2fx.draw_billboard_3d(frame, camera, (3.35, -0.08, local_z + 0.45), (1.35, 1.62), palette.accent, palette.ui, fog, 0.55)
                ps2fx.draw_billboard_3d(frame, camera, (-2.15, 0.42, local_z), (0.28, 0.22), palette.glow, palette.accent, fog, 1.0)
                ps2fx.draw_billboard_3d(frame, camera, (2.15, 0.42, local_z + 0.45), (0.28, 0.22), palette.glow, palette.accent, fog, 1.0)
            ps2fx.debug_end_object()
        elif self._is("fractal_parking_garage"):
            ps2fx.draw_corridor_planes(frame, camera, (52, 54, 60), (42, 44, 50), (28, 30, 34), fog, width=5.0, height=2.1, near_z=1.5, far_z=20.0)
            for z in np.linspace(4.0, 18.0, 8):
                local_z = z - (camera.z % 2.7)
                ps2fx.draw_quad_3d(frame, camera, [(-4.8, -0.8, local_z), (4.8, -0.8, local_z), (3.1, -0.4, local_z + 1.8), (-3.1, -0.4, local_z + 1.8)], (62, 64, 70), (240, 184, 96), fog, 0.92)
                for side in (-1, 1):
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 3.6, -0.2, local_z + 0.4), (0.25, 1.6, 0.25), palette.ui, fog)
        else:
            for side in (-1, 1):
                for i in range(8):
                    x = int(w * 0.5 + side * (30 + i * 20))
                    cv2.line(frame, (x, h), (w // 2 + side * max(8, i * 3), int(h * 0.18)), ps2fx.bgr(palette.mid), 1)
        ps2fx.fog_overlay(frame, 0.04 + f["energy"] * 0.1, palette.glow)

    def _render_aquarium(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        self._render_submerged(frame, t, f, palette)
        h, w = frame.shape[:2]
        if self._is("digital_aquarium"):
            water_band = ps2fx.lerp_color(palette.mid, palette.glow, 0.45)
            cv2.rectangle(frame, (0, int(h * 0.52)), (w, int(h * 0.74)), ps2fx.bgr(water_band), -1)
            for i in range(5):
                x = int(w * (0.12 + i * 0.18))
                cv2.rectangle(frame, (x, int(h * 0.18)), (x + int(w * 0.12), int(h * 0.74)), ps2fx.bgr(palette.ui), 1)
            for i in range(6):
                bx = int(w * (0.16 + i * 0.12))
                cv2.circle(frame, (bx, int(h * 0.3 + math.sin(t + i) * 10)), 8 + (i % 3) * 4, ps2fx.bgr(palette.accent), 1)
            ps2fx.draw_ui_panels(frame, palette.ui, t * 1.2, f["highs"] * 0.12)
        else:
            for i in range(8):
                y = int(h * (0.48 + i * 0.04))
                cv2.line(frame, (0, y), (w, y + int(math.sin(t * 0.28 + i) * 5)), ps2fx.bgr(palette.glow), 1)
            for i in range(4):
                px = int(w * (0.12 + i * 0.2))
                cv2.rectangle(frame, (px, int(h * 0.16)), (px + 34, int(h * 0.28)), ps2fx.bgr(palette.ui), 1)
            ps2fx.draw_ui_panels(frame, palette.ui, t, f["highs"] * 0.08)

    def _render_shrine(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, palette.bg, palette.mid)
        if self._is("bios_temple"):
            grid_y = int(h * 0.66)
            for i in range(10):
                y = grid_y + i * 10
                cv2.line(frame, (0, y), (w, y), ps2fx.bgr(ps2fx.lerp_color(palette.mid, palette.ui, 0.25)), 1)
            for i in range(8):
                scale = 0.12 + i * 0.07
                y = int(h * (0.26 + i * 0.06))
                cv2.rectangle(frame, (int(w * (0.5 - scale)), y), (int(w * (0.5 + scale)), y + 8), ps2fx.bgr(palette.accent), -1)
            for col in (-1, 1):
                for i in range(5):
                    x = int(w * 0.5 + col * (120 + i * 40))
                    cv2.line(frame, (x, int(h * 0.22)), (x, h), ps2fx.bgr(palette.ui), 2)
            for i in range(7):
                ang = t * 0.24 + i * math.tau / 7.0
                px = w // 2 + int(math.cos(ang) * (52 + f["energy"] * 18))
                py = int(h * 0.34 + math.sin(ang * 1.4) * 22)
                cv2.rectangle(frame, (px - 12, py - 8), (px + 12, py + 8), ps2fx.bgr(palette.glow), 1)
                cv2.line(frame, (px - 8, py), (px + 8, py), ps2fx.bgr(palette.ui), 1)
                cv2.line(frame, (px, py - 6), (px, py + 6), ps2fx.bgr(palette.ui), 1)
            ps2fx.scanlines(frame, 0.24 + f["onset"] * 0.1)
            frame[:] = ps2fx.add_dither(frame, 0.7)
        else:
            camera = self._camera(t * 0.55, dolly=t * (0.24 + f["beat"] * 0.08), bob=0.03, orbit=0.6 + f["section"] * 0.3, f=f)
            fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.2)
            ps2fx.draw_corridor_planes(frame, camera, (16, 20, 26), (12, 16, 22), (8, 10, 16), fog, width=4.8, height=2.6, near_z=1.6, far_z=22.0)
            for z in np.linspace(4.0, 18.0, 7):
                local_z = z - (camera.z % 2.8)
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, 0.1, local_z), 1.2 + math.sin(t * 0.45 + z) * 0.08, palette.accent, fog, y_scale=0.9, thickness=2)
                ps2fx.draw_quad_3d(frame, camera, [(-1.8, -1.0, local_z), (1.8, -1.0, local_z), (1.5, -0.82, local_z + 0.9), (-1.5, -0.82, local_z + 0.9)], ps2fx.lerp_color(palette.mid, palette.ui, 0.18), palette.ui, fog, 0.9)
                for side in (-1, 1):
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 1.55, 0.0, local_z), (0.16, 1.7, 0.16), palette.ui, fog)
                    ps2fx.draw_billboard_3d(frame, camera, (side * 1.2, 0.65, local_z + 0.15), (0.28, 0.16), palette.glow, palette.accent, fog, 0.9)
        ps2fx.fog_overlay(frame, 0.06 + f["energy"] * 0.12, palette.glow)

    def _render_city(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, ps2fx.lerp_color(palette.mid, palette.glow, 0.3), palette.bg)
        camera = self._camera(t * 0.4, dolly=t * (0.12 + f["bass"] * 0.08), bob=0.018, orbit=0.42 + f["section"] * 0.22, f=f)
        fog = ps2fx.lerp_color(palette.ui, palette.glow, 0.22)
        for z in (6.0, 9.0, 13.0):
            for x in (-6.0, -3.5, -1.0, 1.5, 4.0):
                height_units = 1.8 + ((int((x + z) * 2) % 4) * 0.6)
                ps2fx.draw_cuboid_3d(frame, camera, (x, -0.45 + height_units * 0.4, z), (1.45, height_units, 1.45), ps2fx.lerp_color(palette.bg, palette.ui, 0.42 + (z - 6.0) * 0.03), fog, edge=palette.glow)
        if self._is("neon_rain_alley"):
            ps2fx.draw_corridor_planes(frame, camera, (18, 18, 24), (14, 14, 18), (8, 10, 14), fog, width=4.0, height=2.1, near_z=1.6, far_z=18.0)
            for z in np.linspace(4.0, 16.0, 6):
                local_z = z - (camera.z % 2.5)
                ps2fx.draw_billboard_3d(frame, camera, (-2.8, -0.1, local_z), (0.75, 0.4), palette.accent, palette.glow, fog, 0.95)
                ps2fx.draw_billboard_3d(frame, camera, (2.9, 0.05, local_z + 0.4), (0.7, 0.35), palette.glow, palette.ui, fog, 0.95)
            self._draw_rain(frame, palette, t, 0.8 + f["highs"] * 0.4)
            ps2fx.draw_reflection_streaks(frame, palette.accent, 0.28 + f["bass"] * 0.26, t)
        elif self._is("skybridge_city"):
            ps2fx.debug_begin_object("skybridge")
            ps2fx.draw_quad_3d(frame, camera, [(-5.8, -0.15, camera.z + 1.4), (5.8, -0.15, camera.z + 1.4), (4.8, -0.45, camera.z + 13.0), (-4.8, -0.45, camera.z + 13.0)], (180, 220, 230), palette.glow, fog, 1.0)
            for z in (camera.z + 3.5, camera.z + 6.5, camera.z + 9.5):
                ps2fx.draw_quad_3d(frame, camera, [(-5.2, 0.25, z), (5.2, 0.25, z), (4.4, -0.05, z + 1.8), (-4.4, -0.05, z + 1.8)], (154, 204, 218), palette.glow, fog, 1.0)
                for x in (-3.2, -1.4, 1.4, 3.2):
                    ps2fx.draw_cuboid_3d(frame, camera, (x, 0.6, z), (0.32, 1.7, 0.32), palette.mid, fog, edge=palette.glow)
            ps2fx.draw_cuboid_3d(frame, camera, (-5.1, 0.9 + math.sin(t * 0.22) * 0.08, camera.z + 2.4), (1.9, 3.4, 1.9), ps2fx.lerp_color(palette.bg, palette.ui, 0.3), fog, edge=palette.glow)
            ps2fx.draw_cuboid_3d(frame, camera, (5.0, 1.1 + math.cos(t * 0.18) * 0.08, camera.z + 3.2), (2.1, 3.8, 2.1), ps2fx.lerp_color(palette.bg, palette.ui, 0.36), fog, edge=palette.glow)
            for z in (camera.z + 3.4, camera.z + 6.8):
                ps2fx.draw_billboard_3d(frame, camera, (-1.6 + math.sin(t * 0.16 + z) * 0.12, 0.65, z), (1.25, 0.68), palette.ui, palette.glow, fog, 0.24)
                ps2fx.draw_billboard_3d(frame, camera, (1.8 + math.cos(t * 0.18 + z) * 0.12, 0.8, z + 0.4), (1.1, 0.58), palette.accent, palette.glow, fog, 0.2)
            ps2fx.debug_end_object()
        elif self._is("hologram_market"):
            for z in np.linspace(4.0, 16.0, 6):
                local_z = z - (camera.z % 2.8)
                for side in (-1, 1):
                    x = side * (1.8 + (z % 2.2))
                    ps2fx.draw_cuboid_3d(frame, camera, (x, -0.7, local_z), (1.0, 0.7, 0.8), (48, 38, 44), fog)
                    ps2fx.draw_billboard_3d(frame, camera, (x, 0.0, local_z), (0.8, 0.3), palette.accent, palette.glow, fog, 0.95)
        elif self._is("rooftop_antenna_field"):
            for z in np.linspace(5.0, 20.0, 8):
                local_z = z - (camera.z % 3.1)
                for x in (-4.5, -2.8, -1.0, 0.8, 2.6, 4.4):
                    ps2fx.draw_cuboid_3d(frame, camera, (x, -1.1, local_z), (0.8, 0.45, 0.8), (30, 34, 42), fog)
                    ps2fx.draw_cuboid_3d(frame, camera, (x, -0.1, local_z), (0.06, 1.6, 0.06), palette.ui, fog)
        elif self._is("arcology_exterior"):
            for z in (8.0, 12.0, 16.0, 22.0):
                for x in (-5.2, -2.4, 0.0, 3.0):
                    ps2fx.draw_cuboid_3d(frame, camera, (x, 0.2 + (z % 3), z), (1.5, 3.2 + (x % 2), 1.4), ps2fx.lerp_color(palette.mid, palette.ui, 0.2), fog)
            for z in (9.0, 14.0):
                ps2fx.draw_quad_3d(frame, camera, [(-6.0, 0.8, z), (6.0, 0.8, z), (6.0, 0.6, z + 1.5), (-6.0, 0.6, z + 1.5)], palette.glow, palette.ui, fog, 0.75)
        elif self._is("data_center_dream"):
            ps2fx.debug_begin_object("server_aisle")
            ps2fx.draw_corridor_planes(frame, camera, (24, 34, 28), (18, 28, 24), (12, 18, 16), fog, width=5.8, height=2.8, near_z=camera.z + 0.9, far_z=camera.z + 14.0)
            ps2fx.draw_quad_3d(frame, camera, [(-1.15, -1.16, camera.z + 1.0), (1.15, -1.16, camera.z + 1.0), (0.9, -1.16, camera.z + 14.0), (-0.9, -1.16, camera.z + 14.0)], (54, 66, 60), palette.ui, fog, 0.95)
            for z in np.linspace(camera.z + 2.0, camera.z + 12.0, 8):
                local_z = z - (camera.z % 1.9)
                for side in (-1, 1):
                    ps2fx.draw_cuboid_3d(frame, camera, (side * 2.9, -0.02, local_z), (1.44, 2.45, 2.05), (52, 86, 78), fog, edge=palette.ui)
                    for led_y in (-0.68, -0.38, -0.08, 0.22, 0.52):
                        ps2fx.draw_billboard_3d(frame, camera, (side * 2.1, led_y, local_z - 0.76), (0.18, 0.16), palette.accent if int((z + led_y * 10 + t * 8)) % 3 == 0 else palette.glow, palette.ui, fog, 1.0)
            ps2fx.draw_cuboid_3d(frame, camera, (-4.25, 0.2, camera.z + 2.2), (1.58, 2.9, 2.3), (42, 72, 66), fog, edge=palette.glow)
            ps2fx.draw_cuboid_3d(frame, camera, (4.25, 0.2, camera.z + 2.7), (1.58, 2.9, 2.3), (42, 72, 66), fog, edge=palette.glow)
            for z in (camera.z + 3.0, camera.z + 6.5, camera.z + 10.0):
                ps2fx.draw_quad_3d(frame, camera, [(-3.7, 1.22, z), (3.7, 1.22, z), (3.1, 1.0, z + 1.2), (-3.1, 1.0, z + 1.2)], (56, 88, 74), palette.glow, fog, 0.82)
            for arc_idx, z in enumerate((camera.z + 3.5, camera.z + 7.4, camera.z + 11.1)):
                ps2fx.draw_billboard_3d(frame, camera, (0.0, 1.32 + math.sin(t * 0.3 + arc_idx) * 0.04, z), (2.2, 0.16), palette.accent, palette.glow, fog, 0.86)
            ps2fx.debug_end_object()
        particle_amount = 0.03 if self._is("data_center_dream") else 0.05 if self._is("skybridge_city") else 0.12
        ps2fx.draw_particles(frame, palette.glow, t, particle_amount + f["highs"] * 0.12, vertical_bias=-0.24)
        ps2fx.fog_overlay(frame, 0.03 + f["energy"] * 0.05, palette.glow)

    def _render_chrome(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, ps2fx.lerp_color(palette.bg, (20, 22, 28), 0.55), ps2fx.lerp_color(palette.mid, (72, 76, 88), 0.45))
        for pool_idx in range(3):
            cx = int(w * (0.24 + pool_idx * 0.26))
            cy = int(h * (0.7 - (pool_idx % 2) * 0.08))
            rx = int(32 + pool_idx * 12 + f["bass"] * 12)
            ry = int(10 + pool_idx * 4)
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, ps2fx.bgr((22, 26, 30)), -1)
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, ps2fx.bgr(palette.ui), 1)
            for ring in range(3):
                cv2.ellipse(frame, (cx, cy), (rx + ring * 8 + int(f["bass"] * 6), ry + ring * 3), 0, 0, 360, ps2fx.bgr(palette.accent), 1)
        for i in range(6):
            cx = int(w * (0.2 + (i % 3) * 0.25) + math.sin(t * 0.4 + i) * 12)
            cy = int(h * (0.26 + (i // 3) * 0.22) + math.cos(t * 0.32 + i) * 8)
            rx = int(24 + f["mids"] * 22 + (i % 3) * 6)
            ry = int(14 + f["bass"] * 14 + (i // 3) * 8)
            angle = (t * 16 + i * 27) % 360
            cv2.ellipse(frame, (cx, cy), (rx, ry), angle, 0, 360, ps2fx.bgr(palette.ui), -1)
            cv2.ellipse(frame, (cx - 4, cy - 3), (max(4, rx // 4), max(4, ry // 4)), angle, 0, 360, ps2fx.bgr(palette.glow), -1)
        for i in range(5):
            y = int(h * (0.2 + i * 0.14))
            cv2.line(frame, (int(w * 0.08), y), (int(w * 0.92), y), ps2fx.bgr(ps2fx.lerp_color(palette.mid, palette.ui, 0.2)), 1)

    def _render_space(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        top = ps2fx.lerp_color((18, 28, 66), palette.bg, 0.45)
        bottom = ps2fx.lerp_color((84, 118, 188), palette.mid, 0.42)
        ps2fx.gradient_background(frame, top, bottom)
        camera = self._camera(t * 0.4, dolly=t * (0.08 + f["bass"] * 0.03), bob=0.012, orbit=0.5 + f["section"] * 0.26, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.18)
        self._draw_space_habitat(frame, camera, palette, fog, t, garden=self._is("solar_sail_space", "moon_pool"))
        if self._is("solar_sail_space"):
            for side in (-1, 1):
                ps2fx.draw_quad_3d(frame, camera, [(side * 0.6, 0.28, camera.z + 3.0), (side * 2.9, 1.36, camera.z + 5.2), (side * 2.4, 1.8, camera.z + 8.8), (side * 0.4, 0.52, camera.z + 6.2)], ps2fx.lerp_color((228, 248, 255), palette.glow, 0.14), palette.ui, fog, 0.68)
            for z in (camera.z + 4.6, camera.z + 7.8):
                ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.74, z), (1.6, 0.54), palette.accent, palette.glow, fog, 0.82)
        elif self._is("moon_pool"):
            ps2fx.draw_quad_3d(frame, camera, [(-4.2, -1.08, camera.z + 3.2), (4.2, -1.08, camera.z + 3.2), (3.1, -1.08, camera.z + 12.2), (-3.1, -1.08, camera.z + 12.2)], (42, 52, 76), palette.ui, fog, 0.95)
            self._draw_water_feature(frame, camera, palette, fog, t, f, center_z=camera.z + 5.8, width=2.8, depth=2.2, splash_scale=0.65)
        else:
            for z in (camera.z + 3.8, camera.z + 7.0, camera.z + 10.2):
                ps2fx.draw_cuboid_3d(frame, camera, (-2.1, -0.38, z), (0.4, 0.9, 0.4), ps2fx.lerp_color(palette.ui, palette.glow, 0.2), fog, edge=palette.glow)
                ps2fx.draw_cuboid_3d(frame, camera, (2.2, -0.38, z + 0.4), (0.4, 0.9, 0.4), ps2fx.lerp_color(palette.ui, palette.glow, 0.14), fog, edge=palette.glow)
        ps2fx.draw_particles(frame, palette.glow, t, 0.06 + f["highs"] * 0.08, vertical_bias=0.0)

    def _render_plaza(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        if self._is("memory_beach"):
            drift = 0.5 + 0.5 * math.sin(t * 0.05)
            sky_top = ps2fx.lerp_color((236, 182, 144), (252, 204, 166), drift)
            sea = ps2fx.lerp_color((94, 142, 182), (122, 176, 206), 0.35 + 0.3 * math.sin(t * 0.04 + 0.8))
            foam = (240, 240, 230)
            ps2fx.draw_ocean_bands(frame, sky_top, sea, foam, t)
            horizon = int(h * 0.46)
            camera = self._camera(t * 0.2, dolly=t * 0.05, bob=0.01, f=f)
            fog = ps2fx.lerp_color(sea, foam, 0.04)
            cv2.rectangle(frame, (0, 0), (w, horizon), ps2fx.bgr(ps2fx.lerp_color((238, 186, 148), (248, 204, 170), drift)), -1)
            for i in range(6):
                bx = int(w * (0.58 + i * 0.065))
                bw = 12 + i * 8
                bh = 22 + i * 12
                shimmer = 0.35 + 0.35 * max(0.0, math.sin(t * 0.22 + i * 0.9))
                tower_color = ps2fx.lerp_color((98, 98, 122), (168, 174, 188), shimmer * 0.35)
                cv2.rectangle(frame, (bx, horizon - bh), (bx + bw, horizon), ps2fx.bgr(tower_color), -1)
            ps2fx.debug_begin_object("beach_scene")
            ps2fx.draw_quad_3d(frame, camera, [(-5.4, -1.02, camera.z + 1.2), (5.4, -1.02, camera.z + 1.2), (5.4, -1.02, camera.z + 14.0), (-5.4, -1.02, camera.z + 14.0)], (162, 140, 116), (188, 164, 138), fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(-5.0, -0.86, camera.z + 2.2), (5.0, -0.86, camera.z + 2.2), (5.0, -0.86, camera.z + 15.5), (-5.0, -0.86, camera.z + 15.5)], sea, foam, fog, 0.9)
            rail_shift = math.sin(t * 0.12) * 0.14
            ps2fx.draw_cuboid_3d(frame, camera, (-4.2 + rail_shift * 0.25, -0.4, camera.z + 1.9), (0.18, 1.0, 2.0), (210, 214, 208), fog)
            ps2fx.draw_quad_3d(frame, camera, [(-4.8 + rail_shift, -0.72, camera.z + 1.55), (-4.3 + rail_shift, -0.72, camera.z + 1.55), (-4.3 + rail_shift * 0.3, 0.0, camera.z + 3.8), (-4.8 + rail_shift * 0.3, 0.0, camera.z + 3.8)], (224, 228, 220), (255, 255, 240), fog, 1.0)
            ps2fx.draw_quad_3d(frame, camera, [(2.9, -0.72, camera.z + 1.6), (5.4, -0.72, camera.z + 1.6), (4.9, -0.52, camera.z + 4.2), (2.5, -0.52, camera.z + 4.2)], (196, 188, 166), (238, 232, 216), fog, 0.96)
            ps2fx.draw_billboard_3d(frame, camera, (-3.0, -0.62, camera.z + 2.0 + math.sin(t * 0.32) * 0.08), (2.7, 0.22 + math.sin(t * 0.18) * 0.02), foam, sea, fog, 1.0)
            ps2fx.draw_billboard_3d(frame, camera, (3.0, -0.58, camera.z + 2.8 + math.cos(t * 0.28) * 0.08), (2.1, 0.2 + math.cos(t * 0.16) * 0.02), foam, sea, fog, 1.0)
            ps2fx.draw_billboard_3d(frame, camera, (0.0, -0.76, camera.z + 5.0 + math.sin(t * 0.12) * 0.1), (7.5, 0.24 + math.sin(t * 0.15) * 0.03), foam, sea, fog, 0.95)
            ps2fx.draw_billboard_3d(frame, camera, (0.6, -0.72, camera.z + 8.0 + math.sin(t * 0.24) * 0.18), (5.2, 0.18 + math.cos(t * 0.13) * 0.02), foam, sea, fog, 0.9)
            ps2fx.draw_billboard_3d(frame, camera, (3.9, -0.44, camera.z + 3.4), (0.28, 0.58), (255, 220, 182), foam, fog, 0.9)
            ps2fx.debug_end_object()
            return
        self._draw_open_sky(frame, palette, t, warm=0.18 if self._is("corporate_fountain_core", "vapor_plaza") else 0.0, sun_strength=0.78, cloud_density=0.86)
        camera = self._camera(t * 0.34, dolly=t * (0.09 + f["bass"] * 0.04), bob=0.014, orbit=0.32 + f["section"] * 0.2, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.12)
        ps2fx.draw_quad_3d(frame, camera, [(-6.2, -1.16, camera.z + 1.2), (6.2, -1.16, camera.z + 1.2), (5.0, -1.16, camera.z + 14.0), (-5.0, -1.16, camera.z + 14.0)], ps2fx.lerp_color((176, 212, 210), palette.mid, 0.14), palette.ui, fog, 0.92)
        ps2fx.draw_quad_3d(frame, camera, [(-5.6, -1.05, camera.z + 2.0), (5.6, -1.05, camera.z + 2.0), (4.2, -1.05, camera.z + 13.5), (-4.2, -1.05, camera.z + 13.5)], ps2fx.lerp_color((124, 194, 208), palette.glow, 0.04), palette.glow, fog, 0.22)
        for x in (-3.8, -1.3, 1.3, 3.8):
            ps2fx.draw_cuboid_3d(frame, camera, (x, -0.06, camera.z + 4.8 + (x % 2.0) * 0.4), (0.3, 1.8, 0.3), ps2fx.lerp_color((160, 208, 220), palette.ui, 0.12), fog, edge=palette.glow)
        if self._is("corporate_fountain_core", "vapor_plaza"):
            self._draw_water_feature(frame, camera, palette, fog, t, f, center_z=camera.z + 5.8, width=3.2, depth=2.6, splash_scale=1.2)
            for z in (camera.z + 4.2, camera.z + 7.8, camera.z + 11.0):
                ps2fx.draw_quad_3d(frame, camera, [(-4.7, 0.42, z), (4.7, 0.42, z), (3.8, 0.16, z + 1.2), (-3.8, 0.16, z + 1.2)], ps2fx.lerp_color((194, 232, 240), palette.glow, 0.1), palette.ui, fog, 0.54)
            if self._is("vapor_plaza"):
                self._draw_people_group(frame, camera, palette, fog, t, f, crowd_center_z=camera.z + 6.2, rows=2, walking=False)
        elif self._is("dream_arcade_floor"):
            for y in range(0, h, 24):
                for x in range(0, w, 24):
                    if ((x // 24) + (y // 24)) % 2 == 0:
                        cv2.rectangle(frame, (x, max(int(h * 0.58), y)), (x + 24, min(h, y + 24)), ps2fx.bgr((28, 28, 42)), -1)
            for i in range(7):
                x = int(w * (0.06 + i * 0.13))
                cv2.rectangle(frame, (x, int(h * 0.3)), (x + 26, int(h * 0.54)), ps2fx.bgr((24, 24, 32)), -1)
                cv2.rectangle(frame, (x + 3, int(h * 0.34)), (x + 23, int(h * 0.44)), ps2fx.bgr(palette.accent), -1)
        else:
            self._draw_water_feature(frame, camera, palette, fog, t, f, center_z=camera.z + 6.0, width=2.7, depth=2.2, splash_scale=0.95)
        ps2fx.draw_reflection_streaks(frame, palette.glow, 0.1 + f["highs"] * 0.1, t)

    def _render_map(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        ps2fx.gradient_background(frame, palette.bg, palette.mid)
        camera = self._camera(t * 0.35, dolly=t * (0.06 + f["beat"] * 0.04), bob=0.02, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.12)
        if self._is("blue_os_desktop_world"):
            for z in (4.0, 7.0, 10.0, 13.0):
                ps2fx.draw_billboard_3d(frame, camera, (-2.6 + (z % 3), 0.8 - (z % 2) * 0.25, z), (1.5, 1.0), palette.ui, palette.glow, fog, 0.28)
                ps2fx.draw_billboard_3d(frame, camera, (-2.6 + (z % 3), 1.18 - (z % 2) * 0.25, z - 0.05), (1.5, 0.18), palette.accent, palette.accent, fog, 0.95)
            ps2fx.draw_quad_3d(frame, camera, [(-4.6, -1.05, 2.0), (4.6, -1.05, 2.0), (4.6, -1.05, 18.0), (-4.6, -1.05, 18.0)], palette.mid, palette.ui, fog, 0.8)
        elif self._is("satellite_weather_map"):
            ps2fx.draw_quad_3d(frame, camera, [(-4.2, -1.1, 3.0), (4.2, -1.1, 3.0), (4.2, -1.1, 18.0), (-4.2, -1.1, 18.0)], palette.mid, palette.ui, fog, 0.55)
            for z in (6.0, 9.0, 12.0):
                ps2fx.draw_billboard_3d(frame, camera, (0.0, 0.7, z), (2.3, 1.2), palette.ui, palette.glow, fog, 0.22)
            center = (int(w * 0.54), int(h * 0.48))
            for i in range(6):
                radius = 18 + i * 16
                angle = t * 0.5 + i * 0.4
                cv2.ellipse(frame, center, (radius, max(10, radius // 2)), angle * 20, 0, 320, ps2fx.bgr(palette.accent), 1)
        elif self._is("crt_navigation_map"):
            ps2fx.draw_horizon_grid(frame, palette.ui, f["beat"], t * 0.8)
            center = (w // 2, h // 2)
            for radius in range(40, min(w, h) // 2, 36):
                cv2.circle(frame, center, int(radius + f["bass"] * 16), ps2fx.bgr(palette.accent), 1)
            for i in range(12):
                ang = t * 0.4 + i * math.tau / 12.0
                x = center[0] + int(math.cos(ang) * w * 0.32)
                y = center[1] + int(math.sin(ang) * h * 0.26)
                cv2.line(frame, center, (x, y), ps2fx.bgr(palette.glow), 1)
        else:
            ps2fx.debug_begin_object("screen_hall")
            ps2fx.draw_corridor_planes(frame, camera, (38, 46, 52), (22, 28, 32), (18, 22, 26), fog, width=4.2, height=2.3, near_z=camera.z + 1.4, far_z=camera.z + 13.0)
            for z in np.linspace(camera.z + 2.0, camera.z + 11.0, 8):
                local_z = z
                for side in (-1, 1):
                    ps2fx.draw_billboard_3d(frame, camera, (side * 2.35, 0.3, local_z), (1.5, 1.0), palette.ui, palette.glow, fog, 0.75)
                    ps2fx.draw_billboard_3d(frame, camera, (side * 2.35, 0.72, local_z - 0.04), (1.5, 0.18), palette.accent, palette.accent, fog, 1.0)
            ps2fx.draw_billboard_3d(frame, camera, (-2.7, 0.38, camera.z + 2.0), (1.9, 1.3), palette.ui, palette.glow, fog, 0.82)
            ps2fx.draw_billboard_3d(frame, camera, (2.7, 0.46, camera.z + 2.6), (1.9, 1.3), palette.ui, palette.glow, fog, 0.82)
            ps2fx.debug_end_object()
        ps2fx.scanlines(frame, 0.18 + f["onset"] * 0.12)

    def _render_garden(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        h, w = frame.shape[:2]
        if self._is("green_wireframe_valley"):
            ps2fx.gradient_background(frame, palette.bg, palette.mid)
            horizon = int(h * 0.46)
            for y in range(horizon, h, 16):
                cv2.line(frame, (0, y), (w, y), ps2fx.bgr(palette.ui), 1)
            for x in range(0, w, 20):
                cv2.line(frame, (x, h), (w // 2 + int((x - w / 2) * 0.14), horizon), ps2fx.bgr(palette.ui), 1)
            ridge = []
            for i in range(10):
                px = int(i / 9 * w)
                py = int(horizon - 10 - math.sin(t * 0.18 + i * 0.7) * 14 - (i % 3) * 10)
                ridge.append((px, py))
            cv2.polylines(frame, [np.array(ridge, dtype=np.int32)], False, ps2fx.bgr(palette.accent), 2)
            cv2.circle(frame, (int(w * 0.76), int(h * 0.2)), 18, ps2fx.bgr(palette.glow), -1)
            return
        self._draw_open_sky(frame, palette, t, warm=0.06, sun_strength=0.72, cloud_density=1.15)
        camera = self._camera(t * 0.3, dolly=t * (0.06 + f["bass"] * 0.03), bob=0.012, orbit=0.26 + f["section"] * 0.18, f=f)
        fog = ps2fx.lerp_color(palette.mid, palette.glow, 0.12)
        self._draw_grass_plane(frame, camera, palette, fog, t, path=not self._is("polygon_garden"))
        self._draw_tree_cluster(frame, camera, palette, fog, t, [(-4.0, -0.48, camera.z + 3.4), (-1.8, -0.48, camera.z + 5.4), (1.5, -0.48, camera.z + 7.2), (3.9, -0.48, camera.z + 9.4)])
        if self._is("weather_simulation_room"):
            for z in (camera.z + 3.8, camera.z + 6.6, camera.z + 9.4):
                ps2fx.draw_billboard_3d(frame, camera, (-2.2, 1.1, z), (1.5, 0.84), palette.ui, palette.glow, fog, 0.22)
                ps2fx.draw_billboard_3d(frame, camera, (2.1, 1.0, z + 0.2), (1.6, 0.86), palette.accent, palette.glow, fog, 0.18)
                ps2fx.draw_portal_ring_3d(frame, camera, (0.0, 0.4, z + 0.2), 0.92, palette.glow, fog, y_scale=0.5, thickness=2)
        elif self._is("polygon_garden"):
            for z in (camera.z + 4.2, camera.z + 7.0, camera.z + 9.8):
                ps2fx.draw_quad_3d(frame, camera, [(-1.2, -0.7, z), (1.2, -0.7, z), (0.8, 0.55, z + 0.7), (-0.8, 0.55, z + 0.7)], ps2fx.lerp_color((194, 246, 236), palette.glow, 0.12), palette.ui, fog, 0.24)
            self._draw_people_group(frame, camera, palette, fog, t, f, crowd_center_z=camera.z + 5.2, rows=2, walking=False)
        else:
            for z in (camera.z + 3.6, camera.z + 6.8, camera.z + 9.6):
                ps2fx.draw_billboard_3d(frame, camera, (-2.6 + math.sin(t * 0.12 + z) * 0.1, 0.9, z), (1.2, 0.72), palette.ui, palette.glow, fog, 0.2)
                ps2fx.draw_billboard_3d(frame, camera, (2.4 + math.cos(t * 0.1 + z) * 0.1, 1.0, z + 0.2), (1.1, 0.66), palette.accent, palette.glow, fog, 0.18)
        ps2fx.draw_reflection_streaks(frame, palette.glow, 0.06 + f["highs"] * 0.06, t)

    def _render_geometry(self, frame: np.ndarray, t: float, f: dict[str, float], palette: Palette) -> None:
        ps2fx.gradient_background(frame, palette.bg, palette.mid)
        h, w = frame.shape[:2]
        for i in range(18):
            ang = t * 0.26 + i * 0.38
            cx = int(w * (0.5 + math.cos(ang) * 0.28))
            cy = int(h * (0.5 + math.sin(ang * 1.3) * 0.18))
            size = int(14 + (i % 4) * 10 + f["mids"] * 26)
            pts = np.array([[cx, cy - size], [cx + size, cy], [cx, cy + size], [cx - size, cy]], dtype=np.int32)
            cv2.polylines(frame, [pts], True, ps2fx.bgr(ps2fx.lerp_color(palette.accent, palette.ui, (i % 5) / 4.0)), 2)
        for i in range(5):
            x0 = int(w * (0.1 + i * 0.16))
            cv2.line(frame, (x0, int(h * 0.2)), (x0 + 40, int(h * 0.72)), ps2fx.bgr(palette.glow), 1)
        ps2fx.draw_particles(frame, palette.glow, t, 0.24 + f["highs"] * 0.28)
