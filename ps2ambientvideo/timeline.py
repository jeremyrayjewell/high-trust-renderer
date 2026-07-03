from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mode_manifest import MODE_MANIFEST
from .modes import ALL_MODE_CLASSES
from .palettes import PALETTES


@dataclass(frozen=True)
class ScheduledMode:
    name: str
    start: float
    end: float
    fade_in: float
    fade_out: float
    palette_name: str
    shot_role: str = "forward_travel"

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) * 0.5


def normalize_preset_name(preset: str) -> str:
    aliases = {
        "city_promise": "lofi",
        "ps2_bios_dream": "bios_dream",
    }
    return aliases.get(preset, preset)


def build_modes(scene_grammar: str = "worlds"):
    modes = [mode_cls() for mode_cls in ALL_MODE_CLASSES]
    for mode in modes:
        if hasattr(mode, "set_scene_grammar"):
            mode.set_scene_grammar(scene_grammar)
    return modes


def _weight_for_segment(t: float, segment: ScheduledMode) -> float:
    if t < segment.start or t > segment.end:
        return 0.0
    weight = 1.0
    if segment.fade_in > 0:
        weight *= np.clip((t - segment.start) / segment.fade_in, 0.0, 1.0)
    if segment.fade_out > 0:
        weight *= np.clip((segment.end - t) / segment.fade_out, 0.0, 1.0)
    return float(weight)


def _mode_names() -> list[str]:
    return [cls().name for cls in ALL_MODE_CLASSES]


def _sequence_timeline(
    duration: float,
    mode_names: list[str | tuple[str, str]],
    segment_length: float,
    overlap: float,
) -> list[ScheduledMode]:
    timeline: list[ScheduledMode] = []
    cursor = 0.0
    step = max(0.5, segment_length - overlap)
    for entry in mode_names:
        if isinstance(entry, tuple):
            name, shot_role = entry
        else:
            name, shot_role = entry, "forward_travel"
        palette_name = MODE_MANIFEST[name].palette
        start = max(0.0, cursor)
        end = min(duration, cursor + segment_length)
        timeline.append(
            ScheduledMode(
                name=name,
                start=start,
                end=end,
                fade_in=0.0 if not timeline else overlap,
                fade_out=0.0,
                palette_name=palette_name,
                shot_role=shot_role,
            )
        )
        cursor += step
        if cursor >= duration:
            break
    if not timeline:
        return []
    for index, segment in enumerate(timeline):
        fade_out = overlap if index < len(timeline) - 1 else 0.0
        timeline[index] = ScheduledMode(
            name=segment.name,
            start=segment.start,
            end=duration if index == len(timeline) - 1 else segment.end,
            fade_in=segment.fade_in,
            fade_out=fade_out,
            palette_name=segment.palette_name,
            shot_role=segment.shot_role,
        )
    return timeline


def _lofi_timeline_legacy(duration: float) -> list[ScheduledMode]:
    if duration <= 36.0:
        mode_names: list[str | tuple[str, str]] = [
            ("airport_moving_walkway", "diagonal_glass_dolly"),
            ("memory_train", "close_foreground_pass"),
            ("glass_mall_atrium", "low_reflective_glide"),
            ("skybridge_city", "side_tracking_towers"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(5.2, duration / 3.8)
        overlap = min(2.2, segment_length * 0.28)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration <= 60.0:
        mode_names = [
            ("airport_moving_walkway", "diagonal_glass_dolly"),
            ("memory_train", "close_foreground_pass"),
            ("glass_mall_atrium", "low_reflective_glide"),
            ("skybridge_city", "side_tracking_towers"),
            ("corporate_fountain_core", "wide_orbit_core"),
            ("digital_aquarium", "slow_orbit_around_object"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(7.0, duration / 4.7)
        overlap = min(2.6, segment_length * 0.24)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration >= 180.0:
        mode_names = [
            ("airport_moving_walkway", "sweeping_crane_rise"),
            ("memory_train", "close_foreground_pass"),
            ("glass_mall_atrium", "low_reflective_glide"),
            ("glass_elevator_shaft", "vertical_elevator_rise"),
            ("skybridge_city", "side_tracking_towers"),
            ("neon_monorail", "fast_forward_flythrough"),
            ("vapor_plaza", "pullback_to_city"),
            ("corporate_fountain_core", "wide_orbit_core"),
            ("shopping_mall_fountain_court", "diagonal_glass_dolly"),
            ("city_hall_plaza", "sweeping_crane_rise"),
            ("digital_aquarium", "slow_orbit_around_object"),
            ("ocean_interface", "long_horizon_glide"),
            ("riverside_promenade", "long_horizon_glide"),
            ("moon_pool", "horizon_water_drift"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(25.0, duration / 12.4)
        overlap = min(7.0, segment_length * 0.2)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration >= 85.0:
        mode_names = [
            ("airport_moving_walkway", "sweeping_crane_rise"),
            ("memory_train", "close_foreground_pass"),
            ("glass_mall_atrium", "low_reflective_glide"),
            ("skybridge_city", "side_tracking_towers"),
            ("neon_monorail", "fast_forward_flythrough"),
            ("corporate_fountain_core", "wide_orbit_core"),
            ("digital_aquarium", "slow_orbit_around_object"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(11.0, duration / 6.8)
        overlap = min(3.6, segment_length * 0.22)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    mode_names = [
        ("airport_moving_walkway", "diagonal_glass_dolly"),
        ("memory_train", "close_foreground_pass"),
        ("glass_mall_atrium", "low_reflective_glide"),
        ("skybridge_city", "side_tracking_towers"),
        ("corporate_fountain_core", "wide_orbit_core"),
        ("digital_aquarium", "slow_orbit_around_object"),
        ("memory_beach", "horizon_water_drift"),
    ]
    return _sequence_timeline(duration, mode_names, 12.0, 3.0)


def _lofi_timeline_worlds(duration: float) -> list[ScheduledMode]:
    if duration <= 36.0:
        mode_names: list[str | tuple[str, str]] = [
            ("blue_sky_aero_terminal", "sweeping_arc"),
            ("polygon_grass_field", "low_flythrough"),
            ("walking_silhouette_commuters", "side_orbit_reveal"),
            ("dancing_polygon_crowd", "orbit_subject"),
            ("splash_fountain_court", "dive_to_fountain"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(5.2, duration / 3.8)
        overlap = min(2.2, segment_length * 0.28)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration <= 60.0:
        mode_names = [
            ("blue_sky_aero_terminal", "sweeping_arc"),
            ("polygon_grass_field", "low_flythrough"),
            ("walking_silhouette_commuters", "side_orbit_reveal"),
            ("dancing_polygon_crowd", "orbit_subject"),
            ("splash_fountain_court", "water_surface_flyover"),
            ("glass_greenhouse", "side_orbit_reveal"),
            ("orbital_garden_station", "rising_spiral"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(7.0, duration / 4.7)
        overlap = min(2.6, segment_length * 0.24)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration >= 180.0:
        mode_names = [
            ("blue_sky_aero_terminal", "sweeping_arc"),
            ("polygon_grass_field", "low_flythrough"),
            ("floating_sky_plaza", "sweeping_arc"),
            ("polygon_grass_field", "low_flythrough"),
            ("dancing_lawn_plaza", "orbit_subject"),
            ("splash_fountain_court", "water_surface_flyover"),
            ("fountain_plaza_dancers", "orbit_subject"),
            ("reflective_pool_garden", "low_reflective_glide"),
            ("white_civic_skybridge", "side_orbit_reveal"),
            ("glass_greenhouse", "side_orbit_reveal"),
            ("orbital_garden_station", "rising_spiral"),
            ("solar_sail_promenade", "sweeping_panorama_deck"),
            ("planet_view_observation_deck", "orbit_subject"),
            ("ocean_glass_terrace", "water_surface_flyover"),
            ("moon_pool_with_stars", "horizon_water_drift"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(25.0, duration / 12.4)
        overlap = min(7.0, segment_length * 0.2)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    if duration >= 85.0:
        mode_names = [
            ("blue_sky_aero_terminal", "sweeping_arc"),
            ("polygon_grass_field", "low_flythrough"),
            ("walking_silhouette_commuters", "side_orbit_reveal"),
            ("dancing_polygon_crowd", "orbit_subject"),
            ("splash_fountain_court", "water_surface_flyover"),
            ("glass_greenhouse", "side_orbit_reveal"),
            ("glass_elevator_shaft", "rising_spiral"),
            ("orbital_garden_station", "rising_spiral"),
            ("ocean_glass_terrace", "water_surface_flyover"),
            ("memory_beach", "horizon_water_drift"),
        ]
        segment_length = max(11.0, duration / 6.8)
        overlap = min(3.6, segment_length * 0.22)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)
    mode_names = [
        ("floating_sky_plaza", "sweeping_arc"),
        ("polygon_grass_field", "low_flythrough"),
        ("walking_silhouette_commuters", "side_orbit_reveal"),
        ("dancing_polygon_crowd", "orbit_subject"),
        ("water_jet_plaza", "water_surface_flyover"),
        ("glass_greenhouse", "side_orbit_reveal"),
        ("planet_view_observation_deck", "orbit_subject"),
        ("memory_beach", "horizon_water_drift"),
    ]
    return _sequence_timeline(duration, mode_names, 12.0, 3.0)


def _curated_presets() -> dict[str, tuple[list[str | tuple[str, str]], float, float]]:
    return {
        "showcase_30s": (
            ["endless_highway", "ring_corridor", "liquid_chrome_room", "bios_temple", "memory_beach"],
            6.5,
            1.5,
        ),
        "depth_showcase": (
            [
                "endless_highway",
                "memory_train",
                "dream_metro",
                "neon_monorail",
                "skybridge_city",
                "glass_mall_atrium",
                "virtual_hotel",
                "tunnel_of_screens",
                "data_center_dream",
                "memory_beach",
            ],
            6.5,
            1.4,
        ),
        "mode_gallery": (_mode_names(), 5.0, 1.2),
        "mode_gallery_large": (_mode_names(), 4.5, 1.0),
        "polygon_motion_gallery": (
            [
                ("polyhedron_swarm_corridor", "spiral_approach"),
                ("rotating_prism_plaza", "slow_orbit_around_object"),
                ("floating_cube_city", "wide_crane_rise"),
                ("polygon_ring_machine", "spiral_approach"),
                ("reflective_shard_basin", "close_object_pass"),
                ("server_cathedral", "vertical_elevator_rise"),
                ("floating_water_panels", "slow_orbit_around_object"),
                ("orbiting_relay_station", "slow_orbit_around_object"),
                ("ribbon_road", "diagonal_dolly"),
                ("crystal_tunnel", "low_flythrough"),
            ],
            7.0,
            1.6,
        ),
        "camera_direction_gallery": (
            [
                ("mountain_tram", "side_tracking_shot"),
                ("glass_elevator_shaft", "vertical_elevator_rise"),
                ("city_hall_plaza", "wide_crane_rise"),
                ("polyhedron_swarm_corridor", "spiral_approach"),
                ("riverside_promenade", "long_horizon_glide"),
                ("telecom_tower_field", "high_overlook_drift"),
                ("server_cathedral", "vertical_elevator_rise"),
                ("orbiting_relay_station", "slow_orbit_around_object"),
            ],
            6.5,
            1.5,
        ),
        "lofi": ([], 12.0, 3.0),
        "frutiger_world": (
            [
                ("floating_sky_plaza", "sweeping_crane_rise"),
                ("blue_sky_aero_terminal", "diagonal_glass_dolly"),
                ("polygon_grass_field", "low_flythrough"),
                ("dancing_lawn_plaza", "wide_orbit_core"),
                ("splash_fountain_court", "dive_to_fountain"),
                ("reflective_pool_garden", "low_reflective_glide"),
                ("orbital_garden_station", "sweeping_panorama_deck"),
                ("planet_view_observation_deck", "slow_orbit_around_object"),
                ("ocean_glass_terrace", "horizon_water_drift"),
                ("memory_beach", "horizon_water_drift"),
            ],
            10.5,
            2.3,
        ),
        "worlds_distinctness_gallery": (
            [
                ("floating_sky_plaza", "sweeping_crane_rise"),
                ("polygon_grass_field", "low_flythrough"),
                ("splash_fountain_court", "dive_to_fountain"),
                ("orbital_garden_station", "sweeping_panorama_deck"),
                ("fountain_plaza_dancers", "wide_orbit_core"),
                ("night_market_grid", "side_tracking_shot"),
                ("glass_greenhouse", "vertical_elevator_rise"),
            ],
            11.5,
            2.2,
        ),
        "worlds_people_water_gallery": (
            [
                ("dancing_polygon_crowd", "wide_orbit_core"),
                ("splash_fountain_court", "dive_to_fountain"),
                ("walking_silhouette_commuters", "pullback_people_skyline"),
                ("water_jet_plaza", "low_reflective_glide"),
                ("fountain_plaza_dancers", "wide_orbit_core"),
                ("waterfall_atrium", "vertical_elevator_rise"),
                ("low_poly_celebration_circle", "pullback_people_skyline"),
            ],
            11.0,
            2.2,
        ),
        "worlds_space_sky_gallery": (
            [
                ("floating_sky_plaza", "sweeping_crane_rise"),
                ("cloud_garden", "wide_crane_rise"),
                ("blue_sky_aero_terminal", "diagonal_glass_dolly"),
                ("orbital_garden_station", "sweeping_panorama_deck"),
                ("solar_sail_promenade", "slow_orbit_around_object"),
                ("planet_view_observation_deck", "pullback_to_city"),
                ("moon_pool_with_stars", "horizon_water_drift"),
            ],
            11.0,
            2.2,
        ),
        "worlds_material_proof": (
            [
                ("blue_sky_aero_terminal", "sweeping_arc"),
                ("dancing_polygon_crowd", "orbit_subject"),
                ("splash_fountain_court", "water_surface_flyover"),
                ("glass_greenhouse", "side_orbit_reveal"),
                ("glass_elevator_shaft", "rising_spiral"),
                ("orbital_garden_station", "orbit_subject"),
                ("ocean_glass_terrace", "water_surface_flyover"),
            ],
            8.5,
            1.8,
        ),
        "cyber_aero_liquid": (
            [
                ("glass_elevator_shaft", "vertical_elevator_rise"),
                ("digital_aquarium", "side_tracking_shot"),
                ("floating_water_panels", "slow_orbit_around_object"),
                ("cyber_shrine", "spiral_approach"),
                ("waterwall_data_lounge", "close_object_pass"),
                ("hologram_control_room", "slow_orbit_around_object"),
                ("polygon_ring_machine", "spiral_approach"),
            ],
            10.0,
            2.3,
        ),
        "night_transit": (
            [
                ("highway_toll_gates", "reverse_pullback"),
                ("memory_train", "close_foreground_pass"),
                ("airport_moving_walkway", "diagonal_dolly"),
                ("dream_metro", "low_flythrough"),
                ("neon_monorail", "side_tracking_shot"),
                ("tunnel_interchange", "spiral_approach"),
                ("parking_garage_descent", "reverse_pullback"),
            ],
            10.5,
            2.0,
        ),
        "frutiger_water": (
            [
                ("glass_elevator_shaft", "vertical_elevator_rise"),
                ("aquarium_tunnel", "low_flythrough"),
                ("indoor_waterfall_lobby", "wide_crane_rise"),
                ("corporate_wellness_pool", "long_horizon_glide"),
                ("storm_glass_boardwalk", "crossing_parallax_pan"),
                ("blue_sky_ui_garden", "high_overlook_drift"),
            ],
            10.5,
            2.1,
        ),
        "bios_dream": (
            [
                ("bios_temple", "wide_establishing"),
                ("blue_os_desktop_world", "vertical_elevator_rise"),
                ("crt_navigation_map", "calm_horizon_hold"),
                ("signal_bridge_archive", "side_tracking_shot"),
                ("radar_weather_dome", "spiral_approach"),
                ("datastream_viaduct", "low_flythrough"),
            ],
            10.5,
            2.0,
        ),
        "high_trust_society": (
            [
                ("glass_bridge_lobby", "wide_crane_rise"),
                ("corporate_fountain_core", "slow_orbit_around_object"),
                ("wellness_conservatory", "high_overlook_drift"),
                ("server_cathedral", "vertical_elevator_rise"),
                ("virtual_hotel", "reverse_pullback"),
                ("city_hall_plaza", "wide_crane_rise"),
            ],
            10.5,
            2.0,
        ),
    }


def build_timeline(duration: float, preset: str = "full_dream_cycle", scene_grammar: str = "worlds") -> list[ScheduledMode]:
    preset = normalize_preset_name(preset)
    if preset == "geometry_calibration":
        return [
            ScheduledMode(
                name="geometry_calibration",
                start=0.0,
                end=duration,
                fade_in=0.0,
                fade_out=0.0,
                palette_name="bios_ocean",
            )
        ]
    curated = _curated_presets()
    if preset in curated:
        if preset == "lofi":
            if scene_grammar == "legacy_plaza":
                return _lofi_timeline_legacy(duration)
            return _lofi_timeline_worlds(duration)
        mode_names, segment_length, overlap = curated[preset]
        if preset in {"mode_gallery", "mode_gallery_large"} and mode_names:
            step = max(4.0, min(5.0, duration / len(mode_names)))
            overlap = 1.0
            segment_length = min(6.0, step + overlap)
        return _sequence_timeline(duration, mode_names, segment_length, overlap)

    if preset != "full_dream_cycle":
        raise ValueError(f"Unknown preset: {preset}")

    names = _mode_names()
    segment_len = max(6.0, min(16.0, duration / max(8, len(names) // 2)))
    overlap = min(segment_len * 0.45, 4.0)
    return _sequence_timeline(duration, names, segment_len, overlap)


def debug_targets_for_timeline(duration: float, preset: str, timeline: list[ScheduledMode]) -> list[float]:
    preset = normalize_preset_name(preset)
    if preset == "showcase_30s":
        targets = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 29.0]
        return [min(duration, value) for value in targets if value <= duration + 1e-6]
    if preset == "depth_showcase":
        return [min(duration, segment.midpoint) for segment in timeline]
    if preset == "geometry_calibration":
        return [min(duration, value) for value in (1.5, 3.5, 5.5, 7.5) if value <= duration + 1e-6]
    if preset in {"mode_gallery", "mode_gallery_large"}:
        return [min(duration, segment.midpoint) for segment in timeline]
    if preset in _curated_presets():
        return [min(duration, segment.midpoint) for segment in timeline]
    if duration <= 0:
        return []
    fractions = [0.15, 0.35, 0.55, 0.75, 0.92]
    return sorted({round(duration * value, 2) for value in fractions})


def active_modes(t: float, timeline: list[ScheduledMode]) -> list[tuple[ScheduledMode, float]]:
    active = []
    for segment in timeline:
        weight = _weight_for_segment(t, segment)
        if weight > 0.001:
            active.append((segment, weight))
    total = sum(weight for _, weight in active)
    if total <= 1e-6:
        return []
    return [(segment, weight / total) for segment, weight in active]


def palette_for(name: str):
    return PALETTES[name]
