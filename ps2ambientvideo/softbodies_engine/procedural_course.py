from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .audio_analysis import AudioFeatures


@dataclass(frozen=True)
class CourseSection:
    section_id: int
    start_time: float
    end_time: float
    intensity: float
    bass: float
    mid: float
    high: float
    brightness: float
    route_x: float
    route_y: float
    z_top: float
    z_bottom: float
    density: float
    theme: str


@dataclass(frozen=True)
class CourseObstacle:
    name: str
    kind: str
    feature_source: str
    time_seconds: float
    section_id: int
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation: tuple[float, float, float]
    color: tuple[float, float, float, float]
    influence_radius: float
    animated: bool = False


@dataclass(frozen=True)
class CourseRelease:
    body_name: str
    time_seconds: float
    trigger: str
    trigger_feature: str
    shape_kind: str = ""
    profile_name: str = ""
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class CourseLayout:
    seed: int
    audio_source: str
    duration: float
    route_points: tuple[tuple[float, float, float, float], ...]
    sections: tuple[CourseSection, ...]
    obstacles: tuple[CourseObstacle, ...]
    releases: tuple[CourseRelease, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "audio_path": self.audio_source,
            "duration": self.duration,
            "route_points": [
                {"time": point[0], "x": point[1], "y": point[2], "z": point[3]}
                for point in self.route_points
            ],
            "sections": [asdict(section) for section in self.sections],
            "obstacles": [asdict(obstacle) for obstacle in self.obstacles],
            "releases": [asdict(release) for release in self.releases],
        }


def _window_level(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    if times.size == 0 or values.size == 0:
        return 0.0
    mask = (times >= start) & (times < end)
    if np.any(mask):
        window = np.asarray(values[mask], dtype=np.float32)
        return float(np.clip(np.mean(window) * 0.45 + np.quantile(window, 0.86) * 0.55, 0.0, 1.0))
    mid = 0.5 * (start + end)
    return float(np.interp(mid, times, values))


def _brightness(mid: float, high: float) -> float:
    return float(np.clip(high * 0.68 + mid * 0.32, 0.0, 1.0))


def _theme_for_section(intensity: float, bass: float, brightness: float) -> str:
    if intensity < 0.16:
        return "reveal"
    if bass > 0.40:
        return "compression"
    if brightness > 0.30:
        return "spark"
    if intensity > 0.48:
        return "dense"
    return "glide"


def _route_for_sections(
    section_count: int,
    intensity_values: list[float],
    brightness_values: list[float],
    seed: int,
) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    current_x = -2.6
    current_y = -0.6
    for index in range(section_count + 1):
        src_index = min(index, section_count - 1)
        intensity = intensity_values[src_index]
        brightness = brightness_values[src_index]
        swing = 1.6 + intensity * 1.7
        x_target = np.clip(
            (rng.uniform(-1.0, 1.0) * 0.8 + np.sin(index * 1.17 + brightness * 2.6)) * swing,
            -3.9,
            3.9,
        )
        current_x = current_x * 0.38 + x_target * 0.62
        y_target = np.clip(
            np.cos(index * 0.83 + intensity * 2.4) * (0.9 + brightness * 0.8) + rng.uniform(-0.3, 0.3),
            -2.2,
            2.2,
        )
        current_y = current_y * 0.42 + y_target * 0.58
        points.append((float(current_x), float(current_y)))
    return points


def _color_for_feature(feature_source: str, brightness: float) -> tuple[float, float, float, float]:
    if feature_source == "beat":
        return (0.22, 0.86, 1.0, 1.0)
    if feature_source == "bass":
        return (1.0, 0.70 + 0.12 * brightness, 0.18, 1.0)
    if feature_source == "onset":
        return (1.0, 0.42, 0.58, 1.0)
    if feature_source == "high":
        return (0.76, 0.54, 1.0, 1.0)
    return (0.40, 0.96, 0.64, 1.0)


def _interp_route(route_points: tuple[tuple[float, float, float, float], ...], time_seconds: float) -> tuple[float, float, float]:
    if not route_points:
        return (0.0, 0.0, 0.0)
    t = float(np.clip(time_seconds, route_points[0][0], route_points[-1][0]))
    times = np.asarray([point[0] for point in route_points], dtype=np.float32)
    xs = np.asarray([point[1] for point in route_points], dtype=np.float32)
    ys = np.asarray([point[2] for point in route_points], dtype=np.float32)
    zs = np.asarray([point[3] for point in route_points], dtype=np.float32)
    return (
        float(np.interp(t, times, xs)),
        float(np.interp(t, times, ys)),
        float(np.interp(t, times, zs)),
    )


def _pick_release_time(
    candidates: np.ndarray,
    fallback: float,
    min_time: float,
    used: list[float],
) -> float:
    if candidates.size:
        for value in candidates:
            time_value = float(value)
            if time_value >= min_time and all(abs(time_value - prior) >= 1.0 for prior in used):
                return time_value
    return float(max(fallback, min_time))


def build_procedural_course_layout(
    features: AudioFeatures,
    duration: float,
    seed: int,
    body_names: tuple[str, ...],
    body_metadata: dict[str, dict[str, object]] | None = None,
) -> CourseLayout:
    duration = float(max(8.0, duration))
    section_count = int(np.clip(round(duration / 4.5), 6, 9))
    section_edges = np.linspace(0.0, duration, section_count + 1, dtype=np.float32)

    intensity_values: list[float] = []
    brightness_values: list[float] = []
    section_values: list[tuple[float, float, float, float, float]] = []
    for index in range(section_count):
        start = float(section_edges[index])
        end = float(section_edges[index + 1])
        rms = _window_level(features.times, features.rms, start, end)
        bass = _window_level(features.times, features.bass, start, end)
        mid = _window_level(features.times, features.mid, start, end)
        high = _window_level(features.times, features.high, start, end)
        brightness = _brightness(mid, high)
        intensity = float(np.clip(rms * 0.42 + bass * 0.34 + high * 0.24, 0.0, 1.0))
        intensity_values.append(intensity)
        brightness_values.append(brightness)
        section_values.append((rms, bass, mid, high, brightness))

    route_xy = _route_for_sections(section_count, intensity_values, brightness_values, seed)
    z_start = 17.5
    z_end = -32.0
    route_points: list[tuple[float, float, float, float]] = []
    for index, edge in enumerate(section_edges):
        progress = index / max(section_count, 1)
        x_pos, y_pos = route_xy[index]
        z_pos = z_start + (z_end - z_start) * progress
        route_points.append((float(edge), float(x_pos), float(y_pos), float(z_pos)))

    sections: list[CourseSection] = []
    for index in range(section_count):
        start = float(section_edges[index])
        end = float(section_edges[index + 1])
        rms, bass, mid, high, brightness = section_values[index]
        intensity = intensity_values[index]
        density = float(np.clip(0.38 + intensity * 0.84 + brightness * 0.28, 0.38, 1.48))
        theme = _theme_for_section(intensity, bass, brightness)
        sections.append(
            CourseSection(
                section_id=index,
                start_time=start,
                end_time=end,
                intensity=intensity,
                bass=bass,
                mid=mid,
                high=high,
                brightness=brightness,
                route_x=route_points[index][1],
                route_y=route_points[index][2],
                z_top=route_points[index][3],
                z_bottom=route_points[index + 1][3],
                density=density,
                theme=theme,
            )
        )

    obstacle_list: list[CourseObstacle] = []
    rng = np.random.default_rng(seed * 17 + 5)
    beat_times = np.asarray(features.beat_times, dtype=np.float32)
    onset_mask = features.onset >= max(0.58, float(np.quantile(features.onset, 0.88))) if features.onset.size else np.zeros((0,), dtype=bool)
    onset_times = features.times[onset_mask] if onset_mask.size else np.zeros((0,), dtype=np.float32)

    for section in sections:
        section_beats = beat_times[(beat_times >= section.start_time) & (beat_times < section.end_time)]
        section_onsets = onset_times[(onset_times >= section.start_time) & (onset_times < section.end_time)]
        local_major_beats = section_beats[:: max(1, len(section_beats) // 3 or 1)][:3]
        local_onsets = section_onsets[:: max(1, len(section_onsets) // 2 or 1)][:2]

        anchor_time = 0.5 * (section.start_time + section.end_time)
        anchor_x, anchor_y, anchor_z = _interp_route(tuple(route_points), anchor_time)
        if section.bass > 0.05:
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-bumper",
                    kind="sphere",
                    feature_source="bass",
                    time_seconds=anchor_time,
                    section_id=section.section_id,
                    position=(anchor_x + rng.uniform(-0.8, 0.8), anchor_y + rng.uniform(-0.5, 0.5), anchor_z - 0.7),
                    size=(0.62 + section.bass * 0.58, 0.62 + section.bass * 0.58, 0.62 + section.bass * 0.58),
                    rotation=(0.0, 0.0, 0.0),
                    color=_color_for_feature("bass", section.brightness),
                    influence_radius=1.25 + section.bass * 0.72,
                    animated=section.bass > 0.26,
                )
            )
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-squeeze-left",
                    kind="box",
                    feature_source="bass",
                    time_seconds=anchor_time + 0.08,
                    section_id=section.section_id,
                    position=(anchor_x - (1.15 + section.bass * 0.38), anchor_y + 0.32, anchor_z - 0.15),
                    size=(0.56 + section.bass * 0.18, 0.16, 0.22),
                    rotation=(0.0, 0.0, 18.0 + section.intensity * 10.0),
                    color=_color_for_feature("bass", section.brightness),
                    influence_radius=1.08 + section.bass * 0.38,
                    animated=section.bass > 0.24,
                )
            )
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-squeeze-right",
                    kind="box",
                    feature_source="bass",
                    time_seconds=anchor_time + 0.12,
                    section_id=section.section_id,
                    position=(anchor_x + (1.15 + section.bass * 0.38), anchor_y - 0.32, anchor_z - 0.18),
                    size=(0.56 + section.bass * 0.18, 0.16, 0.22),
                    rotation=(0.0, 0.0, -(18.0 + section.intensity * 10.0)),
                    color=_color_for_feature("bass", section.brightness),
                    influence_radius=1.08 + section.bass * 0.38,
                    animated=section.bass > 0.24,
                )
            )

        for beat_index, beat_time in enumerate(local_major_beats):
            route_x, route_y, route_z = _interp_route(tuple(route_points), float(beat_time))
            side = -1.0 if (beat_index + section.section_id) % 2 == 0 else 1.0
            gate_kind = "box" if beat_index % 2 == 0 else "sphere"
            if gate_kind == "box":
                size = (
                    0.88 + section.intensity * 0.34,
                    0.14 + section.mid * 0.08,
                    0.14 + section.bass * 0.06,
                )
                rotation = (0.0, 0.0, side * (22.0 + 18.0 * section.intensity))
                offset = (side * (1.05 + 0.28 * section.brightness), 0.12 * side, -0.25)
                influence = 1.18 + section.intensity * 0.42
            else:
                size = (
                    0.46 + section.intensity * 0.26,
                    0.46 + section.intensity * 0.26,
                    0.46 + section.intensity * 0.26,
                )
                rotation = (0.0, 0.0, 0.0)
                offset = (side * (1.15 + 0.22 * section.mid), -0.18 * side, -0.10)
                influence = 0.98 + section.intensity * 0.34
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-beat-{beat_index}",
                    kind=gate_kind,
                    feature_source="beat",
                    time_seconds=float(beat_time),
                    section_id=section.section_id,
                    position=(route_x + offset[0], route_y + offset[1], route_z + offset[2]),
                    size=size,
                    rotation=rotation,
                    color=_color_for_feature("beat", section.brightness),
                    influence_radius=influence,
                    animated=True,
                )
            )

        for onset_index, onset_time in enumerate(local_onsets):
            route_x, route_y, route_z = _interp_route(tuple(route_points), float(onset_time))
            side = 1.0 if (onset_index + section.section_id) % 2 == 0 else -1.0
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-onset-{onset_index}",
                    kind="box",
                    feature_source="onset",
                    time_seconds=float(onset_time),
                    section_id=section.section_id,
                    position=(route_x - side * 0.95, route_y + side * 0.42, route_z - 0.12),
                    size=(0.90 + section.mid * 0.28, 0.16 + section.high * 0.06, 0.14 + section.mid * 0.04),
                    rotation=(0.0, 0.0, side * (34.0 + section.high * 20.0)),
                    color=_color_for_feature("onset", section.brightness),
                    influence_radius=1.05 + section.mid * 0.34,
                    animated=True,
                )
            )

        if section.high > 0.008:
            route_x, route_y, route_z = _interp_route(tuple(route_points), anchor_time + 0.18)
            obstacle_list.append(
                CourseObstacle(
                    name=f"section-{section.section_id}-spark-rail",
                    kind="rail",
                    feature_source="high",
                    time_seconds=anchor_time + 0.18,
                    section_id=section.section_id,
                    position=(route_x, route_y + 1.3 * np.sign(np.sin(section.section_id + 0.5)), route_z - 0.35),
                    size=(1.18 + section.high * 0.36, 0.07, 0.09),
                    rotation=(0.0, 0.0, 0.0),
                    color=_color_for_feature("high", section.brightness),
                    influence_radius=0.92,
                    animated=section.high > 0.55,
                )
            )

    release_targets = [0.0, duration * 0.10, duration * 0.22, duration * 0.36, duration * 0.52]
    candidate_onsets = onset_times
    candidate_beats = beat_times
    used_release_times: list[float] = []
    releases: list[CourseRelease] = []
    trigger_cycle = ("start", "onset", "beat", "phrase", "transition")
    for index, body_name in enumerate(body_names):
        min_time = 0.0 if index == 0 else used_release_times[-1] + 0.75
        if index == 0:
            release_time = 0.0
            trigger = "start"
        elif index % 2 == 1:
            release_time = _pick_release_time(candidate_onsets, release_targets[index], min_time, used_release_times)
            trigger = trigger_cycle[index]
        else:
            release_time = _pick_release_time(candidate_beats, release_targets[index], min_time, used_release_times)
            trigger = trigger_cycle[index]
        used_release_times.append(release_time)
        metadata = (body_metadata or {}).get(body_name, {})
        releases.append(
            CourseRelease(
                body_name=body_name,
                time_seconds=release_time,
                trigger=trigger,
                trigger_feature=trigger,
                shape_kind=str(metadata.get("shape_kind", "")),
                profile_name=str(metadata.get("profile_name", "")),
                color=tuple(metadata.get("color", (1.0, 1.0, 1.0, 1.0))),
            )
        )

    return CourseLayout(
        seed=seed,
        audio_source=features.source,
        duration=duration,
        route_points=tuple(route_points),
        sections=tuple(sections),
        obstacles=tuple(obstacle_list),
        releases=tuple(releases),
    )


def sample_route_position(layout: CourseLayout, time_seconds: float) -> tuple[float, float, float]:
    return _interp_route(layout.route_points, time_seconds)
