from __future__ import annotations

from .base import BaseMode, ModeConfig


def make_mode(
    name: str,
    scene: str,
    palette_hint: str,
    *,
    speed: float = 1.0,
    density: float = 1.0,
    structure: float = 1.0,
    atmosphere: float = 1.0,
    family: str = "",
    scene_recipe: str = "",
    dominant_geometry: tuple[str, ...] = (),
    movement_type: tuple[str, ...] = (),
    camera_roles: tuple[str, ...] = (),
    suitable_presets: tuple[str, ...] = (),
) -> type[BaseMode]:
    class GeneratedMode(BaseMode):
        def __init__(self) -> None:
            super().__init__(
                ModeConfig(
                    name=name,
                    scene=scene,
                    palette_hint=palette_hint,
                    speed=speed,
                    density=density,
                    structure=structure,
                    atmosphere=atmosphere,
                    family=family,
                    scene_recipe=scene_recipe or name,
                    dominant_geometry=dominant_geometry,
                    movement_type=movement_type,
                    camera_roles=camera_roles,
                    suitable_presets=suitable_presets,
                )
            )

    GeneratedMode.__name__ = "".join(part.capitalize() for part in name.split("_"))
    return GeneratedMode
