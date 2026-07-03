from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    bg: tuple[int, int, int]
    mid: tuple[int, int, int]
    accent: tuple[int, int, int]
    glow: tuple[int, int, int]
    ui: tuple[int, int, int]


PALETTES: dict[str, Palette] = {
    "frutiger_aero": Palette((52, 88, 124), (106, 188, 226), (188, 248, 255), (236, 255, 250), (246, 255, 255)),
    "neon_nocturne": Palette((18, 34, 66), (66, 126, 206), (246, 142, 216), (188, 248, 255), (214, 244, 255)),
    "terminal_mist": Palette((26, 58, 64), (82, 176, 148), (170, 255, 220), (146, 246, 255), (226, 255, 240)),
    "bios_ocean": Palette((34, 58, 104), (82, 132, 214), (162, 234, 255), (246, 248, 255), (202, 238, 255)),
    "chrome_dream": Palette((52, 58, 74), (122, 142, 170), (206, 220, 236), (228, 250, 255), (255, 255, 255)),
    "sunken_plaza": Palette((28, 72, 92), (82, 152, 170), (150, 214, 210), (214, 252, 234), (234, 255, 246)),
    "vapor_civic": Palette((88, 126, 162), (154, 206, 238), (248, 250, 220), (226, 248, 255), (244, 255, 255)),
    "rain_alley": Palette((24, 34, 52), (92, 106, 162), (106, 236, 232), (255, 108, 198), (224, 240, 255)),
}


def palette_cycle() -> list[str]:
    return list(PALETTES.keys())
