from __future__ import annotations

import os
import unittest
from unittest import mock

from ps2ambientvideo.cli import build_parser
from ps2ambientvideo.blender_backend import _find_blender
from ps2ambientvideo.mode_manifest import MODE_MANIFEST
from ps2ambientvideo.timeline import build_timeline


class ModeCatalogTest(unittest.TestCase):
    def test_catalog_has_at_least_140_scenes(self) -> None:
        self.assertGreaterEqual(len(MODE_MANIFEST), 140)

    def test_lofi_uses_expanded_scene_pool_for_long_duration(self) -> None:
        timeline = build_timeline(330.0, preset="lofi", scene_grammar="worlds")
        names = [segment.name for segment in timeline]
        self.assertGreaterEqual(len(set(names)), 8)
        self.assertIn("floating_sky_plaza", names)
        self.assertIn("splash_fountain_court", names)
        self.assertIn("orbital_garden_station", names)
        self.assertIn("memory_beach", names)

    def test_frutiger_world_preset_hits_new_world_families(self) -> None:
        timeline = build_timeline(120.0, preset="frutiger_world", scene_grammar="worlds")
        names = [segment.name for segment in timeline]
        self.assertIn("floating_sky_plaza", names)
        self.assertIn("polygon_grass_field", names)
        self.assertIn("splash_fountain_court", names)
        self.assertIn("orbital_garden_station", names)

    def test_worlds_gallery_presets_cover_distinct_families(self) -> None:
        distinctness = [segment.name for segment in build_timeline(90.0, preset="worlds_distinctness_gallery", scene_grammar="worlds")]
        self.assertIn("floating_sky_plaza", distinctness)
        self.assertIn("polygon_grass_field", distinctness)
        self.assertIn("splash_fountain_court", distinctness)
        self.assertIn("orbital_garden_station", distinctness)
        self.assertIn("fountain_plaza_dancers", distinctness)
        self.assertIn("night_market_grid", distinctness)
        self.assertIn("glass_greenhouse", distinctness)

        people_water = [segment.name for segment in build_timeline(90.0, preset="worlds_people_water_gallery", scene_grammar="worlds")]
        self.assertIn("dancing_polygon_crowd", people_water)
        self.assertIn("water_jet_plaza", people_water)
        self.assertIn("waterfall_atrium", people_water)

        space_sky = [segment.name for segment in build_timeline(90.0, preset="worlds_space_sky_gallery", scene_grammar="worlds")]
        self.assertIn("cloud_garden", space_sky)
        self.assertIn("solar_sail_promenade", space_sky)
        self.assertIn("moon_pool_with_stars", space_sky)

    def test_lofi_resolves_under_legacy_plaza(self) -> None:
        timeline = build_timeline(120.0, preset="lofi", scene_grammar="legacy_plaza")
        names = [segment.name for segment in timeline]
        self.assertIn("airport_moving_walkway", names)
        self.assertIn("glass_mall_atrium", names)
        self.assertIn("memory_beach", names)

    def test_city_promise_alias_resolves_to_lofi(self) -> None:
        timeline = build_timeline(120.0, preset="city_promise", scene_grammar="worlds")
        names = [segment.name for segment in timeline]
        self.assertIn("blue_sky_aero_terminal", names)
        self.assertIn("ocean_glass_terrace", names)
        self.assertIn("orbital_garden_station", names)

    def test_cli_accepts_scene_grammar(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--preset",
                "lofi",
                "--scene-grammar",
                "legacy_plaza",
            ]
        )
        self.assertEqual(args.scene_grammar, "legacy_plaza")

    def test_cli_accepts_render_engine(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "blender",
            ]
        )
        self.assertEqual(args.render_engine, "blender")

    def test_cli_accepts_softbodies_render_engine(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "softbodies",
                "--softbodies-scene",
                "translucent",
                "--softbody-preset",
                "stable_medium",
                "--softbody-visualization",
                "shaded",
            ]
        )
        self.assertEqual(args.render_engine, "softbodies")
        self.assertEqual(args.softbodies_scene, "translucent")
        self.assertEqual(args.softbody_preset, "stable_medium")
        self.assertEqual(args.softbody_visualization, "shaded")

    def test_cli_accepts_lowpoly_aesthetic_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--aesthetic",
                "lowpoly",
            ]
        )
        self.assertEqual(args.aesthetic, "lowpoly")

    def test_cli_accepts_retro_clean_compat_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--aesthetic",
                "retro_clean",
            ]
        )
        self.assertEqual(args.aesthetic, "lowpoly")

    def test_cli_accepts_blender_proof_stills(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "blender",
                "--blender-proof-stills",
            ]
        )
        self.assertTrue(args.blender_proof_stills)

    def test_cli_accepts_blender_smoke_scene(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "blender",
                "--blender-smoke-scene",
                "cube",
            ]
        )
        self.assertEqual(args.blender_smoke_scene, "cube")

    def test_cli_accepts_blender_quality(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "blender",
                "--blender-quality",
                "smoke",
            ]
        )
        self.assertEqual(args.blender_quality, "smoke")

    def test_cli_accepts_blender_diagnostic_engine(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render",
                "input.wav",
                "--output",
                "output.mp4",
                "--render-engine",
                "blender",
                "--blender-diagnostic-engine",
                "workbench",
            ]
        )
        self.assertEqual(args.blender_diagnostic_engine, "workbench")

    def test_find_blender_honors_environment_override(self) -> None:
        with mock.patch.dict(os.environ, {"HIGH_TRUST_RENDERER_BLENDER": __file__}, clear=False):
            self.assertEqual(_find_blender(), __file__)


if __name__ == "__main__":
    unittest.main()
