from __future__ import annotations

import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import cv2
import numpy as np

from ps2ambientvideo.blender_backend import _find_blender
from ps2ambientvideo.renderer import render_video


class VisualSmokeTest(unittest.TestCase):
    def test_lofi_renders_under_both_scene_grammars(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            self._write_test_tone(audio_path, duration=6.0)
            for grammar in ("legacy_plaza", "worlds"):
                output_path = root / f"{grammar}.mp4"
                debug_dir = root / f"debug_{grammar}"
                render_video(
                    input_path=audio_path,
                    output_path=output_path,
                    duration=6.0,
                    width=192,
                    height=108,
                    fps=8,
                    seed=5,
                    preset="lofi",
                    crf=28,
                    audio_bitrate="96k",
                    analysis_backend="numpy",
                    render_scale=0.5,
                    bloom_strength=0.4,
                    fog_strength=0.5,
                    exposure=0.92,
                    ps2_jitter=0.5,
                    render_profile="qa",
                    aesthetic="frutiger_cyber",
                    scene_grammar=grammar,
                    debug_frames_dir=debug_dir,
                    debug_labels=False,
                    debug_raw_frames=False,
                )
                self.assertTrue(output_path.exists())
                self.assertGreaterEqual(len(list(debug_dir.glob("*.png"))), 1)

    def test_blender_backend_fails_clearly_when_blender_missing(self) -> None:
        if _find_blender() is not None:
            self.skipTest("blender is available; missing-backend test not applicable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "proof.mp4"
            self._write_test_tone(audio_path, duration=2.0)
            with self.assertRaisesRegex(RuntimeError, r"(?i)blender.*path"):
                render_video(
                    input_path=audio_path,
                    output_path=output_path,
                    duration=2.0,
                    width=192,
                    height=108,
                    fps=8,
                    seed=5,
                    preset="worlds_material_proof",
                    crf=28,
                    audio_bitrate="96k",
                    analysis_backend="numpy",
                    render_scale=0.5,
                    bloom_strength=0.4,
                    fog_strength=0.5,
                    exposure=0.92,
                    ps2_jitter=0.5,
                    render_profile="final",
                    aesthetic="frutiger_cyber",
                    scene_grammar="worlds",
                    render_engine="blender",
                    debug_frames_dir=None,
                    debug_labels=False,
                    debug_raw_frames=False,
                )

    def test_short_render_exports_distinct_debug_frames(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "smoke.mp4"
            debug_dir = root / "debug_frames"
            raw_debug_dir = root / "debug_frames_raw"
            self._write_test_tone(audio_path, duration=8.0)

            render_video(
                input_path=audio_path,
                output_path=output_path,
                duration=12.0,
                width=320,
                height=180,
                fps=12,
                seed=7,
                preset="depth_showcase",
                crf=24,
                audio_bitrate="96k",
                analysis_backend="numpy",
                render_scale=0.5,
                bloom_strength=0.45,
                fog_strength=0.6,
                exposure=0.95,
                ps2_jitter=0.6,
                render_profile="qa",
                scene_grammar="legacy_plaza",
                debug_frames_dir=debug_dir,
                debug_labels=True,
                debug_raw_frames=True,
            )

            frames = sorted(debug_dir.glob("*.png"))
            raw_frames = sorted(raw_debug_dir.glob("*.png"))
            self.assertGreaterEqual(len(frames), 3)
            self.assertGreaterEqual(len(raw_frames), 3)
            self.assertTrue(any("__" in path.name for path in frames))

            images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frames[:3]]
            self.assertTrue(all(image is not None for image in images))
            means = [float(np.mean(image)) for image in images if image is not None]
            self.assertTrue(all(mean > 12.0 for mean in means))

            diffs = []
            for first, second in zip(images, images[1:]):
                assert first is not None and second is not None
                diffs.append(float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)))))
            self.assertTrue(all(diff > 1.5 for diff in diffs))

            for image in images:
                assert image is not None
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edge_density = float(np.mean(cv2.Canny(gray, 40, 120) > 0))
                filled_area = float(np.mean(gray > 28))
                tonal_spread = float(np.std(gray))
                self.assertGreater(edge_density, 0.03)
                self.assertGreater(filled_area, 0.12)
                self.assertGreater(tonal_spread, 16.0)
            self.assertTrue(output_path.exists())

    def test_worlds_distinctness_gallery_exports_multiple_families(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "worlds_gallery.mp4"
            debug_dir = root / "debug_worlds_gallery"
            self._write_test_tone(audio_path, duration=10.0)

            render_video(
                input_path=audio_path,
                output_path=output_path,
                duration=48.0,
                width=256,
                height=144,
                fps=10,
                seed=11,
                preset="worlds_distinctness_gallery",
                crf=26,
                audio_bitrate="96k",
                analysis_backend="numpy",
                render_scale=0.5,
                bloom_strength=0.38,
                fog_strength=0.35,
                exposure=0.98,
                ps2_jitter=0.35,
                render_profile="final",
                aesthetic="frutiger_cyber",
                scene_grammar="worlds",
                debug_frames_dir=debug_dir,
                debug_labels=False,
                debug_raw_frames=False,
            )

            frames = sorted(debug_dir.glob("*.png"))
            self.assertGreaterEqual(len(frames), 3)
            names = [path.name for path in frames]
            self.assertTrue(any("floating_sky_plaza" in name for name in names))
            self.assertTrue(any("polygon_grass_field" in name for name in names))
            images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frames[:3]]
            self.assertTrue(all(image is not None for image in images))
            means = [float(np.mean(image)) for image in images if image is not None]
            self.assertTrue(all(mean > 18.0 for mean in means))
            diffs = []
            for first, second in zip(images, images[1:]):
                assert first is not None and second is not None
                diffs.append(float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)))))
            self.assertTrue(all(diff > 3.0 for diff in diffs))

            frame_map = {path.name: cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frames}

            grass = next(image for name, image in frame_map.items() if "polygon_grass_field" in name and image is not None)
            grass_green = float(np.mean((grass[:, :, 1] > grass[:, :, 2] + 18) & (grass[:, :, 1] > grass[:, :, 0] + 18)))
            self.assertGreater(grass_green, 0.08)

            space = next(image for name, image in frame_map.items() if "orbital_garden_station" in name and image is not None)
            gray = cv2.cvtColor(space, cv2.COLOR_BGR2GRAY)
            dark_ratio = float(np.mean(gray < 60))
            planet_star_ratio = float(
                np.mean(
                    ((space[:, :, 0] > 110) & (space[:, :, 1] > 95) & (space[:, :, 2] > 70))
                    | (gray > 145)
                )
            )
            self.assertGreater(dark_ratio, 0.12)
            self.assertGreater(planet_star_ratio, 0.03)

    def test_worlds_people_and_water_frames_have_subject_contrast(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "people_water.mp4"
            debug_dir = root / "debug_people_water"
            self._write_test_tone(audio_path, duration=10.0)

            render_video(
                input_path=audio_path,
                output_path=output_path,
                duration=28.0,
                width=256,
                height=144,
                fps=10,
                seed=17,
                preset="worlds_people_water_gallery",
                crf=26,
                audio_bitrate="96k",
                analysis_backend="numpy",
                render_scale=0.5,
                bloom_strength=0.38,
                fog_strength=0.35,
                exposure=0.98,
                ps2_jitter=0.35,
                render_profile="final",
                aesthetic="frutiger_cyber",
                scene_grammar="worlds",
                debug_frames_dir=debug_dir,
                debug_labels=False,
                debug_raw_frames=False,
            )

            frames = sorted(debug_dir.glob("*.png"))
            frame_map = {path.name: cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frames}

            people = next(image for name, image in frame_map.items() if "dancing_polygon_crowd" in name and image is not None)
            colorful = float(
                np.mean(
                    ((people[:, :, 2] > 155) & (people[:, :, 0] > 105))
                    | ((people[:, :, 1] > 150) & (people[:, :, 2] < 180))
                    | ((people[:, :, 0] > 150) & (people[:, :, 1] > 140))
                )
            )
            self.assertGreater(colorful, 0.05)

            water = next(image for name, image in frame_map.items() if "splash_fountain_court" in name and image is not None)
            water_gray = cv2.cvtColor(water, cv2.COLOR_BGR2GRAY)
            blue_ratio = float(np.mean((water[:, :, 0] > water[:, :, 1] + 18) & (water[:, :, 0] > water[:, :, 2] + 18)))
            splash_edges = float(np.mean(cv2.Canny(water_gray, 50, 140) > 0))
            self.assertGreater(blue_ratio, 0.09)
            self.assertGreater(splash_edges, 0.03)

    def test_lofi_worlds_contains_broad_family_mix(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "lofi_worlds.mp4"
            debug_dir = root / "debug_lofi_worlds"
            self._write_test_tone(audio_path, duration=10.0)

            render_video(
                input_path=audio_path,
                output_path=output_path,
                duration=56.0,
                width=256,
                height=144,
                fps=10,
                seed=19,
                preset="lofi",
                crf=26,
                audio_bitrate="96k",
                analysis_backend="numpy",
                render_scale=0.5,
                bloom_strength=0.38,
                fog_strength=0.35,
                exposure=0.98,
                ps2_jitter=0.35,
                render_profile="final",
                aesthetic="frutiger_cyber",
                scene_grammar="worlds",
                debug_frames_dir=debug_dir,
                debug_labels=False,
                debug_raw_frames=False,
            )

            names = [path.name for path in debug_dir.glob("*.png")]
            self.assertTrue(any("polygon_grass_field" in name for name in names))
            self.assertTrue(any("dancing_polygon_crowd" in name for name in names))
            self.assertTrue(any("splash_fountain_court" in name for name in names))
            self.assertTrue(any("glass_greenhouse" in name for name in names))
            self.assertTrue(any("orbital_garden_station" in name for name in names))

    def _write_test_tone(self, path: Path, duration: float) -> None:
        sr = 22050
        frame_count = int(sr * duration)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            chunks = []
            for index in range(frame_count):
                t = index / sr
                sample = (
                    0.22 * math.sin(2 * math.pi * 220 * t)
                    + 0.12 * math.sin(2 * math.pi * 440 * t)
                    + 0.08 * math.sin(2 * math.pi * (110 + 30 * math.sin(t * 1.7)) * t)
                )
                chunks.append(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
            wav_file.writeframes(b"".join(chunks))


if __name__ == "__main__":
    unittest.main()
