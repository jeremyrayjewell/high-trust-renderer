from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from ps2ambientvideo.softbodies_backend import render_video_softbodies


class SoftbodiesBackendTest(unittest.TestCase):
    def test_softbodies_backend_reports_missing_optional_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "smoke.wav"
            output_path = root / "softbodies.mp4"
            self._write_silent_wav(audio_path)
            with mock.patch(
                "ps2ambientvideo.softbodies_backend._import_softbodies_runtime",
                side_effect=RuntimeError("Softbodies backend requires optional dependencies and runtime support."),
            ):
                with self.assertRaisesRegex(RuntimeError, "Softbodies backend requires optional dependencies"):
                    render_video_softbodies(
                        input_path=audio_path,
                        output_path=output_path,
                        duration=1.0,
                        width=160,
                        height=90,
                        fps=4,
                        seed=1,
                        preset="lofi",
                        audio_bitrate="96k",
                        analysis_backend="numpy",
                    )

    def _write_silent_wav(self, path: Path, duration: float = 1.0, sr: int = 22050) -> None:
        frame_count = int(sr * duration)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(b"\x00\x00" * frame_count)


if __name__ == "__main__":
    unittest.main()
