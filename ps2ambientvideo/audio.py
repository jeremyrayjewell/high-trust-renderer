from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioAnalysis:
    audio_path: Path
    samples: np.ndarray
    sr: int
    duration: float
    frame_times: np.ndarray
    beat: np.ndarray
    bass: np.ndarray
    mids: np.ndarray
    highs: np.ndarray
    energy: np.ndarray
    onset: np.ndarray
    sections: np.ndarray

    def features_at(self, index: int) -> dict[str, float]:
        last = len(self.frame_times) - 1
        i = max(0, min(index, last))
        return {
            "beat": float(self.beat[i]),
            "bass": float(self.bass[i]),
            "mids": float(self.mids[i]),
            "highs": float(self.highs[i]),
            "energy": float(self.energy[i]),
            "onset": float(self.onset[i]),
            "section": float(self.sections[i]),
        }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / peak, 0.0, 1.0).astype(np.float32)


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1 or len(values) == 0:
        return values.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _resample_to_frames(values: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _section_labels(curve: np.ndarray, num_sections: int = 8) -> np.ndarray:
    if len(curve) == 0:
        return np.zeros(0, dtype=np.float32)
    smooth = _moving_average(curve, max(8, len(curve) // 24))
    gradient = np.abs(np.gradient(smooth))
    thresholds = np.linspace(np.min(gradient), np.max(gradient) + 1e-6, num_sections)
    labels = np.digitize(gradient, thresholds[:-1]).astype(np.float32)
    if np.max(labels) > 0:
        labels /= float(np.max(labels))
    return labels


def _analyze_with_librosa(samples: np.ndarray, sr: int, frame_count: int) -> dict[str, np.ndarray]:
    import librosa

    hop = max(256, int(sr / 60))
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(y=samples, sr=sr, hop_length=hop, units="frames")
    stft = np.abs(librosa.stft(y=samples, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    bass_mask = freqs < 180
    mids_mask = (freqs >= 180) & (freqs < 2200)
    highs_mask = freqs >= 2200
    bass = _normalize(np.mean(stft[bass_mask], axis=0))
    mids = _normalize(np.mean(stft[mids_mask], axis=0))
    highs = _normalize(np.mean(stft[highs_mask], axis=0))
    energy = _normalize(librosa.feature.rms(S=stft)[0])
    onset = _normalize(onset_env)

    beat_curve = np.zeros_like(onset, dtype=np.float32)
    if len(beats):
        beat_curve[np.clip(beats, 0, len(beat_curve) - 1)] = 1.0
    beat_curve = np.maximum(beat_curve, onset * 0.45)

    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    harmonic_motion = _normalize(np.std(chroma, axis=0))
    section_driver = _normalize(0.55 * energy + 0.45 * harmonic_motion)

    return {
        "beat": _resample_to_frames(beat_curve, frame_count),
        "bass": _resample_to_frames(bass, frame_count),
        "mids": _resample_to_frames(mids, frame_count),
        "highs": _resample_to_frames(highs, frame_count),
        "energy": _resample_to_frames(energy, frame_count),
        "onset": _resample_to_frames(onset, frame_count),
        "sections": _resample_to_frames(_section_labels(section_driver), frame_count),
        "tempo": np.array([tempo], dtype=np.float32),
    }


def _analyze_fallback(samples: np.ndarray, sr: int, frame_count: int) -> dict[str, np.ndarray]:
    window = max(512, int(sr / 40))
    hop = max(256, window // 2)
    frames = []
    for start in range(0, max(1, len(samples) - window), hop):
        chunk = samples[start:start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        frames.append(spec)
    stft = np.array(frames, dtype=np.float32).T if frames else np.zeros((1, 1), dtype=np.float32)
    freqs = np.fft.rfftfreq(window, d=1.0 / sr)

    bass = _normalize(np.mean(stft[freqs < 180], axis=0))
    mids = _normalize(np.mean(stft[(freqs >= 180) & (freqs < 2200)], axis=0))
    highs = _normalize(np.mean(stft[freqs >= 2200], axis=0))
    energy = _normalize(np.sqrt(np.mean(stft * stft, axis=0)))
    onset = _normalize(np.maximum(0.0, np.diff(np.concatenate([[0.0], energy]))))
    beat = _normalize(np.maximum(onset, np.maximum(0.0, energy - _moving_average(energy, 7))))
    sections = _section_labels(_normalize(0.65 * energy + 0.35 * mids))

    return {
        "beat": _resample_to_frames(beat, frame_count),
        "bass": _resample_to_frames(bass, frame_count),
        "mids": _resample_to_frames(mids, frame_count),
        "highs": _resample_to_frames(highs, frame_count),
        "energy": _resample_to_frames(energy, frame_count),
        "onset": _resample_to_frames(onset, frame_count),
        "sections": _resample_to_frames(sections, frame_count),
        "tempo": np.array([0.0], dtype=np.float32),
    }


def _should_use_librosa(backend: str) -> bool:
    if backend == "numpy":
        return False
    if backend == "librosa":
        return True
    return sys.version_info < (3, 13)


def analyze_audio(
    path: Path,
    fps: int,
    duration_limit: float | None = None,
    backend: str = "auto",
) -> AudioAnalysis:
    samples, sr = sf.read(str(path), always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = samples.astype(np.float32)

    source_duration = len(samples) / float(sr)
    duration = source_duration
    if duration_limit is not None:
        target_duration = float(duration_limit)
        if target_duration <= source_duration:
            duration = target_duration
            samples = samples[: int(duration * sr)]
        else:
            duration = target_duration
            target_samples = int(duration * sr)
            repeats = int(math.ceil(target_samples / max(1, len(samples))))
            samples = np.tile(samples, repeats)[:target_samples]

    frame_count = max(1, int(math.ceil(duration * fps)))
    frame_times = np.arange(frame_count, dtype=np.float32) / float(fps)

    try:
        features = (
            _analyze_with_librosa(samples, sr, frame_count)
            if _should_use_librosa(backend)
            else _analyze_fallback(samples, sr, frame_count)
        )
    except Exception:
        if backend == "librosa":
            raise
        features = _analyze_fallback(samples, sr, frame_count)

    return AudioAnalysis(
        audio_path=path,
        samples=samples,
        sr=sr,
        duration=duration,
        frame_times=frame_times,
        beat=features["beat"],
        bass=features["bass"],
        mids=features["mids"],
        highs=features["highs"],
        energy=features["energy"],
        onset=features["onset"],
        sections=features["sections"],
    )
