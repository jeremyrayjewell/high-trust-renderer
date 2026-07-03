from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover - allows fallback mode without librosa installed yet
    librosa = None


def safe_normalize(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = np.maximum(values, 0.0)
    peak = float(np.percentile(values, 99.0))
    if peak < floor:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip(values / peak, 0.0, 1.0)
    return normalized.astype(np.float32)


@dataclass(frozen=True)
class AudioSample:
    time_seconds: float
    rms: float
    onset: float
    bass: float
    mid: float
    high: float
    is_beat: bool


@dataclass
class AudioFeatures:
    duration: float
    times: np.ndarray
    rms: np.ndarray
    onset: np.ndarray
    bass: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    beat_times: np.ndarray
    source: str
    sample_period: float

    def sample(self, time_seconds: float) -> AudioSample:
        if self.times.size == 0:
            return AudioSample(time_seconds=max(time_seconds, 0.0), rms=0.0, onset=0.0, bass=0.0, mid=0.0, high=0.0, is_beat=False)

        t = float(np.clip(time_seconds, 0.0, self.duration))
        rms = float(np.interp(t, self.times, self.rms))
        onset = float(np.interp(t, self.times, self.onset))
        bass = float(np.interp(t, self.times, self.bass))
        mid = float(np.interp(t, self.times, self.mid))
        high = float(np.interp(t, self.times, self.high))

        beat_tolerance = max(self.sample_period * 0.6, 1.0 / 60.0)
        if self.beat_times.size:
            insert_at = int(np.searchsorted(self.beat_times, t))
            left_diff = abs(t - float(self.beat_times[insert_at - 1])) if insert_at > 0 else np.inf
            right_diff = abs(float(self.beat_times[insert_at]) - t) if insert_at < self.beat_times.size else np.inf
            is_beat = min(left_diff, right_diff) <= beat_tolerance
        else:
            is_beat = False

        return AudioSample(
            time_seconds=t,
            rms=rms,
            onset=onset,
            bass=bass,
            mid=mid,
            high=high,
            is_beat=is_beat,
        )


def _cache_payload_from_features(features: AudioFeatures) -> dict[str, Any]:
    return {
        "duration": np.array([features.duration], dtype=np.float32),
        "times": features.times.astype(np.float32),
        "rms": features.rms.astype(np.float32),
        "onset": features.onset.astype(np.float32),
        "bass": features.bass.astype(np.float32),
        "mid": features.mid.astype(np.float32),
        "high": features.high.astype(np.float32),
        "beat_times": features.beat_times.astype(np.float32),
        "source": np.array([features.source]),
        "sample_period": np.array([features.sample_period], dtype=np.float32),
    }


def _features_from_npz(npz: Any) -> AudioFeatures:
    return AudioFeatures(
        duration=float(npz["duration"][0]),
        times=np.asarray(npz["times"], dtype=np.float32),
        rms=np.asarray(npz["rms"], dtype=np.float32),
        onset=np.asarray(npz["onset"], dtype=np.float32),
        bass=np.asarray(npz["bass"], dtype=np.float32),
        mid=np.asarray(npz["mid"], dtype=np.float32),
        high=np.asarray(npz["high"], dtype=np.float32),
        beat_times=np.asarray(npz["beat_times"], dtype=np.float32),
        source=str(npz["source"][0]),
        sample_period=float(npz["sample_period"][0]),
    )


def _analysis_cache_path(audio_path: Path) -> Path:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path("output") / "audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in audio_path.stem)[:48] or "audio"
    return cache_dir / f"{safe_stem}_{digest}.analysis.npz"


def generate_synthetic_features(duration: float = 10.0, sample_rate: float = 120.0, seed: int = 42) -> AudioFeatures:
    duration = float(max(duration, 0.1))
    sample_rate = float(max(sample_rate, 10.0))
    rng = np.random.default_rng(seed)
    count = int(np.ceil(duration * sample_rate)) + 1
    times = np.linspace(0.0, duration, count, dtype=np.float32)
    beat_interval = 0.5
    beat_times = np.arange(0.0, duration + beat_interval, beat_interval, dtype=np.float32)

    rms = np.full_like(times, 0.2, dtype=np.float32)
    onset = np.zeros_like(times, dtype=np.float32)
    bass = np.full_like(times, 0.15, dtype=np.float32)
    mid = np.full_like(times, 0.12, dtype=np.float32)
    high = np.full_like(times, 0.08, dtype=np.float32)

    for beat in beat_times:
        pulse = np.exp(-0.5 * ((times - beat) / 0.065) ** 2)
        bass += 0.9 * pulse
        onset += 1.0 * np.exp(-0.5 * ((times - beat) / 0.03) ** 2)
        rms += 0.45 * pulse
        high += 0.12 * np.exp(-0.5 * ((times - beat) / 0.02) ** 2)

    mid += 0.24 * (0.5 + 0.5 * np.sin(times * 2.7 + 0.4))
    high += 0.18 * (0.5 + 0.5 * np.sin(times * 11.0 + 1.2))
    rms += 0.08 * (0.5 + 0.5 * np.sin(times * 0.8))
    bass += 0.06 * (0.5 + 0.5 * np.sin(times * 1.35))

    # A tiny deterministic noise layer keeps the fallback from feeling overly rigid.
    rms += rng.normal(0.0, 0.015, size=times.shape).astype(np.float32)
    onset += rng.normal(0.0, 0.01, size=times.shape).astype(np.float32)
    bass += rng.normal(0.0, 0.01, size=times.shape).astype(np.float32)
    mid += rng.normal(0.0, 0.01, size=times.shape).astype(np.float32)
    high += rng.normal(0.0, 0.01, size=times.shape).astype(np.float32)

    rms = safe_normalize(rms)
    onset = safe_normalize(onset)
    bass = safe_normalize(bass)
    mid = safe_normalize(mid)
    high = safe_normalize(high)

    sample_period = duration / max(count - 1, 1)
    return AudioFeatures(
        duration=duration,
        times=times,
        rms=rms,
        onset=onset,
        bass=bass,
        mid=mid,
        high=high,
        beat_times=beat_times[beat_times <= duration].astype(np.float32),
        source="synthetic",
        sample_period=sample_period,
    )


def _analyze_real_audio(audio_path: Path) -> AudioFeatures:
    if librosa is None:
        raise RuntimeError("librosa is required for real audio analysis but could not be imported.")

    signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    signal = signal.astype(np.float32)
    duration = float(librosa.get_duration(y=signal, sr=sample_rate))
    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sample_rate, hop_length=hop_length)
    del tempo

    stft = np.abs(librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    bass_mask = (freqs >= 20.0) & (freqs < 180.0)
    mid_mask = (freqs >= 180.0) & (freqs < 2000.0)
    high_mask = freqs >= 2000.0
    bass = stft[bass_mask].mean(axis=0) if np.any(bass_mask) else np.zeros(stft.shape[1], dtype=np.float32)
    mid = stft[mid_mask].mean(axis=0) if np.any(mid_mask) else np.zeros(stft.shape[1], dtype=np.float32)
    high = stft[high_mask].mean(axis=0) if np.any(high_mask) else np.zeros(stft.shape[1], dtype=np.float32)

    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sample_rate, hop_length=hop_length).astype(np.float32)
    if times.size == 0:
        return generate_synthetic_features(duration=max(duration, 1.0))

    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length).astype(np.float32)
    sample_period = float(np.median(np.diff(times))) if times.size > 1 else 1.0 / 60.0

    return AudioFeatures(
        duration=duration,
        times=times,
        rms=safe_normalize(rms),
        onset=safe_normalize(onset_env),
        bass=safe_normalize(bass),
        mid=safe_normalize(mid),
        high=safe_normalize(high),
        beat_times=beat_times,
        source=str(audio_path),
        sample_period=sample_period,
    )


def analyze_audio(audio_path: str | Path | None, duration: float = 10.0, seed: int = 42) -> AudioFeatures:
    if not audio_path:
        return generate_synthetic_features(duration=duration, seed=seed)

    path = Path(audio_path)
    if not path.exists():
        return generate_synthetic_features(duration=duration, seed=seed)

    cache_path = _analysis_cache_path(path)
    stat = path.stat()

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_mtime = int(cached["source_mtime_ns"][0])
                cached_size = int(cached["source_size"][0])
                if cached_mtime == stat.st_mtime_ns and cached_size == stat.st_size:
                    return _features_from_npz(cached)
        except Exception:
            pass

    features = _analyze_real_audio(path)
    payload = _cache_payload_from_features(features)
    payload["source_mtime_ns"] = np.array([stat.st_mtime_ns], dtype=np.int64)
    payload["source_size"] = np.array([stat.st_size], dtype=np.int64)
    np.savez_compressed(cache_path, **payload)
    return features
