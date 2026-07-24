"""Stage 2 audio probes, feature extraction, and counterfactuals.

All audio is mono fp32 at the configured sample rate. Probes are deterministic
functions of ``seed`` and exactly frame-aligned to ``config.samples_per_frame``.
Feature extraction yields one row per complete audio frame; no per-clip
normalization is ever applied.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

ProbeName = Literal[
    "silence",
    "constant",
    "impulse",
    "alternating",
    "stepped-bands",
    "review",
]
ConditionName = Literal["correct", "silent", "shuffled", "frozen-mean"]

# Fixed normalization constants (see module docs).
# ``_SPECTRAL_REF`` is the expected rFFT peak magnitude of a unit-amplitude pure
# tone windowed by a Hann window of length ``samples_per_frame``: N * 0.5 / 2.
_RMS_TARGET_CONSTANT = 0.2  # target RMS for moderate seeded-noise segments
_TONE_AMPLITUDE = 0.2  # amplitude for pure-tone probe segments
_LOW_HZ = 80.0
_HIGH_LOW_HZ = 300.0
_BAND_HIGH_LO_HZ = 2500.0
_BAND_HIGH_HI_HZ = 6000.0
_FADE_SAMPLES = 120  # ~5ms at 24kHz short boundary fades
_BURST_SAMPLES = 600  # 25ms audible noise burst
_RMS_TARGET_LOW = 0.04


@dataclass(frozen=True)
class AudioConfig:
    """Sample/frame configuration with exact integer samples per frame."""

    sample_rate: int = 24000
    frame_rate: int = 24

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if self.sample_rate % self.frame_rate != 0:
            raise ValueError("sample_rate must be an exact integer multiple of frame_rate")

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.frame_rate


@dataclass(frozen=True)
class AudioFeatures:
    """Per-frame audio features; columns are rms, onset, spectral_flux, low_band,
    high_band."""

    rms: Tensor
    onset: Tensor
    spectral_flux: Tensor
    low_band: Tensor
    high_band: Tensor

    def __post_init__(self) -> None:
        fields = (
            self.rms,
            self.onset,
            self.spectral_flux,
            self.low_band,
            self.high_band,
        )
        names = ("rms", "onset", "spectral_flux", "low_band", "high_band")
        for name, value in zip(names, fields, strict=True):
            if value.ndim != 1:
                raise ValueError(f"{name} must be 1D")
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        length = fields[0].numel()
        if length == 0:
            raise ValueError("feature tensors must be nonempty")
        for value in fields[1:]:
            if value.numel() != length:
                raise ValueError("all feature tensors must share length")

    def matrix(self) -> Tensor:
        """Return [frames, 5] in column order rms, onset, spectral_flux, low_band,
        high_band."""
        return torch.stack(
            (self.rms, self.onset, self.spectral_flux, self.low_band, self.high_band),
            dim=1,
        )


def _generator(seed: int) -> torch.Generator:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    return torch.Generator().manual_seed(seed)


def _fade(segment: Tensor, fade: int = _FADE_SAMPLES) -> Tensor:
    """Apply linear fade-in/out at the edges of a 1D segment without shifting it."""
    n = segment.numel()
    if n == 0 or fade <= 0:
        return segment
    f = min(fade, n // 2)
    if f <= 0:
        return segment
    ramp = torch.linspace(0.0, 1.0, f, dtype=segment.dtype)
    out = segment.clone()
    out[:f] = out[:f] * ramp
    out[n - f :] = out[n - f :] * ramp.flip(0)
    return out


def _rms(tensor: Tensor) -> float:
    return math.sqrt(float(tensor.pow(2).mean().item()))


def _scaled_noise(gen: torch.Generator, n: int, rms: float) -> Tensor:
    """White gaussian noise scaled to an exact fixed target RMS."""
    noise = torch.randn(n, generator=gen, dtype=torch.float32)
    current = _rms(noise)
    if current == 0.0:
        return noise
    return noise * (rms / current)


def _band_limited_noise(
    gen: torch.Generator, n: int, sample_rate: int, lo: float, hi: float, rms: float
) -> Tensor:
    """Seeded noise band-limited to [lo, hi] Hz at a fixed RMS."""
    noise = torch.randn(n, generator=gen, dtype=torch.float32)
    spectrum = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    spectrum = torch.where(mask, spectrum, torch.zeros_like(spectrum))
    filtered = torch.fft.irfft(spectrum, n=n)
    current = _rms(filtered)
    if current == 0.0:
        return filtered
    return filtered * (rms / current)


def _tone(n: int, sample_rate: int, freq: float, amplitude: float) -> Tensor:
    """Deterministic pure tone of the given amplitude (no seed needed)."""
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return amplitude * torch.cos(2.0 * math.pi * freq * t)


def _const_probe(config: AudioConfig, gen: torch.Generator) -> Tensor:
    n = config.sample_rate * 4
    return _fade(_scaled_noise(gen, n, _RMS_TARGET_CONSTANT))


def _impulse_probe(config: AudioConfig, gen: torch.Generator) -> Tensor:
    n = config.sample_rate * 4
    out = torch.zeros(n, dtype=torch.float32)
    start = n // 2
    burst = _scaled_noise(gen, _BURST_SAMPLES, _RMS_TARGET_CONSTANT)
    out[start : start + _BURST_SAMPLES] = _fade(burst)
    return out


def _alternating_probe(config: AudioConfig, gen: torch.Generator) -> Tensor:
    n = config.sample_rate * 4
    block = config.sample_rate // 2  # 0.5s
    out = torch.zeros(n, dtype=torch.float32)
    for i in range(8):
        s = i * block
        rms = _RMS_TARGET_CONSTANT if i % 2 == 0 else _RMS_TARGET_LOW
        seg = _scaled_noise(gen, block, rms)
        out[s : s + block] = _fade(seg)
    return out


def _stepped_bands_probe(config: AudioConfig, gen: torch.Generator) -> Tensor:
    # Equal-RMS alternating 160Hz and 4000Hz pure-tone blocks (amplitude matched).
    del gen  # tones are deterministic; generator kept for API symmetry only
    n = config.sample_rate * 4
    block = config.sample_rate // 2
    out = torch.zeros(n, dtype=torch.float32)
    for i in range(8):
        s = i * block
        freq = 160.0 if i % 2 == 0 else 4000.0
        seg = _tone(block, config.sample_rate, freq, _TONE_AMPLITUDE)
        out[s : s + block] = _fade(seg)
    return out


def _review_probe(config: AudioConfig, gen: torch.Generator) -> Tensor:
    sr = config.sample_rate
    n = sr * 10
    out = torch.zeros(n, dtype=torch.float32)

    def put(start: int, end: int, seg: Tensor) -> None:
        out[start:end] = seg

    # 0-1s silence (already zero)
    # 1-2s constant moderate seeded noise
    put(1 * sr, 2 * sr, _fade(_scaled_noise(gen, sr, _RMS_TARGET_CONSTANT)))
    # 2-3s silence (already zero)
    # burst exactly at 3s, then silence through 4.5s
    burst = _fade(_scaled_noise(gen, _BURST_SAMPLES, _RMS_TARGET_CONSTANT))
    put(3 * sr, 3 * sr + _BURST_SAMPLES, burst)
    # 4.5-6.5s alternating 0.5s high/low energy blocks
    block = sr // 2
    for i in range(4):
        s = (4 * sr) + (sr // 2) + i * block
        rms = _RMS_TARGET_CONSTANT if i % 2 == 0 else _RMS_TARGET_LOW
        seg = _scaled_noise(gen, block, rms)
        put(s, s + block, _fade(seg))
    # 6.5-7s silence (already zero)
    # 7-8s 160Hz tone
    put(7 * sr, 8 * sr, _fade(_tone(sr, sr, 160.0, _TONE_AMPLITUDE)))
    # 8-9s 4000Hz tone at matched RMS (same amplitude -> same RMS)
    put(8 * sr, 9 * sr, _fade(_tone(sr, sr, 4000.0, _TONE_AMPLITUDE)))
    # 9-10s silence (already zero)
    return out


_PROBE_BUILDERS = {
    "silence": lambda cfg, g: torch.zeros(cfg.sample_rate * 4, dtype=torch.float32),
    "constant": _const_probe,
    "impulse": _impulse_probe,
    "alternating": _alternating_probe,
    "stepped-bands": _stepped_bands_probe,
    "review": _review_probe,
}


def synthesize_probe(name: ProbeName, config: AudioConfig, *, seed: int) -> Tensor:
    """Return a deterministic, frame-aligned mono fp32 waveform for a probe."""
    if name not in _PROBE_BUILDERS:
        raise ValueError(f"unknown probe: {name}")
    gen = _generator(seed)
    audio = _PROBE_BUILDERS[name](config, gen)
    if audio.numel() % config.samples_per_frame != 0:
        raise ValueError("probe length must be frame-aligned")
    return audio


def _band_indices(n: int, sample_rate: int, lo: float, hi: float) -> Tensor:
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate)
    return torch.where(
        (freqs >= lo) & (freqs <= hi), torch.ones_like(freqs), torch.zeros_like(freqs)
    ).to(torch.bool)


def extract_features(waveform: Tensor, config: AudioConfig) -> AudioFeatures:
    """Extract per-frame features; exactly one row per complete audio frame."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1D")
    if waveform.dtype != torch.float32:
        raise ValueError("waveform must be float32")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform must be finite")
    n_frame = config.samples_per_frame
    n_frames = waveform.numel() // n_frame
    if n_frames == 0:
        raise ValueError("waveform too short for one frame")
    frames = waveform[: n_frames * n_frame].view(n_frames, n_frame)

    rms = frames.pow(2).mean(dim=1).clamp_min(0.0).sqrt()

    onset = torch.zeros_like(rms)
    if n_frames > 1:
        diff = rms[1:] - rms[:-1]
        onset[1:] = diff.clamp_min(0.0)
    onset = onset.clamp(0.0, 1.0)

    window = torch.hann_window(n_frame, dtype=torch.float32)
    coherent_gain = window.mean().item()
    ref = (n_frame * coherent_gain) / 2.0  # unit-tone peak magnitude
    win_frames = frames * window
    spec = torch.fft.rfft(win_frames, dim=1)
    mag = spec.abs() / ref
    low_mask = _band_indices(n_frame, config.sample_rate, _LOW_HZ, _HIGH_LOW_HZ)
    high_mask = _band_indices(n_frame, config.sample_rate, _BAND_HIGH_LO_HZ, _BAND_HIGH_HI_HZ)
    low_band = mag[:, low_mask].max(dim=1).values.clamp(0.0, 1.0)
    high_band = mag[:, high_mask].max(dim=1).values.clamp(0.0, 1.0)

    spectral_flux = torch.zeros(n_frames, dtype=torch.float32)
    if n_frames > 1:
        diff = mag[1:] - mag[:-1]
        flux = diff.clamp_min(0.0).sum(dim=1)
        spectral_flux[1:] = flux
    spectral_flux = spectral_flux.clamp(0.0, 1.0)

    return AudioFeatures(
        rms=rms,
        onset=onset,
        spectral_flux=spectral_flux,
        low_band=low_band,
        high_band=high_band,
    )


def interpolate_features(features: AudioFeatures, *, updates_per_frame: int) -> Tensor:
    """Return [frames*updates_per_frame, 5] with linear substep interpolation;
    the final frame is held."""
    if not isinstance(updates_per_frame, int) or isinstance(updates_per_frame, bool):
        raise ValueError("updates_per_frame must be an int")
    if updates_per_frame <= 0:
        raise ValueError("updates_per_frame must be positive")
    matrix = features.matrix()
    n_frames = matrix.shape[0]
    out = torch.zeros(n_frames * updates_per_frame, 5, dtype=torch.float32)
    for f in range(n_frames):
        for s in range(updates_per_frame):
            t = s / updates_per_frame
            row = (1.0 - t) * matrix[f] + t * matrix[f + 1] if f + 1 < n_frames else matrix[f]
            out[f * updates_per_frame + s] = row
    return out


def counterfactual_features(
    features: AudioFeatures, condition: ConditionName, *, seed: int
) -> AudioFeatures:
    """Return a counterfactual feature set for a documented condition."""
    if condition == "correct":
        return AudioFeatures(
            rms=features.rms.clone(),
            onset=features.onset.clone(),
            spectral_flux=features.spectral_flux.clone(),
            low_band=features.low_band.clone(),
            high_band=features.high_band.clone(),
        )
    n = features.rms.numel()
    if condition == "silent":
        zero = torch.zeros(n, dtype=torch.float32)
        return AudioFeatures(
            rms=zero.clone(),
            onset=zero.clone(),
            spectral_flux=zero.clone(),
            low_band=zero.clone(),
            high_band=zero.clone(),
        )
    if condition == "shuffled":
        gen = _generator(seed)
        perm = torch.randperm(n, generator=gen)
        return AudioFeatures(
            rms=features.rms[perm].clone(),
            onset=features.onset[perm].clone(),
            spectral_flux=features.spectral_flux[perm].clone(),
            low_band=features.low_band[perm].clone(),
            high_band=features.high_band[perm].clone(),
        )
    if condition == "frozen-mean":
        cols = features.matrix()
        means = cols.mean(dim=0, keepdim=True).expand(n, 5)
        return AudioFeatures(
            rms=means[:, 0].clone(),
            onset=means[:, 1].clone(),
            spectral_flux=means[:, 2].clone(),
            low_band=means[:, 3].clone(),
            high_band=means[:, 4].clone(),
        )
    raise ValueError(f"unknown condition: {condition}")


def write_pcm16_wav(path: Path, waveform: Tensor, sample_rate: int) -> None:
    """Write a mono PCM16 WAV. Parent directory must already exist; values must be
    finite and within [-1, 1]."""
    if not isinstance(path, Path):
        raise ValueError("path must be a Path")
    if not path.parent.is_dir():
        raise ValueError("parent directory must already exist")
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1D")
    if waveform.dtype != torch.float32:
        raise ValueError("waveform must be float32")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform must be finite")
    if waveform.abs().max().item() > 1.0:
        raise ValueError("waveform samples must be within [-1, 1]")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    scaled = (waveform * 32767.0).round().to(torch.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.numpy().tobytes())
