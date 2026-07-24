"""Stage 2 non-learned audio-conditioned control of the frozen Stage 1 NCA.

The Stage 1 NCA is frozen and driven by a fixed, non-learned control loop that
maps audio features to recurrence speed and bounded channel-1 perturbations.
No training occurs; the NCA architecture is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from synesthete.audio import (
    AudioConfig,
    AudioFeatures,
    ConditionName,
    ProbeName,
    counterfactual_features,
    extract_features,
    interpolate_features,
    synthesize_probe,
    write_pcm16_wav,
)
from synesthete.media import (
    RENDER_MAPPING,
    MediaConfig,
    audio_stream_hash,
    save_audiovisual,
    save_comparison_audiovisual,
    write_review_page,
)
from synesthete.nca import BASELINE_CONFIG, NeuralCellularAutomaton
from synesthete.runtime import collect_runtime_info, require_mps
from synesthete.stage1 import MASK_ALIGNED_RAPID_BUDGET, _sample_fire_masks
from synesthete.stage1_checkpoint import load_stage1_checkpoint
from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    ReactionDiffusionConfig,
    encode_teacher_fields,
    make_teacher_fields,
    masked_reaction_diffusion_step,
    render_luminance,
)

# --- Constants ---

EXPECTED_DISTRIBUTED_HASH = "d037cf586b1a5c86ed8f9ee857cd4940c539bda072f89bfe5805f94acaaa3a21"
RETAINED_CHECKPOINT = Path("outputs/stage1-mask-aligned-rapid-c31be68/checkpoint.pt")
PROBE_NAMES: tuple[ProbeName, ...] = (
    "silence",
    "constant",
    "impulse",
    "alternating",
    "stepped-bands",
    "review",
)
BLIND_CONDITIONS: tuple[ConditionName, ...] = ("correct", "silent", "shuffled")
FEATURE_NAMES = ("rms", "onset", "spectral_flux", "low_band", "high_band")
BAND_DEFINITIONS = {
    "low_band": {"low_hz": 80.0, "high_hz": 300.0},
    "high_band": {"low_hz": 2500.0, "high_hz": 6000.0},
}

# Automated check thresholds (modest, fixed, no auto-tuning).
_THR_BOUND = 2.0
_THR_IMPULSE_WINDOW = 24
_THR_IMPULSE_RATIO = 1.25
_THR_ALTERNATING_RATIO = 1.2
_THR_SPECTRAL_RELATIVE_DIFFERENCE = 0.01
_THR_RECOVERY_RATIO = 0.5
_THR_RECOVERY_MAX_FRAMES = 24
_THR_LAG_MAX_FRAMES = 12
_THR_DIVERGENCE_MIN = 1e-6
_THR_STRUCTURE_RATIO = 0.05
_EVENT_THRESHOLD = 0.03
_ENERGY_SPLIT_RMS = 0.1


# --- Stage2Config ---


@dataclass(frozen=True)
class Stage2Config:
    """Frozen non-learned control parameters for Stage 2."""

    grid_size: int = 96
    updates_per_frame: int = 4
    burn_in_updates: int = 800
    state_seed: int = 4107
    mask_seed: int = 4206
    audio_seed: int = 5201
    shuffle_seed: int = 5202
    speed_min: float = 0.35
    speed_max: float = 1.5
    rms_reference: float = 0.2
    onset_reference: float = 0.08
    spectral_reference: float = 0.3
    band_reference: float = 0.18
    spectral_sustain_amplitude: float = 0.008
    perturbation_amplitude: float = 0.12
    low_radius: float = 14.0
    high_radius: float = 4.0
    state_bound: float = 2.0

    def __post_init__(self) -> None:
        if self.grid_size < 16:
            raise ValueError("grid_size must be at least 16")
        if self.updates_per_frame <= 0:
            raise ValueError("updates_per_frame must be positive")
        if self.burn_in_updates < 0:
            raise ValueError("burn_in_updates must be non-negative")
        for name in ("state_seed", "mask_seed", "audio_seed", "shuffle_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int")
        if self.speed_min <= 0.0:
            raise ValueError("speed_min must be positive")
        if self.speed_max <= self.speed_min:
            raise ValueError("speed_max must exceed speed_min")
        if self.rms_reference <= 0.0:
            raise ValueError("rms_reference must be positive")
        if self.onset_reference <= 0.0:
            raise ValueError("onset_reference must be positive")
        if self.spectral_reference <= 0.0:
            raise ValueError("spectral_reference must be positive")
        if self.band_reference <= 0.0:
            raise ValueError("band_reference must be positive")
        if self.spectral_sustain_amplitude <= 0.0:
            raise ValueError("spectral_sustain_amplitude must be positive")
        if self.perturbation_amplitude <= 0.0:
            raise ValueError("perturbation_amplitude must be positive")
        if self.low_radius <= 0.0:
            raise ValueError("low_radius must be positive")
        if self.high_radius <= 0.0:
            raise ValueError("high_radius must be positive")
        if self.state_bound <= 0.0:
            raise ValueError("state_bound must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_default_config(config: Stage2Config) -> bool:
    return (
        config.state_seed == 4107
        and config.mask_seed == 4206
        and config.grid_size == 96
        and config.burn_in_updates == 800
    )


# --- Rollout dataclass ---


@dataclass(frozen=True)
class ControlledRollout:
    """Result of one controlled rollout, retaining data for exact replay."""

    frames: Tensor
    pulse_timeline: Tensor
    final_state: Tensor
    motion_per_update: Tensor
    initial_state_hash: str
    state_trajectory_hash: str
    fire_mask_hash: str
    per_channel_min: Tensor
    per_channel_max: Tensor
    final_variance: Tensor
    max_magnitude: float
    first_bound_violation: dict[str, Any] | None


# --- Helpers ---


def _tensor_hash(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _require_mps_without_fallback() -> torch.device:
    info = collect_runtime_info()
    require_mps(info)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Stage 2 forbids PYTORCH_ENABLE_MPS_FALLBACK=1")
    return torch.device("mps")


def _source_revision() -> str:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        return f"{revision}+dirty"
    return revision


def _gaussian_blob(grid_size: int, radius: float, device: torch.device) -> Tensor:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    center = (grid_size - 1) / 2.0
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r2 = (x - center).square() + (y - center).square()
    return torch.exp(-r2 / (2.0 * radius * radius))


def _high_band_wavelets(grid_size: int, radius: float, device: torch.device) -> Tensor:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    low_center = (grid_size - 1) * 0.35
    high_center = (grid_size - 1) * 0.65
    pattern = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)
    for center_y, center_x in (
        (low_center, low_center),
        (low_center, high_center),
        (high_center, low_center),
        (high_center, high_center),
    ):
        r2 = (x - center_x).square() + (y - center_y).square()
        gaussian = torch.exp(-r2 / (2.0 * radius * radius))
        pattern = pattern + (1.0 - r2 / (2.0 * radius * radius)) * gaussian
    peak = pattern.abs().max()
    if peak.item() == 0.0:
        raise RuntimeError("high-band wavelet pattern is empty")
    return pattern / peak


def _speed_scale(rms: float, config: Stage2Config) -> float:
    ratio = max(0.0, min(1.0, rms / config.rms_reference))
    return config.speed_min + ratio * (config.speed_max - config.speed_min)


def _low_band_dominance(low_band: float, high_band: float) -> float:
    total = low_band + high_band
    if total == 0.0:
        return 0.5
    return low_band / total


def _event_strength(
    onset: float,
    balance: float,
    prior_balance: float,
    config: Stage2Config,
) -> float:
    onset_term = max(0.0, min(1.0, onset / config.onset_reference))
    spectral_term = _spectral_event_strength(balance, prior_balance, config)
    return min(1.0, onset_term + spectral_term)


def _spectral_event_strength(balance: float, prior_balance: float, config: Stage2Config) -> float:
    balance_change = (
        max(0.0, abs(balance - prior_balance) - _EVENT_THRESHOLD) / config.spectral_reference
    )
    return min(1.0, balance_change)


def _rollouts_match(a: ControlledRollout, b: ControlledRollout) -> bool:
    return (
        a.initial_state_hash == b.initial_state_hash
        and a.fire_mask_hash == b.fire_mask_hash
        and a.state_trajectory_hash == b.state_trajectory_hash
        and torch.equal(a.frames, b.frames)
        and torch.equal(a.pulse_timeline, b.pulse_timeline)
    )


# --- Controlled rollout ---


def _controlled_rollout(
    model: NeuralCellularAutomaton,
    initial_state: Tensor,
    masks: Tensor,
    features: AudioFeatures,
    config: Stage2Config,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
) -> ControlledRollout:
    model.eval()
    n_frames = features.rms.numel()
    n_updates = n_frames * config.updates_per_frame
    if masks.shape[0] < n_updates:
        raise ValueError("not enough masks for this rollout")

    state = initial_state.clone()
    initial_hash = _tensor_hash(state[0])
    digest = hashlib.sha256()
    digest.update(state[0].detach().cpu().contiguous().numpy().tobytes())

    interp = interpolate_features(features, updates_per_frame=config.updates_per_frame)

    frames = torch.empty(
        n_frames, config.grid_size, config.grid_size, dtype=torch.float32, device=device
    )
    pulse_timeline = torch.empty(n_frames, 4, dtype=torch.float32, device=device)
    motion_per_update = torch.empty(n_updates, dtype=torch.float32, device=device)

    chan_min = state.amin(dim=(0, 2, 3)).clone()
    chan_max = state.amax(dim=(0, 2, 3)).clone()
    max_mag = state.abs().max().item()
    first_violation: dict[str, Any] | None = None

    prior_balance = 0.0

    with torch.no_grad():
        for frame_idx in range(n_frames):
            # Record luminance before any modifications.
            frames[frame_idx] = render_luminance(state, teacher_config)[0]

            # Event perturbation at the first update of this frame only.
            balance = float(features.low_band[frame_idx] - features.high_band[frame_idx])
            event = _event_strength(
                float(features.onset[frame_idx]),
                balance,
                prior_balance,
                config,
            )
            spectral_event = _spectral_event_strength(balance, prior_balance, config)
            prior_balance = float(features.low_band[frame_idx] - features.high_band[frame_idx])

            # Radius interpolated by low/high balance.
            lb = float(features.low_band[frame_idx])
            hb = float(features.high_band[frame_idx])
            low_dom = _low_band_dominance(lb, hb)
            radius = config.high_radius + low_dom * (config.low_radius - config.high_radius)
            band_level = min(1.0, max(lb, hb) / config.band_reference)
            spectral_sustain = band_level * config.spectral_sustain_amplitude

            pattern = None
            if event > 0.0:
                if spectral_event > 0.0 and hb > lb:
                    event_radius = config.high_radius
                    pattern = _high_band_wavelets(config.grid_size, config.high_radius, device)
                else:
                    event_radius = radius
                    pattern = _gaussian_blob(config.grid_size, radius, device)
            else:
                event_radius = 0.0
            if hb > lb:
                sustain_radius = config.high_radius
                sustain_pattern = _high_band_wavelets(config.grid_size, config.high_radius, device)
            else:
                sustain_radius = config.low_radius
                sustain_pattern = _gaussian_blob(config.grid_size, config.low_radius, device)
            pulse_timeline[frame_idx] = torch.tensor(
                (event, event_radius, spectral_sustain, sustain_radius),
                device=device,
                dtype=torch.float32,
            )

            for update_idx in range(config.updates_per_frame):
                gu = frame_idx * config.updates_per_frame + update_idx
                mask = masks[gu]

                prev_state = state.clone()

                if update_idx == 0 and event > 0.0 and pattern is not None:
                    state[0, 1] = state[0, 1] + event * config.perturbation_amplitude * pattern
                if update_idx == 0 and spectral_sustain > 0.0:
                    state[0, 1] = state[0, 1] + spectral_sustain * sustain_pattern

                rms = float(interp[gu, 0].item())
                scale = _speed_scale(rms, config)
                base_next = model(state, mask)
                state = state + scale * (base_next - state)

                if not torch.isfinite(state).all():
                    raise RuntimeError(
                        f"Non-finite state at frame {frame_idx}, update {update_idx}"
                    )

                motion_per_update[gu] = (state - prev_state).abs().mean()
                digest.update(state[0].detach().cpu().contiguous().numpy().tobytes())
                chan_min = torch.minimum(chan_min, state.amin(dim=(0, 2, 3)))
                chan_max = torch.maximum(chan_max, state.amax(dim=(0, 2, 3)))
                mag = state.abs().max().item()
                if mag > max_mag:
                    max_mag = mag
                if first_violation is None and mag >= config.state_bound:
                    first_violation = {
                        "frame": frame_idx,
                        "update": update_idx,
                        "magnitude": mag,
                    }

    fire_mask_hash = _tensor_hash(masks[:n_updates, 0])
    return ControlledRollout(
        frames=frames,
        pulse_timeline=pulse_timeline,
        final_state=state[0].clone(),
        motion_per_update=motion_per_update,
        initial_state_hash=initial_hash,
        state_trajectory_hash=digest.hexdigest(),
        fire_mask_hash=fire_mask_hash,
        per_channel_min=chan_min,
        per_channel_max=chan_max,
        final_variance=state.var(dim=(0, 2, 3), unbiased=False),
        max_magnitude=max_mag,
        first_bound_violation=first_violation,
    )


# --- Metric helpers ---


def _raw_motion(frames: Tensor) -> float:
    if frames.shape[0] < 2:
        return 0.0
    return (frames[1:] - frames[:-1]).abs().mean().item()


def _structural_motion(frames: Tensor) -> float:
    motion = _structural_motion_series(frames)
    if motion.numel() == 0:
        return 0.0
    return motion.mean().item()


def _structural_motion_series(frames: Tensor) -> Tensor:
    if frames.shape[0] < 2:
        return torch.empty(0, dtype=torch.float32, device=frames.device)
    centered = frames - frames.mean(dim=(-2, -1), keepdim=True)
    return (centered[1:] - centered[:-1]).abs().mean(dim=(-2, -1))


def _mean_luminance_motion(frames: Tensor) -> float:
    if frames.shape[0] < 2:
        return 0.0
    means = frames.mean(dim=(-2, -1))
    return (means[1:] - means[:-1]).abs().mean().item()


def _total_variation(frames: Tensor) -> float:
    h = (frames - torch.roll(frames, 1, dims=-1)).abs().mean()
    v = (frames - torch.roll(frames, 1, dims=-2)).abs().mean()
    return (h + v).item()


def _spatial_spectral_centroid(frames: Tensor) -> float:
    frames = frames.detach().cpu()
    centered = frames - frames.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(centered).abs().square()
    vf = torch.fft.fftfreq(frames.shape[-2])
    hf = torch.fft.rfftfreq(frames.shape[-1])
    radius = torch.sqrt(vf[:, None].square() + hf[None, :].square())
    total = power.sum()
    if total.item() == 0.0:
        return 0.0
    return (power * radius).sum().div(total).item()


def _response_lag(
    motion_per_frame: Tensor, onset: Tensor, updates_per_frame: int, fps: int
) -> dict[str, float | int]:
    motion_per_frame = motion_per_frame.detach().cpu()
    onset = onset.detach().cpu()
    n = min(motion_per_frame.numel(), onset.numel())
    if n < 2:
        return {"lag_updates": 0, "lag_frames": 0, "lag_ms": 0.0, "peak_correlation": 0.0}
    a = motion_per_frame[:n] - motion_per_frame[:n].mean()
    b = onset[:n] - onset[:n].mean()
    a_std = a.std().item()
    b_std = b.std().item()
    if a_std == 0.0 or b_std == 0.0:
        return {"lag_updates": 0, "lag_frames": 0, "lag_ms": 0.0, "peak_correlation": 0.0}
    max_lag = min(_THR_LAG_MAX_FRAMES * 2, n - 2)
    best_lag = 0
    best_corr = 0.0
    for lag in range(max_lag + 1):
        length = n - lag
        if length < 2:
            break
        response = a[lag:]
        control = b[:length]
        response = response - response.mean()
        control = control - control.mean()
        response_std = response.std(unbiased=False).item()
        control_std = control.std(unbiased=False).item()
        if response_std == 0.0 or control_std == 0.0:
            continue
        corr = (response * control).mean().item() / (response_std * control_std)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    lag_frames = best_lag + 1
    return {
        "lag_updates": lag_frames * updates_per_frame,
        "lag_frames": lag_frames,
        "lag_ms": lag_frames * 1000.0 / fps,
        "peak_correlation": best_corr,
    }


def _onset_response(
    frames: Tensor, impulse_frame: int, fps: int, updates_per_frame: int
) -> dict[str, float | int]:
    if frames.shape[0] < 2:
        return {"baseline": 0.0, "peak": 0.0, "time_to_peak_frames": 0, "recovery_frames": 0}
    motion = _structural_motion_series(frames)
    baseline_start = max(0, impulse_frame - fps // 2)
    baseline_window = motion[baseline_start:impulse_frame]
    if baseline_window.numel() == 0:
        raise ValueError("impulse frame must leave a nonempty baseline window")
    baseline = baseline_window.mean().item()
    if baseline <= 0.0:
        raise ValueError("onset response requires positive baseline motion")
    window = min(fps, motion.numel() - impulse_frame)
    if window <= 0:
        return {"baseline": baseline, "peak": 0.0, "time_to_peak_frames": 0, "recovery_frames": 0}
    response = motion[impulse_frame : impulse_frame + window]
    peak_idx = int(response.argmax().item())
    peak_val = response[peak_idx].item()
    after = motion[impulse_frame + peak_idx :]
    threshold = baseline * (1.0 + _THR_RECOVERY_RATIO)
    recovery = 0
    for i in range(1, after.numel()):
        if after[i].item() <= threshold:
            recovery = i
            break
    time_to_peak_frames = peak_idx + 1
    return {
        "baseline": baseline,
        "peak": peak_val,
        "peak_to_baseline_ratio": peak_val / baseline,
        "time_to_peak_updates": time_to_peak_frames * updates_per_frame,
        "time_to_peak_frames": time_to_peak_frames,
        "time_to_peak_ms": time_to_peak_frames * 1000.0 / fps,
        "recovery_updates": recovery * updates_per_frame,
        "recovery_frames": recovery,
        "recovery_ms": recovery * 1000.0 / fps,
    }


def _block_energy_motion(frames: Tensor, audio_rms: Tensor, block_frames: int) -> dict[str, float]:
    if frames.shape[0] < 2:
        return {"high_energy_motion": 0.0, "low_energy_motion": 0.0, "ratio": 0.0}
    motion = (frames[1:] - frames[:-1]).abs().mean(dim=(-2, -1))
    if block_frames <= 0:
        raise ValueError("block_frames must be positive")
    aligned_rms = audio_rms[: motion.numel()].to(motion.device)
    high = motion[aligned_rms >= _ENERGY_SPLIT_RMS]
    low = motion[aligned_rms < _ENERGY_SPLIT_RMS]
    if high.numel() == 0 or low.numel() == 0:
        raise ValueError("energy probe must contain both high and low frames")
    ha = high.mean().item()
    la = low.mean().item()
    if la <= 0.0:
        raise ValueError("low-energy block motion must be positive")
    ratio = ha / la
    return {"high_energy_motion": ha, "low_energy_motion": la, "ratio": ratio}


def _spectral_block_metrics(
    frames: Tensor, audio_low: Tensor, audio_high: Tensor, block_frames: int
) -> dict[str, float]:
    n_blocks = min(audio_low.numel(), frames.shape[0]) // block_frames
    lt, ht, lc, hc = [], [], [], []
    for i in range(n_blocks):
        s, e = i * block_frames, (i + 1) * block_frames
        if e > frames.shape[0]:
            break
        blk = frames[s:e]
        tv = _total_variation(blk)
        cen = _spatial_spectral_centroid(blk)
        if audio_low[s:e].mean().item() > audio_high[s:e].mean().item():
            lt.append(tv)
            lc.append(cen)
        else:
            ht.append(tv)
            hc.append(cen)
    if not lt or not ht or not lc or not hc:
        raise ValueError("spectral probe must contain both low and high blocks")
    low_tv = sum(lt) / len(lt)
    high_tv = sum(ht) / len(ht)
    low_centroid = sum(lc) / len(lc)
    high_centroid = sum(hc) / len(hc)
    tv_scale = max(abs(low_tv), abs(high_tv))
    centroid_scale = max(abs(low_centroid), abs(high_centroid))
    if tv_scale <= 0.0 or centroid_scale <= 0.0:
        raise ValueError("spectral block metrics require nonzero spatial structure")
    return {
        "low_tv": low_tv,
        "high_tv": high_tv,
        "tv_relative_difference": abs(low_tv - high_tv) / tv_scale,
        "low_centroid": low_centroid,
        "high_centroid": high_centroid,
        "centroid_relative_difference": abs(low_centroid - high_centroid) / centroid_scale,
    }


def _trajectory_divergence(a: Tensor, b: Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        return {"raw": float("inf"), "structural": float("inf")}
    raw = (a - b).abs().mean().item()
    ac = a - a.mean(dim=(-2, -1), keepdim=True)
    bc = b - b.mean(dim=(-2, -1), keepdim=True)
    structural = (ac - bc).abs().mean().item()
    return {"raw": raw, "structural": structural}


def _compute_rollout_metrics(
    rollout: ControlledRollout,
    features: AudioFeatures,
    config: Stage2Config,
    fps: int,
) -> dict[str, Any]:
    frames = rollout.frames
    m: dict[str, Any] = {
        "raw_motion": _raw_motion(frames),
        "structural_motion": _structural_motion(frames),
        "mean_luminance_motion": _mean_luminance_motion(frames),
        "total_variation": _total_variation(frames),
        "spatial_spectral_centroid": _spatial_spectral_centroid(frames),
        "response_lag": _response_lag(
            _structural_motion_series(frames),
            features.onset[:-1],
            config.updates_per_frame,
            fps,
        ),
        "max_magnitude": rollout.max_magnitude,
        "first_bound_violation": rollout.first_bound_violation,
        "initial_state_hash": rollout.initial_state_hash,
        "state_trajectory_hash": rollout.state_trajectory_hash,
        "fire_mask_hash": rollout.fire_mask_hash,
        "per_channel_min": rollout.per_channel_min.cpu().tolist(),
        "per_channel_max": rollout.per_channel_max.cpu().tolist(),
        "final_variance": rollout.final_variance.cpu().tolist(),
    }
    raw = m["raw_motion"]
    if raw > 0.0:
        m["mean_luminance_contribution"] = m["mean_luminance_motion"] / raw
    else:
        m["mean_luminance_contribution"] = 0.0
    return m


# --- Burn-in and mask generation ---


def _generate_shared_state(
    config: Stage2Config,
    teacher_config: ReactionDiffusionConfig,
    max_probe_updates: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Recreate the Stage 1 distributed burn-in and return (initial_state, masks)."""
    total_masks = config.burn_in_updates + max_probe_updates
    fields = make_teacher_fields(
        initialization="distributed",
        batch_size=1,
        grid_size=config.grid_size,
        device=device,
        generator=torch.Generator(device=device).manual_seed(config.state_seed),
    )
    masks = _sample_fire_masks(
        steps=total_masks,
        batch_size=1,
        grid_size=config.grid_size,
        device=device,
        generator=torch.Generator(device=device).manual_seed(config.mask_seed),
    )
    with torch.no_grad():
        for i in range(config.burn_in_updates):
            fields = masked_reaction_diffusion_step(fields, masks[i], teacher_config)
            if not torch.isfinite(fields).all():
                raise RuntimeError(f"Non-finite teacher fields at burn-in step {i}")
    state = encode_teacher_fields(
        fields,
        state_channels=BASELINE_CONFIG.state_channels,
        config=teacher_config,
    )
    return state, masks[config.burn_in_updates :]


# --- Automated checks ---


def _automated_checks(
    all_finite: bool,
    all_bounded: bool,
    exact_replay: bool,
    identical_blind_audio: bool,
    impulse_metrics: dict[str, Any],
    alternating_metrics: dict[str, float],
    spectral_metrics: dict[str, float],
    review_onset: dict[str, Any],
    lag: dict[str, Any],
    div_silent: dict[str, float],
    div_shuffled: dict[str, float],
    structure_ratio: float,
) -> dict[str, bool]:
    return {
        "all_finite": all_finite,
        "all_states_below_2_0": all_bounded,
        "exact_replay": exact_replay,
        "identical_blind_audio": identical_blind_audio,
        "impulse_prompt": (impulse_metrics["peak_to_baseline_ratio"] >= _THR_IMPULSE_RATIO),
        "impulse_localized": (
            impulse_metrics["time_to_peak_frames"] < _THR_IMPULSE_WINDOW
            and impulse_metrics["peak"] > 0.0
        ),
        "alternating_energy": alternating_metrics["ratio"] >= _THR_ALTERNATING_RATIO,
        "spectral_spatial": (
            spectral_metrics["low_centroid"] > 0.0
            and spectral_metrics["high_centroid"] > 0.0
            and max(
                spectral_metrics["tv_relative_difference"],
                spectral_metrics["centroid_relative_difference"],
            )
            >= _THR_SPECTRAL_RELATIVE_DIFFERENCE
        ),
        "recovery": (0 < review_onset["recovery_frames"] <= _THR_RECOVERY_MAX_FRAMES),
        "lag_acceptable": (
            0 < lag["lag_frames"] <= _THR_LAG_MAX_FRAMES and lag["peak_correlation"] > 0.0
        ),
        "correct_vs_silent_divergence": (
            div_silent["raw"] > _THR_DIVERGENCE_MIN
            and div_silent["structural"] > _THR_DIVERGENCE_MIN
        ),
        "correct_vs_shuffled_divergence": (
            div_shuffled["raw"] > _THR_DIVERGENCE_MIN
            and div_shuffled["structural"] > _THR_DIVERGENCE_MIN
        ),
        "response_not_brightness": structure_ratio >= _THR_STRUCTURE_RATIO,
    }


# --- Main run ---


def run_stage2(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config: Stage2Config | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if config is None:
        config = Stage2Config()
    if device is None:
        device = _require_mps_without_fallback()
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    output_dir.mkdir(parents=True, exist_ok=False)

    audio_config = AudioConfig()
    media_config = MediaConfig()
    fps = audio_config.frame_rate
    teacher_config = replace(
        REACTION_DIFFUSION_CONFIG, time_step=MASK_ALIGNED_RAPID_BUDGET.teacher_time_step
    )

    # Load and freeze the checkpoint.
    model, checkpoint_metadata = load_stage1_checkpoint(
        checkpoint_path,
        expected_model_config=BASELINE_CONFIG,
        expected_teacher_config=teacher_config,
        expected_training_config=MASK_ALIGNED_RAPID_BUDGET.to_dict(),
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Synthesize probes, write WAVs and feature timelines.
    probes: dict[str, tuple[Tensor, AudioFeatures]] = {}
    feature_timeline_paths: dict[str, str] = {}
    wav_paths: dict[str, str] = {}
    for name in PROBE_NAMES:
        audio = synthesize_probe(name, audio_config, seed=config.audio_seed)
        features = extract_features(audio, audio_config)
        probes[name] = (audio, features)
        wav_path = output_dir / f"probe-{name}.wav"
        write_pcm16_wav(wav_path, audio, audio_config.sample_rate)
        wav_paths[name] = str(wav_path)
        ft_path = output_dir / f"features-{name}.npz"
        np.savez_compressed(
            ft_path,
            rms=features.rms.numpy(),
            onset=features.onset.numpy(),
            spectral_flux=features.spectral_flux.numpy(),
            low_band=features.low_band.numpy(),
            high_band=features.high_band.numpy(),
        )
        feature_timeline_paths[name] = str(ft_path)

    # Generate shared initial state and masks.
    max_frames = max(f.rms.numel() for _, f in probes.values())
    max_probe_updates = max_frames * config.updates_per_frame
    initial_state, post_masks = _generate_shared_state(
        config, teacher_config, max_probe_updates, device
    )

    # Hash check for default config on MPS.
    if _is_default_config(config) and device.type == "mps":
        actual_hash = _tensor_hash(initial_state[0])
        if actual_hash != EXPECTED_DISTRIBUTED_HASH:
            raise RuntimeError(
                f"Initial state hash mismatch: {actual_hash} != {EXPECTED_DISTRIBUTED_HASH}"
            )

    # Save shared state and masks.
    shared_path = output_dir / "shared-initial-state-and-masks.npz"
    np.savez_compressed(
        shared_path,
        initial_state=initial_state[0].detach().cpu().numpy(),
        fire_masks=post_masks[:max_probe_updates, 0].detach().cpu().numpy(),
    )

    # Rollout standalone probes (correct features).
    standalone_rollouts: dict[str, ControlledRollout] = {}
    standalone_npz_paths: dict[str, str] = {}
    standalone_video_paths: dict[str, str] = {}
    for name in PROBE_NAMES:
        _, features = probes[name]
        n_updates = features.rms.numel() * config.updates_per_frame
        rollout = _controlled_rollout(
            model,
            initial_state,
            post_masks[:n_updates],
            features,
            config,
            teacher_config,
            device,
        )
        standalone_rollouts[name] = rollout
        npz_path = output_dir / f"rollout-{name}.npz"
        np.savez_compressed(
            npz_path,
            frames=rollout.frames.detach().cpu().numpy(),
            feature_timeline=features.matrix().numpy(),
            pulse_timeline=rollout.pulse_timeline.detach().cpu().numpy(),
            motion_per_update=rollout.motion_per_update.detach().cpu().numpy(),
            final_state=rollout.final_state.detach().cpu().numpy(),
        )
        standalone_npz_paths[name] = str(npz_path)
        mp4_path = output_dir / f"video-{name}.mp4"
        save_audiovisual(
            mp4_path,
            frames=rollout.frames.detach().cpu(),
            audio_path=wav_paths[name],
            config=media_config,
            metadata={},
        )
        standalone_video_paths[name] = str(mp4_path)

    # Review counterfactual rollouts.
    review_audio, review_features = probes["review"]
    review_rollouts: dict[str, ControlledRollout] = {}
    review_rollouts["correct"] = standalone_rollouts["review"]
    for cond in ("silent", "shuffled", "frozen-mean"):
        cf_features = counterfactual_features(review_features, cond, seed=config.shuffle_seed)
        n_updates = cf_features.rms.numel() * config.updates_per_frame
        review_rollouts[cond] = _controlled_rollout(
            model,
            initial_state,
            post_masks[:n_updates],
            cf_features,
            config,
            teacher_config,
            device,
        )

    # Exact replay of correct review.
    review_replay = _controlled_rollout(
        model,
        initial_state,
        post_masks[: review_features.rms.numel() * config.updates_per_frame],
        review_features,
        config,
        teacher_config,
        device,
    )
    exact_replay = _rollouts_match(review_rollouts["correct"], review_replay)

    # Save review condition MP4s (all carry original correct review audio).
    review_video_paths: dict[str, str] = {}
    review_video_paths["correct"] = standalone_video_paths["review"]
    for cond in ("silent", "shuffled", "frozen-mean"):
        mp4_path = output_dir / f"review-{cond}.mp4"
        save_audiovisual(
            mp4_path,
            frames=review_rollouts[cond].frames.detach().cpu(),
            audio_path=wav_paths["review"],
            config=media_config,
            metadata={},
        )
        review_video_paths[cond] = str(mp4_path)

    # Save review rollout npz per condition.
    review_npz_paths: dict[str, str] = {}
    review_npz_paths["correct"] = standalone_npz_paths["review"]
    for cond in ("silent", "shuffled", "frozen-mean"):
        npz_path = output_dir / f"rollout-review-{cond}.npz"
        condition_features = counterfactual_features(
            review_features, cond, seed=config.shuffle_seed
        )
        np.savez_compressed(
            npz_path,
            frames=review_rollouts[cond].frames.detach().cpu().numpy(),
            feature_timeline=condition_features.matrix().numpy(),
            pulse_timeline=review_rollouts[cond].pulse_timeline.detach().cpu().numpy(),
            motion_per_update=review_rollouts[cond].motion_per_update.detach().cpu().numpy(),
            final_state=review_rollouts[cond].final_state.detach().cpu().numpy(),
        )
        review_npz_paths[cond] = str(npz_path)

    # Labeled comparison MP4 (correct, silent, shuffled).
    labeled_path = output_dir / "comparison-labeled.mp4"
    save_comparison_audiovisual(
        labeled_path,
        panels=[
            ("Correct features", review_rollouts["correct"].frames.detach().cpu()),
            ("Silent controls", review_rollouts["silent"].frames.detach().cpu()),
            ("Shuffled features", review_rollouts["shuffled"].frames.detach().cpu()),
        ],
        audio_path=wav_paths["review"],
        config=media_config,
        metadata={},
    )

    # Blind mapping from shuffle_seed.
    blind_perm = torch.randperm(3, generator=torch.Generator().manual_seed(config.shuffle_seed))
    blind_order = [BLIND_CONDITIONS[int(i)] for i in blind_perm.tolist()]
    blind_key = {chr(65 + i): blind_order[i] for i in range(3)}
    private_dir = output_dir / "private"
    private_dir.mkdir()
    blind_key_path = private_dir / "blind-key.json"
    blind_key_path.write_text(json.dumps(blind_key, indent=2) + "\n", encoding="utf-8")

    # Blind comparison MP4 (A, B, C).
    blind_panels = [
        (chr(65 + i), review_rollouts[blind_order[i]].frames.detach().cpu()) for i in range(3)
    ]
    blind_comparison_path = output_dir / "comparison-blind.mp4"
    save_comparison_audiovisual(
        blind_comparison_path,
        panels=blind_panels,
        audio_path=wav_paths["review"],
        config=media_config,
        metadata={},
    )

    # Blind individual MP4s.
    blind_individual_paths: list[str] = []
    for i in range(3):
        letter = chr(65 + i)
        ipath = output_dir / f"blind-{letter.lower()}.mp4"
        save_audiovisual(
            ipath,
            frames=review_rollouts[blind_order[i]].frames.detach().cpu(),
            audio_path=wav_paths["review"],
            config=media_config,
            metadata={},
        )
        blind_individual_paths.append(str(ipath))

    # Assert all blind AAC stream hashes equal.
    blind_hashes = [audio_stream_hash(Path(p)) for p in blind_individual_paths]
    blind_hashes.append(audio_stream_hash(blind_comparison_path))
    identical_blind_audio = len(set(blind_hashes)) == 1

    # Review page.
    review_dir = output_dir / "review"
    review_page_path = review_dir / "review.html"
    write_review_page(
        review_page_path,
        labeled_video=labeled_path,
        blind_video=blind_comparison_path,
        blind_individuals=[Path(p) for p in blind_individual_paths],
    )

    # Compute metrics.
    all_rollouts = list(standalone_rollouts.values()) + [
        review_rollouts[c] for c in ("silent", "shuffled", "frozen-mean")
    ]
    all_finite = all(
        torch.isfinite(rollout.frames).all().item()
        and torch.isfinite(rollout.final_state).all().item()
        for rollout in all_rollouts
    )
    all_bounded = all(r.max_magnitude < _THR_BOUND for r in all_rollouts)

    # Per-probe metrics.
    probe_metrics: dict[str, Any] = {}
    for name in PROBE_NAMES:
        _, features = probes[name]
        m = _compute_rollout_metrics(standalone_rollouts[name], features, config, fps)
        if name == "impulse":
            m["onset_response_2s"] = _onset_response(
                standalone_rollouts[name].frames,
                2 * fps,
                fps,
                config.updates_per_frame,
            )
        if name == "review":
            m["onset_response_3s"] = _onset_response(
                standalone_rollouts[name].frames,
                3 * fps,
                fps,
                config.updates_per_frame,
            )
        if name == "alternating":
            block = (audio_config.sample_rate // 2) // audio_config.samples_per_frame
            m["block_energy_motion"] = _block_energy_motion(
                standalone_rollouts[name].frames, features.rms, block
            )
        if name == "stepped-bands":
            block = (audio_config.sample_rate // 2) // audio_config.samples_per_frame
            m["spectral_block_metrics"] = _spectral_block_metrics(
                standalone_rollouts[name].frames, features.low_band, features.high_band, block
            )
        probe_metrics[name] = m

    # Review condition metrics.
    review_condition_metrics: dict[str, Any] = {}
    for cond in ("correct", "silent", "shuffled", "frozen-mean"):
        cf = counterfactual_features(review_features, cond, seed=config.shuffle_seed)
        review_condition_metrics[cond] = _compute_rollout_metrics(
            review_rollouts[cond], cf, config, fps
        )

    # Divergences.
    correct_frames = review_rollouts["correct"].frames
    div_silent = _trajectory_divergence(correct_frames, review_rollouts["silent"].frames)
    div_shuffled = _trajectory_divergence(correct_frames, review_rollouts["shuffled"].frames)

    # Structure ratio for brightness check.
    correct_metrics = review_condition_metrics["correct"]
    raw = correct_metrics["raw_motion"]
    struct = correct_metrics["structural_motion"]
    if raw <= 0.0:
        raise RuntimeError("Stage 2 review requires nonzero visible motion")
    structure_ratio = struct / raw

    # Lag from review correct.
    lag = correct_metrics["response_lag"]

    # Automated checks.
    checks = _automated_checks(
        all_finite=all_finite,
        all_bounded=all_bounded,
        exact_replay=exact_replay,
        identical_blind_audio=identical_blind_audio,
        impulse_metrics=probe_metrics["impulse"]["onset_response_2s"],
        alternating_metrics=probe_metrics["alternating"]["block_energy_motion"],
        spectral_metrics=probe_metrics["stepped-bands"]["spectral_block_metrics"],
        review_onset=probe_metrics["review"]["onset_response_3s"],
        lag=lag,
        div_silent=div_silent,
        div_shuffled=div_shuffled,
        structure_ratio=structure_ratio,
    )

    # Build result.json.
    source_rev = _source_revision()
    result: dict[str, Any] = {
        "stage": 2,
        "source_revision": source_rev,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_fallback_enabled": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "metadata": {
                "source_revision": checkpoint_metadata["source_revision"],
                "optimizer_steps_completed": checkpoint_metadata["optimizer_steps_completed"],
            },
        },
        "model": {
            "config": BASELINE_CONFIG.to_dict(),
            "parameter_count": model.parameter_count,
        },
        "stage2_config": config.to_dict(),
        "audio": {
            "feature_names": list(FEATURE_NAMES),
            "band_definitions": BAND_DEFINITIONS,
            "config": {
                "sample_rate": audio_config.sample_rate,
                "frame_rate": audio_config.frame_rate,
            },
            "feature_timeline_paths": feature_timeline_paths,
            "wav_paths": wav_paths,
        },
        "seeds": {
            "state": config.state_seed,
            "mask": config.mask_seed,
            "audio": config.audio_seed,
            "shuffle": config.shuffle_seed,
        },
        "metrics": {
            "probes": probe_metrics,
            "review_conditions": review_condition_metrics,
        },
        "counterfactual_divergences": {
            "correct_vs_silent": div_silent,
            "correct_vs_shuffled": div_shuffled,
        },
        "exact_replay": {
            "passed": exact_replay,
            "initial_state_hash": review_rollouts["correct"].initial_state_hash,
            "state_trajectory_hash": review_rollouts["correct"].state_trajectory_hash,
            "fire_mask_hash": review_rollouts["correct"].fire_mask_hash,
        },
        "control_config": config.to_dict(),
        "render_mapping": RENDER_MAPPING,
        "fps": fps,
        "sync_offset_seconds": 0,
        "artifacts": {
            "shared_state": str(shared_path),
            "standalone_npz": standalone_npz_paths,
            "standalone_video": standalone_video_paths,
            "review_npz": review_npz_paths,
            "review_video": review_video_paths,
            "labeled_comparison": str(labeled_path),
            "blind_comparison": str(blind_comparison_path),
            "blind_individuals": blind_individual_paths,
            "review_page": str(review_page_path),
            "pulse_timeline_columns": [
                "event_strength",
                "event_radius",
                "spectral_sustain_amplitude",
                "spectral_sustain_radius",
            ],
        },
        "blind_key_path": str(blind_key_path),
        "automated_checks": checks,
        "automated_checks_passed": all(checks.values()),
        "interpretation_boundary": (
            "Automated checks cover finiteness, bounds, replay, audio identity, "
            "and response properties; coherent dynamics and audio causality still "
            "require human review of the rendered artifacts."
        ),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _print_summary(result: dict[str, Any]) -> None:
    checks = result["automated_checks"]
    print(f"Automated checks passed: {result['automated_checks_passed']}")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
    print(f"Result: {result['artifacts']['review_page']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke",))
    parser.add_argument("--checkpoint", type=Path, default=RETAINED_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_stage2(args.checkpoint, args.output_dir)
    _print_summary(result)


if __name__ == "__main__":
    main()
