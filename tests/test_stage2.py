"""Focused CPU unit tests for the Stage 2 non-learned control loop.

These tests exercise the pure helpers and the controlled rollout on a small
16x16 grid on CPU. No checkpoint or media integration is performed.
"""

from __future__ import annotations

import math

import pytest
import torch

from synesthete.audio import AudioFeatures
from synesthete.nca import BASELINE_CONFIG, NeuralCellularAutomaton
from synesthete.stage2 import (
    _EVENT_THRESHOLD,
    ControlledRollout,
    Stage2Config,
    _automated_checks,
    _controlled_rollout,
    _event_strength,
    _gaussian_blob,
    _high_band_wavelets,
    _response_lag,
    _rollouts_match,
    _speed_scale,
    _structural_motion,
    _trajectory_divergence,
)
from synesthete.teacher import REACTION_DIFFUSION_CONFIG

CPU = torch.device("cpu")


def _features(
    rms: torch.Tensor,
    onset: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    flux: torch.Tensor | None = None,
) -> AudioFeatures:
    n = rms.numel()
    if flux is None:
        flux = torch.zeros(n, dtype=torch.float32)
    return AudioFeatures(
        rms=rms.to(torch.float32),
        onset=onset.to(torch.float32),
        spectral_flux=flux.to(torch.float32),
        low_band=low.to(torch.float32),
        high_band=high.to(torch.float32),
    )


# --- Stage2Config validation ---


def test_stage2_config_default_constructs() -> None:
    config = Stage2Config()
    assert config.grid_size == 96
    assert config.updates_per_frame == 4
    assert config.state_bound == 2.0
    keys = set(config.to_dict())
    assert {"grid_size", "updates_per_frame", "state_seed", "speed_min"} <= keys


def test_stage2_config_grid_size_floor() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        Stage2Config(grid_size=15)


def test_stage2_config_updates_per_frame_positive() -> None:
    with pytest.raises(ValueError, match="updates_per_frame"):
        Stage2Config(updates_per_frame=0)


def test_stage2_config_burn_in_non_negative() -> None:
    with pytest.raises(ValueError, match="burn_in_updates"):
        Stage2Config(burn_in_updates=-1)


@pytest.mark.parametrize("name", ["state_seed", "mask_seed", "audio_seed", "shuffle_seed"])
def test_stage2_config_seeds_must_be_int(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Stage2Config(**{name: 1.5})


@pytest.mark.parametrize("name", ["state_seed", "mask_seed", "audio_seed", "shuffle_seed"])
def test_stage2_config_seeds_reject_bool(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Stage2Config(**{name: True})


def test_stage2_config_speed_ordering() -> None:
    with pytest.raises(ValueError, match="speed_min"):
        Stage2Config(speed_min=0.0)
    with pytest.raises(ValueError, match="speed_max"):
        Stage2Config(speed_min=1.5, speed_max=1.5)
    with pytest.raises(ValueError, match="speed_max"):
        Stage2Config(speed_min=2.0, speed_max=1.0)


@pytest.mark.parametrize(
    "name",
    [
        "rms_reference",
        "onset_reference",
        "spectral_reference",
        "band_reference",
        "spectral_sustain_amplitude",
        "perturbation_amplitude",
        "low_radius",
        "high_radius",
        "state_bound",
    ],
)
def test_stage2_config_positive_fields(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Stage2Config(**{name: 0.0})


# --- _speed_scale endpoints ---


def test_speed_scale_endpoints() -> None:
    config = Stage2Config()
    lo, hi, ref = config.speed_min, config.speed_max, config.rms_reference
    assert _speed_scale(0.0, config) == lo
    assert _speed_scale(ref, config) == hi
    # Clamps above the reference.
    assert _speed_scale(10.0 * ref, config) == hi
    # Linear interpolation at the midpoint.
    assert _speed_scale(ref / 2.0, config) == pytest.approx((lo + hi) / 2.0)
    # Negative rms is clamped to zero ratio.
    assert _speed_scale(-1.0, config) == lo


# --- _event_strength ---


def test_event_strength_zero_when_quiet() -> None:
    config = Stage2Config()
    assert _event_strength(0.0, 0.1, 0.1, config) == 0.0


def test_event_strength_onset_dominates() -> None:
    config = Stage2Config()
    assert _event_strength(config.onset_reference, 0.0, 0.0, config) == 1.0
    assert _event_strength(10.0 * config.onset_reference, 0.0, 0.0, config) == 1.0


def test_event_strength_balance_change_must_exceed_threshold() -> None:
    config = Stage2Config()
    # A balance change below the threshold contributes nothing.
    assert _event_strength(0.0, 0.0, _EVENT_THRESHOLD, config) == 0.0
    # A balance change just above the threshold contributes the excess.
    delta = _EVENT_THRESHOLD + 0.2
    assert _event_strength(0.0, delta, 0.0, config) == pytest.approx(
        0.2 / config.spectral_reference
    )
    # Onset plus balance change is capped at one.
    assert _event_strength(10.0 * config.onset_reference, 1.0, 0.0, config) == 1.0


# --- _gaussian_blob ---


def test_gaussian_blob_shape_and_dtype() -> None:
    blob = _gaussian_blob(16, 9.0, CPU)
    assert blob.shape == (16, 16)
    assert blob.dtype == torch.float32


def test_gaussian_blob_peak_is_one_for_odd_grid() -> None:
    blob = _gaussian_blob(15, 9.0, CPU)
    assert blob[7, 7].item() == pytest.approx(1.0)
    assert blob.max().item() == pytest.approx(1.0)


def test_gaussian_blob_broad_spread_exceeds_compact() -> None:
    broad = _gaussian_blob(15, 14.0, CPU)
    compact = _gaussian_blob(15, 2.0, CPU)
    # Both peak at the centre.
    assert broad[7, 7].item() == pytest.approx(1.0)
    assert compact[7, 7].item() == pytest.approx(1.0)
    # Broad profile retains far more energy at the corner than compact.
    assert broad[0, 0].item() > compact[0, 0].item()
    # Broad profile spreads its energy across the grid (higher mean) while the
    # compact profile concentrates it near the centre and decays elsewhere.
    assert broad.mean().item() > compact.mean().item()
    # Compact profile decays to nearly nothing at the corner.
    assert compact[0, 0].item() < 1e-3


def test_high_band_wavelets_are_bounded_signed_and_distributed() -> None:
    pattern = _high_band_wavelets(96, 4.0, CPU)
    assert pattern.shape == (96, 96)
    assert pattern.abs().max().item() == pytest.approx(1.0)
    assert pattern.min().item() < 0.0
    assert torch.count_nonzero(pattern > 0.9) >= 4


# --- _controlled_rollout deterministic exact replay ---


def _nonzero_output_model(seed: int = 0) -> NeuralCellularAutomaton:
    torch.manual_seed(seed)
    model = NeuralCellularAutomaton(BASELINE_CONFIG)
    with torch.no_grad():
        model.update_output.weight.normal_(0.0, 0.01)
        model.update_output.bias.normal_(0.0, 0.01)
    model.eval()
    return model


def _replay_inputs() -> tuple[
    NeuralCellularAutomaton, torch.Tensor, torch.Tensor, AudioFeatures, Stage2Config
]:
    torch.manual_seed(123)
    model = _nonzero_output_model()
    initial = torch.randn(1, 16, 16, 16, dtype=torch.float32).mul_(0.1)
    n_frames = 4
    updates_per_frame = 2
    masks = torch.ones(n_frames * updates_per_frame, 1, 1, 16, 16, dtype=torch.float32)
    features = _features(
        rms=torch.full((n_frames,), 0.2),
        onset=torch.tensor([0.0, 0.2, 0.0, 0.0]),
        low=torch.full((n_frames,), 0.1),
        high=torch.full((n_frames,), 0.1),
    )
    config = Stage2Config(grid_size=16, updates_per_frame=updates_per_frame, burn_in_updates=0)
    return model, initial, masks, features, config


def test_controlled_rollout_frame_count_and_hash_shapes() -> None:
    model, initial, masks, features, config = _replay_inputs()
    rollout = _controlled_rollout(
        model, initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU
    )
    n_frames = features.rms.numel()
    assert rollout.frames.shape == (n_frames, 16, 16)
    assert rollout.pulse_timeline.shape == (n_frames, 4)
    assert rollout.motion_per_update.shape == (n_frames * config.updates_per_frame,)
    assert rollout.final_state.shape == (16, 16, 16)
    for h in (rollout.initial_state_hash, rollout.state_trajectory_hash, rollout.fire_mask_hash):
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # valid hex


def test_controlled_rollout_is_deterministic_exact_replay() -> None:
    model, initial, masks, features, config = _replay_inputs()
    a = _controlled_rollout(model, initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU)
    b = _controlled_rollout(model, initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU)
    assert _rollouts_match(a, b)
    assert torch.equal(a.frames, b.frames)
    assert torch.equal(a.pulse_timeline, b.pulse_timeline)
    assert torch.equal(a.final_state, b.final_state)
    assert a.initial_state_hash == b.initial_state_hash
    assert a.state_trajectory_hash == b.state_trajectory_hash
    assert a.fire_mask_hash == b.fire_mask_hash


def test_controlled_rollout_hash_depends_on_initial_state() -> None:
    model, initial, masks, features, config = _replay_inputs()
    other_initial = initial.clone() + 0.001
    a = _controlled_rollout(model, initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU)
    b = _controlled_rollout(
        model, other_initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU
    )
    assert a.initial_state_hash != b.initial_state_hash
    assert not torch.equal(a.frames, b.frames)


# --- Pulse perturbs only channel 1 immediately ---


def _channel1_inputs() -> tuple[
    NeuralCellularAutomaton, torch.Tensor, torch.Tensor, AudioFeatures, AudioFeatures, Stage2Config
]:
    # Zero output projection: the NCA delta is identically zero, so the only state
    # change over a rollout is the channel-1 pulse applied by the control loop.
    model = NeuralCellularAutomaton(BASELINE_CONFIG)
    model.eval()
    torch.manual_seed(7)
    initial = torch.randn(1, 16, 16, 16, dtype=torch.float32).mul_(0.1)
    masks = torch.ones(3, 1, 1, 16, 16, dtype=torch.float32)
    event = _features(
        rms=torch.full((3,), 0.2),
        onset=torch.tensor([0.0, 0.2, 0.2]),
        low=torch.zeros(3),
        high=torch.zeros(3),
    )
    quiet = _features(
        rms=torch.full((3,), 0.2),
        onset=torch.zeros(3),
        low=torch.zeros(3),
        high=torch.zeros(3),
    )
    config = Stage2Config(grid_size=16, updates_per_frame=1, burn_in_updates=0)
    return model, initial, masks, event, quiet, config


def test_pulse_only_touches_channel_one_immediately() -> None:
    model, initial, masks, event, quiet, config = _channel1_inputs()
    quiet_rollout = _controlled_rollout(
        model, initial, masks, quiet, config, REACTION_DIFFUSION_CONFIG, CPU
    )
    event_rollout = _controlled_rollout(
        model, initial, masks, event, config, REACTION_DIFFUSION_CONFIG, CPU
    )

    # With no event the state is completely unchanged (NCA delta is zero).
    assert torch.equal(quiet_rollout.final_state, initial[0])

    # The pulse timeline records an event only for frames with onset.
    assert torch.equal(quiet_rollout.pulse_timeline[:, 0], torch.zeros(3))
    assert torch.equal(event_rollout.pulse_timeline[:, 0], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(quiet_rollout.pulse_timeline[:, 1], torch.zeros(3))
    assert torch.equal(event_rollout.pulse_timeline[:, 1], torch.tensor([0.0, 9.0, 9.0]))
    assert torch.equal(event_rollout.pulse_timeline[:, 2], torch.zeros(3))
    assert torch.equal(event_rollout.pulse_timeline[:, 3], torch.full((3,), 14.0))

    # Channel 1 receives the two pulses; every other channel is untouched.
    blob = _gaussian_blob(config.grid_size, 9.0, CPU)
    expected_c1 = initial[0, 1] + 2.0 * config.perturbation_amplitude * blob
    assert torch.allclose(event_rollout.final_state[1], expected_c1, atol=1e-6)
    for channel in range(BASELINE_CONFIG.state_channels):
        if channel == 1:
            continue
        assert torch.equal(event_rollout.final_state[channel], initial[0, channel])


# --- _structural_motion ---


def test_structural_motion_rejects_global_brightness() -> None:
    brightness = torch.stack([torch.full((4, 4), v) for v in (0.1, 0.2, 0.3)])
    # Spatially uniform frames carry raw motion but zero structural motion.
    assert _structural_motion(brightness) == 0.0


def test_structural_motion_accepts_spatial_change() -> None:
    spatial = torch.zeros(3, 4, 4)
    spatial[0, 0, 0] = 1.0
    spatial[1, 1, 1] = 1.0
    spatial[2, 2, 2] = 1.0
    assert _structural_motion(spatial) > 0.0


def test_structural_motion_short_sequence_is_zero() -> None:
    assert _structural_motion(torch.zeros(1, 4, 4)) == 0.0


# --- _trajectory_divergence ---


def test_trajectory_divergence_shape_mismatch_is_infinite() -> None:
    a = torch.zeros(3, 4, 4)
    b = torch.zeros(3, 4, 5)
    div = _trajectory_divergence(a, b)
    assert math.isinf(div["raw"])
    assert math.isinf(div["structural"])


def test_trajectory_divergence_identical_is_zero() -> None:
    a = torch.randn(3, 4, 4)
    div = _trajectory_divergence(a, a)
    assert div["raw"] == 0.0
    assert div["structural"] == 0.0


def test_trajectory_divergence_pure_offset_is_structural_zero() -> None:
    a = torch.randn(3, 4, 4)
    b = a + 1.0
    div = _trajectory_divergence(a, b)
    assert div["raw"] == pytest.approx(1.0)
    assert div["structural"] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_divergence_spatial_difference_is_structural() -> None:
    a = torch.zeros(3, 4, 4)
    b = torch.zeros(3, 4, 4)
    b[0, 0, 0] = 1.0
    div = _trajectory_divergence(a, b)
    assert div["raw"] > 0.0
    assert div["structural"] > 0.0


# --- _response_lag units ---


def test_response_lag_units_and_peak_alignment() -> None:
    onset = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    onset_exp = onset
    shift = 2
    # Motion lags the onset by ``shift`` updates: motion[k] = onset_exp[k - shift].
    motion = torch.zeros(onset_exp.numel(), dtype=torch.float32)
    motion[shift:] = onset_exp[:-shift]
    updates_per_frame = 2
    fps = 24
    lag = _response_lag(motion, onset, updates_per_frame, fps)
    assert lag["lag_frames"] == shift + 1
    assert lag["lag_updates"] == (shift + 1) * updates_per_frame
    assert lag["lag_ms"] == pytest.approx(lag["lag_frames"] * 1000.0 / fps)
    assert lag["peak_correlation"] > 0.0


def test_response_lag_returns_zeros_for_flat_inputs() -> None:
    flat = torch.zeros(8, dtype=torch.float32)
    onset = torch.zeros(4, dtype=torch.float32)
    lag = _response_lag(flat, onset, 2, 24)
    assert lag == {"lag_updates": 0, "lag_frames": 0, "lag_ms": 0.0, "peak_correlation": 0.0}


# --- _automated_checks ---


def _passing_checks_kwargs() -> dict:
    return {
        "all_finite": True,
        "all_bounded": True,
        "exact_replay": True,
        "identical_blind_audio": True,
        "impulse_metrics": {
            "baseline": 0.1,
            "peak": 1.0,
            "peak_to_baseline_ratio": 10.0,
            "time_to_peak_frames": 5,
            "recovery_frames": 3,
        },
        "alternating_metrics": {
            "high_energy_motion": 1.0,
            "low_energy_motion": 0.5,
            "ratio": 1.5,
        },
        "spectral_metrics": {
            "low_tv": 1.0,
            "high_tv": 1.0,
            "low_centroid": 0.1,
            "high_centroid": 0.1,
            "tv_relative_difference": 0.02,
            "centroid_relative_difference": 0.0,
        },
        "review_onset": {
            "baseline": 0.1,
            "peak": 1.0,
            "time_to_peak_frames": 5,
            "recovery_frames": 2,
        },
        "lag": {
            "lag_updates": 4,
            "lag_frames": 2,
            "lag_ms": 83.33,
            "peak_correlation": 0.9,
        },
        "div_silent": {"raw": 0.5, "structural": 0.3},
        "div_shuffled": {"raw": 0.5, "structural": 0.3},
        "structure_ratio": 0.5,
    }


def test_automated_checks_all_pass() -> None:
    checks = _automated_checks(**_passing_checks_kwargs())
    assert all(checks.values())


@pytest.mark.parametrize(
    ("field", "value", "key"),
    [
        ("all_finite", False, "all_finite"),
        ("all_bounded", False, "all_states_below_2_0"),
        ("exact_replay", False, "exact_replay"),
        ("identical_blind_audio", False, "identical_blind_audio"),
        ("structure_ratio", 0.01, "response_not_brightness"),
    ],
)
def test_automated_checks_scalar_flips(field: str, value, key: str) -> None:  # noqa: ANN001
    kwargs = _passing_checks_kwargs()
    kwargs[field] = value
    checks = _automated_checks(**kwargs)
    assert checks[key] is False


@pytest.mark.parametrize(
    ("field", "value", "key"),
    [
        # impulse_prompt: peak not above baseline.
        (
            "impulse_metrics",
            {
                "baseline": 0.1,
                "peak": 0.05,
                "peak_to_baseline_ratio": 0.5,
                "time_to_peak_frames": 5,
                "recovery_frames": 3,
            },
            "impulse_prompt",
        ),
        # impulse_localized: time_to_peak beyond the window.
        (
            "impulse_metrics",
            {
                "baseline": 0.1,
                "peak": 1.0,
                "peak_to_baseline_ratio": 10.0,
                "time_to_peak_frames": 30,
                "recovery_frames": 3,
            },
            "impulse_localized",
        ),
        # alternating_energy: ratio below threshold.
        (
            "alternating_metrics",
            {"high_energy_motion": 1.0, "low_energy_motion": 0.5, "ratio": 1.0},
            "alternating_energy",
        ),
        # spectral_spatial: low centroid absent.
        (
            "spectral_metrics",
            {
                "low_tv": 1.0,
                "high_tv": 1.0,
                "tv_relative_difference": 0.0,
                "low_centroid": 0.0,
                "high_centroid": 0.1,
                "centroid_relative_difference": 1.0,
            },
            "spectral_spatial",
        ),
        # recovery: zero recovery frames.
        (
            "review_onset",
            {"baseline": 0.1, "peak": 1.0, "time_to_peak_frames": 5, "recovery_frames": 0},
            "recovery",
        ),
        # lag_acceptable: lag_frames beyond max.
        (
            "lag",
            {"lag_updates": 40, "lag_frames": 20.0, "lag_ms": 833.0, "peak_correlation": 0.9},
            "lag_acceptable",
        ),
        # correct_vs_silent_divergence: raw below threshold.
        ("div_silent", {"raw": 0.0, "structural": 0.0}, "correct_vs_silent_divergence"),
        # correct_vs_shuffled_divergence: raw below threshold.
        ("div_shuffled", {"raw": 0.0, "structural": 0.0}, "correct_vs_shuffled_divergence"),
    ],
)
def test_automated_checks_metric_flips(field: str, value: dict, key: str) -> None:
    kwargs = _passing_checks_kwargs()
    kwargs[field] = value
    checks = _automated_checks(**kwargs)
    assert checks[key] is False


def test_controlled_rollout_is_controlledrollout_dataclass() -> None:
    model, initial, masks, features, config = _replay_inputs()
    rollout = _controlled_rollout(
        model, initial, masks, features, config, REACTION_DIFFUSION_CONFIG, CPU
    )
    assert isinstance(rollout, ControlledRollout)
    assert rollout.first_bound_violation is None  # small state stays bounded
