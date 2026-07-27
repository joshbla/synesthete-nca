"""Focused CPU unit tests for the Stage 3 RMS-forced teacher foundation.

These tests cover the frozen config, the RMS-to-speed tensor mapping, exact
neutral (speed_factor == 1.0) equivalence with the Stage 1 masked teacher
under recorded masks, non-neutral trajectory divergence and finiteness, mask
behavior, speed-sequence replay, and shape/dtype/nonfinite validation. No
training, checkpoint, rendering, or audio integration is exercised.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from synesthete.stage3_teacher import (
    RMSForcingConfig,
    generate_masked_forced_teacher_trajectory,
    masked_forced_teacher_step,
    rms_to_speed,
)
from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    generate_masked_teacher_trajectory,
    masked_reaction_diffusion_step,
    reaction_diffusion_step,
)

CPU = torch.device("cpu")
GRID = 16
BATCH = 1


def _fields(seed: int = 73) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, 2, GRID, GRID, dtype=torch.float32, generator=gen).mul_(0.1)


def _masks(steps: int, seed: int = 99) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(steps, BATCH, 1, GRID, GRID, generator=gen) < 0.5).to(torch.float32)


# --- RMSForcingConfig ---


def test_rms_forcing_config_defaults() -> None:
    config = RMSForcingConfig()
    assert config.speed_min == 0.35
    assert config.speed_max == 1.5
    assert config.rms_reference == 0.2
    assert set(config.to_dict()) == {"speed_min", "speed_max", "rms_reference"}


def test_rms_forcing_config_order_and_positivity() -> None:
    with pytest.raises(ValueError, match="speed_min"):
        RMSForcingConfig(speed_min=0.0)
    with pytest.raises(ValueError, match="speed_max"):
        RMSForcingConfig(speed_min=1.5, speed_max=1.5)
    with pytest.raises(ValueError, match="speed_max"):
        RMSForcingConfig(speed_min=2.0, speed_max=1.0)
    with pytest.raises(ValueError, match="rms_reference"):
        RMSForcingConfig(rms_reference=0.0)


def test_rms_forcing_config_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="speed_min"):
        RMSForcingConfig(speed_min=math.inf)
    with pytest.raises(ValueError, match="speed_max"):
        RMSForcingConfig(speed_max=math.nan)


# --- rms_to_speed endpoints and validation ---


def test_rms_to_speed_endpoints_and_clamp() -> None:
    config = RMSForcingConfig()
    rms = torch.tensor(
        [
            0.0,
            config.rms_reference / 2.0,
            config.rms_reference,
            10.0 * config.rms_reference,
        ],
        dtype=torch.float32,
    )
    speeds = rms_to_speed(rms, config)
    lo, hi = config.speed_min, config.speed_max
    assert speeds[0].item() == pytest.approx(lo)
    assert speeds[1].item() == pytest.approx((lo + hi) / 2.0)
    assert speeds[2].item() == pytest.approx(hi)
    assert speeds[3].item() == pytest.approx(hi)
    assert speeds.dtype == torch.float32


def test_rms_to_speed_matches_stage2_scalar_mapping() -> None:
    from synesthete.stage2 import Stage2Config, _speed_scale

    stage2 = Stage2Config()
    forcing = RMSForcingConfig()
    rms = torch.linspace(0.0, 0.4, 9, dtype=torch.float32)
    tensor_speeds = rms_to_speed(rms, forcing)
    for i in range(rms.numel()):
        assert tensor_speeds[i].item() == pytest.approx(_speed_scale(float(rms[i]), stage2))


def test_rms_to_speed_validation() -> None:
    config = RMSForcingConfig()
    with pytest.raises(ValueError, match="one-dimensional"):
        rms_to_speed(torch.zeros(2, 3, dtype=torch.float32), config)
    with pytest.raises(ValueError, match="fp32"):
        rms_to_speed(torch.zeros(3, dtype=torch.float64), config)
    with pytest.raises(ValueError, match="finite"):
        rms_to_speed(torch.tensor([0.0, math.nan], dtype=torch.float32), config)
    with pytest.raises(ValueError, match="nonnegative"):
        rms_to_speed(torch.tensor([-1.0], dtype=torch.float32), config)


# --- Exact neutral equivalence with Stage 1 masked teacher ---


def test_neutral_step_bit_for_bit_equals_stage1() -> None:
    fields = _fields()
    mask = _masks(1)[0]
    forcing = RMSForcingConfig()
    speed = torch.tensor(1.0, dtype=torch.float32, device=CPU)
    forced = masked_forced_teacher_step(fields, mask, speed, REACTION_DIFFUSION_CONFIG, forcing)
    stage1 = masked_reaction_diffusion_step(fields, mask, REACTION_DIFFUSION_CONFIG)
    assert torch.equal(forced, stage1)


def test_neutral_trajectory_bit_for_bit_equals_stage1() -> None:
    fields = _fields()
    masks = _masks(6)
    forcing = RMSForcingConfig()
    speeds = torch.full((masks.shape[0],), 1.0, dtype=torch.float32)
    forced = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=speeds,
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    stage1 = generate_masked_teacher_trajectory(
        fields, fire_masks=masks, config=REACTION_DIFFUSION_CONFIG
    )
    assert torch.equal(forced, stage1)


# --- Non-neutral trajectories differ, stay finite, and respect masks ---


def test_nonneutral_trajectory_differs_and_stays_finite() -> None:
    fields = _fields()
    masks = _masks(6)
    forcing = RMSForcingConfig()
    speeds_hi = torch.full((masks.shape[0],), forcing.speed_max, dtype=torch.float32)
    speeds_lo = torch.full((masks.shape[0],), forcing.speed_min, dtype=torch.float32)
    fast = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=speeds_hi,
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    slow = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=speeds_lo,
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    assert torch.isfinite(fast).all()
    assert torch.isfinite(slow).all()
    assert not torch.equal(fast, slow)
    # Both share the same initial state.
    assert torch.equal(fast[0], slow[0])
    # Different speeds produce different first-step outputs.
    assert not torch.equal(fast[1], slow[1])


def test_mask_blocks_residual_at_zero_mask() -> None:
    fields = _fields()
    zero_mask = torch.zeros(BATCH, 1, GRID, GRID, dtype=torch.float32)
    forcing = RMSForcingConfig()
    for speed_value in (forcing.speed_min, 1.0, forcing.speed_max):
        speed = torch.tensor(speed_value, dtype=torch.float32)
        out = masked_forced_teacher_step(
            fields, zero_mask, speed, REACTION_DIFFUSION_CONFIG, forcing
        )
        assert torch.equal(out, fields)


def test_mask_passes_residual_at_unit_mask() -> None:
    fields = _fields()
    unit_mask = torch.ones(BATCH, 1, GRID, GRID, dtype=torch.float32)
    forcing = RMSForcingConfig()
    speed = torch.tensor(forcing.speed_max, dtype=torch.float32)
    out = masked_forced_teacher_step(fields, unit_mask, speed, REACTION_DIFFUSION_CONFIG, forcing)
    forced_config = replace(
        REACTION_DIFFUSION_CONFIG,
        time_step=REACTION_DIFFUSION_CONFIG.time_step * forcing.speed_max,
    )
    expected = fields + unit_mask * (reaction_diffusion_step(fields, forced_config) - fields)
    assert torch.equal(out, expected)


# --- Speed-sequence exact replay ---


def test_trajectory_is_deterministic_exact_replay() -> None:
    fields = _fields()
    masks = _masks(5)
    forcing = RMSForcingConfig()
    rms = torch.tensor([0.0, 0.1, 0.2, 0.05, 0.3], dtype=torch.float32)
    speeds = rms_to_speed(rms, forcing)
    a = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=speeds,
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    b = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=speeds,
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    assert torch.equal(a, b)
    # Mixed speed sequence differs from a constant-1.0 trajectory.
    neutral = generate_masked_forced_teacher_trajectory(
        fields,
        fire_masks=masks,
        speed_factors=torch.full_like(speeds, 1.0),
        config=REACTION_DIFFUSION_CONFIG,
        forcing_config=forcing,
    )
    assert not torch.equal(a, neutral)


# --- Step-level validation ---


def test_step_rejects_bad_speed_factor() -> None:
    fields = _fields()
    mask = _masks(1)[0]
    forcing = RMSForcingConfig()
    with pytest.raises(ValueError, match="scalar"):
        masked_forced_teacher_step(
            fields,
            mask,
            torch.tensor([1.0], dtype=torch.float32),
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )
    with pytest.raises(ValueError, match="fp32"):
        masked_forced_teacher_step(
            fields,
            mask,
            torch.tensor(1.0, dtype=torch.float64),
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )
    with pytest.raises(ValueError, match="finite"):
        masked_forced_teacher_step(
            fields,
            mask,
            torch.tensor(math.nan, dtype=torch.float32),
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )
    with pytest.raises(ValueError, match="outside the configured range"):
        masked_forced_teacher_step(
            fields,
            mask,
            torch.tensor(forcing.speed_max + 0.01, dtype=torch.float32),
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )
    with pytest.raises(ValueError, match="outside the configured range"):
        masked_forced_teacher_step(
            fields,
            mask,
            torch.tensor(forcing.speed_min - 0.01, dtype=torch.float32),
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )


def test_step_rejects_mask_shape_and_dtype_mismatch() -> None:
    fields = _fields()
    forcing = RMSForcingConfig()
    speed = torch.tensor(1.0, dtype=torch.float32)
    bad_shape = torch.zeros(BATCH, 1, GRID, GRID + 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="fire mask shape"):
        masked_forced_teacher_step(fields, bad_shape, speed, REACTION_DIFFUSION_CONFIG, forcing)
    bad_dtype = torch.zeros(BATCH, 1, GRID, GRID, dtype=torch.float64)
    with pytest.raises(ValueError, match="same dtype"):
        masked_forced_teacher_step(fields, bad_dtype, speed, REACTION_DIFFUSION_CONFIG, forcing)


def test_step_rejects_bad_fields() -> None:
    forcing = RMSForcingConfig()
    speed = torch.tensor(1.0, dtype=torch.float32)
    mask = _masks(1)[0]
    with pytest.raises(ValueError, match="Reaction-diffusion state"):
        masked_forced_teacher_step(
            torch.zeros(BATCH, 3, GRID, GRID),
            mask,
            speed,
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )
    with pytest.raises(ValueError, match="fp32"):
        masked_forced_teacher_step(
            _fields().to(torch.float64),
            mask,
            speed,
            REACTION_DIFFUSION_CONFIG,
            forcing,
        )


# --- Trajectory-level validation ---


def test_trajectory_rejects_length_and_dtype_mismatches() -> None:
    fields = _fields()
    masks = _masks(4)
    forcing = RMSForcingConfig()
    with pytest.raises(ValueError, match="speed_factors length"):
        generate_masked_forced_teacher_trajectory(
            fields,
            fire_masks=masks,
            speed_factors=torch.full((3,), 1.0, dtype=torch.float32),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )
    with pytest.raises(ValueError, match="fp32"):
        generate_masked_forced_teacher_trajectory(
            fields,
            fire_masks=masks,
            speed_factors=torch.full((masks.shape[0],), 1.0, dtype=torch.float64),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )
    with pytest.raises(ValueError, match="outside the configured range"):
        generate_masked_forced_teacher_trajectory(
            fields,
            fire_masks=masks,
            speed_factors=torch.full(
                (masks.shape[0],), forcing.speed_max + 0.1, dtype=torch.float32
            ),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )
    with pytest.raises(ValueError, match="finite"):
        generate_masked_forced_teacher_trajectory(
            fields,
            fire_masks=masks,
            speed_factors=torch.tensor([1.0, 1.0, math.nan, 1.0], dtype=torch.float32),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )


def test_trajectory_rejects_bad_initial_fields_and_masks() -> None:
    forcing = RMSForcingConfig()
    bad_fields = torch.randn(BATCH, 3, GRID, GRID, dtype=torch.float32).mul_(0.1)
    with pytest.raises(ValueError, match="Initial teacher fields"):
        generate_masked_forced_teacher_trajectory(
            bad_fields,
            fire_masks=_masks(2),
            speed_factors=torch.full((2,), 1.0, dtype=torch.float32),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )
    fields = _fields()
    with pytest.raises(ValueError, match="fire masks"):
        generate_masked_forced_teacher_trajectory(
            fields,
            fire_masks=torch.zeros(2, BATCH, 1, GRID, GRID + 1, dtype=torch.float32),
            speed_factors=torch.full((2,), 1.0, dtype=torch.float32),
            config=REACTION_DIFFUSION_CONFIG,
            forcing_config=forcing,
        )
