"""Deterministic RMS-forced teacher foundation for Stage 3.

The Stage 1 reaction-diffusion teacher is reused as the dynamics kernel; this
module adds a small frozen RMS-forcing configuration and a tensor mapping from
absolute RMS values to recurrence speed factors, then drives the existing
masked teacher step at an effective time step of
``config.time_step * speed_factor``. No training, checkpoints, rendering, or
audio probes live here; the future Stage 3 runner is expected to pass
interpolated RMS into :func:`rms_to_speed` and feed the result into
:func:`generate_masked_forced_teacher_trajectory`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import Tensor

from synesthete.teacher import (
    ReactionDiffusionConfig,
    reaction_diffusion_step,
)


@dataclass(frozen=True)
class RMSForcingConfig:
    """Frozen RMS-forcing parameters for the Stage 3 teacher.

    These fields govern only the RMS-to-speed mapping; the underlying
    reaction-diffusion dynamics remain owned by
    :class:`synesthete.teacher.ReactionDiffusionConfig`.
    """

    speed_min: float = 0.35
    speed_max: float = 1.5
    rms_reference: float = 0.2

    def __post_init__(self) -> None:
        for name in ("speed_min", "speed_max", "rms_reference"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.speed_min <= 0.0:
            raise ValueError("speed_min must be positive")
        if self.speed_max <= self.speed_min:
            raise ValueError("speed_max must exceed speed_min")
        if self.rms_reference <= 0.0:
            raise ValueError("rms_reference must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rms_to_speed(rms: Tensor, config: RMSForcingConfig) -> Tensor:
    """Map 1D absolute RMS fp32 values to speed factors in [speed_min, speed_max].

    The mapping matches the existing Stage 2 scalar mapping: clamp
    ``rms / rms_reference`` to ``[0, 1]`` and linearly interpolate from
    ``speed_min`` (at ratio 0) to ``speed_max`` (at ratio 1). No per-clip
    normalization or fallback is applied.
    """
    if rms.ndim != 1:
        raise ValueError("RMS tensor must be one-dimensional")
    if rms.dtype != torch.float32:
        raise ValueError("RMS tensor must be fp32")
    if not torch.isfinite(rms).all():
        raise ValueError("RMS tensor must be finite")
    if (rms < 0.0).any():
        raise ValueError("RMS tensor must be nonnegative")
    ratio = (rms / config.rms_reference).clamp(0.0, 1.0)
    return config.speed_min + ratio * (config.speed_max - config.speed_min)


def _validate_speed_factor(speed_factor: Tensor, fields: Tensor, config: RMSForcingConfig) -> None:
    if not isinstance(speed_factor, Tensor):
        raise ValueError("speed_factor must be a tensor")
    if speed_factor.ndim != 0:
        raise ValueError("speed_factor must be a scalar (0-dim) tensor")
    if speed_factor.dtype != torch.float32:
        raise ValueError("speed_factor must be fp32")
    if speed_factor.device != fields.device:
        raise ValueError("speed_factor must be on the same device as the fields")
    if not torch.isfinite(speed_factor).all():
        raise ValueError("speed_factor must be finite")
    # Compare against the fp32 representation of the bounds so that outputs of
    # rms_to_speed (which round-trip through fp32) validate at the endpoints.
    lo = torch.tensor(float(config.speed_min), dtype=torch.float32, device=speed_factor.device)
    hi = torch.tensor(float(config.speed_max), dtype=torch.float32, device=speed_factor.device)
    if speed_factor < lo or speed_factor > hi:
        raise ValueError(
            f"speed_factor {float(speed_factor.item())} is outside the configured range "
            f"[{config.speed_min}, {config.speed_max}]"
        )


def masked_forced_teacher_step(
    fields: Tensor,
    fire_mask: Tensor,
    speed_factor: Tensor,
    config: ReactionDiffusionConfig,
    forcing_config: RMSForcingConfig,
) -> Tensor:
    """One masked RMS-forced reaction-diffusion step.

    The effective time step is ``config.time_step * speed_factor``; the
    reaction-diffusion equations themselves are reused verbatim from
    :func:`synesthete.teacher.reaction_diffusion_step` via
    :func:`dataclasses.replace`. The fire mask is applied to the full residual
    exactly as in :func:`synesthete.teacher.masked_reaction_diffusion_step`.
    """
    if fields.ndim != 4 or fields.shape[1] != 2:
        raise ValueError("Reaction-diffusion state must have shape (batch, 2, height, width)")
    if fields.dtype != torch.float32:
        raise ValueError("Stage 3 teacher requires fp32 fields")
    expected_shape = (fields.shape[0], 1, fields.shape[2], fields.shape[3])
    if fire_mask.shape != expected_shape:
        raise ValueError(
            f"Expected teacher fire mask shape {expected_shape}, received {tuple(fire_mask.shape)}"
        )
    if fire_mask.device != fields.device:
        raise ValueError("Teacher fields and fire mask must be on the same device")
    if fire_mask.dtype != fields.dtype:
        raise ValueError("Teacher fields and fire mask must have the same dtype")
    _validate_speed_factor(speed_factor, fields, forcing_config)

    effective_time_step = config.time_step * float(speed_factor.item())
    forced_config = replace(config, time_step=effective_time_step)
    next_fields = reaction_diffusion_step(fields, forced_config)
    out = fields + fire_mask * (next_fields - fields)
    if not torch.isfinite(out).all():
        raise RuntimeError("Forced teacher step produced non-finite state")
    return out


def generate_masked_forced_teacher_trajectory(
    initial_fields: Tensor,
    *,
    fire_masks: Tensor,
    speed_factors: Tensor,
    config: ReactionDiffusionConfig,
    forcing_config: RMSForcingConfig,
) -> Tensor:
    """Generate a masked RMS-forced teacher trajectory.

    ``speed_factors`` is a 1D fp32 tensor of length ``fire_masks.shape[0]``;
    entry ``i`` drives the step that consumes ``fire_masks[i]``.
    """
    if initial_fields.ndim != 4 or initial_fields.shape[1] != 2:
        raise ValueError("Initial teacher fields must have shape (batch, 2, height, width)")
    if initial_fields.dtype != torch.float32:
        raise ValueError("Stage 3 teacher requires fp32 fields")
    if fire_masks.ndim != 5 or fire_masks.shape[0] <= 0:
        raise ValueError("Teacher fire masks must have shape (step, batch, 1, height, width)")
    if fire_masks.shape[1:] != (
        initial_fields.shape[0],
        1,
        initial_fields.shape[2],
        initial_fields.shape[3],
    ):
        raise ValueError("Teacher fire masks do not match the initial fields")
    if fire_masks.device != initial_fields.device or fire_masks.dtype != initial_fields.dtype:
        raise ValueError("Teacher fire masks must match the initial fields device and dtype")
    if speed_factors.ndim != 1:
        raise ValueError("speed_factors must be one-dimensional")
    if speed_factors.shape[0] != fire_masks.shape[0]:
        raise ValueError(
            f"speed_factors length {speed_factors.shape[0]} does not match "
            f"fire_masks step count {fire_masks.shape[0]}"
        )
    if speed_factors.dtype != torch.float32:
        raise ValueError("speed_factors must be fp32")
    if speed_factors.device != initial_fields.device:
        raise ValueError("speed_factors must be on the same device as the initial fields")
    if not torch.isfinite(speed_factors).all():
        raise ValueError("speed_factors must be finite")
    # Compare against the fp32 representation of the bounds to match the
    # scalar path in _validate_speed_factor.
    lo = torch.tensor(
        float(forcing_config.speed_min), dtype=torch.float32, device=speed_factors.device
    )
    hi = torch.tensor(
        float(forcing_config.speed_max), dtype=torch.float32, device=speed_factors.device
    )
    if (speed_factors < lo).any() or (speed_factors > hi).any():
        raise ValueError(
            f"speed_factors contain entries outside the configured range "
            f"[{forcing_config.speed_min}, {forcing_config.speed_max}]"
        )

    trajectory = torch.empty(
        (fire_masks.shape[0] + 1, *initial_fields.shape),
        device=initial_fields.device,
        dtype=torch.float32,
    )
    fields = initial_fields
    trajectory[0].copy_(fields)
    with torch.no_grad():
        for step in range(fire_masks.shape[0]):
            fields = masked_forced_teacher_step(
                fields,
                fire_masks[step],
                speed_factors[step],
                config,
                forcing_config,
            )
            if not torch.isfinite(fields).all():
                raise RuntimeError(f"Forced teacher became non-finite at step {step + 1}")
            trajectory[step + 1].copy_(fields)
    return trajectory


__all__ = [
    "RMSForcingConfig",
    "generate_masked_forced_teacher_trajectory",
    "masked_forced_teacher_step",
    "rms_to_speed",
]
