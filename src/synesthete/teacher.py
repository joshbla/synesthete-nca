"""Deterministic oscillatory reaction-diffusion teacher for Stage 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor

Initialization = Literal["central", "distributed"]


@dataclass(frozen=True)
class ReactionDiffusionConfig:
    diffusion: float
    growth: float
    angular_frequency: float
    saturation: float
    time_step: float
    render_offset: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REACTION_DIFFUSION_CONFIG = ReactionDiffusionConfig(
    diffusion=0.15,
    growth=0.2,
    angular_frequency=0.25,
    saturation=1.0,
    time_step=0.1,
    render_offset=0.5,
)


def periodic_laplacian(field: Tensor) -> Tensor:
    if field.ndim != 4:
        raise ValueError("Teacher fields must have batch, channel, height, and width dimensions")
    return (
        torch.roll(field, 1, dims=2)
        + torch.roll(field, -1, dims=2)
        + torch.roll(field, 1, dims=3)
        + torch.roll(field, -1, dims=3)
        - 4.0 * field
    )


def reaction_diffusion_step(fields: Tensor, config: ReactionDiffusionConfig) -> Tensor:
    if fields.ndim != 4 or fields.shape[1] != 2:
        raise ValueError("Reaction-diffusion state must have shape (batch, 2, height, width)")
    if fields.dtype != torch.float32:
        raise ValueError("Stage 1 teacher requires fp32 fields")

    a = fields[:, 0:1]
    b = fields[:, 1:2]
    radius_squared = a.square() + b.square()
    restoring_scale = config.growth - config.saturation * radius_squared
    da = (
        config.diffusion * periodic_laplacian(a)
        + restoring_scale * a
        - config.angular_frequency * b
    )
    db = (
        config.diffusion * periodic_laplacian(b)
        + config.angular_frequency * a
        + restoring_scale * b
    )
    return torch.cat((a + config.time_step * da, b + config.time_step * db), dim=1)


def masked_reaction_diffusion_step(
    fields: Tensor,
    fire_mask: Tensor,
    config: ReactionDiffusionConfig,
) -> Tensor:
    expected_shape = (fields.shape[0], 1, fields.shape[2], fields.shape[3])
    if fire_mask.shape != expected_shape:
        raise ValueError(
            f"Expected teacher fire mask shape {expected_shape}, received {tuple(fire_mask.shape)}"
        )
    if fire_mask.device != fields.device:
        raise ValueError("Teacher fields and fire mask must be on the same device")
    if fire_mask.dtype != fields.dtype:
        raise ValueError("Teacher fields and fire mask must have the same dtype")
    next_fields = reaction_diffusion_step(fields, config)
    return fields + fire_mask * (next_fields - fields)


def make_teacher_fields(
    *,
    initialization: Initialization,
    batch_size: int,
    grid_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grid_size < 16:
        raise ValueError("grid_size must be at least 16")

    shape = (batch_size, 1, grid_size, grid_size)
    a = torch.zeros(shape, device=device, dtype=torch.float32)
    b = torch.zeros_like(a)
    if initialization == "central":
        half_width = max(2, grid_size // 16)
        center = grid_size // 2
        region = (
            ...,
            slice(center - half_width, center + half_width),
            slice(center - half_width, center + half_width),
        )
        a[region] = 0.35
        a.add_(
            torch.rand(shape, device=device, generator=generator, dtype=torch.float32).mul_(0.001)
        )
    elif initialization == "distributed":
        a.normal_(mean=0.0, std=0.03, generator=generator)
        b.normal_(mean=0.0, std=0.03, generator=generator)
    else:
        raise ValueError(f"Unsupported teacher initialization: {initialization}")
    return torch.cat((a, b), dim=1)


def generate_teacher_trajectory(
    initial_fields: Tensor,
    *,
    steps: int,
    config: ReactionDiffusionConfig,
) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    trajectory = torch.empty(
        (steps + 1, *initial_fields.shape),
        device=initial_fields.device,
        dtype=torch.float32,
    )
    fields = initial_fields
    trajectory[0].copy_(fields)
    with torch.no_grad():
        for step in range(1, steps + 1):
            fields = reaction_diffusion_step(fields, config)
            if not torch.isfinite(fields).all():
                raise RuntimeError(f"Teacher became non-finite at step {step}")
            trajectory[step].copy_(fields)
    return trajectory


def generate_masked_teacher_trajectory(
    initial_fields: Tensor,
    *,
    fire_masks: Tensor,
    config: ReactionDiffusionConfig,
) -> Tensor:
    if initial_fields.ndim != 4 or initial_fields.shape[1] != 2:
        raise ValueError("Initial teacher fields must have shape (batch, 2, height, width)")
    if initial_fields.dtype != torch.float32:
        raise ValueError("Stage 1 teacher requires fp32 fields")
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
    trajectory = torch.empty(
        (fire_masks.shape[0] + 1, *initial_fields.shape),
        device=initial_fields.device,
        dtype=torch.float32,
    )
    fields = initial_fields
    trajectory[0].copy_(fields)
    with torch.no_grad():
        for step, fire_mask in enumerate(fire_masks, start=1):
            fields = masked_reaction_diffusion_step(fields, fire_mask, config)
            if not torch.isfinite(fields).all():
                raise RuntimeError(f"Masked teacher became non-finite at step {step}")
            trajectory[step].copy_(fields)
    return trajectory


def encode_teacher_fields(
    fields: Tensor,
    *,
    state_channels: int,
    config: ReactionDiffusionConfig,
) -> Tensor:
    if fields.shape[-3] != 2:
        raise ValueError("Teacher fields must contain two reaction channels")
    if state_channels < 2:
        raise ValueError("NCA state must have at least two channels")
    state = torch.zeros(
        (*fields.shape[:-3], state_channels, fields.shape[-2], fields.shape[-1]),
        device=fields.device,
        dtype=torch.float32,
    )
    state[..., :2, :, :] = fields
    return state


def render_luminance(state: Tensor, config: ReactionDiffusionConfig) -> Tensor:
    if state.shape[-3] < 1:
        raise ValueError("NCA state must contain a visible channel")
    return (state[..., 0, :, :] + config.render_offset).clamp(0.0, 1.0)
