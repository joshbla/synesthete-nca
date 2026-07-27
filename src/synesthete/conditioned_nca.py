"""Scalar-conditioned Neural Cellular Automaton built on the Stage 0 rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from synesthete.nca import NCAConfig, NeuralCellularAutomaton


@dataclass(frozen=True)
class ConditionedNCAConfig(NCAConfig):
    conditioning_dim: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")


class ConditionedNeuralCellularAutomaton(NeuralCellularAutomaton):
    """Stage 0 NCA with a scalar conditioning path modulating hidden activations."""

    def __init__(self, config: ConditionedNCAConfig) -> None:
        super().__init__(config)
        self.conditioning_projection = nn.Linear(
            config.conditioning_dim,
            2 * config.hidden_channels,
        )
        nn.init.zeros_(self.conditioning_projection.weight)
        nn.init.zeros_(self.conditioning_projection.bias)

    def forward(self, state: Tensor, fire_mask: Tensor, conditioning: Tensor) -> Tensor:
        self._validate_state(state)
        expected_mask_shape = (state.shape[0], 1, state.shape[2], state.shape[3])
        if fire_mask.shape != expected_mask_shape:
            raise ValueError(
                f"Expected fire mask shape {expected_mask_shape}, received {tuple(fire_mask.shape)}"
            )
        if fire_mask.device != state.device:
            raise ValueError("State and fire mask must be on the same device")
        if fire_mask.dtype != state.dtype:
            raise ValueError("State and fire mask must have the same dtype")
        self._validate_conditioning(conditioning, state)

        hidden = torch.relu(self.update_hidden(self.perceive(state)))
        projection = self.conditioning_projection(conditioning)
        gamma, beta = projection.split(self.config.hidden_channels, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        hidden = hidden * (1.0 + gamma) + beta
        delta = torch.tanh(self.update_output(hidden))
        return state + self.config.step_size * fire_mask * delta

    def transfer_unconditioned_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            "conditioning_projection.weight",
            "conditioning_projection.bias",
        }
        if set(missing) != expected_missing:
            raise ValueError(
                f"Expected only {sorted(expected_missing)} to be missing, "
                f"received missing={sorted(missing)}"
            )
        if unexpected:
            raise ValueError(f"Received unexpected keys: {sorted(unexpected)}")

    def _validate_conditioning(self, conditioning: Tensor, state: Tensor) -> None:
        if conditioning.ndim != 2:
            raise ValueError("conditioning must have batch and conditioning dimensions")
        if conditioning.shape != (state.shape[0], self.config.conditioning_dim):
            raise ValueError(
                f"Expected conditioning shape "
                f"{(state.shape[0], self.config.conditioning_dim)}, "
                f"received {tuple(conditioning.shape)}"
            )
        if conditioning.dtype != torch.float32:
            raise ValueError("conditioning must be fp32")
        if conditioning.device != state.device:
            raise ValueError("State and conditioning must be on the same device")
        if not torch.isfinite(conditioning).all():
            raise ValueError("conditioning must be finite")
