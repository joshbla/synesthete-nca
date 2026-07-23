"""The unconditioned Neural Cellular Automaton used by Stage 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

PERCEPTION_VERSION = "identity-sobel-x-sobel-y-laplacian-v1"


@dataclass(frozen=True)
class NCAConfig:
    state_channels: int
    hidden_channels: int
    fire_rate: float
    step_size: float
    visible_channel: int
    boundary: str
    perception: str

    def __post_init__(self) -> None:
        if self.state_channels <= 0:
            raise ValueError("state_channels must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 <= self.fire_rate <= 1.0:
            raise ValueError("fire_rate must be between zero and one")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not 0 <= self.visible_channel < self.state_channels:
            raise ValueError("visible_channel must identify a state channel")
        if self.boundary != "circular":
            raise ValueError("Stage 0 requires circular boundaries")
        if self.perception != PERCEPTION_VERSION:
            raise ValueError(f"Unsupported perception definition: {self.perception}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINE_CONFIG = NCAConfig(
    state_channels=16,
    hidden_channels=96,
    fire_rate=0.5,
    step_size=0.1,
    visible_channel=0,
    boundary="circular",
    perception=PERCEPTION_VERSION,
)


def _perception_kernels(state_channels: int) -> Tensor:
    kernels = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)),
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)),
        ),
        dtype=torch.float32,
    ).unsqueeze(1)
    kernels[1:3].div_(8.0)
    return kernels.repeat(state_channels, 1, 1, 1)


class NeuralCellularAutomaton(nn.Module):
    """A shared local rule with fixed perception and stochastic residual updates."""

    def __init__(self, config: NCAConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "perception_kernels",
            _perception_kernels(config.state_channels),
        )
        self.update_hidden = nn.Conv2d(
            config.state_channels * 4,
            config.hidden_channels,
            kernel_size=1,
        )
        self.update_output = nn.Conv2d(
            config.hidden_channels,
            config.state_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.update_output.weight)
        nn.init.zeros_(self.update_output.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def perceive(self, state: Tensor) -> Tensor:
        self._validate_state(state)
        padded = functional.pad(state, (1, 1, 1, 1), mode="circular")
        responses = functional.conv2d(
            padded,
            self.perception_kernels,
            groups=self.config.state_channels,
        )
        batch, _, height, width = responses.shape
        return (
            responses.view(batch, self.config.state_channels, 4, height, width)
            .transpose(1, 2)
            .reshape(batch, self.config.state_channels * 4, height, width)
        )

    def sample_fire_mask(self, state: Tensor, generator: torch.Generator) -> Tensor:
        self._validate_state(state)
        shape = (state.shape[0], 1, state.shape[2], state.shape[3])
        return (
            torch.rand(shape, device=state.device, generator=generator) < self.config.fire_rate
        ).to(dtype=state.dtype)

    def forward(self, state: Tensor, fire_mask: Tensor) -> Tensor:
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

        hidden = functional.relu(self.update_hidden(self.perceive(state)))
        delta = torch.tanh(self.update_output(hidden))
        return state + self.config.step_size * fire_mask * delta

    def _validate_state(self, state: Tensor) -> None:
        if state.ndim != 4:
            raise ValueError("NCA state must have batch, channel, height, and width dimensions")
        if state.shape[1] != self.config.state_channels:
            raise ValueError(
                f"Expected {self.config.state_channels} state channels, received {state.shape[1]}"
            )
        if state.dtype != torch.float32:
            raise ValueError("Stage 0 requires fp32 state")
        if state.device != self.perception_kernels.device:
            raise ValueError("State and model must be on the same device")


def make_field_state(
    *,
    batch_size: int,
    grid_size: int,
    config: NCAConfig,
    device: torch.device,
    generator: torch.Generator,
    amplitude: float,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")

    state = torch.randn(
        (batch_size, config.state_channels, grid_size, grid_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    state.mul_(amplitude)
    state[:, config.visible_channel].zero_()
    return state
