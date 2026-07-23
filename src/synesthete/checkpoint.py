"""Strict Stage 0 checkpoint save and load behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from synesthete.nca import NCAConfig, NeuralCellularAutomaton

CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_KEYS = {
    "format_version",
    "model_config",
    "parameter_count",
    "model_seed",
    "state_seed",
    "mask_seed",
    "grid_size",
    "batch_size",
    "source_revision",
    "state_dict",
}


def save_checkpoint(
    path: Path,
    model: NeuralCellularAutomaton,
    *,
    model_seed: int,
    state_seed: int,
    mask_seed: int,
    grid_size: int,
    batch_size: int,
    source_revision: str,
) -> None:
    if not source_revision:
        raise ValueError("source_revision must be recorded")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count,
        "model_seed": model_seed,
        "state_seed": state_seed,
        "mask_seed": mask_seed,
        "grid_size": grid_size,
        "batch_size": batch_size,
        "source_revision": source_revision,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    expected_config: NCAConfig,
    expected_grid_size: int,
    expected_batch_size: int,
    device: torch.device,
) -> tuple[NeuralCellularAutomaton, dict[str, int | str]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    if set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("Checkpoint fields do not match the Stage 0 contract")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint format version is incompatible")
    if payload["model_config"] != expected_config.to_dict():
        raise ValueError("Checkpoint model configuration is incompatible")

    model = NeuralCellularAutomaton(expected_config).to(device)
    if payload["parameter_count"] != model.parameter_count:
        raise ValueError("Checkpoint parameter count is incompatible")
    if payload["grid_size"] != expected_grid_size:
        raise ValueError("Checkpoint grid size is incompatible")
    if payload["batch_size"] != expected_batch_size:
        raise ValueError("Checkpoint batch size is incompatible")
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {
        "model_seed": _require_int(payload, "model_seed"),
        "state_seed": _require_int(payload, "state_seed"),
        "mask_seed": _require_int(payload, "mask_seed"),
        "grid_size": _require_int(payload, "grid_size"),
        "batch_size": _require_int(payload, "batch_size"),
        "source_revision": _require_str(payload, "source_revision"),
    }
    return model, metadata


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"Checkpoint field {key} must be an integer")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Checkpoint field {key} must be a nonempty string")
    return value
