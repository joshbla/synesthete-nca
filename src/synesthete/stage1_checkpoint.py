"""Strict model and optimizer checkpoint contract for Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from synesthete.nca import NCAConfig, NeuralCellularAutomaton
from synesthete.teacher import ReactionDiffusionConfig

STAGE1_CHECKPOINT_VERSION = 1
_STAGE1_KEYS = {
    "format_version",
    "stage",
    "model_config",
    "teacher_config",
    "training_config",
    "seeds",
    "source_revision",
    "optimizer_steps_completed",
    "model_state_dict",
    "optimizer_state_dict",
}


def save_stage1_checkpoint(
    path: Path,
    *,
    model: NeuralCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    teacher_config: ReactionDiffusionConfig,
    training_config: dict[str, Any],
    seeds: dict[str, int],
    source_revision: str,
    optimizer_steps_completed: int,
) -> None:
    if not source_revision:
        raise ValueError("source_revision must be recorded")
    if optimizer_steps_completed <= 0:
        raise ValueError("optimizer_steps_completed must be positive")
    payload = {
        "format_version": STAGE1_CHECKPOINT_VERSION,
        "stage": 1,
        "model_config": model.config.to_dict(),
        "teacher_config": teacher_config.to_dict(),
        "training_config": training_config,
        "seeds": seeds,
        "source_revision": source_revision,
        "optimizer_steps_completed": optimizer_steps_completed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_stage1_checkpoint(
    path: Path,
    *,
    expected_model_config: NCAConfig,
    expected_teacher_config: ReactionDiffusionConfig,
    expected_training_config: dict[str, Any],
    device: torch.device,
) -> tuple[NeuralCellularAutomaton, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _STAGE1_KEYS:
        raise ValueError("Checkpoint fields do not match the Stage 1 contract")
    if payload["format_version"] != STAGE1_CHECKPOINT_VERSION or payload["stage"] != 1:
        raise ValueError("Checkpoint stage or format version is incompatible")
    if payload["model_config"] != expected_model_config.to_dict():
        raise ValueError("Checkpoint model configuration is incompatible")
    if payload["teacher_config"] != expected_teacher_config.to_dict():
        raise ValueError("Checkpoint teacher configuration is incompatible")
    if payload["training_config"] != expected_training_config:
        raise ValueError("Checkpoint training configuration is incompatible")
    if not isinstance(payload["seeds"], dict):
        raise ValueError("Checkpoint seeds must be a dictionary")
    if not isinstance(payload["source_revision"], str) or not payload["source_revision"]:
        raise ValueError("Checkpoint source revision is invalid")
    if type(payload["optimizer_steps_completed"]) is not int:
        raise ValueError("Checkpoint optimizer step count is invalid")

    model = NeuralCellularAutomaton(expected_model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metadata = {
        "training_config": payload["training_config"],
        "seeds": payload["seeds"],
        "source_revision": payload["source_revision"],
        "optimizer_steps_completed": payload["optimizer_steps_completed"],
        "optimizer_state_dict": payload["optimizer_state_dict"],
    }
    return model, metadata
