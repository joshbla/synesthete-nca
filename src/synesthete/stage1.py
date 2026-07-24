"""Train and evaluate the Stage 1 unconditioned NCA on a fixed teacher."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from synesthete.nca import BASELINE_CONFIG, NeuralCellularAutomaton
from synesthete.render import save_stage1_visuals
from synesthete.runtime import collect_runtime_info, require_mps
from synesthete.stage1_checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    ReactionDiffusionConfig,
    encode_teacher_fields,
    generate_teacher_trajectory,
    make_teacher_fields,
    reaction_diffusion_step,
    render_luminance,
)

GRID_SIZE = 96
MODEL_SEED = 4101
TRAINING_DATA_SEEDS = (4102, 4103)
SELECTION_SEED = 4104
MASK_SEED = 4105
EVALUATION_SEEDS = (4106, 4107)


@dataclass(frozen=True)
class Stage1Budget:
    name: str
    optimizer_steps: int
    batch_size: int
    minimum_rollout: int
    maximum_rollout: int
    teacher_steps: int
    evaluation_burn_in: int
    evaluation_steps: int
    render_stride: int
    learning_rate: float
    hidden_loss_weight: float
    overflow_loss_weight: float
    stochastic_updates: bool
    teacher_time_step: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SMOKE_BUDGET = Stage1Budget(
    name="smoke",
    optimizer_steps=50,
    batch_size=2,
    minimum_rollout=1,
    maximum_rollout=4,
    teacher_steps=256,
    evaluation_burn_in=64,
    evaluation_steps=64,
    render_stride=2,
    learning_rate=1e-3,
    hidden_loss_weight=1e-4,
    overflow_loss_weight=1.0,
    stochastic_updates=True,
    teacher_time_step=0.05,
)
RAPID_BUDGET = Stage1Budget(
    name="rapid",
    optimizer_steps=2_000,
    batch_size=4,
    minimum_rollout=2,
    maximum_rollout=8,
    teacher_steps=2_400,
    evaluation_burn_in=800,
    evaluation_steps=512,
    render_stride=4,
    learning_rate=1e-3,
    hidden_loss_weight=1e-2,
    overflow_loss_weight=1.0,
    stochastic_updates=False,
    teacher_time_step=0.1,
)
ASYNCHRONOUS_BUDGET = Stage1Budget(
    name="asynchronous",
    optimizer_steps=2_000,
    batch_size=4,
    minimum_rollout=2,
    maximum_rollout=8,
    teacher_steps=2_400,
    evaluation_burn_in=800,
    evaluation_steps=512,
    render_stride=4,
    learning_rate=1e-3,
    hidden_loss_weight=1e-2,
    overflow_loss_weight=1.0,
    stochastic_updates=True,
    teacher_time_step=0.05,
)
STRESS_BUDGET = Stage1Budget(
    name="stress",
    optimizer_steps=8_000,
    batch_size=4,
    minimum_rollout=4,
    maximum_rollout=16,
    teacher_steps=4_800,
    evaluation_burn_in=1_600,
    evaluation_steps=1_024,
    render_stride=8,
    learning_rate=5e-4,
    hidden_loss_weight=1e-2,
    overflow_loss_weight=1.0,
    stochastic_updates=False,
    teacher_time_step=0.1,
)


def _require_mps_without_fallback() -> torch.device:
    info = collect_runtime_info()
    require_mps(info)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Stage 1 forbids PYTORCH_ENABLE_MPS_FALLBACK=1")
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


def _make_model(device: torch.device) -> NeuralCellularAutomaton:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = NeuralCellularAutomaton(BASELINE_CONFIG)
    return model.to(device)


def _build_training_trajectories(
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
) -> Tensor:
    trajectories = []
    for initialization, seed in zip(("central", "distributed"), TRAINING_DATA_SEEDS, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        fields = make_teacher_fields(
            initialization=initialization,
            batch_size=1,
            grid_size=GRID_SIZE,
            device=torch.device("cpu"),
            generator=generator,
        )
        trajectory = generate_teacher_trajectory(
            fields,
            steps=budget.teacher_steps,
            config=teacher_config,
        )
        trajectories.append(trajectory[:, 0])
    return torch.stack(trajectories)


def _sample_training_window(
    trajectories: Tensor,
    *,
    batch_size: int,
    rollout_length: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    family_indices = torch.randint(0, trajectories.shape[0], (batch_size,), generator=generator)
    maximum_start = trajectories.shape[1] - rollout_length
    start_indices = torch.randint(0, maximum_start, (batch_size,), generator=generator)
    families = family_indices.tolist()
    starts = start_indices.tolist()
    initial_fields = torch.stack(
        [
            trajectories[family_index, start_index]
            for family_index, start_index in zip(families, starts, strict=True)
        ]
    )
    targets = torch.stack(
        [
            torch.stack(
                [
                    trajectories[family_index, start_index + offset]
                    for family_index, start_index in zip(families, starts, strict=True)
                ]
            )
            for offset in range(1, rollout_length + 1)
        ]
    )
    return initial_fields, targets


def compute_training_loss(
    model: NeuralCellularAutomaton,
    initial_fields: Tensor,
    target_fields: Tensor,
    *,
    teacher_config: ReactionDiffusionConfig,
    mask_generator: torch.Generator,
    hidden_loss_weight: float,
    overflow_loss_weight: float,
    stochastic_updates: bool,
) -> tuple[Tensor, Tensor]:
    state = encode_teacher_fields(
        initial_fields,
        state_channels=model.config.state_channels,
        config=teacher_config,
    )
    trajectory_loss = torch.zeros((), device=state.device)
    overflow_loss = torch.zeros((), device=state.device)
    hidden_loss = torch.zeros((), device=state.device)
    for target_fields_step in target_fields:
        if stochastic_updates:
            fire_mask = model.sample_fire_mask(state, mask_generator)
        else:
            fire_mask = torch.ones(
                (state.shape[0], 1, state.shape[2], state.shape[3]),
                device=state.device,
                dtype=state.dtype,
            )
        state = model(state, fire_mask)
        target_state = encode_teacher_fields(
            target_fields_step,
            state_channels=model.config.state_channels,
            config=teacher_config,
        )
        trajectory_loss = trajectory_loss + (state[:, :2] - target_state[:, :2]).square().mean()
        overflow_loss = overflow_loss + torch.relu(state.abs() - 4.0).square().mean()
        hidden_loss = hidden_loss + state[:, 2:].square().mean()
    rollout_length = target_fields.shape[0]
    trajectory_loss = trajectory_loss / rollout_length
    overflow_loss = overflow_loss / rollout_length
    hidden_loss = hidden_loss / rollout_length
    total = (
        trajectory_loss + hidden_loss_weight * hidden_loss + overflow_loss_weight * overflow_loss
    )
    return total, state


def _train(
    model: NeuralCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    trajectories: Tensor,
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], float]:
    selection_generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    mask_generator = torch.Generator(device=device).manual_seed(MASK_SEED)
    log_interval = max(1, budget.optimizer_steps // 20)
    log = []
    torch.mps.synchronize()
    started = time.perf_counter()
    for optimizer_step in range(1, budget.optimizer_steps + 1):
        rollout_length = int(
            torch.randint(
                budget.minimum_rollout,
                budget.maximum_rollout + 1,
                (),
                generator=selection_generator,
            ).item()
        )
        initial_fields, target_fields = _sample_training_window(
            trajectories,
            batch_size=budget.batch_size,
            rollout_length=rollout_length,
            generator=selection_generator,
        )
        initial_fields = initial_fields.to(device)
        target_fields = target_fields.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, final_state = compute_training_loss(
            model,
            initial_fields,
            target_fields,
            teacher_config=teacher_config,
            mask_generator=mask_generator,
            hidden_loss_weight=budget.hidden_loss_weight,
            overflow_loss_weight=budget.overflow_loss_weight,
            stochastic_updates=budget.stochastic_updates,
        )
        if not torch.isfinite(loss) or not torch.isfinite(final_state).all():
            raise RuntimeError(f"Non-finite training state at optimizer step {optimizer_step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient at optimizer step {optimizer_step}")
        optimizer.step()
        if optimizer_step == 1 or optimizer_step % log_interval == 0:
            torch.mps.synchronize()
            log.append(
                {
                    "optimizer_step": optimizer_step,
                    "rollout_length": rollout_length,
                    "loss": loss.item(),
                    "gradient_norm": gradient_norm.item(),
                    "maximum_state_magnitude": final_state.abs().max().item(),
                }
            )
    torch.mps.synchronize()
    return log, time.perf_counter() - started


def _total_variation(frames: Tensor) -> float:
    horizontal = (frames - torch.roll(frames, 1, dims=-1)).abs().mean()
    vertical = (frames - torch.roll(frames, 1, dims=-2)).abs().mean()
    return (horizontal + vertical).item()


def _motion_energy(frames: Tensor) -> float:
    return (frames[1:] - frames[:-1]).abs().mean().item()


def _spatial_spectral_centroid(frames: Tensor) -> float:
    frames = frames.detach().cpu()
    centered = frames - frames.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(centered).abs().square()
    vertical_frequency = torch.fft.fftfreq(frames.shape[-2])
    horizontal_frequency = torch.fft.rfftfreq(frames.shape[-1])
    radius = torch.sqrt(
        vertical_frequency[:, None].square() + horizontal_frequency[None, :].square()
    )
    total_power = power.sum()
    if total_power == 0.0:
        return 0.0
    return (power * radius).sum().div(total_power).item()


def _evaluate_initialization(
    model: NeuralCellularAutomaton,
    *,
    initialization: str,
    seed: int,
    mask_seed: int,
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    fields = make_teacher_fields(
        initialization=initialization,
        batch_size=1,
        grid_size=GRID_SIZE,
        device=device,
        generator=generator,
    )
    with torch.no_grad():
        for _ in range(budget.evaluation_burn_in):
            fields = reaction_diffusion_step(fields, teacher_config)
        state = encode_teacher_fields(
            fields,
            state_channels=model.config.state_channels,
            config=teacher_config,
        )
        target_frames = [render_luminance(state, teacher_config)[0]]
        prediction_frames = [target_frames[0].clone()]
        mask_generator = torch.Generator(device=device).manual_seed(mask_seed)
        squared_error = torch.zeros((), device=device)
        short_squared_error = torch.zeros((), device=device)
        per_channel_minimum = state.amin(dim=(0, 2, 3))
        per_channel_maximum = state.amax(dim=(0, 2, 3))
        for step in range(1, budget.evaluation_steps + 1):
            fields = reaction_diffusion_step(fields, teacher_config)
            target_state = encode_teacher_fields(
                fields,
                state_channels=model.config.state_channels,
                config=teacher_config,
            )
            if budget.stochastic_updates:
                fire_mask = model.sample_fire_mask(state, mask_generator)
            else:
                fire_mask = torch.ones(
                    (state.shape[0], 1, state.shape[2], state.shape[3]),
                    device=state.device,
                    dtype=state.dtype,
                )
            state = model(state, fire_mask)
            step_error = (state[:, :2] - target_state[:, :2]).square().mean()
            squared_error = squared_error + step_error
            if step <= 16:
                short_squared_error = short_squared_error + step_error
            per_channel_minimum = torch.minimum(per_channel_minimum, state.amin(dim=(0, 2, 3)))
            per_channel_maximum = torch.maximum(per_channel_maximum, state.amax(dim=(0, 2, 3)))
            if step % budget.render_stride == 0:
                target_frames.append(render_luminance(target_state, teacher_config)[0])
                prediction_frames.append(render_luminance(state, teacher_config)[0])
        target_render = torch.stack(target_frames)
        prediction_render = torch.stack(prediction_frames)
        finite = bool(torch.isfinite(state).all().item())
        final_variance = state.var(dim=(0, 2, 3), unbiased=False)
        metrics = {
            "initialization": initialization,
            "finite": finite,
            "trajectory_mse": (squared_error / budget.evaluation_steps).item(),
            "short_16_step_mse": (short_squared_error / min(16, budget.evaluation_steps)).item(),
            "target_motion_energy": _motion_energy(target_render),
            "prediction_motion_energy": _motion_energy(prediction_render),
            "target_total_variation": _total_variation(target_render),
            "prediction_total_variation": _total_variation(prediction_render),
            "target_spatial_spectral_centroid": _spatial_spectral_centroid(target_render),
            "prediction_spatial_spectral_centroid": _spatial_spectral_centroid(prediction_render),
            "maximum_state_magnitude": state.abs().max().item(),
            "per_channel_minimum": per_channel_minimum.cpu().tolist(),
            "per_channel_maximum": per_channel_maximum.cpu().tolist(),
            "final_per_channel_variance": final_variance.cpu().tolist(),
        }
    return metrics, target_render, prediction_render


def _evaluate(
    model: NeuralCellularAutomaton,
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    evaluations = []
    artifact_target = None
    artifact_prediction = None
    for index, (initialization, seed) in enumerate(
        zip(("central", "distributed"), EVALUATION_SEEDS, strict=True)
    ):
        metrics, target, prediction = _evaluate_initialization(
            model,
            initialization=initialization,
            seed=seed,
            mask_seed=MASK_SEED + 100 + index,
            budget=budget,
            teacher_config=teacher_config,
            device=device,
        )
        evaluations.append(metrics)
        if initialization == "distributed":
            artifact_target = target
            artifact_prediction = prediction
    if artifact_target is None or artifact_prediction is None:
        raise RuntimeError("Distributed evaluation artifact was not produced")
    visuals = save_stage1_visuals(
        output_dir,
        target=artifact_target,
        prediction=artifact_prediction,
        frame_rate=15,
    )
    automated_checks = {
        "all_finite": all(evaluation["finite"] for evaluation in evaluations),
        "all_states_bounded_below_two": all(
            evaluation["maximum_state_magnitude"] < 2.0 for evaluation in evaluations
        ),
        "central_motion_nonzero": evaluations[0]["prediction_motion_energy"] > 1e-6,
        "short_horizon_accurate": all(
            evaluation["short_16_step_mse"] < 1e-4 for evaluation in evaluations
        ),
        "long_horizon_accurate": all(
            evaluation["trajectory_mse"] < 2e-2 for evaluation in evaluations
        ),
        "motion_rate_comparable": all(
            evaluation["target_motion_energy"] > 0.0
            and 0.25
            <= evaluation["prediction_motion_energy"] / evaluation["target_motion_energy"]
            <= 4.0
            for evaluation in evaluations
        ),
    }
    return {
        "initializations": evaluations,
        "automated_checks": automated_checks,
        "automated_checks_passed": all(automated_checks.values()),
        "visuals": visuals,
        "visual_initialization": "distributed",
        "visual_layout": "comparison columns are target, prediction, absolute difference",
    }


def run_stage1(output_dir: Path, budget: Stage1Budget) -> dict[str, Any]:
    device = _require_mps_without_fallback()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_revision = _source_revision()
    teacher_config = replace(
        REACTION_DIFFUSION_CONFIG,
        time_step=budget.teacher_time_step,
    )
    trajectories = _build_training_trajectories(budget, teacher_config)
    model = _make_model(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.learning_rate)
    training_log, training_seconds = _train(
        model,
        optimizer,
        trajectories,
        budget,
        teacher_config,
        device,
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    seeds = {
        "model": MODEL_SEED,
        "training_central": TRAINING_DATA_SEEDS[0],
        "training_distributed": TRAINING_DATA_SEEDS[1],
        "selection": SELECTION_SEED,
        "mask": MASK_SEED,
        "evaluation_central": EVALUATION_SEEDS[0],
        "evaluation_distributed": EVALUATION_SEEDS[1],
    }
    save_stage1_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        teacher_config=teacher_config,
        training_config=budget.to_dict(),
        seeds=seeds,
        source_revision=source_revision,
        optimizer_steps_completed=budget.optimizer_steps,
    )
    loaded_model, checkpoint_metadata = load_stage1_checkpoint(
        checkpoint_path,
        expected_model_config=BASELINE_CONFIG,
        expected_teacher_config=teacher_config,
        expected_training_config=budget.to_dict(),
        device=device,
    )
    loaded_model.eval()
    evaluation = _evaluate(loaded_model, budget, teacher_config, device, output_dir)
    result = {
        "stage": 1,
        "source_revision": source_revision,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_fallback_enabled": False,
        },
        "model": {
            "config": BASELINE_CONFIG.to_dict(),
            "parameter_count": loaded_model.parameter_count,
            "grid_size": GRID_SIZE,
        },
        "teacher": teacher_config.to_dict(),
        "budget": budget.to_dict(),
        "seeds": seeds,
        "training": {
            "seconds": training_seconds,
            "optimizer_steps_per_second": budget.optimizer_steps / training_seconds,
            "log": training_log,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "optimizer_steps_completed": checkpoint_metadata["optimizer_steps_completed"],
            "round_trip_loaded": True,
        },
        "evaluation": evaluation,
        "interpretation_boundary": (
            "Automated checks cover finiteness, bounds, reconstruction, and motion; "
            "coherent dynamics still require review of the fixed artifacts."
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _print_summary(result: dict[str, Any]) -> None:
    training = result["training"]
    print(f"Training seconds: {training['seconds']:.2f}")
    print(f"Optimizer steps/second: {training['optimizer_steps_per_second']:.2f}")
    for evaluation in result["evaluation"]["initializations"]:
        print(
            f"{evaluation['initialization']}: mse={evaluation['trajectory_mse']:.6f}, "
            f"motion={evaluation['prediction_motion_energy']:.6f}, "
            f"max_state={evaluation['maximum_state_magnitude']:.3f}"
        )
    print(f"Automated checks passed: {result['evaluation']['automated_checks_passed']}")
    print(f"Visual comparison: {result['evaluation']['visuals']['comparison_gif']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", choices=("smoke", "rapid", "asynchronous", "stress"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    budgets = {
        "smoke": SMOKE_BUDGET,
        "rapid": RAPID_BUDGET,
        "asynchronous": ASYNCHRONOUS_BUDGET,
        "stress": STRESS_BUDGET,
    }
    result = run_stage1(args.output_dir, budgets[args.budget])
    _print_summary(result)


if __name__ == "__main__":
    main()
