"""Train and evaluate the Stage 1 unconditioned NCA on a fixed teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

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
    generate_masked_teacher_trajectory,
    generate_teacher_trajectory,
    make_teacher_fields,
    masked_reaction_diffusion_step,
    reaction_diffusion_step,
    render_luminance,
)

GRID_SIZE = 96
MODEL_SEED = 4101
TRAINING_DATA_SEEDS = (4102, 4103)
SELECTION_SEED = 4104
MASK_SEED = 4105
EVALUATION_SEEDS = (4106, 4107)
TRAINING_MASK_SEEDS = (MASK_SEED, MASK_SEED + 1)
EVALUATION_MASK_SEEDS = (4205, 4206)
ALTERNATE_MASK_SEEDS = (4305, 4306)

UpdateSchedule = Literal[
    "deterministic",
    "mismatched_asynchronous",
    "mask_aligned_asynchronous",
]


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
    update_schedule: UpdateSchedule
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
    update_schedule="mismatched_asynchronous",
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
    update_schedule="deterministic",
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
    update_schedule="mismatched_asynchronous",
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
    update_schedule="deterministic",
    teacher_time_step=0.1,
)
MASK_ALIGNED_SMOKE_BUDGET = replace(
    SMOKE_BUDGET,
    name="mask-aligned-smoke",
    update_schedule="mask_aligned_asynchronous",
    teacher_time_step=0.1,
)
MASK_ALIGNED_RAPID_BUDGET = replace(
    RAPID_BUDGET,
    name="mask-aligned-rapid",
    update_schedule="mask_aligned_asynchronous",
)


@dataclass(frozen=True)
class TrainingData:
    trajectories: Tensor
    fire_masks: Tensor


@dataclass(frozen=True)
class EvaluationRollout:
    metrics: dict[str, Any]
    target: Tensor
    prediction: Tensor
    initial_state: Tensor
    fire_masks: Tensor


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
) -> TrainingData:
    trajectories = []
    fire_mask_sequences = []
    for initialization, seed, mask_seed in zip(
        ("central", "distributed"),
        TRAINING_DATA_SEEDS,
        TRAINING_MASK_SEEDS,
        strict=True,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        fields = make_teacher_fields(
            initialization=initialization,
            batch_size=1,
            grid_size=GRID_SIZE,
            device=torch.device("cpu"),
            generator=generator,
        )
        if budget.update_schedule == "mask_aligned_asynchronous":
            fire_masks = _sample_fire_masks(
                steps=budget.teacher_steps,
                batch_size=1,
                grid_size=GRID_SIZE,
                device=torch.device("cpu"),
                generator=torch.Generator(device="cpu").manual_seed(mask_seed),
            )
            trajectory = generate_masked_teacher_trajectory(
                fields,
                fire_masks=fire_masks,
                config=teacher_config,
            )
        else:
            fire_masks = torch.ones(
                (budget.teacher_steps, 1, 1, GRID_SIZE, GRID_SIZE),
                dtype=torch.float32,
            )
            trajectory = generate_teacher_trajectory(
                fields,
                steps=budget.teacher_steps,
                config=teacher_config,
            )
        trajectories.append(trajectory[:, 0])
        fire_mask_sequences.append(fire_masks[:, 0])
    return TrainingData(
        trajectories=torch.stack(trajectories),
        fire_masks=torch.stack(fire_mask_sequences),
    )


def _sample_fire_masks(
    *,
    steps: int,
    batch_size: int,
    grid_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    return (
        torch.rand(
            (steps, batch_size, 1, grid_size, grid_size),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        < BASELINE_CONFIG.fire_rate
    ).to(dtype=torch.float32)


def _sample_training_window(
    training_data: TrainingData,
    *,
    batch_size: int,
    rollout_length: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    family_indices = torch.randint(
        0,
        training_data.trajectories.shape[0],
        (batch_size,),
        generator=generator,
    )
    maximum_start = training_data.trajectories.shape[1] - rollout_length
    start_indices = torch.randint(0, maximum_start, (batch_size,), generator=generator)
    families = family_indices.tolist()
    starts = start_indices.tolist()
    initial_fields = torch.stack(
        [
            training_data.trajectories[family_index, start_index]
            for family_index, start_index in zip(families, starts, strict=True)
        ]
    )
    targets = torch.stack(
        [
            torch.stack(
                [
                    training_data.trajectories[family_index, start_index + offset]
                    for family_index, start_index in zip(families, starts, strict=True)
                ]
            )
            for offset in range(1, rollout_length + 1)
        ]
    )
    fire_masks = torch.stack(
        [
            torch.stack(
                [
                    training_data.fire_masks[family_index, start_index + offset - 1]
                    for family_index, start_index in zip(families, starts, strict=True)
                ]
            )
            for offset in range(1, rollout_length + 1)
        ]
    )
    return initial_fields, targets, fire_masks


def compute_training_loss(
    model: NeuralCellularAutomaton,
    initial_fields: Tensor,
    target_fields: Tensor,
    fire_masks: Tensor,
    *,
    teacher_config: ReactionDiffusionConfig,
    mask_generator: torch.Generator,
    hidden_loss_weight: float,
    overflow_loss_weight: float,
    update_schedule: UpdateSchedule,
) -> tuple[Tensor, Tensor]:
    state = encode_teacher_fields(
        initial_fields,
        state_channels=model.config.state_channels,
        config=teacher_config,
    )
    trajectory_loss = torch.zeros((), device=state.device)
    overflow_loss = torch.zeros((), device=state.device)
    hidden_loss = torch.zeros((), device=state.device)
    for target_fields_step, recorded_fire_mask in zip(target_fields, fire_masks, strict=True):
        fire_mask = _model_fire_mask(
            model,
            state,
            recorded_fire_mask,
            update_schedule=update_schedule,
            mask_generator=mask_generator,
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


def _model_fire_mask(
    model: NeuralCellularAutomaton,
    state: Tensor,
    recorded_fire_mask: Tensor,
    *,
    update_schedule: UpdateSchedule,
    mask_generator: torch.Generator,
) -> Tensor:
    if update_schedule == "mismatched_asynchronous":
        return model.sample_fire_mask(state, mask_generator)
    if update_schedule in ("deterministic", "mask_aligned_asynchronous"):
        return recorded_fire_mask
    raise ValueError(f"Unsupported update schedule: {update_schedule}")


def _train(
    model: NeuralCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    training_data: TrainingData,
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
        initial_fields, target_fields, fire_masks = _sample_training_window(
            training_data,
            batch_size=budget.batch_size,
            rollout_length=rollout_length,
            generator=selection_generator,
        )
        initial_fields = initial_fields.to(device)
        target_fields = target_fields.to(device)
        fire_masks = fire_masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, final_state = compute_training_loss(
            model,
            initial_fields,
            target_fields,
            fire_masks,
            teacher_config=teacher_config,
            mask_generator=mask_generator,
            hidden_loss_weight=budget.hidden_loss_weight,
            overflow_loss_weight=budget.overflow_loss_weight,
            update_schedule=budget.update_schedule,
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
) -> EvaluationRollout:
    generator = torch.Generator(device=device).manual_seed(seed)
    fields = make_teacher_fields(
        initialization=initialization,
        batch_size=1,
        grid_size=GRID_SIZE,
        device=device,
        generator=generator,
    )
    if budget.update_schedule == "mask_aligned_asynchronous":
        teacher_fire_masks = _sample_fire_masks(
            steps=budget.evaluation_burn_in + budget.evaluation_steps,
            batch_size=1,
            grid_size=GRID_SIZE,
            device=device,
            generator=torch.Generator(device=device).manual_seed(mask_seed),
        )
    else:
        teacher_fire_masks = torch.ones(
            (
                budget.evaluation_burn_in + budget.evaluation_steps,
                1,
                1,
                GRID_SIZE,
                GRID_SIZE,
            ),
            device=device,
            dtype=torch.float32,
        )
    with torch.no_grad():
        for burn_in_step in range(budget.evaluation_burn_in):
            if budget.update_schedule == "mask_aligned_asynchronous":
                fields = masked_reaction_diffusion_step(
                    fields,
                    teacher_fire_masks[burn_in_step],
                    teacher_config,
                )
            else:
                fields = reaction_diffusion_step(fields, teacher_config)
        state = encode_teacher_fields(
            fields,
            state_channels=model.config.state_channels,
            config=teacher_config,
        )
        initial_state = state[0].clone()
        target_frames = [render_luminance(state, teacher_config)[0]]
        prediction_frames = [target_frames[0].clone()]
        mask_generator = torch.Generator(device=device).manual_seed(mask_seed)
        applied_fire_masks = []
        trajectory_digest = hashlib.sha256()
        trajectory_digest.update(state.detach().cpu().contiguous().numpy().tobytes())
        squared_error = torch.zeros((), device=device)
        short_squared_error = torch.zeros((), device=device)
        per_channel_minimum = state.amin(dim=(0, 2, 3))
        per_channel_maximum = state.amax(dim=(0, 2, 3))
        maximum_state_magnitude = state.abs().max()
        for step in range(1, budget.evaluation_steps + 1):
            recorded_fire_mask = teacher_fire_masks[budget.evaluation_burn_in + step - 1]
            if budget.update_schedule == "mask_aligned_asynchronous":
                fields = masked_reaction_diffusion_step(
                    fields,
                    recorded_fire_mask,
                    teacher_config,
                )
            else:
                fields = reaction_diffusion_step(fields, teacher_config)
            target_state = encode_teacher_fields(
                fields,
                state_channels=model.config.state_channels,
                config=teacher_config,
            )
            fire_mask = _model_fire_mask(
                model,
                state,
                recorded_fire_mask,
                update_schedule=budget.update_schedule,
                mask_generator=mask_generator,
            )
            applied_fire_masks.append(fire_mask[0].clone())
            state = model(state, fire_mask)
            if not torch.isfinite(fields).all() or not torch.isfinite(state).all():
                raise RuntimeError(f"Non-finite {initialization} evaluation state at step {step}")
            trajectory_digest.update(state.detach().cpu().contiguous().numpy().tobytes())
            step_error = (state[:, :2] - target_state[:, :2]).square().mean()
            squared_error = squared_error + step_error
            if step <= 16:
                short_squared_error = short_squared_error + step_error
            per_channel_minimum = torch.minimum(per_channel_minimum, state.amin(dim=(0, 2, 3)))
            per_channel_maximum = torch.maximum(per_channel_maximum, state.amax(dim=(0, 2, 3)))
            maximum_state_magnitude = torch.maximum(maximum_state_magnitude, state.abs().max())
            if step % budget.render_stride == 0:
                target_frames.append(render_luminance(target_state, teacher_config)[0])
                prediction_frames.append(render_luminance(state, teacher_config)[0])
        target_render = torch.stack(target_frames)
        prediction_render = torch.stack(prediction_frames)
        applied_fire_mask_tensor = torch.stack(applied_fire_masks)
        final_variance = state.var(dim=(0, 2, 3), unbiased=False)
        metrics = {
            "initialization": initialization,
            "state_seed": seed,
            "mask_seed": mask_seed,
            "finite": True,
            "trajectory_mse": (squared_error / budget.evaluation_steps).item(),
            "short_16_step_mse": (short_squared_error / min(16, budget.evaluation_steps)).item(),
            "target_motion_energy": _motion_energy(target_render),
            "prediction_motion_energy": _motion_energy(prediction_render),
            "target_total_variation": _total_variation(target_render),
            "prediction_total_variation": _total_variation(prediction_render),
            "target_spatial_spectral_centroid": _spatial_spectral_centroid(target_render),
            "prediction_spatial_spectral_centroid": _spatial_spectral_centroid(prediction_render),
            "maximum_state_magnitude": maximum_state_magnitude.item(),
            "initial_state_hash": _tensor_hash(initial_state),
            "fire_mask_hash": _tensor_hash(applied_fire_mask_tensor),
            "state_trajectory_hash": trajectory_digest.hexdigest(),
            "per_channel_minimum": per_channel_minimum.cpu().tolist(),
            "per_channel_maximum": per_channel_maximum.cpu().tolist(),
            "final_per_channel_variance": final_variance.cpu().tolist(),
        }
    return EvaluationRollout(
        metrics=metrics,
        target=target_render,
        prediction=prediction_render,
        initial_state=initial_state,
        fire_masks=applied_fire_mask_tensor,
    )


def _tensor_hash(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _ratio_is_comparable(prediction: float, target: float, *, factor: float) -> bool:
    return target > 0.0 and 1.0 / factor <= prediction / target <= factor


def _standard_automated_checks(evaluations: list[dict[str, Any]]) -> dict[str, bool]:
    return {
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
            _ratio_is_comparable(
                evaluation["prediction_motion_energy"],
                evaluation["target_motion_energy"],
                factor=4.0,
            )
            for evaluation in evaluations
        ),
    }


def _mask_aligned_automated_checks(
    evaluations: list[dict[str, Any]], *, exact_replay: bool
) -> dict[str, bool]:
    return {
        "all_finite": all(evaluation["finite"] for evaluation in evaluations),
        "all_states_bounded_below_two": all(
            evaluation["maximum_state_magnitude"] < 2.0 for evaluation in evaluations
        ),
        "motion_nonzero": all(
            evaluation["prediction_motion_energy"] > 1e-6 for evaluation in evaluations
        ),
        "short_horizon_accurate": all(
            evaluation["short_16_step_mse"] < 1e-4 for evaluation in evaluations
        ),
        "long_horizon_accurate": all(
            evaluation["trajectory_mse"] < 2e-2 for evaluation in evaluations
        ),
        "motion_rate_within_factor_four": all(
            _ratio_is_comparable(
                evaluation["prediction_motion_energy"],
                evaluation["target_motion_energy"],
                factor=4.0,
            )
            for evaluation in evaluations
        ),
        "total_variation_within_factor_two": all(
            _ratio_is_comparable(
                evaluation["prediction_total_variation"],
                evaluation["target_total_variation"],
                factor=2.0,
            )
            for evaluation in evaluations
        ),
        "spectral_centroid_within_factor_two": all(
            _ratio_is_comparable(
                evaluation["prediction_spatial_spectral_centroid"],
                evaluation["target_spatial_spectral_centroid"],
                factor=2.0,
            )
            for evaluation in evaluations
        ),
        "exact_replay": exact_replay,
    }


def _mask_aligned_smoke_checks(
    evaluations: list[dict[str, Any]], *, exact_replay: bool
) -> dict[str, bool]:
    return {
        "all_finite": all(evaluation["finite"] for evaluation in evaluations),
        "all_states_bounded_below_two": all(
            evaluation["maximum_state_magnitude"] < 2.0 for evaluation in evaluations
        ),
        "motion_nonzero": all(
            evaluation["prediction_motion_energy"] > 1e-6 for evaluation in evaluations
        ),
        "exact_replay": exact_replay,
    }


def _evaluate(
    model: NeuralCellularAutomaton,
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    if budget.update_schedule == "mask_aligned_asynchronous":
        return _evaluate_mask_aligned(model, budget, teacher_config, device, output_dir)

    evaluations = []
    artifact_rollout = None
    for initialization, seed, mask_seed in zip(
        ("central", "distributed"),
        EVALUATION_SEEDS,
        EVALUATION_MASK_SEEDS,
        strict=True,
    ):
        rollout = _evaluate_initialization(
            model,
            initialization=initialization,
            seed=seed,
            mask_seed=mask_seed,
            budget=budget,
            teacher_config=teacher_config,
            device=device,
        )
        evaluations.append(rollout.metrics)
        if initialization == "distributed":
            artifact_rollout = rollout
    if artifact_rollout is None:
        raise RuntimeError("Distributed evaluation artifact was not produced")
    visuals = save_stage1_visuals(
        output_dir,
        target=artifact_rollout.target,
        prediction=artifact_rollout.prediction,
        initial_state=artifact_rollout.initial_state,
        fire_masks=artifact_rollout.fire_masks,
        frame_rate=15,
    )
    automated_checks = _standard_automated_checks(evaluations)
    return {
        "initializations": evaluations,
        "automated_checks": automated_checks,
        "automated_checks_passed": all(automated_checks.values()),
        "visuals": visuals,
        "visual_initialization": "distributed",
        "visual_layout": "comparison columns are target, prediction, absolute difference",
    }


def _evaluate_mask_aligned(
    model: NeuralCellularAutomaton,
    budget: Stage1Budget,
    teacher_config: ReactionDiffusionConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    recorded_evaluations = []
    alternate_evaluations = []
    replay_results = []
    visuals: dict[str, dict[str, dict[str, str]]] = {
        "recorded_mask": {},
        "alternate_mask": {},
    }
    for initialization, state_seed, recorded_mask_seed, alternate_mask_seed in zip(
        ("central", "distributed"),
        EVALUATION_SEEDS,
        EVALUATION_MASK_SEEDS,
        ALTERNATE_MASK_SEEDS,
        strict=True,
    ):
        recorded = _evaluate_initialization(
            model,
            initialization=initialization,
            seed=state_seed,
            mask_seed=recorded_mask_seed,
            budget=budget,
            teacher_config=teacher_config,
            device=device,
        )
        replay = _evaluate_initialization(
            model,
            initialization=initialization,
            seed=state_seed,
            mask_seed=recorded_mask_seed,
            budget=budget,
            teacher_config=teacher_config,
            device=device,
        )
        alternate = _evaluate_initialization(
            model,
            initialization=initialization,
            seed=state_seed,
            mask_seed=alternate_mask_seed,
            budget=budget,
            teacher_config=teacher_config,
            device=device,
        )
        replay_exact = _rollouts_match_exactly(recorded, replay)
        replay_results.append(
            {
                "initialization": initialization,
                "state_seed": state_seed,
                "mask_seed": recorded_mask_seed,
                "exact": replay_exact,
                "state_trajectory_hash": recorded.metrics["state_trajectory_hash"],
            }
        )
        recorded_evaluations.append(recorded.metrics)
        alternate_evaluations.append(alternate.metrics)
        for condition, rollout in (
            ("recorded_mask", recorded),
            ("alternate_mask", alternate),
        ):
            artifact_dir = output_dir / f"{condition.replace('_', '-')}-{initialization}"
            artifact_dir.mkdir()
            visuals[condition][initialization] = save_stage1_visuals(
                artifact_dir,
                target=rollout.target,
                prediction=rollout.prediction,
                initial_state=rollout.initial_state,
                fire_masks=rollout.fire_masks,
                frame_rate=15,
            )
    exact_replay = all(result["exact"] for result in replay_results)
    all_evaluations = [*recorded_evaluations, *alternate_evaluations]
    if budget.name == "mask-aligned-smoke":
        automated_checks = _mask_aligned_smoke_checks(
            all_evaluations,
            exact_replay=exact_replay,
        )
    else:
        automated_checks = _mask_aligned_automated_checks(
            all_evaluations,
            exact_replay=exact_replay,
        )
    return {
        "initializations": recorded_evaluations,
        "alternate_mask_initializations": alternate_evaluations,
        "exact_replay": {
            "passed": exact_replay,
            "initializations": replay_results,
        },
        "automated_checks": automated_checks,
        "automated_checks_passed": all(automated_checks.values()),
        "visuals": visuals,
        "visual_initializations": ["central", "distributed"],
        "visual_layout": "comparison columns are target, prediction, absolute difference",
    }


def _rollouts_match_exactly(first: EvaluationRollout, second: EvaluationRollout) -> bool:
    return (
        first.metrics["initial_state_hash"] == second.metrics["initial_state_hash"]
        and first.metrics["fire_mask_hash"] == second.metrics["fire_mask_hash"]
        and first.metrics["state_trajectory_hash"] == second.metrics["state_trajectory_hash"]
        and torch.equal(first.target, second.target)
        and torch.equal(first.prediction, second.prediction)
    )


def run_stage1(output_dir: Path, budget: Stage1Budget) -> dict[str, Any]:
    device = _require_mps_without_fallback()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_revision = _source_revision()
    teacher_config = replace(
        REACTION_DIFFUSION_CONFIG,
        time_step=budget.teacher_time_step,
    )
    training_data = _build_training_trajectories(budget, teacher_config)
    model = _make_model(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.learning_rate)
    training_log, training_seconds = _train(
        model,
        optimizer,
        training_data,
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
        "mismatched_mask": MASK_SEED,
        "training_mask_central": TRAINING_MASK_SEEDS[0],
        "training_mask_distributed": TRAINING_MASK_SEEDS[1],
        "evaluation_central": EVALUATION_SEEDS[0],
        "evaluation_distributed": EVALUATION_SEEDS[1],
        "evaluation_mask_central": EVALUATION_MASK_SEEDS[0],
        "evaluation_mask_distributed": EVALUATION_MASK_SEEDS[1],
        "alternate_mask_central": ALTERNATE_MASK_SEEDS[0],
        "alternate_mask_distributed": ALTERNATE_MASK_SEEDS[1],
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
    visuals = result["evaluation"]["visuals"]
    if result["budget"]["update_schedule"] == "mask_aligned_asynchronous":
        print(f"Visual comparison: {visuals['recorded_mask']['distributed']['comparison_gif']}")
    else:
        print(f"Visual comparison: {visuals['comparison_gif']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "budget",
        choices=(
            "smoke",
            "rapid",
            "asynchronous",
            "stress",
            "mask-aligned-smoke",
            "mask-aligned-rapid",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    budgets = {
        "smoke": SMOKE_BUDGET,
        "rapid": RAPID_BUDGET,
        "asynchronous": ASYNCHRONOUS_BUDGET,
        "stress": STRESS_BUDGET,
        "mask-aligned-smoke": MASK_ALIGNED_SMOKE_BUDGET,
        "mask-aligned-rapid": MASK_ALIGNED_RAPID_BUDGET,
    }
    result = run_stage1(args.output_dir, budgets[args.budget])
    _print_summary(result)


if __name__ == "__main__":
    main()
