from pathlib import Path

import pytest
import torch

from synesthete.nca import PERCEPTION_VERSION, NCAConfig, NeuralCellularAutomaton
from synesthete.stage1 import (
    EvaluationRollout,
    TrainingData,
    _rollouts_match_exactly,
    _sample_training_window,
    compute_training_loss,
)
from synesthete.stage1_checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    ReactionDiffusionConfig,
    masked_reaction_diffusion_step,
    reaction_diffusion_step,
)

TEST_CONFIG = NCAConfig(
    state_channels=4,
    hidden_channels=8,
    fire_rate=0.5,
    step_size=0.1,
    visible_channel=0,
    boundary="circular",
    perception=PERCEPTION_VERSION,
)
TRAINING_CONFIG = {"name": "test", "rollout": 2}


@pytest.mark.parametrize(
    "update_schedule",
    ["deterministic", "mismatched_asynchronous", "mask_aligned_asynchronous"],
)
def test_training_loss_has_finite_gradients(update_schedule: str) -> None:
    torch.manual_seed(81)
    model = NeuralCellularAutomaton(TEST_CONFIG)
    initial = torch.randn(2, 2, 16, 16).mul_(0.1)
    fire_masks = torch.ones(2, 2, 1, 16, 16)
    fire_masks[:, :, :, ::2, ::2] = 0.0
    if update_schedule == "mask_aligned_asynchronous":
        first_target = masked_reaction_diffusion_step(
            initial,
            fire_masks[0],
            REACTION_DIFFUSION_CONFIG,
        )
        second_target = masked_reaction_diffusion_step(
            first_target,
            fire_masks[1],
            REACTION_DIFFUSION_CONFIG,
        )
    else:
        first_target = reaction_diffusion_step(initial, REACTION_DIFFUSION_CONFIG)
        second_target = reaction_diffusion_step(first_target, REACTION_DIFFUSION_CONFIG)
    targets = torch.stack((first_target, second_target))
    mask_generator = torch.Generator(device="cpu").manual_seed(82)

    loss, state = compute_training_loss(
        model,
        initial,
        targets,
        fire_masks,
        teacher_config=REACTION_DIFFUSION_CONFIG,
        mask_generator=mask_generator,
        hidden_loss_weight=1e-4,
        overflow_loss_weight=1.0,
        update_schedule=update_schedule,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(state).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_mask_aligned_training_respects_zero_fire_mask() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    initial = torch.randn(1, 2, 16, 16).mul_(0.1)
    targets = torch.stack((initial.clone(), initial.clone()))
    fire_masks = torch.zeros(2, 1, 1, 16, 16)

    loss, state = compute_training_loss(
        model,
        initial,
        targets,
        fire_masks,
        teacher_config=REACTION_DIFFUSION_CONFIG,
        mask_generator=torch.Generator(device="cpu").manual_seed(84),
        hidden_loss_weight=1e-4,
        overflow_loss_weight=1.0,
        update_schedule="mask_aligned_asynchronous",
    )

    assert loss == 0.0
    assert torch.equal(state[:, :2], initial)


def test_training_window_keeps_transition_masks_aligned() -> None:
    trajectories = torch.empty(2, 5, 2, 1, 1)
    fire_masks = torch.empty(2, 4, 1, 1, 1)
    for family in range(2):
        for step in range(5):
            trajectories[family, step].fill_(family * 100 + step)
        for step in range(4):
            fire_masks[family, step].fill_(family * 1000 + step)
    initial, targets, sampled_masks = _sample_training_window(
        TrainingData(trajectories=trajectories, fire_masks=fire_masks),
        batch_size=4,
        rollout_length=2,
        generator=torch.Generator(device="cpu").manual_seed(85),
    )

    for batch_index in range(4):
        encoded_start = int(initial[batch_index, 0, 0, 0].item())
        family, start = divmod(encoded_start, 100)
        assert targets[0, batch_index, 0, 0, 0] == family * 100 + start + 1
        assert targets[1, batch_index, 0, 0, 0] == family * 100 + start + 2
        assert sampled_masks[0, batch_index, 0, 0, 0] == family * 1000 + start
        assert sampled_masks[1, batch_index, 0, 0, 0] == family * 1000 + start + 1


def test_exact_rollout_match_checks_masks_states_and_frames() -> None:
    tensor = torch.zeros(2, 2, 2)
    rollout = EvaluationRollout(
        metrics={
            "initial_state_hash": "initial",
            "fire_mask_hash": "masks",
            "state_trajectory_hash": "states",
        },
        target=tensor,
        prediction=tensor,
        initial_state=torch.zeros(4, 2, 2),
        fire_masks=torch.zeros(1, 1, 2, 2),
    )
    changed = EvaluationRollout(
        metrics={**rollout.metrics, "fire_mask_hash": "changed"},
        target=rollout.target,
        prediction=rollout.prediction,
        initial_state=rollout.initial_state,
        fire_masks=rollout.fire_masks,
    )

    assert _rollouts_match_exactly(rollout, rollout)
    assert not _rollouts_match_exactly(rollout, changed)


def test_stage1_checkpoint_round_trip_includes_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(83)
    model = NeuralCellularAutomaton(TEST_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = torch.randn(1, 4, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    model(state, mask).square().mean().backward()
    optimizer.step()
    path = tmp_path / "stage1.pt"
    save_stage1_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        teacher_config=REACTION_DIFFUSION_CONFIG,
        training_config=TRAINING_CONFIG,
        seeds={"model": 83},
        source_revision="test-revision",
        optimizer_steps_completed=1,
    )

    loaded, metadata = load_stage1_checkpoint(
        path,
        expected_model_config=TEST_CONFIG,
        expected_teacher_config=REACTION_DIFFUSION_CONFIG,
        expected_training_config=TRAINING_CONFIG,
        device=torch.device("cpu"),
    )

    assert metadata["optimizer_steps_completed"] == 1
    assert metadata["optimizer_state_dict"]["state"]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(), loaded.state_dict().values(), strict=True
        )
    )


def test_stage1_checkpoint_rejects_teacher_mismatch(tmp_path: Path) -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "stage1.pt"
    save_stage1_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        teacher_config=REACTION_DIFFUSION_CONFIG,
        training_config=TRAINING_CONFIG,
        seeds={"model": 83},
        source_revision="test-revision",
        optimizer_steps_completed=1,
    )
    incompatible_teacher = ReactionDiffusionConfig(
        diffusion=0.17,
        growth=REACTION_DIFFUSION_CONFIG.growth,
        angular_frequency=REACTION_DIFFUSION_CONFIG.angular_frequency,
        saturation=REACTION_DIFFUSION_CONFIG.saturation,
        time_step=REACTION_DIFFUSION_CONFIG.time_step,
        render_offset=REACTION_DIFFUSION_CONFIG.render_offset,
    )

    with pytest.raises(ValueError, match="teacher configuration is incompatible"):
        load_stage1_checkpoint(
            path,
            expected_model_config=TEST_CONFIG,
            expected_teacher_config=incompatible_teacher,
            expected_training_config=TRAINING_CONFIG,
            device=torch.device("cpu"),
        )
