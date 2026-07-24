from pathlib import Path

import pytest
import torch

from synesthete.nca import PERCEPTION_VERSION, NCAConfig, NeuralCellularAutomaton
from synesthete.stage1 import compute_training_loss
from synesthete.stage1_checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    ReactionDiffusionConfig,
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


@pytest.mark.parametrize("stochastic_updates", [True, False])
def test_training_loss_has_finite_gradients(stochastic_updates: bool) -> None:
    torch.manual_seed(81)
    model = NeuralCellularAutomaton(TEST_CONFIG)
    initial = torch.randn(2, 2, 16, 16).mul_(0.1)
    first_target = reaction_diffusion_step(initial, REACTION_DIFFUSION_CONFIG)
    second_target = reaction_diffusion_step(first_target, REACTION_DIFFUSION_CONFIG)
    targets = torch.stack((first_target, second_target))
    mask_generator = torch.Generator(device="cpu").manual_seed(82)

    loss, state = compute_training_loss(
        model,
        initial,
        targets,
        teacher_config=REACTION_DIFFUSION_CONFIG,
        mask_generator=mask_generator,
        hidden_loss_weight=1e-4,
        overflow_loss_weight=1.0,
        stochastic_updates=stochastic_updates,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(state).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


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
