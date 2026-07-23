from pathlib import Path

import pytest
import torch

from synesthete.checkpoint import load_checkpoint, save_checkpoint
from synesthete.nca import PERCEPTION_VERSION, NCAConfig, NeuralCellularAutomaton

TEST_CONFIG = NCAConfig(
    state_channels=2,
    hidden_channels=8,
    fire_rate=0.5,
    step_size=0.1,
    visible_channel=0,
    boundary="circular",
    perception=PERCEPTION_VERSION,
)


def _model_with_probe_dynamics() -> NeuralCellularAutomaton:
    torch.manual_seed(11)
    model = NeuralCellularAutomaton(TEST_CONFIG)
    with torch.no_grad():
        model.update_output.weight.fill_(0.01)
    return model


def _rollout(model: NeuralCellularAutomaton, state: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(8):
        state = model(state, model.sample_fire_mask(state, generator))
    return state


def test_perception_is_circular_and_translation_equivariant() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    state = torch.zeros(1, 2, 7, 7)
    state[0, 0, 0, 0] = 1.0
    shifted_state = torch.roll(state, shifts=(2, -3), dims=(2, 3))

    perceived = model.perceive(state)
    shifted_perceived = model.perceive(shifted_state)

    assert perceived.shape == (1, 8, 7, 7)
    assert torch.equal(
        shifted_perceived,
        torch.roll(perceived, shifts=(2, -3), dims=(2, 3)),
    )
    assert torch.count_nonzero(perceived[:, 1:4, -1, :]) > 0
    assert torch.count_nonzero(perceived[:, 1:4, :, -1]) > 0


def test_constant_field_has_zero_gradient_and_laplacian() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    state = torch.ones(1, 2, 5, 5)

    perceived = model.perceive(state).view(1, 4, 2, 5, 5)

    assert torch.equal(perceived[:, 0], state)
    assert torch.count_nonzero(perceived[:, 1:]) == 0


def test_zero_output_initialization_is_exact_identity() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)

    result = model(state, mask)

    assert torch.count_nonzero(model.update_output.weight) == 0
    assert torch.count_nonzero(model.update_output.bias) == 0
    assert torch.equal(result, state)
    assert model.parameter_count == 90


def test_fire_masks_replay_from_seed() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    state = torch.zeros(1, 2, 32, 32)
    first = torch.Generator(device="cpu").manual_seed(23)
    second = torch.Generator(device="cpu").manual_seed(23)
    different = torch.Generator(device="cpu").manual_seed(24)

    first_masks = [model.sample_fire_mask(state, first) for _ in range(4)]
    second_masks = [model.sample_fire_mask(state, second) for _ in range(4)]
    different_masks = [model.sample_fire_mask(state, different) for _ in range(4)]

    assert all(
        torch.equal(left, right) for left, right in zip(first_masks, second_masks, strict=True)
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first_masks, different_masks, strict=True)
    )


def test_mask_seed_controls_nonzero_trajectory() -> None:
    model = _model_with_probe_dynamics()
    torch.manual_seed(31)
    state = torch.randn(1, 2, 8, 8).mul_(0.01)

    first = _rollout(model, state.clone(), seed=41)
    exact_repeat = _rollout(model, state.clone(), seed=41)
    different_masks = _rollout(model, state.clone(), seed=42)

    assert torch.equal(first, exact_repeat)
    assert not torch.equal(first, state)
    assert not torch.equal(first, different_masks)


def test_checkpoint_round_trip_replays_trajectory(tmp_path: Path) -> None:
    model = _model_with_probe_dynamics()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=11,
        state_seed=51,
        mask_seed=41,
        grid_size=8,
        batch_size=1,
        source_revision="test-revision",
    )

    loaded, metadata = load_checkpoint(
        checkpoint_path,
        expected_config=TEST_CONFIG,
        expected_grid_size=8,
        expected_batch_size=1,
        device=torch.device("cpu"),
    )
    torch.manual_seed(51)
    state = torch.randn(1, 2, 8, 8).mul_(0.01)

    assert metadata["mask_seed"] == 41
    assert metadata["state_seed"] == 51
    assert torch.equal(
        _rollout(model, state.clone(), seed=41),
        _rollout(loaded, state.clone(), seed=41),
    )


def test_checkpoint_rejects_incompatible_configuration(tmp_path: Path) -> None:
    model = _model_with_probe_dynamics()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=11,
        state_seed=51,
        mask_seed=41,
        grid_size=8,
        batch_size=1,
        source_revision="test-revision",
    )
    incompatible = NCAConfig(
        state_channels=2,
        hidden_channels=9,
        fire_rate=0.5,
        step_size=0.1,
        visible_channel=0,
        boundary="circular",
        perception=PERCEPTION_VERSION,
    )

    with pytest.raises(ValueError, match="configuration is incompatible"):
        load_checkpoint(
            checkpoint_path,
            expected_config=incompatible,
            expected_grid_size=8,
            expected_batch_size=1,
            device=torch.device("cpu"),
        )


def test_checkpoint_rejects_incompatible_run_shape(tmp_path: Path) -> None:
    model = _model_with_probe_dynamics()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=11,
        state_seed=51,
        mask_seed=41,
        grid_size=8,
        batch_size=1,
        source_revision="test-revision",
    )

    with pytest.raises(ValueError, match="grid size is incompatible"):
        load_checkpoint(
            checkpoint_path,
            expected_config=TEST_CONFIG,
            expected_grid_size=9,
            expected_batch_size=1,
            device=torch.device("cpu"),
        )


def test_checkpoint_rejects_missing_contract_field(tmp_path: Path) -> None:
    model = _model_with_probe_dynamics()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=11,
        state_seed=51,
        mask_seed=41,
        grid_size=8,
        batch_size=1,
        source_revision="test-revision",
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["mask_seed"]
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="fields do not match"):
        load_checkpoint(
            checkpoint_path,
            expected_config=TEST_CONFIG,
            expected_grid_size=8,
            expected_batch_size=1,
            device=torch.device("cpu"),
        )


def test_long_identity_rollout_remains_finite() -> None:
    model = NeuralCellularAutomaton(TEST_CONFIG)
    state = torch.randn(1, 2, 8, 8)
    initial = state.clone()
    generator = torch.Generator(device="cpu").manual_seed(61)

    with torch.no_grad():
        for _ in range(256):
            state = model(state, model.sample_fire_mask(state, generator))

    assert torch.equal(state, initial)
    assert torch.isfinite(state).all()
