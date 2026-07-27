import pytest
import torch

from synesthete.conditioned_nca import ConditionedNCAConfig, ConditionedNeuralCellularAutomaton
from synesthete.nca import PERCEPTION_VERSION, NCAConfig, NeuralCellularAutomaton

TEST_CONFIG = ConditionedNCAConfig(
    state_channels=2,
    hidden_channels=8,
    fire_rate=0.5,
    step_size=0.1,
    visible_channel=0,
    boundary="circular",
    perception=PERCEPTION_VERSION,
    conditioning_dim=1,
)


def _nontrivial_models() -> tuple[NeuralCellularAutomaton, ConditionedNeuralCellularAutomaton]:
    torch.manual_seed(11)
    ordinary = NeuralCellularAutomaton(
        NCAConfig(
            state_channels=2,
            hidden_channels=8,
            fire_rate=0.5,
            step_size=0.1,
            visible_channel=0,
            boundary="circular",
            perception=PERCEPTION_VERSION,
        )
    )
    with torch.no_grad():
        ordinary.update_output.weight.fill_(0.01)
        ordinary.update_output.bias.fill_(0.001)
    conditioned = ConditionedNeuralCellularAutomaton(TEST_CONFIG)
    conditioned.transfer_unconditioned_state_dict(ordinary.state_dict())
    return ordinary, conditioned


def test_config_rejects_nonpositive_conditioning_dim() -> None:
    with pytest.raises(ValueError, match="conditioning_dim must be positive"):
        ConditionedNCAConfig(
            state_channels=2,
            hidden_channels=8,
            fire_rate=0.5,
            step_size=0.1,
            visible_channel=0,
            boundary="circular",
            perception=PERCEPTION_VERSION,
            conditioning_dim=0,
        )


def test_config_retains_parent_validation() -> None:
    with pytest.raises(ValueError, match="state_channels must be positive"):
        ConditionedNCAConfig(
            state_channels=0,
            hidden_channels=8,
            fire_rate=0.5,
            step_size=0.1,
            visible_channel=0,
            boundary="circular",
            perception=PERCEPTION_VERSION,
            conditioning_dim=1,
        )


def test_zero_initialized_conditioning_matches_ordinary_nca() -> None:
    ordinary, conditioned = _nontrivial_models()
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)

    zero_conditioning = torch.zeros(1, 1)
    nonzero_conditioning = torch.tensor([[0.7]])

    ordinary_result = ordinary(state, mask)
    zero_result = conditioned(state, mask, zero_conditioning)
    nonzero_result = conditioned(state, mask, nonzero_conditioning)

    assert torch.equal(zero_result, ordinary_result)
    assert torch.equal(nonzero_result, ordinary_result)
    assert torch.equal(zero_result, nonzero_result)


def test_transfer_rejects_unexpected_keys() -> None:
    ordinary, _ = _nontrivial_models()
    conditioned = ConditionedNeuralCellularAutomaton(TEST_CONFIG)
    payload = dict(ordinary.state_dict())
    payload["stray.key"] = torch.zeros(1)

    with pytest.raises(ValueError, match="unexpected keys"):
        conditioned.transfer_unconditioned_state_dict(payload)


def test_transfer_rejects_missing_ordinary_parameter() -> None:
    ordinary, _ = _nontrivial_models()
    conditioned = ConditionedNeuralCellularAutomaton(TEST_CONFIG)
    payload = dict(ordinary.state_dict())
    del payload["update_output.weight"]

    with pytest.raises(ValueError, match="missing"):
        conditioned.transfer_unconditioned_state_dict(payload)


def test_transfer_leaves_conditioning_parameters_zero() -> None:
    _, conditioned = _nontrivial_models()

    assert torch.count_nonzero(conditioned.conditioning_projection.weight) == 0
    assert torch.count_nonzero(conditioned.conditioning_projection.bias) == 0


def test_conditioning_shape_validation() -> None:
    _, conditioned = _nontrivial_models()
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="conditioning shape"):
        conditioned(state, mask, torch.zeros(1, 2))


def test_conditioning_batch_validation() -> None:
    _, conditioned = _nontrivial_models()
    state = torch.randn(2, 2, 8, 8)
    mask = torch.ones(2, 1, 8, 8)

    with pytest.raises(ValueError, match="conditioning shape"):
        conditioned(state, mask, torch.zeros(1, 1))


def test_conditioning_dtype_validation() -> None:
    _, conditioned = _nontrivial_models()
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="conditioning must be fp32"):
        conditioned(state, mask, torch.zeros(1, 1, dtype=torch.float64))


def test_conditioning_nonfinite_validation() -> None:
    _, conditioned = _nontrivial_models()
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="conditioning must be finite"):
        conditioned(state, mask, torch.tensor([[float("inf")]]))


def test_gradients_reach_projection_and_update_parameters() -> None:
    _, conditioned = _nontrivial_models()
    state = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    conditioning = torch.tensor([[0.7]])

    conditioned_out = conditioned(state, mask, conditioning)
    conditioned_out.sum().backward()

    assert conditioned.update_hidden.weight.grad is not None
    assert torch.isfinite(conditioned.update_hidden.weight.grad).all()
    assert conditioned.update_output.weight.grad is not None
    assert torch.isfinite(conditioned.update_output.weight.grad).all()

    assert conditioned.conditioning_projection.weight.grad is not None
    assert torch.isfinite(conditioned.conditioning_projection.weight.grad).all()
    assert torch.count_nonzero(conditioned.conditioning_projection.weight.grad) > 0
    assert conditioned.conditioning_projection.bias.grad is not None
    assert torch.isfinite(conditioned.conditioning_projection.bias.grad).all()
    assert torch.count_nonzero(conditioned.conditioning_projection.bias.grad) > 0


def test_parameter_count_delta_matches_conditioning_projection() -> None:
    ordinary = NeuralCellularAutomaton(
        NCAConfig(
            state_channels=2,
            hidden_channels=8,
            fire_rate=0.5,
            step_size=0.1,
            visible_channel=0,
            boundary="circular",
            perception=PERCEPTION_VERSION,
        )
    )
    conditioned = ConditionedNeuralCellularAutomaton(TEST_CONFIG)

    projection_params = (
        conditioned.conditioning_projection.weight.numel()
        + conditioned.conditioning_projection.bias.numel()
    )
    assert conditioned.parameter_count - ordinary.parameter_count == projection_params
    assert projection_params == 2 * 8 * 1 + 2 * 8
