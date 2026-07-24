import torch

from synesthete.teacher import (
    REACTION_DIFFUSION_CONFIG,
    encode_teacher_fields,
    generate_teacher_trajectory,
    make_teacher_fields,
    periodic_laplacian,
    reaction_diffusion_step,
    render_luminance,
)


def test_periodic_laplacian_is_translation_equivariant() -> None:
    field = torch.zeros(1, 1, 9, 9)
    field[0, 0, 0, 0] = 1.0
    shifted = torch.roll(field, shifts=(3, -2), dims=(2, 3))

    assert torch.equal(
        periodic_laplacian(shifted),
        torch.roll(periodic_laplacian(field), shifts=(3, -2), dims=(2, 3)),
    )
    assert periodic_laplacian(field)[0, 0, -1, 0] == 1.0


def test_teacher_trajectory_is_deterministic_finite_and_moving() -> None:
    first_generator = torch.Generator(device="cpu").manual_seed(71)
    second_generator = torch.Generator(device="cpu").manual_seed(71)
    first = make_teacher_fields(
        initialization="central",
        batch_size=1,
        grid_size=32,
        device=torch.device("cpu"),
        generator=first_generator,
    )
    second = make_teacher_fields(
        initialization="central",
        batch_size=1,
        grid_size=32,
        device=torch.device("cpu"),
        generator=second_generator,
    )

    first_trajectory = generate_teacher_trajectory(
        first, steps=32, config=REACTION_DIFFUSION_CONFIG
    )
    second_trajectory = generate_teacher_trajectory(
        second, steps=32, config=REACTION_DIFFUSION_CONFIG
    )

    assert torch.equal(first_trajectory, second_trajectory)
    assert torch.isfinite(first_trajectory).all()
    assert not torch.equal(first_trajectory[0], first_trajectory[-1])
    outside_center = first[:, 0].clone()
    outside_center[:, 14:18, 14:18] = 0.0
    assert 0.0 < outside_center.max() <= 0.001


def test_distributed_initialization_remains_spatially_distributed() -> None:
    generator = torch.Generator(device="cpu").manual_seed(72)
    fields = make_teacher_fields(
        initialization="distributed",
        batch_size=1,
        grid_size=32,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert torch.count_nonzero(fields[:, 1]) > 32
    assert torch.isfinite(reaction_diffusion_step(fields, REACTION_DIFFUSION_CONFIG)).all()


def test_teacher_encoding_and_render_mapping_are_fixed() -> None:
    fields = torch.tensor([[[[0.2]], [[-0.1]]]], dtype=torch.float32)

    state = encode_teacher_fields(fields, state_channels=4, config=REACTION_DIFFUSION_CONFIG)
    luminance = render_luminance(state, REACTION_DIFFUSION_CONFIG)

    assert state.shape == (1, 4, 1, 1)
    assert state[0, 0, 0, 0] == 0.2
    assert state[0, 1, 0, 0] == -0.1
    assert luminance[0, 0, 0] == 0.7
