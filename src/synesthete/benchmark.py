"""Run the Stage 0 NCA runtime and reproducibility benchmark on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from synesthete.checkpoint import load_checkpoint, save_checkpoint
from synesthete.nca import BASELINE_CONFIG, NCAConfig, NeuralCellularAutomaton, make_field_state
from synesthete.runtime import collect_runtime_info, require_mps

GRID_SIZE = 96
BATCH_SIZE = 1
MODEL_SEED = 1701
STATE_SEED = 1702
MASK_SEED = 1703
PROBE_STEPS = 64
ROLLOUT_LENGTHS = (1, 4, 8, 16, 32)


@dataclass(frozen=True)
class BenchmarkBudget:
    name: str
    steady_updates: int
    training_repeats: int
    long_rollout_steps: int
    trace_interval: int


@dataclass(frozen=True)
class MemorySnapshot:
    current_allocated_bytes: int
    driver_allocated_bytes: int
    process_peak_rss_bytes: int


SMOKE_BUDGET = BenchmarkBudget(
    name="smoke",
    steady_updates=20,
    training_repeats=1,
    long_rollout_steps=64,
    trace_interval=16,
)
FULL_BUDGET = BenchmarkBudget(
    name="full",
    steady_updates=100,
    training_repeats=5,
    long_rollout_steps=10_000,
    trace_interval=1_000,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _memory_snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        current_allocated_bytes=torch.mps.current_allocated_memory(),
        driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


@contextmanager
def _seeded_model_construction(seed: int) -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _make_model(config: NCAConfig, device: torch.device, seed: int) -> NeuralCellularAutomaton:
    with _seeded_model_construction(seed):
        model = NeuralCellularAutomaton(config)
    return model.to(device)


def _make_state(config: NCAConfig, device: torch.device, seed: int) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return make_field_state(
        batch_size=BATCH_SIZE,
        grid_size=GRID_SIZE,
        config=config,
        device=device,
        generator=generator,
        amplitude=0.01,
    )


def _new_mask_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _time_inference(
    model: NeuralCellularAutomaton,
    state: Tensor,
    generator: torch.Generator,
    updates: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(updates):
            state = model(state, model.sample_fire_mask(state, generator))
    _synchronize(device)
    return state, time.perf_counter() - started


def _benchmark_inference(device: torch.device, budget: BenchmarkBudget) -> dict[str, Any]:
    model = _make_model(BASELINE_CONFIG, device, MODEL_SEED).eval()
    state = _make_state(BASELINE_CONFIG, device, STATE_SEED)
    generator = _new_mask_generator(device, MASK_SEED)
    memory_before = _memory_snapshot()
    state, first_seconds = _time_inference(model, state, generator, 1, device)
    memory_after_first = _memory_snapshot()
    state, steady_seconds = _time_inference(
        model,
        state,
        generator,
        budget.steady_updates,
        device,
    )
    memory_after_steady = _memory_snapshot()
    return {
        "first_update_seconds": first_seconds,
        "steady_update_count": budget.steady_updates,
        "steady_total_seconds": steady_seconds,
        "steady_seconds_per_update": steady_seconds / budget.steady_updates,
        "steady_updates_per_second": budget.steady_updates / steady_seconds,
        "memory_before": asdict(memory_before),
        "memory_after_first": asdict(memory_after_first),
        "memory_after_steady": asdict(memory_after_steady),
    }


def _training_step(
    model: NeuralCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    initial_state: Tensor,
    generator: torch.Generator,
    rollout_length: int,
    device: torch.device,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    state = initial_state
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(rollout_length):
        state = model(state, model.sample_fire_mask(state, generator))
    loss = state.square().mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError(f"Non-finite gradient at rollout length {rollout_length}")
    optimizer.step()
    _synchronize(device)
    return time.perf_counter() - started, gradient_norm.item()


def _training_memory_probe(
    model: NeuralCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    initial_state: Tensor,
    generator: torch.Generator,
    rollout_length: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    before = _memory_snapshot()
    state = initial_state
    for _ in range(rollout_length):
        state = model(state, model.sample_fire_mask(state, generator))
    loss = state.square().mean()
    _synchronize(device)
    after_forward = _memory_snapshot()
    loss.backward()
    _synchronize(device)
    after_backward = _memory_snapshot()
    optimizer.step()
    _synchronize(device)
    after_step = _memory_snapshot()
    observed_peak = max(
        before.current_allocated_bytes,
        after_forward.current_allocated_bytes,
        after_backward.current_allocated_bytes,
        after_step.current_allocated_bytes,
    )
    return {
        "before": asdict(before),
        "after_forward": asdict(after_forward),
        "after_backward": asdict(after_backward),
        "after_step": asdict(after_step),
        "observed_peak_allocated_bytes": observed_peak,
    }


def _benchmark_training(device: torch.device, budget: BenchmarkBudget) -> list[dict[str, Any]]:
    results = []
    for rollout_length in ROLLOUT_LENGTHS:
        torch.mps.empty_cache()
        model = _make_model(BASELINE_CONFIG, device, MODEL_SEED).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial_state = _make_state(BASELINE_CONFIG, device, STATE_SEED)
        generator = _new_mask_generator(device, MASK_SEED)
        first_seconds, first_gradient_norm = _training_step(
            model,
            optimizer,
            initial_state,
            generator,
            rollout_length,
            device,
        )
        memory = _training_memory_probe(
            model,
            optimizer,
            initial_state,
            generator,
            rollout_length,
            device,
        )
        steady_times = []
        gradient_norms = []
        for _ in range(budget.training_repeats):
            elapsed, gradient_norm = _training_step(
                model,
                optimizer,
                initial_state,
                generator,
                rollout_length,
                device,
            )
            steady_times.append(elapsed)
            gradient_norms.append(gradient_norm)
        mean_seconds = sum(steady_times) / len(steady_times)
        results.append(
            {
                "rollout_length": rollout_length,
                "first_step_seconds": first_seconds,
                "first_gradient_norm": first_gradient_norm,
                "steady_repeats": budget.training_repeats,
                "steady_warmup_optimizer_steps": 2,
                "steady_mean_seconds_per_step": mean_seconds,
                "steady_min_seconds_per_step": min(steady_times),
                "steady_max_seconds_per_step": max(steady_times),
                "steady_gradient_norms": gradient_norms,
                "projected_optimizer_steps_in_10_minutes": 600.0 / mean_seconds,
                "projected_optimizer_steps_in_2_hours": 7200.0 / mean_seconds,
                "memory": memory,
            }
        )
    return results


def _channel_statistics(state: Tensor) -> list[dict[str, float]]:
    dimensions = (0, 2, 3)
    minimum = state.amin(dim=dimensions).cpu().tolist()
    maximum = state.amax(dim=dimensions).cpu().tolist()
    mean = state.mean(dim=dimensions).cpu().tolist()
    variance = state.var(dim=dimensions, unbiased=False).cpu().tolist()
    return [
        {
            "min": minimum[channel],
            "max": maximum[channel],
            "mean": mean[channel],
            "variance": variance[channel],
        }
        for channel in range(state.shape[1])
    ]


def _benchmark_long_inference(
    device: torch.device,
    budget: BenchmarkBudget,
) -> dict[str, Any]:
    model = _make_model(BASELINE_CONFIG, device, MODEL_SEED).eval()
    state = _make_state(BASELINE_CONFIG, device, STATE_SEED)
    generator = _new_mask_generator(device, MASK_SEED)
    trace = [
        {
            "step": 0,
            "finite": bool(torch.isfinite(state).all().item()),
            "memory": asdict(_memory_snapshot()),
            "channels": _channel_statistics(state),
        }
    ]
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(1, budget.long_rollout_steps + 1):
            state = model(state, model.sample_fire_mask(state, generator))
            if step % budget.trace_interval == 0:
                _synchronize(device)
                finite = bool(torch.isfinite(state).all().item())
                trace.append(
                    {
                        "step": step,
                        "finite": finite,
                        "memory": asdict(_memory_snapshot()),
                        "channels": _channel_statistics(state),
                    }
                )
                if not finite:
                    raise RuntimeError(f"Non-finite NCA state at inference step {step}")
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "steps": budget.long_rollout_steps,
        "seconds": elapsed,
        "seconds_per_update": elapsed / budget.long_rollout_steps,
        "updates_per_second": budget.long_rollout_steps / elapsed,
        "trace": trace,
    }


def _initialize_probe_dynamics(model: NeuralCellularAutomaton, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(
        model.update_output.weight.shape,
        dtype=torch.float32,
        generator=generator,
    ).mul_(1e-3)
    with torch.no_grad():
        model.update_output.weight.copy_(weight.to(model.update_output.weight.device))
        model.update_output.bias.zero_()


def _trajectory_hash(
    model: NeuralCellularAutomaton,
    device: torch.device,
    state_seed: int,
    mask_seed: int,
) -> tuple[str, bool, float]:
    state = _make_state(model.config, device, state_seed)
    initial_state = state.clone()
    generator = _new_mask_generator(device, mask_seed)
    digest = hashlib.sha256()
    with torch.no_grad():
        for _ in range(PROBE_STEPS):
            state = model(state, model.sample_fire_mask(state, generator))
            digest.update(state.detach().cpu().contiguous().numpy().tobytes())
    changed = not torch.equal(state, initial_state)
    mean_absolute_change = (state - initial_state).abs().mean().item()
    return digest.hexdigest(), changed, mean_absolute_change


def _reproducibility_probe(
    device: torch.device,
    checkpoint_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    model = _make_model(BASELINE_CONFIG, device, MODEL_SEED).eval()
    _initialize_probe_dynamics(model, MODEL_SEED)
    first_hash, changed, mean_absolute_change = _trajectory_hash(
        model, device, STATE_SEED, MASK_SEED
    )
    second_hash, second_changed, second_mean_absolute_change = _trajectory_hash(
        model, device, STATE_SEED, MASK_SEED
    )
    if not changed or not second_changed or mean_absolute_change == 0.0:
        raise RuntimeError("Reproducibility probe did not produce a nonzero trajectory")
    if first_hash != second_hash:
        raise RuntimeError("Exact MPS trajectory replay failed")
    if mean_absolute_change != second_mean_absolute_change:
        raise RuntimeError("Exact MPS trajectory motion did not replay")

    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=MODEL_SEED,
        state_seed=STATE_SEED,
        mask_seed=MASK_SEED,
        grid_size=GRID_SIZE,
        batch_size=BATCH_SIZE,
        source_revision=source_revision,
    )
    loaded_model, metadata = load_checkpoint(
        checkpoint_path,
        expected_config=BASELINE_CONFIG,
        expected_grid_size=GRID_SIZE,
        expected_batch_size=BATCH_SIZE,
        device=device,
    )
    loaded_model.eval()
    loaded_hash, loaded_changed, loaded_mean_absolute_change = _trajectory_hash(
        loaded_model,
        device,
        int(metadata["state_seed"]),
        int(metadata["mask_seed"]),
    )
    if loaded_hash != first_hash:
        raise RuntimeError("Checkpoint reload changed the state trajectory")
    if not loaded_changed or loaded_mean_absolute_change != mean_absolute_change:
        raise RuntimeError("Checkpoint reload changed the trajectory motion")
    return {
        "probe_steps": PROBE_STEPS,
        "first_hash": first_hash,
        "exact_repeat_hash": second_hash,
        "checkpoint_reload_hash": loaded_hash,
        "bit_exact": True,
        "initial_to_final_mean_absolute_change": mean_absolute_change,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
    }


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


def _require_mps_without_fallback() -> torch.device:
    info = collect_runtime_info()
    require_mps(info)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Stage 0 forbids PYTORCH_ENABLE_MPS_FALLBACK=1")
    return torch.device("mps")


def run_benchmark(output_dir: Path, budget: BenchmarkBudget) -> dict[str, Any]:
    device = _require_mps_without_fallback()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_revision = _source_revision()
    checkpoint_path = output_dir / "checkpoint.pt"
    started = time.perf_counter()
    result = {
        "stage": 0,
        "budget": asdict(budget),
        "source_revision": source_revision,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback_enabled": False,
            "mps_recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
        },
        "model": {
            "config": BASELINE_CONFIG.to_dict(),
            "grid_size": GRID_SIZE,
            "batch_size": BATCH_SIZE,
            "state_shape": [
                BATCH_SIZE,
                BASELINE_CONFIG.state_channels,
                GRID_SIZE,
                GRID_SIZE,
            ],
            "perception_shape": [
                BATCH_SIZE,
                BASELINE_CONFIG.state_channels * 4,
                GRID_SIZE,
                GRID_SIZE,
            ],
            "parameter_count": _make_model(BASELINE_CONFIG, device, MODEL_SEED).parameter_count,
        },
        "inference": _benchmark_inference(device, budget),
        "training": _benchmark_training(device, budget),
        "long_inference": _benchmark_long_inference(device, budget),
        "reproducibility": _reproducibility_probe(
            device,
            checkpoint_path,
            source_revision,
        ),
    }
    result["total_wall_clock_seconds"] = time.perf_counter() - started
    report_path = output_dir / "benchmark.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _print_summary(result: dict[str, Any]) -> None:
    inference = result["inference"]
    long_inference = result["long_inference"]
    reproducibility = result["reproducibility"]
    print(f"Parameters: {result['model']['parameter_count']}")
    print(f"First NCA update: {inference['first_update_seconds']:.6f} s")
    print(f"Steady NCA updates: {inference['steady_updates_per_second']:.2f}/s")
    print(
        f"Long inference: {long_inference['steps']} updates at "
        f"{long_inference['updates_per_second']:.2f}/s"
    )
    for training in result["training"]:
        print(
            f"Train rollout {training['rollout_length']:>2}: "
            f"{training['steady_mean_seconds_per_step']:.4f} s/step"
        )
    print(f"Bit-exact replay: {reproducibility['bit_exact']}")
    print(f"Total wall clock: {result['total_wall_clock_seconds']:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", choices=("smoke", "full"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    budget = SMOKE_BUDGET if args.budget == "smoke" else FULL_BUDGET
    result = run_benchmark(args.output_dir, budget)
    _print_summary(result)


if __name__ == "__main__":
    main()
