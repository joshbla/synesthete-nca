"""Report whether the local runtime is ready for MPS experiments."""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    platform: str
    torch: str
    mps_built: bool
    mps_available: bool


def collect_runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
    )


def format_runtime_info(info: RuntimeInfo) -> str:
    return "\n".join(
        (
            f"Python: {info.python}",
            f"Platform: {info.platform}",
            f"PyTorch: {info.torch}",
            f"MPS built: {info.mps_built}",
            f"MPS available: {info.mps_available}",
        )
    )


def require_mps(info: RuntimeInfo) -> None:
    if not info.mps_available:
        raise RuntimeError("MPS is required for local Synesthete experiments.")


def probe_mps() -> float:
    values = torch.arange(16, device="mps", dtype=torch.float32).reshape(4, 4)
    result = values @ values
    torch.mps.synchronize()
    return result.sum().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Exit unsuccessfully when the MPS backend is unavailable.",
    )
    args = parser.parse_args()

    info = collect_runtime_info()
    print(format_runtime_info(info))
    if args.require_mps:
        require_mps(info)
        print(f"MPS compute probe: {probe_mps():.1f}")


if __name__ == "__main__":
    main()
