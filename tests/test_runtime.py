import pytest
import torch

from synesthete.runtime import RuntimeInfo, format_runtime_info, probe_mps, require_mps


def test_format_runtime_info() -> None:
    info = RuntimeInfo(
        python="3.13.0",
        platform="test-platform",
        torch="2.11.0",
        mps_built=True,
        mps_available=True,
    )

    assert format_runtime_info(info) == "\n".join(
        (
            "Python: 3.13.0",
            "Platform: test-platform",
            "PyTorch: 2.11.0",
            "MPS built: True",
            "MPS available: True",
        )
    )


def test_require_mps_accepts_available_backend() -> None:
    info = RuntimeInfo("3.13.0", "test-platform", "2.11.0", True, True)

    require_mps(info)


def test_require_mps_rejects_unavailable_backend() -> None:
    info = RuntimeInfo("3.13.0", "test-platform", "2.11.0", False, False)

    with pytest.raises(RuntimeError, match="MPS is required"):
        require_mps(info)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_probe_mps_executes_on_device() -> None:
    assert probe_mps() == 3920.0
