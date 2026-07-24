from pathlib import Path

import numpy as np
import torch

from synesthete.render import save_stage1_visuals


def test_visual_artifacts_preserve_raw_native_difference(tmp_path: Path) -> None:
    target = torch.zeros(2, 4, 4)
    prediction = torch.full((2, 4, 4), 0.5)

    paths = save_stage1_visuals(
        tmp_path,
        target=target,
        prediction=prediction,
        frame_rate=15,
    )
    with np.load(paths["native_frames"]) as frames:
        assert np.all(frames["absolute_difference"] == 0.5)
    assert all(Path(path).is_file() for path in paths.values())
