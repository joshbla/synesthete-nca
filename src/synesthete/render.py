"""Fixed monochrome Stage 1 artifact rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor


def save_stage1_visuals(
    output_dir: Path,
    *,
    target: Tensor,
    prediction: Tensor,
    frame_rate: int,
) -> dict[str, str]:
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("Target and prediction must share (frame, height, width) shape")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")

    target_array = target.detach().cpu().numpy().astype(np.float32)
    prediction_array = prediction.detach().cpu().numpy().astype(np.float32)
    difference_array = np.abs(target_array - prediction_array)
    display_difference_array = np.clip(difference_array * 4.0, 0.0, 1.0)
    native_path = output_dir / "evaluation-frames.npz"
    np.savez_compressed(
        native_path,
        target=target_array,
        prediction=prediction_array,
        absolute_difference=difference_array,
    )

    target_images = _images(target_array)
    prediction_images = _images(prediction_array)
    difference_images = _images(display_difference_array)
    comparison_images = [
        _horizontal_triptych(target_frame, prediction_frame, difference_frame)
        for target_frame, prediction_frame, difference_frame in zip(
            target_images,
            prediction_images,
            difference_images,
            strict=True,
        )
    ]
    duration_ms = round(1000 / frame_rate)
    paths = {
        "native_frames": str(native_path),
        "target_gif": str(output_dir / "target.gif"),
        "prediction_gif": str(output_dir / "prediction.gif"),
        "difference_gif": str(output_dir / "absolute-difference.gif"),
        "comparison_gif": str(output_dir / "comparison.gif"),
        "contact_sheet": str(output_dir / "comparison-contact-sheet.png"),
    }
    _save_gif(Path(paths["target_gif"]), target_images, duration_ms)
    _save_gif(Path(paths["prediction_gif"]), prediction_images, duration_ms)
    _save_gif(Path(paths["difference_gif"]), difference_images, duration_ms)
    _save_gif(Path(paths["comparison_gif"]), comparison_images, duration_ms)
    _save_contact_sheet(
        Path(paths["contact_sheet"]), target_images, prediction_images, difference_images
    )
    return paths


def _images(frames: NDArray[np.float32]) -> list[Image.Image]:
    return [
        Image.fromarray(np.rint(frame * 255.0).astype(np.uint8), mode="L").resize(
            (frame.shape[1] * 4, frame.shape[0] * 4),
            resample=Image.Resampling.NEAREST,
        )
        for frame in frames
    ]


def _horizontal_triptych(
    target: Image.Image,
    prediction: Image.Image,
    difference: Image.Image,
) -> Image.Image:
    separator = 4
    result = Image.new("L", (target.width * 3 + separator * 2, target.height), color=255)
    result.paste(target, (0, 0))
    result.paste(prediction, (target.width + separator, 0))
    result.paste(difference, (target.width * 2 + separator * 2, 0))
    return result


def _save_gif(path: Path, images: list[Image.Image], duration_ms: int) -> None:
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _save_contact_sheet(
    path: Path,
    target: list[Image.Image],
    prediction: list[Image.Image],
    difference: list[Image.Image],
) -> None:
    indices = np.linspace(0, len(target) - 1, num=min(8, len(target)), dtype=np.int64)
    tile_width = target[0].width
    tile_height = target[0].height
    sheet = Image.new("L", (tile_width * len(indices), tile_height * 3), color=0)
    for column, index in enumerate(indices):
        x = column * tile_width
        sheet.paste(target[index], (x, 0))
        sheet.paste(prediction[index], (x, tile_height))
        sheet.paste(difference[index], (x, tile_height * 2))
    sheet.save(path)
