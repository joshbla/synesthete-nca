"""Stage 2 browser-playable audiovisual rendering.

Stage 1 has already applied the fixed visual mapping before this module runs.
Inputs are native luminance tensors with shape ``[frame, height, width]``,
``float32``, in ``[0, 1]``. This module only converts that luminance to RGB for
the codec and for review labels; it does not change brightness, contrast, or
color.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

RENDER_MAPPING = "clamp(state[visible_channel] + 0.5, 0, 1)"
UPSCALE_FILTER = "nearest"

_LABEL_HEIGHT = 20
_SEPARATOR_WIDTH = 2
_SEPARATOR_COLOR = (128, 128, 128)
_LABEL_STRIP_COLOR = (24, 24, 24)
_LABEL_TEXT_COLOR = (255, 255, 255)
_LABEL_FONT = ImageFont.load_default()
_BLIND_LETTERS = "ABCDEFGHIJKLMNOP"


@dataclass(frozen=True)
class MediaConfig:
    """Fixed render configuration for Stage 2 audiovisual output."""

    frame_rate: int = 24
    upscale: int = 4

    def __post_init__(self) -> None:
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if self.upscale <= 0:
            raise ValueError("upscale must be positive")


def _require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"required system tool(s) missing: {', '.join(missing)}")


def _validate_luminance(frames: Tensor, *, name: str) -> None:
    if frames.ndim != 3:
        raise ValueError(f"{name} must have shape [frame, height, width]")
    if frames.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if frames.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    if not frames.isfinite().all():
        raise ValueError(f"{name} must be finite")
    if frames.min().item() < 0.0 or frames.max().item() > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_panels(panels: list[tuple[str, Tensor]]) -> tuple[list[str], list[Tensor]]:
    if not panels:
        raise ValueError("panels must contain at least one panel")
    labels = [label for label, _ in panels]
    tensors = [frames for _, frames in panels]
    first = tensors[0]
    for index, (label, frames) in enumerate(panels):
        if not isinstance(label, str) or not label:
            raise ValueError(f"panel {index} label must be a non-empty string")
        _validate_luminance(frames, name=f"panel {index} frames")
        if frames.shape != first.shape:
            raise ValueError("all comparison panels must share the same shape and frame count")
    return labels, tensors


def _upscale_to_rgb(luminance_2d: Tensor, upscale: int) -> Image.Image:
    native = luminance_2d.detach().cpu().numpy()
    arr = np.rint(native * 255.0).clip(0, 255).astype(np.uint8)
    gray = Image.fromarray(arr, mode="L").resize(
        (arr.shape[1] * upscale, arr.shape[0] * upscale),
        resample=Image.Resampling.NEAREST,
    )
    return gray.convert("RGB")


def _single_rgb_frames(frames: Tensor, config: MediaConfig) -> list[Image.Image]:
    return [_upscale_to_rgb(frames[i], config.upscale) for i in range(frames.shape[0])]


def _comparison_rgb_frames(
    panels: list[tuple[str, Tensor]], config: MediaConfig
) -> list[Image.Image]:
    first_shape = panels[0][1].shape
    panel_w = first_shape[2] * config.upscale
    panel_h = first_shape[1] * config.upscale
    n = len(panels)
    total_w = panel_w * n + _SEPARATOR_WIDTH * (n - 1)
    total_h = _LABEL_HEIGHT + panel_h
    frames = []
    for frame_index in range(first_shape[0]):
        canvas = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for panel_index, (label, tensor) in enumerate(panels):
            strip = Image.new("RGB", (panel_w, _LABEL_HEIGHT), color=_LABEL_STRIP_COLOR)
            strip_draw = ImageDraw.Draw(strip)
            strip_draw.text((2, 4), label, fill=_LABEL_TEXT_COLOR, font=_LABEL_FONT)
            canvas.paste(strip, (x, 0))
            canvas.paste(
                _upscale_to_rgb(tensor[frame_index], config.upscale),
                (x, _LABEL_HEIGHT),
            )
            x += panel_w
            if panel_index < n - 1:
                for sx in range(x, x + _SEPARATOR_WIDTH):
                    draw.line([(sx, 0), (sx, total_h)], fill=_SEPARATOR_COLOR)
                x += _SEPARATOR_WIDTH
        frames.append(canvas)
    return frames


def _encode(
    path: Path,
    *,
    rgb_frames: list[Image.Image],
    config: MediaConfig,
    audio_path: Path,
) -> None:
    _require_tools()
    ffmpeg = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for index, image in enumerate(rgb_frames):
            image.save(tmp / f"frame_{index:06d}.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(config.frame_rate),
            "-i",
            str(tmp / "frame_%06d.png"),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(config.frame_rate),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def _probe(path: Path) -> dict[str, object]:
    _require_tools()
    ffprobe = shutil.which("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,codec_type,pix_fmt,r_frame_rate,width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _validate_probe(probe: dict[str, object], config: MediaConfig) -> dict[str, object]:
    streams = probe["streams"]
    if len(streams) != 2:
        raise RuntimeError(f"expected exactly two streams, got {len(streams)}")
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if video is None or audio is None:
        raise RuntimeError("missing video or audio stream")
    if video["codec_name"] != "h264":
        raise RuntimeError(f"video codec is not h264: {video['codec_name']}")
    if video["pix_fmt"] != "yuv420p":
        raise RuntimeError(f"pixel format is not yuv420p: {video['pix_fmt']}")
    num, _, den = video["r_frame_rate"].partition("/")
    if not den:
        raise RuntimeError(f"frame rate is not rational: {video['r_frame_rate']}")
    den_value = int(den)
    if int(num) != config.frame_rate * den_value:
        raise RuntimeError(f"frame rate is {video['r_frame_rate']}, expected {config.frame_rate}")
    if audio["codec_name"] != "aac":
        raise RuntimeError(f"audio codec is not aac: {audio['codec_name']}")
    return {
        "width": video["width"],
        "height": video["height"],
        "duration": float(probe["format"]["duration"]),
    }


def _result(
    path: Path,
    *,
    probe_fields: dict[str, object],
    config: MediaConfig,
    audio_path: Path,
    frame_count: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    duration = float(probe_fields["duration"])
    expected_duration = frame_count / config.frame_rate
    if abs(duration - expected_duration) > 1.0 / config.frame_rate:
        raise RuntimeError(
            f"media duration is {duration}, expected {expected_duration} from frame timing"
        )
    return {
        "path": str(path),
        "codecs": {"video": "h264", "audio": "aac"},
        "pixel_format": "yuv420p",
        "dimensions": {
            "width": probe_fields["width"],
            "height": probe_fields["height"],
        },
        "fps": config.frame_rate,
        "duration": duration,
        "synchronization_offset_seconds": 0,
        "original_audio_path": str(audio_path),
        "frame_count": frame_count,
        "render_mapping": RENDER_MAPPING,
        "upscale_filter": UPSCALE_FILTER,
        "metadata": metadata,
    }


def save_audiovisual(
    path: str | Path,
    *,
    frames: Tensor,
    audio_path: str | Path,
    config: MediaConfig,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Encode a single browser-playable MP4 from native luminance frames."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    _validate_luminance(frames, name="frames")
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = _single_rgb_frames(frames, config)
    _encode(target, rgb_frames=rgb_frames, config=config, audio_path=audio)
    probe = _probe(target)
    probe_fields = _validate_probe(probe, config)
    return _result(
        target,
        probe_fields=probe_fields,
        config=config,
        audio_path=audio,
        frame_count=len(rgb_frames),
        metadata=metadata,
    )


def save_comparison_audiovisual(
    path: str | Path,
    *,
    panels: list[tuple[str, Tensor]],
    audio_path: str | Path,
    config: MediaConfig,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Encode a side-by-side synchronized comparison MP4."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    _validate_panels(panels)
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = _comparison_rgb_frames(panels, config)
    _encode(target, rgb_frames=rgb_frames, config=config, audio_path=audio)
    probe = _probe(target)
    probe_fields = _validate_probe(probe, config)
    return _result(
        target,
        probe_fields=probe_fields,
        config=config,
        audio_path=audio,
        frame_count=len(rgb_frames),
        metadata=metadata,
    )


def audio_stream_hash(path: str | Path) -> str:
    """Return the sha256 of the demuxed AAC stream bytes for blind identity checks."""
    _require_tools()
    ffmpeg = shutil.which("ffmpeg")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-f",
        "data",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def write_review_page(
    path: str | Path,
    *,
    labeled_video: str | Path,
    blind_video: str | Path,
    blind_individuals: Iterable[str | Path],
) -> dict[str, object]:
    """Write a self-contained local HTML review page with labeled and blind sections.

    Blind videos are copied to generic names so the page does not disclose the
    original condition mapping.
    """
    out_path = Path(path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_name = "labeled.mp4"
    shutil.copyfile(labeled_video, out_dir / labeled_name)

    blind_comparison_name = "blind-comparison.mp4"
    shutil.copyfile(blind_video, out_dir / blind_comparison_name)

    blind_entries: list[tuple[str, str]] = []
    for index, video in enumerate(blind_individuals):
        if index >= len(_BLIND_LETTERS):
            raise ValueError("too many blind videos for available labels")
        letter = _BLIND_LETTERS[index]
        name = f"blind-{letter.lower()}.mp4"
        source = Path(video)
        target = out_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        blind_entries.append((letter, name))

    blind_items = "\n".join(
        f'      <div><h3>{letter}</h3><video controls src="{name}"></video></div>'
        for letter, name in blind_entries
    )
    review_name = out_dir.name
    if review_name == "review":
        review_name = out_dir.parent.name
    html = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="utf-8">\n'
        "    <title>Synesthete Review</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>{review_name}</h1>\n"
        '    <section id="labeled">\n'
        "      <h2>Labeled</h2>\n"
        f'      <video controls src="{labeled_name}"></video>\n'
        "    </section>\n"
        '    <section id="blind">\n'
        "      <h2>Blind</h2>\n"
        f'      <video controls src="{blind_comparison_name}"></video>\n'
        f"{blind_items}\n"
        "    </section>\n"
        "  </body>\n"
        "</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")
    return {
        "path": str(out_path),
        "labeled_video": labeled_name,
        "blind_comparison_video": blind_comparison_name,
        "blind_videos": [name for _, name in blind_entries],
    }
