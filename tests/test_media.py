import math
import re
import shutil
import struct
import wave
from pathlib import Path

import pytest
import torch

from synesthete.media import (
    MediaConfig,
    audio_stream_hash,
    save_audiovisual,
    save_comparison_audiovisual,
    write_review_page,
)


def _tools_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


pytestmark = pytest.mark.skipif(not _tools_available(), reason="ffmpeg and ffprobe are required")


def _write_wav(path: Path, *, duration: float = 0.5, rate: int = 8000, freq: float = 220.0) -> None:
    nframes = int(rate * duration)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        chunks = []
        for i in range(nframes):
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / rate))
            chunks.append(struct.pack("<h", sample))
        w.writeframes(b"".join(chunks))


def _tiny_frames() -> torch.Tensor:
    frames = torch.zeros(4, 8, 8, dtype=torch.float32)
    for f in range(4):
        frames[f, : f + 2, :] = (f + 1) / 8.0
    return frames


def test_save_audiovisual_probes_and_returns_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    video = tmp_path / "single.mp4"
    frames = _tiny_frames()
    config = MediaConfig()

    result = save_audiovisual(
        video,
        frames=frames,
        audio_path=audio,
        config=config,
        metadata={},
    )

    assert result["path"] == str(video)
    assert result["codecs"] == {"video": "h264", "audio": "aac"}
    assert result["pixel_format"] == "yuv420p"
    assert result["dimensions"] == {"width": 32, "height": 32}
    assert result["fps"] == 24
    assert result["synchronization_offset_seconds"] == 0
    assert result["original_audio_path"] == str(audio)
    assert result["frame_count"] == 4
    assert result["render_mapping"] == "clamp(state[visible_channel] + 0.5, 0, 1)"
    assert result["upscale_filter"] == "nearest"
    assert result["metadata"] == {}
    assert result["duration"] > 0.0
    assert video.is_file()


def test_mediaconfig_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="frame_rate must be positive"):
        MediaConfig(frame_rate=0)
    with pytest.raises(ValueError, match="upscale must be positive"):
        MediaConfig(upscale=0)


def test_save_audiovisual_rejects_invalid_frames(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    video = tmp_path / "out.mp4"
    config = MediaConfig()
    bad = torch.full((2, 4, 4), 0.5, dtype=torch.float32)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_audiovisual(
            video,
            frames=bad,
            audio_path=audio,
            config=config,
            metadata={},
        )
    out_of_range = torch.full((2, 4, 4), 1.5, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        save_audiovisual(
            video,
            frames=out_of_range,
            audio_path=audio,
            config=config,
            metadata={},
        )


def test_two_outputs_from_same_wav_have_matching_audio_hashes(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    frames = _tiny_frames()
    config = MediaConfig()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    save_audiovisual(first, frames=frames, audio_path=audio, config=config, metadata={})
    save_audiovisual(second, frames=frames, audio_path=audio, config=config, metadata={})

    assert audio_stream_hash(first) == audio_stream_hash(second)


def test_save_comparison_audiovisual_dimensions_and_streams(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    frames_a = _tiny_frames()
    frames_b = torch.roll(frames_a, shifts=1, dims=0)
    panels = [("alpha", frames_a), ("beta", frames_b)]
    video = tmp_path / "comparison.mp4"
    config = MediaConfig()

    result = save_comparison_audiovisual(
        video,
        panels=panels,
        audio_path=audio,
        config=config,
        metadata={},
    )

    assert result["codecs"] == {"video": "h264", "audio": "aac"}
    assert result["pixel_format"] == "yuv420p"
    assert result["dimensions"]["width"] == 32 * 2 + 2
    assert result["dimensions"]["height"] == 20 + 32
    assert result["frame_count"] == 4
    assert result["fps"] == 24
    assert video.is_file()


def test_save_comparison_rejects_mismatched_panels(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    frames_a = _tiny_frames()
    frames_b = torch.zeros(3, 8, 8, dtype=torch.float32)
    panels = [("alpha", frames_a), ("beta", frames_b)]
    with pytest.raises(ValueError, match="same shape and frame count"):
        save_comparison_audiovisual(
            tmp_path / "out.mp4",
            panels=panels,
            audio_path=audio,
            config=MediaConfig(),
            metadata={},
        )


def test_write_review_page_does_not_disclose_blind_conditions(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    frames = _tiny_frames()
    config = MediaConfig()
    labeled = tmp_path / "correct-audio.mp4"
    blind_main = tmp_path / "silence.mp4"
    blind_one = tmp_path / "time-shuffled.mp4"
    blind_two = tmp_path / "batch-shuffled.mp4"
    blind_three = tmp_path / "frozen-mean.mp4"
    for path in (labeled, blind_main, blind_one, blind_two, blind_three):
        save_audiovisual(path, frames=frames, audio_path=audio, config=config, metadata={})

    page = tmp_path / "review.html"
    result = write_review_page(
        page,
        labeled_video=labeled,
        blind_video=blind_main,
        blind_individuals=[blind_one, blind_two, blind_three],
    )

    html = page.read_text(encoding="utf-8")
    blind_section = re.search(r'<section id="blind">.*?</section>', html, flags=re.DOTALL)
    assert blind_section is not None
    blind_html = blind_section.group(0)
    for condition in (
        "silence",
        "time-shuffled",
        "batch-shuffled",
        "correct-audio",
        "frozen-mean",
    ):
        assert condition not in blind_html
    assert "A" in blind_html
    assert "B" in blind_html
    assert "C" in blind_html
    assert result["blind_comparison_video"] == "blind-comparison.mp4"
    assert result["blind_videos"] == ["blind-a.mp4", "blind-b.mp4", "blind-c.mp4"]
    assert (tmp_path / result["blind_comparison_video"]).is_file()
    for name in result["blind_videos"]:
        assert (tmp_path / name).is_file()
    assert (tmp_path / "labeled.mp4").is_file()
