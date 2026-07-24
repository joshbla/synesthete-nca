"""Focused unit tests for the Stage 2 audio module."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
import torch

from synesthete.audio import (
    AudioConfig,
    AudioFeatures,
    counterfactual_features,
    extract_features,
    interpolate_features,
    synthesize_probe,
    write_pcm16_wav,
)

CONFIG = AudioConfig()
NAMES = ["silence", "constant", "impulse", "alternating", "stepped-bands", "review"]


def test_samples_per_frame_exact() -> None:
    assert CONFIG.samples_per_frame == 1000
    with pytest.raises(ValueError):
        AudioConfig(sample_rate=24001, frame_rate=24)


@pytest.mark.parametrize("name", NAMES)
def test_probe_determinism_and_length(name: str) -> None:
    a = synthesize_probe(name, CONFIG, seed=7)
    b = synthesize_probe(name, CONFIG, seed=7)
    c = synthesize_probe(name, CONFIG, seed=8)
    assert a.ndim == 1
    assert a.dtype == torch.float32
    assert a.numel() % CONFIG.samples_per_frame == 0
    assert torch.equal(a, b)
    # Probes that consume the generator are seed-sensitive; silence and the
    # pure-tone stepped-bands probe are deterministic by construction.
    if name not in ("silence", "stepped-bands"):
        assert not torch.equal(a, c)
    if name == "review":
        assert a.numel() == CONFIG.sample_rate * 10
    else:
        assert a.numel() == CONFIG.sample_rate * 4


def test_silence_is_zero() -> None:
    s = synthesize_probe("silence", CONFIG, seed=1)
    assert s.abs().max().item() == 0.0
    feats = extract_features(s, CONFIG)
    assert feats.rms.abs().max().item() == 0.0
    assert feats.onset.abs().max().item() == 0.0
    assert feats.spectral_flux.abs().max().item() == 0.0
    assert feats.low_band.abs().max().item() == 0.0
    assert feats.high_band.abs().max().item() == 0.0


def test_absolute_energy_ordering() -> None:
    silence = extract_features(synthesize_probe("silence", CONFIG, seed=1), CONFIG)
    constant = extract_features(synthesize_probe("constant", CONFIG, seed=1), CONFIG)
    assert silence.rms.max().item() == 0.0
    assert constant.rms.median().item() > 0.0
    # Constant noise RMS sits near the documented target.
    assert 0.15 < constant.rms.median().item() < 0.25
    # Absolute RMS ordering: silence < constant.
    assert silence.rms.mean().item() < constant.rms.mean().item()


def test_impulse_event_placement() -> None:
    audio = synthesize_probe("impulse", CONFIG, seed=3)
    n = audio.numel()
    # Quiet outside a narrow window around the midpoint burst.
    pre = audio[: n // 2 - 1000]
    post = audio[n // 2 + 1000 :]
    assert pre.abs().max().item() == 0.0
    assert post.abs().max().item() == 0.0
    # Audible energy at the midpoint.
    assert audio[n // 2 : n // 2 + 600].abs().max().item() > 0.0


def test_alternating_probe_changes_energy_not_spectral_identity() -> None:
    audio = synthesize_probe("alternating", CONFIG, seed=3)
    block = CONFIG.sample_rate // 2
    block_rms = [
        audio[start : start + block].square().mean().sqrt().item()
        for start in range(0, audio.numel(), block)
    ]
    assert min(block_rms[::2]) > max(block_rms[1::2]) * 4.0


def test_review_event_placement() -> None:
    sr = CONFIG.sample_rate
    audio = synthesize_probe("review", CONFIG, seed=5)
    assert audio.numel() == sr * 10
    # 0-1s silence, 2-3s silence, 9-10s silence.
    assert audio[:sr].abs().max().item() == 0.0
    assert audio[2 * sr : 3 * sr].abs().max().item() == 0.0
    assert audio[9 * sr : 10 * sr].abs().max().item() == 0.0
    # 1-2s constant noise.
    assert audio[1 * sr : 2 * sr].abs().mean().item() > 0.0
    # Burst exactly at 3s.
    assert audio[3 * sr : 3 * sr + 10].abs().max().item() > 0.0
    assert audio[3 * sr + 700 : int(4.5 * sr)].abs().max().item() == 0.0
    # 7-8s and 8-9s tones.
    assert audio[7 * sr : 8 * sr].abs().max().item() > 0.0
    assert audio[8 * sr : 9 * sr].abs().max().item() > 0.0


@pytest.mark.parametrize("name", NAMES)
def test_features_shape_and_finite(name: str) -> None:
    audio = synthesize_probe(name, CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    expected = audio.numel() // CONFIG.samples_per_frame
    for col in (feats.rms, feats.onset, feats.spectral_flux, feats.low_band, feats.high_band):
        assert col.shape == (expected,)
        assert col.dtype == torch.float32
        assert torch.isfinite(col).all()
    assert feats.matrix().shape == (expected, 5)


def test_matrix_column_order() -> None:
    audio = synthesize_probe("constant", CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    m = feats.matrix()
    assert torch.equal(m[:, 0], feats.rms)
    assert torch.equal(m[:, 1], feats.onset)
    assert torch.equal(m[:, 2], feats.spectral_flux)
    assert torch.equal(m[:, 3], feats.low_band)
    assert torch.equal(m[:, 4], feats.high_band)


def test_tone_band_discrimination_matched_rms() -> None:
    audio = synthesize_probe("stepped-bands", CONFIG, seed=4)
    feats = extract_features(audio, CONFIG)
    n_frames = feats.rms.numel()
    block = CONFIG.sample_rate // 2 // CONFIG.samples_per_frame  # frames per 0.5s
    low_frames = list(range(0, n_frames, 2 * block))[:block]
    high_frames = list(range(block, n_frames, 2 * block))[:block]
    high_low = feats.low_band[high_frames].mean().item()
    high_high = feats.high_band[high_frames].mean().item()
    low_low = feats.low_band[low_frames].mean().item()
    low_high = feats.high_band[low_frames].mean().item()
    # 4000Hz blocks: high band dominates low band.
    assert high_high > high_low
    # 160Hz blocks: low band dominates high band.
    assert low_low > low_high
    # Equal RMS: tone blocks share comparable RMS.
    assert abs(feats.rms[high_frames].mean().item() - feats.rms[low_frames].mean().item()) < 0.02


def test_review_tone_discrimination() -> None:
    sr = CONFIG.sample_rate
    audio = synthesize_probe("review", CONFIG, seed=6)
    feats = extract_features(audio, CONFIG)
    f7 = 7 * sr // CONFIG.samples_per_frame
    f8 = 8 * sr // CONFIG.samples_per_frame
    # 160Hz tone frame: low > high.
    assert feats.low_band[f7].item() > feats.high_band[f7].item()
    # 4000Hz tone frame: high > low.
    assert feats.high_band[f8].item() > feats.low_band[f8].item()
    # Matched RMS between the two tones.
    assert abs(feats.rms[f7].item() - feats.rms[f8].item()) < 0.02


def test_interpolation_shape_and_hold() -> None:
    audio = synthesize_probe("constant", CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    upf = 4
    interp = interpolate_features(feats, updates_per_frame=upf)
    n_frames = feats.rms.numel()
    assert interp.shape == (n_frames * upf, 5)
    # First substep equals the frame itself.
    assert torch.allclose(interp[0], feats.matrix()[0])
    # Final frame is held across all its substeps.
    last = feats.matrix()[-1]
    for s in range(upf):
        assert torch.allclose(interp[-(upf - s)], last)
    # Invalid updates_per_frame rejected.
    with pytest.raises(ValueError):
        interpolate_features(feats, updates_per_frame=0)


def test_counterfactual_correct_and_silent() -> None:
    audio = synthesize_probe("alternating", CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    correct = counterfactual_features(feats, "correct", seed=1)
    assert torch.equal(correct.matrix(), feats.matrix())
    # Independent copy.
    assert correct.matrix().data_ptr() != feats.matrix().data_ptr()
    silent = counterfactual_features(feats, "silent", seed=1)
    assert silent.matrix().abs().max().item() == 0.0


def test_counterfactual_shuffled_joint() -> None:
    audio = synthesize_probe("alternating", CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    sh1 = counterfactual_features(feats, "shuffled", seed=11)
    sh1b = counterfactual_features(feats, "shuffled", seed=11)
    sh2 = counterfactual_features(feats, "shuffled", seed=12)
    # Deterministic in seed.
    assert torch.equal(sh1.matrix(), sh1b.matrix())
    assert not torch.equal(sh1.matrix(), sh2.matrix())
    # Joint permutation: derive it from the seed and confirm every column follows
    # the same index permutation.
    gen = torch.Generator().manual_seed(11)
    expected_perm = torch.randperm(feats.rms.numel(), generator=gen)
    assert torch.equal(sh1.rms, feats.rms[expected_perm])
    assert torch.equal(sh1.high_band, feats.high_band[expected_perm])
    assert torch.equal(sh1.low_band, feats.low_band[expected_perm])
    # Shuffled preserves the multiset of values per column.
    assert torch.equal(torch.sort(sh1.rms).values, torch.sort(feats.rms).values)


def test_counterfactual_frozen_mean() -> None:
    audio = synthesize_probe("constant", CONFIG, seed=2)
    feats = extract_features(audio, CONFIG)
    fm = counterfactual_features(feats, "frozen-mean", seed=1)
    m = feats.matrix().mean(dim=0)
    n = feats.rms.numel()
    assert fm.matrix().shape == (n, 5)
    for col in range(5):
        assert torch.allclose(fm.matrix()[:, col], m[col].expand(n))
    # All rows identical.
    assert torch.allclose(fm.matrix()[0], fm.matrix()[-1])


def test_counterfactual_unknown_condition() -> None:
    audio = synthesize_probe("silence", CONFIG, seed=1)
    feats = extract_features(audio, CONFIG)
    with pytest.raises(ValueError):
        counterfactual_features(feats, "bogus", seed=1)  # type: ignore[arg-type]


def test_extract_features_too_short() -> None:
    short = torch.zeros(10, dtype=torch.float32)
    with pytest.raises(ValueError):
        extract_features(short, CONFIG)


def test_wav_header_and_length(tmp_path: Path) -> None:
    audio = synthesize_probe("constant", CONFIG, seed=1)
    # Scale into [-1, 1] safely (constant noise is already bounded near 0.2).
    out = tmp_path / "out.wav"
    write_pcm16_wav(out, audio, CONFIG.sample_rate)
    assert out.is_file()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == CONFIG.sample_rate
        n_frames = wf.getnframes()
        assert n_frames == audio.numel()
        raw = wf.readframes(n_frames)
        assert len(raw) == n_frames * 2
    # Parent must already exist.
    missing = tmp_path / "no_dir" / "x.wav"
    with pytest.raises(ValueError):
        write_pcm16_wav(missing, audio, CONFIG.sample_rate)


def test_wav_range_and_finite_checks(tmp_path: Path) -> None:
    out = tmp_path / "x.wav"
    bad = torch.tensor([0.0, 1.5, -0.5], dtype=torch.float32)
    with pytest.raises(ValueError):
        write_pcm16_wav(out, bad, CONFIG.sample_rate)
    nan = torch.tensor([0.0, float("nan"), 0.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        write_pcm16_wav(out, nan, CONFIG.sample_rate)


def test_audio_features_validation() -> None:
    z = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        AudioFeatures(
            rms=z,
            onset=z,
            spectral_flux=z,
            low_band=z,
            high_band=torch.zeros(3, dtype=torch.float32),
        )
    with pytest.raises(ValueError):
        AudioFeatures(
            rms=torch.zeros(0, dtype=torch.float32),
            onset=z,
            spectral_flux=z,
            low_band=z,
            high_band=z,
        )
    with pytest.raises(ValueError):
        AudioFeatures(
            rms=torch.full((4,), float("inf"), dtype=torch.float32),
            onset=z,
            spectral_flux=z,
            low_band=z,
            high_band=z,
        )
