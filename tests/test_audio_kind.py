"""Tests for audio kind heuristics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import (  # noqa: E402
    KIND_CLONE,
    KIND_DFN3,
    KIND_SEP_INST,
    KIND_SEP_VOCALS,
    KIND_SOURCE,
    classify_audio_kind,
)


def test_source_default():
    assert classify_audio_kind("song.wav", "cx/song.wav") == KIND_SOURCE


def test_sep_vocals():
    assert (
        classify_audio_kind(
            "song_(vocals)_vocals_mel_band_roformer.flac",
            "out/song_(vocals)_vocals_mel_band_roformer.flac",
        )
        == KIND_SEP_VOCALS
    )


def test_sep_inst():
    assert classify_audio_kind("song_(instrumental).flac", "out/x.flac") == KIND_SEP_INST


def test_dfn3():
    assert classify_audio_kind("song_(dfn3).wav", "cx/song_(dfn3).wav") == KIND_DFN3


def test_clone_infer_ab():
    assert (
        classify_audio_kind(
            "my_model_e30.wav",
            "logs/exp/infer_ab_test/my_model_e30.wav",
        )
        == KIND_CLONE
    )


def test_source_copy_in_infer_ab():
    assert (
        classify_audio_kind(
            "_source_test_clip.wav",
            "logs/exp/infer_ab_test/_source_test_clip.wav",
        )
        == KIND_SOURCE
    )


def test_clone_infer_long():
    assert (
        classify_audio_kind("merged.wav", "logs/exp/infer_long/merged.wav") == KIND_CLONE
    )
