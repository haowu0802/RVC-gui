"""Tests for audio kind heuristics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import (  # noqa: E402
    KIND_OTHER,
    KIND_RESULT,
    KIND_SEP_INST,
    KIND_SEP_VOCALS,
    KIND_SOURCE,
    classify_audio_kind,
    kind_from_label,
    kind_label,
    normalize_kind,
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


def test_dfn3_is_other():
    assert classify_audio_kind("song_(dfn3).wav", "cx/song_(dfn3).wav") == KIND_OTHER


def test_result_infer_ab():
    assert (
        classify_audio_kind(
            "my_model_e30.wav",
            "logs/exp/infer_ab_test/my_model_e30.wav",
        )
        == KIND_RESULT
    )


def test_source_copy_in_infer_ab():
    assert (
        classify_audio_kind(
            "_source_test_clip.wav",
            "logs/exp/infer_ab_test/_source_test_clip.wav",
        )
        == KIND_SOURCE
    )


def test_result_infer_long():
    assert (
        classify_audio_kind("merged.wav", "logs/exp/infer_long/merged.wav") == KIND_RESULT
    )


def test_normalize_legacy_kinds():
    assert normalize_kind("clone") == KIND_RESULT
    assert normalize_kind("dfn3") == KIND_OTHER
    assert normalize_kind("vocals") == KIND_SEP_VOCALS


def test_kind_label_roundtrip():
    assert kind_from_label(kind_label(KIND_SOURCE)) == KIND_SOURCE
    assert kind_from_label("instrumental") == KIND_SEP_INST
