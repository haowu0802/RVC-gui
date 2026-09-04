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


def test_merged_convert_output_is_result():
    """GUI Convert writes ``{model}_{source}_(Merged).ext`` — must not be source."""
    cases = [
        "cx_all-cold-noise-badsing__SJ_NTR_SLUT66 女主偷情被抓_GJ_NTR_(Merged).flac",
        "1_B004 纯享之骚货粗口--小美_(Merged).flac",
        "model_song_(Merged).flac",
        "song_(merged).wav",
        "Song_(MERGED).flac",
        # Older MelBand-style replace path without model prefix
        "186 song__(Merged)_vocals_mel_band_roformer.flac",
    ]
    for name in cases:
        assert classify_audio_kind(name, f"_res/{name}") == KIND_RESULT, name
        assert classify_audio_kind(name, name) == KIND_RESULT, name


def test_merged_not_confused_with_plain_source():
    assert classify_audio_kind("song_merged_take.mp3", "a/song_merged_take.mp3") == KIND_SOURCE
    assert classify_audio_kind("merge_me.wav", "a/merge_me.wav") == KIND_SOURCE


def test_vocals_still_wins_over_unrelated_tokens():
    assert (
        classify_audio_kind(
            "song_(vocals)_vocals_mel_band_roformer.flac",
            "_res/song_(vocals)_vocals_mel_band_roformer.flac",
        )
        == KIND_SEP_VOCALS
    )


def test_normalize_legacy_kinds():
    assert normalize_kind("clone") == KIND_RESULT
    assert normalize_kind("dfn3") == KIND_OTHER
    assert normalize_kind("vocals") == KIND_SEP_VOCALS


def test_kind_label_roundtrip():
    assert kind_from_label(kind_label(KIND_SOURCE)) == KIND_SOURCE
    assert kind_from_label("instrumental") == KIND_SEP_INST


def test_source_filter_excludes_merged():
    from audio_kind import kind_matches_filter

    kind = classify_audio_kind(
        "cx_model_clip_(Merged).flac",
        r"E:\_haud\_res\cx_model_clip_(Merged).flac",
    )
    assert kind == KIND_RESULT
    assert not kind_matches_filter(kind, "source")
    assert kind_matches_filter(kind, "result")
