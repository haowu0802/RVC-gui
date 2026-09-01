"""Tests for source / separated stem matching."""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import KIND_SEP_INST, KIND_SEP_VOCALS  # noqa: E402
from audio_scan import AudioFileRow  # noqa: E402
from source_match import find_separated_for_source, stem_key  # noqa: E402


def test_stem_key_source_and_vocals() -> None:
    assert stem_key("song.mp3") == stem_key("song_(vocals)_vocals_mel_band_roformer.flac")
    assert stem_key("song.mp3") == stem_key("song_(other)_vocals_mel_band_roformer.flac")


def test_find_from_scan_rows() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "song.mp3"
        vocals = root / "song_(vocals)_vocals_mel_band_roformer.flac"
        inst = root / "song_(other)_vocals_mel_band_roformer.flac"
        source.write_bytes(b"a")
        vocals.write_bytes(b"b")
        inst.write_bytes(b"c")
        rows = [
            AudioFileRow(
                root=str(root),
                rel_path="song.mp3",
                name="song.mp3",
                path=str(source),
                size_bytes=1,
                mtime_ns=1,
                ctime_ns=1,
                kind="source",
            ),
            AudioFileRow(
                root=str(root),
                rel_path="song_(vocals)_vocals_mel_band_roformer.flac",
                name="song_(vocals)_vocals_mel_band_roformer.flac",
                path=str(vocals),
                size_bytes=1,
                mtime_ns=1,
                ctime_ns=1,
                kind=KIND_SEP_VOCALS,
            ),
            AudioFileRow(
                root=str(root),
                rel_path="song_(other)_vocals_mel_band_roformer.flac",
                name="song_(other)_vocals_mel_band_roformer.flac",
                path=str(inst),
                size_bytes=1,
                mtime_ns=1,
                ctime_ns=1,
                kind=KIND_SEP_INST,
            ),
        ]
        v, i = find_separated_for_source(str(source), rows)
        assert v is not None and "(vocals)" in v
        assert i is not None and "(other)" in i
