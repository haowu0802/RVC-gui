"""Tests for audio scan cache persistence."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import KIND_SOURCE  # noqa: E402
from audio_scan import (  # noqa: E402
    AudioFileRow,
    load_scan_cache,
    save_scan_cache,
)


def test_scan_cache_roundtrip() -> None:
    rows = [
        AudioFileRow(
            root="E:/audio",
            rel_path="song.wav",
            name="song.wav",
            path="E:/audio/song.wav",
            size_bytes=1234,
            mtime_ns=99,
            ctime_ns=88,
            kind=KIND_SOURCE,
        )
    ]
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        save_scan_cache(db_path, rows)
        loaded = load_scan_cache(db_path)
        assert len(loaded) == 1
        assert loaded[0].name == "song.wav"
        assert loaded[0].kind == KIND_SOURCE
