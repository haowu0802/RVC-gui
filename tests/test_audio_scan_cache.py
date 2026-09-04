"""Tests for audio scan cache persistence."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import KIND_OTHER, KIND_RESULT, KIND_SOURCE  # noqa: E402
from audio_scan import (  # noqa: E402
    AudioFileRow,
    load_scan_cache,
    merge_rescan_rows,
    reconcile_auto_kinds,
    save_scan_cache,
)


def _row(
    name: str,
    *,
    kind: str,
    path: str | None = None,
    duration_sec: float = 0.0,
) -> AudioFileRow:
    path = path or f"E:/audio/{name}"
    return AudioFileRow(
        root="E:/audio",
        rel_path=name,
        name=name,
        path=path,
        size_bytes=1234,
        mtime_ns=99,
        ctime_ns=88,
        kind=kind,
        duration_sec=duration_sec,
    )


def test_scan_cache_roundtrip() -> None:
    rows = [_row("song.wav", kind=KIND_SOURCE, duration_sec=42.5)]
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        save_scan_cache(db_path, rows)
        loaded = load_scan_cache(db_path)
        assert len(loaded) == 1
        assert loaded[0].name == "song.wav"
        assert loaded[0].kind == KIND_SOURCE
        assert loaded[0].duration_sec == 42.5


def test_load_cache_reconciles_stale_merged_as_source() -> None:
    """Cached Convert outputs wrongly marked source must upgrade on load."""
    name = "cx_model_clip_(Merged).flac"
    rows = [_row(name, kind=KIND_SOURCE, path=f"E:/_haud/_res/{name}")]
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        save_scan_cache(db_path, rows)
        loaded = load_scan_cache(db_path)
        assert loaded[0].kind == KIND_RESULT


def test_reconcile_auto_kinds_only_upgrades_source() -> None:
    merged = "out_(Merged).flac"
    rows = reconcile_auto_kinds(
        [
            _row(merged, kind=KIND_SOURCE),
            _row(merged, kind=KIND_OTHER, path="E:/audio/other_" + merged),
            _row("plain.mp3", kind=KIND_SOURCE),
        ]
    )
    assert rows[0].kind == KIND_RESULT
    assert rows[1].kind == KIND_OTHER  # manual/non-source kept
    assert rows[2].kind == KIND_SOURCE


def test_merge_rescan_applies_new_merged_heuristic() -> None:
    name = "model_song_(Merged).flac"
    existing = [_row(name, kind=KIND_SOURCE, duration_sec=12.0)]
    scanned = [_row(name, kind=KIND_RESULT, duration_sec=0.0)]
    merged = merge_rescan_rows(existing, scanned)
    assert merged[0].kind == KIND_RESULT
    assert merged[0].duration_sec == 12.0


def test_merge_rescan_uses_fresh_kind_not_stale_cache() -> None:
    """Stale kind in cache must not block heuristic upgrades on rescan."""
    name = "plain.mp3"
    existing = [_row(name, kind=KIND_OTHER, duration_sec=3.0)]
    scanned = [_row(name, kind=KIND_SOURCE, duration_sec=0.0)]
    merged = merge_rescan_rows(existing, scanned)
    assert merged[0].kind == KIND_SOURCE
    assert merged[0].duration_sec == 3.0
