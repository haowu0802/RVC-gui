"""Tests for source audio list sorting."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import KIND_SOURCE  # noqa: E402
from audio_scan import (  # noqa: E402
    AudioFileRow,
    sort_key_from_label,
    sort_label,
    sort_rows,
)


def _row(name: str, *, size: int = 0, mtime: int = 0, ctime: int = 0) -> AudioFileRow:
    return AudioFileRow(
        root="E:/audio",
        rel_path=name,
        name=name,
        path=f"E:/audio/{name}",
        size_bytes=size,
        mtime_ns=mtime,
        ctime_ns=ctime or mtime,
        kind=KIND_SOURCE,
    )


def test_sort_by_name_ascending() -> None:
    rows = [_row("b.wav"), _row("a.wav"), _row("c.wav")]
    out = sort_rows(rows, "name", descending=False)
    assert [r.name for r in out] == ["a.wav", "b.wav", "c.wav"]


def test_sort_by_size_descending() -> None:
    rows = [_row("a.wav", size=10), _row("b.wav", size=100), _row("c.wav", size=50)]
    out = sort_rows(rows, "size", descending=True)
    assert [r.name for r in out] == ["b.wav", "c.wav", "a.wav"]


def test_sort_by_mtime() -> None:
    rows = [_row("a.wav", mtime=3), _row("b.wav", mtime=1), _row("c.wav", mtime=2)]
    out = sort_rows(rows, "mtime", descending=False)
    assert [r.name for r in out] == ["b.wav", "c.wav", "a.wav"]


def test_sort_label_roundtrip() -> None:
    assert sort_key_from_label(sort_label("mtime")) == "mtime"
    assert sort_key_from_label("unknown") == "rel_path"
