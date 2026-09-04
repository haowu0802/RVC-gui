"""Tests for shared_catalog (stem notes + filename kinds)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_kind import KIND_SEP_VOCALS, KIND_SOURCE  # noqa: E402
from audio_scan import AudioFileRow  # noqa: E402
from shared_catalog import (  # noqa: E402
    SharedCatalog,
    apply_catalog_to_state,
    build_catalog_from_state,
    load_shared_catalog,
    name_key,
    save_shared_catalog,
    set_kind_for_name,
    set_note_for_stem,
)
from stem_links import StemLink, set_source_note  # noqa: E402


def test_score_stars_and_click() -> None:
    from stem_links import format_score_stars, score_from_click_x, set_source_score

    assert format_score_stars(0) == "☆☆☆"
    assert format_score_stars(2) == "★★☆"
    assert format_score_stars(9) == "★★★"
    assert score_from_click_x(0, 90) == 1
    assert score_from_click_x(45, 90) == 2
    assert score_from_click_x(89, 90) == 3

    links = {}
    assert set_source_score(links, r"C:\a\clip.wav", 3)
    assert links[list(links)[0]].score == 3


def test_roundtrip_file() -> None:
    cat = SharedCatalog(
        notes_by_stem={"song": "keep"},
        kinds_by_name={"song.mp3": KIND_SOURCE},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shared_catalog.json"
        save_shared_catalog(cat, path)
        loaded = load_shared_catalog(path)
        assert loaded.notes_by_stem["song"] == "keep"
        assert loaded.kinds_by_name["song.mp3"] == KIND_SOURCE


def test_note_by_stem_and_kind_by_name() -> None:
    cat = SharedCatalog()
    assert set_note_for_stem(cat, r"E:\_haud\_song.mp3", "hello")
    # Leading underscore stripped by stem_key
    assert "song" in cat.notes_by_stem or "_song" in cat.notes_by_stem or any(
        "song" in k for k in cat.notes_by_stem
    )
    assert set_kind_for_name(cat, "Song_(Vocals)_x.flac", KIND_SEP_VOCALS)
    assert name_key("Song_(Vocals)_x.flac") in cat.kinds_by_name


def test_build_and_apply() -> None:
    links: dict[str, StemLink] = {}
    set_source_note(links, r"C:\audio\clip_a.wav", "good take")
    rows = [
        AudioFileRow(
            root=r"C:\audio",
            rel_path="clip_a.wav",
            name="clip_a.wav",
            path=r"C:\audio\clip_a.wav",
            size_bytes=1,
            mtime_ns=1,
            ctime_ns=1,
            kind=KIND_SOURCE,
        ),
        AudioFileRow(
            root=r"C:\audio",
            rel_path="other.wav",
            name="other.wav",
            path=r"C:\audio\other.wav",
            size_bytes=1,
            mtime_ns=1,
            ctime_ns=1,
            kind=KIND_SOURCE,
        ),
    ]
    # Force a kind override
    rows[1] = AudioFileRow(
        root=rows[1].root,
        rel_path=rows[1].rel_path,
        name=rows[1].name,
        path=rows[1].path,
        size_bytes=1,
        mtime_ns=1,
        ctime_ns=1,
        kind=KIND_SEP_VOCALS,
    )
    cat = build_catalog_from_state(links, rows, only_kind_overrides=True)
    assert "clip_a" in cat.notes_by_stem
    assert cat.notes_by_stem["clip_a"] == "good take"
    assert name_key("other.wav") in cat.kinds_by_name

    links2: dict[str, StemLink] = {}
    rows2 = [
        AudioFileRow(
            root=r"D:\pool",
            rel_path="clip_a.wav",
            name="clip_a.wav",
            path=r"D:\pool\clip_a.wav",
            size_bytes=1,
            mtime_ns=1,
            ctime_ns=1,
            kind=KIND_SOURCE,
        ),
        AudioFileRow(
            root=r"D:\pool",
            rel_path="other.wav",
            name="other.wav",
            path=r"D:\pool\other.wav",
            size_bytes=1,
            mtime_ns=1,
            ctime_ns=1,
            kind=KIND_SOURCE,
        ),
    ]
    links2, rows2, n_notes, n_scores, n_kinds = apply_catalog_to_state(cat, links2, rows2)
    assert n_notes >= 1
    assert n_kinds >= 1
    assert links2[r"D:\pool\clip_a.wav"].note == "good take" or any(
        (lnk.note or "") == "good take" for lnk in links2.values()
    )
    other = next(r for r in rows2 if r.name == "other.wav")
    assert other.kind == KIND_SEP_VOCALS
