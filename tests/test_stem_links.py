"""Tests for stem link persistence."""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_match import find_separated_for_source  # noqa: E402
from stem_links import (  # noqa: E402
    load_stem_links,
    save_stem_links,
    upsert_stem_link,
)


def test_stem_links_roundtrip() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        source = Path(tmp) / "song.mp3"
        vocals = Path(tmp) / "song_(vocals)_vocals_mel_band_roformer.flac"
        inst = Path(tmp) / "song_(other)_vocals_mel_band_roformer.flac"
        source.write_bytes(b"a")
        vocals.write_bytes(b"b")
        inst.write_bytes(b"c")

        links = load_stem_links(db_path)
        upsert_stem_link(
            links,
            str(source),
            vocals=str(vocals),
            instrumental=str(inst),
        )
        save_stem_links(links, db_path)

        loaded = load_stem_links(db_path)
        v, i = find_separated_for_source(str(source), links=loaded)
        assert v == str(vocals.resolve())
        assert i == str(inst.resolve())


def test_source_note_roundtrip() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        source = Path(tmp) / "song.mp3"
        source.write_bytes(b"a")
        links = load_stem_links(db_path)
        from stem_links import get_source_note, set_source_note

        set_source_note(links, str(source), "test note")
        save_stem_links(links, db_path)
        loaded = load_stem_links(db_path)
        assert get_source_note(str(source), loaded) == "test note"
