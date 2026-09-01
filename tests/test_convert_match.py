"""Tests for convert result matching."""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_match import (  # noqa: E402
    build_convert_count_map,
    convert_result_matches_source,
    find_convert_results_for_source,
    find_source_for_vocals,
)
from source_match import stem_key  # noqa: E402
from stem_links import StemLink, add_convert_result, load_stem_links, save_stem_links  # noqa: E402


def test_convert_result_matches_source() -> None:
    assert convert_result_matches_source(
        "model_song_(vocals)_mel(Merged).flac",
        "song.mp3",
    )
    assert not convert_result_matches_source("song.mp3", "song.mp3")


def test_find_source_for_vocals_from_links() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "song.mp3"
        vocals = root / "song_(vocals)_vocals_mel_band_roformer.flac"
        source.write_bytes(b"a")
        vocals.write_bytes(b"b")
        links = {
            str(source.resolve()): StemLink(vocals=str(vocals.resolve())),
        }
        assert find_source_for_vocals(str(vocals), links) == str(source.resolve())


def test_find_convert_results_persisted_and_on_disk() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "song.mp3"
        merged = root / "model_song_(vocals)_mel(Merged).flac"
        source.write_bytes(b"a")
        merged.write_bytes(b"b")
        cache = Path(tmp) / "test.db"
        links = load_stem_links(cache)
        add_convert_result(links, str(source), str(merged))
        save_stem_links(links, cache)
        loaded = load_stem_links(cache)
        found = find_convert_results_for_source(str(source), loaded, extra_dirs=[str(root)])
        assert str(merged.resolve()) in found


def test_build_convert_count_map_batch() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        s1 = root / "a.mp3"
        s2 = root / "b.mp3"
        m1 = root / "model_a_(vocals)_mel(Merged).flac"
        m2 = root / "model_b_(vocals)_mel(Merged).flac"
        for p in (s1, s2, m1, m2):
            p.write_bytes(b"x")
        counts = build_convert_count_map([str(s1), str(s2)], rows=[])
        assert counts[str(s1.resolve())] == 1
        assert counts[str(s2.resolve())] == 1
