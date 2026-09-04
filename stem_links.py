"""Persist associations between source audio and separated stems."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from app_db import load_stem_links_db, resolve_conn, save_stem_links_db

SCORE_MAX = 3


@dataclass
class StemLink:
    vocals: str | None = None
    instrumental: str | None = None
    convert_results: list[str] | None = None
    note: str | None = None
    score: int = 0
    updated_ns: int = 0


def clamp_score(score: int | None) -> int:
    try:
        n = int(score or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(SCORE_MAX, n))


def format_score_stars(score: int | None) -> str:
    """Display ★/☆ for 0–3 (clickable in the Source Audio tree)."""
    n = clamp_score(score)
    return ("★" * n) + ("☆" * (SCORE_MAX - n))


def score_from_click_x(rel_x: int, width: int, *, max_score: int = SCORE_MAX) -> int:
    """Map a click x within the score cell to 1..max_score."""
    if width <= 0:
        return 1
    slot = int(rel_x * max_score / width) + 1
    return max(1, min(max_score, slot))


def source_key(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def valid_audio_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return str(p.resolve()) if p.is_file() else None


def load_stem_links(
    db_or_path: Path | sqlite3.Connection | None = None,
) -> dict[str, StemLink]:
    conn = resolve_conn(db_or_path)
    return load_stem_links_db(conn)


def save_stem_links(
    links: dict[str, StemLink],
    db_or_path: Path | sqlite3.Connection | None = None,
) -> None:
    conn = resolve_conn(db_or_path)
    save_stem_links_db(conn, links)
    conn.commit()


def get_linked_stems(
    source: str,
    links: dict[str, StemLink],
) -> tuple[str | None, str | None]:
    link = links.get(source_key(source))
    if not link:
        return None, None
    return valid_audio_path(link.vocals), valid_audio_path(link.instrumental)


def _replace_link(cur: StemLink | None, **kwargs) -> StemLink:
    base = cur or StemLink()
    return StemLink(
        vocals=kwargs["vocals"] if "vocals" in kwargs else base.vocals,
        instrumental=(
            kwargs["instrumental"] if "instrumental" in kwargs else base.instrumental
        ),
        convert_results=(
            kwargs["convert_results"]
            if "convert_results" in kwargs
            else (list(base.convert_results) if base.convert_results else None)
        ),
        note=kwargs["note"] if "note" in kwargs else base.note,
        score=clamp_score(kwargs["score"] if "score" in kwargs else base.score),
        updated_ns=int(kwargs.get("updated_ns") or time.time_ns()),
    )


def upsert_stem_link(
    links: dict[str, StemLink],
    source: str,
    *,
    vocals: str | None = None,
    instrumental: str | None = None,
) -> bool:
    key = source_key(source)
    new_v = valid_audio_path(vocals)
    new_i = valid_audio_path(instrumental)
    cur = links.get(key)
    merged_v = new_v or (valid_audio_path(cur.vocals) if cur else None)
    merged_i = new_i or (valid_audio_path(cur.instrumental) if cur else None)
    if not merged_v and not merged_i:
        return False
    prev_v = valid_audio_path(cur.vocals) if cur else None
    prev_i = valid_audio_path(cur.instrumental) if cur else None
    if merged_v == prev_v and merged_i == prev_i:
        return False
    links[key] = _replace_link(
        cur,
        vocals=merged_v,
        instrumental=merged_i,
    )
    return True


def get_convert_results(source: str, links: dict[str, StemLink]) -> list[str]:
    link = links.get(source_key(source))
    if not link or not link.convert_results:
        return []
    out: list[str] = []
    for path_s in link.convert_results:
        valid = valid_audio_path(path_s)
        if valid:
            out.append(valid)
    return out


def add_convert_result(
    links: dict[str, StemLink],
    source: str,
    result_path: str,
) -> bool:
    valid = valid_audio_path(result_path)
    if not valid:
        return False
    key = source_key(source)
    cur = links.get(key)
    prev = list(cur.convert_results or []) if cur else []
    if valid in prev:
        return False
    links[key] = _replace_link(
        cur,
        vocals=valid_audio_path(cur.vocals) if cur else None,
        instrumental=valid_audio_path(cur.instrumental) if cur else None,
        convert_results=prev + [valid],
    )
    return True


def get_source_note(source: str, links: dict[str, StemLink]) -> str:
    link = links.get(source_key(source))
    if not link or not link.note:
        return ""
    return link.note


def set_source_note(
    links: dict[str, StemLink],
    source: str,
    note: str,
) -> bool:
    key = source_key(source)
    cur = links.get(key)
    new_note = note
    prev = cur.note if cur and cur.note else ""
    if new_note == prev:
        return False
    links[key] = _replace_link(
        cur,
        vocals=valid_audio_path(cur.vocals) if cur else None,
        instrumental=valid_audio_path(cur.instrumental) if cur else None,
        note=new_note or None,
    )
    return True


def get_source_score(source: str, links: dict[str, StemLink]) -> int:
    link = links.get(source_key(source))
    if not link:
        return 0
    return clamp_score(link.score)


def set_source_score(
    links: dict[str, StemLink],
    source: str,
    score: int,
) -> bool:
    key = source_key(source)
    cur = links.get(key)
    new_score = clamp_score(score)
    prev = clamp_score(cur.score if cur else 0)
    if new_score == prev and cur is not None:
        return False
    links[key] = _replace_link(
        cur,
        vocals=valid_audio_path(cur.vocals) if cur else None,
        instrumental=valid_audio_path(cur.instrumental) if cur else None,
        score=new_score,
    )
    return True
