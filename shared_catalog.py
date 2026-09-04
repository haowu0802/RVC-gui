"""Portable shared catalog: notes/scores by stem, kinds by filename (git-friendly)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_kind import classify_audio_kind, normalize_kind
from audio_scan import AudioFileRow, update_row_kind
from rvc_env import PACKAGE_DIR
from source_match import stem_key
from stem_links import (
    StemLink,
    clamp_score,
    set_source_note,
    set_source_score,
    source_key,
)

SHARED_CATALOG_PATH = PACKAGE_DIR / "shared_catalog.json"
CATALOG_VERSION = 2


@dataclass
class SharedCatalog:
    notes_by_stem: dict[str, str] = field(default_factory=dict)
    scores_by_stem: dict[str, int] = field(default_factory=dict)
    kinds_by_name: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CATALOG_VERSION,
            "notes_by_stem": dict(sorted(self.notes_by_stem.items())),
            "scores_by_stem": {
                k: v for k, v in sorted(self.scores_by_stem.items()) if v > 0
            },
            "kinds_by_name": dict(sorted(self.kinds_by_name.items())),
        }


def name_key(name: str) -> str:
    """Filename key for kind map (case-insensitive)."""
    return Path(name).name.lower()


def load_shared_catalog(path: Path | None = None) -> SharedCatalog:
    p = path or SHARED_CATALOG_PATH
    if not p.is_file():
        return SharedCatalog()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SharedCatalog()
    notes_raw = data.get("notes_by_stem") or {}
    scores_raw = data.get("scores_by_stem") or {}
    kinds_raw = data.get("kinds_by_name") or {}
    notes = {
        str(k).lower(): str(v)
        for k, v in notes_raw.items()
        if str(v).strip()
    }
    scores = {
        str(k).lower(): clamp_score(v)
        for k, v in scores_raw.items()
        if clamp_score(v) > 0
    }
    kinds = {
        name_key(str(k)): normalize_kind(str(v))
        for k, v in kinds_raw.items()
        if str(v).strip()
    }
    return SharedCatalog(
        notes_by_stem=notes,
        scores_by_stem=scores,
        kinds_by_name=kinds,
    )


def save_shared_catalog(catalog: SharedCatalog, path: Path | None = None) -> Path:
    p = path or SHARED_CATALOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(catalog.to_dict(), ensure_ascii=False, indent=2) + "\n"
    p.write_text(text, encoding="utf-8")
    return p


def set_note_for_stem(catalog: SharedCatalog, source_path: str, note: str) -> bool:
    key = stem_key(Path(source_path).name)
    if not key:
        return False
    cleaned = (note or "").strip()
    prev = catalog.notes_by_stem.get(key, "")
    if cleaned:
        if prev == cleaned:
            return False
        catalog.notes_by_stem[key] = cleaned
    else:
        if key not in catalog.notes_by_stem:
            return False
        del catalog.notes_by_stem[key]
    return True


def set_score_for_stem(catalog: SharedCatalog, source_path: str, score: int) -> bool:
    key = stem_key(Path(source_path).name)
    if not key:
        return False
    n = clamp_score(score)
    prev = catalog.scores_by_stem.get(key, 0)
    if n <= 0:
        if key not in catalog.scores_by_stem:
            return False
        del catalog.scores_by_stem[key]
        return True
    if prev == n:
        return False
    catalog.scores_by_stem[key] = n
    return True


def set_kind_for_name(catalog: SharedCatalog, file_name: str, kind: str) -> bool:
    key = name_key(file_name)
    if not key:
        return False
    nk = normalize_kind(kind)
    if catalog.kinds_by_name.get(key) == nk:
        return False
    catalog.kinds_by_name[key] = nk
    return True


def build_catalog_from_state(
    links: dict[str, StemLink],
    rows: list[AudioFileRow],
    *,
    only_kind_overrides: bool = True,
) -> SharedCatalog:
    """Snapshot notes/scores + kinds from live DB state into a portable catalog."""
    notes: dict[str, str] = {}
    scores: dict[str, int] = {}
    for path, link in links.items():
        sk = stem_key(Path(path).name)
        if not sk:
            continue
        note = (link.note or "").strip()
        if note:
            notes[sk] = note
        sc = clamp_score(link.score)
        if sc > 0:
            scores[sk] = sc

    kinds: dict[str, str] = {}
    for row in rows:
        key = name_key(row.name)
        if not key:
            continue
        if only_kind_overrides:
            auto = classify_audio_kind(row.name, row.rel_path)
            if normalize_kind(row.kind) == normalize_kind(auto):
                continue
        kinds[key] = normalize_kind(row.kind)
    return SharedCatalog(
        notes_by_stem=notes,
        scores_by_stem=scores,
        kinds_by_name=kinds,
    )


def _paths_by_stem(
    links: dict[str, StemLink],
    rows: list[AudioFileRow],
) -> dict[str, list[str]]:
    by_stem: dict[str, list[str]] = {}
    for path in links:
        sk = stem_key(Path(path).name)
        if sk:
            by_stem.setdefault(sk, []).append(path)
    for row in rows:
        sk = stem_key(row.name)
        if sk:
            by_stem.setdefault(sk, []).append(row.path)
    return by_stem


def apply_catalog_to_state(
    catalog: SharedCatalog,
    links: dict[str, StemLink],
    rows: list[AudioFileRow],
) -> tuple[dict[str, StemLink], list[AudioFileRow], int, int, int]:
    """Apply catalog. Returns (links, rows, notes_changed, scores_changed, kinds_changed)."""
    by_stem_paths = _paths_by_stem(links, rows)

    notes_n = 0
    for sk, note in catalog.notes_by_stem.items():
        seen: set[str] = set()
        for path in by_stem_paths.get(sk) or []:
            key = source_key(path)
            if key in seen:
                continue
            seen.add(key)
            prev = links.get(key)
            prev_note = (prev.note if prev and prev.note else "") or ""
            if prev_note == note:
                continue
            if set_source_note(links, path, note):
                notes_n += 1

    scores_n = 0
    for sk, score in catalog.scores_by_stem.items():
        seen: set[str] = set()
        for path in by_stem_paths.get(sk) or []:
            key = source_key(path)
            if key in seen:
                continue
            seen.add(key)
            prev = links.get(key)
            prev_score = clamp_score(prev.score if prev else 0)
            if prev_score == clamp_score(score):
                continue
            if set_source_score(links, path, score):
                scores_n += 1

    kinds_n = 0
    new_rows = list(rows)
    for row in rows:
        key = name_key(row.name)
        want = catalog.kinds_by_name.get(key)
        if want is None or normalize_kind(row.kind) == want:
            continue
        new_rows = update_row_kind(new_rows, row.path, want)
        kinds_n += 1

    return links, new_rows, notes_n, scores_n, kinds_n
