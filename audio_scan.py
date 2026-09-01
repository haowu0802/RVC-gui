"""Scan configured root folders for audio files (recursive)."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from audio_kind import KIND_DISPLAY, classify_audio_kind, kind_matches_filter

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

SRC_SORT_LABELS: dict[str, str] = {
    "rel_path": "Relative path",
    "name": "File name",
    "size": "Size",
    "mtime": "Modified",
    "ctime": "Created",
    "root": "Root folder",
    "path": "Full path",
}
SRC_SORT_KEYS = tuple(SRC_SORT_LABELS.keys())

_SKIP_DIR_NAMES = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}


@dataclass(frozen=True)
class AudioFileRow:
    root: str
    rel_path: str
    name: str
    path: str
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    kind: str

    @property
    def kind_label(self) -> str:
        return KIND_DISPLAY.get(self.kind, self.kind)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def _stat_time_ns(st: os.stat_result, primary: str, fallback: str) -> int:
    for suffix in ("_ns", ""):
        for base in (primary, fallback):
            attr = f"st_{base}{suffix}"
            if hasattr(st, attr):
                val = getattr(st, attr)
                if suffix:
                    return int(val)
                return int(float(val) * 1e9)
    return 0


def sort_label(sort_by: str) -> str:
    return SRC_SORT_LABELS.get(sort_by, SRC_SORT_LABELS["rel_path"])


def sort_key_from_label(label: str) -> str:
    for key, text in SRC_SORT_LABELS.items():
        if text == label:
            return key
    low = (label or "").strip().lower()
    if low in SRC_SORT_LABELS:
        return low
    return "rel_path"


def sort_rows(
    rows: list[AudioFileRow],
    sort_by: str = "rel_path",
    *,
    descending: bool = False,
) -> list[AudioFileRow]:
    key_name = sort_by if sort_by in SRC_SORT_LABELS else "rel_path"
    if key_name == "size":
        key_fn = lambda r: r.size_bytes
    elif key_name == "mtime":
        key_fn = lambda r: r.mtime_ns
    elif key_name == "ctime":
        key_fn = lambda r: r.ctime_ns
    elif key_name == "name":
        key_fn = lambda r: r.name.lower()
    elif key_name == "root":
        key_fn = lambda r: r.root.lower()
    elif key_name == "path":
        key_fn = lambda r: r.path.lower()
    else:
        key_fn = lambda r: (r.root.lower(), r.rel_path.lower())
    return sorted(rows, key=key_fn, reverse=descending)


def _should_skip_dir(name: str) -> bool:
    low = name.lower()
    if low in _SKIP_DIR_NAMES:
        return True
    if name.startswith(".") and low not in {".", ".."}:
        return True
    return False


def scan_audio_roots(roots: list[str]) -> list[AudioFileRow]:
    """Walk each root recursively and collect audio files."""
    out: list[AudioFileRow] = []
    seen: set[str] = set()
    for root_s in roots:
        root = Path(root_s).expanduser()
        if not root.is_dir():
            continue
        root_resolved = str(root.resolve())
        for dirpath, dirnames, filenames in os_walk_sorted(root):
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in AUDIO_EXTS:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                path_s = str(p.resolve())
                if path_s in seen:
                    continue
                seen.add(path_s)
                try:
                    rel = str(p.relative_to(root))
                except ValueError:
                    rel = fn
                rel_norm = rel.replace("\\", "/")
                mtime_ns = _stat_time_ns(st, "mtime", "mtime")
                ctime_ns = _stat_time_ns(st, "birthtime", "ctime") or mtime_ns
                out.append(
                    AudioFileRow(
                        root=root_resolved,
                        rel_path=rel_norm,
                        name=fn,
                        path=path_s,
                        size_bytes=int(st.st_size),
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        kind=classify_audio_kind(fn, rel_norm),
                    )
                )
    out.sort(key=lambda r: (r.root.lower(), r.rel_path.lower()))
    return out


def os_walk_sorted(root: Path):
    """os.walk with sorted dirnames for stable scans."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _e: None):
        dirnames.sort(key=str.lower)
        filenames.sort(key=str.lower)
        yield dirpath, dirnames, filenames


def filter_rows(
    rows: list[AudioFileRow],
    query: str = "",
    *,
    kind_filter: str = "all",
) -> list[AudioFileRow]:
    q = (query or "").strip().lower()
    out: list[AudioFileRow] = []
    for r in rows:
        if not kind_matches_filter(r.kind, kind_filter):
            continue
        if q:
            hay = f"{r.rel_path} {r.name} {r.root} {r.kind_label}".lower()
            if q not in hay:
                continue
        out.append(r)
    return out


def count_by_kind(rows: list[AudioFileRow]) -> dict[str, int]:
    counts: dict[str, int] = {v: 0 for v in KIND_DISPLAY.values()}
    for r in rows:
        label = r.kind_label
        counts[label] = counts.get(label, 0) + 1
    return counts


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def row_to_dict(row: AudioFileRow) -> dict[str, object]:
    return {
        "root": row.root,
        "rel_path": row.rel_path,
        "name": row.name,
        "path": row.path,
        "size_bytes": row.size_bytes,
        "mtime_ns": row.mtime_ns,
        "ctime_ns": row.ctime_ns,
        "kind": row.kind,
    }


def row_from_dict(data: dict[str, object]) -> AudioFileRow | None:
    try:
        return AudioFileRow(
            root=str(data["root"]),
            rel_path=str(data["rel_path"]),
            name=str(data["name"]),
            path=str(data["path"]),
            size_bytes=int(data["size_bytes"]),  # type: ignore[arg-type]
            mtime_ns=int(data["mtime_ns"]),  # type: ignore[arg-type]
            ctime_ns=int(data.get("ctime_ns", data.get("mtime_ns", 0))),  # type: ignore[arg-type]
            kind=str(data["kind"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_scan_cache(
    db_or_path: Path | sqlite3.Connection | None = None,
) -> list[AudioFileRow]:
    from app_db import load_scan_cache_db, resolve_conn

    conn = resolve_conn(db_or_path)
    return load_scan_cache_db(conn)


def save_scan_cache(
    db_or_path: Path | sqlite3.Connection | None,
    rows: list[AudioFileRow],
) -> None:
    from app_db import resolve_conn, save_scan_cache_db

    conn = resolve_conn(db_or_path)
    save_scan_cache_db(conn, rows)
    conn.commit()
