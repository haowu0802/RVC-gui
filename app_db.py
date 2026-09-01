"""SQLite persistence for RVC-gui (settings, stem links, scan cache)."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from rvc_env import PACKAGE_DIR

DB_PATH = PACKAGE_DIR / "rvc_gui.db"
SCHEMA_VERSION = 1

_LEGACY_SETTINGS = PACKAGE_DIR / "settings.json"
_LEGACY_STEM_LINKS = PACKAGE_DIR / "stem_links.json"
_LEGACY_SCAN_CACHE = PACKAGE_DIR / "audio_scan_cache.json"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_setting (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS favorite_path (
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, path)
);

CREATE TABLE IF NOT EXISTS audio_scan_root (
    sort_order INTEGER NOT NULL,
    path       TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS source_link (
    source_path        TEXT PRIMARY KEY,
    vocals_path        TEXT,
    instrumental_path  TEXT,
    note               TEXT,
    updated_ns         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS convert_result (
    source_path  TEXT NOT NULL,
    result_path  TEXT NOT NULL,
    sort_order   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source_path, result_path)
);

CREATE TABLE IF NOT EXISTS audio_scan_row (
    path        TEXT PRIMARY KEY,
    root        TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    name        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    ctime_ns    INTEGER NOT NULL,
    kind        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_convert_result_source ON convert_result(source_path);
CREATE INDEX IF NOT EXISTS idx_audio_scan_row_kind ON audio_scan_row(kind);
"""


def default_db_path(db_path: Path | None = None) -> Path:
    return db_path if db_path is not None else DB_PATH


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = default_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn, path)
    return conn


def resolve_conn(db_or_path: Path | sqlite3.Connection | None) -> sqlite3.Connection:
    if isinstance(db_or_path, sqlite3.Connection):
        return db_or_path
    return connect(db_or_path)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def init_db(conn: sqlite3.Connection, db_path: Path | None = None) -> None:
    conn.executescript(_SCHEMA_SQL)
    if schema_version(conn) < SCHEMA_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    migrate_legacy_json(conn, db_path)


def _json_load(raw: str | None, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def get_setting(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute(
        "SELECT value FROM app_setting WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    return _json_load(row["value"], default)


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO app_setting(key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False)),
    )


def load_settings_map(conn: sqlite3.Connection) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in conn.execute("SELECT key, value FROM app_setting"):
        out[str(row["key"])] = _json_load(row["value"])
    out["favorite_models"] = list_favorites(conn, "model")
    out["favorite_experiments"] = list_favorites(conn, "experiment")
    out["audio_scan_roots"] = list_scan_roots(conn)
    return out


def save_settings_map(conn: sqlite3.Connection, data: dict[str, Any]) -> None:
    reserved = {"favorite_models", "favorite_experiments", "audio_scan_roots"}
    with transaction(conn):
        for key, value in data.items():
            if key in reserved:
                continue
            set_setting(conn, key, value)
        if "favorite_models" in data:
            set_favorites(conn, "model", list(data["favorite_models"] or []))
        if "favorite_experiments" in data:
            set_favorites(conn, "experiment", list(data["favorite_experiments"] or []))
        if "audio_scan_roots" in data:
            set_scan_roots(conn, list(data["audio_scan_roots"] or []))


def list_favorites(conn: sqlite3.Connection, kind: str) -> list[str]:
    rows = conn.execute(
        "SELECT path FROM favorite_path WHERE kind = ? ORDER BY sort_order, path",
        (kind,),
    ).fetchall()
    return [str(r["path"]) for r in rows]


def set_favorites(conn: sqlite3.Connection, kind: str, paths: list[str]) -> None:
    conn.execute("DELETE FROM favorite_path WHERE kind = ?", (kind,))
    for i, raw in enumerate(paths):
        p = str(raw).strip()
        if p:
            conn.execute(
                "INSERT INTO favorite_path(kind, path, sort_order) VALUES (?, ?, ?)",
                (kind, p, i),
            )


def list_scan_roots(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT path FROM audio_scan_root ORDER BY sort_order, path"
    ).fetchall()
    return [str(r["path"]) for r in rows]


def set_scan_roots(conn: sqlite3.Connection, paths: list[str]) -> None:
    conn.execute("DELETE FROM audio_scan_root")
    for i, raw in enumerate(paths):
        p = str(raw).strip()
        if p:
            conn.execute(
                "INSERT INTO audio_scan_root(sort_order, path) VALUES (?, ?)",
                (i, p),
            )


def load_stem_links_db(conn: sqlite3.Connection) -> dict[str, Any]:
    from stem_links import StemLink

    out: dict[str, StemLink] = {}
    for row in conn.execute("SELECT * FROM source_link"):
        source = str(row["source_path"])
        results = [
            str(r["result_path"])
            for r in conn.execute(
                "SELECT result_path FROM convert_result WHERE source_path = ? ORDER BY sort_order",
                (source,),
            )
        ]
        out[source] = StemLink(
            vocals=row["vocals_path"],
            instrumental=row["instrumental_path"],
            convert_results=results or None,
            note=row["note"],
            updated_ns=int(row["updated_ns"] or 0),
        )
    return out


def save_stem_links_db(conn: sqlite3.Connection, links: dict[str, Any]) -> None:
    from stem_links import StemLink

    with transaction(conn):
        conn.execute("DELETE FROM convert_result")
        conn.execute("DELETE FROM source_link")
        for key, link in links.items():
            if not isinstance(link, StemLink):
                continue
            conn.execute(
                """
                INSERT INTO source_link(
                    source_path, vocals_path, instrumental_path, note, updated_ns
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(key),
                    link.vocals,
                    link.instrumental,
                    link.note,
                    int(link.updated_ns or time.time_ns()),
                ),
            )
            for i, result in enumerate(link.convert_results or []):
                if result:
                    conn.execute(
                        """
                        INSERT INTO convert_result(source_path, result_path, sort_order)
                        VALUES (?, ?, ?)
                        """,
                        (str(key), str(result), i),
                    )


def load_scan_cache_db(conn: sqlite3.Connection) -> list[Any]:
    from audio_scan import AudioFileRow, row_from_dict

    rows: list[AudioFileRow] = []
    for row in conn.execute("SELECT * FROM audio_scan_row ORDER BY root, rel_path"):
        item = row_from_dict(
            {
                "root": row["root"],
                "rel_path": row["rel_path"],
                "name": row["name"],
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "mtime_ns": row["mtime_ns"],
                "ctime_ns": row["ctime_ns"],
                "kind": row["kind"],
            }
        )
        if item is not None:
            rows.append(item)
    return rows


def save_scan_cache_db(conn: sqlite3.Connection, rows: list[Any]) -> None:
    from audio_scan import AudioFileRow, row_to_dict

    with transaction(conn):
        conn.execute("DELETE FROM audio_scan_row")
        for row in rows:
            if not isinstance(row, AudioFileRow):
                continue
            d = row_to_dict(row)
            conn.execute(
                """
                INSERT INTO audio_scan_row(
                    path, root, rel_path, name, size_bytes, mtime_ns, ctime_ns, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["path"],
                    d["root"],
                    d["rel_path"],
                    d["name"],
                    d["size_bytes"],
                    d["mtime_ns"],
                    d["ctime_ns"],
                    d["kind"],
                ),
            )


def migrate_legacy_json(conn: sqlite3.Connection, db_path: Path | None = None) -> None:
    if conn.execute(
        "SELECT 1 FROM meta WHERE key = 'legacy_import_done'"
    ).fetchone():
        return

    if default_db_path(db_path).resolve() != DB_PATH.resolve():
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_import_done', ?)",
            ("0",),
        )
        conn.commit()
        return

    imported = False
    if _LEGACY_SETTINGS.is_file():
        try:
            data = json.loads(_LEGACY_SETTINGS.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                save_settings_map(conn, data)
                imported = True
        except Exception:
            pass

    if _LEGACY_STEM_LINKS.is_file():
        try:
            payload = json.loads(_LEGACY_STEM_LINKS.read_text(encoding="utf-8"))
            if payload.get("version") in (1, 2):
                from stem_links import StemLink

                links: dict[str, StemLink] = {}
                for key, item in (payload.get("links") or {}).items():
                    if not isinstance(item, dict):
                        continue
                    raw_results = item.get("convert_results") or []
                    convert_results = (
                        [str(p) for p in raw_results]
                        if isinstance(raw_results, list)
                        else []
                    )
                    links[str(key)] = StemLink(
                        vocals=item.get("vocals") or None,
                        instrumental=item.get("instrumental") or None,
                        convert_results=convert_results,
                        note=item.get("note") or None,
                        updated_ns=int(item.get("updated_ns") or 0),
                    )
                if links:
                    save_stem_links_db(conn, links)
                    imported = True
        except Exception:
            pass

    if _LEGACY_SCAN_CACHE.is_file():
        try:
            payload = json.loads(_LEGACY_SCAN_CACHE.read_text(encoding="utf-8"))
            if payload.get("version") == 1:
                from audio_scan import row_from_dict

                rows = []
                for item in payload.get("rows", []):
                    if isinstance(item, dict):
                        row = row_from_dict(item)
                        if row is not None:
                            rows.append(row)
                if rows:
                    save_scan_cache_db(conn, rows)
                    imported = True
        except Exception:
            pass

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_import_done', ?)",
        ("1" if imported else "0",),
    )
    conn.commit()
