"""Tests for SQLite app database."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_db import connect, load_settings_map, save_settings_map  # noqa: E402


def test_settings_roundtrip() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = connect(db_path)
        save_settings_map(
            conn,
            {
                "rvc_root": "D:/rvc",
                "favorite_models": ["D:/models/a.pth"],
                "favorite_experiments": ["D:/logs/exp1"],
                "audio_scan_roots": ["D:/audio"],
            },
        )
        data = load_settings_map(conn)
        assert data["rvc_root"] == "D:/rvc"
        assert data["favorite_models"] == ["D:/models/a.pth"]
        assert data["favorite_experiments"] == ["D:/logs/exp1"]
        assert data["audio_scan_roots"] == ["D:/audio"]


def test_legacy_settings_import() -> None:
    with TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        settings = pkg / "settings.json"
        settings.write_text(
            json.dumps({"rvc_root": "E:/rvc", "favorite_models": ["E:/m.pth"]}),
            encoding="utf-8",
        )
        import app_db as mod

        old_db = mod.DB_PATH
        old_settings = mod._LEGACY_SETTINGS
        try:
            mod.DB_PATH = pkg / "rvc_gui.db"
            mod._LEGACY_SETTINGS = settings
            conn = connect(mod.DB_PATH)
            data = load_settings_map(conn)
            assert data["rvc_root"] == "E:/rvc"
            assert data["favorite_models"] == ["E:/m.pth"]
        finally:
            mod.DB_PATH = old_db
            mod._LEGACY_SETTINGS = old_settings
