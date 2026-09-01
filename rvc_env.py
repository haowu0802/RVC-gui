"""Resolve the RVC installation root used by this GUI package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DB_PATH = PACKAGE_DIR / "rvc_gui.db"
SCRIPTS_DIR = PACKAGE_DIR / "scripts"


def env_rvc_root() -> Path | None:
    raw = (os.environ.get("RVC_ROOT") or "").strip().strip('"')
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p if p.is_dir() else None


def looks_like_rvc_root(path: Path) -> bool:
    return (path / "train" / "train.py").is_file() and (
        (path / "runtime" / "python.exe").is_file()
        or (path / "runtime" / "python").is_file()
        or True
    )


def resolve_rvc_root(configured: str | None = None) -> Path:
    """
    Resolution order:
      1) explicit configured path (from settings UI)
      2) RVC_ROOT environment variable
      3) parent of this package if it looks like an RVC tree
    """
    if configured:
        p = Path(configured).expanduser().resolve()
        if p.is_dir() and looks_like_rvc_root(p):
            return p
        raise FileNotFoundError(f"Invalid RVC root: {p}")

    env = env_rvc_root()
    if env and looks_like_rvc_root(env):
        return env

    parent = PACKAGE_DIR.parent
    if looks_like_rvc_root(parent):
        return parent

    raise FileNotFoundError(
        "RVC root not set. Set it in the GUI (Settings) or export RVC_ROOT."
    )


def rvc_python(rvc_root: Path) -> Path:
    win = rvc_root / "runtime" / "python.exe"
    if win.is_file():
        return win
    nix = rvc_root / "runtime" / "python"
    if nix.is_file():
        return nix
    # Fall back to the interpreter running the GUI
    return Path(sys.executable)


def rvc_pythonw(rvc_root: Path) -> Path:
    win = rvc_root / "runtime" / "pythonw.exe"
    if win.is_file():
        return win
    return rvc_python(rvc_root)


def prepare_rvc_process_env(rvc_root: Path) -> dict:
    env = os.environ.copy()
    env["RVC_ROOT"] = str(rvc_root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("RVC_CUDA_GRAPH", "0")
    env.setdefault("weight_root", str(rvc_root / "assets" / "weights"))
    env.setdefault("index_root", str(rvc_root / "logs"))
    env.setdefault("outside_index_root", str(rvc_root / "assets" / "indices"))
    env.setdefault("rmvpe_root", str(rvc_root / "assets" / "rmvpe"))
    tmp = rvc_root / "TEMP"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    # Ensure RVC package imports resolve
    py_path = env.get("PYTHONPATH", "")
    root_s = str(rvc_root)
    env["PYTHONPATH"] = root_s if not py_path else root_s + os.pathsep + py_path
    return env


def find_index_for_model(model_path_or_name: str, rvc_root: Path) -> str:
    """Locate an added_*.index matching a weight filename (WebUI-compatible heuristics)."""
    import re

    model_stem = Path(model_path_or_name).stem
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return ""

    roots = [
        rvc_root / "assets" / "indices",
        rvc_root / "logs",
    ]
    candidates: list[tuple[tuple, str]] = []
    for index_root in roots:
        if not index_root.is_dir():
            continue
        for path in index_root.rglob("*.index"):
            name = path.name
            if "trained" in name.lower():
                continue
            index_stem = path.stem
            lower_index = index_stem.lower()
            lower_experiment = experiment_name.lower()
            standard_match = (
                lower_index.startswith(lower_experiment + "_added_")
                or ("_" + lower_experiment + "_v1") in lower_index
                or ("_" + lower_experiment + "_v2") in lower_index
            )
            exact_model_match = model_stem.lower() in lower_index
            if not (standard_match or exact_model_match):
                continue
            score = (
                0 if standard_match else 1,
                0 if index_root == roots[0] else 1,
                -path.stat().st_mtime,
                str(path).lower(),
            )
            candidates.append((score, str(path.resolve())))
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item[0])[1]