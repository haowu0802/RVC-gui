"""Tests for subprocess progress line parsing (tqdm / RVC helpers)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep in sync with gui.py patterns (avoid importing tkinter via gui).
_RE_PROGRESS_FRAC = re.compile(
    r"(?:"
    r"\[progress\]\s*(\d+)\s*/\s*(\d+)"
    r"|进度[：:]\s*(\d+)\s*/\s*(\d+)"
    r"|Write progress:\s*(\d+)\s*/\s*(\d+)"
    r"|写入进度[：:]\s*(\d+)\s*/\s*(\d+)"
    r"|\[Infer\]\s*\((\d+)\s*/\s*(\d+)\)"
    r"|\b(\d+)\s*/\s*(\d+)\s*\["
    r")",
    re.IGNORECASE,
)
_RE_TQDM_PCT = re.compile(r"(?:^|\s)(\d+)%\|")


def _frac_pct(line: str) -> float | None:
    m = _RE_PROGRESS_FRAC.search(line)
    if not m:
        return None
    groups = [g for g in m.groups() if g is not None]
    if len(groups) < 2:
        return None
    cur, total = int(groups[0]), int(groups[1])
    return 100.0 * cur / total if total > 0 else None


def _tqdm_pct(line: str) -> float | None:
    m = _RE_TQDM_PCT.search(line)
    return float(m.group(1)) if m else None


def test_tqdm_bar_fraction() -> None:
    line = " 96%|██████████▋| 192/200 [03:00<00:07, 1.07it/s]"
    assert _frac_pct(line) == 96.0
    assert _tqdm_pct(line) == 96.0


def test_infer_fraction_still_matches() -> None:
    assert _frac_pct("[Infer] (3/10)") == 30.0


def test_carriage_return_chunks() -> None:
    blob = " 50%|██| 100/200 [01:00<01:00, 1.00it/s]\r 96%|██| 192/200 [03:00<00:07, 1.07it/s]"
    last = None
    for line in blob.replace("\r", "\n").splitlines():
        last = _frac_pct(line) or _tqdm_pct(line)
    assert last == 96.0
