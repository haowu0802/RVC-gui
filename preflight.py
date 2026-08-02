#!/usr/bin/env python3
"""Import check before launching the GUI (used by launch_gui.bat)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gui  # noqa: F401
import rvc_env  # noqa: F401

print("preflight ok", flush=True)
