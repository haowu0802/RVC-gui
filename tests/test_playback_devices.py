"""Tests for playback device label helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playback import (  # noqa: E402
    SYSTEM_DEFAULT_DEVICE,
    device_label_matches,
    list_output_devices,
    normalize_output_device,
    pick_default_device_label,
)


def test_normalize_system_default() -> None:
    assert normalize_output_device(SYSTEM_DEFAULT_DEVICE) is None
    assert normalize_output_device("") is None
    assert normalize_output_device("3: Speakers (Realtek)") == "Speakers (Realtek)"


def test_device_label_matches_fuzzy() -> None:
    saved = "Speakers (Realtek(R) Audio)"
    label = "5: Speakers (Realtek(R) Audio)"
    assert device_label_matches(saved, label)


def test_pick_default_prefers_system_default() -> None:
    devices = [SYSTEM_DEFAULT_DEVICE, "0: Speakers"]
    assert pick_default_device_label(devices) == SYSTEM_DEFAULT_DEVICE


def test_list_output_devices_includes_system_default() -> None:
    devices = list_output_devices()
    assert devices
    assert devices[0] == SYSTEM_DEFAULT_DEVICE
