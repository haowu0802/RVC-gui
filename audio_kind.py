"""Heuristic audio kind from filename + relative path (no audio analysis)."""
from __future__ import annotations

import re

# Internal kind keys
KIND_SOURCE = "source"
KIND_SEP_VOCALS = "sep_vocals"
KIND_SEP_INST = "sep_inst"
KIND_RESULT = "result"
KIND_OTHER = "other"

# Deprecated aliases (legacy DB / tests)
KIND_CLONE = KIND_RESULT
KIND_DFN3 = KIND_OTHER

KIND_DISPLAY = {
    KIND_SOURCE: "source audio",
    KIND_SEP_VOCALS: "vocal",
    KIND_SEP_INST: "instrumental",
    KIND_RESULT: "result",
    KIND_OTHER: "other",
}

KIND_FILTER_VALUES = ["all"] + list(KIND_DISPLAY.values())
KIND_EDIT_VALUES = list(KIND_DISPLAY.values())

_KIND_KEYS = frozenset(KIND_DISPLAY)

_LEGACY_KIND_KEYS = {
    "clone": KIND_RESULT,
    "dfn3": KIND_OTHER,
    "vocals": KIND_SEP_VOCALS,
    "inst": KIND_SEP_INST,
    "instrumental": KIND_SEP_INST,
}

_DISPLAY_TO_KIND = {label.lower(): key for key, label in KIND_DISPLAY.items()}

_CLONE_PATH_TOKENS = (
    "infer_ab_test",
    "infer_long",
    "/infer_ab/",
    "\\infer_ab\\",
    "infer_ab/",
)

_RE_CHECKPOINT_WAV = re.compile(r"_e\d+\.wav$", re.I)


def normalize_kind(kind: str) -> str:
    k = (kind or "").strip().lower()
    if not k:
        return KIND_OTHER
    if k in _KIND_KEYS:
        return k
    if k in _LEGACY_KIND_KEYS:
        return _LEGACY_KIND_KEYS[k]
    if k in _DISPLAY_TO_KIND:
        return _DISPLAY_TO_KIND[k]
    return KIND_OTHER


def kind_label(kind: str) -> str:
    return KIND_DISPLAY.get(normalize_kind(kind), KIND_DISPLAY[KIND_OTHER])


def kind_from_label(label: str) -> str:
    return normalize_kind(label)


def _in_clone_output_dir(path_low: str) -> bool:
    for tok in _CLONE_PATH_TOKENS:
        if tok.replace("\\", "/") in path_low:
            return True
    return False


def classify_audio_kind(name: str, rel_path: str = "") -> str:
    """Classify a file as source / vocal / instrumental / result / other."""
    low = name.lower()
    path_low = (rel_path or "").lower().replace("\\", "/")

    if low.startswith("_source_"):
        return KIND_SOURCE

    if _in_clone_output_dir(path_low):
        return KIND_RESULT

    if "(dfn3)" in low:
        return KIND_OTHER

    if "(instrumental)" in low:
        return KIND_SEP_INST
    if "(other)" in low and "mel_band" in low:
        return KIND_SEP_INST
    if "(vocals)" in low or "vocals_mel_band_roformer" in low:
        return KIND_SEP_VOCALS
    if "no_vocals" in low or "_instrumental." in low:
        return KIND_SEP_INST

    if _RE_CHECKPOINT_WAV.search(low):
        return KIND_RESULT

    return KIND_SOURCE


def kind_matches_filter(kind: str, filter_kind: str) -> bool:
    fk = (filter_kind or "all").strip().lower()
    if fk in ("", "all"):
        return True
    nk = normalize_kind(kind)
    if fk == nk:
        return True
    display = kind_label(nk)
    return display.lower() == fk
