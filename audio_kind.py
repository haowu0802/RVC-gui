"""Heuristic audio kind from filename + relative path (no audio analysis)."""
from __future__ import annotations

import re

# Internal kind keys
KIND_SOURCE = "source"
KIND_SEP_VOCALS = "sep_vocals"
KIND_SEP_INST = "sep_inst"
KIND_DFN3 = "dfn3"
KIND_CLONE = "clone"

KIND_DISPLAY = {
    KIND_SOURCE: "source",
    KIND_SEP_VOCALS: "vocals",
    KIND_SEP_INST: "inst",
    KIND_DFN3: "dfn3",
    KIND_CLONE: "clone",
}

KIND_FILTER_VALUES = ["all"] + list(KIND_DISPLAY.values())

_CLONE_PATH_TOKENS = (
    "infer_ab_test",
    "infer_long",
    "/infer_ab/",
    "\\infer_ab\\",
    "infer_ab/",
)

_RE_CHECKPOINT_WAV = re.compile(r"_e\d+\.wav$", re.I)


def _in_clone_output_dir(path_low: str) -> bool:
    for tok in _CLONE_PATH_TOKENS:
        if tok.replace("\\", "/") in path_low:
            return True
    return False


def classify_audio_kind(name: str, rel_path: str = "") -> str:
    """Classify a file as source / sep vocals / sep inst / dfn3 / clone."""
    low = name.lower()
    path_low = (rel_path or "").lower().replace("\\", "/")

    # Infer A/B keeps a copy of the input for reference.
    if low.startswith("_source_"):
        return KIND_SOURCE

    if _in_clone_output_dir(path_low):
        return KIND_CLONE

    if "(dfn3)" in low:
        return KIND_DFN3

    # audio-separator / MelBand naming
    if "(instrumental)" in low:
        return KIND_SEP_INST
    if "(other)" in low and "mel_band" in low:
        return KIND_SEP_INST
    if "(vocals)" in low or "vocals_mel_band_roformer" in low:
        return KIND_SEP_VOCALS
    if "no_vocals" in low or "_instrumental." in low:
        return KIND_SEP_INST

    # Weight-style name outside infer dirs (e.g. copied outputs)
    if _RE_CHECKPOINT_WAV.search(low):
        return KIND_CLONE

    return KIND_SOURCE


def kind_matches_filter(kind: str, filter_kind: str) -> bool:
    fk = (filter_kind or "all").strip().lower()
    if fk in ("", "all"):
        return True
    display = KIND_DISPLAY.get(kind, kind)
    return display.lower() == fk or kind.lower() == fk
