"""Match source audio files to separated vocals / instrumental stems."""
from __future__ import annotations

import re
from pathlib import Path

from audio_kind import KIND_SEP_INST, KIND_SEP_VOCALS, classify_audio_kind
from audio_scan import AUDIO_EXTS, AudioFileRow
from stem_links import StemLink, get_linked_stems, valid_audio_path

_STEM_SUFFIXES = (
    r"_\(dfn3\)$",
    r"_\(Vocals\)$",
    r"_\(vocals\)_vocals_mel_band_roformer$",
    r"_\(other\)_vocals_mel_band_roformer$",
    r"_\(vocals\)$",
    r"_\(instrumental\)$",
    r"_\(other\)$",
    r"_\(merged\)$",
    r"_\(orig\)$",
)


def stem_key(name: str) -> str:
    """Normalize filename to a clip identity stem.

    Leading underscores are stripped because audio-separator drops them when
    writing ``_(vocals)_…`` / ``_(other)_…`` beside a source named ``_foo.mp3``.
    """
    stem = Path(name).stem
    stem = re.sub(r"^1_", "", stem)
    for pat in _STEM_SUFFIXES:
        stem = re.sub(pat, "", stem, flags=re.I)
    stem = stem.lstrip("_")
    return stem.lower()


def find_separated_for_source(
    source_path: str,
    rows: list[AudioFileRow] | None = None,
    links: dict[str, StemLink] | None = None,
) -> tuple[str | None, str | None]:
    """Return (vocals_path, instrumental_path) for a source file, if found."""
    source = Path(source_path)
    if not source_path.strip():
        return None, None
    key = stem_key(source.name)
    vocals: str | None = None
    inst: str | None = None

    def assign(path_s: str, kind: str) -> None:
        nonlocal vocals, inst
        p = Path(path_s)
        if not p.is_file():
            return
        if stem_key(p.name) != key:
            return
        if kind == KIND_SEP_VOCALS and vocals is None:
            vocals = str(p.resolve())
        elif kind == KIND_SEP_INST and inst is None:
            inst = str(p.resolve())

    linked_v, linked_i = get_linked_stems(source_path, links or {})
    vocals = linked_v
    inst = linked_i

    for r in rows or []:
        assign(r.path, r.kind)

    parent = source.parent
    if source.is_file() and parent.is_dir():
        for p in parent.iterdir():
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            kind = classify_audio_kind(p.name, p.name)
            if kind in (KIND_SEP_VOCALS, KIND_SEP_INST):
                assign(str(p), kind)

    return valid_audio_path(vocals), valid_audio_path(inst)
