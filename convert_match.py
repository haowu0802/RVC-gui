"""Match source audio to convert (merged) output files."""
from __future__ import annotations

from pathlib import Path

from audio_scan import AUDIO_EXTS, AudioFileRow
from audio_kind import is_convert_result_name
from source_match import stem_key
from stem_links import StemLink, get_convert_results, source_key, valid_audio_path


def convert_result_matches_source(result_path: str, source_path: str) -> bool:
    if not is_convert_result_name(Path(result_path).name):
        return False
    sk = stem_key(Path(source_path).name)
    if not sk:
        return False
    name_low = Path(result_path).name.lower()
    merged_sk = stem_key(Path(result_path).name)
    return sk in merged_sk or sk in name_low


def find_source_for_vocals(
    vocals_path: str,
    links: dict[str, StemLink] | None = None,
    rows: list[AudioFileRow] | None = None,
) -> str | None:
    if not vocals_path.strip():
        return None
    target = Path(vocals_path)
    if not target.is_file():
        return None
    resolved = str(target.resolve())

    for src, link in (links or {}).items():
        if link.vocals and Path(link.vocals).resolve() == target.resolve():
            if Path(src).is_file():
                return src

    vk = stem_key(target.name)
    for r in rows or []:
        if r.kind != "source":
            continue
        if stem_key(r.name) == vk and Path(r.path).is_file():
            return str(Path(r.path).resolve())
    return None


def _scan_dir_for_convert_results(
    directory: Path,
    source_path: str,
    found: dict[str, float],
) -> None:
    if not directory.is_dir():
        return
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for p in entries:
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        path_s = str(p.resolve())
        if not convert_result_matches_source(path_s, source_path):
            continue
        try:
            found[path_s] = p.stat().st_mtime
        except OSError:
            found[path_s] = 0.0


def _parents_for_sources(
    source_paths: list[str],
    links: dict[str, StemLink] | None,
) -> set[Path]:
    parents: set[Path] = set()
    for sp in source_paths:
        p = Path(sp)
        if p.is_file():
            parents.add(p.parent)
        link = (links or {}).get(source_key(sp))
        if not link:
            continue
        for attr in ("vocals", "instrumental"):
            raw = getattr(link, attr, None)
            if not raw:
                continue
            pp = Path(raw)
            if pp.is_file():
                parents.add(pp.parent)
    return parents


def _collect_merged_files(
    source_paths: list[str],
    links: dict[str, StemLink] | None,
    rows: list[AudioFileRow] | None,
) -> set[str]:
    """Gather merged output paths once (stem_links + scan rows + unique parent dirs)."""
    merged: set[str] = set()
    resolved = [
        str(Path(sp).resolve())
        for sp in source_paths
        if sp.strip() and Path(sp).is_file()
    ]
    for sp in resolved:
        for path_s in get_convert_results(sp, links or {}):
            p = Path(path_s)
            if p.is_file():
                merged.add(str(p.resolve()))
    for r in rows or []:
        if not is_convert_result_name(r.name):
            continue
        p = Path(r.path)
        if p.is_file():
            merged.add(str(p.resolve()))
    for parent in _parents_for_sources(resolved, links):
        try:
            for p in parent.iterdir():
                if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
                    continue
                if is_convert_result_name(p.name):
                    merged.add(str(p.resolve()))
        except OSError:
            continue
    return merged


def build_convert_count_map(
    source_paths: list[str],
    links: dict[str, StemLink] | None = None,
    rows: list[AudioFileRow] | None = None,
) -> dict[str, int]:
    """Map resolved source path -> convert result count (batch; scans each dir once)."""
    resolved = [
        str(Path(sp).resolve())
        for sp in source_paths
        if sp.strip() and Path(sp).is_file()
    ]
    if not resolved:
        return {}
    counts = {sp: 0 for sp in resolved}
    merged_files = _collect_merged_files(resolved, links, rows)
    if not merged_files:
        return counts
    source_sk = {sp: stem_key(Path(sp).name) for sp in resolved}
    found: dict[str, set[str]] = {sp: set() for sp in resolved}
    for mp in merged_files:
        name = Path(mp).name
        name_low = name.lower()
        msk = stem_key(name)
        for sp, sk in source_sk.items():
            if sk and (sk in msk or sk in name_low):
                found[sp].add(mp)
    return {sp: len(found[sp]) for sp in resolved}


def find_convert_results_for_source(
    source_path: str,
    links: dict[str, StemLink] | None = None,
    rows: list[AudioFileRow] | None = None,
    extra_dirs: list[str] | None = None,
) -> list[str]:
    """Return convert result paths for a source, newest first."""
    if not source_path.strip():
        return []
    source = Path(source_path)
    if not source.is_file():
        return []

    found: dict[str, float] = {}
    for path_s in get_convert_results(source_path, links or {}):
        p = Path(path_s)
        if not p.is_file():
            continue
        try:
            found[str(p.resolve())] = p.stat().st_mtime
        except OSError:
            found[str(p.resolve())] = 0.0

    dirs: list[Path] = []
    if source.parent.is_dir():
        dirs.append(source.parent)
    for raw in extra_dirs or []:
        raw = raw.strip()
        if raw:
            dirs.append(Path(raw).expanduser())

    link = (links or {}).get(source_key(source_path))
    if link:
        for attr in ("vocals", "instrumental"):
            p = getattr(link, attr, None)
            if p:
                parent = Path(p).parent
                if parent.is_dir():
                    dirs.append(parent)

    seen_dirs: set[str] = set()
    for directory in dirs:
        key = str(directory.resolve())
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        _scan_dir_for_convert_results(directory, source_path, found)

    for r in rows or []:
        if not is_convert_result_name(r.name):
            continue
        if convert_result_matches_source(r.path, source_path):
            p = Path(r.path)
            if p.is_file():
                try:
                    found[str(p.resolve())] = p.stat().st_mtime
                except OSError:
                    found[str(p.resolve())] = 0.0

    return sorted(found.keys(), key=lambda p: found[p], reverse=True)


def count_convert_results_for_source(
    source_path: str,
    links: dict[str, StemLink] | None = None,
    rows: list[AudioFileRow] | None = None,
) -> int:
    key = str(Path(source_path).resolve()) if source_path.strip() else ""
    if not key or not Path(key).is_file():
        return 0
    return build_convert_count_map([source_path], links, rows).get(key, 0)
