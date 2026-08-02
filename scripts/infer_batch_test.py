#!/usr/bin/env python3
"""Batch-infer one short test clip against multiple RVC .pth weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("RVC_ROOT") or os.getcwd()).resolve()


def _setup_env() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("RVC_CUDA_GRAPH", "0")
    os.environ.setdefault("weight_root", str(ROOT / "assets" / "weights"))
    os.environ.setdefault("index_root", str(ROOT / "logs"))
    os.environ.setdefault("outside_index_root", str(ROOT / "assets" / "indices"))
    os.environ.setdefault("rmvpe_root", str(ROOT / "assets" / "rmvpe"))
    tmp = ROOT / "TEMP"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp)


def _make_config():
    # Config() parses argv; keep only the program name while constructing it.
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        from configs.config import Config

        return Config()
    finally:
        sys.argv = saved


def _safe_stem(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVC multi-weight A/B infer test")
    p.add_argument("--input", required=True, help="Source / test audio path")
    p.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One or more .pth paths (or filenames under weight_root)",
    )
    p.add_argument("--index", default="", help="FAISS added_*.index path (optional)")
    p.add_argument(
        "--out_dir",
        default="",
        help="Output folder (default: <exp-ish>/infer_ab_test next to first weight)",
    )
    p.add_argument("--spk_id", type=int, default=0)
    p.add_argument("--f0_up_key", type=int, default=0)
    p.add_argument("--f0_method", default="rmvpe", choices=["rmvpe", "pm", "fcpe"])
    p.add_argument("--index_rate", type=float, default=0.75)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--rms_mix_rate", type=float, default=0.25)
    p.add_argument(
        "--resample_sr",
        type=int,
        default=0,
        help="0 = keep model tgt_sr; else resample output",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _setup_env()
    args = parse_args(argv)

    inp = Path(args.input).resolve()
    if not inp.is_file():
        print(f"[Infer][fail] input missing: {inp}", flush=True)
        return 1

    weight_paths: list[Path] = []
    for w in args.weights:
        p = Path(w)
        if not p.is_file():
            alt = Path(os.environ["weight_root"]) / w
            p = alt if alt.is_file() else p
        if not p.is_file():
            print(f"[Infer][fail] weight missing: {w}", flush=True)
            return 1
        weight_paths.append(p.resolve())

    index_path = args.index.strip().strip('"')
    if index_path and not Path(index_path).is_file():
        print(f"[Infer][fail] index missing: {index_path}", flush=True)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ROOT / "logs" / "_infer_ab_test"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Point weight_root at each file's parent as needed (usually assets/weights).
    # get_vc loads weight_root/sid — keep sid = basename, ensure parent is weight_root.
    parents = {p.parent for p in weight_paths}
    if len(parents) == 1:
        os.environ["weight_root"] = str(next(iter(parents)))

    print(f"[Infer] input={inp}", flush=True)
    print(f"[Infer] out_dir={out_dir}", flush=True)
    print(f"[Infer] weights={len(weight_paths)} index={index_path or '(none)'}", flush=True)

    import numpy as np
    import soundfile as sf
    from infer.vc.modules import VC

    config = _make_config()
    vc = VC(config)

    manifest: list[dict] = []
    # Copy / write reference note
    ref_out = out_dir / f"_source_{_safe_stem(inp.stem)}{inp.suffix}"
    if not ref_out.exists():
        data, sr = sf.read(str(inp))
        sf.write(str(ref_out), data, sr)

    ok = 0
    fail = 0
    for i, wpath in enumerate(weight_paths):
        sid = wpath.name
        # If this weight lives elsewhere, temporarily retarget weight_root
        if str(wpath.parent) != os.environ.get("weight_root"):
            os.environ["weight_root"] = str(wpath.parent)
        print(
            f"[Infer] ({i + 1}/{len(weight_paths)}) loading {sid}",
            flush=True,
        )
        try:
            vc.get_vc(sid)
            info, opt = vc.vc_single(
                args.spk_id,
                str(inp),
                args.f0_up_key,
                args.f0_method,
                index_path,
                args.index_rate,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
            )
            sr, audio = opt if opt is not None else (None, None)
            if audio is None or sr is None:
                fail += 1
                print(f"[Infer][fail] {sid}\n{info}", flush=True)
                continue
            out_name = f"{_safe_stem(Path(sid).stem)}.wav"
            out_path = out_dir / out_name
            audio = np.asarray(audio)
            sf.write(str(out_path), audio, int(sr))
            ok += 1
            item = {
                "weight": sid,
                "out": str(out_path),
                "sr": int(sr),
                "info": str(info).splitlines()[-1] if info else "",
            }
            manifest.append(item)
            print(
                f"[Infer][ok] {sid} -> {out_path.name} | sr={sr}",
                flush=True,
            )
        except Exception:
            fail += 1
            print(
                f"[Infer][fail] {sid}\n{traceback.format_exc()}",
                flush=True,
            )

    man_path = out_dir / "manifest.json"
    man_path.write_text(
        json.dumps(
            {
                "input": str(inp),
                "source_copy": str(ref_out),
                "index": index_path,
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[Infer] done | ok={ok} fail={fail} | manifest={man_path}",
        flush=True,
    )
    return 0 if fail == 0 and ok > 0 else (0 if ok > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
