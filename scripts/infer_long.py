#!/usr/bin/env python3
"""Long-form RVC inference: split -> convert chunks -> crossfade merge.

Compatible with RVC WebUI layouts that expose infer.vc.modules.VC
(e.g. 2026-07 packages). Requires RVC_ROOT (cwd/env) for imports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _bootstrap_rvc() -> Path:
    root = Path(os.environ.get("RVC_ROOT") or os.getcwd()).resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


PARSER_DEFAULTS = {
    "f0up_key": 0,
    "f0method": "rmvpe",
    "index_rate": 0.0,
    "resample_sr": 0,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "chunk_sec": 120.0,
    "overlap_sec": 0.30,
    "spk_id": 0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RVC long-audio CLI: split, infer, crossfade merge."
    )
    p.add_argument("--input_path", required=True)
    p.add_argument("--opt_path", required=True)
    p.add_argument("--model_name", default="", help="Filename under assets/weights")
    p.add_argument("--index_path", default="")
    p.add_argument("--f0up_key", type=int, default=PARSER_DEFAULTS["f0up_key"])
    p.add_argument("--f0method", default=PARSER_DEFAULTS["f0method"])
    p.add_argument("--index_rate", type=float, default=PARSER_DEFAULTS["index_rate"])
    p.add_argument("--resample_sr", type=int, default=PARSER_DEFAULTS["resample_sr"])
    p.add_argument("--rms_mix_rate", type=float, default=PARSER_DEFAULTS["rms_mix_rate"])
    p.add_argument("--protect", type=float, default=PARSER_DEFAULTS["protect"])
    p.add_argument("--spk_id", type=int, default=PARSER_DEFAULTS["spk_id"])
    p.add_argument("--chunk_sec", type=float, default=PARSER_DEFAULTS["chunk_sec"])
    p.add_argument("--overlap_sec", type=float, default=PARSER_DEFAULTS["overlap_sec"])
    p.add_argument("--temp_dir", default="")
    p.add_argument("--keep_temp", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument("--is_half", type=str2bool, default=None)
    p.add_argument("--preset_json", default="")
    args = p.parse_args(argv)
    # Config() also parses argv; keep only program name for it.
    sys.argv = [sys.argv[0]]
    return args


def read_preset(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_preset(args: argparse.Namespace, preset: dict) -> argparse.Namespace:
    if not preset:
        return args
    for key, default_value in PARSER_DEFAULTS.items():
        current = getattr(args, key, None)
        if current == default_value and key in preset:
            setattr(args, key, preset[key])
    if not args.model_name and "model_name" in preset:
        args.model_name = preset["model_name"]
    if not args.index_path and "index_path" in preset:
        args.index_path = preset["index_path"]
    return args


def ffprobe_duration_sec(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_chunk_to_wav(
    input_path: str, out_wav: str, start_sec: float, dur_sec: float
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{dur_sec:.6f}",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    subprocess.run(cmd, check=True)


def to_float_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        maxv = max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
        return arr.astype(np.float32) / float(maxv)
    return arr.astype(np.float32)


def crossfade_concat(chunks: list[np.ndarray], overlap_samples: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].copy()
    for cur in chunks[1:]:
        if overlap_samples <= 0 or len(out) < overlap_samples or len(cur) < overlap_samples:
            out = np.concatenate([out, cur], axis=0)
            continue
        fade_out = np.linspace(1.0, 0.0, overlap_samples, endpoint=False, dtype=np.float32)
        fade_in = 1.0 - fade_out
        out[-overlap_samples:] = (
            out[-overlap_samples:] * fade_out + cur[:overlap_samples] * fade_in
        )
        out = np.concatenate([out, cur[overlap_samples:]], axis=0)
    return out


def main(argv: list[str] | None = None) -> int:
    rvc_root = _bootstrap_rvc()
    args = parse_args(argv)
    args = apply_preset(args, read_preset(args.preset_json))

    if args.chunk_sec <= 0:
        raise ValueError("--chunk_sec must be > 0")
    if args.overlap_sec < 0:
        raise ValueError("--overlap_sec must be >= 0")
    if args.overlap_sec >= args.chunk_sec:
        raise ValueError("--overlap_sec must be smaller than --chunk_sec")
    if not args.model_name:
        raise ValueError("--model_name is required (CLI or preset_json)")

    from configs.config import Config
    from infer.vc.modules import VC

    config = Config()
    if args.device:
        config.device = args.device
    if args.is_half is not None:
        config.is_half = args.is_half

    os.environ.setdefault("weight_root", str(rvc_root / "assets" / "weights"))
    vc = VC(config)
    vc.get_vc(args.model_name)

    input_path = str(Path(args.input_path).resolve())
    output_path = Path(args.opt_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration_sec(input_path)
    step_sec = args.chunk_sec - args.overlap_sec

    if args.temp_dir:
        temp_root = Path(args.temp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="rvc_long_"))
        cleanup_temp = True

    print(
        f"[info] duration_sec={duration:.2f} chunk_sec={args.chunk_sec} "
        f"overlap_sec={args.overlap_sec}",
        flush=True,
    )
    print(f"[info] temp_dir={temp_root}", flush=True)

    chunk_outputs: list[np.ndarray] = []
    target_sr = None
    idx = 0
    start = 0.0
    try:
        while start < duration:
            this_dur = min(args.chunk_sec, duration - start)
            chunk_in = temp_root / f"chunk_in_{idx:04d}.wav"
            extract_chunk_to_wav(input_path, str(chunk_in), start, this_dur)
            print(
                f"[chunk {idx:04d}] start={start:.2f}s dur={this_dur:.2f}s",
                flush=True,
            )
            info, wav_opt = vc.vc_single(
                args.spk_id,
                str(chunk_in),
                args.f0up_key,
                args.f0method,
                args.index_path,
                args.index_rate,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
            )
            if wav_opt is None or wav_opt[0] is None or wav_opt[1] is None:
                raise RuntimeError(f"Chunk inference failed at {idx}: {info}")
            sr, audio = wav_opt
            if target_sr is None:
                target_sr = int(sr)
            elif int(sr) != target_sr:
                raise RuntimeError(
                    f"Inconsistent sample rate across chunks: {sr} vs {target_sr}"
                )
            chunk_outputs.append(to_float_audio(audio))
            idx += 1
            start += step_sec

        assert target_sr is not None
        overlap_samples = int(round(args.overlap_sec * target_sr))
        merged = crossfade_concat(chunk_outputs, overlap_samples)
        merged = np.clip(merged, -1.0, 1.0)
        wavfile.write(
            str(output_path), target_sr, (merged * 32767.0).astype(np.int16)
        )
        print(f"[done] wrote {output_path}", flush=True)
    finally:
        if cleanup_temp and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
