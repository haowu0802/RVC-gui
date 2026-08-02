#!/usr/bin/env python3
"""Flexible RVC preprocess with all knobs exposed (CLI)."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import traceback
from dataclasses import asdict, dataclass

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

# Resolve RVC install root (must provide imports: infer.*, train.*, configs.*)
ROOT = os.environ.get("RVC_ROOT") or os.getcwd()
ROOT = os.path.abspath(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ["RVC_AUDIO_FORCE_CPU"] = "1"
from infer.audio import load_audio  # noqa: E402
from tools.progress import should_report  # noqa: E402
from train.dataset.slicer2 import Slicer  # noqa: E402


@dataclass
class PreprocessConfig:
    inp_root: str
    exp_dir: str
    sr: int = 48000
    n_p: int = 8
    noparallel: bool = False
    per: float = 3.5
    overlap: float = 0.3
    highpass_hz: float = 48.0
    enable_highpass: bool = True
    threshold: int = -42
    min_length: int = 1500
    min_interval: int = 400
    hop_size: int = 15
    max_sil_kept: int = 500
    norm_max: float = 0.9
    norm_alpha: float = 0.75
    peak_reject: float = 2.5


_LOG = None


def println(msg: str) -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


class PreProcess:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.slicer = Slicer(
            sr=cfg.sr,
            threshold=cfg.threshold,
            min_length=cfg.min_length,
            min_interval=cfg.min_interval,
            hop_size=cfg.hop_size,
            max_sil_kept=cfg.max_sil_kept,
        )
        self.sr = cfg.sr
        self.per = cfg.per
        self.overlap = cfg.overlap
        self.tail = self.per + self.overlap
        self.max = cfg.norm_max
        self.alpha = cfg.norm_alpha
        self.peak_reject = cfg.peak_reject
        self.enable_highpass = cfg.enable_highpass
        hz = max(1.0, min(cfg.highpass_hz, cfg.sr * 0.45))
        self.bh, self.ah = signal.butter(N=5, Wn=hz, btype="high", fs=self.sr)
        self.exp_dir = cfg.exp_dir
        self.gt_wavs_dir = os.path.join(cfg.exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(cfg.exp_dir, "1_16k_wavs")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1) -> bool:
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > self.peak_reject:
            println(f"[skip] bad peak {idx0}_{idx1} peak={tmp_max}")
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        audio_16k = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        ).astype(np.float32)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),
            16000,
            audio_16k,
        )
        return True

    def pipeline(self, path, idx0, total) -> bool:
        try:
            audio = load_audio(path, self.sr)
            if self.enable_highpass:
                audio = signal.lfilter(self.bh, self.ah, audio)
            idx1 = 0
            for chunk in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(chunk[start:]) > self.tail * self.sr:
                        tmp_audio = chunk[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = chunk[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            if should_report(idx0, total):
                println(f"[progress] {idx0 + 1}/{total} | {os.path.basename(path)}")
            return True
        except Exception:
            println(f"[fail] {path}\n{traceback.format_exc()}")
            return False

    def pipeline_mp(self, infos):
        ok = bad = 0
        for path, idx0, total in infos:
            if self.pipeline(path, idx0, total):
                ok += 1
            else:
                bad += 1
        if infos:
            println(f"[worker done] ok={ok} fail={bad}")

    def run(self):
        names = sorted(
            n
            for n in os.listdir(self.cfg.inp_root)
            if os.path.isfile(os.path.join(self.cfg.inp_root, n))
        )
        total = len(names)
        infos = [
            (os.path.join(self.cfg.inp_root, name), idx, total)
            for idx, name in enumerate(names)
        ]
        workers = max(1, min(self.cfg.n_p, max(total, 1)))
        println(f"[start] files={total} workers={workers} cfg={asdict(self.cfg)}")
        if self.cfg.noparallel or workers == 1:
            for i in range(workers):
                self.pipeline_mp(infos[i::workers])
        else:
            ps = []
            for i in range(workers):
                p = multiprocessing.Process(
                    target=self.pipeline_mp, args=(infos[i::workers],)
                )
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
        n_gt = len(os.listdir(self.gt_wavs_dir))
        println(f"[done] slices_in_0_gt_wavs={n_gt}")


def parse_args(argv=None) -> PreprocessConfig:
    p = argparse.ArgumentParser(description="RVC flexible preprocess")
    p.add_argument("--inp_root", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--sr", type=int, default=48000, choices=[32000, 40000, 48000])
    p.add_argument("--n_p", type=int, default=8)
    p.add_argument("--noparallel", action="store_true")
    p.add_argument("--per", type=float, default=3.5)
    p.add_argument("--overlap", type=float, default=0.3)
    p.add_argument("--highpass_hz", type=float, default=48.0)
    p.add_argument("--no_highpass", action="store_true")
    p.add_argument("--threshold", type=int, default=-42)
    p.add_argument("--min_length", type=int, default=1500)
    p.add_argument("--min_interval", type=int, default=400)
    p.add_argument("--hop_size", type=int, default=15)
    p.add_argument("--max_sil_kept", type=int, default=500)
    p.add_argument("--norm_max", type=float, default=0.9)
    p.add_argument("--norm_alpha", type=float, default=0.75)
    p.add_argument("--peak_reject", type=float, default=2.5)
    a = p.parse_args(argv)
    return PreprocessConfig(
        inp_root=a.inp_root,
        exp_dir=a.exp_dir,
        sr=a.sr,
        n_p=a.n_p,
        noparallel=a.noparallel,
        per=a.per,
        overlap=a.overlap,
        highpass_hz=a.highpass_hz,
        enable_highpass=not a.no_highpass,
        threshold=a.threshold,
        min_length=a.min_length,
        min_interval=a.min_interval,
        hop_size=a.hop_size,
        max_sil_kept=a.max_sil_kept,
        norm_max=a.norm_max,
        norm_alpha=a.norm_alpha,
        peak_reject=a.peak_reject,
    )


def main(argv=None) -> int:
    global _LOG
    cfg = parse_args(argv)
    if not os.path.isdir(cfg.inp_root):
        print(f"input missing: {cfg.inp_root}", file=sys.stderr)
        return 1
    os.makedirs(cfg.exp_dir, exist_ok=True)
    _LOG = open(os.path.join(cfg.exp_dir, "preprocess.log"), "a", encoding="utf-8")
    try:
        PreProcess(cfg).run()
    finally:
        _LOG.close()
        _LOG = None
    return 0


if __name__ == "__main__":
    # Windows multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
