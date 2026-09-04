#!/usr/bin/env python3
"""Prepare filelist/config and launch RVC train/train.py (WebUI-equivalent)."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RVC_ROOT") or os.getcwd()).resolve()
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stem_set(folder: Path, suffixes: tuple[str, ...]) -> set[str]:
    if not folder.is_dir():
        return set()
    out: set[str] = set()
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            # match webui: name.split(".")[0]  (first segment only)
            out.add(p.name.split(".")[0])
    return out


def _esc(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def resolve_exp(exp: str) -> tuple[str, Path]:
    """Return (experiment_name_for_-e, absolute_logs_exp_dir)."""
    p = Path(exp)
    if p.is_absolute() or (len(p.parts) > 1 and (p.exists() or str(p).startswith("logs"))):
        p = p.resolve()
        if p.name == "logs":
            raise ValueError(f"Pass experiment folder under logs, not logs itself: {p}")
        # Prefer: .../logs/<name>
        if p.parent.name == "logs":
            return p.name, p
        # Absolute custom path: still require it to live as logs/<name> for train.py
        logs = (ROOT / "logs").resolve()
        try:
            rel = p.relative_to(logs)
        except ValueError as exc:
            raise ValueError(
                f"Experiment must be under {logs} (train.py expects ./logs/<name>):\n{p}"
            ) from exc
        if len(rel.parts) != 1:
            raise ValueError(f"Experiment must be a direct child of logs: {p}")
        return rel.parts[0], p
    name = str(exp).strip().strip("/\\")
    return name, (ROOT / "logs" / name).resolve()


def default_pretrained(version: str, sample_rate: str, if_f0: bool) -> tuple[str, str]:
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if if_f0 else ""
    g = ROOT / "assets" / f"pretrained{path_str}" / f"{f0_str}G{sample_rate}.pth"
    d = ROOT / "assets" / f"pretrained{path_str}" / f"{f0_str}D{sample_rate}.pth"
    return (
        str(g) if g.is_file() else "",
        str(d) if d.is_file() else "",
    )


def write_filelist(
    exp_dir: Path,
    sample_rate: str,
    version: str,
    if_f0: bool,
    spk_id: int,
) -> int:
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not gt.is_dir():
        raise FileNotFoundError(f"Missing 0_gt_wavs: {gt}")
    if not feat.is_dir():
        raise FileNotFoundError(f"Missing feature dir: {feat}")

    names = _stem_set(gt, (".wav",)) & _stem_set(feat, (".npy",))
    if if_f0:
        f0 = exp_dir / "2a_f0"
        f0nsf = exp_dir / "2b-f0nsf"
        names &= _stem_set(f0, (".npy",))
        names &= _stem_set(f0nsf, (".npy",))
        if not names:
            raise RuntimeError(
                "No usable clips (need intersection of 0_gt_wavs ∩ features ∩ F0). "
                "Finish preprocess / F0 / HuBERT first."
            )
    elif not names:
        raise RuntimeError("No usable clips (0_gt_wavs ∩ features empty).")

    opt: list[str] = []
    for name in sorted(names):
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    _esc(gt),
                    name,
                    _esc(feat),
                    name,
                    _esc(exp_dir / "2a_f0"),
                    name,
                    _esc(exp_dir / "2b-f0nsf"),
                    name,
                    spk_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (_esc(gt), name, _esc(feat), name, spk_id)
            )

    fea_dim = 256 if version == "v1" else 768
    mute_root = ROOT / "logs" / "mute"
    if if_f0:
        for _ in range(2):
            opt.append(
                "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|"
                "%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%s"
                % (
                    _esc(mute_root),
                    sample_rate,
                    _esc(mute_root),
                    fea_dim,
                    _esc(mute_root),
                    _esc(mute_root),
                    spk_id,
                )
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%s"
                % (_esc(mute_root), sample_rate, _esc(mute_root), fea_dim, spk_id)
            )

    random.shuffle(opt)
    out = exp_dir / "filelist.txt"
    out.write_text("\n".join(opt) + "\n", encoding="utf8")
    return len(names)


def ensure_config(exp_dir: Path, sample_rate: str, version: str) -> Path:
    cfg_save = exp_dir / "config.json"
    if cfg_save.is_file():
        return cfg_save
    # Match webui: v1 or 40k → configs/v1; else v2
    if version == "v1" or sample_rate == "40k":
        rel = Path("v1") / f"{sample_rate}.json"
    else:
        rel = Path("v2") / f"{sample_rate}.json"
    src = ROOT / "configs" / rel
    if not src.is_file():
        raise FileNotFoundError(f"Missing model config: {src}")
    data = json.loads(src.read_text(encoding="utf8"))
    cfg_save.write_text(
        json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf8",
    )
    return cfg_save


def build_train_cmd(args: argparse.Namespace, exp_name: str) -> list[str]:
    py = sys.executable
    cmd = [
        py,
        str(ROOT / "train" / "train.py"),
        "-e",
        exp_name,
        "-sr",
        args.sample_rate,
        "-f0",
        "1" if args.if_f0 else "0",
        "-bs",
        str(args.batch_size),
        "-g",
        args.gpus,
        "-te",
        str(args.total_epoch),
        "-se",
        str(args.save_every_epoch),
        "-ts",
        str(args.total_steps),
        "-ss",
        str(args.save_every_steps),
        "-l",
        "1" if args.save_latest_only else "0",
        "-c",
        "1" if args.cache_in_gpu else "0",
        "-sw",
        "1" if args.save_every_weights else "0",
        "-v",
        args.version,
    ]
    if args.pretrain_g:
        cmd.extend(["-pg", args.pretrain_g])
    if args.pretrain_d:
        cmd.extend(["-pd", args.pretrain_d])
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVC train launcher (filelist + train.py)")
    p.add_argument(
        "--exp_dir",
        required=True,
        help="Experiment folder: logs/<name> or absolute path under logs/",
    )
    p.add_argument("--sample_rate", default="48k", choices=["32k", "40k", "48k"])
    p.add_argument("--version", default="v2", choices=["v1", "v2"])
    p.add_argument("--if_f0", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--spk_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--total_epoch", type=int, default=200)
    p.add_argument("--save_every_epoch", type=int, default=5)
    p.add_argument(
        "--total_steps",
        type=int,
        default=0,
        help="Stop after N global steps (0=off; whichever of steps/epochs hits first)",
    )
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Export assets/weights every N steps (0=off)",
    )
    p.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. 0 or 0-1")
    p.add_argument(
        "--pretrain_g",
        default="",
        help="Generator pretrained .pth (empty = auto f0G* for version/sr)",
    )
    p.add_argument(
        "--pretrain_d",
        default="",
        help="Discriminator pretrained .pth (empty = auto f0D*)",
    )
    p.add_argument(
        "--save_latest_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep latest G/D checkpoint in logs (saves disk)",
    )
    p.add_argument(
        "--cache_in_gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache dataset in GPU VRAM (faster but hungry)",
    )
    p.add_argument(
        "--save_every_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export inferable weights under assets/weights each save",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write filelist/config and print train command",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = parse_args(argv)
    exp_name, exp_dir = resolve_exp(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not args.pretrain_g and not args.pretrain_d:
        g, d = default_pretrained(args.version, args.sample_rate, args.if_f0)
        args.pretrain_g = g
        args.pretrain_d = d
    elif not args.pretrain_g or not args.pretrain_d:
        g, d = default_pretrained(args.version, args.sample_rate, args.if_f0)
        if not args.pretrain_g:
            args.pretrain_g = g
        if not args.pretrain_d:
            args.pretrain_d = d

    n = write_filelist(
        exp_dir, args.sample_rate, args.version, args.if_f0, args.spk_id
    )
    cfg = ensure_config(exp_dir, args.sample_rate, args.version)
    cmd = build_train_cmd(args, exp_name)

    print(f"[Train] exp={exp_name}")
    print(f"[Train] dir={exp_dir}")
    print(f"[Train] clips={n} (+2 mute)")
    print(f"[Train] config={cfg}")
    print(f"[Train] pretrainG={args.pretrain_g or '(none)'}")
    print(f"[Train] pretrainD={args.pretrain_d or '(none)'}")
    print("[Train] CMD:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if args.dry_run:
        return 0

    # Unbuffered child for live logs in GUI; no extra console on Windows.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
