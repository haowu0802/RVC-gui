"""Parse RVC train.log for per-epoch loss metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RE_EPOCH = re.compile(r"Training epoch:\s*(\d+)", re.IGNORECASE)
_RE_EPOCH_CN = re.compile(r"训练轮次[：:]\s*(\d+)")
_RE_LOSS = re.compile(
    r"loss_disc=([0-9.]+).*?loss_gen=([0-9.]+).*?loss_fm=([0-9.]+)"
    r".*?loss_mel=([0-9.]+).*?loss_kl=([0-9.]+)",
    re.IGNORECASE,
)
_RE_WEIGHT_EPOCH = re.compile(r"_e(\d+)_s\d+", re.IGNORECASE)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss_disc: float
    loss_gen: float
    loss_fm: float
    loss_mel: float
    loss_kl: float
    weight_path: str = ""


def parse_train_log(log_path: Path | str) -> dict[int, EpochMetrics]:
    """
    Return metrics keyed by epoch. If the log contains resumed runs with
    duplicate epochs, the last logged sample for that epoch wins.
    """
    path = Path(log_path)
    if not path.is_file():
        return {}

    cur: int | None = None
    by_epoch: dict[int, EpochMetrics] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    for line in lines:
        m = _RE_EPOCH.search(line) or _RE_EPOCH_CN.search(line)
        if m:
            cur = int(m.group(1))
        m = _RE_LOSS.search(line)
        if not m or cur is None:
            continue
        by_epoch[cur] = EpochMetrics(
            epoch=cur,
            loss_disc=float(m.group(1)),
            loss_gen=float(m.group(2)),
            loss_fm=float(m.group(3)),
            loss_mel=float(m.group(4)),
            loss_kl=float(m.group(5)),
        )
    return by_epoch


def find_weight_files(weights_dir: Path | str, exp_name: str) -> dict[int, Path]:
    """Map epoch -> newest matching exported weight under assets/weights."""
    folder = Path(weights_dir)
    if not folder.is_dir() or not exp_name:
        return {}
    prefix = exp_name.lower() + "_e"
    found: dict[int, Path] = {}
    for path in folder.glob("*.pth"):
        name = path.name
        if not name.lower().startswith(prefix):
            continue
        m = _RE_WEIGHT_EPOCH.search(name)
        if not m:
            continue
        epoch = int(m.group(1))
        prev = found.get(epoch)
        if prev is None or path.stat().st_mtime >= prev.stat().st_mtime:
            found[epoch] = path
    return found


def merge_metrics_with_weights(
    metrics: dict[int, EpochMetrics],
    weights: dict[int, Path],
) -> list[EpochMetrics]:
    rows: list[EpochMetrics] = []
    epochs = sorted(set(metrics) | set(weights))
    for epoch in epochs:
        base = metrics.get(epoch)
        w = weights.get(epoch)
        if base is None:
            rows.append(
                EpochMetrics(
                    epoch=epoch,
                    loss_disc=float("nan"),
                    loss_gen=float("nan"),
                    loss_fm=float("nan"),
                    loss_mel=float("nan"),
                    loss_kl=float("nan"),
                    weight_path=str(w) if w else "",
                )
            )
        else:
            rows.append(
                EpochMetrics(
                    epoch=base.epoch,
                    loss_disc=base.loss_disc,
                    loss_gen=base.loss_gen,
                    loss_fm=base.loss_fm,
                    loss_mel=base.loss_mel,
                    loss_kl=base.loss_kl,
                    weight_path=str(w) if w else "",
                )
            )
    return rows
