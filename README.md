# RVC Pipeline GUI

Portable GUI for Retrieval-based Voice Conversion (RVC) workflows:

1. Preprocess
2. Extract F0 (rmvpe / CUDA)
3. Extract HuBERT features
4. Train
5. Build FAISS index
6. Infer A/B (compare epoch weights on a short clip)
7. Infer + Merge (long-form chunked inference + optional BGM mix)

This repository is **not** a full RVC runtime. Point the GUI at an existing RVC WebUI install (the folder that contains `train/`, `infer/`, `assets/`, and usually `runtime/python`).

## Requirements

- An RVC install compatible with `infer.vc.modules.VC` (tested with 2026-07 style packages)
- That install's Python (typically `runtime/python` / `runtime/python.exe`) with the RVC dependencies already installed
- `ffmpeg` / `ffprobe` on `PATH` (needed for long infer + merge)
- Optional: `sounddevice` inside the RVC Python for in-GUI A/B playback

## Quick start

1. Clone this repo anywhere.
2. Set the RVC root in one of these ways:
   - Open the GUI → **Settings** → browse to your RVC folder, or
   - Set environment variable `RVC_ROOT` to that folder
3. Launch:

```bat
launch_gui.bat
```

or:

```bash
# Prefer the RVC runtime interpreter when available
"$RVC_ROOT/runtime/python" -I gui.py
```

On first launch, configure **Settings → RVC root**. Paths and options are saved to local `settings.json` (gitignored).

## Environment

| Variable   | Meaning                                      |
|-----------|-----------------------------------------------|
| `RVC_ROOT` | Absolute path to the RVC install used by jobs |

Jobs always run with `cwd=$RVC_ROOT` and `PYTHONPATH` including that root so `configs`, `infer`, and `train` imports resolve.

## Scripts

| Script | Role |
|--------|------|
| `scripts/preprocess_flex.py` | Configurable preprocess / slicer |
| `scripts/train_flex.py` | Build `filelist.txt` + launch `train/train.py` |
| `scripts/infer_batch_test.py` | Multi-weight short-clip A/B export |
| `scripts/infer_long.py` | Chunked long-audio inference with crossfade |

## Infer + Merge presets

`examples/preset.example.json` shows the JSON fields accepted by long infer (`model_name`, `index_path`, pitch / index / protect knobs, `chunk_sec`, `overlap_sec`). CLI flags override preset values.

## Notes

- Do not commit `settings.json` (may contain local absolute paths chosen by you).
- Intermediate training checkpoints under `logs/<exp>/G_*.pth` and `D_*.pth` are large; inference uses the smaller exports under `assets/weights/`.
- For epoch picking, A/B with `index_rate=0` first, then enable index for final listens.
