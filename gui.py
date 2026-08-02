#!/usr/bin/env python3
"""RVC training / inference pipeline GUI (portable paths)."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

# Ensure this package directory is importable even under `python -I`.
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from rvc_env import (
    PACKAGE_DIR,
    SCRIPTS_DIR,
    SETTINGS_PATH,
    find_index_for_model,
    looks_like_rvc_root,
    prepare_rvc_process_env,
    resolve_rvc_root,
    rvc_python,
)

AUDIO_FILETYPES = [
    ("Audio Files", "*.wav *.flac *.mp3 *.m4a *.ogg *.aac"),
    ("All Files", "*.*"),
]
JSON_FILETYPES = [("JSON Files", "*.json"), ("All Files", "*.*")]
PTH_FILETYPES = [("PTH Files", "*.pth"), ("All Files", "*.*")]

TAB_SETTINGS = 0
TAB_PREPROCESS = 1
TAB_EXTRACT_F0 = 2
TAB_EXTRACT_HUBERT = 3
TAB_TRAIN = 4
TAB_BUILD_INDEX = 5
TAB_INFER_AB = 6
TAB_INFER_MERGE = 7

TAB_NAMES = [
    "Settings",
    "Preprocess",
    "Extract F0",
    "Extract HuBERT",
    "Train",
    "Build Index",
    "Infer A/B",
    "Infer + Merge",
]


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------


def _windows_apps_use_light_theme() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        try:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return bool(value)
        finally:
            winreg.CloseKey(key)
    except Exception:
        return None


def _apply_theme(root: tk.Tk, light: bool) -> dict[str, str]:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    if light:
        colors = {
            "bg": "#f0f0f0",
            "fg": "#1a1a1a",
            "field": "#ffffff",
            "select": "#cce4ff",
            "select_fg": "#000000",
            "button": "#e1e1e1",
            "border": "#adadad",
            "log_bg": "#ffffff",
            "log_fg": "#1a1a1a",
            "disabled": "#888888",
        }
    else:
        colors = {
            "bg": "#2b2b2b",
            "fg": "#e8e8e8",
            "field": "#3c3c3c",
            "select": "#094771",
            "select_fg": "#ffffff",
            "button": "#3c3c3c",
            "border": "#555555",
            "log_bg": "#1e1e1e",
            "log_fg": "#d4d4d4",
            "disabled": "#777777",
        }

    root.configure(bg=colors["bg"])
    style.configure(".", background=colors["bg"], foreground=colors["fg"], fieldbackground=colors["field"])
    style.configure("TFrame", background=colors["bg"])
    style.configure("TLabelframe", background=colors["bg"], foreground=colors["fg"])
    style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["fg"])
    style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
    style.configure("TButton", background=colors["button"], foreground=colors["fg"])
    style.map(
        "TButton",
        foreground=[("disabled", colors["disabled"])],
        background=[("active", colors["select"])],
    )
    style.configure("TCheckbutton", background=colors["bg"], foreground=colors["fg"])
    style.configure("TRadiobutton", background=colors["bg"], foreground=colors["fg"])
    style.configure("TEntry", fieldbackground=colors["field"], foreground=colors["fg"], insertcolor=colors["fg"])
    style.configure(
        "TCombobox",
        fieldbackground=colors["field"],
        foreground=colors["fg"],
        background=colors["button"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", colors["field"])],
        foreground=[("readonly", colors["fg"])],
    )
    style.configure("TNotebook", background=colors["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", background=colors["button"], foreground=colors["fg"], padding=[10, 4])
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["select"])],
        foreground=[("selected", colors["select_fg"])],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=colors["button"],
        troughcolor=colors["bg"],
        arrowcolor=colors["fg"],
    )
    style.configure("TSeparator", background=colors["border"])
    return colors


# ---------------------------------------------------------------------------
# Scrollable frame
# ---------------------------------------------------------------------------


class ScrollableFrame(ttk.Frame):
    """Canvas + scrollbar body so tab content need not force a maximized window."""

    def __init__(self, parent: tk.Misc, bg: str, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, highlightthickness=0, bg=bg, bd=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        self._vsb.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event: tk.Event | None = None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._win, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event | None = None) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event | None = None) -> None:
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RVC Pipeline")
        self.root.geometry("1080x780")
        self.root.minsize(800, 560)

        light = _windows_apps_use_light_theme()
        if light is None:
            light = True
        self.colors = _apply_theme(root, light)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen[str] | None = None
        self.running = False
        self._persist_trace_ids: list[str] = []
        self._suppress_autosave = False
        self._playback = None  # sounddevice stream / data handle

        # --- shared / settings ---
        self.rvc_root = tk.StringVar(value="")
        self.python_path_display = tk.StringVar(value="(set RVC root first)")
        self.status_var = tk.StringVar(value="Idle")
        self.last_tab = tk.IntVar(value=0)
        self.geometry_var = tk.StringVar(value="1080x780")

        # --- preprocess ---
        self.pp_inp_root = tk.StringVar(value="")
        self.pp_exp_dir = tk.StringVar(value="logs/my_exp")
        self.pp_sr = tk.StringVar(value="48000")
        self.pp_n_p = tk.StringVar(value="8")
        self.pp_per = tk.StringVar(value="3.5")
        self.pp_overlap = tk.StringVar(value="0.3")
        self.pp_noparallel = tk.BooleanVar(value=False)
        self.pp_highpass = tk.BooleanVar(value=True)
        self.pp_highpass_hz = tk.StringVar(value="48.0")
        self.pp_threshold = tk.StringVar(value="-42")
        self.pp_min_length = tk.StringVar(value="1500")
        self.pp_min_interval = tk.StringVar(value="400")
        self.pp_hop_size = tk.StringVar(value="15")
        self.pp_max_sil_kept = tk.StringVar(value="500")
        self.pp_norm_max = tk.StringVar(value="0.9")
        self.pp_norm_alpha = tk.StringVar(value="0.75")
        self.pp_peak_reject = tk.StringVar(value="2.5")

        # --- extract f0 / hubert ---
        self.f0_exp = tk.StringVar(value="my_exp")
        self.f0_gpu = tk.StringVar(value="0")
        self.f0_is_half = tk.BooleanVar(value=True)
        self.hb_exp = tk.StringVar(value="my_exp")
        self.hb_gpu = tk.StringVar(value="0")
        self.hb_version = tk.StringVar(value="v2")
        self.hb_is_half = tk.BooleanVar(value=True)

        # --- train ---
        self.tr_exp = tk.StringVar(value="my_exp")
        self.tr_sample_rate = tk.StringVar(value="48k")
        self.tr_version = tk.StringVar(value="v2")
        self.tr_if_f0 = tk.BooleanVar(value=True)
        self.tr_spk_id = tk.StringVar(value="0")
        self.tr_batch_size = tk.StringVar(value="6")
        self.tr_total_epoch = tk.StringVar(value="200")
        self.tr_save_every_epoch = tk.StringVar(value="5")
        self.tr_gpus = tk.StringVar(value="0")
        self.tr_pretrain_g = tk.StringVar(value="")
        self.tr_pretrain_d = tk.StringVar(value="")
        self.tr_save_latest_only = tk.BooleanVar(value=True)
        self.tr_cache_in_gpu = tk.BooleanVar(value=False)
        self.tr_save_every_weights = tk.BooleanVar(value=True)

        # --- build index ---
        self.ix_exp = tk.StringVar(value="my_exp")
        self.ix_version = tk.StringVar(value="v2")
        self.ix_outside = tk.StringVar(value="")
        self.ix_n_cpu = tk.StringVar(value="4")

        # --- infer A/B ---
        self.ab_input = tk.StringVar(value="")
        self.ab_weights_dir = tk.StringVar(value="")
        self.ab_filter = tk.StringVar(value="")
        self.ab_index = tk.StringVar(value="")
        self.ab_out_dir = tk.StringVar(value="")
        self.ab_pitch = tk.StringVar(value="0")
        self.ab_f0_method = tk.StringVar(value="rmvpe")
        self.ab_index_rate = tk.StringVar(value="0.75")
        self.ab_protect = tk.StringVar(value="0.33")
        self.ab_rms = tk.StringVar(value="0.25")
        self.ab_resample = tk.StringVar(value="0")
        self.ab_spk = tk.StringVar(value="0")
        self.ab_device = tk.StringVar(value="")
        self._weight_vars: dict[str, tk.BooleanVar] = {}
        self._result_paths: list[str] = []

        # --- infer + merge ---
        self.im_model_path = tk.StringVar(value="")
        self.im_model_name = tk.StringVar(value="")
        self.im_index_path = tk.StringVar(value="")
        self.im_input_path = tk.StringVar(value="")
        self.im_infer_output_dir = tk.StringVar(value="")
        self.im_opt_path = tk.StringVar(value="")
        self.im_f0up_key = tk.StringVar(value="0")
        self.im_f0method = tk.StringVar(value="rmvpe")
        self.im_index_rate = tk.StringVar(value="0.95")
        self.im_filter_radius = tk.StringVar(value="0")
        self.im_resample_sr = tk.StringVar(value="0")
        self.im_rms_mix_rate = tk.StringVar(value="0.95")
        self.im_protect = tk.StringVar(value="0.4")
        self.im_chunk_sec = tk.StringVar(value="200")
        self.im_overlap_sec = tk.StringVar(value="0.3")
        self.im_spk_id = tk.StringVar(value="0")
        self.im_infer_result = tk.StringVar(value="")
        self.im_bgm_path = tk.StringVar(value="")
        self.im_merge_output_dir = tk.StringVar(value="")
        self.im_merge_output_path = tk.StringVar(value="")

        self._build_ui()
        self._load_settings()
        self._wire_autosave()
        self._refresh_python_display()
        self._fill_empty_defaults_from_root()
        self._refresh_ab_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._drain_log_queue)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._update_global_start_state()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _configured_root(self) -> Path | None:
        raw = self.rvc_root.get().strip().strip('"')
        if not raw:
            return None
        p = Path(raw).expanduser()
        try:
            p = p.resolve()
        except Exception:
            return None
        return p if p.is_dir() else None

    def _require_rvc_root(self) -> Path | None:
        configured = self.rvc_root.get().strip() or None
        try:
            root = resolve_rvc_root(configured)
        except FileNotFoundError as exc:
            messagebox.showerror("RVC root required", str(exc))
            return None
        if not looks_like_rvc_root(root):
            messagebox.showerror(
                "Invalid RVC root",
                f"Path does not look like an RVC install:\n{root}",
            )
            return None
        # Keep UI in sync if resolved from env / parent
        if not self.rvc_root.get().strip():
            self.rvc_root.set(str(root))
            self._refresh_python_display()
            self._fill_empty_defaults_from_root()
        return root

    def _rvc_path(self, *parts: str) -> str:
        root = self._configured_root()
        if root is None:
            try:
                root = resolve_rvc_root(self.rvc_root.get().strip() or None)
            except FileNotFoundError:
                return str(Path(*parts)) if parts else ""
        return str(root.joinpath(*parts))

    @staticmethod
    def exp_name(exp: str) -> str:
        """If path ends with logs/<name>, return name; else basename / stripped name."""
        s = (exp or "").strip().strip("/\\")
        if not s:
            return ""
        p = Path(s)
        parts = p.parts
        if len(parts) >= 2 and parts[-2].lower() == "logs":
            return parts[-1]
        # Absolute or relative multi-segment → last component
        if len(parts) > 1:
            return p.name
        return s

    def _exp_dir_abs(self, exp: str) -> str:
        name = self.exp_name(exp) or "my_exp"
        return self._rvc_path("logs", name)

    def _fill_empty_defaults_from_root(self) -> None:
        root = self._configured_root()
        if root is None:
            try:
                root = resolve_rvc_root(self.rvc_root.get().strip() or None)
            except FileNotFoundError:
                return
        if not self.ab_weights_dir.get().strip():
            self.ab_weights_dir.set(str(root / "assets" / "weights"))
        if not self.ix_outside.get().strip():
            self.ix_outside.set(str(root / "assets" / "indices"))
        if not self.ab_out_dir.get().strip():
            self.ab_out_dir.set(str(root / "logs" / "my_exp" / "infer_ab_test"))
        if not self.im_infer_output_dir.get().strip():
            self.im_infer_output_dir.set(str(root / "logs" / "my_exp" / "infer_long"))
        if not self.im_merge_output_dir.get().strip():
            self.im_merge_output_dir.set(str(root / "logs" / "my_exp" / "merged"))

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _persist_map(self) -> dict[str, Any]:
        return {
            "geometry": self.root.geometry(),
            "last_tab": self.notebook.index(self.notebook.select()) if hasattr(self, "notebook") else 0,
            "rvc_root": self.rvc_root.get(),
            "pp_inp_root": self.pp_inp_root.get(),
            "pp_exp_dir": self.pp_exp_dir.get(),
            "pp_sr": self.pp_sr.get(),
            "pp_n_p": self.pp_n_p.get(),
            "pp_per": self.pp_per.get(),
            "pp_overlap": self.pp_overlap.get(),
            "pp_noparallel": self.pp_noparallel.get(),
            "pp_highpass": self.pp_highpass.get(),
            "pp_highpass_hz": self.pp_highpass_hz.get(),
            "pp_threshold": self.pp_threshold.get(),
            "pp_min_length": self.pp_min_length.get(),
            "pp_min_interval": self.pp_min_interval.get(),
            "pp_hop_size": self.pp_hop_size.get(),
            "pp_max_sil_kept": self.pp_max_sil_kept.get(),
            "pp_norm_max": self.pp_norm_max.get(),
            "pp_norm_alpha": self.pp_norm_alpha.get(),
            "pp_peak_reject": self.pp_peak_reject.get(),
            "f0_exp": self.f0_exp.get(),
            "f0_gpu": self.f0_gpu.get(),
            "f0_is_half": self.f0_is_half.get(),
            "hb_exp": self.hb_exp.get(),
            "hb_gpu": self.hb_gpu.get(),
            "hb_version": self.hb_version.get(),
            "hb_is_half": self.hb_is_half.get(),
            "tr_exp": self.tr_exp.get(),
            "tr_sample_rate": self.tr_sample_rate.get(),
            "tr_version": self.tr_version.get(),
            "tr_if_f0": self.tr_if_f0.get(),
            "tr_spk_id": self.tr_spk_id.get(),
            "tr_batch_size": self.tr_batch_size.get(),
            "tr_total_epoch": self.tr_total_epoch.get(),
            "tr_save_every_epoch": self.tr_save_every_epoch.get(),
            "tr_gpus": self.tr_gpus.get(),
            "tr_pretrain_g": self.tr_pretrain_g.get(),
            "tr_pretrain_d": self.tr_pretrain_d.get(),
            "tr_save_latest_only": self.tr_save_latest_only.get(),
            "tr_cache_in_gpu": self.tr_cache_in_gpu.get(),
            "tr_save_every_weights": self.tr_save_every_weights.get(),
            "ix_exp": self.ix_exp.get(),
            "ix_version": self.ix_version.get(),
            "ix_outside": self.ix_outside.get(),
            "ix_n_cpu": self.ix_n_cpu.get(),
            "ab_input": self.ab_input.get(),
            "ab_weights_dir": self.ab_weights_dir.get(),
            "ab_filter": self.ab_filter.get(),
            "ab_index": self.ab_index.get(),
            "ab_out_dir": self.ab_out_dir.get(),
            "ab_pitch": self.ab_pitch.get(),
            "ab_f0_method": self.ab_f0_method.get(),
            "ab_index_rate": self.ab_index_rate.get(),
            "ab_protect": self.ab_protect.get(),
            "ab_rms": self.ab_rms.get(),
            "ab_resample": self.ab_resample.get(),
            "ab_spk": self.ab_spk.get(),
            "ab_device": self.ab_device.get(),
            "im_model_path": self.im_model_path.get(),
            "im_model_name": self.im_model_name.get(),
            "im_index_path": self.im_index_path.get(),
            "im_input_path": self.im_input_path.get(),
            "im_infer_output_dir": self.im_infer_output_dir.get(),
            "im_f0up_key": self.im_f0up_key.get(),
            "im_f0method": self.im_f0method.get(),
            "im_index_rate": self.im_index_rate.get(),
            "im_filter_radius": self.im_filter_radius.get(),
            "im_resample_sr": self.im_resample_sr.get(),
            "im_rms_mix_rate": self.im_rms_mix_rate.get(),
            "im_protect": self.im_protect.get(),
            "im_chunk_sec": self.im_chunk_sec.get(),
            "im_overlap_sec": self.im_overlap_sec.get(),
            "im_spk_id": self.im_spk_id.get(),
            "im_infer_result": self.im_infer_result.get(),
            "im_bgm_path": self.im_bgm_path.get(),
            "im_merge_output_dir": self.im_merge_output_dir.get(),
        }

    def _load_settings(self) -> None:
        self._suppress_autosave = True
        if SETTINGS_PATH.is_file():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        str_vars = {
            "rvc_root": self.rvc_root,
            "pp_inp_root": self.pp_inp_root,
            "pp_exp_dir": self.pp_exp_dir,
            "pp_sr": self.pp_sr,
            "pp_n_p": self.pp_n_p,
            "pp_per": self.pp_per,
            "pp_overlap": self.pp_overlap,
            "pp_highpass_hz": self.pp_highpass_hz,
            "pp_threshold": self.pp_threshold,
            "pp_min_length": self.pp_min_length,
            "pp_min_interval": self.pp_min_interval,
            "pp_hop_size": self.pp_hop_size,
            "pp_max_sil_kept": self.pp_max_sil_kept,
            "pp_norm_max": self.pp_norm_max,
            "pp_norm_alpha": self.pp_norm_alpha,
            "pp_peak_reject": self.pp_peak_reject,
            "f0_exp": self.f0_exp,
            "f0_gpu": self.f0_gpu,
            "hb_exp": self.hb_exp,
            "hb_gpu": self.hb_gpu,
            "hb_version": self.hb_version,
            "tr_exp": self.tr_exp,
            "tr_sample_rate": self.tr_sample_rate,
            "tr_version": self.tr_version,
            "tr_spk_id": self.tr_spk_id,
            "tr_batch_size": self.tr_batch_size,
            "tr_total_epoch": self.tr_total_epoch,
            "tr_save_every_epoch": self.tr_save_every_epoch,
            "tr_gpus": self.tr_gpus,
            "tr_pretrain_g": self.tr_pretrain_g,
            "tr_pretrain_d": self.tr_pretrain_d,
            "ix_exp": self.ix_exp,
            "ix_version": self.ix_version,
            "ix_outside": self.ix_outside,
            "ix_n_cpu": self.ix_n_cpu,
            "ab_input": self.ab_input,
            "ab_weights_dir": self.ab_weights_dir,
            "ab_filter": self.ab_filter,
            "ab_index": self.ab_index,
            "ab_out_dir": self.ab_out_dir,
            "ab_pitch": self.ab_pitch,
            "ab_f0_method": self.ab_f0_method,
            "ab_index_rate": self.ab_index_rate,
            "ab_protect": self.ab_protect,
            "ab_rms": self.ab_rms,
            "ab_resample": self.ab_resample,
            "ab_spk": self.ab_spk,
            "ab_device": self.ab_device,
            "im_model_path": self.im_model_path,
            "im_model_name": self.im_model_name,
            "im_index_path": self.im_index_path,
            "im_input_path": self.im_input_path,
            "im_infer_output_dir": self.im_infer_output_dir,
            "im_f0up_key": self.im_f0up_key,
            "im_f0method": self.im_f0method,
            "im_index_rate": self.im_index_rate,
            "im_filter_radius": self.im_filter_radius,
            "im_resample_sr": self.im_resample_sr,
            "im_rms_mix_rate": self.im_rms_mix_rate,
            "im_protect": self.im_protect,
            "im_chunk_sec": self.im_chunk_sec,
            "im_overlap_sec": self.im_overlap_sec,
            "im_spk_id": self.im_spk_id,
            "im_infer_result": self.im_infer_result,
            "im_bgm_path": self.im_bgm_path,
            "im_merge_output_dir": self.im_merge_output_dir,
        }
        bool_vars = {
            "pp_noparallel": self.pp_noparallel,
            "pp_highpass": self.pp_highpass,
            "f0_is_half": self.f0_is_half,
            "hb_is_half": self.hb_is_half,
            "tr_if_f0": self.tr_if_f0,
            "tr_save_latest_only": self.tr_save_latest_only,
            "tr_cache_in_gpu": self.tr_cache_in_gpu,
            "tr_save_every_weights": self.tr_save_every_weights,
        }
        for key, var in str_vars.items():
            if key in data and data[key] is not None:
                var.set(str(data[key]))
        for key, var in bool_vars.items():
            if key in data:
                var.set(bool(data[key]))

        geo = data.get("geometry")
        if isinstance(geo, str) and geo:
            try:
                self.root.geometry(geo)
            except tk.TclError:
                pass

        last = data.get("last_tab", 0)
        try:
            last_i = int(last)
        except (TypeError, ValueError):
            last_i = 0
        last_i = max(0, min(last_i, len(TAB_NAMES) - 1))
        self.root.after(50, lambda: self.notebook.select(last_i))

        self._refresh_im_auto_paths()
        self._suppress_autosave = False

    def _save_settings(self) -> None:
        if self._suppress_autosave:
            return
        try:
            data = self._persist_map()
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _wire_autosave(self) -> None:
        vars_to_trace: list[tk.Variable] = [
            self.rvc_root,
            self.pp_inp_root,
            self.pp_exp_dir,
            self.pp_sr,
            self.pp_n_p,
            self.pp_per,
            self.pp_overlap,
            self.pp_noparallel,
            self.pp_highpass,
            self.pp_highpass_hz,
            self.pp_threshold,
            self.pp_min_length,
            self.pp_min_interval,
            self.pp_hop_size,
            self.pp_max_sil_kept,
            self.pp_norm_max,
            self.pp_norm_alpha,
            self.pp_peak_reject,
            self.f0_exp,
            self.f0_gpu,
            self.f0_is_half,
            self.hb_exp,
            self.hb_gpu,
            self.hb_version,
            self.hb_is_half,
            self.tr_exp,
            self.tr_sample_rate,
            self.tr_version,
            self.tr_if_f0,
            self.tr_spk_id,
            self.tr_batch_size,
            self.tr_total_epoch,
            self.tr_save_every_epoch,
            self.tr_gpus,
            self.tr_pretrain_g,
            self.tr_pretrain_d,
            self.tr_save_latest_only,
            self.tr_cache_in_gpu,
            self.tr_save_every_weights,
            self.ix_exp,
            self.ix_version,
            self.ix_outside,
            self.ix_n_cpu,
            self.ab_input,
            self.ab_weights_dir,
            self.ab_filter,
            self.ab_index,
            self.ab_out_dir,
            self.ab_pitch,
            self.ab_f0_method,
            self.ab_index_rate,
            self.ab_protect,
            self.ab_rms,
            self.ab_resample,
            self.ab_spk,
            self.ab_device,
            self.im_model_path,
            self.im_model_name,
            self.im_index_path,
            self.im_input_path,
            self.im_infer_output_dir,
            self.im_f0up_key,
            self.im_f0method,
            self.im_index_rate,
            self.im_filter_radius,
            self.im_resample_sr,
            self.im_rms_mix_rate,
            self.im_protect,
            self.im_chunk_sec,
            self.im_overlap_sec,
            self.im_spk_id,
            self.im_infer_result,
            self.im_bgm_path,
            self.im_merge_output_dir,
        ]
        for var in vars_to_trace:
            tid = var.trace_add("write", lambda *_a: self._save_settings())
            self._persist_trace_ids.append(tid)

    def _on_close(self) -> None:
        self._save_settings()
        self._stop_playback()
        if self.running and self.proc is not None:
            self._kill_process()
        self.root.destroy()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        self._tab_frames: list[ScrollableFrame] = []
        builders = [
            self._build_tab_settings,
            self._build_tab_preprocess,
            self._build_tab_extract_f0,
            self._build_tab_extract_hubert,
            self._build_tab_train,
            self._build_tab_build_index,
            self._build_tab_infer_ab,
            self._build_tab_infer_merge,
        ]
        for name, builder in zip(TAB_NAMES, builders):
            sf = ScrollableFrame(self.notebook, bg=self.colors["bg"])
            self.notebook.add(sf, text=name)
            self._tab_frames.append(sf)
            builder(sf.inner)

        # Global controls
        ctrl = ttk.Frame(outer)
        ctrl.pack(fill="x", pady=(8, 0))

        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start_current_tab)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.stop_job, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(
            log_frame,
            height=12,
            wrap="word",
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground=self.colors["log_fg"],
            relief="flat",
            bd=0,
        )
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")
        self.log_text.configure(state="disabled")

    def _row_path(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: Callable[[], None],
        browse_label: str = "Browse",
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
        ttk.Button(parent, text=browse_label, command=browse).grid(row=row, column=4, padx=(8, 0), pady=4)

    def _row_entry(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        var: tk.Variable,
        width: int | None = None,
        col: int = 1,
        colspan: int = 1,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0 if col == 1 else col - 1, sticky="w", padx=(0, 8), pady=4)
        kwargs: dict[str, Any] = {"textvariable": var}
        if width is not None:
            kwargs["width"] = width
        ttk.Entry(parent, **kwargs).grid(row=row, column=col, columnspan=colspan, sticky="w", pady=4)

    def _row_readonly(self, parent: tk.Misc, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, state="readonly").grid(
            row=row, column=1, columnspan=4, sticky="ew", pady=4
        )

    def _configure_cols(self, frame: ttk.Frame) -> None:
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    # ---- Tab 0: Settings ----

    def _build_tab_settings(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="RVC installation", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        self._row_path(f, 0, "RVC root", self.rvc_root, self._browse_rvc_root)
        self._row_readonly(f, 1, "Python", self.python_path_display)

        note = (
            "Set RVC root before running jobs. You can also export RVC_ROOT in the environment; "
            "the Settings path takes priority when set. Scripts run with cwd = RVC root."
        )
        ttk.Label(f, text=note, wraplength=900, justify="left").grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(8, 0)
        )

        ttk.Label(
            parent,
            text=f"Settings file: {SETTINGS_PATH}",
            wraplength=900,
        ).pack(anchor="w", padx=8, pady=8)

        self.rvc_root.trace_add("write", lambda *_a: self._on_rvc_root_changed())

    def _on_rvc_root_changed(self) -> None:
        self._refresh_python_display()
        self._fill_empty_defaults_from_root()

    def _browse_rvc_root(self) -> None:
        path = filedialog.askdirectory(title="Select RVC root folder")
        if path:
            self.rvc_root.set(path)
            if not looks_like_rvc_root(Path(path)):
                messagebox.showwarning(
                    "Warning",
                    "Selected folder may not be a full RVC install (missing train/train.py).",
                )

    def _refresh_python_display(self) -> None:
        root = self._configured_root()
        if root is None:
            try:
                root = resolve_rvc_root(None)
            except FileNotFoundError:
                self.python_path_display.set("(set RVC root first)")
                return
        try:
            self.python_path_display.set(str(rvc_python(root)))
        except Exception as exc:
            self.python_path_display.set(f"(error: {exc})")

    # ---- Tab 1: Preprocess ----

    def _build_tab_preprocess(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Flexible preprocess", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        self._row_path(f, 0, "Input audio dir", self.pp_inp_root, self._browse_pp_inp)
        self._row_path(f, 1, "Exp dir", self.pp_exp_dir, self._browse_pp_exp, "Browse")

        ttk.Label(f, text="Sample rate").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f, textvariable=self.pp_sr, values=["32000", "40000", "48000"], width=10, state="readonly"
        ).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(f, text="Workers (n_p)").grid(row=2, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.pp_n_p, width=8).grid(row=2, column=3, sticky="w", pady=4)

        ttk.Label(f, text="per (sec)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.pp_per, width=10).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(f, text="overlap").grid(row=3, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.pp_overlap, width=10).grid(row=3, column=3, sticky="w", pady=4)

        ttk.Checkbutton(f, text="noparallel", variable=self.pp_noparallel).grid(
            row=4, column=0, sticky="w", pady=4
        )
        ttk.Checkbutton(f, text="highpass", variable=self.pp_highpass).grid(
            row=4, column=1, sticky="w", pady=4
        )
        ttk.Label(f, text="highpass Hz").grid(row=4, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.pp_highpass_hz, width=10).grid(row=4, column=3, sticky="w", pady=4)

        slicer = ttk.LabelFrame(parent, text="Slicer", padding=10)
        slicer.pack(fill="x", padx=4, pady=4)
        self._configure_cols(slicer)
        for i, (lab, var) in enumerate(
            [
                ("threshold", self.pp_threshold),
                ("min_length", self.pp_min_length),
                ("min_interval", self.pp_min_interval),
                ("hop_size", self.pp_hop_size),
                ("max_sil_kept", self.pp_max_sil_kept),
            ]
        ):
            r, c = divmod(i, 3)
            ttk.Label(slicer, text=lab).grid(row=r, column=c * 2, sticky="w", padx=(0, 6), pady=3)
            ttk.Entry(slicer, textvariable=var, width=10).grid(row=r, column=c * 2 + 1, sticky="w", pady=3)

        norm = ttk.LabelFrame(parent, text="Normalize", padding=10)
        norm.pack(fill="x", padx=4, pady=4)
        self._configure_cols(norm)
        ttk.Label(norm, text="norm_max").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(norm, textvariable=self.pp_norm_max, width=10).grid(row=0, column=1, sticky="w", pady=3)
        ttk.Label(norm, text="norm_alpha").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=3)
        ttk.Entry(norm, textvariable=self.pp_norm_alpha, width=10).grid(row=0, column=3, sticky="w", pady=3)
        ttk.Label(norm, text="peak_reject").grid(row=0, column=4, sticky="w", padx=(12, 6), pady=3)
        ttk.Entry(norm, textvariable=self.pp_peak_reject, width=10).grid(row=0, column=5, sticky="w", pady=3)

    def _browse_pp_inp(self) -> None:
        path = filedialog.askdirectory(title="Select input audio folder")
        if path:
            self.pp_inp_root.set(path)

    def _browse_pp_exp(self) -> None:
        path = filedialog.askdirectory(title="Select experiment folder")
        if path:
            self.pp_exp_dir.set(path)

    # ---- Tab 2: Extract F0 ----

    def _build_tab_extract_f0(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Extract F0 (rmvpe / CUDA)", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)
        ttk.Label(f, text="Exp name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.f0_exp).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(f, text="GPU id").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.f0_gpu, width=8).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Checkbutton(f, text="is_half", variable=self.f0_is_half).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(
            f,
            text="Runs: train/dataset/extract_f0.py cuda 1 0 <gpu> <exp> <is_half>",
            wraplength=900,
        ).grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))

    # ---- Tab 3: Extract HuBERT ----

    def _build_tab_extract_hubert(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Extract HuBERT features", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)
        ttk.Label(f, text="Exp name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.hb_exp).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(f, text="GPU id").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.hb_gpu, width=8).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Version").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f, textvariable=self.hb_version, values=["v1", "v2"], width=8, state="readonly"
        ).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Checkbutton(f, text="is_half", variable=self.hb_is_half).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(
            f,
            text="Runs: train/dataset/extract_hubert_feature.py cuda 1 0 <gpu> <exp> <version> <is_half>",
            wraplength=900,
        ).grid(row=4, column=0, columnspan=5, sticky="w", pady=(8, 0))

    # ---- Tab 4: Train ----

    def _build_tab_train(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Train", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        ttk.Label(f, text="Exp name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_exp).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(f, text="Sample rate").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f, textvariable=self.tr_sample_rate, values=["32k", "40k", "48k"], width=8, state="readonly"
        ).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Version").grid(row=1, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Combobox(
            f, textvariable=self.tr_version, values=["v1", "v2"], width=8, state="readonly"
        ).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Checkbutton(f, text="if_f0", variable=self.tr_if_f0).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(f, text="spk_id").grid(row=2, column=1, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_spk_id, width=8).grid(row=2, column=2, sticky="w", pady=4)

        ttk.Label(f, text="batch_size").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_batch_size, width=8).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(f, text="total_epoch").grid(row=3, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_total_epoch, width=8).grid(row=3, column=3, sticky="w", pady=4)

        ttk.Label(f, text="save_every_epoch").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_save_every_epoch, width=8).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Label(f, text="gpus").grid(row=4, column=2, sticky="e", padx=(8, 8), pady=4)
        ttk.Entry(f, textvariable=self.tr_gpus, width=10).grid(row=4, column=3, sticky="w", pady=4)

        self._row_path(f, 5, "pretrain G (opt)", self.tr_pretrain_g, self._browse_pretrain_g)
        self._row_path(f, 6, "pretrain D (opt)", self.tr_pretrain_d, self._browse_pretrain_d)
        ttk.Label(f, text="Blank pretrain = auto from assets/pretrained*").grid(
            row=7, column=1, columnspan=4, sticky="w", pady=(0, 4)
        )

        ttk.Checkbutton(f, text="save_latest_only", variable=self.tr_save_latest_only).grid(
            row=8, column=0, sticky="w", pady=4
        )
        ttk.Checkbutton(f, text="cache_in_gpu", variable=self.tr_cache_in_gpu).grid(
            row=8, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(f, text="save_every_weights", variable=self.tr_save_every_weights).grid(
            row=8, column=2, columnspan=2, sticky="w", pady=4
        )

    def _browse_pretrain_g(self) -> None:
        path = filedialog.askopenfilename(title="Select generator pretrained", filetypes=PTH_FILETYPES)
        if path:
            self.tr_pretrain_g.set(path)

    def _browse_pretrain_d(self) -> None:
        path = filedialog.askopenfilename(title="Select discriminator pretrained", filetypes=PTH_FILETYPES)
        if path:
            self.tr_pretrain_d.set(path)

    # ---- Tab 5: Build Index ----

    def _build_tab_build_index(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Build FAISS index", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        ttk.Label(f, text="Exp name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.ix_exp).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(f, text="Version").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f, textvariable=self.ix_version, values=["v1", "v2"], width=8, state="readonly"
        ).grid(row=1, column=1, sticky="w", pady=4)
        self._row_path(f, 2, "Outside index dir", self.ix_outside, self._browse_ix_outside)
        ttk.Label(f, text="n_cpu").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.ix_n_cpu, width=8).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(
            f,
            text="Runs: train/train_index.py <exp_name> <version> <outside_index_dir> <n_cpu>",
            wraplength=900,
        ).grid(row=4, column=0, columnspan=5, sticky="w", pady=(8, 0))

    def _browse_ix_outside(self) -> None:
        path = filedialog.askdirectory(title="Select outside index directory")
        if path:
            self.ix_outside.set(path)

    # ---- Tab 6: Infer A/B ----

    def _build_tab_infer_ab(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Short-clip multi-weight test", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        self._row_path(f, 0, "Test audio", self.ab_input, self._browse_ab_input)
        self._row_path(f, 1, "Weights dir", self.ab_weights_dir, self._browse_ab_weights_dir)
        self._row_path(f, 2, "Index (opt)", self.ab_index, self._browse_ab_index)
        self._row_path(f, 3, "Output dir", self.ab_out_dir, self._browse_ab_out_dir)

        ttk.Label(f, text="Filter").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(f, textvariable=self.ab_filter).grid(row=4, column=1, sticky="ew", pady=4)
        btns = ttk.Frame(f)
        btns.grid(row=4, column=2, columnspan=3, sticky="e", pady=4)
        ttk.Button(btns, text="Refresh", command=self._refresh_weights_list).pack(side="left", padx=2)
        ttk.Button(btns, text="All", command=lambda: self._set_all_weights(True)).pack(side="left", padx=2)
        ttk.Button(btns, text="None", command=lambda: self._set_all_weights(False)).pack(side="left", padx=2)
        ttk.Button(btns, text="Every 10", command=self._select_every_10_epochs).pack(side="left", padx=2)

        list_frame = ttk.LabelFrame(parent, text="Weights (.pth)", padding=6)
        list_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.weights_canvas = tk.Canvas(
            list_frame, height=140, highlightthickness=0, bg=self.colors["field"]
        )
        wsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.weights_canvas.yview)
        self.weights_inner = ttk.Frame(self.weights_canvas)
        self.weights_canvas.create_window((0, 0), window=self.weights_inner, anchor="nw")
        self.weights_canvas.configure(yscrollcommand=wsb.set)
        self.weights_inner.bind(
            "<Configure>",
            lambda _e: self.weights_canvas.configure(scrollregion=self.weights_canvas.bbox("all")),
        )
        self.weights_canvas.pack(side="left", fill="both", expand=True)
        wsb.pack(side="right", fill="y")

        params = ttk.LabelFrame(parent, text="Infer params", padding=10)
        params.pack(fill="x", padx=4, pady=4)
        self._configure_cols(params)

        ttk.Label(params, text="pitch").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_pitch, width=8).grid(row=0, column=1, sticky="w", pady=3)
        ttk.Label(params, text="f0 method").grid(row=0, column=2, sticky="e", padx=(8, 6), pady=3)
        ttk.Combobox(
            params,
            textvariable=self.ab_f0_method,
            values=["rmvpe", "pm", "fcpe"],
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", pady=3)

        ttk.Label(params, text="index_rate").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_index_rate, width=8).grid(row=1, column=1, sticky="w", pady=3)
        ttk.Label(params, text="protect").grid(row=1, column=2, sticky="e", padx=(8, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_protect, width=8).grid(row=1, column=3, sticky="w", pady=3)

        ttk.Label(params, text="rms_mix").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_rms, width=8).grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(params, text="resample_sr").grid(row=2, column=2, sticky="e", padx=(8, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_resample, width=8).grid(row=2, column=3, sticky="w", pady=3)

        ttk.Label(params, text="spk_id").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(params, textvariable=self.ab_spk, width=8).grid(row=3, column=1, sticky="w", pady=3)

        play = ttk.LabelFrame(parent, text="Playback / results", padding=10)
        play.pack(fill="x", padx=4, pady=4)
        self._configure_cols(play)

        ttk.Label(play, text="Output device").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.ab_device_combo = ttk.Combobox(play, textvariable=self.ab_device, width=60, state="readonly")
        self.ab_device_combo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(play, text="Refresh devices", command=self._refresh_ab_devices).grid(
            row=0, column=3, padx=(8, 0), pady=4
        )

        ttk.Label(play, text="Result").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.ab_result_combo = ttk.Combobox(play, values=[], width=60, state="readonly")
        self.ab_result_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        rbtns = ttk.Frame(play)
        rbtns.grid(row=1, column=3, sticky="e", pady=4)
        ttk.Button(rbtns, text="Reload", command=self._reload_ab_results).pack(side="left", padx=2)
        ttk.Button(rbtns, text="Play", command=self._play_ab_result).pack(side="left", padx=2)
        ttk.Button(rbtns, text="Stop", command=self._stop_playback).pack(side="left", padx=2)

    def _browse_ab_input(self) -> None:
        path = filedialog.askopenfilename(title="Select test audio", filetypes=AUDIO_FILETYPES)
        if path:
            self.ab_input.set(path)

    def _browse_ab_weights_dir(self) -> None:
        path = filedialog.askdirectory(title="Select weights folder")
        if path:
            self.ab_weights_dir.set(path)
            self._refresh_weights_list()

    def _browse_ab_index(self) -> None:
        path = filedialog.askopenfilename(
            title="Select index file",
            filetypes=[("Index Files", "*.index"), ("All Files", "*.*")],
        )
        if path:
            self.ab_index.set(path)

    def _browse_ab_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Select infer output folder")
        if path:
            self.ab_out_dir.set(path)

    def _refresh_weights_list(self) -> None:
        for child in self.weights_inner.winfo_children():
            child.destroy()
        self._weight_vars.clear()

        wdir = self.ab_weights_dir.get().strip()
        if not wdir:
            root = self._configured_root()
            if root:
                wdir = str(root / "assets" / "weights")
                self.ab_weights_dir.set(wdir)
        folder = Path(wdir) if wdir else None
        if folder is None or not folder.is_dir():
            ttk.Label(self.weights_inner, text="(weights folder not found — set path and Refresh)").pack(
                anchor="w"
            )
            return

        filt = self.ab_filter.get().strip().lower()
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pth")
        if filt:
            files = [p for p in files if filt in p.name.lower()]
        if not files:
            ttk.Label(self.weights_inner, text="(no .pth files)").pack(anchor="w")
            return
        for p in files:
            var = tk.BooleanVar(value=False)
            self._weight_vars[str(p)] = var
            ttk.Checkbutton(self.weights_inner, text=p.name, variable=var).pack(anchor="w")

    def _set_all_weights(self, value: bool) -> None:
        for var in self._weight_vars.values():
            var.set(value)

    def _select_every_10_epochs(self) -> None:
        """Select weights whose epoch number (e_NNN or _eNNN_) is divisible by 10."""
        for path, var in self._weight_vars.items():
            name = Path(path).stem
            m = re.search(r"(?:^|[_\-])e(\d+)(?:[_\-]|$)", name, re.IGNORECASE)
            if not m:
                # Also try trailing _NNN
                m = re.search(r"_(\d+)$", name)
            if m:
                epoch = int(m.group(1))
                var.set(epoch % 10 == 0 or epoch == 0)
            else:
                var.set(False)

    def _selected_weights(self) -> list[str]:
        return [p for p, v in self._weight_vars.items() if v.get()]

    def _refresh_ab_devices(self) -> None:
        devices: list[str] = []
        try:
            import sounddevice as sd

            for i, d in enumerate(sd.query_devices()):
                if int(d.get("max_output_channels", 0) or 0) > 0:
                    devices.append(f"{i}: {d['name']}")
        except Exception as exc:
            devices = [f"(sounddevice unavailable: {exc})"]
        self.ab_device_combo["values"] = devices
        cur = self.ab_device.get()
        if devices and cur not in devices:
            # Prefer default output if present
            try:
                import sounddevice as sd

                default = sd.default.device
                out_idx = default[1] if isinstance(default, (list, tuple)) else default
                for item in devices:
                    if item.startswith(f"{out_idx}:"):
                        self.ab_device.set(item)
                        break
                else:
                    self.ab_device.set(devices[0])
            except Exception:
                self.ab_device.set(devices[0])

    def _reload_ab_results(self) -> None:
        out_dir = self.ab_out_dir.get().strip()
        if not out_dir:
            out_dir = self._rvc_path("logs", "my_exp", "infer_ab_test")
            self.ab_out_dir.set(out_dir)
        man = Path(out_dir) / "manifest.json"
        labels: list[str] = []
        self._result_paths = []
        if man.is_file():
            try:
                data = json.loads(man.read_text(encoding="utf-8"))
                for item in data.get("items", []):
                    out = item.get("out") or ""
                    weight = item.get("weight") or Path(out).name
                    if out and Path(out).is_file():
                        labels.append(f"{weight} -> {Path(out).name}")
                        self._result_paths.append(out)
                src = data.get("source_copy") or data.get("input")
                if src and Path(src).is_file():
                    labels.insert(0, f"[source] {Path(src).name}")
                    self._result_paths.insert(0, src)
            except Exception as exc:
                messagebox.showerror("Manifest error", str(exc))
        else:
            # Fall back to wav listing
            folder = Path(out_dir)
            if folder.is_dir():
                for p in sorted(folder.glob("*.wav")):
                    labels.append(p.name)
                    self._result_paths.append(str(p))
        self.ab_result_combo["values"] = labels
        if labels:
            self.ab_result_combo.current(0)

    def _play_ab_result(self) -> None:
        self._stop_playback()
        idx = self.ab_result_combo.current()
        if idx < 0 or idx >= len(self._result_paths):
            messagebox.showwarning("No result", "Select a result or Reload from manifest.")
            return
        path = self._result_paths[idx]
        try:
            import sounddevice as sd
            import soundfile as sf

            data, sr = sf.read(path, always_2d=True)
            device = None
            raw = self.ab_device.get().strip()
            if raw and ":" in raw:
                try:
                    device = int(raw.split(":", 1)[0].strip())
                except ValueError:
                    device = None
            sd.play(data, sr, device=device)
            self._playback = (data, sr)
        except Exception as exc:
            messagebox.showerror("Playback error", str(exc))

    def _stop_playback(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        self._playback = None

    # ---- Tab 7: Infer + Merge ----

    def _build_tab_infer_merge(self, parent: ttk.Frame) -> None:
        infer = ttk.LabelFrame(parent, text="Long infer", padding=10)
        infer.pack(fill="x", padx=4, pady=4)
        self._configure_cols(infer)

        self._row_path(infer, 0, "model (.pth)", self.im_model_path, self._browse_im_model)
        self._row_path(infer, 1, "index (.index)", self.im_index_path, self._browse_im_index)
        self._row_path(infer, 2, "input_path", self.im_input_path, self._browse_im_input)
        self._row_path(
            infer, 3, "infer_output_dir", self.im_infer_output_dir, self._browse_im_infer_out_dir
        )
        self._row_readonly(infer, 4, "opt_path", self.im_opt_path)
        self._row_readonly(infer, 5, "model_name", self.im_model_name)

        params = ttk.LabelFrame(parent, text="Infer parameters", padding=10)
        params.pack(fill="x", padx=4, pady=4)
        for c in range(5):
            params.columnconfigure(c, weight=1)

        ttk.Label(params, text="f0up_key").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_f0up_key, width=10).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="f0method").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            params,
            textvariable=self.im_f0method,
            values=["rmvpe", "pm", "fcpe"],
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(params, text="index_rate").grid(row=1, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_index_rate, width=10).grid(
            row=1, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="protect").grid(row=1, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_protect, width=10).grid(
            row=1, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="rms_mix_rate").grid(row=2, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_rms_mix_rate, width=10).grid(
            row=2, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="resample_sr (0=model)").grid(row=2, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_resample_sr, width=10).grid(
            row=2, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="chunk_sec").grid(row=3, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_chunk_sec, width=10).grid(
            row=3, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="overlap_sec").grid(row=3, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_overlap_sec, width=10).grid(
            row=3, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="spk_id").grid(row=4, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_spk_id, width=10).grid(
            row=4, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="filter_radius (harvest only)").grid(
            row=4, column=2, sticky="w"
        )
        ttk.Entry(params, textvariable=self.im_filter_radius, width=10).grid(
            row=4, column=3, sticky="w", padx=4
        )

        self.im_infer_btn = ttk.Button(params, text="Start Infer", command=self.run_infer_long)
        self.im_infer_btn.grid(row=5, column=3, sticky="e", padx=(10, 0), pady=8)

        ttk.Label(
            params,
            text=(
                "Suggested for speech: index_rate 0.5-0.75, rms_mix_rate 0.25-0.5, "
                "protect 0.33. filter_radius unused with rmvpe."
            ),
            wraplength=880,
        ).grid(row=6, column=0, columnspan=5, sticky="w", pady=(4, 0))

        merge = ttk.LabelFrame(parent, text="Merge with BGM", padding=10)
        merge.pack(fill="x", padx=4, pady=4)
        self._configure_cols(merge)

        self._row_path(merge, 0, "infer_result", self.im_infer_result, self._browse_im_infer_result)
        self._row_path(merge, 1, "bgm_path", self.im_bgm_path, self._browse_im_bgm)
        self._row_path(
            merge, 2, "merge_output_dir", self.im_merge_output_dir, self._browse_im_merge_out_dir
        )
        self._row_readonly(merge, 3, "merge_output_path", self.im_merge_output_path)

        self.im_merge_btn = ttk.Button(merge, text="Start Merge", command=self.run_merge)
        self.im_merge_btn.grid(row=4, column=4, sticky="e", padx=(10, 0), pady=4)

        ttk.Label(
            parent,
            text="Use Start Infer / Start Merge on this tab. Global Start is disabled here.",
            wraplength=900,
        ).pack(anchor="w", padx=8, pady=6)

        for var in (
            self.im_model_path,
            self.im_model_name,
            self.im_input_path,
            self.im_infer_output_dir,
            self.im_infer_result,
            self.im_merge_output_dir,
        ):
            var.trace_add("write", lambda *_a: self._refresh_im_auto_paths())

    def _browse_im_model(self) -> None:
        root = self._configured_root()
        initial = str(root / "assets" / "weights") if root else None
        path = filedialog.askopenfilename(
            title="Select model .pth",
            initialdir=initial,
            filetypes=PTH_FILETYPES,
        )
        if not path:
            return
        self.im_model_path.set(path)
        self.im_model_name.set(Path(path).name)
        self._autofill_im_index()
        self._refresh_im_auto_paths()

    def _browse_im_index(self) -> None:
        root = self._configured_root()
        initial = str(root / "assets" / "indices") if root else None
        path = filedialog.askopenfilename(
            title="Select index file",
            initialdir=initial,
            filetypes=[("FAISS index", "*.index"), ("All Files", "*.*")],
        )
        if path:
            self.im_index_path.set(path)

    def _autofill_im_index(self) -> None:
        model = self.im_model_path.get().strip() or self.im_model_name.get().strip()
        if not model:
            return
        root = self._configured_root()
        if root is None:
            try:
                root = resolve_rvc_root(self.rvc_root.get().strip() or None)
            except FileNotFoundError:
                return
        found = find_index_for_model(model, root)
        self.im_index_path.set(found or "")

    def _browse_im_input(self) -> None:
        path = filedialog.askopenfilename(title="Select input audio", filetypes=AUDIO_FILETYPES)
        if path:
            self.im_input_path.set(path)
            self._refresh_im_auto_paths()

    def _browse_im_infer_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Select infer output folder")
        if path:
            self.im_infer_output_dir.set(path)
            self._refresh_im_auto_paths()

    def _browse_im_infer_result(self) -> None:
        path = filedialog.askopenfilename(title="Select infer result", filetypes=AUDIO_FILETYPES)
        if path:
            self.im_infer_result.set(path)
            self._refresh_im_merge_output_path()

    def _browse_im_bgm(self) -> None:
        path = filedialog.askopenfilename(title="Select BGM audio", filetypes=AUDIO_FILETYPES)
        if path:
            self.im_bgm_path.set(path)

    def _browse_im_merge_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Select merge output folder")
        if path:
            self.im_merge_output_dir.set(path)
            self._refresh_im_merge_output_path()

    def _refresh_im_auto_paths(self) -> None:
        input_path = self.im_input_path.get().strip()
        model_name = self.im_model_name.get().strip()
        output_dir = self.im_infer_output_dir.get().strip()
        if not output_dir:
            output_dir = self._rvc_path("logs", "my_exp", "infer_long")
            if output_dir:
                self.im_infer_output_dir.set(output_dir)
        if input_path and model_name and output_dir:
            out_name = f"{Path(model_name).stem}_{Path(input_path).stem}.wav"
            out_path = str(Path(output_dir) / out_name)
            self.im_opt_path.set(out_path)
            if not self.im_infer_result.get().strip():
                self.im_infer_result.set(out_path)
        self._refresh_im_merge_output_path()

    def _refresh_im_merge_output_path(self) -> None:
        src = self.im_infer_result.get().strip() or self.im_opt_path.get().strip()
        out_dir = self.im_merge_output_dir.get().strip()
        if not out_dir:
            out_dir = self._rvc_path("logs", "my_exp", "merged")
        if not src or not out_dir:
            return
        src_name = Path(src).name
        if "(Vocals)" in src_name:
            merged_name = src_name.replace("(Vocals)", "(Merged)")
        else:
            stem = Path(src_name).stem
            merged_name = f"{stem}(Merged){Path(src_name).suffix or '.wav'}"
        self.im_merge_output_path.set(str(Path(out_dir) / merged_name))

    # ------------------------------------------------------------------
    # Logging / process control
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(msg)
        self.root.after(80, self._drain_log_queue)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.start_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")
        if hasattr(self, "im_infer_btn"):
            st = "disabled" if running else "normal"
            self.im_infer_btn.config(state=st)
            self.im_merge_btn.config(state=st)
        self.status_var.set("Running…" if running else "Idle")
        if not running:
            self._update_global_start_state()

    def _on_tab_changed(self, _event: tk.Event | None = None) -> None:
        self._update_global_start_state()
        self._save_settings()

    def _update_global_start_state(self) -> None:
        if self.running:
            return
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        # Global Start disabled on Settings and Infer+Merge (tab-local buttons)
        if idx in (TAB_SETTINGS, TAB_INFER_MERGE):
            self.start_btn.config(state="disabled")
        else:
            self.start_btn.config(state="normal")

    def _creationflags(self) -> int:
        if sys.platform == "win32":
            return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        return 0

    def _kill_process(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if sys.platform == "win32" and proc.pid:
            try:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=self._creationflags(),
                )
            except Exception as exc:
                self.log_queue.put(f"\n[stop error] {exc}\n")
        else:
            try:
                proc.terminate()
            except Exception:
                pass
        self.proc = None

    def stop_job(self) -> None:
        if not self.running:
            return
        self.log_queue.put("\n[stop] killing process tree…\n")
        self._kill_process()
        self.status_var.set("Stopping…")

    def _run_cmd(
        self,
        cmd: list[str],
        rvc_root: Path,
        on_success: Callable[[], None] | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        if self.running:
            messagebox.showwarning("Busy", "Wait for the current job to finish, or Stop it.")
            return

        env = prepare_rvc_process_env(rvc_root)
        env["PYTHONUNBUFFERED"] = "1"
        if env_extra:
            env.update(env_extra)

        def worker() -> None:
            self.log_queue.put("\n" + "=" * 72 + "\n")
            self.log_queue.put(
                "COMMAND:\n"
                + " ".join(f'"{c}"' if " " in c else c for c in cmd)
                + "\n"
            )
            self.log_queue.put(f"cwd: {rvc_root}\n")
            self.log_queue.put("=" * 72 + "\n")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(rvc_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=self._creationflags(),
                )
                self.proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(line)
                code = proc.wait()
                self.log_queue.put(f"\n[exit_code] {code}\n")
                if code == 0 and on_success is not None:
                    self.root.after(0, on_success)
            except Exception as exc:
                self.log_queue.put(f"\n[error] {exc}\n")
            finally:
                self.proc = None
                self.root.after(0, lambda: self._set_running(False))

        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Job builders
    # ------------------------------------------------------------------

    def start_current_tab(self) -> None:
        idx = self.notebook.index(self.notebook.select())
        if idx == TAB_SETTINGS:
            messagebox.showinfo("Settings", "Configure RVC root here; use other tabs to run jobs.")
            return
        if idx == TAB_INFER_MERGE:
            messagebox.showinfo(
                "Infer + Merge",
                "Use Start Infer or Start Merge on this tab.",
            )
            return
        root = self._require_rvc_root()
        if root is None:
            return
        builders = {
            TAB_PREPROCESS: self._cmd_preprocess,
            TAB_EXTRACT_F0: self._cmd_extract_f0,
            TAB_EXTRACT_HUBERT: self._cmd_extract_hubert,
            TAB_TRAIN: self._cmd_train,
            TAB_BUILD_INDEX: self._cmd_build_index,
            TAB_INFER_AB: self._cmd_infer_ab,
        }
        builder = builders.get(idx)
        if builder is None:
            return
        cmd = builder(root)
        if cmd is None:
            return
        on_success = None
        if idx == TAB_INFER_AB:
            on_success = self._reload_ab_results
        self._run_cmd(cmd, root, on_success=on_success)

    def _cmd_preprocess(self, root: Path) -> list[str] | None:
        inp = self.pp_inp_root.get().strip()
        exp = self.pp_exp_dir.get().strip() or "logs/my_exp"
        if not inp or not Path(inp).is_dir():
            messagebox.showerror("Missing input", "Select a valid input audio directory.")
            return None
        exp_path = Path(exp)
        if not exp_path.is_absolute():
            exp_path = (root / exp).resolve()
        else:
            exp_path = exp_path.resolve()
        py = str(rvc_python(root))
        script = str(SCRIPTS_DIR / "preprocess_flex.py")
        cmd = [
            py,
            script,
            "--inp_root",
            inp,
            "--exp_dir",
            str(exp_path),
            "--sr",
            self.pp_sr.get().strip(),
            "--n_p",
            self.pp_n_p.get().strip(),
            "--per",
            self.pp_per.get().strip(),
            "--overlap",
            self.pp_overlap.get().strip(),
            "--highpass_hz",
            self.pp_highpass_hz.get().strip(),
            "--threshold",
            self.pp_threshold.get().strip(),
            "--min_length",
            self.pp_min_length.get().strip(),
            "--min_interval",
            self.pp_min_interval.get().strip(),
            "--hop_size",
            self.pp_hop_size.get().strip(),
            "--max_sil_kept",
            self.pp_max_sil_kept.get().strip(),
            "--norm_max",
            self.pp_norm_max.get().strip(),
            "--norm_alpha",
            self.pp_norm_alpha.get().strip(),
            "--peak_reject",
            self.pp_peak_reject.get().strip(),
        ]
        if self.pp_noparallel.get():
            cmd.append("--noparallel")
        if not self.pp_highpass.get():
            cmd.append("--no_highpass")
        return cmd

    def _cmd_extract_f0(self, root: Path) -> list[str] | None:
        exp = self.exp_name(self.f0_exp.get())
        if not exp:
            messagebox.showerror("Missing exp", "Enter an experiment name.")
            return None
        py = str(rvc_python(root))
        script = str(root / "train" / "dataset" / "extract_f0.py")
        is_half = "True" if self.f0_is_half.get() else "False"
        return [
            py,
            script,
            "cuda",
            "1",
            "0",
            self.f0_gpu.get().strip() or "0",
            exp,
            is_half,
        ]

    def _cmd_extract_hubert(self, root: Path) -> list[str] | None:
        exp = self.exp_name(self.hb_exp.get())
        if not exp:
            messagebox.showerror("Missing exp", "Enter an experiment name.")
            return None
        py = str(rvc_python(root))
        script = str(root / "train" / "dataset" / "extract_hubert_feature.py")
        is_half = "True" if self.hb_is_half.get() else "False"
        return [
            py,
            script,
            "cuda",
            "1",
            "0",
            self.hb_gpu.get().strip() or "0",
            exp,
            self.hb_version.get().strip() or "v2",
            is_half,
        ]

    def _cmd_train(self, root: Path) -> list[str] | None:
        exp = self.exp_name(self.tr_exp.get()) or "my_exp"
        py = str(rvc_python(root))
        script = str(SCRIPTS_DIR / "train_flex.py")
        cmd = [
            py,
            script,
            "--exp_dir",
            f"logs/{exp}",
            "--sample_rate",
            self.tr_sample_rate.get().strip(),
            "--version",
            self.tr_version.get().strip(),
            "--spk_id",
            self.tr_spk_id.get().strip(),
            "--batch_size",
            self.tr_batch_size.get().strip(),
            "--total_epoch",
            self.tr_total_epoch.get().strip(),
            "--save_every_epoch",
            self.tr_save_every_epoch.get().strip(),
            "--gpus",
            self.tr_gpus.get().strip() or "0",
        ]
        if self.tr_if_f0.get():
            cmd.append("--if_f0")
        else:
            cmd.append("--no-if_f0")
        if self.tr_save_latest_only.get():
            cmd.append("--save_latest_only")
        else:
            cmd.append("--no-save_latest_only")
        if self.tr_cache_in_gpu.get():
            cmd.append("--cache_in_gpu")
        else:
            cmd.append("--no-cache_in_gpu")
        if self.tr_save_every_weights.get():
            cmd.append("--save_every_weights")
        else:
            cmd.append("--no-save_every_weights")
        pg = self.tr_pretrain_g.get().strip()
        pd = self.tr_pretrain_d.get().strip()
        if pg:
            cmd.extend(["--pretrain_g", pg])
        if pd:
            cmd.extend(["--pretrain_d", pd])
        return cmd

    def _cmd_build_index(self, root: Path) -> list[str] | None:
        exp = self.exp_name(self.ix_exp.get())
        if not exp:
            messagebox.showerror("Missing exp", "Enter an experiment name.")
            return None
        outside = self.ix_outside.get().strip() or str(root / "assets" / "indices")
        Path(outside).mkdir(parents=True, exist_ok=True)
        py = str(rvc_python(root))
        script = str(root / "train" / "train_index.py")
        return [
            py,
            script,
            exp,
            self.ix_version.get().strip() or "v2",
            outside,
            self.ix_n_cpu.get().strip() or "4",
        ]

    def _cmd_infer_ab(self, root: Path) -> list[str] | None:
        inp = self.ab_input.get().strip()
        if not inp or not Path(inp).is_file():
            messagebox.showerror("Missing input", "Select a valid test audio file.")
            return None
        weights = self._selected_weights()
        if not weights:
            messagebox.showerror("No weights", "Refresh and check at least one .pth weight.")
            return None
        out_dir = self.ab_out_dir.get().strip() or str(root / "logs" / "my_exp" / "infer_ab_test")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        py = str(rvc_python(root))
        script = str(SCRIPTS_DIR / "infer_batch_test.py")
        cmd = [
            py,
            script,
            "--input",
            inp,
            "--weights",
            *weights,
            "--out_dir",
            out_dir,
            "--spk_id",
            self.ab_spk.get().strip() or "0",
            "--f0_up_key",
            self.ab_pitch.get().strip() or "0",
            "--f0_method",
            self.ab_f0_method.get().strip() or "rmvpe",
            "--index_rate",
            self.ab_index_rate.get().strip() or "0.75",
            "--protect",
            self.ab_protect.get().strip() or "0.33",
            "--rms_mix_rate",
            self.ab_rms.get().strip() or "0.25",
            "--resample_sr",
            self.ab_resample.get().strip() or "0",
        ]
        index = self.ab_index.get().strip()
        if index:
            cmd.extend(["--index", index])
        return cmd

    def run_infer_long(self) -> None:
        root = self._require_rvc_root()
        if root is None:
            return
        self._refresh_im_auto_paths()
        model_path = self.im_model_path.get().strip()
        input_path = self.im_input_path.get().strip()
        opt_path = self.im_opt_path.get().strip()
        model_name = self.im_model_name.get().strip()
        index_path = self.im_index_path.get().strip()

        if not model_path or not Path(model_path).is_file():
            messagebox.showerror("Missing model", "Browse and select a valid .pth model.")
            return
        if not input_path or not Path(input_path).is_file():
            messagebox.showerror("Missing input", "Select a valid input_path.")
            return
        if not opt_path:
            messagebox.showerror("Missing opt_path", "Set model and input_path first.")
            return
        if not model_name:
            model_name = Path(model_path).name
            self.im_model_name.set(model_name)
        if index_path and not Path(index_path).is_file():
            messagebox.showerror("Missing index", f"Index file not found:\n{index_path}")
            return

        Path(opt_path).parent.mkdir(parents=True, exist_ok=True)
        py = str(rvc_python(root))
        script = str(SCRIPTS_DIR / "infer_long.py")
        cmd = [
            py,
            script,
            "--input_path",
            input_path,
            "--opt_path",
            opt_path,
            "--model_name",
            model_name,
            "--f0up_key",
            self.im_f0up_key.get().strip() or "0",
            "--f0method",
            self.im_f0method.get().strip() or "rmvpe",
            "--index_rate",
            self.im_index_rate.get().strip() or "0.75",
            "--protect",
            self.im_protect.get().strip() or "0.33",
            "--rms_mix_rate",
            self.im_rms_mix_rate.get().strip() or "0.25",
            "--resample_sr",
            self.im_resample_sr.get().strip() or "0",
            "--spk_id",
            self.im_spk_id.get().strip() or "0",
            "--chunk_sec",
            self.im_chunk_sec.get().strip() or "200",
            "--overlap_sec",
            self.im_overlap_sec.get().strip() or "0.3",
            "--is_half",
            "false",
        ]
        if index_path:
            cmd.extend(["--index_path", index_path])
        # filter_radius is kept in UI for harvest workflows; current VC API ignores it.

        def on_success() -> None:
            self.im_infer_result.set(opt_path)
            self._refresh_im_merge_output_path()
            messagebox.showinfo("Infer done", f"Output:\n{opt_path}")

        self._run_cmd(
            cmd,
            root,
            on_success=on_success,
            env_extra={"weight_root": str(Path(model_path).parent)},
        )

    def run_merge(self) -> None:
        root = self._require_rvc_root()
        if root is None:
            return
        infer_result = self.im_infer_result.get().strip()
        bgm_path = self.im_bgm_path.get().strip()
        merge_out = self.im_merge_output_path.get().strip()

        if not infer_result or not Path(infer_result).is_file():
            messagebox.showerror("Missing result", "Select a valid infer result file.")
            return
        if not bgm_path or not Path(bgm_path).is_file():
            messagebox.showerror("Missing BGM", "Select a valid background audio file.")
            return
        if not merge_out:
            messagebox.showerror("Missing output", "merge_output_path is empty.")
            return

        Path(merge_out).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(merge_out).suffix.lower()
        if ext == ".flac":
            codec_args = ["-c:a", "flac", "-compression_level", "8"]
        elif ext == ".wav":
            codec_args = ["-c:a", "pcm_s16le"]
        elif ext == ".mp3":
            codec_args = ["-c:a", "libmp3lame", "-b:a", "320k"]
        elif ext in {".m4a", ".mp4", ".aac"}:
            codec_args = ["-c:a", "aac", "-b:a", "320k"]
        else:
            codec_args = []

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            infer_result,
            "-i",
            bgm_path,
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:weights=1 1:normalize=0[a]",
            "-map",
            "[a]",
            *codec_args,
            merge_out,
        ]

        def on_success() -> None:
            messagebox.showinfo("Merge done", f"Output:\n{merge_out}")

        self._run_cmd(cmd, root, on_success=on_success)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    _enable_dpi_awareness()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
