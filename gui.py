#!/usr/bin/env python3
"""RVC training / inference pipeline GUI (portable paths)."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

# Ensure this package directory is importable even under `python -I`.
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from audio_kind import KIND_EDIT_VALUES, KIND_FILTER_VALUES, kind_from_label, kind_label
from playback import (
    AudioPlayer,
    PlaybackError,
    SYSTEM_DEFAULT_DEVICE,
    device_label_matches,
    format_time_ms,
    list_output_devices,
    pick_default_device_label,
)
from convert_match import (
    build_convert_count_map,
    find_convert_results_for_source,
    find_source_for_vocals,
)
from source_match import find_separated_for_source
from app_db import DB_PATH, connect, load_settings_map, save_settings_map
from stem_links import (
    SCORE_MAX,
    add_convert_result,
    format_score_stars,
    get_source_note,
    get_source_score,
    load_stem_links,
    save_stem_links,
    score_from_click_x,
    set_source_note,
    set_source_score,
    source_key,
    upsert_stem_link,
)
from shared_catalog import (
    SHARED_CATALOG_PATH,
    apply_catalog_to_state,
    build_catalog_from_state,
    load_shared_catalog,
    save_shared_catalog,
    set_kind_for_name,
    set_note_for_stem,
    set_score_for_stem,
)
from audio_scan import (
    AudioFileRow,
    SRC_SORT_LABELS,
    apply_duration_updates,
    count_by_kind,
    filter_rows,
    format_duration,
    format_size,
    load_scan_cache,
    merge_rescan_rows,
    paths_needing_duration,
    probe_duration_sec,
    save_scan_cache,
    scan_audio_roots,
    sort_key_from_label,
    sort_label,
    sort_rows,
    update_row_kind,
)
from rvc_env import (
    PACKAGE_DIR,
    SCRIPTS_DIR,
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
TAB_AUDIO_SCAN = 1
TAB_SOURCE = 2
TAB_PROCESS = 3
TAB_SEPARATE = 4
TAB_CONVERT = 5

TAB_NAMES = [
    "Settings",
    "Audio Scan",
    "Source Audio",
    "Process",
    "Separate",
    "Convert",
]

SRC_TREE_COLUMNS = ("name", "size", "duration", "note", "score", "converted", "path")
SRC_TREE_COL_DEFAULTS: dict[str, int] = {
    "name": 200,
    "size": 70,
    "duration": 72,
    "note": 160,
    "score": 64,
    "converted": 48,
    "path": 360,
}

SEP_MODELS = {
    "MelBand-RoFormer (recommended)": "vocals_mel_band_roformer.ckpt",
    "BS-RoFormer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}
DEFAULT_SEP_MODEL_DIR = PACKAGE_DIR / "models"
DEFAULT_SEP_VENV = PACKAGE_DIR / ".venv"

BASE_TITLE = "RVC Pipeline"


def reveal_path_in_file_manager(path: Path | str) -> None:
    target = Path(path).expanduser()
    try:
        target = target.resolve()
    except OSError:
        pass
    if not target.exists():
        raise FileNotFoundError(target)
    if sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(target)], check=False)
        return
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(target)], check=False)
        return
    folder = target if target.is_dir() else target.parent
    subprocess.run(["xdg-open", str(folder)], check=False)

# Progress lines emitted by our scripts / RVC train & extract helpers.
_RE_PROGRESS_FRAC = re.compile(
    r"(?:"
    r"\[progress\]\s*(\d+)\s*/\s*(\d+)"
    r"|进度[：:]\s*(\d+)\s*/\s*(\d+)"
    r"|Write progress:\s*(\d+)\s*/\s*(\d+)"
    r"|写入进度[：:]\s*(\d+)\s*/\s*(\d+)"
    r"|\[Infer\]\s*\((\d+)\s*/\s*(\d+)\)"
    r"|\b(\d+)\s*/\s*(\d+)\s*\["  # tqdm: 192/200 [03:00<...
    r")",
    re.IGNORECASE,
)
_RE_TQDM_PCT = re.compile(r"(?:^|\s)(\d+)%\|")
_RE_TRAIN_EPOCH_PCT = re.compile(
    r"(?:训练轮次|Train(?:ing)?\s*epoch)[：:\s]+(\d+)\s*\[(\d+(?:\.\d+)?)%\]",
    re.IGNORECASE,
)
_RE_DURATION_SEC = re.compile(r"duration_sec=([\d.]+)")
_RE_CHUNK_START = re.compile(
    r"\[chunk\s+(\d+)\]\s+start=([\d.]+)s",
    re.IGNORECASE,
)


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
            "converted_bg": "#d8f0d8",
            "converted_fg": "#0d3d0d",
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
            "converted_bg": "#1e3d28",
            "converted_fg": "#b8f0c0",
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
        arrowcolor=colors["fg"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", colors["field"]), ("disabled", colors["field"])],
        foreground=[("readonly", colors["fg"]), ("disabled", colors["disabled"])],
        # Without these, readonly Combobox text is often blank until a second click
        # (Windows / clam dark themes select the whole field with invisible colors).
        selectbackground=[("readonly", colors["field"]), ("!readonly", colors["select"])],
        selectforeground=[("readonly", colors["fg"]), ("!readonly", colors["select_fg"])],
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
    style.configure(
        "Horizontal.TScrollbar",
        background=colors["button"],
        troughcolor=colors["bg"],
        arrowcolor=colors["fg"],
    )
    style.configure(
        "TProgressbar",
        background=colors["select"],
        troughcolor=colors["button"],
        bordercolor=colors["border"],
        lightcolor=colors["select"],
        darkcolor=colors["select"],
    )
    style.configure(
        "Horizontal.TScale",
        background=colors["bg"],
        troughcolor=colors["button"],
    )
    style.configure("TSeparator", background=colors["border"])
    # Treeview: clam defaults to light field colors; force theme-aware contrast.
    style.configure(
        "Treeview",
        background=colors["field"],
        foreground=colors["fg"],
        fieldbackground=colors["field"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        rowheight=24,
    )
    style.configure(
        "Treeview.Heading",
        background=colors["button"],
        foreground=colors["fg"],
        relief="flat",
        bordercolor=colors["border"],
    )
    style.map(
        "Treeview",
        background=[("selected", colors["select"])],
        foreground=[("selected", colors["select_fg"])],
    )
    style.map(
        "Treeview.Heading",
        background=[("active", colors["select"])],
        foreground=[("active", colors["select_fg"])],
    )
    # Tk Treeview ignores custom colors unless '!disabled' map entries are stripped.
    def _tree_map(option: str):
        return [
            elm
            for elm in style.map("Treeview", query_opt=option)
            if elm[:2] != ("!disabled", "!selected")
        ]

    try:
        style.map(
            "Treeview",
            foreground=_tree_map("foreground"),
            background=_tree_map("background"),
        )
    except tk.TclError:
        pass
    try:
        root.option_add("*Treeview*background", colors["field"])
        root.option_add("*Treeview*foreground", colors["fg"])
        root.option_add("*Treeview*fieldBackground", colors["field"])
    except tk.TclError:
        pass
    _configure_action_button_styles(style, light)
    return colors


BTN_STYLE_PROCESS = "Process.TButton"
BTN_STYLE_SEPARATE = "Separate.TButton"
BTN_STYLE_CONVERT = "Convert.TButton"
BTN_STYLE_SCAN = "Scan.TButton"


def _action_button_font() -> tuple[str, int, str]:
    if sys.platform == "win32":
        return ("Segoe UI", 10, "bold")
    return ("TkDefaultFont", 10, "bold")


def _configure_action_button_styles(style: ttk.Style, light: bool) -> None:
    font = _action_button_font()
    pad = [18, 8]
    if light:
        specs = {
            BTN_STYLE_PROCESS: ("#2563eb", "#ffffff", "#1d4ed8", "#93c5fd"),
            BTN_STYLE_SEPARATE: ("#ea580c", "#ffffff", "#c2410c", "#fdba74"),
            BTN_STYLE_CONVERT: ("#059669", "#ffffff", "#047857", "#6ee7b7"),
            BTN_STYLE_SCAN: ("#7c3aed", "#ffffff", "#6d28d9", "#c4b5fd"),
        }
        disabled_bg = "#c8c8c8"
        disabled_fg = "#888888"
    else:
        specs = {
            BTN_STYLE_PROCESS: ("#3b82f6", "#ffffff", "#2563eb", "#60a5fa"),
            BTN_STYLE_SEPARATE: ("#f97316", "#ffffff", "#ea580c", "#fb923c"),
            BTN_STYLE_CONVERT: ("#10b981", "#ffffff", "#059669", "#34d399"),
            BTN_STYLE_SCAN: ("#8b5cf6", "#ffffff", "#7c3aed", "#a78bfa"),
        }
        disabled_bg = "#4a4a4a"
        disabled_fg = "#9a9a9a"

    for name, (bg, fg, active_bg, active_fg) in specs.items():
        style.configure(
            name,
            font=font,
            padding=pad,
            background=bg,
            foreground=fg,
            borderwidth=1,
            focusthickness=2,
            focuscolor=active_bg,
        )
        style.map(
            name,
            background=[("disabled", disabled_bg), ("active", active_bg), ("pressed", active_bg)],
            foreground=[("disabled", disabled_fg), ("active", active_fg), ("pressed", active_fg)],
        )


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
        self.root.title(BASE_TITLE)
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
        self._suppress_autosave = True
        self._settings_hydrated = False
        self._geometry_save_after_id: str | None = None
        self._player = AudioPlayer()
        self._playback_after_id: str | None = None
        self._pb_ignore_seek = False
        self._pb_seeking = False
        self._wrap_labels: list[tuple[tk.Misc, float]] = []
        self._job_pct: float | None = None
        self._job_total_epoch: int | None = None
        self._job_audio_duration: float | None = None
        self.exp_locked = False
        self._exp_lock_entries: list[ttk.Entry] = []
        self._exp_lock_buttons: list[ttk.Button] = []

        # --- shared / settings ---
        self.rvc_root = tk.StringVar(value="")
        self.active_exp_path = tk.StringVar(value="")
        self.python_path_display = tk.StringVar(value="(set RVC root first)")
        self.favorite_experiments: list[str] = []
        self.favorite_models: list[str] = []
        self.fav_exp_pick = tk.StringVar(value="")
        self.fav_model_pick = tk.StringVar(value="")
        self.fav_exp_btn_label = tk.StringVar(value="★ Fav")
        self.fav_model_btn_label = tk.StringVar(value="★ Fav")
        self.status_var = tk.StringVar(value="Idle")
        self.last_tab = tk.IntVar(value=0)
        self.geometry_var = tk.StringVar(value="1080x780")

        # --- global playback ---
        self.pb_device = tk.StringVar(value="")
        self.pb_volume = tk.DoubleVar(value=80.0)

        # --- audio scan ---
        self.as_filter = tk.StringVar(value="")
        self.as_kind_filter = tk.StringVar(value="all")
        self.as_sort_label = tk.StringVar(value=sort_label("rel_path"))
        self.as_sort_desc = tk.BooleanVar(value=False)
        self.src_filter = tk.StringVar(value="")
        self.src_sort_label = tk.StringVar(value=sort_label("rel_path"))
        self.src_sort_desc = tk.BooleanVar(value=False)
        # Multi-select score filter: index 0..SCORE_MAX (☆☆☆ .. ★★★). All on = no filter.
        self.src_score_filters = [tk.BooleanVar(value=True) for _ in range(SCORE_MAX + 1)]
        self.audio_scan_roots: list[str] = []
        self._audio_scan_rows: list[AudioFileRow] = []
        self._audio_scan_running = False
        self._process_source_path: str | None = None
        self._convert_source_path: str | None = None
        self._src_note_editor: tk.Entry | None = None
        self._src_note_edit_item: str | None = None
        self._src_note_edit_path: str | None = None
        self._src_note_committing = False
        self._src_tree_col_widths: dict[str, int] = {}
        self._src_colwidth_save_after_id: str | None = None
        self._src_context_reveal_path: str | None = None
        self._db = connect()
        self._stem_links = load_stem_links(self._db)
        self._shared_catalog = load_shared_catalog()
        self._audio_scan_revision = 0
        self._stem_links_revision = 0
        self._src_display_cache_key: tuple[object, ...] | None = None
        self._src_display_cache_payload: list[tuple[AudioFileRow, str, int, int]] | None = None
        self._duration_backfill_running = False
        self._as_kind_editor: ttk.Combobox | None = None
        self._as_kind_edit_item: str | None = None
        self._as_kind_edit_path: str | None = None
        self._as_kind_committing = False
        self._convert_infer_temp: str | None = None
        self._log_collapsed = True

        # --- infer + merge ---
        self.im_model_path = tk.StringVar(value="")
        self.im_model_name = tk.StringVar(value="")
        self.im_index_path = tk.StringVar(value="")
        self.im_input_path = tk.StringVar(value="")
        self.im_f0up_key = tk.StringVar(value="0")
        self.im_f0method = tk.StringVar(value="rmvpe")
        self.im_index_rate = tk.StringVar(value="0.95")
        self.im_filter_radius = tk.StringVar(value="0")
        self.im_resample_sr = tk.StringVar(value="0")
        self.im_rms_mix_rate = tk.StringVar(value="0.95")
        self.im_protect = tk.StringVar(value="0.4")
        self.im_breath_mix_rate = tk.StringVar(value="0.65")
        self.im_chunk_sec = tk.StringVar(value="200")
        self.im_overlap_sec = tk.StringVar(value="0.3")
        self.im_spk_id = tk.StringVar(value="0")
        self.im_merge_output_dir = tk.StringVar(value="")
        self.im_merge_output_path = tk.StringVar(value="")

        # --- separate (audio-separator / MelBand-RoFormer) ---
        self.sep_input_path = tk.StringVar(value="")
        self.sep_output_dir = tk.StringVar(value="")
        self.sep_model_label = tk.StringVar(value=next(iter(SEP_MODELS)))
        self.sep_format = tk.StringVar(value="FLAC")
        self.sep_segment = tk.StringVar(value="256")
        self.sep_model_dir = tk.StringVar(value=str(DEFAULT_SEP_MODEL_DIR))
        self.sep_venv_dir = tk.StringVar(value=str(DEFAULT_SEP_VENV))
        self.sep_proxy = tk.StringVar(value="")
        self.sep_fill_infer_merge = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_settings()
        self._wire_autosave()
        self._refresh_python_display()
        self._fill_empty_defaults_from_root()

        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._drain_log_queue)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.after_idle(self._refresh_wraplengths)

    def _invalidate_source_display_cache(self) -> None:
        self._src_display_cache_key = None
        self._src_display_cache_payload = None

    def _persist_stem_links(self) -> None:
        save_stem_links(self._stem_links, self._db)
        self._stem_links_revision += 1

    def _persist_shared_catalog(self) -> None:
        try:
            save_shared_catalog(self._shared_catalog)
        except OSError:
            pass

    def _apply_shared_catalog(self, *, persist_db: bool = True) -> tuple[int, int]:
        """Merge shared_catalog.json into live links/rows. Returns (notes, kinds) changed."""
        self._shared_catalog = load_shared_catalog()
        links, rows, n_notes, n_scores, n_kinds = apply_catalog_to_state(
            self._shared_catalog,
            self._stem_links,
            self._audio_scan_rows,
        )
        self._stem_links = links
        self._audio_scan_rows = rows
        if persist_db and (n_notes or n_scores or n_kinds):
            if n_notes or n_scores:
                self._persist_stem_links()
            if n_kinds:
                self._audio_scan_revision += 1
                try:
                    save_scan_cache(self._db, self._audio_scan_rows)
                except OSError:
                    pass
            self._invalidate_source_display_cache()
        return n_notes + n_scores, n_kinds

    def _sync_shared_catalog_from_db(self) -> Path:
        """Rebuild shared_catalog.json from current notes + kind overrides."""
        self._shared_catalog = build_catalog_from_state(
            self._stem_links,
            self._audio_scan_rows,
            only_kind_overrides=True,
        )
        return save_shared_catalog(self._shared_catalog)

    def _src_selected_scores(self) -> frozenset[int] | None:
        """Scores to show. None = show all (nothing or everything checked)."""
        selected = frozenset(
            i for i, var in enumerate(self.src_score_filters) if bool(var.get())
        )
        if not selected or selected == frozenset(range(SCORE_MAX + 1)):
            return None
        return selected

    def _source_display_cache_key(self) -> tuple[object, ...]:
        return (
            self.src_filter.get().strip().lower(),
            self._audio_scan_revision,
            self._stem_links_revision,
            len(self._audio_scan_rows),
        )

    def _build_source_display_payload(
        self, rows: list[AudioFileRow]
    ) -> list[tuple[AudioFileRow, str, int, int]]:
        paths = [r.path for r in rows]
        count_map = build_convert_count_map(paths, self._stem_links, self._audio_scan_rows)
        payload: list[tuple[AudioFileRow, str, int, int]] = []
        for r in rows:
            note = get_source_note(r.path, self._stem_links)
            score = get_source_score(r.path, self._stem_links)
            key = str(Path(r.path).resolve())
            cvt_n = count_map.get(key, 0)
            payload.append((r, note, cvt_n, score))
        return payload

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
        pass

    def _convert_tmp_dir(self) -> Path:
        d = PACKAGE_DIR / ".convert_tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _allocate_infer_temp_path(self, model_path: str, input_path: str) -> Path:
        model_stem = Path(model_path).stem
        input_stem = Path(input_path).stem
        fd, path = tempfile.mkstemp(
            suffix=".wav",
            prefix=f"infer_{model_stem}_{input_stem}_",
            dir=str(self._convert_tmp_dir()),
        )
        os.close(fd)
        return Path(path)

    def _merged_output_basename(self) -> str:
        """Build merge filename: ``{model}_{source_stem}_(Merged).{ext}`` when possible."""
        model_name = self.im_model_name.get().strip()
        model_path = self.im_model_path.get().strip()
        model_stem = (
            Path(model_name).stem
            if model_name
            else (Path(model_path).stem if model_path else "")
        )

        input_path = self.im_input_path.get().strip()
        source = (self._convert_source_path or "").strip()
        if source and Path(source).is_file():
            base_stem = Path(source).stem
            suffix = Path(input_path).suffix if input_path else Path(source).suffix
            if not suffix:
                suffix = ".flac"
            merged_name = f"{base_stem}_(Merged){suffix}"
        elif input_path:
            src_name = Path(input_path).name
            # MelBand stems: keep path but swap (Vocals) → (Merged) (any case).
            if re.search(r"\(vocals\)", src_name, flags=re.IGNORECASE):
                merged_name = re.sub(
                    r"\(vocals\)", "(Merged)", src_name, count=1, flags=re.IGNORECASE
                )
            else:
                suffix = Path(src_name).suffix or ".flac"
                merged_name = f"{Path(input_path).stem}_(Merged){suffix}"
        else:
            return ""

        if model_stem and not merged_name.lower().startswith(model_stem.lower() + "_"):
            merged_name = f"{model_stem}_{merged_name}"
        return merged_name

    def _register_exp_lock_widgets(self, *widgets: tk.Widget) -> None:
        for w in widgets:
            if isinstance(w, ttk.Button):
                self._exp_lock_buttons.append(w)
            elif isinstance(w, (ttk.Entry, tk.Entry)):
                self._exp_lock_entries.append(w)  # type: ignore[arg-type]

    def _set_exp_widgets_locked(self, locked: bool) -> None:
        entry_state = "readonly" if locked else "normal"
        btn_state = "disabled" if locked else "normal"
        for entry in self._exp_lock_entries:
            try:
                entry.configure(state=entry_state)
            except tk.TclError:
                pass
        for btn in self._exp_lock_buttons:
            try:
                btn.configure(state=btn_state)
            except tk.TclError:
                pass
        if hasattr(self, "active_exp_entry"):
            try:
                self.active_exp_entry.configure(state="readonly")
            except tk.TclError:
                pass

    def _apply_active_experiment(
        self,
        path: str,
        *,
        lock: bool = True,
        quiet: bool = False,
    ) -> bool:
        raw = (path or "").strip().strip('"')
        if not raw:
            if not quiet:
                messagebox.showerror("Missing experiment", "Select a logs/<name> folder.")
            return False
        try:
            p = Path(raw).expanduser().resolve()
        except Exception:
            if not quiet:
                messagebox.showerror("Invalid path", f"Cannot resolve:\n{raw}")
            return False
        if not p.is_dir():
            if not quiet:
                messagebox.showerror("Invalid experiment", f"Not a folder:\n{p}")
            return False
        name = self.exp_name(str(p))
        if not name:
            if not quiet:
                messagebox.showerror("Invalid experiment", "Could not derive experiment name.")
            return False
        parts = p.parts
        under_logs = len(parts) >= 2 and parts[-2].lower() == "logs"
        if not under_logs and not quiet:
            messagebox.showwarning(
                "Warning",
                "Selected folder is not under logs/. Using the folder name as the experiment name.",
            )

        was = self._suppress_autosave
        self._suppress_autosave = True
        try:
            self.active_exp_path.set(str(p))
            self._refresh_im_auto_paths()
            self.exp_locked = bool(lock)
            self._set_exp_widgets_locked(self.exp_locked)
        finally:
            self._suppress_autosave = was
        self._refresh_fav_exp_combo()
        self._save_settings()
        return True

    def _clear_active_experiment(self) -> None:
        was = self._suppress_autosave
        self._suppress_autosave = True
        try:
            self.exp_locked = False
            self.active_exp_path.set("")
            self.fav_exp_pick.set("")
            self._set_exp_widgets_locked(False)
        finally:
            self._suppress_autosave = was
        self._update_fav_exp_btn_label()
        self._save_settings()

    def _browse_active_exp(self) -> None:
        root = self._configured_root()
        fallback = str(root / "logs") if root else None
        path = filedialog.askdirectory(
            title="Select experiment folder (logs/<name>)",
            **self._path_dialog_opts(self.active_exp_path.get(), fallback=fallback),
        )
        if path:
            self._apply_active_experiment(path, lock=True)

    @staticmethod
    def _normalize_path_list(items: Any) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, str):
                continue
            p = item.strip().strip('"')
            if not p:
                continue
            try:
                key = str(Path(p).resolve())
            except Exception:
                key = p
            if key.lower() in seen:
                continue
            seen.add(key.lower())
            out.append(key)
        return out

    @staticmethod
    def _fav_display_label(path: str, *, is_dir: bool) -> str:
        p = Path(path)
        name = p.name or path
        # Include path so duplicates stay unique in the Combobox.
        return f"{name}  —  {path}"

    def _refresh_fav_exp_combo(self) -> None:
        labels = [
            self._fav_display_label(p, is_dir=True) for p in self.favorite_experiments
        ]
        if hasattr(self, "fav_exp_combo"):
            self.fav_exp_combo["values"] = labels
        cur = self.active_exp_path.get().strip()
        if cur:
            try:
                cur_key = str(Path(cur).resolve())
            except Exception:
                cur_key = cur
            for i, p in enumerate(self.favorite_experiments):
                if p.lower() == cur_key.lower():
                    self.fav_exp_pick.set(labels[i])
                    break
        self._update_fav_exp_btn_label()

    def _refresh_fav_model_combo(self) -> None:
        labels = [
            self._fav_display_label(p, is_dir=False) for p in self.favorite_models
        ]
        if hasattr(self, "fav_model_combo"):
            self.fav_model_combo["values"] = labels
        cur = self.im_model_path.get().strip()
        if cur:
            try:
                cur_key = str(Path(cur).resolve())
            except Exception:
                cur_key = cur
            for i, p in enumerate(self.favorite_models):
                if p.lower() == cur_key.lower():
                    self.fav_model_pick.set(labels[i])
                    break
        self._update_fav_model_btn_label()

    def _update_fav_exp_btn_label(self) -> None:
        path = self.active_exp_path.get().strip()
        if not path:
            self.fav_exp_btn_label.set("★ Fav")
            return
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        if any(p.lower() == key.lower() for p in self.favorite_experiments):
            self.fav_exp_btn_label.set("★ Unfav")
        else:
            self.fav_exp_btn_label.set("★ Fav")

    def _update_fav_model_btn_label(self) -> None:
        path = self.im_model_path.get().strip()
        if not path:
            self.fav_model_btn_label.set("★ Fav")
            return
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        if any(p.lower() == key.lower() for p in self.favorite_models):
            self.fav_model_btn_label.set("★ Unfav")
        else:
            self.fav_model_btn_label.set("★ Fav")

    def _toggle_favorite_experiment(self) -> None:
        path = self.active_exp_path.get().strip()
        if not path:
            messagebox.showinfo("Favorites", "Browse or select an experiment first.")
            return
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        existing = [p for p in self.favorite_experiments if p.lower() == key.lower()]
        if existing:
            self.favorite_experiments = [
                p for p in self.favorite_experiments if p.lower() != key.lower()
            ]
        else:
            if not Path(key).is_dir():
                messagebox.showerror("Favorites", f"Experiment folder not found:\n{key}")
                return
            self.favorite_experiments.append(key)
        self._refresh_fav_exp_combo()
        self._save_settings()

    def _apply_favorite_experiment(self, _event: Any = None) -> None:
        label = self.fav_exp_pick.get().strip()
        if not label:
            return
        labels = [
            self._fav_display_label(p, is_dir=True) for p in self.favorite_experiments
        ]
        try:
            idx = labels.index(label)
        except ValueError:
            return
        path = self.favorite_experiments[idx]
        if not Path(path).is_dir():
            messagebox.showerror(
                "Favorites",
                f"Favorite experiment missing (removed from list):\n{path}",
            )
            self.favorite_experiments = [
                p for p in self.favorite_experiments if p.lower() != path.lower()
            ]
            self._refresh_fav_exp_combo()
            self._save_settings()
            return
        self._apply_active_experiment(path, lock=True)

    def _toggle_favorite_model(self) -> None:
        path = self.im_model_path.get().strip()
        if not path:
            messagebox.showinfo("Favorites", "Browse or select a model .pth first.")
            return
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        if any(p.lower() == key.lower() for p in self.favorite_models):
            self.favorite_models = [
                p for p in self.favorite_models if p.lower() != key.lower()
            ]
        else:
            if not Path(key).is_file():
                messagebox.showerror("Favorites", f"Model file not found:\n{key}")
                return
            self.favorite_models.append(key)
        self._refresh_fav_model_combo()
        self._save_settings()

    def _apply_favorite_model(self, _event: Any = None) -> None:
        label = self.fav_model_pick.get().strip()
        if not label:
            return
        labels = [
            self._fav_display_label(p, is_dir=False) for p in self.favorite_models
        ]
        try:
            idx = labels.index(label)
        except ValueError:
            return
        path = self.favorite_models[idx]
        if not Path(path).is_file():
            messagebox.showerror(
                "Favorites",
                f"Favorite model file not found:\n{path}",
            )
            return
        self.im_model_path.set(path)
        self.im_model_name.set(Path(path).name)
        self._autofill_im_index()
        self._refresh_im_auto_paths()
        self._update_fav_model_btn_label()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _persist_map(self) -> dict[str, Any]:
        self._capture_src_tree_col_widths()
        return {
            "geometry": self.root.geometry(),
            "last_tab": self.notebook.index(self.notebook.select()) if hasattr(self, "notebook") else 0,
            "rvc_root": self.rvc_root.get(),
            "active_exp_path": self.active_exp_path.get(),
            "exp_locked": self.exp_locked,
            "favorite_experiments": list(self.favorite_experiments),
            "favorite_models": list(self.favorite_models),
            "pb_device": self.pb_device.get(),
            "pb_volume": self.pb_volume.get(),
            "audio_scan_roots": list(self.audio_scan_roots),
            "as_filter": self.as_filter.get(),
            "as_kind_filter": self.as_kind_filter.get(),
            "as_sort_by": sort_key_from_label(self.as_sort_label.get()),
            "as_sort_desc": bool(self.as_sort_desc.get()),
            "src_filter": self.src_filter.get(),
            "src_sort_by": sort_key_from_label(self.src_sort_label.get()),
            "src_sort_desc": bool(self.src_sort_desc.get()),
            "src_score_filter": [
                i for i, var in enumerate(self.src_score_filters) if bool(var.get())
            ],
            "src_tree_col_widths": dict(self._src_tree_col_widths),
            "convert_source_path": self._convert_source_path or "",
            "im_model_path": self.im_model_path.get(),
            "im_model_name": self.im_model_name.get(),
            "im_index_path": self.im_index_path.get(),
            "im_input_path": self.im_input_path.get(),
            "im_f0up_key": self.im_f0up_key.get(),
            "im_f0method": self.im_f0method.get(),
            "im_index_rate": self.im_index_rate.get(),
            "im_filter_radius": self.im_filter_radius.get(),
            "im_resample_sr": self.im_resample_sr.get(),
            "im_rms_mix_rate": self.im_rms_mix_rate.get(),
            "im_protect": self.im_protect.get(),
            "im_breath_mix_rate": self.im_breath_mix_rate.get(),
            "im_chunk_sec": self.im_chunk_sec.get(),
            "im_overlap_sec": self.im_overlap_sec.get(),
            "im_spk_id": self.im_spk_id.get(),
            "im_merge_output_dir": self.im_merge_output_dir.get(),
            "sep_input_path": self.sep_input_path.get(),
            "sep_output_dir": self.sep_output_dir.get(),
            "sep_model_label": self.sep_model_label.get(),
            "sep_format": self.sep_format.get(),
            "sep_segment": self.sep_segment.get(),
            "sep_model_dir": self.sep_model_dir.get(),
            "sep_venv_dir": self.sep_venv_dir.get(),
            "sep_proxy": self.sep_proxy.get(),
            "sep_fill_infer_merge": self.sep_fill_infer_merge.get(),
        }

    def _load_settings(self) -> None:
        self._suppress_autosave = True
        data = load_settings_map(self._db)

        str_vars = {
            "rvc_root": self.rvc_root,
            "active_exp_path": self.active_exp_path,
            "pb_device": self.pb_device,
            "as_filter": self.as_filter,
            "as_kind_filter": self.as_kind_filter,
            "src_filter": self.src_filter,
            "im_model_path": self.im_model_path,
            "im_model_name": self.im_model_name,
            "im_index_path": self.im_index_path,
            "im_input_path": self.im_input_path,
            "im_f0up_key": self.im_f0up_key,
            "im_f0method": self.im_f0method,
            "im_index_rate": self.im_index_rate,
            "im_filter_radius": self.im_filter_radius,
            "im_resample_sr": self.im_resample_sr,
            "im_rms_mix_rate": self.im_rms_mix_rate,
            "im_protect": self.im_protect,
            "im_breath_mix_rate": self.im_breath_mix_rate,
            "im_chunk_sec": self.im_chunk_sec,
            "im_overlap_sec": self.im_overlap_sec,
            "im_spk_id": self.im_spk_id,
            "im_merge_output_dir": self.im_merge_output_dir,
            "sep_input_path": self.sep_input_path,
            "sep_output_dir": self.sep_output_dir,
            "sep_model_label": self.sep_model_label,
            "sep_format": self.sep_format,
            "sep_segment": self.sep_segment,
            "sep_model_dir": self.sep_model_dir,
            "sep_venv_dir": self.sep_venv_dir,
            "sep_proxy": self.sep_proxy,
        }
        bool_vars = {
            "sep_fill_infer_merge": self.sep_fill_infer_merge,
        }
        for key, var in str_vars.items():
            if key in data and data[key] is not None:
                var.set(str(data[key]))
        # Kind filter Combobox values are display labels ("source audio"), not keys.
        kind_raw = self.as_kind_filter.get().strip()
        if kind_raw.lower() in ("", "all"):
            self.as_kind_filter.set("all")
        elif kind_raw in KIND_FILTER_VALUES:
            self.as_kind_filter.set(kind_raw)
        else:
            self.as_kind_filter.set(kind_label(kind_raw))
        if "pb_volume" in data:
            try:
                self.pb_volume.set(float(data["pb_volume"]))
            except (TypeError, ValueError):
                pass
        elif "ab_device" in data and data["ab_device"] and not self.pb_device.get():
            self.pb_device.set(str(data["ab_device"]))
        for key, var in bool_vars.items():
            if key in data:
                var.set(bool(data[key]))

        sort_by = str(data.get("src_sort_by", "rel_path"))
        self.src_sort_label.set(sort_label(sort_by))
        if "src_sort_desc" in data:
            self.src_sort_desc.set(bool(data["src_sort_desc"]))
        as_sort_by = str(data.get("as_sort_by", "rel_path"))
        self.as_sort_label.set(sort_label(as_sort_by))
        if "as_sort_desc" in data:
            self.as_sort_desc.set(bool(data["as_sort_desc"]))
        raw_scores = data.get("src_score_filter")
        if isinstance(raw_scores, list):
            wanted = {int(x) for x in raw_scores if str(x).isdigit() or isinstance(x, int)}
            wanted = {n for n in wanted if 0 <= n <= SCORE_MAX}
            if wanted:
                for i, var in enumerate(self.src_score_filters):
                    var.set(i in wanted)
            else:
                for var in self.src_score_filters:
                    var.set(True)

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
        self.favorite_experiments = self._normalize_path_list(
            data.get("favorite_experiments", [])
        )
        self.favorite_models = self._normalize_path_list(data.get("favorite_models", []))
        self.audio_scan_roots = self._normalize_path_list(data.get("audio_scan_roots", []))
        self._restore_audio_scan_from_settings()
        want_lock = bool(data.get("exp_locked", False))
        active = self.active_exp_path.get().strip()
        if want_lock and active:
            self._apply_active_experiment(active, lock=True, quiet=True)
        else:
            self.exp_locked = False
            self._set_exp_widgets_locked(False)
        self._refresh_fav_exp_combo()
        self._refresh_fav_model_combo()
        self._apply_playback_settings()
        self._src_tree_col_widths = self._parse_src_tree_col_widths(
            data.get("src_tree_col_widths")
        )
        self._apply_src_tree_col_widths()
        csp = str(data.get("convert_source_path", "")).strip()
        if csp and Path(csp).is_file():
            self._convert_source_path = csp
        elif self._convert_source_path is None:
            vocals = self.im_input_path.get().strip()
            if vocals:
                src = find_source_for_vocals(vocals, self._stem_links, self._audio_scan_rows)
                if src:
                    self._convert_source_path = src
        self.root.after(100, self._refresh_convert_results_list)
        self._settings_hydrated = True
        self._suppress_autosave = False

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        self._refresh_wraplengths(event.width)
        if self._suppress_autosave:
            return
        if self._geometry_save_after_id is not None:
            try:
                self.root.after_cancel(self._geometry_save_after_id)
            except tk.TclError:
                pass
        self._geometry_save_after_id = self.root.after(500, self._save_geometry_debounced)

    def _register_wrap_label(self, label: tk.Misc, *, fraction: float = 0.85) -> None:
        self._wrap_labels.append((label, max(0.25, min(0.95, fraction))))

    def _refresh_wraplengths(self, width: int | None = None) -> None:
        if not hasattr(self, "_wrap_labels"):
            return
        if width is None or width <= 1:
            try:
                width = int(self.root.winfo_width())
            except tk.TclError:
                return
        usable = max(360, width - 64)
        for label, fraction in self._wrap_labels:
            try:
                label.configure(wraplength=int(usable * fraction))
            except tk.TclError:
                pass

    def _save_geometry_debounced(self) -> None:
        self._geometry_save_after_id = None
        self._save_settings()

    def _save_settings(self) -> None:
        if self._suppress_autosave or not self._settings_hydrated:
            return
        try:
            save_settings_map(self._db, self._persist_map())
        except Exception:
            pass

    def _wire_autosave(self) -> None:
        vars_to_trace: list[tk.Variable] = [
            self.rvc_root,
            self.active_exp_path,
            self.pb_device,
            self.pb_volume,
            self.as_filter,
            self.as_kind_filter,
            self.src_filter,
            *self.src_score_filters,
            self.im_model_path,
            self.im_model_name,
            self.im_index_path,
            self.im_input_path,
            self.im_f0up_key,
            self.im_f0method,
            self.im_index_rate,
            self.im_filter_radius,
            self.im_resample_sr,
            self.im_rms_mix_rate,
            self.im_protect,
            self.im_breath_mix_rate,
            self.im_chunk_sec,
            self.im_overlap_sec,
            self.im_spk_id,
            self.im_merge_output_dir,
            self.sep_input_path,
            self.sep_output_dir,
            self.sep_model_label,
            self.sep_format,
            self.sep_segment,
            self.sep_model_dir,
            self.sep_venv_dir,
            self.sep_proxy,
            self.sep_fill_infer_merge,
        ]
        for var in vars_to_trace:
            tid = var.trace_add("write", lambda *_a: self._save_settings())
            self._persist_trace_ids.append(tid)

    def _on_close(self) -> None:
        self._save_settings()
        self._playback_stop()
        if self.running and self.proc is not None:
            self._kill_process()
        try:
            self._db.close()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        self._bottom_dock = ttk.Frame(outer)
        self._bottom_dock.pack(side="bottom", fill="x")

        self._build_global_playback(self._bottom_dock)

        ctrl = ttk.Frame(self._bottom_dock)
        ctrl.pack(fill="x", pady=(6, 0))

        self.stop_btn = ttk.Button(
            ctrl, text="Stop job", command=self.stop_job, state="disabled"
        )
        self.stop_btn.pack(side="left")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        self._log_section = ttk.Frame(self._bottom_dock)
        self._log_section.pack(fill="x", pady=(6, 0))

        log_hdr = ttk.Frame(self._log_section)
        log_hdr.pack(fill="x")
        self._log_toggle_label = tk.StringVar(value="▶ Log")
        ttk.Button(
            log_hdr,
            textvariable=self._log_toggle_label,
            command=self._toggle_log_panel,
            width=12,
        ).pack(side="left")

        self._log_body = ttk.Frame(self._log_section)
        log_inner = ttk.Frame(self._log_body, padding=4, relief="groove", borderwidth=1)
        log_inner.pack(fill="x")
        self.log_text = tk.Text(
            log_inner,
            height=8,
            wrap="word",
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground=self.colors["log_fg"],
            relief="flat",
            bd=0,
        )
        log_sb = ttk.Scrollbar(log_inner, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")
        self.log_text.configure(state="disabled")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        # List tabs: plain frame so the Treeview can fill remaining height.
        # Form tabs: ScrollableFrame for long content.
        self._tab_frames: list[ttk.Frame] = []
        builders = [
            (TAB_NAMES[0], self._build_tab_settings, True),
            (TAB_NAMES[1], self._build_tab_audio_scan, False),
            (TAB_NAMES[2], self._build_tab_source_audio, False),
            (TAB_NAMES[3], self._build_tab_process, True),
            (TAB_NAMES[4], self._build_tab_separate, True),
            (TAB_NAMES[5], self._build_tab_convert, True),
        ]
        for name, builder, scrollable in builders:
            if scrollable:
                sf = ScrollableFrame(self.notebook, bg=self.colors["bg"])
                self.notebook.add(sf, text=name)
                self._tab_frames.append(sf)
                builder(sf.inner)
            else:
                frame = ttk.Frame(self.notebook)
                self.notebook.add(frame, text=name)
                self._tab_frames.append(frame)
                builder(frame)

    def _toggle_log_panel(self) -> None:
        self._log_collapsed = not self._log_collapsed
        if self._log_collapsed:
            self._log_body.pack_forget()
            self._log_toggle_label.set("▶ Log")
        else:
            self._log_body.pack(fill="x")
            self._log_toggle_label.set("▼ Log")

    def _row_path(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: Callable[[], None],
        browse_label: str = "Browse",
    ) -> tuple[ttk.Entry, ttk.Button]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
        btn = ttk.Button(parent, text=browse_label, command=browse)
        btn.grid(row=row, column=4, padx=(8, 0), pady=4)
        return entry, btn

    def _path_dialog_opts(
        self,
        current: str,
        *,
        for_file: bool = False,
        fallback: str | Path | None = None,
    ) -> dict[str, str]:
        """Open the file/dir dialog at this field's current path (not last process pick)."""
        opts: dict[str, str] = {}
        candidates: list[Path] = []
        raw = (current or "").strip().strip('"')
        if raw:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                root = self._configured_root()
                if root is not None:
                    p = root / p
            candidates.append(p)
        if fallback:
            candidates.append(Path(fallback))

        for p in candidates:
            try:
                if for_file:
                    if p.is_file():
                        opts["initialdir"] = str(p.parent)
                        opts["initialfile"] = p.name
                        return opts
                    if p.is_dir():
                        opts["initialdir"] = str(p)
                        return opts
                    parent = p.parent
                    if parent.is_dir():
                        opts["initialdir"] = str(parent)
                        if p.name:
                            opts["initialfile"] = p.name
                        return opts
                else:
                    if p.is_dir():
                        opts["initialdir"] = str(p)
                        return opts
                    if p.exists() and p.parent.is_dir():
                        opts["initialdir"] = str(p.parent)
                        return opts
                    parent = p.parent
                    if parent.is_dir():
                        opts["initialdir"] = str(parent)
                        return opts
            except OSError:
                continue
        return opts

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

    def _build_global_playback(self, parent: ttk.Frame) -> None:
        play = ttk.LabelFrame(parent, text="Playback", padding=8)
        play.pack(fill="x", pady=(6, 0))
        play.columnconfigure(0, weight=1)

        self.pb_now_playing = tk.StringVar(value="(no selection)")
        self.pb_now_playing_full = ""
        now_lbl = ttk.Label(play, textvariable=self.pb_now_playing)
        now_lbl.grid(row=0, column=0, columnspan=4, sticky="w")
        self._register_wrap_label(now_lbl, fraction=0.9)

        progress_row = ttk.Frame(play)
        progress_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 2))
        progress_row.columnconfigure(1, weight=1)
        self.pb_time_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(progress_row, textvariable=self.pb_time_var, width=14).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.pb_seek = ttk.Scale(
            progress_row,
            from_=0,
            to=1000,
            orient="horizontal",
            command=self._on_playback_seek,
        )
        self.pb_seek.grid(row=0, column=1, sticky="ew")
        self.pb_seek.bind("<ButtonPress-1>", self._on_playback_seek_press)
        self.pb_seek.bind("<ButtonRelease-1>", self._on_playback_seek_release)

        ctrl = ttk.Frame(play)
        ctrl.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        ctrl.columnconfigure(3, weight=1)
        self.pb_play_btn = ttk.Button(ctrl, text="Play", command=self._playback_toggle_pause)
        self.pb_play_btn.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(ctrl, text="Stop play", command=self._playback_stop).grid(
            row=0, column=1, padx=(0, 16)
        )
        ttk.Label(ctrl, text="Volume").grid(row=0, column=2, padx=(0, 8))
        vol = ttk.Scale(
            ctrl,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.pb_volume,
            command=self._on_playback_volume_changed,
        )
        vol.grid(row=0, column=3, sticky="ew", padx=(0, 16))
        ttk.Label(ctrl, text="Output").grid(row=0, column=4, padx=(0, 8))
        self.pb_device_combo = ttk.Combobox(
            ctrl, textvariable=self.pb_device, width=28, state="readonly"
        )
        self.pb_device_combo.grid(row=0, column=5, sticky="ew")
        self.pb_device_combo.bind("<<ComboboxSelected>>", self._on_playback_device_changed)
        ttk.Button(ctrl, text="Refresh", command=self._refresh_playback_devices).grid(
            row=0, column=6, padx=(8, 0)
        )

        self._refresh_playback_devices()
        self._apply_playback_settings()

    def _apply_playback_settings(self) -> None:
        self._player.set_volume(self.pb_volume.get() / 100.0)
        dev = self.pb_device.get().strip()
        self._player.set_output_device(dev or SYSTEM_DEFAULT_DEVICE)

    def _refresh_playback_devices(self) -> None:
        devices = list_output_devices()
        self.pb_device_combo["values"] = devices
        cur = self.pb_device.get().strip()
        if not devices:
            self.pb_device.set("")
            self._player.set_output_device(None)
            self._save_settings()
            return
        if not cur or not any(device_label_matches(cur, d) for d in devices):
            self.pb_device.set(pick_default_device_label(devices))
        self._apply_playback_settings()
        self._save_settings()

    def _on_playback_volume_changed(self, _value: str = "") -> None:
        self._player.set_volume(self.pb_volume.get() / 100.0)
        self._save_settings()

    def _on_playback_device_changed(self, _event: tk.Event | None = None) -> None:
        self._player.set_output_device(self.pb_device.get())
        self._save_settings()

    def _on_playback_seek_press(self, _event: tk.Event | None = None) -> None:
        self._pb_seeking = True

    def _on_playback_seek_release(self, _event: tk.Event | None = None) -> None:
        self._pb_seeking = False
        try:
            self._on_playback_seek(str(self.pb_seek.get()))
        except tk.TclError:
            pass

    def _on_playback_seek(self, value: str) -> None:
        if self._pb_ignore_seek:
            return
        player = self._player
        if player.path is None:
            return
        dur = player.duration_ms
        if dur <= 0:
            return
        try:
            pct = float(value) / 1000.0
        except (TypeError, ValueError):
            return
        target_ms = int(dur * pct)
        player.seek(target_ms)
        self.pb_time_var.set(f"{format_time_ms(target_ms)} / {format_time_ms(dur)}")
        if player.state() == "playing" and self._playback_after_id is None:
            self._playback_after_id = self.root.after(200, self._playback_tick)

    def _playback_cancel_tick(self) -> None:
        if self._playback_after_id is not None:
            try:
                self.root.after_cancel(self._playback_after_id)
            except tk.TclError:
                pass
            self._playback_after_id = None

    def _set_now_playing_display(self, path: str | None) -> None:
        if not path:
            self.pb_now_playing_full = ""
            self.pb_now_playing.set("(no selection)")
            return
        self.pb_now_playing_full = str(path)
        self.pb_now_playing.set(Path(path).name)

    def _update_playback_ui(self) -> None:
        if not hasattr(self, "pb_play_btn"):
            return
        player = self._player
        state = player.state()
        if player.path is not None:
            self._set_now_playing_display(str(player.path))
        else:
            self._set_now_playing_display(None)
        if state == "paused":
            self.pb_play_btn.config(text="Resume")
        elif state == "playing":
            self.pb_play_btn.config(text="Pause")
        else:
            self.pb_play_btn.config(text="Play")
        dur = player.duration_ms
        pos = player.position_ms() if state in ("playing", "paused") else 0
        if dur > 0 and state in ("playing", "paused"):
            if not self._pb_seeking:
                self._pb_ignore_seek = True
                try:
                    self.pb_seek.set(min(1000.0, pos / dur * 1000.0))
                except tk.TclError:
                    pass
                self._pb_ignore_seek = False
            self.pb_time_var.set(f"{format_time_ms(pos)} / {format_time_ms(dur)}")
        else:
            if not self._pb_seeking:
                self._pb_ignore_seek = True
                try:
                    self.pb_seek.set(0)
                except tk.TclError:
                    pass
                self._pb_ignore_seek = False
            self.pb_time_var.set(
                f"0:00 / {format_time_ms(dur)}" if dur > 0 else "0:00 / 0:00"
            )

    def _playback_tick(self) -> None:
        self._playback_after_id = None
        player = self._player
        state = player.state()
        if state == "playing":
            if not player.is_busy():
                player.stop()
                self._update_playback_ui()
                return
            self._update_playback_ui()
            self._playback_after_id = self.root.after(200, self._playback_tick)
        elif state == "paused":
            self._update_playback_ui()

    def _playback_play_path(self, path: str) -> None:
        try:
            played = self._player.play(path)
        except PlaybackError as exc:
            self._refresh_playback_devices()
            try:
                played = self._player.play(path)
            except PlaybackError:
                messagebox.showerror("Playback error", str(exc))
                return
        self._set_now_playing_display(str(played))
        self._playback_cancel_tick()
        self._update_playback_ui()
        self._playback_after_id = self.root.after(200, self._playback_tick)

    def _playback_toggle_pause(self) -> None:
        state = self._player.state()
        if state == "playing":
            self._player.pause()
            self._playback_cancel_tick()
            self._update_playback_ui()
            return
        if state == "paused":
            self._player.resume()
            self._update_playback_ui()
            self._playback_after_id = self.root.after(200, self._playback_tick)
            return
        path = self._src_selected_path()
        if path:
            self._playback_play_path(path)

    def _playback_stop(self) -> None:
        self._playback_cancel_tick()
        self._player.stop()
        self._update_playback_ui()

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
        note_lbl = ttk.Label(f, text=note, justify="left")
        note_lbl.grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self._register_wrap_label(note_lbl, fraction=0.85)

        exp = ttk.LabelFrame(parent, text="Active experiment", padding=10)
        exp.pack(fill="x", padx=4, pady=4)
        self._configure_cols(exp)
        ttk.Label(exp, text="Experiment").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.active_exp_entry = ttk.Entry(exp, textvariable=self.active_exp_path, state="readonly")
        self.active_exp_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        exp_btns = ttk.Frame(exp)
        exp_btns.grid(row=0, column=3, columnspan=2, sticky="e", pady=4)
        ttk.Button(exp_btns, text="Browse", command=self._browse_active_exp).pack(
            side="left", padx=2
        )
        ttk.Button(
            exp_btns, textvariable=self.fav_exp_btn_label, command=self._toggle_favorite_experiment, width=9
        ).pack(side="left", padx=2)
        ttk.Button(exp_btns, text="Clear", command=self._clear_active_experiment).pack(
            side="left", padx=2
        )

        ttk.Label(exp, text="Favorites").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.fav_exp_combo = ttk.Combobox(
            exp,
            textvariable=self.fav_exp_pick,
            state="readonly",
            width=56,
        )
        self.fav_exp_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        self.fav_exp_combo.bind("<<ComboboxSelected>>", self._apply_favorite_experiment)

        exp_note = ttk.Label(
            exp,
            text="Browse logs/<exp_name>. ★ Fav saves to Favorites; fills Exp fields until Clear.",
            justify="left",
        )
        exp_note.grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self._register_wrap_label(exp_note, fraction=0.85)

        sep = ttk.LabelFrame(parent, text="Audio separator (Separate tab)", padding=10)
        sep.pack(fill="x", padx=4, pady=4)
        self._configure_cols(sep)
        self._row_path(sep, 0, "Separator venv", self.sep_venv_dir, self._browse_sep_venv)
        self._row_path(sep, 1, "Model dir", self.sep_model_dir, self._browse_sep_model_dir)
        ttk.Label(sep, text="Proxy (optional)").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(sep, textvariable=self.sep_proxy).grid(
            row=2, column=1, columnspan=3, sticky="ew", pady=4
        )
        sep_note = ttk.Label(
            sep,
            text=(
                f"Defaults: venv={DEFAULT_SEP_VENV}  models={DEFAULT_SEP_MODEL_DIR}. "
                "Run setup_separator.ps1 if audio-separator is missing."
            ),
            justify="left",
        )
        sep_note.grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self._register_wrap_label(sep_note, fraction=0.85)

        shared = ttk.LabelFrame(parent, text="Shared catalog (git)", padding=10)
        shared.pack(fill="x", padx=4, pady=4)
        self._configure_cols(shared)
        self.shared_catalog_path_var = tk.StringVar(value=str(SHARED_CATALOG_PATH))
        ttk.Label(shared, text="File").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(shared, textvariable=self.shared_catalog_path_var, state="readonly").grid(
            row=0, column=1, columnspan=3, sticky="ew", pady=4
        )
        shared_btns = ttk.Frame(shared)
        shared_btns.grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Button(
            shared_btns,
            text="Export DB → catalog",
            command=self._on_export_shared_catalog,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            shared_btns,
            text="Import catalog → DB",
            command=self._on_import_shared_catalog,
        ).pack(side="left", padx=(0, 8))
        shared_note = ttk.Label(
            shared,
            text=(
                "Portable: notes/scores by stem; kinds by filename. "
                "Commit shared_catalog.json to share triage."
            ),
            justify="left",
        )
        shared_note.grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self._register_wrap_label(shared_note, fraction=0.85)

        db_lbl = ttk.Label(parent, text=f"Local database: {DB_PATH}")
        db_lbl.pack(anchor="w", padx=8, pady=8)
        self._register_wrap_label(db_lbl, fraction=0.85)

        self.rvc_root.trace_add("write", lambda *_a: self._on_rvc_root_changed())

    def _on_export_shared_catalog(self) -> None:
        try:
            path = self._sync_shared_catalog_from_db()
        except OSError as exc:
            messagebox.showerror("Shared catalog", str(exc))
            return
        n_notes = len(self._shared_catalog.notes_by_stem)
        n_scores = len(self._shared_catalog.scores_by_stem)
        n_kinds = len(self._shared_catalog.kinds_by_name)
        messagebox.showinfo(
            "Shared catalog",
            f"Exported notes={n_notes} scores={n_scores} kind_overrides={n_kinds}\n{path}",
        )

    def _on_import_shared_catalog(self) -> None:
        n_notes, n_kinds = self._apply_shared_catalog(persist_db=True)
        self._as_apply_filter()
        self._refresh_source_list()
        messagebox.showinfo(
            "Shared catalog",
            f"Imported from {SHARED_CATALOG_PATH}\n"
            f"notes_updated={n_notes} kinds_updated={n_kinds}",
        )

    def _browse_sep_venv(self) -> None:
        path = filedialog.askdirectory(
            title="Select separator venv folder (.venv)",
            **self._path_dialog_opts(self.sep_venv_dir.get()),
        )
        if path:
            self.sep_venv_dir.set(path)

    def _browse_sep_model_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select separator model folder",
            **self._path_dialog_opts(self.sep_model_dir.get()),
        )
        if path:
            self.sep_model_dir.set(path)

    def _on_rvc_root_changed(self) -> None:
        self._refresh_python_display()
        self._fill_empty_defaults_from_root()

    def _browse_rvc_root(self) -> None:
        path = filedialog.askdirectory(
            title="Select RVC root folder",
            **self._path_dialog_opts(self.rvc_root.get()),
        )
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

    # ---- Tab 1: Audio Scan ----

    def _build_tab_audio_scan(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        roots_frame = ttk.LabelFrame(parent, text="Root directories", padding=8)
        roots_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        roots_frame.columnconfigure(0, weight=1)

        list_wrap = ttk.Frame(roots_frame)
        list_wrap.grid(row=0, column=0, sticky="ew", pady=2)
        list_wrap.columnconfigure(0, weight=1)
        self.as_roots_list = tk.Listbox(
            list_wrap,
            height=3,
            exportselection=False,
            bg=self.colors["field"],
            fg=self.colors["fg"],
            selectbackground=self.colors["select"],
            selectforeground=self.colors.get("select_fg", self.colors["fg"]),
            highlightthickness=0,
        )
        as_roots_sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.as_roots_list.yview)
        self.as_roots_list.configure(yscrollcommand=as_roots_sb.set)
        self.as_roots_list.grid(row=0, column=0, sticky="ew")
        as_roots_sb.grid(row=0, column=1, sticky="ns")

        btns = ttk.Frame(roots_frame)
        btns.grid(row=0, column=1, sticky="n", padx=(8, 0))
        ttk.Button(btns, text="Add folder…", command=self._as_add_root).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self._as_remove_root).pack(fill="x", pady=2)
        self.as_scan_btn = ttk.Button(
            btns, text="Scan", command=self._as_start_scan, style=BTN_STYLE_SCAN
        )
        self.as_scan_btn.pack(fill="x", pady=(10, 2))

        self.as_status = tk.StringVar(value="No scan yet.")
        ttk.Label(roots_frame, textvariable=self.as_status).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.as_progress = ttk.Progressbar(
            roots_frame, mode="determinate", maximum=100, value=0
        )
        self.as_progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        table = ttk.LabelFrame(parent, text="Audio files", padding=6)
        table.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        table.rowconfigure(1, weight=1)
        table.columnconfigure(0, weight=1)

        filt = ttk.Frame(table)
        filt.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        filt.columnconfigure(1, weight=1)
        ttk.Label(filt, text="Filter").grid(row=0, column=0, sticky="w", padx=(0, 8))
        filter_entry = ttk.Entry(filt, textvariable=self.as_filter)
        filter_entry.grid(row=0, column=1, sticky="ew")
        filter_entry.bind("<Return>", lambda _e: self._as_apply_filter())
        ttk.Label(filt, text="Kind").grid(row=0, column=2, sticky="w", padx=(12, 8))
        self.as_kind_combo = ttk.Combobox(
            filt,
            textvariable=self.as_kind_filter,
            values=KIND_FILTER_VALUES,
            state="readonly",
            width=14,
        )
        self.as_kind_combo.grid(row=0, column=3, sticky="w")
        self.as_kind_combo.bind("<<ComboboxSelected>>", self._on_as_kind_filter_changed)
        ttk.Label(filt, text="Sort").grid(row=0, column=4, sticky="w", padx=(12, 8))
        self.as_sort_combo = ttk.Combobox(
            filt,
            textvariable=self.as_sort_label,
            values=list(SRC_SORT_LABELS.values()),
            state="readonly",
            width=14,
        )
        self.as_sort_combo.grid(row=0, column=5, sticky="w")
        self.as_sort_combo.bind("<<ComboboxSelected>>", self._on_as_sort_changed)
        ttk.Checkbutton(
            filt,
            text="Desc",
            variable=self.as_sort_desc,
            command=self._on_as_sort_changed,
        ).grid(row=0, column=6, padx=(8, 0), sticky="w")
        ttk.Button(filt, text="Apply", command=self._as_apply_filter, width=8).grid(
            row=0, column=7, padx=(12, 0)
        )
        self.as_kind_combo.set(self.as_kind_filter.get() or "all")
        self.as_sort_combo.set(self.as_sort_label.get())

        cols = ("kind", "root", "rel", "name", "size", "duration", "path")
        self.as_tree = ttk.Treeview(
            table, columns=cols, show="headings", selectmode="browse"
        )
        headings = {
            "kind": ("Kind", 108),
            "root": ("Root", 120),
            "rel": ("Relative path", 220),
            "name": ("File", 160),
            "size": ("Size", 72),
            "duration": ("Duration", 72),
            "path": ("Full path", 360),
        }
        for key, (label, width) in headings.items():
            self.as_tree.heading(key, text=label)
            anchor = "center" if key in ("kind", "duration") else "w"
            self.as_tree.column(key, width=width, anchor=anchor, stretch=(key == "path"))
        ysb = ttk.Scrollbar(table, orient="vertical", command=self.as_tree.yview)
        xsb = ttk.Scrollbar(table, orient="horizontal", command=self.as_tree.xview)
        self.as_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.as_tree.grid(row=1, column=0, sticky="nsew")
        ysb.grid(row=1, column=1, sticky="ns")
        xsb.grid(row=2, column=0, sticky="ew")
        self.as_tree.bind("<Button-1>", self._as_on_tree_click, add="+")

    def _as_kind_column_id(self) -> str:
        return "#1"

    def _as_path_from_item(self, item: str) -> str | None:
        vals = self.as_tree.item(item, "values")
        if not vals or len(vals) < 7:
            return None
        path = str(vals[6]).strip()
        if not path or not Path(path).is_file():
            return None
        return path

    def _as_row_for_path(self, path: str) -> AudioFileRow | None:
        from audio_scan import _path_key

        target = _path_key(path)
        for row in self._audio_scan_rows:
            if _path_key(row.path) == target:
                return row
        return None

    def _as_destroy_kind_editor(self) -> None:
        if self._as_kind_editor is not None:
            try:
                self._as_kind_editor.destroy()
            except tk.TclError:
                pass
        self._as_kind_editor = None
        self._as_kind_edit_item = None
        self._as_kind_edit_path = None

    def _as_on_tree_click(self, event: tk.Event) -> None:
        if self.as_tree.identify_region(event.x, event.y) != "cell":
            return
        if self.as_tree.identify_column(event.x) != self._as_kind_column_id():
            return
        item = self.as_tree.identify_row(event.y)
        if item:
            self.root.after_idle(lambda i=item: self._as_begin_kind_edit(i))

    def _as_begin_kind_edit(self, item: str) -> None:
        if not item:
            return
        path = self._as_path_from_item(item)
        if not path:
            return
        row = self._as_row_for_path(path)
        if row is None:
            return
        self._as_destroy_kind_editor()
        bbox = self.as_tree.bbox(item, column="kind")
        if not bbox:
            return
        x, y, w, h = bbox
        combo = ttk.Combobox(
            self.as_tree,
            values=KIND_EDIT_VALUES,
            state="readonly",
            width=max(12, w // 8),
        )
        combo.set(row.kind_label)
        combo.place(x=x, y=y, width=max(w, 100), height=h)
        combo.focus_set()
        self._as_kind_editor = combo
        self._as_kind_edit_item = item
        self._as_kind_edit_path = path
        combo.bind("<<ComboboxSelected>>", self._as_commit_kind_edit)
        combo.bind("<FocusOut>", self._as_on_kind_focus_out)
        combo.bind("<Escape>", self._as_cancel_kind_edit)
        # Open on first click: otherwise user must click again to drop the list.
        combo.after(1, lambda c=combo: self._as_post_kind_combo(c))

    def _as_post_kind_combo(self, combo: ttk.Combobox) -> None:
        if combo is not self._as_kind_editor:
            return
        try:
            combo.tk.call("ttk::combobox::Post", str(combo))
        except tk.TclError:
            try:
                combo.event_generate("<Down>")
            except tk.TclError:
                pass

    def _as_on_kind_focus_out(self, _event: tk.Event | None = None) -> None:
        # Delay: opening/closing the popdown can briefly steal focus and would
        # otherwise destroy the editor before a selection registers.
        self.root.after(180, self._as_commit_kind_if_unfocused)

    def _as_commit_kind_if_unfocused(self) -> None:
        editor = self._as_kind_editor
        if editor is None or self._as_kind_committing:
            return
        try:
            focused = self.root.focus_get()
        except tk.TclError:
            focused = None
        if focused is editor:
            return
        self._as_commit_kind_edit()

    def _as_commit_kind_edit(self, _event: tk.Event | None = None) -> None:
        if self._as_kind_committing:
            return
        editor = self._as_kind_editor
        item = self._as_kind_edit_item
        path = self._as_kind_edit_path
        if editor is None or not item or not path:
            self._as_destroy_kind_editor()
            return
        label = editor.get().strip()
        new_kind = kind_from_label(label)
        row = self._as_row_for_path(path)
        if row is None or row.kind == new_kind:
            self._as_destroy_kind_editor()
            return
        self._as_kind_committing = True
        try:
            self._audio_scan_rows = update_row_kind(self._audio_scan_rows, path, new_kind)
            self._audio_scan_revision += 1
            self._invalidate_source_display_cache()
            try:
                save_scan_cache(self._db, self._audio_scan_rows)
            except OSError:
                pass
            row_after = self._as_row_for_path(path)
            if row_after is not None and set_kind_for_name(
                self._shared_catalog, row_after.name, new_kind
            ):
                self._persist_shared_catalog()
            vals = list(self.as_tree.item(item, "values"))
            if vals:
                vals[0] = kind_label(new_kind)
                self.as_tree.item(item, values=vals)
            self._refresh_source_list()
        finally:
            self._as_destroy_kind_editor()
            self._as_kind_committing = False

    def _as_cancel_kind_edit(self, _event: tk.Event | None = None) -> None:
        self._as_destroy_kind_editor()

    def _restore_audio_scan_from_settings(self) -> None:
        self._as_refresh_roots_list()
        self._audio_scan_rows = load_scan_cache(self._db)
        self._apply_shared_catalog(persist_db=True)
        if self._audio_scan_rows:
            self._as_apply_filter()
            self._refresh_source_list()
            self._start_duration_backfill()

    def _start_duration_backfill(self, paths: list[str] | None = None) -> None:
        if self._duration_backfill_running:
            return
        if paths is None:
            paths = paths_needing_duration(self._audio_scan_rows)
        if not paths:
            return
        self._duration_backfill_running = True
        total = len(paths)
        if hasattr(self, "src_status"):
            self.src_status.set(f"Loading durations… 0/{total}")

        def worker() -> None:
            done = 0
            batch: dict[str, float] = {}
            for path in paths:
                dur = probe_duration_sec(path)
                if dur > 0:
                    batch[path] = dur
                done += 1
                if done % 25 == 0 or done == total:
                    chunk = dict(batch)
                    n = done
                    self.root.after(
                        0, lambda c=chunk, n=n: self._duration_backfill_progress(c, n, total)
                    )
            self.root.after(0, self._duration_backfill_done)

        threading.Thread(target=worker, daemon=True).start()

    def _duration_backfill_progress(
        self, updates: dict[str, float], done: int, total: int
    ) -> None:
        if not updates:
            if hasattr(self, "src_status"):
                self.src_status.set(f"Loading durations… {done}/{total}")
            return
        self._audio_scan_rows = apply_duration_updates(self._audio_scan_rows, updates)
        self._audio_scan_revision += 1
        self._invalidate_source_display_cache()
        self._refresh_source_list(sort_only=True)
        if hasattr(self, "src_status"):
            self.src_status.set(f"Loading durations… {done}/{total}")

    def _duration_backfill_done(self) -> None:
        self._duration_backfill_running = False
        try:
            save_scan_cache(self._db, self._audio_scan_rows)
        except OSError:
            pass
        self._refresh_source_list()

    def _set_audio_scan_rows(self, rows: list[AudioFileRow]) -> None:
        self._audio_scan_rows = rows
        self._audio_scan_revision += 1
        self._invalidate_source_display_cache()
        try:
            save_scan_cache(self._db, rows)
        except OSError:
            pass
        self._as_apply_filter()
        self._refresh_source_list()

    def _as_refresh_roots_list(self) -> None:
        if not hasattr(self, "as_roots_list"):
            return
        self.as_roots_list.delete(0, tk.END)
        for p in self.audio_scan_roots:
            self.as_roots_list.insert(tk.END, p)

    def _as_add_root(self) -> None:
        path = filedialog.askdirectory(
            title="Select audio root directory",
            **self._path_dialog_opts(
                self.audio_scan_roots[-1] if self.audio_scan_roots else ""
            ),
        )
        if not path:
            return
        key = str(Path(path).resolve())
        if any(p.lower() == key.lower() for p in self.audio_scan_roots):
            messagebox.showinfo("Audio Scan", "Root already in list.")
            return
        self.audio_scan_roots.append(key)
        self._as_refresh_roots_list()
        self._save_settings()

    def _as_remove_root(self) -> None:
        if not hasattr(self, "as_roots_list"):
            return
        sel = self.as_roots_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.audio_scan_roots):
            del self.audio_scan_roots[idx]
            self._as_refresh_roots_list()
            self._save_settings()

    def _as_start_scan(self) -> None:
        if self._audio_scan_running:
            return
        if not self.audio_scan_roots:
            messagebox.showinfo("Audio Scan", "Add at least one root directory.")
            return
        self._audio_scan_running = True
        self.as_status.set("Scanning…")
        if hasattr(self, "as_scan_btn"):
            self.as_scan_btn.config(state="disabled")
        if hasattr(self, "as_progress"):
            self.as_progress.stop()
            self.as_progress.configure(mode="indeterminate")
            self.as_progress.start(12)
        self._set_window_progress(0.0)
        roots = list(self.audio_scan_roots)

        def on_progress(done: int, total: int, message: str) -> None:
            self.root.after(
                0, lambda d=done, t=total, m=message: self._as_scan_progress(d, t, m)
            )

        def worker() -> None:
            try:
                rows = scan_audio_roots(roots, on_progress=on_progress)
            except Exception as exc:
                self.root.after(0, lambda: self._as_scan_failed(str(exc)))
                return
            self.root.after(0, lambda: self._as_scan_done(rows))

        threading.Thread(target=worker, daemon=True).start()

    def _as_scan_progress(self, done: int, total: int, message: str) -> None:
        if not self._audio_scan_running:
            return
        self.as_status.set(message)
        if not hasattr(self, "as_progress"):
            return
        if total > 0:
            if str(self.as_progress.cget("mode")) != "determinate":
                self.as_progress.stop()
                self.as_progress.configure(mode="determinate", maximum=100)
            pct = 100.0 * done / max(total, 1)
            self.as_progress["value"] = pct
            self._set_window_progress(pct)
        else:
            # Listing phase — keep indeterminate pulse
            if str(self.as_progress.cget("mode")) != "indeterminate":
                self.as_progress.configure(mode="indeterminate")
                self.as_progress.start(12)

    def _as_scan_finish_ui(self) -> None:
        if hasattr(self, "as_scan_btn"):
            self.as_scan_btn.config(state="normal")
        if hasattr(self, "as_progress"):
            self.as_progress.stop()
            self.as_progress.configure(mode="determinate", maximum=100, value=0)
        self._reset_job_progress()

    def _on_as_kind_filter_changed(self, _event: tk.Event | None = None) -> None:
        self._as_apply_filter()

    def _on_as_sort_changed(self, _event: tk.Event | None = None) -> None:
        self._as_apply_filter()
        self._save_settings()

    def _as_scan_failed(self, msg: str) -> None:
        self._audio_scan_running = False
        self._as_scan_finish_ui()
        self.as_status.set(f"Scan failed: {msg}")

    def _as_scan_done(self, rows: list[AudioFileRow]) -> None:
        self._audio_scan_running = False
        self._as_scan_finish_ui()
        rows = merge_rescan_rows(self._audio_scan_rows, rows)
        self._audio_scan_rows = rows
        self._apply_shared_catalog(persist_db=True)
        self._set_audio_scan_rows(self._audio_scan_rows)

    def _as_apply_filter(self) -> None:
        if not hasattr(self, "as_tree"):
            return
        self._as_destroy_kind_editor()
        for item in self.as_tree.get_children():
            self.as_tree.delete(item)
        rows = filter_rows(
            self._audio_scan_rows,
            self.as_filter.get(),
            kind_filter=self.as_kind_filter.get(),
        )
        missing = paths_needing_duration(rows, only_paths={r.path for r in rows})
        if missing:
            self._start_duration_backfill(missing)
        rows = sort_rows(
            rows,
            sort_key_from_label(self.as_sort_label.get()),
            descending=bool(self.as_sort_desc.get()),
        )
        for r in rows:
            root_short = Path(r.root).name or r.root
            self.as_tree.insert(
                "",
                tk.END,
                values=(
                    r.kind_label,
                    root_short,
                    r.rel_path,
                    r.name,
                    format_size(r.size_bytes),
                    format_duration(r.duration_sec),
                    r.path,
                ),
            )
        total_bytes = sum(r.size_bytes for r in rows)
        roots_n = len({r.root for r in rows})
        kind_parts = [
            f"{k}={n}" for k, n in sorted(count_by_kind(self._audio_scan_rows).items()) if n
        ]
        kind_s = "  ".join(kind_parts) if kind_parts else ""
        self.as_status.set(
            f"showing={len(rows)}  roots={roots_n}  total={format_size(total_bytes)}"
            + (f"  |  {kind_s}" if kind_s else "")
        )

    # ---- Tab 2: Source Audio ----

    def _build_tab_source_audio(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(parent, text="Source files from last scan", padding=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.columnconfigure(0, weight=1)

        hint = ttk.Label(
            ctrl,
            text="Note/Score click to edit · Cvt opens Convert · green rows = converted · Auto process via right-click",
        )
        hint.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._register_wrap_label(hint, fraction=0.9)

        filt = ttk.Frame(ctrl)
        filt.grid(row=1, column=0, sticky="ew")
        filt.columnconfigure(1, weight=1)
        ttk.Label(filt, text="Filter").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(filt, textvariable=self.src_filter).grid(row=0, column=1, sticky="ew")
        ttk.Label(filt, text="Sort").grid(row=0, column=2, sticky="w", padx=(12, 8))
        self.src_sort_combo = ttk.Combobox(
            filt,
            textvariable=self.src_sort_label,
            values=list(SRC_SORT_LABELS.values()),
            state="readonly",
            width=14,
        )
        self.src_sort_combo.grid(row=0, column=3, sticky="w")
        self.src_sort_combo.bind("<<ComboboxSelected>>", self._on_src_sort_changed)
        ttk.Checkbutton(
            filt,
            text="Desc",
            variable=self.src_sort_desc,
            command=self._on_src_sort_changed,
        ).grid(row=0, column=4, padx=(8, 0), sticky="w")

        score_row = ttk.Frame(ctrl)
        score_row.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(score_row, text="Score").grid(row=0, column=0, sticky="w", padx=(0, 8))
        for i, var in enumerate(self.src_score_filters):
            ttk.Checkbutton(
                score_row,
                text=format_score_stars(i),
                variable=var,
                command=self._on_src_score_filter_changed,
            ).grid(row=0, column=i + 1, sticky="w", padx=(0, 10))

        actions = ttk.Frame(ctrl)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Apply", command=self._refresh_source_list, width=8).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(actions, text="Refresh", command=self._src_manual_refresh, width=8).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(actions, text="Play", command=self._src_play_selected, width=8).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(
            actions, text="Process", command=self._src_go_process, style=BTN_STYLE_PROCESS, width=9
        ).pack(side="left")

        self.src_status = tk.StringVar(value="No source files yet.")
        ttk.Label(ctrl, textvariable=self.src_status).grid(
            row=4, column=0, sticky="w", pady=(6, 0)
        )

        table = ttk.LabelFrame(parent, text="Source audio", padding=6)
        table.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.src_tree = ttk.Treeview(
            table, columns=SRC_TREE_COLUMNS, show="headings", selectmode="browse"
        )
        headings = {
            "name": "File",
            "size": "Size",
            "duration": "Dur",
            "note": "Note",
            "score": "Score",
            "converted": "Cvt",
            "path": "Full path",
        }
        for key, label in headings.items():
            self.src_tree.heading(key, text=label)
            width = self._src_tree_col_widths.get(key, SRC_TREE_COL_DEFAULTS[key])
            anchor = "center" if key in ("score", "converted", "duration", "size") else "w"
            self.src_tree.column(
                key,
                width=width,
                anchor=anchor,
                stretch=(key in ("note", "path")),
            )
        ysb = ttk.Scrollbar(table, orient="vertical", command=self.src_tree.yview)
        xsb = ttk.Scrollbar(table, orient="horizontal", command=self.src_tree.xview)
        self.src_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.src_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")

        self.src_tree.tag_configure(
            "converted",
            background=self.colors.get("converted_bg", "#1e3d28"),
            foreground=self.colors.get("converted_fg", "#b8f0c0"),
        )

        self.src_tree.bind("<Double-1>", self._src_on_tree_double_click)
        self.src_tree.bind("<Button-1>", self._src_on_tree_click, add="+")
        self.src_tree.bind("<Button-3>", self._src_on_tree_right_click)
        self.src_tree.bind("<ButtonRelease-1>", self._on_src_tree_column_resize, add="+")
        self.src_tree.bind("<<TreeviewSelect>>", self._on_src_tree_select)

        self._src_context_menu = tk.Menu(self.root, tearoff=0)
        self._src_context_menu.add_command(
            label="Play",
            command=self._src_play_context_item,
        )
        self._src_context_menu.add_command(
            label="Auto process",
            command=self._src_auto_process_context_item,
        )
        self._src_context_menu.add_command(
            label="Show in Audio Scan",
            command=self._src_show_in_audio_scan,
        )
        self._src_context_menu.add_command(
            label="Show in Explorer",
            command=self._src_show_context_path_in_explorer,
        )

    @staticmethod
    def _parse_src_tree_col_widths(raw: object) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for key in SRC_TREE_COLUMNS:
            if key not in raw:
                continue
            try:
                width = int(raw[key])
            except (TypeError, ValueError):
                continue
            if width >= 40:
                out[key] = width
        if "name" not in out and "rel" in raw:
            try:
                width = int(raw["rel"])
            except (TypeError, ValueError):
                pass
            else:
                if width >= 40:
                    out["name"] = width
        return out

    def _capture_src_tree_col_widths(self) -> None:
        if not hasattr(self, "src_tree"):
            return
        for key in SRC_TREE_COLUMNS:
            try:
                width = int(self.src_tree.column(key, "width"))
            except (tk.TclError, TypeError, ValueError):
                continue
            if width >= 40:
                self._src_tree_col_widths[key] = width

    def _apply_src_tree_col_widths(self) -> None:
        if not hasattr(self, "src_tree"):
            return
        for key in SRC_TREE_COLUMNS:
            width = self._src_tree_col_widths.get(key, SRC_TREE_COL_DEFAULTS[key])
            try:
                self.src_tree.column(key, width=width)
            except tk.TclError:
                pass

    def _on_src_tree_column_resize(self, event: tk.Event) -> None:
        if not hasattr(self, "src_tree"):
            return
        region = self.src_tree.identify_region(event.x, event.y)
        if region not in ("separator", "heading"):
            return
        self._capture_src_tree_col_widths()
        if self._src_colwidth_save_after_id is not None:
            try:
                self.root.after_cancel(self._src_colwidth_save_after_id)
            except tk.TclError:
                pass
        self._src_colwidth_save_after_id = self.root.after(
            300, self._save_src_col_widths_debounced
        )

    def _save_src_col_widths_debounced(self) -> None:
        self._src_colwidth_save_after_id = None
        self._save_settings()

    def _on_src_tree_select(self, _event: tk.Event | None = None) -> None:
        # Don't rewrite Convert paths while a job is using the locked snapshot.
        if self.running:
            return
        path = self._src_selected_path()
        if path:
            self._set_convert_context_for_source(path)

    def _set_convert_context_for_source(
        self,
        source_path: str,
        *,
        vocals: str | None = None,
        inst: str | None = None,
    ) -> None:
        if not source_path.strip() or not Path(source_path).is_file():
            return
        self._convert_source_path = str(Path(source_path).resolve())
        if vocals is None or inst is None:
            found_v, found_i = find_separated_for_source(
                source_path, self._audio_scan_rows, self._stem_links
            )
            vocals = vocals or found_v
            inst = inst or found_i
        if vocals:
            self.im_input_path.set(vocals)
        self._ensure_convert_stem_paths(self._convert_source_path)
        self._refresh_im_auto_paths()
        self._refresh_convert_results_list()

    def _src_note_column_id(self) -> str:
        return "#4"

    def _src_score_column_id(self) -> str:
        return "#5"

    def _src_converted_column_id(self) -> str:
        return "#6"

    def _src_path_column_id(self) -> str:
        return "#7"

    def _src_path_from_item(self, item: str) -> str | None:
        vals = self.src_tree.item(item, "values")
        if not vals or len(vals) < 7:
            return None
        path = str(vals[6]).strip()
        if not path or not Path(path).is_file():
            return None
        return path

    def _src_on_tree_right_click(self, event: tk.Event) -> None:
        if self.src_tree.identify_region(event.x, event.y) != "cell":
            return
        item = self.src_tree.identify_row(event.y)
        if not item:
            return
        path = self._src_path_from_item(item)
        if not path:
            return
        self.src_tree.selection_set(item)
        self.src_tree.focus(item)
        self._src_context_reveal_path = path
        try:
            self._src_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._src_context_menu.grab_release()

    def _src_play_context_item(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        self._playback_play_path(path)

    def _src_process_context_item(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        self._src_go_process_path(path)

    def _src_auto_process_context_item(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        self._src_auto_process_path(path)

    def _src_auto_process_path(self, path: str) -> None:
        """Separate (if needed) then Convert with the currently selected model."""
        if self.running:
            messagebox.showwarning("Busy", "Wait for the current job to finish, or Stop it.")
            return
        if not path or not Path(path).is_file():
            messagebox.showinfo("Auto process", "Select a valid source file first.")
            return
        model_path = self.im_model_path.get().strip()
        if not model_path or not Path(model_path).is_file():
            messagebox.showerror(
                "Auto process",
                "Select a valid .pth model on the Convert tab first.",
            )
            return
        if not self.im_merge_output_dir.get().strip():
            messagebox.showerror(
                "Auto process",
                "Set merge_output_dir on the Convert tab first.",
            )
            return
        if self._require_rvc_root() is None:
            return

        self._process_source_path = path
        self._set_convert_context_for_source(path)
        vocals, inst = self._resolve_stems_for_source(path)
        self._process_vocals_path = vocals
        self._process_inst_path = inst
        self._refresh_process_tab()

        if vocals and inst and Path(vocals).is_file() and Path(inst).is_file():
            self.log_queue.put(
                f"[auto process] stems already present — skip separate\n"
                f"  vocals: {vocals}\n  instrumental: {inst}\n"
            )
            self._set_convert_context_for_source(path, vocals=vocals, inst=inst)
            self.notebook.select(TAB_CONVERT)
            self.run_convert()
            return

        out = str(Path(path).parent)
        self.log_queue.put(f"[auto process] separate then convert: {path}\n")
        self.notebook.select(TAB_PROCESS)

        def after_separate() -> None:
            self._refresh_process_tab()
            v = self._process_vocals_path
            i = self._process_inst_path
            if not v or not i or not Path(v).is_file() or not Path(i).is_file():
                self._set_running(False)
                messagebox.showerror(
                    "Auto process",
                    "Separate finished but vocals/instrumental stems were not found.",
                )
                return
            self._set_convert_context_for_source(path, vocals=v, inst=i)
            self.notebook.select(TAB_CONVERT)
            # Separate kept running=True (release_running=False); chain into Convert.
            self.run_convert(chain=True)

        self.run_separate(
            inp=path,
            out=out,
            quiet=True,
            fill_infer_merge=True,
            release_running=False,
            on_success=after_separate,
        )

    def _src_show_in_audio_scan(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        if not self._as_focus_path(path):
            messagebox.showinfo(
                "Audio Scan",
                "File not found in the last scan.\nRun Scan on the Audio Scan tab first.",
            )
            return
        self.notebook.select(TAB_AUDIO_SCAN)

    def _src_show_in_convert(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        self._set_convert_context_for_source(path)
        self.notebook.select(TAB_CONVERT)

    def _as_focus_path(self, path: str) -> bool:
        from audio_kind import kind_matches_filter
        from audio_scan import _path_key

        target = _path_key(path)
        row = next(
            (r for r in self._audio_scan_rows if _path_key(r.path) == target),
            None,
        )
        if row is None:
            return False

        changed_filter = False
        if not kind_matches_filter(row.kind, self.as_kind_filter.get()):
            self.as_kind_filter.set("all")
            changed_filter = True
        q = self.as_filter.get().strip().lower()
        if q:
            hay = f"{row.rel_path} {row.name} {row.root} {row.kind_label}".lower()
            if q not in hay:
                self.as_filter.set("")
                changed_filter = True
        if changed_filter:
            self._as_apply_filter()

        for item in self.as_tree.get_children():
            vals = self.as_tree.item(item, "values")
            if len(vals) < 7:
                continue
            if _path_key(str(vals[6])) != target:
                continue
            self.as_tree.selection_set(item)
            self.as_tree.focus(item)
            self.as_tree.see(item)
            return True
        return False

    def _src_show_context_path_in_explorer(self) -> None:
        path = self._src_context_reveal_path or self._src_selected_path()
        if not path:
            return
        try:
            reveal_path_in_file_manager(path)
        except FileNotFoundError:
            messagebox.showerror("Show in Explorer", f"File not found:\n{path}")
        except OSError as exc:
            messagebox.showerror("Show in Explorer", str(exc))

    def _src_destroy_note_editor(self) -> None:
        if self._src_note_editor is not None:
            try:
                self._src_note_editor.destroy()
            except tk.TclError:
                pass
        self._src_note_editor = None
        self._src_note_edit_item = None
        self._src_note_edit_path = None

    def _src_begin_note_edit(self, item: str) -> None:
        if not item:
            return
        path = self._src_path_from_item(item)
        if not path:
            return
        self._src_destroy_note_editor()
        bbox = self.src_tree.bbox(item, column="note")
        if not bbox:
            return
        x, y, w, h = bbox
        note = get_source_note(path, self._stem_links)
        try:
            bg = self.colors["field"]
            fg = self.colors["fg"]
        except Exception:
            bg = "white"
            fg = "black"
        entry = tk.Entry(
            self.src_tree,
            borderwidth=0,
            highlightthickness=1,
            bg=bg,
            fg=fg,
        )
        entry.insert(0, note)
        entry.select_range(0, tk.END)
        entry.place(x=x, y=y, width=max(w, 80), height=h)
        entry.focus_set()
        self._src_note_editor = entry
        self._src_note_edit_item = item
        self._src_note_edit_path = path
        entry.bind("<Return>", self._src_commit_note_edit)
        entry.bind("<Escape>", self._src_cancel_note_edit)
        entry.bind("<FocusOut>", self._src_commit_note_edit)

    def _src_commit_note_edit(self, _event: tk.Event | None = None) -> None:
        if self._src_note_committing:
            return
        editor = self._src_note_editor
        item = self._src_note_edit_item
        path = self._src_note_edit_path
        if editor is None or not item or not path:
            self._src_destroy_note_editor()
            return
        self._src_note_committing = True
        try:
            note = editor.get()
            if set_source_note(self._stem_links, path, note):
                self._persist_stem_links()
                if set_note_for_stem(self._shared_catalog, path, note):
                    self._persist_shared_catalog()
            vals = list(self.src_tree.item(item, "values"))
            if len(vals) >= 7:
                vals[3] = note
                self.src_tree.item(item, values=vals)
        finally:
            self._src_destroy_note_editor()
            self._src_note_committing = False

    def _src_cancel_note_edit(self, _event: tk.Event | None = None) -> None:
        self._src_destroy_note_editor()

    def _src_set_score(self, path: str, score: int, *, item: str | None = None) -> None:
        if not path:
            return
        if not set_source_score(self._stem_links, path, score):
            return
        self._persist_stem_links()
        if set_score_for_stem(self._shared_catalog, path, score):
            self._persist_shared_catalog()
        self._invalidate_source_display_cache()
        stars = format_score_stars(score)
        if item and self.src_tree.exists(item):
            vals = list(self.src_tree.item(item, "values"))
            if len(vals) >= 7:
                vals[4] = stars
                self.src_tree.item(item, values=vals)
        else:
            self._refresh_source_list(sort_only=True)

    def _src_on_tree_click(self, event: tk.Event) -> None:
        if self.src_tree.identify_region(event.x, event.y) != "cell":
            return
        col = self.src_tree.identify_column(event.x)
        item = self.src_tree.identify_row(event.y)
        if not item:
            return
        if col == self._src_note_column_id():
            self.root.after_idle(lambda i=item: self._src_begin_note_edit(i))
            return
        if col == self._src_score_column_id():
            path = self._src_path_from_item(item)
            if not path:
                return
            bbox = self.src_tree.bbox(item, column="score")
            if not bbox:
                return
            x, _y, w, _h = bbox
            clicked = score_from_click_x(event.x - x, w)
            cur = get_source_score(path, self._stem_links)
            new_score = 0 if clicked == cur else clicked
            self._src_set_score(path, new_score, item=item)
            return
        if col == self._src_converted_column_id():
            path = self._src_path_from_item(item)
            if not path:
                return
            self.src_tree.selection_set(item)
            self.src_tree.focus(item)
            self._src_context_reveal_path = path
            self.root.after_idle(self._src_show_in_convert)

    def _src_on_tree_double_click(self, event: tk.Event) -> None:
        col = self.src_tree.identify_column(event.x)
        if col in (
            self._src_note_column_id(),
            self._src_score_column_id(),
            self._src_converted_column_id(),
        ):
            return
        self._src_play_selected()

    def _on_src_sort_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_source_list(sort_only=True)
        self._save_settings()

    def _on_src_score_filter_changed(self) -> None:
        self._refresh_source_list(sort_only=True)
        self._save_settings()

    def _src_selected_path(self) -> str | None:
        if not hasattr(self, "src_tree"):
            return None
        sel = self.src_tree.selection()
        if not sel:
            return None
        return self._src_path_from_item(sel[0])

    def _src_manual_refresh(self) -> None:
        self._stem_links = load_stem_links(self._db)
        self._invalidate_source_display_cache()
        self._refresh_source_list()

    def _src_play_selected(self) -> None:
        path = self._src_selected_path()
        if not path:
            messagebox.showinfo("Source Audio", "Select a file in the list first.")
            return
        self._playback_play_path(path)

    def _src_go_process(self) -> None:
        path = self._src_selected_path()
        if not path:
            messagebox.showinfo("Source Audio", "Select a file in the list first.")
            return
        self._src_go_process_path(path)

    def _src_go_process_path(self, path: str) -> None:
        self._process_source_path = path
        self._refresh_process_tab()
        self.notebook.select(TAB_PROCESS)

    def _refresh_source_list(self, *, sort_only: bool = False) -> None:
        if not hasattr(self, "src_tree"):
            return
        self._src_destroy_note_editor()
        for item in self.src_tree.get_children():
            self.src_tree.delete(item)
        rows = filter_rows(
            self._audio_scan_rows,
            self.src_filter.get(),
            kind_filter="source",
        )
        missing = paths_needing_duration(rows, only_paths={r.path for r in rows})
        if missing:
            self._start_duration_backfill(missing)
        cache_key = self._source_display_cache_key()
        if (
            sort_only
            and cache_key == self._src_display_cache_key
            and self._src_display_cache_payload is not None
        ):
            payload = self._src_display_cache_payload
        else:
            payload = self._build_source_display_payload(rows)
            self._src_display_cache_key = cache_key
            self._src_display_cache_payload = payload

        score_sel = self._src_selected_scores()
        if score_sel is not None:
            payload = [item for item in payload if item[3] in score_sel]

        sort_by = sort_key_from_label(self.src_sort_label.get())
        descending = bool(self.src_sort_desc.get())
        if sort_by == "score":
            ordered = sorted(
                payload,
                key=lambda it: (it[3], it[0].name.lower()),
                reverse=descending,
            )
        else:
            sorted_rows = sort_rows(
                [item[0] for item in payload],
                sort_by,
                descending=descending,
            )
            by_path = {item[0].path: item for item in payload}
            ordered = []
            for r in sorted_rows:
                item = by_path.get(r.path)
                if item is not None:
                    ordered.append(item)

        for row, note, cvt_n, score in ordered:
            self.src_tree.insert(
                "",
                tk.END,
                values=(
                    row.name,
                    format_size(row.size_bytes),
                    format_duration(row.duration_sec),
                    note,
                    format_score_stars(score),
                    str(cvt_n),
                    row.path,
                ),
                tags=("converted",) if cvt_n > 0 else (),
            )
        total_bytes = sum(item[0].size_bytes for item in ordered)
        scored = sum(1 for _r, _n, _c, sc in ordered if sc > 0)
        converted = sum(1 for _r, _n, c, _sc in ordered if c > 0)
        self.src_status.set(
            f"showing={len(ordered)}  scored={scored}  converted={converted}  "
            f"total={format_size(total_bytes)}"
        )

    # ---- Tab 3: Process ----

    def _build_tab_process(self, parent: ttk.Frame) -> None:
        info = ttk.LabelFrame(parent, text="Source audio", padding=10)
        info.pack(fill="x", padx=4, pady=4)
        self._configure_cols(info)

        ttk.Label(info, text="File").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.proc_name_var = tk.StringVar(value="(none)")
        ttk.Label(info, textvariable=self.proc_name_var).grid(
            row=0, column=1, columnspan=2, sticky="w", pady=4
        )
        ttk.Button(info, text="Play", command=self._proc_play_source).grid(
            row=0, column=3, sticky="e", pady=4
        )

        ttk.Label(info, text="Path").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=4)
        self.proc_path_var = tk.StringVar(value="")
        proc_path_lbl = ttk.Label(info, textvariable=self.proc_path_var)
        proc_path_lbl.grid(row=1, column=1, columnspan=3, sticky="w", pady=4)
        self._register_wrap_label(proc_path_lbl, fraction=0.75)

        ttk.Label(info, text="Size").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.proc_size_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.proc_size_var).grid(row=2, column=1, sticky="w", pady=4)

        stems = ttk.LabelFrame(parent, text="Separated stems", padding=10)
        stems.pack(fill="x", padx=4, pady=4)
        self._configure_cols(stems)

        self.proc_vocals_var = tk.StringVar(value="(not found)")
        self.proc_inst_var = tk.StringVar(value="(not found)")
        self._process_vocals_path: str | None = None
        self._process_inst_path: str | None = None

        ttk.Label(stems, text="Vocals").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        vocals_lbl = ttk.Label(stems, textvariable=self.proc_vocals_var)
        vocals_lbl.grid(row=0, column=1, sticky="ew", pady=4)
        self._register_wrap_label(vocals_lbl, fraction=0.7)
        self.proc_vocals_play = ttk.Button(
            stems, text="Play", command=self._proc_play_vocals, state="disabled"
        )
        self.proc_vocals_play.grid(row=0, column=2, sticky="e", pady=4)

        ttk.Label(stems, text="Instrumental").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        inst_lbl = ttk.Label(stems, textvariable=self.proc_inst_var)
        inst_lbl.grid(row=1, column=1, sticky="ew", pady=4)
        self._register_wrap_label(inst_lbl, fraction=0.7)
        self.proc_inst_play = ttk.Button(
            stems, text="Play", command=self._proc_play_inst, state="disabled"
        )
        self.proc_inst_play.grid(row=1, column=2, sticky="e", pady=4)

        self.proc_sep_frame = ttk.Frame(parent)
        self.proc_sep_frame.pack(fill="x", padx=4, pady=8)
        self.proc_sep_btn = ttk.Button(
            self.proc_sep_frame,
            text="Separate",
            command=self._proc_run_separate,
            style=BTN_STYLE_SEPARATE,
        )
        self.proc_sep_btn.pack(side="left", padx=(0, 8))
        self.proc_convert_btn = ttk.Button(
            self.proc_sep_frame,
            text="Convert",
            command=self._proc_go_convert,
            style=BTN_STYLE_CONVERT,
        )
        self.proc_convert_btn.pack(side="left")

        proc_tip = ttk.Label(
            parent,
            text="Open from Source Audio → Process. MelBand outputs are written beside the source.",
        )
        proc_tip.pack(anchor="w", padx=8, pady=6)
        self._register_wrap_label(proc_tip, fraction=0.85)

    def _proc_play_source(self) -> None:
        if self._process_source_path:
            self._playback_play_path(self._process_source_path)

    def _proc_play_vocals(self) -> None:
        if self._process_vocals_path:
            self._playback_play_path(self._process_vocals_path)

    def _proc_play_inst(self) -> None:
        if self._process_inst_path:
            self._playback_play_path(self._process_inst_path)

    def _resolve_stems_for_source(self, source: str) -> tuple[str | None, str | None]:
        vocals, inst = find_separated_for_source(
            source,
            self._audio_scan_rows,
            self._stem_links,
        )
        if upsert_stem_link(
            self._stem_links,
            source,
            vocals=vocals,
            instrumental=inst,
        ):
            self._persist_stem_links()
        return vocals, inst

    def _refresh_process_tab(self) -> None:
        if not hasattr(self, "proc_name_var"):
            return
        src = self._process_source_path
        if not src or not Path(src).is_file():
            self.proc_name_var.set("(none — select Source Audio → Process)")
            self.proc_path_var.set("")
            self.proc_size_var.set("")
            self.proc_vocals_var.set("(not found)")
            self.proc_inst_var.set("(not found)")
            self._process_vocals_path = None
            self._process_inst_path = None
            self.proc_vocals_play.config(state="disabled")
            self.proc_inst_play.config(state="disabled")
            self.proc_sep_btn.pack(side="left", padx=(0, 8))
            self.proc_convert_btn.pack_forget()
            if not self.proc_sep_frame.winfo_ismapped():
                self.proc_sep_frame.pack(fill="x", padx=4, pady=8)
            self.proc_sep_btn.config(state="disabled")
            return

        p = Path(src)
        self.proc_name_var.set(p.name)
        self.proc_path_var.set(str(p.resolve()))
        self.proc_size_var.set(format_size(p.stat().st_size))

        vocals, inst = self._resolve_stems_for_source(src)
        self._process_vocals_path = vocals
        self._process_inst_path = inst

        if vocals:
            self.proc_vocals_var.set(vocals)
            self.proc_vocals_play.config(state="normal")
        else:
            self.proc_vocals_var.set("(not found)")
            self.proc_vocals_play.config(state="disabled")

        if inst:
            self.proc_inst_var.set(inst)
            self.proc_inst_play.config(state="normal")
        else:
            self.proc_inst_var.set("(not found)")
            self.proc_inst_play.config(state="disabled")

        need_sep = not vocals or not inst
        if need_sep:
            if not self.proc_sep_frame.winfo_ismapped():
                self.proc_sep_frame.pack(fill="x", padx=4, pady=8)
            self.proc_sep_btn.pack(side="left", padx=(0, 8))
            self.proc_convert_btn.pack_forget()
            self.proc_sep_btn.config(state="disabled" if self.running else "normal")
        else:
            if not self.proc_sep_frame.winfo_ismapped():
                self.proc_sep_frame.pack(fill="x", padx=4, pady=8)
            self.proc_sep_btn.pack_forget()
            self.proc_convert_btn.pack(side="left")
            self.proc_convert_btn.config(state="disabled" if self.running else "normal")

    def _proc_go_convert(self) -> None:
        vocals = self._process_vocals_path
        inst = self._process_inst_path
        if not vocals or not inst:
            messagebox.showinfo("Process", "Both vocals and instrumental stems are required.")
            return
        if not self._process_source_path:
            messagebox.showinfo("Process", "No source file selected.")
            return
        self._set_convert_context_for_source(
            self._process_source_path,
            vocals=vocals,
            inst=inst,
        )
        self.notebook.select(TAB_CONVERT)

    def _proc_run_separate(self) -> None:
        src = self._process_source_path
        if not src or not Path(src).is_file():
            messagebox.showinfo("Process", "No source file selected.")
            return
        out = str(Path(src).parent)
        self.run_separate(
            inp=src,
            out=out,
            quiet=True,
            fill_infer_merge=False,
            on_success=self._refresh_process_tab,
        )

    # ---- Tab 4: Separate ----

    def _build_tab_separate(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Vocals + Instrumental separation", padding=10)
        f.pack(fill="x", padx=4, pady=4)
        self._configure_cols(f)

        self._row_path(f, 0, "input_path", self.sep_input_path, self._browse_sep_input)
        self._row_path(f, 1, "output_dir", self.sep_output_dir, self._browse_sep_output)

        ttk.Label(f, text="Model").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f,
            textvariable=self.sep_model_label,
            values=list(SEP_MODELS.keys()),
            state="readonly",
        ).grid(row=2, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(f, text="Format").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            f,
            textvariable=self.sep_format,
            values=["FLAC", "WAV", "MP3"],
            width=10,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(f, text="Segment").grid(row=3, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Combobox(
            f,
            textvariable=self.sep_segment,
            values=["128", "160", "256", "320", "512"],
            width=10,
            state="readonly",
        ).grid(row=3, column=3, sticky="w", pady=4)

        ttk.Checkbutton(
            f,
            text="After success, fill Convert (vocals + instrumental stems)",
            variable=self.sep_fill_infer_merge,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 4))

        self.sep_start_btn = ttk.Button(
            f, text="Start Separate", command=self.run_separate, style=BTN_STYLE_SEPARATE
        )
        self.sep_start_btn.grid(row=5, column=0, columnspan=5, sticky="ew", pady=8)

        sep_tip = ttk.Label(
            parent,
            text="Exports Vocals + Instrumental via local audio-separator venv (see Settings).",
        )
        sep_tip.pack(anchor="w", padx=8, pady=6)
        self._register_wrap_label(sep_tip, fraction=0.85)

    def _browse_sep_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio to separate",
            filetypes=AUDIO_FILETYPES,
            **self._path_dialog_opts(self.sep_input_path.get(), for_file=True),
        )
        if path:
            self.sep_input_path.set(path)

    def _browse_sep_output(self) -> None:
        path = filedialog.askdirectory(
            title="Select separation output folder",
            **self._path_dialog_opts(self.sep_output_dir.get()),
        )
        if path:
            self.sep_output_dir.set(path)

    def _resolve_separator_python(self) -> Path | None:
        venv = Path(self.sep_venv_dir.get().strip() or str(DEFAULT_SEP_VENV)).expanduser()
        for c in (
            venv / "Scripts" / "python.exe",
            venv / "Scripts" / "python",
            venv / "bin" / "python",
        ):
            if c.is_file():
                return c
        return None

    def run_separate(
        self,
        *,
        inp: str | None = None,
        out: str | None = None,
        on_success: Callable[[], None] | None = None,
        quiet: bool = False,
        fill_infer_merge: bool | None = None,
        release_running: bool = True,
    ) -> None:
        inp = (inp or self.sep_input_path.get()).strip()
        out = (out or self.sep_output_dir.get()).strip()
        if not inp or not Path(inp).is_file():
            messagebox.showerror("Missing input", "Select a valid input_path.")
            return
        if not out:
            messagebox.showerror("Missing output", "Select a valid output_dir.")
            return
        py = self._resolve_separator_python()
        if py is None:
            messagebox.showerror(
                "Separator missing",
                "python.exe not found in separator venv.\n"
                f"Current venv: {self.sep_venv_dir.get().strip() or DEFAULT_SEP_VENV}\n"
                "Run setup_separator.ps1 or point Settings → Separator venv.",
            )
            return
        wrapper = SCRIPTS_DIR / "run_audio_separator.py"
        if not wrapper.is_file():
            messagebox.showerror("Missing script", f"Not found:\n{wrapper}")
            return
        model_dir = Path(self.sep_model_dir.get().strip() or str(DEFAULT_SEP_MODEL_DIR))
        model_label = self.sep_model_label.get().strip()
        model_file = SEP_MODELS.get(model_label)
        if not model_file:
            messagebox.showerror("Model", f"Unknown model label: {model_label}")
            return
        model_path = model_dir / model_file
        if not model_path.is_file():
            messagebox.showerror(
                "Model missing",
                f"Model file not found:\n{model_path}\n"
                "Put checkpoints in the model dir (see Settings).",
            )
            return

        Path(out).mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Do not call audio-separator.exe: moving a venv leaves broken absolute paths
        # inside the console-script launcher (silent exit code 1).
        cmd = [
            str(py),
            "-u",
            str(wrapper),
            inp,
            "-m",
            model_file,
            "--model_file_dir",
            str(model_dir),
            "--output_dir",
            out,
            "--output_format",
            self.sep_format.get().strip() or "FLAC",
            "--mdxc_segment_size",
            self.sep_segment.get().strip() or "256",
            "--log_level",
            "info",
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        scripts = py.parent
        env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
        proxy = self.sep_proxy.get().strip()
        if proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env[key] = proxy

        def on_done() -> None:
            do_fill = (
                fill_infer_merge
                if fill_infer_merge is not None
                else self.sep_fill_infer_merge.get()
            )
            if do_fill:
                self._maybe_fill_infer_merge_from_sep(out)
            if inp:
                self._resolve_stems_for_source(inp)
            if on_success is not None:
                on_success()
            elif not quiet:
                messagebox.showinfo("Separate done", f"Outputs in:\n{out}")

        self._run_cmd(
            cmd,
            rvc_root=PACKAGE_DIR,
            cwd=PACKAGE_DIR,
            env=env,
            on_success=on_done,
            release_running=release_running,
        )

    def _maybe_fill_infer_merge_from_sep(self, out_dir: str) -> None:
        """Pick newest Vocals / Instrumental (or Other) stems in out_dir."""
        root = Path(out_dir)
        if not root.is_dir():
            return
        files = [p for p in root.iterdir() if p.is_file()]
        if not files:
            return

        def newest(preds: list[Callable[[str], bool]]) -> Path | None:
            hits = [p for p in files if any(pred(p.name.lower()) for pred in preds)]
            if not hits:
                return None
            return max(hits, key=lambda p: p.stat().st_mtime)

        vocals = newest([lambda n: "(vocals)" in n])
        instru = newest(
            [
                lambda n: "(instrumental)" in n,
                lambda n: "(other)" in n,
            ]
        )
        if vocals is not None:
            self.im_input_path.set(str(vocals.resolve()))
            self.log_queue.put(f"[separate] filled Convert vocals: {vocals}\n")
        if instru is not None:
            self.log_queue.put(f"[separate] found instrumental: {instru}\n")
        source = self._resolve_convert_source()
        if source and (vocals is not None or instru is not None):
            if upsert_stem_link(
                self._stem_links,
                source,
                vocals=str(vocals.resolve()) if vocals else None,
                instrumental=str(instru.resolve()) if instru else None,
            ):
                self._persist_stem_links()
        self._refresh_im_auto_paths()
        self._refresh_convert_results_list()

    # ---- Tab 5: Convert ----

    def _build_tab_convert(self, parent: ttk.Frame) -> None:
        top_row = ttk.Frame(parent)
        top_row.pack(fill="both", expand=True, padx=4, pady=4)
        top_row.columnconfigure(0, weight=1, uniform="convert_top")
        top_row.columnconfigure(1, weight=1, uniform="convert_top")
        top_row.rowconfigure(0, weight=1)

        source_box = ttk.LabelFrame(top_row, text="Converting source audio", padding=10)
        source_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._configure_cols(source_box)

        ttk.Label(source_box, text="File").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.conv_source_name_var = tk.StringVar(value="(not linked)")
        ttk.Label(source_box, textvariable=self.conv_source_name_var).grid(
            row=0, column=1, columnspan=3, sticky="w", pady=4
        )
        ttk.Label(source_box, text="Path").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=4)
        self.conv_source_path_var = tk.StringVar(value="")
        conv_src_lbl = ttk.Label(source_box, textvariable=self.conv_source_path_var)
        conv_src_lbl.grid(row=1, column=1, columnspan=3, sticky="w", pady=4)
        self._register_wrap_label(conv_src_lbl, fraction=0.42)
        ttk.Label(source_box, text="Vocals (infer)").grid(
            row=2, column=0, sticky="nw", padx=(0, 8), pady=4
        )
        self.conv_vocals_path_var = tk.StringVar(value="")
        conv_voc_lbl = ttk.Label(source_box, textvariable=self.conv_vocals_path_var)
        conv_voc_lbl.grid(row=2, column=1, columnspan=3, sticky="w", pady=4)
        self._register_wrap_label(conv_voc_lbl, fraction=0.42)
        ttk.Label(source_box, text="Instrumental (merge)").grid(
            row=3, column=0, sticky="nw", padx=(0, 8), pady=4
        )
        self.conv_bgm_path_var = tk.StringVar(value="")
        conv_bgm_lbl = ttk.Label(source_box, textvariable=self.conv_bgm_path_var)
        conv_bgm_lbl.grid(row=3, column=1, columnspan=3, sticky="w", pady=4)
        self._register_wrap_label(conv_bgm_lbl, fraction=0.42)
        conv_tip = ttk.Label(
            source_box,
            text="Set from Source Audio → Process → Convert. Infer uses separated vocals.",
        )
        conv_tip.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self._register_wrap_label(conv_tip, fraction=0.42)

        results_box = ttk.LabelFrame(top_row, text="Convert results", padding=6)
        results_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        results_box.columnconfigure(0, weight=1)
        results_box.rowconfigure(1, weight=1)

        result_ctrl = ttk.Frame(results_box)
        result_ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.conv_results_status = tk.StringVar(value="No source linked.")
        ttk.Label(result_ctrl, textvariable=self.conv_results_status).pack(side="left")
        ttk.Button(result_ctrl, text="Refresh", command=self._refresh_convert_results_list).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(result_ctrl, text="Play selected", command=self._conv_play_selected).pack(
            side="right", padx=(8, 0)
        )

        cols = ("name", "size", "path")
        self.conv_results_tree = ttk.Treeview(
            results_box, columns=cols, show="headings", height=8, selectmode="browse"
        )
        headings = {
            "name": ("File", 140),
            "size": ("Size", 64),
            "path": ("Full path", 220),
        }
        for key, (label, width) in headings.items():
            self.conv_results_tree.heading(key, text=label)
            self.conv_results_tree.column(key, width=width, anchor="w", stretch=(key == "path"))
        conv_ysb = ttk.Scrollbar(results_box, orient="vertical", command=self.conv_results_tree.yview)
        conv_xsb = ttk.Scrollbar(results_box, orient="horizontal", command=self.conv_results_tree.xview)
        self.conv_results_tree.configure(yscrollcommand=conv_ysb.set, xscrollcommand=conv_xsb.set)
        self.conv_results_tree.grid(row=1, column=0, sticky="nsew")
        conv_ysb.grid(row=1, column=1, sticky="ns")
        conv_xsb.grid(row=2, column=0, sticky="ew")
        self.conv_results_tree.bind("<Double-1>", lambda _e: self._conv_play_selected())

        mid_row = ttk.Frame(parent)
        mid_row.pack(fill="x", padx=4, pady=4)
        mid_row.columnconfigure(0, weight=1, uniform="convert_mid")
        mid_row.columnconfigure(1, weight=1, uniform="convert_mid")

        infer = ttk.LabelFrame(mid_row, text="Long infer & merge", padding=10)
        infer.grid(row=0, column=0, sticky="new", padx=(0, 4))
        self._configure_cols(infer)

        ttk.Label(infer, text="model (.pth)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(infer, textvariable=self.im_model_path).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4
        )
        model_btns = ttk.Frame(infer)
        model_btns.grid(row=0, column=3, columnspan=2, sticky="e", pady=4)
        ttk.Button(model_btns, text="Browse", command=self._browse_im_model).pack(
            side="left", padx=2
        )
        ttk.Button(
            model_btns,
            textvariable=self.fav_model_btn_label,
            command=self._toggle_favorite_model,
            width=9,
        ).pack(side="left", padx=2)

        fav_row = ttk.Frame(infer)
        fav_row.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(0, 4))
        fav_row.columnconfigure(1, weight=1)
        ttk.Label(fav_row, text="Favorites").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.fav_model_combo = ttk.Combobox(
            fav_row,
            textvariable=self.fav_model_pick,
            state="readonly",
            width=28,
        )
        self.fav_model_combo.grid(row=0, column=1, sticky="ew")
        self.fav_model_combo.bind("<<ComboboxSelected>>", self._apply_favorite_model)

        self._row_path(infer, 2, "index (.index)", self.im_index_path, self._browse_im_index)
        self._row_readonly(infer, 3, "model_name", self.im_model_name)
        self._row_path(
            infer, 4, "merge_output_dir", self.im_merge_output_dir, self._browse_im_merge_out_dir
        )
        self._row_readonly(infer, 5, "merge_output_path", self.im_merge_output_path)

        params = ttk.LabelFrame(mid_row, text="Infer parameters", padding=10)
        params.grid(row=0, column=1, sticky="new", padx=(4, 0))
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
        ttk.Label(params, text="breath_mix_rate").grid(row=2, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_breath_mix_rate, width=10).grid(
            row=2, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="resample_sr (0=model)").grid(row=3, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_resample_sr, width=10).grid(
            row=3, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="spk_id").grid(row=3, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_spk_id, width=10).grid(
            row=3, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="chunk_sec").grid(row=4, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.im_chunk_sec, width=10).grid(
            row=4, column=1, sticky="w", padx=4
        )
        ttk.Label(params, text="overlap_sec").grid(row=4, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.im_overlap_sec, width=10).grid(
            row=4, column=3, sticky="w", padx=4
        )

        ttk.Label(params, text="filter_radius (harvest only)").grid(
            row=5, column=0, sticky="w"
        )
        ttk.Entry(params, textvariable=self.im_filter_radius, width=10).grid(
            row=5, column=1, sticky="w", padx=4
        )

        conv_params_tip = ttk.Label(
            params,
            text=(
                "Speech tip: protect≈0.33, breath_mix_rate 0.5–0.85 (0=off; needs patched 0718). "
                "index_rate 0.5–0.75. UV F0 no-interp is on in 0718 pipeline."
            ),
        )
        conv_params_tip.grid(row=6, column=0, columnspan=5, sticky="w", pady=(4, 0))
        self._register_wrap_label(conv_params_tip, fraction=0.42)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", padx=4, pady=8)
        self.im_convert_btn = ttk.Button(
            actions, text="Convert", command=self.run_convert, style=BTN_STYLE_CONVERT
        )
        self.im_convert_btn.pack(side="left")

        conv_foot = ttk.Label(
            parent,
            text="Convert runs long infer, then merges with BGM on success.",
        )
        conv_foot.pack(anchor="w", padx=8, pady=6)
        self._register_wrap_label(conv_foot, fraction=0.9)

        for var in (
            self.im_model_path,
            self.im_model_name,
            self.im_input_path,
            self.im_merge_output_dir,
        ):
            var.trace_add("write", lambda *_a: self._refresh_im_auto_paths())
        for var in (self.im_input_path,):
            var.trace_add("write", lambda *_a: self._refresh_convert_results_list())

    def _ensure_convert_stem_paths(self, source: str) -> None:
        link = self._stem_links.get(source_key(source))
        if not link:
            return
        if not self.im_input_path.get().strip() and link.vocals:
            self.im_input_path.set(link.vocals)

    def _resolve_bgm_path(self) -> str | None:
        source = self._resolve_convert_source()
        if not source:
            return None
        _, inst = self._resolve_stems_for_source(source)
        if inst and Path(inst).is_file():
            return inst
        return None

    def _resolve_convert_source(self) -> str | None:
        if self._convert_source_path and Path(self._convert_source_path).is_file():
            return self._convert_source_path
        vocals = self.im_input_path.get().strip()
        if vocals:
            src = find_source_for_vocals(vocals, self._stem_links, self._audio_scan_rows)
            if src:
                self._convert_source_path = src
                return src
        return None

    def _record_convert_result(self, source: str | None, result_path: str) -> None:
        if not source:
            return
        if add_convert_result(self._stem_links, source, result_path):
            self._persist_stem_links()

    def _conv_selected_path(self) -> str | None:
        if not hasattr(self, "conv_results_tree"):
            return None
        sel = self.conv_results_tree.selection()
        if not sel:
            return None
        vals = self.conv_results_tree.item(sel[0], "values")
        if not vals or len(vals) < 3:
            return None
        path = str(vals[2]).strip()
        return path if path and Path(path).is_file() else None

    def _conv_play_selected(self) -> None:
        path = self._conv_selected_path()
        if not path:
            messagebox.showinfo("Convert", "Select a convert result in the list first.")
            return
        self._playback_play_path(path)

    def _refresh_convert_results_list(self) -> None:
        if not hasattr(self, "conv_results_tree"):
            return
        for item in self.conv_results_tree.get_children():
            self.conv_results_tree.delete(item)

        source = self._resolve_convert_source()
        if not source:
            self.conv_source_name_var.set("(not linked)")
            self.conv_source_path_var.set("Open Source Audio → Process → Convert")
            self.conv_vocals_path_var.set("")
            self.conv_bgm_path_var.set("")
            self.conv_results_status.set("No source linked.")
            return

        self._ensure_convert_stem_paths(source)
        p = Path(source)
        self.conv_source_name_var.set(p.name)
        self.conv_source_path_var.set(str(p.resolve()))
        vocals, bgm = self._resolve_stems_for_source(source)
        if not vocals:
            link = self._stem_links.get(source_key(source))
            if link and link.vocals:
                vocals = link.vocals
        if not bgm:
            link = self._stem_links.get(source_key(source))
            if link and link.instrumental:
                bgm = link.instrumental
        self.conv_vocals_path_var.set(vocals or "(not set — run Separate on Process)")
        self.conv_bgm_path_var.set(bgm or "(not set — run Separate on Process)")
        extra_dirs = [self.im_merge_output_dir.get()]
        exp = self.active_exp_path.get().strip()
        if exp:
            extra_dirs.append(str(Path(exp) / "infer_long"))
        results = find_convert_results_for_source(
            source,
            self._stem_links,
            self._audio_scan_rows,
            extra_dirs,
        )
        for path_s in results:
            rp = Path(path_s)
            try:
                size = format_size(rp.stat().st_size)
            except OSError:
                size = "?"
            self.conv_results_tree.insert(
                "",
                tk.END,
                values=(rp.name, size, str(rp.resolve())),
            )
        self.conv_results_status.set(f"source={p.name}  results={len(results)}")

    def _browse_im_model(self) -> None:
        root = self._configured_root()
        fallback = str(root / "assets" / "weights") if root else None
        path = filedialog.askopenfilename(
            title="Select model .pth",
            filetypes=PTH_FILETYPES,
            **self._path_dialog_opts(
                self.im_model_path.get(), for_file=True, fallback=fallback
            ),
        )
        if not path:
            return
        self.im_model_path.set(path)
        self.im_model_name.set(Path(path).name)
        self._autofill_im_index()
        self._refresh_im_auto_paths()
        self._update_fav_model_btn_label()

    def _browse_im_index(self) -> None:
        root = self._configured_root()
        fallback = str(root / "assets" / "indices") if root else None
        path = filedialog.askopenfilename(
            title="Select index file",
            filetypes=[("FAISS index", "*.index"), ("All Files", "*.*")],
            **self._path_dialog_opts(
                self.im_index_path.get(), for_file=True, fallback=fallback
            ),
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

    def _browse_im_merge_out_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select merge output folder",
            **self._path_dialog_opts(self.im_merge_output_dir.get()),
        )
        if path:
            self.im_merge_output_dir.set(path)
            self._refresh_im_merge_output_path()

    def _refresh_im_auto_paths(self) -> None:
        self._refresh_im_merge_output_path()

    def _refresh_im_merge_output_path(self) -> None:
        out_dir = self.im_merge_output_dir.get().strip()
        merged_name = self._merged_output_basename()
        if not merged_name or not out_dir:
            self.im_merge_output_path.set("")
            return
        self.im_merge_output_path.set(str(Path(out_dir) / merged_name))

    # ------------------------------------------------------------------
    # Logging / process control
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reset_job_progress(self) -> None:
        self._job_pct = None
        self._job_total_epoch = None
        self._job_audio_duration = None

    def _set_window_progress(self, pct: float | None) -> None:
        if pct is None:
            self._job_pct = None
            self.root.title(BASE_TITLE)
            return
        pct = max(0.0, min(100.0, float(pct)))
        # Avoid flickering the taskbar title on tiny updates.
        if self._job_pct is not None and abs(pct - self._job_pct) < 0.4:
            return
        self._job_pct = pct
        shown = int(round(pct))
        # Put percent first so Windows taskbar truncation keeps it visible.
        self.root.title(f"[{shown}%] {BASE_TITLE}")
        if self.running:
            self.status_var.set(f"Running… {shown}%")

    def _parse_progress_from_line(self, line: str) -> None:
        m = _RE_DURATION_SEC.search(line)
        if m:
            try:
                self._job_audio_duration = float(m.group(1))
            except ValueError:
                pass

        m = _RE_PROGRESS_FRAC.search(line)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                try:
                    cur, total = int(groups[0]), int(groups[1])
                    if total > 0:
                        self._set_window_progress(100.0 * cur / total)
                        return
                except ValueError:
                    pass

        m = _RE_TQDM_PCT.search(line)
        if m:
            try:
                self._set_window_progress(float(m.group(1)))
                return
            except ValueError:
                pass

        m = _RE_TRAIN_EPOCH_PCT.search(line)
        if m:
            try:
                epoch = int(m.group(1))
                batch_pct = float(m.group(2))
                total = self._job_total_epoch
                if total and total > 0:
                    overall = ((epoch - 1) + batch_pct / 100.0) / total * 100.0
                    self._set_window_progress(overall)
                else:
                    self._set_window_progress(batch_pct)
                return
            except ValueError:
                pass

        m = _RE_CHUNK_START.search(line)
        if m and self._job_audio_duration and self._job_audio_duration > 0:
            try:
                start = float(m.group(2))
                self._set_window_progress(100.0 * start / self._job_audio_duration)
            except ValueError:
                pass

    def _drain_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(msg)
            if self.running:
                for line in msg.replace("\r", "\n").splitlines():
                    if line.strip():
                        self._parse_progress_from_line(line)
        self.root.after(80, self._drain_log_queue)

    def _set_running(self, running: bool) -> None:
        self.running = running
        if hasattr(self, "stop_btn"):
            self.stop_btn.config(state="normal" if running else "disabled")
        st = "disabled" if running else "normal"
        if hasattr(self, "im_convert_btn"):
            self.im_convert_btn.config(state=st)
        if hasattr(self, "sep_start_btn"):
            self.sep_start_btn.config(state=st)
        if hasattr(self, "as_scan_btn"):
            self.as_scan_btn.config(state=st)
        if hasattr(self, "proc_sep_btn"):
            if running:
                self.proc_sep_btn.config(state="disabled")
                if hasattr(self, "proc_convert_btn"):
                    self.proc_convert_btn.config(state="disabled")
            else:
                self._refresh_process_tab()
        if running:
            self._set_window_progress(0.0)
            self.status_var.set("Running… 0%")
        else:
            self._reset_job_progress()
            self.root.title(BASE_TITLE)
            self.status_var.set("Idle")

    def _on_tab_changed(self, _event: tk.Event | None = None) -> None:
        try:
            if self.notebook.index(self.notebook.select()) == TAB_PROCESS:
                self._refresh_process_tab()
            elif self.notebook.index(self.notebook.select()) == TAB_CONVERT:
                self._refresh_convert_results_list()
        except tk.TclError:
            pass
        self._save_settings()

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
        self.root.title(f"[stopping] {BASE_TITLE}")

    def _run_cmd(
        self,
        cmd: list[str],
        rvc_root: Path,
        on_success: Callable[[], None] | None = None,
        env_extra: dict[str, str] | None = None,
        total_epoch: int | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        *,
        chain: bool = False,
        release_running: bool = True,
    ) -> None:
        if self.running and not chain:
            messagebox.showwarning("Busy", "Wait for the current job to finish, or Stop it.")
            return

        if env is None:
            env = prepare_rvc_process_env(rvc_root)
            env["PYTHONUNBUFFERED"] = "1"
            if env_extra:
                env.update(env_extra)
        else:
            env = dict(env)
            env.setdefault("PYTHONUNBUFFERED", "1")
            if env_extra:
                env.update(env_extra)

        work_cwd = cwd if cwd is not None else rvc_root

        def worker() -> None:
            self.log_queue.put("\n" + "=" * 72 + "\n")
            self.log_queue.put(
                "COMMAND:\n"
                + " ".join(f'"{c}"' if " " in c else c for c in cmd)
                + "\n"
            )
            self.log_queue.put(f"cwd: {work_cwd}\n")
            self.log_queue.put("=" * 72 + "\n")
            success = False
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(work_cwd),
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
                success = code == 0
                if success and on_success is not None:
                    self.root.after(0, on_success)
            except Exception as exc:
                self.log_queue.put(f"\n[error] {exc}\n")
            finally:
                self.proc = None
                if release_running or not success:
                    self.root.after(0, lambda: self._set_running(False))

        self._reset_job_progress()
        if total_epoch is not None and total_epoch > 0:
            self._job_total_epoch = total_epoch
        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Job builders
    # ------------------------------------------------------------------

    def _build_infer_cmd(self, root: Path) -> tuple[list[str], str, dict[str, str]] | None:
        self._refresh_im_auto_paths()
        model_path = self.im_model_path.get().strip()
        input_path = self.im_input_path.get().strip()
        model_name = self.im_model_name.get().strip()
        index_path = self.im_index_path.get().strip()

        if not model_path or not Path(model_path).is_file():
            messagebox.showerror("Missing model", "Browse and select a valid .pth model.")
            return None
        if not input_path or not Path(input_path).is_file():
            messagebox.showerror(
                "Missing vocals",
                "Separated vocals not found.\n"
                "Open Source Audio → Process, run Separate, then Convert.",
            )
            return None
        if not model_name:
            model_name = Path(model_path).name
            self.im_model_name.set(model_name)
        if index_path and not Path(index_path).is_file():
            messagebox.showerror("Missing index", f"Index file not found:\n{index_path}")
            return None

        opt_path = str(self._allocate_infer_temp_path(model_path, input_path))
        self._convert_infer_temp = opt_path
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
            "--breath_mix_rate",
            self.im_breath_mix_rate.get().strip() or "0.65",
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
        env_extra = {"weight_root": str(Path(model_path).parent)}
        return cmd, opt_path, env_extra

    def _build_merge_cmd(
        self,
        bgm_path: str,
        *,
        merge_out: str | None = None,
    ) -> tuple[list[str], str, str] | None:
        infer_result = (self._convert_infer_temp or "").strip()
        out_path = (merge_out if merge_out is not None else self.im_merge_output_path.get()).strip()

        if not infer_result or not Path(infer_result).is_file():
            messagebox.showerror("Missing result", "Infer output not found for merge.")
            return None
        if not bgm_path or not Path(bgm_path).is_file():
            messagebox.showerror(
                "Missing instrumental",
                "Instrumental stem not found.\nRun Separate on the Process tab first.",
            )
            return None
        if not self.im_merge_output_dir.get().strip():
            messagebox.showerror("Missing output", "Select a valid merge_output_dir.")
            return None
        if not out_path:
            messagebox.showerror("Missing output", "merge_output_path is empty.")
            return None

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(out_path).suffix.lower()
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
            out_path,
        ]
        return cmd, infer_result, out_path

    def run_convert(self, *, chain: bool = False) -> None:
        root = self._require_rvc_root()
        if root is None:
            if chain:
                self._set_running(False)
            return
        built = self._build_infer_cmd(root)
        if built is None:
            if chain:
                self._set_running(False)
            return
        infer_cmd, opt_path, env_extra = built
        convert_source = self._resolve_convert_source()
        bgm_path = self._resolve_bgm_path()
        if not bgm_path:
            if chain:
                self._set_running(False)
            messagebox.showerror(
                "Missing instrumental",
                "Instrumental stem not found.\nRun Separate on the Process tab first.",
            )
            return
        if not self.im_merge_output_dir.get().strip():
            if chain:
                self._set_running(False)
            messagebox.showerror("Missing output", "Select a valid merge_output_dir.")
            return
        self._refresh_im_merge_output_path()
        locked_merge_out = self.im_merge_output_path.get().strip()
        if not locked_merge_out:
            if chain:
                self._set_running(False)
            messagebox.showerror("Missing output", "merge_output_path is empty.")
            return

        def after_infer() -> None:
            # Keep the merge path locked at convert-start. Refreshing from UI here
            # is wrong: selecting another Source Audio row during a long infer
            # would rewrite merge_output to a different clip (Permission denied /
            # wrong file).
            self.im_merge_output_path.set(locked_merge_out)
            merge_built = self._build_merge_cmd(bgm_path, merge_out=locked_merge_out)
            if merge_built is None:
                self._set_running(False)
                messagebox.showerror(
                    "Convert",
                    f"Infer finished but merge could not start.\nOutput:\n{opt_path}",
                )
                return
            merge_cmd, infer_result, merge_out = merge_built

            def after_merge() -> None:
                note = ""
                try:
                    infer_path = Path(infer_result)
                    merge_path = Path(merge_out)
                    if (
                        infer_path.is_file()
                        and merge_path.is_file()
                        and infer_path.resolve() != merge_path.resolve()
                    ):
                        infer_path.unlink()
                        note = "\n\nDeleted temporary infer file."
                        self.log_queue.put(f"[convert] deleted infer temp: {infer_result}\n")
                except OSError as exc:
                    note = f"\n\nCould not delete infer temp file:\n{exc}"
                    self.log_queue.put(f"[convert] delete infer temp failed: {exc}\n")
                self._convert_infer_temp = None
                self._record_convert_result(convert_source, merge_out)
                self._refresh_convert_results_list()
                self._refresh_source_list()
                messagebox.showinfo("Convert done", f"Output:\n{merge_out}{note}")

            self._run_cmd(merge_cmd, root, on_success=after_merge, chain=True)

        self._run_cmd(
            infer_cmd,
            root,
            on_success=after_infer,
            env_extra=env_extra,
            chain=chain,
            release_running=False,
        )


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
