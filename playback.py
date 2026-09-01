"""Local audio playback (play / pause / stop) via pygame.mixer."""
from __future__ import annotations

from pathlib import Path

SYSTEM_DEFAULT_DEVICE = "(System default)"


class PlaybackError(RuntimeError):
    pass


def normalize_output_device(name: str | None) -> str | None:
    """Map combobox label to pygame devicename (None = OS default)."""
    clean = (name or "").strip()
    if not clean or clean == SYSTEM_DEFAULT_DEVICE:
        return None
    if ":" in clean:
        prefix, rest = clean.split(":", 1)
        if prefix.strip().isdigit():
            rest = rest.strip()
            if rest.endswith("★"):
                rest = rest[:-1].strip()
            return rest or clean
    return clean


def device_label_matches(saved: str, label: str) -> bool:
    if saved == label:
        return True
    saved_n = normalize_output_device(saved) or saved.strip().lower()
    label_n = normalize_output_device(label) or label.strip().lower()
    if saved_n and label_n and saved_n.lower() == label_n.lower():
        return True
    saved_low = saved.strip().lower()
    label_low = label.strip().lower()
    return bool(saved_low) and (saved_low in label_low or label_low in saved_low)


def pick_default_device_label(devices: list[str]) -> str:
    if SYSTEM_DEFAULT_DEVICE in devices:
        return SYSTEM_DEFAULT_DEVICE
    return devices[0] if devices else ""


def format_time_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total_s = ms // 1000
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def audio_duration_ms(path: Path | str) -> int:
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            if f.samplerate <= 0:
                return 0
            return int(len(f) / f.samplerate * 1000)
    except Exception:
        return 0


def list_output_devices() -> list[str]:
    """Playback device labels for the GUI combobox (always includes system default)."""
    labels: list[str] = [SYSTEM_DEFAULT_DEVICE]
    seen: set[str] = set()

    try:
        import sounddevice as sd

        default_out = sd.default.device[1]
        if isinstance(default_out, int) and default_out >= 0:
            try:
                d = sd.query_devices(default_out)
                name = str(d.get("name", "")).strip()
                if name:
                    key = name.lower()
                    seen.add(key)
                    labels.append(f"{default_out}: {name}")
            except Exception:
                pass

        for i, d in enumerate(sd.query_devices()):
            if int(d.get("max_output_channels", 0) or 0) <= 0:
                continue
            name = str(d.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(f"{i}: {name}")
    except Exception:
        pass

    try:
        import pygame

        if not pygame.get_init():
            pygame.init()
        import pygame._sdl2.audio as sdl2_audio

        for raw in sdl2_audio.get_audio_device_names(False):
            name = str(raw).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(name)
    except Exception:
        pass

    return labels


class AudioPlayer:
    """Thin wrapper around pygame.mixer.music for GUI control."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._paused = False
        self._mixer_ready = False
        self.duration_ms = 0
        self._volume = 1.0
        self._device_name: str | None = None
        self._play_start_offset_ms = 0

    def _reinit_mixer(self) -> None:
        import pygame

        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self._mixer_ready = False
        if self._device_name:
            try:
                pygame.mixer.init(devicename=self._device_name)
            except Exception:
                self._device_name = None
                pygame.mixer.init()
        else:
            pygame.mixer.init()
        self._mixer_ready = True
        pygame.mixer.music.set_volume(self._volume)

    def _ensure_mixer(self) -> None:
        if self._mixer_ready:
            return
        try:
            import pygame  # noqa: F401
        except ImportError as e:
            raise PlaybackError(
                "pygame is required for playback. Install with: pip install pygame"
            ) from e
        self._reinit_mixer()

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def device_name(self) -> str | None:
        return self._device_name

    def is_busy(self) -> bool:
        if not self._mixer_ready or self._path is None:
            return False
        import pygame

        return bool(pygame.mixer.music.get_busy()) or self._paused

    def state(self) -> str:
        if self._path is None:
            return "stopped"
        if self._paused:
            return "paused"
        if self.is_busy():
            return "playing"
        return "stopped"

    def set_volume(self, level: float) -> None:
        self._volume = max(0.0, min(1.0, float(level)))
        if self._mixer_ready:
            import pygame

            pygame.mixer.music.set_volume(self._volume)

    def set_output_device(self, name: str | None) -> None:
        clean = normalize_output_device(name)
        if clean == self._device_name:
            return
        self.stop()
        self._device_name = clean
        if self._mixer_ready:
            self._reinit_mixer()

    def play(self, path: Path | str) -> Path:
        p = Path(path)
        if not p.is_file():
            raise PlaybackError(f"file not found: {p}")
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                self._ensure_mixer()
                import pygame

                pygame.mixer.music.load(str(p))
                pygame.mixer.music.set_volume(self._volume)
                pygame.mixer.music.play()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt == 0 and self._device_name:
                    self._device_name = None
                    self._reinit_mixer()
                    continue
                raise PlaybackError(f"cannot play {p.name}: {e}") from e
        if last_exc is not None:
            raise PlaybackError(f"cannot play {p.name}: {last_exc}") from last_exc
        self._path = p
        self._paused = False
        self._play_start_offset_ms = 0
        self.duration_ms = audio_duration_ms(p)
        return p

    def pause(self) -> None:
        if not self._mixer_ready or self._path is None:
            return
        import pygame

        if pygame.mixer.music.get_busy() and not self._paused:
            pygame.mixer.music.pause()
            self._paused = True

    def resume(self) -> None:
        if not self._mixer_ready or self._path is None:
            return
        import pygame

        if self._paused:
            pygame.mixer.music.unpause()
            self._paused = False

    def toggle_pause(self) -> str:
        if self.state() == "paused":
            self.resume()
        elif self.state() == "playing":
            self.pause()
        return self.state()

    def stop(self) -> None:
        if not self._mixer_ready:
            self._path = None
            self._paused = False
            self.duration_ms = 0
            return
        import pygame

        pygame.mixer.music.stop()
        self._path = None
        self._paused = False
        self.duration_ms = 0
        self._play_start_offset_ms = 0

    def position_ms(self) -> int:
        if not self._mixer_ready or self._path is None:
            return 0
        import pygame

        raw = int(pygame.mixer.music.get_pos())
        if raw < 0:
            raw = 0
        return self._play_start_offset_ms + raw

    def seek(self, ms: int) -> None:
        if self._path is None:
            return
        self._ensure_mixer()
        import pygame

        dur = self.duration_ms
        if dur > 0:
            ms = max(0, min(int(ms), dur))
        else:
            ms = max(0, int(ms))
        keep_playing = self.state() == "playing"
        pygame.mixer.music.load(str(self._path))
        pygame.mixer.music.play(start=ms / 1000.0)
        pygame.mixer.music.set_volume(self._volume)
        self._play_start_offset_ms = ms
        if keep_playing:
            self._paused = False
        else:
            pygame.mixer.music.pause()
            self._paused = True
