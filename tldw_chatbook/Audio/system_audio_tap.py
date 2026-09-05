"""System-audio capture for meetings (spec §3.1, §3.6).

Delivers 20 ms PCM16 mono 16 kHz frames of what the computer is playing.
Native routes are a spawned subprocess writing PCM to stdout (the Swift
helper on macOS, ``parec``/``pw-record`` on Linux) or a WASAPI loopback
input device on Windows; the fallback on any OS is a user-chosen input
device (BlackHole, VB-Cable). Textual-free; sounddevice/pyaudio are only
touched through ``AudioRecordingService``.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

FRAME_BYTES = 640
SINK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MACOS_MIN = (14, 2)
HELPER_NAME = "tldw-audiotap"


@dataclass(frozen=True)
class TapMode:
    kind: str
    reason: str
    command: tuple[str, ...] | None = None
    device_name: str | None = None
    device_index: int | None = None


# ---- pure resolvers -------------------------------------------------------

def validate_sink_name(name: str) -> str:
    if not name or not SINK_NAME_RE.match(name):
        raise ValueError(f"unsafe PulseAudio sink name: {name!r}")
    return name


def parse_default_sink(output: str) -> str:
    first = (output or "").strip().splitlines()[:1]
    if not first:
        raise ValueError("pactl returned no default sink")
    return validate_sink_name(first[0].strip())


def linux_capture_command(tool: str, sink: str) -> tuple[str, ...]:
    sink = validate_sink_name(sink)
    if tool == "parec":
        return (
            "parec", f"--device={sink}.monitor", "--format=s16le", "--rate=16000",
            "--channels=1", "--latency-msec=20",
        )
    if tool == "pw-record":
        return (
            "pw-record", "--target", f"{sink}.monitor", "--rate", "16000",
            "--channels", "1", "--format", "s16", "-",
        )
    raise ValueError(f"unsupported capture tool: {tool}")


def resolve_wasapi_loopback(devices: list[dict], default_output_name: str) -> int | None:
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        if (
            name.endswith("[Loopback]")
            and name.startswith(default_output_name)
            and int(device.get("max_input_channels", 0) or 0) > 0
        ):
            return index
    return None


def macos_version_ok(mac_ver: str) -> bool:
    try:
        parts = tuple(int(p) for p in (mac_ver or "").split(".")[:2])
    except ValueError:
        return False
    return len(parts) >= 1 and (parts + (0,))[:2] >= MACOS_MIN


# ---- macOS helper lookup --------------------------------------------------

def helper_source_path() -> Path:
    return Path(__file__).with_name("audiotap") / "main.swift"


def bundled_helper_path(executable: str = sys.executable) -> Path | None:
    candidate = Path(executable).resolve().parent / HELPER_NAME
    return candidate if candidate.exists() else None


def dev_helper_path(data_dir: Path) -> Path:
    digest = hashlib.sha256(helper_source_path().read_bytes()).hexdigest()[:12]
    return Path(data_dir) / "bin" / f"{HELPER_NAME}-{digest}"


def ensure_helper(
    data_dir: Path,
    *,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    executable: str = sys.executable,
) -> Path | None:
    """Return a runnable helper binary, compiling with swiftc if needed."""
    bundled = bundled_helper_path(executable)
    if bundled is not None:
        return bundled
    if not helper_source_path().exists():
        logger.warning(
            "audiotap helper source missing: {}",
            redact_user_paths(str(helper_source_path())),
        )
        return None
    target = dev_helper_path(data_dir)
    if target.exists():
        return target
    swiftc = which("swiftc")
    if swiftc is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [swiftc, "-O", "-o", str(target), str(helper_source_path()),
         "-framework", "CoreAudio", "-framework", "AVFoundation"],
        capture_output=True, text=True,
    )
    if getattr(result, "returncode", 1) != 0 or not target.exists():
        logger.warning(
            "audiotap helper compile failed: {}",
            redact_user_paths(str(getattr(result, "stderr", ""))),
        )
        return None
    return target


# ---- probe ----------------------------------------------------------------

def probe(
    *,
    system_source: str = "auto",
    platform: str = sys.platform,
    mac_ver: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    data_dir: Path | None = None,
    query_devices: Callable[[], list] | None = None,
    default_output_name: str | None = None,
) -> TapMode:
    source = (system_source or "auto").strip()
    if source and source.lower() != "auto":
        return TapMode("virtual_device", f"Virtual device: {source}", device_name=source)
    if platform == "darwin":
        if mac_ver is None:
            import platform as _platform

            mac_ver = _platform.mac_ver()[0]
        if not macos_version_ok(mac_ver):
            return TapMode("unavailable", "Native system audio needs macOS 14.2 or newer; pick a virtual device such as BlackHole")
        if data_dir is None:
            from tldw_chatbook.config import get_user_data_dir

            data_dir = get_user_data_dir()
        helper = ensure_helper(data_dir, run=run, which=which)
        if helper is None:
            return TapMode("unavailable", "System audio helper unavailable (no bundled binary and no swiftc); pick a virtual device")
        return TapMode("native_macos", "Native (macOS tap)", command=(str(helper),))
    if platform.startswith("linux"):
        tool = "parec" if which("parec") else ("pw-record" if which("pw-record") else None)
        if tool is None:
            return TapMode("unavailable", "Neither parec nor pw-record found; install pulseaudio-utils or pipewire-pulse")
        try:
            result = run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5)
            sink = parse_default_sink(getattr(result, "stdout", ""))
        except Exception as exc:  # noqa: BLE001 - reason goes to the rail
            return TapMode("unavailable", f"Could not resolve the default sink: {exc}")
        return TapMode("native_parec", f"Native ({tool})", command=linux_capture_command(tool, sink))
    if platform == "win32":
        if query_devices is None or default_output_name is None:
            try:
                import sounddevice as sd

                query_devices = query_devices or (lambda: [dict(d) for d in sd.query_devices()])
                default_output_name = default_output_name or str(sd.query_devices(kind="output")["name"])
            except Exception as exc:  # noqa: BLE001
                return TapMode("unavailable", f"sounddevice unavailable: {exc}")
        index = resolve_wasapi_loopback(list(query_devices()), default_output_name)
        if index is None:
            return TapMode("unavailable", "No WASAPI [Loopback] device for the default output; pick a virtual device")
        return TapMode("native_wasapi", "Native (WASAPI loopback)", device_index=index)
    return TapMode("unavailable", f"No native system-audio capture on {platform}")


# ---- taps -----------------------------------------------------------------

class SubprocessTap:
    """Reads fixed-size PCM frames from a helper process's stdout."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        frame_bytes: int = FRAME_BYTES,
        restart_delay_s: float = 2.0,
        spawn: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._command = tuple(command)
        self._frame_bytes = frame_bytes
        self._restart_delay_s = restart_delay_s
        self._spawn = spawn
        self._sleep = sleep
        self._proc: Any | None = None
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stopping = False
        self._on_frames: Optional[Callable[[bytes], None]] = None
        self._stderr_lines: deque = deque(maxlen=5)
        self.state = "stopped"
        self.restarts = 0
        self.exit_code: int | None = None

    @property
    def last_stderr(self) -> str:
        return "\n".join(self._stderr_lines)

    def _launch(self) -> bool:
        try:
            self._proc = self._spawn(
                list(self._command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("system audio helper failed to start: {}", exc)
            self._stderr_lines.append(str(exc))
            self.state = "lost"
            return False
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True, name="audiotap-stderr")
        self._stderr_thread.start()
        return True

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            self._stderr_lines.append(raw.decode("utf-8", "replace").rstrip())

    def start(self, on_frames: Callable[[bytes], None]) -> bool:
        self._on_frames = on_frames
        self._stopping = False
        if not self._launch():
            return False
        self.state = "running"
        self._thread = threading.Thread(target=self._reader, daemon=True, name="audiotap-reader")
        self._thread.start()
        return True

    def _reader(self) -> None:
        while True:
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
            while stdout is not None:
                data = stdout.read(self._frame_bytes)
                if not data:
                    break
                if len(data) == self._frame_bytes and self._on_frames is not None:
                    self._on_frames(data)
            if proc is not None:
                self.exit_code = proc.wait()
            if self._stopping:
                self.state = "stopped"
                return
            if self.restarts == 0:
                self.restarts = 1
                logger.warning("system audio helper exited ({}); restarting once", self.exit_code)
                self._sleep(self._restart_delay_s)
                if self._stopping:
                    self.state = "stopped"
                    return
                if self._launch():
                    continue
            if self._stopping:
                self.state = "stopped"
                return
            self.state = "lost"
            logger.error("system audio source lost (exit {}): {}", self.exit_code, self.last_stderr)
            return

    def stop(self) -> None:
        self._stopping = True
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=1.0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("audiotap stop: {}", exc)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.state != "lost":
            self.state = "stopped"


class DeviceTap:
    """System audio through an ordinary input device (loopback or virtual cable)."""

    def __init__(self, device_name: str | None, *, device_index: int | None = None, recorder_factory=None) -> None:
        self._device_name = device_name
        self._device_index = device_index
        self._factory = recorder_factory
        self._recorder: Any | None = None
        self.state = "stopped"

    def start(self, on_frames: Callable[[bytes], None]) -> bool:
        factory = self._factory
        if factory is None:
            from .recording_service import AudioRecordingService

            factory = AudioRecordingService
        try:
            self._recorder = factory(use_vad=False, retain_audio=False, chunk_size=320)
            device_id = self._device_index
            if device_id is None and self._device_name:
                for device in self._recorder.get_audio_devices():
                    if str(device.get("name", "")) == self._device_name:
                        device_id = device.get("id", device.get("index"))
                        break
            if device_id is not None:
                self._recorder.set_device(device_id)
            ok = bool(self._recorder.start_recording(callback=on_frames))
        except Exception as exc:  # noqa: BLE001
            logger.error("device tap failed: {}", exc)
            ok = False
        self.state = "running" if ok else "lost"
        return ok

    def stop(self) -> None:
        if self._recorder is not None:
            try:
                self._recorder.stop_recording()
            except Exception as exc:  # noqa: BLE001
                logger.debug("device tap stop: {}", exc)
        self.state = "stopped"


def build_tap(mode: TapMode, *, recorder_factory=None):
    if mode.kind in ("native_macos", "native_parec") and mode.command:
        return SubprocessTap(mode.command)
    if mode.kind == "native_wasapi":
        return DeviceTap(None, device_index=mode.device_index, recorder_factory=recorder_factory)
    if mode.kind == "virtual_device":
        return DeviceTap(mode.device_name, recorder_factory=recorder_factory)
    return None
