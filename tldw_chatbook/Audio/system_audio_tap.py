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
    """The system-audio route `probe` resolved, and why.

    Attributes:
        kind: ``native_macos``, ``native_parec``, ``native_wasapi``,
            ``virtual_device`` or ``unavailable``.
        reason: User-facing copy for the rail -- the route's name, or why
            there isn't one.
        command: Helper argv for a subprocess route.
        device_name: Input device name for a virtual-cable route.
        device_index: Device index for the WASAPI loopback route.
    """

    kind: str
    reason: str
    command: tuple[str, ...] | None = None
    device_name: str | None = None
    device_index: int | None = None


# ---- pure resolvers -------------------------------------------------------

def validate_sink_name(name: str) -> str:
    """Check a PulseAudio sink name before it reaches a command line.

    Args:
        name: The sink name, as reported by `pactl`.

    Returns:
        The same name, unchanged.

    Raises:
        ValueError: Empty, or containing anything outside
            ``[A-Za-z0-9._-]`` -- the tap spawns no shell, but a sink name
            is still device-supplied text that ends up in argv.
    """
    if not name or not SINK_NAME_RE.match(name):
        raise ValueError(f"unsafe PulseAudio sink name: {name!r}")
    return name


def parse_default_sink(output: str) -> str:
    """Read the default sink name out of `pactl get-default-sink` output.

    Args:
        output: The command's stdout.

    Returns:
        The validated sink name.

    Raises:
        ValueError: No output, or the name fails `validate_sink_name`.
    """
    first = (output or "").strip().splitlines()[:1]
    if not first:
        raise ValueError("pactl returned no default sink")
    return validate_sink_name(first[0].strip())


def linux_capture_command(tool: str, sink: str) -> tuple[str, ...]:
    """Build the argv that captures a sink's monitor as 16 kHz mono PCM16.

    Args:
        tool: ``"parec"`` or ``"pw-record"``.
        sink: The sink whose ``.monitor`` source is captured.

    Returns:
        The command, ready for `SubprocessTap`.

    Raises:
        ValueError: Unknown tool, or an unsafe sink name.
    """
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
    """Find the WASAPI loopback input that mirrors the default output.

    Args:
        devices: `sounddevice.query_devices()` rows, in index order.
        default_output_name: Name of the current default output device.

    Returns:
        The device index, or None when Windows exposes no matching
        ``[Loopback]`` input (the user then picks a virtual cable).
    """
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
    """Whether this macOS is new enough for Core Audio process taps.

    Args:
        mac_ver: `platform.mac_ver()[0]`, e.g. ``"14.6.1"``.

    Returns:
        True from macOS 14.2 on; False for anything older or unparseable.
    """
    try:
        parts = tuple(int(p) for p in (mac_ver or "").split(".")[:2])
    except ValueError:
        return False
    return len(parts) >= 1 and (parts + (0,))[:2] >= MACOS_MIN


# ---- macOS helper lookup --------------------------------------------------

def helper_source_path() -> Path:
    """Return the path to the Swift tap helper's source, shipped in-package."""
    return Path(__file__).with_name("audiotap") / "main.swift"


def bundled_helper_path(executable: str = sys.executable) -> Path | None:
    """Look for a helper binary shipped next to the interpreter.

    Args:
        executable: The running interpreter's path; the packaged app puts
            the helper in the same `Contents/MacOS` directory.

    Returns:
        The helper path, or None when this is not a packaged install.
    """
    candidate = Path(executable).resolve().parent / HELPER_NAME
    return candidate if candidate.exists() else None


def dev_helper_path(data_dir: Path) -> Path:
    """Return where a locally compiled helper lives for the current source.

    The filename carries a digest of the Swift source, so editing the helper
    yields a new path instead of silently reusing a stale binary.

    Args:
        data_dir: The user data dir; the binary goes in its `bin/`.

    Returns:
        The (possibly not-yet-existing) helper path.

    Raises:
        OSError: The helper source cannot be read.
    """
    digest = hashlib.sha256(helper_source_path().read_bytes()).hexdigest()[:12]
    return Path(data_dir) / "bin" / f"{HELPER_NAME}-{digest}"


def ensure_helper(
    data_dir: Path,
    *,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    executable: str = sys.executable,
) -> Path | None:
    """Return a runnable helper binary, compiling with swiftc if needed.

    Args:
        data_dir: Where a locally compiled helper is cached.
        run: Process runner, injectable for tests.
        which: Executable lookup, injectable for tests.
        executable: The running interpreter, used to find a bundled helper.

    Returns:
        A path to a runnable helper, or None when there is none and one
        cannot be built (no source, no swiftc, compile failed) -- the caller
        then reports the native route as unavailable.
    """
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
    """Resolve how (or whether) this machine can capture system audio.

    Pure decision-making: nothing is started here, and no device is opened
    beyond enumeration. An explicit `system_source` short-circuits to that
    virtual device; otherwise the native route for the platform is probed.

    Args:
        system_source: ``"auto"``, or the name of a virtual input device.
        platform: `sys.platform`, injectable for tests.
        mac_ver: macOS version string; probed when None.
        which: Executable lookup, injectable for tests.
        run: Process runner, injectable for tests.
        data_dir: User data dir for the macOS helper cache.
        query_devices: Windows device enumeration, injectable for tests.
        default_output_name: Windows default output name, injectable.

    Returns:
        The resolved `TapMode`; `kind == "unavailable"` carries the reason
        the rail shows, and never raises for a missing tool or backend.
    """
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
    """Reads fixed-size PCM frames from a helper process's stdout.

    Used for the macOS Swift helper and for `parec`/`pw-record` on Linux. A
    helper that exits on its own is restarted exactly once; a second exit
    leaves `state == "lost"` and the meeting continues mic-only (spec §3.6).
    """

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        frame_bytes: int = FRAME_BYTES,
        restart_delay_s: float = 2.0,
        spawn: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build the tap.

        Args:
            command: Helper argv; it must write raw PCM to stdout.
            frame_bytes: Bytes read per frame (20 ms at 16 kHz mono PCM16).
            restart_delay_s: Wait before the single restart attempt.
            spawn: Process spawner, injectable for tests.
            sleep: Sleep function, injectable for tests.
        """
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
        """The helper's last few stderr lines, for the failure message."""
        return "\n".join(self._stderr_lines)

    def _launch(self) -> bool:
        try:
            self._proc = self._spawn(
                list(self._command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            # A spawn failure names the helper's absolute path (which sits
            # under the user's data dir), and `last_stderr` is logged again
            # by `_reader`, so redact once at the point of capture.
            detail = redact_user_paths(str(exc))
            logger.error("system audio helper failed to start: {}", detail)
            self._stderr_lines.append(detail)
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
        """Spawn the helper and stream its frames.

        Args:
            on_frames: Called on the reader thread with one complete frame
                at a time; short reads are discarded.

        Returns:
            True when the helper started. False (state ``"lost"``) means the
            meeting degrades to mic-only; nothing is raised.
        """
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
        """Close the helper's stdin, wait, then terminate if it lingers."""
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
    """System audio through an ordinary input device (loopback or virtual cable).

    Covers the Windows WASAPI loopback route (by index) and any user-chosen
    virtual cable such as BlackHole or VB-Cable (by name).
    """

    def __init__(self, device_name: str | None, *, device_index: int | None = None, recorder_factory=None) -> None:
        """Build the tap.

        Args:
            device_name: The input device to capture, resolved by name.
            device_index: A device index, when already known (WASAPI).
            recorder_factory: Recorder builder, injectable for tests;
                defaults to `AudioRecordingService`.
        """
        self._device_name = device_name
        self._device_index = device_index
        self._factory = recorder_factory
        self._recorder: Any | None = None
        self.state = "stopped"

    def start(self, on_frames: Callable[[bytes], None]) -> bool:
        """Open the configured input device and stream its frames.

        Args:
            on_frames: Called with each captured chunk.

        Returns:
            True when recording started. A configured device that is no
            longer present returns False rather than falling back to the
            default input, which would record the room a second time.
        """
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
                else:
                    # The user named a loopback device (BlackHole, VB-Cable)
                    # that is not plugged in. Falling through to the default
                    # INPUT would silently record the room through the mic
                    # a second time and label it "others" -- worse than no
                    # system audio (final whole-branch review).
                    # The device NAME stays out of the log: audio devices are
                    # routinely named after their owner ("<Name>'s AirPods"),
                    # this is a persistent sink, and nothing here logs device
                    # names today. The user picked the device; the message is
                    # actionable without repeating it back at them.
                    logger.warning(
                        "the configured system audio device was not found; "
                        "not falling back to the default input"
                    )
                    self.state = "lost"
                    return False
            if device_id is not None:
                self._recorder.set_device(device_id)
            ok = bool(self._recorder.start_recording(callback=on_frames))
        except Exception as exc:  # noqa: BLE001
            logger.error("device tap failed: {}", exc)
            ok = False
        self.state = "running" if ok else "lost"
        return ok

    def stop(self) -> None:
        """Stop the underlying recorder; tolerates one that never started."""
        if self._recorder is not None:
            try:
                self._recorder.stop_recording()
            except Exception as exc:  # noqa: BLE001
                logger.debug("device tap stop: {}", exc)
        self.state = "stopped"


def build_tap(mode: TapMode, *, recorder_factory=None):
    """Build the tap object for a probed `TapMode`.

    Args:
        mode: The route `probe` resolved.
        recorder_factory: Recorder builder for device-backed routes,
            injectable for tests.

    Returns:
        A `SubprocessTap`, a `DeviceTap`, or None when the mode carries no
        usable route -- None is what puts the meeting in room mode.
    """
    if mode.kind in ("native_macos", "native_parec") and mode.command:
        return SubprocessTap(mode.command)
    if mode.kind == "native_wasapi":
        return DeviceTap(None, device_index=mode.device_index, recorder_factory=recorder_factory)
    if mode.kind == "virtual_device":
        return DeviceTap(mode.device_name, recorder_factory=recorder_factory)
    return None
