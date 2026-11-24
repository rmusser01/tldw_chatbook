# audio_player.py
# Description: Simple audio playback for TTS - focusing on cross-platform compatibility
#
# Imports
import asyncio
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

#######################################################################################################################
#
# Simple Audio Player for TUI

#: File formats a player binary can definitely decode. macOS `afplay` and
#: Windows delegated playback cover what the OS audio stack supports --
#: but NOT Ogg/Opus (observed on dev: `afplay` can exit successfully
#: without decoding an Ogg/Opus body, which is worse than a clean
#: failure, so opus/ogg are excluded from its set and route to ffplay).
#: The Linux catalogue below is deliberately conservative where a
#: "maybe" exists: `paplay` and `pw-play` decode through libsndfile,
#: whose MP3 support only landed in 1.1.0 (2022) and remains a distro
#: build flag -- relying on it would hand an MP3 to a player that emits
#: nothing on older builds, which is exactly the silent-failure class
#: this catalogue exists to prevent.
_ALL_FILE_FORMATS: frozenset[str] = frozenset(
    {"mp3", "opus", "ogg", "aac", "flac", "wav"}
)
_AFPLAY_FORMATS: frozenset[str] = frozenset({"mp3", "aac", "flac", "wav"})
_AFPLAY_EXECUTABLE = "/usr/bin/afplay"

#: Linux player catalogue, in preference order: (name, base command,
#: supports native pause, definitely-decodable formats).
_LINUX_PLAYER_CATALOGUE: tuple[tuple[str, list[str], bool, frozenset[str]], ...] = (
    (
        "mpv",
        [
            "mpv",
            "--no-video",
            "--really-quiet",
            "--input-ipc-server=/tmp/mpv-socket",
        ],
        True,
        _ALL_FILE_FORMATS,
    ),
    ("mplayer", ["mplayer", "-really-quiet", "-slave"], True, _ALL_FILE_FORMATS),
    (
        "ffplay",
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"],
        False,
        _ALL_FILE_FORMATS,
    ),
    ("pw-play", ["pw-play"], False, frozenset({"wav", "flac"})),
    ("paplay", ["paplay"], False, frozenset({"wav", "flac"})),
    ("aplay", ["aplay", "-q"], False, frozenset({"wav"})),
)


def player_supported_formats(player_name: str | None) -> frozenset[str]:
    """Return the file formats `player_name` can definitely decode.

    macOS `afplay` maps to its OS-decodable subset (no Ogg/Opus); the
    Windows delegated path accepts the formats supported by the OS, including the
    full format set. Unknown names (including ``None``) map to no
    formats -- callers treat that as "cannot serve this format".

    Args:
        player_name: A player name from `_LINUX_PLAYER_CATALOGUE`, or one
            of the platform defaults ("afplay", "windows").

    Returns:
        The frozen set of file-format names the player can decode; empty
        for unknown players.
    """
    if player_name == "afplay":
        return _AFPLAY_FORMATS
    if player_name == "windows":
        return _ALL_FILE_FORMATS
    for _name, _cmd, _pause, formats in _LINUX_PLAYER_CATALOGUE:
        if _name == player_name:
            return formats
    return frozenset()


def find_player_for_format(audio_format: str | None) -> str | None:
    """Return the name of a locally available player for `audio_format`.

    Platform-aware and side-effect free (probes PATH via `shutil.which`
    only): macOS resolves afplay-served formats to `afplay` and Ogg/Opus
    to `ffplay` when installed (afplay cannot decode those containers);
    Linux resolves to the first catalogue player that both exists on PATH
    and can decode `audio_format`; Windows resolves to the delegated player
    path; unknown platforms resolve to None. An unknown/``None`` format
    falls back to the first available player regardless of formats,
    matching the pre-catalogue selection for artifacts with unrecognized
    extensions.

    Args:
        audio_format: A lowercase file-format name ("mp3", "wav", ...) or
            ``None`` for "no format knowledge -- any player will do".

    Returns:
        The selected player's name, or ``None`` when no installed player
        can decode `audio_format` on this platform.
    """
    system = platform.system()
    if system == "Darwin":
        if audio_format is not None and audio_format not in _AFPLAY_FORMATS:
            return "ffplay" if shutil.which("ffplay") else None
        return "afplay"
    if system == "Windows":
        return "windows"
    if system != "Linux":
        return None
    for name, _cmd, _pause, formats in _LINUX_PLAYER_CATALOGUE:
        if audio_format is not None and audio_format not in formats:
            continue
        if shutil.which(name):
            return name
    return None


class PlaybackState(Enum):
    """Audio playback states"""

    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"
    BLOCKED = "blocked"  # native retirement remains unconfirmed
    #: Windows accepted the clip. A launcher handle cannot establish when
    #: external sound stops, even if that launcher was our wmplayer child.
    HANDED_OFF = "handed_off"


#: Ruling (TASK-39040, fix round 2): how long a surface keeps a handed-off
#: clip's FILE after it has released the in-app slot. ShellExecute returns no
#: handle, so no surface can see the default app finish, and deleting on
#: Stop pulled the file from under the app still reading it. One bound for
#: the Console's clip cache and the Speech playground's WAV copy, equal to
#: the Console's owned-playback poll bound; a screen's own exit still
#: releases at once.
HAND_OFF_RETENTION_SECONDS = 300.0


@dataclass
class AudioPlayerInfo:
    """Information about current playback"""

    file_path: Optional[Path] = None
    state: PlaybackState = PlaybackState.IDLE
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None  # When playback started
    pause_time: Optional[float] = None  # When paused
    total_pause_duration: float = 0.0  # Total time spent paused
    duration: Optional[float] = None  # Total duration of the file
    position: float = 0.0  # Current playback position in seconds
    attempt_token: object | None = None
    start_authority: Callable[[], bool] | None = None
    plan: "FilePlaybackPlan | None" = None


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@dataclass(frozen=True)
class FilePlaybackPlan:
    """Worker-prepared artifact identity and exact native launch choice."""

    file_path: Path
    identity: tuple[int, int, int, int]
    name: str
    argv: tuple[str, ...]
    supports_pause: bool

    @property
    def external(self) -> bool:
        return self.name in {"startfile", "wmplayer"}

    def matches_file(self) -> bool:
        """Recheck the prepared artifact after admission or native retirement."""
        try:
            return self.identity == _file_identity(self.file_path)
        except (ValueError, OSError):
            return False

    def executable_available(self) -> bool:
        """Check the selected executable without searching for another player."""
        if self.name == "startfile":
            return True
        executable = self.argv[0]
        return os.path.isfile(executable) and os.access(executable, os.X_OK)


def _kill_player_process_group(process: "subprocess.Popen | None") -> None:
    """SIGKILL a wedged player's process group -- never the app's own.

    TASK-32892 P0-3. A `Popen` child started without `start_new_session`
    inherits OUR process group, so `killpg(getpgid(child.pid))` is suicide:
    it takes down the TUI (and the terminal's whole foreground group) with
    no shutdown, no `close_tts_resources()`, no DB quiesce. The owned player
    starts in its own session, and this refuses the signal when the group
    still resolves to ours -- the same belt-and-braces pairing
    `Event_Handlers/LLM_Management_Events/server_lifecycle.py`
    `_signal_server_process_group` uses (TASK-32806.5).

    Args:
        process: The wedged player handle, if any.
    """
    if os.name != "posix":
        return
    import signal

    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger.debug(f"Failed to resolve player process group: {exc}")
        return
    if pgid <= 0 or pgid == os.getpgrp():
        logger.warning(
            f"Refusing to SIGKILL process group {pgid}: it is the app's own"
        )
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger.debug(f"Failed to kill process group: {exc}")


class SimpleAudioPlayer:
    """
    Enhanced cross-platform audio player with pause/resume support.

    Uses native system commands with pause/resume capabilities where available.
    Falls back to stop/restart for players without native pause support.
    """

    def __init__(self):
        """Initialize audio player"""
        self._current: AudioPlayerInfo = AudioPlayerInfo()
        self._system = platform.system()
        # TASK-38581: this lock MUST be reentrant. pause() and resume()
        # call other lock-taking methods while still holding it:
        # pause() -> get_position() + stop() on players without native
        # pause (afplay, ffplay, paplay, pw-play, aplay), and resume() ->
        # play() -> stop(). With a plain Lock those paths self-deadlock
        # permanently -- one Pause click wedged the call forever, and via
        # AsyncAudioPlayer's single-worker executor it wedged every later
        # TTS audio call with it.
        self._lock = threading.RLock()
        self._launchers: set[subprocess.Popen] = set()
        self._player_name, self._player_cmd, self._supports_pause = self._player_for(
            None
        )

    @property
    def _mpv_socket(self) -> Path:
        return Path(tempfile.gettempdir()) / f"mpv-socket-{id(self)}"

    def _player_for(
        self, audio_format: str | None
    ) -> tuple[str | None, list[str], bool]:
        if self._system == "Windows":
            return "windows", ["windows"], False
        if self._system == "Darwin":
            if audio_format is None or audio_format in _AFPLAY_FORMATS:
                return "afplay", [_AFPLAY_EXECUTABLE], False
            ffplay = shutil.which("ffplay")
            if ffplay:
                return (
                    "ffplay",
                    [
                        os.path.abspath(ffplay),
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "error",
                    ],
                    False,
                )
        if self._system == "Linux":
            for name, cmd, supports_pause, formats in _LINUX_PLAYER_CATALOGUE:
                if audio_format is not None and audio_format not in formats:
                    continue
                executable = shutil.which(name)
                if executable:
                    return name, [os.path.abspath(executable), *cmd[1:]], supports_pause
        return None, [], False

    def prepare(self, file_path: Path) -> FilePlaybackPlan | None:
        """Check the artifact and choose its exact launch route without changing playback."""
        try:
            identity = _file_identity(file_path)
            if identity is None:
                return None
            audio_format = file_path.suffix.lstrip(".").lower()
            name, cmd, supports_pause = self._player_for(
                audio_format if audio_format in _ALL_FILE_FORMATS else None
            )
            if name is None:
                return None
            if name == "windows":
                from .windows_audio_launch import can_hand_off, wmplayer_argv

                argv = wmplayer_argv(file_path)
                if argv is None:
                    if not can_hand_off(file_path):
                        return None
                    name, cmd = "startfile", []
                else:
                    name, cmd = "wmplayer", argv
            else:
                if name == "mpv":
                    cmd[-1] = f"--input-ipc-server={self._mpv_socket}"
                cmd.append(str(file_path))
            plan = FilePlaybackPlan(
                file_path, identity, name, tuple(cmd), supports_pause
            )
            return plan if plan.executable_available() else None
        except (ValueError, OSError):
            return None

    def play(
        self,
        file_path: Path,
        *,
        attempt_token: object | None = None,
        start_authority: Callable[[], bool] | None = None,
    ) -> bool:
        """Prepare and start an ordinary file or delegate it to the OS."""
        plan = self.prepare(file_path)
        return plan is not None and self.play_prepared(
            plan, attempt_token=attempt_token, start_authority=start_authority
        )

    def play_prepared(
        self,
        plan: FilePlaybackPlan,
        *,
        attempt_token: object | None = None,
        start_authority: Callable[[], bool] | None = None,
    ) -> bool:
        """Commit a prepared route on a worker. Accepted external launches return True."""
        token = attempt_token if attempt_token is not None else object()
        with self._lock:
            if not plan.matches_file() or not plan.executable_available():
                return False
            if start_authority is not None and not start_authority():
                return False
            self.stop()
            if self._current.process is not None:  # unconfirmed retirement
                return False

            self._player_name = plan.name
            self._supports_pause = plan.supports_pause
            if plan.name == "afplay":
                time.sleep(0.1)
            if not plan.matches_file():
                return False

            current = AudioPlayerInfo(
                file_path=plan.file_path,
                attempt_token=token,
                start_authority=start_authority,
                plan=plan,
                start_time=time.time(),
            )
            try:
                if plan.name == "startfile":
                    from .windows_audio_launch import hand_off

                    def can_open() -> bool:
                        return plan.matches_file() and (
                            start_authority is None or start_authority()
                        )

                    if not hand_off(
                        plan.file_path, start_authority=can_open, notify=False
                    ):
                        return False
                else:
                    if start_authority is not None and not start_authority():
                        return False
                    process = subprocess.Popen(
                        plan.argv,
                        stdin=subprocess.PIPE if plan.name == "mplayer" else None,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=(os.name == "posix"),
                    )
                    if plan.external:
                        self._launchers.add(process)
                    else:
                        current.process = process

                self._current = current
                if plan.external:
                    # A launcher may forward to another app. The accepted OS open
                    # remains a handoff even if authority was revoked inside Popen.
                    current.state = PlaybackState.HANDED_OFF
                    if plan.name == "wmplayer":
                        threading.Thread(
                            target=self._reap_launcher, args=(process,), daemon=True
                        ).start()
                    if start_authority is not None and not start_authority():
                        logger.debug(
                            "Windows accepted a file handoff during revocation"
                        )
                else:
                    current.state = PlaybackState.PLAYING
                    if start_authority is not None and not start_authority():
                        self.stop(expected_token=token)
                        return False
                    threading.Thread(
                        target=self._monitor_playback, args=(process,), daemon=True
                    ).start()
                    return True
            except Exception as exc:
                logger.warning(
                    "Could not start or observe file player ({})", type(exc).__name__
                )
                if current.state is not PlaybackState.HANDED_OFF:
                    if current.process is not None and not self.stop(
                        expected_token=token
                    ):
                        return False  # retain BLOCKED and its unconfirmed child
                    self._current = AudioPlayerInfo(
                        state=PlaybackState.ERROR, attempt_token=token
                    )
                    return False

        # This may synchronously marshal a toast to Textual. Release the player
        # lock first so an app callback can safely observe the recorded outcome.
        from .windows_audio_launch import HAND_OFF_NOTICE, notify_hand_off

        notice = (
            "Playing in Windows Media Player. Pause or stop it there."
            if plan.name == "wmplayer"
            else HAND_OFF_NOTICE
        )
        notify_hand_off(notice)
        return True

    def _reap_launcher(self, launcher: subprocess.Popen) -> None:
        """Observe only our launcher, independently of the external playback epoch."""
        try:
            launcher.wait()
        except Exception as exc:
            logger.warning(
                "Could not observe audio launcher exit ({})", type(exc).__name__
            )
            return
        with self._lock:
            self._launchers.discard(launcher)

    def pause(self, *, expected_token: object | None = None) -> bool:
        """
        Pause current playback.

        Returns:
            True if paused successfully
        """
        with self._lock:
            if (
                expected_token is not None
                and self._current.attempt_token is not expected_token
            ):
                return False
            logger.debug(
                f"Pause called - current state: {self._current.state}, supports pause: {self._supports_pause}"
            )
            if self._current.state != PlaybackState.PLAYING:
                return False

            if not self._supports_pause:
                # For players without pause, we'll stop and track position
                current = self._current
                position = self.get_position()
                if not self.stop():
                    return False
                self._current = current
                current.process = None
                current.position = position
                self._current.state = PlaybackState.PAUSED
                self._current.pause_time = time.time()
                logger.info(
                    f"Paused at position {self._current.position:.1f}s (using stop)"
                )
                return True

            # Native pause support
            try:
                if self._player_name == "mpv":
                    # Send pause command via socket
                    import socket
                    import json

                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(str(self._mpv_socket))
                        s.send(
                            json.dumps(
                                {"command": ["set_property", "pause", True]}
                            ).encode()
                            + b"\n"
                        )
                elif (
                    self._player_name == "mplayer"
                    and self._current.process
                    and self._current.process.stdin
                ):
                    # Send pause command to mplayer
                    self._current.process.stdin.write(b"pause\n")
                    self._current.process.stdin.flush()
                else:
                    return False

                self._current.state = PlaybackState.PAUSED
                self._current.pause_time = time.time()
                logger.info("Playback paused")
                return True

            except Exception as e:
                logger.error(f"Failed to pause: {e}")
                return False

    def resume(self, *, expected_token: object | None = None) -> bool:
        """
        Resume paused playback.

        Returns:
            True if resumed successfully
        """
        with self._lock:
            current = self._current
            if (
                expected_token is not None
                and current.attempt_token is not expected_token
            ):
                return False
            if (
                self._current.state != PlaybackState.PAUSED
                or not self._current.file_path
            ):
                return False

            if not self._supports_pause:
                # For players without pause, restart from saved position
                # This is a limitation - we can't truly resume from exact position
                logger.info(
                    f"Resuming from start (pause not supported by {self._player_name})"
                )
                if current.plan is not None:
                    return self.play_prepared(
                        current.plan,
                        attempt_token=current.attempt_token,
                        start_authority=current.start_authority,
                    )
                return self.play(
                    current.file_path,
                    attempt_token=current.attempt_token,
                    start_authority=current.start_authority,
                )

            # Native resume support
            try:
                if (
                    current.start_authority is not None
                    and not current.start_authority()
                ):
                    return False
                if self._player_name == "mpv":
                    # Send resume command via socket
                    import socket
                    import json

                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(str(self._mpv_socket))
                        if (
                            current.start_authority is not None
                            and not current.start_authority()
                        ):
                            return False
                        s.send(
                            json.dumps(
                                {"command": ["set_property", "pause", False]}
                            ).encode()
                            + b"\n"
                        )
                elif (
                    self._player_name == "mplayer"
                    and self._current.process
                    and self._current.process.stdin
                ):
                    # Send pause command again to toggle
                    self._current.process.stdin.write(b"pause\n")
                    self._current.process.stdin.flush()
                else:
                    return False

                if (
                    current.start_authority is not None
                    and not current.start_authority()
                ):
                    self.stop(expected_token=current.attempt_token)
                    return False
                self._current.state = PlaybackState.PLAYING
                self._current.total_pause_duration += (
                    time.time() - self._current.pause_time
                )
                self._current.pause_time = None
                logger.info("Playback resumed")
                return True

            except Exception as e:
                logger.error(f"Failed to resume: {e}")
                return False

    def _reap_current_process(self, process: subprocess.Popen) -> bool:
        try:
            if process.poll() is not None:
                return True
        except Exception:
            pass  # still hold the handle and try a bounded wait
        afplay = self._player_name == "afplay"
        for signal_child, timeout in (
            (process.kill if afplay else process.terminate, 1.0 if afplay else 2.0),
            (process.kill, 0.5),
        ):
            try:
                signal_child()
            except Exception:
                pass  # an exited child can refuse a signal; the wait decides
            try:
                process.wait(timeout=timeout)
                return True
            except Exception:
                pass
        if self._system == "Darwin":
            _kill_player_process_group(process)
            try:
                process.wait(timeout=0.5)
                return True
            except Exception:
                pass
        return False

    def stop(self, *, expected_token: object | None = None) -> bool:
        """Stop the exact attempt if supplied. False never certifies physical silence."""
        with self._lock:
            for launcher in tuple(self._launchers):
                try:
                    if launcher.poll() is not None:
                        self._launchers.discard(launcher)
                except Exception:
                    pass  # preserve a launcher whose wait or observer failed
            current = self._current
            if (
                expected_token is not None
                and current.attempt_token is not expected_token
            ):
                return False
            if current.process is not None and not self._reap_current_process(
                current.process
            ):
                current.state = PlaybackState.BLOCKED
                logger.warning("File playback retirement has not been confirmed")
                return False
            if self._player_name == "mpv" and current.process is not None:
                try:
                    self._mpv_socket.unlink(missing_ok=True)
                except OSError:
                    pass
            self._current = AudioPlayerInfo()
            return current.state not in (PlaybackState.IDLE, PlaybackState.HANDED_OFF)

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        with self._lock:
            return self._current.state == PlaybackState.PLAYING

    def get_state(self, *, expected_token: object | None = None) -> PlaybackState:
        """Get current state, returning IDLE for a different attempt."""
        with self._lock:
            if (
                expected_token is not None
                and self._current.attempt_token is not expected_token
            ):
                return PlaybackState.IDLE
            return self._current.state

    def get_position(self) -> float:
        """
        Get current playback position in seconds.

        Returns:
            Current position in seconds
        """
        with self._lock:
            if self._current.state == PlaybackState.IDLE:
                return 0.0
            elif self._current.state == PlaybackState.PAUSED:
                return self._current.position
            elif (
                self._current.state == PlaybackState.PLAYING
                and self._current.start_time
            ):
                # Calculate current position based on elapsed time
                elapsed = (
                    time.time()
                    - self._current.start_time
                    - self._current.total_pause_duration
                )
                return elapsed
            else:
                return self._current.position

    def get_duration(self) -> Optional[float]:
        """Get total duration of current file"""
        with self._lock:
            return self._current.duration

    def get_current_file(self) -> Optional[Path]:
        """Get the file path of the currently loaded/playing clip, if any.

        Returns ``None`` once the player is idle (after `stop()` or before
        any `play()`). Lets a caller check whether a specific file is the
        one this single-slot global player currently owns before deciding
        to stop it -- e.g. so a stop request scoped to one message doesn't
        silence an unrelated message's still-playing clip.
        """
        with self._lock:
            return self._current.file_path

    def _monitor_playback(self, process: subprocess.Popen | None = None) -> None:
        """Monitor playback process"""
        process = process if process is not None else self._current.process
        if process is not None:
            try:
                exit_code = process.wait()
                with self._lock:
                    # Only mark as finished if we're still in playing state
                    # (could have been stopped/paused)
                    if (
                        self._current.process is process
                        and self._current.state == PlaybackState.PLAYING
                    ):
                        self._current.state = (
                            PlaybackState.FINISHED
                            if exit_code == 0
                            else PlaybackState.ERROR
                        )
                        logger.debug(
                            "Playback process finished (exit_code={})", exit_code
                        )
            except Exception as e:
                logger.debug(f"Monitor thread interrupted: {e}")

    def cleanup(self) -> None:
        """Clean up resources - call this before app exit."""
        self.stop()


# Global player instance
_audio_player_instance: Optional[SimpleAudioPlayer] = None


def get_audio_player() -> SimpleAudioPlayer:
    """Get the global audio player instance"""
    global _audio_player_instance
    if _audio_player_instance is None:
        _audio_player_instance = SimpleAudioPlayer()
    return _audio_player_instance


# Keep the original play_audio_file for backward compatibility
def play_audio_file(file_path: Path) -> bool:
    """
    Simple function to play an audio file.

    Args:
        file_path: Path to audio file

    Returns:
        True if playback started
    """
    player = get_audio_player()
    return player.play(file_path)


# Async wrapper for Textual
class AsyncAudioPlayer:
    """Async wrapper for use in Textual"""

    def __init__(self, player: SimpleAudioPlayer | None = None):
        self._player = player if player is not None else get_audio_player()
        self._executor = None
        self._cleanup_task: asyncio.Task[None] | None = None

    def _get_executor(self):
        """Return the transport executor while it accepts work.

        During cleanup, getters can use asyncio's default executor (None).
        They only read the underlying player's locked state.
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return None
        if self._executor is None:
            import concurrent.futures

            # Create a simple thread pool without custom initializer
            # The threads will be daemon by default in Python 3.7+
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="AudioPlayer"
            )
            logger.debug("Created AudioPlayer thread pool executor")

        return self._executor

    async def _transport(self, operation: Callable[..., bool], *args: object) -> bool:
        executor = self._get_executor()
        if executor is None:
            return False
        # Keep the caller's Textual context for the Windows handoff notice.
        return await asyncio.get_running_loop().run_in_executor(
            executor, copy_context().run, operation, *args
        )

    async def play(self, file_path: Path) -> bool:
        """Play audio file asynchronously"""
        return await self._transport(self._player.play, file_path)

    async def pause(self) -> bool:
        """Pause playback asynchronously"""
        return await self._transport(self._player.pause)

    async def resume(self) -> bool:
        """Resume playback asynchronously"""
        return await self._transport(self._player.resume)

    async def stop(self) -> bool:
        """Stop playback asynchronously"""
        return await self._transport(self._player.stop)

    async def is_playing(self) -> bool:
        """Check if playing asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.is_playing)

    async def get_state(self) -> PlaybackState:
        """Get state asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.get_state)

    async def get_position(self) -> float:
        """Get playback position asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_executor(), self._player.get_position
        )

    async def get_duration(self) -> Optional[float]:
        """Get duration asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_executor(), self._player.get_duration
        )

    async def cleanup(self) -> None:
        """Join the retained native cleanup even if another observer was cancelled."""
        if self._cleanup_task is None or self._cleanup_task.done():
            executor = self._get_executor()
            self._cleanup_task = asyncio.create_task(self._cleanup_executor(executor))
        await asyncio.shield(self._cleanup_task)

    async def _cleanup_executor(self, executor) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(
                executor, self._player.cleanup
            )
        except Exception as exc:
            logger.warning("Audio player cleanup failed ({})", type(exc).__name__)
        finally:
            try:
                await asyncio.to_thread(
                    executor.shutdown, wait=True, cancel_futures=True
                )
            finally:
                self._executor = None


#
# End of audio_player.py
#######################################################################################################################
