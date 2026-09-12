# audio_player.py
# Description: Simple audio playback for TTS - focusing on cross-platform compatibility
#
# Imports
import asyncio
import subprocess
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger

#######################################################################################################################
#
# Simple Audio Player for TUI

#: File formats a player binary can definitely decode. macOS `afplay` and
#: the Windows COM path play most of what the OS audio stack supports --
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
    Windows COM path plays everything the OS supports, so it maps to the
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
    and can decode `audio_format`; Windows resolves to the built-in COM
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
        self._lock = threading.Lock()  # Thread safety for state management
        self._supports_pause = False  # Whether the player supports native pause
        self._find_player()

    def _find_player(self) -> None:
        """Find available audio player on the system"""
        if self._system == "Darwin":  # macOS
            self._player_cmd = ["/usr/bin/afplay"]
            self._player_name = "afplay"
            self._supports_pause = False  # afplay doesn't support pause
        elif self._system == "Linux":
            # Initial (format-agnostic) selection from the shared
            # catalogue; `play()` re-selects per artifact format before
            # spawning.
            self._select_linux_player(lambda _formats: True)
            if self._player_name is None:
                logger.warning("No suitable audio player found on Linux")
        elif self._system == "Windows":
            # Windows Media Player supports pause through COM automation
            self._player_cmd = [
                "windows"
            ]  # Placeholder, actual command built in play()
            self._player_name = "windows"
            self._supports_pause = True  # We'll use COM automation for pause
        else:
            self._player_cmd = None
            self._player_name = None
            self._supports_pause = False
            logger.warning(f"Unsupported platform: {self._system}")

    def _select_linux_player(self, formats_accepted) -> None:
        """Point this player at the first catalogue entry matching the filter.

        `formats_accepted` receives a candidate's decodable-format set and
        returns whether it may serve the upcoming artifact; entries are
        visited in the catalogue's preference order and PATH availability
        is always required. Used both by `_find_player`'s format-agnostic
        initial probe and by `play()`'s per-artifact format-aware
        selection.
        """
        for name, cmd, supports_pause, formats in _LINUX_PLAYER_CATALOGUE:
            if not formats_accepted(formats):
                continue
            if not shutil.which(name):
                continue
            self._player_cmd = list(cmd)
            self._player_name = name
            self._supports_pause = supports_pause
            return
        self._player_cmd = None
        self._player_name = None
        self._supports_pause = False

    def _select_player_for_artifact(self, file_path: Path) -> None:
        """Re-select the player for `file_path`'s audio format.

        The artifact's extension decides the format; an unrecognized
        extension keeps whatever player the initial probe selected (the
        pre-catalogue behavior) rather than refusing a file this module
        simply doesn't know the extension of. On macOS this owns the
        Ogg/Opus routing dev observed: `afplay` can exit successfully
        without decoding an Ogg/Opus body, so those containers go to
        ffplay when installed and refuse cleanly when not.
        """
        audio_format = file_path.suffix.lstrip(".").lower() or None
        if audio_format not in _ALL_FILE_FORMATS:
            return
        if self._system == "Darwin":
            if audio_format in _AFPLAY_FORMATS:
                # Reset to afplay even if a previous opus/ogg artifact
                # left ffplay (or a clean refusal) selected -- same
                # re-probe dev's inline special case performed.
                self._player_cmd = ["/usr/bin/afplay"]
                self._player_name = "afplay"
                self._supports_pause = False
                return
            ffplay = shutil.which("ffplay")
            if ffplay is None:
                logger.warning(
                    "Opus playback requires ffplay; choose WAV output instead"
                )
                self._player_cmd = None
                self._player_name = None
                self._supports_pause = False
                return
            self._player_cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "error"]
            self._player_name = "ffplay"
            self._supports_pause = False
            return
        if self._system != "Linux":
            return
        self._select_linux_player(
            lambda formats: audio_format in formats,
        )

    def play(self, file_path: Path) -> bool:
        """
        Play an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if playback started successfully
        """
        logger.debug(f"SimpleAudioPlayer.play called with: {file_path}")
        logger.debug(
            f"Current state before stop: {self._current.state}, process: {self._current.process}"
        )

        # Stop any current playback
        self.stop()

        # Log state after stop
        logger.debug(
            f"State after stop: {self._current.state}, process: {self._current.process}"
        )

        # Validate file
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Audio file not found: {file_path}")
            return False

        # Format-aware player selection: the artifact's extension decides
        # which player may serve it -- an .mp3 must not be handed to a
        # WAV-only aplay, a box with no format-capable player must refuse
        # here rather than spawn garbage, and (dev-observed) Ogg/Opus on
        # macOS must route to ffplay because afplay exits successfully
        # without decoding those containers.
        self._select_player_for_artifact(file_path)

        # Check if we have a player
        if not self._player_cmd:
            logger.error("No audio player available")
            return False

        # Ensure clean state - force cleanup any zombie process
        if self._current.process:
            try:
                # Check if process is zombie/defunct
                if self._current.process.poll() is not None:
                    logger.debug(
                        f"Found zombie process with return code: {self._current.process.poll()}"
                    )
                    self._current.process = None
            except Exception as e:
                logger.debug(f"Error checking process state: {e}")
                self._current.process = None

        # Add a small delay on macOS to ensure previous afplay is fully terminated
        if self._system == "Darwin" and self._player_name == "afplay":
            time.sleep(0.1)

        try:
            if self._system == "Windows":
                # Use Windows built-in wmplayer or start command instead of PowerShell
                # This avoids command injection risks
                wmplayer_path = Path(
                    "C:/Program Files/Windows Media Player/wmplayer.exe"
                )
                if wmplayer_path.exists():
                    # Use Windows Media Player directly
                    cmd = [str(wmplayer_path), "/play", "/close", str(file_path)]
                else:
                    # Use start command to open with default application
                    # The empty string after 'start' is for the window title
                    cmd = ["cmd", "/c", "start", "", "/wait", str(file_path)]
            else:
                # Unix-like systems
                cmd = self._player_cmd + [str(file_path)]

            # Start playback in background
            with self._lock:
                # Special handling for players with pause support
                if self._player_name == "mpv":
                    # Create a unique socket for this instance
                    import tempfile

                    socket_path = Path(tempfile.gettempdir()) / f"mpv-socket-{id(self)}"
                    cmd = self._player_cmd[:-1] + [
                        f"--input-ipc-server={socket_path}",
                        str(file_path),
                    ]
                    self._mpv_socket = socket_path
                    self._current.process = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                elif self._player_name == "mplayer":
                    # For mplayer slave mode, we need pipes
                    self._current.process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    self._current.process = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                self._current.file_path = file_path
                self._current.state = PlaybackState.PLAYING
                self._current.start_time = time.time()
                self._current.pause_time = None
                self._current.total_pause_duration = 0.0
                self._current.position = 0.0

            # Start monitoring thread
            monitor = threading.Thread(
                target=self._monitor_playback,
                args=(self._current.process,),
                daemon=True,
            )
            monitor.start()

            logger.info(
                f"Started playback of {file_path.name} using {self._player_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self._current.state = PlaybackState.ERROR
            return False

    def pause(self) -> bool:
        """
        Pause current playback.

        Returns:
            True if paused successfully
        """
        with self._lock:
            logger.debug(
                f"Pause called - current state: {self._current.state}, supports pause: {self._supports_pause}"
            )
            if self._current.state != PlaybackState.PLAYING:
                return False

            if not self._supports_pause:
                # For players without pause, we'll stop and track position
                self._current.position = self.get_position()
                self.stop()
                self._current.state = PlaybackState.PAUSED
                self._current.pause_time = time.time()
                logger.info(
                    f"Paused at position {self._current.position:.1f}s (using stop)"
                )
                return True

            # Native pause support
            try:
                if self._player_name == "mpv" and hasattr(self, "_mpv_socket"):
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

    def resume(self) -> bool:
        """
        Resume paused playback.

        Returns:
            True if resumed successfully
        """
        with self._lock:
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
                self._current.total_pause_duration += (
                    time.time() - self._current.pause_time
                )
                return self.play(self._current.file_path)

            # Native resume support
            try:
                if self._player_name == "mpv" and hasattr(self, "_mpv_socket"):
                    # Send resume command via socket
                    import socket
                    import json

                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(str(self._mpv_socket))
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

    def stop(self) -> bool:
        """
        Stop current playback.

        Returns:
            True if stopped successfully
        """
        with self._lock:
            logger.debug(
                f"Stop called - current state: {self._current.state}, process exists: {self._current.process is not None}"
            )
            if self._current.process or self._current.state in [
                PlaybackState.PLAYING,
                PlaybackState.PAUSED,
                PlaybackState.FINISHED,
            ]:
                # Only try to terminate if process exists
                if self._current.process:
                    try:
                        # Check if process is still running
                        if self._current.process.poll() is None:
                            # For macOS afplay, use kill directly as terminate doesn't work well
                            if (
                                self._system == "Darwin"
                                and self._player_name == "afplay"
                            ):
                                self._current.process.kill()
                                self._current.process.wait(timeout=1)
                            else:
                                self._current.process.terminate()
                                self._current.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.debug("Process didn't terminate in time, force killing")
                        self._current.process.kill()
                        try:
                            self._current.process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            logger.warning("Process kill timed out")
                            # For macOS, try using os.killpg if process is stuck
                            if self._system == "Darwin":
                                try:
                                    import os
                                    import signal

                                    os.killpg(
                                        os.getpgid(self._current.process.pid),
                                        signal.SIGKILL,
                                    )
                                except Exception as e:
                                    logger.debug(f"Failed to kill process group: {e}")
                    except Exception as e:
                        logger.error(f"Error stopping playback: {e}")
                    finally:
                        # Always nullify the process handle
                        self._current.process = None

                # Clean up mpv socket if exists
                if self._player_name == "mpv" and hasattr(self, "_mpv_socket"):
                    try:
                        Path(self._mpv_socket).unlink(missing_ok=True)
                    except Exception:
                        pass

                # Always reset all state when cleaning up
                self._current.process = None
                self._current.state = PlaybackState.IDLE
                self._current.file_path = None  # Clear the file path too
                self._current.start_time = None
                self._current.pause_time = None
                self._current.total_pause_duration = 0.0
                self._current.position = 0.0
                self._current.duration = None
                logger.info("Playback stopped/cleaned up - all state reset")
                return True
            else:
                logger.debug(
                    f"Cannot stop - process exists: {self._current.process is not None}, state: {self._current.state}"
                )

        return False

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        with self._lock:
            return self._current.state == PlaybackState.PLAYING

    def get_state(self) -> PlaybackState:
        """Get current playback state"""
        with self._lock:
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
        # Clean up any remaining resources
        if hasattr(self, "_mpv_socket") and self._mpv_socket:
            try:
                Path(self._mpv_socket).unlink(missing_ok=True)
            except Exception:
                pass
        logger.debug("Audio player cleaned up")


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

    def __init__(self):
        self._player = get_audio_player()
        self._executor = None

    def _get_executor(self):
        """Get or create thread pool executor"""
        if self._executor is None:
            import concurrent.futures

            # Create a simple thread pool without custom initializer
            # The threads will be daemon by default in Python 3.7+
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="AudioPlayer"
            )
            logger.debug("Created AudioPlayer thread pool executor")

        return self._executor

    async def play(self, file_path: Path) -> bool:
        """Play audio file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._get_executor(), self._player.play, file_path
        )

    async def pause(self) -> bool:
        """Pause playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.pause)

    async def resume(self) -> bool:
        """Resume playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.resume)

    async def stop(self) -> bool:
        """Stop playback asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), self._player.stop)

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
        """Clean up resources asynchronously"""
        try:
            # Stop any active playback
            await self.stop()

            # Clean up the underlying player
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._get_executor(), self._player.cleanup)

            # Force cleanup any remaining threads
            if self._executor:
                try:
                    # First try graceful shutdown with short timeout
                    self._executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    # If that fails, force shutdown
                    self._executor.shutdown(wait=False)
                finally:
                    self._executor = None

            logger.debug("AsyncAudioPlayer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during AsyncAudioPlayer cleanup: {e}")
            # Force cleanup even on error
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None


#
# End of audio_player.py
#######################################################################################################################
