"""Windows audio playback launch, with no command shell (TASK-39040).

On dev before TASK-39040, `SimpleAudioPlayer.play()` fell back to
``cmd /c start "" /wait <path>`` on every Windows host without the legacy
Windows Media Player. `subprocess.list2cmdline` quotes an argument only when
it contains whitespace, so a space-free path such as
``C:\\briefings\\a&calc&.mp3`` reached cmd.exe with a bare ``&``: command
injection from a database row. That fallback now hands the file to
`os.startfile` (ShellExecute), which takes the path as one document name and
never parses it as a command line.

The cost, accepted by the owner on 2026-11-05: ShellExecute returns no
process handle, so the app cannot pause, stop or watch a clip it handed off.
`SimpleAudioPlayer` records such a clip as handed off and never reports a
pause or stop it did not perform, and `hand_off` tells the running app so
(``HAND_OFF_NOTICE``).

`play()` imports this module inside its Windows branch only, so it is never
resident on macOS or Linux.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from .audio_player import _ALL_FILE_FORMATS

#: The legacy player `play()` drives with a list argv when it is installed.
WMPLAYER_PATH = Path("C:/Program Files/Windows Media Player/wmplayer.exe")

#: The toast for a handed-off clip. Literal text: it is shown with
#: ``markup=False`` and never carries the path.
HAND_OFF_NOTICE = "Playing in your default audio app. Pause or stop it there."


def wmplayer_argv(file_path: Path) -> list[str] | None:
    """Return the wmplayer.exe argv for `file_path`, or None when it is absent.

    Args:
        file_path: The clip to play.

    Returns:
        A list argv (no shell parses it), or None when the legacy player is
        not installed and the caller should use `hand_off`.
    """
    if not WMPLAYER_PATH.exists():
        return None
    return [str(WMPLAYER_PATH), "/play", "/close", str(file_path)]


def hand_off(file_path: Path) -> bool:
    """Open `file_path` in the user's default audio app via `os.startfile`.

    ShellExecute runs whatever program an extension is registered to run, so
    only the audio extensions `SimpleAudioPlayer` knows are handed off: a row
    naming ``payload.bat`` is refused here rather than executed.

    Args:
        file_path: The clip to open, passed to `os.startfile` unchanged.

    Returns:
        True when the default app was asked to open the clip.
    """
    if file_path.suffix.lstrip(".").lower() not in _ALL_FILE_FORMATS:
        logger.warning("Windows audio hand-off refused: not an audio file type")
        return False
    try:
        os.startfile(os.fspath(file_path))  # type: ignore[attr-defined]  # nosec B606
    except OSError as exc:
        logger.warning("Windows audio hand-off failed ({})", type(exc).__name__)
        return False
    _tell_running_app(HAND_OFF_NOTICE)
    return True


def _tell_running_app(message: str) -> None:
    """Show `message` as a literal toast in the Textual app that asked to play.

    Every caller reaches `play()` inside that app's context: a handler on the
    event loop, `asyncio.to_thread` (which copies the context), or
    `AsyncAudioPlayer.play` (which runs the call in a copy of it). `App.notify`
    is safe to call from any thread. With no active app (a script or a test),
    the notice goes to the log instead.

    Args:
        message: Literal toast text; never interpolated into markup.
    """
    try:
        from textual._context import active_app

        app = active_app.get()
    except (ImportError, LookupError):
        logger.info("Windows audio handed off to the default app; no app to notify")
        return
    app.notify(message, markup=False)
