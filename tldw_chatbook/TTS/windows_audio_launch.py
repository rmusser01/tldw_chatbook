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
import threading
from pathlib import Path, PureWindowsPath

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


#: The suffixes ShellExecute may open, compared lower-case as Windows does.
_AUDIO_SUFFIXES = frozenset(f".{audio_format}" for audio_format in _ALL_FILE_FORMATS)


def hand_off_refusal(given: PureWindowsPath, resolved: PureWindowsPath) -> str | None:
    """Why `given`, which resolves to `resolved`, must not reach ShellExecute.

    ShellExecute runs whatever program a name's extension is registered to
    run, so both names must end in an audio suffix: a link ``clip.mp3`` whose
    target is ``payload.exe`` is refused (TASK-39040 review). A ``:`` after
    the drive names an alternate data stream (``x.bat:evil.mp3`` is a stream
    of a batch file), and a UNC or device root (``\\\\host\\share``) would open
    a remote file and send this user's credentials to that host.

    Args:
        given: The path as the caller passed it, parsed with Windows rules.
        resolved: The same path with every link followed.

    Returns:
        Which rule refused the path (a fixed phrase, never the path), or None
        when the hand-off may go ahead.
    """
    for path in (given, resolved):
        text = str(path)
        if text.startswith("\\\\"):
            return "a network or device path"
        if ":" in text[len(path.drive) :]:
            return "an alternate data stream"
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            return "not an audio file type"
    return None


def hand_off(file_path: Path) -> bool:
    """Open `file_path` in the user's default audio app via `os.startfile`.

    The path is resolved first (links followed) and both names are checked by
    `hand_off_refusal`; a refused or unresolvable path is never opened.

    Args:
        file_path: The clip whose validated resolved target is opened.

    Returns:
        True when the default app was asked to open the clip.
    """
    try:
        resolved = file_path.resolve(strict=True)
    except (OSError, RuntimeError):
        refusal: str | None = "the file cannot be resolved"
    else:
        refusal = hand_off_refusal(
            PureWindowsPath(file_path), PureWindowsPath(resolved)
        )
    if refusal is not None:
        # Literal text: the category is derived from the path, and no part of
        # a path is logged.
        logger.warning("Windows audio hand-off refused: not a local audio file")
        return False
    try:
        os.startfile(os.fspath(resolved))  # type: ignore[attr-defined]  # nosec B606
    except OSError as exc:
        logger.warning("Windows audio hand-off failed ({})", type(exc).__name__)
        return False
    _tell_running_app(HAND_OFF_NOTICE)
    return True


def _tell_running_app(message: str) -> None:
    """Show `message` as a literal toast in the Textual app that asked to play.

    Every caller reaches `play()` inside that app's context: a handler on the
    event loop, `asyncio.to_thread` (the Console; it copies the context), or
    `AsyncAudioPlayer.play` (the Speech playground; it runs the call in a copy
    of it). The app is found through Textual's `active_app` context variable.
    On the app's own thread this is a direct `App.notify`. From any other
    thread it goes through `App.call_from_thread`, which runs `notify` on the
    app's loop and waits for it: `notify` itself only posts a message, but
    the routing makes the thread crossing explicit (Qodo, fix round 2). The
    thread test is the one `call_from_thread` uses to refuse its own thread
    in Textual 8.2.8 (`textual/app.py`: `App._thread_id`, set by
    `_thread_init` and `run_async` from `threading.get_ident()`, and compared
    against it in `call_from_thread`); that attribute is private API, as
    `active_app` is. With no active app (a script or a test), or an app that
    is not running (`call_from_thread` raises `RuntimeError`), the notice
    goes to the log instead.

    Args:
        message: Literal toast text; never interpolated into markup.
    """
    try:
        from textual._context import active_app

        app = active_app.get()
    except (ImportError, LookupError):
        logger.info("Windows audio handed off to the default app; no app to notify")
        return
    if app._thread_id == threading.get_ident():
        app.notify(message, markup=False)
        return
    try:
        app.call_from_thread(app.notify, message, markup=False)
    except RuntimeError:
        logger.info("Windows audio handed off to the default app; app not running")
