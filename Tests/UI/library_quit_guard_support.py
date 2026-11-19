"""Shared fixtures for the Library Ctrl+Q quit-guard tests (TASK-34000.1).

Two test files use these. ``test_library_quit_guard.py`` is the lean
PR-gated core, enrolled in ``scripts/ui_pr_gate_census.txt``.
``test_library_quit_guard_extended.py`` holds the rest and runs with the full
``Tests/UI`` suite. Not a test module, so pytest does not collect it.

Every test drives the real ``TldwCli`` against a real ChaChaNotes database
(or a real file on disk). The irreversible shutdown is replaced by a recorder
that reads the database at the moment the approved quit would have exited.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual.widgets import Button, Static, TextArea

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: 160x45 is the review's wide repro size.
SIZE = (160, 45)

_SETTLE_SECONDS = 30.0

_TITLE = "Ideas inbox"

_BODY = "- Build a dashboard\n- Try spaced repetition"

_OTHER_TITLE = "Reading list"

_OTHER_BODY = "- Dune"

#: Scaled autosave pair for the burst tests (production: 2 s / 10 s). A key
#: lands every ``_KEY_GAP`` seconds, well inside the debounce.
_DEBOUNCE, _MAX_WAIT, _KEY_GAP = 1.0, 2.0, 0.25


async def _until(pilot, predicate, what: str, timeout: float = _SETTLE_SECONDS):
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


class _NotesProfile:
    """A real ChaChaNotes database behind the real Notes scope service."""

    def __init__(self, tmp_path: Path) -> None:
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="task-34000-1")
        self.note_id = self.db.add_note(_TITLE, _BODY)
        self.other_id = self.db.add_note(_OTHER_TITLE, _OTHER_BODY)
        assert self.note_id and self.other_id
        self.interop = NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="task-34000-1",
            global_db_to_use=self.db,
        )
        self.scope_service = NotesScopeService(
            local_notes_service=self.interop,
            server_service=None,
            folder_repository=LocalNoteFolderRepository(self.db),
        )

    def note(self, note_id: str) -> dict:
        row = self.db.get_note_by_id(note_id)
        assert row is not None, f"note {note_id} is gone"
        return row

    def rows(self) -> list[tuple[str, str]]:
        """Every live note as ``(title, content)``."""
        return [(row["title"], row["content"]) for row in self.db.list_notes(limit=500)]


def _settings_without_splash(section, key=None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _library_app(tmp_path, monkeypatch, events: list, read, *, notes: bool = True):
    """The production app, booted straight into Library, quit recorded.

    Args:
        read: Called with the notes profile (or None) when the approved quit
            would exit; its result is recorded as ``("quit", result)``.

    Returns:
        ``(app, profile)``; ``profile`` is None when ``notes`` is False.
    """
    app = _build_test_app(configured_default="library")
    _seed_conversations(app, [])
    profile = _NotesProfile(tmp_path) if notes else None
    if profile is not None:
        app.chachanotes_db = profile.db
        app.notes_scope_service = profile.scope_service
        app.notes_service = profile.interop

    async def _record_quit() -> None:
        events.append(("quit", read(profile)))

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_quit)
    return app, profile


def _no_autosave(monkeypatch) -> None:
    """Only the quit itself may persist anything in these tests.

    ``raising=False``: the max-wait constant does not exist before the fix,
    and the RED run must fail on behaviour, not on this setup line.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 3600)
    monkeypatch.setattr(
        library_screen_module,
        "LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS",
        3600,
        raising=False,
    )


def _record_toasts(monkeypatch, app) -> list[str]:
    """Record every toast the app shows, still showing it."""
    toasts: list[str] = []
    real_notify = app.notify

    def _notify(message, *args, **kwargs):
        toasts.append(str(message))
        return real_notify(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", _notify)
    return toasts


async def _library(app, pilot) -> LibraryScreen:
    await _until(
        pilot, lambda: isinstance(app.screen, LibraryScreen), "Library to mount"
    )
    screen = app.screen
    await _wait_for_library_shell(screen, pilot)
    return screen


async def _open_notes(screen, pilot) -> None:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")


async def _armed_editor(screen, pilot, note_id: str | None = None) -> None:
    def _ready() -> bool:
        snapshot = screen._library_note_session.snapshot
        return (
            snapshot is not None
            and (note_id is None or snapshot.note_id == note_id)
            and screen._notes_state.editor_armed
            and bool(screen.query("#library-note-body"))
        )

    await _until(pilot, _ready, "the note editor to open and arm")
    await pilot.pause(0.1)


async def _open_note(screen, pilot, title: str, note_id: str) -> TextArea:
    await _open_notes(screen, pilot)
    row = next(
        button
        for button in screen.query(".library-notes-tree-note-row").results(Button)
        if title in str(button.label)
    )
    row.press()
    await _armed_editor(screen, pilot, note_id)
    return screen.query_one("#library-note-body", TextArea)


async def _new_blank_note(screen, pilot) -> None:
    screen.query_one("#library-row-create-note", Button).press()
    blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
    blank.press()
    await _armed_editor(screen, pilot)


async def _type(pilot, text: str) -> None:
    await pilot.press(*("space" if char == " " else char for char in text))


async def _type_at_end(pilot, body: TextArea, text: str) -> None:
    body.focus()
    body.move_cursor(body.document.end)
    await pilot.pause()
    await _type(pilot, text)


async def _ctrl_q(pilot, app, events) -> ConfirmationDialog | None:
    """Press Ctrl+Q; return the quit prompt, or None when the app quit."""
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: isinstance(app.screen, ConfirmationDialog) or bool(events),
        "Ctrl+Q to quit or ask",
        timeout=15.0,
    )
    if isinstance(app.screen, ConfirmationDialog):
        prompt = app.screen
        await _until(pilot, lambda: prompt.focused is not None, "the prompt's focus")
        return prompt
    return None


def _prompt_title(prompt: ConfirmationDialog) -> str:
    return str(prompt.query_one(".dialog-title", Static).renderable)


async def _keep_editing(pilot, app, screen, prompt) -> None:
    """Answer the prompt with its focused default -- Keep editing."""
    focused = prompt.focused
    assert isinstance(focused, Button) and focused.id == "cancel-button", (
        f"Keep editing must be the focused default, not {focused!r}"
    )
    assert str(focused.label) == "Keep editing"
    await pilot.press("enter")
    await _until(pilot, lambda: app.screen is screen, "Keep editing to return")
    await _until(
        pilot, lambda: app._quit_in_progress is False, "the quit guard to clear"
    )
    assert app.is_running and app._shutting_down is False


def _scaled_autosave(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", _DEBOUNCE
    )
    monkeypatch.setattr(
        library_screen_module,
        "LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS",
        _MAX_WAIT,
        raising=False,
    )


async def _type_steadily(pilot, keys: str, sample) -> tuple[list[float], list]:
    """Press ``keys`` one per ``_KEY_GAP``; return the real gaps and samples."""
    gaps: list[float] = []
    samples: list = []
    last = time.monotonic()
    for char in keys:
        await pilot.press(char)
        await pilot.pause(_KEY_GAP)
        now = time.monotonic()
        gaps.append(now - last)
        last = now
        samples.append(sample())
    return gaps, samples
