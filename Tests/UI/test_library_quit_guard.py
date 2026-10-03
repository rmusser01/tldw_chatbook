"""Ctrl+Q saves unsaved Library edits, or asks first (TASK-34000.1, review N-01).

Ctrl+Q in Library ▸ Notes used to throw away whatever was typed since the
last autosave: the quit walk asks only the active screen's ``confirm_quit`` /
``prepare_for_quit``, ``LibraryScreen`` had neither, and its ``on_unmount``
cancels the pending debounced save. A brand-new note was left behind as an
empty "Untitled" row, and continuous typing (one key every 1.2 s re-arms the
2 s debounce forever) was never saved at all.

These drive the real ``TldwCli`` and press the real keys: the quit walk, the
Library screen, its note session coordinator, the save port and a real
ChaChaNotes database on disk are all real. Only the irreversible shutdown is
replaced by a recorder, as the other quit-flow tests do -- and the recorder
reads the DATABASE at the moment the approved quit would have exited, so each
assertion is about what a relaunch would find, not about widget text.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Label, Static, TextArea

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: ``bootstrap_profile`` as in ``test_roleplay_quit_guard.py``: a full-app boot
#: under the per-test profile redirect fails locally at ``_build_test_app`` with
#: ``RecoveryRequired: raw_source_selection_changed`` (lessons-testing-evidence).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: 160x45 is the review's wide repro size.
SIZE = (160, 45)
_SETTLE_SECONDS = 30.0
_TITLE = "Ideas inbox"
_BODY = "- Build a dashboard\n- Try spaced repetition"
_OTHER_TITLE = "Reading list"
_OTHER_BODY = "- Dune"


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


# --- AC#1 / AC#5: a clean flush persists the typed text, with no prompt ------


async def test_ctrl_q_inside_the_autosave_window_persists_the_typed_tail(
    tmp_path, monkeypatch
):
    """Verify capture 03: a tail typed just before Ctrl+Q is in the DB at exit."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " verifytailn01")
            assert screen._library_note_session.snapshot.dirty

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is None, "a flushable edit must quit without asking"
            [(kind, row)] = events
            assert kind == "quit"
            assert row["content"] == _BODY + " verifytailn01", (
                "Ctrl+Q exited before the typed tail reached the database"
            )
            assert row["title"] == _TITLE
    profile.db.close_connection()


async def test_ctrl_q_on_a_new_note_keeps_its_typed_title_and_body(
    tmp_path, monkeypatch
):
    """Capture 10 / j6: the new note persists as typed, not as an empty Untitled."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "riley new ctrlq")
            await _type_at_end(
                pilot, screen.query_one("#library-note-body", TextArea), "body typed"
            )

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, rows)] = events
            assert kind == "quit"
            assert ("riley new ctrlq", "body typed") in rows, rows
            assert len(rows) == 3, f"expected the two seeds plus the new note: {rows}"
            assert not [row for row in rows if row[0] == "Untitled"], rows
    profile.db.close_connection()


# --- AC#3: an untouched new note leaves no empty row ------------------------


async def test_ctrl_q_on_an_untouched_new_note_leaves_no_untitled_row(
    tmp_path, monkeypatch
):
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            assert len(profile.rows()) == 3, "the blank note's row exists while open"

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, rows)] = events
            assert kind == "quit"
            assert sorted(rows) == sorted([(_TITLE, _BODY), (_OTHER_TITLE, _OTHER_BODY)]), (
                f"the untouched new note survived the quit: {rows}"
            )
    profile.db.close_connection()


async def test_ctrl_q_on_a_new_note_with_only_a_blank_title_discards_it(
    tmp_path, monkeypatch
):
    """Review finding #7: quit treats a whitespace-only title like navigation.

    Leaving such a note discards it ("Empty note discarded"). Quitting must
    not first try to save the spaces, hit the whitespace veto and ask.
    """
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "  ")
            assert screen._library_note_session.snapshot.dirty

            assert await _ctrl_q(pilot, app, events) is None, (
                "a whitespace-only new note asked instead of being discarded"
            )
            [(kind, rows)] = events
            assert kind == "quit"
            assert sorted(rows) == sorted(
                [(_TITLE, _BODY), (_OTHER_TITLE, _OTHER_BODY)]
            ), rows
    profile.db.close_connection()


# --- AC#2: a save that cannot complete asks, and only Discard quits ---------


async def test_a_vetoed_save_asks_and_keep_editing_keeps_the_text(
    tmp_path, monkeypatch
):
    """Validation veto: the prompt names the note; Keep editing is the default."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " kept")
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            title.cursor_position = len(title.value)
            await _type(pilot, " ")  # a trailing space vetoes the save

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a vetoed save quit silently"
            assert events == []
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert events == []
            assert body.text == _BODY + " kept"
            assert title.value == _TITLE + " "
            assert screen._library_note_session.snapshot.dirty
            assert profile.note(profile.note_id)["content"] == _BODY

            prompt = await _ctrl_q(pilot, app, events)
            assert prompt is not None, "the second Ctrl+Q must ask again"
            await pilot.click("#confirm-button")
            await _until(pilot, lambda: bool(events), "Discard and quit to quit")
            [(kind, row)] = events
            assert kind == "quit"
            assert (row["title"], row["content"]) == (_TITLE, _BODY), (
                "Discard and quit must not save the discarded draft"
            )
    profile.db.close_connection()


async def test_a_failed_write_asks_instead_of_exiting(tmp_path, monkeypatch):
    """Write failure: the note's text is never silently dropped."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )

    def _disk_full(**_kwargs):
        raise OSError("disk full")

    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " unsaved")
            monkeypatch.setattr(profile.scope_service, "save_note", _disk_full)

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a failed save quit silently"
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert body.text == _BODY + " unsaved"
            assert profile.note(profile.note_id)["content"] == _BODY
    profile.db.close_connection()


async def test_a_conflicting_save_asks_and_keeps_the_other_version(
    tmp_path, monkeypatch
):
    """AC#2 conflict variant: the note changed elsewhere since it opened."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " mine")
            version = profile.note(profile.note_id)["version"]
            assert profile.db.update_note(
                profile.note_id, {"content": "edited elsewhere"}, version
            )

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a conflicting save quit silently"
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            message = str(prompt.query_one(".dialog-message", Label).renderable)
            assert "changed elsewhere" in message, message
            await _keep_editing(pilot, app, screen, prompt)
            assert body.text == _BODY + " mine"
            assert screen._library_note_session.snapshot.in_conflict

            prompt = await _ctrl_q(pilot, app, events)
            assert prompt is not None
            await pilot.click("#confirm-button")
            await _until(pilot, lambda: bool(events), "Discard and quit to quit")
            [(kind, row)] = events
            assert kind == "quit"
            assert row["content"] == "edited elsewhere", (
                "the other version was overwritten"
            )
    profile.db.close_connection()


# --- AC#4: continuous typing is saved before the typist pauses --------------


#: Scaled autosave pair for the burst tests (production: 2 s / 10 s). A key
#: lands every ``_KEY_GAP`` seconds, well inside the debounce.
_DEBOUNCE, _MAX_WAIT, _KEY_GAP = 1.0, 2.0, 0.25


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


async def test_continuous_typing_is_saved_within_the_max_wait(tmp_path, monkeypatch):
    """One key every 0.25 s never lets a 1 s debounce fire; the max wait does.

    The production pair is 2 s / 10 s; scaled down here so the burst fits a
    test. Every keystroke lands well inside the debounce, so without a
    maximum wait nothing reaches the database until typing stops.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _notes: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            body.focus()
            body.move_cursor(body.document.end)
            await pilot.pause()

            gaps, persisted = await _type_steadily(
                pilot,
                "abcdefghijklmnop",  # 16 keys over ~4 s
                lambda: profile.note(profile.note_id)["content"],
            )

            first_save = next(
                (index for index, content in enumerate(persisted) if content != _BODY),
                None,
            )
            assert first_save is not None, (
                "four seconds of continuous typing never reached the database"
            )
            assert persisted[first_save].startswith(_BODY + "a"), persisted[first_save]
            # Only the max wait can have saved it: no gap before that save
            # was long enough for the debounce to fire on its own.
            assert max(gaps[: first_save + 1]) < _DEBOUNCE, (
                f"a {max(gaps[: first_save + 1]):.2f}s pause let the debounce "
                "fire; this run cannot tell the max wait from the debounce"
            )
    profile.db.close_connection()


async def test_a_vetoed_autosave_never_moves_focus_while_typing(
    tmp_path, monkeypatch
):
    """N-07 core guard (TASK-34000.1 review): the max wait fires mid-burst.

    A title the save refuses (a trailing space) vetoes every autosave. If that
    veto moved focus to the title, the next body keys would land in the
    title and the next autosave would save body text as the title.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _notes: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            title.cursor_position = len(title.value)
            await _type(pilot, " ")  # the save refuses a trailing space
            body.focus()
            body.move_cursor(body.document.end)
            await pilot.pause()

            keys = "abcdefghijklmnop"
            _gaps, states = await _type_steadily(
                pilot,
                keys,
                lambda: (
                    screen._notes_state.autosave_state,
                    screen._library_note_session.snapshot.status_message,
                    screen.focused,
                ),
            )

            vetoed = [status for state, status, _focused in states if state == "validation"]
            assert vetoed, (
                "no autosave was vetoed during the burst; the guard was never "
                "exercised"
            )
            # The reason is reported in the save status instead of by moving.
            assert all("whitespace" in status for status in vetoed), vetoed
            assert all(focused is body for _state, _status, focused in states), (
                "a vetoed autosave moved focus off the body mid-typing: "
                f"{[getattr(focused, 'id', None) for *_rest, focused in states]}"
            )
            assert body.text == _BODY + keys
            assert title.value == _TITLE + " "
            row = profile.note(profile.note_id)
            assert (row["title"], row["content"]) == (_TITLE, _BODY)
    profile.db.close_connection()


async def test_keys_typed_while_a_save_is_in_flight_are_saved(tmp_path, monkeypatch):
    """Review finding #5: keys that land during a slow save still persist.

    ``_schedule_library_note_autosave`` arms nothing while the session is
    saving; the coordinator's save loop picks those keys up instead.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.2)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _notes: None)
    original_save = profile.scope_service.save_note
    in_flight: list[bool] = []

    async def _slow_save(**kwargs):
        in_flight.append(True)
        await asyncio.sleep(0.8)
        return await original_save(**kwargs)

    monkeypatch.setattr(profile.scope_service, "save_note", _slow_save)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, "a")
            await _until(
                pilot,
                lambda: bool(screen._library_note_session.snapshot.saving),
                "the autosave to start its slow save",
                timeout=10.0,
            )
            await _type(pilot, "bcd")  # typed while that save is in flight
            assert screen._library_note_session.snapshot.saving

            await _until(
                pilot,
                lambda: profile.note(profile.note_id)["content"] == _BODY + "abcd",
                "the keys typed during the save to reach the database",
                timeout=15.0,
            )
            assert not screen._library_note_session.snapshot.dirty
    profile.db.close_connection()


# --- AC#1: prompt and skill drafts get the same barrier ---------------------


async def test_ctrl_q_with_a_dirty_prompt_draft_asks_and_keeps_it(
    tmp_path, monkeypatch
):
    """Prompts are explicit-Save only: the quit asks instead of dropping them."""
    events: list = []
    app, _ = _library_app(
        tmp_path, monkeypatch, events, lambda _notes: None, notes=False
    )
    prompts_db = PromptsDatabase(tmp_path / "prompts.db", client_id="task-34000-1")
    prompt_id, _uuid, _message = prompts_db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
    )
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db), server_service=None
    )
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-browse-prompts", Button).press()
            row = await _wait_for_selector(
                screen, pilot, f"#library-prompt-row-{prompt_id}"
            )
            row.press()
            name = await _wait_for_selector(screen, pilot, "#library-prompt-name")
            await _until(
                pilot, lambda: screen._prompts_state.editor_armed, "the prompt editor"
            )
            name.focus()
            await pilot.press("!")
            await _until(pilot, lambda: screen._prompts_state.dirty, "a dirty prompt")
            typed = name.value
            toasts.clear()

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "Ctrl+Q quit past a dirty prompt draft"
            # Review finding #6: the prompt says it; no veto toast behind it.
            assert not [t for t in toasts if "Unsaved Prompt" in t], toasts
            assert _prompt_title(prompt) == (
                'Quit and discard unsaved changes to "Summarize"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert screen._prompts_state.dirty
            assert name.value == typed
    prompts_db.close_connection()


async def test_ctrl_q_with_a_dirty_skill_draft_asks(tmp_path, monkeypatch):
    events: list = []
    app, _ = _library_app(
        tmp_path, monkeypatch, events, lambda _notes: None, notes=False
    )
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-create-skill", Button).press()
            name = await _wait_for_selector(screen, pilot, "#library-skill-name")
            await _until(
                pilot, lambda: screen._skills_state.editor_armed, "the skill editor"
            )
            name.focus()
            await _type(pilot, "dirty-demo")
            await _until(pilot, lambda: screen._skills_state.dirty, "a dirty skill")
            toasts.clear()

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "Ctrl+Q quit past a dirty skill draft"
            assert not [t for t in toasts if "Unsaved skill" in t], toasts
            assert _prompt_title(prompt) == (
                'Quit and discard unsaved changes to "new skill"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert screen._skills_state.dirty
            assert name.value == "dirty-demo"


# --- AC#1: Folder files edits are flushed to the file on disk ---------------


async def test_ctrl_q_flushes_a_folder_files_edit_to_disk(tmp_path, monkeypatch):
    """B-static: ``shutdown`` stopped the Folder files autosave without saving."""
    root = tmp_path / "folder"
    root.mkdir()
    source = root / "source.md"
    source.write_text("# Source\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root, replica=replica, autosave_delay=3600, poll_interval=3600
    )
    events: list = []
    app, _ = _library_app(
        tmp_path,
        monkeypatch,
        events,
        lambda _notes: source.read_text(encoding="utf-8"),
        notes=False,
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen._notes_state.file_notes_workspace_factory = lambda: workspace
            screen.query_one("#library-row-browse-notes", Button).press()
            files = await _wait_for_selector(
                screen, pilot, "#library-notes-source-files"
            )
            files.press()
            await _until(
                pilot,
                lambda: workspace.initialized and workspace.is_mounted,
                "Folder files to mount",
            )
            assert await workspace.open_path("source.md")
            editor = workspace.query_one("#file-notes-editor", TextArea)
            await _until(pilot, lambda: not editor.read_only, "an editable file")
            await _type_at_end(pilot, editor, "tail")
            await _until(
                pilot, lambda: workspace.save_state == "dirty", "a dirty file edit"
            )

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, text)] = events
            assert kind == "quit"
            # The workspace keeps the file's final newline on save.
            assert text == "# Source\ntail\n", "Ctrl+Q exited before the file was saved"
    replica.close()
