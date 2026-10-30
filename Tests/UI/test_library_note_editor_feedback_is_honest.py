"""The note editor's feedback says what actually happened (TASK-34000.29).

Review notes-library-ux-review-2026-10-02, verified on ``bc64fb2272``:

- N-12: Info > Export Markdown > Esc in the file dialog painted "Export
  failed — choose another destination and try again." beside the toast "Note
  export cancelled.", and left focus on Delete (the export disabled the
  button that started it; Textual moved focus to the next enabled stop and
  nothing brought it back). A successful export and Copy did the same.
- N-25: a refused leave toasted "Can't leave yet — fix the title or press
  Discard new note." for every validation veto, on notes with no Discard
  control and for vetoes that were not about the title.
- N-26: a note created after opening a linked-to note showed THAT note's
  "Linked from (1)"; a Blank note opened directly read "checking…" forever.

Four boots of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database behind the real Notes scope service (``Index —
start here`` links to ``Ideas inbox`` through ``note://``). Every claim is a
real key press, click or button press, and what the user then sees: painted
text, the toasts, the focused control (with its footer chip), and the file on
disk.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _NotesProfile,
    _open_note,
    _record_toasts,
    _type,
    _until,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_notes_session import (
    NoteFlushOutcome,
    NoteFlushOutcomeKind,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: A behaviour wait: long enough for a worker under CI contention, short
#: enough that a RED arm names its reason in seconds.
WAIT = 10.0

IDEAS = "Ideas inbox"  # ``_NotesProfile``'s first note
INDEX = "Index — start here"

EXPORT_MD = "library-note-context-export-md"
COPY = "library-note-context-copy"
CANCELLED = "Note export cancelled."
TITLE_VETO = "Title begins or ends with whitespace — remove it to save."
NO_LINKS = "Linked from (0) — no notes link here yet"


class _Profile:
    """``_NotesProfile``'s two notes plus ``Index — start here``, whose body
    links to ``Ideas inbox`` the way the importer and the ``[[`` picker do."""

    def __init__(self, tmp_path) -> None:
        self.profile = _NotesProfile(tmp_path)
        self.ideas_id = self.profile.note_id
        self.index_id = self.profile.db.add_note(
            INDEX, f"Start with [[{IDEAS}]](note://{self.ideas_id}) today."
        )
        assert self.index_id


def _mounted(tmp_path, monkeypatch):
    """The real Library hierarchy over the profile; toasts and copies recorded."""
    app = _build_test_app()
    _seed_conversations(app, [])
    seeded = _Profile(tmp_path)
    app.chachanotes_db = seeded.profile.db
    app.notes_scope_service = seeded.profile.scope_service
    app.notes_service = seeded.profile.interop
    host = LibraryProductionCSSHarness(app, screen=LibraryScreen(app))
    app.notify = host.notify
    toasts = _record_toasts(monkeypatch, app)
    copies: list[str] = []
    app.copy_to_clipboard = copies.append
    return host, seeded, toasts, copies


async def _wait(pilot, predicate, what) -> None:
    """``_until`` with a message that may be a callable (read at raise time)."""
    try:
        await _until(pilot, predicate, "", timeout=WAIT)
    except AssertionError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def _painted(host) -> str:
    """Every painted cell of the top screen."""
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def _painted_in(host, widget) -> str:
    """The compositor cells painted inside ``widget``'s region, one line."""
    region = widget.region
    strips = list(host.screen._compositor.render_strips())
    rows = (
        strips[y].crop(region.x, region.right).text
        for y in range(max(region.y, 0), min(region.bottom, len(strips)))
    )
    return " ".join(" ".join(rows).split())


def _painted_of(host, screen, selector: str) -> str:
    found = screen.query(selector)
    return _painted_in(host, found.first()) if found else ""


def _footer(host, screen) -> str:
    found = screen.query(AppFooterStatus)
    return _painted_in(host, found.first()) if found else ""


def _focused_id(screen) -> str:
    return str(getattr(screen.focused, "id", "") or "")


def _open_note_id(screen) -> str:
    snapshot = screen._library_note_session.snapshot
    return snapshot.note_id if snapshot is not None else ""


async def _library(host, pilot) -> LibraryScreen:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    return screen


async def _info(host, screen, pilot) -> None:
    """Open Info on the open note; wait until Reuse & Export paints."""
    screen.query_one("#library-note-context", Button).press()
    await _wait(
        pilot,
        lambda: "Export Markdown" in _painted_of(host, screen, f"#{EXPORT_MD}"),
        "Info to paint Export Markdown",
    )


async def _export_dialog(host, pilot) -> FileSave:
    await _wait(
        pilot,
        lambda: isinstance(host.screen, FileSave)
        and isinstance(host.screen.focused, FileNameInput),
        "the export file dialog with its filename focused",
    )
    return host.screen


async def _back_on_screen(host, screen, pilot) -> None:
    await _wait(pilot, lambda: host.screen is screen, "the file dialog to close")


async def _focus_settles_on(
    host, screen, pilot, control: str, chip: str, *, after: str
) -> None:
    """Focus is on ``control`` and the footer says Enter there does ``chip``."""
    await _wait(
        pilot,
        lambda: _focused_id(screen) == control and chip in _footer(host, screen),
        lambda: (
            f"focus on #{control} after {after} (focus: #{_focused_id(screen)}, "
            f"footer: {_footer(host, screen)!r})"
        ),
    )


async def test_dismissing_the_export_dialog_leaves_no_failure_copy_and_one_toast(
    tmp_path, monkeypatch
) -> None:
    """N-12 (.29 AC#1): Esc in the export dialog is a cancel, not a failure.

    The one quiet signal is the toast; the status line keeps the note's own
    saved state and Info's transfer line stays empty.
    """
    host, seeded, toasts, _copies = _mounted(tmp_path, monkeypatch)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)
        await _wait(
            pilot,
            lambda: "Saved" in _painted_of(host, screen, "#library-note-status"),
            "the open note's Saved status",
        )
        toasts.clear()

        await pilot.click(f"#{EXPORT_MD}")
        await _export_dialog(host, pilot)
        await pilot.press("escape")
        await _back_on_screen(host, screen, pilot)
        # The toast is raised in the callback that settles the export; the
        # running "Export…" line is what the screen painted until then, so
        # its absence from the paint proves the settled state has painted.
        await _wait(
            pilot,
            lambda: CANCELLED in toasts and "Export…" not in _painted(host),
            lambda: f"the export to settle (toasts: {toasts!r})",
        )

        painted = _painted(host)
        assert "Export failed" not in painted, painted
        assert "failed" not in _painted_of(host, screen, "#library-note-status")
        assert "Saved" in _painted_of(host, screen, "#library-note-status")
        assert _painted_of(host, screen, "#library-note-context-transfer-status") == ""
        assert toasts == [CANCELLED], toasts


async def test_focus_returns_to_the_control_that_started_the_export(
    tmp_path, monkeypatch
) -> None:
    """N-12 (.29 AC#2): after a cancelled export, a written export and a Copy,
    focus is on the control that started it -- never on Delete.

    Each landing is followed by a real key on it (Enter re-opens the export;
    Shift+Tab walks to Copy; Tab walks back), and the folder holds exactly one
    export afterwards: returning focus never presses anything.

    Final review, T6 m1: pressing the still-enabled origin behind the open
    picker opens no second picker and claims no second token. T6 m2 is
    pinned on a fake below (``test_the_origin_return_retries_once_after_
    the_refresh``): traced here, a focus moved off the origin while the
    picker is open -- to Back, or to Delete when the origin is disabled --
    is captured by the apply's own canvas sync and replayed OVER the
    cancel's return (``_restore_library_notes_after_targeted_sync``, then
    ``_refocus_pruned_child``), the user's move winning by design, so no
    mounted arm can need the return while the origin stays enabled.
    """
    host, seeded, toasts, copies = _mounted(tmp_path, monkeypatch)
    folder = tmp_path / "exports"
    folder.mkdir()
    destination = folder / "ideas.md"
    pushes: list[str] = []
    original_push_screen = host.push_screen

    def counting_push_screen(dialog, *args, **kwargs):
        if isinstance(dialog, FileSave):
            pushes.append("FileSave")
        return original_push_screen(dialog, *args, **kwargs)

    monkeypatch.setattr(host, "push_screen", counting_push_screen)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)

        # Cancelled.
        await pilot.click(f"#{EXPORT_MD}")
        await _export_dialog(host, pilot)
        assert pushes == ["FileSave"]
        await pilot.press("escape")
        await _back_on_screen(host, screen, pilot)
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown",
            after="the cancelled export",
        )
        assert "delete note" not in _footer(host, screen)

        # Written: Enter on the returned focus re-opens the dialog.
        await pilot.press("enter")
        dialog = await _export_dialog(host, pilot)
        assert pushes == ["FileSave", "FileSave"]
        # m1: the still-enabled origin behind the open picker is refused
        # before any token is claimed -- one picker, one token, no toast.
        token_before = screen._notes_state.operation.token
        assert not screen.query_one(f"#{EXPORT_MD}", Button).disabled
        screen.query_one(f"#{EXPORT_MD}", Button).press()
        await pilot.pause(0.2)
        assert pushes == ["FileSave", "FileSave"], pushes
        assert host.screen is dialog
        assert screen._notes_state.operation.token == token_before
        await pilot.press("ctrl+a")
        filename = dialog.query_one(FileNameInput)
        filename.value = str(destination)
        await pilot.press("enter")
        await _back_on_screen(host, screen, pilot)
        await _wait(
            pilot,
            lambda: destination.exists()
            and "Export complete" in _painted(host),
            "the export to write and report",
        )
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown",
            after="the written export",
        )
        assert sorted(path.name for path in folder.iterdir()) == ["ideas.md"]
        assert f"title: {IDEAS}" in destination.read_text(encoding="utf-8")

        # Copy, reached and pressed from the keyboard.
        await pilot.press("shift+tab")
        await _focus_settles_on(
            host, screen, pilot, COPY, "copy note", after="Shift+Tab"
        )
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: len(copies) == 1 and "Copy complete" in _painted(host),
            "the copy to land and report",
        )
        await _focus_settles_on(host, screen, pilot, COPY, "copy note", after="Copy")
        assert IDEAS in copies[0]
        await pilot.press("tab")
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown", after="Tab"
        )
        assert sorted(path.name for path in folder.iterdir()) == ["ideas.md"]


def test_the_origin_return_retries_once_after_the_refresh() -> None:
    """Final review, T6 m2 (M-3): the deferred retry, on a controller-shaped
    fake. A control the apply has not re-enabled yet (not focusable) gets
    exactly ONE retry after the refresh, marked programmatic when it lands;
    a retry that finds the region changed does nothing; an already-focusable
    origin is focused at once and schedules nothing. RED under a no-op
    ``_return_focus_to_library_notes_operation_origin`` (measured)."""
    from types import MethodType, SimpleNamespace

    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    class _Origin:
        focusable = False

    origin = _Origin()
    deferred: list[tuple] = []
    focused: list[tuple] = []
    fake = SimpleNamespace(
        focused=None,
        query_one=lambda selector, _type=None: origin,
        _library_notes_active_region=lambda: "context",
        call_after_refresh=lambda fn, *args, **kwargs: deferred.append(
            (fn, args, kwargs)
        ),
        set_focus=lambda target, **kwargs: focused.append((target, kwargs)),
        _library_notes_programmatic_focus_target=None,
    )
    fake._return_focus_to_library_notes_operation_origin = MethodType(
        LibraryNotesController._return_focus_to_library_notes_operation_origin, fake
    )
    operation = SimpleNamespace(origin_control=EXPORT_MD, region="context")

    # Not focusable yet: one retry is scheduled, nothing is focused.
    LibraryNotesController._return_focus_to_library_notes_operation_origin(
        fake, operation
    )
    assert focused == []
    assert len(deferred) == 1 and deferred[0][2] == {"deferred": True}

    # The apply re-enabled it: the retry lands, marked programmatic.
    origin.focusable = True
    fn, args, kwargs = deferred[0]
    fn(*args, **kwargs)
    assert focused == [(origin, {"scroll_visible": False})]
    assert fake._library_notes_programmatic_focus_target is origin
    assert len(deferred) == 1, "a retry never schedules another retry"

    # A retry whose region moved on does nothing.
    fake._library_notes_active_region = lambda: "navigator"
    LibraryNotesController._return_focus_to_library_notes_operation_origin(
        fake, operation, deferred=True
    )
    assert len(focused) == 1

    # Focusable from the start: focused at once, nothing scheduled.
    fake._library_notes_active_region = lambda: "context"
    LibraryNotesController._return_focus_to_library_notes_operation_origin(
        fake, operation
    )
    assert len(focused) == 2 and len(deferred) == 1


async def test_a_refused_escape_names_the_field_and_mentions_discard_only_when_shown(
    tmp_path, monkeypatch
) -> None:
    """N-25 (.29 AC#6): the leave-veto toast is the veto's own reason.

    The status line is read first and the toast must repeat ITS sentence --
    the toast never retypes the reason. "Discard new note" is offered only
    while that control is painted: typing into a blank note hides it (the
    first genuine edit clears ``untouched_create_token``), so no user can
    reach a veto with Discard shown today. The last arm pins that branch's
    wiring on an untouched blank note with only the refusal injected. A
    refused leave keeps the editor open, and Backspace (a real key) then
    lets the same Escape leave.
    """
    host, seeded, toasts, _copies = _mounted(tmp_path, monkeypatch)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)

        # An existing note: a trailing space in the title.
        await pilot.click("#library-note-title")
        await pilot.press("end")
        await pilot.press("space")
        await _wait(
            pilot,
            lambda: screen.query_one("#library-note-title", Input).value
            == f"{IDEAS} ",
            "the trailing space in the title",
        )
        toasts.clear()
        await pilot.press("escape")
        await _wait(
            pilot,
            lambda: bool(toasts)
            and TITLE_VETO in _painted_of(host, screen, "#library-note-status"),
            lambda: f"a veto toast (toasts: {toasts!r})",
        )
        status = _painted_of(host, screen, "#library-note-status")
        assert status == TITLE_VETO, status
        assert toasts == [f"Can't leave yet — {status}"], toasts
        assert "Discard" not in toasts[0]
        assert "Discard new note" not in _painted(host)
        assert screen._notes_state.view == "editor"
        assert _open_note_id(screen) == seeded.ideas_id

        await pilot.press("backspace")
        await pilot.press("escape")
        await _wait(
            pilot,
            lambda: screen._notes_state.view == "list",
            "Escape to leave once the title is fixed",
        )
        assert seeded.profile.note(seeded.ideas_id)["title"] == IDEAS

        # A blank note: typing makes it no longer untouched, so Discard goes.
        await pilot.press("n")
        await _armed_editor(screen, pilot)
        blank_id = _open_note_id(screen)
        assert blank_id not in {seeded.ideas_id, seeded.index_id}
        await _wait(
            pilot,
            lambda: "Discard new note" in _painted(host),
            "Discard new note on the untouched blank note",
        )
        await _type(pilot, "Research Note ")
        await _wait(
            pilot,
            lambda: "Discard new note" not in _painted(host),
            "Discard new note to go once the note is touched",
        )
        toasts.clear()
        await pilot.press("escape")
        await _wait(pilot, lambda: bool(toasts), "the blank note's veto toast")
        assert toasts == [f"Can't leave yet — {TITLE_VETO}"], toasts
        assert screen._notes_state.view == "editor"

        # While Discard IS painted (an untouched blank note), the toast names
        # it. No typed edit can veto an untouched note, so the refusal is the
        # one seam this arm replaces; the control and the Escape are real.
        await pilot.press("backspace")
        await pilot.press("escape")
        await _wait(
            pilot,
            lambda: screen._notes_state.view == "list",
            "Escape to leave once the blank note's title is fixed",
        )
        await pilot.press("n")
        await _armed_editor(screen, pilot)
        await _wait(
            pilot,
            lambda: "Discard new note" in _painted(host),
            "Discard new note on the second blank note",
        )

        async def _refuse() -> NoteFlushOutcome:
            return NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO, TITLE_VETO)

        monkeypatch.setattr(screen, "_flush_library_note_save", _refuse)
        toasts.clear()
        await pilot.press("escape")
        await _wait(pilot, lambda: bool(toasts), "the untouched note's veto toast")
        assert toasts == [
            f"Can't leave yet — {TITLE_VETO} Or press Discard new note."
        ], toasts
        assert screen._notes_state.view == "editor"


async def test_a_new_note_settles_on_its_own_backlinks(tmp_path, monkeypatch) -> None:
    """N-26 (.29 AC#7): a created note's Info answers for that note.

    ``n`` after a linked-to note, and a Blank note from the rail's New, both
    settle on the real lookup's ``(0)`` -- never the previous note's ``(1)``
    and never an endless ``checking…``.
    """
    host, seeded, _toasts, _copies = _mounted(tmp_path, monkeypatch)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)
        header = "#library-note-context-backlinks-title"
        rows = "#library-note-context-backlinks"
        await _wait(
            pilot,
            lambda: "Linked from (1)" in _painted_of(host, screen, header)
            and INDEX in _painted_of(host, screen, rows),
            "Ideas inbox's Linked from (1) Index — start here",
        )

        await pilot.press("n")
        await _armed_editor(screen, pilot)
        created = _open_note_id(screen)
        assert created not in {seeded.ideas_id, seeded.index_id}
        await _info(host, screen, pilot)
        await _wait(
            pilot,
            lambda: _painted_of(host, screen, header) == NO_LINKS,
            lambda: f"the new note's own backlinks (header: "
            f"{_painted_of(host, screen, header)!r})",
        )
        assert _painted_of(host, screen, rows) == ""

        # Esc closes Info, Esc leaves the untouched note; then New > Blank.
        await pilot.press("escape")
        await pilot.press("escape")
        await _wait(
            pilot,
            lambda: screen._notes_state.view == "list",
            "Escape back to the list",
        )
        screen.query_one("#library-row-create-note", Button).press()
        await _wait(
            pilot,
            lambda: bool(screen.query("#library-notes-create-blank")),
            "the New note chooser",
        )
        screen.query_one("#library-notes-create-blank", Button).press()
        await _armed_editor(screen, pilot)
        assert _open_note_id(screen) not in {seeded.ideas_id, seeded.index_id, created}
        await _info(host, screen, pilot)
        await _wait(
            pilot,
            lambda: _painted_of(host, screen, header) == NO_LINKS,
            lambda: f"the Blank note's own backlinks (header: "
            f"{_painted_of(host, screen, header)!r})",
        )
