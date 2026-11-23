"""Escape during an Add-from-files import leaves it running (TASK-34000.40 N-34).

The 2026-10-02 review found Escape and the visible "‹ Notes" button disagree
while Import once runs: ‹ Notes returned to the list and the import carried
on, Escape called ``cancel()`` and the receipt read "Cancelled. Finished items
were not rolled back." -- a navigation key that stopped a half-imported
folder. The footer even said so ("esc cancel").

AC#5: Escape during an import returns to Notes and the import keeps running,
as ‹ Notes does; cancelling stays available as the explicit "Cancel import".
Ruling T10-c: re-entering while the import still runs shows the RUNNING
import (progress and Cancel import), never the chooser; a running CHECK keeps
Escape = cancel, because a check that finished while the user was away would
be lost -- Add from files ▸ Import once starts a new selection
(``LibraryNoteImportController.begin_selection``), and a check writes
nothing, so cancelling it loses no data. Ruling T10-e: the import tier's
footer chip names Escape's destination while importing (fix round 1, review
R0 M-4: ``esc back to notes``, the same words as an idle import), and F1
mirrors the footer. Fix round 1 M-7: re-entering the running import lands
focus on its progress region, never on Cancel import, so a blind Enter there
cancels nothing.

Two boots of the real ``LibraryScreen`` on the production stylesheet over a
real ChaChaNotes database and the real import executor. The first note write
is held on a thread event (``LocalNoteImportTarget.create_note``), so the
import is provably mid-flight when the key lands; the executor runs one item
per batch so a cancel lands between items, as it does in any folder larger
than one batch. Every claim is a real keypress or button press and what the
user then sees: painted text, the view, and rows in the database.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.binding import Binding
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_note_import_state import NoteImportPhase
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.note_import_execution_models import ImportSessionState
from tldw_chatbook.Notes.note_import_executor import (
    LocalNoteImportTarget,
    NoteImportExecutor,
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Library_Modules.library_skills_browse_controller import (
    LibrarySkillsBrowseController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)
#: A behaviour wait: long enough for a check or an import step under CI
#: contention, short enough that a RED arm names its reason in seconds.
WAIT = 15.0
#: How long a held write or check waits for the test to release it before it
#: proceeds anyway -- a failing test must never strand an executor thread.
HOLD = 60.0
NOTES = 5
#: The cells a focused or framed compact button paints around its label.
_FRAME = "┃▊▎▌▐│ "


class _Host(LibraryProductionCSSHarness):
    """The production-CSS Library harness plus the app's real F1 seam."""

    BINDINGS = [Binding("f1", "show_workbench_help", "Help")]

    async def action_show_workbench_help(self) -> None:
        handler = getattr(self.screen, "action_show_workbench_help", None)
        if callable(handler):
            result = handler()
            if asyncio.iscoroutine(result):
                await result


class _Gate:
    """Hold the first call of a worker-thread function until released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def wrap(self, function: Callable[..., object]) -> Callable[..., object]:
        def held(*args: object, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                self.release.wait(HOLD)
            return function(*args, **kwargs)

        return held


class _Profile:
    """A real Notes database, scope service and a five-note folder to import."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.database = CharactersRAGDB(
            tmp_path / "notes.sqlite", client_id="task-34000-40-t10"
        )
        folders = LocalNoteFolderRepository(self.database)
        self.interop = NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="task-34000-40-t10",
            global_db_to_use=self.database,
        )
        self.scope = NotesScopeService(
            local_notes_service=self.interop,
            server_service=None,
            folder_repository=folders,
        )
        monkeypatch.setattr(
            library_screen_module,
            "get_notes_sync_state_db_path",
            lambda: tmp_path / "import-receipts.sqlite",
        )
        self.folder = tmp_path / "Field notes"
        self.folder.mkdir()
        for index in range(1, NOTES + 1):
            (self.folder / f"note-{index}.md").write_text(
                f"# Field note {index}\n\nBody {index}.", encoding="utf-8"
            )

    def host(self) -> _Host:
        app = _build_test_app()
        _seed_conversations(app, _two_conversations(), notes=[])
        app.chachanotes_db = self.database
        app.notes_scope_service = self.scope
        return _Host(app, screen=LibraryScreen(app))

    def close(self) -> None:
        self.interop.close_all_user_connections()
        self.database.close_connection()


async def _until(pilot, predicate, what) -> None:
    """Pump until ``predicate()`` holds; ``what`` may be read at raise time."""
    try:
        async with asyncio.timeout(WAIT):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def _painted_in(host, widget) -> str:
    """What the terminal shows in ``widget``'s region, or "" mid-batch.

    Textual recomposes inside ``App.batch_update`` and displays nothing until
    the batch ends, but the compositor can still be read in between: a
    background reload's receipt recompose read there had lost its "Import
    once" heading and "Import completed." line, a frame no user ever sees.
    """
    if host._batch_count:  # noqa: SLF001 - no public "frame displayed" signal
        return ""
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


def _label_of(host, screen, selector: str) -> str:
    """A button's painted label, without the focus frame (``┃ … ┃``)."""
    return _painted_of(host, screen, selector).strip(_FRAME)


def _footer(host, screen) -> str:
    found = screen.query(AppFooterStatus)
    return _painted_in(host, found.first()) if found else ""


def _footer_lacks(host, screen, text: str) -> bool:
    """A negative footer check that holds only on a painted frame.

    Mid-batch ``_footer`` reads "", which lacks everything: a bare ``not in``
    would pass there without reading the footer at all.
    """
    painted = _footer(host, screen)
    return bool(painted) and text not in painted


def _shown(screen, selector: str) -> bool:
    found = screen.query(selector)
    return bool(found) and bool(found.first().display)


async def _press(host, screen, pilot, selector: str) -> None:
    await _until(pilot, lambda: _shown(screen, selector), f"{selector} to mount")
    button = screen.query(selector).first(Button)
    assert not button.disabled, f"{selector} is disabled: {button.label}"
    button.press()


def _batch_of_one(controller) -> None:
    """The real executor, one item per batch: a cancel lands between items."""

    def factory(database, repository, receipts) -> NoteImportExecutor:
        return NoteImportExecutor(
            target=LocalNoteImportTarget(db=database, folder_repository=repository),
            receipt_repository=receipts,
            batch_size=1,
        )

    controller._executor_factory = factory  # noqa: SLF001 - batch size only


async def _open_import_once_on_the_folder(host, screen, pilot, folder: Path) -> None:
    """Notes ▸ Add from files… ▸ Import once, the picker answering ``folder``."""

    original_push_screen = screen.app.push_screen

    def push_screen(dialog, callback=None, *args, **kwargs):
        if not isinstance(dialog, FileOpen):
            return original_push_screen(dialog, callback, *args, **kwargs)
        if callback is not None:
            screen.run_worker(callback(folder))
        return None

    screen.app.push_screen = push_screen
    await _press(host, screen, pilot, "#library-row-browse-notes")
    await _press(host, screen, pilot, "#library-notes-add-from-files")
    await _press(host, screen, pilot, "#notes-add-import-once")
    await _until(
        pilot,
        lambda: screen._library_note_import_controller.snapshot.selected_paths
        == (folder,)
        and _shown(screen, "#note-import-check"),
        "the folder to be chosen",
    )


async def _review_the_folder(host, screen, pilot) -> None:
    await _press(host, screen, pilot, "#note-import-check")
    await _until(
        pilot,
        lambda: _shown(screen, "#note-import-import")
        and not screen.query("#note-import-import").first(Button).disabled,
        "the check to reach its review",
    )


async def _clear_focus(screen, pilot) -> None:
    """Leave the Library with nothing focused, failing closed if it re-lands.

    A restore parked behind a recompose could otherwise put focus back before
    F1 opens, and the pin would test a focused resume instead.
    """
    screen.set_focus(None)
    await pilot.pause()
    assert screen.focused is None, (
        f"a parked restore re-landed focus on {getattr(screen.focused, 'id', None)!r}"
    )


async def _f1_shortcuts(host, screen, pilot) -> tuple[tuple[str, str], ...]:
    """Open F1, read the rows it lists, close it again."""
    await pilot.press("f1")
    await _until(
        pilot,
        lambda: isinstance(host.screen, WorkbenchHelpPanel),
        lambda: f"F1 to open help: stack={host.screen_stack!r}",
    )
    shortcuts = tuple(host.screen.state.shortcuts)
    await pilot.press("escape")
    await _until(pilot, lambda: host.screen is screen, "help to close")
    return shortcuts


async def test_escape_during_an_import_keeps_it_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#5: Escape is ‹ Notes -- the import runs on and completes.

    While the first write is held the footer reads ``esc back to notes``
    (not ``esc cancel``) and F1 lists the same row. Escape lands on the Notes
    list with the import still IMPORTING and its cancel signal unset; the
    toolbar's "View import" re-enters the RUNNING import, Cancel import
    painted and pressable (T10-c) but focus on the progress region (M-7), so
    a blind Enter cancels nothing. Released, all five notes are written and
    the "Last import" receipt says so, with no "Cancelled" caveat.
    """
    profile = _Profile(tmp_path, monkeypatch)
    write = _Gate()
    monkeypatch.setattr(
        LocalNoteImportTarget,
        "create_note",
        write.wrap(LocalNoteImportTarget.create_note),
    )
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            _batch_of_one(controller)
            await _open_import_once_on_the_folder(host, screen, pilot, profile.folder)
            await _review_the_folder(host, screen, pilot)
            await _press(host, screen, pilot, "#note-import-import")
            await _until(pilot, write.entered.is_set, "the first note write to start")
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING

            await _until(
                pilot,
                lambda: _footer(host, screen).startswith("esc back to notes |"),
                lambda: f"the running import's footer, painted {_footer(host, screen)!r}",
            )
            assert _footer_lacks(host, screen, "esc cancel"), _footer(host, screen)
            shortcuts = await _f1_shortcuts(host, screen, pilot)
            assert ("esc", "back to notes") in shortcuts, shortcuts
            assert ("esc", "cancel") not in shortcuts, shortcuts
            # F1 still lists the pane key the footer's global cluster carries.
            assert ("F6", "next pane") in shortcuts, shortcuts

            await pilot.press("escape")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "list"
                and getattr(screen.focused, "id", None) == "library-notes-filter",
                lambda: (
                    "Escape to return to the Notes list: "
                    f"view={screen._notes_state.view!r} "
                    f"cancel_requested={controller.snapshot.cancel_requested} "
                    f"focused={getattr(screen.focused, 'id', None)!r}"
                ),
            )
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING
            assert controller.snapshot.cancel_requested is False
            cancel_event = controller._cancel_event  # noqa: SLF001
            assert cancel_event is not None and not cancel_event.is_set()

            # T10-c: re-entry while it still runs shows the running import.
            await _until(
                pilot,
                lambda: _label_of(host, screen, "#library-notes-add-from-files")
                == "View import",
                lambda: (
                    "the toolbar to offer View import, painted "
                    f"{_label_of(host, screen, '#library-notes-add-from-files')!r}"
                ),
            )
            await _press(host, screen, pilot, "#library-notes-add-from-files")
            await _until(
                pilot,
                lambda: _label_of(host, screen, "#note-import-cancel")
                == "Cancel import"
                and _shown(screen, "#note-import-progress"),
                "the running import with Cancel import to be painted",
            )
            assert not screen.query_one("#note-import-cancel", Button).disabled
            assert screen._notes_state.view == "import"
            assert not screen.query("#notes-add-import-once"), "the chooser came back"
            # M-7: the re-entry lands on the progress region, not on Cancel
            # import, so a blind Enter cancels nothing (the receipt below
            # reads completed, five notes).
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "note-import-body",
                lambda: (
                    "re-entry focus on the progress region, found "
                    f"{getattr(screen.focused, 'id', None)!r}"
                ),
            )
            # Final review (Task 10): a Tab-focused Cancel import names its
            # Enter action in the footer, and F1 mirrors it; Shift+Tab goes
            # back to the progress region, where the blind Enter below
            # cancels nothing. RED at 29d2272b95: the import tier named no
            # Enter action at all ("enter_label_after_tab == ''").
            await pilot.press("tab")
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "note-import-cancel"
                and "enter cancel import" in _footer(host, screen),
                lambda: (
                    "Tab onto Cancel import with its Enter chip painted: "
                    f"focus={getattr(screen.focused, 'id', None)!r} "
                    f"footer={_footer(host, screen)!r}"
                ),
            )
            assert screen._library_focus_enter_label() == "cancel import"
            shortcuts = await _f1_shortcuts(host, screen, pilot)
            assert ("enter", "cancel import") in shortcuts, shortcuts
            assert ("esc", "back to notes") in shortcuts, shortcuts
            await pilot.press("shift+tab")
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "note-import-body"
                and _footer_lacks(host, screen, "enter cancel import"),
                lambda: (
                    "Shift+Tab back to the progress region, chip gone: "
                    f"focus={getattr(screen.focused, 'id', None)!r} "
                    f"footer={_footer(host, screen)!r}"
                ),
            )
            assert controller.snapshot.cancel_requested is False
            await pilot.press("enter")
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "list",
                "a second Escape to leave the running import again",
            )
            assert controller.snapshot.cancel_requested is False

            write.release.set()
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.RECEIPT,
                lambda: f"the import to settle, phase={controller.snapshot.phase}",
            )
            receipt = controller.snapshot.receipt
            assert receipt.state is ImportSessionState.COMPLETED
            assert receipt.imported == NOTES
            assert profile.database.count_notes() == NOTES

            await _press(host, screen, pilot, "#library-notes-import-receipt")
            await _until(
                pilot,
                lambda: _painted_of(host, screen, "#note-import-receipt")
                == f"{NOTES} notes created",
                lambda: (
                    "the Last import receipt, painted "
                    f"{_painted_of(host, screen, '#note-import-receipt')!r}"
                ),
            )
            canvas = _painted_of(host, screen, "#library-note-import-canvas")
            assert "Import completed." in canvas, canvas
            assert "Cancelled" not in canvas, canvas
    finally:
        write.release.set()
        profile.close()


async def test_cancel_import_still_cancels_and_escape_still_cancels_a_running_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative controls: the explicit cancels keep working (green on base).

    T10-c, the rule chosen for CHECKING: Escape cancels the running check and
    stays on the import with the selection kept (footer ``esc cancel``), so a
    check cannot finish unseen and be thrown away by the next Add from files.
    AC#5: the "Cancel import" button stops a running import -- the held first
    note is written, nothing after it, and the receipt says so.
    """
    profile = _Profile(tmp_path, monkeypatch)
    write = _Gate()
    check = _Gate()
    monkeypatch.setattr(
        LocalNoteImportTarget,
        "create_note",
        write.wrap(LocalNoteImportTarget.create_note),
    )
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            _batch_of_one(controller)
            controller._plan_selection = check.wrap(  # noqa: SLF001 - hold the check
                controller._plan_selection  # noqa: SLF001
            )
            await _open_import_once_on_the_folder(host, screen, pilot, profile.folder)

            await _press(host, screen, pilot, "#note-import-check")
            await _until(pilot, check.entered.is_set, "the check to start")
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.CHECKING
                and "esc cancel" in _footer(host, screen),
                lambda: f"the running check's footer, painted {_footer(host, screen)!r}",
            )
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: controller.snapshot.phase is not NoteImportPhase.CHECKING,
                "Escape to cancel the running check",
            )
            check.release.set()
            assert screen._notes_state.view == "import"
            assert controller.snapshot.selected_paths == (profile.folder,)
            assert profile.database.count_notes() == 0

            await _review_the_folder(host, screen, pilot)
            await _press(host, screen, pilot, "#note-import-import")
            await _until(pilot, write.entered.is_set, "the first note write to start")
            await _until(
                pilot,
                lambda: _label_of(host, screen, "#note-import-cancel")
                == "Cancel import",
                lambda: (
                    "Cancel import to be painted: "
                    f"phase={controller.snapshot.phase} "
                    f"view={screen._notes_state.view!r} "
                    f"cancel={_painted_of(host, screen, '#note-import-cancel')!r} "
                    f"canvas={_painted_of(host, screen, '#library-note-import-canvas')!r}"
                ),
            )
            await _press(host, screen, pilot, "#note-import-cancel")
            await _until(
                pilot,
                lambda: controller.snapshot.cancel_requested,
                "Cancel import to request cancellation",
            )
            write.release.set()
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.RECEIPT,
                lambda: f"the cancelled import to settle, phase={controller.snapshot.phase}",
            )
            assert controller.snapshot.receipt.state is ImportSessionState.CANCELLED
            assert profile.database.count_notes() == 1
            await _until(
                pilot,
                lambda: "Cancelled. Finished items were not rolled back."
                in _painted_of(host, screen, "#library-note-import-canvas"),
                "the cancelled receipt to be painted",
            )
    finally:
        check.release.set()
        write.release.set()
        profile.close()


async def test_a_check_started_with_focus_outside_the_import_says_esc_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The running check's footer follows its phase, not a focus move.

    UI Fast Lane 30 (run 45018779351): the test above timed out on a
    CHECKING import whose footer still read ``esc back to notes``. The import
    publish re-derived no footer; ``esc cancel`` arrived only when the
    check's recompose removed the focused control and Textual's focus reset
    refreshed the screen. With focus outside the import -- here on the Items
    pane grip, where that reset parks it -- the footer said Escape goes back
    to Notes while Escape cancelled the check. The grip is focused as a
    click focuses it, so the check's own repaint keeps it (TASK-41600). A
    grip focused in code reads as Textual's reset to that repaint, which
    restores focus into the import -- a focus move that repaints the footer
    by itself, and that under load could land before the publish did
    (nav-home round 3: 1 red in 40 solo runs).
    """
    profile = _Profile(tmp_path, monkeypatch)
    check = _Gate()
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            controller._plan_selection = check.wrap(  # noqa: SLF001 - hold the check
                controller._plan_selection  # noqa: SLF001
            )
            await _open_import_once_on_the_folder(host, screen, pilot, profile.folder)
            # Folder recognition and every queued post-recompose focus restore
            # land first, so moving focus to the grip is the last focus move.
            # (Where that restore lands varies: one run here ended on None.)
            canvases = [
                widget
                for widget in screen.query("*")
                if isinstance(widget, PostRecomposeCallback)
            ]
            assert canvases, "no canvas to settle: the wait would check workers only"
            await _until(
                pilot,
                lambda: not any(
                    worker.group == "library_note_import_recognition"
                    and not worker.is_finished
                    for worker in host.workers
                )
                and not any(
                    widget.has_pending_recompose_callback
                    for widget in screen.query("*")
                    if isinstance(widget, PostRecomposeCallback)
                ),
                "the selection to settle",
            )
            grip = screen.query_one("#library-browse-items-grip")
            # What a click on the grip records (``get_focusable_widget_at``).
            screen._library_input_focus = grip
            screen.set_focus(grip)

            await _press(host, screen, pilot, "#note-import-check")
            await _until(pilot, check.entered.is_set, "the check to start")
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.CHECKING
                and "esc cancel" in _footer(host, screen),
                lambda: (
                    "the running check's footer with focus on "
                    f"{getattr(screen.focused, 'id', None)!r}, painted "
                    f"{_footer(host, screen)!r}"
                ),
            )
            assert screen.focused is grip, "a focus move refreshed the footer"
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.SELECT
                and _footer(host, screen).startswith("esc back to notes |"),
                lambda: (
                    "Escape to cancel the check and the footer to follow, "
                    f"phase={controller.snapshot.phase} "
                    f"painted {_footer(host, screen)!r}"
                ),
            )
            assert screen._notes_state.view == "import"
    finally:
        check.release.set()
        profile.close()


async def test_f1_closing_over_an_unfocused_import_keeps_focus_in_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Library left with nothing focused gets focus back from F1's close.

    Review r11s, Probe B: with Library focus on None over the running
    import, closing F1 let Textual's ``AUTO_FOCUS = "*"`` land on the nav
    bar's first tab (``nav-home``), where a blind Enter leaves Library. The
    Escape after it stayed there: the list-entry arm clears only a pane
    grip, and its expiry lands only from None. That is the lane's end state
    (view 'list', focus 'nav-home').
    """
    profile = _Profile(tmp_path, monkeypatch)
    write = _Gate()
    monkeypatch.setattr(
        LocalNoteImportTarget,
        "create_note",
        write.wrap(LocalNoteImportTarget.create_note),
    )
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            _batch_of_one(controller)
            await _open_import_once_on_the_folder(host, screen, pilot, profile.folder)
            await _review_the_folder(host, screen, pilot)
            await _press(host, screen, pilot, "#note-import-import")
            await _until(pilot, write.entered.is_set, "the first note write to start")
            # The running import's pre-write progress publishes were all
            # scheduled before the write was entered; let their recomposes and
            # focus restores drain, or a pending restore re-lands focus first.
            # Settled is off the grip Textual's reset parks focus on: on Cancel
            # import, or on None (seen once under load, a restore landing none).
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None)
                in {"note-import-cancel", None},
                lambda: f"the import's focus: {getattr(screen.focused, 'id', None)!r}",
            )
            await pilot.pause()

            await _clear_focus(screen, pilot)
            await _f1_shortcuts(host, screen, pilot)
            # Review R0 M-7: the progress region, never Cancel import.
            assert getattr(screen.focused, "id", None) == "note-import-body", (
                f"F1's close landed {getattr(screen.focused, 'id', None)!r}"
            )
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "list"
                and getattr(screen.focused, "id", None) == "library-notes-filter",
                lambda: (
                    "Escape to return to the Notes list: "
                    f"view={screen._notes_state.view!r} "
                    f"cancel_requested={controller.snapshot.cancel_requested} "
                    f"focused={getattr(screen.focused, 'id', None)!r}"
                ),
            )
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING
    finally:
        write.release.set()
        profile.close()


async def test_f1_closing_over_an_unfocused_list_lands_its_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same resume rule outside Notes: a list lands its own entry row.

    Before the fix, Conversations (like Media, Skills, Collections, Ingest
    and New note) ended on ``nav-home`` when F1 closed over nothing focused.
    The entry row lands while the list-entry window is open; a page that
    reconciles after it (seen once in 40 loaded runs, round 3) gets the
    re-check's landing instead, the list's filter (review r11n I-2). The one
    other tolerated miss is the pane grip: the resume's own list reload can
    replace the row just focused, and Textual's reset parks focus there --
    the list-entry channel outside Notes does not take that back (seen once
    in 26 loaded runs). Without the landing it ends on None or ``nav-home``.
    """
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(host, screen, pilot, "#library-row-browse-conversations")
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None)
                == "library-conversation-row-0",
                lambda: f"the entry row: {getattr(screen.focused, 'id', None)!r}",
            )
            await pilot.pause()
            await _clear_focus(screen, pilot)
            await _f1_shortcuts(host, screen, pilot)
            await _until(
                pilot,
                lambda: screen.focused is not None
                and (
                    screen.focused.id
                    in {"library-conversation-row-0", "library-conversations-filter"}
                    or screen.focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS)
                ),
                lambda: (
                    "F1's close to land the Conversations entry row, found "
                    f"{getattr(screen.focused, 'id', None)!r}"
                ),
            )
    finally:
        profile.close()


async def test_f1_closing_over_unfocused_folder_files_lands_its_arrival_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Folder files lands where arriving on it does, never the source toggle.

    Review r11n I-1: the resume landing's last resort was the first content
    control, Notes' "Database" source toggle, where a blind Enter leaves
    Folder files. With no folder linked, arrival lands Choose folder.
    Review r11n2 N-4: that landing is the app's move, so it records no user
    focus intent (which would veto Notes' own deferred restores).
    """
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(host, screen, pilot, "#library-row-browse-notes")
            await _press(host, screen, pilot, "#library-notes-source-files")
            await _until(
                pilot,
                lambda: screen._file_notes_active()
                and getattr(screen.focused, "id", None) == "file-notes-choose-root",
                lambda: f"the arrival, found {getattr(screen.focused, 'id', None)!r}",
            )
            await pilot.pause()
            await _clear_focus(screen, pilot)
            intent = screen._notes_state.focus_intent_generation
            await _f1_shortcuts(host, screen, pilot)
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "file-notes-choose-root",
                lambda: (
                    "F1's close to land Folder files' arrival target, found "
                    f"{getattr(screen.focused, 'id', None)!r}"
                ),
            )
            assert screen._file_notes_active()
            await pilot.pause()
            assert screen._notes_state.focus_intent_generation == intent
    finally:
        profile.close()


async def test_f1_closing_over_a_slow_skills_list_lands_its_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list whose rows outlive the entry window still gets focus back.

    Review r11n I-2: the resume armed the list-entry channel, and the last
    resort stood down while it was armed. Rows arriving after its 2 s window
    ended it with nothing landed (it lands from None on Notes only), so
    focus stayed on None for good. The Skills filter is that route's entry
    while its page loads.
    """
    profile = _Profile(tmp_path, monkeypatch)
    page = threading.Event()
    service_page = LibrarySkillsBrowseController._service_page

    async def held_page(self, *args, **kwargs):
        await asyncio.to_thread(page.wait, HOLD)
        return await service_page(self, *args, **kwargs)

    monkeypatch.setattr(LibrarySkillsBrowseController, "_service_page", held_page)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(host, screen, pilot, "#library-row-browse-skills")
            await _until(
                pilot,
                lambda: screen._library_selected_row_id == "browse-skills"
                and screen._library_skills_browse_controller.result.status
                == "loading",
                "the Skills page to start loading",
            )
            await _clear_focus(screen, pilot)
            await _f1_shortcuts(host, screen, pilot)
            await _until(
                pilot,
                lambda: not screen._library_pending_list_entry_focus,
                "the list-entry window to end",
            )
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "library-skills-filter",
                lambda: (
                    "the Skills filter while its page loads, found "
                    f"{getattr(screen.focused, 'id', None)!r}"
                ),
            )
            page.set()
            await _until(
                pilot,
                lambda: screen._library_skills_browse_controller.result.status
                != "loading"
                and getattr(screen.focused, "id", None) == "library-skills-filter",
                lambda: (
                    "the loaded Skills list to keep its filter, found "
                    f"{getattr(screen.focused, 'id', None)!r}"
                ),
            )
    finally:
        page.set()
        profile.close()
