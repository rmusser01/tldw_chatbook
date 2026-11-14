"""TASK-34000.71: Keep a folder synced is offered on a profile with no synced folder.

Every new user has no lasting-sync root. Boot's TASK-21247 gate therefore
neither constructs nor starts the notes-sync runtime; the Library screen then
builds the owner lazily without starting it, so it sits at its constructor
status "starting", which the setup gate refuses -- and the one force-start
that would move it (``review_setup``) is reachable only through the option
the gate refuses. The chooser offered "○ Keep a folder synced", inert, to
every user who did not already have a synced folder.

This walk boots the real ``TldwCli`` (the production ``on_mount`` gate, the
production lazy owner, the production start-task bookkeeping) straight into
Library over a per-test state-DB path that holds no root, and proves:

- AC#1: the chooser offers Keep a folder synced without the inert prefix, and
  choosing it reaches the configure form, then (Check on a scratch folder)
  the review;
- AC#3: opening the chooser creates no state database -- it appears only
  once the review force-starts the runtime;
- AC#6: the lazily started runtime's start task is settled by app exit and
  the loop saw no unhandled error.

RED on the task's base (50838c3452): the label reads "○ Keep a folder
synced" because the owner is still "starting". One boot, about 10 s.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
    _painted_text,
)
from Tests.UI.test_library_shell import (
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _wait_for_library_screen(app, pilot) -> LibraryScreen:
    await _wait_for_condition(
        pilot,
        lambda: isinstance(app.screen, LibraryScreen),
        message="the app never routed to the Library screen",
        timeout=30.0,
    )
    return app.screen


async def _open_chooser(screen, pilot) -> None:
    """Press Add from files… and wait until the chooser canvas has painted."""

    screen.query_one("#library-notes-add-from-files", Button).press()
    controller = screen._library_notes_sync_controller

    def _canvas_settled() -> bool:
        if screen._notes_state.view != "lasting_add":
            return False
        canvases = list(screen.query(LibraryNotesAddFromFilesCanvas))
        return bool(canvases) and (
            canvases[0].snapshot.phase == controller.snapshot.phase
        )

    await _wait_for_condition(
        pilot,
        _canvas_settled,
        message="Add from files… never mounted the chooser",
    )
    await pilot.pause()


async def test_keep_a_folder_synced_is_offered_without_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "tldw_chatbook_notes_sync_state.db"
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: state_path,
    )
    scratch = (tmp_path / "scratch-folder").resolve()
    scratch.mkdir()
    (scratch / "first.md").write_text("# First\n\nbody\n", encoding="utf-8")

    notes_path = tmp_path / "notes.sqlite3"
    database = CharactersRAGDB(notes_path, client_id="task-34000-71")
    interop = NotesInteropService(
        base_db_directory=notes_path.parent,
        api_client_id="task-34000-71",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(database),
    )

    app = _build_test_app(configured_default="library")
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    _seed_conversations(app, [])
    # The real owner is built by the app's own lazy property from the slots
    # its constructor captured; give those slots a real Notes authority so the
    # review's force-start (legacy migration reads the notes connection) and
    # its reconcile have something real to read.
    app.chachanotes_db = database
    app.notes_scope_service = scope_service
    app._notes_sync_scope_service = scope_service

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await _wait_for_library_screen(app, pilot)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
            assert not state_path.exists(), "Library arrival created the state DB"

            await _open_chooser(screen, pilot)
            controller = screen._library_notes_sync_controller
            keep_label = str(
                screen.query_one("#notes-add-keep-synced", Button).label
            )
            painted = _painted_text(app)
            assert keep_label == "Keep a folder synced", (
                f"Keep a folder synced is inert on a no-root profile: "
                f"{keep_label!r} (runtime status "
                f"{app.notes_sync_runtime_owner.snapshot().status!r})\n{painted}"
            )
            assert controller.snapshot.lasting_available is True
            # AC#3: the chooser alone opens no state store.
            assert not state_path.exists(), "opening the chooser created the state DB"

            # AC#1: choosing it reaches the configure form…
            screen.query_one("#notes-add-keep-synced", Button).press()
            await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
            assert controller.snapshot.phase == "configure", (
                controller.snapshot.status_line
            )
            assert not state_path.exists(), "the configure form created the state DB"

            # …and Check on a scratch folder reaches the review, which is
            # where the runtime is force-started and the store first exists.
            controller.set_setup("display_name", "Scratch")
            controller.set_setup("folder", str(scratch))
            await pilot.pause()
            screen.query_one("#notes-sync-check", Button).press()
            await _wait_for_selector(screen, pilot, "#notes-sync-activate")
            assert controller.snapshot.phase == "review", (
                controller.snapshot.status_line
            )
            assert state_path.exists(), "the review did not create the state DB"
            assert app.notes_sync_runtime_owner.snapshot().status == "active"
        # AC#6: the lazily scheduled start was tracked and settled; nothing
        # was left pending for the loop to complain about at exit.
        start_task = app._notes_sync_runtime_start_task
        assert start_task is not None, "no start task was recorded for the lazy start"
        assert start_task.done() and not start_task.cancelled()
        assert start_task.exception() is None
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)
        interop.close_all_user_connections()
        database.close_connection()
    assert unhandled == []
