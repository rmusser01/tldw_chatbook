"""Notes sync skips a non-UTF-8 or mixed-newline file by name (TASK-34000.16).

Review N-16 (``Docs/QA/notes-library-ux-review-2026-10-02/verify/nl-v-n-16``):
one latin-1 file (byte 0xE9) or one file mixing CRLF and LF stopped the WHOLE
folder -- the setup pane read ``Check failed — NotesSyncRootRefused`` and
nothing else in the folder synced until the file was gone. Each refused file
is now one skipped row in the review that names it and says what to do; the
rest of the folder is planned and activates.

One boot at 160x45 against the real ``LibraryScreen`` with the production
stylesheet. Every claim is what the user sees (painted text, Button state) or
what reached the database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackRecoveryResult,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)
LATIN1_ROW = "latin1.md · Not UTF-8 text — convert it to UTF-8 or move it out of the folder"
MIXED_ROW = "mixed.md · Mixed line endings — save it with one style"


@pytest.fixture(autouse=True)
def _isolate_actor_pack_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated startup recovery off this test's database."""

    monkeypatch.setattr(
        "tldw_chatbook.app.first_profile_created_this_session", lambda: False
    )
    monkeypatch.setattr(
        PersonaActorPackCoordinator,
        "recover",
        lambda _self: PersonaActorPackRecoveryResult(0, 0, 0, ()),
    )


def _painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def write_review_fixture(vault: Path) -> None:
    """The review's two fixtures beside three good files.

    ``latin1.md`` holds byte 0xE9 (not UTF-8); ``mixed.md`` mixes CRLF and LF.
    """

    for index in ("1", "2", "3"):
        (vault / f"ok-{index}.md").write_text(
            f"# Ok {index}\n\nbody {index}\n", encoding="utf-8"
        )
    (vault / "latin1.md").write_bytes(b"caf\xe9\n")
    (vault / "mixed.md").write_bytes(b"alpha\r\nbeta\ngamma\n")


def build_sync_stack(tmp_path: Path):
    """A real notes database, scope service and runtime owner for one test."""

    notes_path = tmp_path / "notes.sqlite3"
    database = CharactersRAGDB(notes_path, client_id="task-34000-16")
    folders = LocalNoteFolderRepository(database)
    interop = NotesInteropService(
        base_db_directory=notes_path.parent,
        api_client_id="task-34000-16",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=folders,
    )
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    return database, interop, scope_service, owner


async def open_keep_synced_review(screen, pilot, host, vault: Path):
    """Notes → Add from files… → Keep a folder synced → Check changes."""

    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
    screen.query_one("#library-notes-add-from-files", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
    screen.query_one("#notes-add-keep-synced", Button).press()
    await _wait_for_selector(screen, pilot, "#notes-sync-display-name")
    controller = screen._library_notes_sync_controller
    controller.set_setup("display_name", "Vault sync")
    controller.set_setup("folder", str(vault))
    await pilot.pause()
    screen.query_one("#notes-sync-check", Button).press()
    return controller


async def test_a_folder_with_two_bad_files_reviews_two_named_skips_and_activates_the_rest(
    tmp_path: Path,
) -> None:
    """AC#1, AC#2 and AC#6's twin: two named skip rows, three creates, Activate works.

    Base (7296c536f2): the whole root is refused -- no review, the setup
    pane's status line reads TASK-32243's refusal copy ("Some files there are
    not UTF-8 text. Convert or move them out of the folder, then Check
    again.") and Activate never enables; the ``Skipped (2)`` wait fails naming
    that line. (The ``NotesSyncRootRefused`` class name of the review's
    capture 03 is the ROOT ROW's fallback under Manage sync folders; the
    ``_extended`` sibling's active-root arm covers that surface.)
    """

    vault = (tmp_path / "vault").resolve()
    vault.mkdir()
    write_review_fixture(vault)
    database, interop, scope_service, owner = build_sync_stack(tmp_path)
    await owner.start()
    app = _build_test_app()
    _seed_conversations(app, [])
    app.notes_scope_service = scope_service
    app.notes_sync_runtime_owner = owner
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            controller = await open_keep_synced_review(screen, pilot, host, vault)

            await _wait_for_condition(
                pilot,
                lambda: "Skipped (2)" in _painted(host),
                message=lambda: (
                    "review never painted 'Skipped (2)'; status line: "
                    f"{controller.snapshot.status_line!r}"
                ),
            )
            painted = _painted(host)
            assert LATIN1_ROW in painted, painted
            assert MIXED_ROW in painted, painted
            assert "NotesSyncRootRefused" not in painted
            assert "Check failed" not in controller.snapshot.status_line

            activate = screen.query_one("#notes-sync-activate", Button)
            await _wait_for_condition(
                pilot,
                lambda: not activate.disabled,
                message=lambda: (
                    "review never admitted activation: "
                    f"{controller.snapshot.status_line!r}"
                ),
            )
            activate.press()
            await _wait_for_selector(screen, pilot, "#notes-sync-receipt")
            await _wait_for_condition(
                pilot,
                lambda: "3 applied" in _painted(host),
                message="receipt never painted '3 applied'",
            )
            assert database.count_notes() == 3
            # The refused files were skipped, never written over: still the
            # review's bytes on disk.
            assert (vault / "latin1.md").read_bytes() == b"caf\xe9\n"
            assert (vault / "mixed.md").read_bytes() == b"alpha\r\nbeta\ngamma\n"
    finally:
        await owner.shutdown()
        interop.close_all_user_connections()
        database.close_connection()
