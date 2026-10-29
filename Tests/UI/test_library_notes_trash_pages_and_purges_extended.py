"""Notes ▸ Recently deleted ▸ Delete forever, through the real Library.

TASK-34000.29 (N-29), part 8b. The owner was asked "Delete forever in Notes:
yes or no" on 2026-10-29 and answered "yes with confimration". The confirm is
ADR-055 Pattern B: it names the note, says "This cannot be undone.", opens
on Cancel, and Escape is a Cancel that writes nothing (ruling T8b-a).

The 8b arms of ``test_library_notes_trash_pages_and_purges.py`` (ruling
T8b-f): that file was already ~33 s alone under load with its three 8a
boots, so these two boots live here.

- 160x45, 25 tombstones plus one rich one (keywords, a folder, a link, a
  title Rich markup would eat): ``x`` on its row opens the confirm, Escape
  leaves everything, ``x`` + Delete forever removes it -- no row anywhere in
  the database names it afterwards (the schema-discovered census, through a
  second connection), search cannot find it, focus is on the row that took
  its place (ruling T8b-e). Then a note bound to a sync folder AFTER the list
  painted is refused at confirm time with the guide's sentence (ruling
  T8b-c), and ``x`` on a live note row in the list view does nothing (T8b-d).
- 100x30, 25 tombstones: the parked R4-2 order. One canvas sync injected
  right after each landing -- a purge of the 3rd row (the ruling's arm), a
  Restore of the last row below the fold (R4-2's own exposure) and a purge
  there -- leaves the landed row on screen with the exact range painted.

Every gesture is a real key; every claim is painted text, a focused control,
a geometry the user sees, or the database read through a second connection.
Base (71f8d8a487): no Delete forever exists -- ``x`` does nothing and the
first wait times out.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.ChaChaNotesDB.note_trace_census import note_traces
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_trash_pages_and_purges import (
    COMPACT,
    WIDE,
    _focused_id,
    _notes_tree_settled,
    _one,
    _painted,
    _settled_on,
    _TrashSeed,
    _until,
    _visible_in,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootRecord,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_models import NotesSyncSerializationProfile
from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner
from tldw_chatbook.RAG_Search import ingestion_indexing
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: A title Rich markup would eat or reject: the confirm and the notice must
#: show it literally (``markup=False``).
SPECIAL = "[Draft] plan · Q3"
#: Only the special note's body has it.
TOKEN = "plutonium"
REFUSAL = (
    "This note is kept in sync with a folder — remove it from the folder's "
    "sync first."
)
_DIGEST = "b" * 64


class _PurgeSeed:
    """The 8a seed (25 tombstones) plus one tombstone referenced everywhere."""

    def __init__(self, tmp_path: Path, *, special: bool) -> None:
        self.trash = _TrashSeed(tmp_path)
        self.path = self.trash.path
        self.ids = self.trash.ids
        self.special_id = ""
        self.state_path = tmp_path / "sync-state.sqlite3"
        self.vault = tmp_path / "vault"
        self.vault.mkdir()
        if not special:
            return
        db = self.trash.profile.db
        self.special_id = db.add_note(
            SPECIAL, f"{TOKEN} body that links [[Ideas inbox]]"
        )
        db.link_note_to_keyword(self.special_id, db.add_keyword("purge-check"))
        folders = LocalNoteFolderRepository(db)
        folder = folders.create_folder(name="Projects", parent_id=None)
        folders.attach_manual(folder_id=folder.folder_id, note_id=self.special_id)
        db.soft_delete_note(
            self.special_id, db.get_note_by_id(self.special_id)["version"]
        )
        with sqlite3.connect(self.path) as connection:
            # Third newest deletion: between "Deleted 02" and "Deleted 03".
            connection.execute(
                "UPDATE notes SET last_modified = ? WHERE id = ?",
                ("2026-10-27T09:57:30.000Z", self.special_id),
            )
            self.special_rowid = connection.execute(
                "SELECT rowid FROM notes WHERE id = ?", (self.special_id,)
            ).fetchone()[0]

    def traces(self, note_id: str, **kwargs) -> dict[str, int]:
        with sqlite3.connect(self.path) as connection:
            return note_traces(connection, note_id, **kwargs)

    def deleted(self) -> set[str]:
        return self.trash.deleted()

    def bind(self, note_id: str) -> None:
        """Bind one note to a sync folder, as lasting sync's own store records it."""
        store = NotesDeviceStateStore(self.state_path)
        try:
            store.initialize()
            store.create_root(
                NotesSyncRootRecord(
                    root_id="root-1",
                    note_scope_id="local_note",
                    logical_folder_id="folder-sync-1",
                    canonical_path=str(self.vault.resolve()),
                    direction=NotesSyncDirection.BIDIRECTIONAL,
                    state=NotesSyncRootState.ACTIVE,
                )
            )
            store.create_binding(
                NotesSyncBindingRecord(
                    binding_id="binding-1",
                    root_id="root-1",
                    note_scope_id="local_note",
                    note_id=note_id,
                    normalized_relative_path="deleted-05.md",
                    stable_identity_digest=_DIGEST,
                    state=NotesSyncBindingState.ACTIVE,
                    serialization=NotesSyncSerializationProfile(
                        utf8_bom=False, newline="lf", final_newline=True, mode=0o644
                    ),
                    content_digest=_DIGEST,
                    note_version=2,
                )
            )
        finally:
            store.close()


def _mounted(tmp_path: Path, *, special: bool):
    app = _build_test_app()
    _seed_conversations(app, [])
    seed = _PurgeSeed(tmp_path, special=special)
    profile = seed.trash.profile
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    # The production runtime over a private store, so the purge's binding
    # check is the real one reading lasting sync's real records.
    runtime = build_notes_sync_runtime_owner(
        notes_scope_service=profile.scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=seed.state_path,
        migrate_legacy=lambda: None,
        local_user_id="default_user",
        recovery_capacity_bytes=1024 * 1024,
    )
    app.notes_sync_runtime_owner = runtime
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), seed, runtime


@pytest.fixture
def index_removals(monkeypatch) -> list[str]:
    """The semantic-index removals purges queue (no RAG runtime in a test)."""
    queued: list[str] = []
    monkeypatch.setattr(
        ingestion_indexing,
        "submit_note_index_removal",
        lambda note_id, *, source_path=None: queued.append(note_id) or True,
        # The base has no such hook; its RED is the missing confirm instead.
        raising=False,
    )
    return queued


def _title(screen) -> str:
    return str(getattr(screen.focused, "note_title", "") or "")


async def _open_trash(host, pilot, count: int) -> LibraryScreen:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _notes_tree_settled(host, pilot, screen, count=count)
    await pilot.click("#library-notes-trash-open")
    await _settled_on(
        pilot,
        host,
        screen,
        lambda: _focused_id(screen) == "library-notes-trash-restore-0",
        lambda: f"the Trash's first row (focus: {_focused_id(screen)!r})",
    )
    return screen


async def _focus_row(host, pilot, screen, title: str, tabs: int) -> None:
    await pilot.press(*(["tab"] * tabs))
    await _settled_on(
        pilot,
        host,
        screen,
        lambda: _title(screen) == title,
        lambda: f"focus on {title!r} (focus: {_focused_id(screen)!r})",
    )


async def _confirm_open(host, pilot) -> ConfirmationDialog:
    await _until(
        pilot,
        lambda: isinstance(host.screen, ConfirmationDialog)
        and getattr(host.screen.focused, "id", "") == "cancel-button"
        and "This cannot be undone." in _painted(host),
        lambda: f"the confirm on Cancel: {_painted(host)}",
    )
    return host.screen


async def _delete_forever(host, pilot) -> None:
    """``x`` on the focused row, then Delete forever in the confirm."""
    await pilot.press("x")
    await _confirm_open(host, pilot)
    await pilot.press("tab")
    await _until(
        pilot,
        lambda: getattr(host.screen.focused, "id", "") == "confirm-button",
        "Delete forever to take focus",
    )
    await pilot.press("enter")


def _landing_then_sync(screen, role: str) -> list[bool]:
    """Fire ONE default Notes sync right after the ``role`` landing returns.

    Task 8 round 4's deterministic window (R3-1): after the landing queued
    its keep-in-view and before the next refresh.
    """
    controller = screen._notes_controller
    real_land = controller._land_library_notes_trash_focus
    fired: list[bool] = []

    def land_then_sync(landing_role: str) -> None:
        real_land(landing_role)
        if not fired and landing_role == role:
            fired.append(True)
            _sync_library_canvas(screen, "notes")

    controller._land_library_notes_trash_focus = land_then_sync
    return fired


async def test_delete_forever_names_the_note_and_leaves_no_trace(
    tmp_path, index_removals
) -> None:
    """AC#9, rulings T8b-a..e, end to end at 160x45."""
    host, seed, runtime = _mounted(tmp_path, special=True)
    try:
        async with host.run_test(size=WIDE) as pilot:
            screen = await _open_trash(host, pilot, count=26)
            assert "1–20 of 26 · Page 1 of 2" in _painted(host)
            await _focus_row(host, pilot, screen, SPECIAL, tabs=2)
            assert "x delete forever" in _painted(host)
            before = seed.traces(seed.special_id, rowid=seed.special_rowid)
            assert {"notes.id", "note_keywords.note_id", "sync_log.entity_id"} <= set(
                before
            ), before

            # The confirm names the note -- literally -- and opens on Cancel.
            await pilot.press("x")
            dialog = await _confirm_open(host, pilot)
            assert f"“{SPECIAL}” will be deleted for good." in _painted(host)
            # Escape is a Cancel: nothing written, the row and focus stay.
            await pilot.press("escape")
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: host.screen is screen and _title(screen) == SPECIAL,
                lambda: f"back on the row after Escape (focus: {_focused_id(screen)!r})",
            )
            assert dialog.result is False
            assert seed.traces(seed.special_id, rowid=seed.special_rowid) == before
            assert index_removals == []

            # Delete forever.
            await _delete_forever(host, pilot)
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: "1–20 of 25 · Page 1 of 2" in _painted(host)
                and f"Deleted forever · {SPECIAL}" in _painted(host)
                and _title(screen) == "Deleted 03"
                and _focused_id(screen) == "library-notes-trash-restore-2",
                lambda: f"the purge landed (focus: {_focused_id(screen)!r}): "
                f"{_painted(host)}",
            )
            listing = screen.query_one("#library-notes-trash-list")
            assert SPECIAL not in "\n".join(
                line for line in _painted(host).splitlines() if "Deleted forever" not in line
            )
            assert _visible_in(listing, screen.focused)
            assert (
                seed.traces(
                    seed.special_id, rowid=seed.special_rowid, tokens=(TOKEN,)
                )
                == {}
            )
            assert await seed.trash.profile.scope_service.search_notes(
                scope="local_note", query=TOKEN, user_id="default_user"
            ) in ([], ())
            assert index_removals == [seed.special_id]

            # Ruling T8b-c: bound AFTER the list painted, refused at confirm.
            bound_id = seed.ids["Deleted 05"]
            seed.bind(bound_id)
            await _focus_row(host, pilot, screen, "Deleted 05", tabs=2)
            await _delete_forever(host, pilot)
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: host.screen is screen and REFUSAL in _painted(host),
                lambda: f"the refusal: {_painted(host)}",
            )
            assert "Deleted 05" in _painted(host)
            assert "1–20 of 25 · Page 1 of 2" in _painted(host)
            assert "Deleted 05" in seed.deleted()
            assert "notes.id" in seed.traces(bound_id)
            assert index_removals == [seed.special_id]

            # Ruling T8b-d: in the list view x is nobody's key.
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: "Recently deleted (25)" in _painted(host),
                lambda: f"the list after Escape: {_painted(host)}",
            )
            row = next(
                row
                for row in screen.query(".library-notes-tree-note-row").results(Button)
                if "Ideas inbox" in str(row.label)
            )
            screen.set_focus(row)
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: screen.focused is row,
                "a live note row to take focus",
            )
            assert not screen.check_action("library_notes_trash_delete", ())
            assert "delete forever" not in _painted(host)
            await pilot.press("x")
            await pilot.pause(0.2)
            assert host.screen is screen
            assert len(seed.deleted()) == 25
            with sqlite3.connect(seed.path) as connection:
                assert connection.execute(
                    "SELECT COUNT(*) FROM notes WHERE deleted = 0"
                ).fetchone()[0] == 2
    finally:
        await runtime.shutdown()


async def test_at_100x30_a_same_page_landing_survives_a_sync_right_after_it(
    tmp_path, index_removals
) -> None:
    """Parked R4-2 (ruling T8b-e): every same-page landing settles first.

    Each arm injects ONE default Notes sync right after its landing returns.
    The purge of the 3rd row is the ruling's arm; the Restore of the last
    row is R4-2's own exposure -- its landing comes from the Restore's own
    canvas sync, not a page read, so nothing withheld the list's offset and
    the injected sync put the new list's pre-layout offset (0) back over the
    landed row below the fold. Then a purge there.
    """
    host, seed, runtime = _mounted(tmp_path, special=False)
    try:
        async with host.run_test(size=COMPACT) as pilot:
            screen = await _open_trash(host, pilot, count=25)
            await _focus_row(host, pilot, screen, "Deleted 03", tabs=2)

            fired = _landing_then_sync(screen, "trash-restore:2")
            await _delete_forever(host, pilot)
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: bool(fired)
                and "1–20 of 24 · Page 1 of 2" in _painted(host)
                and _title(screen) == "Deleted 04",
                lambda: f"the 3rd row purged (focus: {_focused_id(screen)!r})",
            )
            listing = screen.query_one("#library-notes-trash-list")
            assert _visible_in(listing, screen.focused), (
                listing.scroll_y,
                tuple(listing.region),
                tuple(screen.focused.region),
            )
            assert "Deleted 03" not in seed.deleted()

            # R4-2: Restore the last row, below the fold.
            await _focus_row(host, pilot, screen, "Deleted 21", tabs=17)
            assert _one(screen, "#library-notes-trash-list").scroll_y > 0
            fired = _landing_then_sync(screen, "trash-restore:19")
            await pilot.press("r")
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: bool(fired)
                and "Deleted 21" not in seed.deleted()
                and "1–20 of 23 · Page 1 of 2" in _painted(host)
                and _title(screen) == "Deleted 22",
                lambda: f"the last row restored (focus: {_focused_id(screen)!r})",
            )
            listing = screen.query_one("#library-notes-trash-list")
            assert _visible_in(listing, screen.focused), (
                listing.scroll_y,
                tuple(listing.region),
                tuple(screen.focused.region),
            )

            # And a purge there, below the fold.
            fired = _landing_then_sync(screen, "trash-restore:19")
            await _delete_forever(host, pilot)
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: bool(fired)
                and "1–20 of 22 · Page 1 of 2" in _painted(host)
                and _title(screen) == "Deleted 23",
                lambda: f"the last row purged (focus: {_focused_id(screen)!r})",
            )
            listing = screen.query_one("#library-notes-trash-list")
            assert _visible_in(listing, screen.focused), (
                listing.scroll_y,
                tuple(listing.region),
                tuple(screen.focused.region),
            )
            assert seed.traces(seed.ids["Deleted 22"]) == {}
            assert index_removals == [seed.ids["Deleted 03"], seed.ids["Deleted 22"]]
    finally:
        await runtime.shutdown()
