"""Library ▸ Notes Trash view — the second safety net behind the receipt.

Group `i-trash` of the Notes riders wave: task-32144. The delete receipt is
a good immediate affordance and a bad only one; a "Recently deleted (N)" row
under the folder tree opens a list of soft-deleted notes whose Restore goes
through the SAME seam Undo does (``_undo_library_note_delete`` ->
``NotesScopeService.restore_note``). No permanent delete anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import MethodType, SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_notes_folder_navigator import (
    _branch_screen_fake,
    _folder_page,
    _placement_page,
)
from Tests.UI.test_library_notes_wave_list import (
    _CanvasApp,
    _RestoreService,
    _list_state,
    _passthrough_service_call,
)
from tldw_chatbook.Library.library_notes_state import (
    LibraryNotesTrashRow,
    LibraryNotesTrashState,
    build_library_notes_trash_state,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Notes.note_folder_models import FolderPlacementId
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _trash(*rows: LibraryNotesTrashRow, total: int | None = None) -> LibraryNotesTrashState:
    return LibraryNotesTrashState(
        rows=tuple(rows), total=len(rows) if total is None else total
    )


def _row(note_id: str, title: str, age: str = "2h") -> LibraryNotesTrashRow:
    return LibraryNotesTrashRow(
        note_id=note_id, title=title, version=2, age_label=age
    )


# -- the query projection -------------------------------------------------


def test_trash_state_orders_deleted_notes_newest_first_with_an_age() -> None:
    """The repository's page becomes rows carrying title, age and version."""
    state = build_library_notes_trash_state(
        [
            {
                "id": "n2",
                "title": "Groceries",
                "version": 4,
                "last_modified": (NOW - timedelta(minutes=5)).isoformat(),
            },
            {
                "id": "n1",
                "title": "",
                "version": 2,
                "last_modified": (NOW - timedelta(days=2)).isoformat(),
            },
        ],
        total=7,
        now=NOW,
    )

    assert [row.note_id for row in state.rows] == ["n2", "n1"]
    assert state.rows[0].title == "Groceries"
    assert state.rows[0].age_label == "5m"
    # A blank title still names something pressable.
    assert state.rows[1].title == "Untitled"
    # The tombstone's own version is what restore needs handed back.
    assert state.rows[1].version == 2
    assert state.total == 7


def test_trash_state_drops_records_without_an_id() -> None:
    """Degrade-don't-crash, matching the list builder beside it."""
    state = build_library_notes_trash_state(
        [{"title": "no id"}, {"id": "n1", "title": "keeps", "version": 1}],
        now=NOW,
    )

    assert [row.note_id for row in state.rows] == ["n1"]


# -- the row under the folder tree ---------------------------------------


@pytest.mark.asyncio
async def test_recently_deleted_row_names_the_count_under_the_tree() -> None:
    """task-32144: the safety net is visible without a live receipt."""
    app = _CanvasApp(
        pane_width=100,
        list_state=_list_state(),
        trash=_trash(_row("n1", "Groceries"), _row("n2", "Draft"), total=2),
    )
    async with app.run_test(size=(120, 40)):
        opener = app.query_one("#library-notes-trash-open", Button)
        assert "Recently deleted (2)" in str(opener.label)


@pytest.mark.asyncio
async def test_recently_deleted_row_is_absent_when_nothing_is_deleted() -> None:
    """No tombstones, no row -- the tree does not grow a dead affordance."""
    app = _CanvasApp(pane_width=100, list_state=_list_state(), trash=_trash())
    async with app.run_test(size=(120, 40)):
        assert not app.query("#library-notes-trash-open")


# -- the Trash view itself ------------------------------------------------


@pytest.mark.asyncio
async def test_trash_view_lists_every_deleted_note_with_its_own_restore() -> None:
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=_trash(
            _row("n1", "Groceries", "5m"),
            _row("n2", "Draft", "2d"),
            total=2,
        ),
    )
    async with app.run_test(size=(120, 40)):
        labels = [
            LibraryNotesCanvas._static_text(widget)
            for widget in app.query(".library-notes-trash-row-copy").results(Static)
        ]
        assert labels == ["Groceries · 5m", "Draft · 2d"]
        restores = list(app.query(".library-notes-trash-restore").results(Button))
        assert len(restores) == 2
        assert [button.note_id for button in restores] == ["n1", "n2"]
        assert [button.note_version for button in restores] == [2, 2]
        assert app.query_one("#library-notes-trash-back", Button)


@pytest.mark.asyncio
async def test_trash_view_offers_no_permanent_delete() -> None:
    """AC: recovery only -- Danger stays out of this view."""
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=_trash(_row("n1", "Groceries")),
    )
    async with app.run_test(size=(120, 40)):
        labels = " ".join(
            str(button.label) for button in app.query(Button).results(Button)
        ).lower()
        assert "delete" not in labels
        assert "forever" not in labels
        assert not app.query(".library-canvas-danger")


@pytest.mark.asyncio
async def test_empty_trash_view_says_so_and_offers_the_way_back() -> None:
    app = _CanvasApp(pane_width=100, mode="trash", trash=_trash())
    async with app.run_test(size=(120, 40)):
        empty = app.query_one("#library-notes-trash-empty", Static)
        assert "Nothing deleted" in LibraryNotesCanvas._static_text(empty)
        assert app.query_one("#library-notes-trash-back", Button)


# -- Restore rides the Undo seam -----------------------------------------


def _restore_fake(monkeypatch, service: _RestoreService):
    fake = _branch_screen_fake(service)
    fake._notes_state.delete_receipt = None
    fake._library_note_delete_receipt = None
    fake._library_notes_mutation_in_flight = True
    fake._notes_state.view = "trash"
    fake._notes_state.mutation_in_flight = False
    fake._notes_state.notice = ""
    fake._notes_state.tree_pending_target_placement_id = ""
    fake._notes_state.trash = _trash(_row("n1", "n1"))
    fake._local_source_records = {"notes": ()}
    fake._local_source_counts = {"notes": 0}
    fake._selected_note_id = ""
    fake._restore_library_notes_focus_identity = lambda *_a, **_k: None
    fake._focus_library_note_control = lambda *_a, **_k: None
    fake._library_notes_restore_guard_is_current = lambda *_a, **_k: True
    fake._source_record_id = LibraryScreen._source_record_id
    fake._append_library_note_source_record = MethodType(
        LibraryNotesController._append_library_note_source_record, fake
    )
    fake._run_library_service_call = _passthrough_service_call
    fake._locate_library_notes_tree_target = MethodType(
        LibraryScreen._locate_library_notes_tree_target, fake
    )
    fake._reconcile_library_notes_tree_mutation = MethodType(
        LibraryScreen._reconcile_library_notes_tree_mutation, fake
    )
    fake._load_library_notes_tree_slice = MethodType(
        LibraryScreen._load_library_notes_tree_slice, fake
    )
    fake._build_library_notes_tree_projection = MethodType(
        LibraryScreen._build_library_notes_tree_projection, fake
    )
    fake._refresh_library_notes_trash = lambda: None
    fake._undo_library_note_delete = MethodType(
        LibraryNotesController._undo_library_note_delete, fake
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller._sync_library_canvas",
        lambda *_a, **kwargs: (
            kwargs["then"]() if kwargs.get("then") is not None else None
        ),
    )
    return fake


@pytest.mark.asyncio
async def test_trash_restore_returns_the_row_and_the_count_exactly_as_undo(
    monkeypatch,
) -> None:
    """task-32144 AC#2: one seam, so the tree and the rail count both move."""
    service = _RestoreService(None)
    fake = _restore_fake(monkeypatch, service)
    scheduled = []
    fake.run_worker = lambda coroutine, **_kwargs: scheduled.append(coroutine)
    fake._library_notes_mutation_fenced = lambda: False

    for key in (
        NotesBranchKey(None, "folders"),
        NotesBranchKey(None, "placements"),
    ):
        await LibraryScreen._load_library_notes_tree_slice(
            fake, key, direction="replace", offset=0
        )
    before = LibraryScreen._build_library_notes_tree_projection(fake)
    assert all(row.note_id != "n1" for row in before.rows)

    button = Button("Restore", classes="library-notes-trash-restore")
    button.note_id = "n1"
    button.note_title = "n1"
    button.note_version = 2
    LibraryNotesController.handle_library_notes_trash_restore(
        fake, SimpleNamespace(button=button, stop=lambda: None)
    )
    assert scheduled, "Restore never scheduled the shared undo worker"
    await scheduled[0]

    after = LibraryScreen._build_library_notes_tree_projection(fake)
    restored = [row for row in after.rows if row.note_id == "n1"]
    assert restored, "the restored note never came back to the tree"
    assert restored[0].placement_id == FolderPlacementId.unfiled("n1")
    # The same count patch Undo makes.
    assert fake._local_source_counts["notes"] == 1
    assert service.restored is True


def test_trash_restore_refuses_while_a_notes_mutation_is_in_flight(
    monkeypatch,
) -> None:
    """The receipt's Undo fence covers this entry point too."""
    fake = _restore_fake(monkeypatch, _RestoreService(None))
    scheduled = []
    fake.run_worker = lambda coroutine, **_kwargs: scheduled.append(coroutine)
    fake._library_notes_mutation_fenced = lambda: True

    button = Button("Restore", classes="library-notes-trash-restore")
    button.note_id = "n1"
    button.note_title = "n1"
    button.note_version = 2
    LibraryNotesController.handle_library_notes_trash_restore(
        fake, SimpleNamespace(button=button, stop=lambda: None)
    )

    assert scheduled == []


# -- entering and leaving -------------------------------------------------


def _view_fake(monkeypatch, view: str = "list"):
    # A flat fake, the way this cluster's other controller tests build one:
    # the generated `_library_notes_<field>` shims are properties on the real
    # controller, so a SimpleNamespace carries the shim NAME directly.
    fake = SimpleNamespace(
        _library_notes_view=view,
        _library_notes_trash=_trash(_row("n1", "Groceries")),
        _notes_state=SimpleNamespace(
            view=view,
            trash=_trash(_row("n1", "Groceries")),
        ),
        is_mounted=True,
        _synced=[],
        _focus_library_note_control=lambda *_a, **_k: None,
        _apply_library_notes_footer_context=lambda: None,
    )
    fake._leave_library_notes_trash = MethodType(
        LibraryNotesController._leave_library_notes_trash, fake
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller._sync_library_canvas",
        lambda *args, **kwargs: fake._synced.append(kwargs.get("then")),
    )
    return fake


def test_opening_the_trash_switches_the_view_and_reloads_it(monkeypatch) -> None:
    fake = _view_fake(monkeypatch)
    reloads = []
    fake._refresh_library_notes_trash = lambda: reloads.append(True)
    fake._library_notes_mutation_fenced = lambda: False

    LibraryNotesController.handle_library_notes_trash_open(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert fake._library_notes_view == "trash"
    assert reloads == [True]
    assert fake._synced


def test_leaving_the_trash_returns_to_the_notes_list(monkeypatch) -> None:
    fake = _view_fake(monkeypatch, view="trash")

    LibraryNotesController.handle_library_notes_trash_back(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert fake._library_notes_view == "list"
    assert fake._synced


@pytest.mark.asyncio
async def test_escape_from_the_trash_view_returns_to_the_notes_list(
    monkeypatch,
) -> None:
    """AC: Escape returns to the list, not to the rail."""
    fake = _view_fake(monkeypatch, view="trash")
    fake._library_note_session = SimpleNamespace(
        snapshot=None,
        conflict_resolution_running=False,
        destructive_admission=None,
    )
    fake._library_note_context = False
    fake._library_notes_select_mode = False
    fake._library_note_confirming_delete = False
    fake._library_notes_conflict_locked = lambda: False

    await LibraryNotesController.action_library_notes_escape(fake)

    assert fake._library_notes_view == "list"


def test_trash_footer_advertises_restore_and_the_way_back() -> None:
    """An honest footer: only keys this view actually answers."""
    fake = SimpleNamespace(
        _notes_state=SimpleNamespace(
            view="trash",
            compact=False,
            stage="notes",
            select_mode=False,
            sort_choices_visible=False,
            confirming_delete=False,
            trash=_trash(_row("n1", "Groceries")),
        ),
        _library_notes_workflow_active=lambda: True,
        _library_note_session=SimpleNamespace(
            snapshot=None, conflict_resolution_running=False
        ),
        _library_notes_focus_region=lambda: "trash",
        _notes_footer_tier=lambda wide, _compact: wide,
    )

    shortcuts = LibraryScreen._library_notes_footer_shortcuts(fake)

    assert shortcuts == (("r", "restore note"), ("esc", "back to notes"))


@pytest.mark.asyncio
async def test_r_restores_the_focused_trash_row(monkeypatch) -> None:
    """AC: the key stands for the focused row's own button, never a guess."""
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=_trash(_row("n1", "Groceries"), _row("n2", "Draft"), total=2),
    )
    async with app.run_test(size=(120, 40)):
        buttons = list(app.query(".library-notes-trash-restore").results(Button))
        pressed: list[str] = []
        monkeypatch.setattr(
            Button, "press", lambda self: pressed.append(self.note_id) or self
        )
        fake = SimpleNamespace(focused=buttons[1], query=app.query)
        LibraryScreen.action_library_notes_trash_restore(fake)
        assert pressed == ["n2"]

        # Focus elsewhere: the key does nothing rather than restoring a note
        # the reader never pointed at (PR #2553 review).
        pressed.clear()
        fake.focused = app.query_one("#library-notes-trash-back", Button)
        LibraryScreen.action_library_notes_trash_restore(fake)
        assert pressed == []


def test_a_trash_refresh_is_never_dropped_by_an_in_flight_read() -> None:
    """A delete's refresh must outlive a read that started before it.

    The read is superseded by the exclusive worker group, so a request made
    while one is in flight starts a NEW read instead of being discarded
    (PR #2553 review).
    """
    scheduled: list[tuple[object, dict]] = []
    fake = SimpleNamespace(
        run_worker=lambda coroutine, **kwargs: scheduled.append((coroutine, kwargs)),
        _load_library_notes_trash=LibraryNotesController._load_library_notes_trash,
    )
    fake._load_library_notes_trash = MethodType(
        LibraryNotesController._load_library_notes_trash, fake
    )

    LibraryNotesController._refresh_library_notes_trash(fake)
    LibraryNotesController._refresh_library_notes_trash(fake)

    try:
        assert len(scheduled) == 2
        assert all(
            kwargs.get("exclusive") and kwargs.get("group") == "library_notes_trash"
            for _coroutine, kwargs in scheduled
        )
    finally:
        for coroutine, _kwargs in scheduled:
            coroutine.close()
