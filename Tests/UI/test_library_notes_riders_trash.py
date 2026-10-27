"""Library ▸ Notes Trash view — the second safety net behind the receipt.

Group `i-trash` of the Notes riders wave: task-32144. The delete receipt is
a good immediate affordance and a bad only one; a "Recently deleted (N)" row
under the folder tree opens a list of soft-deleted notes whose Restore goes
through the SAME seam Undo does (``_undo_library_note_delete`` ->
``NotesScopeService.restore_note``). No permanent delete anywhere.
"""

from __future__ import annotations

import dataclasses
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
    LibraryNotesFocusIdentity,
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


# -- TASK-34000.29 (N-29, 8a): every deleted note is one page away ----------


def _deleted_records(start: int, stop: int) -> list[dict]:
    """Repository rows for deleted notes ``start``..``stop - 1`` (1-based)."""
    return [
        {"id": f"d{index}", "title": f"Deleted {index:02d}", "version": 2}
        for index in range(start, stop)
    ]


def _painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def _until_painted(pilot, app, fragment: str) -> str:
    for _ in range(200):
        text = _painted(app)
        if fragment in text:
            return text
        await pilot.pause(0.01)
    raise AssertionError(f"{fragment!r} never painted:\n{_painted(app)}")


@pytest.mark.asyncio
async def test_trash_pager_names_the_exact_range_and_page_on_the_first_page() -> None:
    """37 deleted: the first page says so, and Next is the way on.

    Base (734e632f07): ``TypeError`` -- the builder has no ``page``; the view
    painted "Showing the 20 most recently deleted of 37. Restore one to see
    the rest." and no way to reach the other 17.
    """
    trash = build_library_notes_trash_state(
        _deleted_records(1, 21), total=37, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, "1–20 of 37 · Page 1 of 2")
        assert "Restore one to see the rest" not in text
        previous = app.query_one("#library-notes-trash-previous", Button)
        following = app.query_one("#library-notes-trash-next", Button)
        assert previous.disabled and "Previous" in str(previous.label)
        assert not following.disabled and str(following.label) == "Next"


@pytest.mark.asyncio
async def test_trash_pager_on_the_last_page_disables_next() -> None:
    trash = build_library_notes_trash_state(
        _deleted_records(21, 38), total=37, page=2, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, "21–37 of 37 · Page 2 of 2")
        assert "Deleted 21" in text and "Deleted 37" in text
        assert not app.query_one("#library-notes-trash-previous", Button).disabled
        assert app.query_one("#library-notes-trash-next", Button).disabled


@pytest.mark.asyncio
async def test_a_trash_that_fits_one_page_hides_the_pager_controls() -> None:
    """``library_pager_layout``'s one-page rule: the range stays, the rest goes."""
    trash = build_library_notes_trash_state(
        _deleted_records(1, 13), total=12, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, "1–12 of 12")
        assert "Page 1 of 1" not in text
        assert not app.query("#library-notes-trash-previous")
        assert not app.query("#library-notes-trash-next")


def test_trash_state_clamps_a_page_past_the_end_to_the_last_page() -> None:
    """A restore that empties the last page must not leave it numbered."""
    state = build_library_notes_trash_state(
        _deleted_records(1, 21), total=20, page=2, now=NOW
    )

    assert state.page == 1
    assert state.pager.range_copy == "1–20 of 20"
    assert state.pager.single_page


# -- ruling T8-d: where focus goes after a page or a Restore ---------------


def _focus_host(app) -> SimpleNamespace:
    """A controller-shaped receiver over the mounted canvas's own widgets."""
    fake = SimpleNamespace(query=app.query, query_one=app.query_one)
    fake._library_notes_trash_focus_target = MethodType(
        LibraryNotesController._library_notes_trash_focus_target, fake
    )
    return fake


def _role(app, selector: str) -> str:
    receiver = SimpleNamespace(
        _library_landing_focus_control_id=lambda _widget: "",
        _file_notes_active=lambda: False,
    )
    return LibraryScreen._library_notes_semantic_role(
        receiver, app.query_one(selector, Button)
    )


def _target(app, role: str):
    identity = LibraryNotesFocusIdentity(
        stage="notes", region="trash", note_id=None, semantic_role=role
    )
    return LibraryNotesController._library_notes_role_target(_focus_host(app), identity)


@pytest.mark.asyncio
async def test_the_trash_controls_carry_portable_focus_roles_both_halves() -> None:
    """A control with no role loses focus at the next canvas sync.

    Library Notes replays a captured focus by ROLE after every canvas sync
    (lessons-textual, "fix the ROLE, not the timing"); Previous, Next, Back
    and each Restore get one, and each role resolves back to its control.
    """
    trash = build_library_notes_trash_state(
        _deleted_records(1, 21), total=37, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "1–20 of 37")
        assert _role(app, "#library-notes-trash-next") == "trash-next"
        assert _role(app, "#library-notes-trash-previous") == "trash-previous"
        assert _role(app, "#library-notes-trash-back") == "trash-back"
        assert _role(app, "#library-notes-trash-restore-7") == "trash-restore:7"
        assert _target(app, "trash-next") is app.query_one(
            "#library-notes-trash-next", Button
        )
        assert _target(app, "trash-back") is app.query_one(
            "#library-notes-trash-back", Button
        )
        assert _target(app, "trash-restore:7") is app.query_one(
            "#library-notes-trash-restore-7", Button
        )


@pytest.mark.asyncio
async def test_a_disabled_pager_role_hands_focus_to_the_other_control() -> None:
    """Next on the last page is disabled: focus belongs on Previous."""
    trash = build_library_notes_trash_state(
        _deleted_records(21, 38), total=37, page=2, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "21–37 of 37")
        assert _target(app, "trash-next") is app.query_one(
            "#library-notes-trash-previous", Button
        )
        # A Restore index past the shortened list lands on its last row.
        assert _target(app, "trash-restore:19") is app.query_one(
            "#library-notes-trash-restore-16", Button
        )


@pytest.mark.asyncio
async def test_a_pager_role_with_no_pager_lands_on_the_first_row_then_back() -> None:
    trash = build_library_notes_trash_state(
        _deleted_records(1, 4), total=3, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "1–3 of 3")
        assert _target(app, "trash-previous") is app.query_one(
            "#library-notes-trash-restore-0", Button
        )
    empty = _CanvasApp(pane_width=100, mode="trash", trash=_trash())
    async with empty.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, empty, "Nothing deleted")
        assert _target(empty, "trash-next") is empty.query_one(
            "#library-notes-trash-back", Button
        )
        assert _target(empty, "trash-restore:0") is empty.query_one(
            "#library-notes-trash-back", Button
        )


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


class _PagedTrashService:
    """``list_deleted_notes`` over ``total`` tombstones, recording each offset."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.offsets: list[int] = []

    async def list_deleted_notes(self, *, scope, user_id, limit, offset):
        self.offsets.append(offset)
        return {
            "items": _deleted_records(offset + 1, min(offset + limit, self.total) + 1),
            "total": self.total,
        }


def _trash_loader_fake(service: _PagedTrashService, *, page: int = 1):
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(notes_scope_service=service),
        _run_library_service_call=_passthrough_service_call,
        _library_notes_user_id=lambda: "local",
        _library_notes_trash=None,
        _library_notes_trash_page=page,
        is_mounted=False,
    )
    fake._library_notes_trash_focus_target = lambda _role: None
    return fake


@pytest.mark.asyncio
async def test_a_trash_page_past_the_end_reads_the_last_page_instead() -> None:
    """Ruling T8-d, read-side clamp: page 3 of 37 shows page 2, never empty.

    The view keeps no stale total to clamp against, so the read itself does
    it: a page past the (shrunken) total re-reads the last page once.
    Base (734e632f07): ``TypeError`` -- the loader has no ``page`` and read
    ``limit=20`` from offset 0, every time.
    """
    service = _PagedTrashService(total=37)
    fake = _trash_loader_fake(service)

    await LibraryNotesController._load_library_notes_trash(fake, page=3)

    assert service.offsets == [40, 20]
    assert fake._library_notes_trash_page == 2
    trash = fake._library_notes_trash
    assert trash.page == 2 and trash.total == 37
    assert [row.title for row in trash.rows] == [
        f"Deleted {index:02d}" for index in range(21, 38)
    ]
    assert trash.pager.range_copy == "21–37 of 37"


@pytest.mark.asyncio
async def test_a_trash_reload_reads_the_page_the_view_is_on() -> None:
    """A Restore's (or any) reload re-reads the current page, not page 1."""
    service = _PagedTrashService(total=45)
    fake = _trash_loader_fake(service, page=2)

    await LibraryNotesController._load_library_notes_trash(fake)

    assert service.offsets == [20]
    assert fake._library_notes_trash.page == 2
    assert fake._library_notes_trash.pager.range_copy == "21–40 of 45"


def _pager_press_fake(*, fenced: bool):
    """A controller-shaped receiver on page 1 of 37 tombstones."""
    refreshes: list[dict] = []
    fake = SimpleNamespace(
        _library_notes_trash=build_library_notes_trash_state(
            _deleted_records(1, 21), total=37, page=1, now=NOW
        ),
        _library_notes_trash_page=1,
        _library_notes_mutation_fenced=lambda: fenced,
        _refresh_library_notes_trash=lambda **kwargs: refreshes.append(kwargs),
    )
    fake._page_library_notes_trash = MethodType(
        LibraryNotesController._page_library_notes_trash, fake
    )
    return fake, refreshes


def test_next_reads_the_next_page_without_taking_the_notes_interlock() -> None:
    """Paging is a read: no fence, one read, focus owed to the pressed role."""
    fake, refreshes = _pager_press_fake(fenced=False)

    LibraryNotesController.handle_library_notes_trash_next(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert fake._library_notes_trash_page == 2
    assert len(refreshes) == 1
    assert refreshes[0] == {"focus_role": "trash-next"}
    assert fake._trash_focus_role == "trash-next"
    assert not hasattr(fake, "_library_notes_mutation_in_flight")

    # Already on the last page: Next asks for nothing.
    fake._library_notes_trash_page = 2
    LibraryNotesController.handle_library_notes_trash_next(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert len(refreshes) == 1


def test_a_page_press_under_a_running_restore_is_queued_never_dropped() -> None:
    """Ruling T8-c: no page read starts while a Restore holds the interlock.

    The press moves the page and is owed to the interlock's release, which
    reads it exactly once -- the press is not lost and not doubled.
    """
    fake, refreshes = _pager_press_fake(fenced=True)

    LibraryNotesController.handle_library_notes_trash_next(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert refreshes == []
    assert fake._library_notes_trash_page == 2
    assert fake._trash_reload_owed == "trash-next"

    fake._library_notes_mutation_fenced = lambda: False
    fake._library_notes_mutation_in_flight = True
    fake._tree_reload_owed = False
    LibraryNotesController._release_library_notes_mutation(fake)
    LibraryNotesController._release_library_notes_mutation(fake)

    assert fake._library_notes_mutation_in_flight is False
    assert refreshes == [{"focus_role": "trash-next"}]
    assert fake._trash_reload_owed == ""


@pytest.mark.asyncio
async def test_a_trash_page_taller_than_its_pane_scrolls_to_the_focused_row() -> None:
    """Every row of a page is reachable, not only those above the fold.

    Found while checking the pager below 120x36: at 100x30 the work pane
    gives the trash list 15 rows for a 20-row page, and the list was a plain
    ``Vertical`` (``overflow: hidden``) -- rows 16-20 sat under the pager,
    focusable but never painted (the folder tree's TASK-34000.7 shape).
    Base (734e632f07): "Deleted 20" never paints with its Restore focused.
    """
    trash = build_library_notes_trash_state(
        _deleted_records(1, 21), total=37, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 18)) as pilot:
        await _until_painted(pilot, app, "Deleted 01")
        assert "Deleted 20" not in _painted(app)
        last = app.query_one("#library-notes-trash-restore-19", Button)
        last.focus()
        await _until_painted(pilot, app, "Deleted 20")
        listing = app.query_one("#library-notes-trash-list")
        assert listing.region.contains_region(last.region)
        # The pager is the list's last row: the next Tab stop scrolls it in.
        following = app.query_one("#library-notes-trash-next", Button)
        following.focus()
        await _until_painted(pilot, app, "1–20 of 37 · Page 1 of 2")
        assert listing.region.contains_region(following.region)


@pytest.mark.asyncio
async def test_the_pager_follows_a_short_page_instead_of_the_pane_foot() -> None:
    """No blank rows between the last deleted note and its pager.

    The list is ``1fr``; a pager composed after it sat at the foot of the
    pane, a dozen blank rows below a short page (live, 160x45) -- the
    detached-affordance shape task-28015 fixed in the Media Trash.
    """
    trash = build_library_notes_trash_state(
        _deleted_records(1, 4), total=3, page=1, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "1–3 of 3")
        last_row = list(app.query(".library-notes-trash-row"))[-1]
        pager = app.query_one("#library-notes-trash-pager")
        assert pager.region.y == last_row.region.bottom


def test_the_trash_landing_yields_to_the_users_own_input() -> None:
    """Real input wins: any key or click withdraws the owed Trash landing.

    The landing cannot be vetoed by the focus-intent generation: every reload
    prunes the focused control, and Textual's reset of focus onto a pane grip
    bumps that generation as well, which vetoed every landing (traced). The
    role is cleared instead from the screen's one real-input withdrawal.
    """
    landed: list[str] = []
    fake = SimpleNamespace(
        _library_notes_view="trash",
        _trash_focus_role="trash-restore:4",
        _restore_library_notes_focus_identity=lambda identity: landed.append(
            identity.semantic_role
        ),
    )

    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-restore:4")
    assert landed == ["trash-restore:4"]

    screen = SimpleNamespace(
        _library_notes_filter_row_handoff=None,
        _notes_state=SimpleNamespace(pending_focus_identity=None),
        _notes_controller=fake,
    )
    LibraryScreen._withdraw_library_notes_pending_focus_handoffs(screen)
    assert fake._trash_focus_role == ""

    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-restore:4")
    assert landed == ["trash-restore:4"]

    # Leaving the view withdraws it too.
    fake._trash_focus_role = "trash-next"
    fake._library_notes_view = "list"
    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-next")
    assert landed == ["trash-restore:4"]


@pytest.mark.asyncio
async def test_a_roleless_capture_in_the_trash_falls_back_to_the_owed_row() -> None:
    """A replay another sync replaced must not drop focus on the first row.

    Traced under load: a Restore's reload pruned the focused row, Textual
    parked focus on a pane grip, and a later sync captured that grip (no
    role) and REPLACED the pending replay -- whose fallback was the first
    row. The Trash fallback is the owed landing's role while one is owed.
    """
    trash = build_library_notes_trash_state(
        _deleted_records(21, 38), total=37, page=2, now=NOW
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "21–37 of 37")
        fake = _focus_host(app)
        fake._library_notes_compact = False
        fake._library_notes_stage = "notes"
        fake._library_notes_focus_region = lambda: "trash"
        fake._trash_focus_role = "trash-restore:19"
        grip_capture = LibraryNotesFocusIdentity(
            stage="notes", region="trash", note_id=None, semantic_role=""
        )

        target = LibraryNotesController._library_notes_fallback_focus_target(
            fake, grip_capture
        )
        assert target is app.query_one("#library-notes-trash-restore-16", Button)

        fake._trash_focus_role = ""
        target = LibraryNotesController._library_notes_fallback_focus_target(
            fake, grip_capture
        )
        assert target is app.query_one("#library-notes-trash-restore-0", Button)


# -- TASK-34000.29 Task 8 fix round 1 (review R0 minors) -----------------


def test_a_one_row_page_names_its_one_row_without_a_range() -> None:
    """m-8: one row reads "21 of 21", not the range "21–21 of 21"."""
    state = build_library_notes_trash_state(
        _deleted_records(21, 22), total=21, page=2, now=NOW
    )

    assert state.pager.range_copy == "21 of 21"
    assert build_library_notes_trash_state(
        _deleted_records(1, 2), total=1, now=NOW
    ).pager.range_copy == "1 of 1"


class _FailingPagedTrashService(_PagedTrashService):
    """Raises on one offset, the way a locked or failing read does."""

    def __init__(self, total: int, *, fail_offset: int) -> None:
        super().__init__(total)
        self.fail_offset = fail_offset

    async def list_deleted_notes(self, *, scope, user_id, limit, offset):
        if offset == self.fail_offset:
            self.offsets.append(offset)
            raise RuntimeError("database is locked")
        return await super().list_deleted_notes(
            scope=scope, user_id=user_id, limit=limit, offset=offset
        )


@pytest.mark.asyncio
async def test_a_failed_page_read_leaves_the_view_on_the_page_it_paints() -> None:
    """m-1: the page the view asks for never runs ahead of the page it shows.

    Base (c677c8a89a): ``trash_page`` stayed 2 over a painted page 1, so the
    next Next counted from 2, found no page 3 and did nothing -- the user was
    stuck until the view was reopened.
    """
    service = _FailingPagedTrashService(total=25, fail_offset=20)
    fake = _trash_loader_fake(service)
    await LibraryNotesController._load_library_notes_trash(fake)
    assert fake._library_notes_trash.page == 1

    requests: list[dict] = []
    fake._refresh_library_notes_trash = lambda **kwargs: requests.append(kwargs)
    fake._library_notes_mutation_fenced = lambda: False
    fake._page_library_notes_trash = MethodType(
        LibraryNotesController._page_library_notes_trash, fake
    )
    press = SimpleNamespace(stop=lambda: None)

    LibraryNotesController.handle_library_notes_trash_next(fake, press)
    await LibraryNotesController._load_library_notes_trash(fake, **requests[-1])

    assert service.offsets == [0, 20]
    assert fake._library_notes_trash.page == 1
    assert fake._library_notes_trash_page == 1
    assert not fake._library_notes_trash.pager.next_disabled
    # Fix round 2 (ruling T8-i): the painted page stays, and says the read
    # failed (the canvas paints that line above it -- pinned below).
    assert fake._library_notes_trash.failed
    assert fake._library_notes_trash.rows[0].title == "Deleted 01"

    # Next still works: the second press asks for page 2 again.
    LibraryNotesController.handle_library_notes_trash_next(fake, press)
    assert len(requests) == 2
    await LibraryNotesController._load_library_notes_trash(fake, **requests[-1])
    assert service.offsets == [0, 20, 20]

    # A read that succeeds clears the line.
    service.fail_offset = None
    LibraryNotesController.handle_library_notes_trash_next(fake, press)
    await LibraryNotesController._load_library_notes_trash(fake, **requests[-1])
    assert fake._library_notes_trash.page == 2
    assert not fake._library_notes_trash.failed


def test_reopening_the_trash_never_paints_the_page_left_from_the_last_visit(
    monkeypatch,
) -> None:
    """m-7: Back then Recently deleted again starts on page 1, never page N.

    Leaving resets the page the view asks for; opening over a snapshot of
    another page paints page 1's loading state (the Media Trash's own entry:
    a fresh state, then the read) and owes focus to the first row once it
    lands. Base (c677c8a89a): the reopened view painted "21–25 of 25 · Page
    2 of 2" until the page-1 read replaced it.
    """
    fake = _view_fake(monkeypatch, view="trash")
    fake._library_notes_trash = build_library_notes_trash_state(
        _deleted_records(21, 26), total=25, page=2, now=NOW
    )
    fake._library_notes_trash_page = 2
    fake._library_notes_mutation_fenced = lambda: False
    reloads: list[dict] = []
    fake._refresh_library_notes_trash = lambda **kwargs: reloads.append(kwargs)
    press = SimpleNamespace(stop=lambda: None)

    LibraryNotesController.handle_library_notes_trash_back(fake, press)
    assert fake._library_notes_trash_page == 1

    LibraryNotesController.handle_library_notes_trash_open(fake, press)
    shown = fake._library_notes_trash
    assert (shown.page, shown.rows, shown.loading, shown.total) == (1, (), True, 25)
    assert reloads == [{"focus_role": "trash-restore:0"}]
    assert fake._trash_focus_role == "trash-restore:0"


@pytest.mark.asyncio
async def test_a_loading_trash_says_so_instead_of_claiming_it_is_empty() -> None:
    """m-7: page 1's loading state is not the empty Trash's sentence."""
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=LibraryNotesTrashState(total=25, page=1, loading=True),
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "Loading recently deleted notes")
        assert "Nothing deleted recently" not in _painted(app)
        assert app.query_one("#library-notes-trash-back", Button)



# -- Task 8 fix round 2 (review R1; rulings T8-h, T8-i) -------------------

_FAILURE = "Couldn't load recently deleted notes."


@pytest.mark.asyncio
async def test_a_failed_reopen_says_so_and_retry_reads_the_page_again() -> None:
    """N-2 (ruling T8-i): no read leaves the view on "Loading…" forever.

    The reopen's page-1 loading state, then a read that fails at offset 0:
    the failure line and Retry replace the loading state (never "Nothing
    deleted recently", which nothing was read to support). Retry re-runs the
    same read; once it lands the page paints and the line is gone, with
    focus owed to the first row. Base (739adc11c8): the state has no
    failure to carry, and the view stayed on "Loading recently deleted
    notes…".
    """
    service = _FailingPagedTrashService(total=25, fail_offset=0)
    fake = _trash_loader_fake(service)
    fake._library_notes_trash = LibraryNotesTrashState(total=25, page=1, loading=True)
    await LibraryNotesController._load_library_notes_trash(fake)

    failed = fake._library_notes_trash
    assert failed.failed and not failed.loading and failed.rows == ()
    assert failed.total == 25
    app = _CanvasApp(pane_width=100, mode="trash", trash=failed)
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, _FAILURE)
        assert "Loading recently deleted notes" not in text
        assert "Nothing deleted recently" not in text
        retry = app.query_one("#library-notes-trash-retry", Button)
        assert str(retry.label) == "Retry"
        # Its portable role, both halves.
        assert _role(app, "#library-notes-trash-retry") == "trash-retry"
        assert _target(app, "trash-retry") is retry

    service.fail_offset = None
    requests: list[dict] = []
    fake._refresh_library_notes_trash = lambda **kwargs: requests.append(kwargs)
    fake._library_notes_mutation_fenced = lambda: False
    LibraryNotesController.handle_library_notes_trash_retry(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert requests == [{"focus_role": "trash-restore:0"}]
    assert fake._trash_focus_role == "trash-restore:0"
    await LibraryNotesController._load_library_notes_trash(fake, **requests[-1])

    fixed = fake._library_notes_trash
    assert not fixed.failed and fixed.pager.range_copy == "1–20 of 25"
    app = _CanvasApp(pane_width=100, mode="trash", trash=fixed)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until_painted(pilot, app, "1–20 of 25 · Page 1 of 2")
        assert _FAILURE not in _painted(app)
        assert not app.query("#library-notes-trash-retry")


@pytest.mark.asyncio
async def test_a_failed_page_change_shows_the_line_above_the_painted_page() -> None:
    """N-2: a failed Next keeps page 1 on screen and says so above it."""
    shown = build_library_notes_trash_state(
        _deleted_records(1, 21), total=25, page=1, now=NOW
    )
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        # Next to page 2 failed: page 1 stays, Retry asks for page 2.
        trash=dataclasses.replace(shown, retry_page=2),
    )
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, _FAILURE)
        assert "Deleted 01" in text and "1–20 of 25 · Page 1 of 2" in text
        line = app.query_one("#library-notes-trash-error", Static)
        first_row = list(app.query(".library-notes-trash-row"))[0]
        assert line.region.bottom <= first_row.region.y
        assert app.query_one("#library-notes-trash-retry", Button)


class _ShrunkThenFailingService(_PagedTrashService):
    """The total shrank under page 2, and the re-read of the last page fails."""

    async def list_deleted_notes(self, *, scope, user_id, limit, offset):
        self.offsets.append(offset)
        if offset == 0:
            raise RuntimeError("database is locked")
        return {"items": [], "total": self.total}


@pytest.mark.asyncio
async def test_a_failed_clamp_re_read_keeps_the_painted_page_and_says_so() -> None:
    """N-3: the clamp's re-read failing keeps page 2 painted, with the line."""
    service = _ShrunkThenFailingService(total=20)
    fake = _trash_loader_fake(service, page=2)
    fake._library_notes_trash = build_library_notes_trash_state(
        _deleted_records(21, 26), total=25, page=2, now=NOW
    )

    await LibraryNotesController._load_library_notes_trash(fake)

    assert service.offsets == [20, 0]
    shown = fake._library_notes_trash
    assert shown.page == 2 and fake._library_notes_trash_page == 2
    assert [row.title for row in shown.rows] == [
        f"Deleted {index:02d}" for index in range(21, 26)
    ]
    assert shown.failed
