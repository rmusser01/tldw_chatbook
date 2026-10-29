"""Library ▸ Notes Trash view — the second safety net behind the receipt.

Group `i-trash` of the Notes riders wave: task-32144. The delete receipt is
a good immediate affordance and a bad only one; a "Recently deleted (N)" row
under the folder tree opens a list of soft-deleted notes whose Restore goes
through the SAME seam Undo does (``_undo_library_note_delete`` ->
``NotesScopeService.restore_note``). TASK-34000.29 (8b, owner ruling
2026-10-29 "yes with confimration"): each row also offers Delete forever,
behind a confirm that names the note (``NotesScopeService.purge_note``).
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
    assert_every_action_fits,
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
async def test_trash_view_offers_delete_forever_behind_a_confirm() -> None:
    """AC#9: each row's Delete forever, beside its Restore (owner, 2026-10-29).

    Rewritten by name from ``test_trash_view_offers_no_permanent_delete``
    (TASK-34000.29 8b): the owner ruled "yes with confimration". Each row
    carries a danger-marked Delete forever bound to that row's note, painted
    inside the pane. It is not a Tab stop -- a row is one stop, its Restore,
    and ``x`` acts on the focused row as ``r`` does -- so the page keeps one
    Tab per row. The press opens a confirm
    (``test_delete_forever_asks_first_and_names_the_note``); the purpose line
    no longer promises that nothing here removes a note for good.
    """
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=_trash(_row("n1", "Groceries"), _row("n2", "Draft"), total=2),
    )
    async with app.run_test(size=(120, 40)):
        deletes = list(app.query(".library-notes-trash-delete").results(Button))
        assert [str(button.label) for button in deletes] == ["Delete forever"] * 2
        assert [button.id for button in deletes] == [
            "library-notes-trash-delete-0",
            "library-notes-trash-delete-1",
        ]
        assert [button.note_id for button in deletes] == ["n1", "n2"]
        assert [button.note_title for button in deletes] == ["Groceries", "Draft"]
        assert [button.note_version for button in deletes] == [2, 2]
        assert all(button.has_class("library-canvas-danger") for button in deletes)
        assert not any(button.focusable for button in deletes)
        purpose = LibraryNotesCanvas._static_text(
            app.query_one("#library-notes-trash-purpose", Static)
        )
        assert "nothing is removed for good" not in purpose
        assert "Delete forever" in purpose
        assert_every_action_fits(app)


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
        # TASK-34000.29 8b: focus on Back here, so ``x`` is not live and the
        # footer does not offer it (``test_the_x_chip_is_painted_only_...``).
        check_action=lambda _action, _parameters: False,
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
    deferred: list[tuple] = []
    fake = SimpleNamespace(
        _library_notes_view="trash",
        _trash_focus_role="trash-restore:4",
        _restore_library_notes_focus_identity=lambda identity: landed.append(
            identity.semantic_role
        ),
        call_after_refresh=lambda *args: deferred.append(args),
    )

    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-restore:4")
    assert landed == ["trash-restore:4"]
    # Review R2-2: a landing that ran also scrolls its control in once the
    # deferred offset re-applies have run; a vetoed one schedules nothing.
    assert [args[1:] for args in deferred] == [("trash-restore:4",)]
    assert deferred[0][0].func is LibraryNotesController._keep_landed_trash_control_in_view

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
    assert len(deferred) == 1


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


# -- Task 8 fix round 3 (review R2) ----------------------------------------


@pytest.mark.asyncio
async def test_reopening_after_a_failed_read_paints_loading_without_the_failure(
    monkeypatch,
) -> None:
    """R2-3: a reopen starts clean -- "Loading…", no failure line, no Retry.

    A failed read, then Back, then Recently deleted again: the reopen's page-1
    loading state must not carry the old visit's failure, or Retry would
    re-read that visit's failed page. Base (db6910e4f6): the replace kept
    ``retry_page``, so the view painted the failure line over "Loading…".
    """
    fake = _view_fake(monkeypatch, view="trash")
    shown = build_library_notes_trash_state(
        _deleted_records(21, 26), total=25, page=2, now=NOW
    )
    fake._library_notes_trash = dataclasses.replace(shown, retry_page=2)
    fake._library_notes_trash_page = 2
    fake._library_notes_mutation_fenced = lambda: False
    fake._refresh_library_notes_trash = lambda **_kwargs: None
    press = SimpleNamespace(stop=lambda: None)

    LibraryNotesController.handle_library_notes_trash_back(fake, press)
    LibraryNotesController.handle_library_notes_trash_open(fake, press)

    reopened = fake._library_notes_trash
    assert reopened.loading and not reopened.failed and reopened.retry_page == 0
    app = _CanvasApp(pane_width=100, mode="trash", trash=reopened)
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, "Loading recently deleted notes")
        assert _FAILURE not in text
        assert not app.query("#library-notes-trash-retry")



def test_the_landed_control_is_scrolled_in_only_while_it_still_has_focus() -> None:
    """R2-2's constraint: never scroll after the user's own move."""
    scrolled: list[str] = []
    control = SimpleNamespace(
        scroll_visible=lambda **_kwargs: scrolled.append("next"),
    )
    fake = SimpleNamespace(
        _library_notes_view="trash",
        _trash_focus_role="trash-next",
        _library_notes_trash_focus_target=lambda _role: control,
        focused=control,
    )

    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-next")
    assert scrolled == ["next"]

    fake.focused = SimpleNamespace()  # the user moved focus elsewhere
    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-next")
    assert scrolled == ["next"]

    fake.focused = control
    fake._trash_focus_role = ""  # a real input withdrew the landing
    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-next")
    assert scrolled == ["next"]


def test_the_trash_offset_is_withheld_only_while_a_page_landing_settles() -> None:
    """Ruling T8-j (review R3-1): order-proof, in the one scroll-owner seam.

    From a landing read's publish until its keep-in-view has run, the Trash
    list is no capture's scroll owner, so no sync -- on either side of the
    keep-in-view -- puts the new list's pre-layout offset back over it. The
    keep-in-view, a real input and the next press each end it; afterwards a
    same-page reload keeps its offset again (review R0 m-6).
    """
    listing = SimpleNamespace(trash_page=1)
    fake = SimpleNamespace(
        _library_layout_ref=lambda _selector: listing,
        _library_notes_trash=SimpleNamespace(page=1),
        _library_notes_view="trash",
        _trash_focus_role="trash-previous",
        _trash_landing_settling="trash-previous",
        _library_notes_trash_focus_target=lambda _role: None,
        focused=None,
    )

    def owner():
        return LibraryNotesController._library_notes_scroll_owner(fake, "trash")

    assert owner() is None
    fake._trash_focus_role = ""  # a real input withdrew the landing
    assert owner() is listing
    fake._trash_focus_role = "trash-previous"
    assert owner() is None
    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-previous")
    assert fake._trash_landing_settling == ""
    assert owner() is listing

    pressed, _refreshes = _pager_press_fake(fenced=False)
    pressed._trash_focus_role = pressed._trash_landing_settling = "trash-next"
    LibraryNotesController.handle_library_notes_trash_next(
        pressed, SimpleNamespace(stop=lambda: None)
    )
    assert pressed._trash_focus_role == "trash-next"
    assert pressed._trash_landing_settling == ""


# -- TASK-34000.29 part 8b: Delete forever (owner, 2026-10-29) --------------


def _press_event(button) -> SimpleNamespace:
    return SimpleNamespace(stop=lambda: None, button=button)


def _purge_press_fake(*, fenced: bool = False, view: str = "trash"):
    """A controller-shaped receiver whose dialogs and workers are recorded."""
    pushed: list[tuple] = []
    workers: list[tuple] = []
    notices: list[tuple] = []
    fake = SimpleNamespace(
        _library_notes_view=view,
        _trash_focus_role="",
        _trash_landing_settling="settling",
        _library_notes_mutation_in_flight=False,
        _library_notes_mutation_fenced=lambda: fenced,
        app=SimpleNamespace(
            push_screen=lambda screen, callback: pushed.append((screen, callback))
        ),
        app_instance=SimpleNamespace(
            notify=lambda message, **kwargs: notices.append((message, kwargs))
        ),
    )

    def run_worker(work, **kwargs):
        # The coroutine's own arguments, read before it is closed unstarted.
        workers.append((work.cr_code, dict(work.cr_frame.f_locals), kwargs))
        work.close()

    fake.run_worker = run_worker
    fake._purge_library_note_from_trash = MethodType(
        LibraryNotesController._purge_library_note_from_trash, fake
    )
    fake._confirm_library_notes_trash_delete = MethodType(
        LibraryNotesController._confirm_library_notes_trash_delete, fake
    )
    fake._cancel_library_notes_trash_delete = MethodType(
        LibraryNotesController._cancel_library_notes_trash_delete, fake
    )
    return fake, pushed, workers, notices


def test_delete_forever_asks_first_and_names_the_note() -> None:
    """ADR-055 Pattern B, ruling T8b-a: nothing is written before the confirm.

    The confirm is the house ``ConfirmationDialog``: it renders title and
    message with ``markup=False`` (a title like ``[b]x[/b]`` is named, not
    styled), opens on Cancel (``AUTO_FOCUS``), and Escape is its Cancel.
    Cancel writes nothing; only Delete forever starts the purge, under the
    Notes interlock, with focus owed to the row that will take this one's
    place (ruling T8b-e).
    """
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    fake, pushed, workers, _notices = _purge_press_fake()
    button = SimpleNamespace(
        id="library-notes-trash-delete-3",
        note_id="n4",
        note_title="[b]Groceries[/b]",
        note_version=5,
    )

    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))

    ((dialog, answer),) = pushed
    assert isinstance(dialog, ConfirmationDialog)
    assert dialog.title == "Delete this note forever?"
    assert dialog.message == (
        "“[b]Groceries[/b]” will be deleted for good. This cannot be undone."
    )
    assert dialog.confirm_label == "Delete forever"
    assert dialog.cancel_label == "Cancel"
    assert ConfirmationDialog.AUTO_FOCUS == "#cancel-button"
    # Review R0, m-4: the commit held >=3 cells from Cancel (measured below).
    assert dialog.has_class("destructive-confirm")
    assert workers == []

    answer(False)  # Cancel or Escape
    assert workers == []
    assert fake._library_notes_mutation_in_flight is False
    # The dialog's push withdrew every owed landing; the cancel owes focus
    # back to the row ``x`` was pressed on, so the resume's reload that
    # recomposes the list cannot fall back to the first row.
    assert fake._trash_focus_role == "trash-restore:3"

    answer(True)
    ((code, arguments, options),) = workers
    assert options == {"exclusive": True, "group": "library_note_mutation"}
    assert code is LibraryNotesController._purge_library_note_from_trash.__code__
    assert arguments["note_id"] == "n4"
    assert arguments["title"] == "[b]Groceries[/b]"
    assert arguments["version"] == 5
    assert arguments["trash_focus"] == "trash-restore:3"
    assert fake._library_notes_mutation_in_flight is True
    assert fake._trash_focus_role == "trash-restore:3"
    assert fake._trash_landing_settling == ""


def test_a_confirm_that_meets_another_notes_change_deletes_nothing() -> None:
    """The interlock is read again when the confirm returns, and says so."""
    fake, pushed, workers, notices = _purge_press_fake()
    button = SimpleNamespace(
        id="library-notes-trash-delete-0", note_id="n1", note_title="A", note_version=2
    )
    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    fake._library_notes_mutation_fenced = lambda: True

    pushed[0][1](True)

    assert workers == []
    assert notices == [
        (
            "Nothing was deleted — another change to your notes is still running.",
            {"severity": "warning", "markup": False},
        )
    ]


def _trash_key_fake(*, view="trash", rows=True, fenced=False, delete=object()):
    return SimpleNamespace(
        _library_notes_workflow_active=lambda: True,
        _notes_state=SimpleNamespace(
            view=view, trash=_trash(_row("n1", "Groceries")) if rows else _trash()
        ),
        _library_notes_mutation_fenced=lambda: fenced,
        _focused_library_notes_trash_delete=lambda: delete,
    )


def test_x_is_live_only_on_a_focused_trash_row() -> None:
    """Ruling T8b-d: Delete forever's key, and nowhere else.

    ``check_action`` is the one decision the binding, the footer chip and F1
    all read. Review R0: exactly the focused row, where the button shows
    (I-2) -- and still while another Notes change runs, so ``x`` reaches the
    handler that says why it waits (m-5; pinned below).
    """
    check = LibraryScreen.check_action
    assert check(_trash_key_fake(), "library_notes_trash_delete", ()) is True
    # Focus on Back or the pager: no focused row, no key.
    assert not check(_trash_key_fake(delete=None), "library_notes_trash_delete", ())
    # The list view's focused note row: x does nothing there.
    assert not check(_trash_key_fake(view="list"), "library_notes_trash_delete", ())
    assert not check(_trash_key_fake(rows=False), "library_notes_trash_delete", ())
    assert check(_trash_key_fake(fenced=True), "library_notes_trash_delete", ()) is True


def test_the_x_chip_is_painted_only_where_x_deletes() -> None:
    """The footer asks the binding's own gate; F1 lists the same key there."""
    asked: list[str] = []

    def footer(live: bool, compact: bool = False):
        fake = SimpleNamespace(
            _notes_state=SimpleNamespace(
                view="trash",
                compact=compact,
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
            _notes_footer_tier=lambda wide, short: short if compact else wide,
            check_action=lambda action, _params: asked.append(action) or live,
        )
        return LibraryScreen._library_notes_footer_shortcuts(fake)

    assert footer(True) == (
        ("r", "restore note"),
        ("x", "delete forever"),
        ("esc", "back to notes"),
    )
    assert footer(True, compact=True) == (
        ("r", "restore"),
        ("x", "delete forever"),
        ("esc", "notes"),
    )
    assert footer(False) == (("r", "restore note"), ("esc", "back to notes"))
    assert set(asked) == {"library_notes_trash_delete"}

    def f1_keys(live: bool) -> list[tuple[str, str]]:
        fake = SimpleNamespace(
            check_action=lambda action, _params: live
            if action == "library_notes_trash_delete"
            else False,
            BINDINGS=LibraryScreen.BINDINGS,
        )
        return [
            pair
            for pair in LibraryScreen._active_library_binding_shortcuts(fake)
            if pair[0] == "x"
        ]

    assert f1_keys(True) == [("x", "Delete forever")]
    assert f1_keys(False) == []


@pytest.mark.asyncio
async def test_x_deletes_the_focused_rows_note_never_a_guess(monkeypatch) -> None:
    """``x`` stands for the focused row's own Delete forever button."""
    app = _CanvasApp(
        pane_width=100,
        mode="trash",
        trash=_trash(_row("n1", "Groceries"), _row("n2", "Draft"), total=2),
    )
    async with app.run_test(size=(120, 40)):
        restores = list(app.query(".library-notes-trash-restore").results(Button))
        pressed: list[str] = []
        monkeypatch.setattr(
            Button, "press", lambda self: pressed.append(self.id) or self
        )
        fake = SimpleNamespace(focused=restores[1], query=app.query)
        fake._focused_library_notes_trash_delete = MethodType(
            LibraryScreen._focused_library_notes_trash_delete, fake
        )
        LibraryScreen.action_library_notes_trash_delete(fake)
        assert pressed == ["library-notes-trash-delete-1"]

        pressed.clear()
        fake.focused = app.query_one("#library-notes-trash-back", Button)
        LibraryScreen.action_library_notes_trash_delete(fake)
        assert pressed == []


class _PurgeService:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    async def purge_note(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _purge_worker_fake(outcome, *, receipt_note: str = "n4"):
    from tldw_chatbook.Library.library_notes_state import LibraryNoteDeleteReceipt

    service = _PurgeService(outcome)
    notices: list[tuple] = []
    refreshes: list[dict] = []
    runtime = object()
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            notes_scope_service=service,
            notes_sync_runtime_owner=runtime,
            notify=lambda message, **kwargs: notices.append((message, kwargs)),
        ),
        _run_library_service_call=_passthrough_service_call,
        _library_notes_user_id=lambda: "user-1",
        _refresh_library_notes_trash=lambda **kwargs: refreshes.append(kwargs),
        _library_notes_mutation_in_flight=True,
        _tree_reload_owed=False,
        _trash_reload_owed="",
        _trash_focus_role="trash-restore:3",
        _library_note_delete_receipt=LibraryNoteDeleteReceipt(
            note_id=receipt_note, title="Groceries", expected_version=4
        ),
        is_mounted=True,
    )
    return fake, service, notices, refreshes, runtime


@pytest.mark.asyncio
async def test_a_purge_says_so_and_lands_on_the_row_that_took_its_place() -> None:
    """Receipt-free notice (ADR-055 B), then the same page, focus owed.

    The list's own delete receipt for the same note goes too: its Undo
    would try to restore a note that no longer exists.
    """
    fake, service, notices, refreshes, runtime = _purge_worker_fake(True)

    await LibraryNotesController._purge_library_note_from_trash(
        fake, "n4", "Groceries", 5, trash_focus="trash-restore:3"
    )

    assert service.calls == [
        {
            "scope": "local_note",
            "note_id": "n4",
            "version": 5,
            "user_id": "user-1",
            "sync_runtime": runtime,
        }
    ]
    assert notices == [("Deleted forever · Groceries", {"markup": False})]
    assert refreshes == [{"focus_role": "trash-restore:3"}]
    assert fake._library_note_delete_receipt is None
    assert fake._library_notes_mutation_in_flight is False


@pytest.mark.asyncio
async def test_a_refused_purge_keeps_the_row_and_says_why() -> None:
    """Ruling T8b-c: a sync-bound note is refused with the guide's sentence."""
    from tldw_chatbook.Notes.notes_scope_service import (
        PURGE_REFUSED_SYNC_BOUND,
        NotePurgeRefusedError,
    )

    fake, _service, notices, refreshes, _runtime = _purge_worker_fake(
        NotePurgeRefusedError(
            reason_code="sync_bound", user_message=PURGE_REFUSED_SYNC_BOUND
        ),
        receipt_note="n4",
    )

    await LibraryNotesController._purge_library_note_from_trash(
        fake, "n4", "Groceries", 5, trash_focus="trash-restore:3"
    )

    assert notices == [
        (PURGE_REFUSED_SYNC_BOUND, {"severity": "warning", "markup": False})
    ]
    assert refreshes == []
    assert fake._library_note_delete_receipt is not None
    assert fake._library_notes_mutation_in_flight is False


@pytest.mark.asyncio
async def test_a_purge_that_finds_the_tombstone_changed_reloads_the_page() -> None:
    from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

    fake, _service, notices, refreshes, _runtime = _purge_worker_fake(
        ConflictError("version mismatch", entity="notes", entity_id="n4")
    )

    await LibraryNotesController._purge_library_note_from_trash(
        fake, "n4", "Groceries", 5, trash_focus="trash-restore:3"
    )

    assert notices == [
        (
            "This deleted note changed elsewhere — refresh and try again.",
            {"severity": "warning", "markup": False},
        )
    ]
    assert refreshes == [{}]


def test_every_trash_landing_withholds_the_offset_until_it_settles() -> None:
    """Parked R4-2 (ruling T8b-e): a same-page landing settles like a page's.

    A Restore -- and now a purge -- lands focus from its own canvas sync, not
    from a page read's publish, so nothing raised the settling marker and a
    sync captured right after the landing put the new list's pre-layout
    offset back over it. The landing raises it itself.
    """
    fake = SimpleNamespace(
        _library_notes_view="trash",
        _trash_focus_role="trash-restore:2",
        _trash_landing_settling="",
        _restore_library_notes_focus_identity=lambda _identity: None,
        call_after_refresh=lambda *_args: None,
    )

    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-restore:2")

    assert fake._trash_landing_settling == "trash-restore:2"


def test_a_dropped_landing_stops_withholding_the_trash_offset() -> None:
    """Parked R4-1 (ruling T8b-e): the marker never outlives its landing.

    A page change's landing can be dropped -- a later sync's explicit
    ``then`` (a Restore's ``finally``) replaces it -- and a wheel scroll is
    no withdrawal, so the marker stayed up and the next sync jumped the list
    back over the user's own scroll. Two ends now: a landing callback that
    runs for another role (the replacing sync's) clears it, and so does the
    real-input withdrawal.
    """
    listing = SimpleNamespace(trash_page=1)
    fake = SimpleNamespace(
        _library_layout_ref=lambda _selector: listing,
        _library_notes_trash=SimpleNamespace(page=1),
        _library_notes_view="trash",
        _trash_focus_role="trash-previous",
        _trash_landing_settling="trash-previous",
        _restore_library_notes_focus_identity=lambda _identity: None,
        call_after_refresh=lambda *_args: None,
    )

    def owner():
        return LibraryNotesController._library_notes_scroll_owner(fake, "trash")

    assert owner() is None
    # The replacing sync's landing is for the Restore's role: vetoed, and
    # the page change's landing it displaced will never run.
    LibraryNotesController._land_library_notes_trash_focus(fake, "trash-restore:4")
    assert fake._trash_landing_settling == ""
    assert owner() is listing

    fake._trash_landing_settling = "trash-previous"
    screen = SimpleNamespace(
        _library_notes_filter_row_handoff=None,
        _notes_state=SimpleNamespace(pending_focus_identity=None),
        _notes_controller=fake,
    )
    LibraryScreen._withdraw_library_notes_pending_focus_handoffs(screen)
    assert fake._trash_focus_role == ""
    assert fake._trash_landing_settling == ""


def test_a_landing_waits_while_a_pruning_sync_has_focus_parked() -> None:
    """Ruling T8b-e: settle only once the landed control has focus again.

    Traced at 100x30: a sync right after a Restore's landing recomposed the
    list and pruned the landed button, Textual parked focus on the Library
    grip, and the keep-in-view ran THEN -- it scrolled nothing and dropped
    the marker under the replay, so a later capture re-applied offset 0 over
    the row below the fold. Seen once in the 100x30 Pilot before this
    existed; without it, 14 later runs passed, so only this rider can force
    the window. While focus sits on that framework landing the
    check defers to the next refresh -- the offset still withheld -- and a
    bounded number of times.
    """
    scrolled: list[str] = []
    deferred: list[tuple] = []
    control = SimpleNamespace(scroll_visible=lambda **_kwargs: scrolled.append("row"))
    grip = SimpleNamespace(id="library-browse-items-grip")
    parked: dict[str, object] = {"on": grip}
    fake = SimpleNamespace(
        _library_notes_view="trash",
        _trash_focus_role="trash-restore:19",
        _trash_landing_settling="trash-restore:19",
        _library_notes_trash_focus_target=lambda _role: control,
        focused=grip,
        call_after_refresh=lambda *args: deferred.append(args),
        _screen=SimpleNamespace(
            _library_notes_reset_landing_canvas=lambda focused: (
                object() if focused is parked["on"] else None
            )
        ),
    )

    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-restore:19")
    assert fake._trash_landing_settling == "trash-restore:19"
    assert scrolled == []
    ((retry, role, attempt),) = deferred
    assert retry.func is LibraryNotesController._keep_landed_trash_control_in_view
    assert (role, attempt) == ("trash-restore:19", 1)

    fake.focused = control  # the sync's replay put focus back
    retry(role, attempt)
    assert scrolled == ["row"]
    assert fake._trash_landing_settling == ""

    # Bounded: focus that never comes back stops withholding the offset.
    fake.focused = grip
    fake._trash_landing_settling = "trash-restore:19"
    LibraryNotesController._keep_landed_trash_control_in_view(
        fake, "trash-restore:19", 5
    )
    assert fake._trash_landing_settling == ""
    # A focus the USER chose (not the framework's landing) settles at once.
    fake._trash_landing_settling = "trash-restore:19"
    fake.focused = SimpleNamespace(id="library-notes-trash-back")
    deferred.clear()
    LibraryNotesController._keep_landed_trash_control_in_view(fake, "trash-restore:19")
    assert deferred == []
    assert fake._trash_landing_settling == ""


@pytest.mark.asyncio
async def test_delete_forever_leaves_a_dated_title_its_date() -> None:
    """The danger class's 4-cell gap cost the part of the title that differs.

    Live at 160x45 every journal row read "Daily log 2026-09-0…": the rows
    could not be told apart. Delete forever keeps its muted ink with a
    1-cell gap, which leaves the date (the age after it may still be cut).
    In this harness a 54-cell pane is where the gap decides it: with the
    4-cell gap the row reads "Daily log 2026-09-…" (measured).
    """
    app = _CanvasApp(
        pane_width=54,
        mode="trash",
        trash=_trash(
            _row("n1", "Daily log 2026-09-01", "1m"),
            _row("n2", "Daily log 2026-09-02", "1m"),
            total=2,
        ),
    )
    async with app.run_test(size=(120, 40)) as pilot:
        text = await _until_painted(pilot, app, "Delete forever")
        assert "Daily log 2026-09-01" in text
        assert "Daily log 2026-09-02" in text
        delete = app.query_one("#library-notes-trash-delete-0", Button)
        assert delete.has_class("library-media-action-danger")


# -- TASK-34000.29 8b, review R0: m-4, m-5, m-8, m-9 and the R4 residual ----


@pytest.mark.asyncio
async def test_the_delete_forever_confirm_holds_its_commit_three_cells_from_cancel() -> None:
    """m-4 (task-31980's Media rule): a slip off Cancel lands on nothing.

    Only the destructive confirm moves: a plain ``ConfirmationDialog`` keeps
    its gap (``margin: 0 1`` on each button, collapsed: 1 cell, measured).
    Base (780fe1da8b): 1 cell either way.
    """
    from textual.app import App

    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    gaps: dict[str, int] = {}
    for name, extra in (("destructive", {"classes": "destructive-confirm"}), ("plain", {})):
        app = App()
        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(
                ConfirmationDialog(
                    title="Delete this note forever?",
                    message="“A” will be deleted for good.",
                    confirm_label="Delete forever",
                    cancel_label="Cancel",
                    **extra,
                )
            )
            await pilot.pause()
            cancel = app.screen.query_one("#cancel-button")
            confirm = app.screen.query_one("#confirm-button")
            assert cancel.region.y == confirm.region.y
            gaps[name] = confirm.region.x - cancel.region.right
    assert gaps["destructive"] >= 3, gaps
    assert gaps["plain"] == 1, gaps


def test_a_press_while_another_notes_change_runs_says_why_on_the_trash() -> None:
    """m-5: no silent no-op -- the Trash says why, and the release clears it.

    Base (780fe1da8b): the press returned with nothing pushed and nothing
    said.
    """
    from tldw_chatbook.Widgets.Library.library_notes_canvas import PURGE_BUSY_NOTICE

    fake, pushed, workers, notices = _purge_press_fake(fenced=True)
    fake._library_notes_trash = _trash(_row("n1", "Groceries"))
    fake.is_mounted = False
    button = SimpleNamespace(
        id="library-notes-trash-delete-0", note_id="n1", note_title="Groceries", note_version=2
    )

    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))

    assert pushed == [] and workers == [] and notices == []
    assert fake._library_notes_trash.notice == PURGE_BUSY_NOTICE
    LibraryNotesController._clear_library_notes_trash_busy(fake)
    assert fake._library_notes_trash.notice == ""


@pytest.mark.asyncio
async def test_the_trash_paints_the_busy_reason_and_delete_forever_has_a_tooltip() -> None:
    """m-5: the reason sits under the purpose line; the button explains itself."""
    from tldw_chatbook.Widgets.Library.library_notes_canvas import (
        PURGE_BUSY_NOTICE,
        PURGE_TOOLTIP,
    )

    trash = dataclasses.replace(_trash(_row("n1", "Groceries")), notice=PURGE_BUSY_NOTICE)
    app = _CanvasApp(pane_width=100, mode="trash", trash=trash)
    async with app.run_test(size=(120, 40)):
        notice = app.query_one("#library-notes-trash-notice", Static)
        assert LibraryNotesCanvas._static_text(notice) == PURGE_BUSY_NOTICE
        delete = app.query_one("#library-notes-trash-delete-0", Button)
        assert delete.tooltip == PURGE_TOOLTIP
    quiet = _CanvasApp(pane_width=100, mode="trash", trash=_trash(_row("n1", "A")))
    async with quiet.run_test(size=(120, 40)):
        assert not quiet.query("#library-notes-trash-notice")


def test_a_second_press_while_the_confirm_is_up_never_stacks_another() -> None:
    """m-8: a fast double ``x`` reaches the handler twice; one confirm only.

    Base (780fe1da8b): two dialogs were pushed.
    """
    fake, pushed, _workers, _notices = _purge_press_fake()
    button = SimpleNamespace(
        id="library-notes-trash-delete-1", note_id="n2", note_title="B", note_version=2
    )

    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    assert len(pushed) == 1

    pushed[0][1](False)  # Cancel: the next press may ask again.
    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    assert len(pushed) == 2

    # A confirm that left without answering (popped by a quit flow) no
    # longer blocks once it is off the screen stack.
    fake.app.screen_stack = []
    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    assert len(pushed) == 3


def test_the_confirm_says_what_stays_elsewhere_only_where_it_is_known() -> None:
    """m-9: a disconnected folder's file, a server copy -- each only if so."""
    from tldw_chatbook.Widgets.Library.library_notes_canvas import purge_confirm_copy

    assert purge_confirm_copy("A") == (
        "“A” will be deleted for good. This cannot be undone."
    )
    assert purge_confirm_copy("A", kept_file=True, server_copy=True) == (
        "“A” will be deleted for good. This cannot be undone. Its file in the "
        "synced folder stays on disk. Copies on your sync server are not removed."
    )
    fake, pushed, _workers, _notices = _purge_press_fake()
    button = SimpleNamespace(
        id="library-notes-trash-delete-0",
        note_id="n1",
        note_title="A",
        note_version=2,
        note_kept_file=True,
        note_server_copy=False,
    )
    LibraryNotesController.handle_library_notes_trash_delete(fake, _press_event(button))
    assert pushed[0][0].message == (
        "“A” will be deleted for good. This cannot be undone. Its file in the "
        "synced folder stays on disk."
    )
    trash = build_library_notes_trash_state(
        _deleted_records(1, 4), total=3, now=NOW, purge_scope={"d2": (False, True)}
    )
    assert [(row.kept_file, row.server_copy) for row in trash.rows] == [
        (False, False),
        (False, True),
        (False, False),
    ]


@pytest.mark.asyncio
async def test_the_canvas_hands_each_rows_scope_to_its_delete_forever() -> None:
    """m-9: the press reads the page read's facts off the row's own button."""
    rows = (
        dataclasses.replace(_row("n1", "A"), kept_file=True),
        dataclasses.replace(_row("n2", "B"), server_copy=True),
    )
    app = _CanvasApp(pane_width=100, mode="trash", trash=_trash(*rows))
    async with app.run_test(size=(120, 40)):
        deletes = list(app.query(".library-notes-trash-delete").results(Button))
        assert [(b.note_kept_file, b.note_server_copy) for b in deletes] == [
            (True, False),
            (False, True),
        ]


@pytest.mark.asyncio
async def test_a_read_with_no_landing_drops_the_marker_of_a_landing_it_replaced(
    monkeypatch,
) -> None:
    """The R4 residual: a non-landing follow-up replaces a landing's ``then``.

    That landing never runs now, so its settling marker must not stay up
    (it withheld the list's offset until the next input). Deterministic: the
    landing read publishes, then a roleless read publishes before any
    landing ran. Base (780fe1da8b): the marker stayed "trash-restore:3".
    """
    from tldw_chatbook.UI.Library_Modules import library_notes_controller as module

    thens: list[object] = []
    monkeypatch.setattr(
        module,
        "_sync_library_canvas",
        lambda _screen, _kind, then=None, **_kwargs: thens.append(then) or True,
    )
    fake = _trash_loader_fake(_PagedTrashService(total=25))
    fake.is_mounted = True
    fake._land_library_notes_trash_focus = lambda _role: None

    await LibraryNotesController._load_library_notes_trash(
        fake, focus_role="trash-restore:3"
    )
    assert fake._trash_landing_settling == "trash-restore:3"
    assert thens[-1] is not None

    await LibraryNotesController._load_library_notes_trash(fake)

    assert thens[-1] is None
    assert fake._trash_landing_settling == ""
