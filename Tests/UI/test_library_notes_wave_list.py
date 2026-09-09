"""Library ▸ Notes list geometry, receipt, Undo, Sort and ages.

Group `list` of the critique-notes-2026-09 fix wave: tasks 32123, 32124,
32127, 32128 and 32137.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import MethodType, SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_library_notes_folder_navigator import (
    _BranchService,
    _branch_screen_fake,
    _folder_page,
    _membership,
    _placement_page,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_notes_state import (
    LibraryNoteDeleteReceipt,
    LibraryNotesListRow,
    LibraryNotesListState,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
    build_paged_library_notes_tree,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NotePlacementRecord,
    NoteTreeLocation,
    NoteTreePathStep,
)
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_READER_PROFILE,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Utils.adaptive_reader_state import (
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotesCanvas,
    _toolbar_shape,
)

#: The two geometries the critique ran at.
WIDE = (235, 52)
COMPACT = (100, 30)


def _list_state(
    *,
    rows: tuple[LibraryNotesListRow, ...] = (),
    delete_receipt: LibraryNoteDeleteReceipt | None = None,
) -> LibraryNotesListState:
    return LibraryNotesListState(
        rows=rows,
        header_copy="Notes (10)",
        status_copy="",
        empty_copy="No notes yet. Create your first note.",
        delete_receipt=delete_receipt,
    )


class _CanvasApp(ConsolidatedCSSApp):
    """Mount one Notes canvas at the exact width a list pane would give it."""

    def __init__(self, *, pane_width: int, **canvas_kwargs) -> None:
        # The Library's own rules live in the app bundle and the split
        # screen sheet, not in the widget defaults -- without them the
        # canvas mounts unstyled and a geometry assertion measures nothing.
        super().__init__(css_path=[str(path) for path in APP_STYLESHEETS])
        self._pane_width = pane_width
        self._canvas_kwargs = canvas_kwargs

    def compose(self):
        # The canvas is told the same width it is given, the way the
        # controller passes the resolved Items width.
        self._canvas_kwargs.setdefault("pane_width", self._pane_width)
        canvas = LibraryNotesCanvas(id="library-notes-canvas", **self._canvas_kwargs)
        canvas.styles.width = self._pane_width
        canvas.styles.max_width = self._pane_width
        yield canvas


def _visible(widget) -> bool:
    return widget.region.width > 0 and widget.region.height > 0


def assert_every_action_fits(app) -> None:
    """Every action in the mounted frame is painted inside the pane.

    Asserting over the whole `.library-canvas-action` population, not a
    hand-listed few: the hand-listed version passed while two siblings in
    the same frame were off-pane (review round 1).
    """
    canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)

    def rendered(widget) -> bool:
        # A hidden container (the browse row while the sort strip is open)
        # legitimately leaves its children region-less.
        node = widget
        while node is not None and node is not canvas.parent:
            if not node.display:
                return False
            node = node.parent
        return True

    actions = [w for w in app.query(".library-canvas-action") if rendered(w)]
    assert actions, "no actions were composed"
    offenders = [
        (widget.id, widget.region)
        for widget in actions
        if not _visible(widget)
        or widget.region.right > canvas.region.right
        or widget.region.x < canvas.region.x
    ]
    assert not offenders, f"actions painted off the {canvas.region.width}-column pane: {offenders}"


# -- task-32127: the list pane is not starved -----------------------------


def test_notes_list_keeps_sixty_columns_beside_an_open_note() -> None:
    """With a note open at 235 the list keeps min(40% of the canvas, 60)."""
    layout = resolve_adaptive_reader_layout(
        WIDE[0],
        AdaptiveReaderLayoutPreferences(),
        LIBRARY_NOTES_READER_PROFILE,
        reader_has_item=True,
    )

    assert layout.items_open is True
    assert layout.items_width >= min(int(WIDE[0] * 0.4), 60)


def test_notes_list_keeps_its_share_beside_an_open_note_when_compact() -> None:
    """The same floor holds at 100 columns, where 40% is the smaller number."""
    layout = resolve_adaptive_reader_layout(
        COMPACT[0],
        AdaptiveReaderLayoutPreferences(),
        LIBRARY_NOTES_READER_PROFILE,
        reader_has_item=True,
    )

    assert layout.items_open is True
    assert layout.items_width >= min(int(COMPACT[0] * 0.4), 60)


def _layout_screen_fake(*, width: int, view: str):
    shell = SimpleNamespace(
        region=SimpleNamespace(width=width),
        sync_layout=lambda layout, **kwargs: setattr(shell, "applied", layout),
        applied=None,
    )
    fake = SimpleNamespace(
        _notes_state=SimpleNamespace(
            view=view,
            stage="notes",
            reader_layout=resolve_adaptive_reader_layout(
                0,
                AdaptiveReaderLayoutPreferences(),
                LIBRARY_NOTES_READER_PROFILE,
            ),
            reader_preferences=AdaptiveReaderLayoutPreferences(),
        ),
        _library_selected_row_id="browse-notes",
        _library_adaptive_reader_allocation_is_current=lambda _shell: True,
        _library_notes_work_first_preferences=lambda preferences: preferences,
        query_one=lambda *_args, **_kwargs: shell,
        # No canvas is mounted in this fake, so the resolved width has
        # nothing to be pushed to.
        query=lambda *_args, **_kwargs: (),
    )
    return fake, shell


def test_notes_list_uses_the_freed_width_when_no_note_is_open() -> None:
    """task-32127 AC#1: an empty work pane hands its width to the list."""
    fake, shell = _layout_screen_fake(width=WIDE[0], view="list")

    LibraryScreen._sync_library_notes_reader_layout_from_shell(fake)

    assert shell.applied is not None
    assert shell.applied.items_width >= 100


def test_notes_list_narrows_again_once_a_note_is_open() -> None:
    """The work pane takes its width back the moment it has something in it."""
    fake, shell = _layout_screen_fake(width=WIDE[0], view="editor")

    LibraryScreen._sync_library_notes_reader_layout_from_shell(fake)

    assert shell.applied is not None
    assert shell.applied.items_width < 100
    assert shell.applied.items_width >= 60


def _folder_selected_projection() -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.folder("work"),
                kind="folder",
                label="Work",
                depth=0,
                folder_id="work",
                breadcrumb="Work",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.note("work", "n1", "m1"),
                kind="note",
                label="Quarterly plan",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Quarterly plan",
            ),
        )
    )


@pytest.mark.asyncio
def _toolbar_app(pane_width: int) -> _CanvasApp:
    """The heaviest toolbar the list can compose, at one pane width."""
    return _CanvasApp(
        pane_width=pane_width,
        list_state=_list_state(),
        tree_projection=_folder_selected_projection(),
        tree_selected_placement_id=FolderPlacementId.folder("work"),
        import_receipt_available=True,
    )


def _toolbar_rows(app) -> set[int]:
    return {
        app.query_one(selector).region.y
        for selector in (
            "#library-notes-browse-actions",
            "#library-notes-transfer-actions",
            "#library-notes-tree-actions",
        )
    }


@pytest.mark.asyncio
async def test_notes_toolbar_fits_two_rows_with_every_action_visible() -> None:
    """task-32127 AC#3: two toolbar rows at the width 235 columns produces.

    137 is what the resolver hands the list at 235 with no note open (the
    live capture measured the same); the shell is not needed to pin the
    canvas's own composition at that width.
    """
    app = _toolbar_app(137)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        rows = _toolbar_rows(app)
        assert len(rows) <= 2, f"toolbar occupies {len(rows)} rows"
        for selector in (
            "#library-notes-folder-move",
            "#library-notes-folder-remove",
            "#library-notes-import-receipt",
        ):
            assert app.query(selector), f"{selector} is not composed"
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_notes_toolbar_keeps_every_action_on_pane_beside_an_open_note() -> None:
    """task-32127 AC#2's width must still paint every action on the pane.

    62 columns is what the resolver gives the list at 235 with a note open.
    Merging the two action groups onto one row put "Last import" at
    x=53..68 of this pane (review round 1); below the merge threshold the
    groups keep their own rows and everything fits.
    """
    app = _toolbar_app(62)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_notes_toolbar_keeps_every_action_on_a_thirty_eight_column_pane() -> None:
    """task-32127 AC#2 at the narrowest pane the list is ever given.

    A 130-column terminal hands the list 44 columns beside an open note,
    and 38 is the width the delete-receipt pin already uses; the
    folder-selected toolbar is the heaviest frame there (browse +
    transfer + the four tree actions). The docs sweep saw Rename/Move/
    Remove clip here, and "Last import" clipped with them.
    """
    app = _toolbar_app(38)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        for selector in (
            "#library-notes-folder-rename",
            "#library-notes-folder-move",
            "#library-notes-folder-remove",
        ):
            assert app.query(selector), f"{selector} is not composed"
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_narrow_compact_toolbar_groups_stay_on_one_row() -> None:
    """The compact shell pins these rows to one line, so they must not stack.

    `#library-shell-grid.library-notes-compact #library-notes-transfer-actions`
    is `height: 1; overflow-x: hidden` (pinned in
    Tests/UI/test_css_build_integrity.py), so a stacked column there would be
    clipped to its first button -- worse than the off-pane overflow the
    stacking fixes.
    """
    app = _CanvasApp(
        pane_width=38,
        compact=True,
        list_state=_list_state(),
        tree_projection=_folder_selected_projection(),
        tree_selected_placement_id=FolderPlacementId.folder("work"),
        import_receipt_available=True,
    )
    async with app.run_test(size=COMPACT) as pilot:
        await pilot.pause()
        for group in (
            "#library-notes-transfer-actions",
            "#library-notes-tree-actions",
        ):
            lines = {button.region.y for button in app.query(f"{group} Button")}
            assert len(lines) == 1, f"{group} stacked onto {len(lines)} lines"


# -- task-32123: the delete receipt's recovery actions are reachable ------


#: 40 characters exactly -- the AC's title length.
RECEIPT_TITLE = "Groceries and the very long weekly plans"


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.asyncio
async def test_delete_receipt_actions_are_pressable_in_a_thirty_eight_column_pane(
    compact: bool,
) -> None:
    """task-32123 AC#1/#3: both actions survive 38 columns, compact or not."""
    assert len(RECEIPT_TITLE) == 40
    receipt = LibraryNoteDeleteReceipt(
        note_id="n1", title=RECEIPT_TITLE, expected_version=2
    )
    app = _CanvasApp(
        pane_width=38,
        list_state=_list_state(delete_receipt=receipt),
        compact=compact,
    )
    async with app.run_test(size=COMPACT if compact else WIDE) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)
        for selector in (
            "#library-notes-delete-undo",
            "#library-notes-delete-receipt-dismiss",
        ):
            button = app.query_one(selector, Button)
            assert _visible(button), f"{selector} has no region"
            assert button.region.right <= canvas.region.right, (
                f"{selector} is painted off the pane"
            )
        assert_every_action_fits(app)


# -- task-32124: Undo puts the row back in the tree ----------------------


class _RestoreService(_BranchService):
    """A branch service whose slice gains the note back once restored.

    ``parent`` is the branch the note is restored into: ``None`` for
    Unfiled, a folder id for a filed placement (AC#1 names both).
    """

    def __init__(self, parent: str | None = None) -> None:
        super().__init__()
        self.restored = False
        self.parent = parent
        self.folder_pages[None] = _folder_page(None, "ideas")
        self.placement_pages[parent] = _placement_page(parent, "loose")

    async def page_note_placements(self, **kwargs):
        if kwargs["parent_id"] == self.parent:
            self.placement_pages[self.parent] = _placement_page(
                self.parent, *(("loose", "n1") if self.restored else ("loose",))
            )
        return await super().page_note_placements(**kwargs)

    async def load_note_tree_mutation_context(self, **_kwargs):
        # What the real service reports for the restored note: the branches
        # its surviving placements live in.
        return SimpleNamespace(
            parent_ids=(),
            placement_parent_ids=(self.parent,) if self.parent else (),
            folder_ids=(self.parent,) if self.parent else (),
            ancestor_ids=(),
        )

    async def restore_note(self, **_kwargs):
        self.restored = True
        return {"id": "n1", "title": "n1", "version": 2}

    async def locate_note_tree_placement(self, **_kwargs):
        if self.parent is None:
            return NoteTreeLocation(
                placement_id=FolderPlacementId.unfiled("n1"),
                note_id="n1",
                membership_id=None,
                path=(),
                placement_offset=0,
            )
        return NoteTreeLocation(
            placement_id=FolderPlacementId.note(self.parent, "n1", "m-n1"),
            note_id="n1",
            membership_id="m-n1",
            path=(NoteTreePathStep(self.parent, None, 0),),
            placement_offset=0,
        )


@pytest.mark.parametrize("parent", [None, "ideas"])
@pytest.mark.asyncio
async def test_undo_delete_returns_the_row_to_the_tree_projection(
    monkeypatch, parent: str | None
) -> None:
    """task-32124 AC#1/#2: Undo restores the row itself, not only the count.

    Both halves of AC#1's "in its folder (or Unfiled)". The focus half is
    not asserted here: this fake stubs `_restore_library_notes_focus_identity`
    (there is no DOM), so the pin is the SELECTION the restore lands on, and
    focus is evidenced live (caps/05-undo-row-returns.txt).
    """
    service = _RestoreService(parent)
    fake = _branch_screen_fake(service)
    if parent is not None:
        fake._notes_state.tree_expanded_ids = {parent}
    fake._notes_state.delete_receipt = None
    fake._local_source_records = {"notes": ()}
    fake._local_source_counts = {"notes": 0}
    fake._library_notes_mutation_in_flight = True
    fake._library_note_delete_receipt = None
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
    fake._notes_state.tree_pending_target_placement_id = ""
    fake._notes_state.mutation_in_flight = False
    fake._notes_state.notice = ""
    fake._load_library_notes_tree_slice = MethodType(
        LibraryScreen._load_library_notes_tree_slice, fake
    )
    fake._build_library_notes_tree_projection = MethodType(
        LibraryScreen._build_library_notes_tree_projection, fake
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller._sync_library_canvas",
        lambda *_a, **kwargs: (
            kwargs["then"]() if kwargs.get("then") is not None else None
        ),
    )

    # The tree starts with the note already deleted from its branch.
    for key in (
        NotesBranchKey(None, "folders"),
        NotesBranchKey(None, "placements"),
        NotesBranchKey(parent, "placements"),
    ):
        await LibraryScreen._load_library_notes_tree_slice(
            fake, key, direction="replace", offset=0
        )
    before = LibraryScreen._build_library_notes_tree_projection(fake)
    assert all(row.note_id != "n1" for row in before.rows)

    receipt = LibraryNoteDeleteReceipt(
        note_id="n1", title="n1", expected_version=2
    )
    await LibraryNotesController._undo_library_note_delete(fake, receipt)

    after = LibraryScreen._build_library_notes_tree_projection(fake)
    restored = [row for row in after.rows if row.note_id == "n1"]
    assert restored, "the restored note never came back to the tree"
    expected = (
        FolderPlacementId.unfiled("n1")
        if parent is None
        else FolderPlacementId.note(parent, "n1", "m-n1")
    )
    assert restored[0].placement_id == expected
    assert fake._notes_state.tree_selected_placement_id == expected


async def _passthrough_service_call(call, *, isolate_in_worker=False, **kwargs):
    return await call(**kwargs)


# -- task-32128: the Sort control tells the truth -------------------------


# task-32172 retired `test_sort_control_is_absent_from_the_folder_tree`: the
# repository order is a parameter of BOTH paging and the deep-link locator
# now, so the tree offers Sort again. Its replacement, and the blocked state
# that took over for the filter window, live in
# Tests/UI/test_library_notes_riders_r_list.py.


@pytest.mark.asyncio
async def test_every_sort_option_renders_in_the_narrowest_pane() -> None:
    """task-32128 AC#3: where Sort survives, all three options are pressable."""
    state = LibraryNotesListState(
        rows=(LibraryNotesListRow("n1", "Alpha", "2h", False),),
        header_copy="Notes (1)",
        status_copy="",
        empty_copy="",
        sort_choices_visible=True,
    )
    app = _CanvasApp(pane_width=38, list_state=state)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        for selector in (
            "#library-notes-sort-newest",
            "#library-notes-sort-oldest",
            "#library-notes-sort-title",
        ):
            assert app.query(selector), f"{selector} is not composed"
        # Every action in the frame, not only the three options: the strip
        # shares this pane with the transfer actions (review round 1).
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_pressing_a_sort_option_applies_that_sort(monkeypatch) -> None:
    """task-32128 AC#3: the composed option really applies its sort value.

    The press -> apply round trip used to be pinned in the shell by
    `test_library_shell_notes_sort_opens_direct_choices_and_applies_one_value`,
    which now only asserts Sort's ABSENCE (the seeded shell always builds the
    folder tree, so it composes no Sort control to press). Nothing else
    asserted that pressing an option changes the sort, so the two real halves
    are joined here: the option Button this canvas composes, and the
    controller handler the screen routes its press to.
    """
    state = LibraryNotesListState(
        rows=(LibraryNotesListRow("n1", "Alpha", "2h", False),),
        header_copy="Notes (1)",
        status_copy="",
        empty_copy="",
        sort_choices_visible=True,
    )
    app = _CanvasApp(pane_width=38, list_state=state)
    synced: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller"
        "._sync_library_canvas",
        lambda _screen, kind, **_kwargs: synced.append(kind),
    )
    cleared: list[bool] = []
    fake = SimpleNamespace(
        _library_notes_mutation_fenced=lambda: False,
        _library_notes_sort="newest",
        _library_notes_sort_choices_visible=True,
        _library_notes_select_mode=True,
        _library_notes_row_selection=SimpleNamespace(
            clear=lambda: cleared.append(True)
        ),
        # task-32172: a new sort value re-pages the tree rather than
        # re-sorting the loaded window.
        _request_library_notes_tree_initial_load=lambda: None,
    )

    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        option = app.query_one("#library-notes-sort-oldest", Button)
        # The class the screen's `@on(Button.Pressed, ...)` selector matches:
        # without it the press never reaches the handler below.
        assert option.has_class("library-notes-sort-choice")
        LibraryNotesController.handle_library_notes_sort_choice(
            fake, Button.Pressed(option)
        )

    assert fake._library_notes_sort == "oldest"
    assert fake._library_notes_sort_choices_visible is False
    assert fake._library_notes_select_mode is False
    assert cleared == [True]
    assert synced == ["notes"]


# -- task-32137: rows carry an age and duplicates are distinguishable -----


def _placement(note_id: str, title: str, folder_id: str | None, modified: str):
    membership = _membership(f"m-{note_id}", folder_id, note_id) if folder_id else None
    return NotePlacementRecord(
        note={"id": note_id, "title": title, "last_modified": modified},
        folder_id=folder_id,
        membership=membership,
    )


def test_tree_rows_carry_a_relative_age() -> None:
    """task-32137 AC#1: a tree row knows how old its note is."""
    from dataclasses import replace

    from tldw_chatbook.Library.library_notes_tree_paging import empty_notes_slice

    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    record = _placement(
        "n1", "Reading list", None, (now - timedelta(hours=2)).isoformat()
    )
    key = NotesBranchKey(None, "placements")
    state = replace(
        empty_notes_slice(key),
        items=(record,),
        item_ids=(FolderPlacementId.unfiled("n1"),),
        total=1,
    )

    projection = build_paged_library_notes_tree(
        branch_states={key: state},
        expanded_folder_ids=set(),
        now=now,
    )

    note_rows = [row for row in projection.rows if row.kind == "note"]
    assert note_rows and note_rows[0].age_label == "2h"


@pytest.mark.asyncio
async def test_duplicate_titles_render_folder_and_age_suffixes() -> None:
    """task-32137 AC#2: two "Reading list" rows are told apart at render time."""
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.unfiled("n1"),
                kind="note",
                label="Reading list",
                depth=0,
                note_id="n1",
                breadcrumb="Unfiled / Reading list",
                age_label="2h",
            ),
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.unfiled("n2"),
                kind="note",
                label="Reading list",
                depth=0,
                note_id="n2",
                breadcrumb="Unfiled / Reading list",
                age_label="5d",
            ),
        )
    )
    app = _CanvasApp(
        pane_width=100, list_state=_list_state(), tree_projection=projection
    )
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        labels = [
            str(button.label).strip()
            for button in app.query(".library-notes-tree-note-row")
        ]

    assert labels == [
        "Reading list · Unfiled · 2h",
        "Reading list · Unfiled · 5d",
    ]


@pytest.mark.asyncio
async def test_flat_rows_render_the_same_single_line_age() -> None:
    """task-32137 AC#3: one row renderer, so the flat fallback shows age too."""
    state = LibraryNotesListState(
        rows=(LibraryNotesListRow("n1", "Q3 retro", "3m", False),),
        header_copy="Notes (1)",
        status_copy="",
        empty_copy="",
    )
    app = _CanvasApp(pane_width=100, list_state=state)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        label = str(app.query_one("#library-notes-row-0", Button).label)

    assert label == "Q3 retro · 3m"


# -- review round 2 (PR #2544) --------------------------------------------


@pytest.mark.asyncio
async def test_list_toolbar_uses_the_width_the_pane_has_after_a_round_trip() -> None:
    """The toolbar composes from THIS view's width, not the previous view's.

    Qodo review 1 (High): the canvas kwargs were built from
    ``reader_layout`` while ``_sync_library_canvas`` resolved the new
    layout afterwards, so returning from the editor composed the list for
    the editor's narrow Items pane and left the toolbar an extra row until
    some later sync. Driven through the real Back button rather than the
    resolver alone, which is what the review asked for.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        list_width = screen._notes_state.reader_layout.items_width
        # The first frame has no resolved width at all (the shell it lives
        # in is not mounted when it composes), so it must take the shape
        # that clips nothing rather than the narrowest one.
        assert _toolbar_shape(canvas.pane_width, canvas.compact) == (False, False)
        transfer = screen.query_one("#library-notes-transfer-actions")
        assert len({child.region.y for child in transfer.children}) == 1

        screen.query(".library-notes-tree-note-row").first(Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        assert screen._notes_state.reader_layout.items_width < list_width

        screen.query_one("#library-note-back", Button).press()
        # The FIRST frame the list is painted in, not a settled one: the
        # stale-geometry window this pins is one sync wide.
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)

        assert canvas.pane_width == screen._notes_state.reader_layout.items_width
        assert canvas.pane_width == list_width


@pytest.mark.asyncio
async def test_reconcile_abandons_a_tree_visit_that_ended_mid_flight() -> None:
    """Qodo review 4: a reconcile whose Notes visit ended must not reopen it.

    Unmounting clears every branch and bumps the lifecycle generation; a
    reconcile still awaiting its mutation context would otherwise start
    slice loads that capture the NEW generation and repopulate the next
    visit's tree from the old one's mutation.
    """
    service = _RestoreService(None)
    fake = _branch_screen_fake(service)
    fake._load_library_notes_tree_slice = MethodType(
        LibraryScreen._load_library_notes_tree_slice, fake
    )
    fake._build_library_notes_tree_projection = MethodType(
        LibraryScreen._build_library_notes_tree_projection, fake
    )
    fake._locate_library_notes_tree_target = MethodType(
        LibraryScreen._locate_library_notes_tree_target, fake
    )
    fake._notes_state.tree_pending_target_placement_id = ""
    await LibraryScreen._load_library_notes_tree_slice(
        fake, NotesBranchKey(None, "placements"), direction="replace", offset=0
    )

    context = service.load_note_tree_mutation_context

    async def end_the_visit(**kwargs):
        # What `_invalidate_library_notes_tree_for_unmount` does.
        fake._notes_state.tree_branches = {}
        fake._notes_state.tree_lifecycle_generation += 1
        return await context(**kwargs)

    service.load_note_tree_mutation_context = end_the_visit
    service.calls.clear()

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "note_create",
        {"note_id": "n1"},
        before=None,
        result={"id": "n1", "title": "n1"},
    )

    assert service.calls == [], "the ended visit still loaded slices"
    assert fake._notes_state.tree_branches == {}
    assert fake._notes_state.tree_selected_placement_id == ""


def _kwargs_fake(*, tree_projection, sort_choices_visible: bool):
    """The reads `_library_notes_canvas_kwargs` makes in list view."""
    return SimpleNamespace(
        _notes_state=SimpleNamespace(reader_layout=SimpleNamespace(items_width=137)),
        _library_notes_sort="newest",
        _library_notes_filter="",
        _library_notes_sort_choices_visible=sort_choices_visible,
        _library_note_import_snapshot=None,
        _library_note_import_controller=SimpleNamespace(
            snapshot=SimpleNamespace(can_revisit_receipt=False)
        ),
        _library_notes_lasting_sync_snapshot=None,
        _build_library_notes_tree_projection=lambda: tree_projection,
        _library_notes_tree_selected_placement_id="",
        _library_notes_deleted_folder_receipt=None,
        _library_notes_compact=False,
        _library_note_create_running=False,
        _library_note_create_status="",
        _library_note_load_state="",
        _library_note_load_message="",
        _library_selected_row_id="browse-notes",
        _library_notes_view="list",
        _build_library_notes_state=_list_state,
    )


# task-32172 retired `test_the_tree_taking_over_closes_the_flat_sort_chooser`
# with the behaviour it pinned: the tree arriving no longer has to close the
# chooser, because the tree composes Sort itself now. See
# Tests/UI/test_library_notes_riders_r_list.py.


def test_the_flat_list_keeps_its_open_sort_chooser() -> None:
    """The same pass leaves the flat list's chooser exactly as it was."""
    fake = _kwargs_fake(tree_projection=None, sort_choices_visible=True)

    LibraryNotesController._library_notes_canvas_kwargs(fake)

    assert fake._library_notes_sort_choices_visible is True
