from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button

from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_notes_session import (
    NoteFlushOutcome,
    NoteFlushOutcomeKind,
)
from tldw_chatbook.Library.library_notes_state import (
    LibraryNotesListRow,
    LibraryNotesListState,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.UI.Screens.library_screen import (
    LibraryScreen,
    _apply_library_row_toggle,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from Tests.UI.test_library_shell import _painted_text


def _fake(select_mode):
    fake = SimpleNamespace(
        # (wave-8 task 3) `_library_note_dirty` is NOT a `LibraryNotesState`
        # field -- it is one of the five projection `@property` members task 2
        # left screen-resident -- and the three counters are test-local, so
        # all four stay flat. They were moved above the field run so its
        # nesting stayed mechanical.
        _library_note_dirty=False,
        _refreshed=0,
        _opened=[],
        _flushed=0,
        _notes_state=SimpleNamespace(
            select_mode=select_mode,
            row_selection=RowSelection("notes"),
            selected_note_id="",
            view="list",
            # task-15790: production gained this in-flight guard; stale double.
            mutation_in_flight=False,
        ),
    )
    fake._library_notes_mutation_fenced = lambda: False
    return fake


@pytest.mark.asyncio
async def test_notes_row_select_mode_toggles_and_does_not_open_editor():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)

    async def _flush():
        fake._flushed += 1
        return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(button=SimpleNamespace(note_id="n9"), stop=lambda: None)
    await LibraryScreen.handle_library_notes_row(fake, ev)
    assert fake._notes_state.row_selection.is_selected("n9")
    assert fake._notes_state.view == "list"  # editor NOT opened
    assert fake._refreshed == 1


@pytest.mark.asyncio
async def test_rejected_note_navigation_keeps_previous_tree_identity():
    fake = _fake(False)
    fake._notes_state.tree_selected_placement_id = "placement-old"

    async def _flush():
        return NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO)

    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(
        button=SimpleNamespace(note_id="n9", placement_id="placement-new"),
        stop=lambda: None,
    )

    await LibraryScreen.handle_library_notes_row(fake, ev)

    assert fake._notes_state.tree_selected_placement_id == "placement-old"
    assert fake._notes_state.view == "list"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "controller_attr", "action_name"),
    (
        (
            "handle_library_notes_manage_sync_folders",
            "_library_notes_sync_controller",
            "refresh_roots",
        ),
        (
            "handle_library_notes_import_receipt",
            "_library_note_import_controller",
            "revisit_receipt",
        ),
    ),
)
async def test_permanent_navigator_tasks_respect_dirty_draft_veto(
    handler_name: str,
    controller_attr: str,
    action_name: str,
):
    action = Mock()
    fake = _fake(False)
    fake._notes_state.view = "editor"
    fake._library_notes_mutation_fenced = lambda: False
    setattr(fake, controller_attr, SimpleNamespace(**{action_name: action}))

    async def _flush():
        return NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO)

    fake._flush_library_note_save = _flush
    event = SimpleNamespace(stop=lambda: None)

    await getattr(LibraryScreen, handler_name)(fake, event)

    assert fake._notes_state.view == "editor"
    action.assert_not_called()


@pytest.mark.asyncio
async def test_notes_export_selected_scope():
    fake = _fake(True)
    fake._notes_state.row_selection.select_all(["n2", "n1"])

    async def _open(s):
        fake._opened.append(s)

    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_notes_export_selected(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert fake._opened == [ExportScope(kind="notes", ids=("n1", "n2"))]


def test_notes_select_all_uses_unique_note_ids_visible_in_folder_tree():
    fake = _fake(True)
    fake._build_library_notes_state = lambda: LibraryNotesListState(
        rows=(
            LibraryNotesListRow("n1", "One", "", False),
            LibraryNotesListRow("n2", "Two", "", False),
            LibraryNotesListRow("hidden", "Hidden", "", False),
        ),
        header_copy="Notes (3)",
        status_copy="",
        empty_copy="",
    )
    fake._build_library_notes_tree_projection = lambda: LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow("p1", "note", "One", 1, note_id="n1"),
            LibraryNotesTreeRow("p2", "note", "One", 1, note_id="n1"),
            LibraryNotesTreeRow("p3", "note", "Two", 1, note_id="n2"),
        )
    )
    fake.refresh = lambda **kwargs: None

    LibraryScreen.handle_library_notes_select_all(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert fake._notes_state.row_selection.ids == frozenset({"n1", "n2"})


def test_tree_selection_is_not_pruned_by_unrelated_legacy_note_page(monkeypatch):
    fake = _fake(True)
    fake._notes_state.row_selection.select_all(["tree-note"])
    fake._notes_state.filter_records = None
    fake._local_source_records = {"notes": ({"id": "legacy-note"},)}
    fake._local_source_counts = {"notes": 200}
    fake._notes_state.sort = "newest"
    fake._notes_state.filter = ""
    fake._notes_state.sort_choices_visible = False
    fake._notes_state.notice = ""
    fake._library_notes_tree_error = ""
    fake._library_notes_tree_loading = False
    fake._notes_state.delete_receipt = None
    fake._library_notes_operation_for_active_region = lambda: None
    fake._build_library_notes_tree_projection = lambda: LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                "tree-placement",
                "note",
                "Tree note",
                1,
                note_id="tree-note",
            ),
        )
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.build_library_notes_list_state",
        lambda *args, **kwargs: LibraryNotesListState(
            rows=(LibraryNotesListRow("legacy-note", "Legacy", "", False),),
            header_copy="Notes (200)",
            status_copy="",
            empty_copy="",
        ),
    )

    LibraryScreen._build_library_notes_state(fake)

    assert fake._notes_state.row_selection.ids == frozenset({"tree-note"})


# -- F-018: "Export selected" explains its disabled state -----------------


def _select_mode_notes_state(selected_count: int = 0) -> LibraryNotesListState:
    return LibraryNotesListState(
        rows=(
            LibraryNotesListRow(
                note_id="n1",
                title="First note",
                age_label="today",
                checked=False,
            ),
        ),
        header_copy="Notes (1)",
        status_copy="",
        empty_copy="",
        select_mode=True,
        selected_count=selected_count,
    )


class _NotesCanvasApp(ConsolidatedCSSApp):
    def __init__(self, selected_count: int = 0):
        super().__init__()
        self._selected_count = selected_count

    def compose(self):
        yield LibraryNotesCanvas(
            list_state=_select_mode_notes_state(self._selected_count),
            id="library-notes-canvas",
        )


class _DuplicatePlacementNotesCanvasApp(App):
    def __init__(self):
        super().__init__()
        # (wave-8 task 3) `row_selection` is a `LibraryNotesState` field now,
        # and `_apply_library_row_toggle` resolves it through the DOTTED path
        # `_notes_state.row_selection`. This duck-typed stand-in is not a
        # `LibraryScreen`, so it carries the state object itself -- the same
        # shape the media analogue in `test_library_honesty_accessibility.py`
        # took at wave-7 task 3.
        self._notes_state = SimpleNamespace(row_selection=RowSelection("notes"))

    def compose(self):
        state = LibraryNotesListState(
            rows=(
                LibraryNotesListRow(
                    note_id="n1",
                    title="Shared note",
                    age_label="today",
                    checked=False,
                ),
            ),
            header_copy="Notes (1)",
            status_copy="",
            empty_copy="",
            select_mode=True,
            selected_count=0,
        )
        projection = LibraryNotesTreeProjection(
            rows=(
                LibraryNotesTreeRow(
                    "placement-a", "note", "Shared note", 1, note_id="n1"
                ),
                LibraryNotesTreeRow(
                    "placement-b", "note", "Shared note", 1, note_id="n1"
                ),
            )
        )
        yield LibraryNotesCanvas(
            list_state=state,
            tree_projection=projection,
            id="library-notes-canvas",
        )


@pytest.mark.asyncio
async def test_toggling_duplicate_placement_updates_every_visible_checkbox():
    app = _DuplicatePlacementNotesCanvasApp()
    async with app.run_test() as pilot:
        rows = list(app.query(".library-notes-row"))
        app._notes_state.row_selection.toggle("n1")

        _apply_library_row_toggle(app, "notes", rows[0], "n1")
        await pilot.pause()

        assert all(str(row.label).startswith("☑ ") for row in rows)


@pytest.mark.asyncio
async def test_export_selected_tooltip_follows_its_disabled_state():
    """F-018: "Export selected" disabled with zero selection says WHY;
    with a selection the tooltip describes the action."""
    async with _NotesCanvasApp(selected_count=0).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-notes-export-selected", Button)
        assert export_btn.disabled is True
        assert "select" in str(export_btn.tooltip).lower()

    async with _NotesCanvasApp(selected_count=1).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-notes-export-selected", Button)
        assert export_btn.disabled is False
        assert "export" in str(export_btn.tooltip).lower()


# ---------------------------------------------------------------------------
# task-31959: the select-mode "Export selected" label must not move when the
# first selection enables it. The "○ " disabled marker is part of the label,
# so crossing 0 -> 1 selected shifted the word two cells left, right under
# the row the user had just checked (PR J padded Media's bulk row only).
# ---------------------------------------------------------------------------


class _PaintableNotesCanvasApp(_NotesCanvasApp):
    """Notes canvas mounted so its select-mode toolbar actually paints.

    The app's own ``.library-toolbar-count { width: auto; }`` rule does
    not resolve against a bare canvas mount, so the "N selected" counter
    keeps Textual's ``1fr`` default and consumes the whole row, pushing
    every action off-screen (the same runaway task-2853 fixed on the real
    screen). Restated here verbatim, because what this test measures is
    the LABEL's column, not the counter's width -- which has its own pins.
    """

    CSS = ".library-toolbar-count { width: auto; }"


def _painted_word_column(app, button, word: str) -> int:
    """Absolute column where ``word`` is painted inside ``button``."""
    painted = _painted_text(app, button.region)
    assert word in painted, (word, painted)
    return button.region.x + painted.index(word)


@pytest.mark.asyncio
async def test_notes_export_label_holds_its_column_across_the_first_selection():
    """The painted "Export" column is identical before and after selecting.

    Driven through ``_apply_library_row_toggle`` -- the in-place patch a
    row press takes, and the exact path where a padded compose-time label
    would be rebuilt unpadded.
    """
    app = _PaintableNotesCanvasApp(selected_count=0)
    # (wave-8 task 3) Same shape as `_DuplicatePlacementNotesCanvasApp` above:
    # `row_selection` is a `LibraryNotesState` field reached through the DOTTED
    # `_notes_state.row_selection` path, and this duck-typed stand-in is not a
    # `LibraryScreen`, so it carries the state object itself.
    app._notes_state = SimpleNamespace(row_selection=RowSelection("notes"))

    async with app.run_test(size=(120, 30)) as pilot:
        export = app.query_one("#library-notes-export-selected", Button)
        assert export.disabled
        before = _painted_word_column(app, export, "Export")

        row = app.query(".library-notes-row").first(Button)
        app._notes_state.row_selection.toggle("n1")
        _apply_library_row_toggle(app, "notes", row, "n1")
        await pilot.pause()

        export = app.query_one("#library-notes-export-selected", Button)
        assert not export.disabled
        after = _painted_word_column(app, export, "Export")
        assert after == before, (before, after)
