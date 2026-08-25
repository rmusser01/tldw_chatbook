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


def _fake(select_mode):
    fake = SimpleNamespace(
        _library_notes_select_mode=select_mode,
        _library_notes_row_selection=RowSelection("notes"),
        _selected_note_id="",
        _library_note_dirty=False,
        _refreshed=0,
        _opened=[],
        _flushed=0,
        _library_notes_view="list",
        # task-15790: production gained this in-flight guard; stale double.
        _library_notes_mutation_in_flight=False,
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
    assert fake._library_notes_row_selection.is_selected("n9")
    assert fake._library_notes_view == "list"  # editor NOT opened
    assert fake._refreshed == 1


@pytest.mark.asyncio
async def test_rejected_note_navigation_keeps_previous_tree_identity():
    fake = _fake(False)
    fake._library_notes_tree_selected_placement_id = "placement-old"

    async def _flush():
        return NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO)

    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(
        button=SimpleNamespace(note_id="n9", placement_id="placement-new"),
        stop=lambda: None,
    )

    await LibraryScreen.handle_library_notes_row(fake, ev)

    assert fake._library_notes_tree_selected_placement_id == "placement-old"
    assert fake._library_notes_view == "list"


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
    fake._library_notes_view = "editor"
    fake._library_notes_mutation_fenced = lambda: False
    setattr(fake, controller_attr, SimpleNamespace(**{action_name: action}))

    async def _flush():
        return NoteFlushOutcome(NoteFlushOutcomeKind.VALIDATION_VETO)

    fake._flush_library_note_save = _flush
    event = SimpleNamespace(stop=lambda: None)

    await getattr(LibraryScreen, handler_name)(fake, event)

    assert fake._library_notes_view == "editor"
    action.assert_not_called()


@pytest.mark.asyncio
async def test_notes_export_selected_scope():
    fake = _fake(True)
    fake._library_notes_row_selection.select_all(["n2", "n1"])

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

    assert fake._library_notes_row_selection.ids == frozenset({"n1", "n2"})


def test_tree_selection_is_not_pruned_by_unrelated_legacy_note_page(monkeypatch):
    fake = _fake(True)
    fake._library_notes_row_selection.select_all(["tree-note"])
    fake._library_notes_filter_records = None
    fake._local_source_records = {"notes": ({"id": "legacy-note"},)}
    fake._local_source_counts = {"notes": 200}
    fake._library_notes_sort = "newest"
    fake._library_notes_filter = ""
    fake._library_notes_sort_choices_visible = False
    fake._library_notes_notice = ""
    fake._library_notes_tree_error = ""
    fake._library_notes_tree_loading = False
    fake._library_note_delete_receipt = None
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

    assert fake._library_notes_row_selection.ids == frozenset({"tree-note"})


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
        self._library_notes_row_selection = RowSelection("notes")

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
        app._library_notes_row_selection.toggle("n1")

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
