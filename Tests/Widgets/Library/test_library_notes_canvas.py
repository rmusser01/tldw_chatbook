"""Tests for the LibraryNotesCanvas widget (list/sync modes)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    LibraryNoteSessionSnapshot,
    LibraryNotesListState,
    NormalizedDatabaseNote,
)
from tldw_chatbook.Library.library_note_import_state import (
    initial_note_import_snapshot,
    project_library_note_import_snapshot,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
    build_paged_library_notes_tree,
)
from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    empty_notes_slice,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NotePlacementRecord,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotePresentationState,
    LibraryNotesCanvas,
)

pytestmark = pytest.mark.asyncio


def _list_state() -> LibraryNotesListState:
    return LibraryNotesListState(
        rows=(),
        header_copy="Notes (0)",
        status_copy="",
        empty_copy="No notes yet. Create one to see it here.",
    )


def _editor_state(
    *,
    status: str = "Saved",
    saving: bool = False,
    transfer_status: str = "",
    transfer_running: bool = False,
    region: str = "editor",
) -> LibraryNotePresentationState:
    baseline = NormalizedDatabaseNote(
        "n-1",
        "Note",
        "Body",
        (),
        1,
        "2026-08-20T00:00:00+00:00",
        "2026-08-20T00:00:00+00:00",
    )
    snapshot = LibraryNoteSessionSnapshot(
        baseline=baseline,
        draft=DatabaseNoteDraft("n-1", "Note", "Body", "", 0),
        session_generation=1,
        saved_revision=0,
        dirty=False,
        saving=saving,
        in_conflict=False,
        conflict_generation=0,
        status_message=status,
    )
    return LibraryNotePresentationState(
        snapshot=snapshot,
        metadata_line="Updated today",
        status_line=status,
        region=region,
        transfer_status=transfer_status,
        transfer_running=transfer_running,
    )


def _tree_projection() -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="folder:personal",
                kind="folder",
                label="Personal",
                depth=0,
                folder_id="personal",
                breadcrumb="Personal",
                expanded=True,
                version=2,
            ),
            LibraryNotesTreeRow(
                placement_id="folder:ideas",
                kind="folder",
                label="Ideas",
                depth=1,
                folder_id="ideas",
                breadcrumb="Personal / Ideas",
                protected=True,
                semantic_status="connected",
                status_text="⇄ Sync managed",
                expanded=True,
                version=1,
            ),
            LibraryNotesTreeRow(
                placement_id="note:ideas:n1",
                kind="note",
                label="Garden redesign",
                depth=2,
                note_id="n1",
                folder_id="ideas",
                membership_id="m1",
                breadcrumb="Personal / Ideas / Garden redesign",
                ownership="managed",
                protected=True,
                semantic_status="connected",
                status_text="⇄ Synced placement",
            ),
            LibraryNotesTreeRow(
                placement_id="virtual:unfiled",
                kind="unfiled",
                label="Unfiled",
                depth=0,
                breadcrumb="Unfiled",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id="unfiled:n2",
                kind="note",
                label="Loose thought",
                depth=1,
                note_id="n2",
                breadcrumb="Unfiled / Loose thought",
            ),
        ),
    )


def _sync_tree_projection(
    canvas: LibraryNotesCanvas, projection: LibraryNotesTreeProjection
) -> None:
    canvas.sync_state(
        list_state=_list_state(),
        sort_mode="newest",
        filter_value="",
        mode="list",
        presentation_state=None,
        tree_projection=projection,
        tree_selected_placement_id="",
        tree_deleted_folder_available=False,
        title_placeholder_only=False,
        compact=False,
        create_running=False,
        create_status="",
        load_state="loading",
        load_message="",
    )


def _painted_frame(app: ConsolidatedCSSApp) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


# -- TASK-19000: source authority stays pinned across every Notes subview.


@pytest.mark.parametrize(
    ("mode", "kwargs", "status_fragment"),
    (
        (
            "loading",
            {"load_state": "failed", "load_message": "Could not load note."},
            "Could not load note.",
        ),
        ("list", {"list_state": _list_state()}, "Ready"),
        ("editor", {}, "Editor unavailable"),
        (
            "create",
            {"create_status": "Could not create note."},
            "Could not create note.",
        ),
        (
            "import",
            {
                "import_snapshot": project_library_note_import_snapshot(
                    initial_note_import_snapshot()
                )
            },
            "Import once",
        ),
    ),
)
async def test_authority_row_is_first_plain_child_in_every_notes_mode(
    widget_pilot,  # noqa: F811
    mode: str,
    kwargs: dict[str, object],
    status_fragment: str,
):
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode=mode,
        **kwargs,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        assert canvas.children[0] is authority
        assert authority._render_markup is False
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Library notes" in text
        assert "Library database" in text
        assert status_fragment in text
        assert "Next:" in text


async def test_completed_import_receipt_has_focusable_back_action_at_60_columns():
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="receipt",
        status_line="Import finished.",
        receipt_line="1 imported · 0 updated · 0 skipped · 0 failed",
        receipt_detail="All planned items settled.",
    )

    class ImportReceiptApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(mode="import", import_snapshot=snapshot)

    app = ImportReceiptApp()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        back = app.query_one("#library-notes-import-back", Button)
        back.focus()
        await pilot.pause()

        assert app.focused is back
        assert str(back.label) == "Back to Notes"
        assert back.disabled is False


async def test_cancelled_partial_import_offers_retry_without_calling_it_a_failure(
    widget_pilot,  # noqa: F811
):
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="receipt",
        status_line="Import stopped after the current item.",
        receipt_line="1 imported · 0 updated · 0 skipped · 0 failed",
        receipt_detail="Partial completion. Finished items were not rolled back.",
        retry_available=True,
        retry_label="Retry unfinished items",
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="import",
        import_snapshot=snapshot,
    ) as pilot:
        await pilot.pause()
        retry = pilot.app.query_one("#note-import-retry", Button)

        assert str(retry.label) == "Retry unfinished items"
        assert "failure" not in str(retry.label).casefold()


async def test_lasting_setup_retained_wrapper_preserves_input_and_pins_action_at_60x20():
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )

    class LastingSetupApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                mode="lasting_add",
                lasting_sync_snapshot=snapshot,
                compact=True,
            )

    app = LastingSetupApp()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        name = app.query_one("#notes-sync-display-name")
        assert app.focused is name
        await pilot.press(*"Research[2026]")
        folder = app.query_one("#notes-sync-folder-choose", Button)
        folder.focus()
        canvas.sync_state(
            list_state=None,
            sort_mode="newest",
            filter_value="",
            mode="lasting_add",
            presentation_state=None,
            import_snapshot=None,
            import_receipt_available=False,
            tree_projection=None,
            tree_selected_placement_id="",
            tree_deleted_folder_available=False,
            title_placeholder_only=False,
            compact=True,
            create_running=False,
            create_status="",
            load_state="loading",
            load_message="",
            lasting_sync_snapshot=replace(snapshot, status_line="Folder selected."),
        )
        await pilot.pause()

        assert app.query_one("#notes-sync-display-name") is name
        assert name.value == "Research[2026]"
        assert app.focused is folder
        primary = app.query_one("#notes-sync-check", Button)
        primary.scroll_visible(immediate=True)
        await pilot.pause()
        assert primary in app.screen._compositor.visible_widgets
        hint = app.query_one("#notes-sync-fold-hint", Static)
        assert "Additional setup content is scrollable" in str(hint.renderable)
        assert hint in app.screen._compositor.visible_widgets
        assert canvas.region.right <= 60 and canvas.region.bottom <= 20


async def test_import_selection_summary_bounds_many_long_names(widget_pilot):  # noqa: F811
    names = tuple(f"[draft]-{'x' * 80}-{number}.md" for number in range(10))
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="destination",
        selected_names=names,
        selection_kind="files",
        status_line="10 files selected.",
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="import",
        import_snapshot=snapshot,
    ) as pilot:
        await pilot.pause()
        summary = pilot.app.query_one("#note-import-source-summary", Static)
        text = str(summary.renderable)

        assert names[0][:40] in text
        assert names[2][:40] in text
        assert names[0] not in text
        assert names[3] not in text
        assert "and 7 more" in text
        assert len(text) < 200
        assert summary._render_markup is False


async def test_list_authority_running_without_status_uses_updating_fallback(
    widget_pilot,  # noqa: F811
):
    running = replace(
        _list_state(),
        operation_running=True,
        operation_status="",
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=running,
    ) as pilot:
        await pilot.pause()
        authority = pilot.app.query_one("#library-notes-authority", Static)
        text = getattr(authority.renderable, "plain", str(authority.renderable))

        assert "Updating notes…" in text
        assert "Ready" not in text
        assert "Next: Wait for the running notes operation to finish." in text


async def test_editor_authority_tracks_post_mount_save_state(widget_pilot):  # noqa: F811
    initial = _editor_state()
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="editor",
        presentation_state=initial,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        canvas.apply_session_state(_editor_state(status="Saving note…", saving=True))
        assert canvas.query_one("#library-notes-authority", Static) is authority
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Saving note…" in text
        assert "Next: Wait for saving to finish." in text

        canvas.apply_session_state(_editor_state(status="Save failed: database busy"))
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Save failed: database busy" in text
        assert "Next: Review the error, then keep editing." in text


async def test_editor_authority_tracks_transfer_through_context_navigation(
    widget_pilot,  # noqa: F811
):
    initial = _editor_state()
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="editor",
        presentation_state=initial,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        canvas.apply_session_state(
            _editor_state(
                transfer_status="Exporting Markdown…",
                transfer_running=True,
                region="context",
            )
        )
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Saved" in text
        assert "Exporting Markdown…" in text
        assert "Next: Wait for export to finish." in text

        canvas.apply_session_state(
            _editor_state(
                transfer_status="Export failed: permission denied",
                region="editor",
            )
        )
        assert canvas.query_one("#library-notes-authority", Static) is authority
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Export failed: permission denied" in text
        assert "Next: Review the error, then keep editing." in text


async def test_database_mode_list_carries_a_placement_sentence(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
    ) as pilot:
        await pilot.pause()
        purpose = pilot.app.query_one(
            "#library-notes-database-purpose", Static
        ).renderable
        text = getattr(purpose, "plain", str(purpose))
        assert "database" in text.lower()
        assert "Folder files" in text
        assert "Add from files" in text


async def test_tree_projection_renders_hierarchy_and_placement_metadata(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
    ) as pilot:
        await pilot.pause()
        folders = list(pilot.app.query(".library-notes-folder-row"))
        notes = list(pilot.app.query(".library-notes-row"))

        assert [str(button.label) for button in folders] == [
            "▾ Personal",
            "  ▾ Ideas  ⇄ Sync managed",
            "▾ Unfiled",
        ]
        assert [button.note_id for button in notes] == ["n1", "n2"]
        assert notes[0].placement_id == "note:ideas:n1"
        assert notes[0].breadcrumb == "Personal / Ideas / Garden redesign"
        assert notes[0].membership_id == "m1"
        assert notes[0].protected_placement is True
        assert "⇄ Synced placement" in str(notes[0].label)


async def test_paged_tree_projection_renders_branch_controls_at_exact_boundaries(
    widget_pilot,  # noqa: F811
):
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="folder:work",
                kind="folder",
                label="Work",
                depth=0,
                folder_id="work",
                breadcrumb="Work",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:folder:work:folders:earlier",
                kind="pager",
                label="Folders 21–40 of 83  Load earlier",
                depth=1,
                parent_folder_id="work",
                content_kind="folders",
                paging_action="earlier",
                range_copy="Folders 21–40 of 83",
                action_copy="Load earlier",
                focus_id="library-notes-tree-pager-folder-776f726b-folders-earlier",
            ),
            LibraryNotesTreeRow(
                placement_id="note:work:n1:m1",
                kind="note",
                label="[draft] Plan",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / [draft] Plan",
            ),
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:folder:work:placements:more",
                kind="pager",
                label="Notes 1–20 of 146  Loading…",
                depth=1,
                parent_folder_id="work",
                content_kind="placements",
                paging_action="more",
                range_copy="Notes 1–20 of 146",
                action_copy="Loading…",
                focus_id="library-notes-tree-pager-folder-776f726b-placements-more",
                loading=True,
                disabled=True,
            ),
        )
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
    ) as pilot:
        await pilot.pause()
        children = list(pilot.app.query_one("#library-notes-list").children)

        assert [child.id for child in children] == [
            "library-notes-tree-folder-0",
            "library-notes-tree-pager-folder-776f726b-folders-earlier",
            "library-notes-tree-note-2",
            "library-notes-tree-pager-folder-776f726b-placements-more",
        ]
        earlier = children[1]
        loading = children[3]
        assert earlier.has_class("library-notes-tree-pager")
        assert earlier.parent_folder_id == "work"
        assert earlier.content_kind == "folders"
        assert earlier.paging_action == "earlier"
        assert str(earlier.label) == "  Folders 21–40 of 83  Load earlier"
        assert earlier.disabled is False
        earlier.focus()
        await pilot.pause()
        assert pilot.app.focused is earlier
        assert loading.parent_folder_id == "work"
        assert loading.content_kind == "placements"
        assert loading.paging_action == "more"
        assert str(loading.label) == "  Notes 1–20 of 146  Loading…"
        assert loading.disabled is True
        assert not pilot.app.query("#library-notes-tree-more")
        loading.focus()
        await pilot.pause()
        assert pilot.app.focused is earlier
        assert "[draft] Plan" in str(children[2].label)


async def test_tree_pager_renders_copy_as_plain_text_and_retry_stays_focusable(
    widget_pilot,  # noqa: F811
):
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:root:placements:retry",
                kind="pager",
                label="[red]20 placements loaded[/red] · May be out of date · Retry",
                depth=1,
                parent_folder_id=None,
                content_kind="placements",
                paging_action="retry",
                status_text="20 placements loaded · May be out of date",
                action_copy="Retry",
                focus_id="library-notes-tree-pager-root-placements-retry",
            ),
        )
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
    ) as pilot:
        await pilot.pause()
        retry = pilot.app.query_one(
            "#library-notes-tree-pager-root-placements-retry", Button
        )

        assert str(retry.label) == (
            "  [red]20 placements loaded[/red] · May be out of date · Retry"
        )
        assert retry.label.spans == []
        assert retry.paging_action == "retry"
        assert retry.disabled is False
        retry.focus()
        await pilot.pause()
        assert pilot.app.focused is retry


async def test_stale_selected_placement_disables_mutations_but_not_open_or_retry(
    widget_pilot,  # noqa: F811
):
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="note:work:n1:m1",
                kind="note",
                label="Affected",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Affected",
                unsafe_mutation_disabled=True,
            ),
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:folder:work:placements:retry",
                kind="pager",
                label="1 placement loaded · May be out of date · Retry",
                depth=1,
                parent_folder_id="work",
                content_kind="placements",
                paging_action="retry",
                focus_id="library-notes-tree-pager-folder-776f726b-placements-retry",
            ),
        )
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
        tree_selected_placement_id="note:work:n1:m1",
    ) as pilot:
        await pilot.pause()
        selected = pilot.app.query_one(".library-notes-tree-note-row", Button)
        retry = pilot.app.query_one(".library-notes-tree-pager", Button)

        assert selected.disabled is False
        assert retry.disabled is False
        for button_id in (
            "#library-notes-placement-add",
            "#library-notes-placement-move",
            "#library-notes-placement-remove",
        ):
            action = pilot.app.query_one(button_id, Button)
            assert action.disabled is True
            assert "out of date" in str(action.tooltip).lower()


@pytest.mark.parametrize(
    ("selected_id", "protected", "owner_active", "expected_disabled"),
    (
        ("collapsed", True, True, True),
        ("inactive", True, False, True),
        ("normal", False, True, False),
    ),
)
async def test_authoritative_managed_folder_state_gates_empty_folder_mutations(
    widget_pilot,  # noqa: F811
    selected_id: str,
    protected: bool,
    owner_active: bool,
    expected_disabled: bool,
):
    folders = (
        NoteFolder(
            "collapsed", None, "Collapsed", "/Collapsed", "/collapsed", 1, False
        ),
        NoteFolder("inactive", None, "Inactive", "/Inactive", "/inactive", 1, False),
        NoteFolder("normal", None, "Normal", "/Normal", "/normal", 1, False),
    )
    root_key = NotesBranchKey(None, "folders")
    root = replace(
        empty_notes_slice(root_key),
        items=folders,
        item_ids=tuple(
            FolderPlacementId.folder(folder.folder_id) for folder in folders
        ),
        total=3,
        freshness="fresh",
    )
    projection = build_paged_library_notes_tree(
        branch_states={root_key: root},
        expanded_folder_ids={"inactive"},
        protected_folder_ids=frozenset({"collapsed", "inactive"}),
        inactive_managed_folder_ids=frozenset({"inactive"}),
    )
    selected = projection.row(FolderPlacementId.folder(selected_id))
    assert selected is not None
    assert selected.protected is protected
    assert selected.owner_active is owner_active

    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
        tree_selected_placement_id=FolderPlacementId.folder(selected_id),
    ) as pilot:
        await pilot.pause()
        for button_id in (
            "#library-notes-folder-rename",
            "#library-notes-folder-move",
            "#library-notes-folder-remove",
        ):
            assert pilot.app.query_one(button_id, Button).disabled is expected_disabled


async def test_stale_retry_loading_is_disabled_and_cannot_emit_duplicate_press() -> (
    None
):
    key = NotesBranchKey(None, "placements")
    record = NotePlacementRecord({"id": "n1", "title": "One"}, None, None)
    state = replace(
        empty_notes_slice(key),
        items=(record,),
        item_ids=(FolderPlacementId.unfiled("n1"),),
        total=None,
        freshness="stale",
        loading=True,
        requested_direction="replace",
    )
    projection = build_paged_library_notes_tree(
        branch_states={key: state}, expanded_folder_ids=set()
    )

    class PagerApp(ConsolidatedCSSApp):
        pager_presses = 0

        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                list_state=_list_state(), tree_projection=projection
            )

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.has_class("library-notes-tree-pager"):
                self.pager_presses += 1

    app = PagerApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        pager = app.query_one(".library-notes-tree-pager", Button)

        assert str(pager.label) == (
            "  1 placement loaded · May be out of date · Loading…"
        )
        assert pager.disabled is True
        assert pager.paging_loading is True
        pager.press()
        await pilot.pause()
        assert app.pager_presses == 0


@pytest.mark.parametrize(
    ("failed_direction", "expected_label", "expected_suffix"),
    (
        ("previous", "Couldn’t load earlier · Retry", "earlier"),
        ("more", "Couldn’t load more · Retry", "more"),
    ),
)
async def test_failed_branch_retry_keeps_exact_direction_metadata(
    widget_pilot,  # noqa: F811
    failed_direction: str,
    expected_label: str,
    expected_suffix: str,
):
    key = NotesBranchKey(None, "placements")
    record = NotePlacementRecord({"id": "n1", "title": "One"}, None, None)
    state = replace(
        empty_notes_slice(key),
        items=(record,),
        item_ids=(FolderPlacementId.unfiled("n1"),),
        total=3,
        start_offset=1,
        previous_offset=0,
        next_offset=2,
        freshness="fresh",
        failed_direction=failed_direction,
        error="Page request failed.",
    )
    projection = build_paged_library_notes_tree(
        branch_states={key: state}, expanded_folder_ids=set()
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
    ) as pilot:
        await pilot.pause()
        retry = next(
            button
            for button in pilot.app.query(".library-notes-tree-pager")
            if button.paging_action == "retry"
        )

        assert str(retry.label) == f"  {expected_label}"
        assert retry.retry_direction == failed_direction
        assert retry.id.endswith(expected_suffix)


async def test_pager_focus_survives_failure_retry_and_retry_loading_recompose() -> None:
    key = NotesBranchKey(None, "placements")
    record = NotePlacementRecord({"id": "n1", "title": "One"}, None, None)
    middle = replace(
        empty_notes_slice(key),
        items=(record,),
        item_ids=(FolderPlacementId.unfiled("n1"),),
        total=3,
        start_offset=1,
        previous_offset=0,
        next_offset=2,
        freshness="fresh",
    )

    def project(state):
        return build_paged_library_notes_tree(
            branch_states={key: state}, expanded_folder_ids=set()
        )

    idle = project(middle)
    failed = project(
        replace(
            middle,
            failed_direction="more",
            error="Page request failed.",
        )
    )
    retry_loading = project(
        replace(
            middle,
            loading=True,
            requested_direction="more",
        )
    )

    class PagerFocusApp(ConsolidatedCSSApp):
        CSS_PATH = str(BUNDLED_STYLESHEET)

        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(list_state=_list_state(), tree_projection=idle)

    app = PagerFocusApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        boundary = next(
            button
            for button in app.query(".library-notes-tree-pager")
            if button.paging_action == "more"
        )
        stable_id = boundary.id
        boundary.focus()
        await pilot.pause()
        assert app.focused is boundary

        _sync_tree_projection(canvas, failed)
        await pilot.pause()
        await pilot.pause()
        retry = app.query_one(f"#{stable_id}", Button)
        assert retry.paging_action == "retry"
        assert app.focused is retry

        _sync_tree_projection(canvas, retry_loading)
        await pilot.pause()
        await pilot.pause()
        loading = app.query_one(f"#{stable_id}", Button)
        assert loading.paging_loading is True
        assert loading.disabled is True
        assert app.focused is loading


def _compact_pager_projection() -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:root:placements:replace",
                kind="pager",
                label="20 placements loaded · May be out of date · Retry",
                depth=0,
                content_kind="placements",
                paging_action="retry",
                retry_direction="replace",
                focus_id="library-notes-tree-pager-root-placements-replace",
            ),
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:folder:work:placements:earlier",
                kind="pager",
                label="Notes 201–220 of 400  Load earlier",
                depth=1,
                parent_folder_id="work",
                content_kind="placements",
                paging_action="earlier",
                focus_id="library-notes-tree-pager-folder-776f726b-placements-earlier",
            ),
            LibraryNotesTreeRow(
                placement_id="pager:notes-tree:folder:work:placements:more",
                kind="pager",
                label="Notes 201–220 of 400  Load more notes",
                depth=1,
                parent_folder_id="work",
                content_kind="placements",
                paging_action="more",
                focus_id="library-notes-tree-pager-folder-776f726b-placements-more",
            ),
        )
    )


class _CompactPagerApp(ConsolidatedCSSApp):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        shell = Vertical(id="library-shell-grid", classes="library-notes-compact")
        shell.styles.width = 40
        shell.styles.height = 24
        with shell:
            yield LibraryNotesCanvas(
                list_state=_list_state(),
                tree_projection=_compact_pager_projection(),
                compact=True,
            )


async def test_compact_pagers_paint_full_wrapped_copy_at_80x24() -> None:
    app = _CompactPagerApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        painted = " ".join(_painted_frame(app).split())
        pagers = list(app.query(".library-notes-tree-pager"))

        assert "20 placements loaded · May be out of date · Retry" in painted
        assert "Notes 201–220 of 400 Load earlier" in painted
        assert "Notes 201–220 of 400 Load more notes" in painted
        assert pagers[0].region.height >= 2
        assert pagers[-1].region.height >= 2
        assert all(
            pager.region.width <= app.query_one("#library-notes-list").region.width
            for pager in pagers
        )


async def test_nested_pager_paints_projected_depth_indentation() -> None:
    app = _CompactPagerApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        root = app.query_one(
            "#library-notes-tree-pager-root-placements-replace", Button
        )
        nested = app.query_one(
            "#library-notes-tree-pager-folder-776f726b-placements-earlier", Button
        )
        lines = _painted_frame(app).splitlines()
        root_line = next(line for line in lines if "20 placements" in line)
        nested_line = next(line for line in lines if "Notes 201–220" in line)

        assert root.region.x == nested.region.x
        assert root.region.width == nested.region.width
        assert not root.label.plain.startswith("  ")
        assert nested.label.plain.startswith("  ")
        assert (
            nested_line.index("Notes 201–220") >= root_line.index("20 placements") + 2
        )


async def test_tree_managed_state_remains_legible_without_color(widget_pilot):  # noqa: F811
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="note:work:n1",
                kind="note",
                label="Recovered",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Recovered",
                ownership="managed",
                owner_active=False,
                protected=True,
                semantic_status="needs_attention",
                status_text="! Needs owner review",
            ),
        )
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
    ) as pilot:
        await pilot.pause()
        row = pilot.app.query_one(".library-notes-row", Button)
        assert "! Needs owner review" in str(row.label)
        assert row.has_class("library-notes-tree-needs-attention")


async def test_selected_folder_exposes_manual_folder_actions(widget_pilot):  # noqa: F811
    projection = _tree_projection()
    manual_rows = tuple(
        replace(row, protected=False, semantic_status="normal", status_text="")
        if row.placement_id == "folder:ideas"
        else row
        for row in projection.rows
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=LibraryNotesTreeProjection(rows=manual_rows),
        tree_selected_placement_id="folder:ideas",
    ) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#library-notes-folder-new", Button)
        assert (
            pilot.app.query_one("#library-notes-folder-rename", Button).disabled
            is False
        )
        assert (
            pilot.app.query_one("#library-notes-folder-move", Button).disabled is False
        )
        assert (
            pilot.app.query_one("#library-notes-folder-remove", Button).disabled
            is False
        )


async def test_selected_managed_folder_disables_folder_mutations(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="folder:ideas",
    ) as pilot:
        await pilot.pause()
        for button_id in (
            "#library-notes-folder-new",
            "#library-notes-folder-rename",
            "#library-notes-folder-move",
            "#library-notes-folder-remove",
        ):
            button = pilot.app.query_one(button_id, Button)
            assert button.disabled is True
            assert "sync" in str(button.tooltip).lower()


async def test_managed_placement_disables_move_and_remove_but_allows_add(
    widget_pilot,  # noqa: F811
):
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="note:ideas:n1",
    ) as pilot:
        await pilot.pause()
        assert (
            pilot.app.query_one("#library-notes-placement-add", Button).disabled
            is False
        )
        assert (
            pilot.app.query_one("#library-notes-placement-move", Button).disabled
            is True
        )
        remove = pilot.app.query_one("#library-notes-placement-remove", Button)
        assert remove.disabled is True
        assert "sync" in str(remove.tooltip).lower()


async def test_deleted_folder_receipt_exposes_restore_action(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_deleted_folder_available=True,
    ) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#library-notes-folder-restore", Button)
