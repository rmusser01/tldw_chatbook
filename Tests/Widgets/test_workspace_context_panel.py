from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Collapsible, Input, Label, ListView, Static, TextArea

from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from tldw_chatbook.Widgets.Note_Widgets.workspace_context_panel import WorkspaceContextPanel


def _list_item_text(list_view: ListView) -> str:
    return str(list_view.children[0].query_one(Label).renderable)


class PanelTestApp(App[None]):
    def __init__(self, widget):
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


@pytest.mark.asyncio
async def test_workspace_context_panel_shows_four_subviews():
    panel = WorkspaceContextPanel()
    assert list(panel.SUBVIEWS) == [
        "workspace-details",
        "workspace-notes",
        "workspace-sources",
        "workspace-artifacts",
    ]


@pytest.mark.asyncio
async def test_workspace_context_panel_keeps_all_sections_visible_after_mount():
    panel = WorkspaceContextPanel(workspace={"name": "Research"})
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        for view_id in WorkspaceContextPanel.SUBVIEWS:
            section = panel.query_one(f"#{view_id}", Container)
            assert section.display is not False
            assert str(section.styles.display) != "none"


@pytest.mark.asyncio
async def test_workspace_context_panel_renders_workspace_details_fields():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        panel.set_workspace_details(
            {
                "name": "Research Workspace",
                "study_materials_policy": "curated",
                "archived": True,
            }
        )
        workspace_name = panel.query_one("#workspace-name", Label)
        workspace_summary = panel.query_one("#workspace-summary", Label)

        assert str(workspace_name.renderable) == "Research Workspace"
        assert str(workspace_summary.renderable) == "Policy: curated | Archived: True"


@pytest.mark.asyncio
async def test_workspace_context_panel_exposes_open_study_action_only_in_details():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        open_study_button = panel.query_one("#workspace-open-study-button", Button)

        assert str(open_study_button.label) == "Open Study"
        assert panel.query_one("#workspace-save-button", Button)
        assert panel.query_one("#workspace-add-source-button", Button)
        assert panel.query_one("#workspace-save-source-button", Button)
        assert panel.query_one("#workspace-create-artifact-button", Button)
        assert panel.query_one("#workspace-save-artifact-button", Button)
        assert open_study_button.parent.id == "workspace-details"


@pytest.mark.asyncio
async def test_workspace_context_panel_exposes_use_in_chat_actions():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        assert panel.query_one("#workspace-use-in-chat-button", Button)
        assert panel.query_one("#workspace-source-use-in-chat-button", Button)
        assert panel.query_one("#workspace-artifact-use-in-chat-button", Button)


@pytest.mark.asyncio
async def test_notes_sidebar_right_exposes_use_in_chat_action():
    sidebar = NotesSidebarRight()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        assert sidebar.query_one("#notes-use-in-chat-button", Button)


@pytest.mark.asyncio
async def test_workspace_context_panel_populates_workspace_lists():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        await panel.populate_workspace_notes([{"id": 1, "title": "Workspace Note", "version": 2}])
        await panel.populate_workspace_sources([{"id": "src-1", "title": "Source A", "version": 1}])
        await panel.populate_workspace_artifacts([{"id": "art-1", "title": "Artifact A", "version": 3}])

        notes_list = panel.query_one("#workspace-notes-list", ListView)
        sources_list = panel.query_one("#workspace-sources-list", ListView)
        artifacts_list = panel.query_one("#workspace-artifacts-list", ListView)

        assert len(notes_list.children) == 1
        assert len(sources_list.children) == 1
        assert len(artifacts_list.children) == 1


@pytest.mark.asyncio
async def test_notes_sidebar_left_scope_population_keeps_lists_separate():
    sidebar = NotesSidebarLeft()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        await sidebar.populate_notes_list([{"id": 1, "title": "Local A", "version": 1}])
        await sidebar.populate_server_notes_list([{"id": "srv-1", "title": "Server A", "version": 2}])
        await sidebar.populate_workspaces_list([{"id": "ws-1", "name": "Workspace A", "version": 1}])

        local_list = sidebar.query_one("#notes-list-view", ListView)
        server_list = sidebar.query_one("#server-notes-list-view", ListView)
        workspace_list = sidebar.query_one("#workspaces-list-view", ListView)

        assert len(local_list.children) == 1
        assert len(server_list.children) == 1
        assert len(workspace_list.children) == 1

        local_item = local_list.children[0]
        server_item = server_list.children[0]
        workspace_item = workspace_list.children[0]

        assert getattr(local_item, "note_id") == 1
        assert getattr(server_item, "note_id") == "srv-1"
        assert getattr(workspace_item, "workspace_id") == "ws-1"


@pytest.mark.asyncio
async def test_notes_sidebar_left_empty_states_explain_scope_and_creation_routes():
    sidebar = NotesSidebarLeft()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        await sidebar.populate_local_notes_list([])
        await sidebar.populate_server_notes_list([])
        await sidebar.populate_workspaces_list([])

        local_text = _list_item_text(sidebar.query_one("#notes-list-view", ListView))
        server_text = _list_item_text(sidebar.query_one("#server-notes-list-view", ListView))
        workspace_text = _list_item_text(sidebar.query_one("#workspaces-list-view", ListView))

        assert "No local notes yet." in local_text
        assert "Create Blank Note" in local_text
        assert "Import Note" in local_text
        assert "No server notes yet." in server_text
        assert "Server Notes" in server_text
        assert "Create Server Note" in server_text
        assert "No workspaces yet." in workspace_text
        assert "Create Workspace" in workspace_text
        assert "notes, sources, artifacts" in workspace_text


@pytest.mark.asyncio
async def test_notes_sidebar_left_server_empty_state_exposes_scope_transition_marker():
    sidebar = NotesSidebarLeft()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        await sidebar.populate_server_notes_list([])

        server_item = sidebar.query_one("#server-notes-list-view", ListView).children[0]

        assert hasattr(server_item, "note_id")
        assert server_item.note_id is None
        assert server_item.note_scope == "server"


@pytest.mark.asyncio
async def test_workspace_context_panel_empty_states_explain_workspace_resource_routes():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        await panel.populate_workspace_notes([])
        await panel.populate_workspace_sources([])
        await panel.populate_workspace_artifacts([])

        notes_text = _list_item_text(panel.query_one("#workspace-notes-list", ListView))
        sources_text = _list_item_text(panel.query_one("#workspace-sources-list", ListView))
        artifacts_text = _list_item_text(panel.query_one("#workspace-artifacts-list", ListView))

        assert "No workspace notes yet." in notes_text
        assert "Create Workspace Note" in notes_text
        assert "No workspace sources yet." in sources_text
        assert "Add Source" in sources_text
        assert "Media" in sources_text
        assert "No workspace artifacts yet." in artifacts_text
        assert "Create Artifact" in artifacts_text
        assert "outputs" in artifacts_text


@pytest.mark.asyncio
async def test_workspace_context_panel_empty_note_state_exposes_subview_transition_marker():
    panel = WorkspaceContextPanel()
    app = PanelTestApp(panel)

    async with app.run_test() as pilot:
        await panel.populate_workspace_notes([])

        notes_item = panel.query_one("#workspace-notes-list", ListView).children[0]

        assert hasattr(notes_item, "note_id")
        assert notes_item.note_id is None
        assert notes_item.item_version is None


@pytest.mark.asyncio
async def test_notes_sidebar_right_apply_scope_context_updates_workspace_artifact_actions():
    sidebar = NotesSidebarRight()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        sidebar.apply_scope_context(scope_type="workspace", resource_kind="artifact")

        title = sidebar.query_one("#notes-details-sidebar-title", Static)
        save_button = sidebar.query_one("#notes-save-current-button", Button)
        export_actions = sidebar.query_one("#notes-export-actions", Collapsible)
        delete_button = sidebar.query_one("#notes-delete-button", Button)
        title_input = sidebar.query_one("#notes-title-input", Input)

        assert str(title.renderable) == "Workspace Artifact Details"
        assert save_button.display is False
        assert export_actions.display is False
        assert title_input.display is False
        assert str(delete_button.label) == "Delete Selected Artifact"


@pytest.mark.asyncio
async def test_notes_sidebar_right_workspace_note_context_uses_workspace_delete_label():
    sidebar = NotesSidebarRight()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        sidebar.apply_scope_context(scope_type="workspace", resource_kind="note")

        delete_button = sidebar.query_one("#notes-delete-button", Button)
        title = sidebar.query_one("#notes-details-sidebar-title", Static)

        assert str(title.renderable) == "Workspace Note Details"
        assert str(delete_button.label) == "Delete Workspace Note"


@pytest.mark.asyncio
async def test_notes_sidebar_right_normalizes_canonical_note_scope_titles():
    sidebar = NotesSidebarRight()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        sidebar.apply_scope_context(scope_type="server_note", resource_kind="note")
        title = sidebar.query_one("#notes-details-sidebar-title", Static)
        assert str(title.renderable) == "Server Note Details"

        sidebar.apply_scope_context(scope_type="local_note", resource_kind="note")
        assert str(title.renderable) == "Local Note Details"


@pytest.mark.asyncio
async def test_notes_sidebar_right_workspace_details_context_hides_note_editor_controls():
    sidebar = NotesSidebarRight()
    app = PanelTestApp(sidebar)

    async with app.run_test() as pilot:
        sidebar.apply_scope_context(scope_type="workspace", resource_kind="workspace")

        title = sidebar.query_one("#notes-details-sidebar-title", Static)
        delete_button = sidebar.query_one("#notes-delete-button", Button)
        export_actions = sidebar.query_one("#notes-export-actions", Collapsible)
        title_input = sidebar.query_one("#notes-title-input", Input)
        keywords_area = sidebar.query_one("#notes-keywords-area", TextArea)
        save_button = sidebar.query_one("#notes-save-current-button", Button)

        assert str(title.renderable) == "Workspace Details"
        assert str(delete_button.label) == "Delete Workspace"
        assert export_actions.display is False
        assert title_input.display is False
        assert keywords_area.display is False
        assert save_button.display is False
