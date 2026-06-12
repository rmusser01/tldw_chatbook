from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Collapsible, Input, Label, ListView, Static, TextArea

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


