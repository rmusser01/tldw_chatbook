"""Mounted behavior of the responsive Research Workspace shell."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

pytestmark = pytest.mark.bootstrap_profile


class _WorkspaceHarness(ConsolidatedCSSApp):
    async def on_mount(self) -> None:
        from tldw_chatbook.UI.Screens.research_workspace_screen import (
            ResearchWorkspaceScreen,
        )

        await self.push_screen(ResearchWorkspaceScreen(SimpleNamespace()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_workspace", "retryable"),
    (
        ("archived origin", "Restored workspace", False),
        ("missing run", "Run origin", False),
        ("unsupported run owner", "Run origin", False),
        ("connection error", "Run origin", True),
    ),
)
async def test_rejected_run_returns_render_catalog_and_retain_only_transient_claims(
    tmp_path, failure, expected_workspace, retryable
) -> None:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
    from tldw_chatbook.Research_Workspace.controller import ResearchWorkspaceController
    from tldw_chatbook.Research_Workspace.deep_research_context import (
        ResearchWorkspaceRunRef,
    )
    from tldw_chatbook.Research_Workspace.local_adapter import (
        LocalResearchWorkspaceAdapter,
    )
    from tldw_chatbook.Research_Workspace.overlay_store import (
        ResearchPresentationOverlayStore,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
    from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError

    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    registry = LocalWorkspaceRegistryService(db)
    origin = QualifiedWorkspaceRef("local", "run-origin")
    restored = QualifiedWorkspaceRef("local", "restored-workspace")
    registry.create_workspace(workspace_id=origin.workspace_id, name="Run origin")
    registry.create_workspace(
        workspace_id=restored.workspace_id, name="Restored workspace"
    )
    registry.create_workspace(
        workspace_id="another-workspace", name="Another workspace"
    )
    if failure == "archived origin":
        registry.archive_workspace(origin.workspace_id)

    async def get_run(run_id, *, mode):
        assert (run_id, mode) == ("run-id", "local")
        if failure == "connection error":
            raise APIConnectionError("Run store temporarily unavailable")
        if failure == "unsupported run owner":
            raise APIResponseError(501, "Run owner not implemented")
        return None

    class CountingCatalog(LocalResearchWorkspaceAdapter):
        reads = 0

        async def list_workspaces(self, *, include_archived=False):
            self.reads += 1
            return await super().list_workspaces(include_archived=include_archived)

    port = CountingCatalog(registry)
    store = PendingHandoffStore()
    channel = HandoffChannel.RESEARCH_WORKSPACE_RETURN
    target = ResearchWorkspaceRunRef(origin, "launch-id", "run-id")
    store.stage(channel, target)
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(
            pending_handoffs=store,
            research_scope_service=SimpleNamespace(get_run=get_run),
        ),
        controller=ResearchWorkspaceController({origin.data_source: port}),
        overlay_store=ResearchPresentationOverlayStore(tmp_path / "overlay.json"),
    )
    screen.restore_state(
        {"research_workspace_ref": {"workspace_id": restored.workspace_id}}
    )

    class Harness(ConsolidatedCSSApp):
        async def on_mount(self) -> None:
            await self.push_screen(screen)

    try:
        app = Harness()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            selection = screen.query_one("#research-workspace-selection", Static)
            assert expected_workspace in str(selection.render())
            assert screen.controller.selected_ref == (
                restored if failure == "archived origin" else origin
            )
            assert store.has_pending(channel) is retryable
            assert port.reads == 1
            if failure == "archived origin":
                assert any(
                    "origin workspace is unavailable" in notice.message
                    for notice in app._notifications
                )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_workspace_composes_once_with_honest_regions_and_exact_handles() -> None:
    from tldw_chatbook.UI.Research_Workspace_Modules.chat_region import (
        ResearchChatRegion,
    )
    from tldw_chatbook.UI.Research_Workspace_Modules.sources_region import (
        ResearchSourcesRegion,
    )
    from tldw_chatbook.UI.Research_Workspace_Modules.studio_region import (
        ResearchStudioRegion,
    )

    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert len(list(screen.query(ResearchSourcesRegion))) == 1
        assert len(list(screen.query(ResearchChatRegion))) == 1
        assert len(list(screen.query(ResearchStudioRegion))) == 1
        assert len(list(screen.query("#research-workspace-status"))) == 1

        expected = {
            "research-sources-collapse": ("<---", "Collapse Sources pane"),
            "research-sources-reveal": ("--->", "Expand Sources pane"),
            "research-studio-collapse": ("--->", "Collapse Studio pane"),
            "research-studio-reveal": ("<---", "Expand Studio pane"),
        }
        for widget_id, (label, accessible_name) in expected.items():
            button = screen.query_one(f"#{widget_id}", Button)
            assert str(button.label) == label
            assert button.tooltip == accessible_name
            assert button.name == accessible_name

        labels = {str(button.label) for button in screen.query(Button)}
        assert "Generate" not in labels
        assert screen.query_one("#research-chat-send", Button).disabled
        assert not screen.query_one("#research-chat-controls").display
        assert "Add" in labels  # Quick URL intake is now a real implemented action.


@pytest.mark.asyncio
async def test_collapse_and_expand_move_focus_and_remove_hidden_pane_from_cycle() -> (
    None
):
    from tldw_chatbook.UI.Research_Workspace_Modules.sources_region import (
        ResearchSourcesRegion,
    )

    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        sources = screen.query_one("#research-sources-pane", ResearchSourcesRegion)
        sources.focus()
        screen.query_one("#research-sources-collapse", Button).press()
        await pilot.pause()

        reveal = screen.query_one("#research-sources-reveal", Button)
        assert not sources.display
        assert reveal.display
        assert app.focused is reveal

        reveal.press()
        await pilot.pause()
        assert sources.display
        assert not reveal.display
        assert app.focused is sources


@pytest.mark.asyncio
async def test_medium_companion_switch_preserves_wide_preferences_and_widget_identity() -> (
    None
):
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert isinstance(screen, ResearchWorkspaceScreen)
        sources = screen.query_one("#research-sources-pane")
        chat = screen.query_one("#research-chat-pane")
        studio = screen.query_one("#research-studio-pane")

        await pilot.resize_terminal(120, 30)
        await pilot.pause()
        assert sources.display and chat.display and not studio.display

        screen.query_one("#research-pane-mode-chat", Button).press()
        await pilot.pause()
        assert sources.display and chat.display and not studio.display
        assert screen.pane_preferences.sources_open
        assert screen.pane_preferences.studio_open

        screen.query_one("#research-pane-mode-studio", Button).press()
        await pilot.pause()
        assert not sources.display and chat.display and studio.display
        assert screen.pane_preferences.sources_open
        assert screen.pane_preferences.studio_open

        await pilot.resize_terminal(160, 40)
        await pilot.pause()
        assert sources.display and chat.display and studio.display
        assert screen.query_one("#research-sources-pane") is sources
        assert screen.query_one("#research-chat-pane") is chat
        assert screen.query_one("#research-studio-pane") is studio


@pytest.mark.asyncio
async def test_narrow_mode_has_one_pane_and_resize_relocates_hidden_focus() -> None:
    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        studio = screen.query_one("#research-studio-pane")
        studio.focus()

        await pilot.resize_terminal(84, 24)
        await pilot.pause()
        pane_buttons = screen.query_one("#research-pane-mode-strip")
        assert pane_buttons.display
        assert [
            pane_id
            for pane_id in ("sources", "chat", "studio")
            if screen.query_one(f"#research-{pane_id}-pane").display
        ] == ["chat"]
        assert app.focused is screen.query_one("#research-pane-mode-studio", Button)
        assert not screen.query_one("#research-sources-handle").display
        assert not screen.query_one("#research-studio-handle").display

        screen.query_one("#research-pane-mode-sources", Button).press()
        await pilot.pause()
        assert screen.query_one("#research-sources-pane").display
        assert not screen.query_one("#research-chat-pane").display
        assert app.focused is screen.query_one("#research-sources-pane")


@pytest.mark.asyncio
async def test_medium_reflow_moves_focus_from_swapped_handle_to_pane_mode() -> None:
    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        studio_collapse = screen.query_one("#research-studio-collapse", Button)
        studio_collapse.focus()
        await pilot.pause()
        assert app.focused is studio_collapse

        await pilot.resize_terminal(120, 30)
        await pilot.pause()

        assert not studio_collapse.display
        assert app.focused is screen.query_one("#research-pane-mode-studio", Button)
        assert any(
            "Studio pane is hidden" in notification.message
            for notification in app._notifications
        )


@pytest.mark.asyncio
async def test_narrow_reflow_hides_handle_children_and_moves_focus_to_pane_mode() -> (
    None
):
    app = _WorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        sources_collapse = screen.query_one("#research-sources-collapse", Button)
        sources_collapse.focus()
        await pilot.pause()
        assert app.focused is sources_collapse

        await pilot.resize_terminal(84, 24)
        await pilot.pause()

        handle = screen.query_one("#research-sources-handle")
        assert not handle.display
        assert not screen.query_one("#research-sources-collapse", Button).display
        assert not screen.query_one("#research-sources-reveal", Button).display
        assert app.focused is screen.query_one("#research-pane-mode-sources", Button)
        assert any(
            "Sources pane is hidden" in notification.message
            for notification in app._notifications
        )
