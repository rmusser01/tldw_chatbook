"""Post-release Workspaces and Library depth mounted regressions."""

from __future__ import annotations

import pytest
from textual.widgets import Button
from textual.widgets import Static

from tldw_chatbook.UI.Screens import library_screen as library_screen_module

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_selector,
)


def _seed_cross_workspace_library(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [
            {"title": "Workspace B research note", "id": "note-cross"},
            {"title": "Workspace A field note", "id": "note-local"},
        ]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Workspace A transcript", "id": "media-local"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Workspace B planning chat", "id": "chat-cross"}]
    )
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-a",
        name="Workspace A",
    )
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-b",
        name="Workspace B",
    )
    app.workspace_registry_service.set_active_workspace("workspace-a")
    app.workspace_registry_service.link_membership(
        "workspace-b",
        item_type="note",
        item_id="note-cross",
        title="Workspace B research note",
    )
    app.workspace_registry_service.link_membership(
        "workspace-a",
        item_type="note",
        item_id="note-local",
        title="Workspace A field note",
    )
    app.workspace_registry_service.link_membership(
        "workspace-a",
        item_type="media",
        item_id="media-local",
        title="Workspace A transcript",
    )
    app.workspace_registry_service.link_membership(
        "workspace-b",
        item_type="conversation",
        item_id="chat-cross",
        title="Workspace B planning chat",
    )


@pytest.mark.asyncio
async def test_library_workspaces_mode_preserves_global_visibility_and_blocks_cross_workspace_handoff() -> None:
    app = _build_test_app()
    _seed_cross_workspace_library(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        assert "Workspace B research note" in _visible_text(screen)
        assert "Workspace B planning chat" in _visible_text(screen)
        use_button = screen.query_one("#library-use-in-console", Button)
        assert use_button.disabled is True
        assert "Copy or link" in str(use_button.tooltip)

        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        visible = _visible_text(screen)
        assert "Workspace Rules" in visible
        assert "Stage only active-workspace sources into Console, RAG, or agents." in visible
        assert "Handoff Rules" in visible
        assert "Workspace: Workspace A" in visible
        assert "Browse/search: all Library and Notes items remain visible" in visible
        assert "Workspace B research note" in visible
        assert "Workspace A transcript" in visible
        assert visible.count("Handoff Rules") == 1
        table_header = str(
            screen.query_one("#library-workspaces-eligibility-heading", Static).renderable
        )
        assert "|" not in table_header
        assert "Source" in table_header
        assert "Workspace" in table_header
        assert "Visible" in table_header
        assert "Console/RAG" in table_header
        source_rows = [
            str(widget.renderable)
            for widget in screen.query(Static)
            if (widget.id or "").startswith("library-workspaces-source-row-")
        ]
        blocked_source_row = next(
            row for row in source_rows if "Workspace B research note" in row
        )
        eligible_source_row = next(
            row for row in source_rows if "Workspace A transcript" in row
        )
        assert "|" not in blocked_source_row
        assert "Workspace B research note" in blocked_source_row
        assert "Workspace B" in blocked_source_row
        assert "Blocked" in blocked_source_row
        assert "Copy/link to Workspace A" in blocked_source_row
        assert "Workspace A transcript" in eligible_source_row
        assert "Workspace A" in eligible_source_row
        assert "Eligible" in eligible_source_row
        assert "Ready" in eligible_source_row
        assert "Collections: browse and organize; staging is read-only" in visible
        assert "Import/Export: copy or reference sources" in visible
        assert "Workspace selection changes staging, not what you can browse or search" in visible
        assert "Action: Copy/link blocked sources before staging." in visible
        assert "Blocked: some sources are outside Workspace A" in visible
        assert "Fix: Copy/link blocked sources to Workspace A" in visible
        assert "Study Dashboard actions" not in visible
        assert len(screen.query("#library-action-region #library-open-study")) == 0
        snapshot_region = screen.query_one("#library-local-snapshot-region")
        assert snapshot_region.display is False
        assert screen.query_one("#library-use-in-console", Button).disabled is True

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        # The Search/RAG panel renders counts as a pipe row ("... | Notes 2").
        assert "Notes 2" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_workspaces_empty_state_keeps_recovery_copy_compact() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        visible = _visible_text(screen)
        assert "No workspace sources yet." in visible
        assert "Browse/search still shows every Library and Notes item." in visible
        assert "Active workspace: Local Default" in visible
        assert "Browse: all workspaces" in visible
        assert "Handoff unavailable until sources exist or are assigned here." in visible
        assert "Console/RAG handoff: unavailable until sources exist" in visible
        assert "Blocked: no workspace sources" in visible
        assert "Why: no sources are assigned to this workspace yet." in visible
        assert "Fix: import or assign sources to Local Default" in visible
        assert "Action: Import sources or assign sources before staging." in visible
        assert "No Library sources are available for workspace authority preview." not in visible
        assert "Console/RAG handoff: local default source snapshot" not in visible
        assert "Source Workspace Visible Console/RAG" not in visible
        assert "Recovery" not in visible
        assert "Fix: add Library sources or assign sources to Local Default" not in visible
        import_sources_button = screen.query_one("#library-workspace-import-sources", Button)
        assert import_sources_button.disabled is False


@pytest.mark.asyncio
async def test_library_workspaces_can_create_and_select_local_workspace() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        screen.query_one("#library-create-local-workspace", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = app.workspace_registry_service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-1"
        assert active_workspace.name == "Workspace 1"

        visible = _visible_text(screen)
        assert "Workspace: Workspace 1" in visible
        assert "Active workspace: Workspace 1" in visible
        assert "Active workspace: Local Default" not in visible
        assert "Create local workspace" in visible
        assert "Server sync: WIP/unavailable" in visible


@pytest.mark.asyncio
async def test_library_workspaces_create_local_workspace_mouse_clicks() -> None:
    """Verify mouse clicks can create and activate a local workspace."""
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        await pilot.click("#library-mode-workspaces")
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        await pilot.click("#library-create-local-workspace")
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = app.workspace_registry_service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-1"
        assert active_workspace.name == "Workspace 1"

        visible = _visible_text(screen)
        assert "Active workspace: Workspace 1" in visible
        assert "Active workspace: Local Default" not in visible


@pytest.mark.asyncio
async def test_library_workspaces_create_skips_archived_local_workspace_identity() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(
        workspace_id="workspace-local-1",
        name="Workspace 1",
    )
    with service.db.transaction() as conn:
        conn.execute(
            """
            UPDATE workspace_records
            SET archived = 1
            WHERE workspace_id = ?
            """,
            ("workspace-local-1",),
        )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        screen.query_one("#library-create-local-workspace", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-2"
        assert active_workspace.name == "Workspace 2"


@pytest.mark.asyncio
async def test_library_workspaces_rows_escape_markup_text() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "[red]Injected[/red]", "id": "note-markup"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-a",
        name="[bold]Workspace A[/bold]",
    )
    app.workspace_registry_service.set_active_workspace("workspace-a")
    app.workspace_registry_service.link_membership(
        "workspace-a",
        item_type="note",
        item_id="note-markup",
        title="[red]Injected[/red]",
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        row = screen.query_one("#library-workspaces-source-row-0", Static)
        rendered = str(row.renderable)
        assert "\\[red]Injected\\[/red]" in rendered
        assert "\\[bold]Workspace" in rendered
        assert "[red]Injected[/red]" not in rendered


@pytest.mark.asyncio
async def test_library_workspaces_refresh_reuses_depth_state_for_panel_and_actions(
    monkeypatch,
) -> None:
    app = _build_test_app()
    _seed_cross_workspace_library(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        original_builder = library_screen_module.build_library_workspace_depth_state
        calls = 0

        def counting_builder(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_builder(*args, **kwargs)

        monkeypatch.setattr(
            library_screen_module,
            "build_library_workspace_depth_state",
            counting_builder,
        )
        screen.query_one("#library-mode-workspaces", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        assert calls == 1
