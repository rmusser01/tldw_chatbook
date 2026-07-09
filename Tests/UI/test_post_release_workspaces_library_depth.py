"""Post-release Workspaces and Library depth mounted regressions."""

from __future__ import annotations

import time

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
    _wait_for_selector,
)


async def _wait_for_library_shell_ready(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell (not the retired 3-pane workbench).

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites that use the generic ``DestinationHarness``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
    )


async def _open_library_details(screen, pilot) -> None:
    """Expand the Library rail Details section that now always hosts the
    Workspaces depth panel and action controls (see
    ``LibraryRail.workspaces_body_factory``)."""
    screen.query_one("#console-rail-section-toggle-library-details", Button).press()
    await pilot.pause()
    await pilot.pause()


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
        await _wait_for_library_shell_ready(screen, pilot)

        # The Workspaces action controls always live in the rail Details
        # body now (LibraryRail.workspaces_body_factory), independent of the
        # selected canvas mode.
        use_button = screen.query_one("#library-use-in-console", Button)
        assert use_button.disabled is True
        assert "Copy or link" in str(use_button.tooltip)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        visible = _visible_text(screen)
        # "Workspace Rules" is renamed to the bold "Workspace" group header,
        # and the policy-prose lines it used to sit above (global browse
        # rule, staging-scope rule, per-source visibility label) are gone —
        # the Handoff row below carries that signal on its own now.
        active_workspace_row = screen.query_one("#library-workspaces-active-workspace", Static)
        assert active_workspace_row.renderable.plain == "Active · Workspace A"
        assert "Stage only active-workspace sources into Console, RAG, or agents." not in visible
        assert "Workspace: Workspace A" not in visible
        assert "Browse/search: all Library and Notes items remain visible" not in visible
        # The Details body intentionally no longer renders a per-source
        # visibility/assignment table (it duplicated Browse ▸ Conversations).
        # Only the workspace summary + handoff count remain here.
        assert not screen.query("#library-workspaces-eligibility-heading")
        assert not screen.query("#library-workspaces-source-row-0")
        assert not screen.query("#library-workspaces-source-row-1")
        assert not screen.query("#library-workspaces-visibility")
        assert not screen.query("#library-workspaces-global-access-rule")
        assert not screen.query("#library-workspaces-context-rule")
        assert not screen.query("#library-workspaces-collections-membership")
        assert not screen.query("#library-workspaces-import-export")
        handoff_row = screen.query_one("#library-workspaces-handoff", Static)
        assert handoff_row.renderable.plain == "Handoff · 2 eligible, ● 2 blocked"
        assert "Collections: browse and organize; staging is read-only" not in visible
        assert "Import/Export: copy or reference sources" not in visible
        assert "Workspace selection changes staging, not what you can browse or search" not in visible
        # The retired blocked-state callout + "Next:" line are gone; the
        # Handoff row above is now the single source of that signal.
        assert not screen.query("#library-workspace-action-blocked")
        assert not screen.query("#library-workspace-action-next-step")
        assert not screen.query("#library-workspace-action-ready")
        assert "Study Dashboard actions" not in visible
        assert screen.query_one("#library-use-in-console", Button).disabled is True
        assert screen.query_one("#library-create-local-workspace", Button)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        # The Search/RAG panel renders counts as a pipe row ("... | Notes 2").
        assert "Notes 2" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_workspaces_empty_state_keeps_recovery_copy_compact() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        visible = _visible_text(screen)
        # The empty-state-only prose lines (a mini "No workspace sources
        # yet." heading plus two restatements of the browse/handoff rule)
        # are gone; the Active + Handoff rows below carry the same facts.
        assert "No workspace sources yet." not in visible
        assert "Browse/search still shows every Library and Notes item." not in visible
        assert "Handoff unavailable until sources exist or are assigned here." not in visible
        assert not screen.query("#library-workspaces-empty-title")
        assert not screen.query("#library-workspaces-empty-browse")
        assert not screen.query("#library-workspaces-empty-handoff")
        assert not screen.query("#library-workspaces-visibility")
        active_workspace_row = screen.query_one("#library-workspaces-active-workspace", Static)
        assert active_workspace_row.renderable.plain == "Active · Local Default"
        assert "Browse/search: all Library and Notes items remain visible" not in visible
        handoff_row = screen.query_one("#library-workspaces-handoff", Static)
        assert handoff_row.renderable.plain == "Handoff · unavailable until sources exist"
        # The retired blocked/fix callout is gone; the Handoff row above is
        # the only place this state renders now.
        assert not screen.query("#library-workspace-action-blocked")
        assert not screen.query("#library-workspace-action-next-step")
        assert "Blocked: no workspace sources" not in visible
        assert "Fix: import or assign sources to Local Default" not in visible
        assert "No Library sources are available for workspace authority preview." not in visible
        assert "Console/RAG handoff: local default source snapshot" not in visible
        assert "Source Workspace Visible Console/RAG" not in visible
        assert "Recovery" not in visible
        assert "Fix: add Library sources or assign sources to Local Default" not in visible
        # The Import sources action is a real action and survives when there
        # are no workspace sources yet.
        import_sources_button = screen.query_one("#library-workspace-import-sources", Button)
        assert import_sources_button.disabled is False


@pytest.mark.asyncio
async def test_library_workspaces_can_create_and_select_local_workspace() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        screen.query_one("#library-create-local-workspace", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = app.workspace_registry_service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-1"
        assert active_workspace.name == "Workspace 1"

        visible = _visible_text(screen)
        active_workspace_row = screen.query_one("#library-workspaces-active-workspace", Static)
        assert active_workspace_row.renderable.plain == "Active · Workspace 1"
        assert "Create local workspace" in visible
        assert "Server sync WIP · local only" in visible


@pytest.mark.asyncio
async def test_library_workspaces_create_local_workspace_mouse_clicks() -> None:
    """Verify mouse clicks can create and activate a local workspace."""
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        # Collapse the other rail sections first so the Details body has the
        # most headroom at this terminal height.
        for section in ("browse", "create", "ingest"):
            screen.query_one(f"#console-rail-section-toggle-library-{section}", Button).press()
            await pilot.pause()
        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        # DestinationHarness does not load the real app stylesheet, so the
        # `#library-rail { overflow-y: auto; }` rule that makes the rail
        # scrollable never applies here. Set it directly to exercise the same
        # scroll-into-view behavior the real app gets from CSS, then scroll
        # the action button into view before clicking by screen coordinate.
        rail = screen.query_one("#library-rail")
        rail.styles.overflow_y = "auto"
        await pilot.pause()
        create_button = screen.query_one("#library-create-local-workspace", Button)
        create_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.pause()

        await pilot.click("#library-create-local-workspace")
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = app.workspace_registry_service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-1"
        assert active_workspace.name == "Workspace 1"

        active_workspace_row = screen.query_one("#library-workspaces-active-workspace", Static)
        assert active_workspace_row.renderable.plain == "Active · Workspace 1"


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
        await _wait_for_library_shell_ready(screen, pilot)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-create-local-workspace")

        screen.query_one("#library-create-local-workspace", Button).press()
        await _wait_for_selector(screen, pilot, "#library-workspaces-active-workspace")

        active_workspace = service.get_active_workspace()
        assert active_workspace is not None
        assert active_workspace.workspace_id == "workspace-local-2"
        assert active_workspace.name == "Workspace 2"


@pytest.mark.asyncio
async def test_library_workspaces_active_workspace_label_escapes_markup_text() -> None:
    """The workspace summary label still escapes user-provided markup.

    Formerly this also covered per-source title escaping in the now-removed
    assignment table (``library-workspaces-source-row-*``); that surface is
    gone, so only the surviving ``library-workspaces-active-workspace``
    summary widget is exercised here.
    """
    app = _build_test_app()
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-a",
        name="[bold]Workspace A[/bold]",
    )
    app.workspace_registry_service.set_active_workspace("workspace-a")
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        row = screen.query_one("#library-workspaces-active-workspace", Static)
        rendered = row.renderable.plain
        assert rendered == "Active · [bold]Workspace A[/bold]"
        # Renders as literal text, not interpreted Rich markup: the only
        # style span present dims the "Active " label; no additional span
        # (e.g. bold) was produced from the injected workspace name.
        assert all(span.style == "dim" for span in row.renderable.spans)


@pytest.mark.asyncio
async def test_library_workspaces_refresh_reuses_depth_state_for_panel_and_actions(
    monkeypatch,
) -> None:
    """Opening the rail Details disclosure must not rebuild the Workspaces
    depth state: the panel + action widgets built by the last recompose
    (``LibraryRail.workspaces_body_factory``) are only revealed, not rebuilt,
    because ``handle_library_rail_section_toggle`` toggles display in place
    without recomposing."""
    app = _build_test_app()
    _seed_cross_workspace_library(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

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
        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        assert calls == 0


@pytest.mark.asyncio
async def test_library_details_section_renders_grouped_headers_and_drops_policy_prose() -> None:
    """The Details section groups content under three bold headers and no
    longer restates the blocked-handoff state three different ways (a line +
    two bordered callouts + a "Next:" line)."""
    app = _build_test_app()
    _seed_cross_workspace_library(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-depth-panel")

        # The three group headers render as their own Statics, in order.
        status_header = screen.query_one("#library-details-group-status", Static)
        workspace_header = screen.query_one("#library-details-group-workspace", Static)
        actions_header = screen.query_one("#library-details-group-actions", Static)
        assert str(status_header.renderable) == "Status"
        assert str(workspace_header.renderable) == "Workspace"
        assert str(actions_header.renderable) == "Actions"
        details_body = screen.query_one("#library-rail-section-body-details")
        group_header_ids = {
            "library-details-group-status",
            "library-details-group-workspace",
            "library-details-group-actions",
        }
        headers_in_order = [
            widget.id
            for widget in details_body.walk_children()
            if widget.id in group_header_ids
        ]
        assert headers_in_order == [
            "library-details-group-status",
            "library-details-group-workspace",
            "library-details-group-actions",
        ]

        # The full-sentence policy explainers are gone.
        visible = _visible_text(screen)
        for prose in (
            "Workspace Rules",
            "Workspace actions",
            "Browse spans all workspaces; staging follows the active workspace.",
            "Stage only active-workspace sources into Console, RAG, or agents.",
            "Workspace selection changes staging, not what you can browse or search.",
            "Collections: browse and organize; staging is read-only",
            "Import/Export: copy or reference sources",
        ):
            assert prose not in visible

        # The retired blocked-state callouts (ready / blocked / next-step)
        # and the policy-prose Statics they sat beside are gone.
        for removed_id in (
            "library-workspaces-visibility",
            "library-workspaces-global-access-rule",
            "library-workspaces-context-rule",
            "library-workspaces-collections-membership",
            "library-workspaces-import-export",
            "library-workspace-action-blocked",
            "library-workspace-action-next-step",
            "library-workspace-action-ready",
        ):
            assert not screen.query(f"#{removed_id}")

        # The two primary action buttons stay reachable.
        assert screen.query_one("#library-create-local-workspace", Button)
        assert screen.query_one("#library-use-in-console", Button)

        # The Handoff row is the single surviving source of the
        # eligible/blocked counts.
        handoff_row = screen.query_one("#library-workspaces-handoff", Static)
        handoff_text = handoff_row.renderable.plain
        assert "2 eligible" in handoff_text
        assert "2 blocked" in handoff_text
