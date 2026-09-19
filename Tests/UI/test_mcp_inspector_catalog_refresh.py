"""Selected tool details follow refresh without discarding unchanged drafts."""

import asyncio
import threading
from dataclasses import replace

import pytest
from textual.widgets import Button, DataTable, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_root_settings import _open
from Tests.UI.test_mcp_workbench import (
    ToolTestHubService,
    WorkbenchAppWithBundledCSS,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "description", "schema", "removed"])
@private_profile_test
async def test_catalog_refresh_reconciles_selected_tool_without_reselecting(
    request, monkeypatch, change
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).focus()
        await pilot.press("enter")
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        assert inspector.current_tool == original
        inspector.query_one("#mcp-inspector-test-tool", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await _settle(pilot)
        field = inspector.query_one("#mcp-schema-field-0", Input)
        field.focus()
        await pilot.press(*"keep me")
        await _settle(pilot)
        assert field.value == "keep me"
        nonce = inspector._test_preview.nonce

        fresh = replace(original)
        if change == "description":
            fresh = replace(original, description="Search the current catalog.")
        elif change == "schema":
            fresh = replace(original, input_schema={"oneOf": [{"type": "object"}]})
        tools = [
            fresh if tool.tool_id == original.tool_id else tool
            for tool in workbench._last_hub_tools
            if change != "removed" or tool.tool_id != original.tool_id
        ]
        monkeypatch.setattr(workbench, "_collect_hub_tools", lambda: tools)
        await workbench._sync_children()
        await app.workers.wait_for_complete()
        await _settle(pilot)

        if change == "unchanged":
            assert inspector.query_one("#mcp-schema-field-0", Input) is field
            assert field.value == "keep me" and field.has_focus
            assert inspector._test_preview.nonce == nonce
        else:
            assert not inspector.query("#mcp-inspector-test-panel")
            assert inspector._test_preview is None
            assert nonce in app.unified_mcp_service.revoked_nonces
            if change == "removed":
                assert inspector.current_tool is None
                assert not inspector.query_one("#mcp-inspector-tool").display
            else:
                assert inspector.current_tool == fresh
                assert (
                    str(
                        inspector.query_one(
                            "#mcp-inspector-tool-description", Static
                        ).renderable
                    )
                    == fresh.description
                )
                schema = inspector.query_one("#mcp-inspector-tool-schema", Static)
                assert str(schema.renderable) == (
                    "Parameters: raw JSON" if change == "schema" else "Parameters: form"
                )
                assert inspector.query_one("#mcp-inspector-test-tool", Button).has_focus
                note = inspector.query_one("#mcp-inspector-tool-refresh-note", Static)
                assert "changed" in str(note.renderable).lower()
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retire_by", ["refresh", "close"])
@private_profile_test
async def test_delayed_preview_is_revoked_before_retiring_panel_disappears(
    request, monkeypatch, retire_by
):
    app = WorkbenchAppWithBundledCSS()
    service = app.unified_mcp_service = ToolTestHubService()
    mint_started = threading.Event()
    release_mint = threading.Event()
    removal_started = asyncio.Event()
    release_removal = asyncio.Event()
    prepare = service.prepare_hub_test

    def blocked_prepare(tool, **kwargs):
        mint_started.set()
        assert release_mint.wait(10)
        return prepare(tool, **kwargs)

    monkeypatch.setattr(service, "prepare_hub_test", blocked_prepare)
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).action_select_cursor()
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        assert inspector.current_tool == original
        await inspector.open_test_panel()
        assert await asyncio.to_thread(mint_started.wait, 3)
        worker = next(w for w in app.workers if w.name == "mcp-tool-test-preview")
        assert inspector._test_preview is None
        panel = inspector.query_one("#mcp-inspector-test-panel")
        container = inspector.query_one("#mcp-inspector-tool")
        target, method = (
            (container, "remove_children")
            if retire_by == "refresh"
            else (panel, "remove")
        )
        remove = getattr(target, method)

        async def blocked_remove(*args, **kwargs):
            removal_started.set()
            await release_removal.wait()
            await remove(*args, **kwargs)

        monkeypatch.setattr(target, method, blocked_remove)
        if retire_by == "refresh":
            fresh = replace(original, description="Changed while preparing")
            workbench._last_hub_tools = [
                fresh if t == original else t for t in workbench._last_hub_tools
            ]
            retire = asyncio.create_task(workbench._refresh_selected_tool())
        else:
            retire = asyncio.create_task(inspector._close_test_tool_panel())
        try:
            await asyncio.wait_for(removal_started.wait(), 3)
            assert inspector.query_one("#mcp-inspector-test-panel") is panel
            release_mint.set()
            await worker.wait()
            assert inspector._test_preview is None
            assert workbench._tool_test_preview_nonce is None
            assert service.revoked_nonces == ["preview-1"]
            assert service._previews == {}
            assert service.test_calls == []
        finally:
            release_mint.set()
            release_removal.set()
            await retire
            await app.workers.wait_for_complete()


@pytest.mark.asyncio
@private_profile_test
async def test_catalog_refresh_does_not_take_newer_focus_during_removal(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        table = canvas.query_one(DataTable)
        await canvas.select_tool_row(original.tool_id)
        table.action_select_cursor()
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        await inspector.open_test_panel()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        inspector.query_one("#mcp-schema-field-0", Input).focus()
        container = inspector.query_one("#mcp-inspector-tool")
        remove = container.remove_children
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_remove():
            started.set()
            await release.wait()
            await remove()

        monkeypatch.setattr(container, "remove_children", blocked_remove)
        fresh = replace(original, description="Updated description")
        workbench._last_hub_tools = [
            fresh if t == original else t for t in workbench._last_hub_tools
        ]
        refresh = asyncio.create_task(workbench._refresh_selected_tool())
        try:
            await asyncio.wait_for(started.wait(), 3)
            table.focus()
            await pilot.pause()
            assert table.has_focus
        finally:
            release.set()
            await refresh
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert inspector.current_tool == fresh
        assert table.has_focus


@pytest.mark.asyncio
@private_profile_test
async def test_unchanged_raw_form_preserves_editor_draft_focus_and_preview(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        raw = replace(original, input_schema={"oneOf": [{"type": "object"}]})
        tools = [raw if t == original else t for t in workbench._last_hub_tools]
        monkeypatch.setattr(
            workbench, "_collect_hub_tools", lambda: [replace(t) for t in tools]
        )
        await workbench._sync_children()
        await canvas.select_tool_row(raw.tool_id)
        canvas.query_one(DataTable).action_select_cursor()
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        await inspector.open_test_panel()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        editor = inspector.query_one("#mcp-schema-raw", TextArea)
        editor.load_text('{"query": "keep this raw draft"}')
        editor.focus()
        nonce = inspector._test_preview.nonce
        await workbench._sync_children()
        await _settle(pilot)
        assert inspector.query_one("#mcp-schema-raw", TextArea) is editor
        assert editor.text == '{"query": "keep this raw draft"}'
        assert editor.has_focus
        assert inspector._test_preview.nonce == nonce
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_waiting_catalog_refresh_cannot_replace_newer_tool_selection(request):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, _canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        newer = next(t for t in workbench._last_hub_tools if t.name == "fetch")
        context = workbench._tool_policy_profile_context
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(original, profile_context=context)
        await inspector._refresh_lock.acquire()
        selection = asyncio.create_task(
            inspector.show_tool(newer, profile_context=context)
        )
        await asyncio.sleep(0)
        refresh = asyncio.create_task(workbench._refresh_selected_tool())
        await asyncio.sleep(0)
        inspector._refresh_lock.release()
        await asyncio.gather(selection, refresh)
        assert inspector.current_tool is newer
        assert str(
            inspector.query_one("#mcp-inspector-tool-name", Static).renderable
        ).startswith(newer.name)


@pytest.mark.asyncio
@private_profile_test
async def test_queued_preview_request_for_retired_form_does_not_mint(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    service = app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).action_select_cursor()
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        held = []
        post = inspector.post_message

        def hold_preview(message):
            if isinstance(message, MCPInspector.ToolTestPreviewRequested):
                held.append(message)
                return True
            return post(message)

        monkeypatch.setattr(inspector, "post_message", hold_preview)
        await inspector.open_test_panel()
        assert len(held) == 1
        await inspector._close_test_tool_panel()
        workbench.on_mcp_inspector_tool_test_preview_requested(held[0])
        await app.workers.wait_for_complete()
        assert service._preview_count == 0
        assert inspector._test_preview is None


@pytest.mark.asyncio
@private_profile_test
async def test_refused_reopen_cannot_accept_previous_forms_pending_preview(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    service = app.unified_mcp_service = ToolTestHubService()
    started, release = threading.Event(), threading.Event()
    prepare = service.prepare_hub_test

    def blocked_prepare(tool, **kwargs):
        started.set()
        assert release.wait(10)
        return prepare(tool, **kwargs)

    monkeypatch.setattr(service, "prepare_hub_test", blocked_prepare)
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).action_select_cursor()
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        await inspector.open_test_panel()
        assert await asyncio.to_thread(started.wait, 3)
        worker = next(w for w in app.workers if w.name == "mcp-tool-test-preview")
        try:
            await inspector._close_test_tool_panel()
            monkeypatch.setattr(
                workbench, "_validate_profile_context", lambda context: None
            )
            await inspector.open_test_panel()
            await _settle(pilot)
            assert (
                "unavailable"
                in str(
                    inspector.query_one(
                        "#mcp-inspector-test-preview", Static
                    ).renderable
                ).lower()
            )
            release.set()
            await worker.wait()
            assert inspector._test_preview is None
            assert inspector.query_one("#mcp-inspector-test-run", Button).disabled
            assert service.revoked_nonces == ["preview-1"]
            assert service._previews == {}
        finally:
            release.set()
            await app.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_tool_inspector_reveals_focused_actions_and_arguments(request, size):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=size) as pilot:
        workbench, _canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        # A legitimate long description leaves the primary action below the fold.
        tool = replace(
            original, description="Search the connected server catalog. " * 10
        )
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            tool, profile_context=workbench._tool_policy_profile_context
        )
        await _settle(pilot)
        button = inspector.query_one("#mcp-inspector-test-tool", Button)
        button.focus()
        await _settle(pilot)
        assert button.has_focus
        assert button in app.screen._compositor.visible_widgets, (
            inspector.styles.overflow_y,
            inspector.scroll_y,
            inspector.max_scroll_y,
            inspector.virtual_size,
            inspector.size,
            button.region,
            [
                (a.id, a.region, getattr(a.styles, "overflow_y", None))
                for a in button.ancestors
                if hasattr(a, "region")
            ],
        )
        region, clip = app.screen._compositor.visible_widgets[button]
        assert region.intersection(clip) == region
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await _settle(pilot)
        field = inspector.query_one("#mcp-schema-field-0", Input)
        assert field.has_focus
        region, clip = app.screen._compositor.visible_widgets[field]
        assert region.intersection(clip) == region

        await inspector.show_tool(
            replace(tool, description="Updated tool details."),
            profile_context=workbench._tool_policy_profile_context,
            refresh_from=tool,
        )
        await _settle(pilot)
        replacement = inspector.query_one("#mcp-inspector-test-tool", Button)
        assert replacement.has_focus
        region, clip = app.screen._compositor.visible_widgets[replacement]
        assert region.intersection(clip) == region
