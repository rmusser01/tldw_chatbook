"""Selected tool details follow refresh without discarding unchanged drafts."""

import asyncio
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
@pytest.mark.parametrize("raw", [False, True])
@private_profile_test
async def test_catalog_refresh_reconciles_selected_tool_without_reselecting(
    request, monkeypatch, change, raw
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        if raw:
            original = replace(original, input_schema={"oneOf": [{"type": "object"}]})
            workbench._last_hub_tools = [
                original if t.tool_id == original.tool_id else t
                for t in workbench._last_hub_tools
            ]
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
        field = (
            inspector.query_one("#mcp-schema-raw", TextArea)
            if raw
            else inspector.query_one("#mcp-schema-field-0", Input)
        )
        draft = '{"query":"keep me"}' if raw else "keep me"
        field.focus()
        if raw:
            field.load_text(draft)
        else:
            await pilot.press(*draft)
        await _settle(pilot)
        assert (field.text if raw else field.value) == draft
        cursor = field.cursor_location if raw else field.cursor_position
        nonce = inspector._test_preview.nonce

        fresh = replace(original)
        if change == "description":
            fresh = replace(original, description="Search the current catalog.")
        elif change == "schema":
            fresh = replace(
                original,
                input_schema={"type": "object", "properties": {}}
                if raw
                else {"oneOf": [{"type": "object"}]},
            )
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
            assert (
                inspector.query_one("#mcp-schema-raw" if raw else "#mcp-schema-field-0")
                is field
            )
            assert (field.text if raw else field.value) == draft
            assert field.has_focus
            assert (field.cursor_location if raw else field.cursor_position) == cursor
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
                    "Parameters: raw JSON"
                    if raw != (change == "schema")
                    else "Parameters: form"
                )
                assert inspector.query_one("#mcp-inspector-test-tool", Button).has_focus
                note = inspector.query_one("#mcp-inspector-tool-refresh-note", Static)
                assert "changed" in str(note.renderable).lower()
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_revokes_preview_minted_while_old_form_is_retiring(
    request, monkeypatch
):
    """Same-ID replacement must close publication before awaited child removal."""
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
        mint_started, mint_release = asyncio.Event(), asyncio.Event()
        remove_started, remove_release = asyncio.Event(), asyncio.Event()
        original_mint = workbench._mint_test_preview
        container = inspector.query_one("#mcp-inspector-tool")
        original_remove = container.remove_children

        async def delayed_mint(*args):
            mint_started.set()
            await mint_release.wait()
            return await original_mint(*args)

        async def delayed_remove(*args, **kwargs):
            remove_started.set()
            await remove_release.wait()
            await original_remove(*args, **kwargs)

        monkeypatch.setattr(workbench, "_mint_test_preview", delayed_mint)
        inspector.query_one("#mcp-inspector-test-tool", Button).focus()
        await pilot.press("enter")
        await asyncio.wait_for(mint_started.wait(), 3)
        monkeypatch.setattr(container, "remove_children", delayed_remove)
        fresh = replace(original, description="Updated while preparing.")
        tools = [
            fresh if t.tool_id == original.tool_id else t
            for t in workbench._last_hub_tools
        ]
        monkeypatch.setattr(workbench, "_collect_hub_tools", lambda: tools)
        refresh = asyncio.create_task(workbench._sync_children())
        try:
            await asyncio.wait_for(remove_started.wait(), 3)
            assert inspector.query("#mcp-inspector-test-panel")
            mint_release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), 3)
            # Assert during the teardown interval, not only once the old form is gone.
            assert inspector._test_preview is None
            assert app.unified_mcp_service._previews == {}
            assert app.unified_mcp_service.revoked_nonces == ["preview-1"]
        finally:
            mint_release.set()
            remove_release.set()
            await asyncio.wait_for(refresh, 3)
        assert inspector.current_tool == fresh
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_preserves_newer_focus_while_old_form_is_retiring(
    request, monkeypatch
):
    """Returning focus after refresh must not undo the user's newer navigation."""
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        await canvas.select_tool_row(original.tool_id)
        table = canvas.query_one(DataTable)
        table.focus()
        await pilot.press("enter")
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        inspector.query_one("#mcp-inspector-test-tool", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert inspector.query_one("#mcp-schema-field-0", Input).has_focus
        container = inspector.query_one("#mcp-inspector-tool")
        original_remove = container.remove_children
        remove_started, remove_release = asyncio.Event(), asyncio.Event()

        async def delayed_remove(*args, **kwargs):
            remove_started.set()
            await remove_release.wait()
            await original_remove(*args, **kwargs)

        monkeypatch.setattr(container, "remove_children", delayed_remove)
        fresh = replace(original, description="Updated while editing.")
        tools = [
            fresh if t.tool_id == original.tool_id else t
            for t in workbench._last_hub_tools
        ]
        monkeypatch.setattr(workbench, "_collect_hub_tools", lambda: tools)
        refresh = asyncio.create_task(workbench._sync_children())
        try:
            await asyncio.wait_for(remove_started.wait(), 3)
            table.focus()
            await pilot.pause()
            assert table.has_focus
        finally:
            remove_release.set()
            await asyncio.wait_for(refresh, 3)
        await _settle(pilot)
        assert table.has_focus


@pytest.mark.asyncio
@private_profile_test
async def test_queued_old_preview_request_cannot_replace_reopened_preview(
    request, monkeypatch
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
        post_message = inspector.post_message
        queued = []

        def hold_preview(message):
            if isinstance(message, MCPInspector.ToolTestPreviewRequested):
                queued.append(message)
                return True
            return post_message(message)

        monkeypatch.setattr(inspector, "post_message", hold_preview)
        await inspector.open_test_panel()
        assert len(queued) == 1
        monkeypatch.setattr(inspector, "post_message", post_message)
        fresh = replace(original, description="Replacement definition.")
        tools = [
            fresh if t.tool_id == original.tool_id else t
            for t in workbench._last_hub_tools
        ]
        monkeypatch.setattr(workbench, "_collect_hub_tools", lambda: tools)
        await workbench._sync_children()
        await inspector.open_test_panel()
        await _settle(pilot)
        await app.workers.wait_for_complete()
        current = inspector._test_preview
        assert current is not None
        post_message(queued[0])
        await _settle(pilot)
        await app.workers.wait_for_complete()
        assert inspector._test_preview is current
        assert app.unified_mcp_service._preview_count == 1
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_waiting_on_inspector_cannot_replace_newer_selection(request):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, canvas = await _open(pilot)
        original = next(t for t in workbench._last_hub_tools if t.name == "search")
        newer = next(t for t in workbench._last_hub_tools if t.name == "fetch")
        await canvas.select_tool_row(original.tool_id)
        canvas.query_one(DataTable).focus()
        await pilot.press("enter")
        await _settle(pilot)
        inspector = app.query_one(MCPInspector)
        async with inspector._refresh_lock:
            selection = asyncio.create_task(inspector.show_tool(newer))
            await asyncio.sleep(0)
            refresh = asyncio.create_task(workbench._refresh_selected_tool())
            await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.gather(selection, refresh), 3)
        assert inspector.current_tool is newer
        assert newer.name in str(
            inspector.query_one("#mcp-inspector-tool-name", Static).renderable
        )


@pytest.mark.asyncio
@private_profile_test
async def test_failed_profile_reopen_cannot_publish_previous_panel_mint(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    app.unified_mcp_service = ToolTestHubService()
    async with app.run_test(size=(170, 48)) as pilot:
        workbench, _ = await _open(pilot)
        tool = next(t for t in workbench._last_hub_tools if t.name == "search")
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            tool, profile_context=workbench._tool_policy_profile_context
        )
        minted, release = asyncio.Event(), asyncio.Event()
        original_mint = workbench._mint_test_preview

        async def held_mint(*args):
            preview = await original_mint(*args)
            minted.set()
            await release.wait()
            return preview

        monkeypatch.setattr(workbench, "_mint_test_preview", held_mint)
        await inspector.open_test_panel()
        try:
            await asyncio.wait_for(minted.wait(), 3)
            old_panel = inspector.query_one("#mcp-inspector-test-panel")
            await inspector._close_test_tool_panel()
            monkeypatch.setattr(workbench, "_validate_profile_context", lambda *_: None)
            await inspector.open_test_panel()
            await _settle(pilot)
            assert inspector.query_one("#mcp-inspector-test-panel") is not old_panel
            status = inspector.query_one("#mcp-inspector-test-preview", Static)
            assert "profile context" in str(status.renderable)
        finally:
            release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), 3)
        assert inspector._test_preview is None
        assert app.unified_mcp_service._previews == {}
        assert app.unified_mcp_service.revoked_nonces == ["preview-1"]
        assert "profile context" in str(status.renderable)
