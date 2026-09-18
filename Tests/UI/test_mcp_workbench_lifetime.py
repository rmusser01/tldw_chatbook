"""Local-setting receipts only project into a ready, live MCP canvas."""

import asyncio

import pytest
from textual.widget import Widget

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_workbench import WorkbenchAppWithBundledCSS
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.MCP.local_config_saves import (
    ConfigSaveRequest,
    ConfigSaveResult,
    get_mcp_local_config_saves,
)
from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


async def _failed_receipt(app, key):
    owner = get_mcp_local_config_saves(app)
    await owner.submit(
        ConfigSaveRequest(
            False if key == "local_tools_enabled" else "/unsaved-root",
            None if key == "local_tools_enabled" else (object(), 0),
            module.get_cli_config_path(),
            key=key,
        ),
        lambda: ConfigSaveResult("failed"),
    )


def _poll_outcome(workbench):
    try:
        workbench._sync_local_config_save_status()
    except Exception as error:  # noqa: BLE001 - assert after allowing mount/prune to finish
        return type(error).__name__
    return None


async def _ready(pilot):
    workbench = pilot.app.query_one(MCPWorkbench)
    await _wait(pilot, lambda: not workbench.is_loading and not workbench._reloading)
    assert len(workbench.query("Button.mcp-rail-row")) == 3
    return workbench


@pytest.mark.parametrize("key", ["local_tools_enabled", "workspace_root"])
@private_profile_test
async def test_poll_before_deferred_controls_compose_is_safe(request, monkeypatch, key):
    app = WorkbenchAppWithBundledCSS()
    await _failed_receipt(app, key)
    original = MCPToolsMode.mount_composed_widgets
    observations = []

    async def poll_before_children(canvas, widgets):
        workbench = app.query_one(MCPWorkbench)
        observations.append(
            (canvas.is_attached, canvas.is_mounted, _poll_outcome(workbench))
        )
        await original(canvas, widgets)

    monkeypatch.setattr(MCPToolsMode, "mount_composed_widgets", poll_before_children)
    async with app.run_test() as pilot:
        workbench = await _ready(pilot)
        assert observations == [(True, False, None)]
        workbench._sync_local_config_save_status()
        status_id = (
            "#mcp-tools-local-config-status"
            if key == "local_tools_enabled"
            else "#mcp-tools-workspace-status"
        )
        assert "failed" in str(workbench.query_one(status_id).renderable).lower()


@pytest.mark.parametrize("key", ["local_tools_enabled", "workspace_root"])
@pytest.mark.parametrize("remove_target", ["workbench", "canvas"])
@pytest.mark.parametrize("receipt_timing", ["before", "during"])
@private_profile_test
async def test_poll_during_descendant_detach_is_safe_and_remount_recovers(
    request, monkeypatch, key, remove_target, receipt_timing
):
    app = WorkbenchAppWithBundledCSS()
    observations = []
    async with app.run_test() as pilot:
        workbench = await _ready(pilot)
        canvas = workbench.query_one(MCPToolsMode)
        parent = canvas.parent
        button = workbench.query_one("#mcp-tools-local-enabled")
        original = Widget._message_loop_exit
        if receipt_timing == "before":
            await _failed_receipt(app, key)
            workbench._sync_local_config_save_status()

        async def poll_after_control_detach(widget):
            await original(widget)
            if widget is button:
                # A save can settle while descendants are being removed. Its
                # new receipt must neither crash nor be consumed by a dead UI.
                if receipt_timing == "during":
                    await _failed_receipt(app, key)
                observations.append(
                    (
                        workbench.is_attached,
                        workbench._pruning,
                        _poll_outcome(workbench),
                    )
                )

        monkeypatch.setattr(Widget, "_message_loop_exit", poll_after_control_detach)
        if remove_target == "workbench":
            await workbench.remove()
            assert observations == [(True, True, None)]
            await app.mount(MCPWorkbench(app_instance=app, id="replacement-workbench"))
            replacement = await _ready(pilot)
            assert replacement is not workbench
        else:
            await canvas.remove()
            assert observations == [(True, False, None)]
            await parent.mount(MCPToolsMode(id="mcp-mode-canvas-tools"))
            await workbench._sync_children()
            replacement = workbench
        replacement._sync_local_config_save_status()
        status_id = (
            "#mcp-tools-local-config-status"
            if key == "local_tools_enabled"
            else "#mcp-tools-workspace-status"
        )
        assert "failed" in str(replacement.query_one(status_id).renderable).lower()


@private_profile_test
async def test_deferred_initial_load_remains_pending_until_dispatched(
    request, monkeypatch
):
    app = WorkbenchAppWithBundledCSS()
    starts = []
    start_initial_load = MCPWorkbench._start_initial_load
    monkeypatch.setattr(
        MCPWorkbench, "_start_initial_load", lambda workbench: starts.append(workbench)
    )
    async with app.run_test() as pilot:
        await _wait(pilot, lambda: bool(starts))
        workbench = app.query_one(MCPWorkbench)
        await app.workers.wait_for_complete()
        assert workbench.is_loading and workbench._reloading
        assert len(workbench.query("Button.mcp-rail-row")) == 1
        monkeypatch.setattr(MCPWorkbench, "_start_initial_load", start_initial_load)
        start_initial_load(workbench)
        async with asyncio.timeout(10):
            await _ready(pilot)
