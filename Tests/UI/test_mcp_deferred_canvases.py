"""task-2901: MCP defers its three hidden mode canvases past first paint.

Third application of the defer-past-first-paint pattern (2725 Roleplay,
2900 Lab ▸ Models): Tools/Permissions/Audit arrive hidden inside the
ContentSwitcher (Servers is initial). They now mount as the first step of
the workbench's load worker, before `reload()` pushes data into them, so
every push in the load pipeline still sees the full canvas set.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import ContentSwitcher

from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from Tests.UI.test_mcp_workbench import WorkbenchApp

pytestmark = pytest.mark.asyncio

_DEFERRED_TYPES = (MCPToolsMode, MCPPermissionsMode, MCPAuditMode)


async def test_first_paint_excludes_the_hidden_mode_canvases(monkeypatch):
    """Compose alone must not mount the deferred canvases."""
    monkeypatch.setattr(
        MCPWorkbench, "_mount_deferred_canvases", AsyncMock(), raising=False
    )

    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert list(app.query(MCPServersMode)), "initial Servers canvas missing"
        for deferred_type in _DEFERRED_TYPES:
            assert not list(app.query(deferred_type)), (
                f"{deferred_type.__name__} mounted during compose — back on "
                "the click→paint critical path"
            )


async def test_load_mounts_all_canvases_with_servers_current():
    """After the real load, every mode canvas exists and Servers is current."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.pause()

        for canvas_type in (MCPServersMode, *_DEFERRED_TYPES):
            assert len(list(app.query(canvas_type))) == 1, (
                f"{canvas_type.__name__} missing after load"
            )
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-servers"


async def test_mode_request_before_canvases_mount_is_stashed_not_crashed(
    monkeypatch,
):
    """`ContentSwitcher.current` raises for an unmounted id — a fast mode-chip
    click in the pre-mount window must stash the request and apply it once
    the canvases exist, mirroring the `_reloading` restore stash."""
    real_mount = MCPWorkbench._mount_deferred_canvases
    monkeypatch.setattr(MCPWorkbench, "_mount_deferred_canvases", AsyncMock())

    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)

        # The fast click: the audit canvas does not exist yet.
        workbench.set_mode("audit")
        await pilot.pause()

        # Now the deferred mount lands (the real one).
        await real_mount(workbench)
        await pilot.pause()

        assert workbench.active_mode == "audit"
        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-audit"
