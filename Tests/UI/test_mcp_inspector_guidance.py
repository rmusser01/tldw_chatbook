"""Server readiness guidance yields to the inspector's selected detail."""

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _open, _settle
from Tests.UI.test_mcp_inspector import (
    InspectorApp,
    _audit_entry,
    _finding,
    _ready_snap,
    _stale_snap,
    _tool,
)
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.readiness import as_checking
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


def guidance(inspector):
    """Return the mounted server guidance widgets in display order.

    Args:
        inspector: Mounted inspector with state, message and action containers.

    Returns:
        The readiness badge, explanation and actions widgets.
    """
    return [
        inspector.query_one(f"#mcp-inspector-{name}")
        for name in ("state", "message", "actions")
    ]


def assert_hidden(app, inspector):
    """Check that server guidance is neither painted nor keyboard reachable.

    Args:
        app: Running app whose current screen owns the inspector.
        inspector: Mounted inspector after pending layout work has settled.

    Raises:
        AssertionError: Guidance is displayed, painted or in the focus chain.
    """
    for widget in guidance(inspector):
        assert not widget.display, widget.id
        assert widget not in app.screen._compositor.visible_widgets, widget.id
    actions = inspector.query_one("#mcp-inspector-actions")
    for button in actions.query(Button):
        assert button not in app.screen.focus_chain


async def show_detail(inspector, kind):
    """Await rendering one fixture detail in the mounted inspector.

    Args:
        inspector: Inspector receiving the fixture detail.
        kind: One of tool, permission, audit or finding from the test matrix.
    """
    if kind == "tool":
        await inspector.show_tool(_tool())
    elif kind == "permission":
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
    elif kind == "audit":
        await inspector.show_audit_entry(_audit_entry())
    else:
        await inspector.show_finding(_finding(), server_key="local:docs")


async def clear_detail(inspector, kind):
    """Await clearing the specified detail through its existing display owner.

    Args:
        inspector: Inspector containing the fixture detail.
        kind: One of tool, permission, audit or finding from the test matrix.
    """
    if kind in ("tool", "permission"):
        await inspector.show_tool(None)
    elif kind == "audit":
        await inspector.show_audit_entry(None)
    else:
        await inspector.show_finding(None)


@pytest.mark.parametrize("kind", ["tool", "permission", "audit", "finding"])
@private_profile_test
async def test_all_guidance_stays_hidden_through_refresh_then_restores(request, kind):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await show_detail(inspector, kind)
        await pilot.pause()
        assert_hidden(app, inspector)

        # Rebuilt action children must inherit their hidden parent's state.
        operation = object()
        for snapshot in (None, _ready_snap(), as_checking(_stale_snap(), "connect")):
            await inspector.update_readiness(snapshot, cancel_operation=operation)
            await pilot.pause()
            assert_hidden(app, inspector)

        await clear_detail(inspector, kind)
        await pilot.pause()
        assert all(widget.display for widget in guidance(inspector))
        assert (
            inspector.query_one("#mcp-inspector-cancel", Button)
            in app.screen.focus_chain
        )
        message = inspector.query_one("#mcp-inspector-message", Static)
        assert str(message.renderable) == as_checking(_stale_snap(), "connect").message
        await inspector.update_readiness(_ready_snap())
        await pilot.pause()
        assert "4 tools available" in str(message.renderable)


@private_profile_test
async def test_guidance_returns_only_after_last_detail_clears(request):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await inspector.show_audit_entry(_audit_entry())
        await inspector.show_finding(_finding(), server_key="local:docs")
        await inspector.show_finding(None)
        await pilot.pause()
        assert_hidden(app, inspector)
        await inspector.update_readiness(_ready_snap())
        await inspector.show_audit_entry(None)
        await pilot.pause()
        assert all(widget.display for widget in guidance(inspector))
        assert "notes" in str(
            inspector.query_one("#mcp-inspector-state", Static).renderable
        )


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_real_workbench_detail_owns_guidance_visibility(
    request, tmp_path, theme, size
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        inspector = workbench.query_one(MCPInspector)
        await inspector.clear_mode_view()
        await inspector.update_readiness(_stale_snap())
        for kind in ("tool", "permission", "audit", "finding"):
            await show_detail(inspector, kind)
            await _settle(pilot)
            assert_hidden(app, inspector)
            await inspector.update_readiness(_ready_snap())
            await _settle(pilot)
            assert_hidden(app, inspector)
            await clear_detail(inspector, kind)
            await _settle(pilot)
            assert all(widget.display for widget in guidance(inspector))
