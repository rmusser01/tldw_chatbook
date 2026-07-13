# Tests/UI/test_mcp_inspector.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static, TextArea

from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


class FakeAdvService:
    def __init__(self) -> None:
        self.action_calls: list[tuple[str, dict]] = []

    async def load_section(self, section=None):
        return {"source": "local", "section": section or "overview"}

    def available_actions(self):
        return [
            {
                "name": "profile.connect",
                "label": "Connect Profile",
                "action_id": "mcp.external_profiles.configure.local",
                "payload_template": '{"profile_id":"demo"}',
            }
        ]

    async def run_action(self, action_name, payload):
        self.action_calls.append((action_name, dict(payload or {})))
        return {"ok": True}


class InspectorApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.service = FakeAdvService()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        inspector = self.query_one(MCPInspector)
        inspector.set_service_context(self.service, [("Overview", "overview"), ("Inventory", "inventory")])

    def on_mcp_inspector_hub_action_requested(self, event) -> None:
        self.events.append(event)


def _stale_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:docs", label="docs", source="local",
        state=ReadinessState.STALE, reasons=(ReasonCode.RUNTIME_UNAVAILABLE,),
        message="2 tools discovered; not currently connected.",
    )


@pytest.mark.asyncio
async def test_readiness_block_shows_state_message_and_action_buttons():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.update_readiness(_stale_snap())
        await pilot.pause()
        badge = str(app.query_one("#mcp-inspector-state", Static).renderable)
        assert "Stale" in badge
        buttons = {b.id: b for b in app.query("Button.mcp-inspector-action")}
        # connect: not wired in Phase 1 -> disabled; view_details: wired -> enabled
        assert buttons["mcp-inspector-action-connect"].disabled
        assert not buttons["mcp-inspector-action-view_details"].disabled


@pytest.mark.asyncio
async def test_wired_action_posts_hub_action_requested():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.update_readiness(_stale_snap())
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-view_details")
        await pilot.pause()
        assert app.events
        assert app.events[-1].action is HubAction.VIEW_DETAILS
        assert app.events[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_advanced_runner_runs_action_with_template_payload():
    app = InspectorApp()
    # Larger viewport: the Advanced pane's rendered section preview plus two
    # Select controls and the payload TextArea exceed the default 80x24 test
    # screen, and pilot.click requires the target to be within the visible
    # region (it does not auto-scroll).
    async with app.run_test(size=(100, 60)) as pilot:
        select = app.query_one("#mcp-adv-action-select", Select)
        assert select.value == "profile.connect"
        payload = app.query_one("#mcp-adv-payload", TextArea)
        assert "demo" in payload.text
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == [("profile.connect", {"profile_id": "demo"})]
        assert "ok" in str(app.query_one("#mcp-adv-result", Static).renderable)


@pytest.mark.asyncio
async def test_advanced_runner_reports_invalid_json_without_crashing():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        payload = app.query_one("#mcp-adv-payload", TextArea)
        payload.text = "{not json"
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == []
        assert "Invalid JSON" in str(app.query_one("#mcp-adv-result", Static).renderable)
