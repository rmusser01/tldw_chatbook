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


def _ready_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:notes", label="notes", source="local",
        state=ReadinessState.READY, reasons=(),
        message="Connected — 4 tools available.",
    )


@pytest.mark.asyncio
async def test_readiness_block_shows_state_message_and_action_buttons():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        badge = str(app.query_one("#mcp-inspector-state", Static).renderable)
        assert "Stale" in badge
        buttons = {b.id: b for b in app.query("Button.mcp-inspector-action")}
        # connect: not wired in Phase 1 -> disabled; view_details: wired -> enabled
        assert buttons["mcp-inspector-action-connect"].disabled
        assert not buttons["mcp-inspector-action-view_details"].disabled
        # Every rendered action button -- wired or not -- must explain its
        # outcome via a tooltip (destination-wide "every button explains
        # itself" contract; wired buttons previously had none).
        for button in buttons.values():
            assert button.tooltip, f"{button.id} has no tooltip"


@pytest.mark.asyncio
async def test_wired_action_posts_hub_action_requested():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-view_details")
        await pilot.pause()
        assert app.events
        assert app.events[-1].action is HubAction.VIEW_DETAILS
        assert app.events[-1].server_key == "local:docs"


# -- P0: DuplicateIds race on back-to-back readiness updates ----------------
#
# `update_readiness()` rebuilds the action-button list by calling
# `remove_children()` then `mount()`. Before the fix, neither call was
# awaited, so a second `update_readiness()` invocation that starts before the
# first's `remove_children()` has actually pruned its buttons from the DOM
# tries to mount a same-id button (both snapshots below include
# `view_details`, as almost every readiness reason does) while the old one is
# still registered -- Textual raises `DuplicateIds` and the whole app
# crashes. Selecting a second server in Servers mode reproduces this on any
# two-click session. The regression test below drives two updates back to
# back with NO intervening `pilot.pause()` -- the only way to prove the
# remove+mount cycle is now fully serialized within one awaited call, rather
# than merely "usually fast enough in practice".
@pytest.mark.asyncio
async def test_second_update_readiness_does_not_duplicate_action_ids():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        # No pilot.pause() here: the second call must start (and its own
        # remove+mount cycle must fully resolve) before this coroutine
        # returns, exactly like a second rail click arriving while the
        # first selection's inspector refresh is still settling.
        await inspector.update_readiness(_ready_snap())
        await pilot.pause()

        buttons = list(app.query("Button.mcp-inspector-action"))
        ids = [b.id for b in buttons]
        assert len(ids) == len(set(ids)), f"duplicate action button ids: {ids}"

        expected_ids = {
            f"mcp-inspector-action-{action.value}" for action in _ready_snap().allowed_actions
        }
        assert set(ids) == expected_ids, (
            f"actions container should hold exactly the second snapshot's "
            f"buttons; got {set(ids)!r}, expected {expected_ids!r}"
        )


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


class GatedAdvService(FakeAdvService):
    """Fake advanced service exposing the runtime_state_override seam.

    Combined with an app that defines `require_ui_action_allowed`, this makes
    both policy-gate seams present so `MCPInspector._action_allowed` actually
    invokes the gate instead of short-circuiting to permissive.
    """

    def runtime_state_override(self):
        return object()  # the gate fakes below ignore the value


class GatedInspectorApp(App):
    """Like InspectorApp, but with a real (callable) policy-gate seam.

    `gate` is invoked as `gate(action_id, runtime_state_override)` in place of
    `app.require_ui_action_allowed(...)`.
    """

    def __init__(self, gate) -> None:
        super().__init__()
        self.service = GatedAdvService()
        self._gate = gate

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        inspector = self.query_one(MCPInspector)
        inspector.set_service_context(self.service, [("Overview", "overview")])

    def require_ui_action_allowed(self, *, action_id: str, runtime_state_override):
        return self._gate(action_id, runtime_state_override)


class _Decision:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed


@pytest.mark.asyncio
async def test_gate_exception_fails_closed_action_not_offered():
    """A raising policy gate must hide the action, not expose it (fail closed)."""

    def _raise(action_id, runtime_state_override):
        raise RuntimeError("policy engine unavailable")

    app = GatedInspectorApp(_raise)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        select = app.query_one("#mcp-adv-action-select", Select)
        # No allowed descriptors survive -> the select falls back to its
        # empty state (disabled, blank value) exactly like the
        # zero-descriptors case in _refresh_advanced_actions.
        assert select.disabled
        assert select.value is Select.BLANK
        offered_values = [value for _, value in select._options]
        assert "profile.connect" not in offered_values


class SectionAwareFakeService:
    """Mirrors `UnifiedMCPControlPlaneService` semantics: `available_actions()`
    depends on the section that `load_section()` last selected -- Phase 1's
    governance/inventory/advanced actions only exist once the matching
    section has actually been loaded."""

    def __init__(self) -> None:
        self.section = "overview"  # fresh-context default
        self.run_calls: list[tuple[str, dict]] = []

    async def load_section(self, section=None):
        self.section = section or self.section
        return {"source": "local", "section": self.section}

    def available_actions(self):
        if self.section == "external_servers":
            return [
                {
                    "name": "profile.connect",
                    "label": "Connect Profile",
                    "action_id": "x",
                    "payload_template": '{"profile_id":"demo"}',
                }
            ]
        if self.section == "governance":
            return [
                {
                    "name": "governance_rule.save",
                    "label": "Save Governance Rule",
                    "action_id": "y",
                    "payload_template": "{}",
                }
            ]
        return []  # overview / inventory-not-modeled etc.

    async def run_action(self, action_name, payload):
        self.run_calls.append((action_name, dict(payload or {})))
        return {"ok": True}


class SectionAwareInspectorApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.service = SectionAwareFakeService()

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="insp")

    def on_mount(self) -> None:
        self.query_one(MCPInspector).set_service_context(
            self.service,
            [
                ("Overview", "overview"),
                ("External Servers", "external_servers"),
                ("Governance", "governance"),
            ],
        )


@pytest.mark.asyncio
async def test_advanced_actions_follow_section_changes():
    """C2 regression: switching the Advanced section must re-derive actions.

    Before the fix, `available_actions()` was only consulted once, in
    `set_service_context()`. Changing the Advanced section only reloaded the
    rendered content, leaving governance/inventory/advanced actions
    (governance_rule.save, runtime.access.preview, resource.read,
    prompt.get, ...) permanently unreachable -- a capability regression vs
    the legacy panel, which re-synced actions per section.
    """
    app = SectionAwareInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        action_select = app.query_one("#mcp-adv-action-select", Select)
        run_btn = app.query_one("#mcp-adv-run", Button)
        # Fresh-context default section ("overview") has zero descriptors.
        assert action_select.disabled
        assert action_select.value is Select.BLANK
        assert run_btn.disabled

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "governance"
        await pilot.pause()
        await pilot.pause()

        assert app.service.section == "governance"
        assert action_select.value == "governance_rule.save", (
            f"stale actions: {action_select.value!r} (disabled={action_select.disabled})"
        )
        assert not action_select.disabled
        assert not run_btn.disabled


@pytest.mark.asyncio
async def test_advanced_actions_zero_descriptor_section_resets_payload_to_empty_object():
    """C2 fix detail: switching to a zero-descriptor section must reset the
    payload TextArea to "{}" (legacy panel behavior), not leave a stale
    template from whatever action was previously selected."""
    app = SectionAwareInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        section_select = app.query_one("#mcp-adv-section-select", Select)
        payload = app.query_one("#mcp-adv-payload", TextArea)

        section_select.value = "governance"
        await pilot.pause()
        await pilot.pause()
        assert "{}" == payload.text  # governance_rule.save's own template

        payload.text = "not empty"
        section_select.value = "overview"
        await pilot.pause()
        await pilot.pause()

        action_select = app.query_one("#mcp-adv-action-select", Select)
        assert action_select.disabled
        assert action_select.value is Select.BLANK
        assert payload.text == "{}"


class OverlappingActionsService:
    """Two sections that share one action name (by design, not by accident)
    -- used to prove the action re-derivation preserves selection instead of
    always resetting to the new section's first option."""

    def __init__(self) -> None:
        self.section = "overview"

    async def load_section(self, section=None):
        self.section = section or self.section
        return {"source": "local", "section": self.section}

    def available_actions(self):
        if self.section == "alpha":
            return [
                {"name": "action.a", "label": "Action A", "action_id": "a", "payload_template": "{}"},
                {"name": "action.shared", "label": "Shared Action", "action_id": "shared",
                 "payload_template": '{"x":1}'},
            ]
        if self.section == "beta":
            return [
                {"name": "action.shared", "label": "Shared Action", "action_id": "shared",
                 "payload_template": '{"x":1}'},
                {"name": "action.b", "label": "Action B", "action_id": "b", "payload_template": "{}"},
            ]
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class OverlappingActionsApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.service = OverlappingActionsService()

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="insp")

    def on_mount(self) -> None:
        self.query_one(MCPInspector).set_service_context(
            self.service, [("Alpha", "alpha"), ("Beta", "beta")]
        )


@pytest.mark.asyncio
async def test_advanced_action_selection_preserved_across_section_switch_when_still_valid():
    """C2 fix detail: legacy parity -- if the currently selected action name
    is still offered by the new section's descriptor set, keep it selected
    instead of resetting to the new section's first option."""
    app = OverlappingActionsApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        action_select = app.query_one("#mcp-adv-action-select", Select)
        action_select.value = "action.shared"
        await pilot.pause()

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "beta"
        await pilot.pause()
        await pilot.pause()

        assert action_select.value == "action.shared"


@pytest.mark.asyncio
async def test_gate_denied_decision_filters_action():
    """A gate that returns allowed=False must filter the action out.

    This exercises the policy-denied branch already present in
    `_action_allowed` before this fix; included here as coverage, not RED
    evidence for the fail-open bug (it may already pass against current
    code).
    """
    app = GatedInspectorApp(lambda action_id, runtime_state_override: _Decision(False))
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        select = app.query_one("#mcp-adv-action-select", Select)
        assert select.disabled
        assert select.value is Select.BLANK
        offered_values = [value for _, value in select._options]
        assert "profile.connect" not in offered_values
