# Tests/UI/test_mcp_inspector.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Select, Static, TextArea

import tldw_chatbook
import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mcp_inspector_module
from tldw_chatbook.MCP.readiness import (
    REASON_LABELS,
    STATE_CSS_CLASSES,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector

_BUNDLED_CSS_PATH = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")


@pytest.fixture(autouse=True)
def _default_advanced_open(monkeypatch):
    """T12: keep the Advanced disclosure expanded, and never touch the real
    user config file, for every test in this module that isn't specifically
    exercising the collapsed-by-default / persistence behavior itself.

    `MCPInspector.compose()` reads `mcp.hub_state.advanced_open` via this
    module's `get_cli_setting` at mount time; without this fixture every
    test here would hit the developer's real `~/.config/tldw_cli/config.toml`
    (non-deterministic) and the pre-T12 tests that `pilot.click` into the
    Advanced pane (e.g. `test_advanced_runner_runs_action_with_template_
    payload`) would fail outright once collapsed-by-default lands, since a
    collapsed `Collapsible`'s contents are `display: none` (not clickable).
    Individual tests below override this locally via their own
    `monkeypatch.setattr(...)` call, which wins over this fixture's.
    """
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True)


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


def _stale_server_snap() -> ReadinessSnapshot:
    """Same RUNTIME_UNAVAILABLE reason as `_stale_snap()`, but server-source.

    T5 only wires CONNECT/VALIDATE/REFRESH_DISCOVERY for local-source
    snapshots (the workbench can only run the typed T2 lifecycle methods
    against local profiles) -- a server-source server with the same reason
    keeps those actions disabled, pointed at Advanced instead.
    """
    return ReadinessSnapshot(
        server_key="server:main/docs", label="docs", source="server",
        state=ReadinessState.STALE, reasons=(ReasonCode.RUNTIME_UNAVAILABLE,),
        message="2 tools discovered; not currently connected.",
    )


def _ready_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:notes", label="notes", source="local",
        state=ReadinessState.READY, reasons=(),
        message="Connected — 4 tools available.", tool_count=4,
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
        # T5: connect is wired for local-source snapshots (was disabled in
        # Phase 1); view_details was already wired in Phase 1.
        assert not buttons["mcp-inspector-action-connect"].disabled
        assert not buttons["mcp-inspector-action-view_details"].disabled
        # Every rendered action button -- wired or not -- must explain its
        # outcome via a tooltip (destination-wide "every button explains
        # itself" contract; wired buttons previously had none).
        for button in buttons.values():
            assert button.tooltip, f"{button.id} has no tooltip"

        # T5: the same lifecycle action on a server-source snapshot stays
        # disabled -- it's managed server-side, not from this local-lifecycle
        # pane -- with a distinct "use Advanced" tooltip.
        await inspector.update_readiness(_stale_server_snap())
        await pilot.pause()
        server_connect = app.query_one("#mcp-inspector-action-connect", Button)
        assert server_connect.disabled
        assert "server" in (server_connect.tooltip or "").lower()


# -- Task 11: status color class on the readiness badge ----------------------


@pytest.mark.asyncio
async def test_readiness_badge_carries_and_swaps_state_css_class():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert STATE_CSS_CLASSES[ReadinessState.STALE] in badge.classes

        await inspector.update_readiness(_ready_snap())
        await pilot.pause()
        assert STATE_CSS_CLASSES[ReadinessState.READY] in badge.classes
        assert STATE_CSS_CLASSES[ReadinessState.STALE] not in badge.classes

        await inspector.update_readiness(None)
        await pilot.pause()
        assert STATE_CSS_CLASSES[ReadinessState.READY] not in badge.classes


# -- A2: disabled action buttons must stay legible ---------------------------


class InspectorAppWithBundledCSS(App):
    """Mounts MCPInspector under `#mcp-hub-inspector` (the id the real MCP
    workbench uses) and loads the actual bundled stylesheet, so
    `#mcp-hub-inspector Button.mcp-inspector-action:disabled` resolves
    exactly as it does in the live app. A bare `App()` with no `CSS_PATH`
    only exercises Textual's own built-in `Button:disabled` defaults, not the
    project's `_buttons.tcss` override -- `opacity: 50%` stacked on
    `$text-disabled` on `$surface-darken-1` -- that actually causes the
    "nearly invisible" bug this fix addresses. Mirrors `RailAppWithBundledCSS`
    in test_mcp_rail.py.
    """

    CSS_PATH = _BUNDLED_CSS_PATH

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-hub-inspector")


@pytest.mark.asyncio
async def test_disabled_action_buttons_stay_legible_with_bundled_css():
    """A2: `Button.mcp-inspector-action:disabled` must win over the generic
    `Button:disabled` rule and stay at full opacity with a dim-but-readable
    color, instead of the 50%-opacity-on-$text-disabled combination that
    renders as functionally invisible on top of `.console-action-secondary`.
    """
    app = InspectorAppWithBundledCSS()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        # T5 wires CONNECT for local-source snapshots -- use the
        # server-source variant here so this button is still disabled and
        # the legibility contract under test still has something to check.
        await inspector.update_readiness(_stale_server_snap())
        await pilot.pause()
        connect_button = app.query_one("#mcp-inspector-action-connect", Button)
        assert connect_button.disabled
        # The generic Button:disabled rule (_buttons.tcss) sets opacity: 50%;
        # that -- not just the color choice -- is what made the button read
        # as nearly invisible. The dedicated rule must restore full opacity.
        assert connect_button.styles.opacity == 1.0
        # Tooltip must survive (A2 explicitly keeps existing tooltips).
        assert connect_button.tooltip


# -- A3/A5: humanized reason copy, no raw reason codes -----------------------


@pytest.mark.asyncio
async def test_readiness_message_leads_with_humanized_reason_not_raw_code():
    """A3a/A5: the inspector's second line must lead with `Why · <label>`
    from REASON_LABELS, never the bracketed internal reason code, and must
    not just repeat the canvas's own snapshot.message verbatim.
    """
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        snap = _stale_snap()
        await inspector.update_readiness(snap)
        await pilot.pause()
        message = str(app.query_one("#mcp-inspector-message", Static).renderable)
        assert message == f"Why · {REASON_LABELS[ReasonCode.RUNTIME_UNAVAILABLE]}"
        assert "[runtime_unavailable]" not in message
        assert "runtime_unavailable" not in message
        assert snap.message not in message


@pytest.mark.asyncio
async def test_readiness_message_ready_state_shows_tool_count_not_raw_message():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        snap = _ready_snap()
        await inspector.update_readiness(snap)
        await pilot.pause()
        message = str(app.query_one("#mcp-inspector-message", Static).renderable)
        assert message == "Why · Ready — 4 tools available"
        assert snap.message not in message


@pytest.mark.asyncio
async def test_readiness_message_ready_state_without_tool_count_omits_count():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        snap = ReadinessSnapshot(
            server_key="builtin:tldw_chatbook", label="tldw_chatbook (built-in)",
            source="builtin", state=ReadinessState.READY, reasons=(),
            message="Served over stdio when an MCP client launches chatbook.",
        )
        await inspector.update_readiness(snap)
        await pilot.pause()
        message = str(app.query_one("#mcp-inspector-message", Static).renderable)
        assert message == "Why · Ready"
        assert snap.message not in message


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


# -- Task 4: serialized readiness refresh + zero-descriptor Advanced hint ---


@pytest.mark.asyncio
async def test_concurrent_refreshes_serialize_and_last_writer_wins():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        first = _stale_snap()
        second = ReadinessSnapshot(
            server_key="local:web", label="web", source="local",
            state=ReadinessState.READY, reasons=(), message="Connected.",
        )
        await asyncio.gather(
            inspector.update_readiness(first),
            inspector.update_readiness(second),
        )
        await pilot.pause()
        buttons = list(app.query("Button.mcp-inspector-action"))
        assert buttons, "actions must render"
        # last writer wins exactly once: READY action set, no duplicates
        ids = [b.id for b in buttons]
        assert len(ids) == len(set(ids))
        assert inspector._snapshot.server_key == "local:web"


@pytest.mark.asyncio
async def test_zero_descriptor_sections_show_guidance_hint():
    app = InspectorApp()  # FakeAdvService returns one action; override to none
    app.service.available_actions = lambda: []
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(app.service, [("Overview", "overview")])
        await pilot.pause()
        hint = app.query_one("#mcp-adv-empty-hint", Static)
        assert hint.display
        assert "Inventory" in str(hint.renderable)


# -- Task 12: Advanced disclosure (Collapsible) + object label --------------


@pytest.mark.asyncio
async def test_advanced_collapsible_starts_collapsed_by_default(monkeypatch):
    """No persisted preference (fresh install) -> collapsed on mount."""
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: False)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed is True


@pytest.mark.asyncio
async def test_advanced_collapsible_starts_expanded_when_persisted_open(monkeypatch):
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed is False


@pytest.mark.asyncio
async def test_advanced_collapsible_toggle_persists_state(monkeypatch):
    """Expanding the disclosure must persist `advanced_open=True` via
    `save_setting_to_cli_config("mcp.hub_state", "advanced_open", True)`,
    per the task interface's exact call-signature contract."""
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: False)
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed is True

        collapsible.collapsed = False
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("mcp.hub_state", "advanced_open", True) in save_calls


@pytest.mark.asyncio
async def test_advanced_collapsible_recollapse_persists_false(monkeypatch):
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed is False

        collapsible.collapsed = True
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("mcp.hub_state", "advanced_open", False) in save_calls


@pytest.mark.asyncio
async def test_advanced_object_label_defaults_to_local_control_plane():
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: Local control plane"


@pytest.mark.asyncio
async def test_advanced_object_label_reflects_server_source_and_target():
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Overview", "overview")],
            source="server", target_label="Main Server",
        )
        await pilot.pause()
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: server Main Server"


@pytest.mark.asyncio
async def test_advanced_content_cleared_synchronously_on_rebind():
    """A rebind (`set_service_context()` called again -- e.g. on a workbench
    source/target switch) must blank the previous section's rendered dump
    SYNCHRONOUSLY, before the reload worker even starts, so a stale object's
    facts can never linger on screen even for one frame (UX-inputs
    acceptance)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        content = app.query_one("#mcp-adv-content", Static)
        assert str(content.renderable), "sanity: overview section rendered something"

        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Overview", "overview")],
            source="server", target_label="Other Server",
        )
        # No pilot.pause() here: the clear must be visible before the
        # reload worker this call schedules has had any chance to run.
        assert str(content.renderable) == ""
