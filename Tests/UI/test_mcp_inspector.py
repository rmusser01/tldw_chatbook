# Tests/UI/test_mcp_inspector.py
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Collapsible, Input, Select, Static, TextArea

import tldw_chatbook
import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mcp_inspector_module
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.hub_test_execution import ToolTestAdmissionPreview
from tldw_chatbook.MCP.local_control_service import MCPGovernanceDenied
from tldw_chatbook.MCP.local_runtime_delegate import RawToolCallRefusedError
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.unified_control_plane_service import (
    MCPHubGateDeniedError,
    _ADVANCED_EXECUTE_GATE_ERROR_MESSAGE,
)
from tldw_chatbook.MCP.readiness import (
    REASON_LABELS,
    STATE_CSS_CLASSES,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


def _fake_get_cli_setting(**overrides: Any):
    """Key-aware `get_cli_setting(section, key=None, default=None)` fake.

    Task 5 (MCP Hub Phase 6): `MCPInspector.compose()` now reads TWO
    `mcp.hub_state` keys -- the pre-existing `advanced_open` (collapsed
    vs. expanded) and the new `advanced_visible` (composed at all, vs. the
    opt-in reveal Button). A blanket `lambda *a, **k: True/False` (the
    pre-Task-5 shape every fixture/test below used) can no longer express
    "expanded but hidden" or "collapsed but visible" -- it answers both
    keys identically. `overrides` maps a KEY name to the value that key
    should resolve to; any key not in `overrides` falls back to the
    caller's own `default` argument, exactly like the real
    `get_cli_setting`.
    """

    def _fake(section: str, key: str | None = None, default: Any = None) -> Any:
        if key in overrides:
            return overrides[key]
        return default

    return _fake


@pytest.fixture(autouse=True)
def _default_advanced_open(monkeypatch):
    """T12/Task 5: keep the Advanced disclosure expanded AND visible, and
    never touch the real user config file, for every test in this module
    that isn't specifically exercising the collapsed-by-default /
    hidden-by-default / persistence behavior itself.

    `MCPInspector.compose()` reads `mcp.hub_state.advanced_open` and
    `mcp.hub_state.advanced_visible` via this module's `get_cli_setting` at
    mount time; without this fixture every test here would hit the
    developer's real `~/.config/tldw_cli/config.toml` (non-deterministic)
    and would see the opt-in reveal Button instead of the composed
    Collapsible its `#mcp-adv-*` queries assume exist. Individual tests
    below override this locally via their own `monkeypatch.setattr(...)`
    call, which wins over this fixture's.
    """
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=True, advanced_visible=True),
    )
    monkeypatch.setattr(
        mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True
    )


class FakeAdvService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.action_calls: list[tuple[str, dict]] = []
        # Fix Round H (PR-T3 review), Item 2c: this fake used to return
        # `{"ok": True}` unconditionally, so no test built on `InspectorApp`
        # could ever drive `_run_advanced_action()`'s refusal-rendering
        # branch (`except (MCPGovernanceDenied, MCPHubGateDeniedError,
        # RawToolCallRefusedError)`) -- mirrors `ToolExecuteAdvService`'s
        # own `error` constructor param below, the established pattern in
        # this file for a raising service double.
        self.error = error

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
        if self.error is not None:
            raise self.error
        return {"ok": True}


class InspectorApp(ConsolidatedCSSApp):
    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__()
        self.service = FakeAdvService(error=error)
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        inspector = self.query_one(MCPInspector)
        inspector.set_service_context(
            self.service, [("Overview", "overview"), ("Inventory", "inventory")]
        )

    def on_mcp_inspector_hub_action_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_tool_test_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_tool_test_preview_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_tool_test_preview_revocation_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_reallow_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_inspector_change_in_permissions_requested(self, event) -> None:
        self.events.append(event)


def _stale_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:docs",
        label="docs",
        source="local",
        state=ReadinessState.STALE,
        reasons=(ReasonCode.RUNTIME_UNAVAILABLE,),
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
        server_key="server:main/docs",
        label="docs",
        source="server",
        state=ReadinessState.STALE,
        reasons=(ReasonCode.RUNTIME_UNAVAILABLE,),
        message="2 tools discovered; not currently connected.",
    )


def _ready_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:notes",
        label="notes",
        source="local",
        state=ReadinessState.READY,
        reasons=(),
        message="Connected — 4 tools available.",
        tool_count=4,
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


def _auth_missing_local_snap() -> ReadinessSnapshot:
    """A local profile with an unresolved env placeholder -- AUTH_MISSING's
    allowed actions are (OPEN_CREDENTIALS, EDIT_CONFIG, VIEW_DETAILS);
    EDIT_CONFIG/VIEW_DETAILS are wired for local source, but OPEN_CREDENTIALS
    never is (no credentials editor exists for either source -- see
    `_wired_actions()`), so it always renders disabled here."""
    return ReadinessSnapshot(
        server_key="local:docs",
        label="docs",
        source="local",
        state=ReadinessState.NEEDS_SETUP,
        reasons=(ReasonCode.AUTH_MISSING,),
        message="Missing environment variables: API_KEY.",
    )


def _not_configured_builtin_snap() -> ReadinessSnapshot:
    """The built-in server turned off -- NOT_CONFIGURED's only
    allowed action is ADD_SERVER, which is never wired for any source."""
    return ReadinessSnapshot(
        server_key="builtin:tldw_chatbook",
        label="tldw_chatbook (built-in)",
        source="builtin",
        state=ReadinessState.NEEDS_SETUP,
        reasons=(ReasonCode.NOT_CONFIGURED,),
        message="Turned off — open to enable.",
    )


@pytest.mark.asyncio
async def test_disabled_action_tooltips_make_no_phase_promise():
    """I2 (MCP Hub Phase 6 finale, review): the program-close decision
    retired "later phase" framing from every disabled-action tooltip --
    Advanced is a standing escape hatch, not a promised-but-unbuilt future.
    A local AUTH_MISSING snapshot's disabled OPEN_CREDENTIALS button gets an
    action-appropriate substitute (Edit config edits the same env
    placeholders); everything else still-unwired (e.g. ADD_SERVER on a
    disabled built-in server) gets the honest generic fallback -- neither
    makes a phase promise or points at a hidden pane."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)

        await inspector.update_readiness(_auth_missing_local_snap())
        await pilot.pause()
        open_credentials = app.query_one(
            "#mcp-inspector-action-open_credentials", Button
        )
        assert open_credentials.disabled
        tooltip = (open_credentials.tooltip or "").lower()
        assert "later phase" not in tooltip
        assert (
            open_credentials.tooltip
            == "Edit the profile's env placeholders via Edit config."
        )
        # The sibling EDIT_CONFIG button really is the honest substitute --
        # confirm it's actually enabled, not just claimed to be.
        assert not app.query_one("#mcp-inspector-action-edit_config", Button).disabled

        await inspector.update_readiness(_not_configured_builtin_snap())
        await pilot.pause()
        add_server = app.query_one("#mcp-inspector-action-add_server", Button)
        assert add_server.disabled
        assert "later phase" not in (add_server.tooltip or "").lower()
        assert add_server.tooltip == "Not available from this panel."


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


class InspectorAppWithBundledCSS(ConsolidatedCSSApp):
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


@pytest.mark.asyncio
async def test_advanced_reveal_button_renders_with_bundled_css(monkeypatch):
    """Task 5 (MCP Hub Phase 6): real-bundle harness assertion for the
    opt-in reveal button -- under the actual app stylesheet (not a bare
    test App's Textual defaults) the button must actually render (non-zero
    region, displayed) with its tooltip, and pressing it must mount the
    Advanced collapsible without any bundle-rule surprise (e.g. a
    `display: none` ancestor rule swallowing it)."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False),
    )
    monkeypatch.setattr(
        mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True
    )
    app = InspectorAppWithBundledCSS()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        assert not app.query("#mcp-adv-collapsible")
        reveal = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert reveal.display
        assert reveal.region.width > 0 and reveal.region.height > 0
        assert reveal.tooltip == "Show the legacy control-plane action runner."

        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        # F-053: the toggle is no longer removed on reveal -- it flips to
        # its "Hide advanced" state so the choice stays reversible.
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"
        assert not toggle.disabled

        # Task 6 dual-layer CSS audit: the reveal-time forced-open panel
        # (the "resources/prompts" reachable content -- governance_rule.*/
        # runtime.access.preview/resource.read/prompt.get action templates
        # live in `#mcp-adv-action-select`, gated behind this same tree)
        # must actually render under the real bundle, not just exist in the
        # DOM. `#mcp-adv-collapsible`/`#mcp-adv-scroll`/`#mcp-adv-payload`
        # have no bundle-layer mirror of their own DEFAULT_CSS geometry --
        # a bare `Collapsible { height: auto; ... }` rule DOES exist in
        # `_widgets.tcss`, the same bare-type-selector shape as the
        # Select/Checkbox lessons this audit exists to catch -- but the id-
        # scoped `#mcp-adv-collapsible`/`.-collapsed` rules already
        # outrank it on specificity alone, verified here rather than
        # assumed. No bundle-layer rule was added for any of these -- this
        # test is the verification, not a fix.
        assert not collapsible.collapsed, (
            "reveal must land expanded under the real bundle too"
        )
        assert collapsible.size.width > 0 and collapsible.size.height > 0, (
            "Advanced collapsible collapsed to zero geometry under bundled CSS"
        )
        scroll = app.query_one("#mcp-adv-scroll")
        assert scroll.size.width > 0 and scroll.size.height > 0, (
            "#mcp-adv-scroll collapsed to zero geometry under bundled CSS"
        )
        payload = app.query_one("#mcp-adv-payload", TextArea)
        assert payload.size.width > 0 and payload.size.height > 0, (
            "#mcp-adv-payload collapsed to zero geometry under bundled CSS"
        )


# -- A3: inspector action stack is left-aligned -------------------------------


@pytest.mark.asyncio
async def test_inspector_action_buttons_are_left_aligned_with_bundled_css():
    """A3: Button defaults BOTH `text-align` and `content-align` to center
    (see Textual's own Button.DEFAULT_CSS -- the same lesson already
    documented on `Button.mcp-rail-row` in MCPRail.BUNDLED_CSS and
    `Button.mcp-callout` in _agentic_terminal.tcss). `Button.mcp-inspector-
    action` must override both, or the inspector's action stack (Connect/
    Check readiness/Edit config/... and the lone Cancel button during an
    in-flight lifecycle op) renders each label centered in its full-width
    row instead of left-aligned like every other action list in the hub.
    """
    app = InspectorAppWithBundledCSS()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        action_button = app.query_one("#mcp-inspector-action-view_details", Button)
        assert action_button.styles.text_align == "left"
        assert action_button.styles.content_align_horizontal == "left"

        # The lone Cancel button shown during an in-flight (CHECKING)
        # lifecycle op carries the same class (T5) -- must resolve the same.
        checking_snap = ReadinessSnapshot(
            server_key="local:docs",
            label="docs",
            source="local",
            state=ReadinessState.CHECKING,
            reasons=(),
            message="Connecting…",
        )
        await inspector.update_readiness(checking_snap)
        await pilot.pause()
        cancel_button = app.query_one("#mcp-inspector-cancel", Button)
        assert cancel_button.styles.text_align == "left"
        assert cancel_button.styles.content_align_horizontal == "left"


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
            server_key="builtin:tldw_chatbook",
            label="tldw_chatbook (built-in)",
            source="builtin",
            state=ReadinessState.READY,
            reasons=(),
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
            f"mcp-inspector-action-{action.value}"
            for action in _ready_snap().allowed_actions
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
        assert "Invalid JSON" in str(
            app.query_one("#mcp-adv-result", Static).renderable
        )


class GatedAdvService(FakeAdvService):
    """Fake advanced service exposing the runtime_state_override seam.

    Combined with an app that defines `require_ui_action_allowed`, this makes
    both policy-gate seams present so `MCPInspector._action_allowed` actually
    invokes the gate instead of short-circuiting to permissive.
    """

    def runtime_state_override(self):
        return object()  # the gate fakes below ignore the value


class GatedInspectorApp(ConsolidatedCSSApp):
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


class SectionAwareInspectorApp(ConsolidatedCSSApp):
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


@pytest.mark.asyncio
async def test_protected_actions_reachable_after_reveal(monkeypatch):
    """Task 5 (MCP Hub Phase 6): the six protected actions this task's brief
    calls out by name (governance_rule.save/preview/delete, runtime.access.
    preview, resource.read, prompt.get) must stay reachable once a user
    OPTS IN via the reveal Button -- not just when `advanced_visible` is
    already True at mount (every other test in this module, via the
    autouse fixture). `governance_rule.save` (this probe, reusing
    `SectionAwareInspectorApp`'s governance-section fake -- see
    `test_advanced_actions_follow_section_changes` above) stands in for all
    six: nothing about action rendering/selection/running changed, only
    whether the pane composes at all.

    `on_mount()` calls `set_service_context()` BEFORE the Collapsible
    exists (Advanced starts hidden here) -- this also pins that the
    DOM-tolerant guard in `set_service_context()` doesn't crash, and that
    `_reveal_advanced()`'s replay actually binds the recorded context.
    """
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False, advanced_open=True),
    )
    app = SectionAwareInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        assert not app.query("#mcp-adv-collapsible")

        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # _reveal_advanced() itself is a worker that, partway through,
        # schedules a SECOND worker (set_service_context()'s own
        # _load_advanced_section() reload) -- wait once more so that
        # nested worker is also flushed before the test tears the app
        # down (otherwise its coroutine can be torn down mid-flight,
        # producing a harmless but noisy "never awaited" warning).
        await app.workers.wait_for_complete()
        await pilot.pause()
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "governance"
        await pilot.pause()
        await pilot.pause()

        action_select = app.query_one("#mcp-adv-action-select", Select)
        assert action_select.value == "governance_rule.save"
        assert not action_select.disabled


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
                {
                    "name": "action.a",
                    "label": "Action A",
                    "action_id": "a",
                    "payload_template": "{}",
                },
                {
                    "name": "action.shared",
                    "label": "Shared Action",
                    "action_id": "shared",
                    "payload_template": '{"x":1}',
                },
            ]
        if self.section == "beta":
            return [
                {
                    "name": "action.shared",
                    "label": "Shared Action",
                    "action_id": "shared",
                    "payload_template": '{"x":1}',
                },
                {
                    "name": "action.b",
                    "label": "Action B",
                    "action_id": "b",
                    "payload_template": "{}",
                },
            ]
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class OverlappingActionsApp(ConsolidatedCSSApp):
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
            server_key="local:web",
            label="web",
            source="local",
            state=ReadinessState.READY,
            reasons=(),
            message="Connected.",
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
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=False, advanced_visible=True),
    )
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed is True


@pytest.mark.asyncio
async def test_advanced_collapsible_starts_expanded_when_persisted_open(monkeypatch):
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=True, advanced_visible=True),
    )
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
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=False, advanced_visible=True),
    )
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
async def test_mount_with_persisted_open_does_not_write_config(monkeypatch):
    """Review fix (T12): `Collapsible(collapsed=False)` posts one spurious
    Toggled during mount with zero user interaction (`collapsed` is
    `reactive(True, init=False)`, so constructing it expanded differs from
    the reactive default and fires the watcher -- the same documented quirk
    as library_screen.py's `sync_library_ingest_advanced_open`, whose
    handler is a harmless in-memory sync; ours writes the config file to
    disk). Mounting with the preference already open must therefore produce
    ZERO save calls; only a real toggle afterwards persists -- exactly once.
    """
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=True, advanced_visible=True),
    )
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert save_calls == [], (
            f"mount alone must not write the config; got {save_calls!r}"
        )

        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        collapsible.collapsed = True
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert save_calls == [("mcp.hub_state", "advanced_open", False)]


@pytest.mark.asyncio
async def test_advanced_collapsible_recollapse_persists_false(monkeypatch):
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_open=True, advanced_visible=True),
    )
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


# -- Task 5 (MCP Hub Phase 6): Advanced opt-in gate --------------------------


@pytest.mark.asyncio
async def test_advanced_hidden_by_default_composes_reveal_button_not_collapsible(
    monkeypatch,
):
    """No persisted `advanced_visible` (fresh install) -> the Collapsible
    is not composed at all; a reveal Button stands in for it."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False),
    )
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query("#mcp-adv-collapsible")
        reveal = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert reveal.tooltip == "Show the legacy control-plane action runner."


@pytest.mark.asyncio
async def test_advanced_toggle_hides_and_reshows_round_trip(monkeypatch):
    """F-053: the Advanced opt-in is reversible -- the same control that
    shows the runner hides it again, and each direction persists the user's
    explicit choice (True on show, False on hide) so no future visit is
    trapped in a state the user didn't pick."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False),
    )
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()

    async def click_toggle(pilot) -> None:
        """Click the toggle, waiting out Textual's 0.3s `-active` press
        effect first -- `Button._on_click` deliberately swallows a click
        that lands while the previous press's `-active` class is still on
        the button (textual/widgets/_button.py, 8.2.7), so back-to-back
        round-trip clicks need real time between them."""
        await asyncio.sleep(0.4)
        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Advanced…"

        # Show.
        await click_toggle(pilot)
        assert app.query_one("#mcp-adv-collapsible", Collapsible)
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"
        assert not toggle.disabled
        assert toggle.tooltip == "Hide the legacy control-plane action runner."

        # Hide again -- the one-way door is gone.
        await click_toggle(pilot)
        assert not app.query("#mcp-adv-collapsible")
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Advanced…"
        assert not toggle.disabled
        assert toggle.tooltip == "Show the legacy control-plane action runner."

        visible_writes = [
            value for section, key, value in save_calls if key == "advanced_visible"
        ]
        assert visible_writes == [True, False]

        # And back on again -- still no dead end.
        await click_toggle(pilot)
        assert app.query_one("#mcp-adv-collapsible", Collapsible)


@pytest.mark.asyncio
async def test_advanced_visible_true_at_mount_renders_hide_toggle(monkeypatch):
    """A persisted `advanced_visible=True` (a returning opted-in user) ->
    the Collapsible composes immediately, with the toggle rendered in its
    'Hide advanced' state so the choice stays reversible (F-053)."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=True, advanced_open=True),
    )
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#mcp-adv-collapsible", Collapsible)
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"


@pytest.mark.asyncio
async def test_advanced_reveal_button_persists_setting_and_mounts_collapsible(
    monkeypatch,
):
    """Pressing the reveal Button must persist
    `save_setting_to_cli_config("mcp.hub_state", "advanced_visible", True)`
    and mount the Collapsible alongside it -- the button itself flips to
    its "Hide advanced" state (F-053: reversible), it is not removed."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False),
    )
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query("#mcp-adv-collapsible")

        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # _reveal_advanced() itself is a worker that, partway through,
        # schedules a SECOND worker (set_service_context()'s own
        # _load_advanced_section() reload) -- wait once more so that
        # nested worker is also flushed before the test tears the app
        # down (otherwise its coroutine can be torn down mid-flight,
        # producing a harmless but noisy "never awaited" warning).
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("mcp.hub_state", "advanced_visible", True) in save_calls
        assert app.query_one("#mcp-adv-collapsible", Collapsible)
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"
        assert not toggle.disabled


@pytest.mark.asyncio
async def test_advanced_reveal_expands_regardless_of_persisted_collapsed_state(
    monkeypatch,
):
    """Task 6 review fold: a fresh install has never persisted
    `advanced_open` (it reads as `False`, the same as an explicit "keep it
    collapsed" preference). Pressing "Advanced..." must still land the
    panel EXPANDED -- the user just asked to see it -- and must persist
    `advanced_open=True` (via the same helper the disclosure's own toggle
    uses) so a future mount opens directly instead of reverting to
    collapsed."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False, advanced_open=False),
    )
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert not collapsible.collapsed, (
            "explicit reveal must land expanded even though advanced_open "
            "persisted (or defaulted to) False"
        )
        assert ("mcp.hub_state", "advanced_open", True) in save_calls
        assert ("mcp.hub_state", "advanced_visible", True) in save_calls


@pytest.mark.asyncio
async def test_advanced_reveal_button_mount_time_path_keeps_pure_persistence(
    monkeypatch,
):
    """Companion to the reveal-time forcing test above: the mount-time path
    (`compose()`'s `advanced_visible=True` branch, a returning opted-in
    user) must NOT be forced open -- a persisted `advanced_open=False`
    stands, exactly as before this fold."""
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=True, advanced_open=False),
    )
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        assert collapsible.collapsed


@pytest.mark.asyncio
async def test_advanced_reveal_second_press_while_saving_is_a_no_op(monkeypatch):
    """Review fix: a second press while worker A is genuinely mid-save
    (blocked inside the `asyncio.to_thread(save_setting_to_cli_config,
    ...)` call, not merely queued-but-not-started) must be a no-op, not a
    cancel-and-restart. Before the fix, `on_button_pressed` unconditionally
    rescheduled `_reveal_advanced()` into the same `exclusive=True` group
    on every Pressed with no synchronous disable -- a second press landing
    here CANCELLED worker A after it had already set
    `self._advanced_visible = True` but before it removed the button or
    mounted the collapsible, leaving a dead-looking button stuck forever
    (every future call short-circuits on that same flag, since it's
    already True). `save_setting_to_cli_config` runs on a real thread (via
    `asyncio.to_thread`) -- a real `threading.Event` gate lets this test
    hold worker A there deterministically while the second press fires,
    unlike a synchronous double `Button.press()` (which races worker A's
    own task-start and typically cancels it before it runs at all rather
    than mid-save).
    """
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False),
    )
    gate = threading.Event()
    save_calls: list[tuple[str, str, Any]] = []

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        if key == "advanced_visible":
            # Block here (on the real to_thread worker thread) until the
            # test explicitly releases it -- this is "mid-save".
            assert gate.wait(timeout=5), "test gate was never released"
        return True

    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", fake_save)
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        reveal_button = app.query_one("#mcp-inspector-advanced-reveal", Button)
        reveal_button.press()
        # Let worker A actually start and reach the blocked save call --
        # several pumps to give the `asyncio.to_thread` dispatch a real
        # chance to land on the gate before the second press fires.
        for _ in range(5):
            await pilot.pause()

        # With the fix, the handler disabled the button synchronously
        # before scheduling worker A -- `Button.press()` itself refuses to
        # post a second `Pressed` at all for an already-disabled button
        # (returns early without `post_message`), so this second call is a
        # genuine no-op rather than a swallowed message.
        assert reveal_button.disabled
        reveal_button.press()
        await pilot.pause()

        gate.set()  # release worker A's blocked save
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        collapsibles = app.query("#mcp-adv-collapsible")
        assert len(collapsibles) == 1, (
            f"expected exactly one mounted collapsible, got {len(collapsibles)}"
        )
        toggle = app.query_one("#mcp-inspector-advanced-reveal", Button)
        assert str(toggle.label) == "Hide advanced"
        assert not toggle.disabled
        assert save_calls.count(("mcp.hub_state", "advanced_visible", True)) == 1


@pytest.mark.asyncio
async def test_advanced_reveal_replays_recorded_service_context(monkeypatch):
    """`set_service_context()` may be called while Advanced is still
    hidden (the workbench rebinds unconditionally on every reload/source
    switch/selection change) -- that call must not crash on the missing
    `#mcp-adv-*` widgets, and once the user reveals Advanced, the freshly
    mounted panel must bind to whatever was last recorded (source="server",
    a named target) rather than opening on the local-control-plane default.
    """
    monkeypatch.setattr(
        mcp_inspector_module,
        "get_cli_setting",
        _fake_get_cli_setting(advanced_visible=False, advanced_open=True),
    )
    app = InspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        # Recorded while hidden -- must not raise (NoMatches on #mcp-adv-*).
        inspector.set_service_context(
            app.service,
            [("Overview", "overview")],
            source="server",
            target_label="Main Server",
        )
        await pilot.pause()

        await pilot.click("#mcp-inspector-advanced-reveal")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # _reveal_advanced() itself is a worker that, partway through,
        # schedules a SECOND worker (set_service_context()'s own
        # _load_advanced_section() reload) -- wait once more so that
        # nested worker is also flushed before the test tears the app
        # down (otherwise its coroutine can be torn down mid-flight,
        # producing a harmless but noisy "never awaited" warning).
        await app.workers.wait_for_complete()
        await pilot.pause()

        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: server Main Server"


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
            app.service,
            [("Overview", "overview")],
            source="server",
            target_label="Main Server",
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
            app.service,
            [("Overview", "overview")],
            source="server",
            target_label="Other Server",
        )
        # No pilot.pause() here: the clear must be visible before the
        # reload worker this call schedules has had any chance to run.
        assert str(content.renderable) == ""


# -- Task 6: tool detail view + Test Tool runner -----------------------------


def _tool(**overrides: Any) -> HubTool:
    base: dict[str, Any] = dict(
        server_key="local:docs",
        server_label="docs",
        source="local",
        name="search",
        description="Search the docs.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search text"}},
            "required": ["query"],
        },
        tags=(),
        stale=False,
        executable=True,
    )
    base.update(overrides)
    return HubTool(**base)


def _capture_notifications(app: App) -> list[tuple[str, str]]:
    """Shadow `app.notify` with a recorder; returns the (message, severity)
    list it appends to. Mirrors `Tests/UI/test_mcp_workbench.py`'s own
    helper of the same name -- kept as a separate copy since these are
    independent test modules with no shared fixture file."""
    notifications: list[tuple[str, str]] = []

    def recording_notify(message, *, title="", severity="information", **kwargs):
        notifications.append((str(message), severity))

    app.notify = recording_notify
    return notifications


@pytest.mark.asyncio
async def test_show_tool_renders_executable_tool_with_test_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        container = app.query_one("#mcp-inspector-tool")
        assert container.display is True
        name_text = str(app.query_one("#mcp-inspector-tool-name", Static).renderable)
        assert "search" in name_text
        assert "docs" in name_text
        description = str(
            app.query_one("#mcp-inspector-tool-description", Static).renderable
        )
        assert description == "Search the docs."
        schema_line = str(
            app.query_one("#mcp-inspector-tool-schema", Static).renderable
        )
        assert schema_line == "Parameters: form"
        test_button = app.query_one("#mcp-inspector-test-tool", Button)
        assert test_button.tooltip == "Run this tool with test arguments."
        assert not list(app.query("#mcp-inspector-tool-phase-note"))
        assert not list(app.query("#mcp-inspector-tool-stale"))


@pytest.mark.asyncio
async def test_show_tool_raw_schema_reports_raw_json_availability():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool(name="fetch", input_schema=None))
        await pilot.pause()
        schema_line = str(
            app.query_one("#mcp-inspector-tool-schema", Static).renderable
        )
        assert schema_line == "Parameters: raw JSON"


@pytest.mark.asyncio
async def test_show_tool_stale_shows_stale_note():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool(stale=True))
        await pilot.pause()
        stale = app.query_one("#mcp-inspector-tool-stale", Static)
        assert str(stale.renderable)


@pytest.mark.asyncio
async def test_show_tool_non_executable_shows_phase4_note_not_test_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(source="server", server_key="server:main/docs", executable=False)
        )
        await pilot.pause()
        note = app.query_one("#mcp-inspector-tool-phase-note", Static)
        assert str(note.renderable) == "Server-source tools are display-only."
        assert not list(app.query("#mcp-inspector-test-tool"))


@pytest.mark.asyncio
async def test_show_tool_none_hides_and_clears_container():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await inspector.show_tool(None)
        await pilot.pause()
        container = app.query_one("#mcp-inspector-tool")
        assert container.display is False
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_second_show_tool_back_to_back_does_not_duplicate_ids():
    """Mandatory regression: selecting two tools in a row must not raise
    DuplicateIds -- mirrors update_readiness's own back-to-back precedent
    (test_second_update_readiness_does_not_duplicate_action_ids)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool(name="search"))
        # No pilot.pause() here on purpose.
        await inspector.show_tool(_tool(name="fetch"))
        await pilot.pause()
        names = list(app.query("#mcp-inspector-tool-name"))
        assert len(names) == 1
        assert "fetch" in str(names[0].renderable)


@pytest.mark.asyncio
async def test_test_tool_button_mounts_form_run_close_and_result():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        assert app.query_one("#mcp-inspector-test-form")
        assert app.query_one("#mcp-inspector-test-run", Button)
        assert app.query_one("#mcp-inspector-test-close", Button)
        result = app.query_one("#mcp-inspector-test-result", Static)
        assert str(result.renderable) == ""
        assert app.query_one("#mcp-inspector-test-tool", Button).disabled is True


@pytest.mark.asyncio
async def test_test_run_posts_tool_test_requested_with_collected_arguments():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="allow"))
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        events = [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert len(events) == 1
        assert events[0].server_key == tool.server_key
        assert events[0].tool_name == tool.name
        assert events[0].arguments == {"query": "hello"}
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is True


@pytest.mark.asyncio
async def test_test_run_value_error_shows_message_and_does_not_post():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="allow"))
        # required "query" field left empty
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        events = [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert events == []
        result = app.query_one("#mcp-inspector-test-result", Static)
        assert "required" in str(result.renderable)


@pytest.mark.asyncio
async def test_test_run_value_error_result_gets_failed_prefix():
    """F4 (PR-T3 task 3): this was the ONE result write in this module
    with no status prefix at all -- `_handle_test_run()`'s `except
    ValueError` branch used to write the bare exception message
    (`result_widget.update(str(exc))`), unlike every other write here
    ("OK"/"Failed"/"Blocked · not run"). It now leads with "Failed"."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="allow"))
        # required "query" field left empty
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Failed\n")
        assert "required" in result


@pytest.mark.asyncio
async def test_raw_mode_tool_test_panel_shows_raw_textarea():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(name="fetch", input_schema=None)
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="allow"))
        raw_area = app.query_one("#mcp-schema-raw", TextArea)
        raw_area.text = '{"url": "https://example.test"}'
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        events = [
            e for e in app.events if isinstance(e, MCPInspector.ToolTestRequested)
        ]
        assert len(events) == 1
        assert events[0].arguments == {"url": "https://example.test"}


@pytest.mark.asyncio
async def test_show_tool_result_ok_renders_status_line_and_reenables_run():
    """RAG-49 deliberate contract change: an OK result built from the new
    structured `result=`/`source=` kwargs (rather than a pre-flattened
    `text=` string) renders a `OK · <source> · <duration> · N results`
    summary line, and the raw payload moves into a collapsed "Raw
    response" Collapsible instead of being echoed inline in the summary
    Static. The old pin (`OK · 123ms` prefix + `'{"ok": true}'` inline
    containment) is superseded by this."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=123,
            source="local",
            result=[{"id": 1}, {"id": 2}, {"id": 3}],
            raw='{"ok": true}',
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 123ms · 3 results"
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False
        raw_collapsible = app.query_one("#mcp-inspector-test-result-raw", Collapsible)
        assert raw_collapsible.display is not False
        assert raw_collapsible.collapsed is True
        raw_body = str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert '{"ok": true}' in raw_body


@pytest.mark.asyncio
async def test_show_tool_result_ok_list_of_one_uses_singular_result():
    """Behavior contract: N == 1 reads "1 result" (singular), not "1
    results"."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=981,
            source="local",
            result=[{"id": 1}],
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 981ms · 1 result"


@pytest.mark.asyncio
async def test_show_tool_result_ok_empty_list_shows_zero_results_and_quiet_line():
    """Behavior contract: `result == []` reads "0 results" plus a quiet
    interpretation line explaining the empty result, in a sibling Static."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=981,
            source="local",
            result=[],
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 981ms · 0 results"
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert note == "The tool ran and returned no results."


@pytest.mark.asyncio
async def test_show_tool_result_ok_error_shape_result_shows_error_interpretation():
    """Behavior contract: a `[{"error": "..."}]` result (the MCP/tools.py
    tool-error contract) reads "tool returned an error" on the summary line,
    with the error string as the interpretation."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=981,
            source="local",
            result=[{"error": "Tool boom"}],
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 981ms · tool returned an error"
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert note == "Tool boom"


@pytest.mark.asyncio
async def test_show_tool_result_ok_non_list_result_has_no_count_segment():
    """Behavior contract: a non-list result (dict/str/number) gets no count
    segment at all -- just `OK · <source> · <duration>`."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=981,
            source="local",
            result={"ok": True},
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 981ms"


@pytest.mark.asyncio
async def test_show_tool_result_ok_without_source_omits_source_segment():
    """Interfaces contract: the `source` segment is present only when
    known -- an unknown/absent source drops the segment entirely rather
    than rendering an empty one."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            result=[{"id": 1}],
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · 50ms · 1 result"


@pytest.mark.asyncio
async def test_show_tool_result_raw_body_truncated_over_20000_chars():
    """Behavior contract: the raw body is capped at 20,000 chars with a
    trailing `… truncated (showing 20000 of N chars)` note when over --
    the old 500-char cap is retired on this path."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        big_raw = "x" * 25_000
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=10,
            source="local",
            result={"ok": True},
            raw=big_raw,
        )
        await pilot.pause()
        raw_body = str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert raw_body.startswith("x" * 100)
        assert raw_body.endswith("… truncated (showing 20000 of 25000 chars)")
        assert len(raw_body) < 25_000


# PR-T3 task 2 (F1): the MCP result says what it found. `_summarize_tool_
# result()` is generic across every MCP tool -- only a result whose rows
# ALL carry a "score" key (the RAG-search row shape) may grow the honest
# all-weak notice; every other tool's rows (e.g. `list_characters`) must
# render byte-identical to today. These four cover the task's own test
# list verbatim: (a) all-weak -> notice; (b) mixed incl. a strong/moderate
# score -> no notice; (c) no "score" key at all -> byte-identical; (d)
# empty list -> today's quiet line, unchanged.
class TestSummarizeToolResultAllWeakNotice:
    """Pure, UI-harness-free coverage of `_summarize_tool_result()` --
    mirrors `test_error_shape_detection()`'s pattern below."""

    def test_extract_scored_rows_reads_nested_hybrid_vector_score(self):
        """Fusion's RRF score is not a vector-similarity value."""
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _extract_scored_rows,
            _summarize_tool_result,
        )

        rows = [
            {
                "id": 1,
                "score": 0.016,
                "metadata": {
                    "hybrid_fusion": {
                        "fts_rank": 1,
                        "vector_rank": 1,
                        "fts_score": 0.001,
                        "vector_score": 0.8,
                    },
                },
            }
        ]

        extracted = _extract_scored_rows(rows)

        assert extracted is not None
        assert extracted[0].score_kind == "hybrid_fusion"
        assert extracted[0].vector_score == pytest.approx(0.8)
        _, interpretation = _summarize_tool_result(
            ok=True,
            duration_ms=50,
            source="local",
            result=rows,
        )
        assert interpretation is None

    def test_score_provenance_controls_all_weak_notice(self):
        """Only actual vector similarities participate in weak-match bands."""
        from tldw_chatbook.Library.library_rag_state import (
            LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX,
        )
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _extract_scored_rows,
            _summarize_tool_result,
        )

        cases = (
            (
                {"id": "vector", "score": 0.1, "metadata": {}},
                "vector_similarity",
                None,
                True,
            ),
            (
                {
                    "id": "hybrid-strong-vector",
                    "score": 0.016,
                    "metadata": {
                        "hybrid_fusion": {
                            "fts_rank": 1,
                            "vector_rank": 1,
                            "fts_score": 0.001,
                            "vector_score": 0.8,
                        },
                    },
                },
                "hybrid_fusion",
                0.8,
                False,
            ),
            (
                {
                    "id": "hybrid-weak-vector",
                    "score": 0.016,
                    "metadata": {
                        "hybrid_fusion": {
                            "fts_rank": 1,
                            "vector_rank": 1,
                            "fts_score": 0.001,
                            "vector_score": 0.1,
                        },
                    },
                },
                "hybrid_fusion",
                0.1,
                True,
            ),
            (
                {
                    "id": "fts-only-hybrid",
                    "score": 0.016,
                    "metadata": {
                        "hybrid_fusion": {
                            "fts_rank": 1,
                            "vector_rank": None,
                            "fts_score": 0.001,
                            "vector_score": None,
                        },
                    },
                },
                "hybrid_fusion",
                None,
                False,
            ),
            (
                {
                    "id": "reranked",
                    "score": 7.5,
                    "metadata": {"rerank_score": 7.5},
                },
                "reranker",
                None,
                False,
            ),
            (
                {"id": "keyword", "score": None, "metadata": {}},
                "vector_similarity",
                None,
                False,
            ),
            (
                {
                    "id": "reranking-skipped",
                    "score": 0.1,
                    "metadata": {"reranking_skipped": "no credentials"},
                },
                "vector_similarity",
                None,
                True,
            ),
        )

        for row, score_kind, vector_score, expects_weak_notice in cases:
            extracted = _extract_scored_rows([row])

            assert extracted is not None
            assert extracted[0].score_kind == score_kind
            assert extracted[0].vector_score == vector_score
            _, interpretation = _summarize_tool_result(
                ok=True,
                duration_ms=50,
                source="local",
                result=[row],
            )
            assert interpretation == (
                LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX if expects_weak_notice else None
            )

    def test_malformed_scores_are_unscored_and_never_reported_weak(self):
        """Untrusted booleans and non-finite values are not similarities."""
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _extract_scored_rows,
            _summarize_tool_result,
        )

        invalid_scores = (
            True,
            False,
            float("nan"),
            float("inf"),
            float("-inf"),
            10**309,
            -(10**309),
        )
        for score in invalid_scores:
            vector_row = {"id": "vector", "score": score, "metadata": {}}
            hybrid_row = {
                "id": "hybrid",
                "score": 0.016,
                "metadata": {
                    "hybrid_fusion": {
                        "fts_rank": 1,
                        "vector_rank": 1,
                        "fts_score": 0.001,
                        "vector_score": score,
                    },
                },
            }

            for row, score_field in (
                (vector_row, "score"),
                (hybrid_row, "vector_score"),
            ):
                extracted = _extract_scored_rows([row])

                assert extracted is not None
                assert getattr(extracted[0], score_field) is None
                _, interpretation = _summarize_tool_result(
                    ok=True,
                    duration_ms=50,
                    source="local",
                    result=[row],
                )
                assert interpretation is None

    def test_all_rows_scoring_below_moderate_threshold_adds_all_weak_notice(self):
        from tldw_chatbook.Library.library_rag_state import (
            LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX,
        )
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _summarize_tool_result,
        )

        rows = [{"id": i, "score": 0.19 - i * 0.01} for i in range(10)]
        status_line, interpretation = _summarize_tool_result(
            ok=True,
            duration_ms=50,
            source="local",
            result=rows,
        )
        assert status_line == "OK · local · 50ms · 10 results"
        assert interpretation == LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX

    def test_mixed_scores_including_a_strong_match_adds_no_notice(self):
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _summarize_tool_result,
        )

        rows = [{"id": 1, "score": 0.05}, {"id": 2, "score": 0.5}]
        status_line, interpretation = _summarize_tool_result(
            ok=True,
            duration_ms=50,
            source="local",
            result=rows,
        )
        assert status_line == "OK · local · 50ms · 2 results"
        assert interpretation is None

    def test_rows_without_a_score_key_are_byte_identical_to_today(self):
        """`list_characters`-shaped rows -- must never grow banding."""
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _summarize_tool_result,
        )

        rows = [
            {"id": "1", "name": "Alice", "description": "", "message_count": 0},
            {"id": "2", "name": "Bob", "description": "", "message_count": 3},
        ]
        status_line, interpretation = _summarize_tool_result(
            ok=True,
            duration_ms=50,
            source="local",
            result=rows,
        )
        assert status_line == "OK · local · 50ms · 2 results"
        assert interpretation is None

    def test_empty_list_keeps_its_existing_quiet_line(self):
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
            _summarize_tool_result,
        )

        status_line, interpretation = _summarize_tool_result(
            ok=True,
            duration_ms=50,
            source="local",
            result=[],
        )
        assert status_line == "OK · local · 50ms · 0 results"
        assert interpretation == "The tool ran and returned no results."


@pytest.mark.asyncio
async def test_show_tool_result_ok_all_weak_rows_renders_all_weak_notice():
    """End-to-end (not just the pure function): a real `show_tool_result()`
    call with all-weak scored rows renders the notice in the note Static,
    markup=False, alongside the unchanged raw Collapsible evidence."""
    from tldw_chatbook.Library.library_rag_state import (
        LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX,
    )

    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result=[{"id": 1, "score": 0.1}, {"id": 2, "score": 0.05}],
            raw='[{"id": 1, "score": 0.1}, {"id": 2, "score": 0.05}]',
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 50ms · 2 results"
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert str(note_widget.renderable) == LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX
        assert note_widget.display is True
        raw_body = str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert '"score": 0.1' in raw_body


@pytest.mark.asyncio
async def test_show_tool_result_ok_rows_without_score_key_unaffected():
    """End-to-end guard for the `list_characters` case: no "score" key on
    any row must render exactly as it did before this task -- no note
    widget content, nothing new."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result=[{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}],
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK · local · 50ms · 2 results"
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert str(note_widget.renderable) == ""
        assert note_widget.display is False


def test_error_shape_detection():
    """Unit test for the pure MCP/tools.py:326 tool-error-contract detector
    (a length-1 list whose single element is a mapping with exactly one
    key, "error") -- no UI harness needed."""
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import _is_tool_error_shape

    assert _is_tool_error_shape([{"error": "boom"}])
    assert not _is_tool_error_shape([])
    assert not _is_tool_error_shape([{"error": "x", "id": 1}])
    assert not _is_tool_error_shape([{"id": 1}, {"error": "x"}])
    assert not _is_tool_error_shape({"error": "x"})


@pytest.mark.asyncio
async def test_show_tool_result_failed_renders_status_line():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="boom",
            duration_ms=45,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Failed · 45ms")
        assert "boom" in result


# -- Task 5 (RAG-51): name the permission decision -- `decision_note` shares
# the `#mcp-inspector-test-result-note` Static with the RAG-49 structured
# shape's own `interpretation` line; every pre-existing call site above
# never passes `decision_note` (default `None`), so its behavior is
# unchanged by these additions.


@pytest.mark.asyncio
async def test_show_tool_result_decision_note_renders_alone_with_markup_false():
    """A structured OK run with nothing to interpret (a non-list result)
    shows the decision note alone in the note widget -- rendered `markup=
    False` (the Static was mounted that way; `.update()` respects it)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result={"ok": True},
            decision_note="Ran because you approved this run (the tool is set to Ask).",
        )
        await pilot.pause()
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert (
            str(note_widget.renderable)
            == "Ran because you approved this run (the tool is set to Ask)."
        )
        assert note_widget.display is True


@pytest.mark.asyncio
async def test_show_tool_result_decision_note_does_not_interpret_markup():
    """`#mcp-inspector-test-result-note` was mounted `markup=False` (Task 4,
    `_build_test_result_note_static()`) -- a decision note containing a
    literal `[...]` must render as plain text, not be interpreted/stripped
    as Rich markup, keeping the discipline every other result surface in
    this method already follows even though this copy is trusted/quiet-
    register, not user input."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result={"ok": True},
            decision_note="Ran because this tool is set to Allow. [bold]not styled[/bold]",
        )
        await pilot.pause()
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert "[bold]" in note
        assert note == "Ran because this tool is set to Allow. [bold]not styled[/bold]"


@pytest.mark.asyncio
async def test_show_tool_result_decision_note_and_interpretation_stack():
    """Both facts present (a structured OK run with an empty-result
    interpretation AND a decision note) stack, decision_note first, one per
    line -- distinct, non-overwriting content in the shared note widget."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result=[],
            decision_note="Ran because this tool is set to Allow. From this tool's override.",
        )
        await pilot.pause()
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert note == (
            "Ran because this tool is set to Allow. From this tool's override.\n"
            "The tool ran and returned no results."
        )


@pytest.mark.asyncio
async def test_show_tool_result_decision_note_on_blocked_path():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="Blocked — this tool is set to Off in Permissions.",
            duration_ms=0,
            blocked=True,
            decision_note="This tool is set to Off. From this tool's override.",
        )
        await pilot.pause()
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert note == "This tool is set to Off. From this tool's override."
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Blocked · not run")


@pytest.mark.asyncio
async def test_show_tool_result_blocked_heading_uses_the_shared_constant(monkeypatch):
    """Fix Round A, Minor #4: `_ADVANCED_BLOCKED_HEADING` is documented as
    "reused verbatim" by `show_tool_result()`'s blocked path -- this pins
    that it is actually SOURCED from the constant, not a second hand-typed
    copy of the same string that could silently drift from it."""
    monkeypatch.setattr(
        mcp_inspector_module, "_ADVANCED_BLOCKED_HEADING", "Blocked · patched"
    )
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="boom",
            duration_ms=0,
            blocked=True,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Blocked · patched")


@pytest.mark.asyncio
async def test_show_tool_result_decision_note_on_failed_path():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="boom",
            duration_ms=45,
            decision_note="Ran because this tool is set to Allow. From this tool's override.",
        )
        await pilot.pause()
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        assert (
            note == "Ran because this tool is set to Allow. From this tool's override."
        )
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Failed · 45ms")


@pytest.mark.asyncio
async def test_show_tool_result_no_decision_note_leaves_note_widget_hidden():
    """Every pre-existing call site (no `decision_note` given) must keep
    seeing the exact pre-Task-5 contract: a non-empty `interpretation`
    shows, its absence hides the widget -- unaffected by this task."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            duration_ms=50,
            source="local",
            result={"ok": True},
        )
        await pilot.pause()
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert str(note_widget.renderable) == ""
        assert note_widget.display is False


# -- RAG-51 carried finding (Task 4 review): `format_duration_ms(None)` is
# reachable via the failed/legacy-text branches now that `duration_ms`
# defaults to `None` on the signature (RAG-49) -- both branches used to call
# `format_duration_ms(duration_ms)` unconditionally, which raises `TypeError`
# on `duration_ms < 1000` for `None`. Fixed via `_duration_segment()`
# (mirrors `_summarize_tool_result()`'s own `if duration_ms is not None`
# guard); covered directly here.


@pytest.mark.asyncio
async def test_show_tool_result_failed_with_duration_ms_none_does_not_crash():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="boom",
            duration_ms=None,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "Failed\nboom"


@pytest.mark.asyncio
async def test_show_tool_result_legacy_text_ok_with_duration_ms_none_does_not_crash():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            text="{}",
            duration_ms=None,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == "OK\n{}"


@pytest.mark.asyncio
async def test_show_tool_result_for_a_different_tool_is_dropped():
    """I1: a result for tool A arriving after the inspector has moved on to
    tool B must not render in B's panel, and must not re-enable B's Run
    button on A's behalf (B's own Run press is what should control that)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool_b = _tool(name="fetch", server_key="local:docs", input_schema=None)
        await inspector.show_tool(tool_b)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        run_button = app.query_one("#mcp-inspector-test-run", Button)
        await pilot.click(run_button)
        await pilot.pause()
        assert run_button.disabled is True

        # Tool A's late result arrives under B's server_key/tool_name mismatch.
        inspector.show_tool_result(
            server_key="local:docs",
            tool_name="search",
            ok=True,
            text="A's payload",
            duration_ms=10,
        )
        await pilot.pause()

        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "A's payload" not in result
        assert result == ""
        assert run_button.disabled is True


@pytest.mark.asyncio
async def test_show_tool_result_same_name_different_server_is_dropped():
    """I1 (both fields): a result whose `tool_name` matches the currently
    selected tool but whose `server_key` does NOT must still be dropped --
    the stale-drop compare is a (server_key, tool_name) pair, not just the
    name."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(name="search", server_key="local:docs")
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_tool_result(
            server_key="local:notes",
            tool_name="search",
            ok=True,
            text="wrong server's payload",
            duration_ms=5,
        )
        await pilot.pause()

        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "wrong server's payload" not in result
        assert result == ""


@pytest.mark.asyncio
async def test_show_tool_result_same_tool_is_not_dropped():
    """A result that matches BOTH the current tool's server_key and
    tool_name (e.g. a same-tool re-run) must still render -- the stale-drop
    guard must not become a false-positive drop for the tool it's actually
    for."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(name="search", server_key="local:docs")
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_tool_result(
            server_key="local:docs",
            tool_name="search",
            ok=True,
            text="matching payload",
            duration_ms=7,
        )
        await pilot.pause()

        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "matching payload" in result


# -- Task 3 (PR-T3): "a run that ran always says something" -- the two
# stale-drop guards above keep dropping the RENDER (protected -- the tests
# above pin that silence, unmodified), but now also toast, so a completed
# run is never truly silent. `_handle_test_run()`'s own two silent early
# returns get the same treatment.


@pytest.mark.asyncio
async def test_show_tool_result_for_a_different_tool_still_toasts():
    """The stale-drop guard's render stays dropped (see the protected
    `test_show_tool_result_for_a_different_tool_is_dropped` above,
    unmodified) but now also fires a toast naming the tool whose result
    arrived late -- a toast is a different surface from the render."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        notifications = _capture_notifications(app)
        tool_b = _tool(name="fetch", server_key="local:docs", input_schema=None)
        await inspector.show_tool(tool_b)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_tool_result(
            server_key="local:docs",
            tool_name="search",
            ok=True,
            text="A's payload",
            duration_ms=10,
        )
        await pilot.pause()

        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result == ""  # render still dropped -- protected silence
        assert any("search" in msg for msg, _severity in notifications), (
            f"expected a toast naming the late-arriving tool, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_show_tool_result_panel_closed_still_toasts():
    """The second silent drop (`NoMatches` on the result Static -- the SAME
    tool is still selected, but its Test Tool panel was closed, e.g. via
    Close, while the run was in flight) also toasts."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(name="search", server_key="local:docs")
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-close")
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-test-result"))

        notifications = _capture_notifications(app)
        inspector.show_tool_result(
            server_key="local:docs",
            tool_name="search",
            ok=True,
            text="late payload",
            duration_ms=12,
        )
        await pilot.pause()

        assert not list(app.query("#mcp-inspector-test-result"))
        assert any("search" in msg for msg, _severity in notifications), (
            f"expected a toast naming the tool, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_handle_test_run_no_tool_selected_toasts():
    """`_handle_test_run()`'s `tool is None` early return -- defensive
    (the Run button only exists inside a panel `show_tool()` mounts for a
    real tool), but silent isn't the same thing as safe."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        notifications = _capture_notifications(app)
        assert inspector.current_tool is None
        inspector._handle_test_run()
        await pilot.pause()
        assert notifications, "expected a toast for the no-tool-selected guard"


@pytest.mark.asyncio
async def test_handle_test_run_panel_not_mounted_toasts():
    """`_handle_test_run()`'s `NoMatches` early return (the form/result/
    run-button widgets aren't mounted -- a tool is selected, but its Test
    Tool panel was never opened)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-test-run"))

        notifications = _capture_notifications(app)
        inspector._handle_test_run()
        await pilot.pause()
        assert notifications, "expected a toast for the missing-panel guard"


@pytest.mark.asyncio
async def test_close_button_removes_test_panel_and_reenables_test_tool_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-close")
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-test-run"))
        assert not list(app.query("#mcp-inspector-test-result"))
        assert app.query_one("#mcp-inspector-test-tool", Button).disabled is False


@pytest.mark.asyncio
async def test_test_panel_open_moves_focus_inside_and_escape_closes():
    """F-056: opening the Test Tool panel moves keyboard focus into it
    (the schema form's first control), and Escape closes the panel exactly
    like its Close button."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        assert list(app.query("#mcp-inspector-test-panel"))
        focused = app.focused
        assert focused is not None
        assert any(
            ancestor.id == "mcp-inspector-test-panel" for ancestor in focused.ancestors
        ), f"focus did not land inside the test panel: {focused!r}"
        await pilot.press("escape")
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-test-panel"))
        assert app.query_one("#mcp-inspector-test-tool", Button).disabled is False


@pytest.mark.asyncio
async def test_test_panel_mounts_for_an_all_boolean_schema_and_focuses_the_checkbox():
    """task-2740: `_mount_test_tool_panel()`'s F-056 focus query omitted
    `Checkbox`, and `DOMQuery.first()` RAISES `NoMatches` on an empty
    result (the `is None` fallback was dead code) -- so a tool whose
    schema renders ONLY boolean fields killed the mount worker, taking
    the app down (default `exit_on_error`). Found live during PR #1385's
    round-I test writing. The panel must mount, and focus must land on
    the first form control -- the Checkbox -- per F-056's own contract
    ("the schema form's first control when there is one")."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(
                input_schema={
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean", "default": False}},
                }
            )
        )
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.pause()

        assert list(app.query("#mcp-inspector-test-panel"))
        focused = app.focused
        assert focused is not None
        assert focused.id == "mcp-schema-field-0", (
            f"focus must land on the boolean form's Checkbox: {focused!r}"
        )


@pytest.mark.asyncio
async def test_test_panel_mounts_for_a_zero_control_schema_and_focuses_close():
    """task-2740, the shape that made this a LIVE crasher: a schema with
    empty `properties` renders a form with ZERO controls -- and the real
    built-in `list_characters` ships exactly that schema, so opening Test
    Tool on it crashed the app before this fix. With nothing to focus,
    the panel must mount and the F-056 fallback to the Close button must
    actually happen (it was unreachable dead code behind the raising
    `.first()`)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(
                input_schema={
                    "type": "object",
                    "properties": {},
                }
            )
        )
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.pause()

        assert list(app.query("#mcp-inspector-test-panel"))
        focused = app.focused
        assert focused is not None
        assert focused.id == "mcp-inspector-test-close", (
            f"with zero form controls, focus must land on Close: {focused!r}"
        )


def _test_preview(
    tool: HubTool, *, gate: str, nonce: str = "preview-1"
) -> ToolTestAdmissionPreview:
    return ToolTestAdmissionPreview(
        nonce=nonce,
        server_key=tool.server_key,
        tool_name=tool.name,
        definition_hash="definition",
        rendered_gate=gate,
        authority_fingerprint=None,
        safe_authority_label=None,
    )


@pytest.mark.asyncio
async def test_test_tool_preview_preparing_then_allow_is_one_click_run():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(button.label) == "Preparing…"
        assert button.disabled is True
        assert "Preparing" in str(
            app.query_one("#mcp-inspector-test-preview", Static).renderable
        )

        inspector.show_test_preview(_test_preview(tool, gate="allow"))
        await pilot.pause()
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.pause()
        button.focus()
        await pilot.press("enter")
        await pilot.pause()

        events = [
            event
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestRequested)
        ]
        assert len(events) == 1
        assert events[0].preview_nonce == "preview-1"
        assert events[0].intent == "run"


@pytest.mark.asyncio
async def test_test_tool_preview_ask_is_one_click_approve_once_and_keeps_edits():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(_tool(), gate="ask"))

        button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(button.label) == "Approve & run once"
        meaning = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)
        assert "one invocation" in meaning
        assert "does not persist" in meaning

        field = app.query_one("#mcp-schema-field-0", Input)
        field.value = "current argument"
        field.focus()
        await pilot.press("tab")
        button = app.query_one("#mcp-inspector-test-run", Button)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()

        events = [
            event
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestRequested)
        ]
        assert len(events) == 1
        assert events[0].intent == "approve_once"
        assert events[0].arguments == {"query": "current argument"}
        assert "Confirm run" not in str(button.label)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gate", "label", "reason"),
    [("off", "Blocked", "Permissions"), ("unresolved", "Unavailable", "Try again")],
)
async def test_test_tool_preview_non_actionable_is_disabled_with_recovery(
    gate: str, label: str, reason: str
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_preview(_test_preview(tool, gate=gate))
        button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(button.label) == label
        assert button.disabled is True
        assert reason in str(
            app.query_one("#mcp-inspector-test-preview", Static).renderable
        )
        assert not [
            event
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestRequested)
        ]


@pytest.mark.asyncio
async def test_test_tool_preview_close_revokes_nonce_and_late_preview_is_ignored():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="allow", nonce="old"))

        close = app.query_one("#mcp-inspector-test-close", Button)
        close.focus()
        await pilot.press("enter")
        await pilot.pause()

        revocations = [
            event
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestPreviewRevocationRequested)
        ]
        assert [event.preview_nonce for event in revocations] == ["old"]
        inspector.show_test_preview(_test_preview(tool, gate="ask", nonce="late"))
        assert not list(app.query("#mcp-inspector-test-panel"))


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["preview", "result"])
@pytest.mark.parametrize(
    ("absolute_path", "private_marker"),
    [
        ("/Users/alice/private/project/credentials.json", "credentials.json"),
        (
            '"/Users/alice/Private Project/credentials.json"',
            "Private Project",
        ),
        (
            "/Users/alice/Private Project/credentials.json: permission denied",
            "Private Project",
        ),
        (
            "/Users/alice/Private Project/credentials.json failed to open",
            "Project/credentials.json",
        ),
        (
            "/Users/alice/Private Failed Project/credentials.json",
            "Project/credentials.json",
        ),
        (
            "root:/Users/alice/private/credentials.json",
            "credentials.json",
        ),
        (
            "file:///Users/alice/private/credentials.json",
            "credentials.json",
        ),
        (
            r"C:\Private Folder\credentials.json: permission denied",
            "Private Folder",
        ),
        (r"\\server\share\private\credentials.json", "credentials.json"),
        (
            r"'\\server\Shared Folder\private credentials.json'",
            "Shared Folder",
        ),
        (r"\\?\C:\private\credentials.json", "credentials.json"),
        (r"'\\?\C:\Private Folder\credentials.json'", "Private Folder"),
        (r"\\.\pipe\private-token", "private-token"),
    ],
)
async def test_test_tool_failure_surfaces_redact_secrets_paths_and_bound_text(
    surface, absolute_path, private_marker
):
    """The inspector is the final fail-closed boundary for service text."""
    secret = "sk-live-super-secret-value"
    hostile = f"api_key={secret} failed at {absolute_path} " + ("x" * 4_000)
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        if surface == "preview":
            inspector.show_test_unavailable(hostile)
            rendered = str(
                app.query_one("#mcp-inspector-test-preview", Static).renderable
            )
        else:
            inspector.show_tool_result(
                server_key=tool.server_key,
                tool_name=tool.name,
                ok=False,
                text=hostile,
                duration_ms=0,
            )
            rendered = str(
                app.query_one("#mcp-inspector-test-result", Static).renderable
            )

        assert secret not in rendered
        assert absolute_path not in rendered
        assert private_marker not in rendered
        assert "[redacted]" in rendered
        assert "[path]" in rendered
        assert len(rendered) <= 560


@pytest.mark.parametrize(
    "value",
    [
        r"Expected regex \\d+ here.",
        r"Expected regex \\d+\\w+ here.",
        "See https://example.test/docs/private/file.txt for recovery.",
        r"Expected escaped token \\w+ and relative path docs/private.txt.",
        "Status at 12:34/56 remains ready: retry later.",
    ],
)
def test_test_tool_text_scrubber_preserves_urls_regex_and_relative_paths(value: str):
    assert mcp_inspector_module._safe_tool_test_text(value) == value


def test_test_tool_mapping_exception_preserves_innocuous_paths_and_escapes():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "pattern": r"\\d+\\w+",
                "docs": "https://example.test/docs/private/file.txt",
                "relative": "docs/private.txt",
                "timestamp": "12:34/56",
                "status": "ready: retry later",
            }
        )
    )

    assert "[path]" not in rendered
    assert "example.test/docs/private/file.txt" in rendered
    assert "docs/private.txt" in rendered
    assert "12:34/56" in rendered
    assert "ready: retry later" in rendered


def test_test_tool_mapping_exception_redacts_local_uri_and_spaced_suffix():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "path": "file:///Users/alice/Private Project/credentials.json",
                "detail": "Open failed; see https://example.test/recovery.",
            }
        )
    )

    assert "Users/alice" not in rendered
    assert "Project/credentials.json" not in rendered
    assert "[path]" in rendered
    assert "https://example.test/recovery" in rendered


def test_test_tool_path_scrubber_keeps_http_recovery_after_spaced_path():
    rendered = mcp_inspector_module._safe_tool_test_text(
        "Failed at /Users/alice/Private Project/credentials.json; "
        "see docs/recovery.md and https://example.test/recovery."
    )

    assert "Users/alice" not in rendered
    assert "Project/credentials.json" not in rendered
    assert "docs/recovery.md" in rendered
    assert "https://example.test/recovery" in rendered


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            r"failed at /Users/alice/Private Project/credentials.json and see "
            r"docs/recovery.md; with pattern \\d+ for help",
            r"failed at [path]; with pattern \\d+ for help",
        ),
        (
            r"failed at /Users/alice/Private Failed Project/credentials.json and "
            r"see docs/recovery.md; with pattern \\d+ for help",
            r"failed at [path]; with pattern \\d+ for help",
        ),
        (
            r"failed at /Users/alice/Private Project/credentials.json, consult "
            r"docs/recovery.md, pattern \\w+, or visit https://example.test/help.",
            r"failed at [path], consult docs/recovery.md, pattern \\w+, or visit "
            r"https://example.test/help.",
        ),
        (
            r"failed at root:/Users/alice/Private Project/credentials.json then "
            r"read docs/recovery.md; pattern \\s+ before retrying.",
            r"failed at root:[path]; pattern \\s+ before retrying.",
        ),
    ],
)
def test_test_tool_path_scrubber_stops_after_initial_local_path(
    message: str, expected: str
):
    """Ambiguous path-like suffixes redact until a structural boundary."""
    rendered = mcp_inspector_module._safe_tool_test_text(message)

    assert rendered == expected


@pytest.mark.parametrize(
    ("message", "private_fragments", "expected"),
    [
        (
            "failed at /Users/alice/Private Project",
            ("Users/alice", "Private Project"),
            "failed at [path]",
        ),
        (
            "failed at /Users/alice/Very Long Private Project Folder.",
            ("Users/alice", "Long Private Project Folder"),
            "failed at [path].",
        ),
        (
            "failed at file:///Users/alice/Very Long Private Project/credentials.json",
            ("file:", "Users/alice", "Long Private Project", "credentials.json"),
            "failed at [path]",
        ),
        (
            "failed at root:/Users/alice/Very Long Private Project",
            ("Users/alice", "Long Private Project"),
            "failed at root:[path]",
        ),
        (
            r"failed at C:\Very Long Private Project",
            ("Long Private Project",),
            "failed at [path]",
        ),
        (
            r"failed at \\server\share\Very Long Private Project",
            ("server", "Long Private Project"),
            "failed at [path]",
        ),
        (
            r"failed at \\?\C:\Very Long Private Project\credentials.json",
            ("Long Private Project", "credentials.json"),
            "failed at [path]",
        ),
        (
            r"failed at \\.\pipe\Very Long Private Project",
            ("Long Private Project",),
            "failed at [path]",
        ),
    ],
)
def test_test_tool_path_scrubber_redacts_terminal_multiword_components(
    message: str, private_fragments: tuple[str, ...], expected: str
):
    rendered = mcp_inspector_module._safe_tool_test_text(message)

    assert rendered == expected
    for fragment in private_fragments:
        assert fragment not in rendered


def test_test_tool_path_scrubber_redacts_ambiguous_text_after_terminal_directory():
    rendered = mcp_inspector_module._safe_tool_test_text(
        r"failed at /Users/alice/Very Long Private Project and see "
        r"docs/recovery.md with pattern \\d+ or visit https://example.test/help"
    )

    assert rendered == r"failed at [path] https://example.test/help"


@pytest.mark.parametrize(
    "clause",
    [
        "because access was denied",
        "PLEASE see docs/recovery.md",
        r"pattern \\d+ remains",
        "Expected owner root",
        "WHILE preparing the preview",
        "due to missing permission",
        "DUE-TO a stale preview",
    ],
)
def test_test_tool_path_scrubber_fails_closed_on_ambiguous_clause_after_directory(
    clause: str,
):
    rendered = mcp_inspector_module._safe_tool_test_text(
        f"failed at /Users/alice/Very Long Private Project {clause}"
    )

    assert rendered == "failed at [path]"


@pytest.mark.parametrize(
    "private_path",
    [
        "/Users/alice/Node.js Projects",
        r"C:\Node.js Projects",
        r"\\server\share\Report.txt Folder",
        r"\\?\C:\Cache.db Archives",
        r"\\.\pipe\Report.txt Folder",
        "file:///Users/alice/Node.js Projects",
        "/Users/alice/Node.js Projects/Secret Plan.txt",
        "/Users/alice/Node.js Long Term Projects/Secret Plan.txt",
        r"C:\Node.js Projects\Secret Plan.txt",
        r"\\server\share\Report.txt Folder\secret.key",
        r"\\?\C:\Cache.db Archives\Secret Plan.txt",
        r"\\.\pipe\Report.txt Folder\secret.key",
        "file:///Users/alice/Node.js Projects/Secret Plan.txt",
        "/Users/alice/Research and Development/Secret Plan.txt",
        r"C:\Research and Development\Secret Plan.txt",
        r"\\server\share\Research and Development\Secret Plan.txt",
        r"\\?\C:\Research because access\Secret Plan.txt",
        r"\\.\pipe\Please Review\Secret Plan.txt",
        "file:///Users/alice/Please Review/Secret Plan.txt",
        "/Users/alice/Because Project/Secret Plan.txt",
        "/Users/alice/Please Review/Secret Plan.txt",
        "/Users/alice/Pattern Library/Secret Plan.txt",
        "/Users/alice/Expected Results/Secret Plan.txt",
        "/Users/alice/While Away/Secret Plan.txt",
        "/Users/alice/Due To Migration/Secret Plan.txt",
        "/Users/alice/Due-To Migration/Secret Plan.txt",
        "/Users/alice/Very/because/access/secret.txt",
        r"C:\Very\please\see\secret.txt",
    ],
)
def test_test_tool_path_scrubber_keeps_clause_words_inside_path_components_private(
    private_path: str,
):
    rendered = mcp_inspector_module._safe_tool_test_text(f"failed at {private_path}")

    assert rendered == "failed at [path]"


@pytest.mark.parametrize(
    ("message", "preserved"),
    [
        (
            'failed at "/Users/alice/Very Long Private Project" please retry',
            '"[path]" please retry',
        ),
        (
            r"failed at /Users/alice/Very Long Private Project, pattern \\d+ remains",
            r"[path], pattern \\d+ remains",
        ),
        (
            "failed at /Users/alice/Very Long Private Project\n"
            "please see https://example.test/help",
            "[path]\nplease see https://example.test/help",
        ),
        (
            "failed at /Users/alice/Very Long Private Project: please retry",
            "[path]: please retry",
        ),
    ],
)
def test_test_tool_path_scrubber_preserves_only_structurally_delimited_diagnostics(
    message: str,
    preserved: str,
):
    rendered = mcp_inspector_module._safe_tool_test_text(message)

    assert "Users/alice" not in rendered
    assert preserved in rendered


@pytest.mark.parametrize(
    "private_path",
    [
        "/Users/alice/Very Long Secret Credentials.json",
        r"C:\Very Long Secret Credentials.json",
        r"\\server\share\Very Long Secret Credentials.json",
        r"\\?\C:\Very Long Secret Credentials.json",
        r"\\.\pipe\Very Long Secret Credentials.json",
        "file:///Users/alice/Very Long Secret Credentials.json",
    ],
)
def test_test_tool_path_scrubber_preserves_structural_boundary_after_filename(
    private_path: str,
):
    rendered = mcp_inspector_module._safe_tool_test_text(
        f"failed at {private_path}; retry at https://example.test/help"
    )

    assert rendered == "failed at [path]; retry at https://example.test/help"


def test_test_tool_path_scrubber_preserves_regex_after_structural_delimiter():
    rendered = mcp_inspector_module._safe_tool_test_text(
        r"failed at /Users/alice/Very Long Secret Credentials.json; "
        r"pattern \\d+\\w+ remains"
    )

    assert rendered == r"failed at [path]; pattern \\d+\\w+ remains"


def test_test_tool_mapping_exception_preserves_diagnostics_after_initial_path():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "detail": (
                    r"failed at /Users/alice/Private Project/credentials.json and "
                    r"see docs/recovery.md; with pattern \\d+ for help"
                ),
                "recovery": "https://example.test/help",
            }
        )
    )

    assert "Users/alice" not in rendered
    assert "Project/credentials.json" not in rendered
    assert "docs/recovery.md" not in rendered
    assert r"\\d+" in rendered
    assert "for help" in rendered
    assert "https://example.test/help" in rendered


def test_test_tool_mapping_exception_redacts_multiword_file_uri_and_label_path():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "uri": (
                    "file:///Users/alice/Very Long Private Project/credentials.json"
                ),
                "labelled": "root:/Users/alice/Private Project",
                "recovery": "docs/recovery.md",
            }
        )
    )

    assert "file:" not in rendered
    assert "Users/alice" not in rendered
    assert "Long Private Project" not in rendered
    assert "Private Project" not in rendered
    assert "credentials.json" not in rendered
    assert "root:[path]" in rendered
    assert "docs/recovery.md" in rendered


def test_test_tool_mapping_exception_fails_closed_on_ambiguous_path_words():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "posix": "failed at /Users/alice/Node.js Projects/Secret Plan.txt",
                "drive": r"failed at C:\Report.txt Folder\secret.key",
                "unc": r"failed at \\server\share\Report.txt Folder\secret.key",
                "extended": r"failed at \\?\C:\Cache.db Archives\Secret Plan.txt",
                "device": r"failed at \\.\pipe\Report.txt Folder\secret.key",
                "uri": (
                    "failed at file:///Users/alice/Node.js Projects/Secret Plan.txt"
                ),
            }
        )
    )

    assert "Users/alice" not in rendered
    assert "Node.js Projects" not in rendered
    assert "Report.txt Folder" not in rendered
    assert "Cache.db Archives" not in rendered
    assert "Secret Plan.txt" not in rendered
    assert "file:" not in rendered
    assert rendered.count("[path]") == 6


def test_test_tool_nested_mapping_exception_redacts_terminal_extension_directories():
    rendered = mcp_inspector_module._safe_exception_text(
        RuntimeError(
            {
                "paths": {
                    "posix": "failed at /Users/alice/Node.js Projects",
                    "drive": r"failed at C:\Node.js Projects",
                    "unc": r"failed at \\server\share\Report.txt Folder",
                    "uri": "failed at file:///Users/alice/Node.js Projects",
                },
                "special": [
                    r"failed at \\?\C:\Cache.db Archives",
                    r"failed at \\.\pipe\Report.txt Folder",
                ],
            }
        )
    )

    assert "Node.js Projects" not in rendered
    assert "Report.txt Folder" not in rendered
    assert "Cache.db Archives" not in rendered
    assert "file:" not in rendered
    assert rendered.count("[path]") == 6


@pytest.mark.asyncio
async def test_test_tool_unavailable_surface_preserves_nonfilesystem_diagnostics():
    message = (
        r"See https://example.test/docs/private/file.txt at 12:34/56; "
        r"pattern \\d+; relative docs/private.txt."
    )
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_unavailable(message)
        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)

        assert "https://example.test/docs/private/file.txt" in rendered
        assert "12:34/56" in rendered
        assert r"\\d+" in rendered
        assert "docs/private.txt" in rendered
        assert "[path]" not in rendered


@pytest.mark.asyncio
async def test_test_tool_unavailable_surface_preserves_diagnostics_after_initial_path():
    message = (
        r"failed at /Users/alice/Private Project/credentials.json and see "
        r"docs/recovery.md; with pattern \\d+; visit https://example.test/help."
    )
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_unavailable(message)
        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)

        assert "Users/alice" not in rendered
        assert "Project/credentials.json" not in rendered
        assert "docs/recovery.md" not in rendered
        assert r"\\d+" in rendered
        assert "https://example.test/help" in rendered


@pytest.mark.asyncio
async def test_test_tool_unavailable_surface_redacts_terminal_multiword_path():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_unavailable(
            "failed at /Users/alice/Very Long Private Project Folder."
        )
        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)

        assert "Users/alice" not in rendered
        assert "Long Private Project Folder" not in rendered
        assert "failed at [path]." in rendered


@pytest.mark.asyncio
async def test_test_tool_unavailable_surface_preserves_punctuated_recovery_clause():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_unavailable(
            "failed at /Users/alice/Very Long Private Project, "
            "please see docs/recovery.md"
        )
        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)

        assert "Users/alice" not in rendered
        assert "Long Private Project" not in rendered
        assert "please see docs/recovery.md" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("private_path", "private_fragments"),
    [
        ("/Users/alice/Node.js Projects", ("Users/alice", "Node.js Projects")),
        (r"C:\Node.js Projects", ("Node.js Projects",)),
        (r"\\server\share\Report.txt Folder", ("Report.txt Folder",)),
        (r"\\?\C:\Cache.db Archives", ("Cache.db Archives",)),
        (r"\\.\pipe\Report.txt Folder", ("Report.txt Folder",)),
        (
            "file:///Users/alice/Node.js Projects",
            ("file:", "Users/alice", "Node.js Projects"),
        ),
    ],
)
async def test_test_tool_unavailable_surface_redacts_terminal_extension_directory(
    private_path: str,
    private_fragments: tuple[str, ...],
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_test_unavailable(f"failed at {private_path}")
        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)

        for fragment in private_fragments:
            assert fragment not in rendered
        assert "[path]" in rendered


@pytest.mark.asyncio
async def test_test_tool_unavailable_retry_is_keyboard_accessible_and_preserves_form():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        for _ in range(40):
            fields = list(app.query("#mcp-schema-field-0"))
            if fields:
                break
            await pilot.pause()
        assert fields

        inspector.show_test_preview(_test_preview(tool, gate="allow", nonce="old"))
        field = fields[0]
        field.value = "keep this exact value"
        inspector.show_test_unavailable("The preview service timed out.")

        retry = app.query_one("#mcp-inspector-test-retry", Button)
        assert retry.display is True
        assert retry.disabled is False
        retry.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert field.value == "keep this exact value"
        assert app.query_one("#mcp-inspector-test-panel").is_attached
        assert [
            event.preview_nonce
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestPreviewRevocationRequested)
        ] == ["old"]
        requests = [
            event
            for event in app.events
            if isinstance(event, MCPInspector.ToolTestPreviewRequested)
        ]
        assert len(requests) == 2
        assert (requests[-1].server_key, requests[-1].tool_name) == (
            tool.server_key,
            tool.name,
        )
        assert str(app.query_one("#mcp-inspector-test-run", Button).label) == (
            "Preparing…"
        )


# -- Task 7: permission explanation + re-allow -------------------------------


@pytest.mark.asyncio
async def test_show_tool_with_effective_appends_permission_block():
    """`show_tool(tool, effective=...)` (Tools-mode's own call site) appends
    the permission explanation below the existing tool detail -- exact-copy
    origin sentence for a plain server-default inheritance, no notice, no
    Re-allow button."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(), effective=EffectiveToolState(state="ask", origin="server_default")
        )
        await pilot.pause()
        container = app.query_one("#mcp-inspector-permission")
        assert container.display is True
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "Inherited from the server default."
        assert not list(app.query("#mcp-inspector-reallow"))
        assert not list(app.query("#mcp-inspector-permission-notice"))


@pytest.mark.asyncio
async def test_show_tool_without_effective_hides_permission_block():
    """The plain T6 call shape (no `effective` keyword) must not show a
    stale/empty permission block -- backward compatible with every
    pre-Task-7 `show_tool()` call site."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        container = app.query_one("#mcp-inspector-permission")
        assert container.display is False


@pytest.mark.asyncio
async def test_show_tool_none_hides_permission_block_too():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(), effective=EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        await inspector.show_tool(None)
        await pilot.pause()
        container = app.query_one("#mcp-inspector-permission")
        assert container.display is False
        assert not list(app.query("#mcp-inspector-permission-origin"))


@pytest.mark.asyncio
async def test_show_permission_origin_sentence_tool_override():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "From this tool's override."


@pytest.mark.asyncio
async def test_show_permission_origin_sentence_server_default():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="server_default")
        )
        await pilot.pause()
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "Inherited from the server default."


@pytest.mark.asyncio
async def test_show_permission_origin_sentence_global_default():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
        await pilot.pause()
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "Inherited from the global default."


@pytest.mark.asyncio
async def test_show_permission_origin_sentence_falls_back_for_unrecognized_origin():
    """An unknown service origin renders an honest fallback sentence."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="deny", origin="gate_error")
        )
        await pilot.pause()
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "Permission state could not be resolved."


@pytest.mark.asyncio
async def test_show_permission_state_line_reads_unknown_not_off_for_gate_error():
    """Fix Round J (review of Fix Round H): the test above always pinned
    the gate_error ORIGIN sentence while the state line ONE WIDGET ABOVE
    it read "Permission: Off" -- `ui_label` maps the synthesized
    fail-closed `state="deny"` to "Off" with no origin awareness, so the
    block stacked a confident configuration claim directly on top of an
    admission that the configuration could not be read. Exactly the
    contradiction shape earlier rounds removed from the Test Tool body
    and the Advanced hatch, reassembled by composition of two
    individually-truthful widgets. The state line must say what is known
    ("Unknown"), and a GENUINE deny must keep saying "Off" -- both
    directions pinned so the fix cannot silently overreach."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="deny", origin="gate_error")
        )
        await pilot.pause()
        state_line = str(
            app.query_one("#mcp-inspector-permission-state", Static).renderable
        )
        assert state_line == "Permission: Unknown"
        assert "Off" not in state_line

        await inspector.show_permission(
            _tool(), EffectiveToolState(state="deny", origin="tool_override")
        )
        await pilot.pause()
        state_line = str(
            app.query_one("#mcp-inspector-permission-state", Static).renderable
        )
        assert state_line == "Permission: Off"


# -- Task 3 (MCP Hub Phase 6): cascade provenance ----------------------------


@pytest.mark.asyncio
async def test_show_permission_cascade_none_falls_back_to_origin_sentence():
    """The default (`cascade=None`, every pre-Task-3 call shape) must
    render exactly the old single-sentence origin block -- no cascade rungs
    at all."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "From this tool's override."
        assert not list(app.query("#mcp-inspector-permission-cascade-tool"))
        assert not list(app.query("#mcp-inspector-permission-cascade-server"))
        assert not list(app.query("#mcp-inspector-permission-cascade-global"))


@pytest.mark.asyncio
async def test_show_permission_cascade_tool_override_wins():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="allow", origin="tool_override"),
            cascade=("allow", "ask", "ask"),
        )
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-permission-origin"))

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        global_rung = app.query_one("#mcp-inspector-permission-cascade-global", Static)

        assert str(tool_rung.renderable) == "▸ Tool override: Allow •"
        assert str(server_rung.renderable) == "Server default: Ask •"
        assert str(global_rung.renderable) == "Global default: Ask"

        assert "mcp-status-ready" in tool_rung.classes
        assert "mcp-status-muted" in server_rung.classes
        assert "mcp-status-muted" in global_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_server_default_wins_when_tool_unset():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="ask", origin="server_default"),
            cascade=(None, "ask", "allow"),
        )
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        global_rung = app.query_one("#mcp-inspector-permission-cascade-global", Static)

        assert str(tool_rung.renderable) == "Tool override: —"
        assert str(server_rung.renderable) == "▸ Server default: Ask •"
        assert str(global_rung.renderable) == "Global default: Allow"

        assert "mcp-status-muted" in tool_rung.classes
        assert "mcp-status-warning" in server_rung.classes
        assert "mcp-status-muted" in global_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_global_default_wins_when_nothing_overridden():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="ask", origin="global_default"),
            cascade=(None, None, "ask"),
        )
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        global_rung = app.query_one("#mcp-inspector-permission-cascade-global", Static)

        assert str(tool_rung.renderable) == "Tool override: —"
        assert str(server_rung.renderable) == "Server default: —"
        assert str(global_rung.renderable) == "▸ Global default: Ask"

        assert "mcp-status-muted" in tool_rung.classes
        assert "mcp-status-muted" in server_rung.classes
        assert "mcp-status-warning" in global_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_deny_winner_uses_error_class():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="deny", origin="tool_override"),
            cascade=("deny", None, "ask"),
        )
        await pilot.pause()
        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        assert str(tool_rung.renderable) == "▸ Tool override: Off •"
        assert "mcp-status-error" in tool_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_config_changed_winner_renders_warning_not_ready():
    """Critical review fix: a rug-pulled tool (an explicit `allow` whose
    stored `definition_hash` no longer matches the live tool) must not
    render its winning cascade rung READY-green. `resolve_effective_
    state()` already downgrades `effective.state` to `"ask"` for exactly
    this case (`config_changed=True`) -- `_cascade_rungs()` used to ignore
    that real, already-resolved `effective` entirely and build its own
    SYNTHETIC `EffectiveToolState` straight off the raw cascade tuple's
    still-`"allow"` stored value, so the tool rung rendered
    "▸ Tool override: Allow •" GREEN directly under a "Permission: Ask"
    state line and the definition-changed notice."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(
                state="ask", origin="tool_override", config_changed=True
            ),
            cascade=("allow", None, "ask"),
        )
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        assert str(tool_rung.renderable) == "▸ Tool override: Allow ⚠"
        assert "mcp-status-warning" in tool_rung.classes
        assert "mcp-status-ready" not in tool_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_risk_floored_winner_renders_warning_not_ready():
    """Same fix, the OTHER downgrade path: an *inherited* `allow` (here,
    the server default) floored to `"ask"` for a high-risk tool must also
    render its winning rung warning-colored with the ⚑ marker, not the raw
    stored "Allow" rendered ready-green."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="ask", origin="server_default", risk_floored=True),
            cascade=(None, "allow", "ask"),
        )
        await pilot.pause()

        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        assert str(server_rung.renderable) == "▸ Server default: Allow ⚑"
        assert "mcp-status-warning" in server_rung.classes
        assert "mcp-status-ready" not in server_rung.classes


@pytest.mark.asyncio
async def test_show_permission_cascade_muted_rung_dimmed_under_real_bundled_css():
    """Minor 3 (review): `.mcp-status-muted` is scoped to this widget's own
    `DEFAULT_CSS` (the raw `$text-muted` token -- see that rule's own
    comment), while the winner's `.mcp-status-{ready|warning|error}` class
    resolves from the shared bundle's `$ds-status-*` design-system aliases
    (`css/tldw_cli_modular.tcss`). Every other cascade test in this module
    mounts a bundle-less `InspectorApp`, where `.mcp-status-ready` has no
    concrete color to resolve at all -- proving the dimming actually WORKS
    (not just that the class name differs) needs the real bundle
    (`InspectorAppWithBundledCSS`, same harness `test_disabled_action_
    buttons_stay_legible_with_bundled_css` already uses for exactly this
    reason)."""
    app = InspectorAppWithBundledCSS()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="allow", origin="tool_override"),
            cascade=("allow", "ask", "ask"),
        )
        await pilot.pause()
        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        assert "mcp-status-ready" in tool_rung.classes
        assert "mcp-status-muted" in server_rung.classes
        assert tool_rung.styles.color != server_rung.styles.color
        # Task 6 dual-layer CSS audit: the rungs are plain `.ds-field-row`
        # Statics (an established, already-bundle-covered class) with no
        # new geometry properties of their own -- confirmed here rather
        # than assumed, alongside the color check above. No bundle-layer
        # rule was added for them.
        for rung in (tool_rung, server_rung):
            assert rung.size.width > 0 and rung.size.height > 0, (
                f"{rung.id} collapsed to zero geometry under bundled CSS"
            )


@pytest.mark.asyncio
async def test_show_tool_effective_block_never_renders_cascade_rungs():
    """Task 3's cascade wiring is `show_permission()`-only (per the brief) --
    Tools-mode's own combined call (`show_tool(tool, effective=...)`) keeps
    rendering the plain origin sentence, never the cascade rungs, since it
    has no cascade tuple to pass."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(), effective=EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        assert app.query_one("#mcp-inspector-permission-origin", Static)
        assert not list(app.query("#mcp-inspector-permission-cascade-tool"))


# -- Task 3 (MCP Hub Phase 6): Change in Permissions cross-mode jump ---------


@pytest.mark.asyncio
async def test_tools_mode_permission_block_renders_change_in_permissions_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(server_key="local:docs", name="search"),
            effective=EffectiveToolState(state="ask", origin="global_default"),
        )
        await pilot.pause()
        button = app.query_one("#mcp-inspector-goto-permission", Button)
        assert button.tooltip

        await pilot.click("#mcp-inspector-goto-permission")
        await pilot.pause()
        events = [
            e
            for e in app.events
            if isinstance(e, MCPInspector.ChangeInPermissionsRequested)
        ]
        assert len(events) == 1
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name == "search"


@pytest.mark.asyncio
async def test_standalone_show_permission_never_renders_change_in_permissions_button():
    """The standalone Permissions-mode entry point (`show_permission()`) is
    already showing this tool's Permissions-mode row -- jumping there again
    would be a no-op affordance, so no button is rendered."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-goto-permission"))


@pytest.mark.asyncio
async def test_show_tool_result_blocked_shows_test_panel_change_in_permissions_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()

        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="Blocked — this tool is set to Off in Permissions.",
            duration_ms=0,
            blocked=True,
        )
        await pilot.pause()
        goto_button = app.query_one("#mcp-inspector-goto-permission-test", Button)
        assert goto_button.display is True


@pytest.mark.asyncio
async def test_show_tool_result_non_blocked_hides_test_panel_change_in_permissions_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_test_preview(_test_preview(tool, gate="ask"))
        await pilot.pause()
        assert (
            app.query_one("#mcp-inspector-goto-permission-test", Button).display is True
        )

        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            text="{}",
            duration_ms=10,
        )
        await pilot.pause()
        assert (
            app.query_one("#mcp-inspector-goto-permission-test", Button).display
            is False
        )


@pytest.mark.asyncio
async def test_show_permission_config_changed_shows_notice_and_reallow_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(
                state="ask", origin="tool_override", config_changed=True
            ),
        )
        await pilot.pause()
        notice = str(
            app.query_one("#mcp-inspector-permission-notice", Static).renderable
        )
        assert notice == "Definition changed since you allowed it."
        reallow = app.query_one("#mcp-inspector-reallow", Button)
        assert reallow.tooltip == "Store the new definition hash and allow again."


@pytest.mark.asyncio
async def test_show_permission_risk_floored_shows_notice_without_reallow_button():
    """Re-allow only ever appears for a `config_changed` downgrade -- a
    risk-floored inherited "allow" has nothing to re-allow (there's no
    tool-level override to refresh the hash on)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(),
            EffectiveToolState(state="ask", origin="server_default", risk_floored=True),
        )
        await pilot.pause()
        notice = str(
            app.query_one("#mcp-inspector-permission-notice", Static).renderable
        )
        assert (
            notice
            == "High-risk tool — asks even though the inherited default is Allow."
        )
        assert not list(app.query("#mcp-inspector-reallow"))


@pytest.mark.asyncio
async def test_show_permission_plain_state_shows_neither_notice_nor_button():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-permission-notice"))
        assert not list(app.query("#mcp-inspector-reallow"))


@pytest.mark.asyncio
async def test_permission_state_line_carries_semantic_status_class():
    """Task 1 (MCP Hub Phase 6): `#mcp-inspector-permission-state` is a
    non-cell Static -- unlike a DataTable cell, it CAN carry a CSS class, so
    it uses the existing `.mcp-status-{ready|warning|error}` classes
    (`css/tldw_cli_modular.tcss`), the same ones `mcp_rail.py`'s rows and
    `#mcp-inspector-state`'s own readiness badge already use, rather than
    `mcp_permissions_mode.state_text()`'s Rich-style mechanism (which exists
    only because a DataTable cell can't take a class at all)."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)

        await inspector.show_permission(
            _tool(), EffectiveToolState(state="allow", origin="tool_override")
        )
        await pilot.pause()
        state = app.query_one("#mcp-inspector-permission-state", Static)
        assert "mcp-status-ready" in state.classes
        assert "mcp-status-warning" not in state.classes
        assert "mcp-status-error" not in state.classes

        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
        await pilot.pause()
        state = app.query_one("#mcp-inspector-permission-state", Static)
        assert "mcp-status-warning" in state.classes

        await inspector.show_permission(
            _tool(), EffectiveToolState(state="deny", origin="tool_override")
        )
        await pilot.pause()
        state = app.query_one("#mcp-inspector-permission-state", Static)
        assert "mcp-status-error" in state.classes


@pytest.mark.asyncio
async def test_reallow_button_press_posts_reallow_requested_with_server_key_and_tool_name():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(server_key="local:docs", name="search")
        await inspector.show_permission(
            tool,
            EffectiveToolState(
                state="ask", origin="tool_override", config_changed=True
            ),
        )
        await pilot.pause()
        await pilot.click("#mcp-inspector-reallow")
        await pilot.pause()
        events = [e for e in app.events if isinstance(e, MCPInspector.ReallowRequested)]
        assert len(events) == 1
        assert events[0].server_key == "local:docs"
        assert events[0].tool_name == "search"


@pytest.mark.asyncio
async def test_second_show_permission_back_to_back_does_not_duplicate_ids():
    """Mandatory regression: selecting two matrix tool rows in a row must
    not raise DuplicateIds -- mirrors
    `test_second_show_tool_back_to_back_does_not_duplicate_ids`."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(name="search"),
            EffectiveToolState(state="allow", origin="tool_override"),
        )
        # No pilot.pause() here on purpose.
        await inspector.show_permission(
            _tool(name="fetch"),
            EffectiveToolState(state="ask", origin="global_default"),
        )
        await pilot.pause()
        origins = list(app.query("#mcp-inspector-permission-origin"))
        assert len(origins) == 1
        assert str(origins[0].renderable) == "Inherited from the global default."


# -- Phase 4 UX batch item 7: neutral "nothing selected" header --------------


@pytest.mark.asyncio
async def test_no_selection_header_teaches_what_inspection_offers():
    """F-054: the empty state is contextual -- it says what picking a row
    gets you (what's wrong, what you can do) rather than the bare 'Select
    an item to inspect.', and stays that way after a selection is cleared
    back to None."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        badge = str(app.query_one("#mcp-inspector-state", Static).renderable)
        assert badge == (
            "Pick a server, tool, or entry to see what's wrong and what you can do."
        )

        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        await inspector.update_readiness(None)
        await pilot.pause()
        badge = str(app.query_one("#mcp-inspector-state", Static).renderable)
        assert badge == (
            "Pick a server, tool, or entry to see what's wrong and what you can do."
        )


@pytest.mark.asyncio
async def test_empty_state_copy_wraps_instead_of_clipping_at_narrow_width():
    """F-054: at the inspector's narrowest real widths (its min-width is 28)
    the empty-state line must WRAP to multiple rows -- `.ds-status-badge`'s
    shared `height: 1` used to clip it mid-word. The local
    `#mcp-inspector-state` override (height: auto) wins on ID specificity;
    other `.ds-status-badge` consumers are untouched. (The bare harness's
    `width: 3fr` triples a single-child screen's width instead of filling
    it, so the narrow width is pinned with explicit test-app CSS.)"""

    class NarrowInspectorApp(InspectorApp):
        CSS = "MCPInspector { width: 30; }"

    app = NarrowInspectorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        state = app.query_one("#mcp-inspector-state", Static)
        assert state.region.height > 1


# -- Phase 4 UX batch item 8: tool identity above the permission block -------


@pytest.mark.asyncio
async def test_show_permission_standalone_renders_tool_identity_first():
    """Item 8: the standalone permission block (Permissions-mode matrix row
    selection) never mounts `#mcp-inspector-tool` at all -- without its own
    identity line, the explanation below it had no indication of WHICH tool
    it describes."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(name="search", server_label="docs"),
            EffectiveToolState(state="allow", origin="tool_override"),
        )
        await pilot.pause()
        identity = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert identity == "search — docs"


@pytest.mark.asyncio
async def test_show_tool_with_effective_also_renders_tool_identity():
    """Item 8 applies uniformly to `_render_permission_container()` --
    Tools-mode's own combined call (`show_tool(tool, effective=...)`) gets
    the same identity line even though `#mcp-inspector-tool` already shows
    one above it."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(
            _tool(name="fetch", server_label="docs"),
            effective=EffectiveToolState(state="ask", origin="server_default"),
        )
        await pilot.pause()
        identity = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert identity == "fetch — docs"


# -- Phase 4 UX batch items 5 & 13: blocked status + duration formatting -----


@pytest.mark.asyncio
async def test_show_tool_result_sub_second_uses_ms_granularity():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            text="{}",
            duration_ms=999,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("OK · 999ms")


@pytest.mark.asyncio
async def test_show_tool_result_seconds_tier_uses_one_decimal():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="boom",
            duration_ms=45_300,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Failed · 45.3s")


@pytest.mark.asyncio
async def test_show_tool_result_minute_tier_uses_minutes_and_seconds():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=True,
            text="{}",
            duration_ms=125_000,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("OK · 2m 5s")


@pytest.mark.asyncio
async def test_show_tool_result_blocked_renders_not_run_status_line():
    """Item 5: the deny-gate's synthetic result must read "Blocked · not
    run", not "Failed · 0ms" -- the call never reached the tool at all, so
    the ok/duration_ms failure template would misleadingly imply an
    attempted, timed run."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool()
        await inspector.show_tool(tool)
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        inspector.show_tool_result(
            server_key=tool.server_key,
            tool_name=tool.name,
            ok=False,
            text="Blocked — this tool is set to Off in Permissions.",
            duration_ms=0,
            blocked=True,
        )
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Blocked · not run")
        assert "Failed" not in result.split("\n", 1)[0]


# -- Task 2 (MCP Hub Phase 6): finding-detail remediation buttons -----------


def _finding(
    *,
    finding_type: str = "orphaned_path_scope",
    message: str = "Needs review",
    severity: str = "high",
) -> dict[str, Any]:
    return {"severity": severity, "finding_type": finding_type, "message": message}


@pytest.mark.asyncio
async def test_finding_detail_renders_mapped_action_buttons_with_tooltips():
    """A finding whose text matches the discovery/stale/catalog bucket
    renders exactly the two mapped buttons (REFRESH_DISCOVERY, VIEW_DETAILS),
    ids `#mcp-finding-action-<action>`, each with a tooltip. `server_key`
    given -- the "no server context" note (New Minor 3) is a different,
    separately-tested case."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(
            _finding(finding_type="catalog_expired", message="Tool catalog is stale."),
            server_key="server:main",
        )
        await pilot.pause()
        container = app.query_one("#mcp-inspector-finding")
        assert container.display is True
        buttons = {b.id: b for b in container.query(Button)}
        assert set(buttons) == {
            "mcp-finding-action-refresh_discovery",
            "mcp-finding-action-view_details",
        }
        for button in buttons.values():
            assert button.tooltip, f"{button.id} has no tooltip"


@pytest.mark.asyncio
async def test_finding_detail_action_buttons_have_nonzero_geometry_with_bundled_css():
    """Task 6 dual-layer CSS audit: the finding-detail remediation buttons
    (Task 2) carry `classes="console-action-secondary"` -- a class selector
    that already outranks any bare `Button { ... }` type-selector rule in
    the bundle on specificity alone, and `.console-action-secondary`
    itself already ships an explicit `height: 1; min-height: 1;` in
    `_agentic_terminal.tcss` (T5, MCP Hub Phase 4's audit-drill buttons).
    Verified here empirically, under the real bundle, rather than assumed
    from the class reuse -- same Phase 3 lesson as every other bundled-CSS
    check in this suite. No bundle-layer rule was added for these buttons
    -- this test is the verification, not a fix."""
    app = InspectorAppWithBundledCSS()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(
            _finding(finding_type="catalog_expired", message="Tool catalog is stale."),
            server_key="server:main",
        )
        await pilot.pause()
        buttons = list(app.query("#mcp-inspector-finding Button"))
        assert len(buttons) == 2
        for button in buttons:
            assert button.size.width > 0, (
                f"{button.id} collapsed to zero width under bundled CSS"
            )
            assert button.size.height > 0, (
                f"{button.id} collapsed to zero height under bundled CSS"
            )


@pytest.mark.asyncio
async def test_finding_detail_default_mapping_renders_single_view_details_button():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(_finding(), server_key="server:main")
        await pilot.pause()
        container = app.query_one("#mcp-inspector-finding")
        buttons = list(container.query(Button))
        assert [b.id for b in buttons] == ["mcp-finding-action-view_details"]


@pytest.mark.asyncio
async def test_finding_action_button_posts_hub_action_requested_with_given_server_key():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(_finding(), server_key="server:main")
        await pilot.pause()
        await pilot.click("#mcp-finding-action-view_details")
        await pilot.pause()
        assert app.events
        assert app.events[-1].action is HubAction.VIEW_DETAILS
        assert app.events[-1].server_key == "server:main"


@pytest.mark.asyncio
async def test_finding_with_no_server_key_shows_note_instead_of_dead_buttons():
    """New Minor 3 (MCP Hub Phase 6 finale, review): `server_key` defaults
    to `None` -- the caller (`MCPWorkbench._finding_owning_server_key()`)
    could not resolve one (neither the finding nor the rail selection
    carried an owning server). Every remediation button's `HubActionRequested`
    would then post with no server to act on -- `on_mcp_inspector_hub_
    action_requested()` drops all of them (each branch guards on a truthy
    `event.server_key`) -- so rendering them would just be dead chrome.
    `show_finding()` renders an explanatory note instead and mounts no
    buttons at all (was previously the ONLY caller-observable difference:
    the button existed and posted `server_key=None`, silently swallowed
    downstream)."""
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(_finding())
        await pilot.pause()
        container = app.query_one("#mcp-inspector-finding")
        assert container.display is True
        assert not list(container.query(Button))
        note = app.query_one("#mcp-inspector-finding-no-context", Static)
        assert str(note.renderable) == "No server context — select a server first."


@pytest.mark.asyncio
async def test_show_finding_none_clears_action_buttons_and_hides_container():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(_finding(), server_key="server:main")
        await pilot.pause()
        assert list(app.query_one("#mcp-inspector-finding").query(Button))

        await inspector.show_finding(None)
        await pilot.pause()
        container = app.query_one("#mcp-inspector-finding")
        assert container.display is False
        assert not list(container.query(Button))
        # A stray press after clearing must not resurrect a stale server_key.
        assert inspector._current_finding_server_key is None


# -- F2 (PR #722 bot review): `_render_section_payload()` broad fallback ----


def test_render_section_payload_falls_back_to_str_on_non_typeerror_json_failure():
    """F2 (Gemini bot review): `_render_section_payload()` used to catch
    only `TypeError` -- a payload that fails `json.dumps()` with a
    DIFFERENT exception (e.g. `ValueError` from a circular reference, which
    `json.dumps()` raises rather than recursing forever) would raise out of
    the Advanced pane instead of falling back to `str(payload)`. Now catches
    `Exception` broadly.
    """
    circular: dict[str, Any] = {}
    circular["self"] = circular
    result = mcp_inspector_module._render_section_payload("advanced", circular)
    assert isinstance(result, str)
    assert result  # fell back to str(payload) rather than raising


# -- Task 6 (RAG-50): retire the stale empty-state badge over populated ----
# tool detail. `#mcp-inspector-state` is seeded with `_EMPTY_STATE_COPY` at
# compose() time and written only by `update_readiness()`, whose only
# caller (`MCPWorkbench._sync_children()`) is fed by the selected SERVER.
# `show_tool()` never touched it, so Tools mode with no server selected
# left the empty-state badge sitting above fully populated tool detail.


@pytest.mark.asyncio
async def test_empty_state_badge_hidden_when_tool_shown():
    """No prior `update_readiness()` call at all (mirrors Tools mode with
    no server selected) -- `show_tool()` alone must hide the badge."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False


@pytest.mark.asyncio
async def test_empty_state_badge_returns_when_detail_cleared():
    """The clear path is `show_tool(None)` itself (see
    `MCPWorkbench._clear_tool_view()`, which calls exactly that) -- no
    separate blank/clear method exists. Restoring `display = True` must not
    disturb the content `update_readiness()` maintains -- untouched here,
    so it is still the compose()-time `_EMPTY_STATE_COPY`."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        await inspector.show_tool(None)
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is True
        assert str(badge.renderable) == mcp_inspector_module._EMPTY_STATE_COPY


@pytest.mark.asyncio
async def test_update_readiness_does_not_resurrect_badge_over_displayed_tool():
    """A server-selection sync (`update_readiness()`) firing while Tools
    mode has a tool displayed (e.g. a background readiness refresh) must
    not force the badge back visible over the populated tool detail."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_tool(_tool())
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False
        await inspector.update_readiness(_ready_snap())
        await pilot.pause()
        assert badge.display is False
        # Clearing afterwards still restores it correctly.
        await inspector.show_tool(None)
        await pilot.pause()
        assert badge.display is True


# -- task-2270: the RAG-50 badge fix covered Tools mode ONLY. The other -----
# three detail views (Permissions matrix row, Audit entry, Finding) each
# toggled only their own container and never touched `#mcp-inspector-state`,
# so selecting any of them with no server selected left "Pick a server,
# tool, or entry…" sitting above fully populated detail. Confirmed by
# direct code read in the PR-5 task-6 review; fixed here by a single badge
# owner (`_sync_state_badge_display()`): the badge shows exactly when NO
# detail view is displayed, every view's show/clear path funnels through
# it, and `update_readiness()` stays content-only in every mode.


def _audit_entry() -> dict[str, Any]:
    return {"server_key": "local:docs", "tool_name": "search"}


@pytest.mark.asyncio
async def test_empty_state_badge_hidden_when_permission_row_shown():
    """Permissions-matrix row selection (`show_permission()`) must hide
    the badge like Tools mode does, and the shared clear path
    (`show_tool(None)`, which also blanks the permission container via
    `_render_permission_container(None, None)`) must restore it."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False

        await inspector.show_tool(None)
        await pilot.pause()
        assert badge.display is True


@pytest.mark.asyncio
async def test_empty_state_badge_hidden_when_audit_entry_shown():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_audit_entry(_audit_entry())
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False

        await inspector.show_audit_entry(None)
        await pilot.pause()
        assert badge.display is True


@pytest.mark.asyncio
async def test_empty_state_badge_hidden_when_finding_shown():
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_finding(_finding(), server_key="local:docs")
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False

        await inspector.show_finding(None)
        await pilot.pause()
        assert badge.display is True


@pytest.mark.asyncio
async def test_empty_state_badge_stays_hidden_while_any_detail_remains():
    """Audit mode can display an execution entry and a finding at once --
    clearing ONE must not resurrect the badge while the other still shows
    populated detail; only clearing the last one restores it."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await inspector.show_audit_entry(_audit_entry())
        await inspector.show_finding(_finding(), server_key="local:docs")
        await pilot.pause()
        badge = app.query_one("#mcp-inspector-state", Static)
        assert badge.display is False

        await inspector.show_finding(None)
        await pilot.pause()
        assert badge.display is False, (
            "finding cleared but the audit entry still shows detail -- "
            "the badge must stay hidden"
        )

        await inspector.show_audit_entry(None)
        await pilot.pause()
        assert badge.display is True


@pytest.mark.asyncio
async def test_update_readiness_does_not_resurrect_badge_over_other_detail_views():
    """AC3, beyond Tools mode: a background readiness sync while a
    permission row / audit entry shows detail must not force the badge
    back over it -- `update_readiness()` stays content-only."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        badge = app.query_one("#mcp-inspector-state", Static)

        await inspector.show_permission(
            _tool(), EffectiveToolState(state="ask", origin="global_default")
        )
        await pilot.pause()
        await inspector.update_readiness(_ready_snap())
        await pilot.pause()
        assert badge.display is False

        await inspector.show_tool(None)
        await inspector.show_audit_entry(_audit_entry())
        await pilot.pause()
        await inspector.update_readiness(_stale_snap())
        await pilot.pause()
        assert badge.display is False

        await inspector.show_audit_entry(None)
        await pilot.pause()
        assert badge.display is True


# -- Task 6 (PR-T3), Route B: the Advanced runner's `tool.execute` hatch -----
# Every OTHER Advanced action reads or mutates control-plane config; this one
# EXECUTES a tool. It ran on a single press, ungated and unlogged. The service
# now enforces the permission gate (`execute_advanced_tool()`); this pane
# supplies the per-run consent that gate's "ask" verdict requires, and renders
# a refusal as a refusal instead of an "Action failed:" dump.


class ToolExecuteAdvService(FakeAdvService):
    """Advanced service offering the real inventory-section `tool.execute`
    descriptor (unified_control_plane_service.py's own payload template)."""

    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error
        # Fix Round E, Item 4: an explicit, observable call count for
        # `load_section()` -- the surface `_load_advanced_section()`'s
        # scheduled worker calls once it actually runs. Used by the
        # isolation tests below as direct proof the worker has NOT run yet
        # at the point they check `_advanced_confirm_key`, rather than
        # inferring that purely from `set_service_context()`'s own
        # synchronous clear also happening to leave the key `None`.
        self.load_section_calls = 0

    async def load_section(self, section=None):
        self.load_section_calls += 1
        return await super().load_section(section)

    def available_actions(self):
        return [
            {
                "name": "tool.execute",
                "label": "Execute Local Tool",
                "action_id": "mcp.runtime.trigger.local",
                "payload_template": '{"tool_name":"search_notes","arguments":{"query":"example"}}',
            }
        ]

    async def run_action(self, action_name, payload):
        self.action_calls.append((action_name, dict(payload or {})))
        if self.error is not None:
            raise self.error
        return {"ok": True}


class ToolExecuteInspectorApp(ConsolidatedCSSApp):
    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__()
        self.service = ToolExecuteAdvService(error=error)

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        self.query_one(MCPInspector).set_service_context(
            self.service, [("Inventory", "inventory")]
        )


class ToolExecuteAndReadAdvService(ToolExecuteAdvService):
    """Fix Round E, Item 2: adds a second, SAME-SECTION action
    (`resource.read`) alongside `tool.execute` -- mirrors the real
    inventory section's actual membership (`tool.execute` shares
    `inventory` only with `resource.read` and `prompt.get`, both reads)
    so a test can switch the action Select WITHOUT also switching
    section, isolating Item 2's action-switch fix from Item 1's
    section-change fix."""

    def available_actions(self):
        return super().available_actions() + [
            {
                "name": "resource.read",
                "label": "Read Resource",
                "action_id": "mcp.runtime.resource.read.local",
                "payload_template": '{"uri":"note://example"}',
            }
        ]

    async def run_action(self, action_name, payload):
        self.action_calls.append((action_name, dict(payload or {})))
        if self.error is not None:
            raise self.error
        return {"ok": True, "action": action_name}


class ToolExecuteAndReadInspectorApp(ConsolidatedCSSApp):
    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__()
        self.service = ToolExecuteAndReadAdvService(error=error)

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        self.query_one(MCPInspector).set_service_context(
            self.service, [("Inventory", "inventory")]
        )


def _adv_result(app) -> str:
    return str(app.query_one("#mcp-adv-result", Static).renderable)


async def _press_run_again(pilot) -> None:
    """Press Run Action a second time, past Button's own active-effect window.

    `Button._on_click()` ignores a click while the widget still carries the
    0.2s `-active` press-animation class (textual/widgets/_button.py), so
    back-to-back `pilot.click()` calls in one pump window silently drop every
    other press -- nothing to do with the confirm arm under test here.
    """
    await pilot.pause(0.3)
    await pilot.click("#mcp-adv-run")
    await pilot.pause()


@pytest.mark.asyncio
async def test_advanced_tool_execute_first_press_confirms_instead_of_running():
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == []
        result = _adv_result(app)
        assert "again" in result
        assert "search_notes" in result


@pytest.mark.asyncio
async def test_advanced_tool_execute_second_press_runs_it():
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]
        assert "ok" in _adv_result(app)


@pytest.mark.asyncio
async def test_advanced_tool_execute_payload_edit_rearms_the_confirm():
    """The confirm covers the payload it was shown for: editing the tool name
    between presses must re-confirm, never run the thing that was confirmed
    a moment ago with different arguments."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        app.query_one(
            "#mcp-adv-payload", TextArea
        ).text = '{"tool_name":"delete_everything","arguments":{}}'
        await _press_run_again(pilot)
        assert app.service.action_calls == []
        assert "delete_everything" in _adv_result(app)

        await _press_run_again(pilot)
        assert app.service.action_calls == [
            ("tool.execute", {"tool_name": "delete_everything", "arguments": {}})
        ]


@pytest.mark.asyncio
async def test_advanced_tool_execute_refusal_reads_as_blocked_not_failed():
    """An Off tool's `MCPHubGateDeniedError` (the Hub's own gate refusal --
    what `execute_advanced_tool()` really raises for this case, item 2, PR-T3
    fix round D) is a refusal, not a crash: same "Blocked · not run" heading
    `show_tool_result()` gives a blocked test run, never the generic "Action
    failed:" dump.

    Item 2: before fix round D, `_run_advanced_action()`'s `except
    PermissionError` matched the BASE class, so this fake's error object
    only needed to BE a `PermissionError` to prove the rendering -- it used
    a bare one. Now that the handler is narrowed to typed refusals only, the
    fake must raise the SAME type production code raises, or this test would
    no longer describe a reachable scenario (see the sibling `test_advanced_
    tool_execute_tool_body_permission_error_reads_as_failed_not_blocked`
    just below for the case a bare `PermissionError` now falls through to)."""
    app = ToolExecuteInspectorApp(
        error=MCPHubGateDeniedError("search_notes is set to Off in Permissions.")
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        result = _adv_result(app)
        assert "Blocked · not run" in result
        assert "set to Off in Permissions" in result
        assert "Action failed" not in result


@pytest.mark.asyncio
async def test_advanced_tool_execute_tool_body_permission_error_reads_as_failed_not_blocked():
    """Item 2 (PR-T3 fix round D) regression guard: a bare `PermissionError`
    -- what a BUILT-IN TOOL'S OWN body raises for an unrelated reason (e.g. a
    genuine OS EACCES reading a permission-denied path), NOT one of the
    three typed refusals -- must fall through to the generic "Action
    failed:" dump, not misrender as "Blocked · not run" (which would falsely
    claim the call never reached the tool, when it reached the tool and the
    tool is what failed). Mirrors `mcp_workbench.py`'s own
    `test_is_permission_refusal_bare_permission_error_from_tool_body_is_not_
    a_refusal` for the Test Tool surface -- this is the Advanced surface's
    twin of that same guard."""
    app = ToolExecuteInspectorApp(
        error=PermissionError("EACCES: permission denied reading /etc/shadow")
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        result = _adv_result(app)
        assert "Action failed" in result
        assert "EACCES" in result
        assert "Blocked · not run" not in result


@pytest.mark.asyncio
async def test_advanced_tool_execute_governance_denial_reads_as_blocked_not_failed():
    """Item 2 (PR-T3 fix round D): `MCPGovernanceDenied` -- the in-process
    runtime-governance profile's own denial, raised further down inside
    `execute_hub_tool()`'s `coro` -- is a genuine refusal on the Advanced
    surface too, not just the Test Tool panel `_is_permission_refusal()`
    already covers."""
    app = ToolExecuteInspectorApp(
        error=MCPGovernanceDenied("Denied by local governance: tool.execute")
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        result = _adv_result(app)
        assert "Blocked · not run" in result
        assert "Denied by local governance" in result
        assert "Action failed" not in result


@pytest.mark.asyncio
async def test_advanced_raw_tool_call_refusal_reads_as_blocked_not_failed():
    """Item 2 (PR-T3 fix round D): a raw `tools/call` refused by the
    `runtime.request`/`runtime.batch` pre-dispatch scan
    (`RawToolCallRefusedError`) is a genuine refusal on the Advanced
    surface -- both of those are Advanced action descriptors too (Task 6,
    Route B, second door), reachable through this same `run_action()` call
    and this same except clause."""
    app = ToolExecuteInspectorApp(
        error=RawToolCallRefusedError(
            "Tool calls run through the Execute Local Tool action, which "
            "applies your Permissions settings and records the run."
        )
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        result = _adv_result(app)
        assert "Blocked · not run" in result
        assert "Execute Local Tool" in result
        # Fix Round J: restored. 60f2f0f7d (round H) inserted new tests
        # above this line and the diff re-parented it into the inserted
        # function -- it vanished with no `-` line, caught only by the
        # reviewer's AST assertion inventory (a per-commit COUNT sweep
        # misses it too: the same commit added two other copies, so the
        # file total still rose).
        assert "Action failed" not in result


@pytest.mark.asyncio
async def test_advanced_gate_error_refusal_renders_the_unresolved_clause_end_to_end():
    """Fix Round H (PR-T3 review), Item 2c. Two earlier rounds converged the
    Advanced hatch's OWN copy for a resolver failure onto
    `_ADVANCED_EXECUTE_GATE_ERROR_MESSAGE` ("Permission state could not be
    resolved.", derived from `local_runtime_delegate.PERMISSION_STATE_
    UNRESOLVED_CLAUSE`) -- but before this test, nothing drove that
    sentence through `_run_advanced_action()`'s actual render path: the
    ONLY fake capable of raising through `run_action()` (`ToolExecuteAdv
    Service`) was always exercised with the GENUINE-deny message ("... is
    set to Off in Permissions.") or an unrelated refusal type, never this
    one. `execute_advanced_tool()` raises `MCPHubGateDeniedError` with
    EITHER message depending on `state.origin` -- both are the SAME typed
    exception, so this only needed a different message, not a different
    type."""
    app = ToolExecuteInspectorApp(
        error=MCPHubGateDeniedError(_ADVANCED_EXECUTE_GATE_ERROR_MESSAGE)
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        await _press_run_again(pilot)
        result = _adv_result(app)
        assert "Blocked · not run" in result
        assert "Permission state could not be resolved." in result
        assert "is set to Off in Permissions" not in result
        assert "Action failed" not in result


@pytest.mark.asyncio
async def test_fake_adv_service_run_action_raises_when_configured():
    """Fix Round H (PR-T3 review), Item 2c. `FakeAdvService` -- the fake
    `InspectorApp` wires by default, used far more broadly across this file
    than `ToolExecuteAdvService` -- used to return `{"ok": True}`
    unconditionally, so no test built on `InspectorApp` could exercise
    `_run_advanced_action()`'s refusal-rendering branch at all. Proves the
    new `error` constructor param actually reaches `run_action()`'s raise,
    using the single-press `profile.connect` action already wired on this
    fake (no `tool.execute` confirm-arm to press through first)."""
    app = InspectorApp(
        error=MCPGovernanceDenied("Denied by local governance: profile.connect")
    )
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        result = _adv_result(app)
        assert "Blocked · not run" in result
        assert "Denied by local governance" in result
        assert app.service.action_calls == [("profile.connect", {"profile_id": "demo"})]
        assert "Action failed" not in result


@pytest.mark.asyncio
async def test_advanced_non_execute_action_still_runs_on_the_first_press():
    """The confirm is scoped to the one action that executes a tool -- the
    read/config actions keep their single-press behavior."""
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == [("profile.connect", {"profile_id": "demo"})]


# -- Fix Round A, Item 3: the confirm arm has no lifecycle reset -------------
#
# `_advanced_confirm_key` was cleared only on a run (any outcome) or a JSON
# parse error -- never on `set_service_context()` (called UNCONDITIONALLY by
# the workbench on every reload/source-target switch/selection change, per
# that method's own docstring), Advanced panel hide/show, or a section
# change. A same-`(action, payload)` collision across two of those --
# entirely plausible, since `tool.execute`'s default template is often
# identical text regardless of which object is showing -- let a stale arm
# from a PREVIOUS object satisfy a DIFFERENT object's very first press, with
# no confirm text ever shown for it: a raw-JSON tool execution firing from a
# single click.
#
# Fix Round C, Item 3 (review of Fix Round A): the fix clears the arm in
# THREE places -- `set_service_context()`, `_load_advanced_section()`
# (which `set_service_context()` also schedules as a worker on every call),
# and `_hide_advanced()`. Mutation-testing each clear individually found
# `set_service_context()`'s own clear and `_hide_advanced()`'s own clear
# were NOT independently pinned: reverting either one alone left every test
# in this section green, because `set_service_context()`'s scheduled
# `_load_advanced_section()` worker clears the arm again before any test's
# assertions ran (both the rebind test and the hide-then-REVEAL test call
# `set_service_context()`, directly or via `_reveal_advanced()`). Only
# `_load_advanced_section()`'s own clear was actually pinned, by
# `test_section_change_disarms_a_pending_confirm` below (its only trigger
# between arm and check is a direct section-value change, which never
# touches the other two clears). This inverts the original commit
# message's framing of `set_service_context()` as "the headline case" and
# `_load_advanced_section()` as "redundant with the above, harmless" -- the
# opposite was true of the tests, not the code (the belt-and-braces clears
# are all real and correct; only the evidence for two of them was hollow).
#
# Fixed by isolating each seam so reverting THAT clear alone turns THAT
# test red, confirmed by re-running the same revert-and-check mutation
# table below.


@pytest.mark.asyncio
async def test_set_service_context_disarms_a_pending_confirm():
    """`set_service_context()`'s OWN clear, isolated from the
    `_load_advanced_section()` worker it also schedules (which clears too --
    see the section banner comment above). Checks `_advanced_confirm_key`
    immediately after the synchronous `set_service_context()` call returns,
    with no intervening `await`: `run_worker(...)` schedules the section
    worker onto the event loop but cannot run it inline, so at that exact
    point only `set_service_context()`'s own body has had a chance to run.

    Also exercises the full round-trip (a real re-arm and a real run) so
    this remains the end-to-end regression test the original title
    promised, not just a white-box state peek.

    Uses `_press_run_again` (not a bare second `pilot.click`) for every
    press past the first -- `Button._on_click()` ignores a click still
    inside the widget's 0.2s active-effect window, and a silently-dropped
    click would leave `action_calls == []` for the WRONG reason, making a
    broken fix look like a passing test.
    """
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms against the default payload
        await pilot.pause()
        assert app.service.action_calls == []
        assert inspector._advanced_confirm_key is not None  # sanity: really armed

        # Fix Round E, Item 4: baseline BEFORE the rebind -- the initial
        # `on_mount()` schedule already ran `_load_advanced_section()` once
        # during app startup, so `load_section_calls` is not zero here; the
        # observable is the DELTA across this call, not an absolute count.
        calls_before_rebind = app.service.load_section_calls

        # The workbench rebinds Advanced on every reload/selection change --
        # simulate that here with the SAME service, so the re-rendered
        # tool.execute template is byte-identical to what was just armed.
        inspector.set_service_context(app.service, [("Inventory", "inventory")])
        # No `await` between the call above and this assertion: the
        # `_load_advanced_section` worker `set_service_context()` schedules
        # cannot have run yet, so this can only be `set_service_context()`'s
        # own clear at work.
        #
        # Fix Round E, Item 4: that ordering guarantee holds TODAY only
        # because `Worker._start` uses `asyncio.create_task`.
        #
        # Fix Round G, Item 4 (review of Fix Round E): the rationale above
        # used to continue "...if a future change added `thread=True` to
        # the `run_worker(...)` call below, a real OS thread could race
        # ahead and run `_load_advanced_section` (and its own
        # `_advanced_confirm_key = None` clear) before this assertion,
        # silently re-hollowing this test's isolation WITHOUT EITHER
        # ASSERTION HERE GOING RED." Not reproducible: adding `thread=True`
        # to that `run_worker(...)` call reddened 5/5 runs, but at a
        # DIFFERENT, PRE-EXISTING assertion three lines above this one --
        # `assert inspector._advanced_confirm_key is not None  # sanity:
        # really armed` -- because the initial on-MOUNT `set_service_
        # context()` call (`ToolExecuteInspectorApp.on_mount()`) ALSO
        # schedules a `_load_advanced_section` worker, and a genuine OS
        # thread can win the race against the pilot's own arming click:
        # the mount-time worker's clear lands AFTER the click has armed the
        # key, wiping it back to `None` before this test ever reaches the
        # code below. So a real thread does reddened the suite -- just not
        # silently, and not at the pin this comment worried about.
        #
        # The pin below is still worth keeping regardless: it is genuinely
        # non-vacuous (fires on the simulated `thread=True` race just
        # demonstrated in a DIFFERENT reachable way, and on the realistic
        # re-hollowing of a maintainer inserting `await pilot.pause()`
        # before this assertion, which lets the ALREADY-SCHEDULED
        # `asyncio.create_task` worker run first even without `thread=True`
        # -- verified separately: adding a bare `await pilot.pause()` here
        # reddens this exact assertion, 5/5, with no code mutation at all).
        # Residual, not fixed here (cheap-only per this round's brief): the
        # observable is `load_section()`, which `_load_advanced_section()`
        # reaches a few statements AFTER the clear being isolated -- under
        # a genuine thread there is a narrow window where the clear has
        # landed but the counter has not yet incremented, which this
        # delta-based pin cannot see. No cheap fix found that doesn't move
        # the production clear's own position; noted here for the next
        # round rather than worked around.
        assert app.service.load_section_calls == calls_before_rebind, (
            "the scheduled _load_advanced_section worker must not have run "
            "yet at this point -- if it had, the assertion below would no "
            "longer isolate set_service_context()'s own clear from that "
            "worker's independent one"
        )
        assert inspector._advanced_confirm_key is None, (
            "set_service_context() itself must clear the arm synchronously -- "
            "this must not depend on the _load_advanced_section worker it "
            "separately schedules"
        )
        await pilot.pause()

        await _press_run_again(pilot)
        assert app.service.action_calls == [], (
            "a rebind must disarm -- the first press after it must "
            "re-confirm, not execute"
        )
        assert "again" in _adv_result(app)

        # A second (genuinely fresh) press DOES run it -- proving this is a
        # real re-arm, not a button stuck disabled some other way.
        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]


@pytest.mark.asyncio
async def test_hide_advanced_disarms_a_pending_confirm():
    """`_hide_advanced()`'s OWN clear, isolated from `_reveal_advanced()`'s
    subsequent `set_service_context()` replay (which independently clears
    too -- see the section banner comment above, and
    `test_advanced_hide_then_reveal_disarms_a_pending_confirm` below for the
    full round trip). Calls `_hide_advanced()` directly and checks
    immediately, with the panel never revealed again in this test, so only
    `_hide_advanced()`'s own clear can be responsible for the result."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms against the default payload
        await pilot.pause()
        assert app.service.action_calls == []
        assert inspector._advanced_confirm_key is not None  # sanity: really armed

        await inspector._hide_advanced()

        assert inspector._advanced_confirm_key is None, (
            "_hide_advanced() itself must clear the arm -- this must not "
            "depend on a later reveal's set_service_context() replay"
        )


@pytest.mark.asyncio
async def test_advanced_hide_then_reveal_disarms_a_pending_confirm():
    """End-to-end coverage of the full F-053 toggle round trip: the
    Advanced panel's hide/show is an attention-moved transition, and a user
    backing out of the legacy runner and returning later must not find a
    stale arm ready to fire on the first press after showing it again.

    This exercises BOTH clears in the hide-then-reveal path
    (`_hide_advanced()`'s own, and `_reveal_advanced()`'s
    `set_service_context()` replay) together, so on its own it does not
    prove which one is responsible -- that isolation is
    `test_hide_advanced_disarms_a_pending_confirm` above (hide) and
    `test_set_service_context_disarms_a_pending_confirm` (the replay)."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        assert app.service.action_calls == []

        await pilot.click("#mcp-inspector-advanced-reveal")  # hide
        await pilot.pause()
        # Same Button active-effect cooldown as `_press_run_again` guards
        # against, this time on the reveal/hide toggle button itself --
        # two clicks on the SAME button back to back would otherwise drop
        # this second one and leave the panel hidden.
        await pilot.pause(0.3)
        await pilot.click("#mcp-inspector-advanced-reveal")  # reveal again
        await pilot.pause()

        await _press_run_again(pilot)
        assert app.service.action_calls == [], (
            "hide-then-reveal must disarm -- the first press after "
            "revealing must re-confirm, not execute"
        )
        assert "again" in _adv_result(app)

        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]


@pytest.mark.asyncio
async def test_section_change_disarms_a_pending_confirm():
    """A section change within the SAME service (no rebind at all) is the
    narrowest version of the collision: `ToolExecuteAdvService.
    available_actions()` is not itself keyed by section (mirroring the real
    inventory-section template), so switching sections while `tool.execute`
    is still offered reproduces the identical payload template -- exactly
    the case a same-`(action, payload)` confirm key would otherwise
    silently satisfy."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Inventory", "inventory"), ("Overview", "overview")]
        )
        await pilot.pause()
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        assert app.service.action_calls == []

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "overview"
        await pilot.pause()
        await pilot.pause()

        await _press_run_again(pilot)
        assert app.service.action_calls == [], (
            "a section change must disarm -- the first press in the new "
            "section must re-confirm, not execute"
        )
        assert "again" in _adv_result(app)

        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]


class _TallSectionAdvService(ToolExecuteAdvService):
    """`load_section()` returns a payload that renders ~200+ rows -- the
    shape the REAL Inventory section produces (every builtin tool's full
    schema) and the one shape no other fake in this file produces. Fix
    Round K exists because that gap hid a live defect: with a tall
    payload, `#mcp-adv-scroll`'s `height: 1fr` (inside the T12
    Collapsible) resolved without subtracting the rows above the
    collapsible, so the box hung past the screen bottom and the Run
    Action button was unreachable at ANY terminal height."""

    async def load_section(self, section=None):
        self.load_section_calls += 1
        return {
            "tools": [
                {
                    "name": f"tool_{i}",
                    "description": "A long wrapping description. " * 4,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "alpha": {"type": "string"},
                            "beta": {"type": ["integer", "null"]},
                        },
                        "required": ["alpha"],
                    },
                }
                for i in range(10)
            ]
        }


class _TallSectionInspectorApp(ConsolidatedCSSApp):
    """Bundled CSS + real workbench id, like `InspectorAppWithBundledCSS`
    above -- geometry claims are meaningless against Textual's defaults."""

    CSS_PATH = _BUNDLED_CSS_PATH

    def __init__(self) -> None:
        super().__init__()
        self.service = _TallSectionAdvService()

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-hub-inspector")

    def on_mount(self) -> None:
        self.query_one(MCPInspector).set_service_context(
            self.service, [("Inventory", "inventory")]
        )


@pytest.mark.asyncio
async def test_run_action_is_reachable_under_a_tall_section_payload():
    """Fix Round K (live walkthrough of PR #1385): with a real-sized
    section payload, `scroll_visible()` must be able to bring
    `#mcp-adv-run` fully on screen and a real click must land on it --
    before the fix the box's own region hung below the screen bottom and
    the click raised OutOfBounds no matter how far the scroll went
    (reproduced live at 300 terminal rows; this test red-verifies by
    restoring `height: 1fr` on `#mcp-adv-scroll`/`#mcp-adv-collapsible`).
    The press must also be the ARM press (confirm sentence, no
    execution): reachability that ran the tool unconfirmed would be a
    worse defect than the one this fixes."""
    app = _TallSectionInspectorApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.pause()
        run_button = app.query_one("#mcp-adv-run", Button)
        run_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.pause()

        await pilot.click("#mcp-adv-run")  # raises OutOfBounds when clipped
        await pilot.pause()

        assert app.service.action_calls == []  # armed, not executed
        assert inspector_confirm_key(app) is not None


def inspector_confirm_key(app: App):
    return app.query_one(MCPInspector)._advanced_confirm_key


class _ArmDuringLoadAdvService(ToolExecuteAdvService):
    """`load_section()` blocks (once, when told to via `block_next`) until
    the test releases it -- opening a REAL await window inside
    `_load_advanced_section()`, between that method's own deliberate disarm
    (before the await) and `_refresh_advanced_actions()`'s payload rewrite
    (after it). Fix Round I, Item 2 lives entirely inside that window."""

    def __init__(self) -> None:
        super().__init__()
        self.block_next = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def load_section(self, section=None):
        if self.block_next:
            self.block_next = False
            self.entered.set()
            await self.release.wait()
        return await super().load_section(section)


@pytest.mark.asyncio
async def test_arming_during_a_section_load_survives_the_post_load_refresh():
    """Fix Round I, Item 2 (review of Fix Round G): the previous round's
    `_on_advanced_payload_changed()` docstring claimed every programmatic
    `payload.text = ...` write "always runs AFTER that call site's own
    clear, so this is a no-op" -- FALSE for `_load_advanced_section()`,
    whose clear runs BEFORE `await self._service.load_section(...)` while
    `_refresh_advanced_actions()`'s payload write lands AFTER it. A user
    who armed during that window (the Run button is not disabled during a
    load) was silently disarmed by the post-await write, with no user
    action at all -- the exact inverse of the copy's "anything else
    cancels" promise (nothing-the-user-did cancelled), and a fresh copy of
    the "button reads as dead for one press" symptom Fix Round G, Item 2
    existed to eliminate. The write is now wrapped in
    `payload.prevent(TextArea.Changed)`; the arm must survive the refresh
    AND still be a live arm (the follow-up press really runs).

    Counterpoint to `test_section_change_disarms_a_pending_confirm` just
    above: an arm from BEFORE the section change is deliberately cleared
    (attention moved), but an arm placed DURING the load belongs to the
    user standing right there, and background plumbing must not eat it."""
    app = ToolExecuteInspectorApp()
    app.service = _ArmDuringLoadAdvService()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Inventory", "inventory"), ("Overview", "overview")]
        )
        await pilot.pause()
        await pilot.pause()  # let the rebind's own initial load settle

        app.service.block_next = True
        app.query_one("#mcp-adv-section-select", Select).value = "overview"
        await asyncio.wait_for(app.service.entered.wait(), timeout=2)
        # Inside the await window: `_load_advanced_section()` has already
        # done its own deliberate pre-await disarm; the user arms NOW.
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert inspector._advanced_confirm_key is not None  # sanity: armed
        assert app.service.action_calls == []

        app.service.release.set()
        await pilot.pause()
        await pilot.pause()  # post-await refresh (payload rewrite) runs here

        assert inspector._advanced_confirm_key is not None, (
            "the background load's own payload rewrite disarmed a confirm "
            "the user armed during the load window"
        )
        assert "again" in _adv_result(app)

        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ], "the surviving arm must be LIVE -- the confirming press runs"


# -- Fix Round C, Item 4: the confirm sentence survives the disarm -----------
#
# `set_service_context()` and `_load_advanced_section()` both clear
# `_advanced_confirm_key` (Item 3 above) but only ever blanked
# `#mcp-adv-content` -- never `#mcp-adv-result`, which is where
# `_ADVANCED_EXECUTE_CONFIRM` ("Runs <tool> now — press Run Action again to
# confirm...") actually renders. A disarm therefore left the confirm
# sentence on screen describing an arm that no longer existed: the next
# press silently RE-armed (rendering the identical string back) instead of
# running, so the button read as dead for one press with no visible change
# to explain why.


@pytest.mark.asyncio
async def test_set_service_context_blanks_the_stale_confirm_sentence():
    """A rebind must clear the confirm sentence, not just the arm behind
    it -- otherwise the screen still reads "press Run Action again to
    confirm" for a confirm that set_service_context() just disarmed.

    Checked immediately after the synchronous `set_service_context()` call
    returns, with no intervening `await` -- same isolation as
    `test_set_service_context_disarms_a_pending_confirm` above, and for the
    same reason: `set_service_context()` also SCHEDULES
    `_load_advanced_section()` as a worker, which blanks
    `#mcp-adv-result` too, so a version of this check that let the worker
    run first would pass even with `set_service_context()`'s own blank
    removed."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms; renders the confirm sentence
        await pilot.pause()
        assert "again" in _adv_result(app)

        # Fix Round E, Item 4: baseline BEFORE the rebind -- see
        # `test_set_service_context_disarms_a_pending_confirm`'s matching
        # comment for why this is a delta, not an absolute-zero check (the
        # initial `on_mount()` schedule already ran the worker once).
        calls_before_rebind = app.service.load_section_calls

        inspector.set_service_context(app.service, [("Inventory", "inventory")])
        # No `await` here -- see the docstring above. Fix Round E, Item 4:
        # same observable-call-count pin as
        # `test_set_service_context_disarms_a_pending_confirm` above -- see
        # that test for why an inference from state alone would not notice
        # a future `thread=True` letting the scheduled worker win the race.
        assert app.service.load_section_calls == calls_before_rebind, (
            "the scheduled _load_advanced_section worker must not have run "
            "yet at this point -- if it had, this check would no longer "
            "isolate set_service_context()'s own blank from that worker's "
            "independent one"
        )
        assert _adv_result(app) == "", (
            "the stale confirm sentence must be cleared along with the arm, "
            "not just the arm's internal state -- and not by relying on the "
            "_load_advanced_section worker set_service_context() separately "
            "schedules"
        )


@pytest.mark.asyncio
async def test_section_change_blanks_the_stale_confirm_sentence():
    """Same as above for `_load_advanced_section()`'s own disarm, triggered
    by a direct section change (no rebind)."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Inventory", "inventory"), ("Overview", "overview")]
        )
        await pilot.pause()
        await pilot.click("#mcp-adv-run")  # arms; renders the confirm sentence
        await pilot.pause()
        assert "again" in _adv_result(app)

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "overview"
        await pilot.pause()
        await pilot.pause()

        assert _adv_result(app) == "", (
            "the stale confirm sentence must be cleared along with the arm, "
            "not just the arm's internal state"
        )


def test_advanced_execute_confirm_copy_does_not_enumerate_cancel_triggers():
    """Fix Round E, Item 2 (review of Fix Round C): the confirm sentence
    used to enumerate cancel triggers ("Editing, switching object, or
    changing section cancels") -- an incomplete list even after Fix Round
    C's own pass (it omitted switching the action, and hiding the panel),
    and any enumeration here is one more trigger away from being wrong
    again. The corrected copy adopts the house's existing complete
    formulation (`_TEST_RUN_ARMED_HINT`: "anything else cancels") instead,
    which stays true regardless of how many triggers exist or are added."""
    rendered = mcp_inspector_module._ADVANCED_EXECUTE_CONFIRM.format(
        tool="search_notes"
    )
    assert "press Run Action again to confirm" in rendered
    assert "anything else cancels" in rendered
    # Regression guard: no version of this copy should go back to naming
    # triggers by enumeration -- if it does, it is incomplete again.
    assert "Editing" not in rendered
    assert "switching object" not in rendered
    assert "changing section" not in rendered


# -- Fix Round E, Item 1: the disarm blank must not eat real output ----------
#
# Fix Round C's blanking fix for the stale confirm sentence (`#mcp-adv-result`
# above) ran UNCONDITIONALLY in both `set_service_context()` and
# `_load_advanced_section()` -- but that widget is also where genuine RUN
# OUTPUT and refusal text land. Reverting Fix Round E's conditional blank
# (making it unconditional again) is what a reviewer demonstrated makes each
# of the three tests below fail: real output/refusal text present while
# UNARMED must survive a section change or a rebind; only a LIVE arm's
# confirm sentence may be blanked away.


@pytest.mark.asyncio
async def test_section_change_preserves_real_run_output_when_not_armed():
    """Run `tool.execute` to completion (arm, then confirm), THEN change
    section -- the tool's real JSON result must still be on screen.
    Re-reading it by re-running the tool is the exact loss this guards."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Inventory", "inventory"), ("Overview", "overview")]
        )
        await pilot.pause()
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        await _press_run_again(pilot)  # runs it -- real output now showing
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]
        result_before = _adv_result(app)
        assert "ok" in result_before

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "overview"
        await pilot.pause()
        await pilot.pause()

        assert _adv_result(app) == result_before, (
            "a section change while UNARMED must not blank real run output -- "
            "the confirm-sentence blank is only correct while an arm is live"
        )


@pytest.mark.asyncio
async def test_rebind_preserves_real_run_output_when_not_armed():
    """Same defect via the OTHER blanking site: a workbench rebind
    (`set_service_context()`, called unconditionally on every
    reload/source-or-target switch, per that method's own docstring) must
    not erase real run output either."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        await _press_run_again(pilot)  # runs it -- real output now showing
        result_before = _adv_result(app)
        assert "ok" in result_before

        # Simulate a workbench rebind with the same service, same as the
        # existing disarm tests above.
        inspector.set_service_context(app.service, [("Inventory", "inventory")])
        await pilot.pause()

        assert _adv_result(app) == result_before, (
            "a rebind while UNARMED must not blank real run output"
        )


@pytest.mark.asyncio
async def test_section_change_preserves_blocked_refusal_when_not_armed():
    """The third loss: a "Blocked · not run" refusal explaining WHY nothing
    ran must not be erased by a section change exactly as the user
    navigates to go act on it."""
    app = ToolExecuteInspectorApp(
        error=MCPHubGateDeniedError("search_notes is set to Off in Permissions.")
    )
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(
            app.service, [("Inventory", "inventory"), ("Overview", "overview")]
        )
        await pilot.pause()
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        await _press_run_again(pilot)  # blocked
        result_before = _adv_result(app)
        assert "Blocked · not run" in result_before

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "overview"
        await pilot.pause()
        await pilot.pause()

        assert _adv_result(app) == result_before, (
            "a section change while UNARMED must not erase a refusal "
            "explaining why nothing ran"
        )


# -- Fix Round E, Item 2: switching the action is a fourth disarm trigger ----


@pytest.mark.asyncio
async def test_action_switch_disarms_a_pending_confirm():
    """Switching the Advanced action Select must disarm a pending
    `tool.execute` confirm -- `_run_advanced_action()`'s own docstring
    already promised it ("switching action ... re-arms"), but
    `on_select_changed()` never actually cleared `_advanced_confirm_key` on
    an action switch. Live-verified defect: arm `tool.execute`, switch to
    `resource.read` (its own same-section sibling -- both reads, see the
    item's honest scoping), and the FIRST press after the switch used to
    run `resource.read` immediately with no confirm, while the pane still
    read the `tool.execute` confirm sentence. Scoped deliberately to a
    same-section READ action, never a destructive one: this is a
    truthfulness defect on the confirm text, not a path to an unconfirmed
    destructive call (the destructive actions live in other sections, which
    a section change already disarms)."""
    app = ToolExecuteAndReadInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")  # arms tool.execute
        await pilot.pause()
        assert app.service.action_calls == []
        assert "again" in _adv_result(app)

        action_select = app.query_one("#mcp-adv-action-select", Select)
        action_select.value = "resource.read"
        await pilot.pause()

        assert _adv_result(app) == "", (
            "switching the action must blank the stale confirm sentence, "
            "not just clear the internal arm -- otherwise the pane still "
            "promises a confirm the next press will not give"
        )

        await _press_run_again(pilot)
        # A disarmed action switch means the first press on the NEW action
        # runs it directly -- resource.read never required a confirm of its
        # own (only tool.execute does), so this is the ordinary
        # single-press behavior, not a second confirm cycle.
        assert app.service.action_calls == [
            ("resource.read", {"uri": "note://example"})
        ]


@pytest.mark.asyncio
async def test_action_switch_clears_the_confirm_key_synchronously():
    """Item 3 (PR-T3 fix round G, review of Fix Round E): isolates
    `on_select_changed()`'s OWN clear (`self._advanced_confirm_key = None`
    on the action-switch branch) from BOTH of the two things that can now
    mask a dropped clear:

    1. Its own blank (the `_was_armed`-gated `#mcp-adv-result.update("")`
       a few lines below) -- `test_action_switch_disarms_a_pending_
       confirm` (pre-existing, above) only checks that visible blank and
       a later run of `resource.read` (which never confirms at all
       regardless of the arm -- `_run_advanced_action()`'s confirm logic
       only runs for `action_name == _ADVANCED_EXECUTE_ACTION`), so it
       passes with this clear deleted.
    2. Item 2's OWN new `TextArea.Changed` handler
       (`_on_advanced_payload_changed()`), which independently clears the
       SAME key -- `on_select_changed()`'s action-switch branch always
       reassigns `#mcp-adv-payload.text` to the new action's template
       (when `event.value` isn't blank, the only reachable case via real
       UI), and `TextArea.load_text()` posts a `Changed` message
       UNCONDITIONALLY, even when the new text is byte-identical to the
       old. Discovered by mutation: dropping ONLY this clear (Item 2's
       clear intact) and driving the switch through the real Select
       (`action_select.value = ...` + `await pilot.pause()`, draining the
       whole cascade including the queued `TextArea.Changed`) left this
       test GREEN -- Item 2's cascade silently covers for Item 3's own
       clear on every action switch reachable via the UI. That is a
       genuine, additional belt-and-braces safety net (see Item 6 of this
       same fix round's report), not a reason to leave THIS clear
       unpinned -- a future change to either `TextArea`'s posting
       behavior or this branch's unconditional reassignment would silently
       remove that net.

    Isolated by calling `on_select_changed()` DIRECTLY (bypassing the
    message queue entirely) and checking immediately after, no
    `await`/`pilot.pause()` in between -- `post_message()` queues the
    payload's cascaded `TextArea.Changed` for LATER delivery, so at this
    exact point only `on_select_changed()`'s own body has had a chance to
    run. Mirrors `test_set_service_context_disarms_a_pending_confirm`'s
    own no-intervening-await isolation technique.

    See `test_action_switch_round_trip_does_not_execute_tool_execute_
    unconfirmed` below for the genuine-UI reachable consequence -- which,
    because of the same masking, now requires dropping Item 2's clear
    TOO to reproduce via real interaction."""
    app = ToolExecuteAndReadInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms tool.execute
        await pilot.pause()
        assert inspector._advanced_confirm_key is not None  # sanity: really armed

        action_select = app.query_one("#mcp-adv-action-select", Select)
        inspector.on_select_changed(Select.Changed(action_select, "resource.read"))

        assert inspector._advanced_confirm_key is None, (
            "on_select_changed() must clear the arm itself on an action "
            "switch -- not merely blank the rendered confirm sentence, and "
            "not merely rely on the cascaded TextArea.Changed its own "
            "payload-template reassignment will eventually deliver"
        )


@pytest.mark.asyncio
async def test_action_switch_round_trip_does_not_execute_tool_execute_unconfirmed():
    """Item 3 (PR-T3 fix round G, review of Fix Round E): the genuine-UI
    reachable safety consequence the original review demonstrated --
    `tool.execute -> resource.read -> tool.execute`, one press, must never
    execute unconfirmed.

    Reality differed from the brief here: the brief's own mutation
    ("drop `self._advanced_confirm_key = None` at 2603, keep the
    `_was_armed` blank -> executes on one press") was proven against the
    code as Fix Round E left it, BEFORE this same round's Item 2 added
    `_on_advanced_payload_changed()`. With Item 2 also in place, dropping
    ONLY `on_select_changed()`'s own clear no longer reproduces this via
    real UI interaction -- switching to `resource.read` and back to
    `tool.execute` reassigns `#mcp-adv-payload.text` each time, and that
    reassignment's cascaded `TextArea.Changed` clears the arm independently
    (see `test_action_switch_clears_the_confirm_key_synchronously`'s own
    docstring for the mechanism). Reproducing the ORIGINAL finding via
    genuine UI interaction now requires dropping BOTH clears together --
    verified directly: with `on_select_changed()`'s clear (2684) AND
    `_on_advanced_payload_changed()`'s clear (2721) both dropped, this
    exact test goes RED, with `action_calls` showing `tool.execute` ran on
    the single post-round-trip press. `on_select_changed()`'s own clear
    remains correct and worth keeping regardless (defense in depth,
    consistent with this class's established belt-and-braces philosophy --
    see `test_action_switch_clears_the_confirm_key_synchronously` for its
    OWN, still-vacuous-without-it isolation); this test instead pins the
    OBSERVABLE safety property end to end, independent of which internal
    mechanism is doing the disarming."""
    app = ToolExecuteAndReadInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")  # arms tool.execute
        await pilot.pause()
        assert app.service.action_calls == []

        action_select = app.query_one("#mcp-adv-action-select", Select)
        action_select.value = "resource.read"
        await pilot.pause()
        action_select.value = "tool.execute"
        await pilot.pause()

        assert _adv_result(app) == "", (
            "switching back to tool.execute must not resurrect a stale confirm sentence"
        )

        await _press_run_again(pilot)
        assert app.service.action_calls == [], (
            "a single press after the round trip must re-arm, never "
            "execute tool.execute unconfirmed"
        )
        assert "again" in _adv_result(app)

        # A genuinely fresh confirm DOES run it -- proving this is a real
        # re-arm, not a button stuck disabled some other way.
        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]


# -- Fix Round G, Item 1: collapsing the disclosure disarms a pending -------
# confirm too -- `_ADVANCED_EXECUTE_CONFIRM`'s "anything else cancels" was
# untrue for this one interaction: `_on_advanced_collapsible_toggled()` only
# ever persisted the open/collapsed preference. Live-verified: arm
# `tool.execute`, collapse the disclosure, expand it again -- a single press
# ran the tool with the arm still set, no confirm ever shown for that
# viewing.


@pytest.mark.asyncio
async def test_collapsing_advanced_disarms_a_pending_confirm():
    """Direct-state isolation (the RED condition mutation-verified against
    dropping `_on_advanced_collapsible_toggled()`'s clear while keeping its
    `_was_armed` blank) plus the end-to-end behavioral consequence in one
    test: `Collapsible.collapsed = True/False` fires the SAME
    `Collapsible.Toggled` message a real click on the disclosure's title
    does -- this file's own established technique, see e.g.
    `test_advanced_collapsible_toggle_persists_state`."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        assert app.service.action_calls == []

        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        collapsible.collapsed = True
        await pilot.pause()
        assert inspector._advanced_confirm_key is None, (
            "collapsing the disclosure must clear the arm itself, not merely be inert"
        )
        collapsible.collapsed = False
        await pilot.pause()

        await _press_run_again(pilot)
        assert app.service.action_calls == [], (
            "collapse-then-expand must disarm -- the first press after "
            "must re-confirm, not execute"
        )
        assert "again" in _adv_result(app)

        await _press_run_again(pilot)
        assert app.service.action_calls == [
            (
                "tool.execute",
                {"tool_name": "search_notes", "arguments": {"query": "example"}},
            )
        ]


@pytest.mark.asyncio
async def test_collapsing_advanced_preserves_real_run_output_when_not_armed():
    """Same UNARMED-preservation discipline as `test_section_change_
    preserves_real_run_output_when_not_armed`/`test_rebind_preserves_real_
    run_output_when_not_armed` (Fix Round E, Item 1) -- collapsing after a
    completed run (unarmed) must not blank the real result; only a LIVE
    arm's confirm sentence is ever cleared. The Collapsible's children stay
    mounted across a collapse (only CSS display toggles), so this is the
    same "preserve" branch of the Fix Round G, Item 6 rule, not the
    `_hide_advanced()` teardown branch."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        await _press_run_again(pilot)  # runs it -- real output now showing
        result_before = _adv_result(app)
        assert "ok" in result_before

        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        collapsible.collapsed = True
        await pilot.pause()

        assert _adv_result(app) == result_before, (
            "collapsing while UNARMED must not blank real run output"
        )


# -- Fix Round G, Item 2: editing the payload disarms a pending confirm -----
# too -- `_run_advanced_action()`'s own docstring has always promised
# "switching action or editing the payload re-arms"; the action-switch half
# was implemented (Fix Round E, Item 2), the payload-edit half had no
# handler at all.


@pytest.mark.asyncio
async def test_editing_the_payload_disarms_a_pending_confirm_immediately():
    """Checked immediately after the edit, no Run press in between -- proves
    the new `TextArea.Changed` handler disarms on its own, not merely that
    `_run_advanced_action()`'s pre-existing confirm-key mismatch happens to
    save the day the next time Run is pressed (see `test_advanced_tool_
    execute_payload_edit_rearms_the_confirm`, which pins that pre-existing
    safety net and stays green either way -- it is not this gap's guard)."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await pilot.click("#mcp-adv-run")  # arms; renders "Runs search_notes..."
        await pilot.pause()
        assert "search_notes" in _adv_result(app)

        app.query_one(
            "#mcp-adv-payload", TextArea
        ).text = '{"tool_name":"delete_everything","arguments":{}}'
        await pilot.pause()

        assert inspector._advanced_confirm_key is None, (
            "editing the payload must clear the arm itself, immediately -- "
            "not merely be discovered stale the next time Run is pressed"
        )
        assert _adv_result(app) == "", (
            "the stale confirm sentence (naming the OLD tool) must not "
            "survive a payload edit the user hasn't confirmed yet"
        )


@pytest.mark.asyncio
async def test_editing_the_payload_preserves_real_run_output_when_not_armed():
    """Same UNARMED-preservation discipline as the other disarm sites --
    editing the payload after a completed run (unarmed) must not blank the
    real result."""
    app = ToolExecuteInspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#mcp-adv-run")  # arms
        await pilot.pause()
        await _press_run_again(pilot)  # runs it -- real output now showing
        result_before = _adv_result(app)
        assert "ok" in result_before

        app.query_one(
            "#mcp-adv-payload", TextArea
        ).text = '{"tool_name":"search_notes","arguments":{"query":"other"}}'
        await pilot.pause()

        assert _adv_result(app) == result_before, (
            "editing the payload while UNARMED must not blank real run output"
        )
