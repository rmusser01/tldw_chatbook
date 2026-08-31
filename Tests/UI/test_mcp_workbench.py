# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import Vertical
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    DataTable,
    Input,
    Select,
    Static,
    TextArea,
)

import tldw_chatbook
import tldw_chatbook.MCP.local_server_tools as local_server_tools_module
import tldw_chatbook.MCP.unified_control_plane_service as unified_service_module
import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mcp_inspector_module
import tldw_chatbook.UI.MCP_Modules.mcp_workbench as mcp_workbench_module
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.raw_shell_tool_provider import (
    RAW_SHELL_SERVER_KEY,
    RAW_SHELL_TOOL_NAME,
)
from tldw_chatbook.MCP.local_control_service import MCPGovernanceDenied
from tldw_chatbook.MCP.hub_test_execution import (
    LocalHubExecutionOutcome,
    ToolTestAdmissionBlocked,
    ToolTestAdmissionPreview,
    ToolTestAdmissionStale,
)
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    HASH_FREE_SERVER_KEYS,
    EffectiveToolState,
    MCPPermissionStore,
    definition_hash,
    resolve_effective_state,
)
from tldw_chatbook.MCP.readiness import HubAction
from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.MCP.unified_control_plane_service import (
    MCPServerSourceDisplayOnlyError,
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
    MCPInspector,
    audit_entry_detail_payload,
)
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


@pytest.fixture(autouse=True)
def _default_advanced_open(monkeypatch):
    """T12: same rationale as test_mcp_inspector.py's fixture of the same
    name -- `MCPWorkbench` mounts a nested `MCPInspector`, whose `compose()`
    reads `mcp.hub_state.advanced_open` via `mcp_inspector.get_cli_setting`
    at mount time. Keep it expanded and never touch the real user config
    file for every workbench test that isn't specifically exercising T12's
    disclosure/persistence behavior itself.
    """
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setattr(
        mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True
    )
    # Keep this broad workbench fake's historical, deliberately small tool
    # inventory stable. Tests for the production default override this seam
    # explicitly; config/controller tests cover the shipped missing-key
    # default independently.
    original_workbench_get = mcp_workbench_module.get_cli_setting

    def workbench_setting(section, key=None, default=None):
        if section == "console" and key == "local_tools_enabled":
            return False
        return original_workbench_get(section, key, default)

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", workbench_setting)


class FakeTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class FakeTargetStore:
    def list_targets(self):
        return [FakeTarget()]


# Fix Round H (PR-T3 review), Item 2c: a genuine fake-vs-real key-set pin
# candidate (`load_section()` below returns hand-rolled, UI-rendered dict
# shapes per source/section) but NOT pinned this round -- see the POLICY
# comment above `FakeLocalMCPControlService` in
# Tests/MCP/test_unified_control_plane_service.py for the line drawn and
# why this one is flagged rather than pinned or silently skipped.
class FakeHubService:
    def __init__(self) -> None:
        self.target_store = FakeTargetStore()
        self.context = UnifiedMCPContext(
            selected_source="local", selected_section="overview"
        )
        self.disconnect_calls: list[str] = []

    async def disconnect_local_profile(self, profile_id):
        self.disconnect_calls.append(profile_id)
        return True

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        # Mirrors UnifiedMCPControlPlaneService.load_section(): under the
        # local source, every section is a dict EXCEPT "external_servers",
        # which LocalMCPControlService.get_external_servers() returns as a
        # bare list. Only that one section is a shape gap the workbench must
        # normalize (see _AdvancedSectionShim in mcp_workbench.py).
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [{"name": "a"}],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    }
                ]
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class WorkbenchApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = FakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


class HubLocalProjectionService(FakeHubService):
    """Use the production local projection while retaining the compact UI fake."""

    class LocalService:
        def get_inventory(self):
            return {
                "tools": [
                    {
                        "name": "builtin_probe",
                        "description": "Unrelated built-in projection probe.",
                    }
                ]
            }

    def __init__(self) -> None:
        super().__init__()
        self.local_service = self.LocalService()

    def gate_tool_test(self, _tool):
        return EffectiveToolState(state="ask", origin="global_default")

    def local_hub_tools(self):
        return UnifiedMCPControlPlaneService.local_hub_tools(self)


class HubLocalWorkbenchApp(WorkbenchApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = HubLocalProjectionService()


@pytest.mark.asyncio
async def test_workbench_mounts_rail_canvas_inspector_and_loads_local_servers():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        # builtin + docs rows (+ "All servers")
        assert len(list(app.query("Button.mcp-rail-row"))) == 3
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-overview").display


class WorkbenchAppWithBundledCSS(WorkbenchApp):
    """WorkbenchApp under the real bundled stylesheet, so bundle-layer rules
    (e.g. `.ds-status-badge { height: 1; }` in _agentic_terminal.tcss) apply
    exactly as in the live app -- mirrors `CanvasAppWithBundledCSS` in
    test_mcp_servers_mode.py."""

    CSS_PATH = _BUNDLED_CSS_PATH


@pytest.mark.asyncio
async def test_workbench_at_100x30_keeps_primary_content_reachable(monkeypatch):
    """F-057: below ~120 cols the hub must not silently lose primary
    content -- the summary wraps (the bundle's `.ds-status-badge`
    `height: 1` used to clip it mid-sentence), the overview table switches
    to a compact column set that fits the viewport (dropped columns' facts
    stay one click away in the detail pane), primary actions stay on
    screen, and rail rows truncate with an ellipsis instead of cropping
    mid-word."""
    # Deterministic builtin state (off/opt-in -> the Enable affordance).
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Summary wraps to multiple lines instead of clipping at one.
        summary = app.query_one("#mcp-overview-summary", Static)
        assert summary.region.height >= 2
        # Compact columns: everything shown fits the viewport -- no column
        # silently lost behind horizontal overflow.
        table = app.query_one("#mcp-servers-table", DataTable)
        assert [str(c.label) for c in table.ordered_columns] == [
            "Name",
            "Status",
            "Tools",
        ]
        assert table.max_scroll_x == 0
        # Primary actions stay reachable.
        assert app.query_one("#mcp-add-server", Button).region.width > 0
        assert app.query_one("#mcp-builtin-enable", Button).region.width > 0
        # The built-in rail row truncates with an ellipsis (honest
        # truncation) rather than cropping mid-word.
        rows = list(app.query("Button.mcp-rail-row"))
        assert "..." in str(rows[1].label)


@pytest.mark.asyncio
async def test_workbench_at_100x30_keeps_server_master_switch_reachable(monkeypatch):
    """Paint may shorten the label, but its semantics and interaction remain."""
    _, save_calls = _fake_tool_gate_config_seam(monkeypatch)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")
        await pilot.pause()

        checkbox = app.query_one("#mcp-gate-local_tools_enabled", Checkbox)
        assert str(checkbox.label) == (
            "Local workspace, web, and Watchlists tools (master switch)"
        )
        checkbox.scroll_visible(animate=False, force=True, immediate=True)
        checkbox.focus()
        await pilot.pause()
        await pilot.pause()
        assert checkbox.is_on_screen
        assert app.focused is checkbox

        rendered = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )
        assert "Local workspace, web, and Watchlists tools" in rendered

        original = checkbox.value
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("console", "local_tools_enabled", not original) in save_calls

        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        title = app.query_one("#mcp-tools-local-config-title", Static)
        title.scroll_visible(animate=False)
        await pilot.pause()
        assert title.is_on_screen
        assert str(title.renderable) == "Local workspace, web, and Watchlists tools"


class ProblemRecordsService(FakeHubService):
    """FakeHubService whose local catalog is a caller-supplied record list,
    so a test can control exactly how many problem servers load (F-054)."""

    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self._records = records

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if (
            self.context.selected_source == "local"
            and effective_section == "external_servers"
        ):
            return list(self._records)
        return await super().load_section(section)


def _missing_env_record(profile_id: str) -> dict:
    return {
        "profile_id": profile_id,
        "command": "python",
        "args": [],
        "env_placeholders": {"K": "$TLDW_TEST_DEFINITELY_MISSING_VAR"},
        "discovery_snapshot": {
            "tools": [{"name": "a"}],
            "resources": [],
            "prompts": [],
        },
        "is_connected": False,
    }


class ProblemRecordsApp(ConsolidatedCSSApp):
    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self.unified_mcp_service = ProblemRecordsService(records)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_single_problem_row_is_preselected_on_load(monkeypatch):
    """F-054: when the first load surfaces exactly ONE problem server (the
    off/opt-in built-in doesn't count as a problem -- see is_off_opt_in;
    the LONE-row case is task-2240's own preselect, tested below), the
    workbench pre-selects it so the inspector opens on what's wrong and
    what you can do instead of dead space."""
    # Deterministic builtin state: the workbench's own get_cli_setting
    # (separate import from the inspector's fixture-patched one) returns
    # every key's default -- mcp.enabled=False, i.e. off/opt-in.
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ProblemRecordsApp([_missing_env_record("docs")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "local:docs"
        # Observable effect: the inspector is showing the problem server.
        state = app.query_one("#mcp-inspector-state", Static)
        assert "docs" in str(state.renderable)


@pytest.mark.asyncio
async def test_lone_off_builtin_row_is_preselected_on_fresh_install(monkeypatch):
    """task-2240: a fresh install's rail has exactly one row -- the
    off/opt-in built-in, which F-054's problem-only preselect deliberately
    excluded -- so the inspector stayed dead on the exact state every new
    user sees. A lone rail row is now pre-selected even when it isn't a
    "problem": the built-in's detail (what it is, why it's off, the Enable
    affordance) is informational, not alarmist."""
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ProblemRecordsApp([])
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "builtin:tldw_chatbook"
        # Observable effect: the inspector opens on the built-in's own
        # informational detail (task-2239's muted off/opt-in display
        # state), not the dead empty state...
        state = app.query_one("#mcp-inspector-state", Static)
        assert "tldw_chatbook (built-in)" in str(state.renderable)
        assert "Off (opt-in)" in str(state.renderable)
        # ...and the why-line explains the opt-in rather than filing a
        # setup defect ("Why · Not configured").
        message = app.query_one("#mcp-inspector-message", Static)
        assert "Not configured" not in str(message.renderable)
        assert "Off" in str(message.renderable)


@pytest.mark.asyncio
async def test_no_preselection_with_multiple_problems(monkeypatch):
    """F-054: the heuristic only fires for EXACTLY one candidate -- an
    ambiguous two-plus problems leaves the selection alone. (The
    zero-problem lone-row case is task-2240's preselect, covered above.)"""
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    multi = ProblemRecordsApp([_missing_env_record("docs"), _missing_env_record("web")])
    async with multi.run_test() as pilot:
        await pilot.pause()
        await multi.workers.wait_for_complete()
        await pilot.pause()
        workbench = multi.query_one(MCPWorkbench)
        assert workbench._selected_server_key is None


@pytest.mark.asyncio
async def test_cleared_selection_is_not_re_hijacked_by_later_resync(monkeypatch):
    """F-054: the pre-selection is a one-shot load default, not a standing
    policy -- once the user clears the selection ('All servers'), a later
    resync must not force the problem row back into focus."""
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ProblemRecordsApp([_missing_env_record("docs")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "local:docs"
        await workbench._select_server_key(None)
        await pilot.pause()
        assert workbench._selected_server_key is None
        # A further resync (e.g. a lifecycle completion) must not re-select.
        await workbench._sync_children()
        await pilot.pause()
        assert workbench._selected_server_key is None


@pytest.mark.asyncio
async def test_restored_all_servers_selection_wins_over_problem_preselect(monkeypatch):
    """F-054 restore path: a saved view state carrying an EXPLICIT
    `selected_server_key=None` ("All servers") must clear the lone-problem
    preselect the load heuristic just made. The heuristic runs before
    `_consume_pending_view_state()`, but the user's saved "no selection"
    from the previous session wins over the default."""

    class RestoreClearApp(ProblemRecordsApp):
        def compose(self) -> ComposeResult:
            workbench = MCPWorkbench(app_instance=self, id="mcp-workbench")
            # Saved state from a session where the user explicitly cleared
            # the selection ("All servers").
            workbench.set_initial_view_state(
                {
                    "mode": "servers",
                    "source": "local",
                    "selected_server_key": None,
                    "scope": "personal",
                    "scope_ref": None,
                }
            )
            yield workbench

    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = RestoreClearApp([_missing_env_record("docs")])
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key is None


@pytest.mark.asyncio
async def test_server_source_add_button_gated_when_mutations_unavailable():
    """T9: `service.available_actions()` not offering `external_server.create`
    (e.g. scope below team/org/system-admin) must disable the overview
    Add-server button in server source, with an explanatory tooltip.
    `FakeHubService.available_actions()` always returns `[]`, so switching
    to server source alone is enough to exercise the gated-off path.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-add-server", Button)
        assert button.disabled is True
        assert button.tooltip == "Requires team, org, or system-admin scope."


@pytest.mark.asyncio
async def test_import_button_gated_off_under_server_source():
    """I3: `MCPWorkbench._apply_import()` always saves to the LOCAL profile
    store (`save_local_profile()`, unconditionally) -- offering Import under
    server source would silently write somewhere invisible in the current
    view. Mirrors `_update_add_server_button()`'s disabled+tooltip gating
    pattern, but on source alone (no scope/target gating applies -- Import
    never touches server-side records at all).
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-import-server", Button)
        assert button.disabled is False
        assert button.tooltip == (
            "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
        )

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        assert button.disabled is True
        assert button.tooltip == (
            "Import creates LOCAL server profiles — switch Source to Local."
        )

        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        assert button.disabled is False
        assert button.tooltip == (
            "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
        )


@pytest.mark.asyncio
async def test_open_add_server_form_local_source_shows_profile_form():
    """T13: `MCPWorkbench.open_add_server_form()` is the `a` keybinding's
    entry point -- it never presses `#mcp-add-server`, so this drives the
    local-source add-form path directly rather than via a Button.Pressed."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True
        form = app.query_one(MCPProfileForm)
        assert form.is_edit is False


@pytest.mark.asyncio
async def test_open_add_server_form_gated_notifies_with_button_tooltip_copy():
    """T13: unlike a click on the already-disabled `#mcp-add-server` button
    (`test_server_source_add_button_gated_when_mutations_unavailable`), the
    `a` keybinding can reach this gate directly with no button to disable --
    it must notify with the SAME copy the button's own tooltip carries
    rather than silently doing nothing.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()

        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is False
        assert not app.query(MCPServerMutationsPanel)
        assert notifications
        message, severity = notifications[-1]
        assert message == "Requires team, org, or system-admin scope."
        assert severity == "warning"


class MutationsAvailableTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class MutationsAvailableTargetStore:
    def list_targets(self):
        return [MutationsAvailableTarget()]


class MutationsAvailableHubService:
    """Server-source fake whose `available_actions()` DOES offer the
    external_server.* set (team scope), with a mutable external-records
    list that `external_server.create` appends to -- the "available branch"
    counterpart of `FakeHubService`'s always-gated `[]`.
    """

    def __init__(self) -> None:
        self.target_store = MutationsAvailableTargetStore()
        self.context = UnifiedMCPContext(
            selected_source="server",
            selected_active_server_id="main",
            selected_scope="team",
            selected_section="external_servers",
        )
        self.external_records: list[dict[str, Any]] = []
        self.run_action_calls: list[tuple[str, dict[str, Any]]] = []

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        return {
            "source": "server",
            "section": "external_servers",
            "external_servers": [dict(r) for r in self.external_records],
        }

    def available_actions(self):
        return [
            {"name": "external_server.create", "label": "Create External Server"},
            {"name": "external_server.update", "label": "Update External Server"},
            {"name": "external_server.slots.list", "label": "List Credential Slots"},
        ]

    async def run_action(self, action_name, payload):
        self.run_action_calls.append((action_name, dict(payload)))
        if action_name == "external_server.create":
            self.external_records.append(
                {
                    "server_id": payload["server_id"],
                    "name": payload["name"],
                    "transport": payload.get("transport", "http"),
                    "enabled": payload.get("enabled", True),
                }
            )
            return {"server_id": payload["server_id"]}
        if action_name == "external_server.slots.list":
            return {"credential_slots": []}
        return {"ok": True}


class MutationsAvailableApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = MutationsAvailableHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_server_source_add_names_implicit_target_and_create_drills_into_new_record():
    """T9 review fix: the "available branch" of the Add-server gate.

    Add-server is only ever reachable from the overview, where nothing is
    selected -- `external_server.create` then attaches to whatever target
    the SERVICE context has active, invisibly. Two behaviors under test:

    1. The enabled button's tooltip names that implicit target ("Adds to
       server: Main Server.") so the attach point is never silent.
    2. After a successful create, the workbench drills into the new record
       (`server:main/<new_id>`): it appears in the collected snapshots and
       the mutation panel re-opens in edit mode for credential setup.
    """
    app = MutationsAvailableApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # Clear the mount-restored target selection: the scenario under test
        # is the overview with NO visible selection while the service still
        # remembers "main" as its active target.
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected(None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-add-server", Button)
        assert button.disabled is False
        assert "Main Server" in str(button.tooltip)

        await pilot.click("#mcp-add-server")
        await pilot.pause()
        panel = app.query_one(MCPServerMutationsPanel)
        assert not panel.is_edit
        app.query_one("#mcp-srv-id", Input).value = "docs"
        app.query_one("#mcp-srv-name", Input).value = "Docs"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        svc = app.unified_mcp_service
        assert (
            "external_server.create",
            {
                "server_id": "docs",
                "name": "Docs",
                "transport": "http",
                "config": {},
                "enabled": True,
            },
        ) in svc.run_action_calls
        # Post-create drill: the new record is selected...
        assert workbench.get_view_state()["selected_server_key"] == "server:main/docs"
        # ...its snapshot was actually collected (external-record loading is
        # gated on the ACTIVE target, not the UI selection)...
        assert any(
            snap.server_key == "server:main/docs" for snap in workbench._snapshots
        )
        # ...its credential slots were fetched, and the panel re-opened in
        # edit mode for credential setup.
        assert (
            "external_server.slots.list",
            {"server_id": "docs"},
        ) in svc.run_action_calls
        panel = app.query_one(MCPServerMutationsPanel)
        assert panel.is_edit
        assert app.query_one("#mcp-srv-name", Input).value == "Docs"


@pytest.mark.asyncio
async def test_mutations_panel_cancel_clears_selection_and_does_not_reopen_on_resync():
    """I2 regression: `show_server_mutations()` never updates `_detail_snapshot`,
    so Cancel used to restore whatever detail was last shown while
    `_selected_server_key` kept pointing at the external record it was
    hosting -- the very next `_sync_children()` (a background lifecycle
    completion, the `r` keybinding, a runtime-backend refresh) would read
    that stale selection and re-open the SAME mutations panel out of
    nowhere. Cancel must route through the same "clear selection, resync"
    path `ServerRowSelected(None)` uses.
    """
    app = MutationsAvailableApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected(None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Create an external record -- T9's post-create drill leaves the
        # mutations panel open in EDIT mode for it, i.e. exactly the
        # `server:T/R` selection scenario this bug needs.
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-srv-id", Input).value = "docs"
        app.query_one("#mcp-srv-name", Input).value = "Docs"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.get_view_state()["selected_server_key"] == "server:main/docs"
        panel = app.query_one(MCPServerMutationsPanel)
        assert panel.is_edit

        await pilot.click("#mcp-srv-cancel")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.get_view_state()["selected_server_key"] is None
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-overview").display is True
        assert canvas.query_one("#mcp-servers-detail").display is False
        assert not app.query(MCPServerMutationsPanel)

        # The next resync must not re-open the mutations panel now that the
        # selection that used to point at the external record is gone.
        await workbench._sync_children()
        await pilot.pause()
        assert not app.query(MCPServerMutationsPanel)
        assert canvas.query_one("#mcp-servers-overview").display is True


@pytest.mark.asyncio
async def test_mode_switch_shows_audit_canvas():
    """T7: "audit" was the last mode still on the generic phase-placeholder
    path -- it now hosts the real `MCPAuditMode` canvas (see the Task 7
    section below for its own coverage), same as tools (T5) and permissions
    (T6) before it. There is no longer any mode left on the placeholder
    path at all."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-audit"
        assert app.query_one("#mcp-mode-canvas-audit", MCPAuditMode) is not None


@pytest.mark.asyncio
async def test_rail_selection_drives_detail_and_view_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-detail").display
        state = app.query_one(MCPWorkbench).get_view_state()
        assert state["selected_server_key"] == "local:docs"
        assert state["mode"] == "servers"


@pytest.mark.asyncio
async def test_detail_disconnect_button_routes_through_start_lifecycle():
    """T7: the detail toolbar's Disconnect button (rendered because the
    seeded "docs" profile has `is_connected: True`) must route through the
    same `_start_lifecycle()` dispatch T5 wired for connect/test/refresh --
    not a separate, parallel code path."""
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-disconnect")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.disconnect_calls == ["docs"]


@pytest.mark.asyncio
async def test_builtin_flag_toggle_saves_setting_and_reloads_catalog(monkeypatch):
    """Task 10: toggling the built-in detail's "Enabled" Checkbox must call
    `save_setting_to_cli_config("mcp", "enabled", False)` (monkeypatched
    here, per the task interfaces) and then reload the catalog so the
    checkbox itself -- rebuilt fresh from the post-reload snapshot by
    `MCPServersMode.show_detail()` -- reflects the round trip rather than an
    optimistic local flip. A tiny in-memory `flags` dict stands in for the
    `[mcp]` config section on both the read (`get_cli_setting`) and write
    (`save_setting_to_cli_config`) sides so the reload assertion is a real
    signal instead of coincidentally matching a mocked return value.
    """
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as workbench_module

    flags: dict[str, Any] = {
        "enabled": True,
        "expose_tools": True,
        "expose_resources": True,
        "expose_prompts": True,
    }
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "mcp" and key in flags:
            return flags[key]
        return default

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        if section == "mcp":
            flags[key] = value
        return True

    monkeypatch.setattr(workbench_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        workbench_module, "save_setting_to_cli_config", fake_save_setting_to_cli_config
    )

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()
        checkbox = app.query_one("#mcp-builtin-enabled", Checkbox)
        assert checkbox.value is True

        await pilot.click("#mcp-builtin-enabled")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("mcp", "enabled", False) in save_calls
        # Reload picked up the write: the rebuilt checkbox now shows False,
        # not just an optimistic client-side flip.
        reloaded_checkbox = app.query_one("#mcp-builtin-enabled", Checkbox)
        assert reloaded_checkbox.value is False
        workbench = app.query_one(MCPWorkbench)
        builtin_snap = next(
            s for s in workbench._snapshots if s.server_key == "builtin:tldw_chatbook"
        )
        # task-2239: the disabled built-in reports the muted off/opt-in
        # display state, not the old NEEDS_SETUP alarm vocabulary.
        assert builtin_snap.state.value == "off_opt_in"


@pytest.mark.asyncio
async def test_builtin_expose_flag_toggle_saves_matching_key(monkeypatch):
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as workbench_module

    flags: dict[str, Any] = {
        "enabled": True,
        "expose_tools": True,
        "expose_resources": True,
        "expose_prompts": True,
    }
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "mcp" and key in flags:
            return flags[key]
        return default

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        if section == "mcp":
            flags[key] = value
        return True

    monkeypatch.setattr(workbench_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        workbench_module, "save_setting_to_cli_config", fake_save_setting_to_cli_config
    )

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()
        await pilot.click("#mcp-builtin-expose-resources")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("mcp", "expose_resources", False) in save_calls


@pytest.mark.asyncio
async def test_tool_gate_checkbox_toggle_saves_setting_and_reloads_catalog(monkeypatch):
    """task-3240 round-trip, sibling of test_builtin_flag_toggle_saves_
    setting_and_reloads_catalog: toggling a `[tools]`/`[console]` gate
    Checkbox must call `save_setting_to_cli_config(section, key, value)`
    with the checkbox's OWN section (not a hardcoded "mcp"), then reload so
    the checkbox -- rebuilt fresh from `all_tool_gates()` -- reflects the
    real write, not an optimistic local flip.

    Seam namespace (spec review, Minor 6): the write goes through
    `workbench_module.save_setting_to_cli_config` (same seam
    `_save_builtin_flag`'s own test patches), but the RELOAD reads through
    `all_tool_gates()`'s function-local `from ..config import
    get_cli_setting` -- that resolves `tldw_chatbook.config.get_cli_setting`
    at call time, a DIFFERENT name than `workbench_module`'s own imported
    one, so both must be patched here (backed by the SAME `flags` dict) or
    the reload assertion would read real disk config instead of this
    test's fake.

    Fix round 1 (Minor 3): bidirectional -- off->on->off -- rather than
    stopping after the first flip, so a fix that only round-trips ONE
    direction (e.g. an inverted default somewhere in the reload path)
    cannot pass by accident.

    Fix round 1 (Important 1) reorders this test: web_deep_search is a
    LOCAL-group dependent, so its checkbox renders `disabled=True` (a
    click is a no-op) while the master switch is off -- which it is by
    this test's own `flags` default. The master is toggled ON first here
    specifically to exercise web_deep_search's own click/toggle; it is
    toggled back off at the end, mirroring the original bidirectional
    coverage for the master switch itself.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY

    # TASK-14807 changed the product default to enabled. This test exercises
    # the explicit off -> on persistence path, so seed that starting state
    # instead of inheriting the new default from ``default``.
    flags: dict[tuple[str, str], Any] = {("console", "local_tools_enabled"): False}
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        return flags.get((section, key), default)

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        flags[(section, key)] = value
        return True

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        mcp_workbench_module,
        "save_setting_to_cli_config",
        fake_save_setting_to_cli_config,
    )

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()

        # The [console]-section master switch must save under ITS OWN
        # section too -- not "tools", proving `section` really is threaded
        # through end to end rather than hardcoded anywhere on the path.
        # Turned ON first (Important 1): web_deep_search is disabled below
        # it until this happens.
        master_checkbox = app.query_one("#mcp-gate-local_tools_enabled", Checkbox)
        assert master_checkbox.value is False
        await pilot.click("#mcp-gate-local_tools_enabled")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("console", "local_tools_enabled", True) in save_calls
        master_checkbox = app.query_one("#mcp-gate-local_tools_enabled", Checkbox)
        assert master_checkbox.value is True

        checkbox_id = f"#mcp-gate-{WEB_DEEP_SEARCH_GATE_KEY}"
        checkbox = app.query_one(checkbox_id, Checkbox)
        assert checkbox.value is False  # nothing overridden yet -> default off
        assert checkbox.disabled is False  # master is now on

        await pilot.click(checkbox_id)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("tools", WEB_DEEP_SEARCH_GATE_KEY, True) in save_calls
        reloaded_checkbox = app.query_one(checkbox_id, Checkbox)
        assert reloaded_checkbox.value is True

        # Bidirectional (Minor 3): flip it back OFF and confirm the reload
        # reflects that too -- not just the off->on direction.
        await pilot.click(checkbox_id)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("tools", WEB_DEEP_SEARCH_GATE_KEY, False) in save_calls
        reloaded_again = app.query_one(checkbox_id, Checkbox)
        assert reloaded_again.value is False

        # Bidirectional for the master switch too.
        await pilot.click("#mcp-gate-local_tools_enabled")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("console", "local_tools_enabled", False) in save_calls
        master_checkbox_again = app.query_one("#mcp-gate-local_tools_enabled", Checkbox)
        assert master_checkbox_again.value is False


def _fake_tool_gate_config_seam(monkeypatch):
    """Shared fake config/save seam for the fix-round-1 focus tests below.

    Same shape as `test_tool_gate_checkbox_toggle_saves_setting_and_
    reloads_catalog`'s own fakes: an in-memory `(section, key) -> value`
    dict backs BOTH the enumerator's read (`tldw_chatbook.config.
    get_cli_setting`, the seam `all_tool_gates()` actually resolves at
    call time) and the write (`workbench_module.save_setting_to_cli_
    config`), so a real save + real reload are exercised end to end.

    Returns:
        `(flags, save_calls)` -- the two lists/dicts the caller inspects.
    """
    import tldw_chatbook.config as config_module

    flags: dict[tuple[str, str], Any] = {}
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        return flags.get((section, key), default)

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        flags[(section, key)] = value
        return True

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        mcp_workbench_module,
        "save_setting_to_cli_config",
        fake_save_setting_to_cli_config,
    )
    return flags, save_calls


@pytest.mark.asyncio
async def test_focus_is_preserved_across_a_gate_toggle_save_and_resync(monkeypatch):
    """Fix round 1 (Critical 1). A gate toggle's save triggers a real,
    full resync (`MCPWorkbench._save_tool_gate` -> `_collect_snapshots` ->
    `_sync_children` -> `_show_selected_detail` -> `MCPServersMode.
    show_detail`), which destroys and remounts every toggle-group
    Checkbox. Before the fix, focus fell back to whatever DOM sibling
    happened to survive that remount -- for the FIRST gate checkbox
    (`_GATEABLE_BUILTINS[0]`), that is `#mcp-builtin-expose-prompts` (the
    LAST `[mcp]` toggle, immediately preceding it in the DOM): a live,
    actionable Checkbox belonging to a completely different settings
    group. Driven with a real keyboard Space (`pilot.press`), matching how
    the reviewer measured the regression -- not `pilot.click`.
    """
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    _fake_tool_gate_config_seam(monkeypatch)

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()

        first_gate_id = f"mcp-gate-{_GATEABLE_BUILTINS[0].gate_key}"
        checkbox = app.query_one(f"#{first_gate_id}", Checkbox)
        checkbox.focus()
        await pilot.pause()
        assert app.focused is not None and app.focused.id == first_gate_id

        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        # `Widget.focus()` only SCHEDULES the change (`app.call_later`) --
        # give it extra pumps to actually land before asserting on it.
        await pilot.pause()
        await pilot.pause()

        focused = app.focused
        assert focused is not None, "focus must not be dropped by the resync"
        assert focused.id == first_gate_id, (
            f"focus drifted to {focused.id!r} instead of staying on the "
            "toggled gate checkbox"
        )


@pytest.mark.asyncio
async def test_double_space_on_a_gate_checkbox_never_writes_an_mcp_key(monkeypatch):
    """Fix round 1 (Critical 1), the reviewer's exact repro: Space on a
    gate checkbox, then Space again. Before the fix, the SECOND Space hit
    whatever checkbox focus had drifted to post-resync -- for the first
    gate checkbox, `#mcp-builtin-expose-prompts` -- silently writing
    `[mcp] expose_prompts = false` instead of toggling the gate a second
    time. Asserts against the REAL persisted config (`flags`) and the
    save-call list: only the gate's own `[tools]` key is ever written,
    twice (toggle then untoggle), and no `"mcp"`-section key is written at
    all.
    """
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    flags, save_calls = _fake_tool_gate_config_seam(monkeypatch)

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()

        gate_key = _GATEABLE_BUILTINS[0].gate_key
        first_gate_id = f"mcp-gate-{gate_key}"
        checkbox = app.query_one(f"#{first_gate_id}", Checkbox)
        checkbox.focus()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()

        assert save_calls == [
            ("tools", gate_key, True),
            ("tools", gate_key, False),
        ], save_calls
        assert all(section != "mcp" for section, _, _ in save_calls), (
            "a second Space wrote an unrelated [mcp] key -- focus drifted "
            f"off the gate checkbox: {save_calls}"
        )
        assert flags[("tools", gate_key)] is False


@pytest.mark.asyncio
async def test_restore_tolerates_legacy_and_garbage_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # legacy panel shape
        workbench.set_initial_view_state(
            {"selected_source": "local", "selected_section": "inventory"}
        )
        await pilot.pause()
        assert workbench.active_mode == "servers"
        # garbage
        workbench.set_initial_view_state({"mode": "nonsense", "bogus": 1})
        await pilot.pause()
        assert workbench.active_mode == "servers"


@pytest.mark.asyncio
async def test_scope_change_and_restore_round_trip():
    """Finding 1: scope/scope_ref must be tracked, not hardcoded to personal/None."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)

        # Before any scope change, view state reports the personal default.
        assert workbench.get_view_state()["scope"] == "personal"
        assert workbench.get_view_state()["scope_ref"] is None

        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        state = workbench.get_view_state()
        assert state["scope"] == "team"
        assert state["scope_ref"] == "21"

        # A fresh restore (e.g. re-entering the destination) must be able to
        # bring the same scope back.
        workbench.set_initial_view_state(
            {"mode": "servers", "source": "local", "scope": "team", "scope_ref": "21"}
        )
        await pilot.pause()
        restored = workbench.get_view_state()
        assert restored["scope"] == "team"
        assert restored["scope_ref"] == "21"


def test_active_mode_property_rejects_direct_assignment():
    """Finding 4: active_mode is read-only; set_mode() is the only mutator."""
    app = WorkbenchApp()
    workbench = MCPWorkbench(app_instance=app)
    assert workbench.active_mode == "servers"
    with pytest.raises(AttributeError):
        workbench.active_mode = "tools"


def test_set_mode_defers_async_workers_as_callables():
    """Exclusive cancellation must not strand already-created coroutines."""
    switcher = SimpleNamespace(current=None)
    queued: list[tuple[Any, dict[str, Any]]] = []
    posted: list[Any] = []

    async def disarm_canvas_delete() -> None:
        return None

    async def clear_tool_view() -> None:
        return None

    workbench = SimpleNamespace(
        _active_mode="servers",
        ModeChanged=lambda mode: SimpleNamespace(mode=mode),
        # task-2901: set_mode probes for the target canvas before touching
        # the switcher (deferred canvases stash instead); a non-empty result
        # models "canvas mounted".
        query=lambda _selector: [object()],
        query_one=lambda _widget_type: switcher,
        post_message=lambda message: posted.append(message),
        run_worker=lambda work, **kwargs: queued.append((work, kwargs)),
        _disarm_canvas_delete=disarm_canvas_delete,
        _clear_tool_view=clear_tool_view,
    )

    MCPWorkbench.set_mode(workbench, "tools")

    assert switcher.current == "mcp-mode-canvas-tools"
    assert [message.mode for message in posted] == ["tools"]
    assert [work for work, _kwargs in queued] == [
        disarm_canvas_delete,
        clear_tool_view,
    ]
    assert all(callable(work) for work, _kwargs in queued)


@pytest.mark.asyncio
async def test_set_initial_view_state_during_inflight_reload_applies_pending_state_once():
    """Finding 2: a restore requested while a reload is in flight must not race it.

    A fully black-box reproduction of the race (calling set_initial_view_state
    while on_mount's `await self.reload()` is genuinely suspended mid-await)
    isn't reliably forceable through the public API once `app.run_test()` has
    already settled the initial mount. Instead this asserts the `_reloading`
    guard's contract directly: while a reload is marked in flight, a restore
    request is stashed but not applied; the in-flight reload's own
    end-of-method consumption applies it exactly once, and a repeat
    consumption attempt is a no-op.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)

        apply_calls: list[dict] = []
        original_apply = workbench._apply_view_state

        async def tracking_apply(state):
            apply_calls.append(dict(state))
            await original_apply(state)

        workbench._apply_view_state = tracking_apply

        # Simulate set_initial_view_state() arriving while some other reload
        # (e.g. the destination screen refreshing right after mount) is
        # already in flight.
        workbench._reloading = True
        state = {
            "mode": "servers",
            "source": "local",
            "scope": "team",
            "scope_ref": "21",
        }
        workbench.set_initial_view_state(state)
        await pilot.pause()

        # No worker should run while a reload is in flight -- the state is
        # only stashed, not applied.
        assert apply_calls == []
        assert workbench._pending_view_state == state

        # The in-flight reload finishes and, per reload()'s own contract,
        # consumes the pending state exactly once.
        workbench._reloading = False
        await workbench.reload()
        await pilot.pause()

        assert len(apply_calls) == 1
        assert workbench._pending_view_state is None
        restored = workbench.get_view_state()
        assert restored["scope"] == "team"
        assert restored["scope_ref"] == "21"

        # A second consumption attempt must not re-apply.
        await workbench._consume_pending_view_state()
        assert len(apply_calls) == 1


class _StubApp:
    unified_mcp_service = None


def test_screen_hosts_workbench_with_mode_action_and_tolerant_restore():
    screen = MCPScreen(_StubApp())
    # New surface: workbench host + mode action (old screen has mcp_panel, no workbench).
    assert hasattr(screen, "workbench")
    assert not hasattr(screen, "mcp_panel")
    assert callable(getattr(screen, "action_mcp_mode", None))
    # Never crashes on legacy shape, garbage, or empty state pre-mount.
    screen.restore_state({"unified_mcp_view_state": {"selected_source": "server"}})
    screen.restore_state({"mcp_hub_view_state": {"mode": "tools"}})
    screen.restore_state({})
    state = screen.save_state()
    assert isinstance(state, dict)


def test_mcp_hub_modes_registry_is_complete():
    assert list(MCP_HUB_MODES) == ["servers", "tools", "permissions", "audit"]
    for spec in MCP_HUB_MODES.values():
        assert spec["label"] and spec["button_id"].startswith("mcp-mode-")


@pytest.mark.asyncio
async def test_workbench_panes_have_nonzero_geometry():
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for selector in ("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"):
            widget = app.query_one(selector)
            assert widget.size.width > 0, f"{selector} has zero width"
            assert widget.size.height > 0, f"{selector} has zero height"
        table = app.query_one("#mcp-servers-table")
        assert table.size.height > 0, "servers table clipped to zero height"

        # T13: the Add-server form container (`#mcp-servers-form`, styled by
        # the Phase 2 CSS block appended to _agentic_terminal.tcss) must also
        # render with real geometry once shown -- not just the always-visible
        # overview/detail panes checked above.
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        form_container = app.query_one("#mcp-servers-form")
        assert form_container.display is True
        assert form_container.size.width > 0, "add-server form container has zero width"
        assert form_container.size.height > 0, (
            "add-server form container has zero height"
        )


# -- C1: scope-event storm on Server source --------------------------------
#
# Textual 8.2.7 posts a `Select.Changed` for a `Select`'s own constructor
# value as part of mounting it. `MCPRail`'s source select guards this
# mount-echo (`on_select_changed` only forwards when the new value differs
# from `self.source`); the scope and scope-ref selects did not. Because
# `MCPWorkbench._sync_children()` recomposes the rail (new Select instances,
# new mount echoes) and `on_mcp_rail_scope_changed()` used to call
# `_sync_children()` unconditionally, an unguarded echo self-sustained an
# unbounded recompose storm: recompose -> echo -> ScopeChanged ->
# service.select_scope() + `_sync_children()` -> recompose -> echo -> ...


class ScopeTrackingTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class ScopeTrackingTargetStore:
    def list_targets(self):
        return [ScopeTrackingTarget()]


class ScopeTrackingHubService:
    """Like `FakeHubService`, but records every `select_scope()` call."""

    def __init__(self, *, selected_scope: str) -> None:
        self.target_store = ScopeTrackingTargetStore()
        self.context = UnifiedMCPContext(
            selected_source="server", selected_scope=selected_scope
        )
        self.select_scope_calls: list[tuple[object, object]] = []

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        self.select_scope_calls.append((scope, scope_ref))
        self.context = replace(
            self.context, selected_scope=scope, selected_scope_ref=scope_ref
        )
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        return {"external_servers": [], "source": "server", "section": section}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class ScopeTrackingApp(ConsolidatedCSSApp):
    def __init__(self, *, selected_scope: str) -> None:
        super().__init__()
        self.unified_mcp_service = ScopeTrackingHubService(
            selected_scope=selected_scope
        )

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_scope", ["team", "personal"])
async def test_server_source_scope_mount_does_not_storm_select_scope_calls(
    selected_scope,
):
    """C1 regression: mounting on Server source must not spam select_scope().

    Covers both halves called out in review: a restored scope outside
    Phase 1's Personal-only rail options ("team", which the rail's display
    clamps to "personal") and the in-options default ("personal", which
    needs no clamp but was never guarded either).
    """
    app = ScopeTrackingApp(selected_scope=selected_scope)
    async with app.run_test() as pilot:
        svc = app.unified_mcp_service
        counts = []
        for _ in range(8):
            await pilot.pause(0.05)
            counts.append(len(svc.select_scope_calls))
        assert all(c == 0 for c in counts), (
            f"select_scope storm at mount (scope={selected_scope!r}): "
            f"{counts} calls={svc.select_scope_calls}"
        )


@pytest.mark.asyncio
async def test_workbench_dedupes_identical_scope_changed_events():
    """C1 fix (b), defense in depth: a repeat ScopeChanged with the same
    (scope, scope_ref) as the workbench's already-tracked state must not
    call service.select_scope() again.
    """
    app = ScopeTrackingApp(selected_scope="personal")
    async with app.run_test() as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        svc = app.unified_mcp_service
        svc.select_scope_calls.clear()

        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()

        assert svc.select_scope_calls == [("team", "21")]


# -- T7 carry-over: scope_ref key-absent vs key-present-None ----------------


@pytest.mark.asyncio
async def test_apply_view_state_scope_ref_key_absent_keeps_existing_value():
    """A restore blob with no scope_ref/selected_scope_ref key at all must
    not clobber the currently tracked scope_ref."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"

        workbench.set_initial_view_state(
            {"mode": "servers", "source": "local", "scope": "team"}
        )
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"


@pytest.mark.asyncio
async def test_apply_view_state_scope_ref_present_none_clears_existing_value():
    """An explicit `scope_ref: None` key must clear the stale scope_ref
    rather than being treated the same as the key being absent."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"

        workbench.set_initial_view_state(
            {"mode": "servers", "source": "local", "scope": "team", "scope_ref": None}
        )
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] is None


# -- P1: Advanced > External Servers must not leak unredacted secrets -------
#
# `render_external_servers_section()` (legacy renderer, frozen) keys local
# records by "name", which local profile dicts never have (they use
# "profile_id") -- so it falls back to printing the FULL RAW DICT per entry,
# CLI args and env values included. `_AdvancedSectionShim.load_section()` is
# the seam this surface owns: it must redact each record (via
# `redact_mapping`/`redact_args`) before the payload ever reaches the frozen
# renderer, on both the bare-list local-source normalization path and any
# dict payload that already carries an `external_servers` list.


class SecretLeakHubService:
    """Local-source service whose `external_servers` records carry a
    secret-looking CLI arg (`--api-key sk-qa-test-redact-0001`), mirroring
    what QA saw leak through the Advanced pane."""

    SECRET_VALUE = "sk-qa-test-redact-0001"

    def __init__(self) -> None:
        self.context = UnifiedMCPContext(
            selected_source="local", selected_section="overview"
        )

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "external_servers":
            return [
                {
                    "profile_id": "leaky",
                    "command": "npx",
                    "args": ["--api-key", self.SECRET_VALUE, "--verbose"],
                    "env_placeholders": {},
                    "discovery_snapshot": {
                        "tools": [{"name": "a"}],
                        "resources": [],
                        "prompts": [],
                    },
                    "is_connected": True,
                }
            ]
        return {"source": "local", "section": effective_section}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class SecretLeakApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = SecretLeakHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_advanced_external_servers_section_redacts_secret_args():
    app = SecretLeakApp()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "external_servers"
        await pilot.pause()
        await pilot.pause()

        rendered = str(app.query_one("#mcp-adv-content", Static).renderable)
        assert SecretLeakHubService.SECRET_VALUE not in rendered, (
            f"secret arg leaked into Advanced > External Servers: {rendered!r}"
        )
        # Non-secret fields must still render -- this isn't just an empty pane.
        assert "npx" in rendered


# -- T5: connect/test/refresh lifecycle wiring, in-flight CHECKING, cancel --


class LifecycleFakeHubService(FakeHubService):
    """Like `FakeHubService`, but wires the typed T2 lifecycle methods and
    records every call so the workbench's dispatch can be asserted on."""

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle_calls: list[tuple[str, str]] = []
        self.connect_gate: asyncio.Event | None = None

    async def load_section(self, section=None):
        # Same shape as FakeHubService.load_section(), except the docs
        # profile fixture is disconnected with no discovery snapshot -- so it
        # derives NEEDS_SETUP (DISCOVERY_NOT_RUN) rather than READY, whose
        # action set is (CONNECT, VALIDATE, VIEW_DETAILS): both lifecycle
        # buttons the tests below click render enabled. (STALE via
        # RUNTIME_UNAVAILABLE would also wire CONNECT, but its action set
        # offers no VALIDATE button at all -- see REASON_TO_ACTIONS in
        # readiness.py.)
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": None,
                        "is_connected": False,
                    }
                ]
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }

    async def local_external_catalog(self):
        return await self.load_section("external_servers")

    async def connect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("connect", profile_id))
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        return {
            "server_id": profile_id,
            "tools": [{"name": "a"}],
            "resources": [],
            "prompts": [],
        }

    async def test_local_profile(self, profile_id):
        self.lifecycle_calls.append(("test", profile_id))
        return {
            "ok": True,
            "profile_id": profile_id,
            "tools": 1,
            "resources": 0,
            "prompts": 0,
        }

    async def refresh_local_profile(self, profile_id):
        self.lifecycle_calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}

    async def disconnect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("disconnect", profile_id))
        return True


class LifecycleApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = LifecycleFakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_connect_action_runs_lifecycle_and_notifies():
    app = LifecycleApp()
    # Default 80x24 leaves the inspector pane's action buttons out of the
    # visible region (rail+canvas+inspector min-widths sum to 90 > 80) --
    # see test_workbench_panes_have_nonzero_geometry for the same fix.
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # select docs (local profile, never discovered -> NEEDS_SETUP -> CONNECT wired)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-connect")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("connect", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_in_flight_shows_checking_and_cancel_then_completes():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert list(app.query("#mcp-inspector-cancel"))
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("connect", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_in_flight_checking_message_includes_time_bound(monkeypatch):
    """T7 (P3 UX batch): the CHECKING message ("Working — <action>…") gave
    no indication of how long an in-flight lifecycle op might sit there
    before a user gave up on it. `_display_snapshot()` now appends a time
    bound read from `[mcp] hub_lifecycle_timeout_seconds` (the same setting
    `UnifiedMCPControlPlaneService._lifecycle_timeout()` uses to actually
    enforce the timeout) -- default 45s, formatted as an int.

    Monkeypatches `get_cli_setting` to return each call's own default
    (i.e. "nothing configured") so this assertion can't accidentally pass
    or fail depending on a developer's real `~/.config/tldw_cli/config.toml`.
    """
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert "(up to" in selected.message
        assert "(up to 45s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_in_flight_checking_message_time_bound_honors_config_override(
    monkeypatch,
):
    """The time bound is read live from config, not hardcoded -- a
    non-default `hub_lifecycle_timeout_seconds` must show up in the CHECKING
    copy verbatim."""

    def fake_get_cli_setting(section, key=None, default=None):
        if (section, key) == ("mcp", "hub_lifecycle_timeout_seconds"):
            return 12
        return default

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get_cli_setting)

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert "(up to 12s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_in_flight_checking_message_time_bound_survives_malformed_config(
    monkeypatch,
):
    """A non-numeric `hub_lifecycle_timeout_seconds` (e.g. a user fat-fingering
    "soon" into config.toml) must not crash the CHECKING render path --
    `_display_snapshot()` should fall back to the same 45s default that
    `UnifiedMCPControlPlaneService._lifecycle_timeout()` falls back to on the
    same malformed input, rather than letting `float()` raise ValueError
    straight out of a render call."""

    def fake_get_cli_setting(section, key=None, default=None):
        if (section, key) == ("mcp", "hub_lifecycle_timeout_seconds"):
            return "soon"
        return default

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get_cli_setting)

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert "(up to 45s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_cancel_requested_cancels_worker():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()  # never set -> hangs
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        workbench.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs")
        )
        await pilot.pause()
        assert "local:docs" not in workbench._in_flight


def _capture_notifications(app: App) -> list[tuple[str, str]]:
    """Shadow `app.notify` with a recorder; returns the (message, severity)
    list it appends to. The workbench always notifies via `self.app.notify`,
    so an instance-level shadow intercepts every toast."""
    notifications: list[tuple[str, str]] = []

    def recording_notify(message, *, title="", severity="information", **kwargs):
        notifications.append((str(message), severity))

    app.notify = recording_notify
    return notifications


async def _clear_initial_preselection(app: App, pilot) -> None:
    """F-054: the workbench pre-selects a lone problem row on first load
    (`ProfileFormHubService`'s docs profile is AUTH_MISSING, so exactly one
    problem exists) -- form-flow tests drive the OVERVIEW, so clear that
    heuristic selection first, via the same path the rail's 'All servers'
    row drives."""
    workbench = app.query_one(MCPWorkbench)
    if workbench._selected_server_key is not None:
        await workbench._select_server_key(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_validate_action_runs_test_lifecycle_and_notifies_int_tool_count():
    """VALIDATE dispatch through the real click path, and the
    `_lifecycle_tool_count` int-shape branch: `test_local_profile` returns
    `"tools": 1` (a count, not a list), and the success toast must say
    "1 tool" (singular) from that int."""
    app = LifecycleApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        # select docs (never discovered -> NEEDS_SETUP -> VALIDATE wired)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-validate")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("test", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight
        successes = [msg for msg, severity in notifications if severity != "error"]
        assert any("docs" in msg and "1 tool" in msg for msg in successes), (
            f"expected an int-derived '1 tool' success toast, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_refresh_lifecycle_dispatches_refresh_method():
    """REFRESH_DISCOVERY's verb mapping ("refresh" -> refresh_local_profile)
    through `_start_lifecycle` -- the third dispatch-table entry the other
    lifecycle tests (all "connect"/"test") leave uncovered."""
    app = LifecycleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "refresh")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("refresh", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_cancel_after_natural_completion_does_not_toast_cancelled():
    """A stale CancelRequested arriving after the operation already finished
    (and popped itself from `_in_flight`) must be a silent no-op -- toasting
    "Cancelled." for something that actually completed would be a lie."""
    app = LifecycleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle(
            "local:docs", "docs", "connect"
        )  # no gate -> completes
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "local:docs" not in workbench._in_flight
        notifications = _capture_notifications(app)
        workbench.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs")
        )
        await pilot.pause()
        assert notifications == [], (
            f"stale cancel must not toast, got: {notifications!r}"
        )


# -- T6: local profile add/edit form hosting + save wiring ------------------


class ProfileFormHubService(FakeHubService):
    """Like `FakeHubService`, but wires `save_local_profile()` -- configurable
    to raise a store-shaped `ValueError` on the next call (mirrors
    `LocalMCPStore`'s "cannot be stored as a literal" copy) or to succeed and
    grow the catalog, so the reload after a successful save has something new
    to show.
    """

    def __init__(self, *, fail_next: bool = False) -> None:
        super().__init__()
        self.save_calls: list[dict] = []
        self._fail_next = fail_next
        # When set, save_local_profile() records its call then blocks on
        # this gate -- lets the double-submit test hold a save in flight.
        self.save_gate: asyncio.Event | None = None
        self.delete_calls: list[str] = []
        # When set, delete_local_profile() records its call then blocks on
        # this gate -- mirrors save_gate, for the double-confirm test.
        self.delete_gate: asyncio.Event | None = None
        self._records: list[dict] = [
            {
                "profile_id": "docs",
                "command": "python",
                "args": [],
                # An unresolved placeholder derives AUTH_MISSING (see
                # local_profile_readiness()), whose action set is
                # (OPEN_CREDENTIALS, EDIT_CONFIG, VIEW_DETAILS) -- unlike a
                # clean READY profile, which offers no EDIT_CONFIG button at
                # all. Needed so the edit-config test below has a button to
                # click.
                "env_placeholders": {"API_KEY": "$MCP_TEST_MISSING_VAR_XYZ"},
                "env_literals": {},
                "discovery_snapshot": {
                    "tools": [{"name": "a"}],
                    "resources": [],
                    "prompts": [],
                },
                "is_connected": True,
            }
        ]

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return list(self._records)
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }

    async def local_external_catalog(self):
        return list(self._records)

    async def save_local_profile(self, payload):
        self.save_calls.append(dict(payload))
        if self._fail_next:
            self._fail_next = False
            raise ValueError(
                "Secret-bearing env key 'API_KEY' cannot be stored as a literal"
            )
        if self.save_gate is not None:
            await self.save_gate.wait()
        self._records.append(dict(payload))
        return dict(payload)

    async def delete_local_profile(self, profile_id):
        self.delete_calls.append(profile_id)
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        self._records = [r for r in self._records if r.get("profile_id") != profile_id]
        return True


class ProfileFormApp(ConsolidatedCSSApp):
    def __init__(self, *, fail_next: bool = False) -> None:
        super().__init__()
        self.unified_mcp_service = ProfileFormHubService(fail_next=fail_next)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_add_server_requested_shows_add_mode_form():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert not form.is_edit
        assert app.query_one("#mcp-servers-form").display
        assert not app.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_submit_with_service_value_error_renders_store_copy_in_form():
    app = ProfileFormApp(fail_next=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "leaky"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one(
            "#mcp-form-env", TextArea
        ).text = "API_KEY=raw-literal-not-a-placeholder"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        error_text = str(app.query_one("#mcp-form-error", Static).renderable)
        assert "cannot be stored" in error_text
        # Form stays open on failure -- the user can fix the value and retry.
        assert app.query_one("#mcp-servers-form").display
        assert app.unified_mcp_service.save_calls == [
            {
                "profile_id": "leaky",
                "command": "npx",
                "args": [],
                "env_placeholders": {},
                "env_literals": {"API_KEY": "raw-literal-not-a-placeholder"},
            }
        ]


@pytest.mark.asyncio
async def test_submit_success_hides_form_notifies_and_reloads_catalog():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "newprofile"
        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls[-1]["profile_id"] == "newprofile"
        assert any("newprofile" in msg for msg, _ in notifications)

        # Reload actually picked up the new record -- the overview table now
        # shows both the pre-seeded "docs" profile and the new one.
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:newprofile" in keys


@pytest.mark.asyncio
async def test_submit_success_with_secret_shaped_arg_toasts_warning():
    """I4 follow-up (final-review caveat): the in-form
    `#mcp-form-args-warning` Static is unmounted by `hide_form()` sub-second
    after a SUCCESSFUL save, so on exactly the path where the secret got
    persisted the user never saw the warning. The form now carries the
    computed warning on `SubmitRequested`, and the workbench's save-success
    path re-surfaces it as a warning toast alongside the "Saved {id}."
    notify.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "leakyargs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = "-y\nsk-1234567890abcdef"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Save succeeded through the real flow: form gone, record saved.
        assert not app.query_one("#mcp-servers-form").display
        assert app.unified_mcp_service.save_calls[-1]["profile_id"] == "leakyargs"
        assert any("leakyargs" in msg for msg, _ in notifications)
        # ...and the secret-lint warning survived the form's unmount as a toast.
        warnings = [
            msg
            for msg, severity in notifications
            if severity == "warning" and "visible in process listings" in msg
        ]
        assert warnings, (
            f"expected a secret-lint warning toast on save success, "
            f"got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_submit_success_with_clean_args_toasts_no_warning():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "cleanargs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one(
            "#mcp-form-args", TextArea
        ).text = "-y\n@modelcontextprotocol/server-filesystem"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert any("cleanargs" in msg for msg, _ in notifications)
        assert not any(severity == "warning" for _, severity in notifications), (
            f"clean args must not produce a warning toast, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_cancelled_hides_form_without_saving():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        await pilot.click("#mcp-add-server")
        for _ in range(200):
            cancel_buttons = list(app.query("#mcp-form-cancel"))
            if (
                cancel_buttons
                and cancel_buttons[0].region.width > 0
                and cancel_buttons[0].region.height > 0
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("MCP profile-form Cancel did not render")
        await pilot.click("#mcp-form-cancel")
        await pilot.pause()
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls == []


@pytest.mark.asyncio
async def test_reload_while_add_form_open_does_not_stack_overview_and_form():
    """I1 regression (review probe): a background resync -- here `reload()`,
    standing in for the `r` keybinding or a runtime-backend refresh --
    must never re-show the overview UNDERNEATH a still-open add/edit form.
    Typed input must survive a resync: the form is only ever hidden/
    remounted by an explicit close (Save/Cancel), never by a passive
    resync.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-command", Input).value = "still-typing"
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True
        assert canvas.query_one("#mcp-servers-overview").display is False

        workbench = app.query_one(MCPWorkbench)
        await workbench.reload()
        await pilot.pause()

        form_display = canvas.query_one("#mcp-servers-form").display
        overview_display = canvas.query_one("#mcp-servers-overview").display
        detail_display = canvas.query_one("#mcp-servers-detail").display
        assert form_display is True
        assert not (form_display and overview_display), (
            "overview and form are both visible after reload with form open"
        )
        assert not (form_display and detail_display), (
            "detail and form are both visible after reload with form open"
        )
        # Not just "not stacked" -- the SAME form instance, with the typed
        # value intact, proving the resync never hid/remounted it.
        assert app.query_one("#mcp-form-command", Input).value == "still-typing"


@pytest.mark.asyncio
async def test_rail_selection_while_add_form_open_does_not_stack_detail_and_form():
    """I1 regression (review probe): selecting a different rail row while a
    LOCAL add/edit form is open must keep the form on screen rather than
    stack the detail pane underneath it. Selection interaction decision
    (documented in the final-review-fixes report): the underlying
    `_selected_server_key`/`_detail_snapshot` state DOES still update in the
    background here -- only the container-visibility flip is suppressed --
    so once the form closes (Save/Cancel) the view reflects the latest
    selection rather than snapping back to whatever was selected before the
    form opened. That is a deliberate, minimal-scope consequence of this
    fix, not a separate bug.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        app.query_one("#mcp-form-command", Input).value = "still-typing"
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True

        await workbench._select_server_key("local:docs")
        await pilot.pause()

        form_display = canvas.query_one("#mcp-servers-form").display
        detail_display = canvas.query_one("#mcp-servers-detail").display
        overview_display = canvas.query_one("#mcp-servers-overview").display
        assert form_display is True
        assert not (form_display and detail_display), (
            "detail and form are both visible after rail selection with form open"
        )
        assert not (form_display and overview_display), (
            "overview and form are both visible after rail selection with form open"
        )
        assert app.query_one("#mcp-form-command", Input).value == "still-typing"
        # The background selection DID update (see docstring) -- it just
        # isn't rendered while the form has the floor.
        assert workbench.get_view_state()["selected_server_key"] == "local:docs"


@pytest.mark.asyncio
async def test_edit_config_hub_action_opens_prefilled_form_for_local_profile():
    """EDIT_CONFIG on a local-source snapshot (Task 6 wiring of the
    previously-disabled inspector action) opens the form pre-filled from the
    freshly loaded catalog record for that profile_id -- not just the
    readiness snapshot, which doesn't carry command/args/env.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        assert list(app.query("#mcp-inspector-action-edit_config"))
        edit_button = app.query_one("#mcp-inspector-action-edit_config")
        assert not edit_button.disabled
        await pilot.click("#mcp-inspector-action-edit_config")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert form.is_edit
        assert app.query_one("#mcp-form-id", Input).value == "docs"
        assert app.query_one("#mcp-form-id", Input).disabled
        assert app.query_one("#mcp-form-command", Input).value == "python"


@pytest.mark.asyncio
async def test_detail_edit_button_opens_prefilled_form_for_local_profile():
    """T7: the detail toolbar's Edit button must reuse the exact same
    EDIT_CONFIG path as the inspector's own action button (Task 6) -- not a
    parallel implementation that could drift from it or skip the catalog
    record lookup."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-edit")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert form.is_edit
        assert app.query_one("#mcp-form-id", Input).value == "docs"
        assert app.query_one("#mcp-form-command", Input).value == "python"


@pytest.mark.asyncio
async def test_double_submit_dispatches_exactly_one_save():
    """Review fix (Important #1): a second Save while a save is in flight
    must NOT dispatch a second worker. The old handler ran every submit
    through `run_worker(..., exclusive=True)`, so a second click CANCELLED
    the in-flight save mid-write and started a fresh one. Two synchronous
    `Button.press()` calls reproduce it deterministically (pilot.click's
    pump timing masks the race): both `Pressed` messages queue before the
    first handler can disable the button, so two `SubmitRequested` reach
    the workbench -- the in-flight guard must swallow the second with a
    warning toast, leaving exactly one `save_local_profile` call.
    """
    app = ProfileFormApp()
    app.unified_mcp_service.save_gate = asyncio.Event()  # hold save in flight
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _clear_initial_preselection(app, pilot)
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "newprofile"
        app.query_one("#mcp-form-command", Input).value = "npx"
        save_button = app.query_one("#mcp-form-save", Button)
        save_button.press()
        save_button.press()
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second submit must toast a warning, got: {notifications!r}"
        app.unified_mcp_service.save_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.save_calls) == 1, (
            f"expected exactly one save call, got: "
            f"{app.unified_mcp_service.save_calls!r}"
        )
        # The single (uncancelled) save completed: form hidden, record saved.
        assert not app.query_one("#mcp-servers-form").display
        workbench = app.query_one(MCPWorkbench)
        assert "local:newprofile" in {s.server_key for s in workbench._snapshots}


@pytest.mark.asyncio
async def test_save_value_error_with_form_gone_notifies_instead_of_vanishing():
    """Review fix (Important #2): if the form is no longer mounted when the
    service raises ValueError (user cancelled while the save worker was in
    flight), the validation failure must surface as an error toast -- never
    disappear silently."""
    app = ProfileFormApp(fail_next=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        # No form mounted -- drive the worker coroutine directly.
        await workbench._save_local_profile(
            {
                "profile_id": "leaky",
                "command": "npx",
                "args": [],
                "env_placeholders": {},
                "env_literals": {"API_KEY": "raw-literal"},
            }
        )
        await pilot.pause()
        assert any(
            "cannot be stored" in msg and severity == "error"
            for msg, severity in notifications
        ), f"ValueError with no form must notify, got: {notifications!r}"


# -- T7: DeleteConfirmed wiring -----------------------------------------------


@pytest.mark.asyncio
async def test_delete_confirmed_deletes_profile_clears_selection_and_notifies():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench.get_view_state()["selected_server_key"] == "local:docs"

        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        await pilot.click("#mcp-detail-delete-confirm")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.delete_calls == ["docs"]
        assert workbench.get_view_state()["selected_server_key"] is None
        assert any("Deleted" in msg and "docs" in msg for msg, _ in notifications), (
            f"expected a 'Deleted docs.' toast, got: {notifications!r}"
        )
        assert app.query_one("#mcp-servers-overview").display
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:docs" not in keys


@pytest.mark.asyncio
async def test_double_delete_confirm_dispatches_exactly_one_delete():
    """Mirrors test_double_submit_dispatches_exactly_one_save: a second
    `DeleteConfirmed` arriving while a delete worker is already in flight
    must not cancel/duplicate it -- `_profile_delete_in_flight` swallows the
    repeat with a warning toast, leaving exactly one delete call."""
    app = ProfileFormApp()
    app.unified_mcp_service.delete_gate = asyncio.Event()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("local:docs")
        )
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("local:docs")
        )
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second confirm must toast a warning, got: {notifications!r}"
        app.unified_mcp_service.delete_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.delete_calls == ["docs"]


@pytest.mark.asyncio
async def test_mode_round_trip_disarms_pending_delete_confirmation():
    """Review fix (Important): the arm-then-confirm contract says "any other
    interaction disarms" -- switching modes is such an interaction. Before
    the fix, arming Delete, leaving for Tools, and coming back to Servers
    still rendered the live "Confirm delete" button (the ContentSwitcher
    hides the canvas without unmounting it, so nothing reset the arm state).
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))  # armed

        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert list(app.query("#mcp-detail-delete")), (
            "mode round-trip must disarm back to the plain Delete button"
        )
        assert not list(app.query("#mcp-detail-delete-confirm"))
        assert app.unified_mcp_service.delete_calls == []


@pytest.mark.asyncio
async def test_selecting_different_profile_while_armed_disarms():
    """Reviewer's Minor: selecting a DIFFERENT local profile via the rail
    while a delete confirmation is armed must disarm -- otherwise the live
    "Confirm delete" button silently retargets whatever got selected next.
    Already handled by `show_detail()`'s unconditional arm-state reset;
    this locks the behavior in."""
    app = ProfileFormApp()
    app.unified_mcp_service._records.append(
        {
            "profile_id": "web",
            "command": "npx",
            "args": [],
            "env_placeholders": {},
            "env_literals": {},
            "discovery_snapshot": {"tools": [], "resources": [], "prompts": []},
            "is_connected": False,
        }
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))  # armed

        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}3")  # local:web
        await pilot.pause()

        assert list(app.query("#mcp-detail-delete"))
        assert not list(app.query("#mcp-detail-delete-confirm"))
        assert app.unified_mcp_service.delete_calls == []


@pytest.mark.asyncio
async def test_delete_confirmed_ignores_non_local_server_key():
    """Only local-source server_keys are ever produced by the detail
    toolbar's arm-then-confirm flow (built-in/server-source render no
    toolbar at all -- see test_mcp_servers_mode.py), but the handler must
    not misinterpret a server-source key by calling delete_local_profile
    with a bogus profile id derived from it."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("server:main")
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.unified_mcp_service.delete_calls == []


# -- T8: mcpServers import (paste or file) -----------------------------------


class ImportHubService(FakeHubService):
    """Like `FakeHubService`, but wires `save_local_profile()` per-id so a
    test can make one candidate in a batch fail while the rest succeed
    (`fail_ids`), and actually grows its catalog on success -- mirrors
    `ProfileFormHubService` -- so the post-apply reload has something new to
    show. Seeds one "docs" profile, which doubles as the existing-id fixture
    for the overwrite-warning path.
    """

    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        super().__init__()
        self.save_calls: list[dict] = []
        self._fail_ids = set(fail_ids or ())
        self._records: list[dict] = [
            {
                "profile_id": "docs",
                "command": "python",
                "args": [],
                "env_placeholders": {},
                "env_literals": {},
                "discovery_snapshot": {
                    "tools": [{"name": "a"}],
                    "resources": [],
                    "prompts": [],
                },
                "is_connected": True,
            }
        ]

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return list(self._records)
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }

    async def local_external_catalog(self):
        return list(self._records)

    async def save_local_profile(self, payload):
        self.save_calls.append(dict(payload))
        if payload.get("profile_id") in self._fail_ids:
            raise ValueError(f"{payload.get('profile_id')}: cannot be saved")
        self._records = [
            r for r in self._records if r.get("profile_id") != payload.get("profile_id")
        ] + [dict(payload)]
        return dict(payload)


class ImportApp(ConsolidatedCSSApp):
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        super().__init__()
        self.unified_mcp_service = ImportHubService(fail_ids=fail_ids)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_import_paste_preview_apply_calls_save_per_candidate_and_closes_panel():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        assert app.query_one("#mcp-servers-form").display
        assert not app.query_one("#mcp-servers-overview").display

        text = json.dumps(
            {"mcpServers": {"web": {"command": "npx", "args": ["-y", "pkg"]}}}
        )
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        assert not app.query_one("#mcp-import-apply", Button).disabled

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.save_calls == [
            {
                "profile_id": "web",
                "command": "npx",
                "args": ["-y", "pkg"],
                "env_placeholders": {},
                "env_literals": {},
            }
        ]
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert any("web" in msg for msg, _ in notifications)
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:web" in keys


@pytest.mark.asyncio
async def test_import_apply_existing_id_warns_and_overwrites():
    """The seeded "docs" profile (from FakeHubService.load_section) is the
    existing-id fixture: previewing an import that reuses "docs" must both
    warn in the panel and still go through save_local_profile (overwrite).
    """
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps({"mcpServers": {"docs": {"command": "python3"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        body = str(app.query_one("#mcp-import-list Static").renderable)
        assert "overwrite" in body

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.save_calls == [
            {
                "profile_id": "docs",
                "command": "python3",
                "args": [],
                "env_placeholders": {},
                "env_literals": {},
            }
        ]


@pytest.mark.asyncio
async def test_import_apply_failure_produces_summary_notify_without_aborting_rest():
    app = ImportApp(fail_ids={"bad"})
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps(
            {
                "mcpServers": {
                    "good": {"command": "npx"},
                    "bad": {"command": "npx"},
                }
            }
        )
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Both candidates were attempted -- the failure of one did not abort
        # the batch.
        attempted_ids = {
            call["profile_id"] for call in app.unified_mcp_service.save_calls
        }
        assert attempted_ids == {"good", "bad"}

        summary = [
            msg for msg, severity in notifications if "good" in msg or "bad" in msg
        ]
        assert summary, f"expected a combined summary notify, got: {notifications!r}"
        assert any("good" in msg and "bad" in msg for msg in summary), (
            f"expected one summary covering both outcomes, got: {summary!r}"
        )
        # Reload picked up the surviving success.
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:good" in keys
        assert "local:bad" not in keys


@pytest.mark.asyncio
async def test_import_double_apply_dispatches_exactly_one_batch():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps({"mcpServers": {"web": {"command": "npx"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()

        workbench = app.query_one(MCPWorkbench)
        panel = app.query_one(MCPImportPanel)
        workbench.on_mcp_import_panel_import_requested(
            MCPImportPanel.ImportRequested(list(panel._candidates))
        )
        workbench.on_mcp_import_panel_import_requested(
            MCPImportPanel.ImportRequested(list(panel._candidates))
        )
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second apply must toast a warning, got: {notifications!r}"
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.save_calls) == 1


@pytest.mark.asyncio
async def test_import_cancel_closes_panel_without_saving():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        await pilot.click("#mcp-import-cancel")
        await pilot.pause()
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls == []


@pytest.mark.asyncio
async def test_file_requested_pushes_picker_and_loads_selected_file_into_panel(
    tmp_path, monkeypatch
):
    """The production handler opens a JSON picker and schedules the selected file load."""
    monkeypatch.setattr(mcp_workbench_module, "_mcp_import_home", lambda: str(tmp_path))
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"docs": {"command": "npx"}}}))

    pushed: dict[str, Any] = {}
    scheduled: dict[str, Any] = {}
    stopped: list[bool] = []

    async def fake_push_screen(screen, callback=None):
        pushed["screen"] = screen
        pushed["callback"] = callback

    async def fake_load_import_file(file_path: str) -> None:
        scheduled["loaded_path"] = file_path

    def fake_run_worker(coroutine, **kwargs):
        scheduled["coroutine"] = coroutine
        scheduled["kwargs"] = kwargs

    handler = SimpleNamespace(
        app=SimpleNamespace(push_screen=fake_push_screen),
        run_worker=fake_run_worker,
        _load_import_file=fake_load_import_file,
    )
    event = SimpleNamespace(stop=lambda: stopped.append(True))

    await MCPWorkbench.on_mcp_import_panel_file_requested(handler, event)

    assert stopped == [True]
    assert pushed["screen"]._title == "Select MCP config JSON"
    assert pushed["callback"] is not None

    pushed["callback"](config_path)
    assert scheduled["kwargs"] == {
        "group": "mcp-import-file",
        "exclusive": True,
    }
    await scheduled["coroutine"]
    assert scheduled["loaded_path"] == str(config_path)

    loaded_text: list[str] = []
    notifications: list[tuple[str, str]] = []
    loader = SimpleNamespace(
        app=SimpleNamespace(
            notify=lambda message, severity="information": notifications.append(
                (str(message), severity)
            )
        ),
        _import_panel_or_none=lambda: SimpleNamespace(
            set_file_text=lambda text: loaded_text.append(text)
        ),
    )

    await MCPWorkbench._load_import_file(loader, str(config_path))

    assert notifications == []
    assert loaded_text == [config_path.read_text()]


@pytest.mark.asyncio
async def test_non_utf8_import_file_does_not_crash_app(tmp_path, monkeypatch):
    """The production loader contains non-UTF-8 failures and reports them."""
    monkeypatch.setattr(mcp_workbench_module, "_mcp_import_home", lambda: str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_bytes(b'\xff\xfe{"mcpServers": {}}')
    notifications: list[tuple[str, str]] = []
    loader = SimpleNamespace(
        app=SimpleNamespace(
            notify=lambda message, severity="information": notifications.append(
                (str(message), severity)
            )
        ),
        _import_panel_or_none=lambda: None,
    )

    await MCPWorkbench._load_import_file(loader, str(bad))

    assert any(
        "could not read" in msg.lower() and severity == "error"
        for msg, severity in notifications
    ), f"expected an error notify for the unreadable file, got: {notifications!r}"


@pytest.mark.asyncio
async def test_load_import_file_rejects_path_outside_home_directory(
    tmp_path, monkeypatch
):
    """F1 (Qodo compliance finding): the picked import file's path must
    route through `path_validation.is_safe_path()` before it is ever read.
    A path outside the validated root (here, `home`, standing in for
    `expanduser("~")`) is rejected with a plain validation-failure toast --
    no I/O is attempted, and no unread-error message (which would leak the
    fact the path exists) is shown instead.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(mcp_workbench_module, "_mcp_import_home", lambda: str(home))
    outside = tmp_path / "outside" / "mcp.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps({"mcpServers": {}}))

    notifications: list[tuple[str, str]] = []
    loader = SimpleNamespace(
        app=SimpleNamespace(
            notify=lambda message, severity="information": notifications.append(
                (str(message), severity)
            )
        )
    )

    await MCPWorkbench._load_import_file(loader, str(outside))

    assert notifications == [("Import file path failed validation.", "error")], (
        f"expected exactly one path-validation error toast, got: {notifications!r}"
    )


@pytest.mark.asyncio
async def test_load_import_file_rejects_oversized_file(tmp_path, monkeypatch):
    """F1 (Qodo compliance finding): a config JSON over the size cap must be
    rejected with a clear size-limit error before its contents are ever read
    into the import panel -- mirrors `attachment_core.MAX_ATTACHMENT_BYTES`'s
    reject-oversized-files precedent (`Tests/Chat/test_attachment_core.py::
    test_process_attachment_path_rejects_oversized_files`), lowering the cap
    via monkeypatch so the test file itself stays small.
    """
    monkeypatch.setattr(mcp_workbench_module, "_mcp_import_home", lambda: str(tmp_path))
    monkeypatch.setattr(mcp_workbench_module, "MAX_MCP_IMPORT_FILE_BYTES", 16)
    big = tmp_path / "big.json"
    big.write_text("x" * 64)

    notifications: list[tuple[str, str]] = []
    loader = SimpleNamespace(
        app=SimpleNamespace(
            notify=lambda message, severity="information": notifications.append(
                (str(message), severity)
            )
        ),
        _import_panel_or_none=lambda: None,
    )

    await MCPWorkbench._load_import_file(loader, str(big))

    assert any(
        "too large" in msg.lower() and severity == "error"
        for msg, severity in notifications
    ), f"expected an oversized-file error notify, got: {notifications!r}"


@pytest.mark.asyncio
async def test_delete_local_profile_notify_escapes_markup_in_profile_id():
    """F3 (Gemini finding, adapted): the local store keeps a profile id RAW
    on purpose, so a profile id shaped like Rich markup (e.g. embedded via a
    hand-edited config or import) must be escaped by `_toast()` before it
    reaches the "Deleted ..." `app.notify()` toast -- otherwise
    `[red]x[/red]` would be interpreted as styling/control markup instead of
    displayed literally.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        await workbench._delete_local_profile("local:[red]x[/red]", "[red]x[/red]")
        await pilot.pause()
        assert any("\\[red]x\\[/red]" in msg for msg, _ in notifications), (
            f"expected the escaped literal in the toast, got: {notifications!r}"
        )
        assert not any(msg == "Deleted [red]x[/red]." for msg, _ in notifications), (
            f"profile id markup must not reach notify() unescaped, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_notify_survives_markup_bearing_text():
    """Review probe (b): a message that looks like unbalanced Rich markup
    (e.g. embedded in a profile id) must not crash `app.notify()`. Kept as a
    permanent regression guard even though it already passed pre-fix --
    documents that this adjacent surface is NOT the C1 crash and must stay
    that way.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.notify("Saved [/bold]x.")
        await pilot.pause()
        assert app.is_running


# -- Task 12: Advanced disclosure object label + info-callout placeholders --


@pytest.mark.asyncio
async def test_advanced_object_label_updates_on_source_switch():
    """UX-inputs #1: switching source must rebind the inspector's Advanced
    object label (and, per `MCPInspector.set_service_context()`, reset/
    reload its section content) so a previous object's facts never linger.
    `FakeHubService.target_store` labels server_id "main" as "Main Server".
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: Local control plane"

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        assert str(label.renderable) == "Showing: server Main Server"

        # Switching back to local must rebind the label again, not leave the
        # server-source text stuck on screen.
        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        await pilot.pause()
        assert str(label.renderable) == "Showing: Local control plane"


class AuxTarget:
    server_id = "aux"
    label = "Aux Server"
    base_url = "https://aux.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class TwoTargetStore:
    def list_targets(self):
        return [FakeTarget(), AuxTarget()]


@pytest.mark.asyncio
async def test_same_row_reclick_preserves_advanced_section():
    """Review fix (T12): the UX-inputs text says rebind on selection
    CHANGE -- a reclick of the already-selected row is not a change, and
    must not wipe the user's Advanced browsing state (section snapping back
    to Overview). Mirrors the C1 ScopeChanged dedup precedent in this file.
    """
    app = WorkbenchApp()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "inventory"
        await pilot.pause()
        await pilot.pause()
        assert section_select.value == "inventory"

        # Reclick the SAME row: not a selection change -- browsing state
        # must survive.
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()
        assert section_select.value == "inventory"


@pytest.mark.asyncio
async def test_different_target_selection_rebinds_advanced_context():
    """Counterpart guard: a GENUINE selection change (a different server
    target) must still rebind -- the section resets to the first entry and
    the object label follows the new target."""
    app = WorkbenchApp()
    app.unified_mcp_service.target_store = TwoTargetStore()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "inventory"
        await pilot.pause()
        await pilot.pause()

        rail.post_message(MCPRail.ServerSelected("server:aux"))
        await pilot.pause()
        await pilot.pause()

        assert section_select.value == "overview"
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: server Aux Server"


# UX-inputs #4's phase-placeholder styling test
# (`test_mode_placeholder_canvases_use_info_callout_not_recovery_callout`,
# `.ds-info-callout` vs `.ds-recovery-callout`) is retired along with T7:
# "audit" was the last `MCP_HUB_MODES` entry still on the generic
# phase-placeholder path (T5 shipped Tools, T6 shipped Permissions) --
# every mode is now a real canvas, so the loop this test drove would `continue`
# for all four and assert nothing. Mirrors the UX batch items 2+3 kill-switch
# height-rule retirement in test_mcp_permissions_mode.py: a test that can no
# longer exercise anything is worse than no test (it reads as coverage that
# isn't there).


# -- T5: Tools mode canvas registration + workbench-fed catalog -------------


@pytest.mark.asyncio
async def test_tools_mode_canvas_replaces_placeholder():
    """`#mcp-mode-canvas-tools` must host the real `MCPToolsMode` widget --
    not the generic placeholder Vertical/Static every other not-yet-built
    mode still renders."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.query_one("#mcp-mode-canvas-tools"), MCPToolsMode)
        assert not list(app.query("#mcp-mode-canvas-tools > Static"))


@pytest.mark.asyncio
async def test_tools_mode_shows_tools_from_local_catalog_snapshots():
    """T5: `_collect_hub_tools()` derives Tools mode's catalog from the SAME
    local-profile records `_collect_snapshots()` already loaded for the rail/
    overview (`FakeHubService`'s seeded "docs" profile, discovery_snapshot
    tools=[{"name": "a"}]) -- no separate fetch.
    """
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "local:docs::a"
        assert canvas.query_one("#mcp-tools-empty").display is False


class NoServersHubService(FakeHubService):
    """Local source with zero local profiles configured (and no builtin
    inventory, since this fake never sets `local_service`) -- the "no
    servers configured" empty-diagnosis bucket."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return []
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }


class NoServersApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = NoServersHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_empty_diagnosis_no_servers_shows_add_server_and_button_opens_form():
    app = NoServersApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")  # the button must be on-screen for pilot.click()
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        empty = canvas.query_one("#mcp-tools-empty")
        assert empty.display is True
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        # task-3240: a trailing "N tool gate(s) are off ..." breadcrumb may
        # follow (the isolated test config's [tools]/[console] gates all
        # default off) -- startswith isolates this test's own concern from
        # that unrelated, separately-tested addition.
        assert message.startswith("No servers configured — add one to see its tools.")

        # Fix round 1 (Minor 1): this test's own NAME promises the
        # click-opens-form behavior -- restored here (it had drifted onto
        # a differently-named breadcrumb test below when the message
        # assertion above was loosened to `.startswith`).
        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert not form.is_edit


@pytest.mark.asyncio
async def test_empty_diagnosis_names_the_gate_off_count_when_gates_are_off():
    """task-3240 SECONDARY breadcrumb: `_empty_tools_diagnosis()` appends
    "N tool gate(s) are off ..." whenever `all_tool_gates()` finds any --
    TASK-14807 defaults the local master gate on, leaving every other gate
    off. N is DERIVED (TASK-16174): a new gateable built-in changes the
    count, and this test is about the breadcrumb, not the arity."""
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    off_count = len(_GATEABLE_BUILTINS) + 1  # + web_deep_search, - the master
    app = NoServersApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        assert f"{off_count} tool gate(s) are off" in message
        assert "Tools mode" in message


@pytest.mark.asyncio
async def test_empty_diagnosis_omits_gate_breadcrumb_when_all_gates_are_on(monkeypatch):
    """Mirror of the test above: no breadcrumb at all once every gate is on."""
    import tldw_chatbook.config as config_module

    real_get_cli_setting = config_module.get_cli_setting

    def fake_get_cli_setting(section, key=None, default=None):
        if section in ("tools", "console"):
            return True
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

    app = NoServersApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        assert message == "No servers configured — add one to see its tools."
        assert "tool gate" not in message

        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert not form.is_edit


@pytest.mark.asyncio
async def test_empty_diagnosis_connect_routes_to_servers_mode_with_notify():
    """LifecycleFakeHubService's seeded "docs" profile is disconnected with
    no discovery snapshot -> NEEDS_SETUP -- the "servers exist but none
    connected/discovered" empty-diagnosis bucket. The empty state's button
    must switch to Servers mode (where the real connect/refresh actions
    live) and notify, not attempt any lifecycle action itself.
    """
    app = LifecycleApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        # task-3240: see the sibling comment above -- a trailing gate
        # breadcrumb may follow.
        assert message.startswith(
            "No tools discovered yet — connect or refresh a server."
        )

        notifications = _capture_notifications(app)
        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        assert workbench.active_mode == "servers"
        assert notifications and notifications[-1] == (
            "Select a server below to connect or refresh its tools.",
            "information",
        )


class ServerToolsHubService(FakeHubService):
    """Server source with one active target ("main") whose sole external
    record embeds its own `tools` list -- mirrors a backend that returns
    per-record tool inventories inline (see `readiness.py`'s own
    `record.get("tools")` tool_count fallback, the same embedded shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "external_servers":
            return {
                "external_servers": [
                    {
                        "server_id": "docs",
                        "name": "Docs",
                        "tools": [{"name": "search", "description": "Search."}],
                    }
                ],
                "source": "server",
                "section": "external_servers",
            }
        return {
            "external_servers": [],
            "source": "server",
            "section": effective_section,
        }


class ServerToolsApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = ServerToolsHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_tools_mode_shows_server_source_tools_from_embedded_inventory():
    app = ServerToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "server:main/docs::search"
        assert canvas.query_one("#mcp-tools-empty").display is False


class DuplicateNameToolsHubService(FakeHubService):
    """A local external profile whose discovery snapshot carries two
    same-named tools -- C1: this used to crash every mount of the Tools mode
    canvas (Textual `DuplicateKey` on the DataTable row key, which is
    `HubTool.tool_id`), and since discovery snapshots persist verbatim to
    disk, it was a permanent crash-loop, not a one-off."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_source or "overview"
        if (
            self.context.selected_source == "local"
            and effective_section == "external_servers"
        ):
            return [
                {
                    "profile_id": "docs",
                    "command": "python",
                    "args": [],
                    "env_placeholders": {},
                    "discovery_snapshot": {
                        "tools": [
                            {"name": "search", "description": "a"},
                            {"name": "search", "description": "b"},
                        ],
                        "resources": [],
                        "prompts": [],
                    },
                    "is_connected": True,
                }
            ]
        return {"source": "local", "section": effective_section}

    async def local_external_catalog(self):
        return await self.load_section("external_servers")


class DuplicateNameToolsApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = DuplicateNameToolsHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_duplicate_tool_names_do_not_crash_workbench_mount():
    app = DuplicateNameToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "local:docs::search"


# -- Task 6: inspector tool detail + Test Tool runner wiring -----------------
#
# `ToolTestHubService` seeds two local profiles: "docs" (tools "fetch" --
# no inputSchema, raw-mode -- and "search" -- a schema with a required
# "query" string, form-mode) and "notes" (tool "list_notes"). Sorted by
# (server_label, name) per MCPToolsMode._apply_filter(), the table rows are:
# 0=docs::fetch, 1=docs::search, 2=notes::list_notes.


class ToolTestHubService(FakeHubService):
    def __init__(self) -> None:
        super().__init__()
        self.test_calls: list[tuple[str, str, dict]] = []
        self.test_result: Any = {"ok": True}
        self.raise_error: Exception | None = None
        # When set, test_hub_tool() records its call then blocks on this
        # gate -- mirrors LifecycleFakeHubService.connect_gate, for the
        # double-run test.
        self.test_gate: asyncio.Event | None = None
        # Task 5: `gate_tool_test()` (T4) resolution this fake returns,
        # settable per-test. Defaults to "allow" (today's/Phase-3 behavior
        # for every test in this file that predates Task 5 and never
        # touches these fields).
        self.gate_state: str = "allow"
        self.gate_config_changed: bool = False
        self.gate_risk_floored: bool = False
        self.gate_calls: list[tuple[str, str]] = []
        # Store-only resolution calls for catalog-vanished tools, driven by
        # the same `gate_state`, mirroring the real
        # `UnifiedMCPControlPlaneService.gate_tool_test_by_key()`'s "reads
        # the same store `gate_tool_test()` does" contract.
        self.gate_by_key_calls: list[tuple[str, str]] = []
        # Task 5 (RAG-51): the `decision` kwarg `test_hub_tool()` now
        # accepts, recorded separately from `test_calls` -- `test_calls`'s
        # 3-tuple shape is pinned by many pre-existing assertions
        # (`test_calls == [("local:docs", "fetch", {})]`), so the decision
        # string goes in its own list rather than growing that tuple.
        self.decision_calls: list[str] = []
        # Task 4 (PR-T3): the schema-approved argument names the dispatch
        # path (`on_mcp_inspector_tool_test_requested()`) derives from the
        # resolved `HubTool.input_schema` and forwards here -- separate
        # from `test_calls` for the same reason `decision_calls` is (that
        # 3-tuple's shape is pinned by many pre-existing assertions).
        self.registered_argument_names_calls: list[set[str] | None] = []
        self.prepared_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.revoked_nonces: list[str] = []
        self._previews: dict[str, ToolTestAdmissionPreview] = {}
        self._active_tests: set[tuple[str, str]] = set()
        self.active_calls: list[tuple[str, str]] = []
        self._preview_count = 0
        self.next_prepared_outcome: Any | None = None

    def gate_tool_test(self, tool: Any) -> EffectiveToolState:
        self.gate_calls.append((tool.server_key, tool.name))
        return EffectiveToolState(
            state=self.gate_state,
            origin="tool_override",
            config_changed=self.gate_config_changed,
            risk_floored=self.gate_risk_floored,
        )

    def gate_tool_test_by_key(
        self, server_key: str, tool_name: str
    ) -> EffectiveToolState:
        self.gate_by_key_calls.append((server_key, tool_name))
        return EffectiveToolState(
            state=self.gate_state,
            origin="tool_override",
            config_changed=self.gate_config_changed,
            risk_floored=self.gate_risk_floored,
        )

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [
                                {"name": "fetch", "description": "Fetch a doc."},
                                {
                                    "name": "search",
                                    "description": "Search the docs.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "Search text",
                                            }
                                        },
                                        "required": ["query"],
                                    },
                                },
                            ],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                    {
                        "profile_id": "notes",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [
                                {"name": "list_notes", "description": "List notes."}
                            ],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                ]
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }

    async def local_external_catalog(self):
        return await self.load_section("external_servers")

    async def test_hub_tool(
        self,
        server_key,
        tool_name,
        arguments=None,
        *,
        decision="allowed",
        registered_argument_names=None,
    ):
        self.test_calls.append((server_key, tool_name, dict(arguments or {})))
        self.decision_calls.append(decision)
        self.registered_argument_names_calls.append(registered_argument_names)
        if self.test_gate is not None:
            await self.test_gate.wait()
        if self.raise_error is not None:
            raise self.raise_error
        return self.test_result

    def prepare_hub_test(self, tool: HubTool) -> ToolTestAdmissionPreview:
        self._preview_count += 1
        rendered_gate = {
            "allow": "allow",
            "ask": "ask",
            "deny": "off",
        }.get(self.gate_state, "unresolved")
        preview = ToolTestAdmissionPreview(
            nonce=f"preview-{self._preview_count}",
            server_key=tool.server_key,
            tool_name=tool.name,
            definition_hash="definition",
            rendered_gate=rendered_gate,
            authority_fingerprint=None,
            safe_authority_label=None,
        )
        self._previews[preview.nonce] = preview
        return preview

    def revoke_hub_test_preview(self, nonce: str) -> None:
        self.revoked_nonces.append(nonce)
        self._previews.pop(nonce, None)

    def hub_test_active(self, server_key: str, tool_name: str) -> bool:
        self.active_calls.append((server_key, tool_name))
        return (server_key, tool_name) in self._active_tests

    async def execute_prepared_hub_test(
        self, nonce: str, intent: str, arguments: dict[str, Any]
    ) -> Any:
        preview = self._previews.pop(nonce, None)
        if preview is None:
            return ToolTestAdmissionStale(reason="preview_unavailable")
        expected = "approve_once" if preview.rendered_gate == "ask" else "run"
        if preview.rendered_gate == "off":
            return ToolTestAdmissionBlocked(
                reason="permission_denied",
                refreshed_preview=self.prepare_hub_test(
                    next(
                        tool
                        for tool in self._fake_tools_for_preview()
                        if (tool.server_key, tool.name)
                        == (preview.server_key, preview.tool_name)
                    )
                ),
            )
        if intent != expected:
            return ToolTestAdmissionBlocked(reason="intent_mismatch")
        if self.next_prepared_outcome is not None:
            outcome = self.next_prepared_outcome
            self.next_prepared_outcome = None
            return outcome
        key = (preview.server_key, preview.tool_name)
        if key in self._active_tests:
            return ToolTestAdmissionBlocked(reason="already_active")
        self._active_tests.add(key)
        self.prepared_calls.append((nonce, intent, dict(arguments)))
        self.test_calls.append((preview.server_key, preview.tool_name, dict(arguments)))
        self.decision_calls.append(
            "approved" if intent == "approve_once" else "allowed"
        )
        self.registered_argument_names_calls.append(set(arguments))
        try:
            if self.test_gate is not None:
                await self.test_gate.wait()
            if self.raise_error is not None:
                raise self.raise_error
            return self.test_result
        finally:
            self._active_tests.discard(key)

    def _fake_tools_for_preview(self) -> list[HubTool]:
        return [
            HubTool(
                server_key="local:docs",
                server_label="docs",
                name="fetch",
                description="Fetch a doc.",
                input_schema=None,
                source="local",
                executable=True,
            )
        ]


class ToolTestApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = ToolTestHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


class BuiltinToolTestHubService(ToolTestHubService):
    class LocalService:
        @staticmethod
        def get_inventory():
            return {
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Calculate.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                        },
                    }
                ]
            }

    def __init__(self) -> None:
        super().__init__()
        self.local_service = self.LocalService()


class BuiltinToolTestApp(ToolTestApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = BuiltinToolTestHubService()


async def _select_tools_mode_row(app: App, pilot, row: int) -> None:
    table = app.query_one("#mcp-tools-table", DataTable)
    table.focus()
    table.move_cursor(row=row)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _prepared_test_event(
    service: ToolTestHubService,
    tool: HubTool,
    arguments: dict[str, Any],
) -> MCPInspector.ToolTestRequested:
    preview = service.prepare_hub_test(tool)
    return MCPInspector.ToolTestRequested(
        tool.server_key,
        tool.name,
        arguments,
        preview_nonce=preview.nonce,
        intent="approve_once" if preview.rendered_gate == "ask" else "run",
    )


@pytest.mark.asyncio
async def test_tool_row_selection_shows_tool_detail_with_test_button():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        name_text = str(app.query_one("#mcp-inspector-tool-name", Static).renderable)
        assert "docs" in name_text
        test_button = app.query_one("#mcp-inspector-test-tool", Button)
        assert test_button.tooltip == "Run this tool with test arguments."


@pytest.mark.asyncio
async def test_switching_mode_clears_tool_detail():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        assert list(app.query("#mcp-inspector-tool-name"))

        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_switching_selected_server_clears_tool_detail():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        assert list(app.query("#mcp-inspector-tool-name"))

        await workbench._select_server_key("local:notes")
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_second_tool_selection_back_to_back_does_not_duplicate_ids():
    """Mandatory regression (mirrors test_mcp_inspector.py's
    test_second_show_tool_back_to_back_does_not_duplicate_ids): selecting a
    second tool before the first selection's inspector refresh has settled
    must not raise DuplicateIds."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool_a = next(t for t in tools if t.name == "fetch")
        tool_b = next(t for t in tools if t.name == "search")
        await workbench.on_mcp_tools_mode_tool_selected(
            MCPToolsMode.ToolSelected(tool_a.tool_id)
        )
        # No pause here on purpose.
        await workbench.on_mcp_tools_mode_tool_selected(
            MCPToolsMode.ToolSelected(tool_b.tool_id)
        )
        await pilot.pause()
        names = list(app.query("#mcp-inspector-tool-name"))
        assert len(names) == 1
        assert "search" in str(names[0].renderable)


def test_mcp_workbench_source_never_parses_packed_tool_id():
    """task-233 grep-gate: the execute path now carries (server_key,
    tool_name) as separate fields end to end -- nothing in mcp_workbench.py
    may reconstruct them by parsing a packed "server_key::tool_name" string
    anymore. Row keys elsewhere (mcp_tools_mode.py) are unaffected -- this
    only asserts mcp_workbench.py's own source never parses one."""
    source = Path(mcp_workbench_module.__file__).read_text()
    assert 'partition("::")' not in source
    assert "partition('::')" not in source
    assert 'split("::")' not in source
    assert "split('::')" not in source


@pytest.mark.asyncio
async def test_tool_for_resolves_by_server_key_and_tool_name():
    """`_tool_for(server_key, tool_name)` compares fields, not a packed
    string -- distinct from `_tool_for_row_key()`, which still resolves the
    Tools-mode DataTable's packed row key."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        fetch_tool = next(t for t in tools if t.name == "fetch")

        resolved = workbench._tool_for(fetch_tool.server_key, fetch_tool.name)
        assert resolved is fetch_tool

        # Same tool_name, different server -- must not match.
        assert workbench._tool_for("local:notes", "fetch") is None
        # Same server, unknown tool_name -- must not match.
        assert workbench._tool_for(fetch_tool.server_key, "does-not-exist") is None


@pytest.mark.asyncio
async def test_test_tool_run_success_calls_service_and_renders_ok():
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {"ok": True}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (form schema)
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.test_calls == [
            ("local:docs", "search", {"query": "hello"})
        ]
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("OK · ")
        assert first_line.endswith("ms")
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False


@pytest.mark.asyncio
async def test_test_tool_run_error_renders_failed_with_message():
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = RuntimeError("boom")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert first_line.endswith("ms")
        assert "boom" in result


# -- F4 (PR-T3 task 3): a refusal must never read as a failure -------------
#
# task-2537 + task-2539 (PR-T3 fix round B, item 3): `_is_permission_
# refusal()`'s classification contract changed from "any `PermissionError`,
# or a `ValueError` matching an exact string" to "one of two DEDICATED
# exception types" (`MCPGovernanceDenied`, `MCPServerSourceDisplayOnlyError`
# -- see their own docstrings). NOTE: the old contract test this replaces
# (`test_is_permission_refusal_classifies_permission_error_and_display_
# only_value_error`) is not on fix round B's list of two pre-authorized
# assertion changes -- flagged for review in the round's report. It had to
# change because it directly pinned the over-broad/prose-dependent
# behavior this item's own brief identifies as the bug: a bare
# `PermissionError` (not `MCPGovernanceDenied`) must now classify `False`,
# the exact inverse of what it asserted before.


def test_is_permission_refusal_classifies_typed_refusals_only():
    """Pure classification contract, independent of the UI:
    `MCPGovernanceDenied` (the governance seam's typed refusal) and
    `MCPServerSourceDisplayOnlyError` (`execute_hub_tool()`'s typed
    display-only refusal) both classify as a refusal; an unrelated
    `ValueError`/`RuntimeError` does not (a genuine failure must keep
    rendering as `Failed`, not get swept into `Blocked`)."""
    assert (
        mcp_workbench_module._is_permission_refusal(
            MCPGovernanceDenied("Denied by local governance: tool.execute")
        )
        is True
    )
    assert (
        mcp_workbench_module._is_permission_refusal(MCPServerSourceDisplayOnlyError())
        is True
    )
    assert mcp_workbench_module._is_permission_refusal(ValueError("bad json")) is False
    assert mcp_workbench_module._is_permission_refusal(RuntimeError("boom")) is False


def test_is_permission_refusal_bare_permission_error_from_tool_body_is_not_a_refusal():
    """task-2537: a `PermissionError` that is NOT `MCPGovernanceDenied` --
    e.g. a real OS EACCES a tool's own `execute()` body raises reading a
    permission-denied path -- must NOT classify as a refusal. The call DID
    reach the tool; the tool itself is what failed. Before this item, ANY
    `PermissionError` (the bare base class) classified as a refusal, which
    would have misrendered a genuine per-tool failure as `Blocked · not
    run`, falsely claiming the call never reached the tool. This contract
    remains necessary even though the retired `ingest_media` placeholder
    is absent from the standalone inventory."""
    assert (
        mcp_workbench_module._is_permission_refusal(
            PermissionError("EACCES: permission denied")
        )
        is False
    )


def test_is_permission_refusal_display_only_classification_is_type_based_not_message_based():
    """task-2539: classification no longer depends on the exact wording of
    the display-only message -- only on `MCPServerSourceDisplayOnlyError`'s
    TYPE. Before this item, the message was pinned only where it's
    RENDERED, never at its raise site, so an unrelated reword of
    `execute_hub_tool()`'s raise-site string would have silently reverted
    the F4 fix with a fully green suite; this proves that drift is now
    closed."""
    assert (
        mcp_workbench_module._is_permission_refusal(
            MCPServerSourceDisplayOnlyError("A totally different wording.")
        )
        is True
    )


@pytest.mark.asyncio
async def test_test_tool_run_permission_error_renders_blocked_not_failed():
    """F4: `local_control_service.execute_tool()`'s `MCPGovernanceDenied`
    (governance denies `tool.execute`) is a REFUSAL -- the call never
    reached the tool -- not a run failure. It must read as `Blocked · not
    run`, never `Failed · Nms`, and must NOT show the Hub Permissions
    "Change in Permissions" jump (a different permission system --
    jumping there would not fix a governance refusal).

    task-2537 (fix round B, item 3): the fake service now raises
    `MCPGovernanceDenied` (what the real governance seam raises after this
    item), not a bare `PermissionError` -- `_is_permission_refusal()` is
    now type-based and would no longer classify a bare `PermissionError`
    as a refusal (see `test_is_permission_refusal_bare_permission_error_
    from_tool_body_is_not_a_refusal`). The message text is unchanged
    (byte-identical), so every assertion below is untouched.

    Review fix (Important #1): `ToolTestHubService.gate_state` defaults to
    `"allow"`, so the Hub gate's own decision note would otherwise read
    "Ran because this tool is set to Allow..." right next to "Blocked ·
    not run" -- self-contradictory on the same run. The note must be
    empty/hidden for a refusal, not the Hub gate's unrelated dispatch
    reasoning."""
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = MCPGovernanceDenied(
        "Governance profile denies tool.execute."
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Blocked · not run")
        assert not result.startswith("Failed")
        assert "Governance profile denies tool.execute." in result
        goto = app.query_one("#mcp-inspector-goto-permission-test", Button)
        assert goto.display is False
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert str(note_widget.renderable) == ""
        assert note_widget.display is False
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False


@pytest.mark.asyncio
async def test_test_tool_run_server_source_display_only_value_error_renders_blocked_not_failed():
    """F4: `execute_hub_tool()`'s `MCPServerSourceDisplayOnlyError`
    ("Server-source tools are display-only.") is likewise a refusal (a
    structural mismatch), not a run failure.

    task-2539 (fix round B, item 3): the fake service now raises the typed
    `MCPServerSourceDisplayOnlyError`, not a bare `ValueError` --
    `_is_permission_refusal()` is now type-based. The message text is
    unchanged (byte-identical, it's this type's own default), so every
    assertion below is untouched.

    Review fix (Important #1): same self-contradiction guard as the
    `PermissionError` sibling test above -- the note must not carry the
    Hub gate's own "Ran because..." reasoning next to "Blocked · not
    run"."""
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = MCPServerSourceDisplayOnlyError()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.startswith("Blocked · not run")
        assert not result.startswith("Failed")
        assert "Server-source tools are display-only." in result
        note_widget = app.query_one("#mcp-inspector-test-result-note", Static)
        assert str(note_widget.renderable) == ""
        assert note_widget.display is False


@pytest.mark.asyncio
async def test_test_tool_run_bare_permission_error_from_tool_body_renders_failed_not_blocked():
    """task-2537 (fix round B, item 3), end to end -- the whole point of
    item 3(a): a genuine `PermissionError` a TOOL'S OWN body raises (e.g. a
    real OS EACCES reading a permission-denied path) is a run FAILURE, not
    a refusal -- the call DID reach the tool. It must render `Failed ·
    Nms`, never `Blocked · not run`, which would falsely claim the call
    never reached the tool. Before this item, `_is_permission_refusal()`
    matched any bare `PermissionError` regardless of where it was raised,
    so this exact scenario misrendered as a refusal."""
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = PermissionError(
        "EACCES: permission denied reading /etc/shadow"
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert not result.startswith("Blocked")
        assert "EACCES: permission denied reading [path]" in result
        assert "/etc/shadow" not in result


@pytest.mark.asyncio
async def test_test_tool_run_redacts_secret_shaped_result():
    """RAG-49 deliberate contract change: the redacted envelope no longer
    lives in the `#mcp-inspector-test-result` summary Static -- that now
    holds only the terse `OK · <duration>` structured summary (this fake
    envelope has neither a "result" list nor a "source" key, so there is
    no count/source segment to show). The full redacted envelope moved to
    the "Raw response" Collapsible's body Static; the secret-redaction
    guarantee itself is unchanged, just re-targeted to where the content
    now renders."""
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {"ok": True, "api_key": "sk-live-secret"}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "sk-live-secret" not in result
        raw_body = str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert "sk-live-secret" not in raw_body
        assert "***" in raw_body


@pytest.mark.asyncio
async def test_test_tool_run_redacts_secret_in_error_shaped_result_note():
    """Review fix (RAG-49, Important #1): the quiet interpretation line
    must be redacted too, not just the Raw response Collapsible.

    Before this fix, `_run_tool_test()` redacted the envelope only to
    build the raw JSON dump, then fed the ORIGINAL (unredacted) envelope's
    `result`/`source` to `show_tool_result()` -- so a secret embedded
    inside an error-shaped result's `"error"` value would render straight
    into `#mcp-inspector-test-result-note` (via `_summarize_tool_result()`'s
    `str(result[0]["error"])`) even though the raw body correctly showed
    `***`. Now `result`/`source` are derived from the SAME redacted copy
    the raw dump uses, so the secret cannot appear on ANY of the three
    result surfaces (summary, note, raw body)."""
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {
        "ok": True,
        "source": "local",
        "result": [{"error": {"api_key": "sk-live-x"}}],
    }
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        summary = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        note = str(app.query_one("#mcp-inspector-test-result-note", Static).renderable)
        raw_body = str(
            app.query_one("#mcp-inspector-test-result-raw-body", Static).renderable
        )
        assert "sk-live-x" not in summary
        assert "sk-live-x" not in note
        assert "sk-live-x" not in raw_body
        assert "***" in note
        assert "***" in raw_body
        assert summary.startswith("OK · local · ")
        assert summary.endswith("ms · tool returned an error")


@pytest.mark.asyncio
async def test_test_tool_run_error_with_dict_shaped_args_is_redacted():
    """I1 (ledger #5): some errors carry a raw dict payload in `exc.args`
    (e.g. an echoed request/arguments dict) -- `str(exc)` would otherwise
    dump that dict's raw repr, including any secret-shaped values in it,
    straight into the result panel."""
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = RuntimeError(
        {"api_key": "sk-live-secret", "detail": "bad request"}
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "sk-live-secret" not in result
        assert "***" in result
        assert "bad request" in result


@pytest.mark.asyncio
async def test_test_tool_one_click_double_run_service_admits_once():
    """Two delivered clicks race at the service; only one is admitted."""
    app = ToolTestApp()
    app.unified_mcp_service.test_gate = asyncio.Event()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "search")
        preview = app.unified_mcp_service.prepare_hub_test(tool)
        event = MCPInspector.ToolTestRequested(
            tool.server_key,
            tool.name,
            {"query": "hello"},
            preview_nonce=preview.nonce,
            intent="run",
        )
        workbench.on_mcp_inspector_tool_test_requested(event)
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        assert len(app.unified_mcp_service.test_calls) == 1
        app.unified_mcp_service.test_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.test_calls) == 1


@pytest.mark.asyncio
async def test_test_tool_one_click_ask_dispatches_approve_once_from_first_activation():
    app = ToolTestApp()
    app.unified_mcp_service.gate_state = "ask"
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()

        button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(button.label) == "Approve & run once"
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.pause()
        button.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert [call[1] for call in app.unified_mcp_service.prepared_calls] == [
            "approve_once"
        ]
        assert app.unified_mcp_service.test_calls == [
            ("local:docs", "search", {"query": "hello"})
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_method",
    [
        "prepare_hub_test",
        "execute_prepared_hub_test",
        "revoke_hub_test_preview",
        "hub_test_active",
    ],
)
async def test_test_tool_preview_missing_prepared_api_is_unavailable(
    missing_method: str,
):
    app = ToolTestApp()
    setattr(app.unified_mcp_service, missing_method, None)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()

        button = app.query_one("#mcp-inspector-test-run", Button)
        status = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)
        assert str(button.label) == "Unavailable"
        assert button.disabled is True
        assert "not supported" in status
        button.press()
        await pilot.pause()
        assert app.unified_mcp_service.test_calls == []


@pytest.mark.asyncio
async def test_test_tool_panel_open_reads_service_active_state():
    app = ToolTestApp()
    app.unified_mcp_service._active_tests.add(("local:docs", "fetch"))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")

        button = await _wait_for_test_button_label(app, pilot, "Running…")
        assert str(button.label) == "Running…"
        assert button.disabled is True
        assert app.unified_mcp_service._preview_count == 0
        assert app.unified_mcp_service.active_calls
        assert set(app.unified_mcp_service.active_calls) == {("local:docs", "fetch")}

        app.unified_mcp_service._active_tests.discard(("local:docs", "fetch"))
        await _wait_for_test_button_label(app, pilot, "Run")


@pytest.mark.asyncio
async def test_test_tool_preview_stale_refresh_preserves_current_arguments():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()

        field = app.query_one("#mcp-schema-field-0", Input)
        field.value = "keep me"
        tool = next(t for t in workbench._last_hub_tools if t.name == "search")
        refreshed = app.unified_mcp_service.prepare_hub_test(tool)
        app.unified_mcp_service.next_prepared_outcome = ToolTestAdmissionStale(
            reason="definition_changed",
            refreshed_preview=refreshed,
        )
        button = app.query_one("#mcp-inspector-test-run", Button)
        button.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#mcp-schema-field-0", Input).value == "keep me"
        assert str(button.label) == "Run"
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert result.splitlines()[0] == "Changed · not run"
        assert not result.startswith("Failed")
        assert "tool definition changed" in result.lower()


@pytest.mark.asyncio
async def test_test_tool_preview_switch_revokes_old_nonce():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()
        old_nonce = app.query_one(MCPInspector)._test_preview.nonce

        await _select_tools_mode_row(app, pilot, 1)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert old_nonce in app.unified_mcp_service.revoked_nonces


@pytest.mark.asyncio
async def test_test_tool_preview_long_error_is_bounded_and_recoverable():
    app = ToolTestApp()

    def fail_prepare(_tool: HubTool) -> ToolTestAdmissionPreview:
        raise RuntimeError("x" * 2_000)

    app.unified_mcp_service.prepare_hub_test = fail_prepare
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)
        assert status.startswith("Unavailable.")
        assert status.endswith("Try again.")
        assert len(status) <= 280


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(
            "token=sk-live-string-secret failed at /Users/alice/private/key.txt "
            + ("x" * 4_000)
        ),
        RuntimeError(
            {
                "api_key": "sk-live-mapping-secret",
                "path": r"\\?\C:\private\private-token.json",
                "detail": "x" * 4_000,
            }
        ),
    ],
)
async def test_test_tool_preview_failure_redacts_secrets_paths_and_bounds_text(failure):
    app = ToolTestApp()

    def fail_prepare(_tool: HubTool) -> ToolTestAdmissionPreview:
        raise failure

    app.unified_mcp_service.prepare_hub_test = fail_prepare
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.pause()

        rendered = str(app.query_one("#mcp-inspector-test-preview", Static).renderable)
        assert "sk-live" not in rendered
        assert "/Users/" not in rendered
        assert "/private/" not in rendered
        assert "private-token.json" not in rendered
        assert len(rendered) <= 560


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(
            "password=string-secret failed at /Users/alice/private/key.txt "
            + ("x" * 4_000)
        ),
        RuntimeError(
            {
                "access_token": "mapping-secret",
                "path": "/private/tmp/private-token.json",
                "detail": "x" * 4_000,
            }
        ),
    ],
)
async def test_test_tool_execution_failure_redacts_secrets_paths_and_bounds_text(
    failure,
):
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = failure
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.click("#mcp-inspector-test-run")
        await app.workers.wait_for_complete()
        await pilot.pause()

        rendered = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "string-secret" not in rendered
        assert "mapping-secret" not in rendered
        assert "/Users/" not in rendered
        assert "/private/" not in rendered
        assert len(rendered) <= 560


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_branch", ["access", "read"])
@pytest.mark.parametrize(
    ("private_path", "private_marker"),
    [
        (r"\\server\share\private\audit.json", "audit.json"),
        (
            r"'\\server\Shared Folder\private audit.json'",
            "Shared Folder",
        ),
        (r"\\?\C:\private\audit.json", "audit.json"),
        (r"'\\?\C:\Private Folder\audit.json'", "Private Folder"),
        (
            r"C:\Private Folder\audit.json: permission denied",
            "Private Folder",
        ),
        (
            "/Users/alice/Private Project/audit.json failed to open",
            "Project/audit.json",
        ),
        (
            "/Users/alice/Private Failed Project/audit.json",
            "Project/audit.json",
        ),
        ("root:/Users/alice/private/audit.json", "audit.json"),
        ("file:///Users/alice/private/audit.json", "audit.json"),
        (r"\\.\pipe\private-audit", "private-audit"),
    ],
)
async def test_test_tool_audit_sync_log_redacts_real_service_failure_branches(
    failure_branch: str,
    private_path: str,
    private_marker: str,
    caplog: pytest.LogCaptureFixture,
):
    app = ToolTestApp()
    secret = "sk-live-audit-log-secret"
    failure = RuntimeError(
        f"api_key={secret} failed at {private_path} " + ("x" * 4_000)
    )

    class AccessFailureService:
        @property
        def execution_log(self):
            raise failure

    class ReadFailureLog:
        def read_recent(self, _limit: int):
            raise failure

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        app.unified_mcp_service = (
            AccessFailureService()
            if failure_branch == "access"
            else SimpleNamespace(execution_log=ReadFailureLog())
        )
        caplog.clear()
        sink = mcp_workbench_module.logger.add(
            caplog.handler, level="WARNING", format="{message}"
        )
        try:
            await workbench._sync_audit_log_entries()
        finally:
            mcp_workbench_module.logger.remove(sink)

        prefix = f"MCP execution log {failure_branch} failed"
        rendered = "".join(message for message in caplog.messages if prefix in message)
        assert rendered, f"expected {prefix!r} in {caplog.messages!r}"
        assert secret not in rendered
        assert private_path not in rendered
        assert private_marker not in rendered
        assert "[redacted]" in rendered
        assert "[redacted]]" not in rendered
        assert "[path]" in rendered
        assert len(rendered) <= 560
        assert workbench._last_audit_entries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_branch", ["access", "read"])
async def test_test_tool_audit_sync_log_preserves_nonfilesystem_diagnostics(
    failure_branch: str, caplog: pytest.LogCaptureFixture
):
    app = ToolTestApp()
    message = (
        r"See https://example.test/docs/private/file.txt at 12:34/56; "
        r"pattern \\d+; relative docs/private.txt."
    )
    failure = RuntimeError(message)

    class AccessFailureService:
        @property
        def execution_log(self):
            raise failure

    class ReadFailureLog:
        def read_recent(self, _limit: int):
            raise failure

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        app.unified_mcp_service = (
            AccessFailureService()
            if failure_branch == "access"
            else SimpleNamespace(execution_log=ReadFailureLog())
        )
        caplog.clear()
        sink = mcp_workbench_module.logger.add(
            caplog.handler, level="WARNING", format="{message}"
        )
        try:
            await workbench._sync_audit_log_entries()
        finally:
            mcp_workbench_module.logger.remove(sink)

        prefix = f"MCP execution log {failure_branch} failed"
        rendered = "".join(message for message in caplog.messages if prefix in message)
        assert "https://example.test/docs/private/file.txt" in rendered
        assert "12:34/56" in rendered
        assert r"\\d+" in rendered
        assert "docs/private.txt" in rendered
        assert "[path]" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_branch", ["access", "read"])
async def test_test_tool_audit_sync_log_preserves_diagnostics_after_initial_path(
    failure_branch: str, caplog: pytest.LogCaptureFixture
):
    app = ToolTestApp()
    failure = RuntimeError(
        r"failed at /Users/alice/Private Project/credentials.json and see "
        r"docs/recovery.md with pattern \\d+; visit https://example.test/help."
    )

    class AccessFailureService:
        @property
        def execution_log(self):
            raise failure

    class ReadFailureLog:
        def read_recent(self, _limit: int):
            raise failure

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        app.unified_mcp_service = (
            AccessFailureService()
            if failure_branch == "access"
            else SimpleNamespace(execution_log=ReadFailureLog())
        )
        caplog.clear()
        sink = mcp_workbench_module.logger.add(
            caplog.handler, level="WARNING", format="{message}"
        )
        try:
            await workbench._sync_audit_log_entries()
        finally:
            mcp_workbench_module.logger.remove(sink)

        prefix = f"MCP execution log {failure_branch} failed"
        rendered = "".join(message for message in caplog.messages if prefix in message)
        assert "Users/alice" not in rendered
        assert "Project/credentials.json" not in rendered
        assert "docs/recovery.md" in rendered
        assert r"\\d+" in rendered
        assert "https://example.test/help" in rendered


@pytest.mark.asyncio
async def test_test_tool_typed_failure_outcome_is_redacted_and_bounded():
    app = ToolTestApp()
    app.unified_mcp_service.next_prepared_outcome = LocalHubExecutionOutcome(
        decision="allowed",
        status="failed",
        error_category="provider_error",
        final_gate="allow",
        approval_consumed=False,
        dispatch_started=True,
        provider_terminal="error",
        duration_ms=4,
        result=ToolResult(
            ok=False,
            error=(
                "api_key=sk-live-outcome-secret at /Users/alice/private/key.txt "
                + ("x" * 4_000)
            ),
        ),
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        await pilot.click("#mcp-inspector-test-run")
        await app.workers.wait_for_complete()
        await pilot.pause()

        rendered = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "sk-live-outcome-secret" not in rendered
        assert "/Users/alice" not in rendered
        assert "[redacted]" in rendered
        assert "[path]" in rendered
        assert len(rendered) <= 560


async def _open_fetch_test_preview(app: ToolTestApp, pilot) -> str:
    workbench = app.query_one(MCPWorkbench)
    workbench.set_mode("tools")
    await pilot.pause()
    await _select_tools_mode_row(app, pilot, 0)
    await pilot.click("#mcp-inspector-test-tool")
    await app.workers.wait_for_complete()
    await pilot.pause()
    preview = app.query_one(MCPInspector)._test_preview
    assert preview is not None
    return preview.nonce


async def _wait_for_test_button_label(app: ToolTestApp, pilot, label: str) -> Button:
    """Wait for the worker-driven preview render, checking observable state."""
    for _ in range(40):
        buttons = list(app.query("#mcp-inspector-test-run"))
        if buttons and str(buttons[0].label) == label:
            return buttons[0]
        await pilot.pause()
    pytest.fail(f"Test Tool button never reached {label!r}")


async def _wait_for_tools_rows(workbench: MCPWorkbench, pilot) -> DataTable:
    """Wait for the remounted tools canvas and its worker-fed rows."""
    for _ in range(40):
        await workbench.app.workers.wait_for_complete()
        tables = list(workbench.query("#mcp-tools-table"))
        if tables and tables[0].row_count:
            return tables[0]
        await pilot.pause()
    pytest.fail("Remounted tools table never populated")


@pytest.mark.asyncio
async def test_test_tool_preview_escape_revokes_nonce_through_mounted_binding():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        nonce = await _open_fetch_test_preview(app, pilot)
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert nonce in app.unified_mcp_service.revoked_nonces
        assert not list(app.query("#mcp-inspector-test-panel"))


@pytest.mark.asyncio
async def test_test_tool_preview_mode_switch_revokes_nonce_while_mounted():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        nonce = await _open_fetch_test_preview(app, pilot)
        app.query_one(MCPWorkbench).set_mode("permissions")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert nonce in app.unified_mcp_service.revoked_nonces


@pytest.mark.asyncio
async def test_test_tool_preview_source_switch_revokes_nonce_while_mounted():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        nonce = await _open_fetch_test_preview(app, pilot)
        source = app.query_one("#mcp-rail-source", Select)
        source.value = "server"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert nonce in app.unified_mcp_service.revoked_nonces


@pytest.mark.asyncio
async def test_test_tool_preview_unmount_and_remount_revokes_old_nonce():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        nonce = await _open_fetch_test_preview(app, pilot)
        old = app.query_one(MCPWorkbench)
        app.set_focus(None)
        await pilot.pause()
        await old.remove()
        await pilot.pause()
        assert not old.is_attached
        app.unified_mcp_service._active_tests.add(("local:docs", "fetch"))
        preview_count = app.unified_mcp_service._preview_count
        await app.mount(MCPWorkbench(app_instance=app, id="mcp-workbench-remounted"))
        await pilot.pause()
        remounted = app.query_one("#mcp-workbench-remounted", MCPWorkbench)
        await _wait_for_tools_rows(remounted, pilot)
        remounted.set_mode("tools")
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        button = await _wait_for_test_button_label(app, pilot, "Running…")

        assert nonce in app.unified_mcp_service.revoked_nonces
        assert remounted.is_mounted
        assert button.disabled is True
        assert app.unified_mcp_service._preview_count == preview_count
        assert app.unified_mcp_service.active_calls[-1] == ("local:docs", "fetch")


@pytest.mark.asyncio
async def test_test_tool_delayed_preview_cannot_update_closed_stale_generation():
    app = ToolTestApp()
    started = threading.Event()
    release = threading.Event()
    original_prepare = app.unified_mcp_service.prepare_hub_test

    def delayed_prepare(tool: HubTool) -> ToolTestAdmissionPreview:
        started.set()
        assert release.wait(timeout=2)
        return original_prepare(tool)

    app.unified_mcp_service.prepare_hub_test = delayed_prepare
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        assert await asyncio.to_thread(started.wait, 1)
        await pilot.press("escape")
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not list(app.query("#mcp-inspector-test-panel"))
        assert "preview-1" in app.unified_mcp_service.revoked_nonces


@pytest.mark.asyncio
async def test_test_tool_cancelled_unmount_retrieves_and_revokes_late_minted_nonce():
    app = ToolTestApp()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_prepare = app.unified_mcp_service.prepare_hub_test

    def delayed_prepare(tool: HubTool) -> ToolTestAdmissionPreview:
        started.set()
        assert release.wait(timeout=2)
        preview = original_prepare(tool)
        finished.set()
        return preview

    app.unified_mcp_service.prepare_hub_test = delayed_prepare
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        assert await asyncio.to_thread(started.wait, 1)
        worker = next(
            worker for worker in app.workers if worker.name == "mcp-tool-test-preview"
        )

        removal = workbench.remove()
        for _ in range(40):
            if worker.is_cancelled:
                break
            await pilot.pause()
        assert worker.is_cancelled
        release.set()
        await removal
        assert await asyncio.to_thread(finished.wait, 1)
        for _ in range(40):
            if app.unified_mcp_service.revoked_nonces:
                break
            await pilot.pause()

        assert app.unified_mcp_service._previews == {}
        assert app.unified_mcp_service.revoked_nonces == ["preview-1"]
        assert not list(app.query("#mcp-inspector-test-panel"))


@pytest.mark.asyncio
async def test_test_tool_repeated_cancellation_reclaims_abandoned_mint_and_tasks():
    app = ToolTestApp()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_prepare = app.unified_mcp_service.prepare_hub_test
    calls = 0

    def first_prepare_blocks(tool: HubTool) -> ToolTestAdmissionPreview:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=3)
        preview = original_prepare(tool)
        if calls >= 2:
            finished.set()
        return preview

    app.unified_mcp_service.prepare_hub_test = first_prepare_blocks
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        assert await asyncio.to_thread(started.wait, 1)
        abandoned_worker = next(
            worker for worker in app.workers if worker.name == "mcp-tool-test-preview"
        )

        await _select_tools_mode_row(app, pilot, 1)
        await pilot.click("#mcp-inspector-test-tool")
        await _wait_for_test_button_label(app, pilot, "Run")
        removal = workbench.remove()
        for _ in range(5):
            abandoned_worker.cancel()
        release.set()
        await removal
        assert await asyncio.to_thread(finished.wait, 1)

        for _ in range(60):
            live_mints = [
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith("mcp-tool-test-preview-mint:")
                and not task.done()
            ]
            if not app.unified_mcp_service._previews and not live_mints:
                break
            await pilot.pause()

        assert app.unified_mcp_service._previews == {}
        assert set(app.unified_mcp_service.revoked_nonces) == {
            "preview-1",
            "preview-2",
        }
        assert not live_mints
        assert not workbench._tool_test_reclaim_tasks
        assert not workbench.is_attached


@pytest.mark.asyncio
async def test_test_tool_active_watcher_polling_is_bounded_and_stops_on_unmount():
    app = ToolTestApp()
    key = ("local:docs", "fetch")
    app.unified_mcp_service._active_tests.add(key)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await _wait_for_test_button_label(app, pilot, "Running…")
        calls_at_running = len(app.unified_mcp_service.active_calls)

        await pilot.pause(0.75)
        additional_polls = len(app.unified_mcp_service.active_calls) - calls_at_running
        assert 1 <= additional_polls <= 4

        await workbench.remove()
        calls_at_unmount = len(app.unified_mcp_service.active_calls)
        await pilot.pause(0.4)
        assert len(app.unified_mcp_service.active_calls) == calls_at_unmount
        app.unified_mcp_service._active_tests.discard(key)


@pytest.mark.asyncio
async def test_test_tool_rapid_reopen_cleans_cancelled_mint_without_stale_update():
    app = ToolTestApp()
    first_started = threading.Event()
    first_release = threading.Event()
    original_prepare = app.unified_mcp_service.prepare_hub_test
    calls = 0

    def first_prepare_blocks(tool: HubTool) -> ToolTestAdmissionPreview:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert first_release.wait(timeout=2)
        return original_prepare(tool)

    app.unified_mcp_service.prepare_hub_test = first_prepare_blocks
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        assert await asyncio.to_thread(first_started.wait, 1)
        first_worker = next(
            worker for worker in app.workers if worker.name == "mcp-tool-test-preview"
        )

        await _select_tools_mode_row(app, pilot, 1)
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        for _ in range(40):
            buttons = list(app.query("#mcp-inspector-test-run"))
            if buttons and str(buttons[0].label) == "Run":
                break
            await pilot.pause()
        assert str(app.query_one("#mcp-inspector-test-run", Button).label) == "Run"
        current_nonce = app.query_one(MCPInspector)._test_preview.nonce
        assert first_worker.is_cancelled

        first_release.set()
        for _ in range(40):
            if len(app.unified_mcp_service.revoked_nonces) == 1:
                break
            await pilot.pause()

        assert app.query_one(MCPInspector)._test_preview.nonce == current_nonce
        assert set(app.unified_mcp_service._previews) == {current_nonce}
        assert app.unified_mcp_service.revoked_nonces == ["preview-2"]


@pytest.mark.asyncio
async def test_test_tool_active_watcher_recovers_ready_and_preserves_arguments():
    app = ToolTestApp()
    key = ("local:docs", "search")
    app.unified_mcp_service._active_tests.add(key)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)
        await pilot.click("#mcp-inspector-test-tool")
        button = await _wait_for_test_button_label(app, pilot, "Running…")
        field = app.query_one("#mcp-schema-field-0", Input)
        field.value = "preserve while active"

        app.unified_mcp_service._active_tests.discard(key)
        for _ in range(80):
            if str(button.label) == "Run":
                break
            await pilot.pause()

        assert str(button.label) == "Run"
        assert button.disabled is False
        assert field.value == "preserve while active"
        assert app.query_one(MCPInspector)._test_preview is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("leave_by", ["switch", "unmount"])
async def test_test_tool_active_watcher_never_updates_stale_panel(leave_by: str):
    app = ToolTestApp()
    key = ("local:docs", "fetch")
    app.unified_mcp_service._active_tests.add(key)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        await pilot.click("#mcp-inspector-test-tool")
        await _wait_for_test_button_label(app, pilot, "Running…")

        if leave_by == "switch":
            await _select_tools_mode_row(app, pilot, 1)
        else:
            await workbench.remove()
        app.unified_mcp_service._active_tests.discard(key)
        for _ in range(20):
            await pilot.pause()

        assert app.unified_mcp_service._previews == {}
        if leave_by == "switch":
            inspector = app.query_one(MCPInspector)
            assert inspector.current_tool is not None
            assert inspector.current_tool.name == "search"
            assert not list(inspector.query("#mcp-inspector-test-panel"))
        else:
            assert not workbench.is_attached


@pytest.mark.asyncio
async def test_test_tool_retry_preview_uses_service_and_preserves_form_values():
    app = ToolTestApp()
    original_prepare = app.unified_mcp_service.prepare_hub_test
    attempts = 0

    def fail_once(tool: HubTool) -> ToolTestAdmissionPreview:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Preview service is temporarily unavailable.")
        return original_prepare(tool)

    app.unified_mcp_service.prepare_hub_test = fail_once
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        field = app.query_one("#mcp-schema-field-0", Input)
        field.value = "keep through retry"
        retry = app.query_one("#mcp-inspector-test-retry", Button)
        assert retry.display is True

        retry.focus()
        await pilot.press("enter")
        button = await _wait_for_test_button_label(app, pilot, "Run")

        assert attempts == 2
        assert button.disabled is False
        assert field.value == "keep through retry"
        assert retry is app.query_one("#mcp-inspector-test-retry", Button)
        assert retry.display is False


@pytest.mark.asyncio
async def test_test_tool_one_click_builtin_uses_prepared_entry_point():
    app = BuiltinToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        tool = next(
            tool
            for tool in workbench._last_hub_tools
            if tool.server_key == "builtin:tldw_chatbook" and tool.name == "calculator"
        )
        table = app.query_one("#mcp-tools-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(tool.tool_id))
        await pilot.press("enter")
        await pilot.click("#mcp-inspector-test-tool")
        await app.workers.wait_for_complete()
        app.query_one("#mcp-schema-field-0", Input).value = "1"
        await pilot.click("#mcp-inspector-test-run")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(app.unified_mcp_service.prepared_calls) == 1
        assert app.unified_mcp_service.prepared_calls[0][1:] == ("run", {"x": 1})


# -- Task 4 (PR-T3): the dispatch path threads the tool's schema-approved
# argument names through to `test_hub_tool()` -- before this task NO caller
# in the tree supplied `registered_argument_names`, so every audit row
# recorded `argument_names: []` regardless of what was actually called.


@pytest.mark.asyncio
async def test_tool_test_dispatch_supplies_registered_argument_names_from_schema():
    """`docs::search` (row 1) carries `inputSchema.properties: {"query": ...}`
    -- `on_mcp_inspector_tool_test_requested()` must resolve that schema via
    `_tool_for()` and forward its property names to `test_hub_tool()`, not
    the pre-Task-4 `None`/omitted default that always logged `[]`."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "search")
        event = _prepared_test_event(app.unified_mcp_service, tool, {"query": "hello"})
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.registered_argument_names_calls == [{"query"}]


@pytest.mark.asyncio
async def test_tool_test_dispatch_no_schema_supplies_empty_registered_argument_names():
    """`docs::fetch` (row 0) has no `inputSchema` -- the derived set is
    empty, never `None`/omitted, so the execution log records a real
    (if empty) provenance check rather than silently skipping it."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "fetch")
        event = _prepared_test_event(app.unified_mcp_service, tool, {})
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.registered_argument_names_calls == [set()]


@pytest.mark.asyncio
async def test_slow_tool_result_does_not_render_under_a_different_selected_tool():
    """I1: tool A's ("docs::fetch") slow test run must not land in tool B's
    ("notes::list_notes") panel when the user switches selection before A
    resolves -- and must not re-enable B's Run button on A's behalf.
    Mirrors the whole-branch review's end-to-end probe."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        workbench.set_mode("tools")
        await pilot.pause()

        # Select docs::fetch (row 0, raw mode), open Test panel, Run (gated).
        await _select_tools_mode_row(app, pilot, 0)
        app.query_one("#mcp-inspector-test-tool", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        service.test_gate = asyncio.Event()
        service.test_result = {"result": "FETCH-DOC-PAYLOAD"}
        app.query_one("#mcp-inspector-test-run", Button).press()
        await pilot.pause()
        assert service.test_calls and service.test_calls[0][1] == "fetch"

        # While fetch is in flight, select notes::list_notes and open ITS panel.
        await _select_tools_mode_row(app, pilot, 2)
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        assert inspector.current_tool.name == "list_notes"
        app.query_one("#mcp-inspector-test-tool", Button).press()
        await pilot.pause()
        await pilot.pause()
        result_widget = app.query_one("#mcp-inspector-test-result", Static)
        run_button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(result_widget.renderable) == ""

        # Release the gate: fetch's late result must be dropped, not shown
        # under notes::list_notes, and must not touch list_notes's own Run.
        service.test_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(result_widget.renderable)
        assert "FETCH-DOC-PAYLOAD" not in rendered
        assert rendered == ""
        assert inspector.current_tool.name == "list_notes"
        assert run_button.disabled is False  # never pressed for list_notes


@pytest.mark.asyncio
async def test_test_tool_run_non_str_dict_key_result_does_not_crash():
    """Critical regression: `_run_tool_test()`'s success-path result
    formatting (`json.dumps(redact_mapping(result), default=str)`) used to
    sit OUTSIDE the inner try/except. A result dict with a non-str key (a
    tuple, here) makes `json.dumps` raise `TypeError` -- `default=str` only
    covers values, not keys -- and that exception used to escape the worker
    body entirely. Textual's `run_worker()` defaults to `exit_on_error=True`,
    so an uncaught exception there panics the whole app rather than just
    failing this one tool test. After the fix, formatting errors must be
    caught and rendered as a failed result like any other test failure."""
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {("tuple", "key"): 1}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False
        assert app.unified_mcp_service.hub_test_active("local:docs", "fetch") is False


@pytest.mark.asyncio
async def test_test_tool_run_non_mapping_result_str_raises_does_not_crash():
    """Review fix (RAG-49, Important #2): the non-mapping envelope fallback
    (`str(envelope)[:500]`) must share the SAME try/except as the mapping
    branch's `json.dumps(redact_mapping(...))` step.

    Before this fix, the non-mapping `else` branch's `str(envelope)[:500]`
    sat OUTSIDE any try/except -- a non-mapping envelope whose own
    `__str__` raises would escape `_run_tool_test()` entirely uncaught.
    Textual's `run_worker()` defaults to `exit_on_error=True`, so that
    would panic the whole app rather than just failing this one tool
    test -- the exact regression class `test_test_tool_run_non_str_dict_
    key_result_does_not_crash` above already pins for the MAPPING branch.
    This is the non-mapping-branch sibling of that same pin."""

    class _RaisingStr:
        def __str__(self) -> str:
            raise ValueError("cannot stringify this result")

    app = ToolTestApp()
    app.unified_mcp_service.test_result = _RaisingStr()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False
        assert app.unified_mcp_service.hub_test_active("local:docs", "fetch") is False


@pytest.mark.asyncio
async def test_render_failure_in_show_tool_test_result_notifies_instead_of_only_logging(
    monkeypatch,
):
    """Task 3 (PR-T3): `_show_tool_test_result()`'s try/except around
    `MCPInspector.show_tool_result()` used to only `logger.warning()` a
    render failure -- the run genuinely completed (this test's fake
    service returns a normal OK result), but a render bug meant the user
    saw literally nothing. It must also toast, redact the diagnostic log,
    and let the fresh service preview restore the action presentation."""
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {"ok": True}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        run_button = app.query_one("#mcp-inspector-test-run", Button)

        secret = "sk-live-render-log-secret"
        private_path = "/Users/alice/private/render.json"

        def _raise(*args, **kwargs):
            raise RuntimeError(
                f"token={secret} exploded at {private_path} " + ("x" * 4_000)
            )

        monkeypatch.setattr(MCPInspector, "show_tool_result", _raise)

        messages: list[object] = []
        sink = mcp_workbench_module.logger.add(
            messages.append, level="WARNING", format="{message}"
        )
        try:
            await pilot.click(run_button)
            await app.workers.wait_for_complete()
            await pilot.pause()
        finally:
            mcp_workbench_module.logger.remove(sink)

        assert any(
            "search" in msg and severity == "error" for msg, severity in notifications
        ), f"expected an error toast naming the tool, got: {notifications!r}"
        assert run_button.disabled is False
        rendered_log = "".join(
            str(message)
            for message in messages
            if "MCP tool test result render failed" in str(message)
        )
        assert secret not in rendered_log
        assert private_path not in rendered_log
        assert "[redacted]" in rendered_log
        assert "[redacted]]" not in rendered_log
        assert "[path]" in rendered_log
        assert len(rendered_log) <= 560


@pytest.mark.asyncio
async def test_collect_arguments_value_error_does_not_call_service():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (required "query")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        # required "query" field left empty
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        assert app.unified_mcp_service.test_calls == []
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "required" in result


@pytest.mark.asyncio
async def test_raw_mode_tool_run_posts_parsed_json_to_service():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 3
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw mode)
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        raw_area = app.query_one("#mcp-schema-raw", TextArea)
        raw_area.text = '{"id": 42}'
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("local:docs", "fetch", {"id": 42}) in app.unified_mcp_service.test_calls


# -- Task 8: `t` keybinding entry point (open_test_for_selected_tool) --------


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_no_selection_notifies():
    """T8: the `t` keybinding's workbench entry point -- with nothing
    selected in the inspector's tool-detail view, notifies instead of
    silently no-opping, mirroring `open_add_server_form()`'s T13 rationale
    for a keybinding that can reach a state no disabled button gates.

    F-055: it must NOT hijack the mode to get there -- the old
    switch-first behavior force-landed the user in Tools mode with a
    "Select a tool first." toast on top. Now the mode stays put and the
    hint says where the working key lives.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "servers"
        assert not list(app.query("#mcp-inspector-test-panel"))
        assert notifications
        message, severity = notifications[-1]
        assert message == "Select a tool in Tools mode first."
        assert severity == "warning"


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_selection_opens_panel():
    """T8: with a tool already selected in the inspector (Tools mode,
    row already clicked), `open_test_for_selected_tool()` opens the SAME
    Test Tool panel the button's own press handler mounts
    (`MCPInspector._mount_test_tool_panel()`, reused via
    `MCPInspector.open_test_panel()`) -- not a second, duplicate mount path.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (form schema)

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        panel = app.query_one("#mcp-inspector-test-panel", Vertical)
        assert panel.display is not False
        assert app.query_one("#mcp-inspector-test-tool", Button).disabled is True
        # The mounted panel carries the selected tool's schema-driven form,
        # not a blank/duplicate one.
        assert app.query_one("#mcp-inspector-test-form")


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_does_not_duplicate_already_open_panel():
    """T8: pressing `t` a second time while the panel is already open (same
    tool still selected) must not raise `DuplicateIds` -- relies on
    `_mount_test_tool_panel()`'s own existence-check guard, exercised here
    through the keybinding's entry point rather than the button.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch

        await workbench.open_test_for_selected_tool()
        await pilot.pause()
        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert len(list(app.query("#mcp-inspector-test-panel"))) == 1


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_non_executable_selection_notifies_phase_note():
    """T8 regression: `MCPInspector.open_test_panel()` used to return the
    same `False` for BOTH "nothing selected" and "a tool IS selected but
    isn't executable" (server-source), so `open_test_for_selected_tool()`
    notified "Select a tool first." for both -- misleading when a tool is
    in fact selected. With a server-source tool selected (never executable
    -- see `server_tools_from_inventory`), the `t` keybinding must notify
    with the SAME copy the inline detail view already shows for that tool
    (`mcp_inspector.py`'s "Server-source tools are display-only." `Static`),
    not the generic no-selection message, and must not mount a test panel
    (there is no schema-driven form to open for a tool that can't be
    invoked).
    """
    app = ServerToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # server:main/docs::search
        assert app.query_one("#mcp-inspector-tool-phase-note", Static)

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        assert not list(app.query("#mcp-inspector-test-panel"))
        assert notifications
        message, severity = notifications[-1]
        assert message == "Server-source tools are display-only."
        assert severity == "information"


# -- Task 6: Permissions mode canvas# -- Task 6: Permissions mode canvas (matrix, kill switch, policy preview) --
#
# `PermissionsHubService` is wired against a REAL `MCPPermissionStore` (T1)
# and the REAL `resolve_effective_state()` (T2) rather than a hand-rolled
# mock -- an accessor mock here would hide exactly the kind of drift a
# reviewer can't easily catch (see the "unmocked-integration-test" lesson
# from task-222/223: mocked config surfaces have shipped inert before).
# Seeds two local profiles -- "docs" (tools "search"/"fetch") and "notes"
# (tool "list_notes") -- so grouping/sorting is actually exercised: server
# labels "docs" < "notes"; within "docs", tools "fetch" < "search".


class PermissionsHubService(FakeHubService):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self._store = MCPPermissionStore(store_path)

    @property
    def permission_store(self) -> MCPPermissionStore:
        return self._store

    def effective_tool_states(self, tools):
        payload = self._store.load()
        return {
            (t.server_key, t.name): resolve_effective_state(payload, t) for t in tools
        }

    def set_tool_state(self, server_key, tool_name, ui_state, *, tool=None):
        # Mirrors `UnifiedControlPlaneService.set_tool_state()`'s
        # HASH_FREE_SERVER_KEYS exemption (Task 1) -- `agent:builtin`
        # doesn't need a `HubTool` to fingerprint an "allow".
        hash_value = None
        if ui_state == "allow" and server_key not in HASH_FREE_SERVER_KEYS:
            if tool is None:
                raise ValueError("tool is required to set state 'allow'")
            hash_value = definition_hash(tool.description, tool.input_schema)
        self._store.set_tool_state(
            server_key, tool_name, ui_state, definition_hash=hash_value
        )

    def set_server_default(self, server_key, state):
        self._store.set_server_default(server_key, state)

    def set_global_default(self, state):
        self._store.set_global_default(state)

    def get_kill_switch(self):
        return self._store.get_kill_switch()

    def set_kill_switch(self, value):
        self._store.set_kill_switch(value)

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [
                                {"name": "search", "description": "Search docs."},
                                {"name": "fetch", "description": "Fetch a doc."},
                            ],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                    {
                        "profile_id": "notes",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [
                                {"name": "list_notes", "description": "List notes."}
                            ],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                ]
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }


class PermissionsApp(ConsolidatedCSSApp):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self.unified_mcp_service = PermissionsHubService(store_path)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


def _perm_table_texts(app: App, row_index: int) -> list[str]:
    table = app.query_one("#mcp-perm-table", DataTable)
    row = table.get_row_at(row_index)
    return [cell.plain if hasattr(cell, "plain") else str(cell) for cell in row]


def _perm_all_rows(app: App) -> list[list[str]]:
    """Every rendered `#mcp-perm-table` row's cell texts, in table order --
    TASK-627 Task 3's tests use this to locate the built-in section without
    hard-coding its row index against whatever MCP section a given fixture
    also renders."""
    table = app.query_one("#mcp-perm-table", DataTable)
    return [_perm_table_texts(app, i) for i in range(table.row_count)]


def _perm_row_keys(app: App) -> list[str]:
    table = app.query_one("#mcp-perm-table", DataTable)
    return [
        table.coordinate_to_cell_key((i, 0))[0].value for i in range(table.row_count)
    ]


def test_tool_state_label_marker_precedence():
    """`MCPWorkbench._tool_state_label()`'s marker selection, pinned
    directly: config_changed -> "⚠", risk_floored -> "⚑", a plain
    tool_override -> "•", any other origin (inherited, undowngraded) ->
    no marker at all. Local-source `HubTool`s never carry tags (see
    `hub_tool_catalog.local_tools_from_record()`), so a risk-floored
    scenario can't be reached end-to-end through `PermissionsHubService`
    above -- this pins the marker logic itself, independent of how a
    real `EffectiveToolState` gets constructed."""
    label = MCPWorkbench._tool_state_label
    assert label(EffectiveToolState(state="allow", origin="tool_override")) == "Allow •"
    assert label(EffectiveToolState(state="ask", origin="server_default")) == "Ask"
    assert label(EffectiveToolState(state="ask", origin="global_default")) == "Ask"
    assert (
        label(
            EffectiveToolState(state="ask", origin="tool_override", config_changed=True)
        )
        == "Ask ⚠"
    )
    assert (
        label(
            EffectiveToolState(state="ask", origin="server_default", risk_floored=True)
        )
        == "Ask ⚑"
    )


@pytest.mark.asyncio
async def test_permissions_mode_renders_pinned_grouped_sorted_matrix(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-permissions"

        table = app.query_one("#mcp-perm-table", DataTable)
        # TASK-627 Task 3: 6 MCP rows + the built-in section (server default,
        # calculator, get_current_datetime), appended AFTER the MCP sections.
        assert table.row_count == 9
        assert _perm_table_texts(app, 0) == ["Global default", "Ask"]
        assert _perm_table_texts(app, 1) == ["Server default — docs", "Ask"]
        assert _perm_table_texts(app, 2) == ["  fetch", "Ask"]
        assert _perm_table_texts(app, 3) == ["  search", "Ask"]
        assert _perm_table_texts(app, 4) == ["Server default — notes", "Ask"]
        assert _perm_table_texts(app, 5) == ["  list_notes", "Ask"]
        assert _perm_table_texts(app, 6) == [
            "Server default — Built-in (agent runtime)",
            "Allow",
        ]
        assert _perm_table_texts(app, 7) == ["  calculator", "Allow"]
        assert _perm_table_texts(app, 8) == ["  get_current_datetime", "Allow"]

        expected_keys = [
            "__global__",
            "__server__::local:docs",
            "local:docs::fetch",
            "local:docs::search",
            "__server__::local:notes",
            "local:notes::list_notes",
            "__server__::agent:builtin",
            "agent:builtin::calculator",
            "agent:builtin::get_current_datetime",
        ]
        for index, expected_key in enumerate(expected_keys):
            row_key, _ = table.coordinate_to_cell_key((index, 0))
            assert row_key.value == expected_key

        preview = app.query_one("#mcp-perm-preview", Static)
        # Fix 2 (PR #906 review): the preview's override count now DOES
        # include the built-in section -- but this fixture stores no
        # built-in override at all (server/tool rows above all inherit),
        # so the sentence is unchanged from before that fix.
        assert str(preview.renderable) == "global default: ask"


# -- TASK-627 Task 3: agent-runtime built-in section in Permissions mode ----
#
# The built-in section is derived from `builtin_permission_rows()` (Task 2)
# and `resolve_builtin_state` -- NEVER from `_build_permission_rows()`'s own
# `tools`/`effective_tool_states()` path (Constraint 1). These tests exercise
# `MCPWorkbench._sync_permissions_mode()` end to end, the same way the
# `PermissionsHubService` suite above does.


class EmptyCatalogHubService(FakeHubService):
    """A LOCAL-source service exposing a genuinely EMPTY catalog --
    `FakeHubService.load_section("external_servers")` seeds a "docs"
    profile with tool "a", which would let `_last_hub_tools` end up
    non-empty even under a test that claims to cover "no MCP servers".
    This override returns `[]` instead, so `_collect_hub_tools()` really
    does produce zero tools and the built-in section's headline claim (it
    renders independently of the MCP catalog) is actually exercised."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return []
            return {"source": "local", "section": effective_section}
        return {
            "external_servers": [],
            "source": "server",
            "section": "external_servers",
        }


class EmptyCatalogApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = EmptyCatalogHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_permissions_mode_shows_builtin_section_with_no_mcp_servers():
    """The built-in section must not depend on `_collect_hub_tools()`/the
    MCP catalog -- it renders even when the catalog is GENUINELY empty
    (`_last_hub_tools == []`, via `EmptyCatalogHubService` above, not
    merely a fixture whose seeded MCP rows happen to go unasserted) and
    the service exposes no `permission_store`/`effective_tool_states` seam
    at all (`FakeHubService`'s base shape, same as `test_permissions_mode_
    renders_fail_soft_without_t4_seams`)."""
    app = EmptyCatalogApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        # The claim this test exists to pin: the MCP catalog is truly empty.
        assert workbench._last_hub_tools == []

        rows = _perm_all_rows(app)
        # Only the pinned global row plus the built-in section -- zero MCP
        # server/tool rows of any kind.
        assert [row[0] for row in rows] == [
            "Global default",
            "Server default — Built-in (agent runtime)",
            "  calculator",
            "  get_current_datetime",
        ]
        tool_cells = {row[0].strip(): row[1] for row in rows}
        # Untagged built-ins resolve to the built-in ALLOW floor, not MCP's
        # "Ask" default -- proves `resolve_builtin_state` (not the MCP
        # resolver) produced this label.
        assert tool_cells["calculator"] == "Allow"
        assert tool_cells["get_current_datetime"] == "Allow"


class _FakeLocalServiceWithInventory:
    """A minimal `local_service` seam -- `_collect_hub_tools()` only reads
    `get_inventory()` off it -- so `builtin:tldw_chatbook` (the built-in MCP
    *server*) actually appears in `_last_hub_tools` alongside the
    `agent:builtin` section this task adds, letting a test assert the two
    render as genuinely distinct groups rather than merely both existing."""

    def get_inventory(self):
        return {"tools": [{"name": "search_web", "description": "Search the web."}]}


class BuiltinDistinctHubService(PermissionsHubService):
    def __init__(self, store_path: Path) -> None:
        super().__init__(store_path)
        self.local_service = _FakeLocalServiceWithInventory()


class BuiltinDistinctApp(ConsolidatedCSSApp):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self.unified_mcp_service = BuiltinDistinctHubService(store_path)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_builtin_section_is_distinct_from_the_builtin_mcp_server(tmp_path):
    """Constraint 3: `agent:builtin` (the in-process agent-runtime built-ins
    this task renders) must never be grouped with, or labeled the same as,
    `builtin:tldw_chatbook` (the built-in MCP *server*, exposed here via
    `local_service.get_inventory()`)."""
    app = BuiltinDistinctApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        row_keys = _perm_row_keys(app)
        assert "__server__::builtin:tldw_chatbook" in row_keys
        assert "__server__::agent:builtin" in row_keys

        rows = _perm_all_rows(app)
        server_row_labels = {
            row[0] for row in rows if row[0].startswith("Server default")
        }
        mcp_builtin_label = "Server default — tldw_chatbook"
        agent_builtin_label = "Server default — Built-in (agent runtime)"
        assert mcp_builtin_label in server_row_labels
        assert agent_builtin_label in server_row_labels
        assert mcp_builtin_label != agent_builtin_label

        # Neither tool list bleeds into the other's row-key namespace.
        assert "builtin:tldw_chatbook::calculator" not in row_keys
        assert "agent:builtin::search_web" not in row_keys


@pytest.mark.asyncio
async def test_stored_deny_for_builtin_renders_off_with_tool_override_marker(tmp_path):
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        BUILTIN_TOOL_SERVER_KEY, "calculator", "deny"
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        rows = _perm_all_rows(app)
        tool_cells = {row[0].strip(): row[1] for row in rows}
        assert tool_cells["calculator"] == "Off •"
        # Sibling built-in tool is untouched by calculator's own override.
        assert tool_cells["get_current_datetime"] == "Allow"


@pytest.mark.asyncio
async def test_orphaned_builtin_row_is_marked_via_tags_cell_not_tool_name(tmp_path):
    """A stored decision for a built-in tool name no live tool provides
    must still be listed (`orphaned=True`, Task 2) so the user can clear
    it -- marked via its Tags cell ("orphaned"), never by decorating the
    Tool cell/`tool_name` itself: `_row_key()`/`action_cycle_state()`
    (mcp_permissions_mode.py) read `PermRow.tool_name` verbatim as the
    tool's store identity, so a decorated name would corrupt the row
    Task 4's write-back path needs to address."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        BUILTIN_TOOL_SERVER_KEY, "tool_that_no_longer_exists", "allow"
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        row_keys = _perm_row_keys(app)
        assert "agent:builtin::tool_that_no_longer_exists" in row_keys
        index = row_keys.index("agent:builtin::tool_that_no_longer_exists")
        cells = _perm_table_texts(app, index)
        # Tool cell carries the RAW, undecorated stored name -- not
        # "tool_that_no_longer_exists (orphaned)" or similar.
        assert cells[0].strip() == "tool_that_no_longer_exists"
        # The Tags cell (not the Tool cell) carries the orphaned marker.
        assert cells[-1] == "orphaned"

        # A live built-in tool's row is untouched and NOT marked orphaned.
        calculator_index = row_keys.index("agent:builtin::calculator")
        calculator_cells = _perm_table_texts(app, calculator_index)
        assert calculator_cells[-1] != "orphaned"


@pytest.mark.asyncio
async def test_builtin_enumeration_failure_still_shows_a_cyclable_server_default_row(
    tmp_path, monkeypatch
):
    """Review finding: `_builtin_permission_matrix_rows()` must not hide
    the pinned "Server default — Built-in (agent runtime)" row when
    `builtin_permission_rows()` itself fails -- a user's stored built-in
    server default would otherwise become invisible and impossible to
    clear/cycle. Only the per-tool rows beneath it are conditional on
    enumeration succeeding."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_server_default(BUILTIN_TOOL_SERVER_KEY, "deny")

    def _raise(payload):
        raise RuntimeError("builtin registry unavailable")

    monkeypatch.setattr(mcp_workbench_module, "builtin_permission_rows", _raise)

    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        row_keys = _perm_row_keys(app)
        assert "__server__::agent:builtin" in row_keys
        # No tool rows -- enumeration failed -- but the server row survives.
        assert not [key for key in row_keys if key.startswith("agent:builtin::")]

        rows = _perm_all_rows(app)
        server_row = next(
            row for row in rows if row[0] == "Server default — Built-in (agent runtime)"
        )
        # The stored server-level "deny" default is still visible (and, per
        # the "•" override marker, still cyclable back to Inherit).
        assert server_row[1] == "Off •"


# -- TASK-627 Task 4: persisting built-in tool decisions from the UI --------
#
# `on_mcp_permissions_mode_state_cycle_requested()` used to resolve EVERY
# "tool" row's `HubTool` via `_tool_for()` -- which only ever searches
# `_last_hub_tools`, the MCP catalog. Built-in tools never populate that
# list (Task 3's built-in section is rendered from `builtin_permission_
# rows()`, a completely separate path -- Constraint 1), so `_tool_for()`
# always returned `None` for an `agent:builtin` row. Since `cycle_ui_state`
# is Inherit -> Allow -> Ask -> Off, the FIRST Space press from the default
# state always lands on "allow" -- which the vanished-tool guard then
# rejected before any write, with a factually wrong "no longer in the
# catalog" toast. `ask`/`deny` were consequently unreachable: you can't
# advance the ring past the step that's permanently blocked. These tests
# exercise the first press specifically (a pre-seeded "allow" and cycling
# from there would miss the bug entirely), plus the rest of the ring, an
# orphaned row, and an MCP-row regression guard.


@pytest.mark.asyncio
async def test_space_cycle_first_press_on_builtin_tool_persists_allow_no_hash(tmp_path):
    """The headline bug: a built-in tool row starts at Inherit, so the
    FIRST Space press cycles straight to "allow" -- exactly the transition
    the vanished-tool guard used to reject for every built-in row, because
    `_tool_for()` can never find a `HubTool` for `agent:builtin`. Must
    persist with no `definition_hash` (Task 1's HASH_FREE_SERVER_KEYS
    exemption) and raise no error/warning toast.
    """
    store_path = tmp_path / "mcp_permissions.json"
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        notifications = _capture_notifications(app)

        # Sanity: no explicit override exists yet -- this really is the
        # first press from Inherit, not a pre-seeded "allow".
        before = MCPPermissionStore(store_path).get_tool_entry(
            BUILTIN_TOOL_SERVER_KEY, "calculator"
        )
        assert before is None

        workbench.post_message(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="tool",
                server_key=BUILTIN_TOOL_SERVER_KEY,
                tool_name="calculator",
                new_state="allow",
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry(
            BUILTIN_TOOL_SERVER_KEY, "calculator"
        )
        assert entry is not None
        assert entry["state"] == "allow"
        # No REAL hash was computed/stored -- `set_tool_state()` always
        # writes the `definition_hash` key alongside an "allow" entry, but
        # for a HASH_FREE_SERVER_KEYS server it's the sentinel `None`
        # (never a fingerprint), because no `HubTool` was ever resolved
        # or hashed for this row.
        assert entry.get("definition_hash") is None

        assert not any(
            severity in ("warning", "error") for _, severity in notifications
        ), (
            f"expected no error/warning toast for a built-in allow cycle, got: {notifications!r}"
        )
        # Reflected on the next render without a restart.
        rows = _perm_all_rows(app)
        tool_cells = {row[0].strip(): row[1] for row in rows}
        assert tool_cells["calculator"] == "Allow •"


@pytest.mark.asyncio
async def test_space_cycle_ring_continues_through_ask_deny_and_back_to_inherit_for_builtin(
    tmp_path,
):
    """Once the first-press bug above is fixed, the rest of the ring must
    also work: allow -> ask -> deny (Off) -> inherit (cleared), each step
    persisting to the real store."""
    store_path = tmp_path / "mcp_permissions.json"
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        async def cycle(new_state):
            workbench.post_message(
                MCPPermissionsMode.StateCycleRequested(
                    row_kind="tool",
                    server_key=BUILTIN_TOOL_SERVER_KEY,
                    tool_name="calculator",
                    new_state=new_state,
                )
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

        store = MCPPermissionStore(store_path)

        await cycle("allow")
        assert (
            store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "calculator")["state"]
            == "allow"
        )

        await cycle("ask")
        assert (
            store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "calculator")["state"]
            == "ask"
        )

        await cycle("deny")
        assert (
            store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "calculator")["state"]
            == "deny"
        )

        await cycle(None)
        assert store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "calculator") is None


@pytest.mark.asyncio
async def test_space_cycle_on_orphaned_builtin_row_clears_stored_entry(tmp_path):
    """An orphaned built-in row (a stored decision for a tool name no
    longer provided by the live registry, Task 2/3) must still be
    cyclable to Inherit from the UI, clearing its stored entry -- the row
    exists precisely so the user can clean it up."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        BUILTIN_TOOL_SERVER_KEY, "tool_that_no_longer_exists", "allow"
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        row_keys = _perm_row_keys(app)
        assert "agent:builtin::tool_that_no_longer_exists" in row_keys

        workbench.post_message(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="tool",
                server_key=BUILTIN_TOOL_SERVER_KEY,
                tool_name="tool_that_no_longer_exists",
                new_state=None,
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry(
            BUILTIN_TOOL_SERVER_KEY, "tool_that_no_longer_exists"
        )
        assert entry is None
        # The orphaned row itself disappears once its stored entry is
        # cleared -- `builtin_permission_rows()` only lists a name absent
        # from the live registry when a stored decision for it exists.
        row_keys_after = _perm_row_keys(app)
        assert "agent:builtin::tool_that_no_longer_exists" not in row_keys_after


@pytest.mark.asyncio
async def test_mcp_tool_row_cycle_path_unchanged_still_passes_hubtool_and_hashes(
    tmp_path,
):
    """Regression guard for Constraint 2 ("MCP behavior stays byte-
    identical"): an ordinary MCP tool row's cycle-to-allow must still
    resolve and pass its `HubTool` through to `set_tool_state()`, and the
    store must still end up with a `definition_hash` -- the built-in
    branch must not have swallowed or short-circuited the MCP path."""
    store_path = tmp_path / "mcp_permissions.json"
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        workbench.post_message(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="tool",
                server_key="local:docs",
                tool_name="search",
                new_state="allow",
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry("local:docs", "search")
        assert entry is not None
        assert entry["state"] == "allow"
        # An MCP tool's hash is real content -- not merely present, but
        # exactly what `definition_hash()` computes from the catalog
        # `HubTool`'s own description/schema (proves `tool=cycled_tool`,
        # a genuine resolved `HubTool`, was passed through -- a `None`
        # fallback would have raised instead of reaching this assertion).
        matching_tool = workbench._tool_for("local:docs", "search")
        assert matching_tool is not None
        expected_hash = definition_hash(
            matching_tool.description, matching_tool.input_schema
        )
        assert entry["definition_hash"] == expected_hash


class GuardedEffectiveStatesHubService(PermissionsHubService):
    """T3/Constraint 1 regression guard: `effective_tool_states()` (the MCP
    resolver's batched entry point, and transitively `resolve_effective_
    state()`) must NEVER be called with a tool whose `server_key` is the
    built-in namespace -- built-in rows are only ever resolved through
    `resolve_builtin_state()` via `builtin_permission_rows()`. `tools` here
    is always `MCPWorkbench._last_hub_tools`, which the built-in section
    never populates or reads from -- so this failing is a real regression,
    not merely testing that nobody happened to call it with the wrong
    tools this pass."""

    def effective_tool_states(self, tools):
        for tool in tools:
            if tool.server_key == BUILTIN_TOOL_SERVER_KEY:
                pytest.fail(
                    f"effective_tool_states() called with a built-in tool: {tool.name}"
                )
        return super().effective_tool_states(tools)


class GuardedEffectiveStatesApp(ConsolidatedCSSApp):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self.unified_mcp_service = GuardedEffectiveStatesHubService(store_path)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_effective_tool_states_never_called_with_a_builtin_tool(tmp_path):
    app = GuardedEffectiveStatesApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        # The built-in section still rendered -- the guard above didn't
        # short-circuit rendering, it only pins that the MCP resolver was
        # never handed a built-in tool.
        rows = _perm_all_rows(app)
        tool_cells = {row[0].strip(): row[1] for row in rows}
        assert tool_cells["calculator"] == "Allow"


@pytest.mark.asyncio
async def test_space_cycle_round_trip_mutates_store_and_rerenders_override_marker(
    tmp_path,
):
    """`StateCycleRequested` -> the workbench mutates the REAL store via
    T4's typed methods -> the matrix re-renders with the override bullet.
    This is the Task 6 brief's headline round-trip requirement."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        payload = app.unified_mcp_service.permission_store.load()
        tool_entry = payload["profiles"]["default"]["servers"]["local:docs"]["tools"][
            "search"
        ]
        assert tool_entry["state"] == "allow"

        assert _perm_table_texts(app, 3) == ["  search", "Allow •"]
        # Sibling rows are untouched by the single-row mutation.
        assert _perm_table_texts(app, 2) == ["  fetch", "Ask"]
        assert _perm_table_texts(app, 0) == ["Global default", "Ask"]


@pytest.mark.asyncio
async def test_space_on_server_default_row_round_trips_through_store(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=1)  # __server__::local:docs
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        payload = app.unified_mcp_service.permission_store.load()
        assert (
            payload["profiles"]["default"]["servers"]["local:docs"]["default"]
            == "allow"
        )
        assert _perm_table_texts(app, 1) == ["Server default — docs", "Allow •"]


@pytest.mark.asyncio
async def test_space_on_global_row_round_trips_through_store(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=0)  # __global__
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        payload = app.unified_mcp_service.permission_store.load()
        # cycle_global("ask") == "deny"
        assert payload["profiles"]["default"]["global_default"] == "deny"
        assert _perm_table_texts(app, 0) == ["Global default", "Off"]


@pytest.mark.asyncio
async def test_kill_switch_toggle_round_trip_persists_and_resyncs(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        button = app.query_one("#mcp-perm-kill-switch", Button)
        assert str(button.label) == "Block all tool calls in chat: Off ▸"
        await pilot.click("#mcp-perm-kill-switch")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.get_kill_switch() is True
        assert str(button.label) == "Block all tool calls in chat: On ▸"


@pytest.mark.asyncio
async def test_kill_switch_starting_true_renders_without_extra_toggle(tmp_path):
    """A resync that pushes a kill_switch value DIFFERENT from the Button's
    constructor default (False) must not itself post a `KillSwitchToggled`
    -- `MCPPermissionsMode.update_matrix()` only ever relabels the Button,
    which posts no message of its own (unlike the old Checkbox's `.value =`
    assignment, which needed an explicit `prevent(Checkbox.Changed)` guard).
    Proven here by asserting the store still reads back exactly the seeded
    value after the mount-time resync -- a phantom echo would have
    round-tripped through `set_kill_switch()` too (harmlessly idempotent in
    THIS case, but the widget-level test suite pins the zero-events
    contract directly)."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_kill_switch(True)
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        button = app.query_one("#mcp-perm-kill-switch", Button)
        assert str(button.label) == "Block all tool calls in chat: On ▸"
        assert app.unified_mcp_service.get_kill_switch() is True


@pytest.mark.asyncio
async def test_preview_scoped_to_rail_selection(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        preview = app.query_one("#mcp-perm-preview", Static)
        assert str(preview.renderable) == "global default: ask"

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected("local:docs"))
        await pilot.pause()

        assert str(preview.renderable) == (
            "docs: 0 allow · 2 ask · 0 off — global default: ask"
        )


@pytest.mark.asyncio
async def test_preview_shows_override_count_when_no_server_selected(tmp_path):
    """UX batch item 9: with no rail selection, the preview is just the
    global default -- plus an override-count suffix once at least one
    explicit server/tool override exists anywhere in the matrix."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        preview = app.query_one("#mcp-perm-preview", Static)
        assert str(preview.renderable) == "global default: ask"

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Task 3 (MCP Hub Phase 6): the Space-cycle that just ran is a
        # standalone mutation resync, so the preview also carries its
        # transient echo prefix now -- the override-count SUFFIX this test
        # exists to pin is still computed correctly underneath it.
        assert str(preview.renderable) == (
            "search → Allow · global default: ask · 1 override across 1 server"
        )


@pytest.mark.asyncio
async def test_preview_override_count_includes_a_persistent_builtin_override(tmp_path):
    """Fix 2 (PR #906 review): before this fix, the preview's override
    count was computed inside `_build_permission_rows()` on its MCP-only
    `rows` list -- BEFORE `_sync_permissions_mode()` even appended the
    built-in section -- so a persistent built-in override changed the
    table cell but never this summary line, contradicting `update_matrix()`
    's own documented "ALWAYS summarizes the full, UNFILTERED matrix"
    contract. No MCP-side override exists in this fixture, so the ONE
    override this pins is unambiguously the built-in one."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        BUILTIN_TOOL_SERVER_KEY, "calculator", "deny"
    )

    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        preview = app.query_one("#mcp-perm-preview", Static)
        assert str(preview.renderable) == (
            "global default: ask · 1 override across 1 server"
        )


@pytest.mark.asyncio
async def test_preview_override_count_unaffected_when_no_builtin_override_is_set(
    tmp_path,
):
    """Fix 2 companion: the flip side of the test above -- with zero
    built-in overrides (this fixture's default shape), the preview must
    render exactly as it did before Fix 2 (no "0 overrides" segment, no
    phantom count from the built-in section's own always-present pinned
    row or its two always-rendered, uncustomized tool rows)."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        preview = app.query_one("#mcp-perm-preview", Static)
        assert str(preview.renderable) == "global default: ask"


@pytest.mark.asyncio
async def test_permissions_mode_renders_fail_soft_without_t4_seams():
    """A service that hasn't been upgraded with the T4 permission methods
    (the base `FakeHubService` -- no `effective_tool_states`/
    `permission_store`/`get_kill_switch` at all) must not crash
    `_sync_permissions_mode()`: it renders a global-only-effectively "Ask"
    matrix with the kill switch off using the same fail-soft seam pattern.
    """
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        assert table.row_count >= 1
        assert _perm_table_texts(app, 0) == ["Global default", "Ask"]
        button = app.query_one("#mcp-perm-kill-switch", Button)
        assert str(button.label) == "Block all tool calls in chat: Off ▸"


@pytest.mark.asyncio
async def test_double_space_cycle_on_tool_row_stays_on_that_row(tmp_path):
    """Critical regression: `update_matrix()` rebuilds the DataTable with
    `table.clear()` on EVERY resync, and Textual resets `cursor_coordinate`
    to (0, 0) on `clear()`. The workbench resyncs after every Space press,
    so a SECOND press used to land on row 0 (Global default) instead of the
    tool row the user was still looking at -- silently cycling the global
    default instead of the tool. Two Space presses on the tool row must
    advance the TOOL two cycle steps (Inherit -> Allow -> Ask) and must
    leave the global default untouched.
    """
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)

        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)

        payload = app.unified_mcp_service.permission_store.load()
        tool_entry = payload["profiles"]["default"]["servers"]["local:docs"]["tools"][
            "search"
        ]
        # cycle_ui_state(None) == "allow", cycle_ui_state("allow") == "ask"
        assert tool_entry["state"] == "ask"
        assert payload["profiles"]["default"]["global_default"] == "ask"
        assert _perm_table_texts(app, 3) == ["  search", "Ask •"]
        assert _perm_table_texts(app, 0) == ["Global default", "Ask"]


@pytest.mark.asyncio
async def test_state_cycle_requested_with_invalid_state_is_rejected(tmp_path):
    """Trust-boundary validation: `event.new_state` must be one of
    `STORE_STATES` (or `None` for Inherit) before the workbench calls any
    setter. A forged/corrupted `StateCycleRequested` carrying an invalid
    state must be a no-op against the store, with a warning toast, not a
    setter call or an unhandled exception.
    """
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        notifications = _capture_notifications(app)

        before = app.unified_mcp_service.permission_store.load()

        workbench.post_message(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="tool",
                server_key="local:docs",
                tool_name="search",
                new_state="bogus",
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        after = app.unified_mcp_service.permission_store.load()
        assert after == before
        assert any(severity == "warning" for _, severity in notifications), (
            f"expected a warning toast for an invalid cycle state, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_space_to_allow_vanished_tool_toasts_friendly_warning_not_raw_exception(
    tmp_path,
):
    """Minor 5: cycling a tool that's dropped out of the catalog (stale
    selection, or a resync racing a rug-pull refresh) to "allow" used to
    let `set_tool_state(..., "allow", tool=None)` raise, and the generic
    `except` toast the raw internal message
    ("Permission update failed: tool is required to set state 'allow' ...")
    verbatim. It must instead be caught before the setter and toast a
    friendly, actionable message -- no setter call, no store mutation.
    """
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        notifications = _capture_notifications(app)

        before = app.unified_mcp_service.permission_store.load()
        workbench.post_message(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="tool",
                server_key="local:docs",
                tool_name="does-not-exist",
                new_state="allow",
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        after = app.unified_mcp_service.permission_store.load()
        assert after == before
        assert notifications, "expected a toast for a vanished-tool allow cycle"
        message, severity = notifications[-1]
        assert severity == "warning"
        assert message == "Tool is no longer in the catalog — refresh and try again."
        assert "tool is required to set state" not in message


# -- Task 7: inspector permission explanation + re-allow ---------------------


@pytest.mark.asyncio
async def test_tools_mode_selection_shows_permission_block_via_gate_tool_test():
    """`_effective_for_display()` falls back to a single `gate_tool_test()`
    call when the batch `effective_tool_states()` cache is empty (a service
    with `gate_tool_test()` but not the batch method -- `ToolTestHubService`,
    same fake the Task 5/6 Test Tool tests already use)."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        assert app.query_one("#mcp-inspector-permission").display is True
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "From this tool's override."
        assert app.unified_mcp_service.gate_calls[-1] == ("local:docs", "fetch")


@pytest.mark.asyncio
async def test_permissions_mode_tool_row_selection_shows_permission_block(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#mcp-inspector-permission").display is True
        # Task 3 (MCP Hub Phase 6): `show_permission()` routed through the
        # workbench now carries a cascade tuple -- the plain origin sentence
        # is superseded by the three provenance rungs; nothing is overridden
        # here, so the global rung wins.
        assert not list(app.query("#mcp-inspector-permission-origin"))
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-tool", Static
                ).renderable
            )
            == "Tool override: —"
        )
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-server", Static
                ).renderable
            )
            == "Server default: —"
        )
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-global", Static
                ).renderable
            )
            == "▸ Global default: Ask"
        )
        # Routed through show_permission(), NOT show_tool() -- the full
        # tool-detail-plus-Test-Tool block is Tools mode's own surface.
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_cycling_the_selected_tool_refreshes_its_open_permission_block(tmp_path):
    """Minor 3: `on_mcp_permissions_mode_state_cycle_requested()` resyncs
    the MATRIX's own rows, but an already-open `#mcp-inspector-permission`
    block for that same tool is a separate render (`show_permission()`)
    that used to keep showing the PRE-cycle rule until something else
    re-rendered it (only the re-allow handler did). Select docs::search
    (inherits the global default) then Space-cycle that SAME row to
    "allow" -- the still-open inspector block must pick up the new
    tool-override rule without a fresh selection.
    """
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("enter")
        await pilot.pause()
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-global", Static
                ).renderable
            )
            == "▸ Global default: Ask"
        )

        await pilot.press("space")  # cycle_ui_state(None) == "allow"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        state = str(app.query_one("#mcp-inspector-permission-state", Static).renderable)
        assert state == "Permission: Allow"
        # Task 3: the cascade rungs refresh along with the origin used to --
        # the tool rung now wins with the freshly-cycled override.
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-tool", Static
                ).renderable
            )
            == "▸ Tool override: Allow •"
        )


@pytest.mark.asyncio
async def test_permissions_mode_pinned_row_selection_clears_inspector_without_crash(
    tmp_path,
):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search -- populate the block first
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#mcp-inspector-permission").display is True

        table.move_cursor(row=0)  # __global__ -- pinned row, no tool
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#mcp-inspector-permission").display is False
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_permissions_mode_builtin_tool_row_selection_shows_permission_block(
    tmp_path,
):
    """Fix 1 (PR #906 review, code-review round after TASK-627): before
    this fix, `on_mcp_permissions_mode_row_selected()` resolved every
    `"tool"` row's `HubTool` via `_tool_for()`, which only ever searches
    `_last_hub_tools` (the MCP catalog) -- a built-in row is never in that
    list (Constraint 1/5), so `tool` was always `None` and the handler
    fell through to `show_tool(None)`, blanking the inspector. This was
    the only interactive row in the whole matrix that did that -- selected
    here (row 7, `agent:builtin::calculator`) using the SAME `enter`-press
    selection path every other permissions-mode test in this file uses."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=7)  # agent:builtin::calculator
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#mcp-inspector-permission").display is True
        permission_text = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert permission_text == "calculator — Built-in (agent runtime)"
        state = str(app.query_one("#mcp-inspector-permission-state", Static).renderable)
        assert state == "Permission: Allow"
        # Built-ins have no MCP tool/server/global cascade -- falls back to
        # the plain per-tool origin sentence, not the three provenance
        # rungs `show_permission(..., cascade=...)` renders for an MCP row.
        origin = str(
            app.query_one("#mcp-inspector-permission-origin", Static).renderable
        )
        assert origin == "Built-in tools default to allow."
        assert not list(app.query("#mcp-inspector-permission-cascade-tool"))
        # Routed through show_permission(), NOT show_tool() -- same
        # standalone-surface contract as an MCP tool row (Tools mode's own
        # full tool-detail-plus-Test-Tool block never mounts here).
        assert not list(app.query("#mcp-inspector-tool-name"))

        # Selecting an MCP row afterwards is unchanged.
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("enter")
        await pilot.pause()
        permission_text = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert "search" in permission_text and "docs" in permission_text
        assert list(app.query("#mcp-inspector-permission-cascade-tool"))


@pytest.mark.asyncio
async def test_permissions_mode_builtin_pinned_row_selection_clears_inspector(tmp_path):
    """Fix 1 companion: the built-in section's OWN pinned "Server default"
    row (`row_kind == "server"`) must keep the pre-existing pinned-row
    behavior (clears the inspector) -- only the built-in `"tool"` rows
    beneath it gained a permission view. Mirrors `test_permissions_mode_
    pinned_row_selection_clears_inspector_without_crash` above, for the
    built-in section's own pinned row instead of the global one."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(
            row=7
        )  # agent:builtin::calculator -- populate the block first
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#mcp-inspector-permission").display is True

        table.move_cursor(row=6)  # Server default — Built-in (agent runtime)
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#mcp-inspector-permission").display is False
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_reallow_round_trip_clears_config_changed_marker_and_matrix_warning(
    tmp_path,
):
    """The Task 7 headline round-trip: a stale tool-level `allow` (its
    stored `definition_hash` no longer matches the live tool -- the
    rug-pull guard) renders "Ask ⚠" in the matrix; selecting that row shows
    Re-allow in the inspector; pressing it stores the tool's CURRENT
    definition hash via `set_tool_state(..., "allow", tool=tool)` and
    resyncs -- the ⚠ clears to a plain override bullet."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        "local:docs",
        "search",
        "allow",
        definition_hash="stale-hash-from-a-different-tool-shape",
    )

    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        assert _perm_table_texts(app, 3) == ["  search", "Ask ⚠"]

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)
        await pilot.press("enter")
        await pilot.pause()

        notice = str(
            app.query_one("#mcp-inspector-permission-notice", Static).renderable
        )
        assert notice == "Definition changed since you allowed it."
        assert app.query_one("#mcp-inspector-reallow", Button)

        await pilot.click("#mcp-inspector-reallow")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        payload = app.unified_mcp_service.permission_store.load()
        tool_entry = payload["profiles"]["default"]["servers"]["local:docs"]["tools"][
            "search"
        ]
        assert tool_entry["state"] == "allow"

        assert _perm_table_texts(app, 3) == ["  search", "Allow •"]
        assert not list(app.query("#mcp-inspector-reallow"))


# -- Task 3 (MCP Hub Phase 6): cascade provenance end-to-end -----------------


@pytest.mark.asyncio
async def test_permissions_row_selection_cascade_marks_tool_override_as_winner(
    tmp_path,
):
    store_path = tmp_path / "mcp_permissions.json"
    # Critical review fix regression: the stored hash must match "search"'s
    # REAL definition (`PermissionsHubService`'s own discovery_snapshot
    # description, "Search docs.", no inputSchema -- see
    # `hub_tool_catalog.local_tools_from_record()`'s `input_schema=None`
    # default) -- a mismatched hash (this test's own pre-fix
    # `definition_hash="anything"`) trips the SAME rug-pull guard
    # `test_reallow_refreshes_cascade_with_tool_rung_as_winner` exercises
    # on purpose, downgrading `config_changed=True` and coloring the
    # winning rung warning, not the plain undowngraded-override "ready"
    # this test's own name asserts.
    MCPPermissionStore(store_path).set_tool_state(
        "local:docs",
        "search",
        "allow",
        definition_hash=definition_hash("Search docs.", None),
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("enter")
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        assert "▸" in str(tool_rung.renderable)
        assert "mcp-status-ready" in tool_rung.classes


@pytest.mark.asyncio
async def test_permissions_row_selection_cascade_marks_server_default_as_winner(
    tmp_path,
):
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_server_default("local:docs", "deny")
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search -- inherits the server default
        await pilot.press("enter")
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        server_rung = app.query_one("#mcp-inspector-permission-cascade-server", Static)
        assert str(tool_rung.renderable) == "Tool override: —"
        assert str(server_rung.renderable) == "▸ Server default: Off •"
        assert "mcp-status-error" in server_rung.classes
        assert "mcp-status-muted" in tool_rung.classes


@pytest.mark.asyncio
async def test_reallow_refreshes_cascade_with_tool_rung_as_winner(tmp_path):
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        "local:docs",
        "search",
        "allow",
        definition_hash="stale-hash-from-a-different-tool-shape",
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)
        await pilot.press("enter")
        await pilot.pause()

        await pilot.click("#mcp-inspector-reallow")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        tool_rung = app.query_one("#mcp-inspector-permission-cascade-tool", Static)
        assert str(tool_rung.renderable) == "▸ Tool override: Allow •"
        assert "mcp-status-ready" in tool_rung.classes


# -- Task 3 (MCP Hub Phase 6): Change in Permissions cross-mode jump --------


@pytest.mark.asyncio
async def test_tools_mode_permission_block_change_button_jumps_to_permissions_row():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        assert app.query_one("#mcp-inspector-permission").display is True

        await pilot.click("#mcp-inspector-goto-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "permissions"
        perm_table = app.query_one("#mcp-perm-table", DataTable)
        cursor_key, _ = perm_table.coordinate_to_cell_key((perm_table.cursor_row, 0))
        assert cursor_key.value == "local:docs::fetch"
        permission_text = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert "fetch" in permission_text


@pytest.mark.asyncio
async def test_test_tool_blocked_change_button_jumps_to_permissions_row():
    app = ToolTestApp()
    app.unified_mcp_service.gate_state = "deny"
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        goto_button = app.query_one("#mcp-inspector-goto-permission-test", Button)
        assert goto_button.display is True

        await pilot.click("#mcp-inspector-goto-permission-test")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "permissions"
        perm_table = app.query_one("#mcp-perm-table", DataTable)
        cursor_key, _ = perm_table.coordinate_to_cell_key((perm_table.cursor_row, 0))
        assert cursor_key.value == "local:docs::fetch"


@pytest.mark.asyncio
async def test_goto_permission_row_is_the_single_shared_implementation_for_all_three_triggers(
    monkeypatch,
):
    """The brief's "no duplication" requirement, proven behaviorally: the
    audit drill's "Adjust permission" button, the Tools-mode permission
    block's "Change in Permissions" button, and the Test Tool panel's own
    blocked-result button all route through the exact same
    `MCPWorkbench._goto_permission_row()` coroutine -- spied here so a
    regression that reintroduces a second, drifted copy for any of the three
    would be caught even if their end-to-end OUTCOMES still happened to
    match."""
    # "fetch" (not "search") throughout -- "fetch" is raw/no-required-fields
    # (mirrors `test_deny_gate_blocks_without_calling_service`'s own row-0
    # choice for exactly this reason); "search" has a required "query"
    # field, so an empty Test Tool Run would raise inside the inspector's
    # own `form.collect_arguments()` before `ToolTestRequested` is even
    # posted -- never reaching the gate check trigger 3 needs.
    app = AuditApp([_audit_record(server_key="local:docs", tool_name="fetch")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        calls: list[tuple[str, str]] = []
        original = MCPWorkbench._goto_permission_row

        async def _spy(self, server_key, tool_name):
            calls.append((server_key, tool_name))
            await original(self, server_key, tool_name)

        monkeypatch.setattr(MCPWorkbench, "_goto_permission_row", _spy)

        # Trigger 1: the audit drill's own "Adjust permission" button.
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        await pilot.click("#mcp-audit-adjust-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Trigger 2: the Tools-mode permission block's own button.
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        await pilot.click("#mcp-inspector-goto-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Trigger 3: the Test Tool panel's own button (blocked path).
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        app.unified_mcp_service.gate_state = "deny"
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#mcp-inspector-goto-permission-test")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert calls == [
            ("local:docs", "fetch"),
            ("local:docs", "fetch"),
            ("local:docs", "fetch"),
        ]


@pytest.mark.asyncio
async def test_tools_mode_permission_block_change_button_clears_stale_tool_detail():
    """Critical review fix: jumping to Permissions mode via the Tools-mode
    permission block's own "Change in Permissions" button fires from Tools
    mode, where `#mcp-inspector-tool` is populated. `_goto_permission_row()`
    dispatches `_open_audit_permission()` into the SAME exclusive
    `"mcp-tool-clear"` worker group `set_mode()`'s own `_clear_tool_view()`
    just used, cancelling that clear before it runs (the documented trap --
    see `_open_audit_tool()`'s own docstring for the mechanism), so
    `_open_audit_permission()` must explicitly clear `#mcp-inspector-tool`
    itself (mirroring its existing explicit `show_audit_entry(None)` clear)
    or the stale tool detail stays stacked underneath the new Permissions-
    mode block."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        assert app.query_one("#mcp-inspector-tool-name", Static)
        assert app.query_one("#mcp-inspector-tool", Vertical).display is True

        await pilot.click("#mcp-inspector-goto-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "permissions"
        assert app.query_one("#mcp-inspector-permission").display is True
        assert not list(app.query("#mcp-inspector-tool-name"))
        assert app.query_one("#mcp-inspector-tool", Vertical).display is False


# -- Task 3 (MCP Hub Phase 6): mutation echo ---------------------------------


@pytest.mark.asyncio
async def test_space_cycle_prefixes_preview_with_mutation_echo(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("search → Allow · ")


@pytest.mark.asyncio
async def test_double_space_cycle_echo_replaces_not_appends(tmp_path):
    """Minor 4 (review): back-to-back Space-cycles on the same tool row must
    each render their OWN fresh mutation echo, not stack the previous
    press's echo underneath the new one -- `MCPPermissionsMode.
    update_matrix()`'s `#mcp-perm-preview` render is a single
    `Static.update()` call per pass (full replace, never append), and each
    Space press recomputes `echo` fresh from ITS OWN cycle event
    (`_cycled_ui_label(event.new_state)`), so the second press's preview
    must show exactly one "→" -- the SECOND cycle's own arrow -- not two.
    Mirrors `test_double_space_cycle_on_tool_row_stays_on_that_row`'s own
    double-press-same-row choreography (the `pilot.pause(0.3)` settle
    between presses)."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")  # cycle_ui_state(None) == "allow"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("search → Allow · ")
        assert preview.count("→") == 1

        await pilot.press("space")  # cycle_ui_state("allow") == "ask"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("search → Ask · ")
        assert preview.count("→") == 1


@pytest.mark.asyncio
async def test_kill_switch_toggle_prefixes_preview_with_echo(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        await pilot.click("#mcp-perm-kill-switch")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("kill switch → on · ")


@pytest.mark.asyncio
async def test_reallow_prefixes_preview_with_echo(tmp_path):
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        "local:docs",
        "search",
        "allow",
        definition_hash="stale-hash-from-a-different-tool-shape",
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)
        await pilot.press("enter")
        await pilot.pause()

        await pilot.click("#mcp-inspector-reallow")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("search → Allow · ")


@pytest.mark.asyncio
async def test_full_resync_clears_mutation_echo(tmp_path):
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("search → Allow · ")

        # A full `_sync_children()` pass isn't itself a standalone mutation
        # resync -- it must clear the echo without anyone calling a
        # dedicated "clear" method.
        await workbench._sync_children()
        await pilot.pause()

        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert not preview.startswith("search → Allow · ")
        assert preview == "global default: ask · 1 override across 1 server"


@pytest.mark.asyncio
async def test_reallow_guard_tool_not_found_notifies_without_store_call(tmp_path):
    """Guard: an unresolvable tool (dropped out of the catalog, or simply
    never existed) must be a warning toast, never a store call --
    `set_tool_state(..., "allow", ...)` requires a live `HubTool` to
    fingerprint."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        notifications = _capture_notifications(app)

        before = app.unified_mcp_service.permission_store.load()
        workbench.post_message(
            MCPInspector.ReallowRequested("local:docs", "does-not-exist")
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        after = app.unified_mcp_service.permission_store.load()
        assert after == before
        assert any(severity == "warning" for _, severity in notifications), (
            f"expected a warning toast for an unresolvable re-allow target, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_invalid_global_default_renders_ask_instead_of_panicking(tmp_path):
    """I2: a hand-edited `mcp_permissions.json` with an invalid
    `global_default` (e.g. "banana" -- a valid `schema_version`, so
    `load()`'s corruption check does NOT back it up/reset it) must not
    reach `format_tool_state_label()`/`EffectiveToolState.ui_label` as an
    unrecognized state -- that used to `KeyError` out of `_sync_children`
    and panic the whole app on the very first Permissions-mode render.
    docs::fetch has no tool- or server-level override, so it inherits
    straight from the (corrupt) global default -- exactly the path
    `_sync_permissions_mode()`'s OWN "Global default" row already guarded
    but `effective_tool_states()` (feeding every TOOL row) did not.
    """
    store_path = tmp_path / "mcp_permissions.json"
    store_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kill_switch": False,
                "profiles": {"default": {"global_default": "banana", "servers": {}}},
            }
        )
    )
    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        assert _perm_table_texts(app, 0) == ["Global default", "Ask"]
        # row 2: docs::fetch -- inherits the (invalid) global default.
        assert _perm_table_texts(app, 2) == ["  fetch", "Ask"]


# -- T8: Tools-mode State column + server-source governance listing ---------
#
# Reuses `PermissionsHubService` (real store, real `resolve_effective_state`)
# for the State-column wiring, and a new server-source fake for the
# governance listing -- `PermissionsHubService`'s existing local source
# already proves the section stays absent there.


@pytest.mark.asyncio
async def test_tools_mode_state_column_reflects_effective_tool_states(tmp_path):
    """The Tools-mode catalog's State column is populated end-to-end from
    the SAME `effective_tool_states()` resolution the Permissions matrix
    uses: set an explicit override on one tool via the real store, resync,
    and confirm the catalog's State cell reflects it (Textual never
    unmounts the inactive Tools canvas, so this doesn't need
    `set_mode('tools')` first -- mirrors `_sync_tools_mode()`'s own
    "always current" docstring)."""
    app = PermissionsApp(tmp_path / "mcp_permissions_tools.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        search_tool = next(t for t in workbench._last_hub_tools if t.name == "search")

        app.unified_mcp_service.set_tool_state(
            "local:docs", "search", "allow", tool=search_tool
        )
        await workbench._sync_children()
        await pilot.pause()

        table = app.query_one("#mcp-tools-table", DataTable)
        rows_by_tool = {
            table.get_row_at(i)[0].plain: table.get_row_at(i)[1].plain
            for i in range(table.row_count)
        }
        assert rows_by_tool["search"] == "Allow •"
        assert rows_by_tool["fetch"] == "Ask"
        assert rows_by_tool["list_notes"] == "Ask"


def _tools_table_state(app: App, tool_name: str) -> str:
    table = app.query_one("#mcp-tools-table", DataTable)
    for i in range(table.row_count):
        row = table.get_row_at(i)
        if row[0].plain == tool_name:
            return row[1].plain
    raise AssertionError(f"{tool_name!r} not found in #mcp-tools-table")


@pytest.mark.asyncio
async def test_space_cycle_propagates_fresh_states_to_tools_mode_without_full_resync(
    tmp_path,
):
    """Defect 1 (MCP Hub Phase 4 live QA, 2026-07-16): the standalone
    Space-cycle handler (`on_mcp_permissions_mode_state_cycle_requested`)
    deliberately resyncs ONLY the Permissions matrix for latency (T8/T10) --
    but it already resolves a fresh `EffectiveToolState` batch to do that.
    This pins that `MCPToolsMode`'s cached State column must be refreshed
    from that SAME batch, so switching to Tools mode right after a
    Space-cycle -- with no manual `r` refresh and no full `_sync_children()`
    pass -- shows the tool's NEW state, not the pre-mutation one."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        # Tools mode starts with the pre-mutation "Ask" (plain, inherited).
        assert _tools_table_state(app, "search") == "Ask"

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert _perm_table_texts(app, 3) == ["  search", "Allow •"]

        # Switch to Tools mode WITHOUT a full resync (no `r` keypress, no
        # `_sync_children()` call) -- `set_mode()` only swaps the
        # ContentSwitcher's visible pane.
        workbench.set_mode("tools")
        await pilot.pause()

        assert _tools_table_state(app, "search") == "Allow •"


@pytest.mark.asyncio
async def test_reallow_propagates_fresh_states_to_tools_mode_without_full_resync(
    tmp_path,
):
    """Same Defect 1 coverage for the Re-allow standalone handler
    (`on_mcp_inspector_reallow_requested`) -- a rug-pull-downgraded tool's
    "Ask ⚠" marker must also clear in the Tools-mode State column once
    Re-allow is pressed, without a manual refresh."""
    store_path = tmp_path / "mcp_permissions.json"
    MCPPermissionStore(store_path).set_tool_state(
        "local:docs",
        "search",
        "allow",
        definition_hash="stale-hash-from-a-different-tool-shape",
    )

    app = PermissionsApp(store_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        assert _perm_table_texts(app, 3) == ["  search", "Ask ⚠"]
        assert _tools_table_state(app, "search") == "Ask ⚠"

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)
        await pilot.press("enter")
        await pilot.pause()

        await pilot.click("#mcp-inspector-reallow")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert _perm_table_texts(app, 3) == ["  search", "Allow •"]

        workbench.set_mode("tools")
        await pilot.pause()

        assert _tools_table_state(app, "search") == "Allow •"


class CountingEffectiveStatesHubService(PermissionsHubService):
    """T10 regression guard: counts `effective_tool_states()` invocations.

    Before T10, one full `_sync_children()` pass called this TWICE for the
    exact same `tools` list -- once from `_sync_tools_mode()` (State
    column) and once from `_sync_permissions_mode()` (matrix rows), each a
    full store load (plus any mark/audit side effects the real method
    performs). This subclass just counts calls on top of
    `PermissionsHubService`'s real store-backed implementation so a test
    can pin the count directly rather than inferring it from rendered
    output.
    """

    def __init__(self, store_path: Path) -> None:
        super().__init__(store_path)
        self.effective_tool_states_calls = 0

    def effective_tool_states(self, tools):
        self.effective_tool_states_calls += 1
        return super().effective_tool_states(tools)


class CountingEffectiveStatesApp(ConsolidatedCSSApp):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self.unified_mcp_service = CountingEffectiveStatesHubService(store_path)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_sync_children_resolves_effective_states_exactly_once(tmp_path):
    """T10: one full `_sync_children()` pass must call
    `effective_tool_states()` on the service EXACTLY ONCE. Previously
    `_sync_tools_mode()` and `_sync_permissions_mode()` each resolved their
    own `EffectiveToolState` batch independently -- two full resolutions
    over the SAME tools list, back-to-back, every single resync."""
    app = CountingEffectiveStatesApp(tmp_path / "mcp_permissions_count.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        # Reset past whatever the mount-time reload() itself triggered --
        # this test pins the count for ONE explicit full sync, not mount.
        service.effective_tool_states_calls = 0

        await workbench._sync_children()
        await pilot.pause()

        assert service.effective_tool_states_calls == 1


class GovernanceHubService(FakeHubService):
    """Server source with one active target ("main") -- `load_section`
    returns a canned "governance" section carrying a `permission_profiles`
    list (T8's read-only listing source), and an empty external-servers
    section for everything else `_collect_snapshots()`/`_sync_tools_mode()`
    ask for."""

    def __init__(self) -> None:
        super().__init__()
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "governance":
            return {
                "permission_profiles": [
                    {"name": "Docs writers", "id": "prof-1"},
                    {"label": "Analysts", "profile_id": "prof-2"},
                ],
                "source": "server",
                "section": "governance",
            }
        return {
            "external_servers": [],
            "source": "server",
            "section": effective_section,
        }


class GovernanceApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = GovernanceHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_permissions_mode_server_source_shows_governance_profiles_readonly_listing():
    app = GovernanceApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        section = app.query_one("#mcp-perm-server-profiles")
        assert section.display is True
        pointer = str(
            app.query_one("#mcp-perm-server-profiles-pointer", Static).renderable
        )
        assert pointer == (
            "Server-side profiles are managed in the tldw_server webui. The "
            "matrix above is chatbook's client-side gate and still applies."
        )
        rows = [str(s.renderable) for s in app.query(".mcp-perm-server-profile-row")]
        assert rows == ["Docs writers (prof-1)", "Analysts (prof-2)"]


@pytest.mark.asyncio
async def test_permissions_mode_local_source_has_no_governance_section(tmp_path):
    """Local source never calls `load_section("governance")` at all -- the
    section is absent, not merely empty."""
    app = PermissionsApp(tmp_path / "mcp_permissions_local_governance.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 0


class GovernanceFetchFailsHubService(GovernanceHubService):
    """Server source whose `load_section("governance")` raises -- the guard
    in `_load_server_governance_profiles()` must swallow it and leave the
    section absent rather than crashing every `_sync_children()` pass."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "governance":
            raise RuntimeError("governance backend unavailable")
        return await super().load_section(section)


class GovernanceFetchFailsApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = GovernanceFetchFailsHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_permissions_mode_governance_fetch_failure_leaves_section_absent():
    app = GovernanceFetchFailsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        assert len(app.query("#mcp-perm-server-profiles")) == 0
        # The matrix itself must still render -- a governance-section
        # failure is isolated, not a whole-mode crash.
        assert app.query_one("#mcp-perm-table", DataTable)


# -- T11: server-source governance-listing fetch is cached per (source,
# target) identity -----------------------------------------------------------
#
# `GovernanceCachingHubService` combines `GovernanceHubService`'s canned
# governance section with `PermissionsHubService`'s real store-backed
# `effective_tool_states()`/`set_tool_state()`/kill-switch plumbing (T1/T4),
# plus one external server record embedding a tool (mirrors
# `ServerToolsHubService`) so the permissions matrix actually has a "tool"
# row to Space-press cycle, and a `load_section("governance")` call counter.


class GovernanceCachingHubService(FakeHubService):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self._store = MCPPermissionStore(store_path)
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )
        self.governance_fetch_calls = 0

    @property
    def permission_store(self) -> MCPPermissionStore:
        return self._store

    def effective_tool_states(self, tools):
        payload = self._store.load()
        return {
            (t.server_key, t.name): resolve_effective_state(payload, t) for t in tools
        }

    def set_tool_state(self, server_key, tool_name, ui_state, *, tool=None):
        hash_value = None
        if ui_state == "allow":
            if tool is None:
                raise ValueError("tool is required to set state 'allow'")
            hash_value = definition_hash(tool.description, tool.input_schema)
        self._store.set_tool_state(
            server_key, tool_name, ui_state, definition_hash=hash_value
        )

    def get_kill_switch(self):
        return self._store.get_kill_switch()

    def set_kill_switch(self, value):
        self._store.set_kill_switch(value)

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "governance":
            self.governance_fetch_calls += 1
            return {
                "permission_profiles": [{"name": "Docs writers", "id": "prof-1"}],
                "source": "server",
                "section": "governance",
            }
        if (
            self.context.selected_source == "server"
            and effective_section == "external_servers"
        ):
            return {
                "external_servers": [
                    {
                        "server_id": "docs",
                        "name": "Docs",
                        "tools": [{"name": "search", "description": "Search."}],
                    }
                ],
                "source": "server",
                "section": "external_servers",
            }
        return await super().load_section(section)


class GovernanceCachingApp(ConsolidatedCSSApp):
    def __init__(self, store_path: Path) -> None:
        super().__init__()
        self.unified_mcp_service = GovernanceCachingHubService(store_path)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_space_press_resyncs_reuse_cached_governance_profiles(tmp_path):
    """T11 (a): the governance listing is STATIC server-side data -- two
    consecutive Space-press permission cycles under server source (each a
    standalone `_sync_permissions_mode()` resync) must not re-fetch it.
    Only the mount-time full `_sync_children()` pass should ever call
    `load_section("governance")`."""
    app = GovernanceCachingApp(tmp_path / "mcp_governance_cache.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        service = app.unified_mcp_service
        assert service.governance_fetch_calls == 1  # the mount-time full sync

        table = app.query_one("#mcp-perm-table", DataTable)
        # global, server default, "search" tool + the built-in section
        # (server default, calculator, get_current_datetime) -- TASK-627
        # Task 3 appends unconditionally, regardless of source.
        assert table.row_count == 6
        table.focus()

        table.move_cursor(row=2)  # server:main/docs::search
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert service.governance_fetch_calls == 1

        table.move_cursor(row=2)
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert service.governance_fetch_calls == 1

        # The section itself still renders from the cached value -- caching
        # the fetch must not also blank the UI.
        section = app.query_one("#mcp-perm-server-profiles")
        assert section.display is True


@pytest.mark.asyncio
async def test_source_switch_refetches_governance_profiles_once(tmp_path):
    """T11 (b): a source switch is a full `_sync_children()` pass under a
    `(source, target)` key the cache no longer matches for -- exactly one
    additional `load_section("governance")` fetch per switch back into
    server source, never a fetch per resync along the way."""
    app = GovernanceCachingApp(tmp_path / "mcp_governance_switch.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        service = app.unified_mcp_service
        assert service.governance_fetch_calls == 1

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        assert service.governance_fetch_calls == 1  # local source never fetches

        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        assert service.governance_fetch_calls == 2


# -- Minor 2: `_load_server_governance_profiles()` malformed-but-present
# payloads -- both a non-Mapping payload and a `permission_profiles` key
# that isn't a list count as a SUCCESSFUL fetch (not a failure): the section
# still renders, with its pointer text and zero rows, same as a legitimately
# empty profiles list. Only load_section() itself raising renders the
# section absent (see test_permissions_mode_governance_fetch_failure_leaves_
# section_absent above).


class GovernanceNonMappingHubService(GovernanceHubService):
    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "governance":
            return ["not", "a", "mapping"]
        return await super().load_section(section)


class GovernanceNonMappingApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = GovernanceNonMappingHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_permissions_mode_governance_non_mapping_payload_renders_empty_section():
    app = GovernanceNonMappingApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        section = app.query_one("#mcp-perm-server-profiles")
        assert section.display is True
        assert len(app.query(".mcp-perm-server-profile-row")) == 0
        # The matrix itself must still render alongside the empty section.
        assert app.query_one("#mcp-perm-table", DataTable)


class GovernanceProfilesNotListHubService(GovernanceHubService):
    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "governance":
            return {
                "permission_profiles": "not-a-list",
                "source": "server",
                "section": "governance",
            }
        return await super().load_section(section)


class GovernanceProfilesNotListApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = GovernanceProfilesNotListHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_permissions_mode_governance_profiles_not_list_renders_empty_section():
    app = GovernanceProfilesNotListApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        section = app.query_one("#mcp-perm-server-profiles")
        assert section.display is True
        assert len(app.query(".mcp-perm-server-profile-row")) == 0
        assert app.query_one("#mcp-perm-table", DataTable)


# -- T7 (MCP Hub Phase 5): Audit mode canvas + drill-through -----------------


class FakeExecutionLog:
    """Minimal `MCPExecutionLog`-shaped fake -- `read_recent(limit)` returns
    (a copy of) whatever record list the test seeded, newest-first is the
    CALLER's responsibility (the real log guarantees it; this fake just
    hands back records verbatim, same "render whatever it's given" contract
    `MCPAuditMode` itself follows).

    Fix Round H (PR-T3 review), Item 2c: not a fake-vs-real key-set pin
    candidate itself -- it implements exactly ONE method with no real-side
    computation to drift from. `_audit_record()` just below DOES hand-roll
    a dict mirroring `ExecutionRecord`'s dataclass fields, a real but
    orthogonal risk (dataclass-vs-dict, not fake-vs-real method) left as a
    candidate for future work; see the POLICY comment above
    `FakeLocalMCPControlService` in
    Tests/MCP/test_unified_control_plane_service.py for the full policy.
    """

    def __init__(self, records: list[dict] | None = None) -> None:
        self._records = list(records or [])
        self.read_recent_calls: list[int] = []

    def read_recent(self, limit: int = 200) -> list[dict]:
        self.read_recent_calls.append(limit)
        return list(self._records[:limit])


def _audit_record(
    *,
    ts: str = "2026-07-16T21:22:00+00:00",
    server_key: str = "local:docs",
    tool_name: str = "search",
    initiator: str = "test",
    decision: str = "allowed",
    ok: bool = True,
    duration_ms: int = 42,
    status: str = "success",
    error_category: str | None = None,
    exception_type: str | None = None,
    status_code: int | None = None,
    argument_names: list[str] | None = None,
    unknown_argument_count: int = 0,
    result_type: str = "none",
    result_size: int = 0,
) -> dict:
    return {
        "ts": ts,
        "server_key": server_key,
        "tool_name": tool_name,
        "initiator": initiator,
        "decision": decision,
        "ok": ok,
        "status": status,
        "duration_ms": duration_ms,
        "error_category": error_category,
        "exception_type": exception_type,
        "status_code": status_code,
        "argument_names": argument_names or [],
        "unknown_argument_count": unknown_argument_count,
        "result_type": result_type,
        "result_size": result_size,
    }


class AuditHubService(ToolTestHubService):
    """Reuses `ToolTestHubService`'s real docs::fetch/docs::search/
    notes::list_notes catalog (so drill-through has real, resolvable
    `HubTool`s to route to) and adds an `execution_log` attribute -- unlike
    the real `UnifiedMCPControlPlaneService.execution_log` (a lazily
    constructed property), a plain instance attribute is enough here: the
    workbench's own read (`service.execution_log`) doesn't distinguish the
    two."""

    def __init__(self, records: list[dict] | None = None) -> None:
        super().__init__()
        self.execution_log = FakeExecutionLog(records)


class AuditApp(ConsolidatedCSSApp):
    def __init__(self, records: list[dict] | None = None) -> None:
        super().__init__()
        self.unified_mcp_service = AuditHubService(records)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_audit_mode_syncs_execution_log_entries_into_canvas():
    app = AuditApp(
        [_audit_record(tool_name="search"), _audit_record(tool_name="fetch")]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        assert service.execution_log.read_recent_calls == [200]
        workbench.set_mode("audit")
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 2
        assert table.display is True
        assert app.query_one("#mcp-audit-empty").display is False


# -- Task 5 (PR-T3, F3): a completed tool run repopulates the Audit
# entries table without pressing `r` -- before this task, `_sync_audit_
# mode()` had exactly ONE caller (the tail of `_sync_children()`), so
# `_run_tool_test()` completing a run never resynced it, even though the
# JSONL log already had the new row by the time the run finished (real
# `test_hub_tool()`/`execute_hub_tool()` record BEFORE returning).


class AuditRunAppendsHubService(AuditHubService):
    """`test_hub_tool()` mimics the real service's own side effect
    (`execute_hub_tool()` -> `_record_tool_execution()`): the execution
    log already has a new row for THIS run by the time `test_hub_tool()`
    returns -- a test built on this can then check whether the workbench
    re-reads that log after the run, without touching `r`/reload."""

    async def execute_prepared_hub_test(self, nonce, intent, arguments):
        preview = self._previews.get(nonce)
        result = await super().execute_prepared_hub_test(nonce, intent, arguments)
        if preview is not None and not isinstance(
            result, (ToolTestAdmissionBlocked, ToolTestAdmissionStale)
        ):
            self.execution_log._records.insert(
                0,
                _audit_record(
                    server_key=preview.server_key,
                    tool_name=preview.tool_name,
                ),
            )
        return result


class AuditRunAppendsApp(ConsolidatedCSSApp):
    def __init__(self, records: list[dict] | None = None) -> None:
        super().__init__()
        self.unified_mcp_service = AuditRunAppendsHubService(records)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_completed_run_repopulates_audit_entries_without_manual_refresh():
    app = AuditRunAppendsApp([])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._last_audit_entries == []

        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "search")
        event = _prepared_test_event(app.unified_mcp_service, tool, {"query": "hello"})
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._last_audit_entries, (
            "audit entries stayed empty after a completed run -- no manual "
            "refresh was performed"
        )
        assert workbench._last_audit_entries[0]["tool_name"] == "search"


@pytest.mark.asyncio
async def test_completed_run_repopulates_audit_table_when_audit_mode_is_active():
    """Same fix, checked at the rendered-table level (not just the internal
    `_last_audit_entries` list) -- switch to Audit mode FIRST, then run a
    tool test, and confirm the table itself grows without pressing `r`."""
    app = AuditRunAppendsApp([])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 0

        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "search")
        event = _prepared_test_event(app.unified_mcp_service, tool, {"query": "hello"})
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert table.row_count == 1


@pytest.mark.asyncio
async def test_audit_mode_guarded_when_service_has_no_execution_log_attribute():
    """The plain `FakeHubService` (default `WorkbenchApp`) has no
    `execution_log` attribute at all -- `getattr`/attribute access raises
    `AttributeError`, which `_sync_audit_mode()` must catch and render an
    empty window rather than crash the whole `_sync_children()` pass."""
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._last_audit_entries == []
        workbench.set_mode("audit")
        await pilot.pause()
        assert app.query_one("#mcp-audit-empty").display is True
        table = app.query_one("#mcp-audit-table", DataTable)
        assert table.row_count == 0


class RaisingExecutionLogHubService(FakeHubService):
    """`execution_log` is a real PROPERTY that raises something other than
    `AttributeError` -- `getattr(service, "execution_log", None)` alone
    would NOT catch this (only `_sync_audit_mode()`'s explicit try/except
    around the property access itself does); regression coverage for that
    exact guard, mirroring `UnifiedMCPControlPlaneService.execution_log`'s
    own documented N1 lesson (it too can raise, not just resolve to
    `None`)."""

    @property
    def execution_log(self):
        raise RuntimeError("boom")


class RaisingExecutionLogApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = RaisingExecutionLogHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_audit_mode_guarded_when_execution_log_property_raises():
    app = RaisingExecutionLogApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()  # must not crash the whole app
        workbench = app.query_one(MCPWorkbench)
        assert workbench._last_audit_entries == []
        workbench.set_mode("audit")
        await pilot.pause()
        assert app.query_one("#mcp-audit-empty").display is True


async def _select_audit_mode_row(app: App, pilot, row: int) -> None:
    table = app.query_one("#mcp-audit-table", DataTable)
    table.focus()
    table.move_cursor(row=row)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def test_audit_entry_detail_payload_is_metadata_only():
    payload = audit_entry_detail_payload(
        {
            "ts": "2026-07-16T21:22:00+00:00",
            "server_key": "local:docs",
            "tool_name": "search",
            "initiator": "test",
            "decision": "allowed",
            "ok": True,
            "status": "success",
            "duration_ms": 1500,
            "argument_names": ["query"],
            "unknown_argument_count": 1,
            "result_type": "list",
            "result_size": 3,
            # Legacy fields must never cross the public display boundary.
            "arguments": {"api_key": "sk-super-secret", "query": "hello"},
            "result_excerpt": "3 results",
            "error": "private exception text",
        }
    )

    assert payload["tool"] == "local:docs::search"
    assert payload["duration"] == "1.5s"
    assert payload["argument_names"] == ["query"]
    assert payload["unknown_argument_count"] == 1
    assert payload["result_type"] == "list"
    assert payload["result_size"] == 3
    serialized = json.dumps(payload)
    assert "sk-super-secret" not in serialized
    assert "hello" not in serialized
    assert "3 results" not in serialized
    assert "private exception text" not in serialized


@pytest.mark.asyncio
async def test_switching_mode_away_from_audit_clears_entry_detail():
    app = AuditApp([_audit_record()])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        assert list(app.query("#mcp-inspector-audit-name"))

        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-audit-name"))
        assert app.query_one("#mcp-inspector-audit").display is False


@pytest.mark.asyncio
async def test_audit_open_tool_switches_to_tools_mode_selects_row_and_shows_detail():
    app = AuditApp([_audit_record(server_key="local:docs", tool_name="search")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)

        await pilot.click("#mcp-audit-open-tool")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        name_text = str(app.query_one("#mcp-inspector-tool-name", Static).renderable)
        assert "search" in name_text
        tools_table = app.query_one("#mcp-tools-table", DataTable)
        cursor_key, _ = tools_table.coordinate_to_cell_key((tools_table.cursor_row, 0))
        assert cursor_key.value == "local:docs::search"

        # Critical fix: the stale Audit-mode detail (with its own live
        # drill buttons) must not survive the drill-through -- it used to
        # rely on set_mode()'s _clear_tool_view() worker, which the SAME
        # exclusive "mcp-tool-clear" dispatch above cancels before it ever
        # runs (see _open_audit_tool()'s docstring).
        assert app.query_one("#mcp-inspector-audit").display is False
        assert not list(app.query("#mcp-audit-open-tool"))
        assert not list(app.query("#mcp-audit-adjust-permission"))


@pytest.mark.asyncio
async def test_audit_open_tool_missing_tool_notifies_instead_of_crashing():
    app = AuditApp([_audit_record(server_key="local:docs", tool_name="long_gone")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)

        await pilot.click("#mcp-audit-open-tool")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "audit"
        assert notifications
        message, severity = notifications[-1]
        assert message == "local:docs::long_gone: tool no longer available."
        assert severity == "warning"


# -- T8 (MCP Hub Phase 5): Audit mode Findings sub-view ----------------------
#
# Findings are a SERVER-SOURCE-ONLY concept: `_load_server_findings()`
# fetches `load_section("advanced")` and reads its `governance_audit_
# findings.items` list (the same payload key `unified_mcp_sections.
# render_advanced_section()` already extracts, `MCP/server_unified_
# service.py`'s `get_advanced()` ~:392), cached per `(source, target)`
# exactly like `_load_server_governance_profiles()` (T11).


class AuditFindingsHubService(AuditHubService):
    """`AuditHubService` (T7's real docs::fetch/docs::search/notes::
    list_notes catalog plus `execution_log`) with `context` pre-set to
    SERVER source against one active target ("main") -- mirrors
    `GovernanceCachingHubService`'s own constructor-set context, so a test
    doesn't need a rail source-switch click just to exercise the
    server-source-only findings fetch. `load_section("advanced")` returns a
    canned `governance_audit_findings` envelope; every other section falls
    through to `ToolTestHubService.load_section()` unchanged."""

    def __init__(
        self,
        records: list[dict] | None = None,
        *,
        findings_items: list[dict] | None = None,
    ) -> None:
        super().__init__(records)
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )
        self.advanced_fetch_calls = 0
        self.governance_fetch_calls = 0
        self._findings_items = (
            [
                {
                    "severity": "high",
                    "finding_type": "orphaned_path_scope",
                    "object_kind": "path_scope",
                    "object_id": "5",
                    "message": "Needs review",
                    "remediation": "Remove the unused path scope.",
                }
            ]
            if findings_items is None
            else findings_items
        )

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "advanced":
            self.advanced_fetch_calls += 1
            return {
                "server_id": "main",
                "governance_audit_findings": {"items": list(self._findings_items)},
            }
        elif effective_section == "governance":
            self.governance_fetch_calls += 1
            return {
                "server_id": "main",
                "permission_profiles": [],
            }
        return await super().load_section(section)


class MultiTargetAuditFindingsHubService(AuditFindingsHubService):
    """`AuditFindingsHubService` with a second target ("aux", via
    `TwoTargetStore`) and a per-fetch record of which target was ACTIVE
    (`self.context.selected_active_server_id`) at each advanced/governance
    `load_section()` call -- New Minor 2 (MCP Hub Phase 6 finale, linked to
    I1): `_refresh_server_discovery()` must route its refetch through the
    triggering event's OWN server_key (here: a finding whose `target_id`
    field names "aux"), not whatever target happened to already be
    active/rail-selected ("main", the mount-time default) -- this fake makes
    that distinction observable without needing the real target to actually
    return different data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_store = TwoTargetStore()
        self.advanced_fetch_targets: list[str | None] = []
        self.governance_fetch_targets: list[str | None] = []

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "advanced":
            self.advanced_fetch_targets.append(self.context.selected_active_server_id)
        elif effective_section == "governance":
            self.governance_fetch_targets.append(self.context.selected_active_server_id)
        return await super().load_section(section)


class AuditFindingsApp(ConsolidatedCSSApp):
    def __init__(
        self,
        records: list[dict] | None = None,
        *,
        findings_items: list[dict] | None = None,
        service: AuditFindingsHubService | None = None,
    ) -> None:
        super().__init__()
        self.unified_mcp_service = service or AuditFindingsHubService(
            records, findings_items=findings_items
        )

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


async def _select_findings_row(app: App, pilot, row: int) -> None:
    table = app.query_one("#mcp-audit-findings-table", DataTable)
    table.focus()
    table.move_cursor(row=row)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_audit_mode_syncs_server_findings_into_findings_subview():
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        assert service.advanced_fetch_calls == 1  # the mount-time full sync
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.row_count == 1
        row = table.get_row_at(0)
        assert [cell.plain for cell in row] == [
            "high",
            "orphaned_path_scope",
            "Needs review",
        ]


@pytest.mark.asyncio
async def test_findings_fetch_cached_across_resyncs_under_same_source_target():
    """T8, mirrors T11's own `test_space_press_resyncs_reuse_cached_
    governance_profiles`: the findings listing is STATIC server-side data
    -- a second full `_sync_children()` pass under the SAME `(source,
    target)` identity must reuse the cache, not re-fetch."""
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        assert service.advanced_fetch_calls == 1

        await workbench.reload()
        await pilot.pause()
        assert service.advanced_fetch_calls == 1


@pytest.mark.asyncio
async def test_local_source_never_fetches_findings_advanced_section():
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is False
        message = str(
            app.query_one("#mcp-audit-findings-empty-message", Static).renderable
        )
        assert message == "Findings come from a tldw_server target."


class RaisingAdvancedSectionHubService(AuditFindingsHubService):
    """`load_section("advanced")` raises -- the guard in
    `_load_server_findings()` must swallow it and leave the Findings table
    absent with the fail-soft retry-hint copy rather than crashing."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "advanced":
            self.advanced_fetch_calls += 1
            raise RuntimeError("advanced section backend unavailable")
        return await super().load_section(section)


class RaisingAdvancedSectionApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = RaisingAdvancedSectionHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_findings_fetch_failure_is_fail_soft_not_a_crash():
    app = RaisingAdvancedSectionApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()  # must not crash the whole app
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()

        table = app.query_one("#mcp-audit-findings-table", DataTable)
        assert table.display is False
        message = str(
            app.query_one("#mcp-audit-findings-empty-message", Static).renderable
        )
        assert message != "Findings come from a tldw_server target."
        assert message


@pytest.mark.asyncio
async def test_finding_selection_shows_read_only_detail_with_remediation_in_inspector():
    app = AuditFindingsApp(
        findings_items=[
            {
                "severity": "high",
                "finding_type": "orphaned_path_scope",
                "message": "Needs review",
                "remediation": "Remove the unused path scope.",
            }
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        container = app.query_one("#mcp-inspector-finding")
        assert container.display is True
        detail_text = "\n".join(
            str(static.renderable) for static in container.query(Static)
        )
        assert "high" in detail_text
        assert "orphaned_path_scope" in detail_text
        assert "Needs review" in detail_text
        assert "Remove the unused path scope." in detail_text
        # Task 2 (MCP Hub Phase 6): "orphaned_path_scope" matches none of the
        # remediation keyword buckets, so it renders the single default
        # (VIEW_DETAILS) remediation button rather than none at all.
        assert [b.id for b in container.query(Button)] == [
            "mcp-finding-action-view_details"
        ]


# -- Task 2 (MCP Hub Phase 6): finding remediation routing + per-source -----
# -- HubAction routing for `server:` keys ------------------------------------
#
# `AuditFindingsHubService`'s context pre-selects the "main" target -- per
# `MCPWorkbench.reload()`, that means `workbench._selected_server_key` is
# already "server:main" right after mount (no rail click needed), which is
# exactly "the selected rail server" fallback these tests exercise.


@pytest.mark.asyncio
async def test_finding_view_details_action_routes_to_servers_mode_using_rail_fallback():
    """The finding carries no target-identifying field, so the routed
    `server_key` falls back to the already-selected rail server
    ("server:main", set at mount by `reload()`)."""
    app = AuditFindingsApp(
        findings_items=[
            {"severity": "low", "finding_type": "stale_binding", "message": "x"}
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "server:main"
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        await pilot.click("#mcp-finding-action-view_details")
        await pilot.pause()
        assert workbench.active_mode == "servers"
        assert workbench._selected_server_key == "server:main"


@pytest.mark.asyncio
async def test_finding_view_details_action_prefers_findings_own_target_id_over_rail_selection():
    """A finding carrying its own `target_id` wins over the rail's own
    selection -- proves the "target-level server:<target_id>" derivation
    actually takes priority, not just falls through to the rail fallback."""
    app = AuditFindingsApp(
        findings_items=[
            {
                "severity": "low",
                "finding_type": "orphaned_path_scope",
                "message": "x",
                "target_id": "other",
            }
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "server:main"
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        await pilot.click("#mcp-finding-action-view_details")
        await pilot.pause()
        assert workbench.active_mode == "servers"
        assert workbench._selected_server_key == "server:other"


@pytest.mark.asyncio
async def test_finding_open_credentials_action_selects_server_and_notifies_honest_copy():
    app = AuditFindingsApp(
        findings_items=[
            {
                "severity": "high",
                "finding_type": "credential_expired",
                "message": "Server credential needs renewal.",
            }
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        notifications = _capture_notifications(app)
        await pilot.click("#mcp-finding-action-open_credentials")
        await pilot.pause()
        assert workbench.active_mode == "servers"
        assert workbench._selected_server_key == "server:main"
        assert notifications and notifications[-1] == (
            "Credentials are managed in the server's config.",
            "information",
        )


@pytest.mark.asyncio
async def test_finding_refresh_discovery_action_invalidates_caches_resyncs_and_toasts():
    """REFRESH_DISCOVERY on a `server:` key invalidates BOTH the findings
    (T8) and governance-profiles (T11) `(source, target)` caches and runs a
    full resync -- `service.advanced_fetch_calls` (the findings fetch) and
    `service.governance_fetch_calls` (the governance fetch) both go from 1
    (the mount-time sync) to 2 (this resync), proving both cache keys were
    actually reset rather than reused."""
    app = AuditFindingsApp(
        findings_items=[
            {
                "severity": "medium",
                "finding_type": "catalog_expired",
                "message": "Tool catalog is stale.",
            }
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        assert service.advanced_fetch_calls == 1
        assert service.governance_fetch_calls == 1
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        notifications = _capture_notifications(app)
        await pilot.click("#mcp-finding-action-refresh_discovery")
        await pilot.pause()
        assert service.advanced_fetch_calls == 2
        assert service.governance_fetch_calls == 2
        assert notifications and notifications[-1] == (
            "Server discovery refreshed.",
            "information",
        )


@pytest.mark.asyncio
async def test_finding_refresh_routes_to_findings_own_target_not_active_one():
    """New Minor 2 (MCP Hub Phase 6 finale, review, linked to I1):
    REFRESH_DISCOVERY from a Findings-detail remediation button must refresh
    the FINDING's own owning target -- resolved via `_finding_owning_server_
    key()`'s `target_id` field alias -- not whatever target happens to
    already be active/rail-selected. `AuditFindingsApp`'s mount-time default
    active target is "main" (both the service's own `context.selected_
    active_server_id` and, via `reload()`'s auto-select, this workbench's
    own `_selected_server_key`); this finding names "aux" as its OWN target
    via `target_id`, a genuine mismatch. Previously `_refresh_server_
    discovery()` ignored the triggering event's `server_key` entirely and
    always refreshed `_active_service_target_id()` (here: "main"), so the
    finding's real owning target ("aux") was never refetched at all."""
    service = MultiTargetAuditFindingsHubService(
        findings_items=[
            {
                "severity": "high",
                "finding_type": "catalog_expired",
                "message": "Tool catalog is stale.",
                "target_id": "aux",
            }
        ]
    )
    app = AuditFindingsApp(service=service)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._selected_server_key == "server:main"
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        service.advanced_fetch_targets.clear()
        service.governance_fetch_targets.clear()
        await pilot.click("#mcp-finding-action-refresh_discovery")
        await pilot.pause()

        assert service.advanced_fetch_targets == ["aux"]
        assert service.governance_fetch_targets == ["aux"]
        # The workbench's own selection follows the refreshed target too --
        # the NEXT fetch (e.g. a plain resync) must keep landing on it
        # rather than snapping back to "main".
        assert workbench._selected_server_key == "server:aux"


class TargetSwitchTrackingHubService(FakeHubService):
    """F1 (Qodo bot review, PR #722): `on_mcp_inspector_hub_action_
    requested()`'s VIEW_DETAILS/OPEN_CREDENTIALS `server:`-key branches used
    to set `_selected_server_key` directly and resync WITHOUT calling
    `service.select_server_target()` first -- but `_collect_snapshots()`'s
    external-servers fetch is scoped to whatever target the SERVICE ITSELF
    considers active (`self.context.selected_active_server_id`), not the
    workbench's own UI selection. A remediation button naming a target
    other than the one already active would then label/cache the OLD
    target's data under the NEW target's key.

    Two targets ("main"/"aux", via `TwoTargetStore`) so a VIEW_DETAILS
    remediation can name a genuinely non-active one. Tracks, mirroring
    `MultiTargetAuditFindingsHubService`'s own per-fetch target list but for
    the `external_servers` section `_collect_snapshots()` reads (rather
    than "advanced"/"governance"):

    - `select_server_target_calls`: every target id the service was told to
      activate, in order.
    - `external_servers_fetch_targets`: `self.context.selected_active_
      server_id` AT THE MOMENT each `external_servers` fetch ran -- proving
      which target's data actually came back, not just which key the
      workbench labeled it under.
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_store = TwoTargetStore()
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )
        self.select_server_target_calls: list[str] = []
        self.external_servers_fetch_targets: list[str | None] = []

    async def select_server_target(self, server_id):
        self.select_server_target_calls.append(server_id)
        return await super().select_server_target(server_id)

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if (
            self.context.selected_source == "server"
            and effective_section == "external_servers"
        ):
            self.external_servers_fetch_targets.append(
                self.context.selected_active_server_id
            )
            return {
                "external_servers": [{"name": "ext"}],
                "source": "server",
                "section": "external_servers",
            }
        return await super().load_section(section)


class TargetSwitchTrackingApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = TargetSwitchTrackingHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_view_details_remediation_for_non_active_target_switches_service_target_before_sync():
    """F1 (PR #722 Qodo bot review): a VIEW_DETAILS remediation button
    naming a `server:` key for a target OTHER than the one the SERVICE
    already considers active must switch the service's own active target
    BEFORE the resulting resync -- mirroring `_select_server_key()`'s
    already-correct rail/table selection path -- rather than only updating
    the workbench's local `_selected_server_key` and resyncing from
    whatever was already loaded under the OLD target.
    """
    app = TargetSwitchTrackingApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        # Mount-time reload auto-selects the service's own active target.
        assert workbench._selected_server_key == "server:main"
        assert service.external_servers_fetch_targets == ["main"]

        workbench.post_message(
            MCPInspector.HubActionRequested(HubAction.VIEW_DETAILS, "server:aux")
        )
        await pilot.pause()

        # The service was actually told to switch targets...
        assert service.select_server_target_calls == ["aux"]
        # ...and the NEXT external-servers fetch landed on "aux", not the
        # stale "main" -- proving the fetch/cache identity now matches the
        # UI-selected key rather than silently reusing "main"'s data under
        # the "aux" label.
        assert service.external_servers_fetch_targets[-1] == "aux"
        assert workbench.active_mode == "servers"
        assert workbench._selected_server_key == "server:aux"


@pytest.mark.asyncio
async def test_unrouted_action_for_server_key_shows_managed_on_server_toast_not_silent_drop():
    """CONNECT/VALIDATE/EDIT_CONFIG (local-only lifecycle seams) posted with
    a `server:` key must no longer be silently dropped."""
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        notifications = _capture_notifications(app)
        for action in (HubAction.CONNECT, HubAction.VALIDATE, HubAction.EDIT_CONFIG):
            workbench.post_message(
                MCPInspector.HubActionRequested(action, "server:main")
            )
            await pilot.pause()
        assert notifications == [("Managed on the server.", "information")] * 3


@pytest.mark.asyncio
async def test_server_source_empty_tools_diagnosis_uses_refresh_not_disabled_actions():
    """UX item 10 (Task 2, MCP Hub Phase 6): server source's empty-tools
    diagnosis must not point at connect/refresh actions that are disabled
    for server-source snapshots -- and its own "refresh" button must invoke
    the cache-invalidating resync, not just navigate to Servers mode."""
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        assert service.advanced_fetch_calls == 1
        workbench.set_mode("tools")
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        # task-3240: see the sibling comment in test_empty_diagnosis_no_
        # servers_shows_add_server_and_button_opens_form -- a trailing gate
        # breadcrumb may follow.
        assert message.startswith(
            "No tools visible from this server — refresh or check the server."
        )

        notifications = _capture_notifications(app)
        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        assert workbench.active_mode == "tools"  # unlike local source, no mode switch
        assert service.advanced_fetch_calls == 2
        assert notifications and notifications[-1] == (
            "Server discovery refreshed.",
            "information",
        )


@pytest.mark.asyncio
async def test_finding_selection_without_remediation_omits_remediation_line():
    app = AuditFindingsApp(
        findings_items=[
            {
                "severity": "low",
                "finding_type": "stale_binding",
                "message": "Check binding",
            }
        ]
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        container = app.query_one("#mcp-inspector-finding")
        detail_text = "\n".join(
            str(static.renderable) for static in container.query(Static)
        )
        assert "remediation" not in detail_text.lower()


@pytest.mark.asyncio
async def test_switching_mode_away_from_audit_clears_finding_detail():
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)
        assert app.query_one("#mcp-inspector-finding").display is True

        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#mcp-inspector-finding").display is False


@pytest.mark.asyncio
async def test_audit_adjust_permission_switches_to_permissions_mode_and_selects_row():
    app = AuditApp([_audit_record(server_key="local:docs", tool_name="search")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)

        await pilot.click("#mcp-audit-adjust-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "permissions"
        permission_text = str(
            app.query_one("#mcp-inspector-permission-tool", Static).renderable
        )
        assert "search" in permission_text
        perm_table = app.query_one("#mcp-perm-table", DataTable)
        cursor_key, _ = perm_table.coordinate_to_cell_key((perm_table.cursor_row, 0))
        assert cursor_key.value == "local:docs::search"

        # Critical fix: same stale-audit-panel hazard as the "Open tool"
        # drill above -- the previous audit-entry detail must not survive
        # under the new Permissions detail.
        assert app.query_one("#mcp-inspector-audit").display is False
        assert not list(app.query("#mcp-audit-open-tool"))
        assert not list(app.query("#mcp-audit-adjust-permission"))


@pytest.mark.asyncio
async def test_audit_adjust_permission_missing_tool_notifies_instead_of_crashing():
    app = AuditApp([_audit_record(server_key="local:docs", tool_name="long_gone")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)

        await pilot.click("#mcp-audit-adjust-permission")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.active_mode == "audit"
        assert notifications
        message, severity = notifications[-1]
        assert message == "local:docs::long_gone: tool no longer available."
        assert severity == "warning"


# -- Critical fix (T8 review): sub-view toggle must clear the OTHER pane's --
# inspector detail. `MCPAuditMode.on_button_pressed()` used to only flip
# `self._sub_view` and re-render via `_apply_subview_display()` -- no
# message ever reached the workbench, so a still-mounted `#mcp-inspector-
# audit` (Executions selection, WITH its own live "Open tool"/"Adjust
# permission" drill buttons) or `#mcp-inspector-finding` (Findings
# selection) survived a toggle to the other sub-view. Selecting a row in
# the newly-visible pane then left BOTH detail panels mounted and visible
# at once. `MCPAuditMode.SubViewChanged` + `MCPWorkbench.on_mcp_audit_
# mode_sub_view_changed()` close that gap.


@pytest.mark.asyncio
async def test_toggling_to_findings_subview_clears_stale_audit_entry_detail():
    app = AuditFindingsApp([_audit_record(server_key="local:docs", tool_name="search")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        assert app.query_one("#mcp-inspector-audit").display is True

        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#mcp-inspector-audit").display is False
        # The stale detail's own live drill buttons must not linger either
        # -- same hazard `show_audit_entry(None)` already fixes for the
        # Open-tool/Adjust-permission drill-through paths.
        assert not list(app.query("#mcp-audit-open-tool"))
        assert not list(app.query("#mcp-audit-adjust-permission"))


@pytest.mark.asyncio
async def test_toggling_to_executions_subview_clears_stale_finding_detail():
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)
        assert app.query_one("#mcp-inspector-finding").display is True

        await pilot.click("#mcp-audit-subview-executions")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#mcp-inspector-finding").display is False


@pytest.mark.asyncio
async def test_both_sub_view_selections_do_not_stack_visible_detail_panels():
    """Reproduces the exact repro in the review finding: select an
    execution row, toggle to Findings, select a finding row -- both
    `#mcp-inspector-audit` and `#mcp-inspector-finding` must never be
    visible at the same time."""
    app = AuditFindingsApp([_audit_record(server_key="local:docs", tool_name="search")])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await _select_audit_mode_row(app, pilot, 0)
        assert app.query_one("#mcp-inspector-audit").display is True

        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        assert app.query_one("#mcp-inspector-finding").display is True
        assert app.query_one("#mcp-inspector-audit").display is False


@pytest.mark.asyncio
async def test_subview_selection_persists_across_reload_resync():
    """Minor (cheap regression guard): `_sync_audit_mode()` re-pushes
    `update_entries()`/`update_findings()` into `MCPAuditMode` on EVERY
    `_sync_children()` pass (mirrors `_sync_tools_mode()`/`_sync_
    permissions_mode()`), but neither of those touches `self._sub_view` --
    a background `reload()` resync must not silently snap the visible
    sub-view back to Executions."""
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()

        await workbench.reload()
        await pilot.pause()

        assert app.query_one("#mcp-audit-findings-view", Vertical).display is True
        assert app.query_one("#mcp-audit-executions-view", Vertical).display is False
        assert app.query_one("#mcp-audit-subview-findings", Button).has_class(
            "is-active"
        )


# -- Regression tests: MCP Hub Phase 6 source-switch and stale-key guards -----


@pytest.mark.asyncio
async def test_switch_source_clears_finding_detail():
    """Regression: when switching source (local->server or server->local) while
    a finding detail is open, the finding pane (#mcp-inspector-finding) must
    become invisible and any action buttons must be cleared. Verifies the
    workbench properly responds to source switches by clearing stale
    detail state."""
    app = AuditFindingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # Server source is pre-selected in AuditFindingsApp
        assert workbench._source == "server"
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        # Finding detail is now visible with action buttons
        finding_pane = app.query_one("#mcp-inspector-finding")
        assert finding_pane.display is True
        action_buttons = list(finding_pane.query(Button))
        assert len(action_buttons) > 0, "Expected action buttons in finding detail"

        # Switch source via rail (server -> local)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Finding pane must be cleared
        assert app.query_one("#mcp-inspector-finding").display is False
        # No action buttons should remain
        assert not list(app.query_one("#mcp-inspector-finding").query(Button))


@pytest.mark.asyncio
async def test_switching_rail_server_clears_finding_detail():
    """I1 (MCP Hub Phase 6 finale, review -- the program's 6th occurrence of
    this same stale-panel class): switching the RAIL SERVER selection (not
    the source -- `_select_server_key()`'s own path, exercised here via
    `MCPRail.ServerSelected` rather than `SourceChanged`) while a
    Findings-detail pane is open must also clear it. Mirrors
    `test_switch_source_clears_finding_detail()` exactly, but for the other
    stale-panel trigger: leaving the OLD server's finding (remediation
    buttons and all -- whose own "Refresh" would silently refresh the WRONG
    target with a success toast) on screen after switching to a different
    server under the SAME source."""
    app = AuditFindingsApp()
    app.unified_mcp_service.target_store = TwoTargetStore()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._source == "server"
        assert workbench._selected_server_key == "server:main"
        workbench.set_mode("audit")
        await pilot.pause()
        await pilot.click("#mcp-audit-subview-findings")
        await pilot.pause()
        await _select_findings_row(app, pilot, 0)

        # Finding detail is now visible with action buttons
        finding_pane = app.query_one("#mcp-inspector-finding")
        assert finding_pane.display is True
        action_buttons = list(finding_pane.query(Button))
        assert len(action_buttons) > 0, "Expected action buttons in finding detail"

        # Switch the RAIL SERVER selection (same source, different target).
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected("server:aux"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._selected_server_key == "server:aux"
        # Finding pane must be cleared -- not left showing "main"'s finding
        # (and its now-mistargeted remediation buttons) under "aux".
        assert app.query_one("#mcp-inspector-finding").display is False
        assert not list(app.query_one("#mcp-inspector-finding").query(Button))


@pytest.mark.asyncio
async def test_stale_server_key_action_under_local_source_is_harmless():
    """Regression: when source is 'local' but a HubActionRequested with a
    server_key arrives (stale from a previous server-source action),
    the workbench must:
    1. Not update _selected_server_key
    2. Not fetch anything
    3. Fire a "Managed on the server." notification instead of crashing
    Covers both REFRESH_DISCOVERY and OPEN_CREDENTIALS as test examples."""
    app = WorkbenchApp()  # Uses local source
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench._source == "local"
        initial_key = workbench._selected_server_key

        notifications = _capture_notifications(app)

        # Post stale server-keyed actions (as if they arrived from audit findings)
        workbench.post_message(
            MCPInspector.HubActionRequested(HubAction.REFRESH_DISCOVERY, "server:main")
        )
        await pilot.pause()
        workbench.post_message(
            MCPInspector.HubActionRequested(HubAction.OPEN_CREDENTIALS, "server:main")
        )
        await pilot.pause()

        # Selection must not change
        assert workbench._selected_server_key == initial_key

        # The "Managed on the server." toast must appear twice (once per action)
        managed_toasts = [
            (msg, sev) for msg, sev in notifications if "Managed on the server" in msg
        ]
        assert len(managed_toasts) == 2, (
            f"Expected 2 'Managed on the server.' toasts, got {len(managed_toasts)}: "
            f"{notifications!r}"
        )


# -- task-2838: local agent tool catalog in the Hub -----------------------------

TASK_TOOL_NAMES = {"todo_create", "todo_update", "todo_get", "todo_list"}
_LOCAL_AGENT_TOOL_NAMES = {
    "fs_list",
    "fs_read",
    "fs_write",
    "fs_edit",
    "fs_patch",
    "fs_glob",
    "fs_grep",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_branches",
    "web_fetch",
    "web_search",
    "web_crawl",
    "watchlists_search_items",
    "watchlists_get_item",
}
_CONSOLE_ONLY_LOCAL_NAMES = {
    "watchlists_search_items",
    "watchlists_get_item",
    "watchlists_get_briefing",
    "watchlists_create_sources",
    "watchlists_create_collection",
    "watchlists_update_collection_sources",
    "watchlists_check_sources",
    "watchlists_set_briefing_schedule",
    "watchlists_generate_briefing",
}


def _enable_local_tools(monkeypatch):
    """Flip the workbench's `[console] local_tools_enabled` read on, leaving
    every other config key routed to the real `get_cli_setting`."""
    original = mcp_workbench_module.get_cli_setting

    def _patched(section, key, default=None):
        if section == "console" and key == "local_tools_enabled":
            return True
        return original(section, key, default)

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", _patched)
    monkeypatch.setattr(unified_service_module, "get_cli_setting", _patched)


def _missing_local_master_uses_default(monkeypatch):
    """Leave the master key absent so the production fallback is exercised."""
    original = mcp_workbench_module.get_cli_setting

    def _patched(section, key=None, default=None):
        if section == "console" and key == "local_tools_enabled":
            return default
        return original(section, key, default)

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", _patched)
    monkeypatch.setattr(unified_service_module, "get_cli_setting", _patched)


def _disable_local_tools(monkeypatch):
    original = mcp_workbench_module.get_cli_setting

    def _patched(section, key=None, default=None):
        if section == "console" and key == "local_tools_enabled":
            return False
        if section == "mcp" and key == "expose_local_tools":
            return True
        return original(section, key, default)

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", _patched)
    monkeypatch.setattr(unified_service_module, "get_cli_setting", _patched)


@pytest.mark.asyncio
async def test_tools_catalog_includes_local_agent_tools_as_own_group(monkeypatch):
    _enable_local_tools(monkeypatch)
    app = HubLocalWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()

        local = [
            t
            for t in workbench._last_hub_tools
            if t.server_key == "local:__local__" and t.name != RAW_SHELL_TOOL_NAME
        ]
        names = {t.name for t in local}
        assert _LOCAL_AGENT_TOOL_NAMES <= names
        # No Console SessionTodoStore exists at the Hub catalog layer.
        assert "todo_write" not in names
        assert TASK_TOOL_NAMES.isdisjoint(names)
        # The full ordinary catalog remains the inspection source. Only
        # descriptor-approved shared identities gain the executable flag.
        assert all(
            t.server_label == "Local workspace, web, and Watchlists" for t in local
        )
        assert all(t.source == "local" for t in local)
        assert all(
            t.executable is False for t in local if t.name in _CONSOLE_ONLY_LOCAL_NAMES
        )
        assert all(
            t.executable is True
            for t in local
            if t.name not in _CONSOLE_ONLY_LOCAL_NAMES
        )
        assert all(t.stale is False for t in local)
        # Schemas and risk tags ride along for the inspector and the
        # permission risk floor.
        assert all(t.input_schema for t in local)
        assert {t.name: t.tags for t in local}["fs_write"] == ("mutates",)
        permission_rows = app.query_one(MCPPermissionsMode)._all_rows
        local_server_row = next(
            row
            for row in permission_rows
            if row.kind == "server" and row.server_key == "local:__local__"
        )
        assert local_server_row.server_label == "Local workspace, web, and Watchlists"
        labels_by_tool = {
            row.tool_name: row.server_label
            for row in permission_rows
            if row.kind == "tool" and row.server_key == "local:__local__"
        }
        assert {
            labels_by_tool["fs_list"],
            labels_by_tool["web_fetch"],
            labels_by_tool["watchlists_search_items"],
        } == {"Local workspace, web, and Watchlists"}
        # The pre-existing sources are untouched: the fake's "docs" profile
        # tool still lists under its own key.
        assert any(t.server_key == "local:docs" for t in workbench._last_hub_tools)
        assert any(
            t.server_key == "builtin:tldw_chatbook" and t.name == "builtin_probe"
            for t in workbench._last_hub_tools
        )


@pytest.mark.asyncio
async def test_tools_catalog_lists_virtual_cli_as_an_independent_group(monkeypatch):
    _enable_local_tools(monkeypatch)
    app = HubLocalWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()

        virtual = [
            tool
            for tool in workbench._last_hub_tools
            if tool.server_key == "local:__virtual_cli__"
        ]
        assert {tool.name for tool in virtual} == {
            "ls",
            "cat",
            "grep",
            "find",
            "stat",
            "git_status",
            "git_diff",
            "git_log",
            "git_blame",
            "git_branches",
        }
        assert {tool.server_label for tool in virtual} == {"Virtual CLI (read-only)"}
        assert all("independent" in tool.description.lower() for tool in virtual)
        assert all(tool.executable is False for tool in virtual)


@pytest.mark.asyncio
async def test_hub_local_group_stays_visible_but_disabled_when_master_flag_off(
    monkeypatch,
):
    _disable_local_tools(monkeypatch)
    app = HubLocalWorkbenchApp()
    app.raw_cli_runtime = SimpleNamespace(permitted=False, armed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()
        local = [
            tool
            for tool in workbench._last_hub_tools
            if tool.server_key == "local:__local__" and tool.name != RAW_SHELL_TOOL_NAME
        ]
        assert _LOCAL_AGENT_TOOL_NAMES <= {tool.name for tool in local}
        assert all(tool.executable is False for tool in local)
        assert any(
            tool.server_key == "local:docs" for tool in workbench._last_hub_tools
        )
        assert app.query_one("#mcp-tools-local-config").display is True
        assert app.query_one("#mcp-tools-local-enabled", Checkbox).value is False


@pytest.mark.asyncio
async def test_local_agent_group_present_when_master_key_is_missing(monkeypatch):
    _missing_local_master_uses_default(monkeypatch)
    app = HubLocalWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()

        names = {
            tool.name
            for tool in workbench._last_hub_tools
            if tool.server_key == "local:__local__"
        }
        assert _LOCAL_AGENT_TOOL_NAMES <= names


@pytest.mark.asyncio
async def test_tools_mode_local_controls_round_trip_master_and_workspace(
    monkeypatch, tmp_path
):
    values: dict[tuple[str, str], Any] = {}
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get(section, key=None, default=None):
        return values.get((section, key), default)

    def fake_save(section, key, value):
        save_calls.append((section, key, value))
        values[(section, key)] = value
        return True

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get)
    monkeypatch.setattr(unified_service_module, "get_cli_setting", fake_get)
    monkeypatch.setattr(mcp_workbench_module, "save_setting_to_cli_config", fake_save)

    notes_root = tmp_path / "notes-workspace"
    notes_root.mkdir()
    app = HubLocalWorkbenchApp()
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()

        checkbox = app.query_one("#mcp-tools-local-enabled", Checkbox)
        assert checkbox.value is True
        checkbox.value = False
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("console", "local_tools_enabled", False) in save_calls
        assert app.query_one("#mcp-tools-local-enabled", Checkbox).value is False

        checkbox.value = True
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("console", "local_tools_enabled", True) in save_calls
        local_names = {
            tool.name
            for tool in workbench._last_hub_tools
            if tool.server_key == "local:__local__"
        }
        assert _LOCAL_AGENT_TOOL_NAMES <= local_names

        root_input = app.query_one("#mcp-tools-workspace-root", Input)
        root_input.value = str(notes_root)
        app.query_one("#mcp-tools-workspace-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("console", "workspace_root", str(notes_root.resolve())) in save_calls
        assert app.query_one("#mcp-tools-workspace-root", Input).value == str(
            notes_root.resolve()
        )
        status = app.query_one("#mcp-tools-local-config-status", Static)
        assert "next Console agent run" in str(status.renderable)


@pytest.mark.asyncio
async def test_tools_mode_failed_master_save_restores_persisted_truth(monkeypatch):
    def fake_get(section, key=None, default=None):
        if section == "console" and key == "local_tools_enabled":
            return True
        return default

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get)
    monkeypatch.setattr(
        mcp_workbench_module, "save_setting_to_cli_config", lambda *args: False
    )

    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()
        checkbox = app.query_one("#mcp-tools-local-enabled", Checkbox)
        assert checkbox.value is True
        checkbox.value = False
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#mcp-tools-local-enabled", Checkbox).value is True
        status = app.query_one("#mcp-tools-local-config-status", Static)
        assert "persisted setting is shown" in str(status.renderable)
        assert status.has_class("is-error")


@pytest.mark.asyncio
async def test_tools_mode_rejects_non_directory_workspace_root(monkeypatch, tmp_path):
    save_calls: list[tuple[str, str, Any]] = []
    monkeypatch.setattr(
        mcp_workbench_module,
        "save_setting_to_cli_config",
        lambda section, key, value: save_calls.append((section, key, value)) or True,
    )

    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()
        root_input = app.query_one("#mcp-tools-workspace-root", Input)
        root_input.value = str(tmp_path / "missing")
        app.query_one("#mcp-tools-workspace-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not [call for call in save_calls if call[1] == "workspace_root"]
        status = app.query_one("#mcp-tools-local-config-status", Static)
        assert "not saved" in str(status.renderable)
        assert status.has_class("is-error")


@pytest.mark.asyncio
async def test_tools_mode_workspace_root_uses_shared_path_validator(
    monkeypatch, tmp_path
):
    validated_root = tmp_path / "validated-workspace"
    validated_root.mkdir()
    validation_calls: list[tuple[Path, Path, bool, bool]] = []
    save_calls: list[tuple[str, str, Any]] = []

    def fake_validate_path(
        user_path,
        base_directory,
        *,
        redact_paths=False,
        allow_hidden=False,
    ):
        validation_calls.append(
            (
                Path(user_path),
                Path(base_directory),
                redact_paths,
                allow_hidden,
            )
        )
        return validated_root.resolve()

    monkeypatch.setattr(mcp_workbench_module, "validate_path", fake_validate_path)
    monkeypatch.setattr(
        mcp_workbench_module,
        "save_setting_to_cli_config",
        lambda section, key, value: save_calls.append((section, key, value)) or True,
    )

    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()

        root_input = app.query_one("#mcp-tools-workspace-root", Input)
        root_input.value = "relative-workspace"
        app.query_one("#mcp-tools-workspace-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(validation_calls) == 1
        candidate, validation_root, redact_paths, allow_hidden = validation_calls[0]
        assert candidate == Path.cwd() / "relative-workspace"
        assert validation_root == candidate.parent
        assert redact_paths is True
        assert allow_hidden is True
        assert (
            "console",
            "workspace_root",
            str(validated_root.resolve()),
        ) in save_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["root", "filtered_provider"])
async def test_hub_local_projection_failure_keeps_full_disabled_group(
    monkeypatch, failure_stage
):
    _enable_local_tools(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("provider construction exploded")

    if failure_stage == "root":
        monkeypatch.setattr(
            local_server_tools_module, "resolve_server_workspace_root", _boom
        )
    else:
        monkeypatch.setattr(
            local_server_tools_module, "build_hub_local_provider", _boom
        )
    app = HubLocalWorkbenchApp()
    app.raw_cli_runtime = SimpleNamespace(permitted=False, armed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        await workbench._mount_deferred_canvases()
        await workbench._sync_children()

        local = [
            tool
            for tool in workbench._last_hub_tools
            if tool.server_key == "local:__local__" and tool.name != RAW_SHELL_TOOL_NAME
        ]
        assert _LOCAL_AGENT_TOOL_NAMES <= {tool.name for tool in local}
        assert all(tool.executable is False for tool in local)
        assert any(t.server_key == "local:docs" for t in workbench._last_hub_tools)
        assert any(
            t.server_key == "builtin:tldw_chatbook" and t.name == "builtin_probe"
            for t in workbench._last_hub_tools
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("permitted", "armed", "availability"),
    [
        (False, False, "Locked —"),
        (True, False, "Unlocked, not armed —"),
        (True, True, "Armed —"),
    ],
)
async def test_raw_shell_is_always_visible_with_text_labeled_availability(
    tmp_path, permitted, armed, availability
):
    """Discoverability is stable; launch authority is visible but separate."""
    app = PermissionsApp(tmp_path / "mcp_permissions_raw_visibility.json")
    app.raw_cli_runtime = SimpleNamespace(permitted=permitted, armed=armed)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        raw_tool = next(
            tool
            for tool in workbench._last_hub_tools
            if (
                tool.server_key == RAW_SHELL_SERVER_KEY
                and tool.name == RAW_SHELL_TOOL_NAME
            )
        )

        assert raw_tool.executable is False
        assert availability in raw_tool.description
        assert "DANGER" in raw_tool.description
        assert "full authority of the OS user" in raw_tool.description
        assert "not workspace confined" in raw_tool.description

        workbench.set_mode("tools")
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(raw_tool.tool_id))
        await pilot.press("enter")
        await pilot.pause()

        assert app.focused is table
        assert _tools_table_state(app, RAW_SHELL_TOOL_NAME).startswith("Ask")
        description = str(
            app.query_one("#mcp-inspector-tool-description", Static).renderable
        )
        assert availability in description
        assert "DANGER" in description
        assert "not workspace confined" in description
        phase_note = str(
            app.query_one("#mcp-inspector-tool-phase-note", Static).renderable
        )
        assert phase_note == (
            "Policy only — raw shell commands run from Console under its "
            "separate approval flow."
        )


@pytest.mark.asyncio
async def test_raw_shell_hand_edited_allow_renders_ask_and_cycles_only_ask_off(
    monkeypatch, tmp_path
):
    """The exact raw-shell row has no persistent Allow or Inherit rung."""
    _enable_local_tools(monkeypatch)
    store_path = tmp_path / "mcp_permissions_raw_cycle.json"
    app = PermissionsApp(store_path)
    app.raw_cli_runtime = SimpleNamespace(permitted=True, armed=True)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        raw_tool = workbench._tool_for(RAW_SHELL_SERVER_KEY, RAW_SHELL_TOOL_NAME)
        assert raw_tool is not None

        # Simulate a valid hand-edited Allow entry without triggering the
        # definition-change downgrade; the UI must still project it to Ask.
        app.unified_mcp_service.set_tool_state(
            RAW_SHELL_SERVER_KEY,
            RAW_SHELL_TOOL_NAME,
            "allow",
            tool=raw_tool,
        )
        await workbench._sync_children()
        await pilot.pause()

        assert _tools_table_state(app, RAW_SHELL_TOOL_NAME).startswith("Ask")
        assert not _tools_table_state(app, RAW_SHELL_TOOL_NAME).startswith("Allow")

        workbench.set_mode("permissions")
        await pilot.pause()
        raw_row = next(
            row
            for row in app.query_one(MCPPermissionsMode)._all_rows
            if (
                row.kind == "tool"
                and row.server_key == RAW_SHELL_SERVER_KEY
                and row.tool_name == RAW_SHELL_TOOL_NAME
            )
        )
        assert raw_row.state_label.startswith("Ask")

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        raw_row_index = _perm_row_keys(app).index(
            f"{RAW_SHELL_SERVER_KEY}::{RAW_SHELL_TOOL_NAME}"
        )
        table.move_cursor(row=raw_row_index)
        await pilot.press("enter")
        await pilot.pause()
        assert (
            str(
                app.query_one(
                    "#mcp-inspector-permission-cascade-tool", Static
                ).renderable
            )
            == "▸ Tool override: Ask •"
        )

        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry(
            RAW_SHELL_SERVER_KEY, RAW_SHELL_TOOL_NAME
        )
        assert entry is not None and entry["state"] == "deny"
        assert _tools_table_state(app, RAW_SHELL_TOOL_NAME).startswith("Off")

        table.move_cursor(
            row=_perm_row_keys(app).index(
                f"{RAW_SHELL_SERVER_KEY}::{RAW_SHELL_TOOL_NAME}"
            )
        )
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry(
            RAW_SHELL_SERVER_KEY, RAW_SHELL_TOOL_NAME
        )
        assert entry is not None and entry["state"] == "ask"
        assert _tools_table_state(app, RAW_SHELL_TOOL_NAME).startswith("Ask")


@pytest.mark.asyncio
async def test_virtual_cli_permission_cycle_remains_independent_of_raw_shell(
    monkeypatch, tmp_path
):
    """Raw-shell coercion must not remove Allow from virtual CLI commands."""
    _enable_local_tools(monkeypatch)
    store_path = tmp_path / "mcp_permissions_virtual_cli_independent.json"
    app = PermissionsApp(store_path)
    app.raw_cli_runtime = SimpleNamespace(permitted=True, armed=True)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()

        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=_perm_row_keys(app).index("local:__virtual_cli__::ls"))
        await pilot.press("space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        entry = MCPPermissionStore(store_path).get_tool_entry(
            "local:__virtual_cli__", "ls"
        )
        assert entry is not None and entry["state"] == "allow"
        assert _tools_table_state(app, "ls").startswith("Allow")
