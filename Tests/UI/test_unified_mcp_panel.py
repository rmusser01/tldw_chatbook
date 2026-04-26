from __future__ import annotations

from dataclasses import replace

import pytest

from textual.app import App, ComposeResult
from textual.widgets import Select, Static, TextArea

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import (
    ConfiguredServerTarget,
    ServerAccessContext,
    SectionCapabilityFlags,
    UnifiedMCPContext,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.MCP_Modules.unified_mcp_panel import UnifiedMCPPanel


class FakeUnifiedMCPService:
    def __init__(self, target_store: ConfiguredServerTargetStore) -> None:
        self.target_store = target_store
        self.context = UnifiedMCPContext(selected_source="local", selected_section="overview")
        self.action_calls: list[tuple[str, dict]] = []

    async def load_context(self) -> UnifiedMCPContext:
        return self.context

    async def select_source(self, source: str) -> UnifiedMCPContext:
        normalized_source = "server" if source == "server" else "local"
        self.context = replace(self.context, selected_source=normalized_source)
        if normalized_source == "server" and self.context.selected_active_server_id is None:
            default_target = self.target_store.resolve_active_target()
            if default_target is not None:
                return await self.select_server_target(default_target.server_id)
        return self.context

    async def select_server_target(self, server_id: str) -> UnifiedMCPContext:
        target = self.target_store.resolve_active_target(server_id)
        if target is None:
            raise KeyError(server_id)
        context = ServerAccessContext(
            server_id=target.server_id,
            selected_scope="personal",
            selected_scope_ref=None,
            selected_section=self.context.selected_section or "overview",
            can_use_personal_scope=True,
            manageable_team_ids=(21,),
            manageable_org_ids=(11,),
            can_use_system_admin_scope=True,
            section_capabilities=SectionCapabilityFlags(
                overview=True,
                inventory=True,
                catalogs=True,
                external_servers=True,
                governance=True,
                advanced=True,
            ),
        )
        per_server_state = dict(self.context.per_server_state)
        per_server_state[target.server_id] = context
        self.context = replace(
            self.context,
            selected_source="server",
            selected_active_server_id=target.server_id,
            selected_scope=context.selected_scope,
            selected_scope_ref=context.selected_scope_ref,
            selected_section=context.selected_section,
            per_server_state=per_server_state,
        )
        return self.context

    async def select_scope(self, scope: str | None, scope_ref: str | None = None) -> UnifiedMCPContext:
        server_id = self.context.selected_active_server_id
        if server_id is None:
            return self.context
        if scope == "team" and scope_ref is None:
            scope_ref = "21"
        elif scope == "org" and scope_ref is None:
            scope_ref = "11"
        else:
            scope_ref = None if scope in {None, "personal", "system_admin"} else scope_ref
        context = replace(
            self.context.per_server_state[server_id],
            selected_scope=scope or "personal",
            selected_scope_ref=scope_ref,
        )
        per_server_state = dict(self.context.per_server_state)
        per_server_state[server_id] = context
        self.context = replace(
            self.context,
            selected_scope=context.selected_scope,
            selected_scope_ref=context.selected_scope_ref,
            per_server_state=per_server_state,
        )
        return self.context

    async def select_section(self, section: str | None) -> UnifiedMCPContext:
        server_id = self.context.selected_active_server_id
        if server_id is None:
            self.context = replace(self.context, selected_section=section or "overview")
            return self.context
        context = replace(
            self.context.per_server_state[server_id],
            selected_section=section or "overview",
        )
        per_server_state = dict(self.context.per_server_state)
        per_server_state[server_id] = context
        self.context = replace(
            self.context,
            selected_section=context.selected_section,
            per_server_state=per_server_state,
        )
        return self.context

    async def load_section(self, section: str | None = None) -> dict:
        effective_section = section or self.context.selected_section or "overview"
        base = {
            "source": self.context.selected_source,
            "section": effective_section,
            "server_id": self.context.selected_active_server_id,
            "scope": self.context.selected_scope,
            "scope_ref": self.context.selected_scope_ref,
        }
        if effective_section == "catalogs":
            return {**base, "catalogs": [{"id": 9, "name": "Scoped Catalog"}]}
        if effective_section == "external_servers":
            return {**base, "external_servers": [{"id": "docs", "name": "Docs"}]}
        if effective_section == "governance" and self.context.selected_source == "local":
            return {
                **base,
                "rules": [
                    {
                        "rule_id": "rule-a",
                        "capability_id": "mcp.inventory.list.local",
                        "decision": "allow",
                    }
                ],
            }
        if effective_section == "advanced" and self.context.selected_source == "local":
            return {
                **base,
                "runtime_status": {
                    "server_id": "local:tldw_chatbook",
                    "server_label": "tldw_chatbook local MCP",
                    "tool_count": 2,
                },
                "protocol": {
                    "adapter": "direct_in_process",
                    "supports_batch": True,
                    "request_methods": ["tools/list", "resources/list", "prompts/list"],
                },
            }
        if effective_section == "governance":
            return {
                **base,
                "permission_profiles": [{"id": 1, "name": "Default"}],
                "policy_assignments": [{"id": 2, "target_id": "persona-a"}],
                "approval_policies": [{"id": 7, "name": "Default Approval"}],
                "acp_profiles": [{"id": 8, "name": "Workspace ACP"}],
                "effective_policy": {"enabled": True},
                "cache_mode": "live",
            }
        if effective_section == "advanced":
            return {
                **base,
                "tool_registry_summary": {"modules": [{"module": "search"}], "entries": [{"tool_name": "docs.search"}]},
                "tool_registry_entries": [{"tool_name": "docs.search"}],
                "tool_registry_modules": [{"module": "search"}],
                "governance_packs": [{"id": 4, "name": "Baseline"}],
                "cache_mode": "stale_allowed",
            }
        return base

    def available_actions(self) -> list[dict]:
        if self.context.selected_source == "local" and self.context.selected_section == "inventory":
            return [
                {
                    "name": "tool.execute",
                    "label": "Execute Local Tool",
                    "action_id": "mcp.runtime.trigger.local",
                    "payload_template": '{"tool_name":"search_notes","arguments":{"query":"example"}}',
                },
                {
                    "name": "resource.read",
                    "label": "Read Local Resource",
                    "action_id": "mcp.inventory.observe.local",
                    "payload_template": '{"resource_uri":"note://123"}',
                },
                {
                    "name": "prompt.get",
                    "label": "Get Local Prompt",
                    "action_id": "mcp.inventory.observe.local",
                    "payload_template": '{"prompt_name":"summarize_conversation","arguments":{"conversation_id":4}}',
                },
            ]
        if self.context.selected_source == "local" and self.context.selected_section == "external_servers":
            return [
                {
                    "name": "profile.save",
                    "label": "Save Profile",
                    "action_id": "mcp.external_profiles.configure.local",
                    "payload_template": '{"profile_id":"demo","command":"python","args":["-m","demo.server"]}',
                }
            ]
        if self.context.selected_source == "local" and self.context.selected_section == "governance":
            return [
                {
                    "name": "governance_rule.save",
                    "label": "Save Governance Rule",
                    "action_id": "mcp.governance.configure.local",
                    "payload_template": '{"rule_id":"rule-a","capability_id":"mcp.inventory.list.local","decision":"allow"}',
                },
                {
                    "name": "governance_rule.preview",
                    "label": "Preview Governance Decision",
                    "action_id": "mcp.governance.observe.local",
                    "payload_template": '{"capability_id":"mcp.inventory.list.local"}',
                },
                {
                    "name": "governance_rule.delete",
                    "label": "Delete Governance Rule",
                    "action_id": "mcp.governance.configure.local",
                    "payload_template": '{"rule_id":"rule-a"}',
                },
            ]
        if self.context.selected_source == "local" and self.context.selected_section == "advanced":
            return [
                {
                    "name": "runtime.access.preview",
                    "label": "Preview Local Runtime Access",
                    "action_id": "mcp.governance.observe.local",
                    "payload_template": '{"action_name":"tool.execute","payload":{"tool_name":"search_notes","arguments":{"query":"example"}}}',
                },
                {
                    "name": "approval_requests.list",
                    "label": "List Local Approval Requests",
                    "action_id": "mcp.governance.observe.local",
                    "payload_template": '{}',
                },
                {
                    "name": "approval_request.approve",
                    "label": "Approve Local Request",
                    "action_id": "mcp.governance.approve.local",
                    "payload_template": '{"request_id":"approval-a"}',
                },
                {
                    "name": "approval_request.deny",
                    "label": "Deny Local Request",
                    "action_id": "mcp.governance.approve.local",
                    "payload_template": '{"request_id":"approval-a"}',
                },
                {
                    "name": "runtime.activity.list",
                    "label": "List Local Runtime Activity",
                    "action_id": "mcp.runtime.observe.local",
                    "payload_template": '{"limit":5}',
                },
                {
                    "name": "runtime.protocol.inspect",
                    "label": "Inspect Local Protocol",
                    "action_id": "mcp.runtime.observe.local",
                    "payload_template": '{}',
                },
                {
                    "name": "runtime.health.get",
                    "label": "Get Local Runtime Health",
                    "action_id": "mcp.runtime.observe.local",
                    "payload_template": '{}',
                },
                {
                    "name": "runtime.status.get",
                    "label": "Get Local Runtime Status",
                    "action_id": "mcp.runtime.observe.local",
                    "payload_template": '{}',
                },
                {
                    "name": "runtime.request",
                    "label": "Send Local Runtime Request",
                    "action_id": "mcp.runtime.trigger.local",
                    "payload_template": '{"method":"tools/list","params":{}}',
                },
                {
                    "name": "runtime.batch",
                    "label": "Run Local Runtime Batch",
                    "action_id": "mcp.runtime.trigger.local",
                    "payload_template": '{"requests":[{"method":"tools/list"},{"method":"prompts/list"}]}',
                },
            ]
        if self.context.selected_source == "server" and self.context.selected_section == "catalogs":
            return [
                {
                    "name": "catalog.create",
                    "label": "Create Catalog",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"name":"Team Catalog","description":"Scoped"}',
                }
            ]
        if self.context.selected_source == "server" and self.context.selected_section == "external_servers":
            return [
                {
                    "name": "external_server.secret.set",
                    "label": "Set External Server Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","secret":"replace-me"}',
                }
            ]
        if self.context.selected_source == "server" and self.context.selected_section == "governance":
            return [
                {
                    "name": "permission_profile.create",
                    "label": "Create Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Default","mode":"custom","policy_document":{}}',
                },
                {
                    "name": "policy_assignment.workspaces.list",
                    "label": "List Assignment Workspaces",
                    "action_id": "mcp.governance.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "permission_profile.bindings.list",
                    "label": "List Profile Bindings",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"profile_id":1}',
                },
                {
                    "name": "policy_assignment.binding.upsert",
                    "label": "Upsert Assignment Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","binding_mode":"grant","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "policy_assignment.slot_status.get",
                    "label": "Get Assignment Slot Status",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly"}',
                }
            ]
        if self.context.selected_source == "server" and self.context.selected_section == "advanced":
            return [
                {
                    "name": "governance_pack.dry_run",
                    "label": "Dry Run Governance Pack",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"pack":{"manifest":{"pack_id":"baseline","version":"1.0.0"}}}',
                },
                {
                    "name": "governance_pack.source.prepare",
                    "label": "Prepare Governance Pack Source",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"source":{"kind":"git","url":"git@example.com:trusted/repo.git","ref":"main"}}',
                },
                {
                    "name": "governance_pack.import",
                    "label": "Import Governance Pack",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"pack":{"manifest":{"pack_id":"baseline","version":"1.0.0"}}}',
                },
                {
                    "name": "governance_pack.detail.get",
                    "label": "Get Governance Pack Detail",
                    "action_id": "mcp.advanced.observe.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "capability_mapping.preview",
                    "label": "Preview Capability Mapping",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"mapping_id":"filesystem-write","capability_name":"filesystem.write"}',
                },
                {
                    "name": "path_scope_object.create",
                    "label": "Create Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Workspace Root","path_scope_document":{"path_scope_mode":"workspace_root"}}',
                },
                {
                    "name": "workspace_set_object.create",
                    "label": "Create Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Research Set"}',
                },
                {
                    "name": "shared_workspace.create",
                    "label": "Create Shared Workspace",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_id":"shared-ws","display_name":"Shared Workspace","absolute_root":"/srv/shared"}',
                },
            ]
        return []

    def runtime_state_override(self) -> RuntimeSourceState:
        if self.context.selected_source == "server":
            return RuntimeSourceState(
                active_source="server",
                active_server_id=self.context.selected_active_server_id,
                server_configured=True,
                server_reachability="reachable",
                server_auth_state="authenticated",
                last_known_server_label=self.context.selected_active_server_id,
            )
        return RuntimeSourceState(active_source="local")

    async def run_action(self, action_name: str, payload: dict) -> dict:
        self.action_calls.append((action_name, dict(payload)))
        return {"ok": True, "action": action_name, **dict(payload)}


class UnifiedMCPPanelApp(App):
    def __init__(self, service: FakeUnifiedMCPService):
        super().__init__()
        self.notify_messages: list[tuple[str, str]] = []
        self.unified_mcp_service = service

    def notify(self, message: str, severity: str = "information") -> None:
        self.notify_messages.append((message, severity))

    def compose(self) -> ComposeResult:
        yield UnifiedMCPPanel(app_instance=self)


@pytest.mark.asyncio
async def test_unified_mcp_panel_switches_between_local_and_server_views(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    app = UnifiedMCPPanelApp(FakeUnifiedMCPService(target_store))

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_source("server")
        await pilot.pause()

        server_select = panel.query_one("#unified-mcp-server-target", Select)
        content = panel.query_one("#unified-mcp-content", Static)

        assert panel.context.selected_source == "server"
        assert panel.context.selected_active_server_id == "server-a"
        assert server_select.disabled is False
        assert "server-a" in str(content.content)


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_scope_and_section_controls_for_server_context(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    app = UnifiedMCPPanelApp(FakeUnifiedMCPService(target_store))

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_source("server")
        await panel.select_scope("team")
        await panel.select_section("inventory")
        await pilot.pause()

        scope_ref_select = panel.query_one("#unified-mcp-scope-ref", Select)
        section_select = panel.query_one("#unified-mcp-section", Select)
        content = panel.query_one("#unified-mcp-content", Static)

        assert panel.context.selected_scope == "team"
        assert panel.context.selected_scope_ref == "21"
        assert panel.context.selected_section == "inventory"
        assert scope_ref_select.disabled is False
        assert section_select.value == "inventory"
        assert "inventory" in str(content.content).lower()


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_slice2_sections_for_local_and_server_contexts(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    app = UnifiedMCPPanelApp(FakeUnifiedMCPService(target_store))

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        section_select = panel.query_one("#unified-mcp-section", Select)
        local_option_values = [option[0] for option in section_select._options]

        await panel.select_section("external_servers")
        await panel.select_source("server")
        await panel.select_section("catalogs")
        await pilot.pause()

        server_option_values = [option[0] for option in section_select._options]
        content = panel.query_one("#unified-mcp-content", Static)

        assert "External Servers" in local_option_values
        assert "Catalogs" in server_option_values
        assert "External Servers" in server_option_values
        assert "scoped catalog" in str(content.content).lower()


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_local_governance_section_and_actions(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        section_select = panel.query_one("#unified-mcp-section", Select)
        local_option_values = [option[0] for option in section_select._options]

        await panel.select_section("governance")
        await pilot.pause()

        action_select = panel.query_one("#unified-mcp-action", Select)
        content = panel.query_one("#unified-mcp-content", Static)
        action_values = [option[0] for option in action_select._options]

        assert "Governance" in local_option_values
        assert action_select.disabled is False
        assert "Save Governance Rule" in action_values
        assert "Preview Governance Decision" in action_values
        assert "Delete Governance Rule" in action_values
        assert "rule-a" in str(content.content).lower()


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_local_inventory_runtime_actions(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_section("inventory")
        await pilot.pause()

        action_select = panel.query_one("#unified-mcp-action", Select)
        action_values = [option[0] for option in action_select._options]

        assert action_select.disabled is False
        assert "Execute Local Tool" in action_values
        assert "Read Local Resource" in action_values
        assert "Get Local Prompt" in action_values


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_local_advanced_runtime_actions(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_section("advanced")
        await pilot.pause()

        section_select = panel.query_one("#unified-mcp-section", Select)
        action_select = panel.query_one("#unified-mcp-action", Select)
        content = panel.query_one("#unified-mcp-content", Static)

        section_values = [option[0] for option in section_select._options]
        action_values = [option[0] for option in action_select._options]

        assert "Advanced" in section_values
        assert action_select.disabled is False
        assert "local mcp" in str(content.content).lower()
        assert "Preview Local Runtime Access" in action_values
        assert "List Local Runtime Activity" in action_values
        assert "Inspect Local Protocol" in action_values
        assert "Get Local Runtime Health" in action_values
        assert "List Local Approval Requests" in action_values
        assert "Approve Local Request" in action_values
        assert "Deny Local Request" in action_values
        assert "Get Local Runtime Status" in action_values
        assert "Send Local Runtime Request" in action_values
        assert "Run Local Runtime Batch" in action_values


@pytest.mark.asyncio
async def test_unified_mcp_panel_executes_minimal_action_runner(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_section("external_servers")
        payload_area = panel.query_one("#unified-mcp-action-payload", TextArea)
        payload_area.text = '{"profile_id":"local-b","command":"python"}'

        await panel.execute_selected_action()
        await pilot.pause()

        result = panel.query_one("#unified-mcp-action-result", Static)

        assert service.action_calls == [("profile.save", {"profile_id": "local-b", "command": "python"})]
        assert "local-b" in str(result.content)


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_governance_section_and_actions(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_source("server")
        await panel.select_section("governance")
        await pilot.pause()

        section_select = panel.query_one("#unified-mcp-section", Select)
        action_select = panel.query_one("#unified-mcp-action", Select)
        content = panel.query_one("#unified-mcp-content", Static)

        section_values = [option[0] for option in section_select._options]
        assert "Governance" in section_values
        assert action_select.disabled is False
        assert "permission profiles" in str(content.content).lower()


@pytest.mark.asyncio
async def test_unified_mcp_panel_exposes_advanced_section_and_actions(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    service = FakeUnifiedMCPService(target_store)
    app = UnifiedMCPPanelApp(service)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(UnifiedMCPPanel)

        await panel.load_context()
        await panel.select_source("server")
        await panel.select_section("advanced")
        await pilot.pause()

        section_select = panel.query_one("#unified-mcp-section", Select)
        action_select = panel.query_one("#unified-mcp-action", Select)
        content = panel.query_one("#unified-mcp-content", Static)

        section_values = [option[0] for option in section_select._options]
        action_values = [option[0] for option in action_select._options]
        assert "Advanced" in section_values
        assert action_select.disabled is False
        assert "baseline" in str(content.content).lower()
        assert "Dry Run Governance Pack" in action_values
        assert "Prepare Governance Pack Source" in action_values
        assert "Import Governance Pack" in action_values
        assert "Get Governance Pack Detail" in action_values
        assert "Preview Capability Mapping" in action_values
        assert "Create Workspace Set" in action_values
        assert "Create Shared Workspace" in action_values
