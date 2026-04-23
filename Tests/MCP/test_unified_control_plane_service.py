from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_models import (
    ConfiguredServerTarget,
    SectionCapabilityFlags,
    ServerAccessContext,
    UnifiedMCPContext,
)


class FakeLocalMCPControlService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.action_calls: list[tuple[str, dict]] = []

    def get_overview(self) -> dict:
        self.calls.append("overview")
        return {"source": "local", "section": "overview"}

    def get_inventory(self) -> dict:
        self.calls.append("inventory")
        return {"source": "local", "section": "inventory"}

    def get_external_servers(self) -> dict:
        self.calls.append("external_servers")
        return {"source": "local", "section": "external_servers", "profiles": [{"profile_id": "local-a"}]}

    def get_governance(self) -> list[dict]:
        self.calls.append("governance")
        return [
            {
                "rule_id": "rule-a",
                "capability_id": "mcp.inventory.list.local",
                "decision": "allow",
            }
        ]

    def get_advanced(self) -> dict:
        self.calls.append("advanced")
        return {
            "source": "local",
            "section": "advanced",
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

    def save_external_profile(self, payload: dict) -> dict:
        self.action_calls.append(("profile.save", dict(payload)))
        return dict(payload)

    def save_governance_rule(self, payload: dict) -> dict:
        self.action_calls.append(("governance_rule.save", dict(payload)))
        return dict(payload)

    def delete_governance_rule(self, rule_id: str) -> bool:
        self.action_calls.append(("governance_rule.delete", {"rule_id": rule_id}))
        return True

    def preview_governance_decision(self, capability_id: str) -> dict:
        self.action_calls.append(("governance_rule.preview", {"capability_id": capability_id}))
        return {
            "source": "local",
            "capability_id": capability_id,
            "decision": "allow",
            "matched_rule_id": "rule-a",
            "notes": "Inventory is allowed locally.",
        }

    async def execute_tool(self, tool_name: str, arguments: dict | None = None) -> dict:
        payload = {"tool_name": tool_name, "arguments": dict(arguments or {})}
        self.action_calls.append(("tool.execute", payload))
        return {"source": "local", "tool_name": tool_name, "result": dict(arguments or {})}

    async def read_resource(self, resource_uri: str) -> dict:
        payload = {"resource_uri": resource_uri}
        self.action_calls.append(("resource.read", payload))
        return {"source": "local", "resource_uri": resource_uri, "result": {"content": "resource-body"}}

    async def get_prompt(self, prompt_name: str, arguments: dict | None = None) -> dict:
        payload = {"prompt_name": prompt_name, "arguments": dict(arguments or {})}
        self.action_calls.append(("prompt.get", payload))
        return {
            "source": "local",
            "prompt_name": prompt_name,
            "arguments": dict(arguments or {}),
            "messages": [{"role": "assistant", "content": "prompt-body"}],
        }

    def get_runtime_status(self) -> dict:
        self.action_calls.append(("runtime.status.get", {}))
        return {
            "source": "local",
            "status": {
                "server_id": "local:tldw_chatbook",
                "server_label": "tldw_chatbook local MCP",
                "tool_count": 2,
            },
        }

    def get_runtime_health(self) -> dict:
        self.action_calls.append(("runtime.health.get", {}))
        return {
            "source": "local",
            "health": {
                "state": "ready",
                "adapter": "direct_in_process",
                "manifest": {"tools": 2, "resources": 1, "prompts": 1},
            },
        }

    def get_runtime_activity(self, limit: int = 20) -> dict:
        self.action_calls.append(("runtime.activity.list", {"limit": limit}))
        return {
            "source": "local",
            "limit": limit,
            "entries": [
                {
                    "action_name": "tool.execute",
                    "target": "search_notes",
                    "ok": True,
                }
            ],
        }

    def get_runtime_protocol_diagnostics(self) -> dict:
        self.action_calls.append(("runtime.protocol.inspect", {}))
        return {
            "source": "local",
            "diagnostics": {
                "protocol_version": "2025-03-26",
                "transport": "in_process",
                "manifest": {"tools": 2, "resources": 1, "prompts": 1},
            },
        }

    def list_approval_requests(self, status: str | None = None, resolved_action_id: str | None = None) -> list[dict]:
        payload = {}
        if status is not None:
            payload["status"] = status
        if resolved_action_id is not None:
            payload["resolved_action_id"] = resolved_action_id
        self.action_calls.append(("approval_requests.list", payload))
        return [
            {
                "request_id": "approval-a",
                "status": status or "pending",
                "resolved_action_id": resolved_action_id or "notes.list.local",
            }
        ]

    def approve_approval_request(self, request_id: str) -> dict:
        self.action_calls.append(("approval_request.approve", {"request_id": request_id}))
        return {"request_id": request_id, "status": "approved"}

    def deny_approval_request(self, request_id: str) -> dict:
        self.action_calls.append(("approval_request.deny", {"request_id": request_id}))
        return {"request_id": request_id, "status": "denied"}

    def delete_approval_request(self, request_id: str) -> bool:
        self.action_calls.append(("approval_request.delete", {"request_id": request_id}))
        return True

    def preview_runtime_access(self, action_name: str, payload: dict | None = None) -> dict:
        entry = {"action_name": action_name, "payload": dict(payload or {})}
        self.action_calls.append(("runtime.access.preview", entry))
        return {
            "source": "local",
            "action_name": action_name,
            "resolved_action_id": "notes.list.local",
            "registry_capability_id": "notes_workspaces",
            "decision": "deny",
            "matched_rule_id": "rule-deny-notes-list",
            "notes": "Local note listing is blocked.",
            "approval_request_id": None,
            "approval_status": None,
        }

    async def run_runtime_request(self, method: str, params: dict | None = None) -> dict:
        payload = {"method": method, "params": dict(params or {})}
        self.action_calls.append(("runtime.request", payload))
        return {
            "source": "local",
            "method": method,
            "params": dict(params or {}),
            "result": {"ok": True, "method": method},
        }

    async def run_runtime_batch(self, requests: list[dict]) -> dict:
        payload = {"requests": [dict(item) for item in requests]}
        self.action_calls.append(("runtime.batch", payload))
        return {
            "source": "local",
            "results": [
                {"index": index, "method": request.get("method"), "ok": True}
                for index, request in enumerate(requests)
            ],
        }


class FakeServerUnifiedMCPService:
    def __init__(self) -> None:
        self.resolve_calls: list[tuple[str, str | None, str | None, str | None]] = []
        self.overview_calls: list[tuple[str, str | None, str | None]] = []
        self.inventory_calls: list[tuple[str, str | None, str | None]] = []
        self.catalog_calls: list[tuple[str, str | None, str | None]] = []
        self.external_calls: list[tuple[str, str | None, str | None]] = []
        self.governance_calls: list[tuple[str, str | None, str | None]] = []
        self.advanced_calls: list[tuple[str, str | None, str | None]] = []
        self.action_calls: list[tuple[str, str, str | None, dict]] = []

    async def resolve_access_context(
        self,
        *,
        target,
        selected_scope=None,
        selected_scope_ref=None,
        selected_section=None,
    ) -> ServerAccessContext:
        self.resolve_calls.append((target.server_id, selected_scope, selected_scope_ref, selected_section))
        effective_scope = selected_scope or "personal"
        effective_scope_ref = selected_scope_ref if effective_scope in {"team", "org"} else None
        effective_section = selected_section or "overview"
        return ServerAccessContext(
            server_id=target.server_id,
            selected_scope=effective_scope,
            selected_scope_ref=effective_scope_ref,
            selected_section=effective_section,
            can_use_personal_scope=True,
            manageable_team_ids=(21,),
            manageable_org_ids=(11,),
            section_capabilities=SectionCapabilityFlags(
                overview=True,
                inventory=True,
                catalogs=True,
                external_servers=True,
                governance=True,
                advanced=True,
            ),
        )

    async def get_overview(self, *, target, access_context):
        self.overview_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "overview",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
        }

    async def get_inventory(self, *, target, access_context):
        self.inventory_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "inventory",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
        }

    async def get_catalogs(self, *, target, access_context):
        self.catalog_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "catalogs",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
            "catalogs": [{"id": 9, "name": "Scoped Catalog"}],
        }

    async def get_external_servers(self, *, target, access_context):
        self.external_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "external_servers",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
            "external_servers": [{"id": "docs", "name": "Docs"}],
        }

    async def get_governance(self, *, target, access_context):
        self.governance_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "governance",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
            "permission_profiles": [{"id": 1, "name": "Default"}],
            "policy_assignments": [{"id": 2, "target_id": "persona-a"}],
            "approval_policies": [{"id": 7, "name": "Default Approval"}],
            "acp_profiles": [{"id": 8, "name": "Workspace ACP"}],
            "effective_policy": {"enabled": True},
        }

    async def create_catalog(self, *, target, access_context, payload):
        self.action_calls.append(("catalog.create", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"id": 9, **dict(payload)}

    async def update_approval_policy(self, *, target, access_context, approval_policy_id, payload):
        self.action_calls.append(
            ("approval_policy.update", target.server_id, access_context.selected_scope_ref, {"approval_policy_id": approval_policy_id, **dict(payload)})
        )
        return {"id": approval_policy_id, **dict(payload)}

    async def get_advanced(self, *, target, access_context):
        self.advanced_calls.append((target.server_id, access_context.selected_scope, access_context.selected_scope_ref))
        return {
            "source": "server",
            "section": "advanced",
            "server_id": target.server_id,
            "scope": access_context.selected_scope,
            "scope_ref": access_context.selected_scope_ref,
            "tool_registry_summary": {"modules": [{"module": "search"}], "entries": [{"tool_name": "docs.search"}]},
            "tool_registry_entries": [{"tool_name": "docs.search"}],
            "tool_registry_modules": [{"module": "search"}],
            "governance_packs": [{"id": 4, "name": "Baseline"}],
            "cache_mode": "stale_allowed",
        }

    async def create_path_scope_object(self, *, target, access_context, payload):
        self.action_calls.append(("path_scope_object.create", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"id": 11, **dict(payload)}

    async def preview_capability_mapping(self, *, target, access_context, payload):
        self.action_calls.append(("capability_mapping.preview", target.server_id, access_context.selected_scope_ref, dict(payload)))
        merged = dict(payload)
        merged.setdefault("owner_scope_type", access_context.selected_scope)
        if access_context.selected_scope_ref is not None:
            merged.setdefault("owner_scope_id", int(access_context.selected_scope_ref))
        return {"normalized_mapping": merged}

    async def create_workspace_set_object(self, *, target, access_context, payload):
        self.action_calls.append(("workspace_set_object.create", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"id": 12, **dict(payload)}

    async def list_workspace_set_members(self, *, target, access_context, workspace_set_object_id):
        self.action_calls.append(("workspace_set_object.members.list", target.server_id, access_context.selected_scope_ref, {"workspace_set_object_id": workspace_set_object_id}))
        return [{"workspace_id": "ws-1"}]

    async def create_shared_workspace(self, *, target, access_context, payload):
        self.action_calls.append(("shared_workspace.create", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"id": 13, **dict(payload)}

    async def list_policy_assignment_workspaces(self, *, target, access_context, assignment_id):
        self.action_calls.append(("policy_assignment.workspaces.list", target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id}))
        return [{"workspace_id": "ws-1"}]

    async def add_policy_assignment_workspace(self, *, target, access_context, assignment_id, workspace_id):
        self.action_calls.append(("policy_assignment.workspace.add", target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id, "workspace_id": workspace_id}))
        return {"workspace_id": workspace_id}

    async def delete_policy_assignment_workspace(self, *, target, access_context, assignment_id, workspace_id):
        self.action_calls.append(("policy_assignment.workspace.delete", target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id, "workspace_id": workspace_id}))
        return {"ok": True}

    async def list_profile_credential_bindings(self, *, target, access_context, profile_id):
        self.action_calls.append(("permission_profile.bindings.list", target.server_id, access_context.selected_scope_ref, {"profile_id": profile_id}))
        return [{"external_server_id": "docs"}]

    async def upsert_profile_credential_binding(self, *, target, access_context, profile_id, server_id, slot_name=None, payload=None):
        action_name = "permission_profile.slot_binding.upsert" if slot_name else "permission_profile.binding.upsert"
        entry = {"profile_id": profile_id, "server_id": server_id, "slot_name": slot_name, **dict(payload or {})}
        self.action_calls.append((action_name, target.server_id, access_context.selected_scope_ref, entry))
        return {"external_server_id": server_id, "slot_name": slot_name, **dict(payload or {})}

    async def delete_profile_credential_binding(self, *, target, access_context, profile_id, server_id, slot_name=None):
        action_name = "permission_profile.slot_binding.delete" if slot_name else "permission_profile.binding.delete"
        self.action_calls.append((action_name, target.server_id, access_context.selected_scope_ref, {"profile_id": profile_id, "server_id": server_id, "slot_name": slot_name}))
        return {"ok": True}

    async def get_profile_slot_credential_status(self, *, target, access_context, profile_id, server_id, slot_name):
        self.action_calls.append(("permission_profile.slot_status.get", target.server_id, access_context.selected_scope_ref, {"profile_id": profile_id, "server_id": server_id, "slot_name": slot_name}))
        return {"status": "configured"}

    async def list_assignment_credential_bindings(self, *, target, access_context, assignment_id):
        self.action_calls.append(("policy_assignment.bindings.list", target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id}))
        return [{"external_server_id": "docs"}]

    async def upsert_assignment_credential_binding(self, *, target, access_context, assignment_id, server_id, slot_name=None, payload=None):
        action_name = "policy_assignment.slot_binding.upsert" if slot_name else "policy_assignment.binding.upsert"
        entry = {"assignment_id": assignment_id, "server_id": server_id, "slot_name": slot_name, **dict(payload or {})}
        self.action_calls.append((action_name, target.server_id, access_context.selected_scope_ref, entry))
        return {"external_server_id": server_id, "slot_name": slot_name, **dict(payload or {})}

    async def delete_assignment_credential_binding(self, *, target, access_context, assignment_id, server_id, slot_name=None):
        action_name = "policy_assignment.slot_binding.delete" if slot_name else "policy_assignment.binding.delete"
        self.action_calls.append((action_name, target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id, "server_id": server_id, "slot_name": slot_name}))
        return {"ok": True}

    async def get_assignment_slot_credential_status(self, *, target, access_context, assignment_id, server_id, slot_name):
        self.action_calls.append(("policy_assignment.slot_status.get", target.server_id, access_context.selected_scope_ref, {"assignment_id": assignment_id, "server_id": server_id, "slot_name": slot_name}))
        return {"status": "configured"}

    async def set_external_server_secret(self, *, target, access_context, server_id, secret):
        self.action_calls.append(("external_server.secret.set", target.server_id, access_context.selected_scope_ref, {"server_id": server_id, "secret": secret}))
        return {"secret_ref_id": "secret-1"}

    async def dry_run_governance_pack(self, *, target, access_context, payload):
        self.action_calls.append(("governance_pack.dry_run", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"report": {"owner_scope_type": access_context.selected_scope, **dict(payload)}}

    async def prepare_governance_pack_source(self, *, target, access_context, payload):
        self.action_calls.append(("governance_pack.source.prepare", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"candidate_id": "cand-1", **dict(payload)}

    async def check_governance_pack_updates(self, *, target, access_context, governance_pack_id):
        self.action_calls.append(("governance_pack.check_updates", target.server_id, access_context.selected_scope_ref, {"governance_pack_id": governance_pack_id}))
        return {"governance_pack_id": governance_pack_id, "has_update": True}

    async def import_governance_pack(self, *, target, access_context, payload):
        self.action_calls.append(("governance_pack.import", target.server_id, access_context.selected_scope_ref, dict(payload)))
        return {"governance_pack_id": 81, **dict(payload)}

    async def get_governance_pack_detail(self, *, target, access_context, governance_pack_id):
        self.action_calls.append(("governance_pack.detail.get", target.server_id, access_context.selected_scope_ref, {"governance_pack_id": governance_pack_id}))
        return {"id": governance_pack_id, "pack_id": "baseline"}


@pytest.mark.asyncio
async def test_control_plane_service_restores_per_server_context_without_touching_global_runtime_source(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [
            ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True),
            ConfiguredServerTarget(server_id="server-b", label="Server B", base_url="https://b.example/api"),
        ]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    context_store.save(
        UnifiedMCPContext(
            selected_source="local",
            selected_active_server_id="server-b",
            selected_scope="personal",
            selected_section="overview",
            per_server_state={
                "server-a": ServerAccessContext(
                    server_id="server-a",
                    selected_scope="team",
                    selected_scope_ref="21",
                    selected_section="inventory",
                ),
                "server-b": ServerAccessContext(
                    server_id="server-b",
                    selected_scope="org",
                    selected_scope_ref="11",
                    selected_section="overview",
                ),
            },
        )
    )

    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=server_service,
    )

    context = await orchestrator.select_server_target("server-a")

    assert context.selected_source == "server"
    assert context.selected_active_server_id == "server-a"
    assert context.selected_scope == "team"
    assert context.selected_scope_ref == "21"
    assert context.selected_section == "inventory"
    assert server_service.resolve_calls == [("server-a", "team", "21", "inventory")]

    restored = context_store.load()
    assert restored.selected_source == "server"
    assert restored.selected_active_server_id == "server-a"
    assert restored.selected_scope == "team"
    assert restored.selected_scope_ref == "21"
    assert restored.selected_section == "inventory"


@pytest.mark.asyncio
async def test_control_plane_service_routes_overview_and_inventory_by_selected_source(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=server_service,
    )

    await orchestrator.select_source("local")
    local_overview = await orchestrator.load_section("overview")

    await orchestrator.select_server_target("server-a")
    server_inventory = await orchestrator.load_section("inventory")

    assert local_overview == {"source": "local", "section": "overview"}
    assert server_inventory == {
        "source": "server",
        "section": "inventory",
        "server_id": "server-a",
        "scope": "personal",
        "scope_ref": None,
    }
    assert local_service.calls == ["overview"]
    assert server_service.inventory_calls == [("server-a", "personal", None)]


@pytest.mark.asyncio
async def test_control_plane_service_routes_slice2_sections_by_selected_source(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=server_service,
    )

    await orchestrator.select_source("local")
    local_external = await orchestrator.load_section("external_servers")

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    server_catalogs = await orchestrator.load_section("catalogs")
    server_external = await orchestrator.load_section("external_servers")

    assert local_external == {
        "source": "local",
        "section": "external_servers",
        "profiles": [{"profile_id": "local-a"}],
    }
    assert server_catalogs["section"] == "catalogs"
    assert server_catalogs["scope"] == "team"
    assert server_external["section"] == "external_servers"
    assert server_external["scope_ref"] == "21"
    assert local_service.calls == ["external_servers"]
    assert server_service.catalog_calls == [("server-a", "team", "21")]
    assert server_service.external_calls == [("server-a", "team", "21")]


@pytest.mark.asyncio
async def test_control_plane_service_routes_local_governance_section_and_actions(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=FakeServerUnifiedMCPService(),
    )

    await orchestrator.select_source("local")
    local_governance = await orchestrator.load_section("governance")
    action_names = [descriptor["name"] for descriptor in orchestrator.available_actions()]
    saved = await orchestrator.run_action(
        "governance_rule.save",
        {
            "rule_id": "rule-b",
            "capability_id": "mcp.inventory.observe.local",
            "decision": "deny",
        },
    )
    preview = await orchestrator.run_action(
        "governance_rule.preview",
        {"capability_id": "mcp.inventory.list.local"},
    )
    deleted = await orchestrator.run_action("governance_rule.delete", {"rule_id": "rule-b"})

    assert local_governance == {
        "source": "local",
        "section": "governance",
        "rules": [
            {
                "rule_id": "rule-a",
                "capability_id": "mcp.inventory.list.local",
                "decision": "allow",
            }
        ],
    }
    assert "governance_rule.save" in action_names
    assert "governance_rule.preview" in action_names
    assert "governance_rule.delete" in action_names
    assert saved["rule_id"] == "rule-b"
    assert preview["decision"] == "allow"
    assert deleted is True
    assert local_service.calls == ["governance"]
    assert local_service.action_calls == [
        (
            "governance_rule.save",
            {
                "rule_id": "rule-b",
                "capability_id": "mcp.inventory.observe.local",
                "decision": "deny",
            },
        ),
        ("governance_rule.preview", {"capability_id": "mcp.inventory.list.local"}),
        ("governance_rule.delete", {"rule_id": "rule-b"}),
    ]


@pytest.mark.asyncio
async def test_control_plane_service_routes_local_inventory_runtime_actions(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=FakeServerUnifiedMCPService(),
    )

    await orchestrator.select_source("local")
    await orchestrator.select_section("inventory")
    action_names = [descriptor["name"] for descriptor in orchestrator.available_actions()]
    tool_result = await orchestrator.run_action(
        "tool.execute",
        {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
    )
    resource_result = await orchestrator.run_action(
        "resource.read",
        {"resource_uri": "note://123"},
    )
    prompt_result = await orchestrator.run_action(
        "prompt.get",
        {"prompt_name": "summarize_conversation", "arguments": {"conversation_id": 4}},
    )

    assert "tool.execute" in action_names
    assert "resource.read" in action_names
    assert "prompt.get" in action_names
    assert tool_result["tool_name"] == "search_notes"
    assert resource_result["resource_uri"] == "note://123"
    assert prompt_result["prompt_name"] == "summarize_conversation"
    assert local_service.action_calls == [
        ("tool.execute", {"tool_name": "search_notes", "arguments": {"query": "roadmap"}}),
        ("resource.read", {"resource_uri": "note://123"}),
        ("prompt.get", {"prompt_name": "summarize_conversation", "arguments": {"conversation_id": 4}}),
    ]


@pytest.mark.asyncio
async def test_control_plane_service_dispatches_slice2_actions(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=server_service,
    )

    await orchestrator.select_source("local")
    await orchestrator.select_section("external_servers")
    local_result = await orchestrator.run_action(
        "profile.save",
        {"profile_id": "local-b", "command": "python", "args": ["-m", "demo.server"]},
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    await orchestrator.select_section("catalogs")
    server_result = await orchestrator.run_action(
        "catalog.create",
        {"name": "Team Catalog", "description": "Scoped"},
    )

    assert local_result["profile_id"] == "local-b"
    assert server_result["name"] == "Team Catalog"
    assert local_service.action_calls == [
        ("profile.save", {"profile_id": "local-b", "command": "python", "args": ["-m", "demo.server"]})
    ]
    assert server_service.action_calls == [
        ("catalog.create", "server-a", "21", {"name": "Team Catalog", "description": "Scoped"})
    ]


@pytest.mark.asyncio
async def test_control_plane_service_hides_server_mutation_actions_for_personal_scope(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=FakeServerUnifiedMCPService(),
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_section("catalogs")
    personal_actions = orchestrator.available_actions()

    await orchestrator.select_scope("team", "21")
    team_actions = orchestrator.available_actions()

    assert personal_actions == []
    assert any(action["name"] == "catalog.create" for action in team_actions)


@pytest.mark.asyncio
async def test_control_plane_service_routes_governance_section_and_action(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=FakeServerUnifiedMCPService(),
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    governance = await orchestrator.load_section("governance")
    update_result = await orchestrator.run_action(
        "approval_policy.update",
        {"approval_policy_id": 7, "name": "Updated Approval"},
    )

    assert governance["section"] == "governance"
    assert governance["permission_profiles"][0]["name"] == "Default"
    assert update_result["name"] == "Updated Approval"
    assert any(action["name"] == "approval_policy.update" for action in orchestrator.available_actions())


@pytest.mark.asyncio
async def test_control_plane_service_routes_advanced_section_and_action(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=FakeServerUnifiedMCPService(),
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    advanced = await orchestrator.load_section("advanced")
    create_result = await orchestrator.run_action(
        "path_scope_object.create",
        {"name": "Workspace Root", "path_scope_document": {"path_scope_mode": "workspace_root"}},
    )

    assert advanced["section"] == "advanced"
    assert advanced["tool_registry_summary"]["modules"][0]["module"] == "search"
    assert advanced["tool_registry_entries"][0]["tool_name"] == "docs.search"
    assert advanced["tool_registry_modules"][0]["module"] == "search"
    assert create_result["name"] == "Workspace Root"
    assert any(action["name"] == "path_scope_object.create" for action in orchestrator.available_actions())


@pytest.mark.asyncio
async def test_control_plane_service_routes_local_advanced_section_and_runtime_actions(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    local_service = FakeLocalMCPControlService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=local_service,
        server_service=FakeServerUnifiedMCPService(),
    )

    advanced = await orchestrator.load_section("advanced")
    action_names = [descriptor["name"] for descriptor in orchestrator.available_actions()]
    preview_result = await orchestrator.run_action(
        "runtime.access.preview",
        {"action_name": "tool.execute", "payload": {"tool_name": "search_notes", "arguments": {"query": "roadmap"}}},
    )
    activity_result = await orchestrator.run_action("runtime.activity.list", {"limit": 5})
    protocol_result = await orchestrator.run_action("runtime.protocol.inspect", {})
    health_result = await orchestrator.run_action("runtime.health.get", {})
    approvals_result = await orchestrator.run_action(
        "approval_requests.list",
        {"status": "pending", "resolved_action_id": "notes.list.local"},
    )
    approve_result = await orchestrator.run_action("approval_request.approve", {"request_id": "approval-a"})
    deny_result = await orchestrator.run_action("approval_request.deny", {"request_id": "approval-b"})
    delete_result = await orchestrator.run_action("approval_request.delete", {"request_id": "approval-c"})
    status_result = await orchestrator.run_action("runtime.status.get", {})
    request_result = await orchestrator.run_action("runtime.request", {"method": "tools/list", "params": {}})
    batch_result = await orchestrator.run_action(
        "runtime.batch",
        {"requests": [{"method": "tools/list"}, {"method": "prompts/list"}]},
    )

    assert advanced["section"] == "advanced"
    assert advanced["runtime_status"]["server_id"] == "local:tldw_chatbook"
    assert "runtime.access.preview" in action_names
    assert "runtime.activity.list" in action_names
    assert "runtime.protocol.inspect" in action_names
    assert "runtime.health.get" in action_names
    assert "approval_requests.list" in action_names
    assert "approval_request.approve" in action_names
    assert "approval_request.deny" in action_names
    assert "approval_request.delete" in action_names
    assert "runtime.status.get" in action_names
    assert "runtime.request" in action_names
    assert "runtime.batch" in action_names
    assert preview_result["decision"] == "deny"
    assert preview_result["resolved_action_id"] == "notes.list.local"
    assert activity_result["entries"][0]["action_name"] == "tool.execute"
    assert protocol_result["diagnostics"]["protocol_version"] == "2025-03-26"
    assert health_result["health"]["state"] == "ready"
    assert approvals_result[0]["request_id"] == "approval-a"
    assert approve_result["status"] == "approved"
    assert deny_result["status"] == "denied"
    assert delete_result is True
    assert status_result["status"]["server_id"] == "local:tldw_chatbook"
    assert request_result["result"]["method"] == "tools/list"
    assert batch_result["results"][1]["method"] == "prompts/list"
    assert local_service.calls[-1] == "advanced"
    assert local_service.action_calls == [
        (
            "runtime.access.preview",
            {
                "action_name": "tool.execute",
                "payload": {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
            },
        ),
        ("runtime.activity.list", {"limit": 5}),
        ("runtime.protocol.inspect", {}),
        ("runtime.health.get", {}),
        ("approval_requests.list", {"status": "pending", "resolved_action_id": "notes.list.local"}),
        ("approval_request.approve", {"request_id": "approval-a"}),
        ("approval_request.deny", {"request_id": "approval-b"}),
        ("approval_request.delete", {"request_id": "approval-c"}),
        ("runtime.status.get", {}),
        ("runtime.request", {"method": "tools/list", "params": {}}),
        (
            "runtime.batch",
            {
                "requests": [
                    {"method": "tools/list"},
                    {"method": "prompts/list"},
                ]
            },
        ),
    ]


@pytest.mark.asyncio
async def test_control_plane_service_exposes_remaining_advanced_actions_by_scope_and_routes_them(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=server_service,
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    await orchestrator.select_section("advanced")

    actions = {action["name"] for action in orchestrator.available_actions()}
    preview = await orchestrator.run_action(
        "capability_mapping.preview",
        {"mapping_id": "filesystem-write", "capability_name": "filesystem.write"},
    )
    workspace_set = await orchestrator.run_action(
        "workspace_set_object.create",
        {"name": "Research Set"},
    )
    members = await orchestrator.run_action(
        "workspace_set_object.members.list",
        {"workspace_set_object_id": 6},
    )
    shared_workspace = await orchestrator.run_action(
        "shared_workspace.create",
        {"workspace_id": "shared-ws", "display_name": "Shared Workspace", "absolute_root": "/srv/shared"},
    )

    assert "capability_mapping.preview" in actions
    assert "workspace_set_object.create" in actions
    assert "shared_workspace.create" in actions
    assert preview["normalized_mapping"]["owner_scope_type"] == "team"
    assert workspace_set["name"] == "Research Set"
    assert members[0]["workspace_id"] == "ws-1"
    assert shared_workspace["display_name"] == "Shared Workspace"

    await orchestrator.select_scope("personal")
    personal_actions = {action["name"] for action in orchestrator.available_actions()}
    assert "capability_mapping.preview" not in personal_actions
    assert "shared_workspace.create" not in personal_actions


@pytest.mark.asyncio
async def test_control_plane_service_exposes_governance_pack_advanced_actions_and_routes_them(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=server_service,
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_section("advanced")

    actions = {action["name"] for action in orchestrator.available_actions()}
    dry_run = await orchestrator.run_action(
        "governance_pack.dry_run",
        {"pack": {"manifest": {"pack_id": "baseline", "version": "1.0.0"}}},
    )
    prepared = await orchestrator.run_action(
        "governance_pack.source.prepare",
        {"source": {"kind": "git", "url": "git@example.com:trusted/repo.git", "ref": "main"}},
    )
    checked = await orchestrator.run_action(
        "governance_pack.check_updates",
        {"governance_pack_id": 81},
    )
    imported = await orchestrator.run_action(
        "governance_pack.import",
        {"pack": {"manifest": {"pack_id": "baseline", "version": "1.0.0"}}},
    )
    detail = await orchestrator.run_action(
        "governance_pack.detail.get",
        {"governance_pack_id": 81},
    )

    assert "governance_pack.dry_run" in actions
    assert "governance_pack.source.prepare" in actions
    assert "governance_pack.import" in actions
    assert "governance_pack.detail.get" in actions
    assert dry_run["report"]["owner_scope_type"] == "personal"
    assert prepared["candidate_id"] == "cand-1"
    assert checked["has_update"] is True
    assert imported["governance_pack_id"] == 81
    assert detail["pack_id"] == "baseline"


@pytest.mark.asyncio
async def test_control_plane_service_exposes_remaining_governance_and_external_admin_tail_actions_and_routes_them(tmp_path):
    from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    context_store = UnifiedMCPContextStore(tmp_path / "context.json")
    server_service = FakeServerUnifiedMCPService()
    orchestrator = UnifiedMCPControlPlaneService(
        target_store=target_store,
        context_store=context_store,
        local_service=FakeLocalMCPControlService(),
        server_service=server_service,
    )

    await orchestrator.select_server_target("server-a")
    await orchestrator.select_scope("team", "21")
    await orchestrator.select_section("governance")

    governance_actions = {action["name"] for action in orchestrator.available_actions()}
    assignment_workspaces = await orchestrator.run_action(
        "policy_assignment.workspaces.list",
        {"assignment_id": 2},
    )
    added_workspace = await orchestrator.run_action(
        "policy_assignment.workspace.add",
        {"assignment_id": 2, "workspace_id": "ws-1"},
    )
    profile_bindings = await orchestrator.run_action(
        "permission_profile.bindings.list",
        {"profile_id": 1},
    )
    profile_status = await orchestrator.run_action(
        "permission_profile.slot_status.get",
        {"profile_id": 1, "server_id": "docs", "slot_name": "token_readonly"},
    )
    assignment_binding = await orchestrator.run_action(
        "policy_assignment.binding.upsert",
        {"assignment_id": 2, "server_id": "docs", "binding_mode": "grant", "managed_secret_ref_id": "secret-1"},
    )
    assignment_status = await orchestrator.run_action(
        "policy_assignment.slot_status.get",
        {"assignment_id": 2, "server_id": "docs", "slot_name": "token_readonly"},
    )

    await orchestrator.select_section("external_servers")
    external_actions = {action["name"] for action in orchestrator.available_actions()}
    external_secret = await orchestrator.run_action(
        "external_server.secret.set",
        {"server_id": "docs", "secret": "replace-me"},
    )

    assert "policy_assignment.workspaces.list" in governance_actions
    assert "permission_profile.bindings.list" in governance_actions
    assert "permission_profile.slot_status.get" in governance_actions
    assert "policy_assignment.binding.upsert" in governance_actions
    assert "policy_assignment.slot_status.get" in governance_actions
    assert assignment_workspaces[0]["workspace_id"] == "ws-1"
    assert added_workspace["workspace_id"] == "ws-1"
    assert profile_bindings[0]["external_server_id"] == "docs"
    assert profile_status["status"] == "configured"
    assert assignment_binding["external_server_id"] == "docs"
    assert assignment_status["status"] == "configured"
    assert "external_server.secret.set" in external_actions
    assert external_secret["secret_ref_id"] == "secret-1"
