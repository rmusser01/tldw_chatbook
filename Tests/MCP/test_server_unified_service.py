from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.tldw_api.mcp_unified_schemas import (
    MCPAccessBootstrapPrincipal,
    MCPAccessBootstrapResponse,
    MCPPromptsResponse,
    MCPResourcesResponse,
    MCPStatusResponse,
    MCPToolCatalogsResponse,
    MCPToolsResponse,
)


class FakeMCPUnifiedClient:
    def __init__(
        self,
        *,
        status: dict | None = None,
        bootstrap: MCPAccessBootstrapResponse | None = None,
        tools: dict | None = None,
        resources: dict | None = None,
        prompts: dict | None = None,
        catalogs: dict | None = None,
        catalogs_forbidden: bool = False,
        governance_forbidden: bool = False,
    ) -> None:
        self.status = status or {"status": "ok"}
        self.bootstrap = bootstrap or MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=1, username="demo", role="member", is_admin=False),
            manageable_team_ids=[],
            manageable_org_ids=[],
            can_use_system_admin_scope=False,
        )
        self.tools = tools or {"tools": [{"name": "search_docs", "description": "Search docs"}]}
        self.resources = resources or {"resources": [{"uri": "mcp://docs", "name": "Docs"}]}
        self.prompts = prompts or {"prompts": [{"name": "summarize_docs", "description": "Summarize docs"}]}
        self.catalogs = catalogs or {"catalogs": [{"catalog_id": "cat-1", "name": "Default"}]}
        self.scoped_catalogs = [{"id": 9, "name": "Scoped Catalog"}]
        self.external_servers = [{"id": "docs", "name": "Docs", "transport": "http"}]
        self.permission_profiles = [{"id": 1, "name": "Default", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.policy_assignments = [
            {
                "id": 2,
                "target_type": "persona",
                "target_id": "persona-a",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "has_override": True,
                "override_id": 9,
            }
        ]
        self.approval_policies = [{"id": 7, "name": "Default Approval", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.acp_profiles = [{"id": 8, "name": "Workspace ACP", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.effective_policy = {"enabled": True, "approval_mode": "ask_every_time", "selected_assignment_id": 2}
        self.external_access = {"servers": [{"server_id": "docs", "server_name": "Docs", "runtime_executable": True}]}
        self.tool_registry_summary = {
            "entries": [{"tool_name": "docs.search", "display_name": "Docs Search", "module": "search"}],
            "modules": [{"module": "search", "display_name": "Search", "tool_count": 1}],
        }
        self.tool_registry_entries = [{"tool_name": "docs.search", "display_name": "Docs Search", "module": "search"}]
        self.tool_registry_modules = [{"module": "search", "display_name": "Search", "tool_count": 1}]
        self.capability_mappings = [{"id": 3, "mapping_id": "filesystem-write", "capability_name": "filesystem.write"}]
        self.governance_packs = [{"id": 81, "pack_id": "baseline", "name": "Baseline"}]
        self.governance_pack_trust_policy = {"mode": "allowlist", "allowed_sources": ["git@example.com:trusted/repo.git"]}
        self.path_scope_objects = [{"id": 5, "name": "Workspace Root", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.workspace_set_objects = [{"id": 6, "name": "Research Set", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.shared_workspaces = [{"id": 7, "workspace_id": "ws-1", "display_name": "Workspace One", "owner_scope_type": "team", "owner_scope_id": 21}]
        self.policy_assignment_workspaces = [{"workspace_id": "ws-1"}]
        self.profile_credential_bindings = [{"external_server_id": "docs"}]
        self.assignment_credential_bindings = [{"external_server_id": "docs", "binding_mode": "grant"}]
        self.credential_slot_status = {"status": "configured", "has_secret": True}
        self.external_server_secret = {"secret_ref_id": "secret-1", "updated": True}
        self.governance_pack_detail = {"id": 81, "pack_id": "baseline", "owner_scope_type": "team", "owner_scope_id": 21}
        self.governance_pack_upgrade_history = [{"from_version": "1.0.0", "to_version": "1.1.0"}]
        self.governance_audit_findings = {"items": [{"finding_type": "warning", "object_kind": "path_scope", "object_id": "5", "message": "Needs review"}], "total": 1}
        self.catalogs_forbidden = catalogs_forbidden
        self.governance_forbidden = governance_forbidden
        self.calls: list[tuple[str, str | None, str | None]] = []

    async def get_status(self) -> MCPStatusResponse:
        self.calls.append(("get_status", None, None))
        return MCPStatusResponse.from_payload(self.status)

    async def bootstrap_access_context(self) -> MCPAccessBootstrapResponse:
        self.calls.append(("bootstrap_access_context", None, None))
        return self.bootstrap

    async def list_tools(self, *, access_context=None, **_kwargs) -> MCPToolsResponse:
        scope_kind = getattr(access_context, "scope_kind", None)
        scope_ref = getattr(access_context, "scope_ref", None)
        self.calls.append(("list_tools", scope_kind, scope_ref))
        return MCPToolsResponse.from_payload(self.tools)

    async def list_resources(self, *, access_context=None, **_kwargs) -> MCPResourcesResponse:
        scope_kind = getattr(access_context, "scope_kind", None)
        scope_ref = getattr(access_context, "scope_ref", None)
        self.calls.append(("list_resources", scope_kind, scope_ref))
        return MCPResourcesResponse.from_payload(self.resources)

    async def list_prompts(self, *, access_context=None, **_kwargs) -> MCPPromptsResponse:
        scope_kind = getattr(access_context, "scope_kind", None)
        scope_ref = getattr(access_context, "scope_ref", None)
        self.calls.append(("list_prompts", scope_kind, scope_ref))
        return MCPPromptsResponse.from_payload(self.prompts)

    async def list_visible_tool_catalogs(self, *, access_context=None, **_kwargs) -> MCPToolCatalogsResponse:
        scope_kind = getattr(access_context, "scope_kind", None)
        scope_ref = getattr(access_context, "scope_ref", None)
        self.calls.append(("list_visible_tool_catalogs", scope_kind, scope_ref))
        if self.catalogs_forbidden:
            raise RuntimeError("catalog access denied")
        return MCPToolCatalogsResponse.from_payload(self.catalogs)

    async def list_permission_profiles(self, *, access_context=None, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = getattr(access_context, "scope_kind", None)
        scope_ref = getattr(access_context, "scope_ref", None)
        if owner_scope_type is not None:
            scope_kind = str(owner_scope_type)
            scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_permission_profiles", scope_kind, scope_ref))
        if self.governance_forbidden:
            raise RuntimeError("governance access denied")
        return {"items": list(self.permission_profiles)}

    async def list_policy_assignments(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_policy_assignments", scope_kind, scope_ref))
        return {"items": list(self.policy_assignments)}

    async def list_approval_policies(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_approval_policies", scope_kind, scope_ref))
        return {"items": list(self.approval_policies)}

    async def update_approval_policy(self, *, approval_policy_id, request, **_kwargs) -> dict:
        payload = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
        self.calls.append(("update_approval_policy", str(approval_policy_id), payload.get("name")))
        return {"id": approval_policy_id, **payload}

    async def get_effective_policy(self, *, org_id=None, team_id=None, **_kwargs) -> dict:
        scope_kind = "team" if team_id is not None else ("org" if org_id is not None else "personal")
        scope_ref = str(team_id if team_id is not None else org_id) if (team_id is not None or org_id is not None) else None
        self.calls.append(("get_effective_policy", scope_kind, scope_ref))
        return dict(self.effective_policy)

    async def list_acp_profiles(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_acp_profiles", scope_kind, scope_ref))
        return {"items": list(self.acp_profiles)}

    async def get_assignment_external_access(self, *, assignment_id, **_kwargs) -> dict:
        self.calls.append(("get_assignment_external_access", str(assignment_id), None))
        return dict(self.external_access)

    async def list_policy_assignment_workspaces(self, *, assignment_id, **_kwargs) -> dict:
        self.calls.append(("list_policy_assignment_workspaces", str(assignment_id), None))
        return {"items": list(self.policy_assignment_workspaces)}

    async def add_policy_assignment_workspace(self, *, assignment_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("add_policy_assignment_workspace", str(assignment_id), str(payload.get("workspace_id"))))
        return {"workspace_id": payload.get("workspace_id")}

    async def delete_policy_assignment_workspace(self, *, assignment_id, workspace_id, **_kwargs) -> dict:
        self.calls.append(("delete_policy_assignment_workspace", str(assignment_id), str(workspace_id)))
        return {"ok": True}

    async def list_profile_credential_bindings(self, *, profile_id, **_kwargs) -> dict:
        self.calls.append(("list_profile_credential_bindings", str(profile_id), None))
        return {"items": list(self.profile_credential_bindings)}

    async def upsert_profile_credential_binding(self, *, profile_id, server_id, slot_name=None, payload=None, **_kwargs) -> dict:
        payload = dict(payload or {})
        self.calls.append(("upsert_profile_credential_binding", str(profile_id), f"{server_id}:{slot_name or ''}"))
        return {"profile_id": profile_id, "external_server_id": server_id, "slot_name": slot_name, **payload}

    async def upsert_profile_slot_credential_binding(self, *, profile_id, server_id, slot_name, payload=None, **_kwargs) -> dict:
        return await self.upsert_profile_credential_binding(
            profile_id=profile_id,
            server_id=server_id,
            slot_name=slot_name,
            payload=payload,
        )

    async def delete_profile_credential_binding(self, *, profile_id, server_id, slot_name=None, **_kwargs) -> dict:
        self.calls.append(("delete_profile_credential_binding", str(profile_id), f"{server_id}:{slot_name or ''}"))
        return {"ok": True}

    async def delete_profile_slot_credential_binding(self, *, profile_id, server_id, slot_name, **_kwargs) -> dict:
        return await self.delete_profile_credential_binding(
            profile_id=profile_id,
            server_id=server_id,
            slot_name=slot_name,
        )

    async def get_profile_slot_credential_status(self, *, profile_id, server_id, slot_name, **_kwargs) -> dict:
        self.calls.append(("get_profile_slot_credential_status", str(profile_id), f"{server_id}:{slot_name}"))
        return dict(self.credential_slot_status)

    async def list_assignment_credential_bindings(self, *, assignment_id, **_kwargs) -> dict:
        self.calls.append(("list_assignment_credential_bindings", str(assignment_id), None))
        return {"items": list(self.assignment_credential_bindings)}

    async def upsert_assignment_credential_binding(self, *, assignment_id, server_id, slot_name=None, payload=None, **_kwargs) -> dict:
        payload = dict(payload or {})
        self.calls.append(("upsert_assignment_credential_binding", str(assignment_id), f"{server_id}:{slot_name or ''}"))
        return {"assignment_id": assignment_id, "external_server_id": server_id, "slot_name": slot_name, **payload}

    async def upsert_assignment_slot_credential_binding(self, *, assignment_id, server_id, slot_name, payload=None, **_kwargs) -> dict:
        return await self.upsert_assignment_credential_binding(
            assignment_id=assignment_id,
            server_id=server_id,
            slot_name=slot_name,
            payload=payload,
        )

    async def delete_assignment_credential_binding(self, *, assignment_id, server_id, slot_name=None, **_kwargs) -> dict:
        self.calls.append(("delete_assignment_credential_binding", str(assignment_id), f"{server_id}:{slot_name or ''}"))
        return {"ok": True}

    async def delete_assignment_slot_credential_binding(self, *, assignment_id, server_id, slot_name, **_kwargs) -> dict:
        return await self.delete_assignment_credential_binding(
            assignment_id=assignment_id,
            server_id=server_id,
            slot_name=slot_name,
        )

    async def get_assignment_slot_credential_status(self, *, assignment_id, server_id, slot_name, **_kwargs) -> dict:
        self.calls.append(("get_assignment_slot_credential_status", str(assignment_id), f"{server_id}:{slot_name}"))
        return dict(self.credential_slot_status)

    async def get_tool_registry_summary(self, **_kwargs) -> dict:
        self.calls.append(("get_tool_registry_summary", None, None))
        return dict(self.tool_registry_summary)

    async def list_tool_registry_entries(self, **_kwargs) -> list[dict]:
        self.calls.append(("list_tool_registry_entries", None, None))
        return list(self.tool_registry_entries)

    async def list_tool_registry_modules(self, **_kwargs) -> list[dict]:
        self.calls.append(("list_tool_registry_modules", None, None))
        return list(self.tool_registry_modules)

    async def list_capability_mappings(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_capability_mappings", scope_kind, scope_ref))
        return {"items": list(self.capability_mappings)}

    async def list_governance_packs(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_governance_packs", scope_kind, scope_ref))
        return {"items": list(self.governance_packs)}

    async def get_governance_pack_trust_policy(self, **_kwargs) -> dict:
        self.calls.append(("get_governance_pack_trust_policy", None, None))
        return dict(self.governance_pack_trust_policy)

    async def update_governance_pack_trust_policy(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("update_governance_pack_trust_policy", str(payload.get("mode")), None))
        return dict(payload)

    async def dry_run_governance_pack(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("dry_run_governance_pack", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"report": dict(payload)}

    async def prepare_governance_pack_source(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("prepare_governance_pack_source", str((payload.get("source") or {}).get("kind")), None))
        return {"candidate_id": "cand-1", **dict(payload)}

    async def dry_run_governance_pack_source(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("dry_run_governance_pack_source", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"report": dict(payload)}

    async def check_governance_pack_updates(self, *, governance_pack_id, **_kwargs) -> dict:
        self.calls.append(("check_governance_pack_updates", str(governance_pack_id), None))
        return {"governance_pack_id": governance_pack_id, "has_update": True}

    async def prepare_governance_pack_upgrade_candidate(self, *, governance_pack_id, **_kwargs) -> dict:
        self.calls.append(("prepare_governance_pack_upgrade_candidate", str(governance_pack_id), None))
        return {"candidate_id": "cand-2", "governance_pack_id": governance_pack_id}

    async def dry_run_governance_pack_upgrade(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("dry_run_governance_pack_upgrade", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"plan": dict(payload)}

    async def dry_run_governance_pack_source_upgrade(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("dry_run_governance_pack_source_upgrade", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"plan": dict(payload)}

    async def import_governance_pack(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("import_governance_pack", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"governance_pack_id": 81, **dict(payload)}

    async def execute_governance_pack_source_upgrade(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("execute_governance_pack_source_upgrade", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"ok": True, **dict(payload)}

    async def import_governance_pack_source(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("import_governance_pack_source", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"governance_pack_id": 82, **dict(payload)}

    async def execute_governance_pack_upgrade(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("execute_governance_pack_upgrade", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"ok": True, **dict(payload)}

    async def get_governance_pack_detail(self, *, governance_pack_id, **_kwargs) -> dict:
        self.calls.append(("get_governance_pack_detail", str(governance_pack_id), None))
        return dict(self.governance_pack_detail)

    async def list_governance_pack_upgrade_history(self, *, governance_pack_id, **_kwargs) -> list[dict]:
        self.calls.append(("list_governance_pack_upgrade_history", str(governance_pack_id), None))
        return list(self.governance_pack_upgrade_history)

    async def list_path_scope_objects(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_path_scope_objects", scope_kind, scope_ref))
        return {"items": list(self.path_scope_objects)}

    async def create_path_scope_object(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("create_path_scope_object", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"id": 11, **dict(payload)}

    async def update_path_scope_object(self, *, path_scope_object_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("update_path_scope_object", str(path_scope_object_id), str(payload.get("owner_scope_type")) if payload.get("owner_scope_type") is not None else None))
        return {"id": path_scope_object_id, **dict(payload)}

    async def delete_path_scope_object(self, *, path_scope_object_id, **_kwargs) -> dict:
        self.calls.append(("delete_path_scope_object", str(path_scope_object_id), None))
        return {"ok": True}

    async def list_workspace_set_objects(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_workspace_set_objects", scope_kind, scope_ref))
        return {"items": list(self.workspace_set_objects)}

    async def create_workspace_set_object(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("create_workspace_set_object", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"id": 12, **dict(payload)}

    async def update_workspace_set_object(self, *, workspace_set_object_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("update_workspace_set_object", str(workspace_set_object_id), None))
        return {"id": workspace_set_object_id, **dict(payload)}

    async def delete_workspace_set_object(self, *, workspace_set_object_id, **_kwargs) -> dict:
        self.calls.append(("delete_workspace_set_object", str(workspace_set_object_id), None))
        return {"ok": True}

    async def list_workspace_set_members(self, *, workspace_set_object_id, **_kwargs) -> dict:
        self.calls.append(("list_workspace_set_members", str(workspace_set_object_id), None))
        return {"items": [{"workspace_id": "ws-1"}]}

    async def add_workspace_set_member(self, *, workspace_set_object_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("add_workspace_set_member", str(workspace_set_object_id), str(payload.get("workspace_id"))))
        return {"workspace_id": payload.get("workspace_id")}

    async def delete_workspace_set_member(self, *, workspace_set_object_id, workspace_id, **_kwargs) -> dict:
        self.calls.append(("delete_workspace_set_member", str(workspace_set_object_id), str(workspace_id)))
        return {"ok": True}

    async def list_shared_workspaces(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_shared_workspaces", scope_kind, scope_ref))
        return {"items": list(self.shared_workspaces)}

    async def create_shared_workspace(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("create_shared_workspace", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"id": 13, **dict(payload)}

    async def update_shared_workspace(self, *, shared_workspace_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("update_shared_workspace", str(shared_workspace_id), None))
        return {"id": shared_workspace_id, **dict(payload)}

    async def delete_shared_workspace(self, *, shared_workspace_id, **_kwargs) -> dict:
        self.calls.append(("delete_shared_workspace", str(shared_workspace_id), None))
        return {"ok": True}

    async def preview_capability_mapping(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("preview_capability_mapping", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"normalized_mapping": dict(payload)}

    async def create_capability_mapping(self, payload: dict, **_kwargs) -> dict:
        self.calls.append(("create_capability_mapping", str(payload.get("owner_scope_type")), str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None))
        return {"id": 14, **dict(payload)}

    async def update_capability_mapping(self, *, capability_adapter_mapping_id, payload: dict, **_kwargs) -> dict:
        self.calls.append(("update_capability_mapping", str(capability_adapter_mapping_id), None))
        return {"id": capability_adapter_mapping_id, **dict(payload)}

    async def delete_capability_mapping(self, *, capability_adapter_mapping_id, **_kwargs) -> dict:
        self.calls.append(("delete_capability_mapping", str(capability_adapter_mapping_id), None))
        return {"ok": True}

    async def list_governance_audit_findings(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> dict:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_governance_audit_findings", scope_kind, scope_ref))
        return dict(self.governance_audit_findings)

    async def list_external_servers(self, *, owner_scope_type=None, owner_scope_id=None, **_kwargs) -> list[dict]:
        scope_kind = str(owner_scope_type) if owner_scope_type is not None else None
        scope_ref = str(owner_scope_id) if owner_scope_id is not None else None
        self.calls.append(("list_external_servers", scope_kind, scope_ref))
        return list(self.external_servers)

    async def create_external_server(self, request) -> dict:
        payload = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
        scope_kind = str(payload.get("owner_scope_type")) if payload.get("owner_scope_type") is not None else None
        scope_ref = str(payload.get("owner_scope_id")) if payload.get("owner_scope_id") is not None else None
        self.calls.append(("create_external_server", scope_kind, scope_ref))
        return {
            "id": payload.get("server_id") or "created-docs",
            "name": payload.get("name") or "Docs",
            "transport": payload.get("transport") or "http",
            "owner_scope_type": payload.get("owner_scope_type"),
            "owner_scope_id": payload.get("owner_scope_id"),
        }

    async def set_external_server_secret(self, *, server_id, request, **_kwargs) -> dict:
        payload = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
        self.calls.append(("set_external_server_secret", str(server_id), payload.get("secret")))
        return dict(self.external_server_secret)

    async def list_scoped_tool_catalogs(self, *, scope_kind, scope_ref, **_kwargs) -> list[dict]:
        self.calls.append(("list_scoped_tool_catalogs", scope_kind, scope_ref))
        return list(self.scoped_catalogs)


@pytest.mark.asyncio
async def test_server_unified_service_resolves_access_context_and_section_capabilities():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21, 23],
            manageable_org_ids=[11],
            can_use_system_admin_scope=True,
        ),
        governance_forbidden=True,
    )
    service = ServerUnifiedMCPService(client=client)

    context = await service.resolve_access_context(
        target=ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api"),
        selected_scope="team",
        selected_scope_ref="23",
        selected_section="inventory",
    )

    assert context.server_id == "srv"
    assert context.selected_scope == "team"
    assert context.selected_scope_ref == "23"
    assert context.selected_section == "inventory"
    assert context.can_use_personal_scope is True
    assert context.manageable_team_ids == (21, 23)
    assert context.manageable_org_ids == (11,)
    assert context.can_use_system_admin_scope is True
    assert context.section_capabilities.overview is True
    assert context.section_capabilities.inventory is True
    assert context.section_capabilities.catalogs is True
    assert context.section_capabilities.governance is False
    assert context.endpoint_capabilities["tools.list"] is True
    assert context.endpoint_capabilities["catalogs.list"] is True
    assert context.endpoint_capabilities["governance.list"] is False
    assert context.target_status.last_known_reachability == "reachable"
    assert context.target_status.last_known_auth_state == "authenticated"


@pytest.mark.asyncio
async def test_server_unified_service_partitions_inventory_cache_by_server_and_scope():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    personal_context = await service.resolve_access_context(
        target=target,
        selected_scope="personal",
        selected_section="inventory",
    )
    first_inventory = await service.get_inventory(target=target, access_context=personal_context)
    second_inventory = await service.get_inventory(target=target, access_context=personal_context)

    team_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="inventory",
    )
    team_inventory = await service.get_inventory(target=target, access_context=team_context)

    assert first_inventory == second_inventory
    assert team_context.selected_scope == "team"
    assert team_context.selected_scope_ref == "21"
    assert first_inventory["tools"][0]["name"] == "search_docs"
    assert team_inventory["tools"][0]["name"] == "search_docs"
    assert client.calls.count(("list_tools", "personal", None)) == 2
    assert client.calls.count(("list_resources", "personal", None)) == 2
    assert client.calls.count(("list_prompts", "personal", None)) == 2
    assert client.calls.count(("list_tools", "team", "21")) == 2
    assert client.calls.count(("list_resources", "team", "21")) == 2
    assert client.calls.count(("list_prompts", "team", "21")) == 2
    assert service.cache_for(
        section="inventory",
        server_id="srv",
        selected_scope="personal",
        selected_scope_ref=None,
    ) == first_inventory
    assert service.cache_for(
        section="inventory",
        server_id="srv",
        selected_scope="team",
        selected_scope_ref="21",
    ) == team_inventory


@pytest.mark.asyncio
async def test_server_unified_service_caches_catalogs_but_keeps_external_servers_live():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    catalog_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="catalogs",
    )
    first_catalogs = await service.get_catalogs(target=target, access_context=catalog_context)
    second_catalogs = await service.get_catalogs(target=target, access_context=catalog_context)

    external_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="external_servers",
    )
    first_external = await service.get_external_servers(target=target, access_context=external_context)
    second_external = await service.get_external_servers(target=target, access_context=external_context)

    assert first_catalogs == second_catalogs
    assert first_catalogs["catalogs"][0]["name"] == "Scoped Catalog"
    assert first_external["external_servers"][0]["name"] == "Docs"
    assert client.calls.count(("list_scoped_tool_catalogs", "team", "21")) == 1
    assert client.calls.count(("list_external_servers", "team", "21")) == 2
    assert service.cache_for(
        section="catalogs",
        server_id="srv",
        selected_scope="team",
        selected_scope_ref="21",
    ) == first_catalogs
    assert service.cache_for(
        section="external_servers",
        server_id="srv",
        selected_scope="team",
        selected_scope_ref="21",
    ) is None


@pytest.mark.asyncio
async def test_server_unified_service_creates_external_server_with_revalidated_scope():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    access_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="external_servers",
    )
    created = await service.create_external_server(
        target=target,
        access_context=access_context,
        payload={
            "server_id": "docs",
            "name": "Docs",
            "transport": "http",
            "config": {"url": "https://docs.example/mcp"},
        },
    )

    assert created["name"] == "Docs"
    assert created["owner_scope_type"] == "team"
    assert created["owner_scope_id"] == 21
    assert ("create_external_server", "team", "21") in client.calls


@pytest.mark.asyncio
async def test_server_unified_service_keeps_governance_reads_live_and_uncached():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    access_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="governance",
    )
    first_payload = await service.get_governance(target=target, access_context=access_context)
    second_payload = await service.get_governance(target=target, access_context=access_context)

    assert first_payload["permission_profiles"][0]["name"] == "Default"
    assert first_payload["policy_assignments"][0]["target_id"] == "persona-a"
    assert first_payload["approval_policies"][0]["name"] == "Default Approval"
    assert first_payload["acp_profiles"][0]["name"] == "Workspace ACP"
    assert first_payload["effective_policy"]["enabled"] is True
    assert first_payload["cache_mode"] == "live"
    assert second_payload == first_payload
    assert client.calls.count(("list_permission_profiles", "team", "21")) == 3
    assert client.calls.count(("list_policy_assignments", "team", "21")) == 2
    assert client.calls.count(("list_approval_policies", "team", "21")) == 2
    assert client.calls.count(("list_acp_profiles", "team", "21")) == 2
    assert client.calls.count(("get_effective_policy", "team", "21")) == 2
    assert service.cache_for(
        section="governance",
        server_id="srv",
        selected_scope="team",
        selected_scope_ref="21",
    ) is None


@pytest.mark.asyncio
async def test_server_unified_service_updates_approval_policy_and_previews_external_access():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    access_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="governance",
    )
    updated = await service.update_approval_policy(
        target=target,
        access_context=access_context,
        approval_policy_id=7,
        payload={"name": "Updated Approval"},
    )
    external_access = await service.get_assignment_external_access(
        target=target,
        access_context=access_context,
        assignment_id=2,
    )

    assert updated["name"] == "Updated Approval"
    assert external_access["servers"][0]["server_id"] == "docs"
    assert ("update_approval_policy", "7", "Updated Approval") in client.calls
    assert ("get_assignment_external_access", "2", None) in client.calls


@pytest.mark.asyncio
async def test_server_unified_service_caches_advanced_browse_and_mutates_live_admin_paths():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    advanced_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="advanced",
    )
    first_payload = await service.get_advanced(target=target, access_context=advanced_context)
    second_payload = await service.get_advanced(target=target, access_context=advanced_context)

    admin_context = await service.resolve_access_context(
        target=target,
        selected_scope="system_admin",
        selected_section="advanced",
    )
    updated_policy = await service.update_governance_pack_trust_policy(
        target=target,
        access_context=admin_context,
        payload={"mode": "allowlist", "allowed_sources": ["git@example.com:trusted/repo.git"]},
    )
    created_path_scope = await service.create_path_scope_object(
        target=target,
        access_context=advanced_context,
        payload={"name": "Workspace Root", "path_scope_document": {"path_scope_mode": "workspace_root"}},
    )

    assert first_payload["tool_registry_summary"]["modules"][0]["module"] == "search"
    assert first_payload["tool_registry_entries"][0]["tool_name"] == "docs.search"
    assert first_payload["tool_registry_modules"][0]["module"] == "search"
    assert first_payload["governance_packs"][0]["name"] == "Baseline"
    assert first_payload["cache_mode"] == "stale_allowed"
    assert second_payload == first_payload
    assert updated_policy["mode"] == "allowlist"
    assert created_path_scope["owner_scope_type"] == "team"
    assert created_path_scope["owner_scope_id"] == 21
    assert client.calls.count(("get_tool_registry_summary", None, None)) >= 3
    assert service.cache_for(
        section="advanced",
        server_id="srv",
        selected_scope="team",
        selected_scope_ref="21",
    ) is None


@pytest.mark.asyncio
async def test_server_unified_service_routes_remaining_advanced_admin_mutations_with_selected_scope_authority():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")
    access_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="advanced",
    )

    preview = await service.preview_capability_mapping(
        target=target,
        access_context=access_context,
        payload={"mapping_id": "filesystem-write", "capability_name": "filesystem.write", "owner_scope_type": "global"},
    )
    created_mapping = await service.create_capability_mapping(
        target=target,
        access_context=access_context,
        payload={"mapping_id": "filesystem-write", "capability_name": "filesystem.write"},
    )
    updated_mapping = await service.update_capability_mapping(
        target=target,
        access_context=access_context,
        capability_adapter_mapping_id=3,
        payload={"title": "Filesystem Write"},
    )
    deleted_mapping = await service.delete_capability_mapping(
        target=target,
        access_context=access_context,
        capability_adapter_mapping_id=3,
    )
    updated_path_scope = await service.update_path_scope_object(
        target=target,
        access_context=access_context,
        path_scope_object_id=5,
        payload={"name": "Workspace Root Updated"},
    )
    deleted_path_scope = await service.delete_path_scope_object(
        target=target,
        access_context=access_context,
        path_scope_object_id=5,
    )
    created_workspace_set = await service.create_workspace_set_object(
        target=target,
        access_context=access_context,
        payload={"name": "Research Set"},
    )
    members = await service.list_workspace_set_members(
        target=target,
        access_context=access_context,
        workspace_set_object_id=6,
    )
    added_member = await service.add_workspace_set_member(
        target=target,
        access_context=access_context,
        workspace_set_object_id=6,
        payload={"workspace_id": "ws-1"},
    )
    deleted_member = await service.delete_workspace_set_member(
        target=target,
        access_context=access_context,
        workspace_set_object_id=6,
        workspace_id="ws-1",
    )
    updated_workspace_set = await service.update_workspace_set_object(
        target=target,
        access_context=access_context,
        workspace_set_object_id=6,
        payload={"description": "Updated"},
    )
    deleted_workspace_set = await service.delete_workspace_set_object(
        target=target,
        access_context=access_context,
        workspace_set_object_id=6,
    )
    created_shared_workspace = await service.create_shared_workspace(
        target=target,
        access_context=access_context,
        payload={"workspace_id": "shared-ws", "display_name": "Shared Workspace", "absolute_root": "/srv/shared"},
    )
    updated_shared_workspace = await service.update_shared_workspace(
        target=target,
        access_context=access_context,
        shared_workspace_id=7,
        payload={"display_name": "Shared Workspace Updated"},
    )
    deleted_shared_workspace = await service.delete_shared_workspace(
        target=target,
        access_context=access_context,
        shared_workspace_id=7,
    )

    assert preview["normalized_mapping"]["owner_scope_type"] == "team"
    assert preview["normalized_mapping"]["owner_scope_id"] == 21
    assert created_mapping["owner_scope_type"] == "team"
    assert created_mapping["owner_scope_id"] == 21
    assert updated_mapping["title"] == "Filesystem Write"
    assert deleted_mapping["ok"] is True
    assert updated_path_scope["name"] == "Workspace Root Updated"
    assert deleted_path_scope["ok"] is True
    assert created_workspace_set["owner_scope_type"] == "team"
    assert created_workspace_set["owner_scope_id"] == 21
    assert members[0]["workspace_id"] == "ws-1"
    assert added_member["workspace_id"] == "ws-1"
    assert deleted_member["ok"] is True
    assert updated_workspace_set["description"] == "Updated"
    assert deleted_workspace_set["ok"] is True
    assert created_shared_workspace["owner_scope_type"] == "team"
    assert created_shared_workspace["owner_scope_id"] == 21
    assert updated_shared_workspace["display_name"] == "Shared Workspace Updated"
    assert deleted_shared_workspace["ok"] is True


@pytest.mark.asyncio
async def test_server_unified_service_routes_governance_pack_flows_and_personal_scope_uses_principal_user_id():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    personal_context = await service.resolve_access_context(
        target=target,
        selected_scope="personal",
        selected_section="advanced",
    )
    team_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="advanced",
    )

    dry_run = await service.dry_run_governance_pack(
        target=target,
        access_context=personal_context,
        payload={"pack": {"manifest": {"pack_id": "baseline", "version": "1.0.0"}}},
    )
    prepared_source = await service.prepare_governance_pack_source(
        target=target,
        access_context=team_context,
        payload={"source": {"kind": "git", "url": "git@example.com:trusted/repo.git", "ref": "main"}},
    )
    source_dry_run = await service.dry_run_governance_pack_source(
        target=target,
        access_context=team_context,
        payload={"candidate_id": "cand-1"},
    )
    check_updates = await service.check_governance_pack_updates(
        target=target,
        access_context=team_context,
        governance_pack_id=81,
    )
    upgrade_candidate = await service.prepare_governance_pack_upgrade_candidate(
        target=target,
        access_context=team_context,
        governance_pack_id=81,
    )
    dry_run_upgrade = await service.dry_run_governance_pack_upgrade(
        target=target,
        access_context=team_context,
        payload={
            "source_governance_pack_id": 81,
            "pack": {"manifest": {"pack_id": "baseline", "version": "1.1.0"}},
            "planner_inputs_fingerprint": "planner-1",
            "adapter_state_fingerprint": "adapter-1",
        },
    )
    source_dry_run_upgrade = await service.dry_run_governance_pack_source_upgrade(
        target=target,
        access_context=team_context,
        payload={
            "candidate_id": "cand-1",
            "source_governance_pack_id": 81,
            "planner_inputs_fingerprint": "planner-1",
            "adapter_state_fingerprint": "adapter-1",
        },
    )
    imported = await service.import_governance_pack(
        target=target,
        access_context=team_context,
        payload={"pack": {"manifest": {"pack_id": "baseline", "version": "1.0.0"}}},
    )
    source_imported = await service.import_governance_pack_source(
        target=target,
        access_context=team_context,
        payload={"candidate_id": "cand-1"},
    )
    source_upgrade_executed = await service.execute_governance_pack_source_upgrade(
        target=target,
        access_context=team_context,
        payload={
            "candidate_id": "cand-1",
            "source_governance_pack_id": 81,
            "planner_inputs_fingerprint": "planner-1",
            "adapter_state_fingerprint": "adapter-1",
        },
    )
    upgrade_executed = await service.execute_governance_pack_upgrade(
        target=target,
        access_context=team_context,
        payload={
            "source_governance_pack_id": 81,
            "pack": {"manifest": {"pack_id": "baseline", "version": "1.1.0"}},
            "planner_inputs_fingerprint": "planner-1",
            "adapter_state_fingerprint": "adapter-1",
        },
    )
    detail = await service.get_governance_pack_detail(
        target=target,
        access_context=team_context,
        governance_pack_id=81,
    )
    history = await service.list_governance_pack_upgrade_history(
        target=target,
        access_context=team_context,
        governance_pack_id=81,
    )

    assert dry_run["report"]["owner_scope_type"] == "user"
    assert dry_run["report"]["owner_scope_id"] == 7
    assert prepared_source["candidate_id"] == "cand-1"
    assert source_dry_run["report"]["owner_scope_type"] == "team"
    assert source_dry_run["report"]["owner_scope_id"] == 21
    assert check_updates["has_update"] is True
    assert upgrade_candidate["candidate_id"] == "cand-2"
    assert dry_run_upgrade["plan"]["owner_scope_type"] == "team"
    assert source_dry_run_upgrade["plan"]["owner_scope_id"] == 21
    assert imported["governance_pack_id"] == 81
    assert source_imported["governance_pack_id"] == 82
    assert source_upgrade_executed["ok"] is True
    assert upgrade_executed["ok"] is True
    assert detail["pack_id"] == "baseline"
    assert history[0]["to_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_server_unified_service_routes_remaining_governance_workspace_binding_and_secret_admin_tail():
    from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService

    client = FakeMCPUnifiedClient(
        bootstrap=MCPAccessBootstrapResponse(
            principal=MCPAccessBootstrapPrincipal(user_id=7, username="operator", role="admin", is_admin=True),
            manageable_team_ids=[21],
            manageable_org_ids=[],
            can_use_system_admin_scope=True,
        )
    )
    service = ServerUnifiedMCPService(client=client)
    target = ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv.example/api")

    governance_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="governance",
    )
    external_context = await service.resolve_access_context(
        target=target,
        selected_scope="team",
        selected_scope_ref="21",
        selected_section="external_servers",
    )

    assignment_workspaces = await service.list_policy_assignment_workspaces(
        target=target,
        access_context=governance_context,
        assignment_id=2,
    )
    added_workspace = await service.add_policy_assignment_workspace(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        workspace_id="ws-1",
    )
    deleted_workspace = await service.delete_policy_assignment_workspace(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        workspace_id="ws-1",
    )
    profile_bindings = await service.list_profile_credential_bindings(
        target=target,
        access_context=governance_context,
        profile_id=1,
    )
    profile_binding = await service.upsert_profile_credential_binding(
        target=target,
        access_context=governance_context,
        profile_id=1,
        server_id="docs",
        payload={"managed_secret_ref_id": "secret-1"},
    )
    profile_slot_binding = await service.upsert_profile_credential_binding(
        target=target,
        access_context=governance_context,
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
        payload={"managed_secret_ref_id": "secret-1"},
    )
    deleted_profile_binding = await service.delete_profile_credential_binding(
        target=target,
        access_context=governance_context,
        profile_id=1,
        server_id="docs",
    )
    deleted_profile_slot_binding = await service.delete_profile_credential_binding(
        target=target,
        access_context=governance_context,
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
    )
    profile_slot_status = await service.get_profile_slot_credential_status(
        target=target,
        access_context=governance_context,
        profile_id=1,
        server_id="docs",
        slot_name="token_readonly",
    )
    assignment_bindings = await service.list_assignment_credential_bindings(
        target=target,
        access_context=governance_context,
        assignment_id=2,
    )
    assignment_binding = await service.upsert_assignment_credential_binding(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        server_id="docs",
        payload={"binding_mode": "grant", "managed_secret_ref_id": "secret-1"},
    )
    assignment_slot_binding = await service.upsert_assignment_credential_binding(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
        payload={"binding_mode": "grant", "managed_secret_ref_id": "secret-1"},
    )
    deleted_assignment_binding = await service.delete_assignment_credential_binding(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        server_id="docs",
    )
    deleted_assignment_slot_binding = await service.delete_assignment_credential_binding(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
    )
    assignment_slot_status = await service.get_assignment_slot_credential_status(
        target=target,
        access_context=governance_context,
        assignment_id=2,
        server_id="docs",
        slot_name="token_readonly",
    )
    external_secret = await service.set_external_server_secret(
        target=target,
        access_context=external_context,
        server_id="docs",
        secret="replace-me",
    )

    assert assignment_workspaces[0]["workspace_id"] == "ws-1"
    assert added_workspace["workspace_id"] == "ws-1"
    assert deleted_workspace["ok"] is True
    assert profile_bindings[0]["external_server_id"] == "docs"
    assert profile_binding["external_server_id"] == "docs"
    assert profile_slot_binding["slot_name"] == "token_readonly"
    assert deleted_profile_binding["ok"] is True
    assert deleted_profile_slot_binding["ok"] is True
    assert profile_slot_status["status"] == "configured"
    assert assignment_bindings[0]["external_server_id"] == "docs"
    assert assignment_binding["binding_mode"] == "grant"
    assert assignment_slot_binding["slot_name"] == "token_readonly"
    assert deleted_assignment_binding["ok"] is True
    assert deleted_assignment_slot_binding["ok"] is True
    assert assignment_slot_status["status"] == "configured"
    assert external_secret["secret_ref_id"] == "secret-1"
