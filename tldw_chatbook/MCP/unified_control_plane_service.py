from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from .server_target_store import ConfiguredServerTargetStore
from .unified_context_store import UnifiedMCPContextStore
from .unified_control_models import ServerAccessContext, UnifiedMCPContext


class UnifiedMCPControlPlaneService:
    """Destination-local orchestration for local/server Unified MCP browse flows."""

    def __init__(
        self,
        *,
        target_store: ConfiguredServerTargetStore | None,
        context_store: UnifiedMCPContextStore | None,
        local_service: Any,
        server_service: Any,
    ) -> None:
        self.target_store = target_store
        self.context_store = context_store
        self.local_service = local_service
        self.server_service = server_service
        self.context = self.context_store.load() if self.context_store is not None else UnifiedMCPContext()

    @property
    def selected_source(self) -> str:
        return self.context.selected_source

    async def load_context(self) -> UnifiedMCPContext:
        if self.context_store is not None:
            self.context = self.context_store.load()
        return self.context

    async def select_source(self, source: str) -> UnifiedMCPContext:
        normalized_source = "server" if str(source or "").strip() == "server" else "local"
        self.context = replace(self.context, selected_source=normalized_source)
        if normalized_source == "server" and self.context.selected_active_server_id is None:
            default_target = self._resolve_target(None)
            if default_target is not None:
                return await self.select_server_target(default_target.server_id)
        self._persist_context()
        return self.context

    async def select_server_target(self, server_id: str | None) -> UnifiedMCPContext:
        target = self._resolve_target(server_id)
        if target is None:
            raise KeyError(f"Unknown server_id: {server_id}")
        if self.server_service is None:
            raise ValueError("Server Unified MCP service is unavailable.")

        restored_state = self.context.per_server_state.get(target.server_id, ServerAccessContext(server_id=target.server_id))
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=restored_state.selected_scope or self.context.selected_scope,
                selected_scope_ref=restored_state.selected_scope_ref or self.context.selected_scope_ref,
                selected_section=restored_state.selected_section or self.context.selected_section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def select_scope(self, scope: str | None, scope_ref: str | None = None) -> UnifiedMCPContext:
        if self.context.selected_source != "server":
            self.context = replace(
                self.context,
                selected_scope=scope,
                selected_scope_ref=scope_ref,
            )
            self._persist_context()
            return self.context

        target = self._require_active_server_target()
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=scope,
                selected_scope_ref=scope_ref,
                selected_section=self.context.selected_section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def select_section(self, section: str | None) -> UnifiedMCPContext:
        if self.context.selected_source != "server":
            self.context = replace(self.context, selected_section=section)
            self._persist_context()
            return self.context

        target = self._require_active_server_target()
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=self.context.selected_scope,
                selected_scope_ref=self.context.selected_scope_ref,
                selected_section=section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def load_section(self, section: str | None = None) -> dict[str, Any]:
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "server":
            target = self._require_active_server_target()
            access_context = self.context.per_server_state.get(target.server_id)
            if access_context is None:
                await self.select_server_target(target.server_id)
                access_context = self.context.per_server_state.get(target.server_id)
            if access_context is None:
                raise RuntimeError("Failed to resolve Unified MCP server access context.")
            if access_context.selected_section != effective_section:
                await self.select_section(effective_section)
                access_context = self.context.per_server_state[target.server_id]

            if effective_section == "overview":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_overview(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "inventory":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_inventory(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "catalogs":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_catalogs(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "external_servers":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_external_servers(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "governance":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_governance(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "advanced":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_advanced(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            raise ValueError(f"Unsupported Unified MCP section: {effective_section}")

        self.context = replace(self.context, selected_section=effective_section)
        self._persist_context()
        if effective_section == "overview":
            return await self._maybe_await(self.local_service.get_overview())
        if effective_section == "inventory":
            return await self._maybe_await(self.local_service.get_inventory())
        if effective_section == "external_servers":
            return await self._maybe_await(self.local_service.get_external_servers())
        if effective_section == "governance":
            return {
                "source": "local",
                "section": "governance",
                "rules": list(await self._maybe_await(self.local_service.get_governance())),
            }
        if effective_section == "advanced":
            return await self._maybe_await(self.local_service.get_advanced())
        raise ValueError(f"Unsupported Unified MCP section: {effective_section}")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _with_server_context(self, payload: dict[str, Any], *, section: str) -> dict[str, Any]:
        return {
            **dict(payload or {}),
            "source": "server",
            "section": section,
        }

    def available_actions(self) -> list[dict[str, Any]]:
        if self.context.selected_source != "server":
            if (self.context.selected_section or "overview") == "inventory":
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
            if (self.context.selected_section or "overview") == "external_servers":
                return [
                    {
                        "name": "profile.save",
                        "label": "Save Profile",
                        "action_id": "mcp.external_profiles.configure.local",
                        "payload_template": '{"profile_id":"demo","command":"python","args":["-m","demo.server"]}',
                    },
                    {
                        "name": "profile.delete",
                        "label": "Delete Profile",
                        "action_id": "mcp.external_profiles.configure.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.connect",
                        "label": "Connect Profile",
                        "action_id": "mcp.external_profiles.launch.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.disconnect",
                        "label": "Disconnect Profile",
                        "action_id": "mcp.external_profiles.launch.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.test",
                        "label": "Test Profile",
                        "action_id": "mcp.external_profiles.trigger.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.refresh",
                        "label": "Refresh Profile",
                        "action_id": "mcp.external_profiles.observe.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                ]
            if (self.context.selected_section or "overview") == "governance":
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
            if (self.context.selected_section or "overview") == "advanced":
                return [
                    {
                        "name": "runtime.access.preview",
                        "label": "Preview Local Runtime Access",
                        "action_id": "mcp.governance.observe.local",
                        "payload_template": '{"action_name":"tool.execute","payload":{"tool_name":"search_notes","arguments":{"query":"example"}}}',
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
                        "name": "approval_request.delete",
                        "label": "Delete Local Request",
                        "action_id": "mcp.governance.approve.local",
                        "payload_template": '{"request_id":"approval-a"}',
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
            return []

        if (self.context.selected_section or "overview") == "catalogs":
            if (self.context.selected_scope or "personal") not in {"team", "org", "system_admin"}:
                return []
            return [
                {
                    "name": "catalog.create",
                    "label": "Create Catalog",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"name":"Team Catalog","description":"Scoped"}',
                },
                {
                    "name": "catalog.entry.create",
                    "label": "Add Catalog Entry",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9,"tool_name":"media.search"}',
                },
                {
                    "name": "catalog.delete",
                    "label": "Delete Catalog",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9}',
                },
                {
                    "name": "catalog.entry.delete",
                    "label": "Delete Catalog Entry",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9,"tool_name":"media.search"}',
                },
            ]

        if (self.context.selected_section or "overview") == "external_servers":
            if (self.context.selected_scope or "personal") not in {"team", "org", "system_admin"}:
                return []
            return [
                {
                    "name": "external_server.create",
                    "label": "Create External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs","name":"Docs","transport":"http","config":{"url":"https://docs.example/mcp"}}',
                },
                {
                    "name": "external_server.update",
                    "label": "Update External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs","name":"Docs","enabled":true}',
                },
                {
                    "name": "external_server.delete",
                    "label": "Delete External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs"}',
                },
                {
                    "name": "external_server.import",
                    "label": "Import External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"legacy-docs"}',
                },
                {
                    "name": "external_server.auth_template.update",
                    "label": "Update Auth Template",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","mode":"template","mappings":[]}',
                },
                {
                    "name": "external_server.slots.list",
                    "label": "List Credential Slots",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"server_id":"docs"}',
                },
                {
                    "name": "external_server.slot.create",
                    "label": "Create Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","display_name":"Read-only token","secret_kind":"bearer_token","privilege_class":"read","is_required":true}',
                },
                {
                    "name": "external_server.slot.update",
                    "label": "Update Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","display_name":"Read-only token"}',
                },
                {
                    "name": "external_server.slot.delete",
                    "label": "Delete Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "external_server.slot.secret.set",
                    "label": "Set Slot Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","secret":"replace-me"}',
                },
                {
                    "name": "external_server.slot.secret.clear",
                    "label": "Clear Slot Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "external_server.secret.set",
                    "label": "Set External Server Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","secret":"replace-me"}',
                },
            ]

        if (self.context.selected_section or "overview") == "governance":
            return [
                {
                    "name": "permission_profile.create",
                    "label": "Create Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Default","description":"Scoped profile","mode":"custom","policy_document":{},"is_active":true}',
                },
                {
                    "name": "permission_profile.update",
                    "label": "Update Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":1,"name":"Updated Profile"}',
                },
                {
                    "name": "permission_profile.delete",
                    "label": "Delete Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":1}',
                },
                {
                    "name": "policy_assignment.create",
                    "label": "Create Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"target_type":"persona","target_id":"persona-a","profile_id":1,"inline_policy_document":{},"is_active":true}',
                },
                {
                    "name": "policy_assignment.update",
                    "label": "Update Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"approval_policy_id":7}',
                },
                {
                    "name": "policy_assignment.delete",
                    "label": "Delete Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.override.get",
                    "label": "Get Assignment Override",
                    "action_id": "mcp.governance.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.override.upsert",
                    "label": "Upsert Assignment Override",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"override_policy_document":{"allowed_tools":["mcp.tool"]},"is_active":true}',
                },
                {
                    "name": "policy_assignment.override.delete",
                    "label": "Delete Assignment Override",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "approval_policy.create",
                    "label": "Create Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Default Approval","mode":"ask_every_time","rules":{},"is_active":true}',
                },
                {
                    "name": "approval_policy.update",
                    "label": "Update Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"approval_policy_id":7,"name":"Updated Approval"}',
                },
                {
                    "name": "approval_policy.delete",
                    "label": "Delete Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"approval_policy_id":7}',
                },
                {
                    "name": "approval_decision.create",
                    "label": "Record Approval Decision",
                    "action_id": "mcp.governance.approve.server",
                    "payload_template": '{"approval_policy_id":7,"context_key":"user:7:docs","tool_name":"docs.search","scope_key":"team:21","decision":"approved","duration":"once"}',
                },
                {
                    "name": "policy_assignment.external_access.get",
                    "label": "Preview External Access",
                    "action_id": "mcp.effective_access.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.workspaces.list",
                    "label": "List Assignment Workspaces",
                    "action_id": "mcp.governance.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.workspace.add",
                    "label": "Add Assignment Workspace",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"workspace_id":"ws-1"}',
                },
                {
                    "name": "policy_assignment.workspace.delete",
                    "label": "Delete Assignment Workspace",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"workspace_id":"ws-1"}',
                },
                {
                    "name": "permission_profile.bindings.list",
                    "label": "List Profile Bindings",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"profile_id":1}',
                },
                {
                    "name": "permission_profile.binding.upsert",
                    "label": "Upsert Profile Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "permission_profile.slot_binding.upsert",
                    "label": "Upsert Profile Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "permission_profile.binding.delete",
                    "label": "Delete Profile Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs"}',
                },
                {
                    "name": "permission_profile.slot_binding.delete",
                    "label": "Delete Profile Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "permission_profile.slot_status.get",
                    "label": "Get Profile Slot Status",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "policy_assignment.bindings.list",
                    "label": "List Assignment Bindings",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.binding.upsert",
                    "label": "Upsert Assignment Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","binding_mode":"grant","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "policy_assignment.slot_binding.upsert",
                    "label": "Upsert Assignment Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly","binding_mode":"grant","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "policy_assignment.binding.delete",
                    "label": "Delete Assignment Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs"}',
                },
                {
                    "name": "policy_assignment.slot_binding.delete",
                    "label": "Delete Assignment Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "policy_assignment.slot_status.get",
                    "label": "Get Assignment Slot Status",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "acp_profile.create",
                    "label": "Create ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Workspace ACP","profile":{},"is_active":true}',
                },
                {
                    "name": "acp_profile.update",
                    "label": "Update ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":8,"name":"Updated ACP"}',
                },
                {
                    "name": "acp_profile.delete",
                    "label": "Delete ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":8}',
                },
            ]

        if (self.context.selected_section or "overview") == "advanced":
            actions = [
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
                    "name": "governance_pack.source.dry_run",
                    "label": "Dry Run Governance Pack Source",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"candidate_id":"cand-1"}',
                },
                {
                    "name": "governance_pack.check_updates",
                    "label": "Check Governance Pack Updates",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.prepare_upgrade_candidate",
                    "label": "Prepare Governance Pack Upgrade Candidate",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.dry_run_upgrade",
                    "label": "Dry Run Governance Pack Upgrade",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"source_governance_pack_id":81,"pack":{"manifest":{"pack_id":"baseline","version":"1.1.0"}},"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.source.dry_run_upgrade",
                    "label": "Dry Run Governance Pack Source Upgrade",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"candidate_id":"cand-1","source_governance_pack_id":81,"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.import",
                    "label": "Import Governance Pack",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"pack":{"manifest":{"pack_id":"baseline","version":"1.0.0"}}}',
                },
                {
                    "name": "governance_pack.source.import",
                    "label": "Import Governance Pack Source",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"candidate_id":"cand-1"}',
                },
                {
                    "name": "governance_pack.source.execute_upgrade",
                    "label": "Execute Governance Pack Source Upgrade",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"candidate_id":"cand-1","source_governance_pack_id":81,"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.execute_upgrade",
                    "label": "Execute Governance Pack Upgrade",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"source_governance_pack_id":81,"pack":{"manifest":{"pack_id":"baseline","version":"1.1.0"}},"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.detail.get",
                    "label": "Get Governance Pack Detail",
                    "action_id": "mcp.advanced.observe.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.upgrade_history.list",
                    "label": "List Governance Pack Upgrade History",
                    "action_id": "mcp.advanced.observe.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "path_scope_object.create",
                    "label": "Create Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Workspace Root","path_scope_document":{"path_scope_mode":"workspace_root"},"is_active":true}',
                },
                {
                    "name": "path_scope_object.update",
                    "label": "Update Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"path_scope_object_id":5,"name":"Workspace Root Updated"}',
                },
                {
                    "name": "path_scope_object.delete",
                    "label": "Delete Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"path_scope_object_id":5}',
                },
                {
                    "name": "workspace_set_object.create",
                    "label": "Create Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Research Set","description":"Trusted workspaces","is_active":true}',
                },
                {
                    "name": "workspace_set_object.update",
                    "label": "Update Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"description":"Updated"}',
                },
                {
                    "name": "workspace_set_object.delete",
                    "label": "Delete Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6}',
                },
                {
                    "name": "workspace_set_object.members.list",
                    "label": "List Workspace Set Members",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6}',
                },
                {
                    "name": "workspace_set_object.member.add",
                    "label": "Add Workspace Set Member",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"workspace_id":"ws-1"}',
                },
                {
                    "name": "workspace_set_object.member.delete",
                    "label": "Delete Workspace Set Member",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"workspace_id":"ws-1"}',
                },
            ]
            if (self.context.selected_scope or "personal") in {"team", "org", "system_admin"}:
                actions[0:0] = [
                    {
                        "name": "capability_mapping.preview",
                        "label": "Preview Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mapping_id":"filesystem-write","capability_name":"filesystem.write","resolved_policy_document":{"allowed_tools":["filesystem.write"]},"is_active":true}',
                    },
                    {
                        "name": "capability_mapping.create",
                        "label": "Create Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mapping_id":"filesystem-write","title":"Filesystem Write","capability_name":"filesystem.write","resolved_policy_document":{"allowed_tools":["filesystem.write"]},"is_active":true}',
                    },
                    {
                        "name": "capability_mapping.update",
                        "label": "Update Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"capability_adapter_mapping_id":3,"title":"Filesystem Write Updated"}',
                    },
                    {
                        "name": "capability_mapping.delete",
                        "label": "Delete Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"capability_adapter_mapping_id":3}',
                    },
                    {
                        "name": "shared_workspace.create",
                        "label": "Create Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"workspace_id":"shared-ws","display_name":"Shared Workspace","absolute_root":"/srv/shared","is_active":true}',
                    },
                    {
                        "name": "shared_workspace.update",
                        "label": "Update Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"shared_workspace_id":7,"display_name":"Shared Workspace Updated"}',
                    },
                    {
                        "name": "shared_workspace.delete",
                        "label": "Delete Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"shared_workspace_id":7}',
                    },
                ]
            if (self.context.selected_scope or "personal") == "system_admin":
                actions.insert(
                    0,
                    {
                        "name": "governance_pack_trust_policy.update",
                        "label": "Update Trust Policy",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mode":"allowlist","allowed_sources":["git@example.com:trusted/repo.git"]}',
                    },
                )
            return actions

        return []

    def runtime_state_override(self) -> RuntimeSourceState:
        if self.context.selected_source != "server":
            return RuntimeSourceState(active_source="local")

        target = self._resolve_target(self.context.selected_active_server_id)
        access_context = None
        if target is not None:
            access_context = self.context.per_server_state.get(target.server_id)
        target_status = access_context.target_status if access_context is not None else None
        return RuntimeSourceState(
            active_source="server",
            active_server_id=(target.server_id if target is not None else self.context.selected_active_server_id),
            server_configured=target is not None,
            server_reachability=(
                target_status.last_known_reachability
                if target_status is not None and target_status.last_known_reachability is not None
                else "reachable"
            ),
            server_auth_state=(
                target_status.last_known_auth_state
                if target_status is not None and target_status.last_known_auth_state is not None
                else "authenticated"
            ),
            last_known_server_label=(
                target_status.last_known_server_label
                if target_status is not None and target_status.last_known_server_label is not None
                else (target.label if target is not None else None)
            ),
        )

    async def run_action(self, action_name: str, payload: dict[str, Any] | None = None) -> Any:
        payload = dict(payload or {})
        if self.context.selected_source != "server":
            if action_name == "profile.save":
                return await self._maybe_await(self.local_service.save_external_profile(payload))
            if action_name == "profile.delete":
                return await self._maybe_await(self.local_service.delete_external_profile(self._require_field(payload, "profile_id")))
            if action_name == "profile.connect":
                return await self._maybe_await(self.local_service.connect_profile(self._require_field(payload, "profile_id")))
            if action_name == "profile.disconnect":
                return await self._maybe_await(self.local_service.disconnect_profile(self._require_field(payload, "profile_id")))
            if action_name == "profile.test":
                return await self._maybe_await(self.local_service.test_external_profile(self._require_field(payload, "profile_id")))
            if action_name == "profile.refresh":
                return await self._maybe_await(self.local_service.refresh_external_profile(self._require_field(payload, "profile_id")))
            if action_name == "tool.execute":
                return await self._maybe_await(
                    self.local_service.execute_tool(
                        self._require_field(payload, "tool_name"),
                        payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
                    )
                )
            if action_name == "resource.read":
                return await self._maybe_await(
                    self.local_service.read_resource(self._require_field(payload, "resource_uri"))
                )
            if action_name == "prompt.get":
                return await self._maybe_await(
                    self.local_service.get_prompt(
                        self._require_field(payload, "prompt_name"),
                        payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
                    )
                )
            if action_name == "governance_rule.save":
                return await self._maybe_await(self.local_service.save_governance_rule(payload))
            if action_name == "governance_rule.preview":
                return await self._maybe_await(
                    self.local_service.preview_governance_decision(self._require_field(payload, "capability_id"))
                )
            if action_name == "governance_rule.delete":
                return await self._maybe_await(self.local_service.delete_governance_rule(self._require_field(payload, "rule_id")))
            if action_name == "runtime.access.preview":
                nested_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                return await self._maybe_await(
                    self.local_service.preview_runtime_access(
                        self._require_field(payload, "action_name"),
                        nested_payload,
                    )
                )
            if action_name == "runtime.activity.list":
                return await self._maybe_await(self.local_service.get_runtime_activity(int(payload.get("limit", 20))))
            if action_name == "runtime.protocol.inspect":
                return await self._maybe_await(self.local_service.get_runtime_protocol_diagnostics())
            if action_name == "runtime.health.get":
                return await self._maybe_await(self.local_service.get_runtime_health())
            if action_name == "approval_requests.list":
                status = payload.get("status")
                resolved_action_id = payload.get("resolved_action_id")
                return await self._maybe_await(
                    self.local_service.list_approval_requests(
                        str(status).strip() if status is not None else None,
                        str(resolved_action_id).strip() if resolved_action_id is not None else None,
                    )
                )
            if action_name == "approval_request.approve":
                return await self._maybe_await(
                    self.local_service.approve_approval_request(self._require_field(payload, "request_id"))
                )
            if action_name == "approval_request.deny":
                return await self._maybe_await(
                    self.local_service.deny_approval_request(self._require_field(payload, "request_id"))
                )
            if action_name == "approval_request.delete":
                return await self._maybe_await(
                    self.local_service.delete_approval_request(self._require_field(payload, "request_id"))
                )
            if action_name == "runtime.status.get":
                return await self._maybe_await(self.local_service.get_runtime_status())
            if action_name == "runtime.request":
                return await self._maybe_await(
                    self.local_service.run_runtime_request(
                        self._require_field(payload, "method"),
                        payload.get("params") if isinstance(payload.get("params"), dict) else {},
                    )
                )
            if action_name == "runtime.batch":
                requests = payload.get("requests")
                if not isinstance(requests, list):
                    raise ValueError("Unified MCP action requires 'requests'.")
                return await self._maybe_await(self.local_service.run_runtime_batch(requests))
            raise ValueError(f"Unsupported Unified MCP local action: {action_name}")

        target = self._require_active_server_target()
        access_context = self.context.per_server_state.get(target.server_id)
        if access_context is None:
            raise RuntimeError("Failed to resolve Unified MCP server access context.")

        if action_name == "catalog.create":
            return await self._maybe_await(
                self.server_service.create_catalog(target=target, access_context=access_context, payload=payload)
            )
        if action_name == "catalog.entry.create":
            catalog_id = self._require_field(payload, "catalog_id")
            entry_payload = {key: value for key, value in payload.items() if key != "catalog_id"}
            return await self._maybe_await(
                self.server_service.create_catalog_entry(
                    target=target,
                    access_context=access_context,
                    catalog_id=catalog_id,
                    payload=entry_payload,
                )
            )
        if action_name == "catalog.delete":
            return await self._maybe_await(
                self.server_service.delete_catalog(
                    target=target,
                    access_context=access_context,
                    catalog_id=self._require_field(payload, "catalog_id"),
                )
            )
        if action_name == "catalog.entry.delete":
            return await self._maybe_await(
                self.server_service.delete_catalog_entry(
                    target=target,
                    access_context=access_context,
                    catalog_id=self._require_field(payload, "catalog_id"),
                    tool_name=self._require_field(payload, "tool_name"),
                )
            )
        if action_name == "external_server.create":
            return await self._maybe_await(
                self.server_service.create_external_server(target=target, access_context=access_context, payload=payload)
            )
        if action_name == "external_server.update":
            server_id = self._require_field(payload, "server_id")
            update_payload = {key: value for key, value in payload.items() if key != "server_id"}
            return await self._maybe_await(
                self.server_service.update_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=update_payload,
                )
            )
        if action_name == "external_server.delete":
            return await self._maybe_await(
                self.server_service.delete_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.import":
            return await self._maybe_await(
                self.server_service.import_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.auth_template.update":
            server_id = self._require_field(payload, "server_id")
            update_payload = {key: value for key, value in payload.items() if key != "server_id"}
            return await self._maybe_await(
                self.server_service.update_external_server_auth_template(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=update_payload,
                )
            )
        if action_name == "external_server.slots.list":
            return await self._maybe_await(
                self.server_service.list_external_server_credential_slots(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.slot.create":
            server_id = self._require_field(payload, "server_id")
            slot_payload = {key: value for key, value in payload.items() if key != "server_id"}
            return await self._maybe_await(
                self.server_service.create_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=slot_payload,
                )
            )
        if action_name == "external_server.slot.update":
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            slot_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.update_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=slot_payload,
                )
            )
        if action_name == "external_server.slot.delete":
            return await self._maybe_await(
                self.server_service.delete_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "external_server.slot.secret.set":
            return await self._maybe_await(
                self.server_service.set_external_server_slot_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                    secret=self._require_field(payload, "secret"),
                )
            )
        if action_name == "external_server.slot.secret.clear":
            return await self._maybe_await(
                self.server_service.clear_external_server_slot_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "external_server.secret.set":
            return await self._maybe_await(
                self.server_service.set_external_server_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    secret=self._require_field(payload, "secret"),
                )
            )
        if action_name == "permission_profile.create":
            return await self._maybe_await(
                self.server_service.create_permission_profile(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "permission_profile.update":
            profile_id = self._require_field(payload, "profile_id")
            update_payload = {key: value for key, value in payload.items() if key != "profile_id"}
            return await self._maybe_await(
                self.server_service.update_permission_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    payload=update_payload,
                )
            )
        if action_name == "permission_profile.delete":
            return await self._maybe_await(
                self.server_service.delete_permission_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "policy_assignment.create":
            return await self._maybe_await(
                self.server_service.create_policy_assignment(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "policy_assignment.update":
            assignment_id = self._require_field(payload, "assignment_id")
            update_payload = {key: value for key, value in payload.items() if key != "assignment_id"}
            return await self._maybe_await(
                self.server_service.update_policy_assignment(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    payload=update_payload,
                )
            )
        if action_name == "policy_assignment.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.override.get":
            return await self._maybe_await(
                self.server_service.get_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.override.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            override_payload = {key: value for key, value in payload.items() if key != "assignment_id"}
            return await self._maybe_await(
                self.server_service.upsert_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    payload=override_payload,
                )
            )
        if action_name == "policy_assignment.override.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "approval_policy.create":
            return await self._maybe_await(
                self.server_service.create_approval_policy(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "approval_policy.update":
            approval_policy_id = self._require_field(payload, "approval_policy_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "approval_policy_id"
            }
            return await self._maybe_await(
                self.server_service.update_approval_policy(
                    target=target,
                    access_context=access_context,
                    approval_policy_id=approval_policy_id,
                    payload=update_payload,
                )
            )
        if action_name == "approval_policy.delete":
            return await self._maybe_await(
                self.server_service.delete_approval_policy(
                    target=target,
                    access_context=access_context,
                    approval_policy_id=self._require_field(payload, "approval_policy_id"),
                )
            )
        if action_name == "approval_decision.create":
            return await self._maybe_await(
                self.server_service.create_approval_decision(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "policy_assignment.external_access.get":
            return await self._maybe_await(
                self.server_service.get_assignment_external_access(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.workspaces.list":
            return await self._maybe_await(
                self.server_service.list_policy_assignment_workspaces(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.workspace.add":
            return await self._maybe_await(
                self.server_service.add_policy_assignment_workspace(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "policy_assignment.workspace.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment_workspace(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "permission_profile.bindings.list":
            return await self._maybe_await(
                self.server_service.list_profile_credential_bindings(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "permission_profile.binding.upsert":
            profile_id = self._require_field(payload, "profile_id")
            server_id = self._require_field(payload, "server_id")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"profile_id", "server_id"}
            }
            return await self._maybe_await(
                self.server_service.upsert_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    server_id=server_id,
                    payload=binding_payload,
                )
            )
        if action_name == "permission_profile.slot_binding.upsert":
            profile_id = self._require_field(payload, "profile_id")
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"profile_id", "server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.upsert_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=binding_payload,
                )
            )
        if action_name == "permission_profile.binding.delete":
            return await self._maybe_await(
                self.server_service.delete_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "permission_profile.slot_binding.delete":
            return await self._maybe_await(
                self.server_service.delete_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "permission_profile.slot_status.get":
            return await self._maybe_await(
                self.server_service.get_profile_slot_credential_status(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "policy_assignment.bindings.list":
            return await self._maybe_await(
                self.server_service.list_assignment_credential_bindings(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.binding.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            server_id = self._require_field(payload, "server_id")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"assignment_id", "server_id"}
            }
            return await self._maybe_await(
                self.server_service.upsert_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    server_id=server_id,
                    payload=binding_payload,
                )
            )
        if action_name == "policy_assignment.slot_binding.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"assignment_id", "server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.upsert_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=binding_payload,
                )
            )
        if action_name == "policy_assignment.binding.delete":
            return await self._maybe_await(
                self.server_service.delete_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "policy_assignment.slot_binding.delete":
            return await self._maybe_await(
                self.server_service.delete_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "policy_assignment.slot_status.get":
            return await self._maybe_await(
                self.server_service.get_assignment_slot_credential_status(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "acp_profile.create":
            return await self._maybe_await(
                self.server_service.create_acp_profile(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "acp_profile.update":
            profile_id = self._require_field(payload, "profile_id")
            update_payload = {key: value for key, value in payload.items() if key != "profile_id"}
            return await self._maybe_await(
                self.server_service.update_acp_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    payload=update_payload,
                )
            )
        if action_name == "acp_profile.delete":
            return await self._maybe_await(
                self.server_service.delete_acp_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "governance_pack_trust_policy.update":
            return await self._maybe_await(
                self.server_service.update_governance_pack_trust_policy(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.dry_run":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.prepare":
            return await self._maybe_await(
                self.server_service.prepare_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.dry_run":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.check_updates":
            return await self._maybe_await(
                self.server_service.check_governance_pack_updates(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(payload, "governance_pack_id"),
                )
            )
        if action_name == "governance_pack.prepare_upgrade_candidate":
            return await self._maybe_await(
                self.server_service.prepare_governance_pack_upgrade_candidate(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(payload, "governance_pack_id"),
                )
            )
        if action_name == "governance_pack.dry_run_upgrade":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.dry_run_upgrade":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_source_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.import":
            return await self._maybe_await(
                self.server_service.import_governance_pack(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.import":
            return await self._maybe_await(
                self.server_service.import_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.execute_upgrade":
            return await self._maybe_await(
                self.server_service.execute_governance_pack_source_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.execute_upgrade":
            return await self._maybe_await(
                self.server_service.execute_governance_pack_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.detail.get":
            return await self._maybe_await(
                self.server_service.get_governance_pack_detail(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(payload, "governance_pack_id"),
                )
            )
        if action_name == "governance_pack.upgrade_history.list":
            return await self._maybe_await(
                self.server_service.list_governance_pack_upgrade_history(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(payload, "governance_pack_id"),
                )
            )
        if action_name == "path_scope_object.create":
            return await self._maybe_await(
                self.server_service.create_path_scope_object(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "path_scope_object.update":
            path_scope_object_id = self._require_field(payload, "path_scope_object_id")
            update_payload = {key: value for key, value in payload.items() if key != "path_scope_object_id"}
            return await self._maybe_await(
                self.server_service.update_path_scope_object(
                    target=target,
                    access_context=access_context,
                    path_scope_object_id=path_scope_object_id,
                    payload=update_payload,
                )
            )
        if action_name == "path_scope_object.delete":
            return await self._maybe_await(
                self.server_service.delete_path_scope_object(
                    target=target,
                    access_context=access_context,
                    path_scope_object_id=self._require_field(payload, "path_scope_object_id"),
                )
            )
        if action_name == "capability_mapping.preview":
            return await self._maybe_await(
                self.server_service.preview_capability_mapping(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "capability_mapping.create":
            return await self._maybe_await(
                self.server_service.create_capability_mapping(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "capability_mapping.update":
            capability_adapter_mapping_id = self._require_field(payload, "capability_adapter_mapping_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "capability_adapter_mapping_id"
            }
            return await self._maybe_await(
                self.server_service.update_capability_mapping(
                    target=target,
                    access_context=access_context,
                    capability_adapter_mapping_id=capability_adapter_mapping_id,
                    payload=update_payload,
                )
            )
        if action_name == "capability_mapping.delete":
            return await self._maybe_await(
                self.server_service.delete_capability_mapping(
                    target=target,
                    access_context=access_context,
                    capability_adapter_mapping_id=self._require_field(payload, "capability_adapter_mapping_id"),
                )
            )
        if action_name == "workspace_set_object.create":
            return await self._maybe_await(
                self.server_service.create_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "workspace_set_object.update":
            workspace_set_object_id = self._require_field(payload, "workspace_set_object_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "workspace_set_object_id"
            }
            return await self._maybe_await(
                self.server_service.update_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=workspace_set_object_id,
                    payload=update_payload,
                )
            )
        if action_name == "workspace_set_object.delete":
            return await self._maybe_await(
                self.server_service.delete_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(payload, "workspace_set_object_id"),
                )
            )
        if action_name == "workspace_set_object.members.list":
            return await self._maybe_await(
                self.server_service.list_workspace_set_members(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(payload, "workspace_set_object_id"),
                )
            )
        if action_name == "workspace_set_object.member.add":
            workspace_set_object_id = self._require_field(payload, "workspace_set_object_id")
            member_payload = {
                key: value
                for key, value in payload.items()
                if key != "workspace_set_object_id"
            }
            return await self._maybe_await(
                self.server_service.add_workspace_set_member(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=workspace_set_object_id,
                    payload=member_payload,
                )
            )
        if action_name == "workspace_set_object.member.delete":
            return await self._maybe_await(
                self.server_service.delete_workspace_set_member(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(payload, "workspace_set_object_id"),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "shared_workspace.create":
            return await self._maybe_await(
                self.server_service.create_shared_workspace(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "shared_workspace.update":
            shared_workspace_id = self._require_field(payload, "shared_workspace_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "shared_workspace_id"
            }
            return await self._maybe_await(
                self.server_service.update_shared_workspace(
                    target=target,
                    access_context=access_context,
                    shared_workspace_id=shared_workspace_id,
                    payload=update_payload,
                )
            )
        if action_name == "shared_workspace.delete":
            return await self._maybe_await(
                self.server_service.delete_shared_workspace(
                    target=target,
                    access_context=access_context,
                    shared_workspace_id=self._require_field(payload, "shared_workspace_id"),
                )
            )
        raise ValueError(f"Unsupported Unified MCP server action: {action_name}")

    def _apply_server_access_context(
        self,
        server_id: str,
        access_context: ServerAccessContext,
    ) -> None:
        per_server_state = dict(self.context.per_server_state)
        per_server_state[server_id] = access_context
        self.context = replace(
            self.context,
            selected_source="server",
            selected_active_server_id=server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
            selected_section=access_context.selected_section,
            per_server_state=per_server_state,
        )
        self._persist_context()

    def _persist_context(self) -> None:
        if self.context_store is not None:
            self.context_store.save(self.context)

    def _resolve_target(self, server_id: str | None) -> Any:
        if self.target_store is None:
            return None
        return self.target_store.resolve_active_target(server_id)

    def _require_active_server_target(self) -> Any:
        target = self._resolve_target(self.context.selected_active_server_id)
        if target is None:
            raise ValueError("No active Unified MCP server target is selected.")
        return target

    @staticmethod
    def _require_field(payload: dict[str, Any], field_name: str) -> Any:
        value = payload.get(field_name)
        if value in (None, ""):
            raise ValueError(f"Unified MCP action requires '{field_name}'.")
        return value

    # ---- Typed local lifecycle/mutation seam (Phase 2) ----------------------
    # Shared by the Hub UI now and by the Phase 5 chat bridge / agent-runtime
    # MCPToolProvider (task-201) later. Governance enforcement stays inside
    # the local service exactly as run_action's branches rely on it.

    def _lifecycle_timeout(self) -> float:
        try:
            return float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45))
        except (TypeError, ValueError):
            return 45.0

    def _record_local_attempt(self, profile_id: str, action: str, *,
                              ok: bool, error: str | None) -> None:
        store = getattr(self.local_service, "store", None)
        if store is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            previous = store.get_profile_runtime_state(profile_id) or {}
            store.save_profile_runtime_state(profile_id, {
                "last_attempt_at": now,
                "last_action": action,
                "ok": ok,
                "last_ok_at": now if ok else previous.get("last_ok_at"),
                "last_error": None if ok else (error or "")[:300],
            })
        except Exception as exc:
            # Recording is best-effort: it must never mask the lifecycle
            # result or the original exception being propagated.
            logger.warning(f"MCP lifecycle attempt record failed for {profile_id}: {exc}")

    async def _run_local_lifecycle(self, action: str, profile_id: str, coro):
        timeout = self._lifecycle_timeout()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            message = f"Timed out after {timeout:.0f}s"
            self._record_local_attempt(profile_id, action, ok=False, error=message)
            raise RuntimeError(message) from None
        except asyncio.CancelledError:
            self._record_local_attempt(profile_id, action, ok=False, error="Cancelled")
            raise
        except Exception as exc:
            self._record_local_attempt(profile_id, action, ok=False, error=str(exc))
            raise
        self._record_local_attempt(profile_id, action, ok=True, error=None)
        return result

    async def connect_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "connect", profile_id, self.local_service.connect_profile(profile_id))

    async def disconnect_local_profile(self, profile_id: str) -> bool:
        return await self._run_local_lifecycle(
            "disconnect", profile_id, self.local_service.disconnect_profile(profile_id))

    async def test_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "test", profile_id, self.local_service.test_external_profile(profile_id))

    async def refresh_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "refresh", profile_id, self.local_service.refresh_external_profile(profile_id))

    async def save_local_profile(self, payload: dict) -> dict:
        return self.local_service.save_external_profile(dict(payload or {}))

    async def delete_local_profile(self, profile_id: str) -> bool:
        return bool(self.local_service.delete_external_profile(profile_id))

    async def local_external_catalog(self) -> list[dict]:
        records = list(self.local_service.get_external_servers() or [])
        store = getattr(self.local_service, "store", None)
        for record in records:
            profile_id = str(record.get("profile_id") or "")
            record["runtime_state"] = (
                store.get_profile_runtime_state(profile_id) if store else None
            )
        return records
