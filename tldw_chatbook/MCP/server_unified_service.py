from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer, classify_backend_exception
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.tldw_api.mcp_unified_schemas import (
    ACPProfileCreateRequest,
    ACPProfileUpdateRequest,
    ApprovalDecisionCreateRequest,
    ApprovalPolicyCreateRequest,
    ApprovalPolicyUpdateRequest,
    ExternalSecretSetRequest,
    ExternalServerAuthTemplateUpdateRequest,
    ExternalServerCreateRequest,
    ExternalServerCredentialSlotCreateRequest,
    ExternalServerCredentialSlotUpdateRequest,
    ExternalServerUpdateRequest,
    PermissionProfileCreateRequest,
    PermissionProfileUpdateRequest,
    PolicyAssignmentCreateRequest,
    PolicyAssignmentUpdateRequest,
    PolicyOverrideUpsertRequest,
    ScopedToolCatalogCreateRequest,
    ScopedToolCatalogEntryCreateRequest,
    UnifiedMCPAccessContext,
)

from .server_target_store import ConfiguredServerTargetStore
from .unified_control_models import (
    ConfiguredServerTarget,
    SectionCapabilityFlags,
    ServerAccessContext,
    TargetStatusMetadata,
)


class ServerUnifiedMCPService:
    """Resolve server-side MCP browse capabilities and provide section reads."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[ConfiguredServerTarget], Any] | None = None,
        policy_enforcer: ServicePolicyEnforcer | None = None,
        target_store: ConfiguredServerTargetStore | None = None,
    ) -> None:
        self.client = client
        self.client_factory = client_factory
        self.policy_enforcer = policy_enforcer
        self.target_store = target_store
        self._client_cache: dict[str, Any] = {}
        self._browse_cache: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}

    async def resolve_access_context(
        self,
        *,
        target: ConfiguredServerTarget,
        selected_scope: str | None = None,
        selected_scope_ref: str | None = None,
        selected_section: str | None = None,
    ) -> ServerAccessContext:
        client = self._client_for_target(target)

        status_payload = None
        status_error: Exception | None = None
        try:
            status_payload = await client.get_status()
        except Exception as exc:
            status_error = exc
            logger.debug("Unified MCP status probe failed for {}: {}", target.server_id, exc)

        bootstrap = await self._bootstrap_access_context(client)
        effective_scope, effective_scope_ref = self._normalize_scope_selection(
            selected_scope=selected_scope,
            selected_scope_ref=selected_scope_ref,
            manageable_team_ids=bootstrap.manageable_team_ids,
            manageable_org_ids=bootstrap.manageable_org_ids,
            can_use_system_admin_scope=bootstrap.can_use_system_admin_scope,
        )
        api_access_context = self._build_api_access_context(
            selected_scope=effective_scope,
            selected_scope_ref=effective_scope_ref,
        )
        section_capabilities, endpoint_capabilities = await self._probe_section_capabilities(
            client=client,
            access_context=api_access_context,
            overview_allowed=status_error is None,
        )
        effective_section = self._normalize_section_selection(
            selected_section=selected_section,
            section_capabilities=section_capabilities,
        )
        target_status = self._build_target_status(
            target=target,
            status_error=status_error,
        )
        target_status = self._persist_target_status(target, target_status)

        return ServerAccessContext(
            server_id=target.server_id,
            principal_user_id=(
                getattr(getattr(bootstrap, "principal", None), "user_id", None)
                if getattr(bootstrap, "principal", None) is not None
                else None
            ),
            selected_scope=effective_scope,
            selected_scope_ref=effective_scope_ref,
            selected_section=effective_section,
            can_use_personal_scope=True,
            manageable_team_ids=tuple(bootstrap.manageable_team_ids),
            manageable_org_ids=tuple(bootstrap.manageable_org_ids),
            can_use_system_admin_scope=bootstrap.can_use_system_admin_scope,
            section_capabilities=section_capabilities,
            endpoint_capabilities=endpoint_capabilities,
            target_status=target_status,
        )

    async def get_overview(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.runtime.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        cache_key = self._cache_key(
            section="overview",
            server_id=target.server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
        )
        cached = self._browse_cache.get(cache_key)
        if cached is not None:
            return cached

        client = self._client_for_target(target)
        status = await client.get_status()
        payload = {
            "server_id": target.server_id,
            "label": target.label,
            "base_url": target.base_url,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "selected_section": access_context.selected_section,
            "status": self._envelope_payload(status),
            "section_capabilities": access_context.section_capabilities.to_dict(),
            "endpoint_capabilities": dict(access_context.endpoint_capabilities),
            "target_status": access_context.target_status.to_dict(),
        }
        self._browse_cache[cache_key] = payload
        return payload

    async def get_inventory(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.inventory.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        cache_key = self._cache_key(
            section="inventory",
            server_id=target.server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
        )
        cached = self._browse_cache.get(cache_key)
        if cached is not None:
            return cached

        client = self._client_for_target(target)
        api_access_context = self._build_api_access_context(
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
        )
        tools = await client.list_tools(access_context=api_access_context)
        resources = await client.list_resources(access_context=api_access_context)
        prompts = await client.list_prompts(access_context=api_access_context)

        payload = {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "tools": self._response_items(tools),
            "resources": self._response_items(resources),
            "prompts": self._response_items(prompts),
        }
        self._browse_cache[cache_key] = payload
        return payload

    async def get_catalogs(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.catalogs.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        cache_key = self._cache_key(
            section="catalogs",
            server_id=target.server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
        )
        cached = self._browse_cache.get(cache_key)
        if cached is not None:
            return cached

        client = self._client_for_target(target)
        selected_scope = access_context.selected_scope or "personal"
        if selected_scope in {"team", "org", "system_admin"}:
            response = await client.list_scoped_tool_catalogs(
                scope_kind=selected_scope,
                scope_ref=access_context.selected_scope_ref,
            )
        else:
            response = await client.list_visible_tool_catalogs(
                access_context=self._build_api_access_context(
                    selected_scope=access_context.selected_scope,
                    selected_scope_ref=access_context.selected_scope_ref,
                )
            )

        payload = {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "catalogs": self._response_items(response),
            "cache_mode": "stale_allowed",
        }
        self._browse_cache[cache_key] = payload
        return payload

    async def get_external_servers(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.external_servers.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._external_owner_scope(access_context)
        response = await client.list_external_servers(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        return {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "external_servers": self._response_items(response),
            "cache_mode": "live",
        }

    async def get_governance(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(access_context)
        permission_profiles = await client.list_permission_profiles(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        policy_assignments = await client.list_policy_assignments(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        approval_policies = await client.list_approval_policies(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        acp_profiles = await client.list_acp_profiles(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        effective_policy = await client.get_effective_policy(
            **self._effective_policy_scope_filters(access_context)
        )
        return {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "permission_profiles": self._response_items(permission_profiles),
            "policy_assignments": self._response_items(policy_assignments),
            "approval_policies": self._response_items(approval_policies),
            "acp_profiles": self._response_items(acp_profiles),
            "effective_policy": self._envelope_payload(effective_policy),
            "cache_mode": "live",
        }

    async def get_advanced(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        cache_key = self._cache_key(
            section="advanced",
            server_id=target.server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
        )
        cached = self._browse_cache.get(cache_key)
        if cached is not None:
            return cached

        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(access_context)
        capability_scope_type, capability_scope_id = self._capability_mapping_owner_scope(access_context)
        tool_registry_summary = await client.get_tool_registry_summary()
        tool_registry_entries = await client.list_tool_registry_entries()
        tool_registry_modules = await client.list_tool_registry_modules()
        governance_packs = await client.list_governance_packs(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        path_scope_objects = await client.list_path_scope_objects(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        workspace_set_objects = await client.list_workspace_set_objects(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        governance_audit_findings = await client.list_governance_audit_findings(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        capability_mappings = (
            await client.list_capability_mappings(
                owner_scope_type=capability_scope_type,
                owner_scope_id=capability_scope_id,
            )
            if capability_scope_type is not None
            else None
        )
        shared_workspaces = (
            await client.list_shared_workspaces(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
            if owner_scope_type != "user"
            else None
        )
        governance_pack_trust_policy = (
            await client.get_governance_pack_trust_policy()
            if (access_context.selected_scope or "personal") == "system_admin"
            else None
        )
        payload = {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "tool_registry_summary": self._envelope_payload(tool_registry_summary),
            "tool_registry_entries": self._response_items(tool_registry_entries),
            "tool_registry_modules": self._response_items(tool_registry_modules),
            "capability_mappings": self._response_items(capability_mappings) if capability_mappings is not None else [],
            "governance_packs": self._response_items(governance_packs),
            "governance_pack_trust_policy": (
                self._envelope_payload(governance_pack_trust_policy)
                if governance_pack_trust_policy is not None
                else None
            ),
            "path_scope_objects": self._response_items(path_scope_objects),
            "workspace_set_objects": self._response_items(workspace_set_objects),
            "shared_workspaces": self._response_items(shared_workspaces) if shared_workspaces is not None else [],
            "governance_audit_findings": self._envelope_payload(governance_audit_findings),
            "cache_mode": "stale_allowed",
        }
        self._browse_cache[cache_key] = payload
        return payload

    async def create_catalog(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.catalogs.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        created = await client.create_scoped_tool_catalog(
            scope_kind=refreshed_context.selected_scope or "personal",
            scope_ref=refreshed_context.selected_scope_ref,
            request=ScopedToolCatalogCreateRequest(**dict(payload)),
        )
        self.invalidate_cache(
            section="catalogs",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(created)

    async def create_catalog_entry(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        catalog_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.catalogs.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        created = await client.create_scoped_tool_catalog_entry(
            scope_kind=refreshed_context.selected_scope or "personal",
            scope_ref=refreshed_context.selected_scope_ref,
            catalog_id=catalog_id,
            request=ScopedToolCatalogEntryCreateRequest(**dict(payload)),
        )
        self.invalidate_cache(
            section="catalogs",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(created)

    async def delete_catalog(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        catalog_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.catalogs.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_scoped_tool_catalog(
            scope_kind=refreshed_context.selected_scope or "personal",
            scope_ref=refreshed_context.selected_scope_ref,
            catalog_id=catalog_id,
        )
        self.invalidate_cache(
            section="catalogs",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(deleted)

    async def delete_catalog_entry(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        catalog_id: str | int,
        tool_name: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.catalogs.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_scoped_tool_catalog_entry(
            scope_kind=refreshed_context.selected_scope or "personal",
            scope_ref=refreshed_context.selected_scope_ref,
            catalog_id=catalog_id,
            tool_name=tool_name,
        )
        self.invalidate_cache(
            section="catalogs",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(deleted)

    async def create_external_server(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.external_servers.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._external_owner_scope(
            refreshed_context,
            allow_global=True,
        )
        merged_payload = dict(payload)
        if owner_scope_type is not None:
            merged_payload["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            merged_payload["owner_scope_id"] = owner_scope_id
        created = await client.create_external_server(ExternalServerCreateRequest(**merged_payload))
        self.invalidate_cache(
            section="external_servers",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(created)

    async def update_external_server(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.external_servers.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._external_owner_scope(
            refreshed_context,
            allow_global=True,
        )
        merged_payload = dict(payload)
        if owner_scope_type is not None:
            merged_payload["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            merged_payload["owner_scope_id"] = owner_scope_id
        updated = await client.update_external_server(
            server_id=server_id,
            request=ExternalServerUpdateRequest(**merged_payload),
        )
        self.invalidate_cache(
            section="external_servers",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(updated)

    async def delete_external_server(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.external_servers.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_external_server(server_id=server_id)
        self.invalidate_cache(
            section="external_servers",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(deleted)

    async def import_external_server(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.external_servers.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        imported = await client.import_external_server(server_id=server_id)
        self.invalidate_cache(
            section="external_servers",
            server_id=target.server_id,
            selected_scope=refreshed_context.selected_scope,
            selected_scope_ref=refreshed_context.selected_scope_ref,
        )
        return self._envelope_payload(imported)

    async def update_external_server_auth_template(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_external_server_auth_template(
            server_id=server_id,
            request=ExternalServerAuthTemplateUpdateRequest(**dict(payload)),
        )
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def list_external_server_credential_slots(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        response = await client.list_external_server_credential_slots(server_id=server_id)
        return {
            "server_id": target.server_id,
            "selected_scope": access_context.selected_scope,
            "selected_scope_ref": access_context.selected_scope_ref,
            "external_server_id": server_id,
            "credential_slots": self._response_items(response),
            "cache_mode": "live",
        }

    async def create_external_server_credential_slot(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        created = await client.create_external_server_credential_slot(
            server_id=server_id,
            request=ExternalServerCredentialSlotCreateRequest(**dict(payload)),
        )
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(created)

    async def update_external_server_credential_slot(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        slot_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_external_server_credential_slot(
            server_id=server_id,
            slot_name=slot_name,
            request=ExternalServerCredentialSlotUpdateRequest(**dict(payload)),
        )
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def delete_external_server_credential_slot(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_external_server_credential_slot(server_id=server_id, slot_name=slot_name)
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def set_external_server_slot_secret(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        slot_name: str,
        secret: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.set_external_server_slot_secret(
            server_id=server_id,
            slot_name=slot_name,
            request=ExternalSecretSetRequest(secret=secret),
        )
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def clear_external_server_slot_secret(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.clear_external_server_slot_secret(server_id=server_id, slot_name=slot_name)
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def create_permission_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        request_payload = self._strip_scope_fields(payload)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        created = await client.create_permission_profile(
            PermissionProfileCreateRequest(
                **request_payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(created)

    async def update_permission_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_permission_profile(
            profile_id=profile_id,
            request=PermissionProfileUpdateRequest(**self._strip_scope_fields(payload)),
        )
        return self._envelope_payload(updated)

    async def delete_permission_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_permission_profile(profile_id=profile_id)
        return self._envelope_payload(deleted)

    async def create_policy_assignment(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        request_payload = self._strip_scope_fields(payload)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        created = await client.create_policy_assignment(
            PolicyAssignmentCreateRequest(
                **request_payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(created)

    async def update_policy_assignment(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_policy_assignment(
            assignment_id=assignment_id,
            request=PolicyAssignmentUpdateRequest(**self._strip_scope_fields(payload)),
        )
        return self._envelope_payload(updated)

    async def delete_policy_assignment(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_policy_assignment(assignment_id=assignment_id)
        return self._envelope_payload(deleted)

    async def get_policy_assignment_override(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        payload = await client.get_policy_assignment_override(assignment_id=assignment_id)
        return self._envelope_payload(payload)

    async def upsert_policy_assignment_override(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.upsert_policy_assignment_override(
            assignment_id=assignment_id,
            request=PolicyOverrideUpsertRequest(**self._strip_scope_fields(payload)),
        )
        return self._envelope_payload(updated)

    async def delete_policy_assignment_override(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_policy_assignment_override(assignment_id=assignment_id)
        return self._envelope_payload(deleted)

    async def create_approval_policy(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        request_payload = self._strip_scope_fields(payload)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        created = await client.create_approval_policy(
            ApprovalPolicyCreateRequest(
                **request_payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(created)

    async def update_approval_policy(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        approval_policy_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_approval_policy(
            approval_policy_id=approval_policy_id,
            request=ApprovalPolicyUpdateRequest(**self._strip_scope_fields(payload)),
        )
        return self._envelope_payload(updated)

    async def delete_approval_policy(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        approval_policy_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_approval_policy(approval_policy_id=approval_policy_id)
        return self._envelope_payload(deleted)

    async def create_approval_decision(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.approve.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        created = await client.create_approval_decision(
            ApprovalDecisionCreateRequest(**self._strip_scope_fields(payload))
        )
        return self._envelope_payload(created)

    async def create_acp_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        request_payload = self._strip_scope_fields(payload)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        created = await client.create_acp_profile(
            ACPProfileCreateRequest(
                **request_payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(created)

    async def update_acp_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.update_acp_profile(
            profile_id=profile_id,
            request=ACPProfileUpdateRequest(**self._strip_scope_fields(payload)),
        )
        return self._envelope_payload(updated)

    async def delete_acp_profile(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        deleted = await client.delete_acp_profile(profile_id=profile_id)
        return self._envelope_payload(deleted)

    async def get_assignment_external_access(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.effective_access.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        client = self._client_for_target(target)
        payload = await client.get_assignment_external_access(assignment_id=assignment_id)
        return self._envelope_payload(payload)

    async def list_policy_assignment_workspaces(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> list[dict[str, Any]]:
        self._require_allowed(
            action_id="mcp.governance.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=access_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        response = await client.list_policy_assignment_workspaces(assignment_id=assignment_id)
        return self._response_items(response)

    async def add_policy_assignment_workspace(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        workspace_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        created = await client.add_policy_assignment_workspace(
            assignment_id=assignment_id,
            payload={"workspace_id": workspace_id},
        )
        return self._envelope_payload(created)

    async def delete_policy_assignment_workspace(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        workspace_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.governance.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        deleted = await client.delete_policy_assignment_workspace(
            assignment_id=assignment_id,
            workspace_id=workspace_id,
        )
        return self._envelope_payload(deleted)

    async def list_profile_credential_bindings(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
    ) -> list[dict[str, Any]]:
        self._require_allowed(
            action_id="mcp.credentials.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._assert_permission_profile_visible_in_scope(
            target=target,
            access_context=access_context,
            profile_id=profile_id,
        )
        client = self._client_for_target(target)
        response = await client.list_profile_credential_bindings(profile_id=profile_id)
        return self._response_items(response)

    async def upsert_profile_credential_binding(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
        server_id: str,
        payload: dict[str, Any] | None = None,
        slot_name: str | None = None,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_permission_profile_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            profile_id=profile_id,
        )
        client = self._client_for_target(target)
        binding_payload = dict(payload or {})
        if slot_name is not None:
            updated = await client.upsert_profile_slot_credential_binding(
                profile_id=profile_id,
                server_id=server_id,
                slot_name=slot_name,
                payload=binding_payload,
            )
        else:
            updated = await client.upsert_profile_credential_binding(
                profile_id=profile_id,
                server_id=server_id,
                payload=binding_payload,
            )
        return self._envelope_payload(updated)

    async def delete_profile_credential_binding(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
        server_id: str,
        slot_name: str | None = None,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_permission_profile_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            profile_id=profile_id,
        )
        client = self._client_for_target(target)
        if slot_name is not None:
            deleted = await client.delete_profile_slot_credential_binding(
                profile_id=profile_id,
                server_id=server_id,
                slot_name=slot_name,
            )
        else:
            deleted = await client.delete_profile_credential_binding(
                profile_id=profile_id,
                server_id=server_id,
            )
        return self._envelope_payload(deleted)

    async def get_profile_slot_credential_status(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
        server_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._assert_permission_profile_visible_in_scope(
            target=target,
            access_context=access_context,
            profile_id=profile_id,
        )
        client = self._client_for_target(target)
        payload = await client.get_profile_slot_credential_status(
            profile_id=profile_id,
            server_id=server_id,
            slot_name=slot_name,
        )
        return self._envelope_payload(payload)

    async def list_assignment_credential_bindings(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> list[dict[str, Any]]:
        self._require_allowed(
            action_id="mcp.credentials.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=access_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        response = await client.list_assignment_credential_bindings(assignment_id=assignment_id)
        return self._response_items(response)

    async def upsert_assignment_credential_binding(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        server_id: str,
        payload: dict[str, Any],
        slot_name: str | None = None,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        binding_payload = dict(payload)
        if slot_name is not None:
            updated = await client.upsert_assignment_slot_credential_binding(
                assignment_id=assignment_id,
                server_id=server_id,
                slot_name=slot_name,
                payload=binding_payload,
            )
        else:
            updated = await client.upsert_assignment_credential_binding(
                assignment_id=assignment_id,
                server_id=server_id,
                payload=binding_payload,
            )
        return self._envelope_payload(updated)

    async def delete_assignment_credential_binding(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        server_id: str,
        slot_name: str | None = None,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        if slot_name is not None:
            deleted = await client.delete_assignment_slot_credential_binding(
                assignment_id=assignment_id,
                server_id=server_id,
                slot_name=slot_name,
            )
        else:
            deleted = await client.delete_assignment_credential_binding(
                assignment_id=assignment_id,
                server_id=server_id,
            )
        return self._envelope_payload(deleted)

    async def get_assignment_slot_credential_status(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
        server_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.list.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._assert_policy_assignment_visible_in_scope(
            target=target,
            access_context=access_context,
            assignment_id=assignment_id,
        )
        client = self._client_for_target(target)
        payload = await client.get_assignment_slot_credential_status(
            assignment_id=assignment_id,
            server_id=server_id,
            slot_name=slot_name,
        )
        return self._envelope_payload(payload)

    async def set_external_server_secret(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        server_id: str,
        secret: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.credentials.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        updated = await client.set_external_server_secret(
            server_id=server_id,
            request=ExternalSecretSetRequest(secret=secret),
        )
        self.invalidate_cache(section="external_servers", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def update_governance_pack_trust_policy(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        self._require_system_admin_scope(access_context)
        client = self._client_for_target(target)
        updated = await client.update_governance_pack_trust_policy(self._strip_scope_fields(payload))
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def dry_run_governance_pack(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.dry_run_governance_pack(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(result)

    async def prepare_governance_pack_source(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        result = await client.prepare_governance_pack_source(dict(payload))
        return self._envelope_payload(result)

    async def dry_run_governance_pack_source(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.dry_run_governance_pack_source(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(result)

    async def check_governance_pack_updates(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        governance_pack_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=governance_pack_id,
        )
        client = self._client_for_target(target)
        result = await client.check_governance_pack_updates(governance_pack_id=governance_pack_id)
        return self._envelope_payload(result)

    async def prepare_governance_pack_upgrade_candidate(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        governance_pack_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=governance_pack_id,
        )
        client = self._client_for_target(target)
        result = await client.prepare_governance_pack_upgrade_candidate(governance_pack_id=governance_pack_id)
        return self._envelope_payload(result)

    async def dry_run_governance_pack_upgrade(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=self._require_payload_field(payload, "source_governance_pack_id"),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.dry_run_governance_pack_upgrade(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(result)

    async def dry_run_governance_pack_source_upgrade(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.trigger.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=self._require_payload_field(payload, "source_governance_pack_id"),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.dry_run_governance_pack_source_upgrade(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(result)

    async def import_governance_pack(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.import_governance_pack(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(result)

    async def import_governance_pack_source(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.import_governance_pack_source(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(result)

    async def execute_governance_pack_source_upgrade(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=self._require_payload_field(payload, "source_governance_pack_id"),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.execute_governance_pack_source_upgrade(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(result)

    async def execute_governance_pack_upgrade(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=self._require_payload_field(payload, "source_governance_pack_id"),
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        result = await client.execute_governance_pack_upgrade(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(result)

    async def get_governance_pack_detail(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        governance_pack_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=governance_pack_id,
        )
        client = self._client_for_target(target)
        result = await client.get_governance_pack_detail(governance_pack_id=governance_pack_id)
        return self._envelope_payload(result)

    async def list_governance_pack_upgrade_history(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        governance_pack_id: str | int,
    ) -> list[dict[str, Any]]:
        self._require_allowed(
            action_id="mcp.advanced.observe.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        await self._assert_governance_pack_visible_in_scope(
            target=target,
            access_context=refreshed_context,
            governance_pack_id=governance_pack_id,
        )
        client = self._client_for_target(target)
        result = await client.list_governance_pack_upgrade_history(governance_pack_id=governance_pack_id)
        return self._response_items(result)

    async def create_path_scope_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        request_payload = {
            **self._strip_scope_fields(payload),
            "owner_scope_type": owner_scope_type,
            "owner_scope_id": owner_scope_id,
        }
        created = await client.create_path_scope_object(request_payload)
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(created)

    async def preview_capability_mapping(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._capability_mapping_owner_scope(refreshed_context)
        preview = await client.preview_capability_mapping(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        return self._envelope_payload(preview)

    async def create_capability_mapping(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._capability_mapping_owner_scope(refreshed_context)
        created = await client.create_capability_mapping(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(created)

    async def update_capability_mapping(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        capability_adapter_mapping_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._capability_mapping_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=capability_adapter_mapping_id,
            object_label="capability mapping",
            fetcher=lambda: client.list_capability_mappings(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        updated = await client.update_capability_mapping(
            capability_adapter_mapping_id=capability_adapter_mapping_id,
            payload=self._strip_scope_fields(payload),
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def delete_capability_mapping(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        capability_adapter_mapping_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._capability_mapping_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=capability_adapter_mapping_id,
            object_label="capability mapping",
            fetcher=lambda: client.list_capability_mappings(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        deleted = await client.delete_capability_mapping(
            capability_adapter_mapping_id=capability_adapter_mapping_id,
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def update_path_scope_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        path_scope_object_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=path_scope_object_id,
            object_label="path scope object",
            fetcher=lambda: client.list_path_scope_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        updated = await client.update_path_scope_object(
            path_scope_object_id=path_scope_object_id,
            payload=self._strip_scope_fields(payload),
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def delete_path_scope_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        path_scope_object_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=path_scope_object_id,
            object_label="path scope object",
            fetcher=lambda: client.list_path_scope_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        deleted = await client.delete_path_scope_object(path_scope_object_id=path_scope_object_id)
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def create_workspace_set_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        created = await client.create_workspace_set_object(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(created)

    async def update_workspace_set_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        workspace_set_object_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=workspace_set_object_id,
            object_label="workspace set",
            fetcher=lambda: client.list_workspace_set_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        updated = await client.update_workspace_set_object(
            workspace_set_object_id=workspace_set_object_id,
            payload=self._strip_scope_fields(payload),
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def delete_workspace_set_object(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        workspace_set_object_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=workspace_set_object_id,
            object_label="workspace set",
            fetcher=lambda: client.list_workspace_set_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        deleted = await client.delete_workspace_set_object(workspace_set_object_id=workspace_set_object_id)
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def list_workspace_set_members(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        workspace_set_object_id: str | int,
    ) -> list[dict[str, Any]]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=workspace_set_object_id,
            object_label="workspace set",
            fetcher=lambda: client.list_workspace_set_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        members = await client.list_workspace_set_members(workspace_set_object_id=workspace_set_object_id)
        return self._response_items(members)

    async def add_workspace_set_member(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        workspace_set_object_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=workspace_set_object_id,
            object_label="workspace set",
            fetcher=lambda: client.list_workspace_set_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        added = await client.add_workspace_set_member(
            workspace_set_object_id=workspace_set_object_id,
            payload=dict(payload),
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(added)

    async def delete_workspace_set_member(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        workspace_set_object_id: str | int,
        workspace_id: str,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"personal", "team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=workspace_set_object_id,
            object_label="workspace set",
            fetcher=lambda: client.list_workspace_set_objects(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        deleted = await client.delete_workspace_set_member(
            workspace_set_object_id=workspace_set_object_id,
            workspace_id=workspace_id,
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(deleted)

    async def create_shared_workspace(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._shared_workspace_owner_scope(refreshed_context)
        created = await client.create_shared_workspace(
            self._with_owner_scope(
                payload,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(created)

    async def update_shared_workspace(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        shared_workspace_id: str | int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._shared_workspace_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=shared_workspace_id,
            object_label="shared workspace",
            fetcher=lambda: client.list_shared_workspaces(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        updated = await client.update_shared_workspace(
            shared_workspace_id=shared_workspace_id,
            payload=self._strip_scope_fields(payload),
        )
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(updated)

    async def delete_shared_workspace(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        shared_workspace_id: str | int,
    ) -> dict[str, Any]:
        self._require_allowed(
            action_id="mcp.advanced.configure.server",
            runtime_state_override=self._runtime_state_for_target(target, access_context),
        )
        refreshed_context = await self._revalidate_mutation_scope(
            target=target,
            access_context=access_context,
            allowed_scopes={"team", "org", "system_admin"},
        )
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._shared_workspace_owner_scope(refreshed_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=shared_workspace_id,
            object_label="shared workspace",
            fetcher=lambda: client.list_shared_workspaces(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )
        deleted = await client.delete_shared_workspace(shared_workspace_id=shared_workspace_id)
        self.invalidate_cache(section="advanced", server_id=target.server_id)
        return self._envelope_payload(deleted)

    def cache_for(
        self,
        *,
        section: str,
        server_id: str,
        selected_scope: str | None,
        selected_scope_ref: str | None,
    ) -> dict[str, Any] | None:
        return self._browse_cache.get(
            self._cache_key(
                section=section,
                server_id=server_id,
                selected_scope=selected_scope,
                selected_scope_ref=selected_scope_ref,
            )
        )

    def invalidate_cache(
        self,
        *,
        section: str | None = None,
        server_id: str | None = None,
        selected_scope: str | None = None,
        selected_scope_ref: str | None = None,
    ) -> None:
        for key in list(self._browse_cache.keys()):
            key_server_id, key_scope, key_scope_ref, key_section = key
            if section is not None and key_section != section:
                continue
            if server_id is not None and key_server_id != server_id:
                continue
            if selected_scope is not None and key_scope != (selected_scope or "personal"):
                continue
            if selected_scope_ref is not None and key_scope_ref != selected_scope_ref:
                continue
            self._browse_cache.pop(key, None)

    async def _revalidate_mutation_scope(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        allowed_scopes: set[str],
    ) -> ServerAccessContext:
        refreshed_context = await self.resolve_access_context(
            target=target,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
            selected_section=access_context.selected_section,
        )
        if (refreshed_context.selected_scope or "personal") not in allowed_scopes:
            raise ValueError("Selected server scope is not valid for this mutation.")
        return refreshed_context

    async def _bootstrap_access_context(self, client: Any) -> Any:
        method = getattr(client, "bootstrap_access_context", None)
        if callable(method):
            return await method()

        class _FallbackBootstrap:
            manageable_team_ids: list[int] = []
            manageable_org_ids: list[int] = []
            can_use_system_admin_scope: bool = False

        return _FallbackBootstrap()

    async def _probe_section_capabilities(
        self,
        *,
        client: Any,
        access_context: UnifiedMCPAccessContext,
        overview_allowed: bool,
    ) -> tuple[SectionCapabilityFlags, dict[str, bool]]:
        endpoint_capabilities: dict[str, bool] = {}

        async def _probe(endpoint_id: str, method_name: str) -> bool:
            method = getattr(client, method_name, None)
            if not callable(method):
                endpoint_capabilities[endpoint_id] = False
                return False
            try:
                await method(access_context=access_context)
            except Exception as exc:
                endpoint_capabilities[endpoint_id] = False
                logger.debug("Unified MCP endpoint probe {} failed: {}", endpoint_id, exc)
                return False
            endpoint_capabilities[endpoint_id] = True
            return True

        inventory_tools = await _probe("tools.list", "list_tools")
        inventory_resources = await _probe("resources.list", "list_resources")
        inventory_prompts = await _probe("prompts.list", "list_prompts")
        catalogs = await _probe("catalogs.list", "list_visible_tool_catalogs")
        governance = await _probe("governance.list", "list_permission_profiles")
        external_servers = await _probe("external_servers.list", "list_external_servers")
        advanced = await _probe("advanced.list", "get_tool_registry_summary")
        return (
            SectionCapabilityFlags(
                overview=overview_allowed,
                inventory=inventory_tools or inventory_resources or inventory_prompts,
                catalogs=catalogs,
                external_servers=external_servers,
                governance=governance,
                advanced=advanced,
            ),
            endpoint_capabilities,
        )

    def _client_for_target(self, target: ConfiguredServerTarget) -> Any:
        if self.client_factory is not None:
            client = self._client_cache.get(target.server_id)
            if client is None:
                client = self.client_factory(target)
                self._client_cache[target.server_id] = client
            return client
        if self.client is None:
            raise ValueError("A Unified MCP server client or client_factory is required.")
        return self.client

    def _normalize_scope_selection(
        self,
        *,
        selected_scope: str | None,
        selected_scope_ref: str | None,
        manageable_team_ids: list[int],
        manageable_org_ids: list[int],
        can_use_system_admin_scope: bool,
    ) -> tuple[str, str | None]:
        normalized_scope = str(selected_scope or "personal").strip() or "personal"
        normalized_scope_ref = str(selected_scope_ref).strip() if selected_scope_ref not in (None, "") else None
        valid_team_ids = {str(item) for item in manageable_team_ids}
        valid_org_ids = {str(item) for item in manageable_org_ids}

        if normalized_scope == "team":
            if normalized_scope_ref in valid_team_ids:
                return "team", normalized_scope_ref
            return "personal", None
        if normalized_scope == "org":
            if normalized_scope_ref in valid_org_ids:
                return "org", normalized_scope_ref
            return "personal", None
        if normalized_scope == "system_admin":
            if can_use_system_admin_scope:
                return "system_admin", None
            return "personal", None
        return "personal", None

    def _normalize_section_selection(
        self,
        *,
        selected_section: str | None,
        section_capabilities: SectionCapabilityFlags,
    ) -> str | None:
        available_sections = [
            section
            for section, allowed in section_capabilities.to_dict().items()
            if allowed
        ]
        if selected_section in available_sections:
            return selected_section
        if "overview" in available_sections:
            return "overview"
        return available_sections[0] if available_sections else None

    def _build_api_access_context(
        self,
        *,
        selected_scope: str | None,
        selected_scope_ref: str | None,
    ) -> UnifiedMCPAccessContext:
        scope_kind = selected_scope or "personal"
        payload: dict[str, Any] = {"scope_kind": scope_kind}
        if scope_kind in {"team", "org"} and selected_scope_ref:
            payload["scope_ref"] = selected_scope_ref
            if scope_kind == "team":
                payload["team_id"] = selected_scope_ref
            else:
                payload["org_id"] = selected_scope_ref
        return UnifiedMCPAccessContext(**payload)

    @staticmethod
    def _external_owner_scope(
        access_context: ServerAccessContext,
        *,
        allow_global: bool = False,
    ) -> tuple[str | None, int | None]:
        selected_scope = access_context.selected_scope or "personal"
        scope_ref = access_context.selected_scope_ref
        if selected_scope == "team" and scope_ref not in (None, ""):
            return "team", int(scope_ref)
        if selected_scope == "org" and scope_ref not in (None, ""):
            return "org", int(scope_ref)
        if selected_scope == "system_admin" and allow_global:
            return "global", None
        return None, None

    @staticmethod
    def _governance_owner_scope(
        access_context: ServerAccessContext,
    ) -> tuple[str, int | None]:
        selected_scope = access_context.selected_scope or "personal"
        scope_ref = access_context.selected_scope_ref
        if selected_scope == "team" and scope_ref not in (None, ""):
            return "team", int(scope_ref)
        if selected_scope == "org" and scope_ref not in (None, ""):
            return "org", int(scope_ref)
        if selected_scope == "system_admin":
            return "global", None
        if access_context.principal_user_id is not None:
            return "user", int(access_context.principal_user_id)
        return "user", None

    @staticmethod
    def _capability_mapping_owner_scope(
        access_context: ServerAccessContext,
    ) -> tuple[str | None, int | None]:
        selected_scope = access_context.selected_scope or "personal"
        scope_ref = access_context.selected_scope_ref
        if selected_scope == "team" and scope_ref not in (None, ""):
            return "team", int(scope_ref)
        if selected_scope == "org" and scope_ref not in (None, ""):
            return "org", int(scope_ref)
        if selected_scope == "system_admin":
            return "global", None
        return None, None

    @staticmethod
    def _shared_workspace_owner_scope(
        access_context: ServerAccessContext,
    ) -> tuple[str, int | None]:
        owner_scope_type, owner_scope_id = ServerUnifiedMCPService._governance_owner_scope(access_context)
        if owner_scope_type == "user":
            raise ValueError("Selected server scope is not valid for this advanced mutation.")
        return owner_scope_type, owner_scope_id

    @staticmethod
    def _effective_policy_scope_filters(access_context: ServerAccessContext) -> dict[str, int]:
        selected_scope = access_context.selected_scope or "personal"
        scope_ref = access_context.selected_scope_ref
        if selected_scope == "team" and scope_ref not in (None, ""):
            return {"team_id": int(scope_ref)}
        if selected_scope == "org" and scope_ref not in (None, ""):
            return {"org_id": int(scope_ref)}
        return {}

    @staticmethod
    def _strip_scope_fields(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(payload).items()
            if key not in {"owner_scope_type", "owner_scope_id"}
        }

    @classmethod
    def _with_owner_scope(
        cls,
        payload: dict[str, Any],
        *,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> dict[str, Any]:
        merged_payload = cls._strip_scope_fields(payload)
        merged_payload["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            merged_payload["owner_scope_id"] = owner_scope_id
        return merged_payload

    async def _assert_advanced_object_visible_in_scope(
        self,
        *,
        object_id: str | int,
        object_label: str,
        fetcher: Callable[[], Any],
    ) -> None:
        response = await fetcher()
        items = self._response_items(response)
        wanted_id = str(object_id)
        for item in items:
            if isinstance(item, dict) and str(item.get("id")) == wanted_id:
                return
        raise ValueError(f"Selected server scope cannot manage the requested {object_label}.")

    async def _assert_governance_pack_visible_in_scope(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        governance_pack_id: str | int,
    ) -> None:
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(access_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=governance_pack_id,
            object_label="governance pack",
            fetcher=lambda: client.list_governance_packs(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )

    async def _assert_permission_profile_visible_in_scope(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        profile_id: str | int,
    ) -> None:
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(access_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=profile_id,
            object_label="permission profile",
            fetcher=lambda: client.list_permission_profiles(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )

    async def _assert_policy_assignment_visible_in_scope(
        self,
        *,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
        assignment_id: str | int,
    ) -> None:
        client = self._client_for_target(target)
        owner_scope_type, owner_scope_id = self._governance_owner_scope(access_context)
        await self._assert_advanced_object_visible_in_scope(
            object_id=assignment_id,
            object_label="policy assignment",
            fetcher=lambda: client.list_policy_assignments(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            ),
        )

    @staticmethod
    def _require_payload_field(payload: dict[str, Any], field_name: str) -> Any:
        value = payload.get(field_name)
        if value in (None, ""):
            raise ValueError(f"Unified MCP action requires '{field_name}'.")
        return value

    @staticmethod
    def _require_system_admin_scope(access_context: ServerAccessContext) -> None:
        if (access_context.selected_scope or "personal") != "system_admin":
            raise ValueError("Selected server scope is not valid for this advanced mutation.")

    def _build_target_status(
        self,
        *,
        target: ConfiguredServerTarget,
        status_error: Exception | None,
    ) -> TargetStatusMetadata:
        now = datetime.now(timezone.utc)
        if status_error is None:
            return TargetStatusMetadata(
                last_known_server_label=target.label,
                last_known_reachability="reachable",
                last_known_auth_state="authenticated",
                last_connected_at=now,
                updated_at=now,
            )

        classification = classify_backend_exception(status_error)
        reachability = "unknown"
        auth_state = "unknown"
        if classification == "server_unreachable":
            reachability = "unreachable"
        elif classification == "server_auth_required":
            reachability = "reachable"
            auth_state = "auth_required"
        elif classification == "server_session_invalid":
            reachability = "reachable"
            auth_state = "session_invalid"

        return TargetStatusMetadata(
            last_known_server_label=target.label,
            last_known_reachability=reachability,
            last_known_auth_state=auth_state,
            updated_at=now,
        )

    def _persist_target_status(
        self,
        target: ConfiguredServerTarget,
        status: TargetStatusMetadata,
    ) -> TargetStatusMetadata:
        if self.target_store is None:
            return status
        try:
            updated = self.target_store.update_target_status(
                target.server_id,
                last_known_server_label=status.last_known_server_label,
                last_known_reachability=status.last_known_reachability,
                last_known_auth_state=status.last_known_auth_state,
                last_connected_at=status.last_connected_at,
                updated_at=status.updated_at,
            )
        except Exception:
            return status
        return TargetStatusMetadata(
            last_known_server_label=updated.last_known_server_label,
            last_known_reachability=updated.last_known_reachability,
            last_known_auth_state=updated.last_known_auth_state,
            last_connected_at=updated.last_connected_at,
            updated_at=updated.updated_at,
        )

    def _runtime_state_for_target(
        self,
        target: ConfiguredServerTarget,
        access_context: ServerAccessContext,
    ) -> RuntimeSourceState:
        target_status = access_context.target_status
        return RuntimeSourceState(
            active_source="server",
            active_server_id=target.server_id,
            server_configured=True,
            server_reachability=target_status.last_known_reachability or "reachable",
            server_auth_state=target_status.last_known_auth_state or "authenticated",
            last_known_server_label=target_status.last_known_server_label or target.label,
        )

    def _require_allowed(
        self,
        *,
        action_id: str,
        runtime_state_override: RuntimeSourceState,
    ) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(
            action_id=action_id,
            runtime_state_override=runtime_state_override,
        )

    @staticmethod
    def _envelope_payload(response: Any) -> dict[str, Any]:
        payload = getattr(response, "payload", None)
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(response, dict):
            return dict(response)
        return {}

    @staticmethod
    def _response_items(response: Any) -> list[dict[str, Any]]:
        def _normalize_item(item: Any) -> dict[str, Any]:
            if isinstance(item, dict):
                return dict(item)
            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump()
                if isinstance(dumped, dict):
                    return dict(dumped)
            item_dict = getattr(item, "__dict__", None)
            if isinstance(item_dict, dict):
                return {
                    str(key): value
                    for key, value in item_dict.items()
                    if not str(key).startswith("_")
                }
            return {"value": item}

        items = getattr(response, "items", None)
        if isinstance(items, list):
            return [_normalize_item(item) for item in items]
        if isinstance(response, list):
            return [_normalize_item(item) for item in response]
        payload = getattr(response, "payload", None)
        if isinstance(payload, dict):
            for key in ("items", "tools", "resources", "prompts", "catalogs"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [_normalize_item(item) for item in value]
        if isinstance(response, dict):
            for key in ("items", "tools", "resources", "prompts", "catalogs"):
                value = response.get(key)
                if isinstance(value, list):
                    return [_normalize_item(item) for item in value]
        return []

    @staticmethod
    def _cache_key(
        *,
        section: str,
        server_id: str,
        selected_scope: str | None,
        selected_scope_ref: str | None,
    ) -> tuple[str, str, str | None, str]:
        return (
            server_id,
            selected_scope or "personal",
            selected_scope_ref,
            section,
        )
