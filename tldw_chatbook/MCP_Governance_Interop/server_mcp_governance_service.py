"""Policy-gated REST client for server-side Unified MCP governance."""

from __future__ import annotations

from typing import Any, Mapping, Optional, TypeVar

from pydantic import BaseModel

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    MCPApprovalDecisionCreate,
    MCPApprovalPolicyCreate,
    MCPApprovalPolicyUpdate,
    MCPCapabilityMappingCreate,
    MCPCapabilityMappingUpdate,
    MCPCatalogCreate,
    MCPCatalogEntryCreate,
    MCPExternalServerCreate,
    MCPExternalServerUpdate,
    MCPPermissionProfileCreate,
    MCPPermissionProfileUpdate,
    MCPPolicyAssignmentCreate,
    MCPPolicyAssignmentUpdate,
    MCPSecretSetRequest,
    TLDWAPIClient,
)

TModel = TypeVar("TModel", bound=BaseModel)


class ServerMCPGovernanceService:
    """Expose server MCP Hub/catalog administration without using the MCP SDK."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerMCPGovernanceService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerMCPGovernanceService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server MCP governance operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server MCP governance action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerMCPGovernanceService._dump(item) for item in response]
        if isinstance(response, dict):
            return response
        return dict(response or {})

    @staticmethod
    def _model(request_data: TModel | dict[str, Any], model_type: type[TModel]) -> TModel:
        if isinstance(request_data, model_type):
            return request_data
        return model_type.model_validate(request_data)

    async def list_tool_registry(self) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.tool_registry.list.server")
        return self._dump(await self._require_client().list_mcp_tool_registry())

    async def list_tool_registry_modules(self) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.tool_registry.list.server")
        return self._dump(await self._require_client().list_mcp_tool_registry_modules())

    async def get_tool_registry_summary(self) -> dict[str, Any]:
        self._enforce("mcp.governance.tool_registry.detail.server")
        return self._dump(await self._require_client().get_mcp_tool_registry_summary())

    async def list_capability_mappings(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.capability_mappings.list.server")
        return self._dump(
            await self._require_client().list_mcp_capability_mappings(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )

    async def preview_capability_mapping(self, request_data: MCPCapabilityMappingCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.capability_mappings.preview.server")
        request = self._model(request_data, MCPCapabilityMappingCreate)
        return self._dump(await self._require_client().preview_mcp_capability_mapping(request))

    async def create_capability_mapping(self, request_data: MCPCapabilityMappingCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.capability_mappings.create.server")
        request = self._model(request_data, MCPCapabilityMappingCreate)
        return self._dump(await self._require_client().create_mcp_capability_mapping(request))

    async def update_capability_mapping(
        self,
        capability_adapter_mapping_id: int,
        request_data: MCPCapabilityMappingUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.capability_mappings.update.server")
        request = self._model(request_data, MCPCapabilityMappingUpdate)
        return self._dump(await self._require_client().update_mcp_capability_mapping(capability_adapter_mapping_id, request))

    async def delete_capability_mapping(self, capability_adapter_mapping_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.capability_mappings.delete.server")
        return self._dump(await self._require_client().delete_mcp_capability_mapping(capability_adapter_mapping_id))

    async def list_external_servers(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.external_servers.list.server")
        return self._dump(
            await self._require_client().list_mcp_external_servers(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )

    async def create_external_server(self, request_data: MCPExternalServerCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.external_servers.create.server")
        request = self._model(request_data, MCPExternalServerCreate)
        return self._dump(await self._require_client().create_mcp_external_server(request))

    async def import_external_server(self, server_id: str) -> dict[str, Any]:
        self._enforce("mcp.governance.external_servers.create.server")
        return self._dump(await self._require_client().import_mcp_external_server(server_id))

    async def update_external_server(
        self,
        server_id: str,
        request_data: MCPExternalServerUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.external_servers.update.server")
        request = self._model(request_data, MCPExternalServerUpdate)
        return self._dump(await self._require_client().update_mcp_external_server(server_id, request))

    async def delete_external_server(self, server_id: str) -> dict[str, Any]:
        self._enforce("mcp.governance.external_servers.delete.server")
        return self._dump(await self._require_client().delete_mcp_external_server(server_id))

    async def set_external_server_secret(self, server_id: str, request_data: MCPSecretSetRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.external_servers.secrets.update.server")
        request = self._model(request_data, MCPSecretSetRequest)
        return self._dump(await self._require_client().set_mcp_external_server_secret(server_id, request))

    async def list_permission_profiles(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.permission_profiles.list.server")
        return self._dump(
            await self._require_client().list_mcp_permission_profiles(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )

    async def create_permission_profile(self, request_data: MCPPermissionProfileCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.permission_profiles.create.server")
        request = self._model(request_data, MCPPermissionProfileCreate)
        return self._dump(await self._require_client().create_mcp_permission_profile(request))

    async def update_permission_profile(
        self,
        profile_id: int,
        request_data: MCPPermissionProfileUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.permission_profiles.update.server")
        request = self._model(request_data, MCPPermissionProfileUpdate)
        return self._dump(await self._require_client().update_mcp_permission_profile(profile_id, request))

    async def delete_permission_profile(self, profile_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.permission_profiles.delete.server")
        return self._dump(await self._require_client().delete_mcp_permission_profile(profile_id))

    async def list_policy_assignments(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.policy_assignments.list.server")
        return self._dump(await self._require_client().list_mcp_policy_assignments(**kwargs))

    async def create_policy_assignment(self, request_data: MCPPolicyAssignmentCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.policy_assignments.create.server")
        request = self._model(request_data, MCPPolicyAssignmentCreate)
        return self._dump(await self._require_client().create_mcp_policy_assignment(request))

    async def update_policy_assignment(
        self,
        assignment_id: int,
        request_data: MCPPolicyAssignmentUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.policy_assignments.update.server")
        request = self._model(request_data, MCPPolicyAssignmentUpdate)
        return self._dump(await self._require_client().update_mcp_policy_assignment(assignment_id, request))

    async def delete_policy_assignment(self, assignment_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.policy_assignments.delete.server")
        return self._dump(await self._require_client().delete_mcp_policy_assignment(assignment_id))

    async def list_approval_policies(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.approval_policies.list.server")
        return self._dump(
            await self._require_client().list_mcp_approval_policies(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
        )

    async def create_approval_policy(self, request_data: MCPApprovalPolicyCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.approval_policies.create.server")
        request = self._model(request_data, MCPApprovalPolicyCreate)
        return self._dump(await self._require_client().create_mcp_approval_policy(request))

    async def update_approval_policy(
        self,
        approval_policy_id: int,
        request_data: MCPApprovalPolicyUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.approval_policies.update.server")
        request = self._model(request_data, MCPApprovalPolicyUpdate)
        return self._dump(await self._require_client().update_mcp_approval_policy(approval_policy_id, request))

    async def delete_approval_policy(self, approval_policy_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.approval_policies.delete.server")
        return self._dump(await self._require_client().delete_mcp_approval_policy(approval_policy_id))

    async def create_approval_decision(self, request_data: MCPApprovalDecisionCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.approval_decisions.approve.server")
        request = self._model(request_data, MCPApprovalDecisionCreate)
        return self._dump(await self._require_client().create_mcp_approval_decision(request))

    async def get_effective_policy(
        self,
        *,
        persona_id: str | None = None,
        group_id: str | None = None,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.effective_policy.detail.server")
        return self._dump(
            await self._require_client().get_mcp_effective_policy(
                persona_id=persona_id,
                group_id=group_id,
                org_id=org_id,
                team_id=team_id,
            )
        )

    async def list_org_tool_catalogs(self, *, org_id: int) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.catalogs.list.server")
        return self._dump(await self._require_client().list_mcp_org_tool_catalogs(org_id))

    async def create_org_tool_catalog(self, *, org_id: int, request_data: MCPCatalogCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.catalogs.create.server")
        request = self._model(request_data, MCPCatalogCreate)
        return self._dump(await self._require_client().create_mcp_org_tool_catalog(org_id, request))

    async def delete_org_tool_catalog(self, *, org_id: int, catalog_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.catalogs.delete.server")
        return self._dump(await self._require_client().delete_mcp_org_tool_catalog(org_id, catalog_id))

    async def add_org_catalog_entry(
        self,
        *,
        org_id: int,
        catalog_id: int,
        request_data: MCPCatalogEntryCreate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.catalog_entries.create.server")
        request = self._model(request_data, MCPCatalogEntryCreate)
        return self._dump(await self._require_client().add_mcp_org_catalog_entry(org_id, catalog_id, request))

    async def delete_org_catalog_entry(self, *, org_id: int, catalog_id: int, tool_name: str) -> dict[str, Any]:
        self._enforce("mcp.governance.catalog_entries.delete.server")
        return self._dump(await self._require_client().delete_mcp_org_catalog_entry(org_id, catalog_id, tool_name))

    async def list_team_tool_catalogs(self, *, team_id: int) -> list[dict[str, Any]]:
        self._enforce("mcp.governance.catalogs.list.server")
        return self._dump(await self._require_client().list_mcp_team_tool_catalogs(team_id))

    async def create_team_tool_catalog(self, *, team_id: int, request_data: MCPCatalogCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("mcp.governance.catalogs.create.server")
        request = self._model(request_data, MCPCatalogCreate)
        return self._dump(await self._require_client().create_mcp_team_tool_catalog(team_id, request))

    async def delete_team_tool_catalog(self, *, team_id: int, catalog_id: int) -> dict[str, Any]:
        self._enforce("mcp.governance.catalogs.delete.server")
        return self._dump(await self._require_client().delete_mcp_team_tool_catalog(team_id, catalog_id))

    async def add_team_catalog_entry(
        self,
        *,
        team_id: int,
        catalog_id: int,
        request_data: MCPCatalogEntryCreate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("mcp.governance.catalog_entries.create.server")
        request = self._model(request_data, MCPCatalogEntryCreate)
        return self._dump(await self._require_client().add_mcp_team_catalog_entry(team_id, catalog_id, request))

    async def delete_team_catalog_entry(self, *, team_id: int, catalog_id: int, tool_name: str) -> dict[str, Any]:
        self._enforce("mcp.governance.catalog_entries.delete.server")
        return self._dump(await self._require_client().delete_mcp_team_catalog_entry(team_id, catalog_id, tool_name))
