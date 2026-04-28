from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .mcp_unified_schemas import (
    ACPProfileCreateRequest,
    ACPProfileUpdateRequest,
    ApprovalDecisionCreateRequest,
    ApprovalPolicyCreateRequest,
    ApprovalPolicyUpdateRequest,
    CatalogConnectionTestRequest,
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
    MCPAccessBootstrapPrincipal,
    MCPAccessBootstrapResponse,
    MCPCatalogConnectionTestResponse,
    MCPExecuteToolResponse,
    MCPHealthResponse,
    MCPListEnvelope,
    MCPMetricsResponse,
    MCPModuleHealthResponse,
    MCPModulesResponse,
    MCPPayloadEnvelope,
    MCPPromptsResponse,
    MCPResourcesResponse,
    MCPHubOwnerScopeType,
    MCPServerScopeKind,
    MCPStatusResponse,
    MCPToolCatalogsResponse,
    MCPToolsResponse,
    MCPUserProfileIdentity,
    MCPUserProfileMemberships,
    MCPUserProfileResponse,
    ScopedToolCatalogCreateRequest,
    ScopedToolCatalogEntryCreateRequest,
    ToolExecutionRequest,
    UnifiedMCPAccessContext,
)

_MANAGER_LIKE_ROLES = {"owner", "admin", "lead"}


def _coerce_identifier(value: Union[str, int, None]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_entry_value(entry: Any, key: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _coerce_manageable_ids(values: Any, *, id_field: str) -> list[int]:
    if not isinstance(values, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for entry in values:
        role = str(_get_entry_value(entry, "role") or "").strip().lower()
        if role not in _MANAGER_LIKE_ROLES:
            continue
        raw_value = _get_entry_value(entry, id_field)
        if isinstance(raw_value, int) and raw_value not in seen:
            seen.add(raw_value)
            out.append(raw_value)
    return out


class MCPUnifiedClient:
    def __init__(self, root_client: Any):
        self.root_client = root_client

    def normalize_access_context(
        self,
        *,
        scope_kind: Optional[MCPServerScopeKind] = None,
        scope_ref: Union[str, int, None] = None,
        principal_id: Union[str, int, None] = None,
        team_id: Union[str, int, None] = None,
        org_id: Union[str, int, None] = None,
    ) -> UnifiedMCPAccessContext:
        normalized_principal_id = _coerce_identifier(principal_id)
        normalized_team_id = _coerce_identifier(team_id)
        normalized_org_id = _coerce_identifier(org_id)
        normalized_scope_ref = _coerce_identifier(scope_ref)

        effective_scope_kind = scope_kind
        if effective_scope_kind is None:
            if normalized_team_id is not None:
                effective_scope_kind = "team"
            elif normalized_org_id is not None:
                effective_scope_kind = "org"
            else:
                effective_scope_kind = "personal"

        if effective_scope_kind in ("personal", "system_admin"):
            normalized_scope_ref = None
        elif effective_scope_kind == "team":
            normalized_scope_ref = normalized_scope_ref or normalized_team_id
            if normalized_scope_ref is None:
                raise ValueError("team scope requires a team identifier")
        elif effective_scope_kind == "org":
            normalized_scope_ref = normalized_scope_ref or normalized_org_id
            if normalized_scope_ref is None:
                raise ValueError("org scope requires an org identifier")

        return UnifiedMCPAccessContext(
            scope_kind=effective_scope_kind,
            scope_ref=normalized_scope_ref,
            principal_id=normalized_principal_id,
            team_id=normalized_team_id,
            org_id=normalized_org_id,
        )

    def build_access_context_params(
        self,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        **scope_kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if access_context is None:
            if not scope_kwargs:
                return None
            access_context = self.normalize_access_context(**scope_kwargs)
        params = access_context.model_dump(exclude_none=True)
        return params or None

    def _merge_params(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> Optional[Dict[str, Any]]:
        params = dict(base_params or {})
        context_params = self.build_access_context_params(access_context)
        if context_params is not None:
            params.update(context_params)
        return params or None

    async def get_status(self) -> MCPStatusResponse:
        payload = await self.root_client._request("GET", "/api/v1/mcp/status", params=None)
        return MCPStatusResponse.from_payload(payload)

    async def get_health(self) -> MCPHealthResponse:
        payload = await self.root_client._request("GET", "/api/v1/mcp/health", params=None)
        return MCPHealthResponse.from_payload(payload)

    async def get_metrics(self) -> MCPMetricsResponse:
        payload = await self.root_client._request("GET", "/api/v1/mcp/metrics", params=None)
        return MCPMetricsResponse.from_payload(payload)

    async def get_prometheus_metrics(self) -> str:
        request_bytes = getattr(self.root_client, "_request_bytes", None)
        if callable(request_bytes):
            payload = await request_bytes("GET", "/api/v1/mcp/metrics/prometheus")
            return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        payload = await self.root_client._request("GET", "/api/v1/mcp/metrics/prometheus", params=None)
        return payload if isinstance(payload, str) else str(payload)

    async def create_auth_token(
        self,
        *,
        username: str,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> MCPPayloadEnvelope:
        json_data = {"username": username}
        if password is not None:
            json_data["password"] = password
        if api_key is not None:
            json_data["api_key"] = api_key
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/auth/token",
            json_data=json_data,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def refresh_auth_token(
        self,
        *,
        refresh_token: str,
        token_id: Optional[str] = None,
    ) -> MCPPayloadEnvelope:
        json_data = {"refresh_token": refresh_token}
        if token_id is not None:
            json_data["token_id"] = token_id
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/auth/refresh",
            json_data=json_data,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def send_request(
        self,
        request: Dict[str, Any],
        *,
        client_id: Optional[str] = None,
        mcp_session_id: Optional[str] = None,
        config: Optional[str] = None,
    ) -> MCPPayloadEnvelope:
        params: Dict[str, Any] = {}
        if client_id is not None:
            params["client_id"] = client_id
        if config is not None:
            params["config"] = config
        headers = {"mcp-session-id": mcp_session_id} if mcp_session_id is not None else None
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/request",
            json_data=dict(request),
            params=params or None,
            headers=headers,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def send_request_batch(
        self,
        requests: List[Dict[str, Any]],
        *,
        client_id: Optional[str] = None,
        mcp_session_id: Optional[str] = None,
        config: Optional[str] = None,
    ) -> MCPListEnvelope:
        params: Dict[str, Any] = {}
        if client_id is not None:
            params["client_id"] = client_id
        if config is not None:
            params["config"] = config
        headers = {"mcp-session-id": mcp_session_id} if mcp_session_id is not None else None
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/request/batch",
            json_data=[dict(item) for item in requests],
            params=params or None,
            headers=headers,
        )
        return MCPListEnvelope.from_payload(payload)

    async def list_catalog(self, *, archetype_key: Optional[str] = None) -> MCPListEnvelope:
        params = {"archetype_key": archetype_key} if archetype_key is not None else None
        payload = await self.root_client._request("GET", "/api/v1/mcp/catalog", params=params)
        return MCPListEnvelope.from_payload(payload)

    async def list_modules(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPModulesResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/modules",
            params=self.build_access_context_params(access_context),
        )
        return MCPModulesResponse.from_payload(payload)

    async def get_module_health(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPModuleHealthResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/modules/health",
            params=self.build_access_context_params(access_context),
        )
        return MCPModuleHealthResponse.from_payload(payload)

    async def list_tools(
        self,
        *,
        catalog: Optional[str] = None,
        catalog_id: Optional[int] = None,
        module: Optional[Union[str, List[str]]] = None,
        catalog_strict: bool = False,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPToolsResponse:
        params: Dict[str, Any] = {}
        if catalog is not None:
            params["catalog"] = catalog
        if catalog_id is not None:
            params["catalog_id"] = catalog_id
        if module is not None:
            params["module"] = module
        if catalog_strict:
            params["catalog_strict"] = "1"
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/tools",
            params=self._merge_params(params, access_context),
        )
        return MCPToolsResponse.from_payload(payload)

    async def list_resources(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPResourcesResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/resources",
            params=self.build_access_context_params(access_context),
        )
        return MCPResourcesResponse.from_payload(payload)

    async def list_prompts(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPPromptsResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/prompts",
            params=self.build_access_context_params(access_context),
        )
        return MCPPromptsResponse.from_payload(payload)

    async def execute_tool(
        self,
        *,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPExecuteToolResponse:
        request = ToolExecutionRequest(tool_name=tool_name, arguments=arguments or {})
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/tools/execute",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPExecuteToolResponse.from_payload(payload)

    async def list_visible_tool_catalogs(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
    ) -> MCPToolCatalogsResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/tool_catalogs",
            params=self.build_access_context_params(access_context),
        )
        return MCPToolCatalogsResponse.from_payload(payload)

    async def test_catalog_connection(
        self,
        request: CatalogConnectionTestRequest,
    ) -> MCPCatalogConnectionTestResponse:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/catalog/test-connection",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPCatalogConnectionTestResponse.from_payload(payload)

    async def list_scoped_tool_catalogs(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
    ) -> MCPToolCatalogsResponse:
        path = self._catalog_scope_path(scope_kind=scope_kind, scope_ref=scope_ref)
        payload = await self.root_client._request("GET", path, params=None)
        return MCPToolCatalogsResponse.from_payload(payload)

    async def create_scoped_tool_catalog(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
        request: ScopedToolCatalogCreateRequest,
    ) -> MCPPayloadEnvelope:
        path = self._catalog_scope_path(scope_kind=scope_kind, scope_ref=scope_ref)
        payload = await self.root_client._request(
            "POST",
            path,
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_scoped_tool_catalog(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
        catalog_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        path = f"{self._catalog_scope_path(scope_kind=scope_kind, scope_ref=scope_ref)}/{catalog_id}"
        payload = await self.root_client._request("DELETE", path, params=None)
        return MCPPayloadEnvelope.from_payload(payload)

    async def create_scoped_tool_catalog_entry(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
        catalog_id: Union[str, int],
        request: ScopedToolCatalogEntryCreateRequest,
    ) -> MCPPayloadEnvelope:
        path = f"{self._catalog_scope_path(scope_kind=scope_kind, scope_ref=scope_ref)}/{catalog_id}/entries"
        payload = await self.root_client._request(
            "POST",
            path,
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_scoped_tool_catalog_entry(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
        catalog_id: Union[str, int],
        tool_name: str,
    ) -> MCPPayloadEnvelope:
        path = (
            f"{self._catalog_scope_path(scope_kind=scope_kind, scope_ref=scope_ref)}"
            f"/{catalog_id}/entries/{tool_name}"
        )
        payload = await self.root_client._request("DELETE", path, params=None)
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_external_servers(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        owner_scope_type: Optional[str] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        if access_context is not None and owner_scope_type is None:
            owner_scope_type, owner_scope_id = self._owner_scope_from_access_context(access_context)
        params: Dict[str, Any] = {}
        if owner_scope_type is not None:
            params["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            params["owner_scope_id"] = owner_scope_id
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/external-servers",
            params=params or None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_external_server(
        self,
        request: ExternalServerCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/external-servers",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_external_server(
        self,
        *,
        server_id: str,
        request: ExternalServerUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/external-servers/{server_id}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_external_server(self, *, server_id: str) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/external-servers/{server_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def import_external_server(self, *, server_id: str) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{server_id}/import",
            json_data=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_external_server_auth_template(self, *, server_id: str) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/external-servers/{server_id}/auth-template",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_external_server_auth_template(
        self,
        *,
        server_id: str,
        request: ExternalServerAuthTemplateUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/external-servers/{server_id}/auth-template",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_external_server_credential_slots(self, *, server_id: str) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_external_server_credential_slot(
        self,
        *,
        server_id: str,
        request: ExternalServerCredentialSlotCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        request: ExternalServerCredentialSlotUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots/{slot_name}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots/{slot_name}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def set_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
        request: ExternalSecretSetRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots/{slot_name}/secret",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def set_external_server_secret(
        self,
        *,
        server_id: str,
        request: ExternalSecretSetRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/external-servers/{server_id}/secret",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def clear_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/external-servers/{server_id}/credential-slots/{slot_name}/secret",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_permission_profiles(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        owner_scope_type, owner_scope_id = self._resolve_owner_scope_filters(
            access_context=access_context,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/permission-profiles",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_permission_profile(
        self,
        request: PermissionProfileCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/permission-profiles",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_permission_profile(
        self,
        *,
        profile_id: Union[str, int],
        request: PermissionProfileUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_permission_profile(self, *, profile_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_policy_assignments(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> MCPListEnvelope:
        owner_scope_type, owner_scope_id = self._resolve_owner_scope_filters(
            access_context=access_context,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        params = self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id) or {}
        if target_type is not None:
            params["target_type"] = target_type
        if target_id is not None:
            params["target_id"] = target_id
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/policy-assignments",
            params=params or None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_policy_assignment(
        self,
        request: PolicyAssignmentCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/policy-assignments",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_policy_assignment(
        self,
        *,
        assignment_id: Union[str, int],
        request: PolicyAssignmentUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_policy_assignment(self, *, assignment_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_policy_assignment_override(self, *, assignment_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/override",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def upsert_policy_assignment_override(
        self,
        *,
        assignment_id: Union[str, int],
        request: PolicyOverrideUpsertRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/override",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_policy_assignment_override(self, *, assignment_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/override",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_approval_policies(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        owner_scope_type, owner_scope_id = self._resolve_owner_scope_filters(
            access_context=access_context,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/approval-policies",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_approval_policy(
        self,
        request: ApprovalPolicyCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/approval-policies",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_approval_policy(
        self,
        *,
        approval_policy_id: Union[str, int],
        request: ApprovalPolicyUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/approval-policies/{approval_policy_id}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_approval_policy(self, *, approval_policy_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/approval-policies/{approval_policy_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def create_approval_decision(
        self,
        request: ApprovalDecisionCreateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/approval-decisions",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_effective_policy(
        self,
        *,
        persona_id: Optional[str] = None,
        group_id: Optional[str] = None,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> MCPPayloadEnvelope:
        params: Dict[str, Any] = {}
        if persona_id is not None:
            params["persona_id"] = persona_id
        if group_id is not None:
            params["group_id"] = group_id
        if org_id is not None:
            params["org_id"] = org_id
        if team_id is not None:
            params["team_id"] = team_id
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/effective-policy",
            params=params or None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_acp_profiles(
        self,
        *,
        access_context: Optional[UnifiedMCPAccessContext] = None,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        owner_scope_type, owner_scope_id = self._resolve_owner_scope_filters(
            access_context=access_context,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/acp-profiles",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_acp_profile(self, request: ACPProfileCreateRequest) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/acp-profiles",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_acp_profile(
        self,
        *,
        profile_id: Union[str, int],
        request: ACPProfileUpdateRequest,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/acp-profiles/{profile_id}",
            json_data=request.model_dump(exclude_none=True),
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def delete_acp_profile(self, *, profile_id: Union[str, int]) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/acp-profiles/{profile_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_assignment_external_access(
        self,
        *,
        assignment_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/external-access",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_policy_assignment_workspaces(
        self,
        *,
        assignment_id: Union[str, int],
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/workspaces",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def add_policy_assignment_workspace(
        self,
        *,
        assignment_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/workspaces",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_policy_assignment_workspace(
        self,
        *,
        assignment_id: Union[str, int],
        workspace_id: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/workspaces/{workspace_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def get_tool_registry_summary(self) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/tool-registry/summary",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_tool_registry_entries(self) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/tool-registry",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def list_tool_registry_modules(self) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/tool-registry/modules",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def list_capability_mappings(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/capability-mappings",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def preview_capability_mapping(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/capability-mappings/preview",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def create_capability_mapping(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/capability-mappings",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def update_capability_mapping(
        self,
        *,
        capability_adapter_mapping_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/capability-mappings/{capability_adapter_mapping_id}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_capability_mapping(
        self,
        *,
        capability_adapter_mapping_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/capability-mappings/{capability_adapter_mapping_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_governance_packs(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/governance-packs",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def dry_run_governance_pack(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/dry-run",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def prepare_governance_pack_source(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/source/prepare",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def dry_run_governance_pack_source(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/source/dry-run",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def check_governance_pack_updates(
        self,
        *,
        governance_pack_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/governance-packs/{governance_pack_id}/check-updates",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def prepare_governance_pack_upgrade_candidate(
        self,
        *,
        governance_pack_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/governance-packs/{governance_pack_id}/prepare-upgrade-candidate",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def dry_run_governance_pack_upgrade(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/dry-run-upgrade",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def dry_run_governance_pack_source_upgrade(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/source/dry-run-upgrade",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def import_governance_pack(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/import",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def execute_governance_pack_source_upgrade(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/source/execute-upgrade",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def import_governance_pack_source(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/source/import",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def execute_governance_pack_upgrade(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/governance-packs/execute-upgrade",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def get_governance_pack_trust_policy(self) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/governance-packs/trust-policy",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_governance_pack_upgrade_history(
        self,
        *,
        governance_pack_id: Union[str, int],
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/governance-packs/{governance_pack_id}/upgrade-history",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def get_governance_pack_detail(
        self,
        *,
        governance_pack_id: Union[str, int],
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/governance-packs/{governance_pack_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def update_governance_pack_trust_policy(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            "/api/v1/mcp/hub/governance-packs/trust-policy",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_path_scope_objects(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/path-scope-objects",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_path_scope_object(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/path-scope-objects",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def update_path_scope_object(
        self,
        *,
        path_scope_object_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/path-scope-objects/{path_scope_object_id}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_path_scope_object(self, *, path_scope_object_id: Union[str, int]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/path-scope-objects/{path_scope_object_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_workspace_set_objects(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/workspace-set-objects",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_workspace_set_object(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/workspace-set-objects",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def update_workspace_set_object(
        self,
        *,
        workspace_set_object_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/workspace-set-objects/{workspace_set_object_id}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_workspace_set_object(self, *, workspace_set_object_id: Union[str, int]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/workspace-set-objects/{workspace_set_object_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_workspace_set_members(
        self,
        *,
        workspace_set_object_id: Union[str, int],
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/workspace-set-objects/{workspace_set_object_id}/members",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def add_workspace_set_member(
        self,
        *,
        workspace_set_object_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            f"/api/v1/mcp/hub/workspace-set-objects/{workspace_set_object_id}/members",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_workspace_set_member(
        self,
        *,
        workspace_set_object_id: Union[str, int],
        workspace_id: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/workspace-set-objects/{workspace_set_object_id}/members/{workspace_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_shared_workspaces(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/shared-workspaces",
            params=self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id),
        )
        return MCPListEnvelope.from_payload(payload)

    async def create_shared_workspace(self, payload: dict[str, Any]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "POST",
            "/api/v1/mcp/hub/shared-workspaces",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def update_shared_workspace(
        self,
        *,
        shared_workspace_id: Union[str, int],
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/shared-workspaces/{shared_workspace_id}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_shared_workspace(self, *, shared_workspace_id: Union[str, int]) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/shared-workspaces/{shared_workspace_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def list_governance_audit_findings(
        self,
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType] = None,
        owner_scope_id: Optional[int] = None,
        severity: Optional[str] = None,
        finding_type: Optional[str] = None,
        object_kind: Optional[str] = None,
        scope_type: Optional[str] = None,
    ) -> MCPPayloadEnvelope:
        params = self._owner_scope_params(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id) or {}
        if severity is not None:
            params["severity"] = severity
        if finding_type is not None:
            params["finding_type"] = finding_type
        if object_kind is not None:
            params["object_kind"] = object_kind
        if scope_type is not None:
            params["scope_type"] = scope_type
        payload = await self.root_client._request(
            "GET",
            "/api/v1/mcp/hub/audit/findings",
            params=params or None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_profile_credential_bindings(
        self,
        *,
        profile_id: Union[str, int],
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def upsert_profile_credential_binding(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
        payload: dict[str, Any] | None = None,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/{server_id}",
            json_data=dict(payload or {}),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def upsert_profile_slot_credential_binding(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
        slot_name: str,
        payload: dict[str, Any] | None = None,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}",
            json_data=dict(payload or {}),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_profile_credential_binding(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/{server_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_profile_slot_credential_binding(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def get_profile_slot_credential_status(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/status/{server_id}/{slot_name}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_profile_slot_credential_binding_status(
        self,
        *,
        profile_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}/status",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def list_assignment_credential_bindings(
        self,
        *,
        assignment_id: Union[str, int],
    ) -> MCPListEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings",
            params=None,
        )
        return MCPListEnvelope.from_payload(payload)

    async def upsert_assignment_credential_binding(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/{server_id}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def upsert_assignment_slot_credential_binding(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
        slot_name: str,
        payload: dict[str, Any],
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "PUT",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}",
            json_data=dict(payload),
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_assignment_credential_binding(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/{server_id}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def delete_assignment_slot_credential_binding(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        response = await self.root_client._request(
            "DELETE",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(response)

    async def get_assignment_slot_credential_status(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/status/{server_id}/{slot_name}",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_assignment_slot_credential_binding_status(
        self,
        *,
        assignment_id: Union[str, int],
        server_id: str,
        slot_name: str,
    ) -> MCPPayloadEnvelope:
        payload = await self.root_client._request(
            "GET",
            f"/api/v1/mcp/hub/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}/status",
            params=None,
        )
        return MCPPayloadEnvelope.from_payload(payload)

    async def get_current_user_profile(self) -> MCPUserProfileResponse:
        payload = await self.root_client._request(
            "GET",
            "/api/v1/users/me/profile",
            params={"sections": "user,memberships"},
        )
        return MCPUserProfileResponse.model_validate(payload)

    async def bootstrap_access_context(self) -> MCPAccessBootstrapResponse:
        profile = await self.get_current_user_profile()
        user = profile.user or MCPUserProfileIdentity()
        memberships = profile.memberships or MCPUserProfileMemberships()
        manageable_org_ids = _coerce_manageable_ids(memberships.orgs, id_field="org_id")
        manageable_team_ids = _coerce_manageable_ids(memberships.teams, id_field="team_id")
        principal = MCPAccessBootstrapPrincipal(
            user_id=user.id,
            username=user.username,
            role=user.role,
            is_admin=(user.role or "").strip().lower() == "admin",
        )
        return MCPAccessBootstrapResponse(
            profile_version=profile.profile_version,
            catalog_version=profile.catalog_version,
            principal=principal,
            manageable_team_ids=manageable_team_ids,
            manageable_org_ids=manageable_org_ids,
            can_use_system_admin_scope=principal.is_admin,
            profile=profile,
        )

    def _catalog_scope_path(
        self,
        *,
        scope_kind: MCPServerScopeKind,
        scope_ref: Union[str, int, None] = None,
    ) -> str:
        if scope_kind == "team":
            normalized_scope_ref = _coerce_identifier(scope_ref)
            if normalized_scope_ref is None:
                raise ValueError("team-scoped catalog operations require a team identifier")
            return f"/api/v1/teams/{normalized_scope_ref}/mcp/tool_catalogs"
        if scope_kind == "org":
            normalized_scope_ref = _coerce_identifier(scope_ref)
            if normalized_scope_ref is None:
                raise ValueError("org-scoped catalog operations require an org identifier")
            return f"/api/v1/orgs/{normalized_scope_ref}/mcp/tool_catalogs"
        if scope_kind == "system_admin":
            return "/api/v1/admin/mcp/tool_catalogs"
        raise ValueError(f"Unsupported scoped catalog scope: {scope_kind}")

    @staticmethod
    def _owner_scope_from_access_context(
        access_context: UnifiedMCPAccessContext,
    ) -> tuple[Optional[str], Optional[int]]:
        scope_kind = str(access_context.scope_kind or "personal").strip()
        scope_ref = _coerce_identifier(access_context.scope_ref)
        if scope_kind == "team" and scope_ref is not None:
            return "team", int(scope_ref)
        if scope_kind == "org" and scope_ref is not None:
            return "org", int(scope_ref)
        if scope_kind == "system_admin":
            return "global", None
        return "user", None

    @classmethod
    def _resolve_owner_scope_filters(
        cls,
        *,
        access_context: Optional[UnifiedMCPAccessContext],
        owner_scope_type: Optional[MCPHubOwnerScopeType],
        owner_scope_id: Optional[int],
    ) -> tuple[Optional[MCPHubOwnerScopeType], Optional[int]]:
        if access_context is not None and owner_scope_type is None:
            derived_scope_type, derived_scope_id = cls._owner_scope_from_access_context(access_context)
            return derived_scope_type, derived_scope_id
        return owner_scope_type, owner_scope_id

    @staticmethod
    def _owner_scope_params(
        *,
        owner_scope_type: Optional[MCPHubOwnerScopeType],
        owner_scope_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if owner_scope_type is not None:
            params["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            params["owner_scope_id"] = owner_scope_id
        return params or None
