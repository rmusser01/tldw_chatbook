from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .mcp_unified_schemas import (
    CatalogConnectionTestRequest,
    MCPAccessBootstrapPrincipal,
    MCPAccessBootstrapResponse,
    MCPCatalogConnectionTestResponse,
    MCPExecuteToolResponse,
    MCPHealthResponse,
    MCPMetricsResponse,
    MCPModuleHealthResponse,
    MCPModulesResponse,
    MCPPromptsResponse,
    MCPResourcesResponse,
    MCPServerScopeKind,
    MCPStatusResponse,
    MCPToolCatalogsResponse,
    MCPToolsResponse,
    MCPUserProfileIdentity,
    MCPUserProfileMemberships,
    MCPUserProfileResponse,
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
