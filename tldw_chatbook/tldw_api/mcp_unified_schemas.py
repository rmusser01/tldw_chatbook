from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

MCPServerScopeKind = Literal["personal", "team", "org", "system_admin"]
MCPHubOwnerScopeType = Literal["global", "org", "team", "user"]
MCPMembershipRole = Literal["owner", "admin", "lead", "member", "viewer", "contributor"]
MCPPermissionProfileMode = Literal["preset", "custom"]
MCPAssignmentTargetType = Literal["default", "group", "persona"]
MCPApprovalMode = Literal[
    "allow_silently",
    "ask_every_time",
    "ask_outside_profile",
    "ask_on_sensitive_actions",
    "temporary_elevation_allowed",
]
MCPApprovalDecisionType = Literal["approved", "denied"]
MCPApprovalDuration = Literal["once", "session", "conversation"]


class UnifiedMCPAccessContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: MCPServerScopeKind = "personal"
    scope_ref: Optional[str] = None
    principal_id: Optional[str] = None
    team_id: Optional[str] = None
    org_id: Optional[str] = None


class ToolExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class CatalogConnectionTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    auth_type: str = "none"
    secret: Optional[str] = None
    auth_key_name: Optional[str] = None


class ScopedToolCatalogCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    is_active: bool = True


class ScopedToolCatalogEntryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    module_id: Optional[str] = None


class ExternalServerCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    name: str
    transport: str
    config: Dict[str, Any] = Field(default_factory=dict)
    owner_scope_type: Optional[str] = None
    owner_scope_id: Optional[int] = None
    enabled: bool = True


class ExternalServerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    transport: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    owner_scope_type: Optional[str] = None
    owner_scope_id: Optional[int] = None
    enabled: Optional[bool] = None


class ExternalServerAuthTemplateUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    mappings: List[Dict[str, Any]] = Field(default_factory=list)


class ExternalServerCredentialSlotCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_name: str
    display_name: str
    secret_kind: str
    privilege_class: str
    is_required: bool = False


class ExternalServerCredentialSlotUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: Optional[str] = None
    secret_kind: Optional[str] = None
    privilege_class: Optional[str] = None
    is_required: Optional[bool] = None


class ExternalSecretSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secret: str


class PermissionProfileCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    owner_scope_type: MCPHubOwnerScopeType = "user"
    owner_scope_id: Optional[int] = None
    mode: MCPPermissionProfileMode = "custom"
    path_scope_object_id: Optional[int] = None
    policy_document: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class PermissionProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    owner_scope_type: Optional[MCPHubOwnerScopeType] = None
    owner_scope_id: Optional[int] = None
    mode: Optional[MCPPermissionProfileMode] = None
    path_scope_object_id: Optional[int] = None
    policy_document: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class PolicyAssignmentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: MCPAssignmentTargetType
    target_id: Optional[str] = None
    owner_scope_type: MCPHubOwnerScopeType = "user"
    owner_scope_id: Optional[int] = None
    profile_id: Optional[int] = None
    path_scope_object_id: Optional[int] = None
    workspace_source_mode: Optional[str] = None
    workspace_set_object_id: Optional[int] = None
    inline_policy_document: Dict[str, Any] = Field(default_factory=dict)
    approval_policy_id: Optional[int] = None
    is_active: bool = True


class PolicyAssignmentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: Optional[MCPAssignmentTargetType] = None
    target_id: Optional[str] = None
    owner_scope_type: Optional[MCPHubOwnerScopeType] = None
    owner_scope_id: Optional[int] = None
    profile_id: Optional[int] = None
    path_scope_object_id: Optional[int] = None
    workspace_source_mode: Optional[str] = None
    workspace_set_object_id: Optional[int] = None
    inline_policy_document: Optional[Dict[str, Any]] = None
    approval_policy_id: Optional[int] = None
    is_active: Optional[bool] = None


class PolicyOverrideUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override_policy_document: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ApprovalPolicyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    owner_scope_type: MCPHubOwnerScopeType = "user"
    owner_scope_id: Optional[int] = None
    mode: MCPApprovalMode
    rules: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ApprovalPolicyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    owner_scope_type: Optional[MCPHubOwnerScopeType] = None
    owner_scope_id: Optional[int] = None
    mode: Optional[MCPApprovalMode] = None
    rules: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ApprovalDecisionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_policy_id: Optional[int] = None
    context_key: str
    conversation_id: Optional[str] = None
    tool_name: str
    scope_key: str
    decision: MCPApprovalDecisionType
    duration: MCPApprovalDuration = "once"


class ACPProfileCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    owner_scope_type: MCPHubOwnerScopeType = "user"
    owner_scope_id: Optional[int] = None
    profile: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ACPProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    owner_scope_type: Optional[MCPHubOwnerScopeType] = None
    owner_scope_id: Optional[int] = None
    profile: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class MCPUserProfileIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    uuid: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_locked: Optional[bool] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class MCPUserProfileOrgMembership(BaseModel):
    model_config = ConfigDict(extra="ignore")

    org_id: int
    role: Optional[str] = None


class MCPUserProfileTeamMembership(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: int
    role: Optional[str] = None
    org_id: Optional[int] = None


class MCPUserProfileMemberships(BaseModel):
    model_config = ConfigDict(extra="ignore")

    orgs: list[MCPUserProfileOrgMembership] = Field(default_factory=list)
    teams: list[MCPUserProfileTeamMembership] = Field(default_factory=list)


class MCPUserProfileResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_version: Optional[datetime] = None
    catalog_version: Optional[str] = None
    user: Optional[MCPUserProfileIdentity] = None
    memberships: Optional[MCPUserProfileMemberships] = None


class MCPAccessBootstrapPrincipal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None
    is_admin: bool = False


class MCPAccessBootstrapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_version: Optional[datetime] = None
    catalog_version: Optional[str] = None
    principal: Optional[MCPAccessBootstrapPrincipal] = None
    manageable_team_ids: list[int] = Field(default_factory=list)
    manageable_org_ids: list[int] = Field(default_factory=list)
    can_use_system_admin_scope: bool = False
    profile: Optional[MCPUserProfileResponse] = None


class MCPPayloadEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    payload: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "MCPPayloadEnvelope":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            return cls(payload=dict(payload))
        if isinstance(payload, list):
            return cls(payload={"items": list(payload)})
        return cls(payload={"value": payload})


class MCPListEnvelope(MCPPayloadEnvelope):
    items: List[Any] = Field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any) -> "MCPListEnvelope":
        base = super().from_payload(payload)
        items: List[Any] = []
        if isinstance(payload, list):
            items = list(payload)
        elif isinstance(payload, dict):
            for key in ("items", "tools", "modules", "resources", "prompts", "catalogs", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    items = list(value)
                    break
        return cls(payload=base.payload, items=items)


class MCPStatusResponse(MCPPayloadEnvelope):
    status: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Any) -> "MCPStatusResponse":
        base = super().from_payload(payload)
        status = None
        if isinstance(payload, dict):
            raw_status = payload.get("status")
            status = raw_status if isinstance(raw_status, str) else None
        return cls(payload=base.payload, status=status)


class MCPHealthResponse(MCPPayloadEnvelope):
    healthy: Optional[bool] = None
    status: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Any) -> "MCPHealthResponse":
        base = super().from_payload(payload)
        healthy = None
        status = None
        if isinstance(payload, dict):
            raw_healthy = payload.get("healthy")
            if isinstance(raw_healthy, bool):
                healthy = raw_healthy
            raw_status = payload.get("status")
            status = raw_status if isinstance(raw_status, str) else None
        return cls(payload=base.payload, healthy=healthy, status=status)


class MCPMetricsResponse(MCPPayloadEnvelope):
    connections: Dict[str, Any] = Field(default_factory=dict)
    modules: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "MCPMetricsResponse":
        base = super().from_payload(payload)
        connections: Dict[str, Any] = {}
        modules: Dict[str, Any] = {}
        if isinstance(payload, dict):
            raw_connections = payload.get("connections")
            if isinstance(raw_connections, dict):
                connections = dict(raw_connections)
            raw_modules = payload.get("modules")
            if isinstance(raw_modules, dict):
                modules = dict(raw_modules)
        return cls(payload=base.payload, connections=connections, modules=modules)


class MCPModulesResponse(MCPListEnvelope):
    pass


class MCPToolsResponse(MCPListEnvelope):
    pass


class MCPResourcesResponse(MCPListEnvelope):
    pass


class MCPPromptsResponse(MCPListEnvelope):
    pass


class MCPToolCatalogsResponse(MCPListEnvelope):
    pass


class MCPModuleHealthResponse(MCPListEnvelope):
    pass


class MCPExecuteToolResponse(MCPPayloadEnvelope):
    pass


class MCPCatalogConnectionTestResponse(MCPPayloadEnvelope):
    pass
