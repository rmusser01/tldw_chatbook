"""Flexible schemas for the server Unified MCP governance REST control plane."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MCPScopeType = Literal["global", "org", "team", "user"]
MCPApprovalDecision = Literal["approved", "denied"]
MCPApprovalDuration = Literal["once", "session", "conversation"]


class MCPFlexibleModel(BaseModel):
    """Tolerate server-side contract growth while preserving common fields."""

    model_config = ConfigDict(extra="allow")


class MCPGovernanceObject(MCPFlexibleModel):
    id: int | None = None
    name: str | None = None
    server_id: str | None = None
    owner_scope_type: str | None = None
    owner_scope_id: int | None = None
    is_active: bool | None = None
    enabled: bool | None = None
    policy_document: dict[str, Any] | None = None
    inline_policy_document: dict[str, Any] | None = None
    mode: str | None = None
    mapping_id: str | None = None
    normalized_mapping: dict[str, Any] | None = None
    org_id: int | None = None
    team_id: int | None = None
    tool_name: str | None = None
    module_id: str | None = None
    module: str | None = None
    tool_count: int | None = None
    secret_set: bool | None = None
    decision: str | None = None
    duration: str | None = None


class MCPGovernanceSummary(MCPFlexibleModel):
    entries: list[MCPGovernanceObject] = Field(default_factory=list)
    modules: list[MCPGovernanceObject] = Field(default_factory=list)


class MCPGovernanceEvent(MCPFlexibleModel):
    event_id: str
    event_type: str
    action: str | None = None
    source: str | None = None
    actor_id: int | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _unwrap_sse_frame(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = value.get("data")
        if isinstance(data, dict):
            merged = dict(data)
            if value.get("event_id") is not None:
                merged.setdefault("event_id", value.get("event_id"))
            if value.get("event") is not None:
                merged.setdefault("event_type", value.get("event"))
            return merged
        return value


class MCPEffectivePolicyResponse(MCPFlexibleModel):
    policy: dict[str, Any] = Field(default_factory=dict)
    provenance: list[Any] = Field(default_factory=list)


class MCPExternalServerCreate(MCPFlexibleModel):
    server_id: str
    name: str
    transport: str
    config: dict[str, Any] = Field(default_factory=dict)
    owner_scope_type: MCPScopeType = "global"
    owner_scope_id: int | None = None
    enabled: bool = True


class MCPExternalServerUpdate(MCPFlexibleModel):
    name: str | None = None
    transport: str | None = None
    config: dict[str, Any] | None = None
    owner_scope_type: MCPScopeType | None = None
    owner_scope_id: int | None = None
    enabled: bool | None = None


class MCPSecretSetRequest(MCPFlexibleModel):
    secret: str


class MCPPermissionProfileCreate(MCPFlexibleModel):
    name: str
    description: str | None = None
    owner_scope_type: MCPScopeType = "global"
    owner_scope_id: int | None = None
    mode: str = "custom"
    path_scope_object_id: int | None = None
    policy_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class MCPPermissionProfileUpdate(MCPFlexibleModel):
    name: str | None = None
    description: str | None = None
    owner_scope_type: MCPScopeType | None = None
    owner_scope_id: int | None = None
    mode: str | None = None
    path_scope_object_id: int | None = None
    policy_document: dict[str, Any] | None = None
    is_active: bool | None = None


class MCPPolicyAssignmentCreate(MCPFlexibleModel):
    target_type: str
    target_id: str | None = None
    owner_scope_type: MCPScopeType = "global"
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None
    workspace_source_mode: str | None = None
    workspace_set_object_id: int | None = None
    inline_policy_document: dict[str, Any] = Field(default_factory=dict)
    approval_policy_id: int | None = None
    is_active: bool = True


class MCPPolicyAssignmentUpdate(MCPFlexibleModel):
    target_type: str | None = None
    target_id: str | None = None
    owner_scope_type: MCPScopeType | None = None
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None
    workspace_source_mode: str | None = None
    workspace_set_object_id: int | None = None
    inline_policy_document: dict[str, Any] | None = None
    approval_policy_id: int | None = None
    is_active: bool | None = None


class MCPApprovalPolicyCreate(MCPFlexibleModel):
    name: str
    description: str | None = None
    owner_scope_type: MCPScopeType = "global"
    owner_scope_id: int | None = None
    mode: str
    rules: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class MCPApprovalPolicyUpdate(MCPFlexibleModel):
    name: str | None = None
    description: str | None = None
    owner_scope_type: MCPScopeType | None = None
    owner_scope_id: int | None = None
    mode: str | None = None
    rules: dict[str, Any] | None = None
    is_active: bool | None = None


class MCPApprovalDecisionCreate(MCPFlexibleModel):
    approval_policy_id: int | None = None
    context_key: str
    conversation_id: str | None = None
    tool_name: str
    scope_key: str = "default"
    decision: MCPApprovalDecision
    duration: MCPApprovalDuration = "once"


class MCPCapabilityMappingCreate(MCPFlexibleModel):
    mapping_id: str
    title: str | None = None
    description: str | None = None
    owner_scope_type: Literal["global", "org", "team"] = "global"
    owner_scope_id: int | None = None
    capability_name: str
    adapter_contract_version: int = 1
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    is_active: bool = True


class MCPCapabilityMappingUpdate(MCPFlexibleModel):
    mapping_id: str | None = None
    title: str | None = None
    description: str | None = None
    owner_scope_type: Literal["global", "org", "team"] | None = None
    owner_scope_id: int | None = None
    capability_name: str | None = None
    adapter_contract_version: int | None = None
    resolved_policy_document: dict[str, Any] | None = None
    supported_environment_requirements: list[str] | None = None
    is_active: bool | None = None


class MCPCatalogCreate(MCPFlexibleModel):
    name: str
    description: str | None = None
    is_active: bool | None = True


class MCPCatalogEntryCreate(MCPFlexibleModel):
    tool_name: str
    module_id: str | None = None


__all__ = [
    "MCPApprovalDecision",
    "MCPApprovalDecisionCreate",
    "MCPApprovalDuration",
    "MCPApprovalPolicyCreate",
    "MCPApprovalPolicyUpdate",
    "MCPCapabilityMappingCreate",
    "MCPCapabilityMappingUpdate",
    "MCPCatalogCreate",
    "MCPCatalogEntryCreate",
    "MCPEffectivePolicyResponse",
    "MCPExternalServerCreate",
    "MCPExternalServerUpdate",
    "MCPFlexibleModel",
    "MCPGovernanceEvent",
    "MCPGovernanceObject",
    "MCPGovernanceSummary",
    "MCPPermissionProfileCreate",
    "MCPPermissionProfileUpdate",
    "MCPPolicyAssignmentCreate",
    "MCPPolicyAssignmentUpdate",
    "MCPScopeType",
    "MCPSecretSetRequest",
]
