from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConsentRecordResponse(BaseModel):
    id: int | None = None
    user_id: int
    purpose: str
    granted_at: datetime | None = None
    withdrawn_at: datetime | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: str | None = None

    model_config = ConfigDict(extra="allow")


class ConsentPreferencesResponse(BaseModel):
    user_id: int
    consents: list[ConsentRecordResponse]


class PrivilegeDependency(BaseModel):
    id: str
    type: str = "dependency"
    module: str | None = None

    model_config = ConfigDict(extra="allow")


class PrivilegeRecommendedAction(BaseModel):
    privilege_scope_id: str | None = None
    action: str
    reason: str | None = None

    model_config = ConfigDict(extra="allow")


class PrivilegeSelfItem(BaseModel):
    endpoint: str
    method: str
    privilege_scope_id: str
    feature_flag_id: str | None = None
    sensitivity_tier: str | None = None
    ownership_predicates: list[str] = Field(default_factory=list)
    status: Literal["allowed", "blocked"]
    blocked_reason: str | None = None
    dependencies: list[PrivilegeDependency] = Field(default_factory=list)
    dependency_sources: list[str] = Field(default_factory=list)
    rate_limit_class: str | None = None
    rate_limit_resources: list[str] = Field(default_factory=list)
    source_module: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class PrivilegeSelfResponse(BaseModel):
    catalog_version: str
    generated_at: datetime
    items: list[PrivilegeSelfItem]
    recommended_actions: list[PrivilegeRecommendedAction] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class PrivilegeDetailItem(BaseModel):
    user_id: str
    user_name: str
    role: str
    endpoint: str
    method: str
    privilege_scope_id: str
    feature_flag_id: str | None = None
    sensitivity_tier: str
    ownership_predicates: list[str] = Field(default_factory=list)
    status: Literal["allowed", "blocked"]
    blocked_reason: str | None = None
    dependencies: list[PrivilegeDependency] = Field(default_factory=list)
    dependency_sources: list[str] = Field(default_factory=list)
    rate_limit_class: str | None = None
    rate_limit_resources: list[str] = Field(default_factory=list)
    source_module: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class PrivilegeDetailResponse(BaseModel):
    catalog_version: str
    generated_at: datetime
    page: int
    page_size: int
    total_items: int
    items: list[PrivilegeDetailItem]
    recommended_actions: list[PrivilegeRecommendedAction] | None = None

    model_config = ConfigDict(extra="allow")
