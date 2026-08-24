"""Authority-qualified domain contracts for the Research Workspace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Generic, Protocol, TypeVar


class WorkspaceDataSource(StrEnum):
    LOCAL = "local"
    SERVER = "server"


_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)^bearer\s+\S+"),
    re.compile(r"(?i)^sk-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]"),
)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _optional_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    return value.strip()


def _safe_identity(value: object, field_name: str) -> str:
    normalized = _optional_text(value, field_name)
    if any(pattern.search(normalized) for pattern in _SECRET_VALUE_PATTERNS):
        raise ValueError(f"{field_name} must contain identity, not secret material")
    return normalized


@dataclass(frozen=True, slots=True)
class QualifiedWorkspaceRef:
    data_source: WorkspaceDataSource
    workspace_id: str
    server_profile_id: str = ""
    principal_id: str = ""

    def __post_init__(self) -> None:
        try:
            data_source = WorkspaceDataSource(self.data_source)
        except (TypeError, ValueError):
            raise ValueError("data_source must be local or server") from None
        workspace_id = _required_text(self.workspace_id, "workspace_id")
        server_profile_id = _safe_identity(
            self.server_profile_id, "server_profile_id"
        )
        principal_id = _safe_identity(self.principal_id, "principal_id")
        if data_source is WorkspaceDataSource.SERVER and not server_profile_id:
            raise ValueError("server_profile_id is required for Server workspace refs")
        if data_source is WorkspaceDataSource.LOCAL and (
            server_profile_id or principal_id
        ):
            raise ValueError("Local workspace refs cannot carry server identity metadata")
        object.__setattr__(self, "data_source", data_source)
        object.__setattr__(self, "workspace_id", workspace_id)
        object.__setattr__(self, "server_profile_id", server_profile_id)
        object.__setattr__(self, "principal_id", principal_id)


@dataclass(frozen=True, slots=True)
class ResearchCapability:
    available: bool
    reason_code: str
    user_message: str
    owner: str
    recovery_action: str = ""
    capability_revision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reason_code", _required_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "user_message", _required_text(self.user_message, "user_message")
        )
        object.__setattr__(self, "owner", _required_text(self.owner, "owner"))
        object.__setattr__(
            self,
            "recovery_action",
            _optional_text(self.recovery_action, "recovery_action"),
        )
        object.__setattr__(
            self,
            "capability_revision",
            _optional_text(self.capability_revision, "capability_revision"),
        )


class CapabilityUnavailableError(RuntimeError):
    """Raised when the selected authority cannot perform an operation."""

    def __init__(self, capability: ResearchCapability) -> None:
        super().__init__(capability.user_message)
        self.capability = capability


def require_capability(
    capabilities: Mapping[str, ResearchCapability], capability_name: str
) -> ResearchCapability:
    """Return an available capability or fail closed with its exact contract."""

    capability = capabilities.get(capability_name)
    if capability is None:
        capability = ResearchCapability(
            available=False,
            reason_code="unknown_capability",
            user_message=(
                "This action is unavailable because its capability is unknown."
            ),
            owner="research_workspace",
            recovery_action="Refresh capabilities or choose another action.",
        )
    if not capability.available:
        raise CapabilityUnavailableError(capability)
    return capability


@dataclass(frozen=True, slots=True)
class ResearchWorkspaceSummary:
    ref: QualifiedWorkspaceRef
    name: str
    description: str = ""
    archived: bool = False
    version: int | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(
            self, "description", _optional_text(self.description, "description")
        )
        object.__setattr__(
            self, "updated_at", _optional_text(self.updated_at, "updated_at")
        )
        if self.version is not None and (
            type(self.version) is not int or self.version < 0
        ):
            raise ValueError("version must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class ResearchSourceSummary:
    ref: QualifiedWorkspaceRef
    source_id: str
    title: str
    source_type: str
    ready: bool = False
    version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(
            self, "source_type", _required_text(self.source_type, "source_type")
        )
        if self.version is not None and (
            type(self.version) is not int or self.version < 0
        ):
            raise ValueError("version must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class ProcessingRoute:
    data_source: WorkspaceDataSource
    processor: str
    provider: str = ""
    model: str = ""
    remote: bool = False

    def __post_init__(self) -> None:
        try:
            data_source = WorkspaceDataSource(self.data_source)
        except (TypeError, ValueError):
            raise ValueError("data_source must be local or server") from None
        object.__setattr__(self, "data_source", data_source)
        object.__setattr__(self, "processor", _required_text(self.processor, "processor"))
        object.__setattr__(self, "provider", _optional_text(self.provider, "provider"))
        object.__setattr__(self, "model", _optional_text(self.model, "model"))


PageItem = TypeVar("PageItem")


@dataclass(frozen=True, slots=True)
class BoundedPageResult(Generic[PageItem]):
    items: tuple[PageItem, ...]
    limit: int
    offset: int = 0
    total: int | None = None
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if len(self.items) > self.limit:
            raise ValueError("page contains more items than limit")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if self.total is not None and (
            type(self.total) is not int or self.total < 0
        ):
            raise ValueError("total must be a non-negative integer or None")


class ResearchWorkspacePort(Protocol):
    async def list_workspaces(
        self, *, include_archived: bool = False
    ) -> tuple[ResearchWorkspaceSummary, ...]: ...

    async def get_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None: ...

    async def create_workspace(
        self, *, name: str, description: str = "", template_id: str = ""
    ) -> ResearchWorkspaceSummary: ...

    async def update_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        name: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary: ...

    async def duplicate_workspace(
        self, ref: QualifiedWorkspaceRef, *, name: str
    ) -> ResearchWorkspaceSummary: ...

    async def archive_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary: ...

    async def restore_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary: ...

    async def delete_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> bool: ...

    async def capabilities(
        self, ref: QualifiedWorkspaceRef
    ) -> Mapping[str, ResearchCapability]: ...
