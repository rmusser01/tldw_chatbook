"""Authority-qualified domain contracts for the Research Workspace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from .source_operations import ResearchSourceOperation


MAX_RESEARCH_SELECTION_IDS = 10_100
MAX_RESEARCH_SELECTION_ROWS = 100


class WorkspaceDataSource(StrEnum):
    LOCAL = "local"
    SERVER = "server"


class RetrievalMode(StrEnum):
    """Retrieval path requested by grounded Research chat."""

    FTS = "fts"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SourceReadinessState(StrEnum):
    """Closed normalized readiness vocabulary shared by both authorities."""

    ATTACHED = "attached"
    PARSING = "parsing"
    INDEXING = "indexing"
    FTS_READY = "fts_ready"
    VECTOR_READY = "vector_ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


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
        if type(self.available) is not bool:
            raise TypeError("available must be bool")
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


class SourceIdentityMismatchError(ValueError):
    """Raised when an authority read returns a different source owner."""


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
    """One workspace association plus its distinct canonical catalog identity.

    ``source_id`` and ``version`` identify the association owner. For Local,
    ``source_id`` is a membership ID and ``version`` is always ``None``. For
    Server, they are the workspace-source row ID/version. ``catalog_item_id``
    and ``catalog_item_version`` always belong to canonical Media.
    """

    ref: QualifiedWorkspaceRef
    source_id: str
    title: str
    source_type: str
    catalog_item_id: str
    ready: bool = False
    version: int | None = None
    catalog_item_version: int | None = None
    selected: bool = True
    position: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(
            self, "source_type", _required_text(self.source_type, "source_type")
        )
        object.__setattr__(
            self,
            "catalog_item_id",
            _required_text(self.catalog_item_id, "catalog_item_id"),
        )
        if type(self.ready) is not bool:
            raise TypeError("ready must be bool")
        if type(self.selected) is not bool:
            raise TypeError("selected must be bool")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("position must be a non-negative integer")
        if self.version is not None and (
            type(self.version) is not int or self.version < 0
        ):
            raise ValueError("version must be a non-negative integer or None")
        if self.catalog_item_version is not None and (
            type(self.catalog_item_version) is not int
            or self.catalog_item_version < 0
        ):
            raise ValueError(
                "catalog_item_version must be a non-negative integer or None"
            )

    @property
    def workspace_source_id(self) -> str:
        """Explicit alias for the association identity."""

        return self.source_id

    @property
    def workspace_source_version(self) -> int | None:
        """Explicit alias for the association version."""

        return self.version


@dataclass(frozen=True, slots=True)
class ResearchCatalogItem:
    """One canonical Media catalog result under an explicit authority."""

    ref: QualifiedWorkspaceRef
    catalog_item_id: str
    title: str
    source_type: str
    catalog_item_version: int | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_item_id",
            _required_text(self.catalog_item_id, "catalog_item_id"),
        )
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(
            self, "source_type", _required_text(self.source_type, "source_type")
        )
        object.__setattr__(
            self, "updated_at", _optional_text(self.updated_at, "updated_at")
        )
        if self.catalog_item_version is not None and (
            type(self.catalog_item_version) is not int
            or self.catalog_item_version < 0
        ):
            raise ValueError(
                "catalog_item_version must be a non-negative integer or None"
            )


@dataclass(frozen=True, slots=True)
class SourceReadiness:
    """Read projection for one attached source; desired selection is separate."""

    ref: QualifiedWorkspaceRef
    source_id: str
    catalog_item_id: str | None
    state: SourceReadinessState
    metadata_ready: bool = False
    text_ready: bool = False
    fts_ready: bool = False
    vector_ready: bool = False
    citation_ready: bool = False
    summary_ready: bool = False
    tool_ready: bool = False
    stale: bool = False
    retry_eligible: bool = False
    next_action: str = "Refresh status"
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        if self.catalog_item_id is None:
            if self.ref.data_source is not WorkspaceDataSource.SERVER:
                raise ValueError(
                    "catalog_item_id may be null only for Server readiness"
                )
        else:
            object.__setattr__(
                self,
                "catalog_item_id",
                _required_text(self.catalog_item_id, "catalog_item_id"),
            )
        try:
            state = SourceReadinessState(self.state)
        except (TypeError, ValueError):
            raise ValueError("state must be a normalized readiness state") from None
        object.__setattr__(self, "state", state)
        for field_name in (
            "metadata_ready",
            "text_ready",
            "fts_ready",
            "vector_ready",
            "citation_ready",
            "summary_ready",
            "tool_ready",
            "stale",
            "retry_eligible",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")
        object.__setattr__(
            self, "next_action", _required_text(self.next_action, "next_action")
        )
        object.__setattr__(self, "detail", _optional_text(self.detail, "detail"))

    @property
    def desired_id(self) -> str:
        """Return the authority-owned ID used by desired selection."""

        if self.ref.data_source is WorkspaceDataSource.LOCAL:
            if self.catalog_item_id is None:
                raise ValueError("Local readiness requires catalog_item_id")
            return self.catalog_item_id
        return self.source_id


@dataclass(frozen=True, slots=True)
class ResearchSourcePreview:
    ref: QualifiedWorkspaceRef
    source_id: str
    catalog_item_id: str | None
    preview_mode: str
    text: str = ""
    snippets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        preview_mode = _required_text(self.preview_mode, "preview_mode")
        object.__setattr__(self, "preview_mode", preview_mode)
        if self.catalog_item_id is None:
            unavailable_server_preview = (
                self.ref.data_source is WorkspaceDataSource.SERVER
                and preview_mode in {"missing_media", "unavailable"}
            )
            if not unavailable_server_preview:
                raise ValueError(
                    "catalog_item_id may be null only for an unavailable Server preview"
                )
        else:
            object.__setattr__(
                self,
                "catalog_item_id",
                _required_text(self.catalog_item_id, "catalog_item_id"),
            )
        object.__setattr__(self, "text", _optional_text(self.text, "text"))
        object.__setattr__(self, "snippets", tuple(self.snippets))


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


@dataclass(frozen=True, slots=True)
class SourceSelectionResult:
    """Exact owner selection plus an optional bounded row reconciliation."""

    ref: QualifiedWorkspaceRef
    desired_source_ids: tuple[str, ...]
    sources: tuple[ResearchSourceSummary, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.desired_source_ids, tuple):
            raise TypeError("desired_source_ids must be a tuple")
        desired = tuple(
            _required_text(source_id, "desired_source_ids")
            for source_id in self.desired_source_ids
        )
        if any(
            len(source_id) > 1024 or len(source_id.encode("utf-8")) > 4096
            for source_id in desired
        ):
            raise ValueError(
                "desired_source_ids contains an identity that is too long"
            )
        if len(desired) > MAX_RESEARCH_SELECTION_IDS:
            raise ValueError("desired_source_ids exceeds the owner bound")
        if len(desired) != len(set(desired)):
            raise ValueError("desired_source_ids must be unique")
        if not isinstance(self.sources, tuple):
            raise TypeError("sources must be a tuple")
        sources = tuple(self.sources)
        if len(sources) > MAX_RESEARCH_SELECTION_ROWS:
            raise ValueError("selection row reconciliation exceeds the page bound")
        desired_set = set(desired)
        reconciled_ids: list[str] = []
        for source in sources:
            if source.ref != self.ref:
                raise ValueError("selection row has a mismatched workspace ref")
            if not source.selected:
                raise ValueError("selection reconciliation rows must be selected")
            desired_id = (
                source.catalog_item_id
                if self.ref.data_source is WorkspaceDataSource.LOCAL
                else source.source_id
            )
            if desired_id not in desired_set:
                raise ValueError("selection row is outside the desired owner state")
            reconciled_ids.append(desired_id)
        if len(reconciled_ids) != len(set(reconciled_ids)):
            raise ValueError("selection reconciliation rows must be unique")
        object.__setattr__(self, "desired_source_ids", desired)
        object.__setattr__(self, "sources", sources)


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

    async def list_sources(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> BoundedPageResult[ResearchSourceSummary]: ...

    async def search_catalog(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        query: str = "",
        source_types: tuple[str, ...] = (),
        sort_by: str = "updated_desc",
        limit: int = 25,
        offset: int = 0,
    ) -> BoundedPageResult[ResearchCatalogItem]: ...

    async def attach_existing(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        catalog_item_id: str,
        desired_selected: bool = True,
        idempotency_key: str,
    ) -> ResearchSourceOperation: ...

    async def remove_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool: ...

    async def update_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        title: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchSourceSummary: ...

    async def preview_source(
        self,
        ref: QualifiedWorkspaceRef,
        source_id: str,
        *,
        max_chars: int = 3000,
        snippet_limit: int = 3,
    ) -> ResearchSourcePreview: ...

    async def get_readiness(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[SourceReadiness, ...]: ...

    async def set_selected_scope(
        self,
        ref: QualifiedWorkspaceRef,
        source_ids: tuple[str, ...],
    ) -> SourceSelectionResult: ...

    async def reorder_sources(
        self,
        ref: QualifiedWorkspaceRef,
        ordered_source_ids: tuple[str, ...],
    ) -> tuple[ResearchSourceSummary, ...]: ...
