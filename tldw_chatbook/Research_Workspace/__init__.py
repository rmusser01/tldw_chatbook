"""Research Workspace authority adapters and normalized contracts."""

from .contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    ResearchCapability,
    ResearchSourcePreview,
    ResearchSourcePage,
    ResearchSourceSummary,
    SourceSelectionResult,
    ResearchWorkspacePort,
    ResearchWorkspaceSummary,
    RetrievalMode,
    SourceReadiness,
    SourceReadinessState,
    SourceIdentityMismatchError,
    WorkspaceDataSource,
)
from .controller import (
    ResearchRequestContext,
    ResearchSurfaceRequest,
    ResearchWorkspaceCatalogState,
    ResearchWorkspaceController,
)
from .layout_state import ResearchPanePreferences
from .overlay_store import ResearchPresentationOverlayStore
from .local_adapter import LocalResearchWorkspaceAdapter
from .server_adapter import ServerResearchWorkspaceAdapter

__all__ = [
    "BoundedPageResult",
    "CapabilityUnavailableError",
    "LocalResearchWorkspaceAdapter",
    "ProcessingRoute",
    "QualifiedWorkspaceRef",
    "ResearchCatalogItem",
    "ResearchCapability",
    "ResearchPanePreferences",
    "ResearchPresentationOverlayStore",
    "ResearchRequestContext",
    "ResearchSourcePreview",
    "ResearchSourcePage",
    "ResearchSourceSummary",
    "SourceSelectionResult",
    "ResearchSurfaceRequest",
    "ResearchWorkspaceController",
    "ResearchWorkspaceCatalogState",
    "ResearchWorkspacePort",
    "ResearchWorkspaceSummary",
    "ServerResearchWorkspaceAdapter",
    "RetrievalMode",
    "SourceReadiness",
    "SourceReadinessState",
    "SourceIdentityMismatchError",
    "WorkspaceDataSource",
]
