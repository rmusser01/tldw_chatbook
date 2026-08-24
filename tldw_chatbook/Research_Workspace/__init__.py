"""Research Workspace authority adapters and normalized contracts."""

from .contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchSourceSummary,
    ResearchWorkspacePort,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)
from .controller import (
    ResearchRequestContext,
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
    "ResearchCapability",
    "ResearchPanePreferences",
    "ResearchPresentationOverlayStore",
    "ResearchRequestContext",
    "ResearchSourceSummary",
    "ResearchWorkspaceController",
    "ResearchWorkspaceCatalogState",
    "ResearchWorkspacePort",
    "ResearchWorkspaceSummary",
    "ServerResearchWorkspaceAdapter",
    "WorkspaceDataSource",
]
