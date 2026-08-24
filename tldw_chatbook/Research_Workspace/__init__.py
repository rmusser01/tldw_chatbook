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
from .controller import ResearchRequestContext, ResearchWorkspaceController
from .local_adapter import LocalResearchWorkspaceAdapter
from .server_adapter import ServerResearchWorkspaceAdapter

__all__ = [
    "BoundedPageResult",
    "CapabilityUnavailableError",
    "LocalResearchWorkspaceAdapter",
    "ProcessingRoute",
    "QualifiedWorkspaceRef",
    "ResearchCapability",
    "ResearchRequestContext",
    "ResearchSourceSummary",
    "ResearchWorkspaceController",
    "ResearchWorkspacePort",
    "ResearchWorkspaceSummary",
    "ServerResearchWorkspaceAdapter",
    "WorkspaceDataSource",
]
