"""Destinations that own Research Workspace management commands."""

from dataclasses import dataclass
from types import MappingProxyType

from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource


@dataclass(frozen=True, slots=True)
class ResearchOwnerLink:
    label: str
    route: str
    context: tuple[tuple[str, str], ...] = ()


RESEARCH_OWNER_LINKS = MappingProxyType(
    {
        "console": ResearchOwnerLink("Open in Console", "chat"),
        "agent-task": ResearchOwnerLink("Create agent task in Console", "chat"),
        "acp-history": ResearchOwnerLink("ACP history and runtimes", "acp"),
        "sandbox": ResearchOwnerLink("Sandbox diagnostics in Logs", "logs"),
        "mcp": ResearchOwnerLink("MCP settings", "settings", (("category", "tools"),)),
        "provider": ResearchOwnerLink("Provider and model settings", "llm"),
        "library": ResearchOwnerLink("Open Library", "library"),
        "artifacts": ResearchOwnerLink("Open Artifacts", "artifacts"),
        "study": ResearchOwnerLink("Open Study", "study"),
        "settings-workspaces": ResearchOwnerLink(
            "Manage Workspaces in Settings", "settings", (("category", "workspaces"),)
        ),
        "speech": ResearchOwnerLink("Open Speech", "stts"),
        "runs": ResearchOwnerLink("Open Runs", "research"),
    }
)


def can_open_workspace_in_console(ref: QualifiedWorkspaceRef | None) -> bool:
    return ref is not None and ref.data_source is WorkspaceDataSource.LOCAL
