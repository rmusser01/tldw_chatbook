"""Destinations that own Research Workspace management commands."""

from dataclasses import dataclass
from types import MappingProxyType


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
    }
)
