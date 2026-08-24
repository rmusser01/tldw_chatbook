"""Region widgets for the Research Workspace screen."""

from .chat_region import ResearchChatRegion
from .header_region import ResearchHeaderRegion
from .mode_bar import ResearchModeStrip
from .pane_handle import ResearchPaneHandle
from .sources_region import ResearchSourcesRegion
from .studio_region import ResearchStudioRegion
from .workspace_menu import ResearchPaneModeStrip, ResearchWorkspaceMenu

__all__ = [
    "ResearchChatRegion",
    "ResearchHeaderRegion",
    "ResearchModeStrip",
    "ResearchPaneHandle",
    "ResearchPaneModeStrip",
    "ResearchSourcesRegion",
    "ResearchStudioRegion",
    "ResearchWorkspaceMenu",
]
