"""Region widgets for the Research Workspace screen."""

from .chat_region import ResearchChatRegion
from .header_region import ResearchHeaderRegion
from .mode_bar import ResearchModeStrip
from .pane_handle import ResearchPaneHandle
from .sources_region import ResearchSourcesRegion
from .add_source_modal import ResearchAddSourceModal, ResearchSourceIntakeRequest
from .source_receipt import ResearchSourceReceiptList
from .source_list import ResearchSourceList
from .source_inspector import (
    ResearchSourceAnnotationDraft,
    ResearchSourceInspectorModal,
)
from .overlay_conflict_modal import ResearchOverlayConflictModal
from .studio_region import ResearchStudioRegion
from .workspace_menu import ResearchPaneModeStrip, ResearchWorkspaceMenu

__all__ = [
    "ResearchChatRegion",
    "ResearchHeaderRegion",
    "ResearchModeStrip",
    "ResearchPaneHandle",
    "ResearchPaneModeStrip",
    "ResearchSourcesRegion",
    "ResearchAddSourceModal",
    "ResearchSourceIntakeRequest",
    "ResearchSourceReceiptList",
    "ResearchSourceList",
    "ResearchSourceAnnotationDraft",
    "ResearchSourceInspectorModal",
    "ResearchOverlayConflictModal",
    "ResearchStudioRegion",
    "ResearchWorkspaceMenu",
]
