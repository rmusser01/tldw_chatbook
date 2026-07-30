"""Console-native widgets."""

from .console_control_bar import ConsoleControlBar
from .console_composer_bar import (
    ConsoleComposerBar,
    ConsoleComposerUndoHistory,
    ConsoleDraftStash,
)
from .console_background_effect import ConsoleBackgroundEffect, ConsoleTranscriptSurface
from .console_citation_sources_modal import (
    ConsoleCitationSourceRow,
    ConsoleCitationSourcesModal,
    build_console_citation_source_rows,
)
from .console_edit_message_modal import ConsoleEditMessageModal, ConsoleEditResult
from .console_rail_handle import ConsoleRailHandle
from .console_rename_session_modal import ConsoleRenameSessionModal
from .console_retrieval_scope_row import ConsoleRetrievalScopeRow
from .console_run_inspector import ConsoleRunInspector
from .console_save_as_modal import ConsoleSaveAsModal
from .console_session_surface import ConsoleSessionSurface
from .console_settings_modal import ConsoleSettingsModal
from .console_setup_modal import ConsoleSetupModal
from .console_settings_summary import ConsoleSettingsSummary
from .console_staged_context import ConsoleStagedContextTray
from .console_transcript import ConsoleTranscript
from .console_workbench_state import build_console_workbench_state
from .console_workspace_context import ConsoleWorkspaceContextTray
from .console_workspace_switcher_modal import (
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)

__all__ = [
    "build_console_workbench_state",
    "ConsoleComposerBar",
    "ConsoleComposerUndoHistory",
    "ConsoleDraftStash",
    "ConsoleBackgroundEffect",
    "ConsoleCitationSourceRow",
    "ConsoleCitationSourcesModal",
    "ConsoleControlBar",
    "ConsoleEditMessageModal",
    "ConsoleEditResult",
    "ConsoleRailHandle",
    "ConsoleRenameSessionModal",
    "ConsoleRetrievalScopeRow",
    "ConsoleRunInspector",
    "ConsoleSaveAsModal",
    "ConsoleSessionSurface",
    "ConsoleSettingsModal",
    "ConsoleSettingsSummary",
    "ConsoleSetupModal",
    "ConsoleStagedContextTray",
    "ConsoleTranscript",
    "ConsoleTranscriptSurface",
    "ConsoleWorkspaceContextTray",
    "ConsoleWorkspaceRenameModal",
    "ConsoleWorkspaceSwitcherModal",
    "build_console_citation_source_rows",
]
