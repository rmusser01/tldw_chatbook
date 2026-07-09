"""Console-native widgets."""

from .console_control_bar import ConsoleControlBar
from .console_composer_bar import ConsoleComposerBar
from .console_background_effect import ConsoleBackgroundEffect, ConsoleTranscriptSurface
from .console_edit_message_modal import ConsoleEditMessageModal
from .console_rail_handle import ConsoleRailHandle
from .console_rename_session_modal import ConsoleRenameSessionModal
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
from .console_workspace_switcher_modal import ConsoleWorkspaceSwitcherModal

__all__ = [
    "build_console_workbench_state",
    "ConsoleComposerBar",
    "ConsoleBackgroundEffect",
    "ConsoleControlBar",
    "ConsoleEditMessageModal",
    "ConsoleRailHandle",
    "ConsoleRenameSessionModal",
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
    "ConsoleWorkspaceSwitcherModal",
]
