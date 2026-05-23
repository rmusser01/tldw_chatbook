"""Console-native widgets."""

from .console_control_bar import ConsoleControlBar
from .console_composer_bar import ConsoleComposerBar
from .console_run_inspector import ConsoleRunInspector
from .console_save_as_modal import ConsoleSaveAsModal
from .console_session_surface import ConsoleSessionSurface
from .console_staged_context import ConsoleStagedContextTray
from .console_transcript import ConsoleTranscript
from .console_workspace_context import ConsoleWorkspaceContextTray

__all__ = [
    "ConsoleComposerBar",
    "ConsoleControlBar",
    "ConsoleRunInspector",
    "ConsoleSaveAsModal",
    "ConsoleSessionSurface",
    "ConsoleStagedContextTray",
    "ConsoleTranscript",
    "ConsoleWorkspaceContextTray",
]
