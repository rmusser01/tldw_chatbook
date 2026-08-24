"""Console-native widgets."""

from .console_assistant_turn import (
    ConsoleActivityActivated,
    ConsoleActivityDisclosure,
    ConsoleActivityHeader,
    ConsoleAssistantTurnWidget,
)
from .console_control_bar import ConsoleControlBar
from .console_speech_controls import ConsoleSpeechControls
from .console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
)
from .console_composer_bar import (
    ConsoleComposerBar,
    ConsoleComposerUndoHistory,
    ConsoleDraftStash,
)
from .console_command_popup import ConsoleCommandPopup
from .console_background_effect import ConsoleBackgroundEffect, ConsoleTranscriptSurface
from .console_bounded_section import ConsoleBoundedSection
from .console_inspector_ownership import (
    InspectorOwnershipPolicy,
    UnownedInspectorContentError,
)
from .console_changed_files_section import (
    ConsoleChangedFilesSection,
    ConsoleChangedFilesState,
)
from .console_citation_sources_modal import (
    ConsoleCitationSourceRow,
    ConsoleCitationSourcesModal,
    build_console_citation_source_rows,
)
from .console_edit_message_modal import ConsoleEditMessageModal, ConsoleEditResult
from .console_rail_handle import ConsoleRailHandle
from .console_prompts_modal import ConsolePromptsModal
from .console_prompts_state import ConsolePromptsState, PromptBrowseResult
from .console_project_instructions import (
    ConsoleProjectInstructionContextPanel,
    ConsoleProjectInstructionStatusRow,
    ProjectInstructionBindingOption,
    ProjectInstructionNoticeModal,
    ProjectInstructionSetupModal,
    ProjectInstructionSetupResult,
)
from .console_rename_session_modal import ConsoleRenameSessionModal
from .console_retrieval_scope_row import ConsoleRetrievalScopeRow
from .console_run_inspector import ConsoleRunInspector
from .console_save_as_modal import ConsoleSaveAsModal
from .console_session_surface import ConsoleSessionSurface
from .console_settings_modal import ConsoleSettingsModal
from .console_setup_modal import ConsoleSetupModal
from .console_settings_summary import ConsoleSettingsSummary
from .console_staged_context import ConsoleStagedContextTray
from .console_staged_evidence_strip import ConsoleStagedEvidenceStrip
from .console_transcript import ConsoleTranscript
from .console_workbench_state import build_console_workbench_state
from .console_workspace_context import ConsoleWorkspaceContextTray
from .console_workspace_switcher_modal import (
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)

__all__ = [
    "build_console_workbench_state",
    "ConsoleActivityActivated",
    "ConsoleActivityDisclosure",
    "ConsoleActivityHeader",
    "ConsoleAssistantTurnWidget",
    "ConsoleComposerBar",
    "ConsoleComposerUndoHistory",
    "ConsoleDraftStash",
    "ConsoleCommandPopup",
    "ConsoleBackgroundEffect",
    "ConsoleBoundedSection",
    "ConsoleChangedFilesSection",
    "ConsoleChangedFilesState",
    "ConsoleCitationSourceRow",
    "ConsoleCitationSourcesModal",
    "ConsoleControlBar",
    "ConsoleSpeechControls",
    "ConsoleContextControlState",
    "ConsoleEditMessageModal",
    "ConsoleEditResult",
    "InspectorOwnershipPolicy",
    "ConsoleRailHandle",
    "ConsolePromptsModal",
    "ConsolePromptsState",
    "ConsoleProjectInstructionContextPanel",
    "ConsoleProjectInstructionStatusRow",
    "ConsoleRenameSessionModal",
    "ConsoleRetrievalScopeRow",
    "ConsoleRunInspector",
    "ConsoleSaveAsModal",
    "ConsoleSessionSurface",
    "ConsoleSettingsModal",
    "ConsoleSettingsSummary",
    "ConsoleSetupModal",
    "ConsoleStagedContextTray",
    "ConsoleStagedEvidenceStrip",
    "ConsoleTranscript",
    "ConsoleTranscriptSurface",
    "ConsoleWorkspaceContextTray",
    "ConsoleWorkspaceRenameModal",
    "ConsoleWorkspaceSwitcherModal",
    "PromptBrowseResult",
    "ProjectInstructionBindingOption",
    "ProjectInstructionNoticeModal",
    "ProjectInstructionSetupModal",
    "ProjectInstructionSetupResult",
    "UnownedInspectorContentError",
    "build_console_citation_source_rows",
    "build_console_context_control_state",
]
