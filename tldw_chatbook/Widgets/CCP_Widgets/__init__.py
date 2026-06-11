"""CCP Widget Components.

Surviving prompt/dictionary editor widgets. The legacy CCP screen chrome
(sidebar, character card/editor, conversation view, persona card/editor)
was retired in favor of the Personas workbench
(tldw_chatbook/Widgets/Persona_Widgets/).
"""

from .ccp_prompt_editor_widget import (
    CCPPromptEditorWidget,
    PromptSaveRequested,
    PromptDeleteRequested,
    PromptTestRequested,
    PromptEditorCancelled,
    PromptVariableAdded,
    PromptVariableRemoved,
)

from .ccp_dictionary_editor_widget import (
    CCPDictionaryEditorWidget,
    DictionarySaveRequested,
    DictionaryDeleteRequested,
    DictionaryEntryAdded,
    DictionaryEntryRemoved,
    DictionaryEntryUpdated,
    DictionaryImportRequested,
    DictionaryExportRequested,
    DictionaryEditorCancelled,
)

__all__ = [
    # Widgets
    'CCPPromptEditorWidget',
    'CCPDictionaryEditorWidget',

    # Prompt Editor Messages
    'PromptSaveRequested',
    'PromptDeleteRequested',
    'PromptTestRequested',
    'PromptEditorCancelled',
    'PromptVariableAdded',
    'PromptVariableRemoved',

    # Dictionary Editor Messages
    'DictionarySaveRequested',
    'DictionaryDeleteRequested',
    'DictionaryEntryAdded',
    'DictionaryEntryRemoved',
    'DictionaryEntryUpdated',
    'DictionaryImportRequested',
    'DictionaryExportRequested',
    'DictionaryEditorCancelled',
]
