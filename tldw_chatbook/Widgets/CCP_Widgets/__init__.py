"""CCP Widget Components.

This module contains focused, reusable widget components for the CCP screen,
following Textual best practices for component separation.
"""

from .ccp_sidebar_widget import (
    CCPSidebarWidget,
    ConversationSearchRequested,
    ConversationLoadRequested,
    CharacterLoadRequested,
    PromptLoadRequested,
    DictionaryLoadRequested,
    ImportRequested,
    CreateRequested,
    RefreshRequested,
)

from .ccp_conversation_view_widget import (
    CCPConversationViewWidget,
    ConversationMessageWidget,
    MessageSelected,
    MessageEditRequested,
    MessageDeleteRequested,
    RegenerateRequested,
    ContinueConversationRequested,
)

from .ccp_character_card_widget import (
    CCPCharacterCardWidget,
    EditCharacterRequested,
    CloneCharacterRequested,
    ExportCharacterRequested,
    DeleteCharacterRequested,
    StartChatRequested,
)

from .ccp_character_editor_widget import (
    CCPCharacterEditorWidget,
    CharacterSaveRequested,
    CharacterFieldGenerateRequested,
    CharacterImageUploadRequested,
    CharacterImageGenerateRequested,
    CharacterEditorCancelled,
    AlternateGreetingAdded,
    AlternateGreetingRemoved,
)

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
    'CCPSidebarWidget',
    'CCPConversationViewWidget',
    'ConversationMessageWidget',
    'CCPCharacterCardWidget',
    'CCPCharacterEditorWidget',
    'CCPPromptEditorWidget',
    'CCPDictionaryEditorWidget',
    
    # Sidebar Messages
    'ConversationSearchRequested',
    'ConversationLoadRequested',
    'CharacterLoadRequested',
    'PromptLoadRequested',
    'DictionaryLoadRequested',
    'ImportRequested',
    'CreateRequested',
    'RefreshRequested',
    
    # Conversation View Messages
    'MessageSelected',
    'MessageEditRequested',
    'MessageDeleteRequested',
    'RegenerateRequested',
    'ContinueConversationRequested',
    
    # Character Card Messages
    'EditCharacterRequested',
    'CloneCharacterRequested',
    'ExportCharacterRequested',
    'DeleteCharacterRequested',
    'StartChatRequested',
    
    # Character Editor Messages
    'CharacterSaveRequested',
    'CharacterFieldGenerateRequested',
    'CharacterImageUploadRequested',
    'CharacterImageGenerateRequested',
    'CharacterEditorCancelled',
    'AlternateGreetingAdded',
    'AlternateGreetingRemoved',
    
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