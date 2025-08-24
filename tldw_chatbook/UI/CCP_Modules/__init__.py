"""CCP (Conversations, Characters & Prompts) modular handlers.

This module provides modular handlers for the CCP window functionality,
following the same pattern as the Chat window for consistency and maintainability.

Now enhanced with:
- Pydantic validation for input data
- Loading indicators for async operations
- Performance tracking integration with existing stats system
"""

from .ccp_messages import (
    CCPMessage,
    ConversationMessage,
    CharacterMessage,
    PromptMessage,
    DictionaryMessage,
    SidebarMessage,
    ViewChangeMessage
)

from .ccp_conversation_handler import CCPConversationHandler
from .ccp_character_handler import CCPCharacterHandler
from .ccp_prompt_handler import CCPPromptHandler
from .ccp_dictionary_handler import CCPDictionaryHandler
from .ccp_message_manager import CCPMessageManager
from .ccp_sidebar_handler import CCPSidebarHandler

# Import validation models
from .ccp_validators import (
    ConversationInput,
    CharacterCardInput,
    PromptInput,
    DictionaryInput,
    SearchInput,
    FileImportInput,
    validate_with_model
)

# Import validation decorators
from .ccp_validation_decorators import (
    validate_input,
    validate_search,
    validate_file_import,
    sanitize_output,
    require_selection
)

# Import loading indicators
from .ccp_loading_indicators import (
    CCPLoadingWidget,
    LoadingManager,
    InlineLoadingIndicator,
    with_loading,
    with_progress
)

# Import enhancement setup
from .ccp_enhanced_handlers import setup_ccp_enhancements

__all__ = [
    # Messages
    'CCPMessage',
    'ConversationMessage',
    'CharacterMessage', 
    'PromptMessage',
    'DictionaryMessage',
    'SidebarMessage',
    'ViewChangeMessage',
    
    # Handlers
    'CCPConversationHandler',
    'CCPCharacterHandler',
    'CCPPromptHandler',
    'CCPDictionaryHandler',
    'CCPMessageManager',
    'CCPSidebarHandler',
    
    # Validation models
    'ConversationInput',
    'CharacterCardInput',
    'PromptInput',
    'DictionaryInput',
    'SearchInput',
    'FileImportInput',
    'validate_with_model',
    
    # Decorators
    'validate_input',
    'validate_search',
    'validate_file_import',
    'sanitize_output',
    'require_selection',
    
    # Loading indicators
    'CCPLoadingWidget',
    'LoadingManager',
    'InlineLoadingIndicator',
    'with_loading',
    'with_progress',
    
    # Enhancement setup
    'setup_ccp_enhancements'
]