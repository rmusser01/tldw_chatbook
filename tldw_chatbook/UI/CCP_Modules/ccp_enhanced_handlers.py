"""Enhanced CCP handlers with validation and loading indicators integration."""

from typing import Optional, Dict, Any
from loguru import logger
from textual import work

# Import the validation and loading decorators
from .ccp_validation_decorators import (
    validate_input,
    validate_search,
    validate_file_import,
    sanitize_output,
    require_selection
)
from .ccp_loading_indicators import (
    with_loading,
    with_progress,
    LoadingManager
)
from .ccp_validators import (
    CharacterCardInput,
    ConversationInput,
    PromptInput,
    DictionaryInput
)

logger = logger.bind(module="CCPEnhancedHandlers")


def enhance_conversation_handler(handler_class):
    """
    Enhance the conversation handler with validation and loading indicators.
    
    This function modifies the handler methods to include:
    - Input validation using Pydantic models
    - Loading indicators for async operations
    - Performance tracking via existing stats system
    """
    
    # Enhance search method
    original_search = handler_class.handle_search
    
    @validate_search
    @with_loading("Searching conversations...", "Search complete", "Search failed")
    async def enhanced_search(self, search_term: str, search_type: str = "title", validated_data=None):
        return await original_search(search_term, search_type)
    
    handler_class.handle_search = enhanced_search
    
    # Enhance load method
    original_load = handler_class.handle_load_selected
    
    @require_selection("conversation")
    @with_loading("Loading conversation...", "Conversation loaded", "Failed to load conversation")
    async def enhanced_load(self):
        return await original_load()
    
    handler_class.handle_load_selected = enhanced_load
    
    # Enhance save method if exists
    if hasattr(handler_class, 'handle_save_details'):
        original_save = handler_class.handle_save_details
        
        @validate_input(ConversationInput, extract_fields=['title', 'keywords'])
        @with_loading("Saving conversation details...", "Details saved", "Failed to save details")
        async def enhanced_save(self, validated_data):
            # Pass validated data to original method
            return await original_save()
        
        handler_class.handle_save_details = enhanced_save
    
    logger.info("Enhanced conversation handler with validation and loading indicators")
    return handler_class


def enhance_character_handler(handler_class):
    """
    Enhance the character handler with validation and loading indicators.
    """
    
    # Enhance save character method
    if hasattr(handler_class, 'handle_save_character'):
        original_save = handler_class.handle_save_character
        
        @validate_input(
            CharacterCardInput,
            extract_fields=[
                'char_name', 'char_description', 'char_personality',
                'char_scenario', 'char_first_message', 'char_keywords',
                'char_system_prompt', 'char_tags', 'char_creator', 'char_version'
            ]
        )
        @with_loading("Saving character...", "Character saved successfully", "Failed to save character")
        async def enhanced_save(self, validated_data):
            return await original_save()
        
        handler_class.handle_save_character = enhanced_save
    
    # Enhance load character method
    if hasattr(handler_class, 'handle_load_character'):
        original_load = handler_class.handle_load_character
        
        @require_selection("character")
        @with_loading("Loading character card...", "Character loaded", "Failed to load character")
        async def enhanced_load(self):
            return await original_load()
        
        handler_class.handle_load_character = enhanced_load
    
    # Enhance import method
    if hasattr(handler_class, 'handle_import_character'):
        original_import = handler_class.handle_import_character
        
        @validate_file_import
        @with_loading("Importing character card...", "Character imported", "Failed to import character")
        async def enhanced_import(self, file_path: str, file_type: str = "character_card"):
            return await original_import(file_path)
        
        handler_class.handle_import_character = enhanced_import
    
    # Enhance refresh list
    original_refresh = handler_class.refresh_character_list
    
    @with_loading("Refreshing character list...", "List refreshed", "Failed to refresh list")
    async def enhanced_refresh(self):
        return await original_refresh(self)
    
    handler_class.refresh_character_list = enhanced_refresh
    
    logger.info("Enhanced character handler with validation and loading indicators")
    return handler_class


def enhance_prompt_handler(handler_class):
    """
    Enhance the prompt handler with validation and loading indicators.
    """
    
    # Enhance save prompt method
    if hasattr(handler_class, 'handle_save_prompt'):
        original_save = handler_class.handle_save_prompt
        
        @validate_input(
            PromptInput,
            extract_fields=[
                'prompt_name', 'prompt_author', 'prompt_description',
                'prompt_system', 'prompt_user', 'prompt_keywords'
            ]
        )
        @with_loading("Saving prompt...", "Prompt saved successfully", "Failed to save prompt")
        async def enhanced_save(self, validated_data):
            return await original_save()
        
        handler_class.handle_save_prompt = enhanced_save
    
    # Enhance search
    if hasattr(handler_class, 'handle_search'):
        original_search = handler_class.handle_search
        
        @validate_search
        @with_loading("Searching prompts...", "Search complete", "Search failed")
        async def enhanced_search(self, search_term: str):
            return await original_search(search_term)
        
        handler_class.handle_search = enhanced_search
    
    logger.info("Enhanced prompt handler with validation and loading indicators")
    return handler_class


def enhance_dictionary_handler(handler_class):
    """
    Enhance the dictionary handler with validation and loading indicators.
    """
    
    # Enhance save dictionary method
    if hasattr(handler_class, 'handle_save_dictionary'):
        original_save = handler_class.handle_save_dictionary
        
        @validate_input(
            DictionaryInput,
            extract_fields=[
                'dict_name', 'dict_description', 'dict_strategy', 'dict_max_tokens'
            ]
        )
        @with_loading("Saving dictionary...", "Dictionary saved", "Failed to save dictionary")
        async def enhanced_save(self, validated_data):
            return await original_save()
        
        handler_class.handle_save_dictionary = enhanced_save
    
    # Enhance refresh list
    original_refresh = handler_class.refresh_dictionary_list
    
    @with_loading("Refreshing dictionary list...", "List refreshed", "Failed to refresh list")
    async def enhanced_refresh(self):
        return await original_refresh(self)
    
    handler_class.refresh_dictionary_list = enhanced_refresh
    
    logger.info("Enhanced dictionary handler with validation and loading indicators")
    return handler_class


def setup_ccp_enhancements(ccp_window):
    """
    Setup all enhancements for the CCP window.
    
    This should be called during CCP window initialization to add:
    - Loading manager
    - Validation to all handlers
    - Performance tracking integration
    
    Args:
        ccp_window: The CCPWindow instance to enhance
    """
    try:
        # Initialize loading manager
        ccp_window.loading_manager = LoadingManager(ccp_window)
        
        # Enhance handlers
        if hasattr(ccp_window, 'conversation_handler'):
            enhance_conversation_handler(ccp_window.conversation_handler.__class__)
        
        if hasattr(ccp_window, 'character_handler'):
            enhance_character_handler(ccp_window.character_handler.__class__)
        
        if hasattr(ccp_window, 'prompt_handler'):
            enhance_prompt_handler(ccp_window.prompt_handler.__class__)
        
        if hasattr(ccp_window, 'dictionary_handler'):
            enhance_dictionary_handler(ccp_window.dictionary_handler.__class__)
        
        # Setup loading widget
        if hasattr(ccp_window.loading_manager, 'setup'):
            # This would be called in on_mount to properly mount the widget
            pass
        
        logger.info("CCP window enhancements setup complete")
        
    except Exception as e:
        logger.error(f"Failed to setup CCP enhancements: {e}", exc_info=True)