"""Validation decorators for CCP handler methods."""

from functools import wraps
from typing import Callable, Type, Any, Optional
from pydantic import BaseModel, ValidationError
from loguru import logger

from .ccp_validators import (
    ConversationInput,
    CharacterCardInput,
    PromptInput,
    DictionaryInput,
    SearchInput,
    FileImportInput,
    validate_with_model
)

logger = logger.bind(module="CCPValidationDecorators")


def validate_input(model_class: Type[BaseModel], extract_fields: Optional[list] = None):
    """
    Decorator to validate input data using Pydantic models.
    
    Args:
        model_class: The Pydantic model class to use for validation
        extract_fields: List of field names to extract from the handler's widgets
        
    Example:
        @validate_input(CharacterCardInput, extract_fields=['name', 'description'])
        async def save_character(self):
            # Method will receive validated_data as first argument after self
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                # Extract data based on the context
                if extract_fields:
                    # Extract from widgets
                    data = {}
                    for field in extract_fields:
                        widget_id = f"ccp-editor-{field.replace('_', '-')}-input"
                        alt_widget_id = f"ccp-editor-{field.replace('_', '-')}-textarea"
                        
                        try:
                            # Try input field first
                            widget = self.window.query_one(f"#{widget_id}")
                            data[field] = widget.value if hasattr(widget, 'value') else widget.text
                        except:
                            try:
                                # Try textarea
                                widget = self.window.query_one(f"#{alt_widget_id}")
                                data[field] = widget.text if hasattr(widget, 'text') else widget.value
                            except:
                                # Field not found, set as None
                                data[field] = None
                else:
                    # Expect data as first argument
                    if args and isinstance(args[0], dict):
                        data = args[0]
                        args = args[1:]  # Remove data from args
                    else:
                        data = kwargs.get('data', {})
                
                # Validate the data
                is_valid, validated_data, error_msg = validate_with_model(model_class, data)
                
                if not is_valid:
                    logger.warning(f"Validation failed for {func.__name__}: {error_msg}")
                    # Show error to user
                    if hasattr(self, 'window') and hasattr(self.window, 'app_instance'):
                        self.window.app_instance.notify(
                            f"Validation Error: {error_msg}",
                            severity="error"
                        )
                    return None
                
                # Call the original function with validated data
                return await func(self, validated_data, *args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error in validation decorator for {func.__name__}: {e}", exc_info=True)
                # Call original function without validation as fallback
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_search(func: Callable) -> Callable:
    """
    Specialized decorator for search operations.
    Automatically extracts search parameters from the handler.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            # Extract search parameters
            data = {
                'search_term': args[0] if args else kwargs.get('search_term', ''),
                'search_type': args[1] if len(args) > 1 else kwargs.get('search_type', 'title')
            }
            
            # Try to get additional parameters from checkboxes
            try:
                include_char = self.window.query_one("#conv-char-search-include-character-checkbox")
                data['include_character_chats'] = include_char.value if hasattr(include_char, 'value') else True
            except:
                data['include_character_chats'] = True
            
            try:
                all_chars = self.window.query_one("#conv-char-search-all-characters-checkbox")
                data['all_characters'] = all_chars.value if hasattr(all_chars, 'value') else True
            except:
                data['all_characters'] = True
            
            # Validate
            is_valid, validated_data, error_msg = validate_with_model(SearchInput, data)
            
            if not is_valid:
                logger.warning(f"Search validation failed: {error_msg}")
                # Still proceed with original search term but log the issue
                return await func(self, args[0] if args else '', *args[1:], **kwargs)
            
            # Call with validated data
            return await func(self, validated_data.search_term, validated_data.search_type, 
                            validated_data, *args[2:], **kwargs)
            
        except Exception as e:
            logger.error(f"Error in search validation: {e}", exc_info=True)
            return await func(self, *args, **kwargs)
    
    return wrapper


def validate_file_import(func: Callable) -> Callable:
    """
    Decorator specifically for file import operations.
    Validates file path and type before processing.
    """
    @wraps(func)
    async def wrapper(self, file_path: str, file_type: str = None, *args, **kwargs):
        try:
            from pathlib import Path
            
            # Determine file type from context or extension
            if not file_type:
                path = Path(file_path)
                ext = path.suffix.lower()
                if ext in ['.json', '.yaml', '.yml']:
                    # Could be various types, need to check content
                    file_type = 'character_card'  # Default assumption
                elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    file_type = 'image'
                else:
                    file_type = 'character_card'  # Default
            
            data = {
                'file_path': Path(file_path),
                'file_type': file_type,
                'overwrite_existing': kwargs.get('overwrite', False)
            }
            
            # Validate
            is_valid, validated_data, error_msg = validate_with_model(FileImportInput, data)
            
            if not is_valid:
                logger.error(f"File import validation failed: {error_msg}")
                if hasattr(self, 'window') and hasattr(self.window, 'app_instance'):
                    self.window.app_instance.notify(
                        f"Invalid file: {error_msg}",
                        severity="error"
                    )
                return None
            
            # Call with validated path
            return await func(self, str(validated_data.file_path), validated_data.file_type, 
                            *args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in file import validation: {e}", exc_info=True)
            return await func(self, file_path, file_type, *args, **kwargs)
    
    return wrapper


def sanitize_output(func: Callable) -> Callable:
    """
    Decorator to sanitize output data before displaying to user.
    Prevents XSS and other injection attacks in displayed content.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            import html
            
            result = await func(self, *args, **kwargs)
            
            # If result is a string, sanitize it
            if isinstance(result, str):
                # Basic HTML escaping
                result = html.escape(result)
            elif isinstance(result, dict):
                # Sanitize string values in dictionary
                for key, value in result.items():
                    if isinstance(value, str):
                        result[key] = html.escape(value)
            elif isinstance(result, list):
                # Sanitize strings in list
                result = [html.escape(item) if isinstance(item, str) else item for item in result]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in output sanitization: {e}", exc_info=True)
            return await func(self, *args, **kwargs)
    
    return wrapper


def require_selection(item_type: str = "item"):
    """
    Decorator to ensure an item is selected before operation proceeds.
    
    Args:
        item_type: Type of item that must be selected (for error messages)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check for various selection attributes
            selection_attrs = [
                f'selected_{item_type}_id',
                f'current_{item_type}_id',
                f'{item_type}_id'
            ]
            
            selected_id = None
            for attr in selection_attrs:
                if hasattr(self, attr):
                    selected_id = getattr(self, attr)
                    if selected_id:
                        break
            
            if not selected_id:
                logger.warning(f"No {item_type} selected for operation {func.__name__}")
                if hasattr(self, 'window') and hasattr(self.window, 'app_instance'):
                    self.window.app_instance.notify(
                        f"Please select a {item_type} first",
                        severity="warning"
                    )
                return None
            
            return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator