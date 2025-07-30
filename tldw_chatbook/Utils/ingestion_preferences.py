# tldw_chatbook/Utils/ingestion_preferences.py
# Utility functions for managing ingestion UI preferences

from typing import Optional
from loguru import logger
from ..config import save_setting_to_cli_config, get_cli_setting

def get_ingestion_mode_preference(media_type: str) -> bool:
    """
    Get the saved preference for Simple/Advanced mode for a media type.
    
    Args:
        media_type: The media type (e.g., 'video', 'audio', 'document')
    
    Returns:
        bool: True for simple mode (default), False for advanced mode
    """
    try:
        # Get from ingestion_preferences section
        mode = get_cli_setting("ingestion_preferences", f"{media_type}_simple_mode", True)
        return bool(mode)
    except Exception as e:
        logger.error(f"Error loading ingestion mode preference for {media_type}: {e}")
        return True  # Default to simple mode

def save_ingestion_mode_preference(media_type: str, simple_mode: bool) -> None:
    """
    Save the preference for Simple/Advanced mode for a media type.
    
    Args:
        media_type: The media type (e.g., 'video', 'audio', 'document')
        simple_mode: True for simple mode, False for advanced mode
    """
    try:
        save_setting_to_cli_config("ingestion_preferences", f"{media_type}_simple_mode", simple_mode)
        logger.debug(f"Saved ingestion mode preference for {media_type}: simple_mode={simple_mode}")
    except Exception as e:
        logger.error(f"Error saving ingestion mode preference for {media_type}: {e}")

def get_global_ingestion_mode_preference() -> bool:
    """
    Get the global default for Simple/Advanced mode across all media types.
    
    Returns:
        bool: True for simple mode (default), False for advanced mode
    """
    try:
        mode = get_cli_setting("ingestion_preferences", "global_simple_mode", True)
        return bool(mode)
    except Exception as e:
        logger.error(f"Error loading global ingestion mode preference: {e}")
        return True  # Default to simple mode

def save_global_ingestion_mode_preference(simple_mode: bool) -> None:
    """
    Save the global default for Simple/Advanced mode.
    
    Args:
        simple_mode: True for simple mode, False for advanced mode
    """
    try:
        save_setting_to_cli_config("ingestion_preferences", "global_simple_mode", simple_mode)
        logger.debug(f"Saved global ingestion mode preference: simple_mode={simple_mode}")
    except Exception as e:
        logger.error(f"Error saving global ingestion mode preference: {e}")

# End of ingestion_preferences.py