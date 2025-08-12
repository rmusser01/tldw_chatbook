# sidebar_compatibility.py
# Description: Backward compatibility layer for migrating from dual sidebars to unified sidebar
#
# This module provides a compatibility layer that maps old widget IDs and event handlers
# to the new unified sidebar structure, allowing gradual migration without breaking existing code.
#
# Imports
from typing import TYPE_CHECKING, Optional, Any
import logging
#
# 3rd-Party Imports
from loguru import logger
from textual.widget import Widget
#
# Local Imports

if TYPE_CHECKING:
    from ...app import TldwCli
    from .unified_chat_sidebar import UnifiedChatSidebar

#
#######################################################################################################################
#
# Compatibility Mappings
#

# Map old widget IDs to new widget IDs in the unified sidebar
WIDGET_ID_MAPPINGS = {
    # Left sidebar (settings) mappings
    "chat-api-provider": "settings-provider",
    "chat-api-model": "settings-model",
    "chat-temperature": "settings-temperature",
    "chat-system-prompt": "settings-system-prompt",
    "chat-top-p": "settings-top-p",
    "chat-top-k": "settings-top-k",
    "chat-min-p": "settings-min-p",
    "chat-streaming-enabled-checkbox": "settings-streaming-enabled",
    "chat-rag-enabled": "settings-rag-enabled",
    "chat-rag-pipeline": "settings-rag-pipeline",
    
    # Right sidebar (session/character) mappings
    "chat-conversation-uuid-display": "session-chat-id",
    "chat-conversation-title-input": "session-title",
    "chat-conversation-keywords-input": "session-keywords",
    "chat-save-current-chat-button": "session-save-chat",
    "chat-new-conversation-button": "session-new-chat",
    "chat-clone-current-chat-button": "session-clone-chat",
    "chat-convert-to-note-button": "session-to-note",
    "chat-strip-thinking-tags-checkbox": "session-strip-tags",
    
    # Media search mappings
    "chat-media-search-input": "content-search-input",
    "chat-media-search-button": "content-search-btn",
    "chat-media-search-results-listview": "content-results-list",
    "chat-media-prev-page-button": "content-prev",
    "chat-media-next-page-button": "content-next",
    "chat-media-page-label": "content-page-label",
    
    # Notes mappings (now in content tab)
    "chat-notes-search-input": "content-search-input",
    "chat-notes-listview": "content-results-list",
    "chat-notes-load-button": "content-load",
    
    # Prompts mappings (now in content tab)
    "chat-prompt-search-input": "content-search-input",
    "chat-prompts-listview": "content-results-list",
    "chat-prompt-load-selected-button": "content-load",
    
    # Toggle buttons
    "toggle-chat-left-sidebar": "toggle-unified-sidebar",
    "toggle-chat-right-sidebar": "toggle-unified-sidebar",
}

# Map old event handler names to new handlers
EVENT_HANDLER_MAPPINGS = {
    "handle_chat_tab_sidebar_toggle": "handle_unified_sidebar_toggle",
    "handle_sidebar_shrink": "handle_unified_sidebar_resize",
    "handle_sidebar_expand": "handle_unified_sidebar_resize",
}

#
#######################################################################################################################
#
# Compatibility Adapter Class
#

class LegacySidebarAdapter:
    """
    Adapter class that provides backward compatibility for code expecting the old dual-sidebar structure.
    This allows gradual migration to the unified sidebar without breaking existing functionality.
    """
    
    def __init__(self, unified_sidebar: 'UnifiedChatSidebar', app_instance: 'TldwCli'):
        self.sidebar = unified_sidebar
        self.app = app_instance
        self._widget_cache = {}
        logger.debug("LegacySidebarAdapter initialized")
    
    def query_one(self, selector: str, expect_type: Optional[type] = None) -> Optional[Widget]:
        """
        Map old selectors to new widget structure.
        
        Args:
            selector: CSS selector or widget ID (old format)
            expect_type: Expected widget type for validation
            
        Returns:
            The mapped widget or None if not found
        """
        # Remove # if present
        if selector.startswith("#"):
            selector = selector[1:]
        
        # Check if we have a mapping for this ID
        if selector in WIDGET_ID_MAPPINGS:
            new_selector = WIDGET_ID_MAPPINGS[selector]
            logger.debug(f"Mapping old selector '{selector}' to new selector '{new_selector}'")
            
            try:
                # Try to find the widget in the unified sidebar
                widget = self.sidebar.query_one(f"#{new_selector}")
                
                # Cache for faster subsequent lookups
                self._widget_cache[selector] = widget
                
                # Validate type if specified
                if expect_type and not isinstance(widget, expect_type):
                    logger.warning(f"Widget type mismatch: expected {expect_type}, got {type(widget)}")
                    return None
                    
                return widget
                
            except Exception as e:
                logger.debug(f"Widget not found with new selector '{new_selector}': {e}")
                
                # Try fallback strategies
                return self._fallback_query(selector, expect_type)
        
        # No mapping found - try direct query as fallback
        try:
            return self.sidebar.query_one(f"#{selector}")
        except:
            return None
    
    def _fallback_query(self, selector: str, expect_type: Optional[type] = None) -> Optional[Widget]:
        """
        Fallback strategies for finding widgets when direct mapping fails.
        """
        # Strategy 1: Check if it's a settings-related widget
        if selector.startswith("chat-") and any(x in selector for x in ["api", "temperature", "model", "provider"]):
            # Try with "settings-" prefix
            alt_selector = selector.replace("chat-", "settings-")
            try:
                return self.sidebar.query_one(f"#{alt_selector}")
            except:
                pass
        
        # Strategy 2: Check if it's a session-related widget
        if "conversation" in selector or "chat" in selector:
            alt_selector = selector.replace("chat-", "session-").replace("conversation-", "")
            try:
                return self.sidebar.query_one(f"#{alt_selector}")
            except:
                pass
        
        # Strategy 3: Content tab widgets
        if any(x in selector for x in ["media", "notes", "prompt"]):
            # These are now unified in content tab
            if "search" in selector:
                return self._get_content_search_widget()
            elif "list" in selector or "results" in selector:
                return self._get_content_results_widget()
        
        return None
    
    def _get_content_search_widget(self):
        """Get the unified content search widget."""
        try:
            # First ensure we're on the content tab
            self.sidebar.action_switch_tab("content")
            return self.sidebar.query_one("#content-search-input")
        except:
            return None
    
    def _get_content_results_widget(self):
        """Get the unified content results widget."""
        try:
            # First ensure we're on the content tab
            self.sidebar.action_switch_tab("content")
            return self.sidebar.query_one("#content-results-list")
        except:
            return None
    
    def handle_legacy_event(self, event_name: str, *args, **kwargs):
        """
        Route legacy event handlers to new unified handlers.
        
        Args:
            event_name: Name of the old event handler
            *args, **kwargs: Event handler arguments
        """
        if event_name in EVENT_HANDLER_MAPPINGS:
            new_handler_name = EVENT_HANDLER_MAPPINGS[event_name]
            logger.debug(f"Routing legacy event '{event_name}' to '{new_handler_name}'")
            
            # Get the new handler method
            if hasattr(self.sidebar, new_handler_name):
                handler = getattr(self.sidebar, new_handler_name)
                return handler(*args, **kwargs)
            else:
                logger.warning(f"New handler '{new_handler_name}' not found")
        else:
            logger.warning(f"No mapping for legacy event '{event_name}'")
    
    def get_sidebar_state(self) -> dict:
        """
        Get the current state in legacy format for backward compatibility.
        
        Returns:
            Dictionary with state information in the old format
        """
        state = self.sidebar.state
        
        # Map new state to old format
        legacy_state = {
            # Settings from left sidebar
            "provider": self._safe_get_value("settings-provider"),
            "model": self._safe_get_value("settings-model"),
            "temperature": self._safe_get_value("settings-temperature"),
            "system_prompt": self._safe_get_value("settings-system-prompt"),
            
            # Session info from right sidebar
            "conversation_id": self._safe_get_value("session-chat-id"),
            "conversation_title": self._safe_get_value("session-title"),
            "conversation_keywords": self._safe_get_value("session-keywords"),
            "is_ephemeral": "Temp" in str(self._safe_get_value("session-chat-id", "")),
            
            # Advanced settings
            "advanced_mode": state.advanced_mode,
            "rag_enabled": self._safe_get_value("settings-rag-enabled", False),
        }
        
        return legacy_state
    
    def _safe_get_value(self, widget_id: str, default: Any = None) -> Any:
        """Safely get value from a widget."""
        try:
            widget = self.sidebar.query_one(f"#{widget_id}")
            if hasattr(widget, 'value'):
                return widget.value
            elif hasattr(widget, 'text'):
                return widget.text
            else:
                return default
        except:
            return default
    
    def ensure_tab_for_legacy_widget(self, old_widget_id: str):
        """
        Ensure the correct tab is active for accessing a legacy widget.
        
        Args:
            old_widget_id: The old widget ID being accessed
        """
        # Determine which tab contains this widget
        if any(x in old_widget_id for x in ["api", "model", "temperature", "system", "rag"]):
            self.sidebar.action_switch_tab("settings")
        elif any(x in old_widget_id for x in ["conversation", "save", "clone", "character"]):
            self.sidebar.action_switch_tab("session")
        elif any(x in old_widget_id for x in ["media", "notes", "prompt", "search"]):
            self.sidebar.action_switch_tab("content")

#
#######################################################################################################################
#
# Helper Functions for Migration
#

def create_compatibility_adapter(app_instance: 'TldwCli') -> Optional[LegacySidebarAdapter]:
    """
    Create a compatibility adapter for the app instance.
    
    Args:
        app_instance: The TldwCli app instance
        
    Returns:
        LegacySidebarAdapter instance or None if unified sidebar not found
    """
    try:
        # Find the unified sidebar
        unified_sidebar = app_instance.query_one("UnifiedChatSidebar")
        return LegacySidebarAdapter(unified_sidebar, app_instance)
    except Exception as e:
        logger.error(f"Could not create compatibility adapter: {e}")
        return None

def migrate_event_handler(old_handler_name: str, app_instance: 'TldwCli', *args, **kwargs):
    """
    Helper to migrate old event handlers to new ones.
    
    Args:
        old_handler_name: Name of the old event handler
        app_instance: The app instance
        *args, **kwargs: Handler arguments
    """
    adapter = create_compatibility_adapter(app_instance)
    if adapter:
        return adapter.handle_legacy_event(old_handler_name, *args, **kwargs)
    else:
        logger.error(f"Could not migrate event handler '{old_handler_name}'")

#
# End of sidebar_compatibility.py
#######################################################################################################################