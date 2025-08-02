"""
Event Dispatcher - Centralized button event routing system.

This module provides a clean, extensible architecture for handling button
press events throughout the application.
"""

from typing import TYPE_CHECKING, Dict, Callable, Optional, Any
from textual.widgets import Button
from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EventDispatcher:
    """Centralized dispatcher for button events."""
    
    def __init__(self, app: 'TldwCli'):
        """
        Initialize the event dispatcher.
        
        Args:
            app: The TldwCli application instance
        """
        self.app = app
        self.logger = logger
        self._handler_map: Dict[str, Callable] = {}
        self._prefix_handlers: Dict[str, Callable] = {}
        self._build_handler_map()
    
    def _build_handler_map(self) -> None:
        """Build the master handler map from all event modules."""
        # Import here to avoid circular imports
        from tldw_chatbook.Event_Handlers import (
            tab_events, chat_events, conv_char_events, notes_events,
            media_events, search_events, llm_nav_events, ingest_events,
            embeddings_events, subscription_events, template_events
        )
        from tldw_chatbook.Event_Handlers.Chat_Events import (
            chat_events_sidebar, chat_events_tabs, chat_events_worldbooks,
            chat_events_dictionaries
        )
        from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
            llm_management_events, llm_management_events_llamacpp,
            llm_management_events_llamafile, llm_management_events_vllm,
            llm_management_events_ollama, llm_management_events_mlx_lm,
            llm_management_events_onnx, llm_management_events_transformers
        )
        
        # Build the complete handler map
        self._handler_map = {
            # Tab navigation (handled specially)
            **self._get_tab_handlers(),
            
            # LLM Management
            **llm_management_events.LLM_MANAGEMENT_BUTTON_HANDLERS,
            **llm_nav_events.LLM_NAV_BUTTON_HANDLERS,
            **llm_management_events_llamacpp.LLAMACPP_BUTTON_HANDLERS,
            **llm_management_events_llamafile.LLAMAFILE_BUTTON_HANDLERS,
            **llm_management_events_vllm.VLLM_BUTTON_HANDLERS,
            **llm_management_events_ollama.OLLAMA_BUTTON_HANDLERS,
            **llm_management_events_mlx_lm.MLX_LM_BUTTON_HANDLERS,
            **llm_management_events_onnx.ONNX_BUTTON_HANDLERS,
            **llm_management_events_transformers.TRANSFORMERS_BUTTON_HANDLERS,
            
            # Chat
            **chat_events.CHAT_BUTTON_HANDLERS,
            **chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS,
            **chat_events_tabs.CHAT_TABS_BUTTON_HANDLERS,
            **chat_events_worldbooks.CHAT_WORLDBOOKS_BUTTON_HANDLERS,
            **chat_events_dictionaries.CHAT_DICTIONARIES_BUTTON_HANDLERS,
            
            # Conversations/Characters/Prompts
            **conv_char_events.CCP_BUTTON_HANDLERS,
            
            # Notes
            **notes_events.NOTES_BUTTON_HANDLERS,
            
            # Media
            **media_events.MEDIA_BUTTON_HANDLERS,
            
            # Search
            **search_events.SEARCH_BUTTON_HANDLERS,
            
            # Ingest
            **ingest_events.INGEST_BUTTON_HANDLERS,
            
            # Other modules
            **embeddings_events.EMBEDDINGS_BUTTON_HANDLERS,
            **subscription_events.SUBSCRIPTION_BUTTON_HANDLERS,
            **template_events.TEMPLATE_BUTTON_HANDLERS,
        }
        
        # Build prefix handlers for dynamic routing
        self._prefix_handlers = {
            "tab-": self._handle_tab_button,
            "llm-nav-": self._create_nav_handler("llm", "llm_active_view"),
            "ingest-nav-": self._create_nav_handler("ingest", "ingest_active_view"),
            "search-nav-": self._create_nav_handler("search", "search_active_sub_tab"),
            "tools-settings-nav-": self._create_nav_handler("tools-settings", "tools_settings_active_view"),
            "media-nav-": self._create_nav_handler("media", "media_active_view"),
        }
        
        self.logger.info(f"Event dispatcher initialized with {len(self._handler_map)} handlers")
    
    def _get_tab_handlers(self) -> Dict[str, Callable]:
        """Get tab button handlers."""
        from tldw_chatbook.Event_Handlers import tab_events
        return {
            f"tab-{tab_id}": tab_events.handle_tab_button_pressed
            for tab_id in ["chat", "ccp", "notes", "search", "llm", "media", 
                          "ingest", "tools_settings", "evals", "coding", "stts", 
                          "study", "subscriptions", "chatbooks"]
        }
    
    def _create_nav_handler(self, prefix: str, reactive_attr: str) -> Callable:
        """Create a navigation handler for a specific prefix."""
        async def handler(app: 'TldwCli', event: Button.Pressed) -> None:
            """Generic handler for switching views within a tab."""
            view_to_activate = event.button.id.replace(f"{prefix}-nav-", f"{prefix}-view-")
            self.logger.info(
                f"Nav button '{event.button.id}' pressed. "
                f"Activating view '{view_to_activate}'"
            )
            setattr(app, reactive_attr, view_to_activate)
        
        return handler
    
    async def _handle_tab_button(self, app: 'TldwCli', event: Button.Pressed) -> None:
        """Handle tab button presses."""
        from tldw_chatbook.Event_Handlers import tab_events
        await tab_events.handle_tab_button_pressed(app, event)
    
    def _create_sidebar_toggle_handler(self, reactive_attr: str) -> Callable:
        """Create a sidebar toggle handler."""
        async def handler(app: 'TldwCli', event: Button.Pressed) -> None:
            """Toggle a sidebar's collapsed state."""
            setattr(app, reactive_attr, not getattr(app, reactive_attr))
        
        return handler
    
    async def dispatch(self, event: Button.Pressed) -> bool:
        """
        Dispatch a button press event to the appropriate handler.
        
        Args:
            event: The button pressed event
            
        Returns:
            True if the event was handled, False otherwise
        """
        button_id = event.button.id
        if not button_id:
            return False
        
        self.logger.info(f"Dispatching button press: '{button_id}'")
        
        # Check exact match first
        if button_id in self._handler_map:
            handler = self._handler_map[button_id]
            try:
                await handler(self.app, event)
                return True
            except Exception as e:
                self.logger.error(
                    f"Error in handler for button '{button_id}': {e}",
                    exc_info=True
                )
                return False
        
        # Check prefix handlers
        for prefix, handler in self._prefix_handlers.items():
            if button_id.startswith(prefix):
                try:
                    await handler(self.app, event)
                    return True
                except Exception as e:
                    self.logger.error(
                        f"Error in prefix handler for button '{button_id}': {e}",
                        exc_info=True
                    )
                    return False
        
        # Try window delegation as fallback
        return await self._try_window_delegation(event)
    
    async def _try_window_delegation(self, event: Button.Pressed) -> bool:
        """
        Try to delegate button press to the appropriate window component.
        
        Args:
            event: The button pressed event
            
        Returns:
            True if handled by a window, False otherwise
        """
        # Import here to avoid circular imports
        from tldw_chatbook.Constants import (
            TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH,
            TAB_LLM, TAB_INGEST, TAB_TOOLS_SETTINGS, TAB_EVALS
        )
        
        # Map tabs to their window IDs
        window_id_map = {
            TAB_CHAT: "chat-window",
            TAB_CCP: "conversations_characters_prompts-window",
            TAB_NOTES: "notes-window",
            TAB_MEDIA: "media-window",
            TAB_SEARCH: "search-window",
            TAB_LLM: "llm-window",
            TAB_INGEST: "ingest-window",
            TAB_TOOLS_SETTINGS: "tools_settings-window",
            TAB_EVALS: "evals-window",
        }
        
        current_tab = self.app.current_tab
        window_id = window_id_map.get(current_tab)
        
        if not window_id:
            return False
        
        try:
            window = self.app.query_one(f"#{window_id}")
            
            # Check if window has button handler
            if hasattr(window, "on_button_pressed") and callable(window.on_button_pressed):
                self.logger.info(f"Delegating to window's on_button_pressed")
                result = window.on_button_pressed(event)
                
                # Handle async results
                if hasattr(result, "__await__"):
                    await result
                
                return True
                
        except Exception as e:
            self.logger.debug(f"Window delegation failed: {e}")
        
        return False