"""
Chat Sidebar Handler Module

Handles all sidebar functionality including:
- Sidebar toggling (left/right)
- Character loading/clearing
- Prompt management
- Notes expansion
- Sidebar resizing
- Settings sidebar interactions
"""

from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual.widgets import Button, TextArea
from textual.css.query import NoMatches

if TYPE_CHECKING:
    from ..Chat_Window_Enhanced import ChatWindowEnhanced

logger = logger.bind(module="ChatSidebarHandler")


class ChatSidebarHandler:
    """Handles sidebar interactions and management."""
    
    def __init__(self, chat_window: 'ChatWindowEnhanced'):
        """Initialize the sidebar handler.
        
        Args:
            chat_window: Parent ChatWindowEnhanced instance
        """
        self.chat_window = chat_window
        self.app_instance = chat_window.app_instance
    
    async def handle_sidebar_toggle(self, sidebar_id: str, event):
        """Handle sidebar toggle button clicks.
        
        Args:
            sidebar_id: ID of the sidebar to toggle
            event: Button.Pressed event
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        logger.debug(f"Toggling sidebar: {sidebar_id}")
        await chat_events.handle_chat_tab_sidebar_toggle(self.app_instance, event)
    
    async def handle_character_buttons(self, button_id: str, event):
        """Handle character-related button clicks.
        
        Args:
            button_id: ID of the button pressed
            event: Button.Pressed event
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        character_handlers = {
            "chat-load-character-button": chat_events.handle_chat_load_character_button_pressed,
            "chat-clear-active-character-button": chat_events.handle_chat_clear_active_character_button_pressed,
        }
        
        if button_id in character_handlers:
            logger.debug(f"Handling character button: {button_id}")
            await character_handlers[button_id](self.app_instance, event)
    
    async def handle_prompt_buttons(self, button_id: str, event):
        """Handle prompt-related button clicks.
        
        Args:
            button_id: ID of the button pressed
            event: Button.Pressed event
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        prompt_handlers = {
            "chat-prompt-load-selected-button": chat_events.handle_chat_view_selected_prompt_button_pressed,
            "chat-prompt-copy-system-button": chat_events.handle_chat_copy_system_prompt_button_pressed,
            "chat-prompt-copy-user-button": chat_events.handle_chat_copy_user_prompt_button_pressed,
        }
        
        if button_id in prompt_handlers:
            logger.debug(f"Handling prompt button: {button_id}")
            await prompt_handlers[button_id](self.app_instance, event)
    
    async def handle_notes_expand_button(self, event):
        """Handle the notes expand/collapse button.
        
        Args:
            event: Button.Pressed event
        """
        try:
            # Use cached widgets if available, fallback to query
            button = self.chat_window._notes_expand_button if self.chat_window._notes_expand_button else self.app_instance.query_one("#chat-notes-expand-button", Button)
            textarea = self.chat_window._notes_textarea if self.chat_window._notes_textarea else self.app_instance.query_one("#chat-notes-content-textarea", TextArea)
            
            # Toggle between expanded and normal states
            if "notes-textarea-expanded" in textarea.classes:
                # Collapse
                textarea.remove_class("notes-textarea-expanded")
                textarea.add_class("notes-textarea-normal")
                textarea.styles.height = 10
                button.label = "Expand Notes"
            else:
                # Expand
                textarea.remove_class("notes-textarea-normal")
                textarea.add_class("notes-textarea-expanded")
                textarea.styles.height = 30
                button.label = "Collapse Notes"
                
            logger.debug(f"Notes area toggled - expanded: {'notes-textarea-expanded' in textarea.classes}")
            
        except NoMatches as e:
            logger.warning(f"Notes expand button or textarea not found: {e}")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error toggling notes area: {e}")
    
    def resize_sidebar(self, sidebar_id: str, direction: str):
        """Resize a sidebar.
        
        Args:
            sidebar_id: ID of the sidebar to resize
            direction: 'shrink' or 'expand'
        """
        try:
            sidebar = self.app_instance.query_one(f"#{sidebar_id}")
            current_width = sidebar.styles.width
            
            if direction == "shrink":
                # Decrease width
                if isinstance(current_width, int) and current_width > 20:
                    sidebar.styles.width = current_width - 5
                    logger.debug(f"Sidebar {sidebar_id} shrunk to {current_width - 5}")
            elif direction == "expand":
                # Increase width
                if isinstance(current_width, int) and current_width < 60:
                    sidebar.styles.width = current_width + 5
                    logger.debug(f"Sidebar {sidebar_id} expanded to {current_width + 5}")
                    
        except NoMatches:
            logger.warning(f"Sidebar {sidebar_id} not found")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error resizing sidebar: {e}")
    
    def toggle_sidebar_visibility(self, sidebar_id: str):
        """Toggle visibility of a sidebar.
        
        Args:
            sidebar_id: ID of the sidebar to toggle
        """
        try:
            sidebar = self.app_instance.query_one(f"#{sidebar_id}")
            sidebar.display = not sidebar.display
            
            # Update toggle button state
            button_id = f"toggle-{sidebar_id}"
            try:
                button = self.app_instance.query_one(f"#{button_id}", Button)
                if sidebar.display:
                    button.remove_class("sidebar-hidden")
                else:
                    button.add_class("sidebar-hidden")
            except NoMatches:
                pass
                
            logger.debug(f"Sidebar {sidebar_id} visibility toggled to {sidebar.display}")
            
        except NoMatches:
            logger.warning(f"Sidebar {sidebar_id} not found")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error toggling sidebar visibility: {e}")
    
    def update_sidebar_content(self, sidebar_id: str, content: str):
        """Update the content of a sidebar.
        
        Args:
            sidebar_id: ID of the sidebar to update
            content: New content for the sidebar
        """
        try:
            # Find the content area within the sidebar
            sidebar = self.app_instance.query_one(f"#{sidebar_id}")
            
            # Look for common content containers
            content_areas = sidebar.query("TextArea, Static, ListView")
            if content_areas:
                content_area = content_areas[0]
                if hasattr(content_area, 'value'):
                    content_area.value = content
                elif hasattr(content_area, 'update'):
                    content_area.update(content)
                    
                logger.debug(f"Updated sidebar {sidebar_id} content")
            else:
                logger.warning(f"No content area found in sidebar {sidebar_id}")
                
        except NoMatches:
            logger.warning(f"Sidebar {sidebar_id} not found")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error updating sidebar content: {e}")
    
    def get_sidebar_state(self, sidebar_id: str) -> dict:
        """Get the current state of a sidebar.
        
        Args:
            sidebar_id: ID of the sidebar
            
        Returns:
            Dictionary with sidebar state information
        """
        state = {
            "visible": False,
            "width": None,
            "collapsed": False
        }
        
        try:
            sidebar = self.app_instance.query_one(f"#{sidebar_id}")
            state["visible"] = sidebar.display
            state["width"] = sidebar.styles.width
            
            # Check if sidebar has collapsed class
            state["collapsed"] = "collapsed" in sidebar.classes
            
        except NoMatches:
            logger.debug(f"Sidebar {sidebar_id} not found")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error getting sidebar state: {e}")
        
        return state
    
    async def handle_sidebar_action(self, action: str, **kwargs):
        """Handle generic sidebar actions.
        
        Args:
            action: Action to perform
            **kwargs: Additional arguments for the action
        """
        actions = {
            "toggle": self.toggle_sidebar_visibility,
            "resize": self.resize_sidebar,
            "update": self.update_sidebar_content,
            "get_state": self.get_sidebar_state
        }
        
        if action in actions:
            result = actions[action](**kwargs)
            if asyncio.iscoroutine(result):
                await result
            return result
        else:
            logger.warning(f"Unknown sidebar action: {action}")