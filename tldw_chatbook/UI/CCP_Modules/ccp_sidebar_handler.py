"""Handler for sidebar operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger
from textual.widgets import Collapsible, Button, Input, TextArea
from textual.css.query import NoMatches

from .ccp_messages import SidebarMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPSidebarHandler")


class CCPSidebarHandler:
    """Handles all sidebar-related operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the sidebar handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.sidebar_collapsed: bool = False
        self.active_section: Optional[str] = None
        self.section_states: Dict[str, bool] = {}  # Track collapsed state of sections
        
        logger.debug("CCPSidebarHandler initialized")
    
    async def toggle_sidebar(self) -> None:
        """Toggle the sidebar visibility."""
        try:
            sidebar = self.window.query_one("#ccp-sidebar")
            toggle_button = self.window.query_one("#toggle-ccp-sidebar")
            
            if self.sidebar_collapsed:
                # Show sidebar
                sidebar.remove_class("collapsed")
                toggle_button.label = "◀"  # Arrow pointing left (to collapse)
                self.sidebar_collapsed = False
                logger.info("Sidebar expanded")
            else:
                # Hide sidebar
                sidebar.add_class("collapsed")
                toggle_button.label = "▶"  # Arrow pointing right (to expand)
                self.sidebar_collapsed = True
                logger.info("Sidebar collapsed")
            
            # Post message for other components
            self.window.post_message(SidebarMessage.ToggleRequested())
            
        except NoMatches as e:
            logger.error(f"Sidebar element not found: {e}")
        except Exception as e:
            logger.error(f"Error toggling sidebar: {e}", exc_info=True)
    
    def expand_section(self, section_id: str) -> None:
        """Expand a specific collapsible section.
        
        Args:
            section_id: The ID of the section to expand
        """
        try:
            section = self.window.query_one(f"#{section_id}", Collapsible)
            if section.collapsed:
                section.collapsed = False
                self.section_states[section_id] = False
                logger.debug(f"Expanded section: {section_id}")
                
                # Post message
                self.window.post_message(
                    SidebarMessage.CollapsibleToggled(section_id, False)
                )
                
        except NoMatches:
            logger.warning(f"Section not found: {section_id}")
        except Exception as e:
            logger.error(f"Error expanding section {section_id}: {e}", exc_info=True)
    
    def collapse_section(self, section_id: str) -> None:
        """Collapse a specific collapsible section.
        
        Args:
            section_id: The ID of the section to collapse
        """
        try:
            section = self.window.query_one(f"#{section_id}", Collapsible)
            if not section.collapsed:
                section.collapsed = True
                self.section_states[section_id] = True
                logger.debug(f"Collapsed section: {section_id}")
                
                # Post message
                self.window.post_message(
                    SidebarMessage.CollapsibleToggled(section_id, True)
                )
                
        except NoMatches:
            logger.warning(f"Section not found: {section_id}")
        except Exception as e:
            logger.error(f"Error collapsing section {section_id}: {e}", exc_info=True)
    
    def toggle_section(self, section_id: str) -> None:
        """Toggle a specific collapsible section.
        
        Args:
            section_id: The ID of the section to toggle
        """
        try:
            section = self.window.query_one(f"#{section_id}", Collapsible)
            section.collapsed = not section.collapsed
            self.section_states[section_id] = section.collapsed
            
            logger.debug(f"Toggled section {section_id}: collapsed={section.collapsed}")
            
            # Post message
            self.window.post_message(
                SidebarMessage.CollapsibleToggled(section_id, section.collapsed)
            )
            
        except NoMatches:
            logger.warning(f"Section not found: {section_id}")
        except Exception as e:
            logger.error(f"Error toggling section {section_id}: {e}", exc_info=True)
    
    def set_active_section(self, section_id: str) -> None:
        """Set the active section and ensure it's visible.
        
        Args:
            section_id: The ID of the section to make active
        """
        try:
            # Expand the target section
            self.expand_section(section_id)
            
            # Optionally collapse other sections based on configuration
            if self.app_instance.app_config.get("ccp", {}).get("auto_collapse_sections", True):
                self._collapse_other_sections(section_id)
            
            self.active_section = section_id
            logger.info(f"Set active section: {section_id}")
            
        except Exception as e:
            logger.error(f"Error setting active section: {e}", exc_info=True)
    
    def _collapse_other_sections(self, except_section: str) -> None:
        """Collapse all sections except the specified one.
        
        Args:
            except_section: The section ID to keep expanded
        """
        sections = [
            "ccp-characters-collapsible",
            "ccp-conversations-collapsible", 
            "ccp-prompts-collapsible",
            "ccp-dictionaries-collapsible",
            "ccp-worldbooks-collapsible"
        ]
        
        for section_id in sections:
            if section_id != except_section:
                self.collapse_section(section_id)
    
    def focus_search_input(self, search_type: str) -> None:
        """Focus a specific search input field.
        
        Args:
            search_type: Type of search ("conversation", "character", "prompt", etc.)
        """
        try:
            input_map = {
                "conversation": "#conv-char-search-input",
                "content": "#conv-char-keyword-search-input",
                "tags": "#conv-char-tags-search-input",
                "prompt": "#ccp-prompt-search-input",
                "worldbook": "#ccp-worldbook-search-input"
            }
            
            input_id = input_map.get(search_type)
            if input_id:
                search_input = self.window.query_one(input_id, Input)
                search_input.focus()
                
                # Expand the relevant section
                section_map = {
                    "conversation": "ccp-conversations-collapsible",
                    "content": "ccp-conversations-collapsible",
                    "tags": "ccp-conversations-collapsible",
                    "prompt": "ccp-prompts-collapsible",
                    "worldbook": "ccp-worldbooks-collapsible"
                }
                
                section_id = section_map.get(search_type)
                if section_id:
                    self.expand_section(section_id)
                
                # Post message
                self.window.post_message(SidebarMessage.SearchFocused(search_type))
                
                logger.debug(f"Focused search input for: {search_type}")
            else:
                logger.warning(f"Unknown search type: {search_type}")
                
        except NoMatches as e:
            logger.error(f"Search input not found: {e}")
        except Exception as e:
            logger.error(f"Error focusing search input: {e}", exc_info=True)
    
    def update_conversation_details(self, title: str, keywords: str) -> None:
        """Update the conversation details in the sidebar.
        
        Args:
            title: The conversation title
            keywords: The conversation keywords
        """
        try:
            title_input = self.window.query_one("#conv-char-title-input", Input)
            keywords_input = self.window.query_one("#conv-char-keywords-input", TextArea)
            
            title_input.value = title
            keywords_input.text = keywords
            
            logger.debug(f"Updated conversation details: {title}")
            
        except NoMatches as e:
            logger.error(f"Detail inputs not found: {e}")
        except Exception as e:
            logger.error(f"Error updating conversation details: {e}", exc_info=True)
    
    def clear_conversation_details(self) -> None:
        """Clear the conversation details in the sidebar."""
        self.update_conversation_details("", "")
    
    def enable_conversation_controls(self, enabled: bool = True) -> None:
        """Enable or disable conversation-related controls.
        
        Args:
            enabled: Whether to enable the controls
        """
        try:
            controls = [
                "#conv-char-title-input",
                "#conv-char-keywords-input",
                "#conv-char-save-details-button",
                "#conv-char-export-text-button",
                "#conv-char-export-json-button"
            ]
            
            for control_id in controls:
                try:
                    widget = self.window.query_one(control_id)
                    widget.disabled = not enabled
                except NoMatches:
                    continue
            
            logger.debug(f"Conversation controls {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error setting conversation controls: {e}", exc_info=True)
    
    def show_context_buttons(self, context: str) -> None:
        """Show context-appropriate buttons in the sidebar.
        
        Args:
            context: The current context ("conversation", "character", "prompt", "dictionary")
        """
        try:
            # Hide all context buttons first
            button_groups = {
                "conversation": ["#conv-char-export-text-button", "#conv-char-export-json-button"],
                "character": ["#ccp-character-delete-button", "#ccp-export-character-button"],
                "prompt": ["#ccp-editor-prompt-delete-button"],
                "dictionary": ["#ccp-dict-delete-button", "#ccp-dict-clone-button"]
            }
            
            # Hide all buttons
            for buttons in button_groups.values():
                for button_id in buttons:
                    try:
                        button = self.window.query_one(button_id, Button)
                        button.add_class("hidden")
                    except NoMatches:
                        continue
            
            # Show buttons for current context
            if context in button_groups:
                for button_id in button_groups[context]:
                    try:
                        button = self.window.query_one(button_id, Button)
                        button.remove_class("hidden")
                    except NoMatches:
                        continue
            
            logger.debug(f"Showing buttons for context: {context}")
            
        except Exception as e:
            logger.error(f"Error showing context buttons: {e}", exc_info=True)
    
    def restore_section_states(self) -> None:
        """Restore the collapsed/expanded state of all sections."""
        for section_id, collapsed in self.section_states.items():
            try:
                section = self.window.query_one(f"#{section_id}", Collapsible)
                section.collapsed = collapsed
            except NoMatches:
                continue
            except Exception as e:
                logger.warning(f"Error restoring section {section_id}: {e}")
    
    def get_sidebar_width(self) -> str:
        """Get the current sidebar width setting.
        
        Returns:
            The sidebar width as a CSS value (e.g., "25%", "300px")
        """
        default_width = "25%"
        try:
            return self.app_instance.app_config.get("ccp", {}).get("sidebar_width", default_width)
        except Exception:
            return default_width
    
    def set_sidebar_width(self, width: str) -> None:
        """Set the sidebar width.
        
        Args:
            width: The width as a CSS value (e.g., "25%", "300px")
        """
        try:
            sidebar = self.window.query_one("#ccp-sidebar")
            sidebar.styles.width = width
            logger.debug(f"Set sidebar width to: {width}")
        except NoMatches:
            logger.error("Sidebar not found")
        except Exception as e:
            logger.error(f"Error setting sidebar width: {e}", exc_info=True)