# chat_tab_container.py
# Description: Container widget that manages multiple chat sessions with tabs
#
# Imports
import re
import uuid
from typing import TYPE_CHECKING, Dict, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static
from textual.reactive import reactive
from textual.binding import Binding
#
# Local Imports
from .chat_tab_bar import ChatTabBar
from .chat_session import ChatSession
from .confirmation_dialog import UnsavedChangesDialog
from ..Chat.chat_models import ChatSessionData
from ..config import get_cli_setting
from ..Utils.input_validation import validate_text_input
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Classes:

class ChatTabContainer(Container):
    """
    Container that manages multiple chat sessions with a tab interface.
    
    This widget contains a ChatTabBar at the top and displays the active
    ChatSession below it. It handles tab creation, deletion, and switching.
    """
    
    BINDINGS = [
        Binding("ctrl+t", "new_tab", "New Chat Tab", show=False),
        Binding("ctrl+w", "close_tab", "Close Current Tab", show=False),
        Binding("ctrl+tab", "next_tab", "Next Tab", show=False),
        Binding("ctrl+shift+tab", "previous_tab", "Previous Tab", show=False),
    ]
    
    # Current active session
    active_session_id: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sessions: Dict[str, ChatSession] = {}
        self.tab_bar: Optional[ChatTabBar] = None
        self.max_tabs = get_cli_setting("chat", "max_tabs", 10)
        # Pattern for valid tab IDs
        self._tab_id_pattern = re.compile(r'^[a-f0-9]{8}$')
        logger.debug(f"ChatTabContainer initialized with max_tabs: {self.max_tabs}")
    
    def compose(self) -> ComposeResult:
        """Compose the tab container UI."""
        # Tab bar at the top
        self.tab_bar = ChatTabBar()
        yield self.tab_bar
        
        # Container for chat sessions
        with Container(id="chat-sessions-container", classes="chat-sessions-container"):
            # Placeholder when no sessions
            yield Static(
                "Click '+' to start a new chat",
                id="no-sessions-placeholder",
                classes="no-sessions-placeholder"
            )
    
    async def on_mount(self) -> None:
        """Initialize with a default tab when mounted."""
        # Create initial tab
        await self.create_new_tab()
    
    async def create_new_tab(self, title: Optional[str] = None) -> str:
        """
        Create a new chat tab.
        
        Args:
            title: Optional title for the tab
            
        Returns:
            The tab ID of the created tab, or empty string on failure
        """
        try:
            # Check max tabs limit
            if len(self.sessions) >= self.max_tabs:
                self.app_instance.notify(
                    f"Maximum number of tabs ({self.max_tabs}) reached",
                    severity="warning"
                )
                return ""
            
            # Validate and sanitize title
            if title:
                try:
                    title = validate_text_input(title, max_length=100)
                except Exception as e:
                    logger.warning(f"Invalid tab title, using default: {e}")
                    title = None
            
            # Generate unique tab ID
            tab_id = str(uuid.uuid4())[:8]
            
            # Ensure tab ID is unique
            attempts = 0
            while tab_id in self.sessions and attempts < 10:
                tab_id = str(uuid.uuid4())[:8]
                attempts += 1
            
            if tab_id in self.sessions:
                logger.error("Failed to generate unique tab ID")
                self.app_instance.notify("Failed to create new tab", severity="error")
                return ""
            
            # Create session data
            session_data = ChatSessionData(
                tab_id=tab_id,
                title=title or f"Chat {len(self.sessions) + 1}"
            )
            
            # Create session widget
            try:
                session = ChatSession(
                    self.app_instance,
                    session_data,
                    id=f"chat-session-{tab_id}",
                    classes="chat-session"
                )
            except Exception as e:
                logger.error(f"Failed to create ChatSession widget: {e}")
                self.app_instance.notify("Failed to create chat session", severity="error")
                return ""
            
            # Add to container
            try:
                container = self.query_one("#chat-sessions-container", Container)
                await container.mount(session)
            except Exception as e:
                logger.error(f"Failed to mount session widget: {e}")
                self.app_instance.notify("Failed to add tab to container", severity="error")
                return ""
            
            # Hide by default
            session.styles.display = "none"
            
            # Store session
            self.sessions[tab_id] = session
            
            # Add tab to tab bar
            if self.tab_bar:
                try:
                    await self.tab_bar.add_tab(session_data)
                except Exception as e:
                    logger.error(f"Failed to add tab to tab bar: {e}")
                    # Continue anyway as the session is created
            
            # Hide placeholder if this is the first session
            if len(self.sessions) == 1:
                try:
                    placeholder = self.query_one("#no-sessions-placeholder", Static)
                    placeholder.styles.display = "none"
                except Exception as e:
                    logger.debug(f"Could not hide placeholder: {e}")
                
                # Automatically switch to the first tab
                await self.switch_to_tab_async(tab_id)
            
            logger.info(f"Created new chat tab: {tab_id}")
            return tab_id
            
        except Exception as e:
            logger.error(f"Unexpected error creating new tab: {e}")
            self.app_instance.notify("Failed to create new tab", severity="error")
            return ""
    
    async def close_tab(self, tab_id: str) -> None:
        """
        Close a chat tab.
        
        Args:
            tab_id: The ID of the tab to close
        """
        if tab_id not in self.sessions:
            return
        
        session = self.sessions[tab_id]
        
        # Check if this is a non-ephemeral chat that might have unsaved changes
        if not session.session_data.is_ephemeral and session.session_data.has_unsaved_changes:
            # Show confirmation dialog
            async def confirm_close():
                await self._force_close_tab(tab_id)
            
            dialog = UnsavedChangesDialog(
                tab_title=session.session_data.title,
                confirm_callback=confirm_close
            )
            await self.app_instance.push_screen(dialog)
            return
        
        # No unsaved changes, close directly
        await self._force_close_tab(tab_id)
    
    async def _force_close_tab(self, tab_id: str) -> None:
        """
        Force close a tab without confirmation.
        
        Args:
            tab_id: The ID of the tab to close
        """
        if tab_id not in self.sessions:
            return
            
        session = self.sessions[tab_id]
        
        # If this was the active tab, we need to switch to another
        was_active = self.active_session_id == tab_id
        
        # Clean up the session before removing
        await session.cleanup()
        
        # Remove from tab bar
        if self.tab_bar:
            self.tab_bar.remove_tab(tab_id)
        
        # Remove session widget
        await session.remove()
        
        # Remove from sessions dict
        del self.sessions[tab_id]
        
        # Handle UI state after closing
        if not self.sessions:
            # Show placeholder if no sessions left
            placeholder = self.query_one("#no-sessions-placeholder", Static)
            placeholder.styles.display = "block"
            self.active_session_id = None
        elif was_active:
            # Switch to another tab if this was active
            remaining_tabs = list(self.sessions.keys())
            if remaining_tabs:
                await self.switch_to_tab_async(remaining_tabs[0])
        
        logger.info(f"Closed chat tab: {tab_id}")
    
    def switch_to_tab(self, tab_id: str) -> None:
        """
        Switch to a specific tab (sync version for UI events).
        
        Args:
            tab_id: The ID of the tab to switch to
        """
        # Use call_later to schedule the async version
        self.app_instance.call_later(self.switch_to_tab_async, tab_id)
    
    async def switch_to_tab_async(self, tab_id: str) -> None:
        """
        Switch to a specific tab (async version with proper lifecycle management).
        
        Args:
            tab_id: The ID of the tab to switch to
        """
        try:
            # Validate tab ID
            if not self._validate_tab_id(tab_id):
                logger.error(f"Invalid tab ID format: {tab_id}")
                return
                
            if tab_id not in self.sessions:
                logger.warning(f"Attempted to switch to non-existent tab: {tab_id}")
                return
            
            # If switching to the same tab, just ensure it's resumed
            if self.active_session_id == tab_id:
                try:
                    await self.sessions[tab_id].resume()
                except Exception as e:
                    logger.error(f"Error resuming current tab {tab_id}: {e}")
                return
            
            # Suspend the current active session
            if self.active_session_id and self.active_session_id in self.sessions:
                old_session = self.sessions[self.active_session_id]
                try:
                    old_session.styles.display = "none"
                    await old_session.suspend()
                    old_session.session_data.is_active = False
                except Exception as e:
                    logger.error(f"Error suspending tab {self.active_session_id}: {e}")
                    # Continue with tab switch even if suspend fails
            
            # Update active session
            self.active_session_id = tab_id
            new_session = self.sessions[tab_id]
            
            # Show and resume the new session
            try:
                new_session.styles.display = "block"
                new_session.session_data.is_active = True
                await new_session.resume()
            except Exception as e:
                logger.error(f"Error resuming tab {tab_id}: {e}")
                # Still update UI even if resume fails
            
            # Update tab bar
            if self.tab_bar:
                try:
                    self.tab_bar.set_active_tab(tab_id)
                except Exception as e:
                    logger.warning(f"Error updating tab bar: {e}")
            
            logger.debug(f"Switched to tab: {tab_id}")
            
        except Exception as e:
            logger.error(f"Unexpected error switching to tab {tab_id}: {e}")
            self.app_instance.notify("Error switching tabs", severity="error")
    
    def get_active_session(self) -> Optional[ChatSession]:
        """Get the currently active chat session."""
        if self.active_session_id and self.active_session_id in self.sessions:
            return self.sessions[self.active_session_id]
        return None
    
    # Message handlers for tab bar events
    async def on_chat_tab_bar_tab_selected(self, message: ChatTabBar.TabSelected) -> None:
        """Handle tab selection from tab bar."""
        self.switch_to_tab(message.tab_id)
    
    async def on_chat_tab_bar_tab_closed(self, message: ChatTabBar.TabClosed) -> None:
        """Handle tab close request from tab bar."""
        await self.close_tab(message.tab_id)
    
    async def on_chat_tab_bar_new_tab_requested(self, message: ChatTabBar.NewTabRequested) -> None:
        """Handle new tab request from tab bar."""
        tab_id = await self.create_new_tab()
        if tab_id:
            self.switch_to_tab(tab_id)
    
    # Key binding actions
    async def action_new_tab(self) -> None:
        """Create a new tab via keyboard shortcut."""
        tab_id = await self.create_new_tab()
        if tab_id:
            self.switch_to_tab(tab_id)
    
    async def action_close_tab(self) -> None:
        """Close current tab via keyboard shortcut."""
        if self.active_session_id:
            await self.close_tab(self.active_session_id)
    
    def action_next_tab(self) -> None:
        """Switch to next tab via keyboard shortcut."""
        if self.active_session_id and self.tab_bar:
            next_id = self.tab_bar.get_next_tab_id(self.active_session_id)
            if next_id:
                self.switch_to_tab(next_id)
    
    def action_previous_tab(self) -> None:
        """Switch to previous tab via keyboard shortcut."""
        if self.active_session_id and self.tab_bar:
            prev_id = self.tab_bar.get_previous_tab_id(self.active_session_id)
            if prev_id:
                self.switch_to_tab(prev_id)
    
    def update_active_tab_title(self, new_title: str, character_name: Optional[str] = None) -> None:
        """Update the title of the active tab."""
        if not self.active_session_id:
            return
        
        session = self.sessions.get(self.active_session_id)
        if session:
            session.session_data.title = new_title
            session.session_data.character_name = character_name
            
            if self.tab_bar:
                self.tab_bar.update_tab_title(
                    self.active_session_id,
                    new_title,
                    character_name
                )
    
    def _validate_tab_id(self, tab_id: str) -> bool:
        """
        Validate a tab ID.
        
        Args:
            tab_id: The tab ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not tab_id:
            return False
        
        # Check format (8 hex characters)
        if not self._tab_id_pattern.match(tab_id):
            logger.warning(f"Invalid tab ID format: {tab_id}")
            return False
        
        return True
    
    def update_tab_title(self, tab_id: str, new_title: str, character_name: Optional[str] = None) -> None:
        """
        Update the title of a specific tab.
        
        Args:
            tab_id: The ID of the tab to update
            new_title: The new title
            character_name: Optional character name
        """
        if not self._validate_tab_id(tab_id):
            return
            
        if tab_id not in self.sessions:
            logger.warning(f"Tab {tab_id} not found")
            return
        
        session = self.sessions[tab_id]
        session.session_data.title = new_title
        session.session_data.character_name = character_name
        
        if self.tab_bar:
            self.tab_bar.update_tab_title(tab_id, new_title, character_name)

#
# End of chat_tab_container.py
#######################################################################################################################