# chat_tab_container.py
# Description: Container widget that manages multiple chat sessions with tabs
#
# Imports
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
from ..Chat.chat_models import ChatSessionData
from ..config import get_cli_setting
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
            The tab ID of the created tab
        """
        # Check max tabs limit
        if len(self.sessions) >= self.max_tabs:
            self.app_instance.notify(
                f"Maximum number of tabs ({self.max_tabs}) reached",
                severity="warning"
            )
            return ""
        
        # Generate unique tab ID
        tab_id = str(uuid.uuid4())[:8]
        
        # Create session data
        session_data = ChatSessionData(
            tab_id=tab_id,
            title=title or f"Chat {len(self.sessions) + 1}"
        )
        
        # Create session widget
        session = ChatSession(
            self.app_instance,
            session_data,
            id=f"chat-session-{tab_id}",
            classes="chat-session"
        )
        
        # Add to container
        container = self.query_one("#chat-sessions-container", Container)
        await container.mount(session)
        
        # Hide by default
        session.styles.display = "none"
        
        # Store session
        self.sessions[tab_id] = session
        
        # Add tab to tab bar
        if self.tab_bar:
            self.tab_bar.add_tab(session_data)
        
        # Hide placeholder if this is the first session
        if len(self.sessions) == 1:
            placeholder = self.query_one("#no-sessions-placeholder", Static)
            placeholder.styles.display = "none"
        
        logger.info(f"Created new chat tab: {tab_id}")
        return tab_id
    
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
            # Show confirmation dialog (simplified for now)
            self.app_instance.notify(
                "This chat has unsaved changes. Are you sure you want to close it?",
                severity="warning"
            )
            # TODO: Implement proper confirmation dialog
            return
        
        # Remove from tab bar
        if self.tab_bar:
            self.tab_bar.remove_tab(tab_id)
        
        # Remove session widget
        await session.remove()
        
        # Remove from sessions dict
        del self.sessions[tab_id]
        
        # Show placeholder if no sessions left
        if not self.sessions:
            placeholder = self.query_one("#no-sessions-placeholder", Static)
            placeholder.styles.display = "block"
            self.active_session_id = None
        
        logger.info(f"Closed chat tab: {tab_id}")
    
    def switch_to_tab(self, tab_id: str) -> None:
        """
        Switch to a specific tab.
        
        Args:
            tab_id: The ID of the tab to switch to
        """
        if tab_id not in self.sessions:
            logger.warning(f"Attempted to switch to non-existent tab: {tab_id}")
            return
        
        # Hide all sessions
        for session in self.sessions.values():
            session.styles.display = "none"
        
        # Show selected session
        self.sessions[tab_id].styles.display = "block"
        
        # Update active session
        self.active_session_id = tab_id
        
        # Update tab bar
        if self.tab_bar:
            self.tab_bar.set_active_tab(tab_id)
        
        # Focus the input area
        try:
            input_area = self.sessions[tab_id].get_chat_input()
            input_area.focus()
        except Exception as e:
            logger.debug(f"Could not focus input area: {e}")
        
        logger.debug(f"Switched to tab: {tab_id}")
    
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

#
# End of chat_tab_container.py
#######################################################################################################################