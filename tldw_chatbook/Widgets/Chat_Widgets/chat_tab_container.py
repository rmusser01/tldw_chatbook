# chat_tab_container.py
# Description: Container widget that manages multiple chat sessions with tabs
#
# Imports
import re
import uuid
from typing import TYPE_CHECKING, Dict, Optional, Tuple
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
#
# Local Imports
from .chat_tab_bar import ChatTabBar
from .chat_session import ChatSession
from tldw_chatbook.Widgets.confirmation_dialog import UnsavedChangesDialog
from tldw_chatbook.Chat.chat_conversation_service import derive_conversation_title
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.input_validation import validate_text_input
#
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
#
#######################################################################################################################
#
# Classes:


def _derive_session_title(
    session_data: ChatSessionData,
    fallback_title: Optional[str] = None,
) -> str:
    effective_fallback_title = session_data.title if fallback_title is None else fallback_title
    assistant_name = None
    if session_data.assistant_kind == "character":
        assistant_name = session_data.character_name
    elif session_data.assistant_kind == "persona" and session_data.assistant_id:
        assistant_name = f"Persona {session_data.assistant_id}"

    return derive_conversation_title(
        assistant_kind=session_data.assistant_kind,
        assistant_name=assistant_name,
        fallback_title=effective_fallback_title,
        character_id=session_data.character_id,
    )


def _session_reuse_key(session_data: ChatSessionData) -> Tuple[str, Optional[str]]:
    return (session_data.runtime_backend, session_data.conversation_id)

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

    class ActiveSessionChanged(Message):
        """Sent when the live active chat session changes."""

        def __init__(self, session_data: Optional[ChatSessionData]) -> None:
            super().__init__()
            self.session_data = session_data
    
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
    
    async def create_new_tab(
        self,
        title: Optional[str] = None,
        session_data: Optional[ChatSessionData] = None,
    ) -> str:
        """
        Create a new chat tab.
        
        Args:
            title: Optional title for the tab
            session_data: Optional session contract to seed the new tab

        Returns:
            The tab ID of the created tab, or empty string on failure
        """
        try:
            if session_data is not None and session_data.conversation_id:
                reuse_key = _session_reuse_key(session_data)
                for existing_tab_id, existing_session in self.sessions.items():
                    existing_session_data = existing_session.session_data
                    if not existing_session_data.conversation_id:
                        continue
                    if _session_reuse_key(existing_session_data) == reuse_key:
                        logger.info(
                            f"Reusing existing chat tab {existing_tab_id} for conversation {reuse_key}"
                        )
                        await self.switch_to_tab_async(existing_tab_id)
                        return existing_tab_id

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
            if session_data is not None:
                session_data = ChatSessionData.from_dict(session_data.to_dict())
                session_data.tab_id = tab_id
                session_data.title = _derive_session_title(
                    session_data,
                    fallback_title=title or session_data.title,
                )
            else:
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
        
        # Remove session widget. In tests and recovery paths a session may
        # already be detached from a live Textual app; state cleanup should
        # still proceed.
        try:
            await session.remove()
        except Exception as e:
            logger.debug(f"Could not remove chat session widget {tab_id}: {e}")
        
        # Remove from sessions dict
        del self.sessions[tab_id]
        
        # Handle UI state after closing
        if not self.sessions:
            # Show placeholder if no sessions left
            placeholder = self.query_one("#no-sessions-placeholder", Static)
            placeholder.styles.display = "block"
            self.active_session_id = None
            self._publish_active_session_changed(None)
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
                    session = self.sessions[tab_id]
                    await session.resume()
                    self._publish_active_session_changed(session.session_data)
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

            self._publish_active_session_changed(new_session.session_data)
            
            logger.debug(f"Switched to tab: {tab_id}")
            
        except Exception as e:
            logger.error(f"Unexpected error switching to tab {tab_id}: {e}")
            self.app_instance.notify("Error switching tabs", severity="error")

    def _publish_active_session_changed(
        self,
        session_data: Optional[ChatSessionData],
    ) -> None:
        """Notify listeners that the live active chat session changed."""
        try:
            self.post_message(self.ActiveSessionChanged(session_data))
        except Exception as e:
            logger.debug(f"Could not publish active session change: {e}")
    
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
    
    def get_all_sessions_state(self) -> Dict[str, ChatSessionData]:
        """
        Get the state of all sessions for saving.
        
        Returns:
            Dictionary mapping tab IDs to session data
        """
        state = {}
        for tab_id, session in self.sessions.items():
            state[tab_id] = session.session_data
        return state
    
    async def restore_sessions_from_state(self, state: Dict[str, ChatSessionData]) -> None:
        """
        Restore sessions from saved state.
        
        Args:
            state: Dictionary mapping tab IDs to session data
        """
        # Clear all existing live tabs so restore replaces the mounted UI state.
        for tab_id in list(self.sessions.keys()):
            await self._force_close_tab(tab_id)

        restored_reuse_keys = set()

        # Restore each session
        for _, session_data in state.items():
            restored_session_data = ChatSessionData.from_dict(session_data.to_dict())
            restored_session_data.title = _derive_session_title(restored_session_data)
            reuse_key = None
            if restored_session_data.conversation_id:
                reuse_key = _session_reuse_key(restored_session_data)

            # Create new session
            new_tab_id = await self.create_new_tab(session_data=restored_session_data)
            if new_tab_id and new_tab_id in self.sessions:
                if reuse_key is not None and reuse_key in restored_reuse_keys:
                    continue

                # Preserve the saved session state while rebinding it to the new widget tab ID.
                restored_session_data.tab_id = new_tab_id
                self.sessions[new_tab_id].session_data = restored_session_data
                if self.tab_bar:
                    self.tab_bar.update_tab_title(
                        new_tab_id,
                        restored_session_data.title,
                        restored_session_data.character_name,
                    )
                if reuse_key is not None:
                    restored_reuse_keys.add(reuse_key)

#
# End of chat_tab_container.py
#######################################################################################################################
