# chat_tab_bar.py
# Description: Widget for managing chat tabs navigation
#
# Imports
from typing import TYPE_CHECKING, List, Optional, Dict
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll
from textual.widgets import Button, Static
from textual.reactive import reactive
from textual.message import Message
#
# Local Imports
from ..Chat.chat_models import ChatSessionData
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Classes:

class ChatTabBar(Horizontal):
    """
    A tab bar widget for managing multiple chat sessions.
    
    Displays tabs for each chat session with the ability to switch between them,
    create new tabs, and close existing ones.
    """
    
    # Messages for tab events
    class TabSelected(Message):
        """Sent when a tab is selected."""
        def __init__(self, tab_id: str) -> None:
            super().__init__()
            self.tab_id = tab_id
    
    class TabClosed(Message):
        """Sent when a tab close is requested."""
        def __init__(self, tab_id: str) -> None:
            super().__init__()
            self.tab_id = tab_id
    
    class NewTabRequested(Message):
        """Sent when a new tab is requested."""
        pass
    
    # Current active tab
    active_tab_id: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "chat-tab-bar"
        self.tab_buttons: Dict[str, Button] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the tab bar UI."""
        with HorizontalScroll(id="chat-tabs-scroll", classes="chat-tabs-scroll"):
            # New tab button
            yield Button(
                "+",
                id="new-chat-tab-button",
                classes="new-tab-button",
                tooltip="New chat tab (Ctrl+T)"
            )
    
    def add_tab(self, session_data: ChatSessionData) -> None:
        """
        Add a new tab to the tab bar.
        
        Args:
            session_data: The session data for the new tab
        """
        tab_id = session_data.tab_id
        
        # Create tab button with close button
        tab_container = Horizontal(
            classes="chat-tab-container",
            id=f"tab-container-{tab_id}"
        )
        
        # Character icon if assigned
        icon = ""
        if session_data.character_name:
            icon = "👤 "
        
        # Tab button
        tab_button = Button(
            f"{icon}{session_data.title}",
            id=f"chat-tab-{tab_id}",
            classes="chat-tab",
            name=tab_id  # Store tab_id in name for easy access
        )
        
        # Close button
        close_button = Button(
            "×",
            id=f"close-tab-{tab_id}",
            classes="close-tab-button",
            name=tab_id  # Store tab_id in name for easy access
        )
        
        # Mount the buttons in the scroll container
        scroll_container = self.query_one("#chat-tabs-scroll", HorizontalScroll)
        
        # Mount before the new tab button
        new_tab_button = self.query_one("#new-chat-tab-button", Button)
        tab_container.mount(tab_button)
        tab_container.mount(close_button)
        scroll_container.mount(tab_container, before=new_tab_button)
        
        # Store reference
        self.tab_buttons[tab_id] = tab_button
        
        # If this is the first tab or no active tab, make it active
        if not self.active_tab_id:
            self.set_active_tab(tab_id)
        
        logger.info(f"Added tab: {tab_id} with title: {session_data.title}")
    
    def remove_tab(self, tab_id: str) -> None:
        """
        Remove a tab from the tab bar.
        
        Args:
            tab_id: The ID of the tab to remove
        """
        if tab_id not in self.tab_buttons:
            return
        
        # Remove the tab container
        try:
            tab_container = self.query_one(f"#tab-container-{tab_id}")
            tab_container.remove()
        except Exception as e:
            logger.error(f"Error removing tab container: {e}")
        
        # Remove from tracking
        del self.tab_buttons[tab_id]
        
        # If this was the active tab, activate another
        if self.active_tab_id == tab_id:
            if self.tab_buttons:
                # Activate the first remaining tab
                next_tab_id = next(iter(self.tab_buttons.keys()))
                self.set_active_tab(next_tab_id)
            else:
                self.active_tab_id = None
        
        logger.info(f"Removed tab: {tab_id}")
    
    def set_active_tab(self, tab_id: str) -> None:
        """
        Set the active tab.
        
        Args:
            tab_id: The ID of the tab to activate
        """
        if tab_id not in self.tab_buttons:
            logger.warning(f"Attempted to activate non-existent tab: {tab_id}")
            return
        
        # Remove active class from all tabs
        for button in self.tab_buttons.values():
            button.remove_class("active")
        
        # Add active class to selected tab
        self.tab_buttons[tab_id].add_class("active")
        self.active_tab_id = tab_id
        
        # Post message about tab selection
        self.post_message(self.TabSelected(tab_id))
        
        logger.debug(f"Activated tab: {tab_id}")
    
    def update_tab_title(self, tab_id: str, new_title: str, character_name: Optional[str] = None) -> None:
        """
        Update the title of a tab.
        
        Args:
            tab_id: The ID of the tab to update
            new_title: The new title for the tab
            character_name: Optional character name to show icon
        """
        if tab_id not in self.tab_buttons:
            return
        
        icon = ""
        if character_name:
            icon = "👤 "
        
        self.tab_buttons[tab_id].label = f"{icon}{new_title}"
        logger.debug(f"Updated tab {tab_id} title to: {new_title}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the tab bar."""
        button = event.button
        
        if button.id == "new-chat-tab-button":
            # Request new tab
            self.post_message(self.NewTabRequested())
            event.stop()
        elif button.id and button.id.startswith("chat-tab-"):
            # Tab selection
            tab_id = button.name
            if tab_id:
                self.set_active_tab(tab_id)
                event.stop()
        elif button.id and button.id.startswith("close-tab-"):
            # Tab close request
            tab_id = button.name
            if tab_id:
                self.post_message(self.TabClosed(tab_id))
                event.stop()
    
    def get_tab_count(self) -> int:
        """Get the number of tabs."""
        return len(self.tab_buttons)
    
    def get_tab_ids(self) -> List[str]:
        """Get list of all tab IDs."""
        return list(self.tab_buttons.keys())
    
    def get_next_tab_id(self, current_id: str) -> Optional[str]:
        """Get the next tab ID in order."""
        tab_ids = self.get_tab_ids()
        if not tab_ids or current_id not in tab_ids:
            return None
        
        current_index = tab_ids.index(current_id)
        next_index = (current_index + 1) % len(tab_ids)
        return tab_ids[next_index]
    
    def get_previous_tab_id(self, current_id: str) -> Optional[str]:
        """Get the previous tab ID in order."""
        tab_ids = self.get_tab_ids()
        if not tab_ids or current_id not in tab_ids:
            return None
        
        current_index = tab_ids.index(current_id)
        prev_index = (current_index - 1) % len(tab_ids)
        return tab_ids[prev_index]

#
# End of chat_tab_bar.py
#######################################################################################################################