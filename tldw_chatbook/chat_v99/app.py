"""Main chat application following Textual patterns."""

from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive
from typing import Optional
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .screens.chat_screen import ChatScreen
    from .models import ChatSession, Settings
    from .messages import SessionChanged, SidebarToggled
except ImportError:
    # Fallback for direct execution
    from screens.chat_screen import ChatScreen
    from models import ChatSession, Settings
    from messages import SessionChanged, SidebarToggled


class ChatV99App(App):
    """Main chat application following Textual patterns.
    
    References:
    - App basics: https://textual.textualize.io/guide/app/
    - Reactive attributes: https://textual.textualize.io/guide/reactivity/#reactive-attributes
    """
    
    # Inline CSS per documentation
    CSS = """
    ChatV99App {
        background: $surface;
    }
    
    /* Define custom color scheme */
    $surface: #1e1e1e;
    $panel: #2a2a2a;
    $primary: #0066cc;
    $accent: #00aa66;
    $text: #ffffff;
    $text-muted: #888888;
    """
    
    TITLE = "Chat Interface v99"
    
    # Key bindings
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New Chat", priority=True),
        Binding("ctrl+s", "save_session", "Save", priority=True),
        Binding("ctrl+o", "open_session", "Open", priority=True), 
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+\\", "toggle_sidebar", "Toggle Sidebar"),
        Binding("ctrl+k", "clear_messages", "Clear Chat"),
    ]
    
    # Reactive state with proper typing
    current_session: reactive[Optional[ChatSession]] = reactive(None, init=False)
    settings: reactive[Settings] = reactive(Settings)
    sidebar_visible: reactive[bool] = reactive(True)
    is_loading: reactive[bool] = reactive(False)
    
    def on_mount(self):
        """Initialize app after mounting.
        Per https://textual.textualize.io/guide/app/#mounting
        Apps push screens, not compose them."""
        # Push the main screen
        self.push_screen(ChatScreen())
        
        # Create initial session
        self.current_session = ChatSession()
        
    def watch_current_session(self, old_session: Optional[ChatSession], new_session: Optional[ChatSession]):
        """React to session changes.
        Per https://textual.textualize.io/guide/reactivity/#watch-methods
        Watchers receive both old and new values."""
        # Update title
        self.title = f"Chat - {new_session.title if new_session else 'No Session'}"
        
        # Post message to current screen
        if self.screen:
            self.screen.post_message(SessionChanged(new_session))
    
    def watch_sidebar_visible(self, old_value: bool, new_value: bool):
        """React to sidebar visibility changes."""
        if self.screen:
            self.screen.post_message(SidebarToggled(new_value))
    
    def action_new_session(self):
        """Create new chat session.
        Per https://textual.textualize.io/guide/actions/"""
        self.current_session = ChatSession()
    
    def action_toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar_visible = not self.sidebar_visible
    
    def action_clear_messages(self):
        """Clear current session messages.
        Create new session object to trigger reactive update."""
        if self.current_session:
            # Create new session to trigger reactive update (not mutation)
            self.current_session = ChatSession(
                id=self.current_session.id,
                title=self.current_session.title,
                messages=[],
                created_at=self.current_session.created_at,
                updated_at=datetime.now(),
                metadata=self.current_session.metadata
            )
    
    def action_save_session(self):
        """Save current session."""
        if self.current_session:
            # This would integrate with the database
            self.notify(f"Session '{self.current_session.title}' saved")
    
    def action_open_session(self):
        """Open a saved session."""
        # This would show a file picker or session list
        self.notify("Open session dialog would appear here")


if __name__ == "__main__":
    app = ChatV99App()
    app.run()