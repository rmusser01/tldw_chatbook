"""Main chat application following Textual patterns."""

from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive
from typing import Optional
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .screens.chat_screen import ChatScreen
    from .models import ChatSession, Settings, ChatMessage
    from .messages import SessionChanged, SidebarToggled
except ImportError:
    # Fallback for direct execution
    from screens.chat_screen import ChatScreen
    from models import ChatSession, Settings, ChatMessage
    from messages import SessionChanged, SidebarToggled

# Use the REAL, EXISTING database and chat functions!
from tldw_chatbook.Chat.Chat_Functions import (
    save_chat_history_to_db_wrapper,
    save_chat_history
)
from tldw_chatbook.config import get_chachanotes_db_lazy
from loguru import logger


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
        Binding("ctrl+k", "clear_messages", "Clear Chat", priority=True),
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
            new_session = ChatSession(
                id=self.current_session.id,
                title=self.current_session.title,
                messages=[],  # Clear messages
                created_at=self.current_session.created_at,
                updated_at=datetime.now(),
                metadata=self.current_session.metadata
            )
            self.current_session = new_session
    
    def action_save_session(self):
        """Save current session using REAL Chat_Functions."""
        if self.current_session:
            # Convert ChatSession messages to OpenAI format expected by save function
            chatbot_history = []
            for msg in self.current_session.messages:
                chatbot_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Get the database instance
            db = get_chachanotes_db_lazy()
            
            # Ensure Default Character exists to avoid database errors
            try:
                from tldw_chatbook.Chat.Chat_Functions import DEFAULT_CHARACTER_NAME, DEFAULT_CHARACTER_DESCRIPTION
                default_char = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
                if not default_char:
                    # Create Default Character if it doesn't exist
                    db.add_character_card(
                        name=DEFAULT_CHARACTER_NAME,
                        description=DEFAULT_CHARACTER_DESCRIPTION,
                        personality="",
                        scenario=""
                    )
                    logger.info(f"Created '{DEFAULT_CHARACTER_NAME}' in database")
            except Exception as e:
                logger.warning(f"Could not ensure Default Character exists: {e}")
            
            # Use the REAL save function with correct parameters
            conversation_id, status = save_chat_history_to_db_wrapper(
                db=db,
                chatbot_history=chatbot_history,
                conversation_id=self.current_session.id,
                media_content_for_char_assoc=None,
                media_name_for_char_assoc=None,
                character_name_for_chat=None
            )
            
            if conversation_id:
                self.current_session.id = conversation_id
                self.notify(f"✅ Saved: {self.current_session.title}")
            else:
                self.notify(f"Failed to save: {status}", severity="error")
    
    def action_open_session(self):
        """Load sessions using REAL database."""
        self.run_worker(self._load_sessions)
    
    async def _load_sessions(self):
        """Load sessions from REAL database."""
        try:
            db = get_chachanotes_db_lazy()
            conversations = db.get_all_conversations(limit=20)
            
            if conversations:
                # Update sidebar with real conversations
                if self.screen:
                    sidebar = self.screen.query_one("#sidebar")
                    if sidebar:
                        session_list = [(conv['conversation_id'], conv['title']) 
                                      for conv in conversations]
                        sidebar.load_session_history(session_list)
                        self.notify(f"Found {len(conversations)} conversations")
            else:
                self.notify("No conversations found", severity="warning")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            self.notify(f"Error: {str(e)}", severity="error")
    
    async def load_session_by_id(self, conversation_id: str):
        """Load a specific conversation from REAL database."""
        try:
            # Use REAL database to load messages
            db = get_chachanotes_db_lazy()
            messages = db.get_messages_for_conversation(conversation_id)
            
            if messages:
                # Convert to ChatSession format
                chat_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('message', '')
                        if content:
                            chat_messages.append(ChatMessage(role=role, content=content))
                
                # Get conversation details
                db = get_chachanotes_db_lazy()
                details = db.get_conversation_details(conversation_id)
                
                self.current_session = ChatSession(
                    id=conversation_id,
                    title=details.get('title', 'Untitled') if details else 'Untitled',
                    messages=chat_messages
                )
                self.notify(f"Loaded: {self.current_session.title}")
            else:
                self.notify("Conversation not found", severity="error")
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            self.notify(f"Error: {str(e)}", severity="error")


if __name__ == "__main__":
    app = ChatV99App()
    app.run()