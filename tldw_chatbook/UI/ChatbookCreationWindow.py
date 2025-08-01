# ChatbookCreationWindow.py
# Description: UI for creating chatbooks/knowledge packs
#
"""
Chatbook Creation Window
------------------------

Provides an interface for selecting content and creating chatbooks.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, TYPE_CHECKING
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button, Input, TextArea, Label, Checkbox, Tree
from textual.widgets.tree import TreeNode
from loguru import logger

from ..Chatbooks.chatbook_creator import ChatbookCreator
from ..Chatbooks.chatbook_models import ContentType
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Prompts_DB import PromptsDatabase
from ..DB.Client_Media_DB_v2 import MediaDatabase

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookCreationWindow(ModalScreen):
    """Window for creating chatbooks."""
    
    DEFAULT_CSS = """
    ChatbookCreationWindow {
        align: center middle;
    }
    
    ChatbookCreationWindow > Container {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    .chatbook-title {
        text-style: bold;
        color: $text;
        padding: 1 0;
        text-align: center;
    }
    
    .form-section {
        padding: 1 0;
        border-bottom: solid $background-darken-1;
        margin-bottom: 1;
    }
    
    .form-label {
        color: $text-muted;
        padding: 0 0 0 0;
    }
    
    .content-tree {
        height: 15;
        border: round $background-darken-1;
        background: $boost;
        padding: 1;
        margin: 1 0;
    }
    
    .button-container {
        dock: bottom;
        height: auto;
        padding: 1 0;
        align: center middle;
    }
    
    .button-container Button {
        margin: 0 1;
    }
    
    .stats-container {
        layout: horizontal;
        height: auto;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin: 1 0;
    }
    
    .stat-item {
        width: 1fr;
        text-align: center;
        padding: 0 1;
    }
    
    .stat-value {
        text-style: bold;
        color: $primary;
    }
    """
    
    def __init__(self, app_instance: "TldwCli"):
        """Initialize the chatbook creation window."""
        super().__init__()
        self.app = app_instance
        self.selected_content: Dict[ContentType, Set[str]] = {
            ContentType.CONVERSATION: set(),
            ContentType.NOTE: set(),
            ContentType.CHARACTER: set(),
            ContentType.PROMPT: set(),
            ContentType.MEDIA: set()
        }
        
        # Get database paths from config
        db_config = self.app.config_data.get("database", {})
        self.db_paths = {
            "chachanotes": Path(db_config.get("chachanotes_db_path", "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")).expanduser(),
            "prompts": Path(db_config.get("prompts_db_path", "~/.local/share/tldw_cli/tldw_prompts_db.db")).expanduser(),
            "media": Path(db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_media_db.db")).expanduser()
        }
        
        self.creator = ChatbookCreator({
            name: str(path) for name, path in self.db_paths.items()
        })
    
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        with Container():
            yield Static("Create Chatbook", classes="chatbook-title")
            
            with VerticalScroll():
                # Basic Information
                with Container(classes="form-section"):
                    yield Label("Chatbook Name:", classes="form-label")
                    yield Input(
                        placeholder="Enter chatbook name...",
                        id="chatbook-name"
                    )
                    
                    yield Label("Description:", classes="form-label")
                    yield TextArea(
                        "",
                        id="chatbook-description"
                    )
                    
                    yield Label("Author (optional):", classes="form-label")
                    yield Input(
                        placeholder="Your name...",
                        id="chatbook-author"
                    )
                    
                    yield Label("Tags (comma-separated):", classes="form-label")
                    yield Input(
                        placeholder="research, tutorial, guide...",
                        id="chatbook-tags"
                    )
                
                # Content Selection
                with Container(classes="form-section"):
                    yield Label("Select Content:", classes="form-label")
                    yield Tree(
                        "Content",
                        id="content-tree",
                        classes="content-tree"
                    )
                
                # Options
                with Container(classes="form-section"):
                    yield Label("Options:", classes="form-label")
                    yield Checkbox("Include media files", id="include-media")
                    yield Checkbox("Include embeddings", id="include-embeddings")
                
                # Statistics
                with Container(classes="stats-container"):
                    with Container(classes="stat-item"):
                        yield Static("Conversations", classes="form-label")
                        yield Static("0", id="stat-conversations", classes="stat-value")
                    with Container(classes="stat-item"):
                        yield Static("Notes", classes="form-label")
                        yield Static("0", id="stat-notes", classes="stat-value")
                    with Container(classes="stat-item"):
                        yield Static("Characters", classes="form-label")
                        yield Static("0", id="stat-characters", classes="stat-value")
                    with Container(classes="stat-item"):
                        yield Static("Total Items", classes="form-label")
                        yield Static("0", id="stat-total", classes="stat-value")
            
            # Buttons
            with Horizontal(classes="button-container"):
                yield Button("Create Chatbook", variant="success", id="create-button")
                yield Button("Cancel", variant="default", id="cancel-button")
    
    async def on_mount(self) -> None:
        """Called when the window is mounted."""
        # Load content tree
        await self._populate_content_tree()
        
        # Set default author from config
        username = self.app.config_data.get("general", {}).get("users_name")
        if username:
            self.query_one("#chatbook-author", Input).value = username
    
    async def _populate_content_tree(self) -> None:
        """Populate the content tree with available content."""
        tree = self.query_one("#content-tree", Tree)
        tree.clear()
        
        # Load conversations
        if self.db_paths["chachanotes"].exists():
            db = CharactersRAGDB(str(self.db_paths["chachanotes"]), "chatbook_ui")
            
            # Add conversations node
            conv_node = tree.root.add("📚 Conversations", expand=True)
            conversations = db.list_all_active_conversations(limit=100)
            
            for conv in conversations:
                node = conv_node.add(
                    f"{conv['conversation_name']} ({conv['message_count']} messages)",
                    data={"type": ContentType.CONVERSATION, "id": str(conv['id'])}
                )
                node.allow_expand = False
            
            # Add notes node
            notes_node = tree.root.add("📝 Notes", expand=True)
            notes = db.list_notes(limit=100)
            
            for note in notes:
                node = notes_node.add(
                    note['title'],
                    data={"type": ContentType.NOTE, "id": str(note['id'])}
                )
                node.allow_expand = False
            
            # Add characters node
            chars_node = tree.root.add("👤 Characters", expand=True)
            characters = db.list_all_characters()
            
            for char in characters:
                node = chars_node.add(
                    char['name'],
                    data={"type": ContentType.CHARACTER, "id": str(char['id'])}
                )
                node.allow_expand = False
        
        # Load prompts
        if self.db_paths["prompts"].exists():
            db = PromptsDatabase(str(self.db_paths["prompts"]), "chatbook_ui")
            
            prompts_node = tree.root.add("💬 Prompts", expand=False)
            prompts = db.list_prompts()
            
            for prompt in prompts:
                node = prompts_node.add(
                    prompt['name'],
                    data={"type": ContentType.PROMPT, "id": str(prompt['id'])}
                )
                node.allow_expand = False
    
    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        
        if hasattr(node, 'data') and node.data:
            content_type = node.data.get('type')
            content_id = node.data.get('id')
            
            if content_type and content_id:
                # Toggle selection
                if content_id in self.selected_content[content_type]:
                    self.selected_content[content_type].remove(content_id)
                    # Update visual indicator
                    if "✓ " in node.label:
                        node.set_label(node.label.replace("✓ ", ""))
                else:
                    self.selected_content[content_type].add(content_id)
                    # Update visual indicator
                    if "✓ " not in node.label:
                        node.set_label(f"✓ {node.label}")
                
                # Update statistics
                self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update the statistics display."""
        conv_count = len(self.selected_content[ContentType.CONVERSATION])
        note_count = len(self.selected_content[ContentType.NOTE])
        char_count = len(self.selected_content[ContentType.CHARACTER])
        total_count = conv_count + note_count + char_count + len(self.selected_content[ContentType.PROMPT])
        
        self.query_one("#stat-conversations", Static).update(str(conv_count))
        self.query_one("#stat-notes", Static).update(str(note_count))
        self.query_one("#stat-characters", Static).update(str(char_count))
        self.query_one("#stat-total", Static).update(str(total_count))
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "create-button":
            await self._create_chatbook()
    
    async def _create_chatbook(self) -> None:
        """Create the chatbook."""
        # Get form values
        name = self.query_one("#chatbook-name", Input).value.strip()
        description = self.query_one("#chatbook-description", TextArea).text.strip()
        author = self.query_one("#chatbook-author", Input).value.strip()
        tags_input = self.query_one("#chatbook-tags", Input).value.strip()
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
        
        include_media = self.query_one("#include-media", Checkbox).value
        include_embeddings = self.query_one("#include-embeddings", Checkbox).value
        
        # Validate
        if not name:
            self.app.notify("Please enter a chatbook name", severity="error")
            return
        
        if not description:
            self.app.notify("Please enter a description", severity="error")
            return
        
        # Check if any content selected
        total_selected = sum(len(items) for items in self.selected_content.values())
        if total_selected == 0:
            self.app.notify("Please select at least one content item", severity="error")
            return
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        output_dir = Path.home() / ".local" / "share" / "tldw_cli" / "chatbooks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{safe_name}_{timestamp}.zip"
        
        # Convert selected content to list format
        content_selections = {
            content_type: list(items)
            for content_type, items in self.selected_content.items()
            if items
        }
        
        # Create chatbook in background
        self.app.notify(f"Creating chatbook '{name}'...", severity="information")
        
        try:
            success, message = self.creator.create_chatbook(
                name=name,
                description=description,
                content_selections=content_selections,
                output_path=output_path,
                author=author or None,
                include_media=include_media,
                include_embeddings=include_embeddings,
                tags=tags
            )
            
            if success:
                self.app.notify(message, severity="success")
                self.dismiss(str(output_path))
            else:
                self.app.notify(message, severity="error")
                
        except Exception as e:
            logger.error(f"Error creating chatbook: {e}")
            self.app.notify(f"Error creating chatbook: {str(e)}", severity="error")