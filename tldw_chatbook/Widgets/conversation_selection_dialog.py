"""Conversation selection dialog for TTS audio generation"""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Label, Static, Input, RadioButton, RadioSet
from textual.screen import ModalScreen
from textual.reactive import reactive
from typing import Optional, Dict, Any, List
from loguru import logger


class ConversationItem(Container):
    """A single conversation item in the selection list"""
    
    def __init__(
        self, 
        conversation_id: int, 
        title: str, 
        model_name: str,
        message_count: int,
        created_at: str,
        updated_at: str
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.title = title
        self.model_name = model_name
        self.message_count = message_count
        self.created_at = created_at
        self.updated_at = updated_at
    
    def compose(self) -> ComposeResult:
        """Build the conversation item UI"""
        with Horizontal(classes="conversation-item"):
            yield RadioButton(
                label="",
                id=f"conv-radio-{self.conversation_id}",
                name="conversation-selection"
            )
            with Vertical(classes="conversation-details"):
                yield Label(self.title or "Untitled Conversation", classes="conversation-title")
                yield Static(
                    f"Model: {self.model_name} | Messages: {self.message_count}",
                    classes="conversation-info"
                )
                yield Static(
                    f"Created: {self.created_at} | Updated: {self.updated_at}",
                    classes="conversation-date"
                )


class ConversationSelectionDialog(ModalScreen[Optional[Dict[str, Any]]]):
    """Dialog for selecting a conversation to convert to audio"""
    
    CSS = """
    ConversationSelectionDialog {
        align: center middle;
    }
    
    #conversation-selection-container {
        width: 80;
        height: 50;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .search-row {
        height: 3;
        margin-bottom: 1;
    }
    
    #conversation-search-input {
        width: 100%;
    }
    
    #conversations-list-container {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }
    
    .conversation-item {
        padding: 1;
        margin-bottom: 1;
        border-bottom: dashed $secondary;
    }
    
    .conversation-item:hover {
        background: $boost;
    }
    
    .conversation-details {
        width: 1fr;
        margin-left: 1;
    }
    
    .conversation-title {
        text-style: bold;
    }
    
    .conversation-info {
        color: $text-muted;
    }
    
    .conversation-date {
        color: $text-disabled;
        font-size: 10;
    }
    
    .options-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $secondary;
    }
    
    .option-label {
        width: 20;
        margin-right: 1;
    }
    
    .button-row {
        dock: bottom;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    
    .button-row Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, conversations: List[Dict[str, Any]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.conversations = conversations
        self.conversation_items: List[ConversationItem] = []
        self.selected_conversation_id: Optional[int] = None
    
    def compose(self) -> ComposeResult:
        """Build the dialog UI"""
        with Container(id="conversation-selection-container"):
            yield Label("Select Conversation for Audio Generation", classes="dialog-title")
            
            # Search input
            with Horizontal(classes="search-row"):
                yield Input(
                    placeholder="Search conversations by title...",
                    id="conversation-search-input"
                )
            
            # Conversations list
            with ScrollableContainer(id="conversations-list-container"):
                with RadioSet(id="conversations-radio-set"):
                    yield Vertical(id="conversations-list")
            
            # Options section
            with Vertical(classes="options-section"):
                yield Label("Export Options:", classes="section-title")
                
                with Horizontal(classes="form-row"):
                    yield Label("Include:", classes="option-label")
                    with Vertical():
                        yield RadioSet(
                            RadioButton("All messages", id="include-all", value=True),
                            RadioButton("User messages only", id="include-user"),
                            RadioButton("Assistant messages only", id="include-assistant"),
                            id="include-options"
                        )
                
                with Horizontal(classes="form-row"):
                    yield Label("Format:", classes="option-label")
                    with Vertical():
                        yield RadioSet(
                            RadioButton("Include speaker names", id="format-speakers", value=True),
                            RadioButton("Messages only", id="format-messages"),
                            id="format-options"
                        )
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Generate Audio", id="generate-btn", variant="primary", disabled=True)
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    def on_mount(self) -> None:
        """Initialize with conversations data"""
        self.load_conversations(self.conversations)
    
    def load_conversations(self, conversations: List[Dict[str, Any]]) -> None:
        """Load conversations into the list"""
        conversations_list = self.query_one("#conversations-list", Vertical)
        conversations_list.clear()
        self.conversation_items.clear()
        
        for conv in conversations:
            # Create conversation item
            item = ConversationItem(
                conversation_id=conv["conversation_id"],
                title=conv.get("title", ""),
                model_name=conv.get("model_name", "Unknown"),
                message_count=conv.get("message_count", 0),
                created_at=conv.get("created_at", "Unknown"),
                updated_at=conv.get("updated_at", "Unknown")
            )
            
            self.conversation_items.append(item)
            conversations_list.mount(item)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "conversation-search-input":
            search_term = event.value.lower()
            self.filter_conversations(search_term)
    
    def filter_conversations(self, search_term: str) -> None:
        """Filter conversations based on search term"""
        for item in self.conversation_items:
            if search_term:
                # Search in title
                visible = search_term in item.title.lower()
                item.display = visible
            else:
                item.display = True
    
    def on_radio_button_changed(self, event: RadioButton.Changed) -> None:
        """Handle radio button selection"""
        if event.radio_button.name == "conversation-selection" and event.value:
            # Extract conversation ID from the radio button ID
            radio_id = event.radio_button.id
            if radio_id and radio_id.startswith("conv-radio-"):
                try:
                    self.selected_conversation_id = int(radio_id.replace("conv-radio-", ""))
                    self.query_one("#generate-btn", Button).disabled = False
                except ValueError:
                    logger.error(f"Invalid conversation ID from radio button: {radio_id}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "generate-btn":
            self.generate_audio()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)
    
    def generate_audio(self) -> None:
        """Generate audio for selected conversation"""
        if not self.selected_conversation_id:
            self.app.notify("No conversation selected", severity="warning")
            return
        
        # Get selected options
        include_options = self.query_one("#include-options", RadioSet)
        include_all = include_options.pressed_button.id == "include-all"
        include_user = include_options.pressed_button.id == "include-user"
        include_assistant = include_options.pressed_button.id == "include-assistant"
        
        format_options = self.query_one("#format-options", RadioSet)
        include_speakers = format_options.pressed_button.id == "format-speakers"
        
        # Build result
        result = {
            "conversation_id": self.selected_conversation_id,
            "include_all": include_all,
            "include_user": include_user,
            "include_assistant": include_assistant,
            "include_speakers": include_speakers
        }
        
        self.dismiss(result)