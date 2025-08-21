"""Character card display widget for the CCP screen.

This widget displays character card information in a read-only format.
Following Textual best practices with focused, reusable components.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Label, TextArea, Button
from textual.reactive import reactive
from textual import on
from textual.message import Message

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPCharacterCardWidget")


# ========== Messages ==========

class CharacterCardMessage(Message):
    """Base message for character card events."""
    pass


class EditCharacterRequested(CharacterCardMessage):
    """User requested to edit the character."""
    def __init__(self, character_id: int) -> None:
        super().__init__()
        self.character_id = character_id


class CloneCharacterRequested(CharacterCardMessage):
    """User requested to clone the character."""
    def __init__(self, character_id: int) -> None:
        super().__init__()
        self.character_id = character_id


class ExportCharacterRequested(CharacterCardMessage):
    """User requested to export the character."""
    def __init__(self, character_id: int, format: str = "json") -> None:
        super().__init__()
        self.character_id = character_id
        self.format = format


class DeleteCharacterRequested(CharacterCardMessage):
    """User requested to delete the character."""
    def __init__(self, character_id: int) -> None:
        super().__init__()
        self.character_id = character_id


class StartChatRequested(CharacterCardMessage):
    """User requested to start a chat with the character."""
    def __init__(self, character_id: int) -> None:
        super().__init__()
        self.character_id = character_id


# ========== Character Card Widget ==========

class CCPCharacterCardWidget(Container):
    """
    Character card display widget for the CCP screen.
    
    This widget displays character information in a clean, read-only format
    following Textual best practices for focused components.
    """
    
    DEFAULT_CSS = """
    CCPCharacterCardWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    CCPCharacterCardWidget.hidden {
        display: none !important;
    }
    
    .character-header {
        width: 100%;
        height: 3;
        background: $primary-background-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    .character-content {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .character-image-container {
        width: 100%;
        height: 20;
        border: round $surface;
        margin-bottom: 2;
        align: center middle;
        background: $surface-darken-1;
    }
    
    .character-image {
        width: 100%;
        height: 100%;
        align: center middle;
    }
    
    .no-image-placeholder {
        text-align: center;
        color: $text-muted;
    }
    
    .field-section {
        width: 100%;
        margin-bottom: 2;
    }
    
    .field-label {
        margin-bottom: 0;
        color: $text-muted;
        text-style: bold;
    }
    
    .field-value {
        margin-top: 0;
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
    }
    
    .field-textarea {
        width: 100%;
        height: 8;
        margin-top: 0;
        margin-bottom: 1;
        border: round $surface;
        background: $surface;
    }
    
    .field-textarea.small {
        height: 5;
    }
    
    .field-textarea.large {
        height: 12;
    }
    
    .tags-container {
        layout: horizontal;
        flex-wrap: wrap;
        width: 100%;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
    }
    
    .tag-item {
        padding: 0 1;
        margin: 0 1 1 0;
        background: $primary-background-darken-1;
        border: round $primary-darken-1;
        height: 3;
    }
    
    .character-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 1;
        background: $surface;
        border-top: thick $background-darken-1;
    }
    
    .character-action-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .character-action-button:last-child {
        margin-right: 0;
    }
    
    .character-action-button.primary {
        background: $success;
    }
    
    .character-action-button.primary:hover {
        background: $success-lighten-1;
    }
    
    .character-action-button.danger {
        background: $error-darken-1;
    }
    
    .character-action-button.danger:hover {
        background: $error;
    }
    
    .alternate-greetings-list {
        width: 100%;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
    }
    
    .greeting-item {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        background: $surface-lighten-1;
        border: round $surface;
    }
    
    .greeting-item:last-child {
        margin-bottom: 0;
    }
    
    .no-character-message {
        width: 100%;
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    # Current character data
    character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the character card widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for Container
        """
        super().__init__(id="ccp-character-card-view", classes="ccp-view-area hidden", **kwargs)
        self.parent_screen = parent_screen
        
        # Cache references to frequently updated fields
        self._name_display: Optional[Static] = None
        self._description_display: Optional[TextArea] = None
        self._personality_display: Optional[TextArea] = None
        self._scenario_display: Optional[TextArea] = None
        self._first_message_display: Optional[TextArea] = None
        self._tags_container: Optional[Container] = None
        
        logger.debug("CCPCharacterCardWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the character card UI."""
        # Header
        yield Static("Character Card", classes="character-header pane-title")
        
        # Content container with scroll
        with VerticalScroll(classes="character-content"):
            # Default message when no character is loaded
            yield Static(
                "No character loaded.\nSelect a character from the sidebar to view details.",
                classes="no-character-message",
                id="no-character-placeholder"
            )
            
            # Character details container (hidden by default)
            with Container(id="character-details-container", classes="hidden"):
                # Character image
                with Container(classes="character-image-container", id="character-image-container"):
                    yield Static("No image", classes="no-image-placeholder", id="ccp-card-image-placeholder")
                
                # Basic fields
                yield Label("Name:", classes="field-label")
                yield Static("", id="ccp-card-name-display", classes="field-value")
                
                yield Label("Description:", classes="field-label")
                yield TextArea("", id="ccp-card-description-display", read_only=True, classes="field-textarea")
                
                yield Label("Personality:", classes="field-label")
                yield TextArea("", id="ccp-card-personality-display", read_only=True, classes="field-textarea")
                
                yield Label("Scenario:", classes="field-label")
                yield TextArea("", id="ccp-card-scenario-display", read_only=True, classes="field-textarea")
                
                yield Label("First Message:", classes="field-label")
                yield TextArea("", id="ccp-card-first-message-display", read_only=True, classes="field-textarea")
                
                # V2 fields
                yield Label("Creator Notes:", classes="field-label")
                yield TextArea("", id="ccp-card-creator-notes-display", read_only=True, classes="field-textarea small")
                
                yield Label("System Prompt:", classes="field-label")
                yield TextArea("", id="ccp-card-system-prompt-display", read_only=True, classes="field-textarea large")
                
                yield Label("Post History Instructions:", classes="field-label")
                yield TextArea("", id="ccp-card-post-history-instructions-display", read_only=True, 
                             classes="field-textarea small")
                
                # Alternate greetings
                yield Label("Alternate Greetings:", classes="field-label")
                with Container(id="ccp-card-alternate-greetings-container", classes="alternate-greetings-list"):
                    yield Static("No alternate greetings", classes="no-image-placeholder")
                
                # Tags
                yield Label("Tags:", classes="field-label")
                with Container(id="ccp-card-tags-container", classes="tags-container"):
                    yield Static("No tags", classes="no-image-placeholder")
                
                # Metadata
                yield Label("Creator:", classes="field-label")
                yield Static("", id="ccp-card-creator-display", classes="field-value")
                
                yield Label("Version:", classes="field-label")
                yield Static("", id="ccp-card-version-display", classes="field-value")
        
        # Action buttons
        with Container(classes="character-actions"):
            yield Button("Start Chat", classes="character-action-button primary", id="start-chat-btn")
            yield Button("Edit", classes="character-action-button", id="edit-character-btn")
            yield Button("Clone", classes="character-action-button", id="clone-character-btn")
            yield Button("Export", classes="character-action-button", id="export-character-btn")
            yield Button("Delete", classes="character-action-button danger", id="delete-character-btn")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache field references
        self._cache_field_references()
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPCharacterCardWidget mounted")
    
    def _cache_field_references(self) -> None:
        """Cache references to frequently updated fields."""
        try:
            self._name_display = self.query_one("#ccp-card-name-display", Static)
            self._description_display = self.query_one("#ccp-card-description-display", TextArea)
            self._personality_display = self.query_one("#ccp-card-personality-display", TextArea)
            self._scenario_display = self.query_one("#ccp-card-scenario-display", TextArea)
            self._first_message_display = self.query_one("#ccp-card-first-message-display", TextArea)
            self._tags_container = self.query_one("#ccp-card-tags-container", Container)
        except Exception as e:
            logger.warning(f"Could not cache all field references: {e}")
    
    # ===== Public Methods =====
    
    def load_character(self, character_data: Dict[str, Any]) -> None:
        """Load and display character data.
        
        Args:
            character_data: Dictionary containing character information
        """
        self.character_data = character_data
        
        # Hide placeholder, show details
        try:
            placeholder = self.query_one("#no-character-placeholder")
            placeholder.add_class("hidden")
            
            details = self.query_one("#character-details-container")
            details.remove_class("hidden")
        except:
            pass
        
        # Update fields
        self._update_basic_fields(character_data)
        self._update_v2_fields(character_data)
        self._update_metadata_fields(character_data)
        self._update_image(character_data)
        self._update_tags(character_data.get('tags', []))
        self._update_alternate_greetings(character_data.get('alternate_greetings', []))
        
        logger.info(f"Loaded character: {character_data.get('name', 'Unknown')}")
    
    def _update_basic_fields(self, data: Dict[str, Any]) -> None:
        """Update basic character fields."""
        if self._name_display:
            self._name_display.update(data.get('name', 'Unnamed Character'))
        
        if self._description_display:
            self._description_display.text = data.get('description', '')
        
        if self._personality_display:
            self._personality_display.text = data.get('personality', '')
        
        if self._scenario_display:
            self._scenario_display.text = data.get('scenario', '')
        
        if self._first_message_display:
            self._first_message_display.text = data.get('first_mes', data.get('first_message', ''))
    
    def _update_v2_fields(self, data: Dict[str, Any]) -> None:
        """Update V2 character card fields."""
        try:
            creator_notes = self.query_one("#ccp-card-creator-notes-display", TextArea)
            creator_notes.text = data.get('creator_notes', '')
            
            system_prompt = self.query_one("#ccp-card-system-prompt-display", TextArea)
            system_prompt.text = data.get('system_prompt', data.get('system', ''))
            
            post_history = self.query_one("#ccp-card-post-history-instructions-display", TextArea)
            post_history.text = data.get('post_history_instructions', '')
        except Exception as e:
            logger.warning(f"Could not update V2 fields: {e}")
    
    def _update_metadata_fields(self, data: Dict[str, Any]) -> None:
        """Update metadata fields."""
        try:
            creator = self.query_one("#ccp-card-creator-display", Static)
            creator.update(data.get('creator', 'Unknown'))
            
            version = self.query_one("#ccp-card-version-display", Static)
            version.update(str(data.get('character_version', data.get('version', '1.0'))))
        except Exception as e:
            logger.warning(f"Could not update metadata fields: {e}")
    
    def _update_image(self, data: Dict[str, Any]) -> None:
        """Update character image display."""
        try:
            image_container = self.query_one("#character-image-container")
            placeholder = self.query_one("#ccp-card-image-placeholder", Static)
            
            # Check for image data
            image_data = data.get('image') or data.get('avatar')
            if image_data:
                # In a real implementation, we'd display the actual image
                # For now, just show a placeholder with the image info
                placeholder.update(f"[Image: {len(str(image_data))} bytes]")
            else:
                placeholder.update("No image")
        except Exception as e:
            logger.warning(f"Could not update image: {e}")
    
    def _update_tags(self, tags: List[str]) -> None:
        """Update tags display."""
        if not self._tags_container:
            return
        
        # Clear existing tags
        self._tags_container.remove_children()
        
        if tags:
            for tag in tags:
                tag_widget = Static(tag, classes="tag-item")
                self._tags_container.mount(tag_widget)
        else:
            placeholder = Static("No tags", classes="no-image-placeholder")
            self._tags_container.mount(placeholder)
    
    def _update_alternate_greetings(self, greetings: List[str]) -> None:
        """Update alternate greetings display."""
        try:
            container = self.query_one("#ccp-card-alternate-greetings-container")
            container.remove_children()
            
            if greetings:
                for i, greeting in enumerate(greetings, 1):
                    greeting_widget = Static(
                        f"Greeting {i}: {greeting[:100]}{'...' if len(greeting) > 100 else ''}",
                        classes="greeting-item"
                    )
                    container.mount(greeting_widget)
            else:
                placeholder = Static("No alternate greetings", classes="no-image-placeholder")
                container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not update alternate greetings: {e}")
    
    def clear_character(self) -> None:
        """Clear the character display."""
        self.character_data = None
        
        # Show placeholder, hide details
        try:
            placeholder = self.query_one("#no-character-placeholder")
            placeholder.remove_class("hidden")
            
            details = self.query_one("#character-details-container")
            details.add_class("hidden")
        except:
            pass
        
        # Clear all fields
        if self._name_display:
            self._name_display.update("")
        if self._description_display:
            self._description_display.text = ""
        if self._personality_display:
            self._personality_display.text = ""
        if self._scenario_display:
            self._scenario_display.text = ""
        if self._first_message_display:
            self._first_message_display.text = ""
    
    # ===== Event Handlers =====
    
    @on(Button.Pressed, "#start-chat-btn")
    async def handle_start_chat(self, event: Button.Pressed) -> None:
        """Handle start chat button press."""
        event.stop()
        if self.character_data:
            char_id = self.character_data.get('id')
            if char_id:
                self.post_message(StartChatRequested(char_id))
    
    @on(Button.Pressed, "#edit-character-btn")
    async def handle_edit_character(self, event: Button.Pressed) -> None:
        """Handle edit character button press."""
        event.stop()
        if self.character_data:
            char_id = self.character_data.get('id')
            if char_id:
                self.post_message(EditCharacterRequested(char_id))
    
    @on(Button.Pressed, "#clone-character-btn")
    async def handle_clone_character(self, event: Button.Pressed) -> None:
        """Handle clone character button press."""
        event.stop()
        if self.character_data:
            char_id = self.character_data.get('id')
            if char_id:
                self.post_message(CloneCharacterRequested(char_id))
    
    @on(Button.Pressed, "#export-character-btn")
    async def handle_export_character(self, event: Button.Pressed) -> None:
        """Handle export character button press."""
        event.stop()
        if self.character_data:
            char_id = self.character_data.get('id')
            if char_id:
                self.post_message(ExportCharacterRequested(char_id))
    
    @on(Button.Pressed, "#delete-character-btn")
    async def handle_delete_character(self, event: Button.Pressed) -> None:
        """Handle delete character button press."""
        event.stop()
        if self.character_data:
            char_id = self.character_data.get('id')
            if char_id:
                self.post_message(DeleteCharacterRequested(char_id))