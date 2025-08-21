"""Character editor widget for the CCP screen.

This widget provides a comprehensive form for editing character cards,
including V2 character card fields, following Textual best practices.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Label, Input, TextArea, Button, Switch
from textual.reactive import reactive
from textual import on
from textual.message import Message
from textual.validation import Length

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPCharacterEditorWidget")


# ========== Messages ==========

class CharacterEditorMessage(Message):
    """Base message for character editor events."""
    pass


class CharacterSaveRequested(CharacterEditorMessage):
    """User requested to save the character."""
    def __init__(self, character_data: Dict[str, Any]) -> None:
        super().__init__()
        self.character_data = character_data


class CharacterFieldGenerateRequested(CharacterEditorMessage):
    """User requested to generate a field with AI."""
    def __init__(self, field_name: str, character_data: Dict[str, Any]) -> None:
        super().__init__()
        self.field_name = field_name
        self.character_data = character_data


class CharacterImageUploadRequested(CharacterEditorMessage):
    """User requested to upload an image."""
    pass


class CharacterImageGenerateRequested(CharacterEditorMessage):
    """User requested to generate an image with AI."""
    def __init__(self, character_data: Dict[str, Any]) -> None:
        super().__init__()
        self.character_data = character_data


class CharacterEditorCancelled(CharacterEditorMessage):
    """User cancelled character editing."""
    pass


class AlternateGreetingAdded(CharacterEditorMessage):
    """User added an alternate greeting."""
    def __init__(self, greeting: str) -> None:
        super().__init__()
        self.greeting = greeting


class AlternateGreetingRemoved(CharacterEditorMessage):
    """User removed an alternate greeting."""
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index


# ========== Character Editor Widget ==========

class CCPCharacterEditorWidget(Container):
    """
    Character editor widget for the CCP screen.
    
    This widget provides a comprehensive editing form for character cards,
    including all V2 fields and AI generation capabilities.
    """
    
    DEFAULT_CSS = """
    CCPCharacterEditorWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    CCPCharacterEditorWidget.hidden {
        display: none !important;
    }
    
    .editor-header {
        width: 100%;
        height: 3;
        background: $primary-background-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    .editor-content {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .editor-section {
        width: 100%;
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
        background: $surface-darken-1;
    }
    
    .section-title {
        margin-bottom: 1;
        text-style: bold;
        color: $primary;
    }
    
    .field-container {
        width: 100%;
        margin-bottom: 1;
    }
    
    .field-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    
    .field-with-button {
        layout: horizontal;
        width: 100%;
        height: auto;
    }
    
    .field-input {
        width: 1fr;
        margin-right: 1;
    }
    
    .field-textarea {
        width: 100%;
        height: 8;
        margin-top: 0;
        border: round $surface;
        background: $surface;
    }
    
    .field-textarea.small {
        height: 5;
    }
    
    .field-textarea.large {
        height: 12;
    }
    
    .generate-button {
        width: auto;
        height: 3;
        padding: 0 1;
        background: $secondary;
    }
    
    .generate-button:hover {
        background: $secondary-lighten-1;
    }
    
    .image-section {
        width: 100%;
        height: 25;
        border: round $surface;
        background: $surface-darken-1;
        align: center middle;
        margin-bottom: 2;
    }
    
    .image-preview {
        width: 100%;
        height: 20;
        align: center middle;
        border: round $surface;
        background: $surface-darken-2;
        margin-bottom: 1;
    }
    
    .image-controls {
        layout: horizontal;
        height: 3;
        width: 100%;
    }
    
    .image-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .image-button:last-child {
        margin-right: 0;
    }
    
    .greeting-item {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
        border: round $surface-lighten-1;
    }
    
    .greeting-controls {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-top: 1;
    }
    
    .greeting-remove-button {
        width: auto;
        height: 3;
        background: $error-darken-1;
    }
    
    .greeting-remove-button:hover {
        background: $error;
    }
    
    .tags-input-container {
        width: 100%;
        margin-bottom: 1;
    }
    
    .tags-display {
        layout: horizontal;
        flex-wrap: wrap;
        width: 100%;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
        margin-top: 1;
    }
    
    .tag-item {
        padding: 0 1;
        margin: 0 1 1 0;
        background: $primary-background-darken-1;
        border: round $primary-darken-1;
        height: 3;
    }
    
    .tag-remove {
        margin-left: 1;
        color: $error;
        cursor: pointer;
    }
    
    .v2-toggle-container {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-bottom: 2;
        align: left middle;
    }
    
    .v2-toggle-label {
        width: auto;
        margin-right: 2;
    }
    
    .editor-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 1;
        background: $surface;
        border-top: thick $background-darken-1;
    }
    
    .editor-action-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .editor-action-button:last-child {
        margin-right: 0;
    }
    
    .editor-action-button.primary {
        background: $success;
    }
    
    .editor-action-button.primary:hover {
        background: $success-lighten-1;
    }
    
    .editor-action-button.cancel {
        background: $warning-darken-1;
    }
    
    .editor-action-button.cancel:hover {
        background: $warning;
    }
    
    .greetings-list {
        width: 100%;
        max-height: 30;
        overflow-y: auto;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
    }
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    # Current character data being edited
    character_data: reactive[Dict[str, Any]] = reactive({})
    
    # V2 features enabled
    v2_enabled: reactive[bool] = reactive(False)
    
    # Alternate greetings list
    alternate_greetings: reactive[List[str]] = reactive([])
    
    # Tags list
    tags: reactive[List[str]] = reactive([])
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the character editor widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for Container
        """
        super().__init__(id="ccp-character-editor-view", classes="ccp-view-area hidden", **kwargs)
        self.parent_screen = parent_screen
        
        # Field references for quick access
        self._name_input: Optional[Input] = None
        self._description_area: Optional[TextArea] = None
        self._personality_area: Optional[TextArea] = None
        self._scenario_area: Optional[TextArea] = None
        self._first_message_area: Optional[TextArea] = None
        self._creator_notes_area: Optional[TextArea] = None
        self._system_prompt_area: Optional[TextArea] = None
        self._post_history_area: Optional[TextArea] = None
        self._creator_input: Optional[Input] = None
        self._version_input: Optional[Input] = None
        self._tags_input: Optional[Input] = None
        self._new_greeting_area: Optional[TextArea] = None
        self._v2_toggle: Optional[Switch] = None
        
        logger.debug("CCPCharacterEditorWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the character editor UI."""
        # Header
        yield Static("Character Editor", classes="editor-header pane-title")
        
        # Content container with scroll
        with VerticalScroll(classes="editor-content"):
            # Image section
            with Container(classes="image-section"):
                with Container(classes="image-preview"):
                    yield Static("No image", id="ccp-editor-image-preview")
                
                with Container(classes="image-controls"):
                    yield Button("Upload Image", classes="image-button", id="upload-image-btn")
                    yield Button("Generate Image", classes="image-button generate-button", id="generate-image-btn")
                    yield Button("Remove Image", classes="image-button", id="remove-image-btn")
            
            # Basic Information Section
            with Container(classes="editor-section"):
                yield Static("Basic Information", classes="section-title")
                
                # Name field
                with Container(classes="field-container"):
                    yield Label("Name:", classes="field-label")
                    yield Input(placeholder="Character name", id="ccp-editor-name", classes="field-input",
                              validators=[Length(1, 100)])
                
                # Description with AI generate
                with Container(classes="field-container"):
                    yield Label("Description:", classes="field-label")
                    with Container(classes="field-with-button"):
                        yield TextArea("", id="ccp-editor-description", classes="field-textarea")
                        yield Button("Generate", classes="generate-button", id="generate-description-btn")
                
                # Personality with AI generate
                with Container(classes="field-container"):
                    yield Label("Personality:", classes="field-label")
                    with Container(classes="field-with-button"):
                        yield TextArea("", id="ccp-editor-personality", classes="field-textarea")
                        yield Button("Generate", classes="generate-button", id="generate-personality-btn")
                
                # Scenario with AI generate
                with Container(classes="field-container"):
                    yield Label("Scenario:", classes="field-label")
                    with Container(classes="field-with-button"):
                        yield TextArea("", id="ccp-editor-scenario", classes="field-textarea")
                        yield Button("Generate", classes="generate-button", id="generate-scenario-btn")
                
                # First Message with AI generate
                with Container(classes="field-container"):
                    yield Label("First Message:", classes="field-label")
                    with Container(classes="field-with-button"):
                        yield TextArea("", id="ccp-editor-first-message", classes="field-textarea")
                        yield Button("Generate", classes="generate-button", id="generate-first-message-btn")
            
            # V2 Toggle
            with Container(classes="v2-toggle-container"):
                yield Label("Enable V2 Character Card Features:", classes="v2-toggle-label")
                yield Switch(id="ccp-editor-v2-toggle", value=False)
            
            # V2 Fields Section (hidden by default)
            with Container(classes="editor-section hidden", id="v2-fields-section"):
                yield Static("V2 Character Card Fields", classes="section-title")
                
                # Creator Notes
                with Container(classes="field-container"):
                    yield Label("Creator Notes:", classes="field-label")
                    yield TextArea("", id="ccp-editor-creator-notes", classes="field-textarea small")
                
                # System Prompt with AI generate
                with Container(classes="field-container"):
                    yield Label("System Prompt:", classes="field-label")
                    with Container(classes="field-with-button"):
                        yield TextArea("", id="ccp-editor-system-prompt", classes="field-textarea large")
                        yield Button("Generate", classes="generate-button", id="generate-system-prompt-btn")
                
                # Post History Instructions
                with Container(classes="field-container"):
                    yield Label("Post History Instructions:", classes="field-label")
                    yield TextArea("", id="ccp-editor-post-history", classes="field-textarea small")
            
            # Alternate Greetings Section
            with Container(classes="editor-section"):
                yield Static("Alternate Greetings", classes="section-title")
                
                # List of existing greetings
                with Container(classes="greetings-list", id="ccp-editor-greetings-list"):
                    yield Static("No alternate greetings", classes="no-greetings-placeholder")
                
                # Add new greeting
                with Container(classes="field-container"):
                    yield Label("Add New Greeting:", classes="field-label")
                    yield TextArea("", id="ccp-editor-new-greeting", classes="field-textarea small")
                    yield Button("Add Greeting", id="add-greeting-btn")
            
            # Tags Section
            with Container(classes="editor-section"):
                yield Static("Tags", classes="section-title")
                
                with Container(classes="tags-input-container"):
                    yield Label("Add Tag:", classes="field-label")
                    yield Input(placeholder="Enter tag and press Enter", id="ccp-editor-tags-input")
                    
                    # Tags display
                    with Container(classes="tags-display", id="ccp-editor-tags-display"):
                        yield Static("No tags", classes="no-tags-placeholder")
            
            # Metadata Section
            with Container(classes="editor-section"):
                yield Static("Metadata", classes="section-title")
                
                with Container(classes="field-container"):
                    yield Label("Creator:", classes="field-label")
                    yield Input(placeholder="Creator name", id="ccp-editor-creator")
                
                with Container(classes="field-container"):
                    yield Label("Version:", classes="field-label")
                    yield Input(placeholder="1.0", id="ccp-editor-version", value="1.0")
        
        # Action buttons
        with Container(classes="editor-actions"):
            yield Button("Save Character", classes="editor-action-button primary", id="save-character-btn")
            yield Button("Reset", classes="editor-action-button", id="reset-character-btn")
            yield Button("Cancel", classes="editor-action-button cancel", id="cancel-edit-btn")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache field references
        self._cache_field_references()
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPCharacterEditorWidget mounted")
    
    def _cache_field_references(self) -> None:
        """Cache references to frequently used fields."""
        try:
            self._name_input = self.query_one("#ccp-editor-name", Input)
            self._description_area = self.query_one("#ccp-editor-description", TextArea)
            self._personality_area = self.query_one("#ccp-editor-personality", TextArea)
            self._scenario_area = self.query_one("#ccp-editor-scenario", TextArea)
            self._first_message_area = self.query_one("#ccp-editor-first-message", TextArea)
            self._creator_notes_area = self.query_one("#ccp-editor-creator-notes", TextArea)
            self._system_prompt_area = self.query_one("#ccp-editor-system-prompt", TextArea)
            self._post_history_area = self.query_one("#ccp-editor-post-history", TextArea)
            self._creator_input = self.query_one("#ccp-editor-creator", Input)
            self._version_input = self.query_one("#ccp-editor-version", Input)
            self._tags_input = self.query_one("#ccp-editor-tags-input", Input)
            self._new_greeting_area = self.query_one("#ccp-editor-new-greeting", TextArea)
            self._v2_toggle = self.query_one("#ccp-editor-v2-toggle", Switch)
        except Exception as e:
            logger.warning(f"Could not cache all field references: {e}")
    
    # ===== Public Methods =====
    
    def load_character(self, character_data: Dict[str, Any]) -> None:
        """Load character data into the editor.
        
        Args:
            character_data: Dictionary containing character information
        """
        self.character_data = character_data.copy()
        
        # Load basic fields
        if self._name_input:
            self._name_input.value = character_data.get('name', '')
        if self._description_area:
            self._description_area.text = character_data.get('description', '')
        if self._personality_area:
            self._personality_area.text = character_data.get('personality', '')
        if self._scenario_area:
            self._scenario_area.text = character_data.get('scenario', '')
        if self._first_message_area:
            self._first_message_area.text = character_data.get('first_mes', 
                                                              character_data.get('first_message', ''))
        
        # Load V2 fields if present
        has_v2_fields = any(character_data.get(field) for field in 
                           ['creator_notes', 'system_prompt', 'post_history_instructions'])
        
        if has_v2_fields:
            self.v2_enabled = True
            if self._v2_toggle:
                self._v2_toggle.value = True
            self._show_v2_fields()
            
            if self._creator_notes_area:
                self._creator_notes_area.text = character_data.get('creator_notes', '')
            if self._system_prompt_area:
                self._system_prompt_area.text = character_data.get('system_prompt', 
                                                                   character_data.get('system', ''))
            if self._post_history_area:
                self._post_history_area.text = character_data.get('post_history_instructions', '')
        
        # Load metadata
        if self._creator_input:
            self._creator_input.value = character_data.get('creator', '')
        if self._version_input:
            self._version_input.value = str(character_data.get('character_version', 
                                                              character_data.get('version', '1.0')))
        
        # Load alternate greetings
        self.alternate_greetings = character_data.get('alternate_greetings', []).copy()
        self._update_greetings_display()
        
        # Load tags
        self.tags = character_data.get('tags', []).copy()
        self._update_tags_display()
        
        # Load image if present
        image_data = character_data.get('image') or character_data.get('avatar')
        if image_data:
            self._update_image_preview(f"[Image loaded: {len(str(image_data))} bytes]")
        
        logger.info(f"Loaded character for editing: {character_data.get('name', 'Unknown')}")
    
    def new_character(self) -> None:
        """Initialize the editor for a new character."""
        self.character_data = {}
        self.alternate_greetings = []
        self.tags = []
        self.v2_enabled = False
        
        # Clear all fields
        if self._name_input:
            self._name_input.value = ""
        if self._description_area:
            self._description_area.text = ""
        if self._personality_area:
            self._personality_area.text = ""
        if self._scenario_area:
            self._scenario_area.text = ""
        if self._first_message_area:
            self._first_message_area.text = ""
        if self._creator_notes_area:
            self._creator_notes_area.text = ""
        if self._system_prompt_area:
            self._system_prompt_area.text = ""
        if self._post_history_area:
            self._post_history_area.text = ""
        if self._creator_input:
            self._creator_input.value = ""
        if self._version_input:
            self._version_input.value = "1.0"
        if self._new_greeting_area:
            self._new_greeting_area.text = ""
        if self._v2_toggle:
            self._v2_toggle.value = False
        
        self._hide_v2_fields()
        self._update_greetings_display()
        self._update_tags_display()
        self._update_image_preview("No image")
        
        logger.info("Initialized editor for new character")
    
    def get_character_data(self) -> Dict[str, Any]:
        """Get the current character data from the editor.
        
        Returns:
            Dictionary containing all character data
        """
        data = self.character_data.copy()
        
        # Update with current field values
        if self._name_input:
            data['name'] = self._name_input.value
        if self._description_area:
            data['description'] = self._description_area.text
        if self._personality_area:
            data['personality'] = self._personality_area.text
        if self._scenario_area:
            data['scenario'] = self._scenario_area.text
        if self._first_message_area:
            data['first_mes'] = self._first_message_area.text
        
        # V2 fields if enabled
        if self.v2_enabled:
            if self._creator_notes_area:
                data['creator_notes'] = self._creator_notes_area.text
            if self._system_prompt_area:
                data['system_prompt'] = self._system_prompt_area.text
            if self._post_history_area:
                data['post_history_instructions'] = self._post_history_area.text
        
        # Metadata
        if self._creator_input:
            data['creator'] = self._creator_input.value
        if self._version_input:
            data['character_version'] = self._version_input.value
        
        # Lists
        data['alternate_greetings'] = self.alternate_greetings.copy()
        data['tags'] = self.tags.copy()
        
        return data
    
    # ===== Private Helper Methods =====
    
    def _show_v2_fields(self) -> None:
        """Show V2 character card fields."""
        try:
            v2_section = self.query_one("#v2-fields-section")
            v2_section.remove_class("hidden")
        except:
            pass
    
    def _hide_v2_fields(self) -> None:
        """Hide V2 character card fields."""
        try:
            v2_section = self.query_one("#v2-fields-section")
            v2_section.add_class("hidden")
        except:
            pass
    
    def _update_image_preview(self, text: str) -> None:
        """Update the image preview display."""
        try:
            preview = self.query_one("#ccp-editor-image-preview", Static)
            preview.update(text)
        except:
            pass
    
    def _update_greetings_display(self) -> None:
        """Update the alternate greetings display."""
        try:
            container = self.query_one("#ccp-editor-greetings-list")
            container.remove_children()
            
            if self.alternate_greetings:
                for i, greeting in enumerate(self.alternate_greetings):
                    greeting_container = Container(classes="greeting-item")
                    
                    greeting_text = Static(
                        f"Greeting {i+1}: {greeting[:100]}{'...' if len(greeting) > 100 else ''}"
                    )
                    greeting_container.mount(greeting_text)
                    
                    remove_btn = Button(f"Remove", classes="greeting-remove-button remove-greeting-btn", 
                                      id=f"remove-greeting-{i}")
                    greeting_container.mount(remove_btn)
                    
                    container.mount(greeting_container)
            else:
                placeholder = Static("No alternate greetings", classes="no-greetings-placeholder")
                container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not update greetings display: {e}")
    
    def _update_tags_display(self) -> None:
        """Update the tags display."""
        try:
            container = self.query_one("#ccp-editor-tags-display")
            container.remove_children()
            
            if self.tags:
                for i, tag in enumerate(self.tags):
                    tag_container = Container(classes="tag-item")
                    tag_btn = Button(f"{tag} ×", id=f"remove-tag-{i}", classes="remove-tag-btn tag-button")
                    tag_container.mount(tag_btn)
                    container.mount(tag_container)
            else:
                placeholder = Static("No tags", classes="no-tags-placeholder")
                container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not update tags display: {e}")
    
    # ===== Event Handlers =====
    
    @on(Switch.Changed, "#ccp-editor-v2-toggle")
    async def handle_v2_toggle(self, event: Switch.Changed) -> None:
        """Handle V2 features toggle."""
        self.v2_enabled = event.value
        if event.value:
            self._show_v2_fields()
        else:
            self._hide_v2_fields()
    
    @on(Button.Pressed, "#save-character-btn")
    async def handle_save_character(self, event: Button.Pressed) -> None:
        """Handle save character button press."""
        event.stop()
        character_data = self.get_character_data()
        
        # Validate required fields
        if not character_data.get('name'):
            logger.warning("Cannot save character without name")
            return
        
        self.post_message(CharacterSaveRequested(character_data))
    
    @on(Button.Pressed, "#reset-character-btn")
    async def handle_reset_character(self, event: Button.Pressed) -> None:
        """Handle reset character button press."""
        event.stop()
        if self.character_data:
            self.load_character(self.character_data)
        else:
            self.new_character()
    
    @on(Button.Pressed, "#cancel-edit-btn")
    async def handle_cancel_edit(self, event: Button.Pressed) -> None:
        """Handle cancel edit button press."""
        event.stop()
        self.post_message(CharacterEditorCancelled())
    
    @on(Button.Pressed, "#add-greeting-btn")
    async def handle_add_greeting(self, event: Button.Pressed) -> None:
        """Handle add greeting button press."""
        event.stop()
        if self._new_greeting_area and self._new_greeting_area.text.strip():
            greeting = self._new_greeting_area.text.strip()
            self.alternate_greetings.append(greeting)
            self._new_greeting_area.text = ""
            self._update_greetings_display()
            self.post_message(AlternateGreetingAdded(greeting))
    
    @on(Button.Pressed, ".remove-greeting-btn")
    async def handle_remove_greeting(self, event: Button.Pressed) -> None:
        """Handle remove greeting button press."""
        event.stop()
        if event.button.id and event.button.id.startswith("remove-greeting-"):
            index = int(event.button.id.replace("remove-greeting-", ""))
            if 0 <= index < len(self.alternate_greetings):
                del self.alternate_greetings[index]
                self._update_greetings_display()
                self.post_message(AlternateGreetingRemoved(index))
    
    @on(Input.Submitted, "#ccp-editor-tags-input")
    async def handle_add_tag(self, event: Input.Submitted) -> None:
        """Handle tag input submission."""
        if event.value.strip():
            tag = event.value.strip()
            if tag not in self.tags:
                self.tags.append(tag)
                self._update_tags_display()
            event.input.value = ""
    
    @on(Button.Pressed, ".remove-tag-btn")
    async def handle_remove_tag(self, event: Button.Pressed) -> None:
        """Handle tag removal click."""
        if event.button.id and event.button.id.startswith("remove-tag-"):
            index = int(event.button.id.replace("remove-tag-", ""))
            if 0 <= index < len(self.tags):
                del self.tags[index]
                self._update_tags_display()
    
    # AI Generation button handlers
    @on(Button.Pressed, "#generate-description-btn")
    async def handle_generate_description(self, event: Button.Pressed) -> None:
        """Handle generate description button press."""
        event.stop()
        self.post_message(CharacterFieldGenerateRequested("description", self.get_character_data()))
    
    @on(Button.Pressed, "#generate-personality-btn")
    async def handle_generate_personality(self, event: Button.Pressed) -> None:
        """Handle generate personality button press."""
        event.stop()
        self.post_message(CharacterFieldGenerateRequested("personality", self.get_character_data()))
    
    @on(Button.Pressed, "#generate-scenario-btn")
    async def handle_generate_scenario(self, event: Button.Pressed) -> None:
        """Handle generate scenario button press."""
        event.stop()
        self.post_message(CharacterFieldGenerateRequested("scenario", self.get_character_data()))
    
    @on(Button.Pressed, "#generate-first-message-btn")
    async def handle_generate_first_message(self, event: Button.Pressed) -> None:
        """Handle generate first message button press."""
        event.stop()
        self.post_message(CharacterFieldGenerateRequested("first_message", self.get_character_data()))
    
    @on(Button.Pressed, "#generate-system-prompt-btn")
    async def handle_generate_system_prompt(self, event: Button.Pressed) -> None:
        """Handle generate system prompt button press."""
        event.stop()
        self.post_message(CharacterFieldGenerateRequested("system_prompt", self.get_character_data()))
    
    # Image handlers
    @on(Button.Pressed, "#upload-image-btn")
    async def handle_upload_image(self, event: Button.Pressed) -> None:
        """Handle upload image button press."""
        event.stop()
        self.post_message(CharacterImageUploadRequested())
    
    @on(Button.Pressed, "#generate-image-btn")
    async def handle_generate_image(self, event: Button.Pressed) -> None:
        """Handle generate image button press."""
        event.stop()
        self.post_message(CharacterImageGenerateRequested(self.get_character_data()))
    
    @on(Button.Pressed, "#remove-image-btn")
    async def handle_remove_image(self, event: Button.Pressed) -> None:
        """Handle remove image button press."""
        event.stop()
        if 'image' in self.character_data:
            del self.character_data['image']
        if 'avatar' in self.character_data:
            del self.character_data['avatar']
        self._update_image_preview("No image")