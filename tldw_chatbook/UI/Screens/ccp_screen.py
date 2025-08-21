"""Conversations, Characters & Prompts (CCP) Screen.

This screen provides a unified interface for managing conversations, characters,
prompts, and dictionaries following Textual best practices with Screen-based architecture.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Input, ListView, Select, Collapsible, Label, TextArea, Checkbox
from textual.reactive import reactive
from textual import on, work
from textual.css.query import NoMatches

from ..Navigation.base_app_screen import BaseAppScreen
from ...Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE

# Import modular handlers and enhancements
from ..CCP_Modules import (
    CCPConversationHandler,
    CCPCharacterHandler,
    CCPPromptHandler,
    CCPDictionaryHandler,
    CCPMessageManager,
    CCPSidebarHandler,
    ConversationMessage,
    CharacterMessage,
    PromptMessage,
    DictionaryMessage,
    ViewChangeMessage,
    SidebarMessage,
    LoadingManager,
    setup_ccp_enhancements
)

if TYPE_CHECKING:
    from ...app import TldwCli

logger = logger.bind(module="CCPScreen")


class CCPScreen(BaseAppScreen):
    """
    Screen for the Conversations, Characters & Prompts (CCP) interface.
    
    This screen follows Textual best practices:
    - Extends BaseAppScreen for proper screen management
    - Uses reactive properties for state management
    - Implements modern event handling with @on decorators
    - Utilizes message system for inter-component communication
    - Employs modular handlers for separation of concerns
    """
    
    # CSS embedded directly
    DEFAULT_CSS = """
    /* CCP Screen Styles */
    #ccp-main-container {
        layout: horizontal;
        height: 100%;
    }
    
    /* Sidebar Styling */
    .ccp-sidebar {
        width: 30%;
        min-width: 25;
        max-width: 40%;
        height: 100%;
        background: $boost;
        padding: 1;
        border-right: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    .ccp-sidebar.collapsed {
        width: 0 !important;
        min-width: 0 !important;
        border-right: none !important;
        padding: 0 !important;
        overflow: hidden !important;
        display: none !important;
    }
    
    .ccp-sidebar-toggle-button {
        width: 3;
        height: 100%;
        min-width: 3;
        border: none;
        background: $surface-darken-1;
        color: $text;
        dock: left;
    }
    
    .ccp-sidebar-toggle-button:hover {
        background: $surface;
    }
    
    /* Content Area */
    .ccp-content-area {
        width: 1fr;
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }
    
    .ccp-view-area {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1;
    }
    
    .ccp-view-area.hidden {
        display: none !important;
    }
    
    .hidden {
        display: none !important;
    }
    
    /* Titles and Labels */
    .pane-title {
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
        background: $primary-background-darken-1;
        padding: 0 1;
        height: 3;
    }
    
    .sidebar-title {
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
        color: $primary;
    }
    
    .sidebar-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
    }
    
    .field-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
        text-style: bold;
    }
    
    .field-value {
        margin-bottom: 1;
        padding: 0 1;
    }
    
    /* Input Components */
    .sidebar-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .sidebar-textarea {
        width: 100%;
        height: 5;
        margin-bottom: 1;
        border: round $surface;
    }
    
    .sidebar-textarea.small {
        height: 3;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
        height: 3;
    }
    
    .sidebar-button.small {
        width: 45%;
        margin-right: 1;
    }
    
    .sidebar-button.danger {
        background: $error-darken-1;
    }
    
    .sidebar-button.danger:hover {
        background: $error;
    }
    
    .sidebar-listview {
        height: 10;
        margin-bottom: 1;
        border: round $surface;
    }
    
    /* Editor Components */
    .editor-scroll {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .editor-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .editor-textarea {
        width: 100%;
        height: 10;
        margin-bottom: 1;
        border: round $surface;
    }
    
    .editor-textarea.small {
        height: 5;
    }
    
    .field-textarea {
        width: 100%;
        height: 8;
        margin-bottom: 1;
        border: round $surface;
    }
    
    /* AI Generation */
    .field-with-ai {
        layout: horizontal;
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }
    
    .field-with-ai TextArea {
        width: 85%;
        margin-right: 1;
    }
    
    .ai-generate-button {
        width: 12%;
        height: 3;
        margin-top: 0;
        background: $primary;
    }
    
    .ai-generate-button:hover {
        background: $primary-lighten-1;
    }
    
    .ai-generate-button.full-width {
        width: 100%;
        margin-bottom: 1;
    }
    
    /* Action Buttons */
    .editor-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-top: 2;
        margin-bottom: 1;
    }
    
    .editor-actions Button {
        width: 1fr;
        margin-right: 1;
    }
    
    .editor-actions Button:last-child {
        margin-right: 0;
    }
    
    .primary-button {
        background: $success;
    }
    
    .primary-button:hover {
        background: $success-lighten-1;
    }
    
    .secondary-button {
        background: $surface;
    }
    
    .secondary-button:hover {
        background: $surface-lighten-1;
    }
    
    /* Export buttons */
    .export-buttons {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .export-buttons Button {
        width: 1fr;
        margin-right: 1;
    }
    
    .export-buttons Button:last-child {
        margin-right: 0;
    }
    
    /* Image controls */
    .image-controls {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .image-controls Button {
        width: 1fr;
        margin-right: 1;
    }
    
    .image-controls Button:last-child {
        margin-right: 0;
    }
    
    .image-status {
        margin-bottom: 1;
        padding: 0 1;
        color: $text-muted;
    }
    
    .character-image {
        width: 100%;
        height: 15;
        border: round $surface;
        margin-bottom: 1;
        align: center middle;
        background: $surface-darken-1;
    }
    
    /* Dictionary styles */
    .dict-entries-list {
        height: 12;
        margin-bottom: 1;
        border: round $surface;
    }
    
    .dict-entry-controls {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .dict-entry-controls Button {
        width: 1fr;
        margin-right: 1;
    }
    
    .dict-entry-controls Button:last-child {
        margin-right: 0;
    }
    """
    
    # Reactive properties for state management
    active_view: reactive[str] = reactive("conversations", layout=False)
    selected_character_id: reactive[Optional[int]] = reactive(None, layout=False)
    selected_conversation_id: reactive[Optional[int]] = reactive(None, layout=False)
    selected_prompt_id: reactive[Optional[int]] = reactive(None, layout=False)
    selected_dictionary_id: reactive[Optional[int]] = reactive(None, layout=False)
    sidebar_collapsed: reactive[bool] = reactive(False, layout=False)
    
    # Cached widget references
    _sidebar: Optional[Container] = None
    _content_area: Optional[Container] = None
    _message_area: Optional[Container] = None

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the CCP Screen with modular handlers.
        
        Args:
            app_instance: Reference to the main application instance
            **kwargs: Additional keyword arguments for Screen
        """
        super().__init__(app_instance, "ccp", **kwargs)
        
        # Initialize modular handlers
        self.conversation_handler = CCPConversationHandler(self)
        self.character_handler = CCPCharacterHandler(self)
        self.prompt_handler = CCPPromptHandler(self)
        self.dictionary_handler = CCPDictionaryHandler(self)
        self.message_manager = CCPMessageManager(self)
        self.sidebar_handler = CCPSidebarHandler(self)
        
        # Initialize loading manager for async operation feedback
        self.loading_manager = LoadingManager(self)
        
        # Setup enhancements (validation, loading indicators)
        setup_ccp_enhancements(self)
        
        logger.debug("CCPScreen initialized with modular handlers and enhancements")

    def compose_content(self) -> ComposeResult:
        """Compose the CCP UI with single sidebar design.
        
        This overrides the base class method to provide CCP-specific content.
        
        Yields:
            The widgets that make up the CCP interface
        """
        logger.debug("Composing CCPScreen UI")
        
        # Main container for CCP content
        with Container(id="ccp-main-container", classes="ccp-main-container"):
            # Sidebar toggle button
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="toggle-ccp-sidebar",
                classes="ccp-sidebar-toggle-button",
                tooltip="Toggle sidebar (Ctrl+[)"
            )
            
            # Single unified sidebar with all controls
            with VerticalScroll(id="ccp-sidebar", classes="ccp-sidebar"):
                yield Static("CCP Navigation", classes="sidebar-title")
                
                # Conversations section
                with Collapsible(title="Conversations", id="ccp-conversations-collapsible"):
                    yield Button("Import Conversation", id="ccp-import-conversation-button", 
                               classes="sidebar-button")
                    
                    # Search controls
                    yield Label("Search by Title:", classes="sidebar-label")
                    yield Input(id="conv-char-search-input", placeholder="Search by title...", 
                              classes="sidebar-input")
                    
                    yield Label("Search by Content:", classes="sidebar-label")
                    yield Input(id="conv-char-keyword-search-input", placeholder="Search keywords...", 
                              classes="sidebar-input")
                    
                    yield Label("Filter by Tags:", classes="sidebar-label")
                    yield Input(id="conv-char-tags-search-input", placeholder="Tags (comma-separated)...", 
                              classes="sidebar-input")
                    
                    # Search options
                    yield Checkbox("Include Character Chats", id="conv-char-search-include-character-checkbox", 
                                 value=True)
                    yield Checkbox("All Characters", id="conv-char-search-all-characters-checkbox", 
                                 value=True)
                    
                    # Results list
                    yield ListView(id="conv-char-search-results-list", classes="sidebar-listview")
                    yield Button("Load Selected", id="conv-char-load-button", classes="sidebar-button")
                    
                    # Conversation details (shown when a conversation is loaded)
                    with Container(id="conv-details-container", classes="hidden"):
                        yield Label("Title:", classes="sidebar-label")
                        yield Input(id="conv-char-title-input", placeholder="Conversation title...", 
                                  classes="sidebar-input")
                        yield Label("Keywords:", classes="sidebar-label")
                        yield TextArea(id="conv-char-keywords-input", classes="sidebar-textarea")
                        yield Button("Save Details", id="conv-char-save-details-button", 
                                   classes="sidebar-button")
                        
                        # Export options
                        yield Label("Export:", classes="sidebar-label")
                        with Horizontal(classes="export-buttons"):
                            yield Button("Text", id="conv-char-export-text-button", 
                                       classes="sidebar-button small")
                            yield Button("JSON", id="conv-char-export-json-button", 
                                       classes="sidebar-button small")
                
                # Characters section
                with Collapsible(title="Characters", id="ccp-characters-collapsible", collapsed=True):
                    yield Button("Import Character Card", id="ccp-import-character-button", 
                               classes="sidebar-button")
                    yield Button("Create Character", id="ccp-create-character-button", 
                               classes="sidebar-button")
                    yield Select([], prompt="Select Character...", allow_blank=True, 
                               id="conv-char-character-select")
                    yield Button("Load Character", id="ccp-right-pane-load-character-button", 
                               classes="sidebar-button")
                    yield Button("Refresh List", id="ccp-refresh-character-list-button", 
                               classes="sidebar-button")
                    
                    # Character actions (shown when a character is loaded)
                    with Container(id="char-actions-container", classes="hidden"):
                        yield Button("Edit Character", id="ccp-edit-character-button", 
                                   classes="sidebar-button")
                        yield Button("Clone Character", id="ccp-clone-character-button", 
                                   classes="sidebar-button")
                        yield Button("Export Character", id="ccp-export-character-button", 
                                   classes="sidebar-button")
                        yield Button("Delete Character", id="ccp-delete-character-button", 
                                   classes="sidebar-button danger")
                
                # Prompts section
                with Collapsible(title="Prompts", id="ccp-prompts-collapsible", collapsed=True):
                    yield Button("Import Prompt", id="ccp-import-prompt-button", classes="sidebar-button")
                    yield Button("Create New Prompt", id="ccp-prompt-create-new-button", 
                               classes="sidebar-button")
                    yield Input(id="ccp-prompt-search-input", placeholder="Search prompts...", 
                              classes="sidebar-input")
                    yield ListView(id="ccp-prompts-listview", classes="sidebar-listview")
                    yield Button("Load Selected", id="ccp-prompt-load-selected-button", 
                               classes="sidebar-button")
                    
                    # Prompt actions (shown when a prompt is loaded)
                    with Container(id="prompt-actions-container", classes="hidden"):
                        yield Button("Clone Prompt", id="ccp-prompt-clone-button", 
                                   classes="sidebar-button")
                        yield Button("Delete Prompt", id="ccp-prompt-delete-button", 
                                   classes="sidebar-button danger")
                
                # Dictionaries section
                with Collapsible(title="Chat Dictionaries", id="ccp-dictionaries-collapsible", collapsed=True):
                    yield Button("Import Dictionary", id="ccp-import-dictionary-button", 
                               classes="sidebar-button")
                    yield Button("Create Dictionary", id="ccp-create-dictionary-button", 
                               classes="sidebar-button")
                    yield Select([], prompt="Select Dictionary...", allow_blank=True, 
                               id="ccp-dictionary-select")
                    yield Button("Load Dictionary", id="ccp-load-dictionary-button", 
                               classes="sidebar-button")
                    yield Button("Refresh List", id="ccp-refresh-dictionary-list-button", 
                               classes="sidebar-button")
                    
                    # Dictionary actions (shown when a dictionary is loaded)
                    with Container(id="dict-actions-container", classes="hidden"):
                        yield Button("Edit Dictionary", id="ccp-edit-dictionary-button", 
                                   classes="sidebar-button")
                        yield Button("Clone Dictionary", id="ccp-clone-dictionary-button", 
                                   classes="sidebar-button")
                        yield Button("Delete Dictionary", id="ccp-delete-dictionary-button", 
                                   classes="sidebar-button danger")
                
                # World Books section
                with Collapsible(title="World/Lore Books", id="ccp-worldbooks-collapsible", collapsed=True):
                    yield Button("Import World Book", id="ccp-import-worldbook-button", 
                               classes="sidebar-button")
                    yield Button("Create World Book", id="ccp-create-worldbook-button", 
                               classes="sidebar-button")
                    yield Input(id="ccp-worldbook-search-input", placeholder="Search world books...", 
                              classes="sidebar-input")
                    yield ListView(id="ccp-worldbooks-listview", classes="sidebar-listview")
                    yield Button("Load Selected", id="ccp-worldbook-load-button", 
                               classes="sidebar-button")
                    yield Button("Edit Selected", id="ccp-worldbook-edit-button", 
                               classes="sidebar-button")
                    yield Button("Refresh List", id="ccp-refresh-worldbook-list-button", 
                               classes="sidebar-button")

            # Main Content Area
            with Container(id="ccp-content-area", classes="ccp-content-area"):
                # Container for conversation messages
                with Container(id="ccp-conversation-messages-view", classes="ccp-view-area"):
                    yield Static("Conversation History", classes="pane-title")
                    # Messages will be mounted dynamically here by message_manager
                
                # Container for character card display
                with Container(id="ccp-character-card-view", classes="ccp-view-area hidden"):
                    yield Static("Character Card", classes="pane-title")
                    yield Static(id="ccp-card-image-placeholder", classes="character-image")
                    yield Label("Name:", classes="field-label")
                    yield Static(id="ccp-card-name-display", classes="field-value")
                    yield Label("Description:", classes="field-label")
                    yield TextArea(id="ccp-card-description-display", read_only=True, classes="field-textarea")
                    yield Label("Personality:", classes="field-label")
                    yield TextArea(id="ccp-card-personality-display", read_only=True, classes="field-textarea")
                    yield Label("Scenario:", classes="field-label")
                    yield TextArea(id="ccp-card-scenario-display", read_only=True, classes="field-textarea")
                    yield Label("First Message:", classes="field-label")
                    yield TextArea(id="ccp-card-first-message-display", read_only=True, classes="field-textarea")
                    # V2 fields
                    yield Label("Creator Notes:", classes="field-label")
                    yield TextArea(id="ccp-card-creator-notes-display", read_only=True, classes="field-textarea")
                    yield Label("System Prompt:", classes="field-label")
                    yield TextArea(id="ccp-card-system-prompt-display", read_only=True, classes="field-textarea")
                    yield Label("Post History Instructions:", classes="field-label")
                    yield TextArea(id="ccp-card-post-history-instructions-display", read_only=True, 
                                 classes="field-textarea")
                    yield Label("Alternate Greetings:", classes="field-label")
                    yield TextArea(id="ccp-card-alternate-greetings-display", read_only=True, 
                                 classes="field-textarea")
                    yield Label("Tags:", classes="field-label")
                    yield Static(id="ccp-card-tags-display", classes="field-value")
                    yield Label("Creator:", classes="field-label")
                    yield Static(id="ccp-card-creator-display", classes="field-value")
                    yield Label("Version:", classes="field-label")
                    yield Static(id="ccp-card-version-display", classes="field-value")
                
                # Container for character editor
                with Container(id="ccp-character-editor-view", classes="ccp-view-area hidden"):
                    yield Static("Character Editor", classes="pane-title")
                    with VerticalScroll(classes="editor-scroll"):
                        yield Label("Character Name:", classes="field-label")
                        yield Input(id="ccp-editor-char-name-input", placeholder="Character name...", 
                                  classes="editor-input")
                        yield Button("✨ Generate All Fields", id="ccp-generate-all-button", 
                                   classes="ai-generate-button full-width")
                        
                        # Image controls
                        yield Label("Character Image:", classes="field-label")
                        with Horizontal(classes="image-controls"):
                            yield Button("Choose Image", id="ccp-editor-char-image-button", 
                                       classes="sidebar-button")
                            yield Button("Clear Image", id="ccp-editor-char-clear-image-button", 
                                       classes="sidebar-button")
                        yield Static("No image selected", id="ccp-editor-char-image-status", 
                                   classes="image-status")
                        yield Label("Image URL (optional):", classes="field-label")
                        yield Input(id="ccp-editor-char-avatar-input", placeholder="URL to avatar image...", 
                                  classes="editor-input")
                        
                        # Character fields with AI generation
                        yield Label("Description:", classes="field-label")
                        with Horizontal(classes="field-with-ai"):
                            yield TextArea(id="ccp-editor-char-description-textarea", classes="editor-textarea")
                            yield Button("✨", id="ccp-generate-description-button", 
                                       classes="ai-generate-button")
                        
                        yield Label("Personality:", classes="field-label")
                        with Horizontal(classes="field-with-ai"):
                            yield TextArea(id="ccp-editor-char-personality-textarea", classes="editor-textarea")
                            yield Button("✨", id="ccp-generate-personality-button", 
                                       classes="ai-generate-button")
                        
                        yield Label("Scenario:", classes="field-label")
                        with Horizontal(classes="field-with-ai"):
                            yield TextArea(id="ccp-editor-char-scenario-textarea", classes="editor-textarea")
                            yield Button("✨", id="ccp-generate-scenario-button", 
                                       classes="ai-generate-button")
                        
                        yield Label("First Message:", classes="field-label")
                        with Horizontal(classes="field-with-ai"):
                            yield TextArea(id="ccp-editor-char-first-message-textarea", 
                                         classes="editor-textarea")
                            yield Button("✨", id="ccp-generate-first-message-button", 
                                       classes="ai-generate-button")
                        
                        # Additional fields
                        yield Label("Keywords (comma-separated):", classes="field-label")
                        yield TextArea(id="ccp-editor-char-keywords-textarea", classes="editor-textarea small")
                        
                        # V2 fields
                        yield Label("Creator Notes:", classes="field-label")
                        yield TextArea(id="ccp-editor-char-creator-notes-textarea", classes="editor-textarea")
                        
                        yield Label("System Prompt:", classes="field-label")
                        with Horizontal(classes="field-with-ai"):
                            yield TextArea(id="ccp-editor-char-system-prompt-textarea", 
                                         classes="editor-textarea")
                            yield Button("✨", id="ccp-generate-system-prompt-button", 
                                       classes="ai-generate-button")
                        
                        yield Label("Post History Instructions:", classes="field-label")
                        yield TextArea(id="ccp-editor-char-post-history-instructions-textarea", 
                                     classes="editor-textarea")
                        
                        yield Label("Alternate Greetings (one per line):", classes="field-label")
                        yield TextArea(id="ccp-editor-char-alternate-greetings-textarea", 
                                     classes="editor-textarea")
                        
                        yield Label("Tags (comma-separated):", classes="field-label")
                        yield Input(id="ccp-editor-char-tags-input", placeholder="e.g., fantasy, anime", 
                                  classes="editor-input")
                        
                        yield Label("Creator:", classes="field-label")
                        yield Input(id="ccp-editor-char-creator-input", placeholder="Creator name", 
                                  classes="editor-input")
                        
                        yield Label("Character Version:", classes="field-label")
                        yield Input(id="ccp-editor-char-version-input", placeholder="e.g., 1.0", 
                                  classes="editor-input")
                        
                        # Action buttons
                        with Horizontal(classes="editor-actions"):
                            yield Button("Save Character", id="ccp-editor-char-save-button", 
                                       classes="primary-button")
                            yield Button("Cancel", id="ccp-editor-char-cancel-button", 
                                       classes="secondary-button")
                
                # Container for prompt editor
                with Container(id="ccp-prompt-editor-view", classes="ccp-view-area hidden"):
                    yield Static("Prompt Editor", classes="pane-title")
                    with VerticalScroll(classes="editor-scroll"):
                        yield Label("Prompt Name:", classes="field-label")
                        yield Input(id="ccp-editor-prompt-name-input", placeholder="Unique prompt name...", 
                                  classes="editor-input")
                        yield Label("Author:", classes="field-label")
                        yield Input(id="ccp-editor-prompt-author-input", placeholder="Author name...", 
                                  classes="editor-input")
                        yield Label("Details/Description:", classes="field-label")
                        yield TextArea(id="ccp-editor-prompt-description-textarea", classes="editor-textarea")
                        yield Label("System Prompt:", classes="field-label")
                        yield TextArea(id="ccp-editor-prompt-system-textarea", classes="editor-textarea")
                        yield Label("User Prompt (Template):", classes="field-label")
                        yield TextArea(id="ccp-editor-prompt-user-textarea", classes="editor-textarea")
                        yield Label("Keywords (comma-separated):", classes="field-label")
                        yield TextArea(id="ccp-editor-prompt-keywords-textarea", classes="editor-textarea small")
                        
                        # Action buttons
                        with Horizontal(classes="editor-actions"):
                            yield Button("Save Prompt", id="ccp-editor-prompt-save-button", 
                                       classes="primary-button")
                            yield Button("Cancel", id="ccp-editor-prompt-cancel-button", 
                                       classes="secondary-button")
                
                # Container for dictionary view
                with Container(id="ccp-dictionary-view", classes="ccp-view-area hidden"):
                    yield Static("Chat Dictionary", classes="pane-title")
                    yield Label("Dictionary Name:", classes="field-label")
                    yield Static(id="ccp-dict-name-display", classes="field-value")
                    yield Label("Description:", classes="field-label")
                    yield TextArea(id="ccp-dict-description-display", read_only=True, classes="field-textarea")
                    yield Label("Strategy:", classes="field-label")
                    yield Static(id="ccp-dict-strategy-display", classes="field-value")
                    yield Label("Max Tokens:", classes="field-label")
                    yield Static(id="ccp-dict-max-tokens-display", classes="field-value")
                    yield Label("Entries:", classes="field-label")
                    yield ListView(id="ccp-dict-entries-list", classes="dict-entries-list")
                
                # Container for dictionary editor
                with Container(id="ccp-dictionary-editor-view", classes="ccp-view-area hidden"):
                    yield Static("Dictionary Editor", classes="pane-title")
                    with VerticalScroll(classes="editor-scroll"):
                        yield Label("Dictionary Name:", classes="field-label")
                        yield Input(id="ccp-editor-dict-name-input", placeholder="Dictionary name...", 
                                  classes="editor-input")
                        yield Label("Description:", classes="field-label")
                        yield TextArea(id="ccp-editor-dict-description-textarea", classes="editor-textarea")
                        yield Label("Replacement Strategy:", classes="field-label")
                        yield Select([
                            ("sorted_evenly", "sorted_evenly"),
                            ("character_lore_first", "character_lore_first"),
                            ("global_lore_first", "global_lore_first")
                        ], value="sorted_evenly", id="ccp-editor-dict-strategy-select")
                        yield Label("Max Tokens:", classes="field-label")
                        yield Input(id="ccp-editor-dict-max-tokens-input", placeholder="1000", value="1000", 
                                  classes="editor-input")
                        
                        yield Label("Dictionary Entries:", classes="field-label")
                        yield ListView(id="ccp-editor-dict-entries-list", classes="dict-entries-list")
                        
                        with Horizontal(classes="dict-entry-controls"):
                            yield Button("Add Entry", id="ccp-dict-add-entry-button", 
                                       classes="sidebar-button")
                            yield Button("Remove Entry", id="ccp-dict-remove-entry-button", 
                                       classes="sidebar-button")
                        
                        yield Label("Entry Key/Pattern:", classes="field-label")
                        yield Input(id="ccp-dict-entry-key-input", placeholder="Key or /regex/flags", 
                                  classes="editor-input")
                        yield Label("Entry Value:", classes="field-label")
                        yield TextArea(id="ccp-dict-entry-value-textarea", classes="editor-textarea small")
                        yield Label("Group (optional):", classes="field-label")
                        yield Input(id="ccp-dict-entry-group-input", placeholder="e.g., character, global", 
                                  classes="editor-input")
                        yield Label("Probability (0-100):", classes="field-label")
                        yield Input(id="ccp-dict-entry-probability-input", placeholder="100", value="100", 
                                  classes="editor-input")
                        
                        # Action buttons
                        with Horizontal(classes="editor-actions"):
                            yield Button("Save Dictionary", id="ccp-editor-dict-save-button", 
                                       classes="primary-button")
                            yield Button("Cancel", id="ccp-editor-dict-cancel-button", 
                                       classes="secondary-button")

    async def on_mount(self) -> None:
        """Handle post-composition setup."""
        super().on_mount()  # Don't await - parent's on_mount is not async
        
        # Cache widget references
        self._cache_widget_references()
        
        # Setup loading manager widget
        await self.loading_manager.setup()
        
        # Initialize UI state
        await self._initialize_ui_state()
        
        logger.debug("CCPScreen mounted and initialized with enhancements")

    def _cache_widget_references(self) -> None:
        """Cache frequently accessed widgets."""
        try:
            self._sidebar = self.query_one("#ccp-sidebar")
            self._content_area = self.query_one("#ccp-content-area")
            self._message_area = self.query_one("#ccp-conversation-messages-view")
        except NoMatches as e:
            logger.error(f"Failed to cache widget: {e}")

    async def _initialize_ui_state(self) -> None:
        """Initialize the UI state."""
        # Refresh lists
        await self.character_handler.refresh_character_list()
        await self.dictionary_handler.refresh_dictionary_list()
        
        # Set initial view
        self.active_view = "conversations"

    # ===== Event Handlers using @on decorators =====
    
    @on(Button.Pressed, "#toggle-ccp-sidebar")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Handle sidebar toggle button press."""
        event.stop()
        await self.sidebar_handler.toggle_sidebar()
    
    @on(Button.Pressed, "#conv-char-load-button")
    async def handle_load_conversation(self, event: Button.Pressed) -> None:
        """Handle loading selected conversation."""
        event.stop()
        await self.conversation_handler.handle_load_selected()
    
    @on(Button.Pressed, "#ccp-right-pane-load-character-button")
    async def handle_load_character(self, event: Button.Pressed) -> None:
        """Handle loading selected character."""
        event.stop()
        await self.character_handler.handle_load_character()
    
    @on(Button.Pressed, "#ccp-prompt-load-selected-button")
    async def handle_load_prompt(self, event: Button.Pressed) -> None:
        """Handle loading selected prompt."""
        event.stop()
        await self.prompt_handler.handle_load_selected()
    
    @on(Button.Pressed, "#ccp-load-dictionary-button")
    async def handle_load_dictionary(self, event: Button.Pressed) -> None:
        """Handle loading selected dictionary."""
        event.stop()
        await self.dictionary_handler.handle_load_dictionary()
    
    @on(Button.Pressed, "#ccp-refresh-character-list-button")
    async def handle_refresh_characters(self, event: Button.Pressed) -> None:
        """Handle refreshing character list."""
        event.stop()
        await self.character_handler.refresh_character_list()
    
    @on(Button.Pressed, "#ccp-refresh-dictionary-list-button")
    async def handle_refresh_dictionaries(self, event: Button.Pressed) -> None:
        """Handle refreshing dictionary list."""
        event.stop()
        await self.dictionary_handler.refresh_dictionary_list()
    
    @on(Button.Pressed, "#ccp-editor-char-save-button")
    async def handle_save_character(self, event: Button.Pressed) -> None:
        """Handle saving character from editor."""
        event.stop()
        await self.character_handler.handle_save_character()
    
    @on(Button.Pressed, "#ccp-editor-prompt-save-button")
    async def handle_save_prompt(self, event: Button.Pressed) -> None:
        """Handle saving prompt from editor."""
        event.stop()
        await self.prompt_handler.handle_save_prompt()
    
    @on(Button.Pressed, "#ccp-editor-dict-save-button")
    async def handle_save_dictionary(self, event: Button.Pressed) -> None:
        """Handle saving dictionary from editor."""
        event.stop()
        await self.dictionary_handler.handle_save_dictionary()
    
    @on(Input.Changed, "#conv-char-search-input")
    async def handle_conversation_search(self, event: Input.Changed) -> None:
        """Handle conversation title search."""
        await self.conversation_handler.handle_search(event.value, "title")
    
    @on(Input.Changed, "#conv-char-keyword-search-input")
    async def handle_content_search(self, event: Input.Changed) -> None:
        """Handle conversation content search."""
        await self.conversation_handler.handle_search(event.value, "content")
    
    @on(Input.Changed, "#ccp-prompt-search-input")
    async def handle_prompt_search(self, event: Input.Changed) -> None:
        """Handle prompt search."""
        await self.prompt_handler.handle_search(event.value)
    
    # ===== Message Handlers =====
    
    async def on_view_change_message_requested(self, message: ViewChangeMessage.Requested) -> None:
        """Handle view change requests."""
        await self._switch_view(message.view_name)
    
    async def on_conversation_message_loaded(self, message: ConversationMessage.Loaded) -> None:
        """Handle conversation loaded message."""
        # Update UI to show conversation details
        self.selected_conversation_id = message.conversation_id
        await self.message_manager.load_conversation_messages(message.conversation_id)
        
        # Show conversation details section
        try:
            details_container = self.query_one("#conv-details-container")
            details_container.remove_class("hidden")
        except NoMatches:
            pass
    
    async def on_character_message_loaded(self, message: CharacterMessage.Loaded) -> None:
        """Handle character loaded message."""
        self.selected_character_id = message.character_id
        
        # Show character actions
        try:
            actions_container = self.query_one("#char-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass
    
    async def on_prompt_message_loaded(self, message: PromptMessage.Loaded) -> None:
        """Handle prompt loaded message."""
        self.selected_prompt_id = message.prompt_id
        
        # Show prompt actions
        try:
            actions_container = self.query_one("#prompt-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass
    
    async def on_dictionary_message_loaded(self, message: DictionaryMessage.Loaded) -> None:
        """Handle dictionary loaded message."""
        self.selected_dictionary_id = message.dictionary_id
        
        # Show dictionary actions
        try:
            actions_container = self.query_one("#dict-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass
    
    # ===== Reactive Watchers =====
    
    def watch_active_view(self, old_view: str, new_view: str) -> None:
        """Watch for active view changes."""
        logger.debug(f"Active view changed from {old_view} to {new_view}")
        
        # Post view changed message
        self.post_message(ViewChangeMessage.Changed(old_view, new_view))
    
    def watch_sidebar_collapsed(self, collapsed: bool) -> None:
        """Watch for sidebar collapse state changes."""
        logger.debug(f"Sidebar collapsed: {collapsed}")
    
    # ===== Private Helper Methods =====
    
    async def _switch_view(self, view_name: str) -> None:
        """Switch the active view in the content area.
        
        Args:
            view_name: Name of the view to switch to
        """
        try:
            # Hide all views
            view_containers = [
                "#ccp-conversation-messages-view",
                "#ccp-character-card-view",
                "#ccp-character-editor-view",
                "#ccp-prompt-editor-view",
                "#ccp-dictionary-view",
                "#ccp-dictionary-editor-view"
            ]
            
            for container_id in view_containers:
                try:
                    container = self.query_one(container_id)
                    container.add_class("hidden")
                except NoMatches:
                    continue
            
            # Show the requested view
            view_map = {
                "conversations": "#ccp-conversation-messages-view",
                "conversation_messages": "#ccp-conversation-messages-view",
                "character_card": "#ccp-character-card-view",
                "character_editor": "#ccp-character-editor-view",
                "prompt_editor": "#ccp-prompt-editor-view",
                "dictionary_view": "#ccp-dictionary-view",
                "dictionary_editor": "#ccp-dictionary-editor-view"
            }
            
            target_id = view_map.get(view_name)
            if target_id:
                target_view = self.query_one(target_id)
                target_view.remove_class("hidden")
                self.active_view = view_name
                logger.info(f"Switched to view: {view_name}")
            else:
                logger.warning(f"Unknown view requested: {view_name}")
                
        except Exception as e:
            logger.error(f"Error switching view: {e}", exc_info=True)
    
    # ===== State Management (Override from BaseAppScreen) =====
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the CCP screen."""
        return {
            "active_view": self.active_view,
            "selected_character_id": self.selected_character_id,
            "selected_conversation_id": self.selected_conversation_id,
            "selected_prompt_id": self.selected_prompt_id,
            "selected_dictionary_id": self.selected_dictionary_id,
            "sidebar_collapsed": self.sidebar_collapsed,
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state."""
        if "active_view" in state:
            self.active_view = state["active_view"]
        if "selected_character_id" in state:
            self.selected_character_id = state["selected_character_id"]
        if "selected_conversation_id" in state:
            self.selected_conversation_id = state["selected_conversation_id"]
        if "selected_prompt_id" in state:
            self.selected_prompt_id = state["selected_prompt_id"]
        if "selected_dictionary_id" in state:
            self.selected_dictionary_id = state["selected_dictionary_id"]
        if "sidebar_collapsed" in state:
            self.sidebar_collapsed = state["sidebar_collapsed"]