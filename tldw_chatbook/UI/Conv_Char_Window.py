# Conv_Char_Window.py
# Description: This file contains the UI functions for the Conv_Char_Window tab
#
# Imports
from typing import TYPE_CHECKING
#
# Third-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Input, ListView, Select, Collapsible, Label, TextArea
#
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Constants import TAB_CCP

# Configure logger with context
logger = logger.bind(module="Conv_Char_Window")

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class CCPWindow(Container):
    """
    Container for the Conversations, Characters & Prompts (CCP) Tab's UI.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("CCPWindow initialized.")

    def compose(self) -> ComposeResult:
        logger.debug("Composing CCPWindow UI")
        # Left Pane
        with VerticalScroll(id="conv-char-left-pane", classes="cc-left-pane"):
            yield Static("CCP Menu", classes="sidebar-title cc-section-title-text")
            with Collapsible(title="Characters", id="conv-char-characters-collapsible"):
                yield Button("Import Character Card", id="ccp-import-character-button",
                             classes="sidebar-button")
                yield Button("Create Character", id="ccp-create-character-button",
                             classes="sidebar-button")
                yield Select([], prompt="Select Character...", allow_blank=True, id="conv-char-character-select")
                yield Button("Load Character", id="ccp-right-pane-load-character-button", classes="sidebar-button")
                yield Button("Refresh List", id="ccp-refresh-character-list-button", classes="sidebar-button")
            with Collapsible(title="Conversations", id="conv-char-conversations-collapsible"):
                yield Button("Import Conversation", id="ccp-import-conversation-button",
                             classes="sidebar-button")
                yield Input(id="conv-char-search-input", placeholder="Search conversations...", classes="sidebar-input")
                yield Button("Search", id="conv-char-conversation-search-button", classes="sidebar-button")
                yield ListView(id="conv-char-search-results-list")
                yield Button("Load Selected", id="conv-char-load-button", classes="sidebar-button")
            with Collapsible(title="Prompts", id="ccp-prompts-collapsible"):
                yield Button("Import Prompt", id="ccp-import-prompt-button", classes="sidebar-button")
                yield Button("Create New Prompt", id="ccp-prompt-create-new-button", classes="sidebar-button")
                yield Input(id="ccp-prompt-search-input", placeholder="Search prompts...", classes="sidebar-input")
                yield ListView(id="ccp-prompts-listview", classes="sidebar-listview")
                yield Button("Load Selected Prompt", id="ccp-prompt-load-selected-button", classes="sidebar-button")

        yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), id="toggle-conv-char-left-sidebar",
                     classes="cc-sidebar-toggle-button", tooltip="Toggle left sidebar")

        # Center Pane
        logger.debug("Composing center pane")
        with VerticalScroll(id="conv-char-center-pane", classes="cc-center-pane"):
            # Container for conversation messages
            with Container(id="ccp-conversation-messages-view", classes="ccp-view-area"):
                yield Static("Conversation History", classes="pane-title", id="ccp-center-pane-title-conv")
                # Messages will be mounted dynamically here

            # Container for character card display (initially hidden by CSS)
            with Container(id="ccp-character-card-view", classes="ccp-view-area"):
                yield Static("Character Card Details", classes="pane-title", id="ccp-center-pane-title-char-card")
                # Character card details will be displayed here
                yield Static(id="ccp-card-image-placeholder") # Placeholder for character image
                yield Label("Name:")
                yield Static(id="ccp-card-name-display")
                yield Label("Description:")
                yield TextArea(id="ccp-card-description-display", read_only=True, classes="ccp-card-textarea")
                yield Label("Personality:")
                yield TextArea(id="ccp-card-personality-display", read_only=True, classes="ccp-card-textarea")
                yield Label("Scenario:")
                yield TextArea(id="ccp-card-scenario-display", read_only=True, classes="ccp-card-textarea")
                yield Label("First Message:")
                yield TextArea(id="ccp-card-first-message-display", read_only=True, classes="ccp-card-textarea")
                # V2 Character Card fields
                yield Label("Creator Notes:")
                yield TextArea(id="ccp-card-creator-notes-display", read_only=True, classes="ccp-card-textarea")
                yield Label("System Prompt:")
                yield TextArea(id="ccp-card-system-prompt-display", read_only=True, classes="ccp-card-textarea")
                yield Label("Post History Instructions:")
                yield TextArea(id="ccp-card-post-history-instructions-display", read_only=True, classes="ccp-card-textarea")
                yield Label("Alternate Greetings:")
                yield TextArea(id="ccp-card-alternate-greetings-display", read_only=True, classes="ccp-card-textarea")
                yield Label("Tags:")
                yield Static(id="ccp-card-tags-display")
                yield Label("Creator:")
                yield Static(id="ccp-card-creator-display")
                yield Label("Character Version:")
                yield Static(id="ccp-card-version-display")
                yield Label("Keywords:")
                yield Static(id="ccp-card-keywords-display")
                with Horizontal(classes="ccp-card-action-buttons"): # Added a class for potential styling
                    yield Button("Edit this Character", id="ccp-card-edit-button", variant="default")
                    yield Button("Save Changes", id="ccp-card-save-button", variant="success") # Added variant
                    yield Button("Clone Character", id="ccp-card-clone-button", variant="primary") # Added variant
            # Container for character editing UI (initially hidden by CSS)
            with Container(id="ccp-character-editor-view", classes="ccp-view-area"):
                yield Static("Character Editor", classes="pane-title", id="ccp-center-pane-title-char-editor")
                yield Label("Character Name:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-name-input", placeholder="Character name...", classes="sidebar-input")
                yield Button("✨ Generate All Fields", id="ccp-generate-all-button", classes="ai-generate-all-button", variant="success")
                yield Label("Avatar Path/URL:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-avatar-input", placeholder="Path or URL to avatar image...", classes="sidebar-input")
                yield Label("Description:", classes="sidebar-label")
                with Horizontal(classes="field-with-ai-button"):
                    yield TextArea(id="ccp-editor-char-description-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                    yield Button("✨ Generate", id="ccp-generate-description-button", classes="ai-generate-button", variant="primary")
                yield Label("Personality:", classes="sidebar-label")
                with Horizontal(classes="field-with-ai-button"):
                    yield TextArea(id="ccp-editor-char-personality-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                    yield Button("✨ Generate", id="ccp-generate-personality-button", classes="ai-generate-button", variant="primary")
                yield Label("Scenario:", classes="sidebar-label")
                with Horizontal(classes="field-with-ai-button"):
                    yield TextArea(id="ccp-editor-char-scenario-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                    yield Button("✨ Generate", id="ccp-generate-scenario-button", classes="ai-generate-button", variant="primary")
                yield Label("First Message (Greeting):", classes="sidebar-label")
                with Horizontal(classes="field-with-ai-button"):
                    yield TextArea(id="ccp-editor-char-first-message-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                    yield Button("✨ Generate", id="ccp-generate-first-message-button", classes="ai-generate-button", variant="primary")
                yield Label("Keywords (comma-separated):", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-keywords-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                # V2 Character Card Fields
                yield Label("Creator Notes:", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-creator-notes-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("System Prompt:", classes="sidebar-label")
                with Horizontal(classes="field-with-ai-button"):
                    yield TextArea(id="ccp-editor-char-system-prompt-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                    yield Button("✨ Generate", id="ccp-generate-system-prompt-button", classes="ai-generate-button", variant="primary")
                yield Label("Post History Instructions:", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-post-history-instructions-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Alternate Greetings (one per line):", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-alternate-greetings-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Tags (comma-separated):", classes="sidebar-label")
                yield Input(id="ccp-editor-char-tags-input", placeholder="e.g., fantasy, anime, helpful", classes="sidebar-input")
                yield Label("Creator:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-creator-input", placeholder="Creator name", classes="sidebar-input")
                yield Label("Character Version:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-version-input", placeholder="e.g., 1.0", classes="sidebar-input")
                with Horizontal(classes="ccp-prompt-action-buttons"):
                    yield Button("Save Character", id="ccp-editor-char-save-button", variant="success", classes="sidebar-button")
                    yield Button("Clone Character", id="ccp-editor-char-clone-button", classes="sidebar-button")
                    yield Button("Cancel Edit", id="ccp-editor-char-cancel-button", variant="error", classes="sidebar-button hidden")

            # Container for prompt editing UI (initially hidden by CSS)
            with Container(id="ccp-prompt-editor-view", classes="ccp-view-area"):
                yield Static("Prompt Editor", classes="pane-title", id="ccp-center-pane-title-prompt")
                yield Label("Prompt Name:", classes="sidebar-label")
                yield Input(id="ccp-editor-prompt-name-input", placeholder="Unique prompt name...",
                            classes="sidebar-input")
                yield Label("Author:", classes="sidebar-label")
                yield Input(id="ccp-editor-prompt-author-input", placeholder="Author name...", classes="sidebar-input")
                yield Label("Details/Description:", classes="sidebar-label")
                yield TextArea("", id="ccp-editor-prompt-description-textarea",
                               classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("System Prompt:", classes="sidebar-label")
                yield TextArea("", id="ccp-editor-prompt-system-textarea",
                               classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("User Prompt (Template):", classes="sidebar-label")
                yield TextArea("", id="ccp-editor-prompt-user-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Keywords (comma-separated):", classes="sidebar-label")
                yield TextArea("", id="ccp-editor-prompt-keywords-textarea",
                               classes="sidebar-textarea ccp-prompt-textarea")
                with Horizontal(classes="ccp-prompt-action-buttons"):
                    yield Button("Save Prompt", id="ccp-editor-prompt-save-button", variant="success",
                                 classes="sidebar-button")
                    yield Button("Clone Prompt", id="ccp-editor-prompt-clone-button", classes="sidebar-button")

        # Button to toggle the right sidebar for CCP tab
        yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                     id="toggle-conv-char-right-sidebar", classes="cc-sidebar-toggle-button", tooltip="Toggle right sidebar")

        # Right Pane
        logger.debug("Composing right pane")
        with VerticalScroll(id="conv-char-right-pane", classes="cc-right-pane"):
            yield Static("Details & Settings", classes="sidebar-title") # This title is for the whole pane

            # Conversation Details Collapsible
            with Collapsible(title="Conversation Details", id="ccp-conversation-details-collapsible",
                             collapsed=True):
                yield Static("Title:", classes="sidebar-label")
                yield Input(id="conv-char-title-input", placeholder="Conversation title...", classes="sidebar-input")
                yield Static("Keywords:", classes="sidebar-label")
                yield TextArea("", id="conv-char-keywords-input", classes="conv-char-keywords-textarea")
                yield Button("Save Conversation Details", id="conv-char-save-details-button", classes="sidebar-button")
                yield Static("Export Options", classes="sidebar-label export-label")
                yield Button("Export as Text", id="conv-char-export-text-button", classes="sidebar-button")
                yield Button("Export as JSON", id="conv-char-export-json-button", classes="sidebar-button")

            # Prompt Details Collapsible (for the right-pane prompt editor)
            with Collapsible(title="Prompt Options", id="ccp-prompt-details-collapsible", collapsed=True):
                yield Static("Prompt metadata or non-editor actions will appear here.", classes="sidebar-label")
            with Collapsible(title="Prompt Deletion", id="ccp-prompt-details-collapsible-2", collapsed=True):
                yield Button("Delete Prompt", id="ccp-editor-prompt-delete-button", variant="error",
                             classes="sidebar-button")
            # Characters Collapsible
            with Collapsible(title="Delete Character", id="ccp-characters-collapsible", collapsed=True):
                yield Button("Delete Character", id="ccp-character-delete-button", variant="error",)
                # Add other character related widgets here if needed in the future

#
# End of Conv_Char_Window.py
#######################################################################################################################
