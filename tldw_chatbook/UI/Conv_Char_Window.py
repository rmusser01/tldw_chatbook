# Conv_Char_Window.py
# Description: This file contains the UI functions for the Conv_Char_Window tab
#
# Imports
from typing import TYPE_CHECKING
#
# Third-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Input, ListView, Select, Collapsible, Label, TextArea
#
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Constants import TAB_CCP

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

    def compose(self) -> ComposeResult:
        # Left Pane
        with VerticalScroll(id="conv-char-left-pane", classes="cc-left-pane"):
            yield Static("CCP Menu", classes="sidebar-title cc-section-title-text")
            with Collapsible(title="Characters", id="conv-char-characters-collapsible"):
                yield Button("Import Character Card", id="ccp-import-character-button",
                             classes="sidebar-button")
                yield Select([], prompt="Select Character...", allow_blank=True, id="conv-char-character-select")
                yield Button("Load Character", id="ccp-right-pane-load-character-button", classes="sidebar-button")
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
                     classes="cc-sidebar-toggle-button")

        # Center Pane
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
                yield TextArea(id="ccp-card-first-message-display", classes="ccp-card-textarea")
                with Horizontal(classes="ccp-card-action-buttons"): # Added a class for potential styling
                    yield Button("Edit this Character", id="ccp-card-edit-button", variant="default")
                    yield Button("Save Changes", id="ccp-card-save-button", variant="success") # Added variant
                    yield Button("Clone Character", id="ccp-card-clone-button", variant="primary") # Added variant
            # Container for character editing UI (initially hidden by CSS)
            with Container(id="ccp-character-editor-view", classes="ccp-view-area"):
                yield Static("Character Editor", classes="pane-title", id="ccp-center-pane-title-char-editor")
                yield Label("Character Name:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-name-input", placeholder="Character name...", classes="sidebar-input")
                yield Label("Avatar Path/URL:", classes="sidebar-label")
                yield Input(id="ccp-editor-char-avatar-input", placeholder="Path or URL to avatar image...", classes="sidebar-input")
                yield Label("Description:", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-description-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Personality:", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-personality-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Scenario:", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-scenario-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("First Message (Greeting):", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-first-message-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                yield Label("Keywords (comma-separated):", classes="sidebar-label")
                yield TextArea(id="ccp-editor-char-keywords-textarea", classes="sidebar-textarea ccp-prompt-textarea")
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
                     id="toggle-conv-char-right-sidebar", classes="cc-sidebar-toggle-button")

        # Right Pane
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
