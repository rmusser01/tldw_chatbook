"""Conversations, Characters & Prompts (CCP) Screen.

This screen provides a unified interface for managing conversations, characters,
prompts, and dictionaries following Textual best practices with Screen-based architecture.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button, Input, ListView, Select, Collapsible, Label, TextArea, Checkbox
from textual.reactive import reactive
from textual import on, work
from textual.css.query import NoMatches
from textual.message import Message

from ..Navigation.base_app_screen import BaseAppScreen
from ...Widgets.destination_workbench import DestinationModeStrip
from ...Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from ...Chat.chat_handoff_models import ChatHandoffPayload

# Import widget components
from ...Widgets.CCP_Widgets import (
    CCPSidebarWidget,
    ContinueConversationRequested,
    ConversationSearchRequested,
    ConversationLoadRequested,
    CharacterLoadRequested,
    CharacterEditorCancelled,
    CharacterSaveRequested,
    StartChatRequested,
    EditCharacterRequested,
    EditPersonaRequested,
    PersonaLoadRequested,
    PersonaSaveRequested,
    StartPersonaChatRequested,
    PromptLoadRequested,
    DictionaryLoadRequested,
    ImportRequested,
    CreateRequested,
    RefreshRequested,
)

# Import modular handlers and enhancements
from ..CCP_Modules import (
    CCPConversationHandler,
    CCPCharacterHandler,
    CCPPersonaHandler,
    CCPPromptHandler,
    CCPDictionaryHandler,
    CCPMessageManager,
    CCPSidebarHandler,
    ConversationMessage,
    CharacterMessage,
    PersonaMessage,
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

CCP_DESTINATION_MODE_VIEWS = {
    "ccp-personas-mode-button": ("persona_profiles", "Personas"),
    "ccp-characters-mode-button": ("character_card", "Characters"),
    "ccp-prompts-mode-button": ("prompt_editor", "Prompts"),
    "ccp-dictionaries-mode-button": ("dictionary_view", "Dictionaries"),
    "ccp-lore-mode-button": ("lore_view", "Lore"),
    "ccp-import-export-mode-button": ("import_export_view", "Import/Export"),
}

CCP_MODE_PLACEHOLDERS = {
    "lore_view": (
        "Lore",
        "Lore mode is not wired yet. Character cards and dictionaries remain available in this slice.",
    ),
    "import_export_view": (
        "Import / Export",
        "Use the inspector actions to import or export local character cards.",
    ),
}

CCP_ACTION_BUTTON_TOOLTIPS = {
    "continue-conversation-btn": "Continue the loaded conversation from CCP.",
    "export-conversation-btn": "Export the loaded conversation from CCP.",
    "clear-conversation-btn": "Clear the loaded CCP conversation history.",
    "ccp-attach-selected-to-console": "Select a character before attaching it to Console.",
    "ccp-start-selected-chat": "Select a character before starting a Console chat.",
    "ccp-import-character-native": "Import a local character card into CCP.",
    "ccp-export-character-native": "Select a character before exporting it from CCP.",
    "start-chat-btn": "Start a Console chat with this character.",
    "edit-character-btn": "Open this character in the CCP editor.",
    "clone-character-btn": "Create a copy of this character.",
    "export-character-btn": "Export this character card.",
    "delete-character-btn": "Delete this character card.",
    "ccp-persona-start-chat-button": "Start a Console chat with this persona.",
    "ccp-persona-edit-button": "Open this persona in the CCP editor.",
    "ccp-persona-save-button": "Save the current persona profile.",
}

CCP_ACTION_TOOLTIP_PREFIXES = (
    ("generate-", "Generate AI-assisted content for this CCP field."),
    ("add-", "Add a new item or value to this CCP editor."),
    ("remove-", "Remove this item or value from the CCP editor."),
    ("save-", "Save the current CCP edits."),
    ("reset-", "Reset unsaved changes in this CCP editor."),
    ("cancel-", "Cancel editing and return to the previous CCP view."),
    ("delete-", "Delete the selected CCP item."),
    ("edit-", "Open the selected CCP item for editing."),
    ("clone-", "Create a copy of the selected CCP item."),
    ("export-", "Export the selected CCP item."),
    ("import-", "Import content into this CCP editor."),
    ("test-", "Test the current CCP prompt configuration."),
    ("clear-", "Clear the current CCP search or editor value."),
    ("update-", "Update the selected CCP entry."),
)


# ========== Custom Messages ==========

class ConversationSelected(Message):
    """Message sent when a conversation is selected."""
    def __init__(self, conversation_id: int, title: str) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.title = title


class CharacterSelected(Message):
    """Message sent when a character is selected."""
    def __init__(self, character_id: int, name: str) -> None:
        super().__init__()
        self.character_id = character_id
        self.name = name


class PromptSelected(Message):
    """Message sent when a prompt is selected."""
    def __init__(self, prompt_id: int, name: str) -> None:
        super().__init__()
        self.prompt_id = prompt_id
        self.name = name


class DictionarySelected(Message):
    """Message sent when a dictionary is selected."""
    def __init__(self, dictionary_id: int, name: str) -> None:
        super().__init__()
        self.dictionary_id = dictionary_id
        self.name = name


class ViewSwitchRequested(Message):
    """Message sent when a view switch is requested."""
    def __init__(self, view_name: str) -> None:
        super().__init__()
        self.view_name = view_name


# ========== State Management ==========

@dataclass
class CCPScreenState:
    """Encapsulates all state for the CCP screen.
    
    This dataclass centralizes all state management for the Conversations,
    Characters & Prompts screen, following Textual best practices.
    """
    
    # Current view
    active_view: str = "conversations"  # conversations, character_card, character_editor, etc.
    
    # Selected items
    selected_conversation_id: Optional[str] = None
    selected_conversation_title: str = ""
    selected_conversation_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    selected_character_id: Optional[str] = None
    selected_persona_id: Optional[str] = None
    selected_character_name: str = ""
    selected_character_data: Dict[str, Any] = field(default_factory=dict)
    is_editing_character: bool = False
    
    selected_prompt_id: Optional[Union[int, str]] = None
    selected_prompt_name: str = ""
    selected_prompt_data: Dict[str, Any] = field(default_factory=dict)
    is_editing_prompt: bool = False
    
    selected_dictionary_id: Optional[int] = None
    selected_dictionary_name: str = ""
    selected_dictionary_data: Dict[str, Any] = field(default_factory=dict)
    is_editing_dictionary: bool = False
    
    # Search state
    conversation_search_term: str = ""
    conversation_search_type: str = "title"  # title, content, tags
    conversation_search_results: List[Dict[str, Any]] = field(default_factory=list)
    include_character_chats: bool = True
    search_all_characters: bool = True
    
    prompt_search_term: str = ""
    prompt_search_results: List[Dict[str, Any]] = field(default_factory=list)
    
    worldbook_search_term: str = ""
    worldbook_search_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # UI state
    sidebar_collapsed: bool = False
    conversation_details_visible: bool = False
    character_actions_visible: bool = False
    prompt_actions_visible: bool = False
    dictionary_actions_visible: bool = False
    
    # Lists cache
    character_list: List[Dict[str, Any]] = field(default_factory=list)
    dictionary_list: List[Dict[str, Any]] = field(default_factory=list)
    worldbook_list: List[Dict[str, Any]] = field(default_factory=list)
    
    # Loading states
    is_loading_conversation: bool = False
    is_loading_character: bool = False
    is_loading_prompt: bool = False
    is_loading_dictionary: bool = False
    is_saving: bool = False
    
    # Validation flags
    has_unsaved_changes: bool = False
    validation_errors: Dict[str, str] = field(default_factory=dict)

    def reset_for_backend_change(self) -> None:
        """Clear entity/session state when the backing mode changes."""
        self.selected_character_id = None
        self.selected_persona_id = None
        self.selected_conversation_id = None
        self.selected_character_name = ""
        self.selected_character_data = {}
        self.selected_conversation_title = ""
        self.selected_conversation_messages = []
        self.conversation_search_results = []
        self.character_actions_visible = False
        self.conversation_details_visible = False


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
        min-width: 15; /* Ensure buttons don't get too small */
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

    #ccp-shell {
        width: 100%;
        height: 100%;
        min-height: 0;
    }

    #ccp-destination-title,
    #ccp-destination-purpose,
    #ccp-status-row,
    #ccp-mode-strip {
        height: 1;
        min-height: 1;
    }

    .ccp-mode-button {
        height: 1;
        min-height: 1;
        width: 15;
        min-width: 12;
        margin-right: 1;
        border: none;
        background: $surface;
    }

    .ccp-mode-button.is-active {
        text-style: bold;
        background: $surface-lighten-1;
        color: white;
    }

    #ccp-mode-label {
        width: 7;
        min-width: 7;
    }

    #ccp-workbench {
        width: 100%;
        height: 1fr;
        min-height: 0;
        padding: 1;
        margin: 0;
    }

    #ccp-character-library-pane.ds-inspector,
    #ccp-behavior-detail-pane.ds-inspector,
    #ccp-attachment-inspector-pane.ds-inspector {
        height: 1fr !important;
        min-height: 0 !important;
        min-width: 0 !important;
        padding: 1 !important;
    }

    #ccp-character-library-pane {
        width: 27%;
        min-width: 28;
    }

    #ccp-behavior-detail-pane {
        width: 1fr;
        min-width: 45;
    }

    #ccp-attachment-inspector-pane {
        width: 27%;
        min-width: 30;
    }

    #ccp-list-detail-divider,
    #ccp-detail-inspector-divider {
        width: 1;
        min-width: 1;
        height: 100%;
        min-height: 0;
        background: #6f6f6f;
    }

    #ccp-character-list,
    #ccp-detail-widget-stack {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
    }

    .ccp-character-list-button {
        width: 100%;
        height: 1;
        min-height: 1;
        margin: 0 0 1 0;
        border: none;
    }

    .ccp-character-list-button.is-active {
        text-style: bold;
    }

    #ccp-attachment-actions Button {
        width: 100%;
        height: 1;
        min-height: 1;
        margin-bottom: 1;
    }
    """
    
    # Reactive state using proper Textual patterns
    state: reactive[CCPScreenState] = reactive(CCPScreenState)
    
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
        
        # Initialize state with a fresh instance
        self.state = CCPScreenState()
        
        # Initialize modular handlers
        self.conversation_handler = CCPConversationHandler(self)
        self.character_handler = CCPCharacterHandler(self)
        self.persona_handler = CCPPersonaHandler(self)
        self.prompt_handler = CCPPromptHandler(self)
        self.dictionary_handler = CCPDictionaryHandler(self)
        self.message_manager = CCPMessageManager(self)
        self.sidebar_handler = CCPSidebarHandler(self)
        
        # Initialize loading manager for async operation feedback
        self.loading_manager = LoadingManager(self)
        self._character_button_to_id: Dict[str, str] = {}
        
        # Setup enhancements (validation, loading indicators)
        setup_ccp_enhancements(self)
        
        logger.debug("CCPScreen initialized with reactive state and modular handlers")

    def compose_content(self) -> ComposeResult:
        """Compose the destination-native Personas workbench for the CCP route."""
        logger.debug("Composing destination-native CCPScreen UI")
        
        # Import our widget components
        from ...Widgets.CCP_Widgets import (
            CCPConversationViewWidget,
            CCPCharacterCardWidget,
            CCPCharacterEditorWidget,
            CCPPersonaCardWidget,
            CCPPersonaEditorWidget,
            CCPPromptEditorWidget,
            CCPDictionaryEditorWidget,
        )
        
        with Vertical(id="ccp-shell"):
            yield Static(
                "Personas | Behavior, characters, prompts, lore | Ready | Local/Server",
                id="ccp-destination-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Behavior Profile Workbench for personas, characters, prompts, dictionaries, lore, and Console attachments.",
                id="ccp-destination-purpose",
                classes="destination-purpose",
            )
            yield Static(
                "Mode: Characters | Source: local character DB | Attachments: Console / Workflows / ACP / Skills",
                id="ccp-status-row",
                classes="destination-status-row",
            )
            with DestinationModeStrip(id="ccp-mode-strip", classes="destination-mode-strip"):
                yield Static("Modes:", id="ccp-mode-label", classes="destination-section")
                for button_id, label in (
                    ("ccp-personas-mode-button", "Personas"),
                    ("ccp-characters-mode-button", "Characters"),
                    ("ccp-prompts-mode-button", "Prompts"),
                    ("ccp-dictionaries-mode-button", "Dictionaries"),
                    ("ccp-lore-mode-button", "Lore"),
                    ("ccp-import-export-mode-button", "Import/Export"),
                ):
                    classes = "ccp-mode-button"
                    button = Button(label, id=button_id, classes=classes)
                    button.tooltip = f"Switch CCP workbench to {label} mode."
                    yield button

            with Horizontal(id="ccp-workbench", classes="ds-panel destination-workbench"):
                with Vertical(
                    id="ccp-character-library-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield Static("Character Library", classes="destination-section ccp-column-title")
                    yield Static(
                        "Select a local character card to inspect, edit, or attach.",
                        id="ccp-character-library-help",
                        classes="destination-purpose",
                    )
                    with Vertical(id="ccp-character-list"):
                        yield Static("Loading local characters...", id="ccp-character-list-loading")

                yield self._column_divider("ccp-list-detail-divider")

                with Vertical(
                    id="ccp-behavior-detail-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield Static("Character Detail", classes="destination-section ccp-column-title")
                    yield Static(
                        "No character selected. Choose a character from the library.",
                        id="ccp-detail-selection-summary",
                    )
                    with Container(id="ccp-detail-widget-stack"):
                        yield CCPConversationViewWidget(parent_screen=self)
                        yield CCPCharacterCardWidget(parent_screen=self)
                        yield CCPCharacterEditorWidget(parent_screen=self)
                        yield CCPPersonaCardWidget(parent_screen=self)
                        yield CCPPersonaEditorWidget(parent_screen=self)
                        yield CCPPromptEditorWidget(parent_screen=self)
                        yield CCPDictionaryEditorWidget(parent_screen=self)
                        with Vertical(id="ccp-mode-placeholder-view", classes="hidden"):
                            yield Static("Mode unavailable", id="ccp-mode-placeholder-title")
                            yield Static(
                                "This mode is not wired in the destination-native route yet.",
                                id="ccp-mode-placeholder-body",
                            )

                yield self._column_divider("ccp-detail-inspector-divider")

                with Vertical(
                    id="ccp-attachment-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield Static("Attach / Validate", classes="destination-section ccp-column-title")
                    yield Static("Selected target", classes="destination-section")
                    yield Static("Selected: none", id="ccp-selected-target-name")
                    yield Static("Type: character", id="ccp-selected-target-kind")
                    yield Static("Runtime target: not staged", id="ccp-selected-runtime-target")
                    yield Static("Attachment readiness", classes="destination-section")
                    yield Static(
                        "Console: blocked until a character is selected",
                        id="ccp-console-readiness",
                    )
                    yield Static("Workflows: ready after selection", id="ccp-workflows-readiness")
                    yield Static("ACP: needs runtime payloads", id="ccp-acp-readiness")
                    yield Static(
                        "Skills: ready for compatible behavior skills",
                        id="ccp-skills-readiness",
                    )
                    with Vertical(id="ccp-attachment-actions"):
                        yield Button(
                            "Attach to Console",
                            id="ccp-attach-selected-to-console",
                            disabled=True,
                        )
                        yield Button(
                            "Start Chat",
                            id="ccp-start-selected-chat",
                            disabled=True,
                        )
                        yield Button("Import Character", id="ccp-import-character-native")
                        yield Button(
                            "Export Character",
                            id="ccp-export-character-native",
                            disabled=True,
                        )

    async def on_mount(self) -> None:
        """Handle post-composition setup."""
        super().on_mount()  # Don't await - parent's on_mount is not async
        
        # Cache widget references
        self._cache_widget_references()
        
        # Setup loading manager widget
        await self.loading_manager.setup()
        
        # Initialize UI state
        await self._initialize_ui_state()
        self._ensure_action_button_tooltips()
        
        logger.debug("CCPScreen mounted and initialized with enhancements")
    def _cache_widget_references(self) -> None:
        """Cache frequently accessed widgets."""
        self._sidebar = None
        try:
            self._content_area = self.query_one("#ccp-behavior-detail-pane")
            self._message_area = self.query_one("#ccp-conversation-messages-view")
        except NoMatches as e:
            logger.debug(f"Optional CCP widget reference unavailable during cache: {e}")

    async def _initialize_ui_state(self) -> None:
        """Initialize the UI state."""
        # Refresh lists
        await self.character_handler.refresh_character_list()
        await self.persona_handler.refresh_persona_list()
        await self.dictionary_handler.refresh_dictionary_list()
        
        # Characters mode is the first destination-native CCP slice.
        await self._switch_view("character_card")

    @staticmethod
    def _column_divider(widget_id: str) -> Static:
        divider = Static("", id=widget_id, classes="destination-pane-divider")
        divider.styles.width = 1
        divider.styles.min_width = 1
        return divider

    async def refresh_character_library_list(self, characters: List[Dict[str, Any]]) -> None:
        """Refresh the destination-native character list pane."""
        try:
            character_list = self.query_one("#ccp-character-list")
        except NoMatches:
            return

        await character_list.remove_children()
        self._character_button_to_id = {}
        self.state.character_list = characters

        if not characters:
            await character_list.mount(
                Static(
                    "No characters yet. Import a character card or create one here.",
                    id="ccp-character-list-empty",
                )
            )
            return

        buttons: list[Button] = []
        for index, character in enumerate(characters):
            character_id = str(character.get("id", ""))
            name = str(character.get("name") or "Unnamed")
            button_id = f"ccp-character-list-item-{index}"
            self._character_button_to_id[button_id] = character_id
            classes = "ccp-character-list-button"
            if character_id == self.state.selected_character_id:
                classes = f"{classes} is-active"
            buttons.append(
                Button(
                    name,
                    id=button_id,
                    classes=classes,
                    tooltip=f"Load {name}.",
                )
            )
        if buttons:
            await character_list.mount(*buttons)

    def _update_destination_selection_summary(self) -> None:
        """Update destination-native selected-target summary copy."""
        selected_name = self.state.selected_character_name or "none"
        selected_id = self.state.selected_character_id or "not staged"
        has_selection = bool(self.state.selected_character_id)

        updates = {
            "#ccp-detail-selection-summary": (
                f"Selected character: {selected_name}" if has_selection
                else "No character selected. Choose a character from the library."
            ),
            "#ccp-selected-target-name": f"Selected: {selected_name}",
            "#ccp-selected-target-kind": "Type: character",
            "#ccp-selected-runtime-target": f"Runtime target: local:character:{selected_id}",
            "#ccp-console-readiness": (
                "Console: ready to attach selected character"
                if has_selection
                else "Console: blocked until a character is selected"
            ),
            "#ccp-workflows-readiness": (
                "Workflows: ready to use selected behavior context"
                if has_selection
                else "Workflows: ready after selection"
            ),
        }
        for selector, text in updates.items():
            try:
                self.query_one(selector, Static).update(text)
            except NoMatches:
                continue

        action_tooltips = {
            "#ccp-attach-selected-to-console": (
                "Attach the selected character to Console context."
                if has_selection
                else "Select a character before attaching it to Console."
            ),
            "#ccp-start-selected-chat": (
                "Start a Console chat with the selected character."
                if has_selection
                else "Select a character before starting a Console chat."
            ),
            "#ccp-export-character-native": (
                "Export the selected local character card."
                if has_selection
                else "Select a character before exporting it from CCP."
            ),
        }
        for selector, tooltip in action_tooltips.items():
            try:
                button = self.query_one(selector, Button)
                button.disabled = not has_selection
                button.tooltip = tooltip
            except NoMatches:
                continue

    def _ensure_action_button_tooltips(self) -> None:
        """Fill missing CCP action tooltips after legacy child widgets compose."""
        for button in self.query(Button):
            tooltip = getattr(button, "tooltip", None)
            if tooltip is not None and str(tooltip).strip().lower() not in {"", "none"}:
                continue
            button.tooltip = self._default_button_tooltip(button)

    def _default_button_tooltip(self, button: Button) -> str:
        button_id = str(button.id or "")
        if button_id in CCP_ACTION_BUTTON_TOOLTIPS:
            return CCP_ACTION_BUTTON_TOOLTIPS[button_id]
        for prefix, tooltip in CCP_ACTION_TOOLTIP_PREFIXES:
            if button_id.startswith(prefix):
                return tooltip
        label = self._button_label_text(button)
        if label:
            return f"{label} in the CCP workbench."
        return "Run this CCP workbench action."

    @staticmethod
    def _button_label_text(button: Button) -> str:
        label = getattr(button.label, "plain", None)
        if label is None:
            label = str(button.label or "")
        return " ".join(str(label).split())

    # ===== Event Handlers using @on decorators =====
    
    @on(Button.Pressed, "#toggle-ccp-sidebar")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Handle sidebar toggle button press."""
        event.stop()
        
        # Update state
        new_state = self.state
        new_state.sidebar_collapsed = not new_state.sidebar_collapsed
        self.state = new_state
        
        # Let the handler do any additional work
        await self.sidebar_handler.toggle_sidebar()

    @on(Button.Pressed, ".ccp-character-list-button")
    async def handle_character_library_selection(self, event: Button.Pressed) -> None:
        """Load a character from the destination-native character library pane."""
        event.stop()
        character_id = self._character_button_to_id.get(str(event.button.id or ""))
        if character_id:
            await self.character_handler.load_character(character_id)

    @on(Button.Pressed, "#ccp-import-character-native")
    async def handle_native_character_import(self, event: Button.Pressed) -> None:
        """Import a character from the destination-native inspector action."""
        event.stop()
        await self.character_handler.handle_import()

    @on(Button.Pressed, ".ccp-mode-button")
    async def handle_destination_mode_button(self, event: Button.Pressed) -> None:
        """Switch destination-native CCP modes from the mode strip."""
        event.stop()
        view_config = CCP_DESTINATION_MODE_VIEWS.get(str(event.button.id or ""))
        if view_config is None:
            return
        view_name, _ = view_config
        await self._switch_view(view_name)

    @on(Button.Pressed, "#ccp-attach-selected-to-console")
    async def handle_attach_selected_character_to_console(self, event: Button.Pressed) -> None:
        """Stage the selected character card into Console context."""
        event.stop()
        self._attach_selected_character_to_console()

    @on(Button.Pressed, "#ccp-start-selected-chat")
    async def handle_start_selected_character_chat(self, event: Button.Pressed) -> None:
        """Start a character-backed Console chat from the inspector action."""
        event.stop()
        await self._launch_character_in_chat(self.state.selected_character_id)

    @on(Button.Pressed, "#ccp-export-character-native")
    async def handle_export_selected_character(self, event: Button.Pressed) -> None:
        """Export the selected character card from the inspector action."""
        event.stop()
        await self.character_handler.handle_export_character()
    
    # Note: These button handlers are now handled by the sidebar widget
    # The sidebar widget posts messages that we handle in the message handlers above
    
    # Editor button handlers - these remain here as they're part of the main content area
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

    @on(Button.Pressed, "#ccp-editor-prompt-record-usage-button")
    async def handle_record_prompt_usage(self, event: Button.Pressed) -> None:
        """Handle recording usage for the selected prompt."""
        event.stop()
        await self.prompt_handler.handle_record_prompt_usage()

    @on(Button.Pressed, "#ccp-editor-prompt-list-versions-button")
    async def handle_list_prompt_versions(self, event: Button.Pressed) -> None:
        """Handle listing server prompt versions."""
        event.stop()
        await self.prompt_handler.handle_list_prompt_versions()

    @on(Button.Pressed, "#ccp-editor-prompt-restore-version-button")
    async def handle_restore_prompt_version(self, event: Button.Pressed) -> None:
        """Handle restoring a server prompt version."""
        event.stop()
        await self.prompt_handler.handle_restore_prompt_version()
    
    @on(Button.Pressed, "#ccp-editor-dict-save-button")
    async def handle_save_dictionary(self, event: Button.Pressed) -> None:
        """Handle saving dictionary from editor."""
        event.stop()
        await self.dictionary_handler.handle_save_dictionary()
    
    # Note: Input change handlers are now handled by the sidebar widget
    # which posts messages that we handle above
    
    # ===== Message Handlers =====
    
    async def on_view_change_message_requested(self, message: ViewChangeMessage.Requested) -> None:
        """Handle view change requests."""
        await self._switch_view(message.view_name)
    
    # ===== Sidebar Widget Message Handlers =====
    
    async def on_conversation_search_requested(self, message: ConversationSearchRequested) -> None:
        """Handle conversation search request from sidebar."""
        await self.conversation_handler.handle_search(message.search_term, message.search_type)
    
    async def on_conversation_load_requested(self, message: ConversationLoadRequested) -> None:
        """Handle conversation load request from sidebar."""
        if message.conversation_id:
            await self.conversation_handler.load_conversation(message.conversation_id)
        else:
            await self.conversation_handler.handle_load_selected()
    
    async def on_character_load_requested(self, message: CharacterLoadRequested) -> None:
        """Handle character load request from sidebar."""
        if message.character_id:
            await self.character_handler.load_character(message.character_id)
        else:
            await self.character_handler.handle_load_character()

    async def on_persona_load_requested(self, message: PersonaLoadRequested) -> None:
        """Handle persona load request from sidebar."""
        if message.persona_id:
            await self.persona_handler.load_persona(message.persona_id)
    
    async def on_prompt_load_requested(self, message: PromptLoadRequested) -> None:
        """Handle prompt load request from sidebar."""
        if message.prompt_id:
            await self.prompt_handler.load_prompt(message.prompt_id)
        else:
            await self.prompt_handler.handle_load_selected()
    
    async def on_dictionary_load_requested(self, message: DictionaryLoadRequested) -> None:
        """Handle dictionary load request from sidebar."""
        if message.dictionary_id:
            await self.dictionary_handler.load_dictionary(message.dictionary_id)
        else:
            await self.dictionary_handler.handle_load_dictionary()
    
    async def on_import_requested(self, message: ImportRequested) -> None:
        """Handle import request from sidebar."""
        if message.item_type == "conversation":
            await self.conversation_handler.handle_import()
        elif message.item_type == "character":
            await self.character_handler.handle_import()
        elif message.item_type == "prompt":
            await self.prompt_handler.handle_import()
        elif message.item_type == "dictionary":
            await self.dictionary_handler.handle_import()
        elif message.item_type == "worldbook":
            # Handle worldbook import
            pass
    
    async def on_create_requested(self, message: CreateRequested) -> None:
        """Handle create request from sidebar."""
        if message.item_type == "character":
            await self.character_handler.handle_create()
        elif message.item_type == "persona":
            await self.persona_handler.handle_create_persona()
        elif message.item_type == "prompt":
            await self.prompt_handler.handle_create()
        elif message.item_type == "dictionary":
            await self.dictionary_handler.handle_create()
        elif message.item_type == "worldbook":
            # Handle worldbook creation
            pass

    async def on_edit_persona_requested(self, message: EditPersonaRequested) -> None:
        """Handle persona edit requests from the persona card."""
        await self.persona_handler.handle_edit_persona(message.persona_id)

    async def on_start_chat_requested(self, message: StartChatRequested) -> None:
        """Launch a new character-backed chat session in main chat."""
        await self._launch_character_in_chat(message.character_id)

    async def on_edit_character_requested(self, message: EditCharacterRequested) -> None:
        """Handle character edit requests from the character card."""
        await self.character_handler.handle_edit_character()

    async def on_character_save_requested(self, message: CharacterSaveRequested) -> None:
        """Handle character save requests from the character editor."""
        await self.character_handler.save_character_data(message.character_data)

    async def on_character_editor_cancelled(self, message: CharacterEditorCancelled) -> None:
        """Return to the loaded character card when character editing is cancelled."""
        await self._switch_view("character_card")

    async def on_start_persona_chat_requested(self, message: StartPersonaChatRequested) -> None:
        """Launch a new persona-backed chat session in main chat."""
        await self._launch_persona_in_chat(message.persona_id)

    async def on_persona_save_requested(self, message: PersonaSaveRequested) -> None:
        """Handle persona save requests from the persona editor."""
        await self.persona_handler.save_persona(message.persona_data)
    
    async def on_refresh_requested(self, message: RefreshRequested) -> None:
        """Handle refresh request from sidebar."""
        if message.list_type == "character":
            await self.character_handler.refresh_character_list()
        elif message.list_type == "persona":
            await self.persona_handler.refresh_persona_list()
        elif message.list_type == "dictionary":
            await self.dictionary_handler.refresh_dictionary_list()
        elif message.list_type == "worldbook":
            # Handle worldbook refresh
            pass
    
    async def on_conversation_message_loaded(self, message: ConversationMessage.Loaded) -> None:
        """Handle conversation loaded message."""
        # Update state with loaded conversation
        new_state = self.state
        new_state.selected_conversation_id = message.conversation_id
        if getattr(message, "conversation_data", None):
            new_state.selected_conversation_title = message.conversation_data.get("title", "")
        new_state.conversation_details_visible = True
        self.state = new_state
        
        await self.message_manager.load_conversation_messages(message.conversation_id)
        
        # Show conversation details section
        try:
            details_container = self.query_one("#conv-details-container")
            details_container.remove_class("hidden")
        except NoMatches:
            pass

    async def on_character_message_updated(self, message: CharacterMessage.Updated) -> None:
        """Handle character update messages."""
        new_state = self.state
        new_state.selected_character_id = str(message.character_id)
        new_state.selected_persona_id = None
        new_state.selected_character_data = message.card_data
        new_state.selected_character_name = message.card_data.get("name", "")
        new_state.character_actions_visible = True
        self.state = new_state

        try:
            card_widget = self.query_one("#ccp-character-card-view")
            if hasattr(card_widget, "load_character"):
                card_widget.load_character(message.card_data)
        except NoMatches:
            pass
        self._update_destination_selection_summary()

    async def on_character_message_created(self, message: CharacterMessage.Created) -> None:
        """Handle character creation messages."""
        new_state = self.state
        new_state.selected_character_id = str(message.character_id)
        new_state.selected_persona_id = None
        new_state.selected_character_data = message.card_data
        new_state.selected_character_name = message.name
        new_state.character_actions_visible = True
        self.state = new_state

        try:
            card_widget = self.query_one("#ccp-character-card-view")
            if hasattr(card_widget, "load_character"):
                card_widget.load_character(message.card_data)
        except NoMatches:
            pass
        self._update_destination_selection_summary()
    
    async def on_character_message_loaded(self, message: CharacterMessage.Loaded) -> None:
        """Handle character loaded message."""
        # Update state with loaded character
        new_state = self.state
        new_state.selected_character_id = message.character_id
        new_state.selected_persona_id = None
        new_state.selected_character_data = message.card_data
        new_state.selected_character_name = message.card_data.get("name", "")
        new_state.character_actions_visible = True
        self.state = new_state
        self._update_destination_selection_summary()
        
        # Show character actions
        try:
            actions_container = self.query_one("#char-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass

    async def on_persona_message_loaded(self, message: PersonaMessage.Loaded) -> None:
        """Handle persona loaded message."""
        new_state = self.state
        new_state.selected_persona_id = message.persona_id
        new_state.selected_character_id = None
        new_state.selected_character_data = message.persona_data
        new_state.selected_character_name = message.persona_data.get("name", "")
        new_state.character_actions_visible = False
        self.state = new_state

        try:
            persona_card = self.query_one("#ccp-persona-card-view")
            if hasattr(persona_card, "load_persona"):
                persona_card.load_persona(message.persona_data)
        except NoMatches:
            pass

        try:
            persona_editor = self.query_one("#ccp-persona-editor-view")
            if hasattr(persona_editor, "load_persona"):
                persona_editor.load_persona(message.persona_data)
        except NoMatches:
            pass

    async def on_continue_conversation_requested(self, message: ContinueConversationRequested) -> None:
        """Handle request to continue the loaded CCP conversation in main chat."""
        await self._launch_selected_conversation_in_chat()
    
    async def on_prompt_message_loaded(self, message: PromptMessage.Loaded) -> None:
        """Handle prompt loaded message."""
        # Update state with loaded prompt
        new_state = self.state
        new_state.selected_prompt_id = message.prompt_id
        new_state.prompt_actions_visible = True
        self.state = new_state
        
        # Show prompt actions
        try:
            actions_container = self.query_one("#prompt-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass
    
    async def on_dictionary_message_loaded(self, message: DictionaryMessage.Loaded) -> None:
        """Handle dictionary loaded message."""
        # Update state with loaded dictionary
        new_state = self.state
        new_state.selected_dictionary_id = message.dictionary_id
        new_state.dictionary_actions_visible = True
        self.state = new_state
        
        # Show dictionary actions
        try:
            actions_container = self.query_one("#dict-actions-container")
            actions_container.remove_class("hidden")
        except NoMatches:
            pass
    
    # ===== Reactive Watchers =====
    
    def watch_state(self, old_state: CCPScreenState, new_state: CCPScreenState) -> None:
        """Watch for state changes and update UI accordingly."""
        # Check for active view change
        if old_state.active_view != new_state.active_view:
            logger.debug(f"Active view changed from {old_state.active_view} to {new_state.active_view}")
            self.post_message(ViewChangeMessage.Changed(old_state.active_view, new_state.active_view))
            self._update_view_visibility(new_state.active_view)
        
        # Check for sidebar collapse change
        if old_state.sidebar_collapsed != new_state.sidebar_collapsed:
            logger.debug(f"Sidebar collapsed: {new_state.sidebar_collapsed}")
            self._update_sidebar_visibility(new_state.sidebar_collapsed)
        
        # Check for loading state changes
        if old_state.is_loading_conversation != new_state.is_loading_conversation:
            self._update_loading_indicator("conversation", new_state.is_loading_conversation)
        
        if old_state.is_loading_character != new_state.is_loading_character:
            self._update_loading_indicator("character", new_state.is_loading_character)
    
    def validate_state(self, state: CCPScreenState) -> CCPScreenState:
        """Validate state changes."""
        # Ensure active view is valid
        valid_views = [
            "conversations", "conversation_messages", "character_card", 
            "character_editor", "persona_profiles", "persona_editor",
            "prompt_editor", "dictionary_view", 
            "dictionary_editor", "lore_view", "import_export_view"
        ]
        if state.active_view not in valid_views:
            state.active_view = "conversations"
        
        return state
    
    # ===== Private Helper Methods =====

    def _update_mode_placeholder(self, view_name: str) -> None:
        """Update placeholder content for destination modes without full widgets."""
        placeholder = CCP_MODE_PLACEHOLDERS.get(view_name)
        if placeholder is None:
            return
        title, body = placeholder
        try:
            self.query_one("#ccp-mode-placeholder-title", Static).update(title)
            self.query_one("#ccp-mode-placeholder-body", Static).update(body)
        except NoMatches:
            return

    def _update_destination_mode_chrome(self, view_name: str) -> None:
        """Keep mode-strip active state and status copy synchronized."""
        active_button_id = None
        active_label = None
        for button_id, (candidate_view, label) in CCP_DESTINATION_MODE_VIEWS.items():
            if candidate_view == view_name:
                active_button_id = button_id
                active_label = label
                break

        for button in self.query(".ccp-mode-button"):
            if button.id == active_button_id:
                button.add_class("is-active")
            else:
                button.remove_class("is-active")

        if active_label is not None:
            try:
                self.query_one("#ccp-status-row", Static).update(
                    f"Mode: {active_label} | Source: local character DB | Attachments: Console / Workflows / ACP / Skills"
                )
            except NoMatches:
                return
    
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
                "#ccp-persona-card-view",
                "#ccp-persona-editor-view",
                "#ccp-prompt-editor-view",
                "#ccp-dictionary-view",
                "#ccp-dictionary-editor-view",
                "#ccp-mode-placeholder-view",
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
                "persona_profiles": "#ccp-persona-card-view",
                "persona_editor": "#ccp-persona-editor-view",
                "prompt_editor": "#ccp-prompt-editor-view",
                "dictionary_view": "#ccp-dictionary-view",
                "dictionary_editor": "#ccp-dictionary-editor-view",
                "lore_view": "#ccp-mode-placeholder-view",
                "import_export_view": "#ccp-mode-placeholder-view",
            }
            
            target_id = view_map.get(view_name)
            if target_id:
                self._update_mode_placeholder(view_name)
                target_view = self.query_one(target_id)
                target_view.remove_class("hidden")
                
                # Update state with new view
                new_state = self.state
                new_state.active_view = view_name
                self.state = new_state
                self._update_destination_mode_chrome(view_name)
                
                logger.info(f"Switched to view: {view_name}")
            else:
                logger.warning(f"Unknown view requested: {view_name}")
                
        except Exception as e:
            logger.error(f"Error switching view: {e}", exc_info=True)
    
    def _update_view_visibility(self, view_name: str) -> None:
        """Update view visibility based on active view.
        
        This is called from the state watcher to ensure UI stays in sync.
        
        Args:
            view_name: Name of the view to show
        """
        # This will be handled by the _switch_view method
        # We just need to ensure it's called when state changes
        pass
    
    def _update_sidebar_visibility(self, collapsed: bool) -> None:
        """Update sidebar visibility based on collapsed state.
        
        Args:
            collapsed: Whether the sidebar should be collapsed
        """
        try:
            sidebar = self.query_one("#ccp-sidebar")
            if collapsed:
                sidebar.add_class("collapsed")
            else:
                sidebar.remove_class("collapsed")
        except NoMatches:
            logger.warning("Sidebar not found for visibility update")
    
    def _update_loading_indicator(self, component: str, is_loading: bool) -> None:
        """Update loading indicator for a component.
        
        Args:
            component: Name of the component (conversation, character, etc.)
            is_loading: Whether the component is loading
        """
        # This will be implemented when we have proper loading indicators
        # For now, just log the state change
        logger.debug(f"Loading state for {component}: {is_loading}")
    
    # ===== State Management (Override from BaseAppScreen) =====
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the CCP screen."""
        return {
            "ccp_state": {
                "active_view": self.state.active_view,
                "selected_character_id": self.state.selected_character_id,
                "selected_persona_id": self.state.selected_persona_id,
                "selected_conversation_id": self.state.selected_conversation_id,
                "selected_prompt_id": self.state.selected_prompt_id,
                "selected_dictionary_id": self.state.selected_dictionary_id,
                "sidebar_collapsed": self.state.sidebar_collapsed,
                "conversation_search_term": self.state.conversation_search_term,
                "conversation_search_type": self.state.conversation_search_type,
                "include_character_chats": self.state.include_character_chats,
                "search_all_characters": self.state.search_all_characters,
            }
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state."""
        if "ccp_state" in state:
            ccp_state = state["ccp_state"]
            
            # Create new state instance with restored values
            new_state = CCPScreenState(
                active_view=ccp_state.get("active_view", "conversations"),
                selected_character_id=ccp_state.get("selected_character_id"),
                selected_persona_id=ccp_state.get("selected_persona_id"),
                selected_conversation_id=ccp_state.get("selected_conversation_id"),
                selected_prompt_id=ccp_state.get("selected_prompt_id"),
                selected_dictionary_id=ccp_state.get("selected_dictionary_id"),
                sidebar_collapsed=ccp_state.get("sidebar_collapsed", False),
                conversation_search_term=ccp_state.get("conversation_search_term", ""),
                conversation_search_type=ccp_state.get("conversation_search_type", "title"),
                include_character_chats=ccp_state.get("include_character_chats", True),
                search_all_characters=ccp_state.get("search_all_characters", True),
            )
            self.state = new_state
            
            # Reload selected items if needed
            if self.state.selected_conversation_id:
                logger.debug(f"Restoring conversation {self.state.selected_conversation_id}")
                # Use call_after_refresh to properly await the async method
                async def load_restored_conversation():
                    await self.conversation_handler.load_conversation(self.state.selected_conversation_id)
                self.call_after_refresh(load_restored_conversation)

    def _get_chat_tab_container(self):
        """Return the main chat tab container if the chat screen is mounted."""
        try:
            chat_window = self.app.query_one("#chat-window")
        except Exception:
            return None

        getter = getattr(chat_window, "_get_tab_container", None)
        if callable(getter):
            container = getter()
            if container is not None:
                return container

        for attr_name in ("_tab_container", "tab_container"):
            container = getattr(chat_window, attr_name, None)
            if container is not None:
                return container

        return None

    def _selected_character_handoff_body(self) -> str:
        """Build a compact Console handoff body for the selected character."""
        character_data = self.state.selected_character_data or getattr(
            self.character_handler,
            "current_character_data",
            {},
        ) or {}
        character_name = self.state.selected_character_name or character_data.get("name") or "Selected character"
        lines = [f"Character: {character_name}"]
        for label, key in (
            ("Description", "description"),
            ("Personality", "personality"),
            ("Scenario", "scenario"),
            ("First message", "first_message"),
        ):
            value = str(character_data.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def _attach_selected_character_to_console(self) -> None:
        """Stage selected CCP character context in Console via the handoff adapter."""
        character_id = self.state.selected_character_id or getattr(
            self.character_handler,
            "current_character_id",
            None,
        )
        if character_id in {None, ""}:
            self.notify("Load a character before attaching it to Console.", severity="warning")
            return

        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            self.notify("Console handoff is unavailable for Personas in this runtime.", severity="warning")
            return

        character_data = self.state.selected_character_data or getattr(
            self.character_handler,
            "current_character_data",
            {},
        ) or {}
        character_name = self.state.selected_character_name or character_data.get("name") or str(character_id)
        target_id = f"local:character:{character_id}"
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="personas",
                item_type="character-card",
                title=f"Character: {character_name}",
                body=self._selected_character_handoff_body(),
                display_summary=f"{character_name} character staged.",
                suggested_prompt=f"Use {character_name} as the active character context.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={
                    "selected_kind": "character",
                    "selected_name": str(character_name),
                    "selected_record_id": str(character_id),
                    "selected_target_id": target_id,
                    "backend": "local",
                },
            )
        )

    def _current_runtime_backend(self) -> str:
        """Resolve the active runtime backend for CCP-launched chats."""
        getter = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(getter):
            normalized = str(getter() or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        candidates = (
            getattr(getattr(self, "state", None), "runtime_backend", None),
            getattr(self, "runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
        )
        for candidate in candidates:
            if candidate in {"local", "server"}:
                return candidate
        return "local"

    async def _launch_chat_session(self, session_contract, conversation_id: Optional[str] = None) -> None:
        """Open a new or existing CCP-backed session in the main chat UI."""
        from ...Event_Handlers.Chat_Events.chat_events_tabs import display_conversation_in_chat_tab_ui_with_tabs

        tab_container = self._get_chat_tab_container()
        if tab_container is None:
            self.notify("Main chat tabs are not available right now.", severity="warning")
            return

        existing_tab_ids = set(getattr(tab_container, "sessions", {}).keys())
        tab_id = await tab_container.create_new_tab(session_data=session_contract)
        if not tab_id:
            return

        if hasattr(tab_container, "switch_to_tab_async"):
            await tab_container.switch_to_tab_async(tab_id)

        if not conversation_id or tab_id in existing_tab_ids:
            return

        sessions = getattr(tab_container, "sessions", {})
        session = sessions.get(tab_id)
        session_data = session.session_data if session is not None else session_contract
        await display_conversation_in_chat_tab_ui_with_tabs(
            self.app_instance,
            conversation_id,
            session_data=session_data,
        )

    async def _launch_character_in_chat(self, character_id: Any) -> None:
        """Open a new character-backed main chat tab from the CCP screen."""
        from ...Chat.chat_models import ChatSessionData

        raw_character_id = character_id
        if raw_character_id in {None, ""}:
            raw_character_id = self.state.selected_character_id
        if raw_character_id in {None, ""}:
            raw_character_id = getattr(self.character_handler, "current_character_id", None)
        if raw_character_id in {None, ""}:
            self.notify("Load a character before starting chat.", severity="warning")
            return

        character_data = self.state.selected_character_data or getattr(self.character_handler, "current_character_data", {}) or {}
        character_name = (
            self.state.selected_character_name
            or character_data.get("name")
            or str(raw_character_id)
        )
        try:
            normalized_character_id = int(raw_character_id)
        except (TypeError, ValueError):
            normalized_character_id = None
        canonical_id = str(raw_character_id)

        session_contract = ChatSessionData(
            tab_id="ccp-character-launch",
            title=f"Chat with {character_name}",
            conversation_id=None,
            is_ephemeral=True,
            runtime_backend=self._current_runtime_backend(),
            discovery_owner="ccp_character",
            discovery_entity_id=canonical_id,
            character_id=normalized_character_id,
            character_name=character_name,
            assistant_kind="character",
            assistant_id=canonical_id,
        )
        await self._launch_chat_session(session_contract)

    async def _launch_persona_in_chat(self, persona_id: Any) -> None:
        """Open a new persona-backed main chat tab from the CCP screen."""
        from ...Chat.chat_models import ChatSessionData

        raw_persona_id = persona_id
        if raw_persona_id in {None, ""}:
            raw_persona_id = self.state.selected_persona_id
        if raw_persona_id in {None, ""}:
            raw_persona_id = getattr(self.persona_handler, "current_persona_id", None)
        if raw_persona_id in {None, ""}:
            self.notify("Load a persona before starting chat.", severity="warning")
            return

        persona_data = getattr(self.persona_handler, "current_persona_data", {}) or {}
        persona_name = (
            self.state.selected_character_name
            or persona_data.get("name")
            or str(raw_persona_id)
        )
        canonical_id = str(raw_persona_id)

        session_contract = ChatSessionData(
            tab_id="ccp-persona-launch",
            title=f"Chat with {persona_name}",
            conversation_id=None,
            is_ephemeral=True,
            runtime_backend=self._current_runtime_backend(),
            discovery_owner="ccp_persona",
            discovery_entity_id=canonical_id,
            character_id=None,
            character_name=persona_name,
            assistant_kind="persona",
            assistant_id=canonical_id,
        )
        await self._launch_chat_session(session_contract)

    async def _launch_selected_conversation_in_chat(self) -> None:
        """Open the currently selected CCP conversation in the main chat UI."""
        from ...Chat.chat_models import ChatSessionData

        contract = self.conversation_handler.get_conversation_contract(self.state.selected_conversation_id)
        conversation_id = contract.get("id")
        if not conversation_id:
            self.notify("Load a conversation before continuing it in chat.", severity="warning")
            return

        session_contract = ChatSessionData(
            tab_id="ccp-launch",
            title=contract.get("title", ""),
            conversation_id=conversation_id,
            is_ephemeral=False,
            runtime_backend=contract.get("runtime_backend", "local"),
            discovery_owner=contract.get("discovery_owner", "general_chat"),
            discovery_entity_id=contract.get("discovery_entity_id"),
            character_id=contract.get("character_id"),
            character_name=self.state.selected_character_name or contract.get("character_name"),
            assistant_kind=contract.get("assistant_kind"),
            assistant_id=contract.get("assistant_id"),
            persona_memory_mode=contract.get("persona_memory_mode"),
            scope_type=contract.get("scope_type"),
            workspace_id=contract.get("workspace_id"),
        )
        await self._launch_chat_session(session_contract, conversation_id=conversation_id)
