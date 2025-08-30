"""Sidebar widget for the CCP screen.

This widget encapsulates the entire sidebar functionality including:
- Conversations search and management
- Characters management
- Prompts management
- Dictionaries management
- World Books management

Following Textual best practices with focused, reusable components.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Input, ListView, Select, Collapsible, Label, TextArea, Checkbox
from textual.reactive import reactive
from textual import on
from textual.message import Message

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPSidebarWidget")


# ========== Messages ==========

class CCPSidebarMessage(Message):
    """Base message for sidebar events."""
    pass


class ConversationSearchRequested(CCPSidebarMessage):
    """User requested a conversation search."""
    def __init__(self, search_term: str, search_type: str = "title") -> None:
        super().__init__()
        self.search_term = search_term
        self.search_type = search_type


class ConversationLoadRequested(CCPSidebarMessage):
    """User requested to load a conversation."""
    def __init__(self, conversation_id: Optional[int] = None) -> None:
        super().__init__()
        self.conversation_id = conversation_id


class CharacterLoadRequested(CCPSidebarMessage):
    """User requested to load a character."""
    def __init__(self, character_id: Optional[int] = None) -> None:
        super().__init__()
        self.character_id = character_id


class PromptLoadRequested(CCPSidebarMessage):
    """User requested to load a prompt."""
    def __init__(self, prompt_id: Optional[int] = None) -> None:
        super().__init__()
        self.prompt_id = prompt_id


class DictionaryLoadRequested(CCPSidebarMessage):
    """User requested to load a dictionary."""
    def __init__(self, dictionary_id: Optional[int] = None) -> None:
        super().__init__()
        self.dictionary_id = dictionary_id


class ImportRequested(CCPSidebarMessage):
    """User requested to import an item."""
    def __init__(self, item_type: str) -> None:
        super().__init__()
        self.item_type = item_type  # conversation, character, prompt, dictionary, worldbook


class CreateRequested(CCPSidebarMessage):
    """User requested to create a new item."""
    def __init__(self, item_type: str) -> None:
        super().__init__()
        self.item_type = item_type  # character, prompt, dictionary, worldbook


class RefreshRequested(CCPSidebarMessage):
    """User requested to refresh a list."""
    def __init__(self, list_type: str) -> None:
        super().__init__()
        self.list_type = list_type  # character, dictionary, worldbook


# ========== Sidebar Widget ==========

class CCPSidebarWidget(VerticalScroll):
    """
    Sidebar widget for the CCP screen.
    
    This widget encapsulates all sidebar functionality and communicates
    with the parent screen via messages, following Textual best practices.
    """
    
    DEFAULT_CSS = """
    CCPSidebarWidget {
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
    
    CCPSidebarWidget.collapsed {
        display: none !important;
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
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the sidebar widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for VerticalScroll
        """
        super().__init__(id="ccp-sidebar", classes="ccp-sidebar", **kwargs)
        self.parent_screen = parent_screen
        
        # Cache references to frequently accessed widgets
        self._conv_search_input: Optional[Input] = None
        self._conv_results_list: Optional[ListView] = None
        self._character_select: Optional[Select] = None
        self._dictionary_select: Optional[Select] = None
        
        logger.debug("CCPSidebarWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the sidebar UI."""
        yield Static("CCP Navigation", classes="sidebar-title")
        
        # ===== Conversations Section =====
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
        
        # ===== Characters Section =====
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
        
        # ===== Prompts Section =====
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
        
        # ===== Dictionaries Section =====
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
        
        # ===== World Books Section =====
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
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache widget references
        self._cache_widget_references()
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPSidebarWidget mounted")
    
    def _cache_widget_references(self) -> None:
        """Cache frequently accessed widget references."""
        try:
            self._conv_search_input = self.query_one("#conv-char-search-input", Input)
            self._conv_results_list = self.query_one("#conv-char-search-results-list", ListView)
            self._character_select = self.query_one("#conv-char-character-select", Select)
            self._dictionary_select = self.query_one("#ccp-dictionary-select", Select)
        except Exception as e:
            logger.warning(f"Could not cache all widget references: {e}")
    
    # ===== Event Handlers =====
    
    @on(Input.Changed, "#conv-char-search-input")
    async def handle_conversation_title_search(self, event: Input.Changed) -> None:
        """Handle conversation title search input change."""
        self.post_message(ConversationSearchRequested(event.value, "title"))
    
    @on(Input.Changed, "#conv-char-keyword-search-input")
    async def handle_conversation_content_search(self, event: Input.Changed) -> None:
        """Handle conversation content search input change."""
        self.post_message(ConversationSearchRequested(event.value, "content"))
    
    @on(Input.Changed, "#conv-char-tags-search-input")
    async def handle_conversation_tags_search(self, event: Input.Changed) -> None:
        """Handle conversation tags search input change."""
        self.post_message(ConversationSearchRequested(event.value, "tags"))
    
    @on(Button.Pressed, "#conv-char-load-button")
    async def handle_load_conversation(self, event: Button.Pressed) -> None:
        """Handle load conversation button press."""
        event.stop()
        # Get selected conversation from list
        if self._conv_results_list and self._conv_results_list.highlighted_child:
            # Extract ID from the list item
            item_id = self._conv_results_list.highlighted_child.id
            if item_id and item_id.startswith("conv-result-"):
                conv_id = int(item_id.replace("conv-result-", ""))
                self.post_message(ConversationLoadRequested(conv_id))
        else:
            self.post_message(ConversationLoadRequested())
    
    @on(Button.Pressed, "#ccp-import-conversation-button")
    async def handle_import_conversation(self, event: Button.Pressed) -> None:
        """Handle import conversation button press."""
        event.stop()
        self.post_message(ImportRequested("conversation"))
    
    @on(Button.Pressed, "#ccp-import-character-button")
    async def handle_import_character(self, event: Button.Pressed) -> None:
        """Handle import character button press."""
        event.stop()
        self.post_message(ImportRequested("character"))
    
    @on(Button.Pressed, "#ccp-create-character-button")
    async def handle_create_character(self, event: Button.Pressed) -> None:
        """Handle create character button press."""
        event.stop()
        self.post_message(CreateRequested("character"))
    
    @on(Button.Pressed, "#ccp-right-pane-load-character-button")
    async def handle_load_character(self, event: Button.Pressed) -> None:
        """Handle load character button press."""
        event.stop()
        # Get selected character from select widget
        if self._character_select and self._character_select.value:
            try:
                char_id = int(self._character_select.value)
                self.post_message(CharacterLoadRequested(char_id))
            except (ValueError, TypeError):
                self.post_message(CharacterLoadRequested())
        else:
            self.post_message(CharacterLoadRequested())
    
    @on(Button.Pressed, "#ccp-refresh-character-list-button")
    async def handle_refresh_characters(self, event: Button.Pressed) -> None:
        """Handle refresh character list button press."""
        event.stop()
        self.post_message(RefreshRequested("character"))
    
    @on(Button.Pressed, "#ccp-prompt-load-selected-button")
    async def handle_load_prompt(self, event: Button.Pressed) -> None:
        """Handle load prompt button press."""
        event.stop()
        # Get selected prompt from list
        prompts_list = self.query_one("#ccp-prompts-listview", ListView)
        if prompts_list.highlighted_child:
            # Extract ID from the list item
            item_id = prompts_list.highlighted_child.id
            if item_id and item_id.startswith("prompt-result-"):
                prompt_id = int(item_id.replace("prompt-result-", ""))
                self.post_message(PromptLoadRequested(prompt_id))
        else:
            self.post_message(PromptLoadRequested())
    
    @on(Button.Pressed, "#ccp-load-dictionary-button")
    async def handle_load_dictionary(self, event: Button.Pressed) -> None:
        """Handle load dictionary button press."""
        event.stop()
        # Get selected dictionary from select widget
        if self._dictionary_select and self._dictionary_select.value:
            try:
                dict_id = int(self._dictionary_select.value)
                self.post_message(DictionaryLoadRequested(dict_id))
            except (ValueError, TypeError):
                self.post_message(DictionaryLoadRequested())
        else:
            self.post_message(DictionaryLoadRequested())
    
    @on(Button.Pressed, "#ccp-refresh-dictionary-list-button")
    async def handle_refresh_dictionaries(self, event: Button.Pressed) -> None:
        """Handle refresh dictionary list button press."""
        event.stop()
        self.post_message(RefreshRequested("dictionary"))
    
    # ===== Public Methods =====
    
    def update_conversation_results(self, results: List[Dict[str, Any]]) -> None:
        """Update the conversation search results list.
        
        Args:
            results: List of conversation search results
        """
        if self._conv_results_list:
            self._conv_results_list.clear()
            for conv in results:
                from textual.widgets import ListItem, Static
                title = conv.get('name', 'Untitled')
                conv_id = conv.get('conversation_id', conv.get('id'))
                list_item = ListItem(Static(title), id=f"conv-result-{conv_id}")
                self._conv_results_list.append(list_item)
    
    def update_character_list(self, characters: List[Dict[str, Any]]) -> None:
        """Update the character select options.
        
        Args:
            characters: List of available characters
        """
        if self._character_select:
            options = [(str(char.get('id')), char.get('name', 'Unnamed')) 
                      for char in characters]
            self._character_select.set_options(options)
    
    def update_dictionary_list(self, dictionaries: List[Dict[str, Any]]) -> None:
        """Update the dictionary select options.
        
        Args:
            dictionaries: List of available dictionaries
        """
        if self._dictionary_select:
            options = [(str(d.get('id')), d.get('name', 'Unnamed')) 
                      for d in dictionaries]
            self._dictionary_select.set_options(options)
    
    def show_conversation_details(self, show: bool = True) -> None:
        """Show or hide the conversation details section.
        
        Args:
            show: Whether to show the details section
        """
        try:
            details = self.query_one("#conv-details-container")
            if show:
                details.remove_class("hidden")
            else:
                details.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle conversation details: {e}")
    
    def show_character_actions(self, show: bool = True) -> None:
        """Show or hide the character actions section.
        
        Args:
            show: Whether to show the actions section
        """
        try:
            actions = self.query_one("#char-actions-container")
            if show:
                actions.remove_class("hidden")
            else:
                actions.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle character actions: {e}")
    
    def show_prompt_actions(self, show: bool = True) -> None:
        """Show or hide the prompt actions section.
        
        Args:
            show: Whether to show the actions section
        """
        try:
            actions = self.query_one("#prompt-actions-container")
            if show:
                actions.remove_class("hidden")
            else:
                actions.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle prompt actions: {e}")
    
    def show_dictionary_actions(self, show: bool = True) -> None:
        """Show or hide the dictionary actions section.
        
        Args:
            show: Whether to show the actions section
        """
        try:
            actions = self.query_one("#dict-actions-container")
            if show:
                actions.remove_class("hidden")
            else:
                actions.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle dictionary actions: {e}")