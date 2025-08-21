"""Dictionary editor widget for the CCP screen.

This widget provides a comprehensive form for editing dictionaries/world books,
following Textual best practices with focused components.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Label, Input, TextArea, Button, Switch, DataTable
from textual.reactive import reactive
from textual import on
from textual.message import Message
from textual.validation import Length

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPDictionaryEditorWidget")


# ========== Messages ==========

class DictionaryEditorMessage(Message):
    """Base message for dictionary editor events."""
    pass


class DictionarySaveRequested(DictionaryEditorMessage):
    """User requested to save the dictionary."""
    def __init__(self, dictionary_data: Dict[str, Any]) -> None:
        super().__init__()
        self.dictionary_data = dictionary_data


class DictionaryDeleteRequested(DictionaryEditorMessage):
    """User requested to delete the dictionary."""
    def __init__(self, dictionary_id: int) -> None:
        super().__init__()
        self.dictionary_id = dictionary_id


class DictionaryEntryAdded(DictionaryEditorMessage):
    """User added an entry to the dictionary."""
    def __init__(self, key: str, value: str) -> None:
        super().__init__()
        self.key = key
        self.value = value


class DictionaryEntryRemoved(DictionaryEditorMessage):
    """User removed an entry from the dictionary."""
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key


class DictionaryEntryUpdated(DictionaryEditorMessage):
    """User updated an entry in the dictionary."""
    def __init__(self, key: str, value: str) -> None:
        super().__init__()
        self.key = key
        self.value = value


class DictionaryImportRequested(DictionaryEditorMessage):
    """User requested to import dictionary data."""
    pass


class DictionaryExportRequested(DictionaryEditorMessage):
    """User requested to export dictionary data."""
    def __init__(self, format: str = "json") -> None:
        super().__init__()
        self.format = format


class DictionaryEditorCancelled(DictionaryEditorMessage):
    """User cancelled dictionary editing."""
    pass


# ========== Dictionary Editor Widget ==========

class CCPDictionaryEditorWidget(Container):
    """
    Dictionary editor widget for the CCP screen.
    
    This widget provides a comprehensive editing form for dictionaries/world books,
    including entries management, import/export, and search capabilities.
    """
    
    DEFAULT_CSS = """
    CCPDictionaryEditorWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    CCPDictionaryEditorWidget.hidden {
        display: none !important;
    }
    
    .dictionary-editor-header {
        width: 100%;
        height: 3;
        background: $primary-background-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    .dictionary-editor-content {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .dictionary-section {
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
    
    .field-input {
        width: 100%;
        margin-top: 0;
    }
    
    .field-textarea {
        width: 100%;
        height: 5;
        margin-top: 0;
        border: round $surface;
        background: $surface;
    }
    
    .entries-table {
        width: 100%;
        height: 20;
        border: round $surface;
        background: $surface;
    }
    
    .entry-editor-container {
        width: 100%;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
        margin-top: 1;
    }
    
    .entry-key-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .entry-value-textarea {
        width: 100%;
        height: 8;
        margin-bottom: 1;
        border: round $surface-lighten-1;
        background: $surface-lighten-1;
    }
    
    .entry-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
    }
    
    .entry-action-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .entry-action-button:last-child {
        margin-right: 0;
    }
    
    .entry-action-button.add {
        background: $success-darken-1;
    }
    
    .entry-action-button.add:hover {
        background: $success;
    }
    
    .entry-action-button.update {
        background: $primary;
    }
    
    .entry-action-button.update:hover {
        background: $primary-lighten-1;
    }
    
    .entry-action-button.remove {
        background: $error-darken-1;
    }
    
    .entry-action-button.remove:hover {
        background: $error;
    }
    
    .search-container {
        layout: horizontal;
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    
    .search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    .search-button {
        width: auto;
        padding: 0 1;
    }
    
    .import-export-section {
        layout: horizontal;
        width: 100%;
        height: 3;
        margin-top: 1;
    }
    
    .import-export-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .import-export-button:last-child {
        margin-right: 0;
    }
    
    .stats-container {
        width: 100%;
        padding: 1;
        background: $surface-darken-2;
        border: round $surface-darken-1;
        margin-top: 1;
    }
    
    .stats-row {
        layout: horizontal;
        width: 100%;
        height: 3;
    }
    
    .stats-label {
        width: auto;
        margin-right: 1;
        color: $text-muted;
    }
    
    .stats-value {
        width: 1fr;
        text-align: right;
        text-style: bold;
    }
    
    .dictionary-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 1;
        background: $surface;
        border-top: thick $background-darken-1;
    }
    
    .dictionary-action-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .dictionary-action-button:last-child {
        margin-right: 0;
    }
    
    .dictionary-action-button.primary {
        background: $success;
    }
    
    .dictionary-action-button.primary:hover {
        background: $success-lighten-1;
    }
    
    .dictionary-action-button.danger {
        background: $error-darken-1;
    }
    
    .dictionary-action-button.danger:hover {
        background: $error;
    }
    
    .dictionary-action-button.cancel {
        background: $warning-darken-1;
    }
    
    .dictionary-action-button.cancel:hover {
        background: $warning;
    }
    
    .no-dictionary-message {
        width: 100%;
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    
    .active-toggle-container {
        layout: horizontal;
        height: 3;
        width: 100%;
        align: left middle;
        margin-bottom: 1;
    }
    
    .toggle-label {
        width: auto;
        margin-right: 2;
    }
    
    .no-entries-placeholder {
        width: 100%;
        padding: 2;
        text-align: center;
        color: $text-muted;
    }
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    # Current dictionary data being edited
    dictionary_data: reactive[Dict[str, Any]] = reactive({})
    
    # Dictionary entries
    entries: reactive[Dict[str, str]] = reactive({})
    
    # Selected entry for editing
    selected_entry_key: reactive[Optional[str]] = reactive(None)
    
    # Search filter
    search_filter: reactive[str] = reactive("")
    
    # Is active/enabled
    is_active: reactive[bool] = reactive(True)
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the dictionary editor widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for Container
        """
        super().__init__(id="ccp-dictionary-editor-view", classes="ccp-view-area hidden", **kwargs)
        self.parent_screen = parent_screen
        
        # Field references for quick access
        self._name_input: Optional[Input] = None
        self._description_area: Optional[TextArea] = None
        self._active_toggle: Optional[Switch] = None
        self._entries_table: Optional[DataTable] = None
        self._entry_key_input: Optional[Input] = None
        self._entry_value_area: Optional[TextArea] = None
        self._search_input: Optional[Input] = None
        self._stats_entries: Optional[Static] = None
        self._stats_size: Optional[Static] = None
        
        logger.debug("CCPDictionaryEditorWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the dictionary editor UI."""
        # Header
        yield Static("Dictionary/World Book Editor", classes="dictionary-editor-header pane-title")
        
        # Content container with scroll
        with VerticalScroll(classes="dictionary-editor-content"):
            # No dictionary placeholder (shown when no dictionary is loaded)
            yield Static(
                "No dictionary loaded.\nSelect a dictionary from the sidebar or create a new one.",
                classes="no-dictionary-message",
                id="no-dictionary-placeholder"
            )
            
            # Editor container (hidden by default)
            with Container(id="dictionary-editor-container", classes="hidden"):
                # Basic Information Section
                with Container(classes="dictionary-section"):
                    yield Static("Basic Information", classes="section-title")
                    
                    # Name field
                    with Container(classes="field-container"):
                        yield Label("Dictionary Name:", classes="field-label")
                        yield Input(
                            placeholder="Enter dictionary name",
                            id="ccp-dictionary-name",
                            classes="field-input",
                            validators=[Length(1, 100)]
                        )
                    
                    # Description field
                    with Container(classes="field-container"):
                        yield Label("Description:", classes="field-label")
                        yield TextArea(
                            "",
                            id="ccp-dictionary-description",
                            classes="field-textarea"
                        )
                    
                    # Active toggle
                    with Container(classes="active-toggle-container"):
                        yield Label("Active/Enabled:", classes="toggle-label")
                        yield Switch(id="ccp-dictionary-active-toggle", value=True)
                
                # Entries Section
                with Container(classes="dictionary-section"):
                    yield Static("Dictionary Entries", classes="section-title")
                    
                    # Search
                    with Container(classes="search-container"):
                        yield Input(
                            placeholder="Search entries...",
                            id="ccp-dictionary-search",
                            classes="search-input"
                        )
                        yield Button("Clear", id="clear-search-btn", classes="search-button")
                    
                    # Entries table
                    yield DataTable(
                        id="ccp-entries-table",
                        classes="entries-table",
                        show_header=True,
                        zebra_stripes=True,
                        cursor_type="row"
                    )
                    
                    # Entry editor
                    with Container(classes="entry-editor-container"):
                        yield Label("Entry Editor", classes="field-label")
                        
                        yield Input(
                            placeholder="Entry key/term",
                            id="ccp-entry-key",
                            classes="entry-key-input"
                        )
                        
                        yield TextArea(
                            "",
                            placeholder="Entry value/definition",
                            id="ccp-entry-value",
                            classes="entry-value-textarea"
                        )
                        
                        with Container(classes="entry-actions"):
                            yield Button("Add Entry", id="add-entry-btn", classes="entry-action-button add")
                            yield Button("Update Entry", id="update-entry-btn", classes="entry-action-button update")
                            yield Button("Remove Entry", id="remove-entry-btn", classes="entry-action-button remove")
                
                # Import/Export Section
                with Container(classes="dictionary-section"):
                    yield Static("Import/Export", classes="section-title")
                    
                    with Container(classes="import-export-section"):
                        yield Button("Import JSON", id="import-json-btn", classes="import-export-button")
                        yield Button("Import CSV", id="import-csv-btn", classes="import-export-button")
                        yield Button("Export JSON", id="export-json-btn", classes="import-export-button")
                        yield Button("Export CSV", id="export-csv-btn", classes="import-export-button")
                
                # Statistics Section
                with Container(classes="dictionary-section"):
                    yield Static("Statistics", classes="section-title")
                    
                    with Container(classes="stats-container"):
                        with Container(classes="stats-row"):
                            yield Static("Total Entries:", classes="stats-label")
                            yield Static("0", id="stats-entries", classes="stats-value")
                        
                        with Container(classes="stats-row"):
                            yield Static("Dictionary Size:", classes="stats-label")
                            yield Static("0 KB", id="stats-size", classes="stats-value")
                        
                        with Container(classes="stats-row"):
                            yield Static("Last Modified:", classes="stats-label")
                            yield Static("Never", id="stats-modified", classes="stats-value")
        
        # Action buttons
        with Container(classes="dictionary-actions"):
            yield Button("Save Dictionary", classes="dictionary-action-button primary", id="save-dictionary-btn")
            yield Button("Delete", classes="dictionary-action-button danger", id="delete-dictionary-btn")
            yield Button("Reset", classes="dictionary-action-button", id="reset-dictionary-btn")
            yield Button("Cancel", classes="dictionary-action-button cancel", id="cancel-dictionary-btn")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache field references
        self._cache_field_references()
        
        # Setup entries table
        self._setup_entries_table()
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPDictionaryEditorWidget mounted")
    
    def _cache_field_references(self) -> None:
        """Cache references to frequently used fields."""
        try:
            self._name_input = self.query_one("#ccp-dictionary-name", Input)
            self._description_area = self.query_one("#ccp-dictionary-description", TextArea)
            self._active_toggle = self.query_one("#ccp-dictionary-active-toggle", Switch)
            self._entries_table = self.query_one("#ccp-entries-table", DataTable)
            self._entry_key_input = self.query_one("#ccp-entry-key", Input)
            self._entry_value_area = self.query_one("#ccp-entry-value", TextArea)
            self._search_input = self.query_one("#ccp-dictionary-search", Input)
            self._stats_entries = self.query_one("#stats-entries", Static)
            self._stats_size = self.query_one("#stats-size", Static)
        except Exception as e:
            logger.warning(f"Could not cache all field references: {e}")
    
    def _setup_entries_table(self) -> None:
        """Setup the entries data table."""
        if self._entries_table:
            self._entries_table.add_column("Key", width=20)
            self._entries_table.add_column("Value", width=50)
    
    # ===== Public Methods =====
    
    def load_dictionary(self, dictionary_data: Dict[str, Any]) -> None:
        """Load dictionary data into the editor.
        
        Args:
            dictionary_data: Dictionary containing dictionary/world book information
        """
        self.dictionary_data = dictionary_data.copy()
        self.entries = dictionary_data.get('entries', {}).copy()
        
        # Hide placeholder, show editor
        try:
            placeholder = self.query_one("#no-dictionary-placeholder")
            placeholder.add_class("hidden")
            
            editor = self.query_one("#dictionary-editor-container")
            editor.remove_class("hidden")
        except:
            pass
        
        # Load basic fields
        if self._name_input:
            self._name_input.value = dictionary_data.get('name', '')
        if self._description_area:
            self._description_area.text = dictionary_data.get('description', '')
        if self._active_toggle:
            self._active_toggle.value = dictionary_data.get('active', True)
        
        # Load entries
        self._update_entries_table()
        
        # Update statistics
        self._update_statistics()
        
        logger.info(f"Loaded dictionary for editing: {dictionary_data.get('name', 'Unknown')}")
    
    def new_dictionary(self) -> None:
        """Initialize the editor for a new dictionary."""
        self.dictionary_data = {}
        self.entries = {}
        self.selected_entry_key = None
        self.search_filter = ""
        self.is_active = True
        
        # Hide placeholder, show editor
        try:
            placeholder = self.query_one("#no-dictionary-placeholder")
            placeholder.add_class("hidden")
            
            editor = self.query_one("#dictionary-editor-container")
            editor.remove_class("hidden")
        except:
            pass
        
        # Clear all fields
        if self._name_input:
            self._name_input.value = ""
        if self._description_area:
            self._description_area.text = ""
        if self._active_toggle:
            self._active_toggle.value = True
        if self._entry_key_input:
            self._entry_key_input.value = ""
        if self._entry_value_area:
            self._entry_value_area.text = ""
        if self._search_input:
            self._search_input.value = ""
        
        # Clear table
        if self._entries_table:
            self._entries_table.clear()
        
        # Update statistics
        self._update_statistics()
        
        logger.info("Initialized editor for new dictionary")
    
    def get_dictionary_data(self) -> Dict[str, Any]:
        """Get the current dictionary data from the editor.
        
        Returns:
            Dictionary containing all dictionary data
        """
        data = self.dictionary_data.copy()
        
        # Update with current field values
        if self._name_input:
            data['name'] = self._name_input.value
        if self._description_area:
            data['description'] = self._description_area.text
        if self._active_toggle:
            data['active'] = self._active_toggle.value
        
        # Add entries
        data['entries'] = self.entries.copy()
        
        return data
    
    # ===== Private Helper Methods =====
    
    def _update_entries_table(self, filter_text: str = "") -> None:
        """Update the entries table display."""
        if not self._entries_table:
            return
        
        # Clear existing rows
        self._entries_table.clear()
        
        # Filter entries if needed
        entries_to_show = self.entries
        if filter_text:
            filter_lower = filter_text.lower()
            entries_to_show = {
                k: v for k, v in self.entries.items()
                if filter_lower in k.lower() or filter_lower in v.lower()
            }
        
        # Add rows
        if entries_to_show:
            for key, value in sorted(entries_to_show.items()):
                # Truncate value for display
                display_value = value[:100] + "..." if len(value) > 100 else value
                self._entries_table.add_row(key, display_value)
        else:
            # Show placeholder if no entries
            if not self.entries:
                self._entries_table.add_row("No entries", "Add entries using the editor below")
            else:
                self._entries_table.add_row("No matches", "Try a different search term")
    
    def _update_statistics(self) -> None:
        """Update the statistics display."""
        if self._stats_entries:
            self._stats_entries.update(str(len(self.entries)))
        
        if self._stats_size:
            # Calculate approximate size
            size_bytes = sum(len(k) + len(v) for k, v in self.entries.items())
            size_kb = size_bytes / 1024
            self._stats_size.update(f"{size_kb:.2f} KB")
        
        # Update last modified
        try:
            stats_modified = self.query_one("#stats-modified", Static)
            if self.dictionary_data.get('last_modified'):
                stats_modified.update(self.dictionary_data['last_modified'])
            else:
                stats_modified.update("Never")
        except:
            pass
    
    def _load_entry_for_editing(self, key: str) -> None:
        """Load an entry into the editor fields."""
        if key in self.entries:
            if self._entry_key_input:
                self._entry_key_input.value = key
            if self._entry_value_area:
                self._entry_value_area.text = self.entries[key]
            self.selected_entry_key = key
    
    # ===== Event Handlers =====
    
    @on(DataTable.RowSelected, "#ccp-entries-table")
    async def handle_entry_selected(self, event: DataTable.RowSelected) -> None:
        """Handle entry selection in the table."""
        if event.row_key and self._entries_table:
            # Get the key from the first column
            row_data = self._entries_table.get_row(event.row_key.value)
            if row_data and len(row_data) > 0:
                key = str(row_data[0])
                if key != "No entries" and key != "No matches":
                    self._load_entry_for_editing(key)
    
    @on(Button.Pressed, "#add-entry-btn")
    async def handle_add_entry(self, event: Button.Pressed) -> None:
        """Handle add entry button press."""
        event.stop()
        
        if self._entry_key_input and self._entry_value_area:
            key = self._entry_key_input.value.strip()
            value = self._entry_value_area.text.strip()
            
            if key and value:
                self.entries[key] = value
                self._entry_key_input.value = ""
                self._entry_value_area.text = ""
                self._update_entries_table(self.search_filter)
                self._update_statistics()
                self.post_message(DictionaryEntryAdded(key, value))
    
    @on(Button.Pressed, "#update-entry-btn")
    async def handle_update_entry(self, event: Button.Pressed) -> None:
        """Handle update entry button press."""
        event.stop()
        
        if self._entry_key_input and self._entry_value_area:
            key = self._entry_key_input.value.strip()
            value = self._entry_value_area.text.strip()
            
            if key and value and key in self.entries:
                self.entries[key] = value
                self._update_entries_table(self.search_filter)
                self._update_statistics()
                self.post_message(DictionaryEntryUpdated(key, value))
    
    @on(Button.Pressed, "#remove-entry-btn")
    async def handle_remove_entry(self, event: Button.Pressed) -> None:
        """Handle remove entry button press."""
        event.stop()
        
        if self._entry_key_input:
            key = self._entry_key_input.value.strip()
            
            if key and key in self.entries:
                del self.entries[key]
                self._entry_key_input.value = ""
                self._entry_value_area.text = ""
                self._update_entries_table(self.search_filter)
                self._update_statistics()
                self.post_message(DictionaryEntryRemoved(key))
    
    @on(Input.Changed, "#ccp-dictionary-search")
    async def handle_search_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        self.search_filter = event.value
        self._update_entries_table(event.value)
    
    @on(Button.Pressed, "#clear-search-btn")
    async def handle_clear_search(self, event: Button.Pressed) -> None:
        """Handle clear search button press."""
        event.stop()
        if self._search_input:
            self._search_input.value = ""
        self.search_filter = ""
        self._update_entries_table()
    
    @on(Button.Pressed, "#save-dictionary-btn")
    async def handle_save_dictionary(self, event: Button.Pressed) -> None:
        """Handle save dictionary button press."""
        event.stop()
        dictionary_data = self.get_dictionary_data()
        
        # Validate required fields
        if not dictionary_data.get('name'):
            logger.warning("Cannot save dictionary without name")
            return
        
        self.post_message(DictionarySaveRequested(dictionary_data))
    
    @on(Button.Pressed, "#delete-dictionary-btn")
    async def handle_delete_dictionary(self, event: Button.Pressed) -> None:
        """Handle delete dictionary button press."""
        event.stop()
        if self.dictionary_data and 'id' in self.dictionary_data:
            self.post_message(DictionaryDeleteRequested(self.dictionary_data['id']))
    
    @on(Button.Pressed, "#reset-dictionary-btn")
    async def handle_reset_dictionary(self, event: Button.Pressed) -> None:
        """Handle reset dictionary button press."""
        event.stop()
        if self.dictionary_data:
            self.load_dictionary(self.dictionary_data)
        else:
            self.new_dictionary()
    
    @on(Button.Pressed, "#cancel-dictionary-btn")
    async def handle_cancel_edit(self, event: Button.Pressed) -> None:
        """Handle cancel edit button press."""
        event.stop()
        self.post_message(DictionaryEditorCancelled())
    
    # Import/Export handlers
    @on(Button.Pressed, "#import-json-btn")
    async def handle_import_json(self, event: Button.Pressed) -> None:
        """Handle import JSON button press."""
        event.stop()
        self.post_message(DictionaryImportRequested())
    
    @on(Button.Pressed, "#import-csv-btn")
    async def handle_import_csv(self, event: Button.Pressed) -> None:
        """Handle import CSV button press."""
        event.stop()
        self.post_message(DictionaryImportRequested())
    
    @on(Button.Pressed, "#export-json-btn")
    async def handle_export_json(self, event: Button.Pressed) -> None:
        """Handle export JSON button press."""
        event.stop()
        self.post_message(DictionaryExportRequested("json"))
    
    @on(Button.Pressed, "#export-csv-btn")
    async def handle_export_csv(self, event: Button.Pressed) -> None:
        """Handle export CSV button press."""
        event.stop()
        self.post_message(DictionaryExportRequested("csv"))
    
    @on(Switch.Changed, "#ccp-dictionary-active-toggle")
    async def handle_active_toggle(self, event: Switch.Changed) -> None:
        """Handle active toggle changes."""
        self.is_active = event.value