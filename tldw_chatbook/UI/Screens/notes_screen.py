"""Notes screen implementation following Textual best practices."""

from typing import TYPE_CHECKING, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import asyncio
import time

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, TextArea, Label, Input, ListView, Select
from textual.reactive import reactive, var
from textual.timer import Timer
from textual.css.query import QueryError
from textual.message import Message

from ..Navigation.base_app_screen import BaseAppScreen
from ...Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from ...Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from ...Widgets.Note_Widgets.notes_sync_widget_improved import NotesSyncWidgetImproved
from ...Widgets.emoji_picker import EmojiSelected, EmojiPickerScreen
from ...Event_Handlers.Audio_Events.dictation_integration_events import InsertDictationTextEvent
from ...DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


# ========== Custom Messages ==========

class NoteSelected(Message):
    """Message sent when a note is selected."""
    def __init__(self, note_id: int, note_data: Dict[str, Any]) -> None:
        super().__init__()
        self.note_id = note_id
        self.note_data = note_data


class NoteSaved(Message):
    """Message sent when a note is saved."""
    def __init__(self, note_id: int, success: bool) -> None:
        super().__init__()
        self.note_id = note_id
        self.success = success


class NoteDeleted(Message):
    """Message sent when a note is deleted."""
    def __init__(self, note_id: int) -> None:
        super().__init__()
        self.note_id = note_id


class AutoSaveTriggered(Message):
    """Message sent when auto-save is triggered."""
    def __init__(self, note_id: int) -> None:
        super().__init__()
        self.note_id = note_id


class SyncRequested(Message):
    """Message sent when sync is requested."""
    pass


# ========== State Management ==========

@dataclass
class NotesScreenState:
    """Encapsulates all state for the Notes screen."""
    
    # Current note
    selected_note_id: Optional[int] = None
    selected_note_version: Optional[int] = None
    selected_note_title: str = ""
    selected_note_content: str = ""
    
    # Editor state
    has_unsaved_changes: bool = False
    is_preview_mode: bool = False
    word_count: int = 0
    
    # Auto-save
    auto_save_enabled: bool = True
    auto_save_status: str = ""  # "", "saving", "saved"
    last_save_time: Optional[float] = None
    
    # Search and filter
    search_query: str = ""
    keyword_filter: str = ""
    sort_by: str = "date_created"
    sort_ascending: bool = False
    
    # UI state
    left_sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False
    
    # Notes list cache
    notes_list: List[Dict[str, Any]] = field(default_factory=list)


class NotesScreen(BaseAppScreen):
    """
    Notes management screen with complete functionality.
    Follows Textual best practices for Screen implementation.
    """
    
    DEFAULT_CSS = """
    NotesScreen {
        background: $background;
    }
    
    #notes-main-content {
        width: 100%;
        height: 100%;
    }
    
    #notes-controls-area {
        height: 3;
        align: center middle;
        overflow-x: auto;
    }
    
    .unsaved-indicator {
        color: $text-muted;
        margin: 0 1;
    }
    
    .unsaved-indicator.has-unsaved {
        color: $error;
        text-style: bold;
    }
    
    .unsaved-indicator.auto-saving {
        color: $primary;
        text-style: italic;
    }
    
    .unsaved-indicator.saved {
        color: $success;
    }
    
    .word-count {
        color: $text-muted;
        margin: 0 1;
    }
    
    #notes-preview-toggle {
        margin: 0 1;
    }
    
    .sidebar-toggle {
        min-width: 4;
    }
    """
    
    # Reactive attributes using proper Textual patterns
    state: reactive[NotesScreenState] = reactive(NotesScreenState)
    
    # Timer for auto-save (not reactive)
    _auto_save_timer: Optional[Timer] = None
    _search_timer: Optional[Timer] = None
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the Notes screen with proper state management."""
        super().__init__(app_instance, "notes", **kwargs)
        
        # Initialize state with a fresh instance
        self.state = NotesScreenState()
        
        # Get notes service from app (will be abstracted later)
        self.notes_service = getattr(app_instance, 'notes_service', None)
        self.notes_user_id = "default_user"
        
        logger.debug("NotesScreen initialized with reactive state")
    
    def compose_content(self) -> ComposeResult:
        """Compose the notes interface directly in the screen."""
        # Left sidebar
        yield NotesSidebarLeft(id="notes-sidebar-left")
        
        # Main content area
        with Container(id="notes-main-content"):
            # Text editor
            yield TextArea(
                id="notes-editor-area", 
                classes="notes-editor",
                disabled=False
            )
            
            # Control buttons
            with Horizontal(id="notes-controls-area"):
                yield Button(
                    "☰ L", 
                    id="toggle-notes-sidebar-left", 
                    classes="sidebar-toggle", 
                    tooltip="Toggle left sidebar"
                )
                yield Label(
                    "Ready", 
                    id="notes-unsaved-indicator", 
                    classes="unsaved-indicator"
                )
                yield Label(
                    "Words: 0",
                    id="notes-word-count",
                    classes="word-count"
                )
                yield Button(
                    "Save Note", 
                    id="notes-save-button", 
                    variant="primary"
                )
                yield Button(
                    "Preview", 
                    id="notes-preview-toggle", 
                    variant="default"
                )
                yield Button(
                    "Sync 🔄", 
                    id="notes-sync-button", 
                    variant="default"
                )
                yield Button(
                    "R ☰", 
                    id="toggle-notes-sidebar-right", 
                    classes="sidebar-toggle", 
                    tooltip="Toggle right sidebar"
                )
        
        # Right sidebar
        yield NotesSidebarRight(id="notes-sidebar-right")
    
    # ========== Reactive Watchers ==========
    
    def watch_state(self, old_state: NotesScreenState, new_state: NotesScreenState) -> None:
        """Watch for state changes and update UI accordingly."""
        # Update unsaved indicator
        if old_state.has_unsaved_changes != new_state.has_unsaved_changes:
            self._update_unsaved_indicator()
        
        # Update save status
        if old_state.auto_save_status != new_state.auto_save_status:
            self._update_save_status()
        
        # Update word count
        if old_state.word_count != new_state.word_count:
            self._update_word_count_display()
        
        # Handle sidebar collapses
        if old_state.left_sidebar_collapsed != new_state.left_sidebar_collapsed:
            self._toggle_left_sidebar_visibility()
        
        if old_state.right_sidebar_collapsed != new_state.right_sidebar_collapsed:
            self._toggle_right_sidebar_visibility()
    
    def validate_state(self, state: NotesScreenState) -> NotesScreenState:
        """Validate state changes."""
        # Ensure word count is non-negative
        state.word_count = max(0, state.word_count)
        
        # Validate auto-save status
        if state.auto_save_status not in ("", "saving", "saved"):
            state.auto_save_status = ""
        
        return state
    
    # ========== Button Event Handlers ==========
    
    @on(Button.Pressed, "#notes-save-button")
    async def handle_save_button(self, event: Button.Pressed) -> None:
        """Handle the main save button press."""
        event.stop()
        logger.debug("Save button pressed")
        await self._save_current_note()
        
        # Post message for other components
        self.post_message(NoteSaved(self.state.selected_note_id, True))
    
    @on(Button.Pressed, "#notes-sync-button")
    def handle_sync_button(self, event: Button.Pressed) -> None:
        """Handle the sync button press."""
        event.stop()
        logger.debug("Sync button pressed")
        
        # Post sync message
        self.post_message(SyncRequested())
        
        # Push sync screen
        self.app.push_screen(NotesSyncWidgetImproved(self.app_instance))
    
    @on(Button.Pressed, "#notes-preview-toggle")
    async def handle_preview_toggle(self, event: Button.Pressed) -> None:
        """Handle the preview toggle button."""
        event.stop()
        logger.debug("Preview toggle pressed")
        
        # Toggle preview mode in state
        new_state = self.state
        new_state.is_preview_mode = not new_state.is_preview_mode
        self.state = new_state
        
        await self._toggle_preview_mode()
    
    @on(Button.Pressed, "#toggle-notes-sidebar-left")
    def handle_left_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Handle left sidebar toggle."""
        event.stop()
        
        # Update state
        new_state = self.state
        new_state.left_sidebar_collapsed = not new_state.left_sidebar_collapsed
        self.state = new_state
        
        logger.debug(f"Left sidebar toggled: {self.state.left_sidebar_collapsed}")
    
    @on(Button.Pressed, "#toggle-notes-sidebar-right")
    def handle_right_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Handle right sidebar toggle."""
        event.stop()
        
        # Update state
        new_state = self.state
        new_state.right_sidebar_collapsed = not new_state.right_sidebar_collapsed
        self.state = new_state
        
        logger.debug(f"Right sidebar toggled: {self.state.right_sidebar_collapsed}")
    
    @on(Button.Pressed, "#notes-create-new-button")
    async def handle_create_new_button(self, event: Button.Pressed) -> None:
        """Handle creating a new note."""
        event.stop()
        await self._create_new_note()
    
    @on(Button.Pressed, "#notes-delete-button")
    async def handle_delete_button(self, event: Button.Pressed) -> None:
        """Handle deleting the current note."""
        event.stop()
        
        if self.state.selected_note_id:
            await self._delete_current_note()
            self.post_message(NoteDeleted(self.state.selected_note_id))
    
    @on(Button.Pressed, "#notes-sidebar-emoji-button")
    def handle_emoji_button(self, event: Button.Pressed) -> None:
        """Handle emoji picker button."""
        event.stop()
        self.app.push_screen(EmojiPickerScreen(), self._handle_emoji_picker_result)
    
    # ========== Text Input Event Handlers ==========
    
    @on(TextArea.Changed, "#notes-editor-area")
    async def handle_editor_changed(self, event: TextArea.Changed) -> None:
        """Handle changes to the notes editor."""
        if not self.state.selected_note_id:
            return
        
        current_content = event.text_area.text
        
        # Update state
        new_state = self.state
        new_state.has_unsaved_changes = (current_content != new_state.selected_note_content)
        new_state.word_count = len(current_content.split()) if current_content else 0
        self.state = new_state
        
        # Start auto-save timer if enabled
        if self.state.auto_save_enabled and self.state.has_unsaved_changes:
            self._start_auto_save_timer()
    
    @on(Input.Changed, "#notes-title-input")
    async def handle_title_changed(self, event: Input.Changed) -> None:
        """Handle title input changes."""
        if not self.state.selected_note_id:
            return
        
        current_title = event.input.value
        
        # Update state
        new_state = self.state
        new_state.has_unsaved_changes = (current_title != new_state.selected_note_title)
        self.state = new_state
        
        # Start auto-save timer if enabled
        if self.state.auto_save_enabled and self.state.has_unsaved_changes:
            self._start_auto_save_timer()
    
    @on(Input.Changed, "#notes-search-input")
    async def handle_search_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes with debouncing."""
        search_term = event.value.strip()
        logger.debug(f"Notes search input changed to: '{search_term}'")
        
        # Update state
        new_state = self.state
        new_state.search_query = search_term
        self.state = new_state
        
        # Cancel previous timer
        if self._search_timer is not None:
            self._search_timer.stop()
        
        # Start new debounced search
        self._search_timer = self.set_timer(0.5, lambda: self._perform_search(search_term))
    
    @on(Input.Changed, "#notes-keyword-filter-input")
    async def handle_keyword_filter_changed(self, event: Input.Changed) -> None:
        """Handle keyword filter input changes."""
        keyword_filter = event.value.strip()
        logger.debug(f"Notes keyword filter changed to: '{keyword_filter}'")
        
        # Update state
        new_state = self.state
        new_state.keyword_filter = keyword_filter
        self.state = new_state
        
        # Perform filtered search
        await self._perform_filtered_search(self.state.search_query, keyword_filter)
    
    # ========== List/Select Event Handlers ==========
    
    @on(ListView.Selected, "#notes-list-view")
    async def handle_list_selection(self, event: ListView.Selected) -> None:
        """Handle selecting a note from the list."""
        if event.item and hasattr(event.item, 'note_id'):
            await self._load_note(event.item.note_id)
            
            # Post selection message
            self.post_message(NoteSelected(
                event.item.note_id,
                {"title": self.state.selected_note_title}
            ))
    
    @on(Select.Changed, "#notes-sort-select")
    async def handle_sort_changed(self, event: Select.Changed) -> None:
        """Handle changes to the sort dropdown."""
        # Update state
        new_state = self.state
        new_state.sort_by = event.select.value
        self.state = new_state
        
        logger.debug(f"Sort by changed to: {self.state.sort_by}")
        self.run_worker(self._load_and_display_notes, thread=True)
    
    # ========== Special Event Handlers ==========
    
    def on_insert_dictation_text_event(self, event: InsertDictationTextEvent) -> None:
        """Handle dictation text insertion."""
        if event.text:
            try:
                editor = self.query_one("#notes-editor-area", TextArea)
                cursor_location = editor.cursor_location
                row, col = cursor_location
                
                # Get current text
                current_text = editor.text
                lines = current_text.split('\n') if current_text else ['']
                
                # Ensure we have enough lines
                while len(lines) <= row:
                    lines.append('')
                
                # Insert text at cursor position
                line = lines[row]
                lines[row] = line[:col] + event.text + line[col:]
                
                # Update editor
                new_text = '\n'.join(lines)
                editor.load_text(new_text)
                
                # Move cursor after inserted text
                new_col = col + len(event.text)
                editor.cursor_location = (row, new_col)
                
            except Exception as e:
                self.app.notify(f"Failed to insert voice input: {e}", severity="error")
    
    def on_emoji_picker_emoji_selected(self, message: EmojiSelected) -> None:
        """Handle emoji selection from the emoji picker."""
        try:
            notes_editor = self.query_one("#notes-editor-area", TextArea)
            notes_editor.insert(message.emoji)
            notes_editor.focus()
            message.stop()
        except Exception as e:
            logger.error(f"Failed to insert emoji: {e}")
    
    def _handle_emoji_picker_result(self, emoji_char: str) -> None:
        """Callback for when the EmojiPickerScreen is dismissed."""
        if emoji_char:
            self.post_message(EmojiSelected(emoji_char))
    
    # ========== UI Update Methods ==========
    
    def _update_unsaved_indicator(self) -> None:
        """Update the unsaved changes indicator based on state."""
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            
            if self.state.auto_save_status == "saving":
                indicator.update("⟳ Auto-saving...")
                indicator.remove_class("has-unsaved", "saved")
                indicator.add_class("auto-saving")
            elif self.state.auto_save_status == "saved":
                indicator.update("✓ Saved")
                indicator.remove_class("has-unsaved", "auto-saving")
                indicator.add_class("saved")
            elif self.state.has_unsaved_changes:
                indicator.update("● Unsaved")
                indicator.remove_class("saved", "auto-saving")
                indicator.add_class("has-unsaved")
            else:
                indicator.update("✓ Ready")
                indicator.remove_class("has-unsaved", "auto-saving", "saved")
        except QueryError:
            pass
    
    def _update_save_status(self) -> None:
        """Update save status display."""
        self._update_unsaved_indicator()
        
        # Clear saved status after 2 seconds
        if self.state.auto_save_status == "saved":
            self.set_timer(2.0, self._clear_save_status)
    
    def _clear_save_status(self) -> None:
        """Clear the save status."""
        if self.state.auto_save_status == "saved":
            new_state = self.state
            new_state.auto_save_status = ""
            self.state = new_state
    
    def _update_word_count_display(self) -> None:
        """Update the word count display."""
        try:
            word_count_label = self.query_one("#notes-word-count", Label)
            word_count_label.update(f"Words: {self.state.word_count}")
        except QueryError:
            pass
    
    def _toggle_left_sidebar_visibility(self) -> None:
        """Toggle left sidebar visibility."""
        try:
            sidebar = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            sidebar.display = not self.state.left_sidebar_collapsed
        except QueryError:
            pass
    
    def _toggle_right_sidebar_visibility(self) -> None:
        """Toggle right sidebar visibility."""
        try:
            sidebar = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar.display = not self.state.right_sidebar_collapsed
        except QueryError:
            pass
    
    # ========== Auto-save Methods ==========
    
    def _start_auto_save_timer(self) -> None:
        """Start or restart the auto-save timer."""
        # Cancel existing timer if any
        if self._auto_save_timer:
            self._auto_save_timer.stop()
        
        # Start new timer (3 seconds delay)
        self._auto_save_timer = self.set_timer(3.0, lambda: self.run_worker(self._perform_auto_save))
    
    @work(exclusive=True)
    async def _perform_auto_save(self) -> None:
        """Perform auto-save of the current note using a worker."""
        logger.debug("Performing auto-save")
        
        if not self.notes_service:
            logger.error("Notes service not available for auto-save")
            return
        
        if not self.state.selected_note_id or self.state.selected_note_version is None:
            logger.warning("No note selected or version missing for auto-save")
            return
        
        if not self.state.auto_save_enabled:
            logger.debug("Auto-save disabled, skipping")
            return
        
        # Update status
        new_state = self.state
        new_state.auto_save_status = "saving"
        self.state = new_state
        
        try:
            # Get current content from UI
            editor = self.query_one("#notes-editor-area", TextArea)
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            title_input = sidebar_right.query_one("#notes-title-input", Input)
            
            current_content = editor.text
            current_title = title_input.value.strip() or "Untitled Note"
            
            # Save note
            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.state.selected_note_id,
                update_data={'title': current_title, 'content': current_content},
                expected_version=self.state.selected_note_version
            )
            
            if success:
                # Update state
                updated_note = self.notes_service.get_note_by_id(
                    user_id=self.notes_user_id,
                    note_id=self.state.selected_note_id
                )
                if updated_note:
                    new_state = self.state
                    new_state.selected_note_version = updated_note.get('version')
                    new_state.selected_note_title = updated_note.get('title')
                    new_state.selected_note_content = updated_note.get('content')
                    new_state.has_unsaved_changes = False
                    new_state.auto_save_status = "saved"
                    new_state.last_save_time = time.time()
                    self.state = new_state
                
                logger.debug(f"Auto-saved note {self.state.selected_note_id}")
                
                # Post auto-save message
                self.post_message(AutoSaveTriggered(self.state.selected_note_id))
            else:
                logger.warning(f"Auto-save failed for note {self.state.selected_note_id}")
                new_state = self.state
                new_state.auto_save_status = ""
                self.state = new_state
                
        except Exception as e:
            logger.error(f"Error during auto-save: {e}")
            new_state = self.state
            new_state.auto_save_status = ""
            self.state = new_state
    
    # ========== Note Operations ==========
    
    async def _save_current_note(self) -> bool:
        """Save the current note."""
        if not self.state.selected_note_id or not self.notes_service:
            return False
        
        try:
            # Get current content
            editor = self.query_one("#notes-editor-area", TextArea)
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            title_input = sidebar_right.query_one("#notes-title-input", Input)
            
            current_content = editor.text
            current_title = title_input.value.strip() or "Untitled Note"
            
            # Save note
            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.state.selected_note_id,
                update_data={'title': current_title, 'content': current_content},
                expected_version=self.state.selected_note_version
            )
            
            if success:
                # Update state
                new_state = self.state
                new_state.has_unsaved_changes = False
                new_state.selected_note_version += 1
                new_state.selected_note_title = current_title
                new_state.selected_note_content = current_content
                self.state = new_state
                
                self.app.notify("Note saved!", severity="information")
                return True
            else:
                self.app.notify("Failed to save note", severity="error")
                return False
                
        except Exception as e:
            logger.error(f"Error saving note: {e}")
            self.app.notify(f"Error saving note: {e}", severity="error")
            return False
    
    async def _create_new_note(self) -> None:
        """Create a new note."""
        if self.notes_service:
            new_note_id = self.notes_service.add_note(
                user_id=self.notes_user_id,
                title="New Note",
                content=""
            )
            if new_note_id:
                await self._load_note(new_note_id)
                self.run_worker(self._load_and_display_notes, thread=True)
                self.app.notify("New note created", severity="information")
    
    async def _delete_current_note(self) -> None:
        """Delete the currently selected note."""
        if self.state.selected_note_id and self.notes_service:
            success = self.notes_service.delete_note(
                user_id=self.notes_user_id,
                note_id=self.state.selected_note_id
            )
            if success:
                # Clear state
                new_state = self.state
                new_state.selected_note_id = None
                new_state.selected_note_version = None
                new_state.selected_note_title = ""
                new_state.selected_note_content = ""
                new_state.has_unsaved_changes = False
                self.state = new_state
                
                await self._clear_editor()
                self.run_worker(self._load_and_display_notes, thread=True)
                self.app.notify("Note deleted", severity="information")
    
    async def _load_note(self, note_id: int) -> None:
        """Load a specific note into the editor."""
        if not self.notes_service:
            return
        
        # Cancel any pending auto-save
        if self._auto_save_timer:
            self._auto_save_timer.stop()
            self._auto_save_timer = None
        
        # Save current note if there are unsaved changes
        if self.state.has_unsaved_changes and self.state.auto_save_enabled:
            await self._perform_auto_save()
        
        # Load the new note
        note_details = self.notes_service.get_note_by_id(
            user_id=self.notes_user_id,
            note_id=note_id
        )
        
        if note_details:
            # Update state
            new_state = self.state
            new_state.selected_note_id = note_id
            new_state.selected_note_version = note_details.get('version')
            new_state.selected_note_title = note_details.get('title', '')
            new_state.selected_note_content = note_details.get('content', '')
            new_state.has_unsaved_changes = False
            new_state.word_count = len(new_state.selected_note_content.split()) if new_state.selected_note_content else 0
            self.state = new_state
            
            # Update UI
            editor = self.query_one("#notes-editor-area", TextArea)
            editor.load_text(self.state.selected_note_content)
            
            try:
                sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
                title_input = sidebar_right.query_one("#notes-title-input", Input)
                title_input.value = self.state.selected_note_title
            except QueryError:
                pass
    
    async def _clear_editor(self) -> None:
        """Clear the editor and related fields."""
        editor = self.query_one("#notes-editor-area", TextArea)
        editor.clear()
        
        try:
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            title_input = sidebar_right.query_one("#notes-title-input", Input)
            title_input.value = ""
        except QueryError:
            pass
    
    async def _toggle_preview_mode(self) -> None:
        """Toggle between edit and preview mode."""
        # This would render markdown preview in the future
        mode = "Preview" if self.state.is_preview_mode else "Edit"
        self.app.notify(f"{mode} mode activated", severity="information")
    
    async def _perform_search(self, search_term: str) -> None:
        """Perform a search for notes."""
        await self._perform_filtered_search(search_term, self.state.keyword_filter)
    
    async def _perform_filtered_search(self, search_term: str, keyword_filter: str) -> None:
        """Perform a filtered search for notes."""
        logger.debug(f"Searching for: '{search_term}' with keyword filter: '{keyword_filter}'")
        
        # Update notes list based on search
        self.run_worker(self._load_and_display_notes, thread=True)
    
    def _load_and_display_notes(self) -> None:
        """Load and display notes in the sidebar."""
        if not self.notes_service:
            logger.error("Notes service not available")
            return
        
        try:
            sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            
            # Get notes from service
            notes_list_data = self.notes_service.list_notes(
                user_id=self.notes_user_id,
                limit=200
            )
            
            # Apply search filter if present
            if self.state.search_query:
                query = self.state.search_query.lower()
                notes_list_data = [
                    n for n in notes_list_data
                    if query in (n.get('title', '') or '').lower() or
                       query in (n.get('content', '') or '').lower()
                ]
            
            # Apply keyword filter if present
            if self.state.keyword_filter:
                keyword = self.state.keyword_filter.lower()
                notes_list_data = [
                    n for n in notes_list_data
                    if keyword in (n.get('keywords', '') or '').lower()
                ]
            
            # Sort notes based on current settings
            if self.state.sort_by == "title":
                notes_list_data.sort(
                    key=lambda n: (n.get('title', '') or '').lower(),
                    reverse=not self.state.sort_ascending
                )
            elif self.state.sort_by == "date_modified":
                notes_list_data.sort(
                    key=lambda n: n.get('updated_at', ''),
                    reverse=not self.state.sort_ascending
                )
            else:  # date_created (default)
                notes_list_data.sort(
                    key=lambda n: n.get('created_at', ''),
                    reverse=not self.state.sort_ascending
                )
            
            # Update state cache
            new_state = self.state
            new_state.notes_list = notes_list_data
            self.state = new_state
            
            # Update the sidebar - use app.call_from_thread for async method
            self.app.call_from_thread(sidebar_left.populate_notes_list, notes_list_data)
            logger.info(f"Loaded {len(notes_list_data)} notes")
            
        except Exception as e:
            logger.error(f"Error loading notes: {e}")
    
    # ========== Lifecycle Methods ==========
    
    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        super().on_mount()  # Don't await - parent's on_mount is not async
        logger.info("NotesScreen mounted")
        
        # Load initial notes data
        if self.notes_service:
            self.run_worker(self._load_and_display_notes, thread=True)
    
    def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        # Cancel any pending timers
        if self._auto_save_timer:
            self._auto_save_timer.stop()
        
        if self._search_timer:
            self._search_timer.stop()
        
        # Perform final save if needed
        if self.state.has_unsaved_changes and self.state.auto_save_enabled:
            # Use run_worker for async save on unmount
            self.run_worker(self._perform_auto_save, exclusive=True)
        
        super().on_unmount()
        logger.info("NotesScreen unmounted")
    
    # ========== State Persistence ==========
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the notes screen."""
        state = super().save_state()
        
        # Convert dataclass to dict for serialization
        state.update({
            'notes_state': {
                'selected_note_id': self.state.selected_note_id,
                'selected_note_version': self.state.selected_note_version,
                'selected_note_title': self.state.selected_note_title,
                'has_unsaved_changes': self.state.has_unsaved_changes,
                'auto_save_enabled': self.state.auto_save_enabled,
                'sort_by': self.state.sort_by,
                'sort_ascending': self.state.sort_ascending,
                'search_query': self.state.search_query,
                'keyword_filter': self.state.keyword_filter,
                'left_sidebar_collapsed': self.state.left_sidebar_collapsed,
                'right_sidebar_collapsed': self.state.right_sidebar_collapsed,
            }
        })
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state."""
        super().restore_state(state)
        
        if 'notes_state' in state:
            notes_state = state['notes_state']
            
            # Create new state instance with restored values
            new_state = NotesScreenState(
                selected_note_id=notes_state.get('selected_note_id'),
                selected_note_version=notes_state.get('selected_note_version'),
                selected_note_title=notes_state.get('selected_note_title', ''),
                has_unsaved_changes=notes_state.get('has_unsaved_changes', False),
                auto_save_enabled=notes_state.get('auto_save_enabled', True),
                sort_by=notes_state.get('sort_by', 'date_created'),
                sort_ascending=notes_state.get('sort_ascending', False),
                search_query=notes_state.get('search_query', ''),
                keyword_filter=notes_state.get('keyword_filter', ''),
                left_sidebar_collapsed=notes_state.get('left_sidebar_collapsed', False),
                right_sidebar_collapsed=notes_state.get('right_sidebar_collapsed', False),
            )
            self.state = new_state
            
            # Reload the note content if selected_note_id is set
            if self.state.selected_note_id:
                logger.debug(f"Restoring note {self.state.selected_note_id}")
                # Use call_after_refresh with an async lambda to properly await the async method
                async def load_restored_note():
                    await self._load_note(self.state.selected_note_id)
                self.call_after_refresh(load_restored_note)