"""Note selection dialog for TTS audio generation"""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Label, Static, Input, Checkbox
from textual.screen import ModalScreen
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
from loguru import logger


class NoteItem(Container):
    """A single note item in the selection list"""
    
    def __init__(self, note_id: int, title: str, content_preview: str, created_at: str) -> None:
        super().__init__()
        self.note_id = note_id
        self.title = title
        self.content_preview = content_preview
        self.created_at = created_at
        self.selected = False
    
    def compose(self) -> ComposeResult:
        """Build the note item UI"""
        with Horizontal(classes="note-item"):
            yield Checkbox(value=False, id=f"note-checkbox-{self.note_id}")
            with Vertical(classes="note-details"):
                yield Label(self.title or "Untitled Note", classes="note-title")
                yield Static(self.content_preview, classes="note-preview")
                yield Static(f"Created: {self.created_at}", classes="note-date")


class NoteSelectionDialog(ModalScreen[Optional[List[int]]]):
    """Dialog for selecting notes to convert to audio"""
    
    CSS = """
    NoteSelectionDialog {
        align: center middle;
    }
    
    #note-selection-container {
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
    
    #note-search-input {
        width: 100%;
    }
    
    #notes-list-container {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }
    
    .note-item {
        padding: 1;
        margin-bottom: 1;
        border-bottom: dashed $secondary;
    }
    
    .note-item:hover {
        background: $boost;
    }
    
    .note-details {
        width: 1fr;
        margin-left: 1;
    }
    
    .note-title {
        text-style: bold;
    }
    
    .note-preview {
        color: $text-muted;
        text-style: italic;
        max-height: 2;
        overflow: hidden;
    }
    
    .note-date {
        color: $text-disabled;
        font-size: 10;
    }
    
    .selection-info {
        height: 3;
        text-align: center;
        margin-bottom: 1;
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
    
    def __init__(self, notes: List[Dict[str, Any]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.notes = notes
        self.note_items: List[NoteItem] = []
        self.selected_count = reactive(0)
    
    def compose(self) -> ComposeResult:
        """Build the dialog UI"""
        with Container(id="note-selection-container"):
            yield Label("Select Notes for Audio Generation", classes="dialog-title")
            
            # Search input
            with Horizontal(classes="search-row"):
                yield Input(
                    placeholder="Search notes by title or content...",
                    id="note-search-input"
                )
            
            # Notes list
            with ScrollableContainer(id="notes-list-container"):
                yield Vertical(id="notes-list")
            
            # Selection info
            yield Static(
                "0 notes selected",
                id="selection-info",
                classes="selection-info"
            )
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Select All", id="select-all-btn", variant="default")
                yield Button("Clear All", id="clear-all-btn", variant="default")
                yield Button("Generate Audio", id="generate-btn", variant="primary", disabled=True)
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    def on_mount(self) -> None:
        """Initialize with notes data"""
        self.load_notes(self.notes)
    
    def load_notes(self, notes: List[Dict[str, Any]]) -> None:
        """Load notes into the list"""
        notes_list = self.query_one("#notes-list", Vertical)
        notes_list.clear()
        self.note_items.clear()
        
        for note in notes:
            # Create preview from content (first 100 chars)
            content = note.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            preview = preview.replace("\n", " ")  # Single line preview
            
            # Create note item
            item = NoteItem(
                note_id=note["note_id"],
                title=note.get("title", ""),
                content_preview=preview,
                created_at=note.get("created_at", "Unknown")
            )
            
            self.note_items.append(item)
            notes_list.mount(item)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "note-search-input":
            search_term = event.value.lower()
            self.filter_notes(search_term)
    
    def filter_notes(self, search_term: str) -> None:
        """Filter notes based on search term"""
        for item in self.note_items:
            if search_term:
                # Search in title and content preview
                visible = (
                    search_term in item.title.lower() or
                    search_term in item.content_preview.lower()
                )
                item.display = visible
            else:
                item.display = True
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes"""
        # Update selection count
        self.update_selection_count()
    
    def update_selection_count(self) -> None:
        """Update the selected notes count"""
        count = 0
        for item in self.note_items:
            checkbox = item.query_one(f"#note-checkbox-{item.note_id}", Checkbox)
            if checkbox.value:
                count += 1
        
        self.selected_count = count
        
        # Update UI
        info = self.query_one("#selection-info", Static)
        info.update(f"{count} note{'s' if count != 1 else ''} selected")
        
        # Enable/disable generate button
        self.query_one("#generate-btn", Button).disabled = count == 0
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "select-all-btn":
            self.select_all_notes()
        elif event.button.id == "clear-all-btn":
            self.clear_all_notes()
        elif event.button.id == "generate-btn":
            self.generate_audio()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)
    
    def select_all_notes(self) -> None:
        """Select all visible notes"""
        for item in self.note_items:
            if item.display:
                checkbox = item.query_one(f"#note-checkbox-{item.note_id}", Checkbox)
                checkbox.value = True
        self.update_selection_count()
    
    def clear_all_notes(self) -> None:
        """Clear all selections"""
        for item in self.note_items:
            checkbox = item.query_one(f"#note-checkbox-{item.note_id}", Checkbox)
            checkbox.value = False
        self.update_selection_count()
    
    def generate_audio(self) -> None:
        """Generate audio for selected notes"""
        selected_ids = []
        for item in self.note_items:
            checkbox = item.query_one(f"#note-checkbox-{item.note_id}", Checkbox)
            if checkbox.value:
                selected_ids.append(item.note_id)
        
        if selected_ids:
            self.dismiss(selected_ids)
        else:
            self.app.notify("No notes selected", severity="warning")