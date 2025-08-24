"""
Notes state management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class Note:
    """Represents a single note."""
    
    id: str
    title: str
    content: str
    version: int = 1
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    is_pinned: bool = False
    is_archived: bool = False
    
    def update_content(self, content: str) -> None:
        """Update note content."""
        self.content = content
        self.modified_at = datetime.now()
        self.version += 1
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the note."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the note."""
        if tag in self.tags:
            self.tags.remove(tag)


@dataclass
class NotesState:
    """Manages notes-related state."""
    
    # Current selection
    selected_note_id: Optional[str] = None
    notes: Dict[str, Note] = field(default_factory=dict)
    
    # Editor state
    unsaved_changes: bool = False
    preview_mode: bool = False
    
    # Auto-save settings
    auto_save_enabled: bool = True
    auto_save_interval: int = 30  # seconds
    last_save_time: Optional[datetime] = None
    auto_save_status: str = ""
    
    # View settings
    sort_by: str = "date_created"  # date_created, date_modified, title
    sort_ascending: bool = False
    filter_tags: List[str] = field(default_factory=list)
    search_query: str = ""
    
    # Sidebar state
    left_sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False
    
    def create_note(self, title: str, content: str = "") -> Note:
        """Create a new note."""
        note_id = f"note_{datetime.now().timestamp()}"
        note = Note(id=note_id, title=title, content=content)
        self.notes[note_id] = note
        self.selected_note_id = note_id
        return note
    
    def get_selected_note(self) -> Optional[Note]:
        """Get the currently selected note."""
        if self.selected_note_id:
            return self.notes.get(self.selected_note_id)
        return None
    
    def delete_note(self, note_id: str) -> None:
        """Delete a note."""
        if note_id in self.notes:
            del self.notes[note_id]
            if self.selected_note_id == note_id:
                self.selected_note_id = None
    
    def select_note(self, note_id: str) -> Optional[Note]:
        """Select a note for editing."""
        if note_id in self.notes:
            self.selected_note_id = note_id
            return self.notes[note_id]
        return None
    
    def mark_unsaved(self) -> None:
        """Mark current note as having unsaved changes."""
        self.unsaved_changes = True
        self.auto_save_status = "pending"
    
    def mark_saved(self) -> None:
        """Mark current note as saved."""
        self.unsaved_changes = False
        self.auto_save_status = "saved"
        self.last_save_time = datetime.now()
    
    def toggle_preview(self) -> bool:
        """Toggle preview mode."""
        self.preview_mode = not self.preview_mode
        return self.preview_mode
    
    def get_sorted_notes(self) -> List[Note]:
        """Get notes sorted according to current settings."""
        notes_list = list(self.notes.values())
        
        # Apply filters
        if self.filter_tags:
            notes_list = [
                n for n in notes_list
                if any(tag in n.tags for tag in self.filter_tags)
            ]
        
        if self.search_query:
            query = self.search_query.lower()
            notes_list = [
                n for n in notes_list
                if query in n.title.lower() or query in n.content.lower()
            ]
        
        # Sort
        if self.sort_by == "title":
            notes_list.sort(key=lambda n: n.title.lower())
        elif self.sort_by == "date_modified":
            notes_list.sort(key=lambda n: n.modified_at)
        else:  # date_created
            notes_list.sort(key=lambda n: n.created_at)
        
        if not self.sort_ascending:
            notes_list.reverse()
        
        return notes_list