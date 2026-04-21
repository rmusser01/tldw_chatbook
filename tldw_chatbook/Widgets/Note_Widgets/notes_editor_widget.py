"""Enhanced notes editor widget with built-in state management."""

from typing import Optional, Callable
from loguru import logger

from textual.widgets import TextArea
from textual.reactive import reactive
from textual.message import Message
from textual import work


class EditorContentChanged(Message):
    """Message emitted when editor content changes."""
    def __init__(self, content: str, word_count: int) -> None:
        super().__init__()
        self.content = content
        self.word_count = word_count


class NotesEditorWidget(TextArea):
    """
    Enhanced TextArea for notes editing with additional features.
    Follows Textual best practices with reactive state.
    """
    
    DEFAULT_CSS = """
    NotesEditorWidget {
        height: 100%;
        border: none;
        padding: 1;
    }
    
    NotesEditorWidget:focus {
        border: none;
    }
    
    NotesEditorWidget.preview-mode {
        opacity: 0.8;
    }
    """
    
    # Reactive attributes
    word_count: reactive[int] = reactive(0)
    is_preview_mode: reactive[bool] = reactive(False)
    has_unsaved_changes: reactive[bool] = reactive(False)
    
    # Store original content for comparison
    _original_content: str = ""
    _auto_save_callback: Optional[Callable] = None
    
    def __init__(
        self,
        text: str = "",
        *,
        auto_save_callback: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """
        Initialize the notes editor.
        
        Args:
            text: Initial text content
            auto_save_callback: Optional callback for auto-save
            **kwargs: Additional TextArea arguments
        """
        super().__init__(text, **kwargs)
        self._original_content = text
        self._auto_save_callback = auto_save_callback
        self.word_count = self._calculate_word_count(text)
    
    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("NotesEditorWidget mounted")
    
    def watch_is_preview_mode(self, is_preview: bool) -> None:
        """React to preview mode changes."""
        if is_preview:
            self.add_class("preview-mode")
            self.disabled = True
        else:
            self.remove_class("preview-mode")
            self.disabled = False
    
    def watch_text(self, text: str) -> None:
        """React to text changes."""
        # Calculate word count
        self.word_count = self._calculate_word_count(text)
        
        # Check if content has changed
        self.has_unsaved_changes = (text != self._original_content)
        
        # Post change message
        self.post_message(EditorContentChanged(text, self.word_count))
        
        # Trigger auto-save callback if provided
        if self.has_unsaved_changes and self._auto_save_callback:
            self._auto_save_callback()
    
    def _calculate_word_count(self, text: str) -> int:
        """Calculate the word count of the text."""
        if not text:
            return 0
        return len(text.split())
    
    def load_content(self, content: str, mark_as_saved: bool = True) -> None:
        """
        Load new content into the editor.
        
        Args:
            content: The content to load
            mark_as_saved: Whether to mark content as saved (no unsaved changes)
        """
        self.load_text(content)
        if mark_as_saved:
            self._original_content = content
            self.has_unsaved_changes = False
        else:
            # Content loaded but not marked as saved - has unsaved changes
            self.has_unsaved_changes = (content != self._original_content)
    
    def mark_as_saved(self) -> None:
        """Mark current content as saved."""
        self._original_content = self.text
        self.has_unsaved_changes = False
    
    def get_content(self) -> str:
        """Get the current content."""
        return self.text
    
    def toggle_preview_mode(self) -> bool:
        """Toggle preview mode on/off."""
        self.is_preview_mode = not self.is_preview_mode
        return self.is_preview_mode
    
    def clear_content(self) -> None:
        """Clear the editor content."""
        self.clear()
        self._original_content = ""
        self.has_unsaved_changes = False
    
    def insert_at_cursor(self, text: str) -> None:
        """
        Insert text at the current cursor position.
        
        Args:
            text: Text to insert
        """
        self.insert(text)
    
    def get_selection(self) -> str:
        """Get the currently selected text."""
        # This would need implementation based on TextArea's selection API
        return ""
    
    def replace_selection(self, text: str) -> None:
        """
        Replace the currently selected text.
        
        Args:
            text: Text to replace selection with
        """
        # This would need implementation based on TextArea's selection API
        pass