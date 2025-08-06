# chapter_editor_widget.py
# Description: Enhanced chapter editor widget for audiobook generation
#
# Imports
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import re
from datetime import timedelta
#
# Third-party imports  
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal, Vertical, Container
from textual.widget import Widget
from textual.widgets import (
    DataTable, Button, Label, TextArea, 
    Input, Switch, Select, Static, Rule, Collapsible
)
from textual.reactive import reactive
from textual.message import Message
from textual.coordinate import Coordinate
from rich.text import Text
from rich.console import RenderableType
#
# Local imports
from tldw_chatbook.TTS.audiobook_generator import Chapter
#
#######################################################################################################################
#
# Data Models

@dataclass
class ChapterEditEvent(Message):
    """Event emitted when a chapter is edited"""
    chapter_index: int
    chapter: Chapter
    action: str  # 'edit', 'split', 'merge', 'delete', 'reorder'

@dataclass 
class ChapterPreviewEvent(Message):
    """Event emitted when chapter preview is requested"""
    chapter: Chapter
    preview_type: str  # 'text', 'audio'

#######################################################################################################################
#
# Chapter Editor Widget

class ChapterEditorWidget(Widget):
    """Enhanced chapter editor with visual editing capabilities"""
    
    DEFAULT_CSS = """
    ChapterEditorWidget {
        height: 100%;
        width: 100%;
    }
    
    .chapter-editor-container {
        height: 100%;
        layout: horizontal;
    }
    
    .chapter-list-section {
        width: 50%;
        padding: 1;
        border-right: solid $primary;
    }
    
    .chapter-preview-section {
        width: 50%;
        padding: 1;
    }
    
    .chapter-table {
        height: 80%;
        margin: 1 0;
    }
    
    .chapter-controls {
        height: auto;
        layout: horizontal;
        margin: 1 0;
    }
    
    .chapter-controls Button {
        margin: 0 1;
    }
    
    .chapter-preview {
        height: 70%;
        border: solid $secondary;
        padding: 1;
    }
    
    .chapter-metadata {
        height: 25%;
        padding: 1;
        background: $surface;
        border: solid $secondary;
        margin-top: 1;
    }
    
    .chapter-title-input {
        width: 100%;
        margin: 1 0;
    }
    
    .duration-estimate {
        color: $text-muted;
        text-align: right;
    }
    
    .narrator-notes-area {
        height: 3;
    }
    """
    
    # Reactive properties
    chapters = reactive([], recompose=True)
    selected_chapter_index = reactive(-1)
    preview_content = reactive("")
    
    def __init__(self, chapters: Optional[List[Chapter]] = None, **kwargs):
        super().__init__(**kwargs)
        self.chapters = chapters if chapters else []
        self.dragging_row = None
        self.drop_target_row = None
    
    def compose(self) -> ComposeResult:
        """Compose the chapter editor UI"""
        with Container(classes="chapter-editor-container"):
            # Left side - Chapter list
            with Vertical(classes="chapter-list-section"):
                yield Label("📖 Chapter Editor", classes="section-title")
                
                # Chapter controls
                with Horizontal(classes="chapter-controls"):
                    yield Button("➕ Add", id="add-chapter-btn", variant="default")
                    yield Button("✂️ Split", id="split-chapter-btn", variant="default")
                    yield Button("🔗 Merge", id="merge-chapter-btn", variant="default")
                    yield Button("🗑️ Delete", id="delete-chapter-btn", variant="warning")
                
                # Chapter table
                chapter_table = DataTable(
                    id="chapter-table",
                    classes="chapter-table",
                    show_cursor=True,
                    cursor_type="row",
                    zebra_stripes=True
                )
                chapter_table.add_columns(
                    "📑", "Chapter", "Title", "Words", "Est. Duration"
                )
                yield chapter_table
                
                # Auto-detect controls
                with Horizontal(classes="form-row"):
                    yield Label("Chapter Pattern:")
                    yield Input(
                        id="chapter-pattern-input",
                        value="Chapter \\d+",
                        placeholder="Regex pattern"
                    )
                    yield Button("🔍 Detect", id="detect-chapters-btn", variant="primary")
            
            # Right side - Chapter preview and metadata
            with Vertical(classes="chapter-preview-section"):
                yield Label("👁️ Chapter Preview", classes="section-title")
                
                # Chapter metadata editor
                with Collapsible(title="Chapter Details", collapsed=False):
                    with Vertical(classes="chapter-metadata"):
                        yield Label("Title:")
                        yield Input(
                            id="chapter-title-input",
                            classes="chapter-title-input",
                            placeholder="Enter chapter title..."
                        )
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Voice Override:")
                            yield Select(
                                options=[
                                    ("narrator", "Use Narrator Voice"),
                                    ("custom", "Custom Voice"),
                                ],
                                id="chapter-voice-select"
                            )
                        
                        yield Label("Narrator Notes:")
                        yield TextArea(
                            id="narrator-notes",
                            classes="narrator-notes-area"
                        )
                
                # Preview area
                yield Label("Content:")
                yield TextArea(
                    id="chapter-preview",
                    classes="chapter-preview",
                    read_only=True,
                    show_line_numbers=True
                )
                
                # Preview controls
                with Horizontal(classes="form-row"):
                    yield Button("🔊 Preview Audio", id="preview-audio-btn", variant="success")
                    yield Label("", id="duration-estimate", classes="duration-estimate")
    
    def on_mount(self) -> None:
        """Initialize the chapter table when mounted"""
        self._refresh_chapter_table()
    
    def watch_chapters(self) -> None:
        """React to chapter list changes"""
        # Only refresh if the widget is mounted and composed
        if self.is_mounted:
            self._refresh_chapter_table()
    
    def watch_selected_chapter_index(self) -> None:
        """React to chapter selection changes"""
        # Only update if the widget is mounted and composed
        if self.is_mounted:
            self._update_preview()
    
    def _refresh_chapter_table(self) -> None:
        """Refresh the chapter table with current chapters"""
        try:
            table = self.query_one("#chapter-table", DataTable)
            table.clear()
        except Exception as e:
            # Widget not ready yet
            logger.debug(f"Chapter table not ready: {e}")
            return
        
        for i, chapter in enumerate(self.chapters):
            # Calculate word count and estimated duration
            word_count = len(chapter.content.split())
            # Rough estimate: 150 words per minute for narration
            duration_minutes = word_count / 150
            duration_str = self._format_duration(duration_minutes * 60)
            
            # Add row with drag handle
            table.add_row(
                "≡",  # Drag handle
                f"{chapter.number}",
                chapter.title,
                f"{word_count:,}",
                duration_str,
                key=str(i)
            )
    
    def _update_preview(self) -> None:
        """Update the preview pane with selected chapter"""
        if 0 <= self.selected_chapter_index < len(self.chapters):
            chapter = self.chapters[self.selected_chapter_index]
            
            try:
                # Update preview text
                preview = self.query_one("#chapter-preview", TextArea)
                preview.text = chapter.content
            except Exception as e:
                logger.debug(f"Preview area not ready: {e}")
                return
            
            try:
                # Update metadata fields
                title_input = self.query_one("#chapter-title-input", Input)
                title_input.value = chapter.title
                
                if chapter.narrator_notes:
                    notes_area = self.query_one("#narrator-notes", TextArea)
                    notes_area.text = chapter.narrator_notes
                
                # Update duration estimate
                word_count = len(chapter.content.split())
                duration_minutes = word_count / 150
                duration_label = self.query_one("#duration-estimate", Label)
                duration_label.update(f"Est. duration: {self._format_duration(duration_minutes * 60)}")
            except Exception as e:
                logger.debug(f"Some UI elements not ready: {e}")
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle chapter selection"""
        if event.row_key is not None:
            self.selected_chapter_index = int(event.row_key.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "detect-chapters-btn":
            self._detect_chapters()
        elif button_id == "add-chapter-btn":
            self._add_chapter()
        elif button_id == "split-chapter-btn":
            self._split_chapter()
        elif button_id == "merge-chapter-btn":
            self._merge_chapters()
        elif button_id == "delete-chapter-btn":
            self._delete_chapter()
        elif button_id == "preview-audio-btn":
            self._preview_audio()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        if event.input.id == "chapter-title-input" and 0 <= self.selected_chapter_index < len(self.chapters):
            # Update chapter title
            self.chapters[self.selected_chapter_index].title = event.value
            self._refresh_chapter_table()
    
    def _detect_chapters(self) -> None:
        """Run chapter detection with custom pattern"""
        pattern_input = self.query_one("#chapter-pattern-input", Input)
        pattern = pattern_input.value
        
        if not pattern:
            self.app.notify("Please enter a chapter pattern", severity="warning")
            return
        
        # This would integrate with ChapterDetector from audiobook_generator.py
        self.app.notify(f"Detecting chapters with pattern: {pattern}", severity="information")
        # Implementation would go here
    
    def _add_chapter(self) -> None:
        """Add a new chapter at current position"""
        if self.selected_chapter_index >= 0:
            # Insert after selected chapter
            insert_pos = self.selected_chapter_index + 1
        else:
            # Add at end
            insert_pos = len(self.chapters)
        
        new_chapter = Chapter(
            number=insert_pos + 1,
            title=f"New Chapter {insert_pos + 1}",
            content="",
            start_position=0,
            end_position=0
        )
        
        self.chapters.insert(insert_pos, new_chapter)
        self._renumber_chapters()
        self.post_message(ChapterEditEvent(insert_pos, new_chapter, "add"))
    
    def _split_chapter(self) -> None:
        """Split the selected chapter at cursor position"""
        if not (0 <= self.selected_chapter_index < len(self.chapters)):
            self.app.notify("Please select a chapter to split", severity="warning")
            return
        
        chapter = self.chapters[self.selected_chapter_index]
        preview = self.query_one("#chapter-preview", TextArea)
        
        # Get cursor position
        cursor_pos = preview.cursor_location[0]
        lines = chapter.content.split('\n')
        
        if cursor_pos < len(lines):
            # Split at cursor line
            first_content = '\n'.join(lines[:cursor_pos])
            second_content = '\n'.join(lines[cursor_pos:])
            
            # Create two chapters
            chapter.content = first_content
            new_chapter = Chapter(
                number=chapter.number + 1,
                title=f"{chapter.title} (Part 2)",
                content=second_content,
                start_position=chapter.start_position + cursor_pos,
                end_position=chapter.end_position
            )
            
            self.chapters.insert(self.selected_chapter_index + 1, new_chapter)
            self._renumber_chapters()
            self.post_message(ChapterEditEvent(self.selected_chapter_index, chapter, "split"))
    
    def _merge_chapters(self) -> None:
        """Merge selected chapter with next chapter"""
        if not (0 <= self.selected_chapter_index < len(self.chapters) - 1):
            self.app.notify("Select a chapter to merge with the next one", severity="warning")
            return
        
        current = self.chapters[self.selected_chapter_index]
        next_chapter = self.chapters[self.selected_chapter_index + 1]
        
        # Merge content
        current.content = f"{current.content}\n\n{next_chapter.content}"
        current.end_position = next_chapter.end_position
        
        # Remove next chapter
        self.chapters.pop(self.selected_chapter_index + 1)
        self._renumber_chapters()
        self.post_message(ChapterEditEvent(self.selected_chapter_index, current, "merge"))
    
    def _delete_chapter(self) -> None:
        """Delete the selected chapter"""
        if not (0 <= self.selected_chapter_index < len(self.chapters)):
            self.app.notify("Please select a chapter to delete", severity="warning")
            return
        
        if len(self.chapters) <= 1:
            self.app.notify("Cannot delete the last chapter", severity="warning")
            return
        
        deleted = self.chapters.pop(self.selected_chapter_index)
        self._renumber_chapters()
        self.selected_chapter_index = min(self.selected_chapter_index, len(self.chapters) - 1)
        self.post_message(ChapterEditEvent(self.selected_chapter_index, deleted, "delete"))
    
    def _preview_audio(self) -> None:
        """Request audio preview for selected chapter"""
        if not (0 <= self.selected_chapter_index < len(self.chapters)):
            self.app.notify("Please select a chapter to preview", severity="warning")
            return
        
        chapter = self.chapters[self.selected_chapter_index]
        self.post_message(ChapterPreviewEvent(chapter, "audio"))
    
    def _renumber_chapters(self) -> None:
        """Renumber all chapters sequentially"""
        for i, chapter in enumerate(self.chapters):
            chapter.number = i + 1
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string"""
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_chapters(self) -> List[Chapter]:
        """Get the current chapter list"""
        return self.chapters
    
    def set_chapters(self, chapters: List[Chapter]) -> None:
        """Set the chapter list"""
        self.chapters = chapters
        self.selected_chapter_index = 0 if chapters else -1