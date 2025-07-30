# transcription_history_viewer.py
"""
Widget for viewing and managing transcription history with search and export capabilities.
"""

from typing import Optional, List
from datetime import datetime
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Button, Label, Input, Select, Static, DataTable,
    TextArea, Rule, Collapsible, LoadingIndicator
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from loguru import logger

from ..Audio.transcription_history import (
    TranscriptionHistory, TranscriptionEntry, get_transcription_history
)
from ..Widgets.enhanced_file_picker import EnhancedFileSave as FileSave
from ..Third_Party.textual_fspicker import Filters


class TranscriptionHistoryViewer(Widget):
    """
    Widget for viewing and managing transcription history.
    
    Features:
    - Search and filter transcriptions
    - View full transcript details
    - Export selected entries
    - Delete entries
    - Encryption status indicator
    """
    
    DEFAULT_CSS = """
    TranscriptionHistoryViewer {
        height: 100%;
        layout: vertical;
    }
    
    .history-header {
        height: auto;
        padding: 1;
        border-bottom: solid $surface;
    }
    
    .search-bar {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    .search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    .filter-controls {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    .filter-controls Select {
        width: 20;
        margin-right: 1;
    }
    
    .history-content {
        height: 1fr;
        layout: horizontal;
    }
    
    .history-list {
        width: 2fr;
        border-right: solid $surface;
        padding: 1;
    }
    
    .history-table {
        height: 1fr;
    }
    
    .transcript-detail {
        width: 1fr;
        padding: 1;
    }
    
    .detail-header {
        height: auto;
        margin-bottom: 1;
    }
    
    .detail-meta {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .transcript-text {
        height: 1fr;
        border: round $surface;
    }
    
    .action-buttons {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    
    .action-buttons Button {
        margin-right: 1;
    }
    
    .encryption-status {
        padding: 0.5 1;
        margin-bottom: 1;
        border-radius: 3;
    }
    
    .encryption-status.encrypted {
        background: $success-darken-3;
        color: $success;
    }
    
    .encryption-status.unencrypted {
        background: $warning-darken-3;
        color: $warning;
    }
    
    .stats-display {
        color: $text-muted;
        margin-bottom: 1;
    }
    """
    
    # Reactive state
    search_text = reactive("")
    selected_entry_id = reactive(None)
    is_loading = reactive(False)
    
    def __init__(self):
        """Initialize history viewer."""
        super().__init__()
        self.history = get_transcription_history()
        self.current_entries: List[TranscriptionEntry] = []
        self.selected_entry: Optional[TranscriptionEntry] = None
    
    def compose(self) -> ComposeResult:
        """Compose the history viewer UI."""
        with Vertical(classes="history-container"):
            # Header
            with Container(classes="history-header"):
                yield Label("📜 Transcription History", classes="section-title")
                
                # Encryption status
                with Container(id="encryption-container"):
                    # Will be populated on mount
                    pass
                
                # Search bar
                with Horizontal(classes="search-bar"):
                    yield Input(
                        placeholder="Search transcriptions...",
                        id="search-input",
                        classes="search-input"
                    )
                    yield Button("🔍 Search", id="search-btn")
                    yield Button("🔄 Refresh", id="refresh-btn")
                
                # Filter controls
                with Horizontal(classes="filter-controls"):
                    yield Select(
                        options=[
                            ("All Languages", "all"),  # Changed empty string to "all"
                            ("English", "en"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                        ],
                        id="language-filter",
                        value="all"
                    )
                    
                    yield Select(
                        options=[
                            ("All Time", "all"),
                            ("Today", "today"),
                            ("This Week", "week"),
                            ("This Month", "month"),
                        ],
                        id="time-filter",
                        value="all"
                    )
                    
                    yield Button("🗑️ Clear All", id="clear-all-btn", variant="error")
                
                # Stats
                yield Static(
                    "0 entries",
                    id="stats-display",
                    classes="stats-display"
                )
            
            # Main content
            with Horizontal(classes="history-content"):
                # History list
                with Container(classes="history-list"):
                    yield DataTable(
                        id="history-table",
                        classes="history-table",
                        cursor_type="row"
                    )
                
                # Transcript detail
                with Container(classes="transcript-detail"):
                    yield Label("Select an entry to view details", id="detail-title")
                    yield Static("", id="detail-meta", classes="detail-meta")
                    yield TextArea(
                        "",
                        id="transcript-text",
                        classes="transcript-text",
                        read_only=True
                    )
                    
                    # Action buttons
                    with Horizontal(classes="action-buttons"):
                        yield Button("📋 Copy", id="copy-btn", disabled=True)
                        yield Button("💾 Export", id="export-btn", disabled=True)
                        yield Button("🗑️ Delete", id="delete-btn", disabled=True, variant="error")
    
    def on_mount(self):
        """Initialize on mount."""
        self._update_encryption_status()
        self._setup_table()
        self.load_history()
    
    def _update_encryption_status(self):
        """Update encryption status indicator."""
        container = self.query_one("#encryption-container", Container)
        container.remove_children()
        
        if self.history.is_encrypted():
            status = Static(
                "🔒 History is encrypted",
                classes="encryption-status encrypted"
            )
        else:
            status = Static(
                "⚠️ History is not encrypted",
                classes="encryption-status unencrypted"
            )
        
        container.mount(status)
    
    def _setup_table(self):
        """Set up the history table columns."""
        table = self.query_one("#history-table", DataTable)
        
        table.add_column("Date", width=12)
        table.add_column("Time", width=8)
        table.add_column("Duration", width=10)
        table.add_column("Words", width=8)
        table.add_column("Preview", width=40)
    
    @work(exclusive=True)
    async def load_history(self, password: Optional[str] = None):
        """Load history entries."""
        self.is_loading = True
        
        try:
            # Load from file
            success = await self.run_worker(
                self.history.load,
                password=password
            ).wait()
            
            if not success and self.history.is_encrypted():
                # Need password
                self.app.notify(
                    "History is encrypted. Please enter password.",
                    severity="warning"
                )
                # TODO: Show password dialog
                return
            
            # Get filtered entries
            await self._refresh_entries()
            
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.app.notify(
                f"Failed to load history: {str(e)}",
                severity="error"
            )
        finally:
            self.is_loading = False
    
    async def _refresh_entries(self):
        """Refresh the entry list based on filters."""
        # Get filter values
        search = self.search_text
        language = self.query_one("#language-filter", Select).value
        time_filter = self.query_one("#time-filter", Select).value
        
        # Calculate date range
        start_date = None
        if time_filter == "today":
            start_date = datetime.now().replace(hour=0, minute=0, second=0)
        elif time_filter == "week":
            start_date = datetime.now().replace(hour=0, minute=0, second=0)
            start_date = start_date.replace(day=start_date.day - 7)
        elif time_filter == "month":
            start_date = datetime.now().replace(hour=0, minute=0, second=0, day=1)
        
        # Get entries
        self.current_entries = await self.run_worker(
            self.history.get_entries,
            search=search if search else None,
            language=language if language != "all" else None,  # Handle "all" filter
            start_date=start_date
        ).wait()
        
        # Update table
        self._populate_table()
        
        # Update stats
        stats = self.query_one("#stats-display", Static)
        stats.update(f"{len(self.current_entries)} entries")
    
    def _populate_table(self):
        """Populate the history table."""
        table = self.query_one("#history-table", DataTable)
        table.clear()
        
        for entry in self.current_entries:
            # Format values
            date_str = entry.timestamp.strftime("%Y-%m-%d")
            time_str = entry.timestamp.strftime("%H:%M:%S")
            duration_str = f"{entry.duration:.1f}s"
            words_str = str(entry.word_count)
            
            # Preview (first 50 chars)
            preview = entry.transcript[:50]
            if len(entry.transcript) > 50:
                preview += "..."
            
            table.add_row(
                date_str,
                time_str,
                duration_str,
                words_str,
                preview,
                key=entry.id
            )
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.data_table.id == "history-table":
            self.selected_entry_id = event.row_key.value
            
            # Find entry
            for entry in self.current_entries:
                if entry.id == self.selected_entry_id:
                    self.selected_entry = entry
                    self._show_entry_details(entry)
                    break
    
    def _show_entry_details(self, entry: TranscriptionEntry):
        """Show details of selected entry."""
        # Update title
        title = self.query_one("#detail-title", Label)
        title.update(f"Transcription from {entry.timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Update metadata
        meta = self.query_one("#detail-meta", Static)
        meta_text = f"Duration: {entry.duration:.1f}s | "
        meta_text += f"Words: {entry.word_count} | "
        meta_text += f"Language: {entry.language} | "
        meta_text += f"Provider: {entry.provider}"
        meta.update(meta_text)
        
        # Update transcript
        text_area = self.query_one("#transcript-text", TextArea)
        text_area.load_text(entry.transcript)
        
        # Enable action buttons
        self.query_one("#copy-btn", Button).disabled = False
        self.query_one("#export-btn", Button).disabled = False
        self.query_one("#delete-btn", Button).disabled = False
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "search-btn":
            self.search_text = self.query_one("#search-input", Input).value
            self.run_worker(self._refresh_entries())
        elif event.button.id == "refresh-btn":
            self.load_history()
        elif event.button.id == "clear-all-btn":
            self._confirm_clear_all()
        elif event.button.id == "copy-btn":
            self._copy_transcript()
        elif event.button.id == "export-btn":
            self._export_entries()
        elif event.button.id == "delete-btn":
            self._delete_entry()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "search-input":
            self.search_text = event.value
            self.run_worker(self._refresh_entries())
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        if event.select.id in ["language-filter", "time-filter"]:
            self.run_worker(self._refresh_entries())
    
    def _copy_transcript(self):
        """Copy selected transcript to clipboard."""
        if not self.selected_entry:
            return
        
        try:
            import pyperclip
            pyperclip.copy(self.selected_entry.transcript)
            self.app.notify("Transcript copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy: {e}")
            self.app.notify("Failed to copy transcript", severity="error")
    
    def _export_entries(self):
        """Export selected or all entries."""
        # Show file save dialog
        self.app.push_screen(
            FileSave(
                filters=Filters(
                    ("*.txt", "*.md", "*.json", "*.csv"),
                    {"Text": ["*.txt"], "Markdown": ["*.md"], "JSON": ["*.json"], "CSV": ["*.csv"]}
                ),
                filename="transcription_history.txt"
            ),
            callback=self._handle_export_file
        )
    
    def _handle_export_file(self, path: Optional[Path]):
        """Handle export file selection."""
        if not path:
            return
        
        # Determine format from extension
        ext = path.suffix.lower()
        format_map = {'.txt': 'txt', '.md': 'md', '.json': 'json', '.csv': 'csv'}
        format = format_map.get(ext, 'txt')
        
        # Export selected or all
        entries = [self.selected_entry] if self.selected_entry else self.current_entries
        
        success = self.history.export_to_file(path, format, entries)
        
        if success:
            self.app.notify(f"Exported {len(entries)} entries to {path.name}")
        else:
            self.app.notify("Failed to export entries", severity="error")
    
    def _delete_entry(self):
        """Delete selected entry."""
        if not self.selected_entry:
            return
        
        # Confirm deletion
        self.app.notify(
            f"Delete this entry? This cannot be undone.",
            title="Confirm Deletion",
            severity="warning"
        )
        
        # TODO: Show confirmation dialog
        # For now, just delete
        if self.history.delete_entry(self.selected_entry.id):
            self.app.notify("Entry deleted")
            self.selected_entry = None
            self.selected_entry_id = None
            self.run_worker(self._refresh_entries())
        else:
            self.app.notify("Failed to delete entry", severity="error")
    
    def _confirm_clear_all(self):
        """Confirm clearing all history."""
        # TODO: Show confirmation dialog
        self.app.notify(
            "Clear all history? This cannot be undone!",
            title="Confirm Clear All",
            severity="warning"
        )
        
        # For now, don't actually clear
        # In real implementation, would show dialog and then:
        # if confirmed:
        #     self.history.clear_history()
        #     self.run_worker(self._refresh_entries())