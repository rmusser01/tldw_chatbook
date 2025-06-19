# notes_sync_widget.py
# Description: Widget for managing note synchronization
#
# Imports
from pathlib import Path
from typing import Optional, Dict, Any, List
#
# Third-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, Select, Checkbox, Label, ListView, ListItem, ProgressBar
from textual.reactive import reactive
from textual.binding import Binding
from textual.css.query import NoMatches
from rich.text import Text
from rich.table import Table
#
# Local Imports
from ..Notes.sync_service import SyncDirection, ConflictResolution
from textual.screen import ModalScreen
#
########################################################################################################################
#
# Classes:

class SyncStatusIcon(Static):
    """Widget to display sync status with an icon."""
    
    def __init__(self, status: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self.status = status
    
    def on_mount(self):
        self.update_status(self.status)
    
    def update_status(self, status: str):
        """Update the displayed status."""
        self.status = status
        
        status_icons = {
            "synced": "✓",
            "file_changed": "↓",
            "db_changed": "↑",
            "conflict": "⚠",
            "file_missing": "✗",
            "file_error": "!",
            "syncing": "⟳",
            "unknown": "?"
        }
        
        status_colors = {
            "synced": "green",
            "file_changed": "yellow",
            "db_changed": "yellow",
            "conflict": "red",
            "file_missing": "red",
            "file_error": "red",
            "syncing": "blue",
            "unknown": "dim"
        }
        
        icon = status_icons.get(status, "?")
        color = status_colors.get(status, "white")
        
        self.update(Text(icon, style=color))


class SyncProgressWidget(Container):
    """Widget to display sync progress."""
    
    DEFAULT_CSS = """
    SyncProgressWidget {
        height: auto;
        padding: 1;
        border: solid $primary;
        display: none;
    }
    
    SyncProgressWidget.active {
        display: block;
    }
    
    #sync-progress-bar {
        margin: 1 0;
    }
    
    #sync-progress-status {
        text-align: center;
    }
    
    #sync-progress-details {
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("Sync in Progress", id="sync-progress-title")
        yield ProgressBar(id="sync-progress-bar", total=100)
        yield Static("Initializing...", id="sync-progress-status")
        yield Static("", id="sync-progress-details")
        yield Button("Cancel", id="sync-cancel-button", variant="error")
    
    def start_sync(self, total_files: int):
        """Start showing sync progress."""
        self.add_class("active")
        progress_bar = self.query_one("#sync-progress-bar", ProgressBar)
        progress_bar.total = total_files
        progress_bar.progress = 0
        self.query_one("#sync-progress-status", Static).update("Starting sync...")
    
    def update_progress(self, processed: int, total: int, status: str = None):
        """Update sync progress."""
        progress_bar = self.query_one("#sync-progress-bar", ProgressBar)
        progress_bar.progress = processed
        
        if status:
            self.query_one("#sync-progress-status", Static).update(status)
        
        percent = (processed / total * 100) if total > 0 else 0
        self.query_one("#sync-progress-details", Static).update(
            f"Processed: {processed}/{total} ({percent:.1f}%)"
        )
    
    def complete_sync(self, summary: Dict[str, Any]):
        """Show sync completion summary."""
        self.query_one("#sync-progress-status", Static).update("Sync completed!")
        
        details = []
        if summary.get('created_notes'):
            details.append(f"Created {summary['created_notes']} notes")
        if summary.get('updated_notes'):
            details.append(f"Updated {summary['updated_notes']} notes")
        if summary.get('created_files'):
            details.append(f"Created {summary['created_files']} files")
        if summary.get('updated_files'):
            details.append(f"Updated {summary['updated_files']} files")
        if summary.get('conflicts'):
            details.append(f"[red]{summary['conflicts']} conflicts[/red]")
        if summary.get('errors'):
            details.append(f"[red]{summary['errors']} errors[/red]")
        
        self.query_one("#sync-progress-details", Static).update(" | ".join(details))
        self.query_one("#sync-cancel-button", Button).label = "Close"
    
    def hide_progress(self):
        """Hide the progress widget."""
        self.remove_class("active")


class NotesSyncWidget(ModalScreen):
    """Main widget for note synchronization management."""
    
    def __init__(self, app_instance, **kwargs):
        """Initialize the sync widget with app instance."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sync_service = None  # Will be initialized on mount
    
    DEFAULT_CSS = """
    NotesSyncWidget {
        align: center middle;
    }
    
    NotesSyncWidget > Container {
        background: $surface;
        border: thick $primary;
        width: 80%;
        max-width: 100;
        height: 80%;
        overflow-y: auto;
    }
    
    .sync-section {
        margin: 1;
        padding: 1;
        border: solid $primary;
    }
    
    .sync-controls {
        layout: horizontal;
        height: auto;
        margin: 1 0;
    }
    
    .sync-controls > * {
        margin: 0 1;
    }
    
    #sync-folder-input {
        width: 50;
    }
    
    #sync-profiles-list {
        height: 10;
        margin: 1 0;
    }
    
    #sync-history-list {
        height: 15;
        margin: 1 0;
    }
    
    .sync-note-item {
        layout: horizontal;
        height: 1;
        padding: 0 1;
    }
    
    .sync-note-item > Static {
        width: 1fr;
    }
    
    .sync-note-item > SyncStatusIcon {
        width: 3;
    }
    
    .modal-header {
        height: 3;
        width: 100%;
        background: $primary-background;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .modal-title {
        width: 1fr;
        text-align: center;
        text-style: bold;
        content-align: center middle;
    }
    
    .close-button {
        width: 3;
        min-width: 3;
        background: transparent;
        border: none;
    }
    
    .close-button:hover {
        background: $error;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container():
            # Header with close button
            with Horizontal(classes="modal-header"):
                yield Static("📁 Notes Sync Manager", classes="modal-title")
                yield Button("✕", id="sync-close-button", classes="close-button")
            
            # Quick Sync Section
            with Container(classes="sync-section"):
                yield Static("Quick Sync", classes="section-title")
                
                with Horizontal(classes="sync-controls"):
                    yield Input(
                        placeholder="Select folder to sync...",
                        id="sync-folder-input"
                    )
                    yield Button("Browse", id="sync-browse-button")
                
                with Horizontal(classes="sync-controls"):
                    yield Select(
                    [(SyncDirection.BIDIRECTIONAL.value, "Bidirectional"),
                     (SyncDirection.DISK_TO_DB.value, "Disk → Database"),
                     (SyncDirection.DB_TO_DISK.value, "Database → Disk")],
                    id="sync-direction-select",
                    value=SyncDirection.BIDIRECTIONAL.value
                )
                    
                    yield Select(
                    [(ConflictResolution.ASK.value, "Ask on Conflict"),
                     (ConflictResolution.NEWER_WINS.value, "Newer Wins"),
                     (ConflictResolution.DB_WINS.value, "Database Wins"),
                     (ConflictResolution.DISK_WINS.value, "Disk Wins")],
                    id="sync-conflict-select",
                    value=ConflictResolution.ASK.value
                )
                
                with Horizontal(classes="sync-controls"):
                    yield Button("Start Sync", id="sync-start-button", variant="primary")
                    yield Button("Save as Profile", id="sync-save-profile-button")
            
            # Sync Profiles Section
            with Container(classes="sync-section"):
                yield Static("Sync Profiles", classes="section-title")
                yield ListView(id="sync-profiles-list")
                
                with Horizontal(classes="sync-controls"):
                    yield Button("Sync Selected", id="sync-profile-run-button")
                    yield Button("Edit", id="sync-profile-edit-button")
                    yield Button("Delete", id="sync-profile-delete-button", variant="error")
            
            # Sync Status Section
            with Container(classes="sync-section"):
                yield Static("Notes Sync Status", classes="section-title")
                
                with Horizontal(classes="sync-controls"):
                    yield Input(
                        placeholder="Filter notes...",
                        id="sync-notes-filter"
                    )
                    yield Select(
                        [("all", "All Notes"),
                         ("synced", "Synced"),
                         ("changed", "Changed"),
                         ("conflicts", "Conflicts")],
                        id="sync-status-filter",
                        value="all"
                    )
                    yield Button("Refresh", id="sync-refresh-status-button")
                
                    yield ScrollableContainer(id="sync-notes-status-container")
            
            # Sync History Section
            with Container(classes="sync-section"):
                yield Static("Sync History", classes="section-title")
                yield ListView(id="sync-history-list")
                
                with Horizontal(classes="sync-controls"):
                    yield Button("View Details", id="sync-history-details-button")
                    yield Button("View Conflicts", id="sync-history-conflicts-button")
            
            # Progress widget (hidden by default)
            yield SyncProgressWidget(id="sync-progress-widget")
    
    def on_mount(self):
        """Initialize the widget when mounted."""
        # Initialize sync service
        from ..Notes.sync_service import NotesSyncService
        if hasattr(self.app_instance, 'notes_service') and hasattr(self.app_instance, 'db'):
            self.sync_service = NotesSyncService(
                notes_service=self.app_instance.notes_service,
                db=self.app_instance.db
            )
        
        self.load_sync_profiles()
        self.load_sync_history()
        self.refresh_notes_status()
    
    def load_sync_profiles(self):
        """Load and display sync profiles."""
        # This would be populated from the sync service
        profiles_list = self.query_one("#sync-profiles-list", ListView)
        profiles_list.clear()
        
        # Example profiles (would come from sync service)
        example_profiles = [
            ("Work Notes", "/Users/me/Documents/WorkNotes", "Bidirectional"),
            ("Personal Journal", "/Users/me/Documents/Journal", "Database → Disk"),
        ]
        
        for name, folder, direction in example_profiles:
            item = ListItem(
                Static(f"{name} - {folder} ({direction})")
            )
            profiles_list.append(item)
    
    def load_sync_history(self):
        """Load and display sync history."""
        history_list = self.query_one("#sync-history-list", ListView)
        history_list.clear()
        
        # Example history (would come from sync service)
        example_history = [
            ("2024-01-15 10:30", "Work Notes", "Completed", "10 synced"),
            ("2024-01-15 09:15", "Personal Journal", "Completed", "5 synced, 1 conflict"),
            ("2024-01-14 18:00", "Work Notes", "Failed", "Connection error"),
        ]
        
        for timestamp, profile, status, summary in example_history:
            status_style = "green" if status == "Completed" else "red"
            item = ListItem(
                Static(f"{timestamp} - {profile} - [{status_style}]{status}[/{status_style}] - {summary}")
            )
            history_list.append(item)
    
    def refresh_notes_status(self):
        """Refresh the notes sync status display."""
        container = self.query_one("#sync-notes-status-container", ScrollableContainer)
        container.remove_children()
        
        # Example notes status (would come from sync service)
        example_notes = [
            ("Meeting Notes 2024-01-15", "synced"),
            ("Project TODO", "file_changed"),
            ("Ideas Brainstorm", "db_changed"),
            ("Design Review", "conflict"),
        ]
        
        with container:
            for title, status in example_notes:
                with Horizontal(classes="sync-note-item"):
                    container.mount(Static(title))
                    container.mount(SyncStatusIcon(status))
    
    async def start_sync(self, folder: Path, direction: SyncDirection, 
                        conflict_resolution: ConflictResolution):
        """Start a sync operation."""
        if not self.sync_service:
            from textual.widgets import MessageBox
            await self.app.push_screen(
                MessageBox("Sync service not initialized. Please try again.", title="Error")
            )
            return
        
        progress_widget = self.query_one("#sync-progress-widget", SyncProgressWidget)
        progress_widget.start_sync(100)  # Will be updated with actual total
        
        # Define progress callback
        def progress_callback(progress: 'SyncProgress'):
            progress_widget.update_progress(
                progress.processed_files,
                progress.total_files,
                f"Processing: {progress.current_file or 'Scanning...'}"
            )
        
        try:
            # Get current user ID from app
            user_id = getattr(self.app_instance, 'current_user_id', 'default_user')
            
            # Start the actual sync
            session_id = await self.sync_service.sync(
                user_id=user_id,
                sync_root=folder,
                direction=direction,
                conflict_resolution=conflict_resolution,
                progress_callback=progress_callback
            )
            
            # Get sync results
            results = self.sync_service.get_session_results(session_id)
            if results:
                progress_widget.complete_sync({
                    'created_notes': results.notes_created,
                    'updated_notes': results.notes_updated,
                    'created_files': results.files_created,
                    'conflicts': len(results.conflicts_resolved)
                })
            else:
                progress_widget.complete_sync({
                    'created_notes': 0,
                    'updated_notes': 0,
                    'created_files': 0,
                    'conflicts': 0
                })
                
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            progress_widget.hide_progress()
            from textual.widgets import MessageBox
            await self.app.push_screen(
                MessageBox(f"Sync failed: {str(e)}", title="Sync Error")
            )
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "sync-browse-button":
            # Handle browse button
            from ..Event_Handlers.notes_sync_events import handle_sync_browse_button
            await handle_sync_browse_button(self.app_instance, event)
            
        elif event.button.id == "sync-start-button":
            # Get sync parameters
            folder_input = self.query_one("#sync-folder-input", Input)
            direction_select = self.query_one("#sync-direction-select", Select)
            conflict_select = self.query_one("#sync-conflict-select", Select)
            
            if folder_input.value:
                folder = Path(folder_input.value)
                direction = SyncDirection(direction_select.value)
                conflict_resolution = ConflictResolution(conflict_select.value)
                
                await self.start_sync(folder, direction, conflict_resolution)
        
        elif event.button.id == "sync-refresh-status-button":
            self.refresh_notes_status()
        
        elif event.button.id == "sync-cancel-button":
            progress_widget = self.query_one("#sync-progress-widget", SyncProgressWidget)
            if event.button.label == "Close":
                progress_widget.hide_progress()
        
        elif event.button.id == "sync-close-button":
            self.dismiss()

#
# End of notes_sync_widget.py
########################################################################################################################