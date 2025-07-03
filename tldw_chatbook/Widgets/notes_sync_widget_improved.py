# notes_sync_widget_improved.py
# Description: Improved modal sync widget with better UX
#
# Imports
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
#
# Third-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import Static, Button, Label, ProgressBar, Switch, Input, Select
from textual.reactive import reactive
from textual.screen import ModalScreen
from rich.text import Text
from loguru import logger
#
# Local Imports
from ..Notes.sync_engine import SyncDirection, ConflictResolution
from ..Third_Party.textual_fspicker import SelectDirectory
#
########################################################################################################################
#
# Classes:

class SyncStatusCard(Container):
    """Compact card showing sync status."""
    
    DEFAULT_CSS = """
    SyncStatusCard {
        height: auto;
        padding: 1 2;
        background: $boost;
        border: solid $primary;
        margin-bottom: 1;
    }
    
    .status-header {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .status-detail {
        color: $text-muted;
    }
    
    .status-synced {
        color: $success;
    }
    
    .status-pending {
        color: $warning;
    }
    
    .status-error {
        color: $error;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("📁 My Notes", classes="status-header")
        yield Static("Last synced: Never", id="last-sync-time", classes="status-detail")
        yield Static("Status: Ready to sync", id="sync-status", classes="status-detail status-synced")


class QuickSyncSection(Container):
    """Sync configuration and controls."""
    
    DEFAULT_CSS = """
    QuickSyncSection {
        height: auto;
        padding: 1;
        margin-bottom: 2;
    }
    
    .folder-row {
        margin-bottom: 1;
        height: 3;
    }
    
    #sync-folder-input {
        width: 1fr;
        margin-right: 1;
    }
    
    #browse-folder-btn {
        width: 10;
    }
    
    .sync-settings-row {
        margin-bottom: 1;
        height: auto;
    }
    
    .sync-setting {
        width: 50%;
        padding-right: 1;
    }
    
    .setting-label {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    .sync-button-container {
        align: center middle;
        height: 5;
        margin-top: 1;
    }
    
    #quick-sync-btn {
        width: 30;
        height: 3;
    }
    
    .sync-options {
        margin-top: 1;
        align: center middle;
    }
    
    .sync-option {
        margin: 0 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        # Folder selection
        with Horizontal(classes="folder-row"):
            yield Input(
                placeholder="Select folder to sync...",
                id="sync-folder-input"
            )
            yield Button("Browse", id="browse-folder-btn", variant="default")
        
        # Sync settings
        with Horizontal(classes="sync-settings-row"):
            with Vertical(classes="sync-setting"):
                yield Label("Sync Direction:", classes="setting-label")
                yield Select(
                    [("↔ Two-way sync", "bidirectional"),
                     ("→ Import from files", "disk_to_db"),
                     ("← Export to files", "db_to_disk")],
                    id="sync-direction",
                    value="bidirectional"
                )
            
            with Vertical(classes="sync-setting"):
                yield Label("If conflict occurs:", classes="setting-label")
                yield Select(
                    [("Keep newer version", "newer_wins"),
                     ("Ask me each time", "ask"),
                     ("Always use file version", "disk_wins"),
                     ("Always use app version", "db_wins")],
                    id="conflict-resolution",
                    value="newer_wins"
                )
        
        # Sync button
        with Center(classes="sync-button-container"):
            yield Button("🔄 Sync Now", id="quick-sync-btn", variant="primary")
        
        # Auto-sync option
        with Horizontal(classes="sync-options"):
            with Horizontal(classes="sync-option"):
                yield Label("Auto-sync: ")
                yield Switch(id="auto-sync-switch", value=False)


class SyncProgressSection(Container):
    """Progress display during sync."""
    
    DEFAULT_CSS = """
    SyncProgressSection {
        height: auto;
        padding: 1;
        display: none;
    }
    
    SyncProgressSection.active {
        display: block;
    }
    
    .progress-header {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    #sync-progress-bar {
        margin: 1 0;
    }
    
    .progress-status {
        text-align: center;
        color: $text-muted;
    }
    """
    
    progress_text = reactive("Starting sync...")
    
    def compose(self) -> ComposeResult:
        yield Static("Syncing...", classes="progress-header")
        yield ProgressBar(id="sync-progress-bar", total=100)
        yield Static(self.progress_text, id="progress-status", classes="progress-status")
    
    def start_progress(self, total: int = 100):
        """Start showing progress."""
        self.add_class("active")
        bar = self.query_one("#sync-progress-bar", ProgressBar)
        bar.total = total
        bar.progress = 0
    
    def update_progress(self, current: int, total: int, status: str = None):
        """Update progress display."""
        bar = self.query_one("#sync-progress-bar", ProgressBar)
        bar.progress = current
        
        if status:
            self.query_one("#progress-status", Static).update(status)
    
    def complete_progress(self):
        """Hide progress section."""
        self.remove_class("active")


class RecentActivitySection(ScrollableContainer):
    """Shows recent sync activity."""
    
    DEFAULT_CSS = """
    RecentActivitySection {
        height: 10;
        border: solid $primary;
        padding: 1;
        margin-top: 2;
    }
    
    .activity-header {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .activity-item {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    .activity-success {
        color: $success;
    }
    
    .activity-error {
        color: $error;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("Recent Activity", classes="activity-header")
    
    def add_activity(self, message: str, status: str = "info"):
        """Add an activity entry."""
        time_str = datetime.now().strftime("%H:%M")
        style_class = f"activity-{status}" if status in ["success", "error"] else ""
        
        item = Static(
            f"[{time_str}] {message}",
            classes=f"activity-item {style_class}"
        )
        self.mount(item)
        
        # Keep only last 10 items
        while len(self.children) > 11:  # +1 for header
            self.children[-1].remove()


class NotesSyncWidgetImproved(ModalScreen):
    """Improved sync modal with better UX."""
    
    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]
    
    DEFAULT_CSS = """
    NotesSyncWidgetImproved {
        align: center middle;
    }
    
    NotesSyncWidgetImproved > Container {
        background: $surface;
        border: thick $primary;
        width: 60;
        max-width: 80;
        height: auto;
        max-height: 40;
        padding: 0;
    }
    
    .modal-header {
        height: 3;
        width: 100%;
        background: $primary-background;
        padding: 0 1;
        margin-bottom: 1;
        layout: horizontal;
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
    
    .modal-content {
        padding: 2;
    }
    
    .modal-footer {
        margin-top: 2;
        padding-top: 1;
        border-top: solid $primary;
        align: center middle;
    }
    
    #close-btn {
        width: 20;
    }
    """
    
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sync_service = None
        self.sync_in_progress = False
    
    def compose(self) -> ComposeResult:
        with Container():
            # Header with close button
            with Horizontal(classes="modal-header"):
                yield Static("📂 Notes Sync", classes="modal-title")
                yield Button("✕", id="sync-close-button", classes="close-button")
            
            # Main content
            with Container(classes="modal-content"):
                # Status card
                yield SyncStatusCard(id="status-card")
                
                # Quick sync section
                yield QuickSyncSection(id="quick-sync-section")
                
                # Progress section (hidden by default)
                yield SyncProgressSection(id="progress-section")
                
                # Recent activity
                yield RecentActivitySection(id="activity-section")
                
                # Footer
                with Center(classes="modal-footer"):
                    yield Button("Close", id="close-btn", variant="default")
    
    def on_mount(self):
        """Initialize when mounted."""
        # Get sync service
        try:
            from ..Notes.sync_service import NotesSyncService
            if hasattr(self.app_instance, 'notes_service') and hasattr(self.app_instance, 'db'):
                self.sync_service = NotesSyncService(
                    notes_service=self.app_instance.notes_service,
                    db=self.app_instance.db
                )
        except ImportError:
            logger.warning("Sync service not available - running in demo mode")
            self.sync_service = None
        
        # Update status
        self.update_status()
        
        # Load saved sync directory
        from tldw_chatbook.config import get_cli_setting
        saved_dir = get_cli_setting("notes", "sync_directory", "~/Documents/Notes")
        folder_input = self.query_one("#sync-folder-input", Input)
        folder_input.value = str(Path(saved_dir).expanduser())
        
        # Add some example activities
        activity = self.query_one("#activity-section", RecentActivitySection)
        activity.add_activity("Sync widget opened", "info")
    
    def update_status(self):
        """Update the status display."""
        # Get last sync time from config or storage
        last_sync = getattr(self.app_instance, 'last_sync_time', None)
        
        if last_sync:
            time_str = last_sync.strftime("%Y-%m-%d %H:%M")
            self.query_one("#last-sync-time", Static).update(f"Last synced: {time_str}")
        else:
            self.query_one("#last-sync-time", Static).update("Last synced: Never")
        
        # Check for pending changes
        # This would check actual file changes in a real implementation
        self.query_one("#sync-status", Static).update("Status: Ready to sync")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close-btn" or event.button.id == "sync-close-button":
            if not self.sync_in_progress:
                self.dismiss()
        
        elif event.button.id == "browse-folder-btn":
            # Handle folder browse
            await self.browse_folder()
        
        elif event.button.id == "quick-sync-btn":
            if not self.sync_in_progress:
                await self.perform_sync()
    
    async def browse_folder(self):
        """Open folder browser."""
        async def folder_selected(path: Optional[Path]) -> None:
            if path:
                folder_input = self.query_one("#sync-folder-input", Input)
                folder_input.value = str(path)
                
                # Save the selected directory to config
                from tldw_chatbook.config import set_cli_setting
                if set_cli_setting("notes", "sync_directory", str(path)):
                    logger.info(f"Saved notes sync directory to config: {path}")
                
                # Update status card with folder name
                status_card = self.query_one("#status-card", SyncStatusCard)
                folder_name = path.name if path.name else "Notes"
                status_header = status_card.query_one(".status-header", Static)
                status_header.update(f"📁 {folder_name}")
        
        # Get the configured notes sync directory with intelligent fallback
        from tldw_chatbook.config import get_cli_setting
        default_notes_dir = get_cli_setting("notes", "sync_directory", "~/Documents/Notes")
        
        # Expand user path and resolve
        current_path = Path(default_notes_dir).expanduser().resolve()
        
        # Intelligent fallback if directory doesn't exist
        if not current_path.exists():
            # Try to create it
            try:
                current_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created notes directory: {current_path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create notes directory {current_path}: {e}")
                # Fallback to user's home directory
                current_path = Path.home()
                self.notify(f"Notes directory not accessible, using: {current_path}", severity="warning")
        
        await self.app.push_screen(
            SelectDirectory(str(current_path), title="Select Notes Folder"),
            callback=folder_selected
        )
    
    async def perform_sync(self):
        """Perform the sync operation."""
        # Get sync parameters
        folder_input = self.query_one("#sync-folder-input", Input)
        if not folder_input.value:
            self.notify("Please select a folder to sync", severity="warning")
            return
        
        sync_folder = Path(folder_input.value)
        if not sync_folder.exists():
            self.notify("Selected folder does not exist", severity="error")
            return
        
        # Get sync direction and conflict resolution
        direction_select = self.query_one("#sync-direction", Select)
        conflict_select = self.query_one("#conflict-resolution", Select)
        
        sync_direction = SyncDirection(direction_select.value)
        conflict_resolution = ConflictResolution(conflict_select.value)
        
        # Check if we have a sync service
        if not self.sync_service:
            # Run in demo mode
            await self.perform_demo_sync(sync_folder, sync_direction, conflict_resolution)
            return
        
        self.sync_in_progress = True
        
        # Disable buttons
        self.query_one("#quick-sync-btn", Button).disabled = True
        self.query_one("#close-btn", Button).label = "Syncing..."
        self.query_one("#sync-close-button", Button).disabled = True
        
        # Show progress
        progress = self.query_one("#progress-section", SyncProgressSection)
        progress.start_progress()
        
        # Get activity section
        activity = self.query_one("#activity-section", RecentActivitySection)
        activity.add_activity(f"Starting sync: {sync_folder.name}", "info")
        
        try:
            # Define progress callback
            def progress_callback(sync_progress):
                # Update progress in UI thread
                progress.update_progress(
                    sync_progress.processed_files,
                    sync_progress.total_files,
                    f"Processing: {sync_progress.current_file or 'Scanning...'}"
                )
            
            # Get user ID
            user_id = getattr(self.app_instance, 'current_user_id', 'default_user')
            
            # Perform actual sync
            session_id = await self.sync_service.sync(
                user_id=user_id,
                sync_root=sync_folder,
                direction=sync_direction,
                conflict_resolution=conflict_resolution,
                progress_callback=progress_callback
            )
            
            # Get results
            results = self.sync_service.get_session_results(session_id)
            
            if results:
                # Build summary
                summary_parts = []
                if results.notes_created:
                    summary_parts.append(f"{results.notes_created} notes created")
                if results.notes_updated:
                    summary_parts.append(f"{results.notes_updated} notes updated")
                if results.files_created:
                    summary_parts.append(f"{results.files_created} files created")
                if results.files_updated:
                    summary_parts.append(f"{results.files_updated} files updated")
                
                if summary_parts:
                    activity.add_activity(f"Sync complete: {', '.join(summary_parts)}", "success")
                else:
                    activity.add_activity("Sync complete: No changes", "success")
                
                if results.conflicts_resolved:
                    activity.add_activity(f"{len(results.conflicts_resolved)} conflicts resolved", "info")
            else:
                activity.add_activity("Sync completed", "success")
            
            self.app_instance.last_sync_time = datetime.now()
            self.update_status()
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            activity.add_activity(f"Sync failed: {str(e)}", "error")
            self.notify("Sync failed", severity="error")
        
        finally:
            # Hide progress
            progress.complete_progress()
            
            # Re-enable buttons
            self.query_one("#quick-sync-btn", Button).disabled = False
            self.query_one("#close-btn", Button).label = "Close"
            self.query_one("#sync-close-button", Button).disabled = False
            
            self.sync_in_progress = False
    
    async def perform_demo_sync(self, sync_folder: Path, direction: SyncDirection, resolution: ConflictResolution):
        """Perform a demo sync when service is not available."""
        self.sync_in_progress = True
        
        # Disable buttons
        self.query_one("#quick-sync-btn", Button).disabled = True
        self.query_one("#close-btn", Button).label = "Syncing..."
        self.query_one("#sync-close-button", Button).disabled = True
        
        # Show progress
        progress = self.query_one("#progress-section", SyncProgressSection)
        progress.start_progress()
        
        # Get activity section
        activity = self.query_one("#activity-section", RecentActivitySection)
        activity.add_activity(f"Demo sync: {sync_folder.name}", "info")
        activity.add_activity(f"Direction: {direction.value}", "info")
        activity.add_activity(f"Conflict resolution: {resolution.value}", "info")
        
        try:
            # Simulate sync with progress updates
            total_steps = 5
            for i in range(total_steps):
                progress.update_progress(i + 1, total_steps, f"Processing step {i + 1} of {total_steps}")
                await asyncio.sleep(0.5)
            
            # Simulate results
            activity.add_activity("Demo sync complete: 3 notes synced", "success")
            self.app_instance.last_sync_time = datetime.now()
            self.update_status()
            
        finally:
            # Hide progress
            progress.complete_progress()
            
            # Re-enable buttons
            self.query_one("#quick-sync-btn", Button).disabled = False
            self.query_one("#close-btn", Button).label = "Close"
            self.query_one("#sync-close-button", Button).disabled = False
            
            self.sync_in_progress = False
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle auto-sync toggle."""
        if event.switch.id == "auto-sync-switch":
            # Store auto-sync preference
            if hasattr(self.app_instance, 'config'):
                self.app_instance.config.set('notes', 'auto_sync', event.value)
            
            activity = self.query_one("#activity-section", RecentActivitySection)
            status = "enabled" if event.value else "disabled"
            activity.add_activity(f"Auto-sync {status}", "info")

#
# End of notes_sync_widget_improved.py
########################################################################################################################