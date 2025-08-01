# ChatbookExportManagementWindow.py
# Description: Window for managing exported chatbooks
#
"""
Chatbook Export Management Window
---------------------------------

Provides interface for:
- Viewing all exported chatbooks
- Re-exporting with different settings
- Deleting old exports
- Sharing chatbooks
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Button, DataTable, Label, Checkbox,
    OptionList, Header, Footer
)
from textual.reactive import reactive
from textual.coordinate import Coordinate
from loguru import logger

from ..Chatbooks.chatbook_importer import ChatbookImporter
from ..Chatbooks.chatbook_models import ChatbookManifest

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookExportManagementWindow(ModalScreen):
    """Window for managing exported chatbooks."""
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("d", "delete_selected", "Delete"),
        ("r", "refresh", "Refresh"),
        ("e", "re_export", "Re-export"),
        ("s", "share", "Share"),
        ("o", "open_location", "Open Location")
    ]
    
    DEFAULT_CSS = """
    ChatbookExportManagementWindow {
        align: center middle;
    }
    
    ChatbookExportManagementWindow > Container {
        width: 90%;
        height: 90%;
        max-width: 120;
        background: $surface;
        border: thick $primary;
    }
    
    .window-header {
        height: 3;
        padding: 1;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    .window-title {
        text-style: bold;
        text-align: center;
        color: $primary;
    }
    
    .toolbar {
        height: 4;
        padding: 1;
        background: $panel;
        border-bottom: solid $background-darken-1;
    }
    
    .toolbar-buttons {
        layout: horizontal;
        height: auto;
        align: left middle;
    }
    
    .toolbar Button {
        margin-right: 1;
        min-width: 16;
    }
    
    .main-content {
        layout: horizontal;
        height: 1fr;
    }
    
    .chatbook-list-container {
        width: 35%;
        border-right: solid $background-darken-1;
        padding: 1;
    }
    
    .list-header {
        height: 3;
        padding: 0 1;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 1;
    }
    
    .list-title {
        text-style: bold;
        margin-bottom: 0;
    }
    
    .list-subtitle {
        color: $text-muted;
        font-size: 90%;
        margin-top: 0;
    }
    
    #chatbook-list {
        height: 1fr;
        background: $boost;
        border: round $background-darken-1;
    }
    
    .details-container {
        width: 65%;
        padding: 1;
    }
    
    .details-header {
        height: auto;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 1;
    }
    
    .details-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .details-meta {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        margin-top: 1;
    }
    
    .meta-label {
        text-align: right;
        color: $text-muted;
        padding-right: 1;
    }
    
    .meta-value {
        text-align: left;
    }
    
    .content-preview {
        height: auto;
        padding: 1;
        background: $boost;
        border: round $background-darken-1;
        margin-bottom: 1;
    }
    
    .preview-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .content-table {
        height: 12;
        background: $background;
        border: round $background-darken-1;
    }
    
    .action-buttons {
        height: auto;
        padding: 1;
        background: $panel;
        border: round $background-darken-1;
    }
    
    .action-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    .action-row Button {
        margin-right: 1;
        min-width: 20;
    }
    
    .no-selection {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    
    .status-bar {
        dock: bottom;
        height: 3;
        padding: 1;
        background: $panel;
        border-top: solid $background-darken-1;
    }
    
    .status-text {
        text-align: left;
        color: $text-muted;
    }
    """
    
    # Reactive properties
    selected_chatbook = reactive(None)
    chatbook_count = reactive(0)
    total_size = reactive(0)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.chatbooks_dir = Path.home() / "Documents" / "Chatbooks"
        self.chatbook_files: List[Dict[str, Any]] = []
        self.current_manifest: Optional[ChatbookManifest] = None
        
    def compose(self) -> ComposeResult:
        """Compose the management UI."""
        with Container():
            # Header
            with Container(classes="window-header"):
                yield Static("📦 Chatbook Export Management", classes="window-title")
            
            # Toolbar
            with Container(classes="toolbar"):
                with Horizontal(classes="toolbar-buttons"):
                    yield Button("🔄 Refresh", id="refresh-list", variant="default")
                    yield Button("🗑️ Delete", id="delete-selected", variant="warning", disabled=True)
                    yield Button("📤 Re-export", id="re-export", variant="default", disabled=True)
                    yield Button("🔗 Share", id="share-selected", variant="default", disabled=True)
                    yield Button("📁 Open Location", id="open-location", variant="default", disabled=True)
            
            # Main content area
            with Container(classes="main-content"):
                # Left: Chatbook list
                with Container(classes="chatbook-list-container"):
                    with Container(classes="list-header"):
                        yield Static("Exported Chatbooks", classes="list-title")
                        yield Static("", id="list-count", classes="list-subtitle")
                    
                    yield OptionList(id="chatbook-list")
                
                # Right: Details
                with Container(classes="details-container"):
                    yield Container(id="no-selection-container", classes="no-selection")
                    yield Container(id="details-content", display=False)
            
            # Status bar
            with Container(classes="status-bar"):
                yield Static("", id="status-text", classes="status-text")
            
            # Footer
            yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize when mounted."""
        # Add no-selection message
        no_selection = self.query_one("#no-selection-container", Container)
        no_selection.mount(Static("Select a chatbook to view details"))
        
        # Create details content structure
        details = self.query_one("#details-content", Container)
        
        # Header section
        with details:
            header_container = Container(classes="details-header")
            header_container.mount(Static("", id="chatbook-name", classes="details-title"))
            
            meta_grid = Grid(classes="details-meta")
            meta_grid.mount(Static("Size:", classes="meta-label"))
            meta_grid.mount(Static("", id="meta-size", classes="meta-value"))
            meta_grid.mount(Static("Created:", classes="meta-label"))
            meta_grid.mount(Static("", id="meta-created", classes="meta-value"))
            meta_grid.mount(Static("Author:", classes="meta-label"))
            meta_grid.mount(Static("", id="meta-author", classes="meta-value"))
            meta_grid.mount(Static("Version:", classes="meta-label"))
            meta_grid.mount(Static("", id="meta-version", classes="meta-value"))
            
            header_container.mount(meta_grid)
            details.mount(header_container)
            
            # Content preview
            preview_container = Container(classes="content-preview")
            preview_container.mount(Static("Content Summary:", classes="preview-title"))
            
            content_table = DataTable(
                id="content-table",
                classes="content-table",
                cursor_type="row",
                zebra_stripes=True
            )
            content_table.add_columns("Type", "Count", "Size")
            preview_container.mount(content_table)
            details.mount(preview_container)
            
            # Action buttons
            actions_container = Container(classes="action-buttons")
            
            row1 = Horizontal(classes="action-row")
            row1.mount(Button("📤 Re-export with Options", id="re-export-options", variant="primary"))
            row1.mount(Button("📋 Copy Path", id="copy-path", variant="default"))
            actions_container.mount(row1)
            
            row2 = Horizontal(classes="action-row")
            row2.mount(Button("📧 Email", id="share-email", variant="default"))
            row2.mount(Button("☁️ Upload to Cloud", id="share-cloud", variant="default"))
            actions_container.mount(row2)
            
            details.mount(actions_container)
        
        # Load chatbooks
        await self.refresh_chatbook_list()
    
    async def refresh_chatbook_list(self) -> None:
        """Refresh the list of chatbooks."""
        self.chatbook_files.clear()
        
        try:
            # Find all chatbook files
            if self.chatbooks_dir.exists():
                for file_path in self.chatbooks_dir.glob("*.zip"):
                    try:
                        stat = file_path.stat()
                        self.chatbook_files.append({
                            "name": file_path.stem,
                            "path": file_path,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                            "created": datetime.fromtimestamp(stat.st_ctime)
                        })
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
            
            # Sort by modification time (newest first)
            self.chatbook_files.sort(key=lambda x: x["modified"], reverse=True)
            
            # Update the list widget
            option_list = self.query_one("#chatbook-list", OptionList)
            option_list.clear_options()
            
            for i, chatbook in enumerate(self.chatbook_files):
                size_mb = chatbook["size"] / (1024 * 1024)
                modified = self._format_relative_time(chatbook["modified"])
                option_list.add_option(
                    f"{chatbook['name']}\n  {size_mb:.1f} MB • {modified}",
                    id=str(i)
                )
            
            # Update counts
            self.chatbook_count = len(self.chatbook_files)
            self.total_size = sum(cb["size"] for cb in self.chatbook_files)
            
            # Update UI
            self._update_list_count()
            self._update_status()
            
        except Exception as e:
            logger.error(f"Error refreshing chatbook list: {e}")
            self.app_instance.notify(f"Error loading chatbooks: {str(e)}", severity="error")
    
    def _format_relative_time(self, dt: datetime) -> str:
        """Format datetime as relative time."""
        now = datetime.now()
        delta = now - dt
        
        if delta.days == 0:
            if delta.seconds < 60:
                return "just now"
            elif delta.seconds < 3600:
                return f"{delta.seconds // 60}m ago"
            else:
                return f"{delta.seconds // 3600}h ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days}d ago"
        elif delta.days < 30:
            return f"{delta.days // 7}w ago"
        else:
            return dt.strftime("%Y-%m-%d")
    
    def _update_list_count(self) -> None:
        """Update the list count display."""
        count_label = self.query_one("#list-count", Static)
        total_mb = self.total_size / (1024 * 1024)
        count_label.update(f"{self.chatbook_count} chatbooks • {total_mb:.1f} MB total")
    
    def _update_status(self) -> None:
        """Update status bar."""
        status = self.query_one("#status-text", Static)
        if self.selected_chatbook is not None:
            chatbook = self.chatbook_files[self.selected_chatbook]
            status.update(f"Selected: {chatbook['path']}")
        else:
            status.update(f"Storage: {self.chatbooks_dir}")
    
    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle chatbook selection."""
        try:
            index = int(event.option_id)
            self.selected_chatbook = index
            
            # Enable action buttons
            self.query_one("#delete-selected", Button).disabled = False
            self.query_one("#re-export", Button).disabled = False
            self.query_one("#share-selected", Button).disabled = False
            self.query_one("#open-location", Button).disabled = False
            
            # Load and display details
            await self._load_chatbook_details(index)
            
        except Exception as e:
            logger.error(f"Error selecting chatbook: {e}")
    
    async def _load_chatbook_details(self, index: int) -> None:
        """Load details for selected chatbook."""
        chatbook = self.chatbook_files[index]
        
        # Show details container
        self.query_one("#no-selection-container", Container).display = False
        self.query_one("#details-content", Container).display = True
        
        # Update basic info
        self.query_one("#chatbook-name", Static).update(f"📚 {chatbook['name']}")
        self.query_one("#meta-size", Static).update(f"{chatbook['size'] / (1024 * 1024):.2f} MB")
        self.query_one("#meta-created", Static).update(chatbook['created'].strftime("%Y-%m-%d %H:%M"))
        
        # Try to load manifest
        try:
            # Create importer to preview
            db_config = self.app_instance.config_data.get("database", {})
            db_paths = {
                "ChaChaNotes": str(Path(db_config.get("chachanotes_db_path", 
                    "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")).expanduser()),
                "Prompts": str(Path(db_config.get("prompts_db_path", 
                    "~/.local/share/tldw_cli/tldw_prompts.db")).expanduser()),
                "Media": str(Path(db_config.get("media_db_path", 
                    "~/.local/share/tldw_cli/media_db_v2.db")).expanduser())
            }
            
            importer = ChatbookImporter(db_paths)
            manifest, error = importer.preview_chatbook(chatbook['path'])
            
            if manifest and not error:
                self.current_manifest = manifest
                self.query_one("#meta-author", Static).update(manifest.author or "Unknown")
                self.query_one("#meta-version", Static).update(manifest.version.value)
                
                # Update content table
                table = self.query_one("#content-table", DataTable)
                table.clear()
                
                if manifest.total_conversations > 0:
                    table.add_row("Conversations", str(manifest.total_conversations), "-")
                if manifest.total_notes > 0:
                    table.add_row("Notes", str(manifest.total_notes), "-")
                if manifest.total_characters > 0:
                    table.add_row("Characters", str(manifest.total_characters), "-")
                if manifest.total_media_items > 0:
                    table.add_row("Media", str(manifest.total_media_items), "-")
                if len([i for i in manifest.content_items if i.type.value == "prompt"]) > 0:
                    prompt_count = len([i for i in manifest.content_items if i.type.value == "prompt"])
                    table.add_row("Prompts", str(prompt_count), "-")
            else:
                self.query_one("#meta-author", Static).update("Error loading")
                self.query_one("#meta-version", Static).update("-")
                
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            self.query_one("#meta-author", Static).update("Error")
            self.query_one("#meta-version", Static).update("-")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "refresh-list":
            await self.refresh_chatbook_list()
            self.app_instance.notify("Chatbook list refreshed", severity="success")
            
        elif button_id == "delete-selected" and self.selected_chatbook is not None:
            await self._delete_selected()
            
        elif button_id == "re-export" and self.selected_chatbook is not None:
            await self._re_export_chatbook()
            
        elif button_id == "share-selected" and self.selected_chatbook is not None:
            self.app_instance.notify("Share functionality coming soon!", severity="info")
            
        elif button_id == "open-location" and self.selected_chatbook is not None:
            await self._open_location()
            
        elif button_id == "copy-path" and self.selected_chatbook is not None:
            chatbook = self.chatbook_files[self.selected_chatbook]
            # Would copy to clipboard
            self.app_instance.notify(f"Path: {chatbook['path']}", severity="info")
            
        elif button_id == "re-export-options":
            self.app_instance.notify("Re-export with options coming soon!", severity="info")
            
        elif button_id == "share-email":
            self.app_instance.notify("Email sharing coming soon!", severity="info")
            
        elif button_id == "share-cloud":
            self.app_instance.notify("Cloud upload coming soon!", severity="info")
    
    async def _delete_selected(self) -> None:
        """Delete selected chatbook."""
        chatbook = self.chatbook_files[self.selected_chatbook]
        
        # Confirm deletion
        from ..Widgets.confirmation_dialog import ConfirmationDialog
        
        dialog = ConfirmationDialog(
            title="Delete Chatbook",
            message=f"Are you sure you want to delete '{chatbook['name']}'?\n\nThis action cannot be undone.",
            confirm_text="Delete",
            cancel_text="Cancel"
        )
        
        result = await self.app_instance.push_screen(dialog, wait_for_dismiss=True)
        
        if result:
            try:
                # Delete the file
                chatbook['path'].unlink()
                
                # Refresh list
                await self.refresh_chatbook_list()
                
                # Clear selection
                self.selected_chatbook = None
                self.query_one("#no-selection-container", Container).display = True
                self.query_one("#details-content", Container).display = False
                
                # Disable buttons
                self.query_one("#delete-selected", Button).disabled = True
                self.query_one("#re-export", Button).disabled = True
                self.query_one("#share-selected", Button).disabled = True
                self.query_one("#open-location", Button).disabled = True
                
                self.app_instance.notify(f"Deleted '{chatbook['name']}'", severity="success")
                
            except Exception as e:
                logger.error(f"Error deleting chatbook: {e}")
                self.app_instance.notify(f"Error deleting: {str(e)}", severity="error")
    
    async def _re_export_chatbook(self) -> None:
        """Re-export the selected chatbook."""
        # For now, just show notification
        # In future, could launch creation wizard with pre-filled selections
        self.app_instance.notify("Re-export functionality coming soon!", severity="info")
    
    async def _open_location(self) -> None:
        """Open the folder containing the chatbook."""
        chatbook = self.chatbook_files[self.selected_chatbook]
        
        try:
            import subprocess
            import platform
            
            folder = str(chatbook['path'].parent)
            
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder])
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", folder])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", folder])
                
        except Exception as e:
            logger.error(f"Error opening folder: {e}")
            self.app_instance.notify(f"Error opening folder: {str(e)}", severity="error")
    
    def action_close(self) -> None:
        """Close the window."""
        self.dismiss()
    
    def action_delete_selected(self) -> None:
        """Delete selected chatbook."""
        if self.selected_chatbook is not None:
            self.run_worker(self._delete_selected())
    
    def action_refresh(self) -> None:
        """Refresh the list."""
        self.run_worker(self.refresh_chatbook_list())
    
    def action_re_export(self) -> None:
        """Re-export selected chatbook."""
        if self.selected_chatbook is not None:
            self.run_worker(self._re_export_chatbook())
    
    def action_share(self) -> None:
        """Share selected chatbook."""
        if self.selected_chatbook is not None:
            self.app_instance.notify("Share functionality coming soon!", severity="info")
    
    def action_open_location(self) -> None:
        """Open chatbook location."""
        if self.selected_chatbook is not None:
            self.run_worker(self._open_location())