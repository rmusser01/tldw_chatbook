# tldw_chatbook/Widgets/Media_Ingest/components/file_selector.py
"""
Enhanced file selector component with drag-drop, browse, and URL support.
"""

from typing import List, Callable, Optional, Tuple
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, TextArea, Label
from textual.reactive import reactive
from textual import on
from loguru import logger


class FileSelector(Container):
    """
    File selection component supporting:
    - File browsing with filters
    - URL input
    - File list display with metadata
    - Remove individual files
    - Clear all functionality
    """
    
    selected_files = reactive([])
    selected_urls = reactive([])
    
    def __init__(
        self, 
        file_filters: Optional[List[Tuple[str, str]]] = None,
        on_files_changed: Optional[Callable] = None,
        media_type: str = "media",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.file_filters = file_filters or [("All Files", "*")]
        self.on_files_changed = on_files_changed
        self.media_type = media_type
    
    def compose(self) -> ComposeResult:
        """Compose the file selector."""
        with Container(classes="file-selector-container"):
            yield Label("Select Files", classes="form-label-primary")
            
            # Action buttons
            with Horizontal(classes="file-actions-row"):
                yield Button("Browse Files", id="browse-files", variant="primary", classes="file-action-button")
                yield Button("Add URLs", id="add-urls", variant="default", classes="file-action-button") 
                yield Button("Clear All", id="clear-all", variant="default", classes="file-action-button")
            
            # File list display
            yield Container(id="file-display", classes="file-display-container")
            
            # URL input section (initially hidden)
            with Container(id="url-section", classes="url-input-container hidden"):
                yield Label(f"Enter {self.media_type} URLs (one per line):")
                yield TextArea(
                    placeholder=f"https://example.com/{self.media_type}.mp4",
                    id="url-textarea",
                    classes="url-textarea"
                )
                with Horizontal(classes="url-button-row"):
                    yield Button("Add URLs", id="confirm-urls", variant="primary")
                    yield Button("Cancel", id="cancel-urls", variant="default")
    
    @on(Button.Pressed, "#browse-files")
    async def handle_browse_files(self):
        """Open file browser."""
        try:
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
            
            files = await self.app.push_screen_wait(FileOpen(filters=self.file_filters))
            if files:
                self.add_files(files)
                logger.debug(f"[FileSelector] Added {len(files)} files")
        except Exception as e:
            logger.error(f"[FileSelector] Error browsing files: {e}")
            self.app.notify(f"Error selecting files: {e}", severity="error")
    
    @on(Button.Pressed, "#add-urls")
    def handle_show_url_input(self):
        """Show URL input section."""
        url_section = self.query_one("#url-section")
        url_section.remove_class("hidden")
        
        # Focus the textarea
        try:
            self.query_one("#url-textarea").focus()
        except:
            pass
    
    @on(Button.Pressed, "#confirm-urls")
    def handle_add_urls(self):
        """Add URLs from textarea."""
        try:
            textarea = self.query_one("#url-textarea")
            urls_text = textarea.value.strip()
            
            if urls_text:
                # Parse URLs (one per line)
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                self.add_urls(urls)
                
                # Clear and hide URL input
                textarea.value = ""
                self.query_one("#url-section").add_class("hidden")
                
                logger.debug(f"[FileSelector] Added {len(urls)} URLs")
            else:
                self.app.notify("Please enter at least one URL", severity="warning")
        except Exception as e:
            logger.error(f"[FileSelector] Error adding URLs: {e}")
            self.app.notify(f"Error adding URLs: {e}", severity="error")
    
    @on(Button.Pressed, "#cancel-urls")
    def handle_cancel_urls(self):
        """Cancel URL input."""
        textarea = self.query_one("#url-textarea")
        textarea.value = ""
        self.query_one("#url-section").add_class("hidden")
    
    @on(Button.Pressed, "#clear-all")
    def handle_clear_all(self):
        """Clear all selected files and URLs."""
        self.selected_files = []
        self.selected_urls = []
        self.update_display()
        self.notify_changes()
        logger.debug("[FileSelector] Cleared all files and URLs")
    
    def add_files(self, files: List[Path]):
        """Add files to the selection."""
        current_files = list(self.selected_files)
        # Avoid duplicates
        for file_path in files:
            if file_path not in current_files:
                current_files.append(file_path)
        
        self.selected_files = current_files
        self.update_display()
        self.notify_changes()
    
    def add_urls(self, urls: List[str]):
        """Add URLs to the selection."""
        current_urls = list(self.selected_urls)
        # Avoid duplicates
        for url in urls:
            if url not in current_urls:
                current_urls.append(url)
        
        self.selected_urls = current_urls
        self.update_display()
        self.notify_changes()
    
    def remove_file(self, index: int):
        """Remove a file by index."""
        if 0 <= index < len(self.selected_files):
            files = list(self.selected_files)
            files.pop(index)
            self.selected_files = files
            self.update_display()
            self.notify_changes()
    
    def remove_url(self, index: int):
        """Remove a URL by index."""
        if 0 <= index < len(self.selected_urls):
            urls = list(self.selected_urls)
            urls.pop(index)
            self.selected_urls = urls
            self.update_display()
            self.notify_changes()
    
    def update_display(self):
        """Update the file list display."""
        try:
            display = self.query_one("#file-display")
            display.remove_children()
            
            total_items = len(self.selected_files) + len(self.selected_urls)
            
            if total_items == 0:
                display.mount(Static("No files or URLs selected", classes="empty-message"))
            else:
                # Summary
                summary = f"Selected: {len(self.selected_files)} files, {len(self.selected_urls)} URLs ({total_items} total)"
                display.mount(Static(summary, classes="file-summary"))
                
                # File list (limit to prevent UI overflow)
                max_display = 8
                items_shown = 0
                
                # Show files
                for i, file_path in enumerate(self.selected_files[:max_display - items_shown]):
                    file_item = self.create_file_item(i, file_path, "file")
                    display.mount(file_item)
                    items_shown += 1
                
                # Show URLs
                remaining_slots = max_display - items_shown
                for i, url in enumerate(self.selected_urls[:remaining_slots]):
                    url_item = self.create_file_item(i, url, "url")
                    display.mount(url_item)
                    items_shown += 1
                
                if total_items > max_display:
                    display.mount(Static(f"... and {total_items - max_display} more items", classes="more-items"))
                    
        except Exception as e:
            logger.error(f"[FileSelector] Error updating display: {e}")
    
    def create_file_item(self, index: int, item, item_type: str) -> Container:
        """Create a file or URL list item with remove button."""
        with Container(classes="file-item-container"):
            with Horizontal(classes="file-item-row"):
                if item_type == "file":
                    icon = "📁"
                    name = item.name
                    size = f" ({item.stat().st_size // 1024} KB)" if item.exists() else ""
                    display_text = f"{icon} {name}{size}"
                else:  # URL
                    icon = "🔗"
                    display_text = f"{icon} {item[:60]}{'...' if len(item) > 60 else ''}"
                
                yield Static(display_text, classes="file-item-text")
                yield Button("✕", id=f"remove-{item_type}-{index}", classes="remove-button")
        
        return container
    
    @on(Button.Pressed)
    def handle_remove_button(self, event):
        """Handle remove button presses."""
        button_id = event.button.id
        if button_id and button_id.startswith("remove-"):
            try:
                parts = button_id.split("-")
                item_type = parts[1]  # "file" or "url"
                index = int(parts[2])
                
                if item_type == "file":
                    self.remove_file(index)
                elif item_type == "url":
                    self.remove_url(index)
                    
                logger.debug(f"[FileSelector] Removed {item_type} at index {index}")
            except (ValueError, IndexError) as e:
                logger.error(f"[FileSelector] Error removing item: {e}")
    
    def notify_changes(self):
        """Notify parent of file selection changes."""
        if self.on_files_changed:
            self.on_files_changed(self.selected_files, self.selected_urls)
    
    def get_all_items(self) -> Tuple[List[Path], List[str]]:
        """Get all selected files and URLs."""
        return list(self.selected_files), list(self.selected_urls)
    
    def has_items(self) -> bool:
        """Check if any files or URLs are selected."""
        return len(self.selected_files) > 0 or len(self.selected_urls) > 0
    
    # Watch reactive properties
    def watch_selected_files(self, files: List[Path]):
        """Update display when files change."""
        self.update_display()
    
    def watch_selected_urls(self, urls: List[str]):
        """Update display when URLs change."""
        self.update_display()
    
    # Default styling
    DEFAULT_CSS = """
    .file-selector-container {
        margin-bottom: 2;
    }
    
    .file-actions-row {
        height: 3;
        margin-bottom: 1;
        gap: 1;
    }
    
    .file-action-button {
        margin-right: 1;
    }
    
    .file-display-container {
        min-height: 5;
        max-height: 12;
        border: round $primary;
        background: $surface;
        padding: 1;
        margin-bottom: 1;
        overflow-y: auto;
    }
    
    .empty-message {
        color: $text-muted;
        text-align: center;
        padding: 2;
        text-style: italic;
    }
    
    .file-summary {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
        border-bottom: solid $primary;
        padding-bottom: 1;
    }
    
    .file-item-container {
        margin-bottom: 1;
        padding: 1;
        border: round $surface-lighten-1;
        background: $surface-lighten-1;
    }
    
    .file-item-container:hover {
        background: $surface-lighten-2;
        border: round $primary;
    }
    
    .file-item-row {
        height: auto;
        align: left middle;
    }
    
    .file-item-text {
        width: 1fr;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .remove-button {
        width: 3;
        height: 1;
        min-width: 3;
        background: $error;
        color: $text;
        text-align: center;
    }
    
    .remove-button:hover {
        background: $error-darken-1;
        text-style: bold;
    }
    
    .more-items {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    
    .url-input-container {
        margin-top: 1;
        padding: 1;
        border: round $primary;
        background: $surface-lighten-1;
    }
    
    .url-textarea {
        min-height: 5;
        max-height: 8;
        margin-bottom: 1;
    }
    
    .url-button-row {
        gap: 1;
    }
    
    .hidden {
        display: none;
    }
    """