# tldw_chatbook/Widgets/file_list_item_enhanced.py
# Enhanced file list item widget with metadata display

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Button, ListItem
from textual.widget import Widget
from textual.reactive import reactive
from loguru import logger

class FileListItemEnhanced(ListItem):
    """
    Enhanced list item for file display with metadata.
    
    Shows:
    - File name with appropriate icon
    - File size (human readable)
    - Last modified date
    - File type/extension
    - Remove button
    """
    
    def __init__(
        self, 
        file_path: Path,
        show_size: bool = True,
        show_date: bool = True,
        show_remove: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.show_size = show_size
        self.show_date = show_date
        self.show_remove = show_remove
        self._metadata = self._get_file_metadata()
    
    def _get_file_metadata(self) -> Dict[str, Any]:
        """Get file metadata."""
        metadata = {
            "name": self.file_path.name,
            "extension": self.file_path.suffix.lower(),
            "exists": self.file_path.exists(),
            "is_url": str(self.file_path).startswith(("http://", "https://")),
            "size": 0,
            "modified": None,
            "size_str": "Unknown",
            "date_str": "Unknown"
        }
        
        if metadata["exists"] and not metadata["is_url"]:
            try:
                stat = self.file_path.stat()
                metadata["size"] = stat.st_size
                metadata["modified"] = datetime.fromtimestamp(stat.st_mtime)
                metadata["size_str"] = self._format_file_size(stat.st_size)
                metadata["date_str"] = metadata["modified"].strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                logger.error(f"Error getting file metadata for {self.file_path}: {e}")
        elif metadata["is_url"]:
            metadata["size_str"] = "URL"
            metadata["date_str"] = "Remote"
        
        return metadata
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def _get_file_icon(self) -> str:
        """Get appropriate icon for file type."""
        if self._metadata["is_url"]:
            return "🌐"
        
        ext = self._metadata["extension"]
        
        # Video files
        if ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']:
            return "🎬"
        # Audio files
        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus']:
            return "🎵"
        # Document files
        elif ext in ['.doc', '.docx', '.odt', '.rtf']:
            return "📄"
        # PDF files
        elif ext == '.pdf':
            return "📕"
        # Ebook files
        elif ext in ['.epub', '.mobi', '.azw', '.azw3', '.fb2']:
            return "📚"
        # Text files
        elif ext in ['.txt', '.md', '.log']:
            return "📝"
        # Archive files
        elif ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
            return "📦"
        # Image files
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']:
            return "🖼️"
        # Unknown
        else:
            return "📎"
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced file list item."""
        with Horizontal(classes="file-list-item-container"):
            # File icon and name
            with Vertical(classes="file-info-main"):
                icon = self._get_file_icon()
                with Horizontal(classes="file-name-row"):
                    yield Static(icon, classes="file-icon")
                    yield Static(
                        self._metadata["name"], 
                        classes="file-name",
                        tooltip=str(self.file_path)
                    )
                
                # Metadata row
                metadata_parts = []
                if self.show_size:
                    metadata_parts.append(self._metadata["size_str"])
                if self.show_date and self._metadata["date_str"] != "Unknown":
                    metadata_parts.append(self._metadata["date_str"])
                if self._metadata["extension"]:
                    metadata_parts.append(self._metadata["extension"][1:].upper())
                
                if metadata_parts:
                    yield Static(
                        " • ".join(metadata_parts),
                        classes="file-metadata"
                    )
            
            # Remove button
            if self.show_remove:
                yield Button(
                    "✕",
                    id=f"remove-{hash(str(self.file_path))}",
                    classes="file-remove-button",
                    variant="error",
                    tooltip="Remove this file"
                )
    
    @property
    def path(self) -> Path:
        """Get the file path."""
        return self.file_path
    
    def refresh_metadata(self) -> None:
        """Refresh file metadata and update display."""
        self._metadata = self._get_file_metadata()
        self.refresh()


class FileListEnhanced(Widget):
    """
    Enhanced file list widget that uses FileListItemEnhanced.
    
    Features:
    - Displays files with metadata
    - Supports adding/removing files
    - Shows total size and file count
    - Keyboard navigation
    """
    
    files = reactive([], recompose=True)
    
    def __init__(
        self,
        files: Optional[list[Path]] = None,
        show_summary: bool = True,
        max_height: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.show_summary = show_summary
        self.max_height = max_height
        if files:
            self.files = files
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced file list."""
        with Vertical(classes="file-list-enhanced"):
            # File items
            if self.files:
                with Vertical(
                    classes="file-items-container",
                    id="file-items"
                ):
                    for file_path in self.files:
                        yield FileListItemEnhanced(
                            file_path,
                            id=f"file-item-{hash(str(file_path))}"
                        )
            else:
                yield Static(
                    "No files selected",
                    classes="no-files-message"
                )
            
            # Summary footer
            if self.show_summary and self.files:
                yield Static(
                    self._get_summary_text(),
                    id="file-list-summary",
                    classes="file-list-summary"
                )
    
    def _get_summary_text(self) -> str:
        """Get summary text for the file list."""
        total_size = 0
        file_count = len(self.files)
        
        for file_path in self.files:
            if file_path.exists() and not str(file_path).startswith(("http://", "https://")):
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
        
        size_str = self._format_file_size(total_size) if total_size > 0 else "0 B"
        
        if file_count == 1:
            return f"1 file • {size_str}"
        else:
            return f"{file_count} files • {size_str} total"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def add_file(self, file_path: Path) -> None:
        """Add a file to the list."""
        if file_path not in self.files:
            self.files = self.files + [file_path]
    
    def remove_file(self, file_path: Path) -> None:
        """Remove a file from the list."""
        self.files = [f for f in self.files if f != file_path]
    
    def clear(self) -> None:
        """Clear all files."""
        self.files = []
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle remove button clicks."""
        if event.button.id and event.button.id.startswith("remove-"):
            # Find the file item that contains this button
            file_item = event.button.parent.parent
            if isinstance(file_item, FileListItemEnhanced):
                self.remove_file(file_item.path)

# End of file_list_item_enhanced.py