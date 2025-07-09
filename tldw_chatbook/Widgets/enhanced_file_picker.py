# enhanced_file_picker.py
# Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, and search

from pathlib import Path
from typing import List, Optional, Callable, Set, Dict, Any, Tuple
from datetime import datetime
import json
import os
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, ListView, ListItem, Input, Select
from textual.reactive import reactive
from textual.worker import Worker, get_current_worker
from loguru import logger

from ..Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from ..Third_Party.textual_fspicker.file_dialog import BaseFileDialog
from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
from ..config import get_cli_setting, save_setting_to_cli_config


class RecentLocations:
    """Manages recently accessed file locations with persistent storage"""
    
    def __init__(self, max_items: int = 20, context: str = "default"):
        self.max_items = max_items
        self.context = context  # Context for different file picker uses
        self._recent: List[Dict[str, Any]] = []
        self.load_from_config()
    
    def load_from_config(self):
        """Load recent locations from config"""
        try:
            recent_data = get_cli_setting("filepicker", f"recent_{self.context}", [])
            if isinstance(recent_data, list):
                self._recent = recent_data[:self.max_items]
            else:
                self._recent = []
        except Exception as e:
            logger.error(f"Failed to load recent locations: {e}")
            self._recent = []
    
    def save_to_config(self):
        """Save recent locations to config"""
        try:
            save_setting_to_cli_config("filepicker", f"recent_{self.context}", self._recent)
        except Exception as e:
            logger.error(f"Failed to save recent locations: {e}")
    
    def add(self, path: Path, file_type: str = "file"):
        """Add a path to recent locations"""
        path_str = str(path.resolve())
        
        # Remove if already exists
        self._recent = [item for item in self._recent if item.get("path") != path_str]
        
        # Add to front
        self._recent.insert(0, {
            "path": path_str,
            "name": path.name,
            "type": file_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim to max
        self._recent = self._recent[:self.max_items]
        self.save_to_config()
    
    def get_recent(self) -> List[Dict[str, Any]]:
        """Get recent locations"""
        return self._recent
    
    def clear(self):
        """Clear all recent locations"""
        self._recent = []
        self.save_to_config()


class BookmarksManager:
    """Manages bookmarked directories for quick access"""
    
    def __init__(self, context: str = "default"):
        self.context = context
        self._bookmarks: List[Dict[str, Any]] = []
        self._default_bookmarks = self._get_default_bookmarks()
        self.load_from_config()
    
    def _get_default_bookmarks(self) -> List[Dict[str, Any]]:
        """Get platform-specific default bookmarks"""
        home = Path.home()
        bookmarks = [
            {"name": "Home", "path": str(home), "icon": "🏠"},
            {"name": "Desktop", "path": str(home / "Desktop"), "icon": "🖥️"},
            {"name": "Documents", "path": str(home / "Documents"), "icon": "📄"},
            {"name": "Downloads", "path": str(home / "Downloads"), "icon": "⬇️"},
        ]
        
        # Add platform-specific paths
        if os.name == 'posix':  # Unix/Linux/Mac
            if (home / "Pictures").exists():
                bookmarks.append({"name": "Pictures", "path": str(home / "Pictures"), "icon": "🖼️"})
        
        # Filter out non-existent directories
        return [b for b in bookmarks if Path(b["path"]).exists()]
    
    def load_from_config(self):
        """Load bookmarks from config"""
        try:
            saved_bookmarks = get_cli_setting("filepicker", f"bookmarks_{self.context}", None)
            if saved_bookmarks is None:
                # First time - use defaults
                self._bookmarks = self._default_bookmarks.copy()
                self.save_to_config()
            elif isinstance(saved_bookmarks, list):
                self._bookmarks = saved_bookmarks
            else:
                self._bookmarks = self._default_bookmarks.copy()
        except Exception as e:
            logger.error(f"Failed to load bookmarks: {e}")
            self._bookmarks = self._default_bookmarks.copy()
    
    def save_to_config(self):
        """Save bookmarks to config"""
        try:
            save_setting_to_cli_config("filepicker", f"bookmarks_{self.context}", self._bookmarks)
        except Exception as e:
            logger.error(f"Failed to save bookmarks: {e}")
    
    def add(self, path: Path, name: Optional[str] = None, icon: str = "📁"):
        """Add a bookmark"""
        path_str = str(path.resolve())
        
        # Check if already bookmarked
        if any(b.get("path") == path_str for b in self._bookmarks):
            return False
        
        bookmark = {
            "name": name or path.name,
            "path": path_str,
            "icon": icon,
            "custom": True  # Mark as user-added
        }
        
        self._bookmarks.append(bookmark)
        self.save_to_config()
        return True
    
    def remove(self, path: Path):
        """Remove a bookmark"""
        path_str = str(path.resolve())
        self._bookmarks = [b for b in self._bookmarks if b.get("path") != path_str]
        self.save_to_config()
    
    def is_bookmarked(self, path: Path) -> bool:
        """Check if a path is bookmarked"""
        path_str = str(path.resolve())
        return any(b.get("path") == path_str for b in self._bookmarks)
    
    def get_bookmarks(self) -> List[Dict[str, Any]]:
        """Get all bookmarks"""
        return self._bookmarks.copy()
    
    def reset_to_defaults(self):
        """Reset bookmarks to defaults"""
        self._bookmarks = self._default_bookmarks.copy()
        self.save_to_config()


class PathBreadcrumbs(Horizontal):
    """Clickable breadcrumb navigation for paths"""
    
    DEFAULT_CSS = """
    PathBreadcrumbs {
        height: 3;
        padding: 1;
        background: $surface;
        overflow: hidden;
    }
    
    PathBreadcrumbs Button {
        min-width: 0;
        padding: 0 1;
        margin: 0;
        height: 1;
    }
    
    PathBreadcrumbs .breadcrumb-separator {
        margin: 0 1;
        color: $text-muted;
    }
    """
    
    from textual.message import Message
    
    class PathChanged(Message):
        """Emitted when a breadcrumb is clicked"""
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()
    
    def __init__(self, initial_path: Optional[Path] = None):
        super().__init__()
        self.current_path = initial_path or Path.cwd()
    
    def update_path(self, path: Path):
        """Update the breadcrumb display with a new path"""
        self.current_path = path
        self.refresh_breadcrumbs()
    
    @work(exclusive=True)
    async def refresh_breadcrumbs(self):
        """Refresh the breadcrumb display"""
        await self.remove_children()
        
        parts = self.current_path.parts
        for i, part in enumerate(parts):
            partial_path = Path(*parts[:i+1])
            
            # Create button for each part
            btn = Button(part, variant="subtle")
            btn.data = partial_path  # Store the path in the button
            await self.mount(btn)
            
            # Add separator if not last
            if i < len(parts) - 1:
                await self.mount(Label("/", classes="breadcrumb-separator"))
    
    @on(Button.Pressed)
    def handle_breadcrumb_click(self, event: Button.Pressed):
        """Handle clicks on breadcrumb buttons"""
        if hasattr(event.button, 'data'):
            self.post_message(self.PathChanged(event.button.data))


class DirectorySearch(Horizontal):
    """Search widget for filtering directory contents"""
    
    DEFAULT_CSS = """
    DirectorySearch {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    
    DirectorySearch Input {
        width: 1fr;
    }
    
    DirectorySearch Button {
        margin-left: 1;
    }
    """
    
    from textual.message import Message
    
    class SearchChanged(Message):
        """Emitted when search text changes"""
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search files...", id="search-input")
        yield Button("Clear", id="clear-search", variant="subtle")
    
    @on(Input.Changed, "#search-input")
    def handle_search_change(self, event: Input.Changed):
        """Handle search input changes"""
        self.post_message(self.SearchChanged(event.value))
    
    @on(Button.Pressed, "#clear-search")
    def handle_clear(self):
        """Clear the search"""
        search_input = self.query_one("#search-input", Input)
        search_input.value = ""
        self.post_message(self.SearchChanged(""))


class EnhancedFileDialog(BaseFileDialog):
    """Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, and search"""
    
    DEFAULT_CSS = BaseFileDialog.DEFAULT_CSS + """
    BaseFileDialog {
        Dialog {
            height: 90%;
            width: 90%;
        }
        
        #recent-locations, #bookmarks-panel {
            height: 10;
            border: solid $primary;
            background: $surface;
            margin-bottom: 1;
            display: none;  /* Hidden by default */
        }
        
        #recent-locations.visible, #bookmarks-panel.visible {
            display: block;
        }
        
        #recent-list, #bookmarks-list {
            height: 8;
            background: $surface;
        }
        
        .recent-item, .bookmark-item {
            padding: 0 1;
        }
        
        .bookmark-item-icon {
            margin-right: 1;
        }
        
        .bookmarks-header {
            height: 2;
            padding: 0 1;
        }
        
        .bookmark-button {
            min-width: 3;
            height: 1;
            margin: 0;
        }
        
        .search-active {
            border-title-style: bold;
            border-title-color: $warning;
        }
        
        .section-title {
            width: 1fr;
            text-style: bold;
        }
    }
    """
    
    BINDINGS = [
        Binding("ctrl+h", "toggle_hidden", "Toggle hidden files"),
        Binding("ctrl+l", "focus_path_input", "Edit path directly"),
        Binding("ctrl+r", "toggle_recent", "Show recent files"),
        Binding("ctrl+b", "toggle_bookmarks", "Show bookmarks"),
        Binding("ctrl+d", "bookmark_current", "Bookmark current directory"),
        Binding("f5", "refresh", "Refresh directory"),
        Binding("ctrl+f", "focus_search", "Search files"),
        Binding("ctrl+1", "quick_access", "Quick access", key_display="1-9"),
        Binding("escape", "dismiss(None)", "Cancel"),
    ]
    
    show_recent = reactive(False)
    show_bookmarks = reactive(False)
    search_query = reactive("")
    
    def __init__(self, *args, context: str = "default", **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context
        self.recent_locations = RecentLocations(context=context)
        self.bookmarks_manager = BookmarksManager(context=context)
        self._original_entries = []  # Store original entries for search filtering
        self._last_directory = self._get_last_directory()
    
    def _get_last_directory(self) -> Optional[Path]:
        """Get the last used directory for this context"""
        try:
            last_dir = get_cli_setting("filepicker", f"last_dir_{self.context}", None)
            if last_dir and Path(last_dir).exists():
                return Path(last_dir)
        except Exception:
            pass
        return None
    
    def _save_last_directory(self, path: Path):
        """Save the last used directory for this context"""
        try:
            dir_path = path if path.is_dir() else path.parent
            save_setting_to_cli_config("filepicker", f"last_dir_{self.context}", str(dir_path))
        except Exception as e:
            logger.error(f"Failed to save last directory: {e}")
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced file picker UI"""
        from ..Third_Party.textual_fspicker.parts import DriveNavigation
        from ..Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
        import sys
        
        # Use last directory if available, otherwise use provided location
        initial_location = self._last_directory or Path(self._location)
        
        with Dialog() as dialog:
            dialog.border_title = self._title
            
            # Recent locations panel (hidden by default)
            with Container(id="recent-locations"):
                yield Label("📋 Recent Locations", classes="section-title")
                yield ListView(id="recent-list")
            
            # Bookmarks panel (hidden by default)
            with Container(id="bookmarks-panel"):
                with Horizontal(classes="bookmarks-header"):
                    yield Label("⭐ Bookmarks", classes="section-title")
                    yield Button("➕", id="add-bookmark", classes="bookmark-button")
                yield ListView(id="bookmarks-list")
            
            # Breadcrumb navigation
            yield PathBreadcrumbs(initial_location)
            
            # Search bar
            yield DirectorySearch()
            
            # Main file browser
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(str(initial_location))
                yield DirectoryNavigation(str(initial_location))
            
            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")
    
    def on_mount(self) -> None:
        """Initialize the dialog on mount"""
        super().on_mount()
        self._update_recent_list()
        self._update_bookmarks_list()
        
        # Update breadcrumbs with initial location
        dir_nav = self.query_one(DirectoryNavigation)
        breadcrumbs = self.query_one(PathBreadcrumbs)
        breadcrumbs.update_path(dir_nav.location)
        
        # Update bookmark button state
        self._update_bookmark_button_state(dir_nav.location)
    
    def watch_show_recent(self, show: bool) -> None:
        """Toggle recent locations visibility"""
        try:
            recent_panel = self.query_one("#recent-locations")
            recent_panel.set_class(show, "visible")
            # Hide bookmarks if showing recent
            if show:
                self.show_bookmarks = False
        except Exception:
            pass
    
    def watch_show_bookmarks(self, show: bool) -> None:
        """Toggle bookmarks panel visibility"""
        try:
            bookmarks_panel = self.query_one("#bookmarks-panel")
            bookmarks_panel.set_class(show, "visible")
            # Hide recent if showing bookmarks
            if show:
                self.show_recent = False
        except Exception:
            pass
    
    def watch_search_query(self, query: str) -> None:
        """Filter directory entries based on search query"""
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            from ..Third_Party.textual_fspicker.base_dialog import Dialog
            dialog = self.query_one(Dialog)
            
            if query:
                dialog.add_class("search-active")
                # Implement filtering logic here
                # This would require modifying DirectoryNavigation or creating a wrapper
            else:
                dialog.remove_class("search-active")
        except Exception as e:
            logger.error(f"Error filtering entries: {e}")
    
    def _update_recent_list(self):
        """Update the recent locations list"""
        try:
            recent_list = self.query_one("#recent-list", ListView)
            recent_list.clear()
            
            for item in self.recent_locations.get_recent():
                path = item["path"]
                name = item["name"]
                file_type = item.get("type", "file")
                icon = "📁" if file_type == "directory" else "📄"
                list_item = ListItem(Label(f"{icon} {name} - {path}", classes="recent-item"))
                list_item.data = path
                recent_list.append(list_item)
        except Exception as e:
            logger.error(f"Error updating recent list: {e}")
    
    def _update_bookmarks_list(self):
        """Update the bookmarks list"""
        try:
            bookmarks_list = self.query_one("#bookmarks-list", ListView)
            bookmarks_list.clear()
            
            for bookmark in self.bookmarks_manager.get_bookmarks():
                path = bookmark["path"]
                name = bookmark["name"]
                icon = bookmark.get("icon", "📁")
                list_item = ListItem(
                    Horizontal(
                        Label(icon, classes="bookmark-item-icon"),
                        Label(name, classes="bookmark-item")
                    )
                )
                list_item.data = path
                bookmarks_list.append(list_item)
        except Exception as e:
            logger.error(f"Error updating bookmarks list: {e}")
    
    def _update_bookmark_button_state(self, path: Path):
        """Update the bookmark button based on current directory"""
        try:
            btn = self.query_one("#add-bookmark", Button)
            if self.bookmarks_manager.is_bookmarked(path):
                btn.label = "⭐"  # Already bookmarked
                btn.tooltip = "Remove bookmark"
            else:
                btn.label = "➕"
                btn.tooltip = "Add bookmark"
        except Exception:
            pass
    
    def action_toggle_hidden(self) -> None:
        """Toggle showing hidden files"""
        self.query_one(DirectoryNavigation).toggle_hidden()
        self.notify("Hidden files toggled", timeout=2)
    
    def action_focus_path_input(self) -> None:
        """Focus the path input field"""
        try:
            from textual.widgets import Input
            input_widget = self.query_one(Input)
            input_widget.focus()
            input_widget.action_select_all()
        except Exception:
            pass
    
    def action_toggle_recent(self) -> None:
        """Toggle recent locations panel"""
        self.show_recent = not self.show_recent
    
    def action_toggle_bookmarks(self) -> None:
        """Toggle bookmarks panel"""
        self.show_bookmarks = not self.show_bookmarks
    
    def action_bookmark_current(self) -> None:
        """Add or remove current directory from bookmarks"""
        dir_nav = self.query_one(DirectoryNavigation)
        current_path = dir_nav.location
        
        if self.bookmarks_manager.is_bookmarked(current_path):
            self.bookmarks_manager.remove(current_path)
            self.notify(f"Removed bookmark: {current_path.name}", timeout=2)
        else:
            self.bookmarks_manager.add(current_path)
            self.notify(f"Added bookmark: {current_path.name}", timeout=2)
        
        self._update_bookmarks_list()
        self._update_bookmark_button_state(current_path)
    
    def action_refresh(self) -> None:
        """Refresh the current directory"""
        dir_nav = self.query_one(DirectoryNavigation)
        # Trigger a refresh by resetting the location
        current = dir_nav.location
        dir_nav.location = current
        self.notify("Directory refreshed", timeout=2)
    
    def action_focus_search(self) -> None:
        """Focus the search input"""
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.focus()
        except Exception:
            pass
    
    def action_quick_access(self) -> None:
        """Quick access to bookmarks via number keys"""
        # This is handled by key bindings 1-9
        pass
    
    def on_key(self, event):
        """Handle key presses for quick bookmark access"""
        if event.key in "123456789" and not event.ctrl:
            index = int(event.key) - 1
            bookmarks = self.bookmarks_manager.get_bookmarks()
            
            if 0 <= index < len(bookmarks):
                path = Path(bookmarks[index]["path"])
                if path.exists():
                    dir_nav = self.query_one(DirectoryNavigation)
                    dir_nav.location = path
                    self._update_bookmark_button_state(path)
                    self.notify(f"Jumped to: {bookmarks[index]['name']}", timeout=1)
                else:
                    self.notify(f"Path no longer exists: {path}", severity="warning")
                
                event.prevent_default()
                event.stop()
    
    @on(DirectoryNavigation.Changed)
    def handle_directory_changed(self, event: DirectoryNavigation.Changed):
        """Handle directory changes to update UI"""
        try:
            # Get the new location from the navigation widget
            dir_nav = event.navigation
            new_path = dir_nav.location
            # Update breadcrumbs
            breadcrumbs = self.query_one(PathBreadcrumbs)
            breadcrumbs.update_path(new_path)
            # Update bookmark button
            self._update_bookmark_button_state(new_path)
        except Exception as e:
            logger.debug(f"Error handling directory change: {e}")
    
    @on(ListView.Selected, "#recent-list")
    def handle_recent_selection(self, event: ListView.Selected):
        """Handle selection from recent list"""
        if hasattr(event.item, 'data'):
            path = Path(event.item.data)
            if path.exists():
                dir_nav = self.query_one(DirectoryNavigation)
                
                if path.is_dir():
                    dir_nav.location = path
                else:
                    dir_nav.location = path.parent
                    # TODO: Select the file in the list
                
                self.show_recent = False
                self._update_bookmark_button_state(dir_nav.location)
    
    @on(ListView.Selected, "#bookmarks-list")
    def handle_bookmark_selection(self, event: ListView.Selected):
        """Handle selection from bookmarks list"""
        if hasattr(event.item, 'data'):
            path = Path(event.item.data)
            if path.exists():
                dir_nav = self.query_one(DirectoryNavigation)
                dir_nav.location = path
                self.show_bookmarks = False
                self._update_bookmark_button_state(path)
            else:
                self.notify(f"Path no longer exists: {path}", severity="warning")
    
    @on(Button.Pressed, "#add-bookmark")
    def handle_bookmark_button(self, event: Button.Pressed):
        """Handle bookmark button press"""
        self.action_bookmark_current()
    
    @on(PathBreadcrumbs.PathChanged)
    def handle_breadcrumb_navigation(self, event: PathBreadcrumbs.PathChanged):
        """Handle breadcrumb navigation"""
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.location = event.path
        self._update_bookmark_button_state(event.path)
    
    @on(DirectorySearch.SearchChanged)
    def handle_search_change(self, event: DirectorySearch.SearchChanged):
        """Handle search query changes"""
        self.search_query = event.query
    
    def dismiss(self, result: Optional[Path]) -> None:
        """Override dismiss to save recent location and last directory"""
        if result:
            # Save to recent locations
            self.recent_locations.add(result, "file" if result.is_file() else "directory")
            # Save last directory
            self._save_last_directory(result)
        
        # Always save current directory even if cancelled
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            self._save_last_directory(dir_nav.location)
        except Exception:
            pass
        
        super().dismiss(result)


class EnhancedFileOpen(EnhancedFileDialog):
    """Enhanced file open dialog with bookmarks and recent files"""
    
    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Open File",
        *,
        filters: Filters | None = None,
        must_exist: bool = True,
        context: str = "file_open",
        select_button: str = "Open",
        cancel_button: str = "Cancel",
        **kwargs
    ):
        super().__init__(
            location=location,
            title=title,
            select_button=select_button,
            cancel_button=cancel_button,
            filters=filters,
            context=context,
            **kwargs
        )
        self.filters = filters
        self.must_exist = must_exist
    
    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file selection"""
        from textual.widgets import Input, Select
        
        yield Input(placeholder="File name...")
        if self.filters:
            yield Select(
                [(f.name, i) for i, f in enumerate(self.filters.filters)],
                prompt="File type",
                value=0,
                id="file-filter"
            )


class EnhancedFileSave(EnhancedFileDialog):
    """Enhanced file save dialog with bookmarks and recent files"""
    
    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Save File",
        *,
        filters: Filters | None = None,
        default_filename: str = "",
        context: str = "file_save",
        select_button: str = "Save",
        cancel_button: str = "Cancel",
        **kwargs
    ):
        super().__init__(
            location=location,
            title=title,
            select_button=select_button,
            cancel_button=cancel_button,
            filters=filters,
            default_file=default_filename,
            context=context,
            **kwargs
        )
        self.filters = filters
        self.default_filename = default_filename
    
    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file saving"""
        from textual.widgets import Input, Select
        
        yield Input(value=self.default_filename, placeholder="File name...")
        if self.filters:
            yield Select(
                [(f.name, i) for i, f in enumerate(self.filters.filters)],
                prompt="File type",
                value=0,
                id="file-filter"
            )