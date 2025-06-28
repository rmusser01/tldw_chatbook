# enhanced_file_picker.py
# Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, and search

from pathlib import Path
from typing import List, Optional, Callable, Set, Dict, Any
from datetime import datetime
import json
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.message import Message
from textual.widgets import Button, Label, Static, ListView, ListItem, Input, Select
from textual.reactive import reactive
from textual.worker import Worker, get_current_worker
from loguru import logger

from ..Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from ..Third_Party.textual_fspicker.file_dialog import BaseFileDialog
from ..Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
from ..config import get_cli_setting


class RecentLocations:
    """Manages recently accessed file locations"""
    
    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self._recent: List[Dict[str, Any]] = []
        self.load_from_config()
    
    def load_from_config(self):
        """Load recent locations from config"""
        try:
            recent_data = get_cli_setting("general", "filepicker_recent_locations", [])
            self._recent = recent_data[:self.max_items]
        except Exception as e:
            logger.error(f"Failed to load recent locations: {e}")
            self._recent = []
    
    def save_to_config(self):
        """Save recent locations to config"""
        # Note: Currently disabled as set_cli_setting is not available
        # Would need to implement proper config persistence
        pass
    
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


class FilePickerToolbar(Horizontal):
    """Toolbar widget for file picker controls"""
    
    DEFAULT_CSS = """
    FilePickerToolbar {
        height: 3;
        padding: 0 1;
        background: $surface;
        align: center middle;
    }
    
    FilePickerToolbar Input {
        width: 30;
        margin-right: 1;
    }
    
    FilePickerToolbar Button {
        margin: 0 1;
        min-width: 0;
    }
    
    FilePickerToolbar Select {
        width: 20;
        margin: 0 1;
    }
    
    #hidden-files-btn.active {
        background: $primary;
    }
    """
    
    class SearchChanged(Message):
        """Emitted when search text changes"""
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()
    
    class SortChanged(Message):
        """Emitted when sort order changes"""
        def __init__(self, sort_by: str) -> None:
            self.sort_by = sort_by
            super().__init__()
    
    class HiddenToggled(Message):
        """Emitted when hidden files toggle changes"""
        def __init__(self, show_hidden: bool) -> None:
            self.show_hidden = show_hidden
            super().__init__()
    
    def __init__(self, show_hidden: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.show_hidden = show_hidden
    
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search files...", id="search-input")
        yield Button("Clear", id="clear-search", variant="default")
        yield Button("Hidden Files", id="hidden-files-btn", variant="primary" if self.show_hidden else "default")
        yield Label("Sort by:")
        yield Select(
            [("Name", "name"), ("Size", "size"), ("Type", "type"), ("Date", "time")],
            prompt="Sort by",
            value="name",
            id="sort-select"
        )
    
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
    
    @on(Button.Pressed, "#hidden-files-btn")
    def handle_hidden_toggle(self):
        """Toggle hidden files display"""
        self.show_hidden = not self.show_hidden
        btn = self.query_one("#hidden-files-btn", Button)
        btn.variant = "primary" if self.show_hidden else "default"
        self.post_message(self.HiddenToggled(self.show_hidden))
    
    @on(Select.Changed, "#sort-select")
    def handle_sort_change(self, event: Select.Changed):
        """Handle sort order changes"""
        if event.value:
            self.post_message(self.SortChanged(str(event.value)))


class EnhancedFileDialog(BaseFileDialog):
    """Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, and search"""
    
    DEFAULT_CSS = BaseFileDialog.DEFAULT_CSS + """
    BaseFileDialog {
        Dialog {
            height: 90%;
            width: 90%;
        }
        
        #recent-locations {
            height: 8;
            border: solid $primary;
            background: $surface;
            margin-bottom: 1;
            display: none;  /* Hidden by default */
        }
        
        #recent-locations.visible {
            display: block;
        }
        
        #recent-list {
            height: 6;
            background: $surface;
        }
        
        .recent-item {
            padding: 0 1;
        }
        
        .search-active {
            border-title-style: bold;
            border-title-color: $warning;
        }
    }
    """
    
    BINDINGS = [
        Binding("ctrl+h", "toggle_hidden", "Toggle hidden files"),
        Binding("ctrl+l", "focus_path_input", "Edit path directly"),
        Binding("ctrl+r", "toggle_recent", "Show recent files"),
        Binding("f5", "refresh", "Refresh directory"),
        Binding("ctrl+f", "focus_search", "Search files"),
        Binding("escape", "dismiss(None)", "Cancel"),
    ]
    
    show_recent = reactive(False)
    search_query = reactive("")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_locations = RecentLocations()
        self._original_entries = []  # Store original entries for search filtering
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced file picker UI"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation, DriveNavigation
        import sys
        
        with Dialog() as dialog:
            dialog.border_title = self._title
            
            # Recent locations panel (hidden by default)
            with Container(id="recent-locations"):
                yield Label("Recent Locations", classes="section-title")
                yield ListView(id="recent-list")
            
            # Breadcrumb navigation
            yield PathBreadcrumbs(Path(self._location))
            
            # Toolbar with search and controls
            yield FilePickerToolbar()
            
            # Main file browser
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(self._location)
                yield DirectoryNavigation(self._location)
            
            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")
    
    def on_mount(self) -> None:
        """Initialize the dialog on mount"""
        super().on_mount()
        self._update_recent_list()
        
        # Update breadcrumbs with initial location
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        breadcrumbs = self.query_one(PathBreadcrumbs)
        breadcrumbs.update_path(dir_nav.location)
        
        # Initialize toolbar state
        toolbar = self.query_one(FilePickerToolbar)
        toolbar.show_hidden = dir_nav.show_hidden
    
    def watch_show_recent(self, show: bool) -> None:
        """Toggle recent locations visibility"""
        try:
            recent_panel = self.query_one("#recent-locations")
            recent_panel.set_class(show, "visible")
        except Exception:
            pass
    
    def watch_search_query(self, query: str) -> None:
        """Filter directory entries based on search query"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        from ..Third_Party.textual_fspicker.base_dialog import Dialog
        
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            dialog = self.query_one(Dialog)
            
            if query:
                dialog.add_class("search-active")
                dir_nav.search_filter = query
            else:
                dialog.remove_class("search-active")
                dir_nav.search_filter = ""
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
                list_item = ListItem(Label(f"{name} - {path}", classes="recent-item"))
                list_item.data = path
                recent_list.append(list_item)
        except Exception as e:
            logger.error(f"Error updating recent list: {e}")
    
    def action_toggle_hidden(self) -> None:
        """Toggle showing hidden files"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.toggle_hidden()
        # Update the toolbar button state
        toolbar = self.query_one(FilePickerToolbar)
        toolbar.show_hidden = dir_nav.show_hidden
        btn = toolbar.query_one("#hidden-files-btn", Button)
        btn.variant = "primary" if dir_nav.show_hidden else "default"
        self.notify(f"Hidden files {'shown' if dir_nav.show_hidden else 'hidden'}", timeout=2)
    
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
    
    def action_refresh(self) -> None:
        """Refresh the current directory"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
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
    
    @on(ListView.Selected, "#recent-list")
    def handle_recent_selection(self, event: ListView.Selected):
        """Handle selection from recent list"""
        if hasattr(event.item, 'data'):
            path = Path(event.item.data)
            if path.exists():
                from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
                dir_nav = self.query_one(DirectoryNavigation)
                
                if path.is_dir():
                    dir_nav.location = path
                else:
                    dir_nav.location = path.parent
                    # TODO: Select the file in the list
                
                self.show_recent = False
    
    @on(PathBreadcrumbs.PathChanged)
    def handle_breadcrumb_navigation(self, event: PathBreadcrumbs.PathChanged):
        """Handle breadcrumb navigation"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.location = event.path
    
    @on(FilePickerToolbar.SearchChanged)
    def handle_search_change(self, event: FilePickerToolbar.SearchChanged):
        """Handle search query changes"""
        self.search_query = event.query
    
    @on(FilePickerToolbar.SortChanged)
    def handle_sort_change(self, event: FilePickerToolbar.SortChanged):
        """Handle sort order changes"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.sort_by = event.sort_by
        self.notify(f"Sorting by {event.sort_by}", timeout=2)
    
    @on(FilePickerToolbar.HiddenToggled)
    def handle_hidden_toggle(self, event: FilePickerToolbar.HiddenToggled):
        """Handle hidden files toggle from toolbar"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.show_hidden = event.show_hidden
        self.notify(f"Hidden files {'shown' if event.show_hidden else 'hidden'}", timeout=2)
    
    def on_mount(self) -> None:
        """Initialize on mount"""
        # Don't call parent's on_mount as it expects widgets we don't have
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        # Initialize toolbar state
        toolbar = self.query_one(FilePickerToolbar)
        dir_nav = self.query_one(DirectoryNavigation)
        toolbar.show_hidden = dir_nav.show_hidden
        
        # Focus the directory navigation
        dir_nav.focus()
    
    def _update_breadcrumbs(self, path: Path):
        """Stub for parent's breadcrumb update"""
        pass
    
    def _load_recent_locations(self):
        """Stub for parent's recent locations"""
        pass
    
    def dismiss(self, result: Optional[Path]) -> None:
        """Override dismiss to save recent location"""
        if result:
            self.recent_locations.add(result, "file" if result.is_file() else "directory")
        super().dismiss(result)


class EnhancedFileOpen(FileOpen):
    """Enhanced file open dialog"""
    
    DEFAULT_CSS = FileOpen.DEFAULT_CSS + """
    #file-picker-toolbar {
        height: 3;
        margin-bottom: 1;
        background: $surface;
    }
    """
    
    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Open File",
        *,
        filters: Filters | None = None,
        must_exist: bool = True,
        **kwargs
    ):
        super().__init__(location, title, **kwargs)
        self.filters = filters
        self.must_exist = must_exist
    
    def compose(self) -> ComposeResult:
        """Override compose to add our toolbar"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation, DriveNavigation
        import sys
        
        with Dialog() as dialog:
            dialog.border_title = self._title
            
            # Add the path display that parent expects
            yield Label("", id="current_path_display", classes="hidden")
            
            # Add our custom toolbar
            yield FilePickerToolbar(id="file-picker-toolbar")
            
            # Main file browser
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(self._location)
                yield DirectoryNavigation(self._location)
            
            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")
    
    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file selection"""
        from textual.widgets import Input, Select
        
        yield Input(placeholder="File name...")
        if self.filters:
            yield Select(
                self.filters.selections,
                prompt="File type",
                value=0,
                id="file-filter"
            )
    
    def watch_search_query(self, query: str) -> None:
        """Filter directory entries based on search query"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            dir_nav.search_filter = query
        except Exception as e:
            logger.error(f"Error filtering entries: {e}")
    
    @on(FilePickerToolbar.SearchChanged)
    def handle_search_change(self, event: FilePickerToolbar.SearchChanged):
        """Handle search query changes"""
        self.search_query = event.query
    
    @on(FilePickerToolbar.SortChanged)
    def handle_sort_change(self, event: FilePickerToolbar.SortChanged):
        """Handle sort order changes"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.sort_by = event.sort_by
        self.notify(f"Sorting by {event.sort_by}", timeout=2)
    
    @on(FilePickerToolbar.HiddenToggled)
    def handle_hidden_toggle(self, event: FilePickerToolbar.HiddenToggled):
        """Handle hidden files toggle from toolbar"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.show_hidden = event.show_hidden
        self.notify(f"Hidden files {'shown' if event.show_hidden else 'hidden'}", timeout=2)
    
    def on_mount(self) -> None:
        """Initialize on mount"""
        # Don't call parent's on_mount as it expects widgets we don't have
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        # Initialize toolbar state
        toolbar = self.query_one(FilePickerToolbar)
        dir_nav = self.query_one(DirectoryNavigation)
        toolbar.show_hidden = dir_nav.show_hidden
        
        # Focus the directory navigation
        dir_nav.focus()
    
    def _update_breadcrumbs(self, path: Path):
        """Stub for parent's breadcrumb update"""
        pass
    
    def _load_recent_locations(self):
        """Stub for parent's recent locations"""
        pass
    
    def on_mount(self) -> None:
        """Initialize on mount"""
        # Don't call parent's on_mount as it expects widgets we don't have
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        # Initialize toolbar state
        toolbar = self.query_one(FilePickerToolbar)
        dir_nav = self.query_one(DirectoryNavigation)
        toolbar.show_hidden = dir_nav.show_hidden
        
        # Focus the directory navigation
        dir_nav.focus()
    
    def _update_breadcrumbs(self, path: Path):
        """Stub for parent's breadcrumb update"""
        pass
    
    def _load_recent_locations(self):
        """Stub for parent's recent locations"""
        pass


class EnhancedFileSave(FileSave):
    """Enhanced file save dialog"""
    
    DEFAULT_CSS = FileSave.DEFAULT_CSS + """
    #file-picker-toolbar {
        height: 3;
        margin-bottom: 1;
        background: $surface;
    }
    """
    
    search_query = reactive("")
    
    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Save File",
        *,
        filters: Filters | None = None,
        default_filename: str = "",
        **kwargs
    ):
        super().__init__(location, title, **kwargs)
        self.filters = filters
        self.default_filename = default_filename
    
    def compose(self) -> ComposeResult:
        """Override compose to add our toolbar"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation, DriveNavigation
        import sys
        
        with Dialog() as dialog:
            dialog.border_title = self._title
            
            # Add the path display that parent expects
            yield Label("", id="current_path_display", classes="hidden")
            
            # Add our custom toolbar
            yield FilePickerToolbar(id="file-picker-toolbar")
            
            # Main file browser
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(self._location)
                yield DirectoryNavigation(self._location)
            
            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")
    
    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file saving"""
        from textual.widgets import Input, Select
        
        yield Input(value=self.default_filename, placeholder="File name...")
        if self.filters:
            yield Select(
                self.filters.selections,
                prompt="File type",
                value=0,
                id="file-filter"
            )
    
    def watch_search_query(self, query: str) -> None:
        """Filter directory entries based on search query"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            dir_nav.search_filter = query
        except Exception as e:
            logger.error(f"Error filtering entries: {e}")
    
    @on(FilePickerToolbar.SearchChanged)
    def handle_search_change(self, event: FilePickerToolbar.SearchChanged):
        """Handle search query changes"""
        self.search_query = event.query
    
    @on(FilePickerToolbar.SortChanged)
    def handle_sort_change(self, event: FilePickerToolbar.SortChanged):
        """Handle sort order changes"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.sort_by = event.sort_by
        self.notify(f"Sorting by {event.sort_by}", timeout=2)
    
    @on(FilePickerToolbar.HiddenToggled)
    def handle_hidden_toggle(self, event: FilePickerToolbar.HiddenToggled):
        """Handle hidden files toggle from toolbar"""
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.show_hidden = event.show_hidden
        self.notify(f"Hidden files {'shown' if event.show_hidden else 'hidden'}", timeout=2)
    
    def on_mount(self) -> None:
        """Initialize on mount"""
        # Don't call parent's on_mount as it expects widgets we don't have
        from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
        
        # Initialize toolbar state
        toolbar = self.query_one(FilePickerToolbar)
        dir_nav = self.query_one(DirectoryNavigation)
        toolbar.show_hidden = dir_nav.show_hidden
        
        # Focus the directory navigation
        dir_nav.focus()
    
    def _update_breadcrumbs(self, path: Path):
        """Stub for parent's breadcrumb update"""
        pass
    
    def _load_recent_locations(self):
        """Stub for parent's recent locations"""
        pass