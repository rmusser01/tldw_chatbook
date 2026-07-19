# enhanced_file_picker.py
# Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, bookmarks, and search

import os
import sys
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
from rich.console import RenderableType
from rich.table import Table
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ..Third_Party.textual_fspicker import Filters
from ..Third_Party.textual_fspicker.base_dialog import FileSystemPickerScreen
from ..Third_Party.textual_fspicker.file_dialog import BaseFileDialog
from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
from ..Third_Party.textual_fspicker.parts.directory_navigation import DirectoryEntry
from ..Third_Party.textual_fspicker.path_maker import MakePath
from ..Third_Party.textual_fspicker.safe_tests import is_dir
from ..Utils.path_validation import validate_path_simple
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
    """Manages bookmarked directories for quick access.

    task-261: the constructor is I/O-free. It used to run five synchronous
    ``Path.exists()`` probes (a stall hazard when $HOME dirs live on a cloud
    mount) plus, on first run, a full TOML config write — on EVERY picker
    construction. Default computation and the config load (including the
    first-run defaults write) are now deferred to the first call that
    actually needs bookmark data.
    """

    def __init__(self, context: str = "default"):
        """Initialize the manager without touching the filesystem or config.

        Args:
            context: Names the ``[filepicker] bookmarks_<context>`` config key
                so different picker surfaces keep separate bookmark lists.
        """
        self.context = context
        # None = not loaded yet; load_from_config() always leaves a list.
        self._bookmarks: Optional[List[Dict[str, Any]]] = None
        self._default_bookmarks_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def _default_bookmarks(self) -> List[Dict[str, Any]]:
        """Platform default bookmarks, computed (with I/O) at most once."""
        if self._default_bookmarks_cache is None:
            self._default_bookmarks_cache = self._get_default_bookmarks()
        return self._default_bookmarks_cache

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

    def _ensure_loaded(self) -> None:
        """Load bookmarks from config on first actual use (task-261)."""
        if self._bookmarks is None:
            self.load_from_config()

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
        if self._bookmarks is None:
            # Never loaded, so there is nothing to persist -- and saving here
            # would break the constructor's I/O-free guarantee (task-261).
            return
        try:
            save_setting_to_cli_config("filepicker", f"bookmarks_{self.context}", self._bookmarks)
        except Exception as e:
            logger.error(f"Failed to save bookmarks: {e}")

    def add(self, path: Path, name: Optional[str] = None, icon: str = "📁"):
        """Add a bookmark.

        Args:
            path: Directory to bookmark.
            name: Display name; defaults to the path's basename.
            icon: Emoji shown next to the bookmark.

        Returns:
            True if the bookmark was added, False if it already existed.
        """
        self._ensure_loaded()
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
        """Remove a bookmark.

        Args:
            path: Bookmarked directory to remove.
        """
        self._ensure_loaded()
        path_str = str(path.resolve())
        self._bookmarks = [b for b in self._bookmarks if b.get("path") != path_str]
        self.save_to_config()

    def is_bookmarked(self, path: Path) -> bool:
        """Check if a path is bookmarked.

        Args:
            path: Directory to look up.

        Returns:
            True if the path is currently bookmarked.
        """
        self._ensure_loaded()
        path_str = str(path.resolve())
        return any(b.get("path") == path_str for b in self._bookmarks)

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        """Get all bookmarks.

        Returns:
            A copy of the current bookmark records.
        """
        self._ensure_loaded()
        return self._bookmarks.copy()

    def reset_to_defaults(self):
        """Reset bookmarks to defaults"""
        self._bookmarks = self._default_bookmarks.copy()
        self.save_to_config()


class PathBreadcrumbs(Horizontal):
    """Clickable breadcrumb navigation for paths.

    NOTE: ``EnhancedFileDialog`` now uses the base ``#path-breadcrumbs``
    container directly. This widget is kept for compatibility and may be used
    by other consumers.
    """

    DEFAULT_CSS = """
    /* Local fallbacks so DEFAULT_CSS parses without the app bundle. */
    $ds-focus-bg: $surface;
    $ds-focus-fg: $text;

    PathBreadcrumbs {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-bottom: tall $primary-lighten-1;
        overflow: hidden;
    }

    PathBreadcrumbs .breadcrumb-button {
        min-width: 0;
        padding: 0 1;
        margin: 0;
        height: 1;
        background: transparent;
        border: none;
        color: $text;
        text-style: none;
    }

    PathBreadcrumbs .breadcrumb-button:hover {
        background: $primary 20%;
        text-style: underline;
    }

    PathBreadcrumbs .breadcrumb-button:focus {
        background: $ds-focus-bg;
        color: $ds-focus-fg;
        text-style: bold underline;
    }

    PathBreadcrumbs .breadcrumb-separator {
        margin: 0;
        padding: 0 1;
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

    def refresh_breadcrumbs(self):
        """Refresh the breadcrumb display synchronously.

        This previously ran inside a worker and awaited ``mount()``, which
        raised ``MountError`` when called before the widget was attached.
        """
        if not self.is_attached:
            return

        self.remove_children()

        parts = self.current_path.parts
        for i, part in enumerate(parts):
            partial_path = Path(*parts[:i+1])

            # Create button for each part
            btn = Button(part, variant="default", classes="breadcrumb-button")
            btn.data = partial_path  # Store the path in the button
            self.mount(btn)

            # Add separator if not last
            if i < len(parts) - 1:
                self.mount(Label("/", classes="breadcrumb-separator"))

    @on(Button.Pressed)
    def handle_breadcrumb_click(self, event: Button.Pressed):
        """Handle clicks on breadcrumb buttons"""
        if hasattr(event.button, 'data'):
            self.post_message(self.PathChanged(event.button.data))


class DirectorySearch(Horizontal):
    """Search widget for filtering directory contents.

    NOTE: ``EnhancedFileDialog`` uses the base ``#search-container`` input
    and a subclassed ``DirectoryNavigation`` for live filtering. This widget
    is kept for compatibility and may be used by other consumers.
    """

    DEFAULT_CSS = """
    DirectorySearch {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-bottom: tall $primary-lighten-1;
    }

    DirectorySearch Input {
        width: 1fr;
        margin-right: 1;
    }

    DirectorySearch Button {
        min-width: 7;
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
        yield Button("Clear", id="clear-search", variant="default")

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


def _make_glob_filter(patterns: Union[str, List[str], Tuple[str, ...]]) -> Callable[[Path], bool]:
    """Build a case-insensitive path filter from glob patterns.

    Args:
        patterns: One or more glob patterns. A semicolon-separated string or an
            iterable of strings is accepted.

    Returns:
        A callable that returns ``True`` when a path matches any pattern.
    """
    if isinstance(patterns, str):
        pattern_list = [p.strip().lower() for p in patterns.split(";") if p.strip()]
    else:
        pattern_list = [p.strip().lower() for p in patterns if p and isinstance(p, str)]

    def filter_func(path: Path) -> bool:
        name = path.name.lower()
        return any(fnmatch(name, pattern) for pattern in pattern_list)

    return filter_func


def _human_readable_size(size: int) -> str:
    """Return a concise, human-readable byte size."""
    if size < 1024:
        return f"{size} B"
    units = ["KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    value = size
    unit = units[-1]
    for u in units:
        value /= 1024
        if value < 1024:
            unit = u
            break
    formatted = f"{value:.1f}"
    # Drop trailing zeros and the decimal point when not needed.
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return f"{formatted} {unit}"


class FormattedDirectoryEntry(DirectoryEntry):
    """Directory entry with human-readable sizes and no size for directories."""

    @staticmethod
    def _size(location: Path) -> str:
        if is_dir(location):
            return ""
        try:
            entry_size = location.stat().st_size
        except (FileNotFoundError, OSError):
            entry_size = 0
        return _human_readable_size(entry_size)


class MultiSelectDirectoryEntry(FormattedDirectoryEntry):
    """A directory entry that renders a selection marker for multi-select."""

    def __init__(self, location: Path, styles: Any, selected: bool = False) -> None:
        self.selected = selected
        super().__init__(location, styles)

    def _as_renderable(self, location: Path) -> RenderableType:
        """Render with an extra leading check-mark column."""
        prompt = Table.grid(expand=True)
        prompt.add_column(no_wrap=True, width=1)
        prompt.add_column(
            no_wrap=True,
            width=1,
            style=self._style(self._styles.name, location),
        )
        prompt.add_column(no_wrap=True, width=3)
        prompt.add_column(
            no_wrap=True,
            justify="left",
            ratio=1,
            style=self._style(self._styles.name, location),
        )
        prompt.add_column(
            no_wrap=True,
            justify="right",
            width=10,
            style=self._style(self._styles.size, location),
        )
        prompt.add_column(
            no_wrap=True,
            justify="right",
            width=20,
            style=self._style(self._styles.time, location),
        )
        prompt.add_column(no_wrap=True, width=1)
        marker = "✓" if self.selected else " "
        prompt.add_row(
            "",
            marker,
            self.FOLDER_ICON if is_dir(location) else self.FILE_ICON,
            self._name(location),
            self._size(location),
            self._mtime(location),
            "",
        )
        return prompt


class SearchableDirectoryNavigation(DirectoryNavigation):
    """Directory navigation that also filters by a free-text search term.

    The base ``DirectoryNavigation`` references ``search_filter`` in its watch
    method but does not declare the reactive variable, and its
    ``_repopulate_display`` does not apply the term. This subclass adds the
    missing reactive and applies it without editing third-party code.

    Additional enhancements:
    - Type-ahead jumping to the next entry whose name starts with the typed
      prefix.
    - Optional multi-select rendering via :class:`MultiSelectDirectoryEntry`.
    """

    search_filter = reactive("")
    """Free-text filter applied to entry names."""

    class SearchCountChanged(Message):
        """Posted when the number of visible options changes."""
        def __init__(self, navigation: DirectoryNavigation, count: int, query: str) -> None:
            self.navigation = navigation
            self.count = count
            self.query = query
            super().__init__()

    class ToggleSelection(Message):
        """Posted when the user asks to toggle the highlighted entry."""
        def __init__(self, navigation: DirectoryNavigation) -> None:
            self.navigation = navigation
            super().__init__()

    def __init__(self, location: Path | str = ".") -> None:
        super().__init__(location)
        self._type_ahead_buffer = ""
        self._type_ahead_timer: Optional[Timer] = None

    def _restart_type_ahead_timer(self) -> None:
        """Reset the inactivity timeout that clears the type-ahead buffer."""
        if self._type_ahead_timer is not None:
            self._type_ahead_timer.stop()
        self._type_ahead_timer = self.app.set_timer(0.8, self._reset_type_ahead)

    def _reset_type_ahead(self) -> None:
        """Clear the type-ahead buffer after a period of inactivity."""
        self._type_ahead_buffer = ""
        self._type_ahead_timer = None

    def _jump_to_prefix(self, prefix: str) -> None:
        """Move the highlight to the next entry whose name starts with prefix."""
        prefix = prefix.lower()
        start = (self.highlighted or -1) + 1
        options = self.options

        def match_at(index: int) -> bool:
            option = options[index]
            if not isinstance(option, DirectoryEntry):
                return False
            name = option.location.name.lower()
            return name.startswith(prefix)

        # Search from the item after the current highlight, wrapping around.
        for idx in range(start, len(options)):
            if match_at(idx):
                self.highlighted = idx
                return
        for idx in range(0, start):
            if match_at(idx):
                self.highlighted = idx
                return

    def _on_key(self, event: events.Key) -> None:
        """Handle type-ahead jumping and multi-select toggling.

        Navigation keys (arrows, page, home, end, enter, backspace) are left
        for the default OptionList bindings.  The space bar toggles selection
        in multi-select mode.  Printable letters drive type-ahead jumping;
        digits are reserved for the screen's bookmark-jump bindings unless a
        type-ahead prefix is already active.
        """
        if event.key in ("up", "down", "pageup", "pagedown", "home", "end", "enter", "backspace"):
            return

        if event.key == "space":
            if getattr(self.screen, "multi_select", False):
                event.stop()
                event.prevent_default()
                self.post_message(self.ToggleSelection(self))
            return

        if not event.is_printable or not event.character or len(event.character) != 1:
            return

        char = event.character
        if char.isspace():
            return

        # Digits jump bookmarks when no prefix is active; otherwise extend it.
        if char.isdigit():
            if self._type_ahead_buffer:
                event.stop()
                event.prevent_default()
                self._type_ahead_buffer += char
                self._jump_to_prefix(self._type_ahead_buffer)
                self._restart_type_ahead_timer()
            return

        # Printable characters start or extend the type-ahead prefix.
        event.stop()
        event.prevent_default()
        self._type_ahead_buffer += char.lower()
        self._jump_to_prefix(self._type_ahead_buffer)
        self._restart_type_ahead_timer()

    def _repopulate_display(self) -> None:
        """Repopulate the display, honouring file, hidden, and search filters."""
        styles = self._styles
        query = self.search_filter.strip().lower()

        # Remember the currently highlighted path so we can restore it after
        # rebuilding the option list.
        previous_path: Optional[Path] = None
        try:
            if self.highlighted is not None:
                highlighted_option = self.get_option_at_index(self.highlighted)
                if isinstance(highlighted_option, DirectoryEntry):
                    previous_path = highlighted_option.location
        except Exception:
            pass

        # Determine whether the hosting dialog is in multi-select mode.
        screen = self.screen
        multi_select = getattr(screen, "multi_select", False)
        selected = getattr(screen, "_selected_paths", set()) if multi_select else set()

        with self.app.batch_update():
            self.clear_options()
            if not self.is_root:
                if multi_select:
                    self.add_option(MultiSelectDirectoryEntry(self._location / "..", styles, False))
                else:
                    self.add_option(FormattedDirectoryEntry(self._location / "..", styles))
            self.add_options(
                self._sort(
                    (
                        MultiSelectDirectoryEntry(entry.location, styles, entry.location in selected)
                        if multi_select
                        else FormattedDirectoryEntry(entry.location, styles)
                    )
                    for entry in self._entries
                    if not self.hide(entry.location)
                    and (not query or query in entry.location.name.lower())
                )
            )
        self._settle_highlight()

        # Restore the previous highlight if the entry still exists.
        if previous_path is not None:
            for idx, option in enumerate(self.options):
                if isinstance(option, DirectoryEntry) and option.location == previous_path:
                    self.highlighted = idx
                    break

        self.post_message(self.SearchCountChanged(self, self.option_count, query))


class EnhancedFileDialog(BaseFileDialog):
    """Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, bookmarks, and search"""

    DEFAULT_CSS = BaseFileDialog.DEFAULT_CSS + """
    .hidden {
        display: none;
    }

    EnhancedFileDialog Dialog {
        height: 95%;
        width: 95%;
        border-title-align: center;
        border-title-style: bold;
    }

    #dialog-body {
        height: 1fr;
        width: 1fr;
    }

    #filepicker-sidebar {
        width: 28;
        height: 1fr;
        border-right: solid $surface-lighten-2;
        background: $surface;
        display: none;
    }

    #filepicker-main {
        width: 1fr;
        height: 1fr;
    }

    #file-list-container {
        width: 1fr;
        height: 1fr;
    }

    #file-list-pane {
        width: 1fr;
        height: 1fr;
    }

    EnhancedFileDialog #file-list-header {
        height: auto;
        min-height: 1;
        padding: 0 1 0 2;
        color: $text-muted;
        text-style: bold;
        background: $surface;
        border-bottom: solid $surface-lighten-2;
    }

    #recent-locations, #bookmarks-panel {
        height: 1fr;
        border: none;
        background: $surface;
        padding: 0 1;
    }

    #recent-list, #bookmarks-list {
        height: 1fr;
        background: $surface;
    }

    .recent-item, .bookmark-item {
        padding: 0 1;
    }

    .empty-state {
        color: $text-muted;
        text-style: italic;
    }

    EnhancedFileDialog #bookmarks-list {
        height: 1fr;
        overflow-y: auto;
    }

    EnhancedFileDialog #bookmarks-list ListItem {
        height: 3;
        margin: 0;
        padding: 0 1;
    }

    .bookmark-item {
        padding: 0 1;
        height: 3;
        overflow: hidden;
    }

    .bookmark-item-icon {
        margin-right: 1;
    }

    .bookmarks-header {
        height: 2;
        padding: 0 1;
    }

    .bookmark-button, #add-bookmark {
        min-width: 1;
        width: 3;
        height: 1;
        margin: 0;
        padding: 0;
        text-align: center;
    }

    .bookmark-container {
        width: 100%;
        height: 3;
        align: left middle;
    }

    #current_path_display {
        display: none;
    }

    EnhancedFileDialog #path-breadcrumbs {
        height: 3;
        min-height: 3;
        padding: 0 1;
        margin-bottom: 1;
        background: $surface;
        border-bottom: tall $primary-lighten-1;
        align: center middle;
    }

    EnhancedFileDialog #path-breadcrumbs .breadcrumb-btn {
        min-width: 0;
        padding: 0 1;
        margin: 0;
        height: 1;
        background: transparent;
        border: none;
        color: $text;
        text-style: none;
    }

    EnhancedFileDialog #path-breadcrumbs .breadcrumb-btn:hover {
        background: $primary 20%;
        text-style: underline;
    }

    EnhancedFileDialog #path-breadcrumbs .breadcrumb-separator {
        margin: 0;
        padding: 0 1;
        color: $text-muted;
    }

    EnhancedFileDialog #path-breadcrumbs .breadcrumb-ellipsis {
        color: $text-disabled;
        padding: 0;
    }

    EnhancedFileDialog #path-input-container {
        height: 3;
        padding: 0 1;
    }

    EnhancedFileDialog #path-input {
        width: 1fr;
    }

    EnhancedFileDialog #go-to-path,
    EnhancedFileDialog #cancel-path-input {
        min-width: 7;
    }

    .search-active {
        border-title-style: bold;
        border-title-color: $warning;
    }

    .section-title {
        width: 1fr;
        text-style: bold;
    }

    .shortcut-hints {
        height: auto;
        min-height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
        text-align: center;
        text-style: italic;
    }

    .shortcut-hints.collapsed {
        text-align: right;
        text-style: none;
    }

    #select {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    EnhancedFileDialog #error-line {
        height: auto;
        min-height: 1;
        padding: 0 1;
        color: $error;
        text-style: bold;
        display: none;
    }

    EnhancedFileDialog SearchableDirectoryNavigation > .option-list--option-highlighted {
        background: $primary 30%;
        color: $text;
        text-style: bold;
    }

    EnhancedFileDialog SearchableDirectoryNavigation:focus > .option-list--option-highlighted {
        background: $primary 50%;
        color: $text;
        text-style: bold;
    }

    EnhancedFileDialog #search-status {
        width: auto;
        min-width: 10;
        padding: 0 1;
        color: $text-muted;
        text-align: right;
        content-align: center middle;
    }

    EnhancedFileDialog #search-no-match {
        height: auto;
        padding: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }

    EnhancedFileDialog #multi-select-info {
        height: auto;
        min-height: 1;
        padding: 0 1;
        color: $text-muted;
        text-align: right;
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_bookmarks", "Show bookmarks"),
        Binding("question_mark", "toggle_hints", "Toggle shortcuts"),
        *[Binding(str(n), f"jump_bookmark('{n}')", f"Bookmark {n}", show=False) for n in range(1, 10)],
    ]

    show_bookmarks = reactive(False)
    show_hints = reactive(True)

    # Base class handlers that this subclass replaces. Textual dispatches
    # decorated handlers from the whole MRO, so simply renaming the subclass
    # methods and adding no-op overrides is not enough to stop the base
    # implementations from firing. ``_get_dispatch_methods`` filters them out.
    _SUPPRESSED_BASE_HANDLERS = {
        BaseFileDialog._select_file,
        BaseFileDialog._confirm_file,
        FileSystemPickerScreen._on_clear_search,
        FileSystemPickerScreen._on_directory_changed,
    }

    def _get_dispatch_methods(self, method_name: str, message: Message):
        """Yield dispatch methods, skipping base handlers we replace."""
        for cls, method in super()._get_dispatch_methods(method_name, message):
            if method.__func__ in self._SUPPRESSED_BASE_HANDLERS:
                continue
            yield cls, method

    def __init__(
        self,
        location: Union[str, Path] = ".",
        title: str = "",
        select_button: str = "",
        cancel_button: str = "",
        *,
        filters: Optional[Union[Filters, List[str], Tuple[str, ...]]] = None,
        default_file: Optional[Union[str, Path]] = None,
        context: str = "default",
        multi_select: bool = False,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        # ``BaseFileDialog`` does not accept Textual screen kwargs such as
        # ``id`` or ``classes``; apply them manually after the base init.
        # Normalize legacy list/tuple filters to a Filters instance so the
        # dialog can safely read ``self.filters.selections`` during compose.
        filters = self._normalize_filters(filters)
        super().__init__(
            location,
            title,
            select_button,
            cancel_button,
            filters=filters,
            default_file=default_file,
        )
        # The base class stores filters internally as ``_filters``; expose the
        # normalized value as ``self.filters`` for the input bar and callers.
        self.filters = filters
        if id is not None:
            self.id = id
        if classes is not None:
            self.classes = classes
        if name is not None:
            self.name = name
        self.context = context
        self.multi_select = multi_select
        self._selected_paths: set[Path] = set()
        self.recent_locations = RecentLocations(context=context)
        self.bookmarks_manager = BookmarksManager(context=context)
        self._last_directory = self._get_last_directory()
        if self._last_directory is not None:
            self._location = self._last_directory

    @staticmethod
    def _normalize_filters(
        filters: Optional[Union[Filters, List[str], Tuple[str, ...]]]
    ) -> Optional[Filters]:
        """Convert legacy list/tuple filters into a ``Filters`` instance.

        Args:
            filters: A ``Filters`` collection, an iterable of glob strings, or
                ``None``.

        Returns:
            A ``Filters`` instance or ``None``.
        """
        if filters is None or isinstance(filters, Filters):
            return filters
        patterns = list(filters)
        if not patterns:
            return None
        return Filters(("Filtered files", _make_glob_filter(patterns)))

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
        """Compose the enhanced file picker UI.

        The base ``FileSystemPickerScreen`` (defined in
        ``Third_Party/textual_fspicker/base_dialog.py``) does not expose any
        compositional hooks for adding a bookmarks panel or for replacing its
        ``DirectoryNavigation`` with a search-aware subclass. To keep the
        enhancement isolated in this file, we deliberately mirror the base
        layout here and inject the extra widgets at the documented IDs.
        """
        from ..Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
        from ..Third_Party.textual_fspicker.parts import DriveNavigation

        with Dialog() as dialog:
            dialog.border_title = self._title

            with Horizontal(id="dialog-body"):
                # Sidebar for bookmarks/recent (hidden by default)
                with VerticalScroll(id="filepicker-sidebar"):
                    with VerticalScroll(id="recent-locations"):
                        yield Label("📋 Recent", classes="section-title")
                        yield ListView(id="recent-list")

                    with VerticalScroll(id="bookmarks-panel"):
                        with Horizontal(classes="bookmarks-header"):
                            yield Label("⭐ Bookmarks", classes="section-title")
                            yield Button("➕", id="add-bookmark", classes="bookmark-button")
                        yield ListView(id="bookmarks-list")

                with Vertical(id="filepicker-main"):
                    # Path display (kept for base on_mount, hidden visually)
                    yield Label(id="current_path_display")
                    with Horizontal(id="path-breadcrumbs"):
                        pass

                    # Path input field (hidden by default, shown with Ctrl+L)
                    with Horizontal(id="path-input-container", classes="hidden"):
                        yield Input(placeholder="Enter path...", id="path-input")
                        yield Button("Go", id="go-to-path", variant="primary")
                        yield Button("Cancel", id="cancel-path-input", variant="default")

                    # Search container (hidden by default)
                    with Horizontal(id="search-container"):
                        yield Input(placeholder="Search files...", id="search-input")
                        yield Button("Clear", id="clear-search", variant="default")
                        yield Label("", id="search-status")

                    # Shown when a search returns no results.
                    yield Static("", id="search-no-match", classes="hidden")

                    # Multi-select status (visible only in multi-select mode).
                    yield Static("", id="multi-select-info")

                    # Main directory navigation
                    with Horizontal(id="file-list-container"):
                        if sys.platform == "win32":
                            yield DriveNavigation(self._location)
                        with Vertical(id="file-list-pane"):
                            yield Static(self._file_list_header(), id="file-list-header")
                            yield SearchableDirectoryNavigation(self._location)

                    yield Static(
                        "",
                        id="shortcut-hints",
                        classes="shortcut-hints",
                    )

                    # Dedicated error line above the input bar.
                    yield Static("", id="error-line")

                    # Input bar with buttons
                    with InputBar():
                        yield from self._input_bar()
                        yield Button(
                            self._label(self._select_button, "Select"),
                            id="select",
                            variant="primary",
                        )
                        yield Button(
                            self._label(self._cancel_button, "Cancel"),
                            id="cancel",
                            variant="default",
                        )

    def on_mount(self) -> None:
        """Initialize the dialog on mount.

        The base ``on_mount`` expects ``#path-breadcrumbs`` and
        ``#recent-list`` to exist; our compose provides them, so calling
        ``super().on_mount()`` is safe.

        Hidden panels rely on inline ``styles.display`` rather than CSS
        ``display: none`` rules because the vendored base selectors do not
        reliably override container defaults in this subclass.
        """
        super().on_mount()
        self._update_bookmarks_list()
        self._update_bookmark_button_state(
            self.query_one(SearchableDirectoryNavigation).location
        )
        try:
            self.query_one("#path-input-container").styles.display = "none"
            self.query_one("#search-container").styles.display = "none"
            self.query_one("#recent-locations").styles.display = "none"
            self.query_one("#bookmarks-panel").styles.display = "none"
            self.query_one("#filepicker-sidebar").styles.display = "none"
            # Breadcrumbs already show the path; the label is redundant and
            # often truncated.
            self.query_one("#current_path_display").styles.display = "none"
            self.watch_show_hints(self.show_hints)
            if self.multi_select:
                self.query_one("#multi-select-info").styles.display = "block"
                self._update_multi_select_ui()
        except Exception:
            pass

    def _sync_sidebar(self) -> None:
        """Show the sidebar if either panel is open, otherwise hide it."""
        try:
            sidebar = self.query_one("#filepicker-sidebar")
            sidebar.styles.display = (
                "block" if self.show_recent or self.show_bookmarks else "none"
            )
        except Exception:
            pass

    def watch_show_recent(self, show: bool) -> None:
        """Toggle recent locations visibility."""
        try:
            self.query_one("#recent-locations").styles.display = (
                "block" if show else "none"
            )
            if show:
                self.show_bookmarks = False
            self._sync_sidebar()
        except Exception:
            pass

    def action_show_recent(self) -> None:
        """Toggle the recent locations panel (explicit override)."""
        self.show_recent = not self.show_recent

    def watch_show_bookmarks(self, show: bool) -> None:
        """Toggle bookmarks panel visibility."""
        try:
            self.query_one("#bookmarks-panel").styles.display = (
                "block" if show else "none"
            )
            if show:
                self.show_recent = False
            self._sync_sidebar()
        except Exception:
            pass

    def watch_search_active(self, active: bool) -> None:
        """Toggle search container visibility."""
        try:
            self.query_one("#search-container").styles.display = (
                "block" if active else "none"
            )
            if active:
                self.query_one("#search-input", Input).focus()
        except Exception:
            pass

    def _shortcut_hint_text(self) -> str:
        """Full shortcut-hint text shown when the footer is expanded."""
        select_hint = self._label(self._select_button, "Select")
        hints = [
            "Ctrl+B Bookmarks",
            "Ctrl+R Recent",
            "Ctrl+F Search",
            "Ctrl+L Path",
            "1-9 Jump",
        ]
        if self.multi_select:
            hints.append("Space Toggle")
        hints.extend([f"Enter {select_hint}", "Esc Cancel", "? Hide"])
        return "  ".join(hints)

    def watch_show_hints(self, show: bool) -> None:
        """Expand or collapse the shortcut-hints footer."""
        try:
            hints = self.query_one("#shortcut-hints", Static)
            if show:
                hints.update(self._shortcut_hint_text())
                hints.remove_class("collapsed")
            else:
                hints.update("? Show shortcuts")
                hints.add_class("collapsed")
        except Exception:
            pass

    def action_toggle_hints(self) -> None:
        """Toggle the shortcut-hints footer."""
        self.show_hints = not self.show_hints

    def action_focus_path_input(self) -> None:
        """Toggle and focus the path input field."""
        try:
            path_container = self.query_one("#path-input-container")
            path_input = self.query_one("#path-input", Input)
            if path_container.styles.display == "none":
                path_container.styles.display = "block"
                path_input.value = str(
                    self.query_one(SearchableDirectoryNavigation).location
                )
                path_input.focus()
                path_input.selection = (0, len(path_input.value))
            else:
                path_container.styles.display = "none"
                self.query_one(SearchableDirectoryNavigation).focus()
        except Exception as e:
            self.notify(f"Error toggling path input: {e}", severity="error", timeout=2)

    def _select_file(self, event: DirectoryNavigation.Selected) -> None:
        """No-op override of ``BaseFileDialog._select_file``.

        The real selection handling happens in ``_on_select_file``. The base
        decorated handler is suppressed via ``_get_dispatch_methods`` because
        Textual dispatches decorated handlers from the whole MRO; a simple
        name-shadowing override is not sufficient to stop it from firing.
        """
        pass

    @on(DirectoryNavigation.Selected)
    def _on_select_file(self, event: DirectoryNavigation.Selected) -> None:
        """Handle a file being selected in the picker.

        In multi-select mode the list selection toggles the file in the
        selected set.  In single-select mode the file name is copied into the
        filename input for editing/confirmation.
        """
        if self.multi_select:
            self._toggle_path_selection(event.path)
            return

        try:
            file_name = self.query_one("#filename-input", Input)
        except Exception:
            return
        file_name.value = str(event.path.name)
        file_name.focus()

    def _confirm_file(self, event: Input.Submitted | Button.Pressed) -> None:
        """No-op override of ``BaseFileDialog._confirm_file``.

        The real confirmation handling happens in ``_on_confirm_file`` and
        ``_on_select_button``. The base decorated handler is suppressed via
        ``_get_dispatch_methods``.
        """
        pass

    def _confirm_single(self) -> None:
        """Confirm the single filename currently in the input box."""
        if self.multi_select:
            # Multi-select uses _confirm_multi_select; this path has no filename input.
            return

        file_name = self.query_one("#filename-input", Input)

        # Only even try and process this if there's some input.
        if not file_name.value:
            self._set_error(self.ERROR_A_FILE_MUST_BE_CHOSEN)
            return

        # If it looks like the user is typing in some sort of home
        # directory path...
        try:
            if file_name.value.startswith("~"):
                # ...let's simply expand and go with that.
                chosen = MakePath.of(file_name.value).expanduser().resolve()
            else:
                # It's not a home directory path, so let's combine with the
                # location of the directory navigator widget.
                chosen = (
                    self.query_one(DirectoryNavigation).location / file_name.value
                ).resolve()
        except (RuntimeError, OSError) as error:
            self._set_error(str(error))
            return

        # If it's a directory, approach it like it's the user simply
        # doing a "cd".
        try:
            if chosen.is_dir():
                if sys.platform == "win32":
                    if drive_letter := MakePath.of(chosen).drive:
                        # Ensure DriveNavigation is present before querying
                        try:
                            from ..Third_Party.textual_fspicker.parts import DriveNavigation
                            drive_nav = self.query_one(DriveNavigation)
                            drive_nav.drive = drive_letter
                        except Exception:  # QueryError if not present
                            pass  # Silently ignore if DriveNavigation isn't there
                self.query_one(DirectoryNavigation).location = chosen
                self.query_one(DirectoryNavigation).focus()
                file_name.value = ""
                return
        except PermissionError:
            self._set_error(self.ERROR_PERMISSION_ERROR)
            return

        # If the chosen file passes the final tests...
        if self._should_return(chosen):
            # ...return it.
            self.dismiss(result=chosen)

    def _confirm_multi_select(self) -> None:
        """Confirm a multi-select pick and return the selected files."""
        if not self._selected_paths:
            self._set_error("Select at least one file")
            return
        self.dismiss(result=list(self._selected_paths))

    @on(Input.Submitted, "#filename-input")
    def _on_confirm_file(self, event: Input.Submitted) -> None:
        """Handle Enter in the filename input.

        In single-select mode this confirms the file.  In multi-select mode a
        non-empty value adds the file to the selection and an empty value
        confirms the current selection.
        """
        event.stop()

        if not self.multi_select:
            self._confirm_single()
            return

        try:
            file_name = self.query_one("#filename-input", Input)
        except Exception:
            self._confirm_multi_select()
            return

        if not file_name.value:
            self._confirm_multi_select()
            return

        try:
            if file_name.value.startswith("~"):
                chosen = MakePath.of(file_name.value).expanduser().resolve()
            else:
                chosen = (
                    self.query_one(DirectoryNavigation).location / file_name.value
                ).resolve()
        except (RuntimeError, OSError) as error:
            self._set_error(str(error))
            return

        if self._should_return(chosen):
            self._toggle_path_selection(chosen)
            file_name.value = ""

    @on(Button.Pressed, "#select")
    def _on_select_button(self, event: Button.Pressed) -> None:
        """Handle the main select/save/open button."""
        event.stop()
        if self.multi_select:
            self._confirm_multi_select()
        else:
            self._confirm_single()

    def _toggle_path_selection(self, path: Path) -> None:
        """Add or remove a file from the multi-select set."""
        if not path.is_file():
            self.notify("Only files can be selected", severity="warning", timeout=2)
            return
        if path in self._selected_paths:
            self._selected_paths.discard(path)
        else:
            self._selected_paths.add(path)
        self._update_multi_select_ui()
        try:
            self.query_one(SearchableDirectoryNavigation)._repopulate_display()
        except Exception:
            pass

    def action_toggle_selection(self) -> None:
        """Toggle selection for the currently highlighted file."""
        if not self.multi_select:
            return
        nav = self.query_one(SearchableDirectoryNavigation)
        highlighted = nav.highlighted
        if highlighted is None:
            return
        option = nav.get_option_at_index(highlighted)
        self._toggle_path_selection(option.location)

    def _update_multi_select_ui(self) -> None:
        """Refresh the multi-select status label and select button."""
        if not self.multi_select:
            return
        count = len(self._selected_paths)
        try:
            info = self.query_one("#multi-select-info", Static)
            if count == 0:
                info.update("No files selected")
            elif count == 1:
                info.update("1 file selected")
            else:
                info.update(f"{count} files selected")
        except Exception:
            pass
        try:
            btn = self.query_one("#select", Button)
            base = self._label(self._select_button, "Select")
            btn.label = f"{base} ({count})" if count else base
        except Exception:
            pass

    _MAX_VISIBLE_BREADCRUMBS = 5

    def _update_breadcrumbs(self, path: Path) -> None:
        """Update breadcrumb navigation using the base container.

        Very deep paths are collapsed in the middle so the current directory
        always remains readable.
        """
        try:
            breadcrumb_container = self.query_one("#path-breadcrumbs", Horizontal)
            breadcrumb_container.remove_children()

            parts = path.parts
            max_visible = self._MAX_VISIBLE_BREADCRUMBS
            if len(parts) > max_visible:
                # Root + ellipsis + tail so the current directory is visible.
                visible_indices = [0] + list(range(len(parts) - max_visible + 2, len(parts)))
            else:
                visible_indices = list(range(len(parts)))

            for position, i in enumerate(visible_indices):
                part = parts[i]
                # Show the root as a home-ish icon rather than a raw "/" that
                # would collide with the separator.
                label = "🏠" if part == "/" else part
                partial_path = Path(*parts[: i + 1])

                if i > 0 and i != visible_indices[position - 1] + 1:
                    breadcrumb_container.mount(
                        Label("…", classes="breadcrumb-separator breadcrumb-ellipsis")
                    )

                btn = Button(label, variant="default", classes="breadcrumb-btn")
                btn.tooltip = str(partial_path)
                btn.data = partial_path
                breadcrumb_container.mount(btn)

                if position < len(visible_indices) - 1:
                    breadcrumb_container.mount(
                        Label("›", classes="breadcrumb-separator")
                    )
        except Exception:
            pass

    @on(Button.Pressed, ".breadcrumb-btn")
    def _on_breadcrumb_click(self, event: Button.Pressed) -> None:
        """Navigate to the directory represented by a breadcrumb button."""
        path = getattr(event.button, "data", None)
        if isinstance(path, Path) and path.exists():
            try:
                self.query_one(SearchableDirectoryNavigation).location = path
            except Exception:
                pass

    def _load_recent_locations(self) -> None:
        """Populate the recent locations list from persistent storage."""
        try:
            recent_list = self.query_one("#recent-list", ListView)
            recent_list.clear()

            recent = self.recent_locations.get_recent()
            if not recent:
                empty_item = ListItem(
                    Label("No recent files yet. Open a file to see it here.", classes="recent-item empty-state")
                )
                empty_item.data = None
                recent_list.append(empty_item)
                return

            for item in recent:
                path_str = item["path"]
                name = item["name"]
                file_type = item.get("type", "file")
                icon = "📁" if file_type == "directory" else "📄"
                list_item = ListItem(
                    Label(f"{icon} {name} - {path_str}", classes="recent-item")
                )
                list_item.data = path_str
                recent_list.append(list_item)
        except Exception:
            pass

    def _add_to_recent(self, path: Path, file_type: str) -> None:
        """Persist a recent location and refresh the list."""
        self.recent_locations.add(path, file_type)
        self._save_last_directory(path)
        self._load_recent_locations()

    def _update_bookmarks_list(self):
        """Update the bookmarks list."""
        try:
            bookmarks_list = self.query_one("#bookmarks-list", ListView)
            bookmarks_list.clear()

            bookmarks = self.bookmarks_manager.get_bookmarks()
            if not bookmarks:
                empty_item = ListItem(
                    Label(
                        "No bookmarks. Press Ctrl+D to bookmark the current directory.",
                        classes="bookmark-item empty-state"
                    )
                )
                empty_item.data = None
                bookmarks_list.append(empty_item)
                return

            for bookmark in bookmarks:
                path = bookmark["path"]
                name = bookmark["name"]
                icon = bookmark.get("icon", "📁")
                if len(name) > 15:
                    name = name[:12] + "..."
                list_item = ListItem(
                    Horizontal(
                        Label(icon, classes="bookmark-item-icon"),
                        Label(name, classes="bookmark-item"),
                        classes="bookmark-container"
                    )
                )
                list_item.data = path
                bookmarks_list.append(list_item)
        except Exception as e:
            logger.error(f"Error updating bookmarks list: {e}")

    def _update_bookmark_button_state(self, path: Path):
        """Update the bookmark button based on current directory."""
        try:
            btn = self.query_one("#add-bookmark", Button)
            if self.bookmarks_manager.is_bookmarked(path):
                btn.label = "⭐"
                btn.tooltip = "Remove bookmark"
            else:
                btn.label = "➕"
                btn.tooltip = "Add bookmark"
        except Exception:
            pass

    def _filename_placeholder(self) -> str:
        """Placeholder text for the filename input."""
        if getattr(self, "filters", None):
            return "File name (filtered by selected type)"
        return "File name"

    def _file_list_header(self) -> RenderableType:
        """Return a column header matching DirectoryEntry's layout."""
        grid = Table.grid(expand=True)
        # Optional multi-select marker column.
        if getattr(self, "multi_select", False):
            grid.add_column(no_wrap=True, width=1)
            grid.add_column(no_wrap=True, width=1)
        else:
            grid.add_column(no_wrap=True, width=1)
        grid.add_column(no_wrap=True, width=3)
        grid.add_column(no_wrap=True, ratio=1)
        grid.add_column(no_wrap=True, justify="right", width=10)
        grid.add_column(no_wrap=True, justify="right", width=20)
        grid.add_column(no_wrap=True, width=1)
        if self.multi_select:
            grid.add_row("", "", "", "Name", "Size", "Modified", "")
        else:
            grid.add_row("", "", "Name", "Size", "Modified", "")
        return grid

    def action_toggle_bookmarks(self) -> None:
        """Toggle bookmarks panel."""
        self.show_bookmarks = not self.show_bookmarks

    def action_toggle_recent(self) -> None:
        """Toggle recent locations panel (compatibility alias)."""
        self.show_recent = not self.show_recent

    def action_toggle_hidden(self) -> None:
        """Toggle showing hidden files (compatibility alias)."""
        self.query_one(SearchableDirectoryNavigation).toggle_hidden()
        self.notify("Hidden files toggled", timeout=2)

    def _action_hidden(self) -> None:
        """Override the base ``hidden`` action to use the notifying toggle."""
        self.action_toggle_hidden()

    def action_bookmark_current(self) -> None:
        """Add or remove current directory from bookmarks."""
        dir_nav = self.query_one(SearchableDirectoryNavigation)
        current_path = dir_nav.location

        if self.bookmarks_manager.is_bookmarked(current_path):
            self.bookmarks_manager.remove(current_path)
            self.notify(f"Removed bookmark: {current_path.name}", timeout=2)
        else:
            self.bookmarks_manager.add(current_path)
            self.notify(f"Added bookmark: {current_path.name}", timeout=2)

        self._update_bookmarks_list()
        self._update_bookmark_button_state(current_path)

    def _jump_to_bookmark(self, index: int) -> None:
        """Jump to the bookmark at ``index`` if one exists.

        Does nothing when an input field is focused so typing filenames or
        search queries is not hijacked.
        """
        focused = self.screen.focused if self.screen else None
        if isinstance(focused, Input):
            return

        bookmarks = self.bookmarks_manager.get_bookmarks()
        if 0 <= index < len(bookmarks):
            raw_path = bookmarks[index]["path"]
            try:
                path = validate_path_simple(raw_path, require_exists=True)
            except ValueError as exc:
                self.notify(f"Invalid bookmark path: {exc}", severity="warning")
                return
            dir_nav = self.query_one(SearchableDirectoryNavigation)
            dir_nav.location = path
            self._update_bookmark_button_state(path)
            self.notify(f"Jumped to: {bookmarks[index]['name']}", timeout=1)

    def action_jump_bookmark(self, index: str) -> None:
        """Jump to the bookmark at the 1-based index supplied by the binding."""
        try:
            idx = int(index) - 1
        except ValueError:
            return
        self._jump_to_bookmark(idx)

    def _set_error(self, message: str = "") -> None:
        """Show or clear the dedicated error line.

        Overrides the base implementation that paints errors into the dialog's
        border subtitle, which is too easy to miss.
        """
        try:
            error_line = self.query_one("#error-line", Static)
        except Exception:
            # Fall back to the base border-subtitle behavior if the dedicated
            # line is not in the DOM.
            super()._set_error(message)
            return

        if message:
            error_line.update(message)
            error_line.styles.display = "block"
        else:
            error_line.update("")
            error_line.styles.display = "none"

    @on(DirectoryNavigation.Changed)
    def _on_directory_changed(self, event: DirectoryNavigation.Changed) -> None:
        """React to directory navigation.

        Mirrors the base handler but deliberately skips ``_add_to_recent`` so
        the config is not rewritten on every directory change. Persistence
        happens in ``dismiss`` instead.
        """
        self._set_error()
        try:
            current_path_label = self.query_one("#current_path_display", Label)
            current_path_label.update(str(event.control.location))
        except Exception:
            pass
        self._update_breadcrumbs(event.control.location)
        self._update_bookmark_button_state(event.control.location)

    @on(ListView.Selected, "#bookmarks-list")
    def handle_bookmark_selection(self, event: ListView.Selected):
        """Handle selection from bookmarks list."""
        if hasattr(event.item, 'data') and event.item.data is not None:
            path = Path(event.item.data)
            if path.exists():
                dir_nav = self.query_one(SearchableDirectoryNavigation)
                dir_nav.location = path
                self.show_bookmarks = False
                self._update_bookmark_button_state(path)
            else:
                self.notify(f"Path no longer exists: {path}", severity="warning")

    @on(Button.Pressed, "#add-bookmark")
    def handle_bookmark_button(self, event: Button.Pressed):
        """Handle bookmark button press."""
        self.action_bookmark_current()

    def _on_clear_search(self, event: Button.Pressed) -> None:
        """No-op override of ``FileSystemPickerScreen._on_clear_search``.

        The real clear handling happens in ``_on_clear_search_enhanced``. The
        base decorated handler is suppressed via ``_get_dispatch_methods``.
        """
        pass

    @on(Button.Pressed, "#clear-search")
    def _on_clear_search_enhanced(self, event: Button.Pressed) -> None:
        """Clear the search input and reset the directory filter."""
        event.stop()
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
        except Exception:
            pass
        self.search_active = False
        try:
            self.query_one(SearchableDirectoryNavigation).search_filter = ""
        except Exception:
            pass

    @on(SearchableDirectoryNavigation.SearchCountChanged)
    def _on_search_count_changed(self, event: SearchableDirectoryNavigation.SearchCountChanged) -> None:
        """Update the search status label and no-match notice."""
        try:
            status = self.query_one("#search-status", Label)
            no_match = self.query_one("#search-no-match", Static)
            if event.query:
                status.update(f"{event.count} result{'s' if event.count != 1 else ''}")
                if event.count == 0:
                    no_match.update(f"No files match '{event.query}'")
                    no_match.styles.display = "block"
                else:
                    no_match.styles.display = "none"
            else:
                status.update("")
                no_match.styles.display = "none"
        except Exception:
            pass

    @on(SearchableDirectoryNavigation.ToggleSelection)
    def _on_toggle_selection(self, event: SearchableDirectoryNavigation.ToggleSelection) -> None:
        """Toggle selection for the currently highlighted file."""
        self.action_toggle_selection()

    def dismiss(self, result: Optional[Union[Path, List[Path]]]) -> None:
        """Override dismiss to save recent location(s) and last directory."""
        if isinstance(result, list):
            for path in result:
                self.recent_locations.add(
                    path, "file" if path.is_file() else "directory"
                )
        elif result:
            self.recent_locations.add(
                result, "file" if result.is_file() else "directory"
            )
            self._save_last_directory(result)

        try:
            dir_nav = self.query_one(SearchableDirectoryNavigation)
            self._save_last_directory(dir_nav.location)
        except Exception:
            pass

        super().dismiss(result)


class EnhancedFileOpen(EnhancedFileDialog):
    """Enhanced file open dialog with bookmarks and recent files"""

    ERROR_FILE_MUST_EXIST = "The file must exist"

    def __init__(
        self,
        location: Union[str, Path] = ".",
        title: str = "Open File",
        *,
        filters: Optional[Union[Filters, List[str], Tuple[str, ...]]] = None,
        must_exist: bool = True,
        multi_select: bool = False,
        context: str = "file_open",
        select_button: str = "Open",
        cancel_button: str = "Cancel",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            location=location,
            title=title,
            select_button=select_button,
            cancel_button=cancel_button,
            filters=filters,
            context=context,
            multi_select=multi_select,
            id=id,
            classes=classes,
            name=name,
        )
        self.must_exist = must_exist
        self.multi_select = multi_select

    def _should_return(self, candidate: Path) -> bool:
        """Final check on a picked file before returning it."""
        if self.must_exist and not candidate.exists():
            self._set_error(self.ERROR_FILE_MUST_EXIST)
            return False
        return True

    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file selection"""
        from textual.widgets import Input, Select

        if not self.multi_select:
            yield Input(placeholder=self._filename_placeholder(), id="filename-input")
        if self.filters:
            yield Select(
                self.filters.selections,
                prompt="File type",
                value=0,
                id="file-filter"
            )


class EnhancedFileSave(EnhancedFileDialog):
    """Enhanced file save dialog with bookmarks and recent files"""

    def __init__(
        self,
        location: Union[str, Path] = ".",
        title: str = "Save File",
        *,
        filters: Optional[Union[Filters, List[str], Tuple[str, ...]]] = None,
        default_filename: str = "",
        context: str = "file_save",
        select_button: str = "Save",
        cancel_button: str = "Cancel",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            location=location,
            title=title,
            select_button=select_button,
            cancel_button=cancel_button,
            filters=filters,
            default_file=default_filename,
            context=context,
            id=id,
            classes=classes,
            name=name,
        )
        self.default_filename = default_filename

    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file saving"""
        from textual.widgets import Input, Select

        yield Input(
            value=self.default_filename,
            placeholder=self._filename_placeholder(),
            id="filename-input",
        )
        if self.filters:
            yield Select(
                self.filters.selections,
                prompt="File type",
                value=0,
                id="file-filter"
            )
