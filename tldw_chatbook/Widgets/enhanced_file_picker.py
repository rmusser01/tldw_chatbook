# enhanced_file_picker.py
# Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, bookmarks, and search

import sys
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union
from datetime import datetime
import os
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll

from textual.widgets import Button, Label, ListView, ListItem, Input
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

from ..Third_Party.textual_fspicker import Filters
from ..Third_Party.textual_fspicker.file_dialog import BaseFileDialog
from ..Third_Party.textual_fspicker.base_dialog import FileSystemPickerScreen
from ..Third_Party.textual_fspicker.parts import DirectoryNavigation
from ..Third_Party.textual_fspicker.parts.directory_navigation import DirectoryEntry
from ..Third_Party.textual_fspicker.path_maker import MakePath
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


class SearchableDirectoryNavigation(DirectoryNavigation):
    """Directory navigation that also filters by a free-text search term.

    The base ``DirectoryNavigation`` references ``search_filter`` in its watch
    method but does not declare the reactive variable, and its
    ``_repopulate_display`` does not apply the term. This subclass adds the
    missing reactive and applies it without editing third-party code.
    """

    search_filter = reactive("")
    """Free-text filter applied to entry names."""

    def _repopulate_display(self) -> None:
        """Repopulate the display, honouring file, hidden, and search filters."""
        styles = self._styles
        query = self.search_filter.strip().lower()
        with self.app.batch_update():
            self.clear_options()
            if not self.is_root:
                self.add_option(DirectoryEntry(self._location / "..", styles))
            self.add_options(
                self._sort(
                    entry
                    for entry in self._entries
                    if not self.hide(entry.location)
                    and (not query or query in entry.location.name.lower())
                )
            )
        self._settle_highlight()


class EnhancedFileDialog(BaseFileDialog):
    """Enhanced file picker with keyboard shortcuts, recent files, breadcrumbs, bookmarks, and search"""

    DEFAULT_CSS = BaseFileDialog.DEFAULT_CSS + """
    BaseFileDialog {
        .hidden {
            display: none;
        }

        Dialog {
            height: 90%;
            width: 90%;
        }

        #recent-locations, #bookmarks-panel {
            height: 10;
            border: solid $primary;
            background: $surface;
            margin-bottom: 1;
            display: none;
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

        #bookmarks-list {
            layout: grid;
            grid-size: 2;
            grid-columns: 1fr 1fr;
            grid-gutter: 1;
            height: 8;
            overflow-y: auto;
        }

        #bookmarks-list ListItem {
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
            height: 100%;
            align: left middle;
        }

        #path-breadcrumbs {
            height: 3;
            padding: 0 1;
            background: $surface;
            border-bottom: tall $primary-lighten-1;
            overflow: hidden;
        }

        #path-breadcrumbs .breadcrumb-btn {
            min-width: 0;
            padding: 0 1;
            margin: 0;
            height: 1;
            background: transparent;
            border: none;
            color: $text;
            text-style: none;
        }

        #path-breadcrumbs .breadcrumb-btn:hover {
            background: $primary 20%;
            text-style: underline;
        }

        #path-breadcrumbs .breadcrumb-separator {
            margin: 0;
            padding: 0 1;
            color: $text-muted;
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
        Binding("ctrl+b", "toggle_bookmarks", "Show bookmarks"),
        *[Binding(str(n), f"jump_bookmark_{n}", f"Bookmark {n}", show=False) for n in range(1, 10)],
    ]

    show_bookmarks = reactive(False)

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
        filters: Optional[Filters] = None,
        default_file: Optional[Union[str, Path]] = None,
        context: str = "default",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        # ``BaseFileDialog`` does not accept Textual screen kwargs such as
        # ``id`` or ``classes``; apply them manually after the base init.
        super().__init__(
            location,
            title,
            select_button,
            cancel_button,
            filters=filters,
            default_file=default_file,
        )
        if id is not None:
            self.id = id
        if classes is not None:
            self.classes = classes
        if name is not None:
            self.name = name
        self.context = context
        self.recent_locations = RecentLocations(context=context)
        self.bookmarks_manager = BookmarksManager(context=context)
        self._last_directory = self._get_last_directory()
        if self._last_directory is not None:
            self._location = self._last_directory

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

            # Recent locations panel (hidden by default)
            with VerticalScroll(id="recent-locations"):
                yield Label("📋 Recent Locations", classes="section-title")
                yield ListView(id="recent-list")

            # Bookmarks panel (hidden by default)
            with VerticalScroll(id="bookmarks-panel"):
                with Horizontal(classes="bookmarks-header"):
                    yield Label("⭐ Bookmarks", classes="section-title")
                    yield Button("➕", id="add-bookmark", classes="bookmark-button")
                yield ListView(id="bookmarks-list")

            # Path display and breadcrumbs
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

            # Main directory navigation
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(self._location)
                yield SearchableDirectoryNavigation(self._location)

            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")

    def on_mount(self) -> None:
        """Initialize the dialog on mount.

        The base ``on_mount`` expects ``#path-breadcrumbs`` and
        ``#recent-list`` to exist; our compose provides them, so calling
        ``super().on_mount()`` is safe.
        """
        super().on_mount()
        self._update_bookmarks_list()
        self._update_bookmark_button_state(
            self.query_one(SearchableDirectoryNavigation).location
        )

    def watch_show_recent(self, show: bool) -> None:
        """Toggle recent locations visibility."""
        try:
            recent_panel = self.query_one("#recent-locations")
            recent_panel.set_class(show, "visible")
            if show:
                self.show_bookmarks = False
        except Exception:
            pass

    def action_show_recent(self) -> None:
        """Toggle the recent locations panel (explicit override)."""
        self.show_recent = not self.show_recent

    def watch_show_bookmarks(self, show: bool) -> None:
        """Toggle bookmarks panel visibility."""
        try:
            bookmarks_panel = self.query_one("#bookmarks-panel")
            bookmarks_panel.set_class(show, "visible")
            if show:
                self.show_recent = False
        except Exception:
            pass

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

        Mirrors ``BaseFileDialog._select_file`` but targets the dedicated
        ``#filename-input`` so the hidden ``#path-input`` and
        ``#search-input`` are not selected by ``query_one(Input)``.
        """
        try:
            file_name = self.query_one("#filename-input", Input)
        except Exception:
            return
        file_name.value = str(event.path.name)
        file_name.focus()

    def _confirm_file(self, event: Input.Submitted | Button.Pressed) -> None:
        """No-op override of ``BaseFileDialog._confirm_file``.

        The real confirmation handling happens in ``_on_confirm_file``. The
        base decorated handler is suppressed via ``_get_dispatch_methods``.
        """
        pass

    @on(Input.Submitted, "#filename-input")
    @on(Button.Pressed, "#select")
    def _on_confirm_file(self, event: Input.Submitted | Button.Pressed) -> None:
        """Confirm the selection of the file in the input box.

        Mirrors ``BaseFileDialog._confirm_file`` but targets the dedicated
        ``#filename-input`` so the hidden ``#path-input`` and
        ``#search-input`` are not selected by ``query_one(Input)``.
        """
        event.stop()
        file_name = self.query_one("#filename-input", Input)

        # Only even try and process this if there's some input.
        if not file_name.value:
            self._set_error(self.ERROR_A_FILE_MUST_BE_CHOSEN)
            return

        # If it looks like the user is typing in some sort of home
        # directory path...
        if file_name.value.startswith("~"):
            # ...let's simply expand and go with that.
            try:
                chosen = MakePath.of(file_name.value).expanduser()
            except RuntimeError as error:
                self._set_error(str(error))
                return
        else:
            # It's not a home directory path, so let's combine with the
            # location of the directory navigator widget.
            chosen = (
                self.query_one(DirectoryNavigation).location / file_name.value
            ).resolve()

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

    def _update_breadcrumbs(self, path: Path) -> None:
        """Update breadcrumb navigation using the base container."""
        try:
            breadcrumb_container = self.query_one("#path-breadcrumbs", Horizontal)
            breadcrumb_container.remove_children()

            parts = path.parts
            for i, part in enumerate(parts):
                partial_path = Path(*parts[:i+1])

                btn = Button(part, variant="default", classes="breadcrumb-btn")
                btn.tooltip = str(partial_path)
                breadcrumb_container.mount(btn)

                if i < len(parts) - 1:
                    breadcrumb_container.mount(Label("/", classes="breadcrumb-separator"))
        except Exception:
            pass

    def _load_recent_locations(self) -> None:
        """Populate the recent locations list from persistent storage."""
        try:
            recent_list = self.query_one("#recent-list", ListView)
            recent_list.clear()

            for item in self.recent_locations.get_recent():
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

            for bookmark in self.bookmarks_manager.get_bookmarks():
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
            path = Path(bookmarks[index]["path"])
            if path.exists():
                dir_nav = self.query_one(SearchableDirectoryNavigation)
                dir_nav.location = path
                self._update_bookmark_button_state(path)
                self.notify(f"Jumped to: {bookmarks[index]['name']}", timeout=1)
            else:
                self.notify(f"Path no longer exists: {path}", severity="warning")

    def action_jump_bookmark_1(self) -> None:
        self._jump_to_bookmark(0)

    def action_jump_bookmark_2(self) -> None:
        self._jump_to_bookmark(1)

    def action_jump_bookmark_3(self) -> None:
        self._jump_to_bookmark(2)

    def action_jump_bookmark_4(self) -> None:
        self._jump_to_bookmark(3)

    def action_jump_bookmark_5(self) -> None:
        self._jump_to_bookmark(4)

    def action_jump_bookmark_6(self) -> None:
        self._jump_to_bookmark(5)

    def action_jump_bookmark_7(self) -> None:
        self._jump_to_bookmark(6)

    def action_jump_bookmark_8(self) -> None:
        self._jump_to_bookmark(7)

    def action_jump_bookmark_9(self) -> None:
        self._jump_to_bookmark(8)

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
        if hasattr(event.item, 'data'):
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

    def dismiss(self, result: Optional[Path]) -> None:
        """Override dismiss to save recent location and last directory."""
        if result:
            self.recent_locations.add(result, "file" if result.is_file() else "directory")
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
        filters: Optional[Filters] = None,
        must_exist: bool = True,
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
            id=id,
            classes=classes,
            name=name,
        )
        self.filters = filters
        self.must_exist = must_exist

    def _should_return(self, candidate: Path) -> bool:
        """Final check on a picked file before returning it."""
        if self.must_exist and not candidate.exists():
            self._set_error(self.ERROR_FILE_MUST_EXIST)
            return False
        return True

    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file selection"""
        from textual.widgets import Input, Select

        yield Input(placeholder="File name...", id="filename-input")
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
        filters: Optional[Filters] = None,
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
        self.filters = filters
        self.default_filename = default_filename

    def _input_bar(self) -> ComposeResult:
        """Provide input widgets for file saving"""
        from textual.widgets import Input, Select

        yield Input(value=self.default_filename, placeholder="File name...", id="filename-input")
        if self.filters:
            yield Select(
                self.filters.selections,
                prompt="File type",
                value=0,
                id="file-filter"
            )
