"""The app-integrated base dialog code for the other picker dialogs.

This vendored fork intentionally uses the application's safe-modal contract and
therefore requires a small patch when syncing from or extracting to upstream.
"""

##############################################################################
# Backward compatibility.
from __future__ import annotations

##############################################################################
# Python imports.
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeAlias, Union

##############################################################################
# Third-party imports.
from rich.console import RenderableType
from rich.table import Table
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Input, ListView, ListItem, Static

##############################################################################
# Local imports.
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .parts import DirectoryNavigation, DriveNavigation
from .path_maker import MakePath


##############################################################################
def resolve_default_location(location: Union[str, Path]) -> Union[str, Path]:
    """Swap the bare "." default for the user's home directory.

    A raw ``FileSystemPickerScreen`` subclass with no other location logic
    of its own (``FileOpen``, ``SelectDirectory``) otherwise opens wherever
    the process happened to be launched from -- an implementation detail
    no user picked (task-32122 Step 4: "Import once" opened at the process
    cwd). Scoped to those two callers rather than ``FileSystemPickerScreen.
    __init__`` itself: ``EnhancedFileDialog`` already resolves its own
    start location (a per-context remembered directory, see
    ``resolve_file_picker_start`` in ``enhanced_file_picker.py``) before
    calling super().__init__, and several of ITS tests rely on a literal
    "." reaching ``DirectoryNavigation`` unchanged (it means "the actual
    process cwd" there, e.g. a ``monkeypatch.chdir(tmp_path)`` fixture) --
    resolving "." at the shared base broke those.

    Args:
        location: The caller's requested starting location, ``str`` or
            ``Path``. Only the bare current-directory spelling (``"."``,
            the ``__init__`` default) is special-cased.

    Returns:
        ``Path.home()`` when ``location`` is exactly ``"."``; otherwise
        ``location`` unchanged, same type and value as passed in (a
        non-default string stays a string, a ``Path`` stays a ``Path``).
    """
    return Path.home() if Path(location) == Path(".") else location


##############################################################################
def resolve_typed_directory(value: str, current: Path) -> Union[Path, str]:
    """Resolve a typed field value to an absolute directory, or an error.

    The single validation block shared by every folder-picker Select action:
    Enter-to-navigate and Select-to-confirm across ``EnhancedSelectDirectory``
    (``enhanced_file_picker.py``), the vendored ``SelectDirectory``, and
    ``FileOpen(offer_select_folder=True)`` all call this one function
    (task-32122 round 2) instead of keeping three copies that had already
    drifted -- one resolved a relative path against the process cwd instead
    of the browsed directory, one dropped the NUL-byte guard, and the
    caught-exception width differed between them.

    Args:
        value: The raw field text, used exactly as typed -- leading and
            trailing spaces are significant and valid in POSIX folder
            names (Qodo review round 4), so only a *whitespace-only* value
            is treated as "nothing typed"; a genuinely padded name is kept
            intact for comparison and path construction. Callers that need
            to first decide whether a non-empty value should even count as
            "typed" (e.g. it merely echoes a just-clicked file's name) do
            that before calling this.
        current: The directory currently being browsed -- the fallback for
            an empty/unchanged value, and the base a relative typed value
            resolves against.

    Returns:
        The resolved absolute ``Path`` when it names an existing directory,
        or an error string ready for ``_set_error`` when it does not.
    """
    if not value.strip() or value == str(current):
        # Unchanged from what's being browsed: return the exact object,
        # not a re-resolved reconstruction (macOS "/tmp" -> "/private/tmp"
        # would otherwise change the result of a plain, no-typing Select).
        return current
    if "\x00" in value:
        return "Path cannot contain null characters."
    try:
        target = MakePath.of(value).expanduser()
        if not target.is_absolute():
            target = current / target
        target = target.resolve()
    except PermissionError:
        # A friendly, errno-free message (matches
        # FileSystemPickerScreen.ERROR_PERMISSION_ERROR) -- the generic
        # `str(error)` branch below would otherwise leak a raw
        # "[Errno 13] Permission denied: ..." to the user.
        return FileSystemPickerScreen.ERROR_PERMISSION_ERROR
    except (RuntimeError, OSError, ValueError) as error:
        return str(error)
    if target.is_dir():
        return target
    if target.exists():
        # A real path that is not a directory is a different mistake than
        # a nonexistent one; the vendored SelectDirectory distinguishes
        # them too.
        return f"Not a directory: {target.name}"
    return f"Path not found: {value}"


##############################################################################
def _listing_column_headers() -> RenderableType:
    """Build the Name / Size / Modified header row for the listing.

    (task-3304, MI-15) The directory listing renders three data columns
    with nothing naming them -- a bare right-aligned number next to a
    timestamp read as noise. The grid mirrors ``DirectoryEntry
    ._as_renderable``'s exact column recipe (pad 1 / icon 3 / name 1fr /
    size 10 / time 20 / pad 1) so the headers land over their columns.
    Known drift: when the list overflows, the OptionList's vertical
    scrollbar shifts the data columns left by its width relative to this
    header; and on Windows the DriveNavigation pane sits left of the
    listing. Both offsets are cosmetic and accepted.
    """
    headers = Table.grid(expand=True)
    headers.add_column(no_wrap=True, width=1)
    headers.add_column(no_wrap=True, justify="left", width=3)
    headers.add_column(no_wrap=True, justify="left", ratio=1)
    headers.add_column(no_wrap=True, justify="right", width=10)
    headers.add_column(no_wrap=True, justify="right", width=20)
    headers.add_column(no_wrap=True, width=1)
    headers.add_row("", "", "Name", "Size", "Modified", "")
    return headers


##############################################################################
class Dialog(Vertical):
    """Layout class for the main dialog area."""


##############################################################################
class InputBar(Horizontal):
    """The input bar area of the dialog."""


##############################################################################
ButtonLabel: TypeAlias = Union[str, Callable[[str], str]]
"""The type for a button label value."""


##############################################################################
class FileSystemPickerScreen(SafeModalDismissMixin, ModalScreen[Path | None]):
    """Base screen for the dialogs in this library."""

    SAFE_MODAL_CONTENT = "#file-system-picker-dialog"

    DEFAULT_CSS = """
    FileSystemPickerScreen {
        align: center middle;

        Dialog {
            width: 80%;
            height: 80%;
            border: $border;
            background: $panel;
            border-title-color: $text;
            border-title-background: $panel;
            border-subtitle-color: $text;
            border-subtitle-background: $error;

            OptionList, OptionList:focus {
                background: $panel;
                background-tint: $panel;
            }
        }

        #current_path_display {
            width: 1fr;
            padding: 0 1;
            margin-bottom: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            color: $text-muted;
        }
        
        #path-breadcrumbs {
            height: 3;
            padding: 1;
            background: $surface;
            margin-bottom: 1;
        }
        
        #path-breadcrumbs Button {
            min-width: 0;
            padding: 0 1;
            margin: 0;
            height: 1;
            background: transparent;
            border: none;
        }
        
        #path-breadcrumbs Button:hover {
            background: $boost;
        }
        
        #clear-search {
            background: $boost;
            border: none;
        }
        
        #path-breadcrumbs .breadcrumb-separator {
            margin: 0 1;
            color: $text-muted;
        }
        
        #recent-locations {
            height: 10;
            border: solid $primary;
            background: $surface;
            margin-bottom: 1;
            display: none;
        }
        
        #recent-locations.visible {
            display: block;
        }
        
        #search-container {
            height: 3;
            padding: 0 1;
            margin-bottom: 1;
            display: none;
        }
        
        #search-container.visible {
            display: block;
        }
        
        #search-input {
            width: 1fr;
        }

        /* task-3304 (MI-15) / task-14825 #5: the header cells must land on
           their own data columns. The listing insets its content THREE
           ways, and the original padding compensated for only the first:
             1. DirectoryNavigation's `border: blank`      -> 1 cell each side
             2. OptionList's own default `padding: 0 1`    -> 1 cell each side
             3. the vertical scrollbar, on the right only  -> 2 cells
           (2) cost a permanent one-cell skew (the `Size` header ended at
           col 188 over values ending at 186 in the live capture) and (3)
           added two more the moment the list overflowed -- shipped as
           "cosmetic and accepted", which reads as a broken table. The
           scrollbar half is made deterministic by reserving the gutter
           below rather than by guessing whether it is showing. */
        #file-dialog-column-headers {
            height: 1;
            padding: 0 4 0 2;
            color: $text-muted;
        }

        DirectoryNavigation {
            height: 1fr;
            /* Reserve the scrollbar column whether or not it is showing:
               without this the whole listing shifts 2 cells left as soon
               as a directory overflows, and no static header can be right
               in both states. */
            scrollbar-gutter: stable;
        }

        InputBar {
            height: auto;
            align: right middle;
            padding-top: 1;
            padding-right: 1;
            padding-bottom: 1;
            Button {
                margin-left: 1;
            }
        }

        /* task-32122 Step 3: SelectDirectory/FileOpen(offer_select_folder)
           had no on-screen Enter-vs-Select hint at all -- only the
           EnhancedFileDialog family had one, and neither of those two
           pickers uses it. Empty text (the default -- see _hint_text())
           renders as a blank line, so compose() only yields this when a
           dialog overrides _hint_text() to return something. */
        #picker-hint-line {
            height: 1;
            padding: 0 1;
            color: $text-muted;
            text-style: italic;
        }
    }
    """

    ERROR_PERMISSION_ERROR = "Permission error"
    """Error to tell there user there was a problem with permissions."""

    BINDINGS = [
        Binding("full_stop", "hidden", "Toggle hidden"),
        Binding("ctrl+h", "hidden", "Toggle hidden files"),
        Binding("ctrl+l", "focus_path_input", "Edit path directly"),
        Binding("f5", "refresh", "Refresh directory"),
        Binding("ctrl+d", "bookmark_current", "Bookmark directory"),
        Binding("ctrl+r", "show_recent", "Show recent locations"),
        Binding("ctrl+f", "focus_search", "Search in directory"),
        Binding("escape", "request_safe_cancel", "Cancel"),
        Binding("ctrl+s", "select_current_folder", "Select this folder"),
    ]
    """The bindings for the dialog."""

    show_recent = reactive(False)
    """Whether to show recent locations panel."""

    search_active = reactive(False)
    """Whether search is active."""

    def __init__(
        self,
        location: str | Path = ".",
        title: str = "",
        select_button: ButtonLabel = "",
        cancel_button: ButtonLabel = "",
    ) -> None:
        """Initialise the dialog.

        Args:
            location: Optional starting location.
            title: Optional title.
            select_button: Label or format function for the select button.
            cancel_button: Label or format function for the cancel button.
        """
        super().__init__()
        self._location = location
        """The starting location."""
        self._title = title
        """The title for the dialog."""
        self._select_button = select_button
        """The text prompt for the select button, or a function to format it."""
        self._cancel_button = cancel_button
        """The text prompt for the cancel button, or a function to format it."""
        self._recent_locations: List[Dict[str, Any]] = []
        """Recent file/directory locations."""

    def _input_bar(self) -> ComposeResult:
        """Provide any widgets for the input bar, before the buttons."""
        yield from ()

    def _hint_text(self) -> str:
        """One-line Enter-vs-Select hint shown above the input bar.

        Empty by default: most ``FileSystemPickerScreen`` dialogs (a plain
        file-only ``FileOpen``, ``FileSave``) have no folder-selection
        ambiguity to clarify, and this line simply isn't yielded in
        ``compose()`` when it returns "" -- so callers unrelated to
        task-32122 (character import, skill-folder import, TTS model
        directories, ...) render exactly as before. ``SelectDirectory`` and
        ``FileOpen(offer_select_folder=True)`` override this: both let
        Enter descend into a directory while a separate action confirms
        "use this one", and neither dialog had any on-screen hint for that
        at all.
        """
        return ""

    @staticmethod
    def _label(label: ButtonLabel, default: str) -> str:
        """Create a label for use with a button.

        Args:
            label: The label value for the button.
            default: The default label for the button.

        Returns:
            The formatted label.
        """
        # If the label is callable, then call it with the default as a
        # parameter; otherwise use it as-is as it'll be a string.
        return label(default) if callable(label) else label or default

    def compose(self) -> ComposeResult:
        """Compose the child widgets.

        Returns:
            The widgets to compose.
        """
        with Dialog(id="file-system-picker-dialog") as dialog:
            dialog.border_title = self._title

            # Recent locations panel (hidden by default)
            with VerticalScroll(id="recent-locations"):
                yield Label("Recent Locations", classes="section-title")
                yield ListView(id="recent-list")

            # Path display and breadcrumbs
            yield Label(id="current_path_display")
            with Horizontal(id="path-breadcrumbs"):
                # Breadcrumbs will be dynamically populated
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

            # Column headers for the listing below (task-3304, MI-15).
            yield Static(
                _listing_column_headers(), id="file-dialog-column-headers"
            )

            # Main directory navigation
            with Horizontal():
                if sys.platform == "win32":
                    yield DriveNavigation(self._location)
                yield DirectoryNavigation(self._location)

            if hint_text := self._hint_text():
                yield Static(hint_text, id="picker-hint-line")

            # Input bar with buttons
            with InputBar():
                yield from self._input_bar()
                yield Button(self._label(self._select_button, "Select"), id="select")
                # (task-2222) Opt-in folder affordance: a file picker whose
                # caller also accepts a directory can offer "this folder"
                # without a second dialog. Off by default, so every other
                # caller's bar is unchanged.
                if getattr(self, "_offer_select_folder", False):
                    yield Button(
                        "Select folder", id="select-current-folder"
                    )
                yield Button(self._label(self._cancel_button, "Cancel"), id="cancel")

    def on_mount(self) -> None:
        """Focus the initial widget on mount and set the initial path."""
        dir_nav = self.query_one(DirectoryNavigation)
        current_path_label = self.query_one("#current_path_display", Label)
        current_path_label.update(str(dir_nav.location))

        # Initialize breadcrumbs
        self._update_breadcrumbs(dir_nav.location)

        # Load recent locations
        self._load_recent_locations()

        self._focus_initial_widget()

    def _focus_initial_widget(self) -> None:
        """Focus whichever widget should hold focus right after mounting.

        Defaults to the directory listing. Subclasses override this to
        steer initial focus elsewhere -- e.g. ``FileSave`` (file_save.py)
        focuses its filename input instead, so a keyboard user can press
        Enter immediately to confirm the seeded default filename rather
        than have Enter activate the highlighted directory row (usually
        ``..``) (task-1479).

        This is a plain method call, not a message handler: overriding it
        resolves via normal Python MRO, unlike Textual's ``on_mount``/`@on`
        dispatch, which invokes a handler defined on *every* class in the
        MRO rather than just the most-derived one -- a subclass adding its
        own ``on_mount`` here would run *before*, not instead of, this
        class's own ``on_mount`` (dispatch order walks the MRO
        most-derived-first, so a naming-convention override defined earlier
        in the walk fires and then gets clobbered by this method's own
        ``dir_nav.focus()`` call afterwards).
        """
        self.query_one(DirectoryNavigation).focus()

    def _set_error(self, message: str = "") -> None:
        """Set or clear the error message.

        Args:
            message: Optional message to show as an error.
        """
        self.query_one(Dialog).border_subtitle = message

    @on(DriveNavigation.DriveSelected)
    def _change_drive(self, event: DriveNavigation.DriveSelected) -> None:
        """Reload DirectoryNavigation in response to drive change."""
        """Reload DirectoryNavigation in response to drive change."""
        dir_nav = self.query_one(DirectoryNavigation)
        dir_nav.location = event.drive_root

    @on(DirectoryNavigation.Changed)
    def _on_directory_changed(self, event: DirectoryNavigation.Changed) -> None:
        """Clear any error and update the path display."""
        self._set_error()
        current_path_label = self.query_one("#current_path_display", Label)
        current_path_label.update(str(event.control.location))

        # Update breadcrumbs
        self._update_breadcrumbs(event.control.location)

        # Add to recent locations
        self._add_to_recent(event.control.location, "directory")

    def _clear_error(self) -> None:
        """Clear any error that might be showing."""
        self._set_error()

    @on(DirectoryNavigation.PermissionError)
    def _show_permission_error(self) -> None:
        """Show any permission error bubbled up from the directory navigator."""
        self._set_error(self.ERROR_PERMISSION_ERROR)

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Hide the folder shortcut on dialogs that do not offer it.

        (task-2222 Qodo round) The binding is declared on the shared base,
        so without this every picker advertised a ctrl+s that did nothing
        -- including in the F1 help. Returning None removes it from both
        the key map and the listing.

        Args:
            action: The action name being checked.
            parameters: The action's parameters.

        Returns:
            ``None`` to hide the folder action when this dialog does not
            offer it; otherwise the base class's decision.
        """
        if action == "select_current_folder" and not getattr(
            self, "_offer_select_folder", False
        ):
            return None
        return super().check_action(action, parameters)

    def _resolve_select_folder_target(self) -> Union[Path, str]:
        """Resolve the input bar's typed field, else the directory being viewed.

        Used to always dismiss with ``DirectoryNavigation.location`` -- the
        directory merely being browsed -- silently discarding a path the
        user typed into the "File name" field but never pressed Enter on
        (task-32122 AC#1). The field lives in ``BaseFileDialog._input_bar``
        (``file_dialog.py``), not here, so it's read generically via
        ``InputBar``'s one ``Input`` child rather than importing that
        module (this base is shared by ``SelectDirectory`` too, which has
        its own differently-shaped input bar and never sets
        ``_offer_select_folder``, so this path never runs for it).

        Returns:
            The resolved absolute ``Path`` when the field is empty or
            names an existing directory, or an error string ready for
            ``_set_error`` when it names something else.
        """
        dir_nav = self.query_one(DirectoryNavigation)
        try:
            field = self.query_one(InputBar).query_one(Input)
            raw_value = field.value
        except Exception:
            raw_value = ""
        value = raw_value.strip()
        if value and raw_value == getattr(self, "_select_folder_click_fill", None):
            # A single click on a file fills this field with its basename
            # (file_dialog.py's _select_file, so a click-then-Open flow
            # works) -- that is not the user typing a folder path, so
            # Select folder must not mistake it for one and error with
            # "Not a directory: <file>" (review round 2, Important 1).
            # ``_select_folder_click_fill`` (set/cleared by _select_file /
            # _update_field_label) is the click-provenance flag itself --
            # NOT "does the value match whatever's highlighted": that
            # earlier approach falsely swallowed a typed, never-clicked
            # ".." too, since DirectoryNavigation defaults `highlighted`
            # to 0 on every repopulate and ".." is always option 0 in a
            # non-root directory (review round 3).
            value = ""
        return resolve_typed_directory(value, dir_nav.location)

    def _confirm_select_folder(self) -> None:
        """Resolve the typed field first, then the directory being viewed.

        "Open" keeps descending into directories; this returns the one on
        screen (or the one typed), which is how every OS folder picker
        behaves (task-2222, task-32122).
        """
        result = self._resolve_select_folder_target()
        if isinstance(result, Path):
            self.dismiss(result)
            return
        self._set_error(result)
        try:
            self.query_one(InputBar).query_one(Input).focus()
        except Exception:
            pass

    def action_select_current_folder(self) -> None:
        """Keyboard route to the folder affordance (task-2222)."""
        if getattr(self, "_offer_select_folder", False):
            self._confirm_select_folder()

    @on(Button.Pressed, "#select-current-folder")
    def _select_current_folder(self, event: Button.Pressed) -> None:
        """Handle the "Select folder" button (task-2222).

        Args:
            event: The button press event.
        """
        event.stop()
        self._confirm_select_folder()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Peel transient surfaces for Escape, otherwise cancel immediately."""
        if source != "escape":
            self.dismiss_safe_once(None)
            return

        path_container = self.query_one("#path-input-container")
        if not path_container.has_class("hidden"):
            self._on_cancel_path_input()
            return

        if self.search_active:
            self.query_one("#search-input", Input).value = ""
            self.search_active = False
            self.query_one(DirectoryNavigation).search_filter = ""
            self.query_one(DirectoryNavigation).focus()
            return

        if self.show_recent:
            self.show_recent = False
            self.query_one(DirectoryNavigation).focus()
            return

        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        """Cancel the dialog.

        Args:
            event: The even to handle.
        """
        event.stop()
        await self.request_safe_cancel(source="visible")

    def _action_hidden(self) -> None:
        """Action for toggling the display of hidden entries."""
        self.query_one(DirectoryNavigation).toggle_hidden()
        self.notify("Hidden files toggled", timeout=2)

    def action_focus_path_input(self) -> None:
        """Toggle and focus the path input field."""
        try:
            path_container = self.query_one("#path-input-container")
            path_input = self.query_one("#path-input", Input)

            # Toggle visibility
            if path_container.has_class("hidden"):
                path_container.remove_class("hidden")
                # Set current path as the initial value
                dir_nav = self.query_one(DirectoryNavigation)
                path_input.value = str(dir_nav.location)
                path_input.focus()
                # Select all text in the input
                path_input.selection = (0, len(path_input.value))
            else:
                path_container.add_class("hidden")
                # Return focus to directory navigation
                self.query_one(DirectoryNavigation).focus()
        except Exception as e:
            self.notify(f"Error toggling path input: {e}", severity="error", timeout=2)

    def action_refresh(self) -> None:
        """Refresh the current directory listing."""
        dir_nav = self.query_one(DirectoryNavigation)
        # Force refresh by resetting location
        current = dir_nav.location
        dir_nav.location = current
        self.notify("Directory refreshed", timeout=2)

    def action_bookmark_current(self) -> None:
        """Bookmark the current directory."""
        dir_nav = self.query_one(DirectoryNavigation)
        current_path = dir_nav.location
        # This would need to be implemented with proper bookmark storage
        self.notify(f"Bookmarked: {current_path.name}", timeout=2)

    def action_show_recent(self) -> None:
        """Toggle the recent locations panel."""
        self.show_recent = not self.show_recent

    def action_focus_search(self) -> None:
        """Toggle search mode and focus search input."""
        self.search_active = not self.search_active
        if self.search_active:
            try:
                search_input = self.query_one("#search-input", Input)
                search_input.focus()
            except Exception:
                pass

    def _update_breadcrumbs(self, path: Path) -> None:
        """Update breadcrumb navigation."""
        try:
            breadcrumb_container = self.query_one("#path-breadcrumbs", Horizontal)
            breadcrumb_container.remove_children()

            parts = path.parts
            for i, part in enumerate(parts):
                partial_path = Path(*parts[: i + 1])

                # Create button for each path component
                btn = Button(part, variant="default", classes="breadcrumb-btn")
                btn.tooltip = str(partial_path)  # Store full path in tooltip
                breadcrumb_container.mount(btn)

                # Add separator if not last
                if i < len(parts) - 1:
                    breadcrumb_container.mount(
                        Label("/", classes="breadcrumb-separator")
                    )
        except Exception:
            # Silently fail if breadcrumbs can't be updated
            pass

    def _load_recent_locations(self) -> None:
        """Load recent locations from storage."""
        # This is a placeholder - in real implementation,
        # this would load from a config file or database
        try:
            recent_list = self.query_one("#recent-list", ListView)
            recent_list.clear()

            # Add some example recent locations
            for path in self._get_recent_paths():
                item = ListItem(Label(str(path)))
                item.data = path  # Store path in data attribute
                recent_list.append(item)
        except Exception:
            pass

    def _get_recent_paths(self) -> List[Path]:
        """Get list of recent paths."""
        # Placeholder - would load from persistent storage
        return []

    def _add_to_recent(self, path: Path, file_type: str) -> None:
        """Add a path to recent locations."""
        # Placeholder - would save to persistent storage
        pass

    @on(Button.Pressed, ".breadcrumb-btn")
    def _on_breadcrumb_click(self, event: Button.Pressed) -> None:
        """Handle breadcrumb navigation clicks."""
        if event.button.tooltip:
            try:
                path = Path(event.button.tooltip)
                dir_nav = self.query_one(DirectoryNavigation)
                dir_nav.location = path
            except Exception:
                pass

    @on(ListView.Selected, "#recent-list")
    def _on_recent_selected(self, event: ListView.Selected) -> None:
        """Handle selection from recent locations."""
        if hasattr(event.item, "data") and event.item.data:
            try:
                path = Path(event.item.data)
                if path.exists():
                    dir_nav = self.query_one(DirectoryNavigation)
                    if path.is_dir():
                        dir_nav.location = path
                    else:
                        dir_nav.location = path.parent
                    self.show_recent = False
            except Exception:
                pass

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        try:
            dir_nav = self.query_one(DirectoryNavigation)
            dir_nav.search_filter = event.value
        except Exception:
            pass

    @on(Button.Pressed, "#clear-search")
    def _on_clear_search(self) -> None:
        """Clear the search input."""
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            self.search_active = False
        except Exception:
            pass

    def watch_show_recent(self, show: bool) -> None:
        """React to show_recent changes."""
        try:
            recent_panel = self.query_one("#recent-locations")
            recent_panel.set_class(show, "visible")
        except Exception:
            pass

    def watch_search_active(self, active: bool) -> None:
        """React to search_active changes."""
        try:
            search_container = self.query_one("#search-container")
            search_container.set_class(active, "visible")
        except Exception:
            pass

    @on(Button.Pressed, "#go-to-path")
    @on(Input.Submitted, "#path-input")
    def _on_path_input_submit(self, event=None) -> None:
        """Handle path input submission."""
        try:
            path_input = self.query_one("#path-input", Input)
            path_str = path_input.value.strip()

            if not path_str:
                return

            # Expand user home directory if needed
            if path_str.startswith("~"):
                path = Path(path_str).expanduser()
            else:
                path = Path(path_str)

            # Make path absolute if it's relative
            if not path.is_absolute():
                dir_nav = self.query_one(DirectoryNavigation)
                path = dir_nav.location / path

            # Resolve the path
            path = path.resolve()

            # Check if path exists
            if not path.exists():
                self.notify(f"Path does not exist: {path}", severity="error", timeout=3)
                return

            # Navigate to the path
            dir_nav = self.query_one(DirectoryNavigation)
            if path.is_dir():
                dir_nav.location = path
            else:
                # If it's a file, navigate to its parent directory AND highlight
                # the file in the list, so a keyboard user who typed a full path
                # lands on the file instead of on '..' (TASK-378).
                dir_nav.show_and_highlight(path)

            # Hide the path input
            path_container = self.query_one("#path-input-container")
            path_container.add_class("hidden")
            dir_nav.focus()

        except Exception as e:
            self.notify(f"Error navigating to path: {e}", severity="error", timeout=3)

    @on(Button.Pressed, "#cancel-path-input")
    def _on_cancel_path_input(self) -> None:
        """Cancel path input and hide the container."""
        try:
            path_container = self.query_one("#path-input-container")
            path_container.add_class("hidden")
            self.query_one(DirectoryNavigation).focus()
        except Exception:
            pass


### base_dialog.py ends here
