"""Small keyboard-first dialogs for Database Note folder mutations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from ...Utils.input_validation import escape_markup
from ..modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from ...Notes.note_folder_models import NoteFolderChildPage

#: How many folders one picker page shows. The Library's loader asks the
#: scope service for exactly this many (TASK-34000.28, N-23).
FOLDER_TARGET_PAGE_SIZE = 20

#: ``(query, offset)`` -> one exact page of every active folder matching it.
FolderTargetLoader = Callable[[str, int], Awaitable["NoteFolderChildPage"]]


class LibraryNoteFolderNameDialog(SafeModalDismissMixin, ModalScreen[str | None]):
    """Collect a validated-by-service folder name without transforming it."""

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#library-note-folder-name-dialog"
    AUTO_FOCUS = "#library-note-folder-name"

    BUNDLED_CSS = """
    LibraryNoteFolderNameDialog { align: center middle; }
    LibraryNoteFolderNameDialog > Vertical {
        width: 56; height: auto; padding: 1 2;
        border: tall $accent; background: $surface;
    }
    #library-note-folder-dialog-actions { height: auto; align-horizontal: right; }
    """

    def __init__(self, *, title: str, initial_name: str = "") -> None:
        super().__init__()
        self._title = title
        self._initial_name = initial_name

    def compose(self) -> ComposeResult:
        with Vertical(id="library-note-folder-name-dialog"):
            yield Static(
                self._title,
                id="library-note-folder-dialog-title",
                classes="destination-section",
            )
            yield Input(
                value=self._initial_name,
                placeholder="Folder name",
                id="library-note-folder-name",
            )
            with Horizontal(id="library-note-folder-dialog-actions"):
                yield Button("Cancel", id="library-note-folder-dialog-cancel")
                yield Button(
                    "Save", id="library-note-folder-dialog-confirm", variant="primary"
                )

    def _submit(self) -> None:
        value = self.query_one("#library-note-folder-name", Input).value.strip()
        if value:
            self.dismiss(value)

    @on(Button.Pressed, "#library-note-folder-dialog-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#library-note-folder-dialog-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#library-note-folder-name")
    def _name_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()


class LibraryNoteFolderTargetDialog(SafeModalDismissMixin, ModalScreen[str | None]):
    """Choose one folder destination from EVERY active folder, type-to-filter.

    TASK-34000.28 (N-23): this was a ``Select`` over the folders the tree had
    already loaded this visit, so a nested folder under a collapsed parent
    (``Projects / Thesis``) could not be chosen at all, and the title never
    named what was moving. It now pages every active folder at any depth
    through ``loader``, ``page_size`` at a time with the exact total, narrowed
    by the filter as the user types.

    Keys: the filter has focus on open; Down from it focuses the first row;
    Up/Down walk the rows (Up on the first returns to the filter); Enter on a
    row, or a click, chooses it; Choose confirms the row last focused (Top
    level when ``include_root`` and nothing else was focused); Escape, Cancel
    or the backdrop dismiss with ``None``. Nothing but Enter, a click or
    Choose ever dismisses with a folder id.
    """

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#library-note-folder-target-dialog"
    AUTO_FOCUS = "#library-note-folder-target-filter"

    #: The filter waits this long after the last keystroke before it asks.
    FILTER_DEBOUNCE_SECONDS = 0.2

    BUNDLED_CSS = """
    LibraryNoteFolderTargetDialog { align: center middle; }
    LibraryNoteFolderTargetDialog > Vertical {
        width: 64; height: auto; max-height: 100%; padding: 1 2;
        border: tall $accent; background: $surface;
    }
    #library-note-folder-target-list { height: auto; max-height: 50vh; }
    #library-note-folder-target-list > .library-note-folder-target-row {
        width: 1fr; text-align: left; content-align: left middle;
    }
    #library-note-folder-target-list > .library-note-folder-target-row.-highlighted {
        text-style: bold;
    }
    #library-note-folder-target-pager { height: auto; }
    #library-note-folder-target-actions { height: auto; align-horizontal: right; }
    """

    def __init__(
        self,
        *,
        title: str,
        loader: FolderTargetLoader,
        include_root: bool = False,
        page_size: int = FOLDER_TARGET_PAGE_SIZE,
    ) -> None:
        """Build the picker; the first page is requested on mount.

        Args:
            title: The heading, naming what moves (user text: painted
                literally, never as markup).
            loader: ``(query, offset)`` -> one exact page of every active
                folder whose path contains ``query``.
            include_root: Offer "Top level" (a folder move to the root) as the
                first row of the unfiltered first page.
            page_size: The page size ``loader`` serves.
        """
        super().__init__()
        self._title = title
        self._loader = loader
        self._include_root = include_root
        self._page_size = page_size
        self._query = ""
        self._page: NoteFolderChildPage | None = None
        # Top level is a real choice before the first page has even landed.
        self._highlighted: str | None = "" if include_root else None
        self._load_generation = 0
        self._filter_timer: Timer | None = None
        self._focus_first_row_on_load = False

    def compose(self) -> ComposeResult:
        with Vertical(id="library-note-folder-target-dialog"):
            yield Static(
                self._title,
                id="library-note-folder-target-title",
                classes="destination-section",
                markup=False,
            )
            yield Input(
                placeholder="Type to narrow folders…",
                max_length=200,
                id="library-note-folder-target-filter",
            )
            yield VerticalScroll(id="library-note-folder-target-list")
            yield Static(
                "Loading folders…", id="library-note-folder-target-total", markup=False
            )
            with Horizontal(id="library-note-folder-target-pager"):
                yield Button(
                    "‹ Previous",
                    id="library-note-folder-target-previous",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Next ›",
                    id="library-note-folder-target-next",
                    classes="library-canvas-action",
                    compact=True,
                )
            with Horizontal(id="library-note-folder-target-actions"):
                yield Button("Cancel", id="library-note-folder-target-cancel")
                yield Button(
                    "Choose", id="library-note-folder-target-confirm", variant="primary"
                )

    def on_mount(self) -> None:
        """Hide the pager and ask for page one.

        Textual runs every class's own ``on_mount`` along the MRO, so the
        mixin's (which remembers the opener) runs too -- calling it again
        here would run it twice.
        """
        self.query_one("#library-note-folder-target-pager").display = False
        self._request_page(0)

    # -- loading -----------------------------------------------------------

    def _request_page(self, offset: int) -> None:
        self._load_generation += 1
        self.run_worker(
            self._load_page(self._query, offset, self._load_generation),
            exclusive=True,
            group="library-note-folder-target",
        )

    async def _load_page(self, query: str, offset: int, generation: int) -> None:
        try:
            page = await self._loader(query, offset)
        except Exception as exc:  # noqa: BLE001 - one normalized picker boundary
            logger.bind(
                event="library_note_folder_target_page_failed",
                exception_class=type(exc).__name__,
            ).warning("library_note_folder_target_page_failed")
            if generation == self._load_generation and self.is_mounted:
                for total in self.query("#library-note-folder-target-total").results(
                    Static
                ):
                    total.update("Could not load folders.")
            return
        if generation != self._load_generation or not self.is_mounted:
            return
        await self._show_page(page)

    def _root_row_shown(self, page: NoteFolderChildPage) -> bool:
        return (
            self._include_root
            and page.start_offset == 0
            and (not self._query or self._query.casefold() in "top level")
        )

    async def _show_page(self, page: NoteFolderChildPage) -> None:
        """Replace the rows in place: never a recompose over the filter."""
        listing = next(
            iter(self.query("#library-note-folder-target-list").results(VerticalScroll)),
            None,
        )
        if listing is None:
            return
        self._page = page
        rows: list[Button] = []
        if self._root_row_shown(page):
            rows.append(self._row("Top level", ""))
        rows.extend(
            self._row(
                escape_markup(folder.path.strip("/").replace("/", " / ")),
                folder.folder_id,
            )
            for folder in page.folders
        )
        listed = {row.choice_value for row in rows}
        if self._highlighted not in listed:
            self._highlighted = None
        for row in rows:
            row.set_class(row.choice_value == self._highlighted, "-highlighted")
        # A row that held focus is about to be removed (Down pressed before
        # the filter's debounce answered): hand focus to the new first row
        # rather than letting Textual drop it somewhere else.
        refocus = self._focus_first_row_on_load or self.focused in self._rows()
        self._focus_first_row_on_load = False
        await listing.remove_children()
        await listing.mount_all(rows)
        listing.scroll_home(animate=False)
        for total in self.query("#library-note-folder-target-total").results(Static):
            total.update(self._total_text(page))
        paged = (
            page.total_folders > self._page_size
            or page.previous_offset is not None
            or page.next_offset is not None
        )
        for pager in self.query("#library-note-folder-target-pager"):
            pager.display = paged
        for button in self.query("#library-note-folder-target-previous").results(
            Button
        ):
            button.disabled = page.previous_offset is None
        for button in self.query("#library-note-folder-target-next").results(Button):
            button.disabled = page.next_offset is None
        if refocus:
            self._focus_row(0)

    @staticmethod
    def _row(label: str, value: str) -> Button:
        row = Button(
            label,
            classes="library-note-folder-target-row library-canvas-action",
            compact=True,
        )
        row.choice_value = value
        return row

    def _total_text(self, page: NoteFolderChildPage) -> str:
        total = page.total_folders
        if total == 0:
            if self._query:
                return f"No folder matches “{self._query}”"
            return "No folders yet"
        if page.previous_offset is None and page.next_offset is None:
            return f"{total} folder" if total == 1 else f"{total} folders"
        start = page.start_offset + 1
        end = page.start_offset + len(page.folders)
        return f"Showing {start}–{end} of {total} folders"

    # -- the filter ----------------------------------------------------------

    @on(Input.Changed, "#library-note-folder-target-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._filter_timer is not None:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(
            self.FILTER_DEBOUNCE_SECONDS, self._apply_filter
        )

    def _apply_filter(self) -> None:
        self._filter_timer = None
        value = next(
            iter(self.query("#library-note-folder-target-filter").results(Input)), None
        )
        if value is None:
            return
        query = value.value.strip()
        if query == self._query and self._page is not None:
            # Same text (a typo typed and deleted): keep the rows, and any
            # focus on them, rather than remounting an identical page.
            return
        self._query = query
        self._request_page(0)

    @on(Input.Submitted, "#library-note-folder-target-filter")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        """Enter in the filter moves to the first row; it never chooses."""
        event.stop()
        self._focus_row(0)

    # -- rows, keys, choosing ------------------------------------------------

    def _rows(self) -> list[Button]:
        return list(self.query(".library-note-folder-target-row").results(Button))

    def _focus_row(self, index: int) -> bool:
        rows = self._rows()
        if not rows:
            return False
        rows[max(0, min(index, len(rows) - 1))].focus()
        return True

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """A focused row becomes the one Choose confirms."""
        widget = event.widget
        if not isinstance(widget, Button) or not widget.has_class(
            "library-note-folder-target-row"
        ):
            return
        self._highlighted = widget.choice_value
        for row in self._rows():
            row.set_class(row is widget, "-highlighted")

    def on_key(self, event: events.Key) -> None:
        if event.key not in ("down", "up"):
            return
        focused = self.focused
        if (
            isinstance(focused, Input)
            and focused.id == "library-note-folder-target-filter"
        ):
            if event.key == "down" and self._focus_row(0):
                event.stop()
                event.prevent_default()
            return
        rows = self._rows()
        if focused not in rows:
            return
        index = rows.index(focused)
        event.stop()
        event.prevent_default()
        if event.key == "down":
            self._focus_row(index + 1)
        elif index == 0:
            self.query_one("#library-note-folder-target-filter", Input).focus()
        else:
            self._focus_row(index - 1)

    @on(Button.Pressed, ".library-note-folder-target-row")
    def _row_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(str(event.button.choice_value))

    @on(Button.Pressed, "#library-note-folder-target-previous")
    def _previous_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page is not None and self._page.previous_offset is not None:
            self._focus_first_row_on_load = True
            self._request_page(self._page.previous_offset)

    @on(Button.Pressed, "#library-note-folder-target-next")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page is not None and self._page.next_offset is not None:
            self._focus_first_row_on_load = True
            self._request_page(self._page.next_offset)

    def _submit(self) -> None:
        # Choose with no row ever focused (and no Top level) stays open: it
        # must never dismiss with a folder id the user did not pick.
        if self._highlighted is not None:
            self.dismiss(self._highlighted)

    @on(Button.Pressed, "#library-note-folder-target-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#library-note-folder-target-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()
