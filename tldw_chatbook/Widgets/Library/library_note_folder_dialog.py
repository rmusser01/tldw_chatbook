"""Small keyboard-first dialogs for Database Note folder mutations."""

from __future__ import annotations

from collections.abc import Sequence

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ..modal_dismissal import SafeModalDismissMixin


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
    """Choose one bounded, already-loaded folder destination."""

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#library-note-folder-target-dialog"
    AUTO_FOCUS = "#library-note-folder-target"

    BUNDLED_CSS = """
    LibraryNoteFolderTargetDialog { align: center middle; }
    LibraryNoteFolderTargetDialog > Vertical {
        width: 64; height: auto; padding: 1 2;
        border: tall $accent; background: $surface;
    }
    #library-note-folder-target-actions { height: auto; align-horizontal: right; }
    """

    def __init__(
        self,
        *,
        title: str,
        folders: Sequence[tuple[str, str]],
        include_root: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._folders = tuple(folders)
        self._include_root = include_root

    def compose(self) -> ComposeResult:
        options = list(self._folders)
        if self._include_root:
            options.insert(0, ("Top level", ""))
        with Vertical(id="library-note-folder-target-dialog"):
            yield Static(
                self._title,
                id="library-note-folder-target-title",
                classes="destination-section",
            )
            # TASK-32533 (critique #3 P0): the blank sentinel is `Select.NULL`.
            # `Select.BLANK` is not a Select attribute on this Textual version
            # -- it silently resolves to `Widget.BLANK` (`False`), which is not
            # one of the options, so opening Add to folder / Move note raised
            # InvalidSelectValueError at mount and exited the whole app.
            yield Select(
                options,
                value="" if self._include_root else Select.NULL,
                allow_blank=not self._include_root,
                id="library-note-folder-target",
            )
            with Horizontal(id="library-note-folder-target-actions"):
                yield Button("Cancel", id="library-note-folder-target-cancel")
                yield Button(
                    "Choose", id="library-note-folder-target-confirm", variant="primary"
                )

    def _submit(self) -> None:
        value = self.query_one("#library-note-folder-target", Select).value
        # Same sentinel bug: the old `is not Select.BLANK` compared against
        # `False`, so Choose with nothing picked dismissed with the literal
        # folder id "Select.NULL". "" is a real choice here (Top level).
        if value is not Select.NULL:
            self.dismiss(str(value))

    @on(Button.Pressed, "#library-note-folder-target-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#library-note-folder-target-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()
