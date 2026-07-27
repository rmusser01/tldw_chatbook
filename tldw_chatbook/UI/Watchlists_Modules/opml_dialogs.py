"""Modal dialogs for the Watchlists screen.

OPML import/export and delete confirmation, plus (task-895) the two prompts
the watchlist tree's write verbs need: a name entry for create/rename, and a
source picker for membership editing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, TextArea

from ...Utils.input_validation import sanitize_string, validate_text_input


class OpmlImportDialog(ModalScreen[str | None]):
    """Modal dialog that prompts the user for OPML XML to import."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-import-dialog", classes="opml-dialog"):
            yield Static("Import OPML", classes="dialog-title")
            yield TextArea("", id="opml-import-text")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Import", id="opml-import-confirm", variant="success")
                yield Button("Cancel", id="opml-import-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "opml-import-confirm":
            text_area = self.query_one("#opml-import-text", TextArea)
            self.dismiss(text_area.text)
        elif button_id == "opml-import-cancel":
            self.dismiss(None)
        event.stop()


class OpmlExportDialog(ModalScreen[None]):
    """Modal dialog that displays OPML XML exported from watchlist sources."""

    BINDINGS = []

    def __init__(self, xml_text: str) -> None:
        super().__init__()
        self.xml_text = xml_text

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-export-dialog", classes="opml-dialog"):
            yield Static("Export OPML", classes="dialog-title")
            yield TextArea(self.xml_text, id="opml-export-text", read_only=True)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Close", id="opml-export-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if str(event.button.id) == "opml-export-close":
            self.dismiss(None)
        event.stop()


class ConfirmDeleteDialog(ModalScreen[bool]):
    """Modal dialog that asks the user to confirm deletion of an entity."""

    BINDINGS = []

    def __init__(self, item_name: str) -> None:
        super().__init__()
        self.item_name = item_name

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-delete-dialog", classes="opml-dialog"):
            yield Static("Confirm delete", classes="dialog-title")
            yield Label(f"Delete {self.item_name}?")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Delete", id="confirm-delete-confirm", variant="error")
                yield Button("Cancel", id="confirm-delete-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "confirm-delete-confirm":
            self.dismiss(True)
        elif button_id == "confirm-delete-cancel":
            self.dismiss(False)
        event.stop()


class WatchlistNameDialog(ModalScreen[str | None]):
    """Prompt for a watchlist name, for create and for rename (task-895).

    Rejects an empty or duplicate name with a *visible* reason rather than
    a silent no-op: the dialog stays open with `#watchlist-name-error`
    filled in, and only dismisses once the name is one the service will
    actually store as typed.

    The duplicate check here is a user-facing guard, not a second
    implementation of `WatchlistBundleService._unique_name`. That method
    still resolves genuine collisions (an OPML re-import, a race with
    another writer) by suffixing; this exists so a user who types a name
    that already exists is told so, instead of silently ending up with
    "Security (2)" and no explanation.

    Args:
        dialog_title: Heading, e.g. "New watchlist" or "Rename watchlist".
        submit_label: Label for the confirming button.
        initial_name: Value the input starts with (the current name, when
            renaming).
        taken_names: Names already in use. Compared case-insensitively,
            matching how `_unique_name` compares. When renaming, the
            watchlist's own name must be excluded by the caller so
            re-submitting it unchanged is not reported as a collision.
    """

    BINDINGS = []

    def __init__(
        self,
        *,
        dialog_title: str,
        submit_label: str,
        initial_name: str = "",
        taken_names: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.dialog_title = dialog_title
        self.submit_label = submit_label
        self.initial_name = initial_name
        self._taken = {
            str(name).strip().lower() for name in taken_names if str(name).strip()
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="watchlist-name-dialog", classes="opml-dialog"):
            yield Static(self.dialog_title, classes="dialog-title")
            yield Input(
                value=self.initial_name,
                placeholder="Watchlist name",
                id="watchlist-name-input",
            )
            yield Static("", id="watchlist-name-error", classes="dialog-error")
            with Horizontal(classes="dialog-buttons"):
                yield Button(
                    self.submit_label, id="watchlist-name-submit", variant="success"
                )
                yield Button("Cancel", id="watchlist-name-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#watchlist-name-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if str(event.button.id) == "watchlist-name-submit":
            self._submit()
        elif str(event.button.id) == "watchlist-name-cancel":
            self.dismiss(None)

    def _show_error(self, message: str) -> None:
        """Render `message` where the user is already looking.

        `Text.from_markup` over pre-escaped content, the same convention the
        Watchlists screen uses everywhere a name reaches a label: a
        watchlist name is user-authored free text and `Static` parses markup
        by default.
        """
        self.query_one("#watchlist-name-error", Static).update(
            Text.from_markup(message)
        )

    def _submit(self) -> None:
        raw = self.query_one("#watchlist-name-input", Input).value
        name = sanitize_string(raw.strip(), max_length=255)
        if not name:
            self._show_error("A watchlist name cannot be empty. Type a name.")
            return
        if not validate_text_input(name, max_length=255):
            self._show_error(
                "That name contains invalid characters or is too long."
            )
            return
        if name.lower() in self._taken:
            self._show_error(
                f'A watchlist named "{escape_markup(name)}" already exists. '
                "Choose a different name."
            )
            return
        self.dismiss(name)


class WatchlistSourcePickerDialog(ModalScreen[int | None]):
    """Pick one source to add to a watchlist (task-895).

    One compact button per candidate rather than a `Select`: the button id
    is built from the subscription's integer id (always a legal Textual id,
    unlike the free-text name), the label is escaped, and a `Select` on this
    screen posts `Changed` on mount, which is a recompose storm this repo
    has already paid for once.

    Args:
        watchlist_name: Name of the target watchlist, for the heading.
        candidates: Source rows (`id`/`name`/`type`) not already in the
            watchlist. An empty sequence renders an explained empty state
            with the confirming affordance absent rather than dead.
    """

    BINDINGS = []

    def __init__(
        self, watchlist_name: str, candidates: Sequence[Mapping[str, Any]]
    ) -> None:
        super().__init__()
        self.watchlist_name = watchlist_name
        self.candidates = list(candidates)

    def compose(self) -> ComposeResult:
        with Vertical(id="watchlist-add-source-dialog", classes="opml-dialog"):
            yield Static(
                Text.from_markup(
                    f"Add a source to {escape_markup(self.watchlist_name)}"
                ),
                classes="dialog-title",
            )
            if self.candidates:
                with VerticalScroll(id="watchlist-add-source-options"):
                    for row in self.candidates:
                        source_id = int(row["id"])
                        name = escape_markup(str(row.get("name") or "Untitled"))
                        source_type = escape_markup(str(row.get("type") or ""))
                        yield Button(
                            Text.from_markup(f"{name}  ({source_type})"),
                            id=f"wl-add-source-option-{source_id}",
                            compact=True,
                        )
            else:
                yield Static(
                    "Every source already belongs to this watchlist. "
                    "Create a source in the Sources tab first.",
                    id="watchlist-add-source-empty",
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="watchlist-add-source-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id)
        prefix = "wl-add-source-option-"
        if button_id.startswith(prefix):
            self.dismiss(int(button_id[len(prefix):]))
        elif button_id == "watchlist-add-source-cancel":
            self.dismiss(None)
