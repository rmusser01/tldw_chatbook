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

    # Escape dismisses, exactly as Cancel does (TASK-1300). These dialogs are
    # modal, so without it a keyboard user can neither back out nor do anything
    # else -- during the third Watchlists UAT a `Delete` click was silently
    # swallowed by a Rename dialog that would not close.
    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-import-dialog", classes="opml-dialog"):
            yield Static("Import OPML", classes="dialog-title")
            yield TextArea("", id="opml-import-text")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Import", id="opml-import-confirm", variant="success")
                yield Button("Cancel", id="opml-import-cancel", variant="default")

    def action_cancel(self) -> None:
        """Back out without applying anything.
        """
        self.dismiss(None)

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

    # Escape dismisses, exactly as Cancel does (TASK-1300). These dialogs are
    # modal, so without it a keyboard user can neither back out nor do anything
    # else -- during the third Watchlists UAT a `Delete` click was silently
    # swallowed by a Rename dialog that would not close.
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, xml_text: str) -> None:
        super().__init__()
        self.xml_text = xml_text

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-export-dialog", classes="opml-dialog"):
            yield Static("Export OPML", classes="dialog-title")
            yield TextArea(self.xml_text, id="opml-export-text", read_only=True)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Close", id="opml-export-close", variant="primary")

    def action_cancel(self) -> None:
        """Back out without applying anything.
        """
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if str(event.button.id) == "opml-export-close":
            self.dismiss(None)
        event.stop()


class ConfirmDeleteDialog(ModalScreen[bool]):
    """Modal dialog that asks the user to confirm deletion of an entity."""

    # Escape dismisses, exactly as Cancel does (TASK-1300). These dialogs are
    # modal, so without it a keyboard user can neither back out nor do anything
    # else -- during the third Watchlists UAT a `Delete` click was silently
    # swallowed by a Rename dialog that would not close.
    BINDINGS = [("escape", "cancel", "Cancel")]

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

    def action_cancel(self) -> None:
        """Back out without applying anything.

        Dismisses `False`, not `None`: the caller is asking
        "should I delete this?" and must get the same answer Cancel gives.
        """
        self.dismiss(False)

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

    # Escape dismisses, exactly as Cancel does (TASK-1300). These dialogs are
    # modal, so without it a keyboard user can neither back out nor do anything
    # else -- during the third Watchlists UAT a `Delete` click was silently
    # swallowed by a Rename dialog that would not close.
    BINDINGS = [("escape", "cancel", "Cancel")]

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

    def action_cancel(self) -> None:
        """Back out without applying anything.
        """
        self.dismiss(None)

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


#: TASK-2303 AC#3. Both pickers below are bare lists of buttons, and a bare
#: list of names does not say what pressing one DOES -- the 2026-08-04 UAT
#: read the source picker as a place where a source might be created. Each
#: dialog states the effect and, explicitly, that nothing new is made: the
#: whole point of this task is that ADD and NEW are different operations, and
#: the modal is the last place the distinction can be drawn before the write.
_PICKER_CREATES_NOTHING = "No new source is created."


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

    # Escape dismisses, exactly as Cancel does (TASK-1300). These dialogs are
    # modal, so without it a keyboard user can neither back out nor do anything
    # else -- during the third Watchlists UAT a `Delete` click was silently
    # swallowed by a Rename dialog that would not close.
    BINDINGS = [("escape", "cancel", "Cancel")]

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
                    f"Add an existing source to {escape_markup(self.watchlist_name)}"
                ),
                classes="dialog-title",
            )
            if self.candidates:
                yield Static(
                    "Choose a source below to add it to this watchlist. "
                    + _PICKER_CREATES_NOTHING,
                    id="watchlist-add-source-instructions",
                    classes="watchlist-picker-instructions",
                )
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
                    "Use New source under Sources to make another one.",
                    id="watchlist-add-source-empty",
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="watchlist-add-source-cancel")

    def action_cancel(self) -> None:
        """Back out without applying anything.
        """
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id)
        prefix = "wl-add-source-option-"
        if button_id.startswith(prefix):
            self.dismiss(int(button_id[len(prefix):]))
        elif button_id == "watchlist-add-source-cancel":
            self.dismiss(None)


class WatchlistPickerDialog(ModalScreen[int | None]):
    """Pick one watchlist to add a source to (TASK-2303).

    The mirror image of `WatchlistSourcePickerDialog`, and deliberately the
    same shape: one compact button per candidate, ids built from the
    watchlist's integer id, labels escaped. It exists because assignment was
    reachable from exactly one place -- the rail, with a watchlist already in
    scope -- so a user looking at the source they wanted to file had no way
    to file it. This is the source-first direction of the same write.

    Args:
        source_name: Name of the source being added, for the heading.
        candidates: Watchlist rows (`id`/`name`) the source is not already
            in. An empty sequence renders an explained empty state with the
            confirming affordance absent rather than dead.
        total_watchlists: How many watchlists exist at all. Needed because an
            empty `candidates` has TWO causes with opposite remedies (review
            wave, M2), and the dialog cannot tell them apart from the empty
            list alone. Defaults to the length of `candidates`, so a caller
            that does not pass it gets the "none exist" reading for an empty
            list -- the safer of the two, since it never claims membership
            that is not there.
    """

    # Escape dismisses, exactly as Cancel does (TASK-1300).
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        source_name: str,
        candidates: Sequence[Mapping[str, Any]],
        total_watchlists: int | None = None,
    ) -> None:
        super().__init__()
        self.source_name = source_name
        self.candidates = list(candidates)
        self.total_watchlists = (
            len(self.candidates) if total_watchlists is None else int(total_watchlists)
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="watchlist-pick-dialog", classes="opml-dialog"):
            yield Static(
                Text.from_markup(
                    f"Add {escape_markup(self.source_name)} to a watchlist"
                ),
                classes="dialog-title",
            )
            if self.candidates:
                yield Static(
                    "Choose a watchlist below to add this source to it. "
                    + _PICKER_CREATES_NOTHING,
                    id="watchlist-pick-instructions",
                    classes="watchlist-picker-instructions",
                )
                with VerticalScroll(id="watchlist-pick-options"):
                    for row in self.candidates:
                        watchlist_id = int(row["id"])
                        name = escape_markup(str(row.get("name") or "Untitled"))
                        yield Button(
                            Text.from_markup(name),
                            id=f"wl-pick-option-{watchlist_id}",
                            compact=True,
                        )
            else:
                # Two empty states, not one sentence stretched across both
                # (review wave, M2). "Already belongs to every watchlist" is
                # simply false on a profile that has none -- which is exactly
                # the profile a first-run user reaches this dialog on.
                yield Static(
                    (
                        "This source already belongs to every watchlist. "
                        "Use New in the rail to make another one."
                        if self.total_watchlists
                        else "There are no watchlists yet. Use New in the "
                        "rail to make one, then add this source to it."
                    ),
                    id="watchlist-pick-empty",
                )
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="watchlist-pick-cancel")

    def action_cancel(self) -> None:
        """Back out without applying anything."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id)
        prefix = "wl-pick-option-"
        if button_id.startswith(prefix):
            self.dismiss(int(button_id[len(prefix):]))
        elif button_id == "watchlist-pick-cancel":
            self.dismiss(None)
