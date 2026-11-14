"""Name-prompt modal for the RAG profile-manager Clone/Rename actions.

Moved out of ``UI/Screens/settings_screen.py`` (TASK-36944.8) so the theme
editor widget can import it without a circular-import workaround; the screen
re-imports it, so ``settings_screen.RagProfileNameModal`` still resolves.
"""

from __future__ import annotations

from collections.abc import Callable

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class RagProfileNameModal(ModalScreen[str | None]):
    """Minimal name-prompt modal for the RAG profile-manager Clone/Rename actions.

    No existing text-prompt modal precedent lives in this screen (task-2
    brief) -- this follows the same dismiss-with-a-value + push_screen(modal,
    callback) shape as ``ConsoleSystemPromptModal``. Dismisses with the
    trimmed name, or ``None`` on Cancel/Escape/a blank submission.

    TASK-33070: ``validate`` (optional) is called with the trimmed value on
    submit; a returned reason is shown inside the dialog and the typed value
    kept, ``None`` dismisses as before.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(
        self,
        *,
        title: str,
        initial: str = "",
        confirm_label: str = "Save",
        validate: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__()
        self._modal_title = title
        self._initial = initial
        self._confirm_label = confirm_label
        self._validate = validate

    def compose(self) -> ComposeResult:
        with Vertical(
            id="settings-rag-profile-name-modal", classes="settings-rag-profile-modal"
        ):
            yield Static(self._modal_title, classes="destination-section")
            yield Input(value=self._initial, id="settings-rag-profile-name-input")
            # Untrusted text (a file's reason, a typed path): never markup.
            error = Static(
                "",
                id="settings-rag-profile-name-error",
                classes="settings-rag-profile-name-error",
                markup=False,
            )
            error.display = False
            yield error
            with Horizontal(classes="settings-action-row"):
                yield Button("Cancel", id="settings-rag-profile-name-cancel")
                yield Button(
                    self._confirm_label,
                    id="settings-rag-profile-name-confirm",
                    variant="primary",
                )

    def on_mount(self) -> None:
        try:
            self.query_one("#settings-rag-profile-name-input", Input).focus()
        except QueryError:
            pass

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#settings-rag-profile-name-cancel")
    def _handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#settings-rag-profile-name-confirm")
    def _handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit_current_value()

    @on(Input.Submitted, "#settings-rag-profile-name-input")
    def _handle_submit(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit_current_value()

    def _submit_current_value(self) -> None:
        try:
            value = self.query_one("#settings-rag-profile-name-input", Input).value
        except QueryError:
            value = ""
        value = value.strip()
        reason = self._validate(value) if value and self._validate is not None else None
        if reason:
            from ..css.Themes.themes import printable

            error = self.query_one("#settings-rag-profile-name-error", Static)
            error.update(printable(reason))
            error.display = True
            self.query_one("#settings-rag-profile-name-input", Input).focus()
            return
        self.dismiss(value or None)
