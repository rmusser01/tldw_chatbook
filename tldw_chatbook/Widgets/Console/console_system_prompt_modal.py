"""Console system prompt editor modal (Console `/system`, Task 14).

Edits (or clears) the active native Console session's system prompt --
seeded from the session's current value, applied back via the caller's
session-settings plumbing (Task 13's ``ConsoleChatStore.set_session_system_
prompt``). "Save to Library" is a side action that persists the current
editor text as a brand-new Library prompt's ``system_prompt`` (never an
update to an existing one) through a caller-injected async callable, mirroring
``ConsolePromptPickerModal``'s injected ``prompt_search`` -- this widget never
reaches into the scope service directly.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

MODAL_ID = "console-system-prompt-modal"
SCOPE_STATIC_ID = "console-system-prompt-scope"
TEXT_AREA_ID = "console-system-prompt-text"
NAME_INPUT_ID = "console-system-prompt-name"
SAVE_LIBRARY_BUTTON_ID = "console-system-prompt-save-library"
SAVE_STATUS_ID = "console-system-prompt-save-status"
APPLY_BUTTON_ID = "console-system-prompt-apply"
CLEAR_BUTTON_ID = "console-system-prompt-clear"
CANCEL_BUTTON_ID = "console-system-prompt-cancel"

SCOPE_COPY = "Applies to this session."
NAME_INPUT_PLACEHOLDER = "Prompt name"
SAVE_LIBRARY_BUTTON_LABEL = "Save to Library"
MISSING_NAME_COPY = "Enter a name to save this system prompt to Library."
MISSING_TEXT_COPY = "Enter a system prompt to save."
SAVE_TO_LIBRARY_ERROR_COPY = "Couldn't save this prompt. Try again."

# Async callable the caller binds to the scope service's create-prompt seam
# (Task 4's save flow); receives the modal's current (name, text) and
# returns outcome copy to display inline -- the modal never dismisses on
# either success or failure, it only reports the outcome.
SaveToLibrary = Callable[[str, str], Awaitable[str]]


class ConsoleSystemPromptModal(ModalScreen[Optional[str]]):
    """Edit (or clear) the active Console session's system prompt.

    Dismisses with:
      * ``None`` -- Cancel/Escape: no change.
      * ``str`` (possibly ``""``) -- Apply/Clear: the new session system
        prompt text (``""`` for Clear; the caller's session-settings
        plumbing already normalizes a blank string to ``None``).
    """

    BINDINGS = [("escape", "dismiss_editor", "Cancel")]

    def __init__(
        self,
        *,
        system_prompt: str | None,
        save_to_library: SaveToLibrary,
    ) -> None:
        """Initialize the system prompt editor.

        Args:
            system_prompt: Current session system prompt text (or ``None``)
                to seed the editor with.
            save_to_library: Async callable bound by the caller to the
                scope service's create-prompt seam; receives the modal's
                current (name, text) and returns outcome copy to display
                inline.
        """
        super().__init__()
        self._initial_text = system_prompt or ""
        self._save_to_library = save_to_library

    def compose(self) -> ComposeResult:
        """Build the editor's text area, name/save row, and action buttons."""
        with Vertical(id=MODAL_ID):
            yield Static("Edit system prompt", classes="console-modal-header")
            yield Static(
                SCOPE_COPY,
                id=SCOPE_STATIC_ID,
                classes="console-system-prompt-row",
                markup=False,
            )
            yield TextArea(self._initial_text, id=TEXT_AREA_ID)
            with Horizontal(classes="console-system-prompt-row"):
                yield Static("Name", classes="console-system-prompt-label")
                yield Input(placeholder=NAME_INPUT_PLACEHOLDER, id=NAME_INPUT_ID)
                yield Button(
                    SAVE_LIBRARY_BUTTON_LABEL,
                    id=SAVE_LIBRARY_BUTTON_ID,
                    compact=True,
                )
            yield Static(
                "",
                id=SAVE_STATUS_ID,
                classes="console-system-prompt-row",
                markup=False,
            )
            with Horizontal(
                id="console-system-prompt-actions",
                classes="console-system-prompt-row console-system-prompt-actions",
            ):
                yield Button("Clear", id=CLEAR_BUTTON_ID)
                yield Button("Cancel", id=CANCEL_BUTTON_ID)
                yield Button("Apply", id=APPLY_BUTTON_ID, variant="primary")

    def on_mount(self) -> None:
        """Focus the system prompt text area so editing can start immediately."""
        try:
            self.query_one(f"#{TEXT_AREA_ID}", TextArea).focus()
        except (NoMatches, QueryError):
            pass

    def action_dismiss_editor(self) -> None:
        """Dismiss with ``None`` (no change), bound to the Escape key."""
        self.dismiss(None)

    @on(Button.Pressed, f"#{CANCEL_BUTTON_ID}")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, f"#{APPLY_BUTTON_ID}")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        text = self.query_one(f"#{TEXT_AREA_ID}", TextArea).text
        self.dismiss(text.strip())

    @on(Button.Pressed, f"#{CLEAR_BUTTON_ID}")
    def _clear(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("")

    @on(Button.Pressed, f"#{SAVE_LIBRARY_BUTTON_ID}")
    def _save_to_library_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        name = self.query_one(f"#{NAME_INPUT_ID}", Input).value.strip()
        text = self.query_one(f"#{TEXT_AREA_ID}", TextArea).text.strip()
        if not name:
            self._set_save_status(MISSING_NAME_COPY)
            return
        if not text:
            self._set_save_status(MISSING_TEXT_COPY)
            return
        self.run_worker(
            self._run_save_to_library(name, text),
            exclusive=True,
            group="console-system-prompt-save-library",
        )

    async def _run_save_to_library(self, name: str, text: str) -> None:
        try:
            status = await self._save_to_library(name, text)
        except Exception:
            status = SAVE_TO_LIBRARY_ERROR_COPY
        self._set_save_status(status)

    def _set_save_status(self, text: str) -> None:
        try:
            self.query_one(f"#{SAVE_STATUS_ID}", Static).update(text)
        except (NoMatches, QueryError):
            pass
