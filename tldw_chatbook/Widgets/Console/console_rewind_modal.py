"""Console `/rewind` menu modal.

Two-level flow, modeled on ``ConsoleSessionSwitcherModal``: the first level
lists the active path's prior USER prompts (newest first) as plain Button
rows; selecting one reveals a second-level action row offering **Restore to
here**, **Summarize up to here**, and **Never mind**. The result is a tagged
union (``ConsoleRewindChoice``) so the caller can dispatch on ``kind``
exactly like the switcher's ``ConsoleSwitcherChoice``.

Rows are ordinary focusable Buttons (click/Tab flow) -- v1 does not add the
prompt-picker's non-focusable-rows + synthetic-highlight keyboard discipline
since no arrow-key navigation is implemented yet; a future task can borrow
that pattern verbatim if arrow-nav is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Static

ROW_ID_PREFIX = "console-rewind-row-"
ROW_CLASS = "console-rewind-row"

MODAL_ID = "console-rewind-modal"
PROMPTS_CONTAINER_ID = "console-rewind-prompts"
ACTIONS_CONTAINER_ID = "console-rewind-actions"
SELECTED_LABEL_ID = "console-rewind-selected-label"
EMPTY_STATIC_ID = "console-rewind-empty"

RESTORE_ACTION_ID = "console-rewind-action-restore"
SUMMARIZE_ACTION_ID = "console-rewind-action-summarize"
CANCEL_ACTION_ID = "console-rewind-action-cancel"

KIND_RESTORE = "restore"
KIND_SUMMARIZE_UP_TO = "summarize-up-to"

EMPTY_PROMPTS_COPY = "No prior prompts to rewind to."


@dataclass(frozen=True)
class RewindPromptRow:
    """One prior USER prompt offered by the `/rewind` menu.

    Args:
        message_id: Native Console message id of the USER turn.
        index_label: Short display tag (e.g. ``"#3"``) for the turn's
            chronological position.
        preview: Collapsed, truncated single-line preview of the prompt's
            content shown on the row and (on restore) inserted back into the
            composer.
    """

    message_id: str
    index_label: str
    preview: str


@dataclass(frozen=True)
class ConsoleRewindChoice:
    """Result returned by the rewind modal.

    Args:
        kind: ``"restore"`` or ``"summarize-up-to"``.
        message_id: Native id of the selected prompt row.
        prompt_text: The selected row's DISPLAY-ONLY preview text (may be
            truncated). Restore deliberately re-fetches the FULL original
            text via ``store.get_message(message_id).content`` rather than
            using this field — never treat it as the message content.
    """

    kind: str
    message_id: str
    prompt_text: str


class ConsoleRewindModal(ModalScreen["ConsoleRewindChoice | None"]):
    """Pick a prior Console USER prompt, then Restore / Summarize / cancel."""

    DEFAULT_CSS = """
    ConsoleRewindModal {
        align: center middle;
    }

    #console-rewind-modal {
        width: 76;
        height: auto;
        max-height: 32;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-rewind-prompts {
        height: auto;
        max-height: 20;
        margin: 1 0 0 0;
    }

    .console-rewind-row {
        width: 100%;
        height: 2;
        min-height: 2;
        margin: 0;
    }

    #console-rewind-actions {
        height: auto;
        margin: 1 0 0 0;
    }

    #console-rewind-actions Button {
        width: 100%;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [("escape", "dismiss_rewind", "Cancel")]

    def __init__(self, *, prompts: tuple[RewindPromptRow, ...], **kwargs: Any) -> None:
        """Initialize the modal.

        Args:
            prompts: The active path's USER prompts, newest first.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._prompts = prompts
        self._selected: RewindPromptRow | None = None

    def compose(self) -> ComposeResult:
        """Build the prompt-row list and an empty actions container."""
        with Vertical(id=MODAL_ID):
            yield Static("Rewind", classes="console-modal-header")
            with Vertical(id=PROMPTS_CONTAINER_ID):
                if not self._prompts:
                    yield Static(EMPTY_PROMPTS_COPY, id=EMPTY_STATIC_ID, markup=False)
                for index, row in enumerate(self._prompts):
                    label = escape_markup(f"{row.index_label}  {row.preview}")
                    yield Button(
                        label,
                        id=f"{ROW_ID_PREFIX}{index}",
                        classes=ROW_CLASS,
                        compact=True,
                    )
            yield Vertical(id=ACTIONS_CONTAINER_ID)

    def action_dismiss_rewind(self) -> None:
        """Dismiss the modal with no result (Escape)."""
        self.dismiss(None)

    @on(Button.Pressed, f".{ROW_CLASS}")
    async def _row_pressed(self, event: Button.Pressed) -> None:
        """Select a prompt row and reveal the action row.

        Args:
            event: The pressed row button's event.
        """
        event.stop()
        index = self._row_index(event.button.id or "")
        if index is None or not (0 <= index < len(self._prompts)):
            return
        self._selected = self._prompts[index]
        await self._show_actions()

    async def _show_actions(self) -> None:
        """(Re)render the action row for the currently selected prompt.

        Awaited (mirrors ``ConsoleSessionSwitcherModal._refresh_results``):
        the removal must complete before mounting same-id replacements, or a
        second row selection can race a `DuplicateIds` error.
        """
        try:
            actions = self.query_one(f"#{ACTIONS_CONTAINER_ID}", Vertical)
        except NoMatches:
            return
        await actions.remove_children()
        selected = self._selected
        if selected is None:
            return
        await actions.mount_all(
            [
                Static(
                    f"Selected {selected.index_label}: {selected.preview}",
                    id=SELECTED_LABEL_ID,
                    markup=False,
                ),
                Button("Restore to here", id=RESTORE_ACTION_ID, variant="primary"),
                Button("Summarize up to here", id=SUMMARIZE_ACTION_ID),
                Button("Never mind", id=CANCEL_ACTION_ID),
            ]
        )

    @on(Button.Pressed, f"#{RESTORE_ACTION_ID}")
    def _restore_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with a ``restore`` choice for the selected prompt.

        Args:
            event: The Restore button's press event.
        """
        event.stop()
        if self._selected is None:
            return
        self.dismiss(
            ConsoleRewindChoice(
                kind=KIND_RESTORE,
                message_id=self._selected.message_id,
                prompt_text=self._selected.preview,
            )
        )

    @on(Button.Pressed, f"#{SUMMARIZE_ACTION_ID}")
    def _summarize_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with a ``summarize-up-to`` choice for the selected prompt.

        Args:
            event: The Summarize button's press event.
        """
        event.stop()
        if self._selected is None:
            return
        self.dismiss(
            ConsoleRewindChoice(
                kind=KIND_SUMMARIZE_UP_TO,
                message_id=self._selected.message_id,
                prompt_text=self._selected.preview,
            )
        )

    @on(Button.Pressed, f"#{CANCEL_ACTION_ID}")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with no result ("Never mind").

        Args:
            event: The Never-mind button's press event.
        """
        event.stop()
        self.dismiss(None)

    @staticmethod
    def _row_index(widget_id: str) -> int | None:
        """Parse the row index out of a ``console-rewind-row-N`` id.

        Args:
            widget_id: Candidate widget id.

        Returns:
            The parsed index, or ``None`` if ``widget_id`` doesn't match the
            expected row-button id shape.
        """
        if not widget_id.startswith(ROW_ID_PREFIX):
            return None
        try:
            return int(widget_id[len(ROW_ID_PREFIX) :])
        except ValueError:
            return None
