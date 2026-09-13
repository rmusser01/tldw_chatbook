"""Compact receipt-keyed destination notice for Console activity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_switcher_state import CapturedReceipt
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label

MarkSeenHandler = Callable[
    ["ConsoleActivityOutcomePresentation", int], bool | Awaitable[bool]
]


@dataclass(frozen=True)
class ConsoleActivityOutcomePresentation:
    """Frozen destination and receipt evidence rendered by one notice."""

    title: str
    profile_authority: str
    authority_token: str
    session_id: str | None
    conversation_id: str | None
    receipts: tuple[CapturedReceipt, ...]

    def __post_init__(self) -> None:
        if not self.profile_authority or not self.authority_token:
            raise ValueError("Outcome notice authority is required.")
        if not (self.session_id or self.conversation_id):
            raise ValueError("Outcome notice destination is required.")
        if not self.receipts:
            raise ValueError("Outcome notice receipts are required.")


class ConsoleActivityOutcomeNotice(Horizontal):
    """Always-mounted notice that reveals exact selected outcomes in place."""

    BUNDLED_CSS = """
    ConsoleActivityOutcomeNotice {
        width: 100%;
        height: 1;
        min-height: 1;
        max-height: 1;
        background: $boost;
    }
    #console-activity-outcome-copy {
        width: 1fr;
        height: 1;
    }
    #console-activity-outcome-mark-seen {
        width: 11;
        min-width: 11;
        height: 1;
        min-height: 1;
        border: none;
    }
    #console-activity-outcome-dismiss {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        border: none;
    }
    """

    def __init__(
        self,
        *,
        mark_seen: MarkSeenHandler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mark_seen = mark_seen
        self._presentation: ConsoleActivityOutcomePresentation | None = None
        self._presentation_generation = 0
        self._retry_all_generation: int | None = None
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1
        self.styles.display = "none"

    @property
    def presentation_generation(self) -> int:
        """Return the monotonically increasing paint-generation fence."""
        return self._presentation_generation

    @property
    def presentation(self) -> ConsoleActivityOutcomePresentation | None:
        """Return current frozen evidence, if visible."""
        return self._presentation

    def compose(self) -> ComposeResult:
        content = Static("", id="console-activity-outcome-copy", markup=False)
        content.styles.width = "1fr"
        content.styles.height = 1
        yield content
        mark_seen = Button(
            "Mark seen",
            id="console-activity-outcome-mark-seen",
            compact=True,
        )
        mark_seen.styles.width = 11
        mark_seen.styles.min_width = 11
        mark_seen.styles.height = 1
        mark_seen.styles.min_height = 1
        yield mark_seen
        dismiss = Button(
            "×", id="console-activity-outcome-dismiss", compact=True
        )
        dismiss.styles.width = 3
        dismiss.styles.min_width = 3
        dismiss.styles.height = 1
        dismiss.styles.min_height = 1
        yield dismiss

    def set_mark_seen_handler(self, handler: MarkSeenHandler | None) -> None:
        """Replace the late-bound exact acknowledgement callback."""
        self._mark_seen = handler

    def show(self, presentation: ConsoleActivityOutcomePresentation) -> int:
        """Reveal a replacement presentation and return its generation."""
        self._presentation_generation += 1
        self._presentation = presentation
        self._retry_all_generation = None
        self.styles.display = "block"
        title = sanitize_character_display_label(
            presentation.title,
            max_characters=64,
        ) or "Conversation activity"
        statuses = tuple(receipt.status.upper() for receipt in presentation.receipts)
        summary = " + ".join(dict.fromkeys(statuses))
        self.query_one("#console-activity-outcome-copy", Static).update(
            Text(f"{summary} · {title}")
        )
        requires_mark = any(
            receipt.status != "done" for receipt in presentation.receipts
        )
        self.query_one(
            "#console-activity-outcome-mark-seen", Button
        ).styles.display = "block" if requires_mark else "none"
        return self._presentation_generation

    def require_mark_seen(self, generation: int) -> None:
        """Expose an exact all-receipt retry for the current presentation."""
        if generation != self._presentation_generation or self._presentation is None:
            return
        self._retry_all_generation = generation
        self.query_one(
            "#console-activity-outcome-mark-seen", Button
        ).styles.display = "block"

    def should_retry_all(self, generation: int) -> bool:
        """Return whether automatic success acknowledgement failed here."""
        return self._retry_all_generation == generation

    def hide(self) -> None:
        """Hide and invalidate the current presentation without acknowledging."""
        self._presentation_generation += 1
        self._presentation = None
        self._retry_all_generation = None
        self.styles.display = "none"

    def is_current(
        self,
        generation: int,
        presentation: ConsoleActivityOutcomePresentation,
    ) -> bool:
        """Return whether this exact evidence is still the visible paint owner."""
        return bool(
            self.display
            and generation == self._presentation_generation
            and presentation == self._presentation
        )

    @on(Button.Pressed, "#console-activity-outcome-mark-seen")
    async def _mark_seen_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        presentation = self._presentation
        generation = self._presentation_generation
        handler = self._mark_seen
        if presentation is None or handler is None:
            return
        accepted = handler(presentation, generation)
        if inspect.isawaitable(accepted):
            accepted = await accepted
        if accepted and self.is_current(generation, presentation):
            self.hide()

    @on(Button.Pressed, "#console-activity-outcome-dismiss")
    def _dismiss_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.hide()

    def on_unmount(self) -> None:
        """Invalidate callbacks captured before structural teardown/remount."""
        self._presentation_generation += 1
        self._presentation = None
        self._retry_all_generation = None
