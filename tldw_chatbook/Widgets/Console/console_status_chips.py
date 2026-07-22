"""Console status-pill strip (provider/model/persona/RAG/source/tool/approval).

Extracted from ConsoleControlBar so the pills can render in their own strip
directly above the composer. The widget owns the chip classes, the chip
builder, chip labelling + emphasis sync, and the approvals-review action.
"""

from __future__ import annotations

from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


class ConsoleChip(Static):
    """Focusable Console readiness chip.

    Chips ellipsize at 22 cells; focusing a chip lifts that cap (see
    ``.console-control-chip:focus`` in ``_agentic_terminal.tcss``) so the full
    label is reachable from the keyboard, while the tooltip keeps carrying the
    same full text on hover.
    """

    can_focus = True


class ConsoleApprovalsChip(ConsoleChip):
    """Approvals readiness chip that doubles as an approval-review action.

    Activating it (Enter/Space while focused, or click) asks the strip to
    focus the pending approval card in the transcript.
    """

    BINDINGS = [
        Binding("enter", "review_approval", "Review pending approval", show=False),
        Binding("space", "review_approval", "Review pending approval", show=False),
    ]

    class ReviewRequested(Message):
        """Posted when the approvals chip is activated from keyboard or mouse."""

    def action_review_approval(self) -> None:
        self.post_message(self.ReviewRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.ReviewRequested())


class ConsoleStatusChips(Horizontal):
    """Full-width strip of the seven Console readiness pills.

    Exposes ``sync_state`` so ``ChatScreen`` can refresh the pill labels and
    counter emphasis after provider/model/source/tool/approval state changes.
    """

    def __init__(self, state: ConsoleControlState, **kwargs: Any) -> None:
        """Initialize the strip.

        Args:
            state: Display-state snapshot for the readiness labels.
            **kwargs: Additional Textual widget arguments (id/classes).
        """
        classes = kwargs.pop("classes", "")
        # Reuse the existing chip-row class so its CSS continues to apply.
        super().__init__(
            classes=f"console-control-chip-row console-status-chips {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    @staticmethod
    def _chip(
        label: str,
        *,
        id: str,
        emphasis: bool | None = None,
        chip_class: type[ConsoleChip] = ConsoleChip,
    ) -> ConsoleChip:
        """Build one readiness chip. Mirrors the former ConsoleControlBar._chip."""
        classes = "console-control-chip"
        if emphasis is False:
            classes += " console-chip-dim"
        elif emphasis is True:
            classes += " console-chip-alert"
        chip = chip_class(label, id=id, classes=classes)
        chip.tooltip = label
        return chip

    def compose(self) -> ComposeResult:
        yield self._chip(self.state.provider_label, id="console-provider-chip")
        yield self._chip(self.state.model_label, id="console-model-chip")
        yield self._chip(self.state.persona_label, id="console-persona-chip")
        yield self._chip(self.state.rag_label, id="console-rag-chip")
        yield self._chip(
            self.state.sources_label,
            id="console-sources-chip",
            emphasis=self.state.sources_active,
        )
        yield self._chip(
            self.state.tools_label,
            id="console-tools-chip",
            emphasis=self.state.tools_active,
        )
        yield self._chip(
            self.state.approvals_label,
            id="console-approvals-chip",
            emphasis=self.state.approvals_active,
            chip_class=ConsoleApprovalsChip,
        )

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh pill labels and counter emphasis from a new snapshot."""
        if state == self.state:
            return
        self.state = state
        label_values = {
            "#console-provider-chip": state.provider_label,
            "#console-model-chip": state.model_label,
            "#console-persona-chip": state.persona_label,
            "#console-rag-chip": state.rag_label,
            "#console-sources-chip": state.sources_label,
            "#console-tools-chip": state.tools_label,
            "#console-approvals-chip": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.update(label)
            chip.tooltip = label
        chip_emphasis = {
            "#console-sources-chip": state.sources_active,
            "#console-tools-chip": state.tools_active,
            "#console-approvals-chip": state.approvals_active,
        }
        for selector, active in chip_emphasis.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.set_class(not active, "console-chip-dim")
            chip.set_class(active, "console-chip-alert")

    @on(ConsoleApprovalsChip.ReviewRequested)
    def on_approval_review_requested(
        self, event: ConsoleApprovalsChip.ReviewRequested
    ) -> None:
        """Focus the pending approval card in the transcript.

        Falls back to a notification when no approval is pending so the
        keyboard-only path never dead-ends silently.
        """
        event.stop()
        self._focus_pending_approval_card()

    def _focus_pending_approval_card(self) -> None:
        """Scroll the displayed approval card into view and focus its action."""
        try:
            cards = list(self.screen.query("#chat-approval-card"))
        except Exception:
            cards = []
        card = next(
            (
                candidate
                for candidate in cards
                if isinstance(candidate, ChatApprovalCard) and candidate.display
            ),
            None,
        )
        if card is None:
            self.app.notify(CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning")
            return
        try:
            card.scroll_visible(animate=False)
        except Exception:
            pass
        try:
            batch_visible = card.query_one("#approval-batch-body").display
        except NoMatches:
            batch_visible = False
        target_id = "#approval-submit" if batch_visible else "#approval-allow-once"
        try:
            card.query_one(target_id, Button).focus()
        except NoMatches:
            pass
