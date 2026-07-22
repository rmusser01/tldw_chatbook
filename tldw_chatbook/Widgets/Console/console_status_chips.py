"""Console status-pill strip (provider/model/persona/RAG/source/tool/approval)
plus the retrieval-scope chip.

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
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Chat.rag_scope import SCOPE_EMPTY_NOTICE_TEMPLATE, SCOPE_REASON_EMPTY
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_retrieval_scope_row import (
    UNSCOPED_LABEL as SCOPE_ROW_UNSCOPED_LABEL,
)


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


class ConsoleScopeChip(ConsoleChip):
    """Retrieval-scope chip that opens the scope picker when activated.

    Mirrors ``ConsoleApprovalsChip`` exactly: Enter/Space while focused, or
    a click, opens the same RAG retrieval-scope picker modal the Inspector
    row's Edit/Narrow… button opens
    (``ChatScreen._open_console_retrieval_scope_picker``, task-9) -- the
    same handler seam, task-10 just adds a second entry point into it.
    """

    BINDINGS = [
        Binding("enter", "open_scope_picker", "Open retrieval scope picker", show=False),
        Binding("space", "open_scope_picker", "Open retrieval scope picker", show=False),
    ]

    class OpenRequested(Message):
        """Posted when the scope chip is activated from keyboard or mouse."""

    def action_open_scope_picker(self) -> None:
        self.post_message(self.OpenRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.OpenRequested())


class ConsoleStatusChips(Horizontal):
    """Full-width strip of the eight Console readiness pills (provider/model/
    persona/RAG/source/tool/approval plus the retrieval-scope chip).

    Exposes ``sync_state`` so ``ChatScreen`` can refresh the pill labels and
    counter emphasis after provider/model/source/tool/approval state changes.
    """

    def __init__(
        self,
        state: ConsoleControlState,
        *,
        scope_state: ConsoleRetrievalScopeState | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the strip.

        Args:
            state: Display-state snapshot for the readiness labels.
            scope_state: Display-state snapshot for the "Scope" chip
                (task-10) -- the same ``ConsoleRetrievalScopeState`` the
                Inspector's retrieval-scope row renders from. ``None``
                renders as unscoped (hidden).
            **kwargs: Additional Textual widget arguments (id/classes).
        """
        classes = kwargs.pop("classes", "")
        # Reuse the existing chip-row class so its CSS continues to apply.
        super().__init__(
            classes=f"console-control-chip-row console-status-chips {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.scope_state = scope_state
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
        # task-10: the retrieval-scope chip -- unlike the chips above,
        # hidden entirely when unscoped rather than showing a
        # "Scope: everything" default (see ``_scope_chip_render``).
        yield self._scope_chip()

    def _scope_chip(self) -> ConsoleScopeChip:
        label, tooltip, hidden, alert = self._scope_chip_render(self.scope_state)
        chip = self._chip(
            label,
            id="console-scope-chip",
            emphasis=True if alert else None,
            chip_class=ConsoleScopeChip,
        )
        chip.tooltip = tooltip
        chip.display = not hidden
        return chip

    @staticmethod
    def _scope_chip_render(
        state: ConsoleRetrievalScopeState | None,
    ) -> tuple[str, str, bool, bool]:
        """Pure ``(label, tooltip, hidden, alert)`` render for the scope chip.

        ``state`` is conversation-scope-only until Phase 3 of the
        rag-scope-narrowing program wires workspace-level scoping through
        this same seam: ``item_count`` is simply the conversation scope's
        own (already zero-item-normalized -- see
        ``ConsoleRetrievalScopeState``) item count, and the scoped tooltip
        below will widen from "conversation N items" to the intersection
        breakdown ("conversation A ∩ workspace B → N") once that lands.

        Args:
            state: Display-state snapshot, or ``None`` (renders unscoped).

        Returns:
            ``label``: chip text. ``tooltip``: hover/focus text (the
            EMPTY branch folds the cause in). ``hidden``: ``True`` when
            unscoped (chip carries no useful information -- hidden rather
            than shown as "everything", matching the brief). ``alert``:
            ``True`` only for EMPTY, reusing the same
            ``console-chip-alert`` action-required styling the
            sources/tools/approvals chips use when their own count is
            active.
        """
        if state is None or (not state.is_scoped and not state.is_empty):
            return SCOPE_ROW_UNSCOPED_LABEL, "", True, False
        if state.is_empty:
            cause = state.cause or SCOPE_REASON_EMPTY
            return (
                "Scope: empty",
                SCOPE_EMPTY_NOTICE_TEMPLATE.format(cause=cause),
                False,
                True,
            )
        label = f"Scope: {state.item_count}"
        tooltip = f"conversation {state.item_count} items"
        return label, tooltip, False, False

    def sync_scope_chip(self, scope_state: ConsoleRetrievalScopeState | None) -> None:
        """Refresh the "Scope" chip from a new snapshot (task-10).

        Deliberately NOT folded into ``sync_state`` above: this is pushed
        directly from ``ChatScreen._sync_console_retrieval_scope_row`` with
        the exact same ``ConsoleRetrievalScopeState`` instance passed to the
        Inspector row's own ``sync_state`` in the same call -- one state,
        two renderers, computed once. Keeping it a separate method also
        keeps the chip's refresh triggers identical to the row's: the
        general ``sync_state`` refresh (called far more often, e.g. every
        control-bar sync tick) never touches this chip at all.

        Args:
            scope_state: Updated display-state snapshot to render.
        """
        if scope_state == self.scope_state:
            return
        self.scope_state = scope_state
        try:
            chip = self.query_one("#console-scope-chip", ConsoleScopeChip)
        except NoMatches:
            return
        label, tooltip, hidden, alert = self._scope_chip_render(scope_state)
        chip.update(label)
        chip.tooltip = tooltip
        chip.display = not hidden
        chip.set_class(alert, "console-chip-alert")

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
