"""Late-bound feedback projection without retaining approval DOM."""

from __future__ import annotations
from collections.abc import Callable
from textual.message import Message
from tldw_chatbook.Chat.console_approval_feedback import (
    ApprovalFeedback,
    format_approval_feedback,
)


class ApprovalFeedbackChanged(Message):
    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.run_id = run_id


class ApprovalFeedbackController:
    def __init__(
        self,
        *,
        read_snapshot: Callable[[str, str], tuple[ApprovalFeedback, ...]],
        active_session: Callable[[], str],
        card_identity: Callable[[], tuple[str, int] | None],
        paint_card: Callable[[str], None],
        refresh_status: Callable[[], None],
    ) -> None:
        self._read_snapshot = read_snapshot
        self._active_session = active_session
        self._card_identity = card_identity
        self._paint_card = paint_card
        self._refresh_status = refresh_status

    def refresh(self, session_id: str, run_id: str) -> None:
        if self._active_session() != session_id:
            return
        identity = self._card_identity()
        facts = self._read_snapshot(session_id, run_id)
        matching = tuple(
            fact
            for fact in facts
            if identity == (fact.identity.round_id, fact.identity.revision)
        )
        if matching:
            text = "\n".join(
                dict.fromkeys(format_approval_feedback(fact) for fact in matching)
            )
            if text:
                self._paint_card(text)
        self._refresh_status()
