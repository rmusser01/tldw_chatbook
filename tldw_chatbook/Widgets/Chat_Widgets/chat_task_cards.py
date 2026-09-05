from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)
from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
    SkillScriptConfirmCard,
)
from tldw_chatbook.Widgets.Chat_Widgets.watchlists_operation_card import (
    WatchlistsOperationCard,
)


class ChatTaskCards(Container):
    """Inline task-surface wrapper for approvals, skill-install/script, questions, and resume."""

    class QuestionAnswered(Message):
        """The user submitted the ask_user card (PRD Feature A).

        Defined here, not on the card, so ``ChatScreen``'s ``@on`` handler
        needs no import of the lazily-loaded card module (ADR-097).
        """

        def __init__(self, answers: list[dict[str, Any]], request_id: str | None) -> None:
            """Carry the answers and the round id they resolve.

            Args:
                answers: One PRD A6 answer dict per question, in order.
                request_id: The pending round's id, echoed back unchanged.
            """
            super().__init__()
            self.answers = answers
            self.request_id = request_id

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the surface hidden.

        task-17500: construction state, not an ``on_mount`` write. The
        screen's mount-time ``sync_task_resume_state`` (a 0.05s timer) and
        this widget's own Mount handler had no ordering contract, so a
        sync that landed first had its ``display = True`` clobbered by the
        late mount hide -- the same race family as the approval card's
        deferred batch-body hide, which produced the live title-only card.
        """
        super().__init__(*args, **kwargs)
        self.display = False

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")
        yield SkillInstallConfirmCard(id="chat-skill-install-card")
        yield SkillScriptConfirmCard(id="chat-skill-script-card")
        yield Vertical(id="console-watchlists-operation-cards")
        yield ChatResumePanel(id="chat-resume-panel")

    def _question_card(self, *, create: bool):
        """Return the question card, mounting it on first use.

        The card module is NOT imported at boot: ``ChatTaskCards`` composes
        during startup and ADR-097's UI-ready module census sits at its cap.
        The first pending question mounts the card after the skill-script
        card, so nothing about it loads until an agent actually asks.

        Args:
            create: Mount the card when it is absent.

        Returns:
            The ``ChatQuestionCard``, or None when absent and not created.
        """
        try:
            return self.query_one("#chat-question-card")
        except NoMatches:
            if not create:
                return None
            from tldw_chatbook.Widgets.Chat_Widgets.chat_question_card import (
                ChatQuestionCard,
            )

            card = ChatQuestionCard(id="chat-question-card")
            self.mount(card, after=self.query_one(SkillScriptConfirmCard))
            return card

    def sync_state(
        self,
        task_state,
        *,
        operation_rows: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Sync the approval, skill-install/script, and resume cards from task state.

        Args:
            task_state: The ``TaskResumeState`` snapshot to render.
            operation_rows: Current safe status projections keyed by canonical
                local Watchlists receipt ID.
        """
        operations_container = self.query_one("#console-watchlists-operation-cards")
        resume_panel = self.query_one(ChatResumePanel)

        # A pending approval payload (task-5) is a dict carrying a "calls"
        # list; `set_batch` also accepts an empty/absent list to mean
        # "nothing pending" (task-914: the once-legacy non-batch shape
        # produced by no live caller was removed alongside its dead
        # single-approval card body).
        # task-31384: one routing table, kind -> (state field, card, setter).
        # Each card keeps its own round-identity guard, so a re-sync for an
        # unrelated field never rebuilds a card the user is mid-decision on.
        for field, card in self._routes(task_state):
            card(getattr(task_state, field))
        self._sync_watchlists_operations(
            operations_container,
            task_state.followed_watchlists_operations,
            operation_rows or {},
        )
        resume_panel.set_resume_state(task_state)
        self.display = (
            task_state.has_pending_approval()
            or task_state.has_pending_skill_install()
            or task_state.has_pending_skill_script()
            or task_state.has_pending_question()
            or bool(task_state.followed_watchlists_operations)
            or task_state.has_resume_content()
        )

    def _routes(self, task_state):
        """The kind -> card routing table for one sync (task-31384).

        Args:
            task_state: The ``TaskResumeState`` being synced.

        Returns:
            ``(state field, setter)`` pairs in mount order. The question
            card's entry is present only once a question exists, because
            that card mounts lazily on first use (ADR-097).
        """
        yield ("pending_approval", self._set_approval)
        yield ("pending_skill_install", self.query_one(SkillInstallConfirmCard).set_install)
        yield ("pending_skill_script", self.query_one(SkillScriptConfirmCard).set_script)
        # Generated lazily so the question card is created (mounted) only
        # after the three fixed cards have synced, as before the table.
        question_card = self._question_card(create=bool(task_state.pending_question))
        if question_card is not None:
            yield ("pending_question", question_card.set_questions)

    def _set_approval(self, payload) -> None:
        """Route an approval payload (a dict carrying ``calls``) to the approval card.

        Args:
            payload: ``TaskResumeState.pending_approval``; anything but a
                dict means "nothing pending" (task-914 removed the legacy
                non-batch shape).
        """
        approval = payload if isinstance(payload, dict) else {}
        self.query_one(ChatApprovalCard).set_batch(
            approval.get("calls") or [],
            timeout_seconds=approval.get("timeout_seconds", 0.0),
            # Task 9 fix round 1: round-trip the round's stamped id so
            # a decision on THIS card resolves THIS round, never
            # "whichever session is active" -- see
            # `ConsoleChatController.resolve_pending_approval`.
            round_id=approval.get("round_id"),
            phase=str(approval.get("phase") or "approval"),
            # ADR-090 (task 5): the payload-carried advisory summary -- the
            # slot starts None and is filled by the advisory summarizer;
            # payload carriage means any remount re-renders it.
            summary=approval.get("summary"),
        )

    @staticmethod
    def _sync_watchlists_operations(
        container: Vertical,
        operation_ids: tuple[str, ...],
        operation_rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Reconcile receipt cards without retaining tool payloads."""
        wanted = set(operation_ids)
        current = {
            card.operation_id: card
            for card in container.query(WatchlistsOperationCard)
        }
        for operation_id, card in current.items():
            if operation_id not in wanted:
                card.remove()
        for operation_id in operation_ids:
            row = operation_rows.get(operation_id, {"id": operation_id})
            card = current.get(operation_id)
            if card is None:
                container.mount(
                    WatchlistsOperationCard(operation_id, operation=row)
                )
            else:
                card.set_operation(row)
