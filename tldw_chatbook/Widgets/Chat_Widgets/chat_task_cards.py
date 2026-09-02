from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Container, Vertical
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
    """Inline task-surface wrapper for approvals, skill-install/script, and resume."""

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
        approval_card = self.query_one(ChatApprovalCard)
        install_card = self.query_one(SkillInstallConfirmCard)
        script_card = self.query_one(SkillScriptConfirmCard)
        operations_container = self.query_one("#console-watchlists-operation-cards")
        resume_panel = self.query_one(ChatResumePanel)

        # A pending approval payload (task-5) is a dict carrying a "calls"
        # list; `set_batch` also accepts an empty/absent list to mean
        # "nothing pending" (task-914: the once-legacy non-batch shape
        # produced by no live caller was removed alongside its dead
        # single-approval card body).
        approval = task_state.pending_approval
        approval = approval if isinstance(approval, dict) else {}
        approval_card.set_batch(
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
        install_card.set_install(task_state.pending_skill_install)
        script_card.set_script(task_state.pending_skill_script)
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
            or bool(task_state.followed_watchlists_operations)
            or task_state.has_resume_content()
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
