from textual.app import ComposeResult
from textual.containers import Container
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)
from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
    SkillScriptConfirmCard,
)


class ChatTaskCards(Container):
    """Inline task-surface wrapper for approvals, skill-install/script, and resume."""

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")
        yield SkillInstallConfirmCard(id="chat-skill-install-card")
        yield SkillScriptConfirmCard(id="chat-skill-script-card")
        yield ChatResumePanel(id="chat-resume-panel")

    def on_mount(self) -> None:
        self.display = False

    def sync_state(self, task_state) -> None:
        """Sync the approval, skill-install/script, and resume cards from task state.

        Args:
            task_state: The ``TaskResumeState`` snapshot (pending_approval,
                pending_skill_install, pending_skill_script, and resume
                fields) to render.
        """
        approval_card = self.query_one(ChatApprovalCard)
        install_card = self.query_one(SkillInstallConfirmCard)
        script_card = self.query_one(SkillScriptConfirmCard)
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
        )
        install_card.set_install(task_state.pending_skill_install)
        script_card.set_script(task_state.pending_skill_script)
        resume_panel.set_resume_state(task_state)
        self.display = (
            task_state.has_pending_approval()
            or task_state.has_pending_skill_install()
            or task_state.has_pending_skill_script()
            or task_state.has_resume_content()
        )
