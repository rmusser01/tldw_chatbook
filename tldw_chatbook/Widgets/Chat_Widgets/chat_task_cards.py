from textual.app import ComposeResult
from textual.containers import Container
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)


class ChatTaskCards(Container):
    """Inline task-surface wrapper for approvals, skill-install, and resume."""

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")
        yield SkillInstallConfirmCard(id="chat-skill-install-card")
        yield ChatResumePanel(id="chat-resume-panel")

    def on_mount(self) -> None:
        self.display = False

    def sync_state(self, task_state) -> None:
        """Sync the approval, skill-install, and resume cards from task state."""
        approval_card = self.query_one(ChatApprovalCard)
        install_card = self.query_one(SkillInstallConfirmCard)
        resume_panel = self.query_one(ChatResumePanel)

        approval = task_state.pending_approval
        # A batch approval payload (task-5) is a dict carrying a "calls"
        # list -- anything else (None, or the legacy {"summary", "details",
        # ...} single-approval shape) goes through the original API
        # unchanged.
        calls = approval.get("calls") if isinstance(approval, dict) else None
        if calls:
            approval_card.set_batch(
                calls, timeout_seconds=approval.get("timeout_seconds", 0.0)
            )
        else:
            approval_card.set_approval(approval)
        install_card.set_install(task_state.pending_skill_install)
        resume_panel.set_resume_state(task_state)
        self.display = (
            task_state.has_pending_approval()
            or task_state.has_pending_skill_install()
            or task_state.has_resume_content()
        )
