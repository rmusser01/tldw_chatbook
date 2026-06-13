from textual.app import ComposeResult
from textual.containers import Container
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel


class ChatTaskCards(Container):
    """Inline task-surface wrapper for approvals and resume context."""

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")
        yield ChatResumePanel(id="chat-resume-panel")

    def on_mount(self) -> None:
        self.display = False

    def sync_state(self, task_state) -> None:
        """Sync the approval and resume cards from the latest task state."""
        approval_card = self.query_one(ChatApprovalCard)
        resume_panel = self.query_one(ChatResumePanel)

        approval_card.set_approval(task_state.pending_approval)
        resume_panel.set_resume_state(task_state)
        self.display = task_state.has_pending_approval() or task_state.has_resume_content()
