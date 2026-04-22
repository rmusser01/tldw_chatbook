from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static


class ChatResumePanel(Container):
    """Inline resume summary that keeps the current task visible in chat."""

    def compose(self) -> ComposeResult:
        yield Static("Current task", id="resume-title")
        yield Static("", id="resume-summary")
        yield Static("", id="resume-last-step")
        yield Static("", id="resume-diff-summary")
        yield Static("", id="resume-next-action")

    def on_mount(self) -> None:
        self.display = False

    def set_resume_state(self, task_state) -> None:
        """Update the panel with the latest task summary."""
        self.display = task_state.has_resume_content()
        if not task_state.has_resume_content():
            return

        self.query_one("#resume-summary", Static).update(task_state.summary)
        self.query_one("#resume-last-step", Static).update(task_state.last_step)
        self.query_one("#resume-diff-summary", Static).update(task_state.diff_summary)
        self.query_one("#resume-next-action", Static).update(task_state.next_action)
