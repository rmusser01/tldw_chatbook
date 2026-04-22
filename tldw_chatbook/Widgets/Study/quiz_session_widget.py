"""Focused shell widget for starting and monitoring quiz sessions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static


class QuizSessionWidget(Widget):
    """Compact shell-level summary for quiz launch and progress."""

    DEFAULT_CSS = """
    QuizSessionWidget {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #quiz-session {
        border: round $surface;
        padding: 1;
        width: 100%;
        height: auto;
    }

    .quiz-session-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .quiz-session-meta {
        color: $text-muted;
        margin-bottom: 1;
    }

    .quiz-session-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="quiz-session"):
            yield Static("Quiz Session", classes="quiz-session-title")
            yield Static("Global study", id="quiz-scope-summary", classes="quiz-session-meta")
            yield Static("Select a quiz to begin.", id="quiz-session-summary")
            yield Static("", id="quiz-session-status")
            with Horizontal(classes="quiz-session-actions"):
                yield Button("Start quiz", id="quiz-start", variant="primary", disabled=True)
                yield Button("Review in chat", id="quiz-open-in-chat")

    def update_scope_summary(self, summary: str) -> None:
        if self.is_mounted:
            self.query_one("#quiz-scope-summary", Static).update(summary)

    def update_session_summary(self, summary: str) -> None:
        if self.is_mounted:
            self.query_one("#quiz-session-summary", Static).update(summary)

    def update_status(self, status: str) -> None:
        if self.is_mounted:
            self.query_one("#quiz-session-status", Static).update(status)

    def set_start_enabled(self, enabled: bool) -> None:
        if self.is_mounted:
            self.query_one("#quiz-start", Button).disabled = not enabled
