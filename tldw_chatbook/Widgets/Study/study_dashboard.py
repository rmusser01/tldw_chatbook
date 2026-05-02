"""Shell-level dashboard for the Study destination."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static


RESUME_DISABLED_TOOLTIP = (
    "No study session to resume. Open flashcards or quizzes to start a session."
)
RESUME_ENABLED_TOOLTIP = "Resume the most recent study session."


class StudyDashboard(Widget):
    """Summarize due work, recents, and resume actions."""

    DEFAULT_CSS = """
    StudyDashboard {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #study-dashboard {
        border: round $surface;
        padding: 1;
        width: 100%;
        height: auto;
    }

    .study-dashboard-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .study-dashboard-meta {
        color: $text-muted;
        margin-bottom: 1;
    }

    .study-dashboard-column {
        width: 1fr;
        margin-right: 1;
    }

    .study-dashboard-heading {
        text-style: bold;
        margin-bottom: 1;
    }

    .study-dashboard-value {
        margin-bottom: 1;
    }

    .study-dashboard-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="study-dashboard"):
            yield Static("Study Dashboard", classes="study-dashboard-title")
            yield Static("Global study", id="study-scope-summary", classes="study-dashboard-meta")
            with Horizontal():
                with Vertical(classes="study-dashboard-column"):
                    yield Static("Due Today", classes="study-dashboard-heading")
                    yield Static("0 due today", id="study-due-today", classes="study-dashboard-value")
                with Vertical(classes="study-dashboard-column"):
                    yield Static("Recent Decks", classes="study-dashboard-heading")
                    yield Static("No recent decks yet.", id="study-recent-decks", classes="study-dashboard-value")
                with Vertical(classes="study-dashboard-column"):
                    yield Static("Recent Quizzes", classes="study-dashboard-heading")
                    yield Static("No recent quizzes yet.", id="study-recent-quizzes", classes="study-dashboard-value")
            with Horizontal(classes="study-dashboard-actions"):
                yield Button(
                    "Resume last session",
                    id="study-resume-last",
                    disabled=True,
                    variant="primary",
                    tooltip=RESUME_DISABLED_TOOLTIP,
                )
                yield Button("Open flashcards", id="study-open-flashcards")
                yield Button("Open quizzes", id="study-open-quizzes")

    def update_scope_summary(self, summary: str) -> None:
        if self.is_mounted:
            self.query_one("#study-scope-summary", Static).update(summary)

    def update_due_today(self, due_count: int) -> None:
        if self.is_mounted:
            self.query_one("#study-due-today", Static).update(f"{due_count} due today")

    def update_recent_decks(self, deck_names: list[str]) -> None:
        text = ", ".join(deck_names) if deck_names else "No recent decks yet."
        if self.is_mounted:
            self.query_one("#study-recent-decks", Static).update(text)

    def update_recent_quizzes(self, quiz_names: list[str]) -> None:
        text = ", ".join(quiz_names) if quiz_names else "No recent quizzes yet."
        if self.is_mounted:
            self.query_one("#study-recent-quizzes", Static).update(text)

    def update_resume_action(self, summary: str | None) -> None:
        if not self.is_mounted:
            return
        button = self.query_one("#study-resume-last", Button)
        if summary:
            button.label = f"Resume {summary}"
            button.disabled = False
            button.tooltip = RESUME_ENABLED_TOOLTIP
        else:
            button.label = "Resume last session"
            button.disabled = True
            button.tooltip = RESUME_DISABLED_TOOLTIP
