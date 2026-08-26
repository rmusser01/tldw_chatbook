"""Startup recovery prompt for an interrupted first-run setup."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

SetupRecoveryResult = Literal["resume", "start_over", "later"]


class SetupRecoveryDialog(ModalScreen[SetupRecoveryResult]):
    """Offer the three bounded recovery actions and nothing else."""

    BINDINGS = [Binding("escape", "later", "Later", show=False)]

    BUNDLED_CSS = """
    SetupRecoveryDialog {
        align: center middle;
    }

    SetupRecoveryDialog > Container {
        width: 72;
        max-width: 92%;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    SetupRecoveryDialog .setup-recovery-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    SetupRecoveryDialog .setup-recovery-message {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    SetupRecoveryDialog .setup-recovery-actions {
        width: 100%;
        height: auto;
    }

    SetupRecoveryDialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.message = (
            "A previous setup stopped before it finished. Resume from the last "
            "completed step, start over, or continue later. Credentials are not "
            "retained in setup recovery and may need to be re-entered."
        )

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Continue setup?", classes="setup-recovery-title")
            yield Static(self.message, classes="setup-recovery-message")
            with Vertical(classes="setup-recovery-actions"):
                yield Button(
                    "Resume",
                    id="setup-recovery-resume",
                    variant="primary",
                )
                yield Button("Start over", id="setup-recovery-start_over")
                yield Button("Later", id="setup-recovery-later")

    def on_mount(self) -> None:
        self.query_one("#setup-recovery-resume", Button).focus()

    @on(Button.Pressed)
    def handle_action(self, event: Button.Pressed) -> None:
        action = (event.button.id or "").removeprefix("setup-recovery-")
        if action in {"resume", "start_over", "later"}:
            self.dismiss(action)

    def action_later(self) -> None:
        self.dismiss("later")
