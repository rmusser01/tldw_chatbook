"""Keyboard-readable restored-root details with persistent confirmation choices."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Label, Static

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class MCPRecoveryReviewDialog(ConfirmationDialog):
    """Keep long root paths scrollable independently of the safe-cancel actions."""

    AUTO_FOCUS = "#mcp-recovery-review-body"

    def compose(self) -> ComposeResult:
        """Compose the existing confirmation contract around a focusable body.

        Returns:
            ComposeResult: An iterator yielding the title, scrollable review body,
                and standard Cancel and confirmation actions.
        """
        with Container(id="confirmation-dialog"):
            yield Static(self.title, classes="dialog-title")
            with VerticalScroll(id="mcp-recovery-review-body"):
                yield Label(self.message, classes="dialog-message")
            with Horizontal(classes="button-container"):
                yield Button(
                    self.cancel_label,
                    id="cancel-button",
                    classes="cancel-button",
                    variant="primary",
                )
                yield Button(
                    self.confirm_label,
                    id="confirm-button",
                    classes="confirm-button",
                    variant="error",
                )
