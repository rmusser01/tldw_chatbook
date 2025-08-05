# tldw_chatbook/Widgets/cancel_confirmation_dialog.py
"""
Cancel confirmation dialog for media ingestion processes.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Container
from textual.screen import ModalScreen
from textual.widgets import Label, Button, Static


class CancelConfirmationDialog(ModalScreen[bool]):
    """Modal dialog for confirming cancellation of media processing."""
    
    DEFAULT_CSS = """
    CancelConfirmationDialog {
        align: center middle;
    }
    
    CancelConfirmationDialog > Container {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    CancelConfirmationDialog .dialog-title {
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }
    
    CancelConfirmationDialog .dialog-message {
        text-align: center;
        margin-bottom: 1;
    }
    
    CancelConfirmationDialog .button-container {
        align: center middle;
        height: auto;
        width: 100%;
        margin-top: 1;
    }
    
    CancelConfirmationDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    def __init__(
        self,
        title: str = "Cancel Transcription?",
        message: str = "Are you sure you want to cancel the transcription?\nAlready processed files will be kept.",
        confirm_text: str = "Yes, Cancel",
        cancel_text: str = "Continue Processing",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the cancel confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
        """
        super().__init__(name=name, id=id, classes=classes)
        self.title_text = title
        self.message_text = message
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
    
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        with Container():
            yield Static(self.title_text, classes="dialog-title")
            yield Label(self.message_text, classes="dialog-message")
            with Horizontal(classes="button-container"):
                yield Button(self.cancel_text, variant="primary", id="continue-btn")
                yield Button(self.confirm_text, variant="error", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.dismiss(True)  # User confirmed cancellation
        else:
            self.dismiss(False)  # User wants to continue processing