# tldw_chatbook/Widgets/cancel_confirmation_dialog.py
"""
Cancel confirmation dialog for media ingestion processes.
"""

from textual.binding import Binding

from .confirmation_dialog import ConfirmationDialog


class CancelConfirmationDialog(ConfirmationDialog):
    """Modal dialog for confirming cancellation of media processing.

    TASK-32864: this was a hand-rolled copy of ConfirmationDialog's shape
    with different accents; it is now a thin specialization of the
    documented pattern class (ADR-161). Only the accent CSS and the
    default wording differ; compose, bindings, safe dismissal, and the
    markup-disabled prose contract come from the parent.
    """

    DEFAULT_CSS = """
    CancelConfirmationDialog {
        align: center middle;
    }

    CancelConfirmationDialog > Container {
        border: thick $primary;
    }

    CancelConfirmationDialog .dialog-title {
        text-align: center;
        color: $error;
    }

    CancelConfirmationDialog .dialog-message {
        text-align: center;
    }

    CancelConfirmationDialog .button-container {
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    def __init__(
        self,
        title: str = "Cancel Transcription?",
        message: str = "Are you sure you want to cancel the transcription?\nAlready processed files will be kept.",
        confirm_text: str = "Yes, Cancel",
        cancel_text: str = "Continue Processing",
        **kwargs: object,
    ) -> None:
        """Initialize the cancel confirmation dialog.

        Args:
            title: Dialog title.
            message: Confirmation message.
            confirm_text: Text for the confirm (cancel-the-job) button.
            cancel_text: Text for the cancel (keep-going) button.
            **kwargs: Forwarded to ModalScreen (name/id/classes).
        """
        super().__init__(
            title=title,
            message=message,
            confirm_label=confirm_text,
            cancel_label=cancel_text,
            **kwargs,
        )
