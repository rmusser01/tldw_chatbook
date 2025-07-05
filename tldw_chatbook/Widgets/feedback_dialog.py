# feedback_dialog.py
# Description: Modal dialog for entering message ranking feedback
#
"""
Message Ranking Feedback Dialog
------------------------------

Provides a modal dialog for users to optionally add comments
when providing thumbs up/down feedback on messages.
"""

from typing import Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, TextArea, Static
from loguru import logger


class FeedbackDialog(ModalScreen):
    """Dialog for adding optional comments to message feedback."""
    
    # CSS for the dialog
    DEFAULT_CSS = """
    FeedbackDialog {
        align: center middle;
    }
    
    FeedbackDialog > Container {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
        max-height: 20;
    }
    
    FeedbackDialog .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    FeedbackDialog .feedback-type-label {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    FeedbackDialog .comment-area {
        margin-bottom: 1;
        height: 6;
        width: 100%;
    }
    
    FeedbackDialog .dialog-buttons {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    FeedbackDialog .dialog-buttons Button {
        margin: 0 1;
        width: 12;
    }
    """
    
    def __init__(self, 
                 feedback_type: str,  # "1" for thumbs up, "2" for thumbs down
                 existing_comment: Optional[str] = None,
                 callback: Optional[Callable[[Optional[tuple[str, str]]], None]] = None,
                 **kwargs):
        """
        Initialize the feedback dialog.
        
        Args:
            feedback_type: "1" for thumbs up, "2" for thumbs down
            existing_comment: Existing comment if editing feedback
            callback: Function to call with (feedback_type, comment) or None
        """
        super().__init__(**kwargs)
        self.feedback_type = feedback_type
        self.existing_comment = existing_comment or ""
        self.callback = callback
        
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        feedback_label = "👍 Thumbs Up" if self.feedback_type == "1" else "👎 Thumbs Down"
        
        with Container(classes="config-dialog"):
            yield Label("Message Ranking Feedback", classes="dialog-title")
            
            # Show which feedback type is selected
            yield Static(f"Feedback: {feedback_label}", classes="feedback-type-label")
            
            # Comment area
            yield Label("Additional Comments (optional):")
            yield TextArea(
                self.existing_comment,
                id="comment-area",
                classes="comment-area"
            )
            
            # Buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Save", id="save-button", variant="primary")
    
    @on(Button.Pressed, "#save-button")
    def handle_save(self):
        """Save the feedback with optional comment."""
        try:
            comment_widget = self.query_one("#comment-area", TextArea)
            comment = comment_widget.text.strip()
            
            # Create the feedback string
            if comment:
                feedback_str = f"{self.feedback_type};{comment}"
            else:
                feedback_str = f"{self.feedback_type};"
            
            logger.debug(f"Saving feedback: {feedback_str[:50]}...")
            
            if self.callback:
                self.callback((self.feedback_type, comment))
            
            self.dismiss(True)
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            if self.callback:
                self.callback(None)
            self.dismiss(False)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel without saving."""
        logger.debug("Feedback dialog cancelled")
        if self.callback:
            self.callback(None)
        self.dismiss(False)
    
    async def on_mount(self) -> None:
        """Focus the comment area when dialog opens."""
        try:
            comment_area = self.query_one("#comment-area", TextArea)
            comment_area.focus()
        except Exception:
            pass