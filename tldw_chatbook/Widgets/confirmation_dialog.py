# confirmation_dialog.py
# Description: Modal confirmation dialog for unsaved changes and other confirmations
#
# Imports
from typing import Optional, Callable
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static
#
#######################################################################################################################
#
# Classes:

class ConfirmationDialog(ModalScreen):
    """
    A modal confirmation dialog for user actions.
    
    This dialog displays a message and provides confirm/cancel options.
    """
    
    # CSS for styling
    DEFAULT_CSS = """
    ConfirmationDialog {
        align: center middle;
    }
    
    ConfirmationDialog > Container {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    ConfirmationDialog .dialog-title {
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }
    
    ConfirmationDialog .dialog-message {
        margin-bottom: 2;
        width: 100%;
    }
    
    ConfirmationDialog .button-container {
        align: center middle;
        height: 3;
        width: 100%;
    }
    
    ConfirmationDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    ConfirmationDialog .confirm-button {
        background: $error;
    }
    
    ConfirmationDialog .cancel-button {
        background: $primary;
    }
    """
    
    def __init__(
        self,
        title: str = "Confirm Action",
        message: str = "Are you sure you want to proceed?",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        confirm_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the confirmation dialog.
        
        Args:
            title: Dialog title
            message: Message to display
            confirm_label: Label for confirm button
            cancel_label: Label for cancel button
            confirm_callback: Callback when confirmed
            cancel_callback: Callback when cancelled
        """
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.confirm_callback = confirm_callback
        self.cancel_callback = cancel_callback
        self.result = None
    
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        with Container():
            yield Static(self.title, classes="dialog-title")
            yield Label(self.message, classes="dialog-message")
            
            with Horizontal(classes="button-container"):
                yield Button(
                    self.cancel_label,
                    id="cancel-button",
                    classes="cancel-button",
                    variant="primary"
                )
                yield Button(
                    self.confirm_label,
                    id="confirm-button",
                    classes="confirm-button",
                    variant="error"
                )
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "confirm-button":
            self.result = True
            if self.confirm_callback:
                await self.confirm_callback()
            self.dismiss(True)
        elif event.button.id == "cancel-button":
            self.result = False
            if self.cancel_callback:
                await self.cancel_callback()
            self.dismiss(False)


class UnsavedChangesDialog(ConfirmationDialog):
    """
    Specialized confirmation dialog for unsaved changes.
    """
    
    def __init__(
        self,
        tab_title: str = "Untitled",
        confirm_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize unsaved changes dialog.
        
        Args:
            tab_title: Title of the tab with unsaved changes
            confirm_callback: Callback when user confirms close
            cancel_callback: Callback when user cancels
        """
        super().__init__(
            title="Unsaved Changes",
            message=f'The tab "{tab_title}" has unsaved changes.\n\nAre you sure you want to close it?',
            confirm_label="Close Without Saving",
            cancel_label="Keep Open",
            confirm_callback=confirm_callback,
            cancel_callback=cancel_callback,
            **kwargs
        )

#
# End of confirmation_dialog.py
#######################################################################################################################