# delete_confirmation_dialog.py
# Description: Specialized confirmation dialog for delete operations
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
# Local Imports
from .confirmation_dialog import ConfirmationDialog
#
#######################################################################################################################
#
# Classes:

class DeleteConfirmationDialog(ConfirmationDialog):
    """
    Specialized confirmation dialog for delete operations.
    
    Provides consistent UI/UX for all delete confirmations with 
    appropriate warnings based on the type of item being deleted.
    """
    
    # CSS for additional styling
    DEFAULT_CSS = """
    DeleteConfirmationDialog {
        align: center middle;
    }
    
    DeleteConfirmationDialog > Container {
        width: 70;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    
    DeleteConfirmationDialog .dialog-title {
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
        color: $error;
    }
    
    DeleteConfirmationDialog .dialog-message {
        margin-bottom: 1;
        width: 100%;
    }
    
    DeleteConfirmationDialog .warning-message {
        margin-bottom: 2;
        width: 100%;
        color: $warning;
        text-style: italic;
    }
    
    DeleteConfirmationDialog .item-details {
        margin-bottom: 2;
        padding: 0 1;
        width: 100%;
    }
    
    DeleteConfirmationDialog .button-container {
        align: center middle;
        height: auto;
        width: 100%;
        padding: 1 0;
    }
    
    DeleteConfirmationDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    DeleteConfirmationDialog .confirm-button {
        background: $error;
    }
    
    DeleteConfirmationDialog .cancel-button {
        background: $primary;
    }
    """
    
    def __init__(
        self,
        item_type: str,
        item_name: str = "",
        additional_warning: str = "",
        permanent: bool = False,
        confirm_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the delete confirmation dialog.
        
        Args:
            item_type: Type of item being deleted (e.g., "Character", "Conversation")
            item_name: Name/identifier of the item being deleted
            additional_warning: Optional additional warning message
            permanent: Whether this is a permanent deletion (vs soft delete)
            confirm_callback: Callback when deletion is confirmed
            cancel_callback: Callback when deletion is cancelled
        """
        # Generate appropriate messages based on item type
        title = f"Delete {item_type}?"
        
        # Build the main message
        if item_name:
            message = f'Are you sure you want to delete the {item_type.lower()} "{item_name}"?'
        else:
            message = f"Are you sure you want to delete this {item_type.lower()}?"
        
        # Add permanent deletion warning if applicable
        if permanent:
            message += "\n\nThis action cannot be undone!"
        
        # Initialize parent with generated messages
        super().__init__(
            title=title,
            message=message,
            confirm_label="Delete",
            cancel_label="Cancel",
            confirm_callback=confirm_callback,
            cancel_callback=cancel_callback,
            **kwargs
        )
        
        self.item_type = item_type
        self.item_name = item_name
        self.additional_warning = additional_warning
        self.permanent = permanent
    
    def compose(self) -> ComposeResult:
        """Compose the dialog UI with additional warning if provided."""
        with Container():
            yield Static(self.title, classes="dialog-title")
            yield Label(self.message, classes="dialog-message")
            
            # Add item details if name is provided
            if self.item_name:
                yield Static(f"{self.item_type}: {self.item_name}", classes="item-details")
            
            # Add additional warning if provided
            if self.additional_warning:
                yield Label(self.additional_warning, classes="warning-message")
            
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


def create_delete_confirmation(
    item_type: str,
    item_name: str = "",
    additional_warning: str = "",
    permanent: bool = False
) -> DeleteConfirmationDialog:
    """
    Factory function to create delete confirmation dialogs with appropriate warnings.
    
    Args:
        item_type: Type of item being deleted
        item_name: Name/identifier of the item
        additional_warning: Optional additional warning
        permanent: Whether this is permanent deletion
        
    Returns:
        Configured DeleteConfirmationDialog instance
    """
    # Add default warnings based on item type
    default_warnings = {
        "Character": "All associated conversation data will be preserved.",
        "Conversation": "This will remove the conversation from your history.",
        "Prompt": "This prompt will no longer be available for use.",
        "Dictionary": "This dictionary will be removed from all conversations using it.",
        "Media": "This will permanently remove the media and all associated data.",
        "Note": "This note will be moved to trash and can be recovered later.",
        "Embedding": "This will delete the vector embeddings for this item.",
        "Collection": "All embeddings in this collection will be permanently deleted.",
        "Template": "This template will no longer be available for creating new items.",
        "Transcription": "The transcription history for this item will be removed.",
        "Model": "This will uninstall the model from your system."
    }
    
    # Use provided warning or default based on item type
    if not additional_warning and item_type in default_warnings:
        additional_warning = default_warnings[item_type]
    
    return DeleteConfirmationDialog(
        item_type=item_type,
        item_name=item_name,
        additional_warning=additional_warning,
        permanent=permanent
    )

#
# End of delete_confirmation_dialog.py
#######################################################################################################################