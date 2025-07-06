# eval_error_dialog.py
# Description: Enhanced error dialog for evaluation system
#
"""
Evaluation Error Dialog Widget
-----------------------------

Provides a user-friendly error dialog with:
- Clear error messages
- Technical details (expandable)
- Suggested actions
- Retry options
"""

from typing import Dict, Any, Optional, Callable
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid, ScrollableContainer
from textual.widgets import Button, Label, Static, TextArea, Collapsible
from textual.screen import ModalScreen
from textual.reactive import reactive
from loguru import logger

from ..Evals.eval_errors import ErrorContext, ErrorSeverity, EvaluationError

class ErrorDetailsDialog(ModalScreen):
    """Modal dialog for displaying error details."""
    
    CSS = """
    ErrorDetailsDialog {
        align: center middle;
    }
    
    #error-dialog {
        width: 80;
        height: auto;
        max-height: 40;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }
    
    #error-title {
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }
    
    #error-message {
        margin-bottom: 1;
    }
    
    #error-suggestion {
        background: $primary 20%;
        padding: 1;
        margin: 1 0;
        border: solid $primary 50%;
    }
    
    #error-actions {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    #error-actions Button {
        margin: 0 1;
    }
    
    .error-severity-critical {
        color: $error;
        text-style: bold;
    }
    
    .error-severity-error {
        color: $error;
    }
    
    .error-severity-warning {
        color: $warning;
    }
    
    .error-severity-information {
        color: $primary;
    }
    
    #technical-details {
        margin-top: 1;
    }
    
    #details-content {
        background: $background;
        padding: 1;
        height: auto;
        max-height: 15;
    }
    """
    
    def __init__(self, 
                 error: Exception,
                 title: str = "Evaluation Error",
                 on_retry: Optional[Callable] = None,
                 on_dismiss: Optional[Callable] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.error = error
        self.title = title
        self.on_retry = on_retry
        self.on_dismiss = on_dismiss
        self.error_context = self._extract_error_context(error)
    
    def _extract_error_context(self, error: Exception) -> Optional[ErrorContext]:
        """Extract error context from exception."""
        if isinstance(error, EvaluationError):
            return error.context
        elif hasattr(error, 'context') and isinstance(error.context, ErrorContext):
            return error.context
        return None
    
    def compose(self) -> ComposeResult:
        with Container(id="error-dialog"):
            # Title
            severity_class = ""
            if self.error_context:
                severity_class = f"error-severity-{self.error_context.severity.value}"
            
            yield Label(self.title, id="error-title", classes=severity_class)
            
            # Main error message
            error_message = str(self.error)
            if self.error_context:
                error_message = self.error_context.message
            
            yield Static(error_message, id="error-message")
            
            # Suggestion box (if available)
            if self.error_context and self.error_context.suggestion:
                with Container(id="error-suggestion"):
                    yield Label("💡 Suggestion", classes="suggestion-title")
                    yield Static(self.error_context.suggestion)
            
            # Technical details (collapsible)
            if self.error_context:
                with Collapsible(title="Technical Details", id="technical-details", collapsed=True):
                    details = self._format_technical_details()
                    yield TextArea(details, id="details-content", read_only=True)
            
            # Actions
            with Horizontal(id="error-actions"):
                if self.error_context and self.error_context.is_retryable and self.on_retry:
                    yield Button("Retry", id="retry-button", variant="primary")
                
                yield Button("Copy Details", id="copy-button", variant="default")
                yield Button("Dismiss", id="dismiss-button", variant="default")
    
    def _format_technical_details(self) -> str:
        """Format technical details for display."""
        details = []
        
        if self.error_context:
            details.append(f"Category: {self.error_context.category.value}")
            details.append(f"Severity: {self.error_context.severity.value}")
            details.append(f"Timestamp: {self.error_context.timestamp.isoformat()}")
            
            if self.error_context.error_code:
                details.append(f"Error Code: {self.error_context.error_code}")
            
            if self.error_context.details:
                details.append(f"\nDetails:\n{self.error_context.details}")
            
            if self.error_context.retry_after:
                details.append(f"\nRetry After: {self.error_context.retry_after} seconds")
        
        # Add original error info if it's an EvaluationError
        if isinstance(self.error, EvaluationError) and self.error.original_error:
            details.append(f"\nOriginal Error: {type(self.error.original_error).__name__}")
            details.append(f"{str(self.error.original_error)}")
        
        # Add stack trace for debugging (in debug mode)
        import traceback
        if logger.level <= 10:  # DEBUG level
            details.append("\nStack Trace:")
            details.append(traceback.format_exc())
        
        return "\n".join(details)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "retry-button":
            if self.on_retry:
                self.on_retry()
            self.dismiss()
        
        elif event.button.id == "copy-button":
            # Copy technical details to clipboard
            try:
                import pyperclip
                details = self._format_technical_details()
                pyperclip.copy(details)
                self.notify("Error details copied to clipboard", severity="information")
            except:
                self.notify("Could not copy to clipboard", severity="warning")
        
        elif event.button.id == "dismiss-button":
            if self.on_dismiss:
                self.on_dismiss()
            self.dismiss()

class ErrorSummaryWidget(Container):
    """Widget for displaying error summary in the main UI."""
    
    error_count: reactive[int] = reactive(0)
    
    CSS = """
    ErrorSummaryWidget {
        height: auto;
        background: $error 10%;
        border: solid $error 50%;
        padding: 1;
        margin: 1 0;
    }
    
    #error-summary-title {
        text-style: bold;
        color: $error;
    }
    
    .error-item {
        margin-left: 2;
        margin-top: 0;
    }
    
    #show-all-errors {
        margin-top: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.errors: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        yield Label("⚠️ Errors Encountered", id="error-summary-title")
        yield Container(id="error-list")
        yield Button("Show All Errors", id="show-all-errors", variant="error")
    
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the summary."""
        error_info = {
            'error': error,
            'context': context,
            'timestamp': datetime.now()
        }
        
        self.errors.append(error_info)
        self.error_count = len(self.errors)
        self.update_display()
    
    def update_display(self):
        """Update the error display."""
        error_list = self.query_one("#error-list")
        error_list.remove_children()
        
        # Show last 3 errors
        recent_errors = self.errors[-3:]
        
        for error_info in recent_errors:
            error = error_info['error']
            
            # Extract message
            if isinstance(error, EvaluationError):
                message = error.context.message
                category = error.context.category.value
            else:
                message = str(error)
                category = "unknown"
            
            # Truncate if too long
            if len(message) > 60:
                message = message[:57] + "..."
            
            error_item = Static(
                f"• [{category}] {message}",
                classes="error-item"
            )
            error_list.mount(error_item)
        
        # Show count if more errors
        if len(self.errors) > 3:
            more_count = len(self.errors) - 3
            error_list.mount(Static(f"... and {more_count} more", classes="error-item"))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press to show all errors."""
        if event.button.id == "show-all-errors":
            # Would open a dialog with all errors
            self.notify("Error history dialog not implemented yet", severity="information")
    
    def clear_errors(self):
        """Clear all errors."""
        self.errors.clear()
        self.error_count = 0
        self.update_display()

def show_error_dialog(app, error: Exception, 
                     title: str = "Evaluation Error",
                     allow_retry: bool = True) -> None:
    """Helper function to show error dialog."""
    
    def on_retry():
        app.notify("Retrying operation...", severity="information")
        # Actual retry logic would be implemented by caller
    
    def on_dismiss():
        logger.info("Error dialog dismissed by user")
    
    # Determine if retry is allowed based on error type
    can_retry = allow_retry
    if isinstance(error, EvaluationError):
        can_retry = allow_retry and error.context.is_retryable
    
    dialog = ErrorDetailsDialog(
        error=error,
        title=title,
        on_retry=on_retry if can_retry else None,
        on_dismiss=on_dismiss
    )
    
    app.push_screen(dialog)

# Enhanced notification helper
def notify_error(app, error: Exception, title: str = None) -> None:
    """Show error notification with appropriate severity."""
    if isinstance(error, EvaluationError):
        # Use error context for better notifications
        context = error.context
        message = context.message
        
        if context.suggestion:
            message += f" - {context.suggestion}"
        
        # Map severity
        severity_map = {
            ErrorSeverity.CRITICAL: "error",
            ErrorSeverity.ERROR: "error",
            ErrorSeverity.WARNING: "warning",
            ErrorSeverity.INFO: "information"
        }
        
        severity = severity_map.get(context.severity, "error")
        
        app.notify(message, severity=severity, title=title)
    else:
        # Generic error notification
        app.notify(f"Error: {str(error)}", severity="error", title=title)