# NotificationHelper.py
# Description: Helper functions for transitioning from notify to toast notifications
#
# Imports
from typing import TYPE_CHECKING, Optional
#
# Type checking imports
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

def show_notification(
    app: 'TldwCli',
    message: str,
    severity: str = "information",
    timeout: Optional[float] = None
) -> None:
    """
    Show a notification using toast if available, fallback to notify.
    Maps Textual's notify severity levels to toast severity levels.
    
    Args:
        app: The app instance
        message: Message to display
        severity: Textual severity level (information, warning, error)
        timeout: Timeout in seconds
    """
    # Map Textual severity to toast severity
    severity_map = {
        "information": "info",
        "warning": "warning", 
        "error": "error"
    }
    
    toast_severity = severity_map.get(severity, "info")
    
    # Try to use toast notifications if available
    if hasattr(app, 'show_toast'):
        # Default timeout if not specified
        if timeout is None:
            timeout = 5.0 if severity != "error" else None  # Errors stay longer
            
        app.show_toast(
            message=message,
            severity=toast_severity,
            timeout=timeout,
            persistent=(timeout is None)
        )
    else:
        # Fallback to built-in notify
        app.notify(message, severity=severity, timeout=timeout)

#
# End of NotificationHelper.py
#######################################################################################################################