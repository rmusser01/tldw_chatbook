# Helper for transitioning from TextArea to EnhancedStatusWidget
"""
Provides compatibility layer for status updates during migration from TextArea to EnhancedStatusWidget.
"""

from typing import Union, Optional
from textual.widgets import TextArea
from textual.css.query import QueryError
from loguru import logger


def update_status(app, widget_id: str, message: str, level: str = "info", append: bool = True) -> bool:
    """
    Update status in either TextArea or EnhancedStatusWidget.
    
    Args:
        app: The app instance
        widget_id: The ID of the status widget (without # prefix)
        message: The status message
        level: Message level (info, success, warning, error, debug)
        append: Whether to append (True) or replace (False) the content
    
    Returns:
        bool: True if update was successful
    """
    try:
        # Try to find EnhancedStatusWidget first (new approach)
        from ..Widgets.status_widget import EnhancedStatusWidget
        try:
            widget = app.query_one(f"#{widget_id}", EnhancedStatusWidget)
            # Use the appropriate method based on level
            if level == "info":
                widget.add_info(message)
            elif level == "success":
                widget.add_success(message)
            elif level == "warning":
                widget.add_warning(message)
            elif level == "error":
                widget.add_error(message)
            elif level == "debug":
                widget.add_debug(message)
            else:
                widget.add_message(message, level)
            return True
        except QueryError:
            pass  # Not an EnhancedStatusWidget, try TextArea
        
        # Fallback to TextArea (old approach)
        widget = app.query_one(f"#{widget_id}", TextArea)
        if append and widget.text:
            widget.text += "\n" + message
        else:
            widget.text = message
        
        # Apply color based on level for TextArea
        if level == "error":
            widget.styles.color = "red"
        elif level == "warning":
            widget.styles.color = "yellow"
        elif level == "success":
            widget.styles.color = "green"
            
        return True
        
    except QueryError as e:
        logger.error(f"Status widget {widget_id} not found: {e}")
        # Show notification as fallback
        severity = "error" if level == "error" else "warning" if level == "warning" else "information"
        app.notify(message, severity=severity)
        return False


def clear_status(app, widget_id: str) -> bool:
    """
    Clear status in either TextArea or EnhancedStatusWidget.
    
    Args:
        app: The app instance
        widget_id: The ID of the status widget (without # prefix)
    
    Returns:
        bool: True if clear was successful
    """
    try:
        # Try EnhancedStatusWidget first
        from ..Widgets.status_widget import EnhancedStatusWidget
        try:
            widget = app.query_one(f"#{widget_id}", EnhancedStatusWidget)
            widget.clear()
            return True
        except QueryError:
            pass
        
        # Fallback to TextArea
        widget = app.query_one(f"#{widget_id}", TextArea)
        widget.clear()
        return True
        
    except QueryError as e:
        logger.error(f"Status widget {widget_id} not found for clearing: {e}")
        return False


# Mapping of old TextArea IDs to new EnhancedStatusWidget IDs
STATUS_WIDGET_MAPPING = {
    "prompt-import-status-area": "prompt-import-status-widget",
    "ingest-character-import-status-area": "ingest-character-import-status-widget",
    "ingest-notes-import-status-area": "ingest-notes-import-status-widget",
}


def get_status_widget_id(old_id: str) -> str:
    """Get the new widget ID from the old TextArea ID."""
    return STATUS_WIDGET_MAPPING.get(old_id, old_id)