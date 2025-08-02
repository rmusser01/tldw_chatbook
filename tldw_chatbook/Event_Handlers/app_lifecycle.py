# app_lifecycle.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING

from textual.css.query import QueryError
#
# 3rd-Party Imports
from textual.widgets import RichLog, Button

#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_copy_logs_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Copy All Logs to Clipboard' button press."""
    logger = getattr(app, 'loguru_logger', logging)  # Use app's logger
    logger.info("Copy logs button pressed.")
    try:
        # Use the actual RichLog type, not a string
        log_widget = app.query_one("#app-log-display", RichLog)  # <--- FIX HERE

        logger.info(f"RichLog widget found. Number of lines: {len(log_widget.lines)}")
        logger.info(f"Type of lines: {type(log_widget.lines)}")
        
        # Check if the widget has the expected attributes
        logger.info(f"Widget has .lines: {hasattr(log_widget, 'lines')}")
        logger.info(f"Widget mounted: {log_widget.is_mounted}")
        
        if not log_widget.lines:
            logger.warning("RichLog widget has no lines!")
            app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
            return
            
        if log_widget.lines:
            # Extract plain text from Strip objects
            all_log_text_parts = []
            for i, strip in enumerate(log_widget.lines):
                # Debug: log the type of the first few items
                if i < 3:
                    logger.debug(f"Line {i} type: {type(strip)}, has .text: {hasattr(strip, 'text')}")
                
                if hasattr(strip, 'text'):  # Strip objects have a .text attribute
                    text = strip.text
                    all_log_text_parts.append(text)
                    if i < 3:  # Log first few lines for debugging
                        logger.debug(f"Line {i} text: {text[:50]}...")
                else:
                    # Fallback - try to convert to string
                    text = str(strip)
                    all_log_text_parts.append(text)
                    logger.warning(f"Line {i} doesn't have .text attribute, using str(): {text[:50]}...")

            all_log_text = "\n".join(all_log_text_parts)
            logger.info(f"Total text length: {len(all_log_text)} characters")
            
            if not all_log_text or all_log_text.isspace():
                logger.warning("Extracted text is empty or only whitespace!")
                app.notify("Log appears to be empty.", title="Error", severity="warning", timeout=4)
                return

            # Try to copy to clipboard
            logger.info(f"Attempting to copy {len(all_log_text)} characters to clipboard...")
            app.copy_to_clipboard(all_log_text)  # Assuming app has this method
            app.notify(
                "Logs copied to clipboard!",
                title="Clipboard",
                severity="information",
                timeout=4
            )
            logger.debug(
                f"Copied {len(log_widget.lines)} lines ({len(all_log_text)} chars) from RichLog to clipboard.")
        else:
            app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)

    except QueryError:
        app.notify("Log widget not found. Cannot copy.", title="Error", severity="error", timeout=4)
        logger.error("Could not find #app-log-display to copy logs.")
    except AttributeError as ae:
        app.notify(f"Error processing log line: {str(ae)}", title="Error", severity="error", timeout=6)
        logger.error(f"AttributeError while processing RichLog lines: {ae}", exc_info=True)
    except Exception as e:  # General catch-all
        app.notify(f"Error copying logs: {str(e)}", title="Error", severity="error", timeout=6)
        logger.error(f"Failed to copy logs: {e}", exc_info=True)

# --- Button Handler Map ---
APP_LIFECYCLE_BUTTON_HANDLERS = {
    "copy-logs-button": handle_copy_logs_button_pressed,
}

#
# End of app_lifecycle.py
########################################################################################################################
