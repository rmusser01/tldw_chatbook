# Web_Server package initialization
"""
Web server functionality for tldw_chatbook using textual-serve.

This module provides the ability to run the Textual TUI application 
in a web browser, making it accessible without requiring terminal access.
"""

from ..Utils.optional_deps import check_web_server_deps

# Check if web server dependencies are available dynamically
def is_web_server_available():
    """Check if web server dependencies are available."""
    return check_web_server_deps()

# For backward compatibility
WEB_SERVER_AVAILABLE = is_web_server_available()

__all__ = ['WEB_SERVER_AVAILABLE', 'is_web_server_available']