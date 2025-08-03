# Web_Server package initialization
"""
Web server functionality for tldw_chatbook using textual-serve.

This module provides the ability to run the Textual TUI application 
in a web browser, making it accessible without requiring terminal access.
"""

from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Check if web server dependencies are available
WEB_SERVER_AVAILABLE = DEPENDENCIES_AVAILABLE.get('web', False)

__all__ = ['WEB_SERVER_AVAILABLE']