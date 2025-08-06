"""
Debug Helpers for Textual UI Development
=========================================

This module contains useful debugging utilities for troubleshooting Textual UI issues.
"""

from textual.widgets import Static
from textual.containers import Container
from textual.app import ComposeResult


def add_debug_test_message(container_or_compose: bool = False) -> Static:
    """
    Creates a highly visible test message widget for debugging UI rendering issues.
    
    This is useful when a window or container appears blank and you need to verify
    if the container is actually rendering but just has no visible content.
    
    Args:
        container_or_compose: If False, returns a Static widget to mount.
                            If True, returns code snippet for compose method.
    
    Returns:
        Static widget with bright test message or code snippet string
        
    Example usage in compose():
        ```python
        def compose(self) -> ComposeResult:
            # ... other widgets ...
            
            # Add this to test if window is rendering:
            yield Static("🚨 TEST: If you see this, the window is rendering! 🚨", 
                        classes="debug-test-message")
        ```
        
    Example usage with mount():
        ```python
        from tldw_chatbook.Utils.debug_helpers import add_debug_test_message
        
        def on_mount(self):
            # Add test message to see if container is visible
            container = self.query_one("#my-container")
            container.mount(add_debug_test_message())
        ```
        
    Required CSS (add to your widget's DEFAULT_CSS):
        ```css
        .debug-test-message {
            color: red;
            background: yellow;
            text-align: center;
            padding: 2;
            text-style: bold;
            border: thick red;
        }
        ```
    """
    if container_or_compose:
        return '''
# DEBUG: Add this to compose() to test if window is rendering:
yield Static("🚨 TEST: If you see this, the window is rendering! 🚨", 
            classes="debug-test-message")

# And add this CSS to DEFAULT_CSS:
.debug-test-message {
    color: red;
    background: yellow;
    text-align: center;
    padding: 2;
    text-style: bold;
    border: thick red;
}
'''
    
    return Static("🚨 TEST: If you see this, the container is rendering! 🚨", 
                  classes="debug-test-message")


def debug_window_visibility_snippet():
    """
    Returns a code snippet for debugging window visibility issues.
    
    This includes logging statements and test content to diagnose why
    a window might not be showing.
    """
    return '''
# Add to your window's compose() method:
def compose(self) -> ComposeResult:
    from loguru import logger
    logger.debug(f"{self.__class__.__name__}.compose() called")
    logger.debug(f"{self.__class__.__name__} display property: {self.display}")
    logger.debug(f"{self.__class__.__name__} styles.display: {self.styles.display}")
    
    # Your normal compose content here...
    
    # Add test message to verify rendering:
    yield Static("🚨 TEST: Window is rendering! 🚨", classes="debug-test-message")

# Add to your window's on_mount() method:
async def on_mount(self) -> None:
    from loguru import logger
    logger.debug(f"{self.__class__.__name__}.on_mount() called, display={self.display}")
    # Your normal on_mount content...

# Add to your window's DEFAULT_CSS:
.debug-test-message {
    color: red;
    background: yellow;
    text-align: center;
    padding: 2;
    text-style: bold;
    border: thick red;
}
'''


def check_widget_hierarchy(widget, indent=0):
    """
    Recursively logs the widget hierarchy to help debug layout issues.
    
    Args:
        widget: The widget to start from
        indent: Current indentation level (internal use)
    """
    from loguru import logger
    
    indent_str = "  " * indent
    display_info = f"display={widget.display}, visible={widget.visible if hasattr(widget, 'visible') else 'N/A'}"
    logger.debug(f"{indent_str}{widget.__class__.__name__}(id={widget.id}) - {display_info}")
    
    if hasattr(widget, 'children'):
        for child in widget.children:
            check_widget_hierarchy(child, indent + 1)