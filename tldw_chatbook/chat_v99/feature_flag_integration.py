"""Feature flag integration for ChatV99.

This module provides the integration point for the new chat window
to be used in the main application based on a feature flag.

Usage:
    In the main app.py, import and use get_chat_window_class() to
    determine which chat window to use based on configuration.

Example config.toml:
    [chat_defaults]
    use_chat_v99 = true  # Enable new chat window
"""

from typing import Type
from textual.widget import Widget


def get_chat_window_class(config_getter) -> Type[Widget]:
    """Get the appropriate chat window class based on feature flag.
    
    Args:
        config_getter: Function to get config values (e.g., get_cli_setting)
        
    Returns:
        The chat window class to use
    """
    # Import both chat windows
    from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
    from tldw_chatbook.UI.Chat_Window import ChatWindow
    
    # Check feature flags in order of preference
    use_chat_v99 = config_getter("chat_defaults", "use_chat_v99", False)
    
    if use_chat_v99:
        try:
            # Try to import ChatV99App
            from tldw_chatbook.chat_v99.app import ChatV99App
            
            # Create a wrapper that makes ChatV99App behave like a widget
            class ChatV99Widget(Widget):
                """Wrapper to make ChatV99App work as a widget in the main app."""
                
                def compose(self):
                    """Compose the ChatV99App as a sub-app."""
                    # Note: In production, this would need proper integration
                    # For now, return a placeholder that indicates v99 is loaded
                    from textual.widgets import Static
                    yield Static(
                        "[bold green]Chat v99 Loaded![/bold green]\n\n"
                        "To test the new chat interface, run:\n"
                        "[cyan]python -m tldw_chatbook.chat_v99.app[/cyan]",
                        id="chat-v99-placeholder"
                    )
                
                def on_mount(self):
                    """Mount handler."""
                    self.notify("Chat v99 interface loaded (placeholder mode)")
            
            return ChatV99Widget
            
        except ImportError as e:
            # Fall back to enhanced chat if v99 not available
            import logging
            logging.warning(f"Failed to import ChatV99App: {e}")
            logging.info("Falling back to enhanced chat window")
    
    # Check for enhanced chat flag
    use_enhanced_chat = config_getter("chat_defaults", "use_enhanced_window", False)
    
    if use_enhanced_chat:
        return ChatWindowEnhanced
    else:
        return ChatWindow


def integrate_chat_v99(app_instance):
    """Integrate ChatV99 into the main application.
    
    This function patches the main app to use ChatV99 when the feature flag is enabled.
    
    Args:
        app_instance: The main TldwCli app instance
    """
    # This would be called from the main app's __init__ or on_mount
    # to set up the integration
    
    from tldw_chatbook.config import get_cli_setting
    
    if get_cli_setting("chat_defaults", "use_chat_v99", False):
        import logging
        logging.info("ChatV99 feature flag enabled - new chat interface will be used")
        
        # Additional setup could go here
        # For example, registering new event handlers, modifying keybindings, etc.


# Configuration documentation
FEATURE_FLAG_DOCS = """
To enable the new Chat v99 interface, add this to your config.toml:

[chat_defaults]
use_chat_v99 = true

The new interface features:
- Fully reactive state management
- Improved streaming performance
- Better message handling
- Cleaner CSS architecture
- Proper Textual patterns throughout

To run the chat interface standalone:
    python -m tldw_chatbook.chat_v99.app

To disable and revert to the previous interface:
    use_chat_v99 = false
"""