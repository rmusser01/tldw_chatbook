"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING, Optional
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.message import Message
from textual import on

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class NavigateToScreen(Message):
    """Message to request navigation to a specific screen."""
    
    def __init__(self, screen_name: str):
        super().__init__()
        self.screen_name = screen_name


class MainNavigationBar(Container):
    """
    Main navigation bar for the application.
    Replaces the tab-based navigation with screen-based navigation.
    """
    
    DEFAULT_CSS = """
    MainNavigationBar {
        height: 3;
        width: 100%;
        dock: top;
        background: $panel;
        border-bottom: solid $primary;
        overflow-x: auto;
    }
    
    .main-nav {
        height: 100%;
        width: auto;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }
    
    .nav-button {
        margin: 0;
        padding: 0 1;
        min-width: 6;
        background: transparent;
        border: none;
        height: 3;
    }
    
    .nav-button:hover {
        background: $primary-lighten-2;
        text-style: bold;
    }
    
    .nav-button.active {
        background: $primary;
        text-style: bold;
        color: $text;
    }
    
    .nav-separator {
        margin: 0;
        padding: 0 0;
        color: $text-muted;
        width: 1;
    }
    """
    
    def __init__(self, active: str = "chat", **kwargs):
        super().__init__(**kwargs)
        self.active_screen = active
        
        # Define the navigation items
        self.nav_items = [
            ("chat", "Chat"),
            ("ccp", "Conv/Char"),
            ("notes", "Notes"),
            ("media", "Media"),
            ("search", "Search"),
            ("ingest", "Ingest"),
            ("tools_settings", "Settings"),
            ("llm", "LLM"),
            ("customize", "Customize"),
            ("logs", "Logs"),
            ("coding", "Coding"),
            ("stats", "Stats"),
            ("evals", "Evals"),
        ]
    
    def compose(self) -> ComposeResult:
        """Compose the navigation bar."""
        with Horizontal(classes="main-nav"):
            for i, (screen_id, label) in enumerate(self.nav_items):
                # Add separator between items (except before first)
                if i > 0:
                    yield Static("|", classes="nav-separator")
                
                # Create button with active class if needed
                button = Button(
                    label,
                    id=f"nav-{screen_id}",
                    classes="nav-button"
                )
                if screen_id == self.active_screen:
                    button.add_class("active")
                yield button
    
    @on(Button.Pressed, ".nav-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Extract screen name from button ID (nav-chat -> chat)
        screen_name = button_id.replace("nav-", "")
        
        # Don't navigate if already on this screen
        if screen_name == self.active_screen:
            return
        
        # Update active state
        for button in self.query(".nav-button"):
            button.remove_class("active")
        event.button.add_class("active")
        self.active_screen = screen_name
        
        # Post navigation message to app
        self.post_message(NavigateToScreen(screen_name))
        
        logger.info(f"Navigation requested to screen: {screen_name}")