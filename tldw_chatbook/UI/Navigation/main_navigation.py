"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING, Optional
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.message import Message
from textual import on

from tldw_chatbook.Constants import TAB_RESEARCH, TAB_WRITING

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

    # Navigation items organized by visual group
    NAV_GROUPS = [
        ("Workspace", [("chat", "Chat"), ("coding", "Coding"), ("chatbooks", "Chatbooks")]),
        ("Content", [("notes", "Notes"), ("media", "Media"), ("ingest", "Ingest"), ("search", "Search"), ("subscriptions", "Subscriptions")]),
        ("Characters", [("ccp", "Conv/Char"), ("study", "Study")]),
        ("AI Config", [("llm", "LLM"), ("stts", "S/TT/S"), ("evals", "Evals")]),
        ("System", [("tools_settings", "Settings"), ("customize", "Customize"), ("logs", "Logs"), ("stats", "Stats")]),
    ]

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
        padding: 0;
        color: $text-muted;
        width: auto;
    }

    .nav-group-separator {
        margin: 0;
        padding: 0 1;
        color: $accent;
        width: auto;
        text-style: bold;
    }
    """

    def __init__(self, active: str = "chat", **kwargs):
        super().__init__(**kwargs)
        self.active_screen = active

    def compose(self) -> ComposeResult:
        """Compose the navigation bar with visual grouping."""
        with Horizontal(classes="main-nav"):
            first_group = True
            for group_name, items in self.NAV_GROUPS:
                if not first_group:
                    yield Static("┃", classes="nav-group-separator")
                first_group = False

                for i, (screen_id, label) in enumerate(items):
                    if i > 0:
                        yield Static("·", classes="nav-separator")

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
