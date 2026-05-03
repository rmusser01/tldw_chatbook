"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.message import Message
from textual import on

from .shell_destinations import SHELL_DESTINATION_ORDER, get_shell_destination, resolve_shell_route

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

    .nav-button.is-active {
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
        self.active_screen = resolve_shell_route(active).destination_id

    def compose(self) -> ComposeResult:
        """Compose the navigation bar from master-shell destination metadata."""
        with Horizontal(classes="main-nav"):
            for index, destination in enumerate(SHELL_DESTINATION_ORDER):
                if index > 0:
                    yield Static("·", classes="nav-separator")

                button = Button(
                    destination.label,
                    id=f"nav-{destination.destination_id}",
                    classes="nav-button",
                    tooltip=destination.tooltip,
                )
                if destination.destination_id == self.active_screen:
                    button.add_class("is-active")
                yield button
    
    @on(Button.Pressed, ".nav-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        destination_id = button_id.replace("nav-", "")
        destination = get_shell_destination(destination_id)
        screen_name = destination.primary_route
        
        # Don't navigate if already on this screen
        if destination.destination_id == self.active_screen:
            return
        
        # Update active state
        for button in self.query(".nav-button"):
            button.remove_class("is-active")
        event.button.add_class("is-active")
        self.active_screen = destination.destination_id
        
        # Post navigation message to app
        self.post_message(NavigateToScreen(screen_name))
        
        logger.info(f"Navigation requested to screen: {screen_name}")
