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


class NavigationButton(Button):
    """Navigation button that remains pressable when mounted in hidden chrome."""

    def __init__(self, *args, target_route: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_route = target_route

    def press(self):
        if not self.display:
            self.app.post_message(NavigateToScreen(self._target_route))
            return self
        return super().press()


class MainNavigationBar(Container):
    """
    Main navigation bar for the application.
    Replaces the tab-based navigation with screen-based navigation.
    """

    DEFAULT_CSS = """
    MainNavigationBar {
        height: 4;
        min-height: 4;
        width: 100%;
        dock: top;
        background: $background;
        border-bottom: solid $surface-lighten-2;
        overflow-x: auto;
    }

    .main-nav {
        height: 100%;
        width: auto;
        layout: horizontal;
        align: left middle;
        padding: 0;
    }

    .nav-button {
        margin: 0;
        padding: 0;
        min-width: 4;
        background: $surface-darken-1;
        border: solid $surface-lighten-2;
        height: 4;
        min-height: 4;
        content-align: center middle;
    }

    .nav-button:hover {
        background: $surface;
        border: solid $primary-lighten-1;
        text-style: bold;
    }

    .nav-button:focus {
        background: $surface;
        border: solid $primary;
        text-style: bold underline;
        color: $text;
        outline: none;
    }

    .nav-button.is-active {
        background: $primary-darken-1;
        border: solid $primary;
        text-style: bold;
        color: $text;
    }

    .nav-button.is-active:focus {
        background: $primary-darken-1;
        border: solid $primary;
        text-style: bold underline;
        color: $text;
        outline: none;
    }

    .nav-group-separator {
        margin: 0;
        padding: 0 1;
        color: $accent;
        width: auto;
        text-style: bold;
    }

    .nav-overflow-hint {
        width: auto;
        padding: 0;
        height: 4;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def __init__(self, active: str = "chat", active_route: str | None = None, **kwargs):
        """Initialize the navigation bar with destination and route state.

        Args:
            active: Current screen or destination used to highlight the owning
                top-level destination.
            active_route: Canonical active route when the highlighted
                destination owns a subroute. When omitted, `active` is used.
            **kwargs: Additional Textual container keyword arguments.
        """
        super().__init__(**kwargs)
        resolved_active = resolve_shell_route(active)
        self.active_destination_id = resolved_active.destination_id
        self.active_route = resolve_shell_route(active_route or active).canonical_route
        self.active_screen = self.active_destination_id

    def compose(self) -> ComposeResult:
        """Compose the navigation bar from master-shell destination metadata."""
        with Horizontal(classes="main-nav"):
            for destination in SHELL_DESTINATION_ORDER:
                button = NavigationButton(
                    destination.label,
                    id=f"nav-{destination.destination_id}",
                    classes="nav-button ascii-nav-tab",
                    tooltip=destination.tooltip,
                    target_route=destination.primary_route,
                )
                if destination.destination_id == self.active_destination_id:
                    button.add_class("is-active")
                yield button
            overflow_hint = Static("More: Ctrl+P", id="nav-overflow-hint", classes="nav-overflow-hint")
            overflow_hint.tooltip = "Open command palette"
            yield overflow_hint

    @on(Button.Pressed, ".nav-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        destination_id = button_id.replace("nav-", "")
        destination = get_shell_destination(destination_id)
        screen_name = destination.primary_route
        
        # A destination-owned subroute may highlight the same top-level destination;
        # clicking the destination should still return to its primary route.
        if (
            destination.destination_id == self.active_destination_id
            and screen_name == self.active_route
        ):
            return
        
        # Update active state
        for button in self.query(".nav-button"):
            button.remove_class("is-active")
        event.button.add_class("is-active")
        self.active_destination_id = destination.destination_id
        self.active_route = screen_name
        self.active_screen = self.active_destination_id
        
        # Post navigation message to app
        self.post_message(NavigateToScreen(screen_name))
        
        logger.info(f"Navigation requested to screen: {screen_name}")
