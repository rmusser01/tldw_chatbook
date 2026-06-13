"""Base screen class for all application screens."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Footer

from .main_navigation import MainNavigationBar

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class BaseAppScreen(Screen):
    """
    Base screen class for all application screens.
    Provides common functionality like navigation bar and state management.
    """
    
    DEFAULT_CSS = """
    BaseAppScreen {
        background: $background;
    }
    
    #screen-content {
        width: 100%;
        height: 1fr;
        min-height: 0;
        padding-top: 0;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', screen_name: str, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.screen_name = screen_name
        self.state_data: Dict[str, Any] = {}
        
        logger.debug(f"Initializing {self.__class__.__name__} screen: {screen_name}")
    
    def compose(self) -> ComposeResult:
        """Compose the screen with navigation bar and content."""
        # Navigation bar at the top
        yield MainNavigationBar(active=self.screen_name, active_route=self.screen_name)
        
        # Content area below navigation
        with Container(id="screen-content"):
            yield from self.compose_content()

        yield Footer()
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide screen-specific content."""
        yield Container()  # Default empty container
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the screen."""
        # Override in subclasses to save specific state
        return self.state_data
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state."""
        # Override in subclasses to restore specific state
        self.state_data = state
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        logger.info(f"Screen {self.screen_name} mounted")
    
    def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        logger.info(f"Screen {self.screen_name} unmounted")
