"""
Evaluation Window V3 - Navigation-based implementation.

This version integrates with the new navigation system for better UX.
"""

from typing import TYPE_CHECKING, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen

from loguru import logger

from .navigation import EvalNavigationScreen, NavigateToEvalScreen
from .screens import QuickTestScreen

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsWindowV3(Container):
    """
    Main evaluation window with navigation-based UI.
    
    This is a container that manages different evaluation screens
    with a navigation-first approach for better UX.
    """
    
    DEFAULT_CSS = """
    EvalsWindowV3 {
        width: 100%;
        height: 100%;
        layout: vertical;
    }
    """
    
    def __init__(self, app_instance: Optional['TldwCli'] = None, **kwargs):
        """Initialize evaluation window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_screen: Optional[Screen] = None
        self.screen_stack: list[Screen] = []
        
        logger.info("Evaluation Window V3 initialized")
    
    def compose(self) -> ComposeResult:
        """Compose with navigation screen as default."""
        # Start with the navigation hub
        self.current_screen = EvalNavigationScreen(self.app_instance)
        yield self.current_screen
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("Evaluation Window V3 mounted")
        
        # Set up message handling
        self.watch(self, "current_screen", self._handle_screen_change)
    
    @on(NavigateToEvalScreen)
    def handle_navigation(self, message: NavigateToEvalScreen) -> None:
        """Handle navigation to different eval screens."""
        screen_id = message.screen_id
        logger.info(f"Navigating to screen: {screen_id}")
        
        # Create the appropriate screen
        new_screen = self._create_screen(screen_id)
        
        if new_screen:
            # Push current screen to stack
            if self.current_screen:
                self.screen_stack.append(self.current_screen)
                self.current_screen.remove()
            
            # Mount new screen
            self.current_screen = new_screen
            self.mount(new_screen)
        else:
            logger.warning(f"Unknown screen ID: {screen_id}")
            if self.app_instance:
                self.app_instance.notify(
                    f"Screen '{screen_id}' not yet implemented",
                    severity="warning"
                )
    
    def _create_screen(self, screen_id: str) -> Optional[Screen]:
        """Create a screen based on ID."""
        screen_map = {
            "eval_home": lambda: EvalNavigationScreen(self.app_instance),
            "quick_test": lambda: QuickTestScreen(self.app_instance),
            # Add more screens as they're implemented:
            # "comparison": lambda: ComparisonScreen(self.app_instance),
            # "batch_eval": lambda: BatchEvalScreen(self.app_instance),
            # "results": lambda: ResultsBrowserScreen(self.app_instance),
            # "tasks": lambda: TaskManagerScreen(self.app_instance),
            # "models": lambda: ModelManagerScreen(self.app_instance),
        }
        
        screen_factory = screen_map.get(screen_id)
        if screen_factory:
            return screen_factory()
        
        return None
    
    def go_back(self) -> None:
        """Navigate back to the previous screen."""
        if self.screen_stack:
            # Remove current screen
            if self.current_screen:
                self.current_screen.remove()
            
            # Pop and mount previous screen
            self.current_screen = self.screen_stack.pop()
            self.mount(self.current_screen)
            
            logger.info(f"Navigated back to: {self.current_screen.__class__.__name__}")
        else:
            logger.info("No screen to go back to")
    
    def _handle_screen_change(self, old_screen: Optional[Screen], new_screen: Optional[Screen]) -> None:
        """Handle screen change events."""
        if old_screen:
            logger.debug(f"Left screen: {old_screen.__class__.__name__}")
        if new_screen:
            logger.debug(f"Entered screen: {new_screen.__class__.__name__}")
    
    def reset_to_home(self) -> None:
        """Reset to the home navigation screen."""
        # Clear stack
        self.screen_stack.clear()
        
        # Remove current screen
        if self.current_screen:
            self.current_screen.remove()
        
        # Create and mount home screen
        self.current_screen = EvalNavigationScreen(self.app_instance)
        self.mount(self.current_screen)
        
        logger.info("Reset to navigation home")