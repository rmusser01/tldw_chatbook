"""
Navigation manager for screen-based navigation.
"""

from typing import Optional, TYPE_CHECKING
from textual.screen import Screen
from loguru import logger

from .screen_registry import ScreenRegistry
from ..state.navigation_state import NavigationState

if TYPE_CHECKING:
    from textual.app import App


class NavigationManager:
    """
    Manages screen navigation for the application.
    Handles screen switching, history, and state management.
    """
    
    def __init__(self, app: 'App', state: NavigationState):
        self.app = app
        self.state = state
        self.registry = ScreenRegistry()
        self._screen_cache = {}
        
    async def navigate_to(self, screen_name: str) -> bool:
        """
        Navigate to a screen by name.
        
        Args:
            screen_name: Name of the screen to navigate to
            
        Returns:
            True if navigation was successful, False otherwise
        """
        # Get screen class from registry
        screen_class = self.registry.get_screen_class(screen_name)
        if not screen_class:
            logger.error(f"Unknown screen: {screen_name}")
            return False
        
        # Check if we're already on this screen
        if self.state.current_screen == screen_name:
            logger.debug(f"Already on screen: {screen_name}")
            return True
        
        try:
            # Create screen instance (could implement caching here)
            screen = self._get_or_create_screen(screen_name, screen_class)
            
            # Switch to the new screen
            await self.app.switch_screen(screen)
            
            # Update state
            self.state.navigate_to(screen_name)
            
            logger.info(f"Navigated to screen: {screen_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate to {screen_name}: {e}")
            return False
    
    async def go_back(self) -> bool:
        """
        Navigate to the previous screen.
        
        Returns:
            True if navigation was successful, False otherwise
        """
        previous = self.state.go_back()
        if previous:
            return await self.navigate_to(previous)
        
        logger.debug("No previous screen to go back to")
        return False
    
    async def go_home(self) -> bool:
        """
        Navigate to the home screen (chat).
        
        Returns:
            True if navigation was successful, False otherwise
        """
        return await self.navigate_to("chat")
    
    def _get_or_create_screen(self, name: str, screen_class: type) -> Screen:
        """
        Get a screen from cache or create a new one.
        
        Args:
            name: Screen name
            screen_class: Screen class to instantiate
            
        Returns:
            Screen instance
        """
        # For now, always create new screens
        # Could implement caching for performance
        return screen_class(self.app)
    
    def clear_cache(self) -> None:
        """Clear the screen cache."""
        self._screen_cache.clear()
        logger.debug("Screen cache cleared")
    
    def get_current_screen(self) -> str:
        """Get the name of the current screen."""
        return self.state.current_screen
    
    def get_history(self) -> list:
        """Get navigation history."""
        return self.state.history.copy()
    
    def can_go_back(self) -> bool:
        """Check if we can navigate back."""
        return self.state.previous_screen is not None
    
    def register_screen(self, name: str, screen_class: type) -> None:
        """
        Register a new screen with the navigation system.
        
        Args:
            name: Screen name
            screen_class: Screen class
        """
        self.registry.register_screen(name, screen_class)
    
    def register_alias(self, alias: str, screen_name: str) -> None:
        """
        Register an alias for a screen.
        
        Args:
            alias: Alias name
            screen_name: Target screen name
        """
        self.registry.register_alias(alias, screen_name)