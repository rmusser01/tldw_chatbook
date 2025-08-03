"""
Base Tab Initializer - Abstract base class for tab initialization logic.

This module provides the foundation for handling tab-specific initialization
when tabs are shown or hidden in the application.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class BaseTabInitializer(ABC):
    """Abstract base class for tab initializers."""
    
    def __init__(self, app: 'TldwCli'):
        """
        Initialize the tab initializer.
        
        Args:
            app: The TldwCli application instance
        """
        self.app = app
        self.logger = logger
    
    @abstractmethod
    def get_tab_id(self) -> str:
        """
        Return the tab ID this initializer handles.
        
        Returns:
            The tab ID constant (e.g., TAB_CHAT, TAB_NOTES)
        """
        pass
    
    @abstractmethod
    async def on_tab_shown(self) -> None:
        """
        Called when the tab is shown.
        
        This method should handle all initialization logic needed when
        the tab becomes active.
        """
        pass
    
    async def on_tab_hidden(self) -> None:
        """
        Called when the tab is hidden.
        
        This method should handle any cleanup needed when the tab
        becomes inactive. Default implementation does nothing.
        """
        pass
    
    def is_window_ready(self, window_id: str) -> bool:
        """
        Check if a window is ready (not a placeholder).
        
        Args:
            window_id: The ID of the window to check
            
        Returns:
            True if the window is ready, False otherwise
        """
        try:
            from tldw_chatbook.app import PlaceholderWindow
            window = self.app.query_one(f"#{window_id}")
            return not isinstance(window, PlaceholderWindow)
        except Exception:
            return False
    
    def schedule_initialization(self, func: callable, delay: float = 0.1) -> None:
        """
        Schedule an initialization function with a delay.
        
        Args:
            func: The function to call
            delay: Delay in seconds before calling the function
        """
        self.app.set_timer(delay, func)
    
    def call_async_handler(self, handler: callable, *args, **kwargs) -> None:
        """
        Safely call an async handler function.
        
        Args:
            handler: The async handler function
            *args: Positional arguments for the handler
            **kwargs: Keyword arguments for the handler
        """
        self.app.call_after_refresh(handler, *args, **kwargs)
    
    def log_initialization(self, message: str) -> None:
        """
        Log a tab initialization message.
        
        Args:
            message: The message to log
        """
        self.logger.debug(f"[{self.get_tab_id()}] {message}")


class TabInitializerRegistry:
    """Registry for managing tab initializers."""
    
    def __init__(self, app: 'TldwCli'):
        """
        Initialize the tab initializer registry.
        
        Args:
            app: The TldwCli application instance
        """
        self.app = app
        self.initializers: Dict[str, BaseTabInitializer] = {}
        self.logger = logger
    
    def register(self, initializer: BaseTabInitializer) -> None:
        """
        Register a tab initializer.
        
        Args:
            initializer: The initializer to register
        """
        tab_id = initializer.get_tab_id()
        self.initializers[tab_id] = initializer
        self.logger.debug(f"Registered tab initializer for: {tab_id}")
    
    async def handle_tab_change(self, old_tab: Optional[str], new_tab: str) -> None:
        """
        Handle a tab change event.
        
        Args:
            old_tab: The previously active tab ID (may be None)
            new_tab: The newly active tab ID
        """
        # Handle old tab cleanup
        if old_tab and old_tab in self.initializers:
            try:
                await self.initializers[old_tab].on_tab_hidden()
            except Exception as e:
                self.logger.error(
                    f"Error handling tab hidden for {old_tab}: {e}",
                    exc_info=True
                )
        
        # Handle new tab initialization
        if new_tab in self.initializers:
            try:
                await self.initializers[new_tab].on_tab_shown()
            except Exception as e:
                self.logger.error(
                    f"Error handling tab shown for {new_tab}: {e}",
                    exc_info=True
                )
        else:
            self.logger.debug(f"No initializer registered for tab: {new_tab}")