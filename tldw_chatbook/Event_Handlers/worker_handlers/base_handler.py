"""
Base Worker Handler - Abstract base class for worker state change handlers.

This module provides the foundation for handling different types of worker
state changes in a modular, extensible way.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List
from textual.worker import Worker, WorkerState
from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class BaseWorkerHandler(ABC):
    """Abstract base class for worker state change handlers."""
    
    def __init__(self, app: 'TldwCli'):
        """
        Initialize the worker handler.
        
        Args:
            app: The TldwCli application instance
        """
        self.app = app
        self.logger = logger
    
    @abstractmethod
    def can_handle(self, worker_name: str, worker_group: Optional[str] = None) -> bool:
        """
        Check if this handler can process the given worker.
        
        Args:
            worker_name: The name attribute of the worker
            worker_group: The group attribute of the worker
            
        Returns:
            True if this handler can process the worker, False otherwise
        """
        pass
    
    @abstractmethod
    async def handle(self, event: Worker.StateChanged) -> None:
        """
        Handle the worker state change event.
        
        Args:
            event: The worker state changed event
        """
        pass
    
    def get_worker_info(self, event: Worker.StateChanged) -> dict:
        """
        Extract common worker information from the event.
        
        Args:
            event: The worker state changed event
            
        Returns:
            Dictionary containing worker information
        """
        return {
            'name': event.worker.name,
            'group': event.worker.group,
            'state': event.state,
            'description': event.worker.description,
            'is_finished': event.worker.is_finished,
            'is_cancelled': event.worker.is_cancelled,
            'is_running': event.worker.is_running,
        }
    
    def log_state_change(self, worker_info: dict, prefix: str = "") -> None:
        """
        Log a worker state change with consistent formatting.
        
        Args:
            worker_info: Dictionary containing worker information
            prefix: Optional prefix for the log message
        """
        msg = f"{prefix}Worker '{worker_info['name']}' (Group: {worker_info['group']}) -> State: {worker_info['state']}"
        if worker_info.get('description'):
            msg += f" | Desc: {worker_info['description']}"
        self.logger.debug(msg)
    
    def log_error(self, worker_name: str, error: Exception, context: str = "") -> None:
        """
        Log an error with consistent formatting.
        
        Args:
            worker_name: The name of the worker
            error: The exception that occurred
            context: Optional context for the error
        """
        msg = f"Error in worker '{worker_name}'"
        if context:
            msg += f" ({context})"
        msg += f": {error}"
        self.logger.error(msg, exc_info=True)
    
    async def update_button_state(self, button_id: str, disabled: bool) -> None:
        """
        Safely update a button's disabled state.
        
        Args:
            button_id: The ID of the button to update
            disabled: Whether to disable (True) or enable (False) the button
        """
        try:
            from textual.widgets import Button
            button = self.app.query_one(f"#{button_id}", Button)
            button.disabled = disabled
        except Exception as e:
            self.logger.warning(f"Could not update button '{button_id}': {e}")
    
    async def update_ui_element(self, element_id: str, update_func: callable) -> None:
        """
        Safely update a UI element with error handling.
        
        Args:
            element_id: The ID of the element to update
            update_func: Function to call with the element as argument
        """
        try:
            element = self.app.query_one(f"#{element_id}")
            update_func(element)
        except Exception as e:
            self.logger.warning(f"Could not update element '{element_id}': {e}")


class WorkerHandlerRegistry:
    """Registry for managing worker handlers."""
    
    def __init__(self, app: 'TldwCli'):
        """
        Initialize the handler registry.
        
        Args:
            app: The TldwCli application instance
        """
        self.app = app
        self.handlers: List[BaseWorkerHandler] = []
        self.logger = logger
    
    def register(self, handler: BaseWorkerHandler) -> None:
        """
        Register a worker handler.
        
        Args:
            handler: The handler to register
        """
        self.handlers.append(handler)
        self.logger.debug(f"Registered worker handler: {handler.__class__.__name__}")
    
    async def handle_event(self, event: Worker.StateChanged) -> bool:
        """
        Route the event to the appropriate handler.
        
        Args:
            event: The worker state changed event
            
        Returns:
            True if the event was handled, False otherwise
        """
        worker_name = event.worker.name
        worker_group = event.worker.group
        
        for handler in self.handlers:
            if handler.can_handle(worker_name, worker_group):
                try:
                    await handler.handle(event)
                    return True
                except Exception as e:
                    self.logger.error(
                        f"Handler {handler.__class__.__name__} failed for worker '{worker_name}': {e}",
                        exc_info=True
                    )
                    # Continue to next handler if one fails
        
        return False