"""Loading indicators for CCP async operations following Textual best practices."""

from typing import Optional, Any, Callable
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime

from textual.widgets import LoadingIndicator, Static
from textual.containers import Center
from textual.reactive import reactive
from textual import work
from loguru import logger

logger = logger.bind(module="CCPLoadingIndicators")


class CCPLoadingWidget(Static):
    """
    A loading widget that follows Textual's reactive patterns.
    Can be used as an overlay or inline loading indicator.
    """
    
    DEFAULT_CSS = """
    CCPLoadingWidget {
        height: auto;
        width: auto;
        align: center middle;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        layer: overlay;
        display: none;
    }
    
    CCPLoadingWidget.visible {
        display: block;
    }
    
    CCPLoadingWidget.inline {
        layer: default;
        border: none;
        padding: 0;
    }
    
    CCPLoadingWidget .loading-text {
        text-align: center;
        margin-left: 1;
    }
    """
    
    loading_text: reactive[str] = reactive("Loading...")
    is_loading: reactive[bool] = reactive(False)
    
    def __init__(self, text: str = "Loading...", inline: bool = False, **kwargs):
        """
        Initialize the loading widget.
        
        Args:
            text: The loading message to display
            inline: If True, displays inline rather than as overlay
        """
        super().__init__(**kwargs)
        self.loading_text = text
        self._inline = inline
        if inline:
            self.add_class("inline")
    
    def compose(self):
        """Compose the loading widget with a LoadingIndicator and text."""
        with Center():
            yield LoadingIndicator()
            yield Static(self.loading_text, classes="loading-text")
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """React to loading state changes."""
        if is_loading:
            self.add_class("visible")
        else:
            self.remove_class("visible")
    
    def start_loading(self, text: Optional[str] = None) -> None:
        """Start showing the loading indicator."""
        if text:
            self.loading_text = text
            self.query_one(".loading-text", Static).update(text)
        self.is_loading = True
    
    def stop_loading(self) -> None:
        """Stop showing the loading indicator."""
        self.is_loading = False
    
    def update_text(self, text: str) -> None:
        """Update the loading text while loading."""
        self.loading_text = text
        self.query_one(".loading-text", Static).update(text)


class LoadingManager:
    """
    Manages loading states for the CCP window.
    Follows Textual's notification patterns.
    """
    
    def __init__(self, window):
        """
        Initialize the loading manager.
        
        Args:
            window: Reference to the CCP window
        """
        self.window = window
        self.active_operations = {}
        self._loading_widget: Optional[CCPLoadingWidget] = None
    
    async def setup(self):
        """Setup the loading overlay in the window."""
        try:
            # Create and mount the loading widget if not exists
            if not self._loading_widget:
                self._loading_widget = CCPLoadingWidget()
                await self.window.mount(self._loading_widget)
                logger.debug("Loading widget mounted to CCP window")
        except Exception as e:
            logger.error(f"Failed to setup loading widget: {e}")
    
    @asynccontextmanager
    async def loading(self, text: str = "Loading...", operation_id: Optional[str] = None):
        """
        Context manager for loading operations.
        
        Usage:
            async with self.loading_manager.loading("Fetching data..."):
                await some_async_operation()
        """
        if not operation_id:
            operation_id = f"op_{datetime.now().timestamp()}"
        
        try:
            # Start loading
            await self.start_loading(text, operation_id)
            yield
        finally:
            # Stop loading
            await self.stop_loading(operation_id)
    
    async def start_loading(self, text: str = "Loading...", operation_id: str = None) -> str:
        """
        Start a loading operation.
        
        Args:
            text: The loading message
            operation_id: Unique ID for this operation
            
        Returns:
            The operation ID
        """
        if not operation_id:
            operation_id = f"op_{datetime.now().timestamp()}"
        
        self.active_operations[operation_id] = text
        
        # Update loading widget
        if self._loading_widget:
            self._loading_widget.start_loading(text)
        
        # Also use Textual's notify for important operations
        if hasattr(self.window, 'notify'):
            self.window.notify(f"⏳ {text}", timeout=2)
        
        logger.debug(f"Started loading operation: {operation_id} - {text}")
        return operation_id
    
    async def stop_loading(self, operation_id: str) -> None:
        """
        Stop a loading operation.
        
        Args:
            operation_id: The ID of the operation to stop
        """
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        
        # If no more operations, hide loading widget
        if not self.active_operations and self._loading_widget:
            self._loading_widget.stop_loading()
        elif self.active_operations and self._loading_widget:
            # Update with the last operation's text
            last_text = list(self.active_operations.values())[-1]
            self._loading_widget.update_text(last_text)
        
        logger.debug(f"Stopped loading operation: {operation_id}")
    
    async def update_loading_text(self, operation_id: str, text: str) -> None:
        """
        Update the text for an active loading operation.
        
        Args:
            operation_id: The operation ID
            text: New loading text
        """
        if operation_id in self.active_operations:
            self.active_operations[operation_id] = text
            if self._loading_widget:
                self._loading_widget.update_text(text)


def with_loading(loading_text: str = "Processing...", success_text: str = "Complete!", 
                 error_text: str = "Operation failed"):
    """
    Decorator to automatically show loading indicators for async operations.
    
    Args:
        loading_text: Text to show while loading
        success_text: Text to show on success
        error_text: Text to show on error
    
    Usage:
        @with_loading("Saving character...", "Character saved!", "Failed to save character")
        async def save_character(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get loading manager
            loading_manager = getattr(self, 'loading_manager', None)
            if not loading_manager and hasattr(self, 'window'):
                loading_manager = getattr(self.window, 'loading_manager', None)
            
            if not loading_manager:
                # Fallback: just run the function without loading indicator
                return await func(self, *args, **kwargs)
            
            operation_id = f"{func.__name__}_{datetime.now().timestamp()}"
            
            try:
                # Start loading
                await loading_manager.start_loading(loading_text, operation_id)
                
                # Run the actual function
                result = await func(self, *args, **kwargs)
                
                # Show success notification
                if hasattr(self.window, 'notify'):
                    self.window.notify(f"✅ {success_text}", severity="information", timeout=2)
                
                return result
                
            except Exception as e:
                # Show error notification
                if hasattr(self.window, 'notify'):
                    self.window.notify(f"❌ {error_text}: {str(e)}", severity="error", timeout=4)
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
                
            finally:
                # Stop loading
                await loading_manager.stop_loading(operation_id)
        
        return wrapper
    return decorator


def with_progress(total_steps: int = None):
    """
    Decorator for operations with progress tracking.
    Updates loading text with progress percentage.
    
    Args:
        total_steps: Total number of steps (if known)
    
    Usage:
        @with_progress(total_steps=5)
        async def import_multiple_files(self, files):
            for i, file in enumerate(files):
                await self.update_progress(i + 1, len(files), f"Importing {file.name}")
                await process_file(file)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Add progress tracking method to self temporarily
            operation_id = f"{func.__name__}_{datetime.now().timestamp()}"
            loading_manager = getattr(self, 'loading_manager', None)
            if not loading_manager and hasattr(self, 'window'):
                loading_manager = getattr(self.window, 'loading_manager', None)
            
            async def update_progress(current: int, total: int = None, text: str = ""):
                nonlocal total_steps
                if total:
                    total_steps = total
                if total_steps:
                    percentage = int((current / total_steps) * 100)
                    progress_text = f"{text} ({percentage}%)" if text else f"Processing... ({percentage}%)"
                else:
                    progress_text = f"{text} ({current})" if text else f"Processing... ({current})"
                
                if loading_manager:
                    await loading_manager.update_loading_text(operation_id, progress_text)
            
            # Temporarily add the method to self
            original_update_progress = getattr(self, 'update_progress', None)
            self.update_progress = update_progress
            
            try:
                if loading_manager:
                    await loading_manager.start_loading("Starting...", operation_id)
                
                result = await func(self, *args, **kwargs)
                
                if hasattr(self.window, 'notify'):
                    self.window.notify("✅ Operation completed", severity="information", timeout=2)
                
                return result
                
            finally:
                # Restore original method or remove it
                if original_update_progress:
                    self.update_progress = original_update_progress
                else:
                    delattr(self, 'update_progress')
                
                if loading_manager:
                    await loading_manager.stop_loading(operation_id)
        
        return wrapper
    return decorator


class InlineLoadingIndicator(Static):
    """
    A simpler inline loading indicator for individual widgets.
    Follows Textual's patterns for inline feedback.
    """
    
    DEFAULT_CSS = """
    InlineLoadingIndicator {
        height: 1;
        width: auto;
        color: $text-muted;
        display: none;
    }
    
    InlineLoadingIndicator.active {
        display: block;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._loading = False
        self._dots = 0
    
    @work(exclusive=True)
    async def animate(self):
        """Animate the loading dots."""
        import asyncio
        while self._loading:
            self._dots = (self._dots + 1) % 4
            self.update("Loading" + "." * self._dots)
            await asyncio.sleep(0.5)
    
    def start(self):
        """Start the loading animation."""
        self._loading = True
        self.add_class("active")
        self.animate()
    
    def stop(self):
        """Stop the loading animation."""
        self._loading = False
        self.remove_class("active")
        self.update("")