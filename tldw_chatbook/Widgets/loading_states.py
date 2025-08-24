"""
Loading state widgets for improved UX during async operations.

This module provides various loading indicators and states:
- Inline loading indicators
- Skeleton screens for content placeholders
- Progress bars for long operations
- Error/retry states for failed operations
"""

from typing import Optional, Callable, Any
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, LoadingIndicator, ProgressBar
from textual.reactive import reactive
from textual.message import Message
from textual import work
from datetime import datetime
from loguru import logger


class LoadingState(Container):
    """A container that shows loading state while content is being fetched."""
    
    # Reactive properties
    is_loading = reactive(True, layout=False)
    has_error = reactive(False, layout=False)
    error_message = reactive("", layout=False)
    progress = reactive(0.0, layout=False)
    
    def __init__(
        self,
        loader: Optional[Callable] = None,
        placeholder_text: str = "Loading...",
        show_progress: bool = False,
        auto_start: bool = True,
        **kwargs
    ):
        """Initialize the loading state widget.
        
        Args:
            loader: Async function to load content
            placeholder_text: Text to show while loading
            show_progress: Whether to show progress bar
            auto_start: Whether to start loading automatically
        """
        super().__init__(**kwargs)
        self.loader = loader
        self.placeholder_text = placeholder_text
        self.show_progress = show_progress
        self.auto_start = auto_start
        self.content = None
        self.start_time = None
        
    def compose(self) -> ComposeResult:
        """Compose the loading state UI."""
        with Container(classes="loading-state-container"):
            # Loading view
            with Container(classes="loading-view", id="loading-view"):
                yield LoadingIndicator()
                yield Static(self.placeholder_text, classes="loading-text")
                if self.show_progress:
                    yield ProgressBar(total=100, id="loading-progress")
            
            # Error view
            with Container(classes="error-view hidden", id="error-view"):
                yield Static("⚠️ Error", classes="error-icon")
                yield Static("", id="error-message", classes="error-message")
                with Horizontal(classes="error-actions"):
                    yield Button("Retry", id="retry-button", variant="primary")
                    yield Button("Cancel", id="cancel-button", variant="default")
            
            # Content view (initially hidden)
            with Container(classes="content-view hidden", id="content-view"):
                pass
    
    async def on_mount(self) -> None:
        """Handle mount event."""
        if self.auto_start and self.loader:
            self.start_loading()
    
    @work(exclusive=True)
    async def start_loading(self) -> None:
        """Start the loading process."""
        self.is_loading = True
        self.has_error = False
        self.progress = 0.0
        self.start_time = datetime.now()
        
        # Show loading view
        self._show_loading_view()
        
        try:
            if self.loader:
                # Support progress callback
                async def progress_callback(value: float):
                    self.progress = value
                    if self.show_progress:
                        progress_bar = self.query_one("#loading-progress", ProgressBar)
                        progress_bar.update(progress=int(value))
                
                # Call loader with progress callback if it accepts it
                import inspect
                sig = inspect.signature(self.loader)
                if 'progress_callback' in sig.parameters:
                    self.content = await self.loader(progress_callback=progress_callback)
                else:
                    self.content = await self.loader()
                
                # Success - show content
                self._show_content_view()
                
                # Post success message
                self.post_message(LoadingComplete(self.content))
                
        except Exception as e:
            logger.error(f"Loading failed: {e}")
            self.has_error = True
            self.error_message = str(e)
            self._show_error_view()
            
            # Post error message
            self.post_message(LoadingFailed(str(e)))
            
        finally:
            self.is_loading = False
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.debug(f"Loading completed in {elapsed:.2f}s")
    
    def _show_loading_view(self) -> None:
        """Show the loading view."""
        self.query_one("#loading-view").remove_class("hidden")
        self.query_one("#error-view").add_class("hidden")
        self.query_one("#content-view").add_class("hidden")
    
    def _show_error_view(self) -> None:
        """Show the error view."""
        self.query_one("#loading-view").add_class("hidden")
        self.query_one("#error-view").remove_class("hidden")
        self.query_one("#content-view").add_class("hidden")
        
        # Update error message
        error_msg = self.query_one("#error-message", Static)
        error_msg.update(self.error_message)
    
    def _show_content_view(self) -> None:
        """Show the content view."""
        self.query_one("#loading-view").add_class("hidden")
        self.query_one("#error-view").add_class("hidden")
        
        content_view = self.query_one("#content-view")
        content_view.remove_class("hidden")
        
        # Add content if it's a widget
        if self.content and hasattr(self.content, 'compose'):
            content_view.mount(self.content)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "retry-button":
            self.start_loading()
        elif event.button.id == "cancel-button":
            self.post_message(LoadingCancelled())
    
    def watch_progress(self, progress: float) -> None:
        """Watch progress changes."""
        if self.show_progress and self.is_loading:
            try:
                progress_bar = self.query_one("#loading-progress", ProgressBar)
                progress_bar.update(progress=int(progress))
            except:
                pass


class SkeletonLoader(Container):
    """A skeleton screen placeholder for content that's loading."""
    
    def __init__(
        self,
        lines: int = 3,
        show_avatar: bool = False,
        **kwargs
    ):
        """Initialize the skeleton loader.
        
        Args:
            lines: Number of text lines to show
            show_avatar: Whether to show avatar placeholder
        """
        super().__init__(**kwargs)
        self.lines = lines
        self.show_avatar = show_avatar
        
    def compose(self) -> ComposeResult:
        """Compose the skeleton UI."""
        with Container(classes="skeleton-container"):
            if self.show_avatar:
                with Horizontal(classes="skeleton-header"):
                    yield Static("", classes="skeleton-avatar")
                    with Vertical(classes="skeleton-title-group"):
                        yield Static("", classes="skeleton-title")
                        yield Static("", classes="skeleton-subtitle")
            
            for i in range(self.lines):
                width_class = "skeleton-line-full" if i == 0 else f"skeleton-line-{90 - (i * 10)}"
                yield Static("", classes=f"skeleton-line {width_class}")


class InlineLoader(Static):
    """An inline loading indicator for small async operations."""
    
    def __init__(
        self,
        loading_text: str = "Loading",
        success_text: str = "Done",
        error_text: str = "Failed",
        **kwargs
    ):
        """Initialize the inline loader.
        
        Args:
            loading_text: Text to show while loading
            success_text: Text to show on success
            error_text: Text to show on error
        """
        super().__init__(loading_text, **kwargs)
        self.loading_text = loading_text
        self.success_text = success_text
        self.error_text = error_text
        self.state = "loading"  # loading, success, error
        self.dots = 0
        
    async def on_mount(self) -> None:
        """Start the loading animation."""
        self.set_interval(0.5, self._update_dots)
    
    def _update_dots(self) -> None:
        """Update the loading dots animation."""
        if self.state == "loading":
            self.dots = (self.dots + 1) % 4
            dots_str = "." * self.dots
            self.update(f"{self.loading_text}{dots_str}")
    
    def set_success(self) -> None:
        """Set the loader to success state."""
        self.state = "success"
        self.update(f"✓ {self.success_text}")
        self.add_class("success")
        self.remove_class("error", "loading")
    
    def set_error(self, message: Optional[str] = None) -> None:
        """Set the loader to error state."""
        self.state = "error"
        error_text = message or self.error_text
        self.update(f"✗ {error_text}")
        self.add_class("error")
        self.remove_class("success", "loading")
    
    def reset(self) -> None:
        """Reset to loading state."""
        self.state = "loading"
        self.dots = 0
        self.update(self.loading_text)
        self.add_class("loading")
        self.remove_class("success", "error")


# Messages
class LoadingComplete(Message):
    """Message sent when loading completes successfully."""
    
    def __init__(self, content: Any):
        super().__init__()
        self.content = content


class LoadingFailed(Message):
    """Message sent when loading fails."""
    
    def __init__(self, error: str):
        super().__init__()
        self.error = error


class LoadingCancelled(Message):
    """Message sent when loading is cancelled."""
    pass