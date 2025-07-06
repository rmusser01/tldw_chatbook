# loading_states.py
# Description: Loading state widgets and transitions for evaluation UI
#
"""
Loading States and Transitions
-----------------------------

Provides loading state indicators and smooth transitions:
- Loading overlays
- Skeleton screens
- Progress indicators
- State transitions
"""

from typing import Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static, LoadingIndicator, ProgressBar
from textual.containers import Container, Center
from textual.timer import Timer
from textual.css.transition import Transition
from loguru import logger

class LoadingOverlay(Container):
    """Full-screen loading overlay with message."""
    
    message = reactive("Loading...")
    
    def __init__(self, message: str = "Loading...", **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.add_class("loading-overlay")
    
    def compose(self) -> ComposeResult:
        with Center(classes="loading-center"):
            yield LoadingIndicator()
            yield Static(self.message, id="loading-message", classes="loading-text")
    
    def update_message(self, message: str) -> None:
        """Update the loading message."""
        self.message = message
        try:
            self.query_one("#loading-message", Static).update(message)
        except:
            pass

class SkeletonLoader(Container):
    """Skeleton screen for loading content."""
    
    def __init__(self, num_items: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_items = num_items
        self.add_class("skeleton-loader")
    
    def compose(self) -> ComposeResult:
        for i in range(self.num_items):
            with Container(classes="skeleton-item"):
                yield Static("", classes="skeleton-line skeleton-title")
                yield Static("", classes="skeleton-line skeleton-subtitle")
                yield Static("", classes="skeleton-line skeleton-content")

class StateTransition(Container):
    """Smooth state transition container."""
    
    TRANSITIONS = {
        "fade": {"opacity": Transition("opacity", duration=0.3)},
        "slide": {"offset": Transition("offset", duration=0.3)},
        "scale": {"scale": Transition("scale", duration=0.2)}
    }
    
    current_state = reactive("idle")
    
    def __init__(self, transition_type: str = "fade", **kwargs):
        super().__init__(**kwargs)
        self.transition_type = transition_type
        self._content_cache = {}
        self._timer: Optional[Timer] = None
    
    def set_state(self, state: str, content: Optional[ComposeResult] = None) -> None:
        """Set the current state with optional content."""
        old_state = self.current_state
        self.current_state = state
        
        # Apply transition
        self._apply_transition(old_state, state, content)
    
    def _apply_transition(self, old_state: str, new_state: str, content: Optional[ComposeResult]) -> None:
        """Apply transition between states."""
        # Start transition out
        self.add_class("transitioning-out")
        
        # Schedule content update
        if self._timer:
            self._timer.stop()
        
        self._timer = self.set_timer(0.15, lambda: self._update_content(new_state, content))
    
    def _update_content(self, state: str, content: Optional[ComposeResult]) -> None:
        """Update content after transition."""
        # Clear existing content
        self.remove_children()
        
        # Add new content
        if content:
            self.mount(*content)
        elif state in self._content_cache:
            self.mount(*self._content_cache[state])
        
        # Transition in
        self.remove_class("transitioning-out")
        self.add_class("transitioning-in")
        
        # Clean up transition classes
        self.set_timer(0.3, lambda: self.remove_class("transitioning-in"))
    
    def cache_state_content(self, state: str, content: ComposeResult) -> None:
        """Cache content for a state."""
        self._content_cache[state] = content

class LoadingButton(Container):
    """Button with loading state."""
    
    is_loading = reactive(False)
    label = reactive("Click Me")
    
    def __init__(
        self, 
        label: str = "Click Me",
        on_click: Optional[Callable] = None,
        variant: str = "primary",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label = label
        self._on_click = on_click
        self.variant = variant
        self.add_class(f"loading-button {variant}")
    
    def compose(self) -> ComposeResult:
        if self.is_loading:
            yield LoadingIndicator(classes="button-spinner")
            yield Static("Loading...", classes="button-label loading")
        else:
            yield Static(self.label, classes="button-label")
    
    async def on_click(self) -> None:
        """Handle button click."""
        if self.is_loading or not self._on_click:
            return
        
        self.is_loading = True
        self.refresh()
        
        try:
            # Call the callback
            result = self._on_click()
            if hasattr(result, "__await__"):
                await result
        except Exception as e:
            logger.error(f"Error in loading button callback: {e}")
        finally:
            self.is_loading = False
            self.refresh()
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Update button state when loading changes."""
        if is_loading:
            self.add_class("is-loading")
        else:
            self.remove_class("is-loading")

class ProgressStep(Container):
    """Single step in a progress workflow."""
    
    status = reactive("pending")  # pending, active, completed, error
    
    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.add_class("progress-step")
    
    def compose(self) -> ComposeResult:
        with Container(classes="step-indicator"):
            if self.status == "completed":
                yield Static("✓", classes="step-icon completed")
            elif self.status == "active":
                yield LoadingIndicator(classes="step-icon active")
            elif self.status == "error":
                yield Static("✗", classes="step-icon error")
            else:
                yield Static("○", classes="step-icon pending")
        
        yield Static(self.label, classes="step-label")
    
    def set_status(self, status: str) -> None:
        """Update step status."""
        self.status = status
        self.refresh()
        
        # Update CSS classes
        self.remove_class("pending", "active", "completed", "error")
        self.add_class(status)

class WorkflowProgress(Container):
    """Multi-step workflow progress indicator."""
    
    current_step = reactive(0)
    
    def __init__(self, steps: list[str], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self._step_widgets = []
        self.add_class("workflow-progress")
    
    def compose(self) -> ComposeResult:
        yield Static("Progress", classes="progress-title")
        
        with Container(classes="steps-container"):
            for i, step_label in enumerate(self.steps):
                step = ProgressStep(step_label, id=f"step-{i}")
                self._step_widgets.append(step)
                yield step
                
                # Add connector between steps
                if i < len(self.steps) - 1:
                    yield Static("", classes="step-connector")
    
    def set_step(self, step_index: int, status: str = "active") -> None:
        """Set the current step and update statuses."""
        self.current_step = step_index
        
        for i, step_widget in enumerate(self._step_widgets):
            if i < step_index:
                step_widget.set_status("completed")
            elif i == step_index:
                step_widget.set_status(status)
            else:
                step_widget.set_status("pending")
    
    def complete_step(self, step_index: int) -> None:
        """Mark a step as completed."""
        if step_index < len(self._step_widgets):
            self._step_widgets[step_index].set_status("completed")
    
    def error_step(self, step_index: int) -> None:
        """Mark a step as errored."""
        if step_index < len(self._step_widgets):
            self._step_widgets[step_index].set_status("error")

class DataLoadingCard(Container):
    """Card with loading state for data display."""
    
    is_loading = reactive(True)
    has_error = reactive(False)
    
    def __init__(self, title: str = "Data", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.add_class("data-loading-card")
    
    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="card-title")
        
        with Container(classes="card-content", id="card-content"):
            if self.is_loading:
                yield SkeletonLoader(num_items=2)
            elif self.has_error:
                yield Static("❌ Failed to load data", classes="error-message")
                yield Static("Click to retry", classes="retry-hint")
            else:
                yield Container(id="actual-content")
    
    def set_loading(self, is_loading: bool) -> None:
        """Set loading state."""
        self.is_loading = is_loading
        self.has_error = False
        self.refresh()
    
    def set_error(self, error: bool = True) -> None:
        """Set error state."""
        self.has_error = error
        self.is_loading = False
        self.refresh()
    
    def set_content(self, content: ComposeResult) -> None:
        """Set the actual content when loaded."""
        self.is_loading = False
        self.has_error = False
        
        # Update content
        content_container = self.query_one("#card-content")
        content_container.remove_children()
        
        actual = Container(id="actual-content")
        actual.mount(*content)
        content_container.mount(actual)

# CSS Helper for smooth transitions
LOADING_STATES_CSS = """
/* Loading Overlay */
.loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    z-index: 1000;
}

.loading-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.loading-text {
    margin-top: 1;
    color: $text-muted;
}

/* Skeleton Loader */
.skeleton-item {
    padding: 1 2;
    margin-bottom: 1;
}

.skeleton-line {
    height: 1;
    background: $surface-lighten-1;
    animation: skeleton-pulse 1.5s infinite;
}

.skeleton-title {
    width: 60%;
    margin-bottom: 0.5;
}

.skeleton-subtitle {
    width: 40%;
    margin-bottom: 0.5;
}

.skeleton-content {
    width: 80%;
}

@keyframes skeleton-pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 0.7; }
}

/* State Transitions */
.transitioning-out {
    opacity: 0.3;
    transition: opacity 0.15s ease-out;
}

.transitioning-in {
    opacity: 1;
    transition: opacity 0.15s ease-in;
}

/* Loading Button */
.loading-button {
    border: solid $primary;
    padding: 0 2;
    height: 3;
    content-align: center middle;
}

.loading-button.is-loading {
    opacity: 0.7;
}

.button-spinner {
    display: none;
}

.loading-button.is-loading .button-spinner {
    display: block;
}

/* Progress Steps */
.workflow-progress {
    padding: 1 2;
    background: $surface;
    border: solid $border;
}

.steps-container {
    display: flex;
    align-items: center;
    margin-top: 1;
}

.progress-step {
    display: flex;
    align-items: center;
    margin-right: 1;
}

.step-icon {
    width: 3;
    height: 3;
    text-align: center;
    border: solid $border;
    border-radius: 50%;
}

.step-icon.completed {
    background: $success;
    color: $text;
}

.step-icon.active {
    border-color: $primary;
}

.step-icon.error {
    background: $error;
    color: $text;
}

.step-connector {
    width: 4;
    height: 1;
    border-top: dashed $border;
    margin: 0 1;
}

/* Data Loading Card */
.data-loading-card {
    border: solid $border;
    padding: 1 2;
}

.card-title {
    text-style: bold;
    margin-bottom: 1;
}

.error-message {
    color: $error;
    text-align: center;
    margin: 2 0;
}

.retry-hint {
    color: $text-muted;
    text-align: center;
    text-style: italic;
}
"""