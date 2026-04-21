"""Breadcrumb navigation widget for eval screens."""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from textual import on
from textual.widgets import Static, Button
from textual.containers import Horizontal
from textual.app import ComposeResult
from textual.message import Message

from loguru import logger


@dataclass
class BreadcrumbItem:
    """Single breadcrumb item."""
    label: str
    screen_id: Optional[str] = None
    is_active: bool = False


class BreadcrumbClicked(Message):
    """Message when a breadcrumb is clicked."""
    
    def __init__(self, screen_id: str):
        super().__init__()
        self.screen_id = screen_id


class BreadcrumbTrail(Horizontal):
    """
    Breadcrumb trail widget for navigation.
    
    Shows the current navigation path and allows
    clicking to go back to previous screens.
    """
    
    DEFAULT_CSS = """
    BreadcrumbTrail {
        height: 3;
        width: 100%;
        background: $panel;
        padding: 0 2;
        align: left middle;
        border-bottom: solid $primary-background;
    }
    
    .breadcrumb-item {
        margin: 0;
        padding: 0 1;
        background: transparent;
        border: none;
        color: $text;
    }
    
    .breadcrumb-item:hover {
        color: $accent;
        text-style: underline;
    }
    
    .breadcrumb-item.active {
        color: $primary;
        text-style: bold;
    }
    
    .breadcrumb-separator {
        margin: 0 1;
        color: $text-muted;
    }
    
    .breadcrumb-home {
        margin-right: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trail: List[BreadcrumbItem] = [
            BreadcrumbItem("Evaluation Lab", "eval_home", True)
        ]
    
    def compose(self) -> ComposeResult:
        """Compose the breadcrumb trail."""
        # Home icon
        yield Button("🏠", id="breadcrumb-home", classes="breadcrumb-item breadcrumb-home")
        
        # Build trail
        for i, item in enumerate(self.trail):
            if i > 0:
                yield Static("›", classes="breadcrumb-separator")
            
            if item.is_active:
                yield Static(item.label, classes="breadcrumb-item active")
            else:
                yield Button(
                    item.label,
                    id=f"breadcrumb-{item.screen_id}",
                    classes="breadcrumb-item"
                )
    
    def push(self, label: str, screen_id: str) -> None:
        """Add a new breadcrumb to the trail."""
        # Mark all existing as inactive
        for item in self.trail:
            item.is_active = False
        
        # Add new active item
        self.trail.append(BreadcrumbItem(label, screen_id, True))
        
        # Refresh display
        self.refresh()
        logger.debug(f"Pushed breadcrumb: {label} ({screen_id})")
    
    def pop(self) -> Optional[BreadcrumbItem]:
        """Remove the last breadcrumb."""
        if len(self.trail) > 1:
            removed = self.trail.pop()
            # Mark new last as active
            if self.trail:
                self.trail[-1].is_active = True
            self.refresh()
            logger.debug(f"Popped breadcrumb: {removed.label}")
            return removed
        return None
    
    def pop_to(self, screen_id: str) -> None:
        """Pop breadcrumbs until reaching the specified screen."""
        # Find the target in trail
        target_index = -1
        for i, item in enumerate(self.trail):
            if item.screen_id == screen_id:
                target_index = i
                break
        
        if target_index >= 0:
            # Remove everything after target
            self.trail = self.trail[:target_index + 1]
            # Mark target as active
            if self.trail:
                for item in self.trail:
                    item.is_active = False
                self.trail[-1].is_active = True
            self.refresh()
            logger.debug(f"Popped to: {screen_id}")
    
    def clear(self) -> None:
        """Clear the trail and reset to home."""
        self.trail = [
            BreadcrumbItem("Evaluation Lab", "eval_home", True)
        ]
        self.refresh()
    
    @on(Button.Pressed, ".breadcrumb-item")
    def handle_breadcrumb_click(self, event: Button.Pressed) -> None:
        """Handle clicking on a breadcrumb."""
        button_id = event.button.id
        
        if button_id == "breadcrumb-home":
            # Go to home
            self.clear()
            self.post_message(BreadcrumbClicked("eval_home"))
        elif button_id and button_id.startswith("breadcrumb-"):
            # Extract screen ID
            screen_id = button_id.replace("breadcrumb-", "")
            self.pop_to(screen_id)
            self.post_message(BreadcrumbClicked(screen_id))