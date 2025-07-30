# tldw_chatbook/Widgets/tooltip.py
# Tooltip widget for providing contextual help
#
# Imports
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer

class Tooltip(Widget):
    """Tooltip widget that shows on hover/focus.
    
    Features:
    - Auto-positioning (above/below target)
    - Auto-dismiss after delay
    - Keyboard accessible
    """
    
    DEFAULT_CLASSES = "tooltip"
    
    # Whether tooltip is currently visible
    visible: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        text: str,
        delay: float = 0.5,
        dismiss_after: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text = text
        self.delay = delay
        self.dismiss_after = dismiss_after
        self._show_timer: Optional[Timer] = None
        self._hide_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Compose the tooltip content."""
        yield Static(self.text, classes="tooltip-content")
    
    def watch_visible(self, visible: bool) -> None:
        """Handle visibility changes."""
        self.display = visible
        
        if visible and self.dismiss_after > 0:
            # Start auto-dismiss timer
            if self._hide_timer:
                self._hide_timer.stop()
            self._hide_timer = self.set_timer(self.dismiss_after, self.hide)
        elif not visible and self._hide_timer:
            self._hide_timer.stop()
            self._hide_timer = None
    
    def show_delayed(self) -> None:
        """Show tooltip after delay."""
        if self._show_timer:
            self._show_timer.stop()
        self._show_timer = self.set_timer(self.delay, self.show)
    
    def show(self) -> None:
        """Show the tooltip immediately."""
        self.visible = True
    
    def hide(self) -> None:
        """Hide the tooltip."""
        self.visible = False
        if self._show_timer:
            self._show_timer.stop()
            self._show_timer = None


class TooltipMixin:
    """Mixin to add tooltip support to any widget."""
    
    def __init__(self, *args, tooltip: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tooltip: Optional[Tooltip] = None
        self._tooltip_text = tooltip
    
    def on_mount(self) -> None:
        """Create tooltip on mount if text provided."""
        super().on_mount()
        if self._tooltip_text:
            self._create_tooltip()
    
    def _create_tooltip(self) -> None:
        """Create the tooltip widget."""
        if not self._tooltip and self._tooltip_text:
            self._tooltip = Tooltip(self._tooltip_text)
            # Mount tooltip to the screen/app level for proper positioning
            self.screen.mount(self._tooltip)
            self._tooltip.visible = False
    
    def on_enter(self) -> None:
        """Show tooltip on mouse enter."""
        super().on_enter()
        if self._tooltip:
            self._position_tooltip()
            self._tooltip.show_delayed()
    
    def on_leave(self) -> None:
        """Hide tooltip on mouse leave."""
        super().on_leave()
        if self._tooltip:
            self._tooltip.hide()
    
    def on_focus(self) -> None:
        """Show tooltip on focus for keyboard users."""
        super().on_focus()
        if self._tooltip:
            self._position_tooltip()
            self._tooltip.show_delayed()
    
    def on_blur(self) -> None:
        """Hide tooltip on blur."""
        super().on_blur()
        if self._tooltip:
            self._tooltip.hide()
    
    def _position_tooltip(self) -> None:
        """Position tooltip relative to this widget."""
        if not self._tooltip:
            return
            
        # Get widget position
        x, y = self.offset
        widget_width = self.size.width
        widget_height = self.size.height
        
        # Position above widget by default
        tooltip_x = x + (widget_width // 2)  # Center horizontally
        tooltip_y = y - 2  # Above widget
        
        # Adjust if too close to top
        if tooltip_y < 2:
            tooltip_y = y + widget_height + 1  # Below widget
        
        # Apply position
        self._tooltip.styles.offset = (tooltip_x, tooltip_y)


class HelpIcon(Static, TooltipMixin):
    """Help icon widget with built-in tooltip support."""
    
    DEFAULT_CLASSES = "help-icon"
    
    def __init__(self, help_text: str, icon: str = "ℹ️", **kwargs):
        super().__init__(icon, tooltip=help_text, **kwargs)