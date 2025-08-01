# tldw_chatbook/Widgets/toast_notification.py
# Toast notification system for temporary messages
#
# Imports
from __future__ import annotations
from typing import Optional, Literal, List
from datetime import datetime

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static, Button
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from textual.message import Message
from loguru import logger

# Configure logger
logger = logger.bind(module="toast_notification")

# Type definitions
ToastSeverity = Literal["success", "info", "warning", "error"]


class ToastDismissed(Message):
    """Message sent when a toast is dismissed."""
    
    def __init__(self, toast_id: str) -> None:
        self.toast_id = toast_id
        super().__init__()


class ToastNotification(Widget):
    """Individual toast notification widget.
    
    Features:
    - Auto-dismiss after duration
    - Click to dismiss
    - Severity-based styling
    - Slide animations
    """
    
    DEFAULT_CLASSES = "toast-notification"
    
    # Duration before auto-dismiss (seconds)
    DEFAULT_DURATION = 5.0
    
    def __init__(
        self,
        message: str,
        severity: ToastSeverity = "info",
        duration: float = DEFAULT_DURATION,
        toast_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.message = message
        self.severity = severity
        self.duration = duration
        self.toast_id = toast_id or f"toast-{id(self)}"
        self._dismiss_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Compose the toast content."""
        with Container(classes=f"toast-container toast-{self.severity}"):
            # Icon based on severity
            icon = self._get_severity_icon()
            yield Static(icon, classes="toast-icon")
            
            # Message
            yield Static(self.message, classes="toast-message")
            
            # Close button
            yield Button("×", id="toast-close", classes="toast-close", variant="plain")
    
    def on_mount(self) -> None:
        """Start auto-dismiss timer when mounted."""
        if self.duration > 0:
            self._dismiss_timer = self.set_timer(self.duration, self.dismiss)
        
        # Add slide-in animation class
        self.add_class("toast-slide-in")
    
    def _get_severity_icon(self) -> str:
        """Get icon based on severity."""
        icons = {
            "success": "✅",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌"
        }
        return icons.get(self.severity, "ℹ️")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle close button click."""
        if event.button.id == "toast-close":
            event.stop()
            self.dismiss()
    
    def on_click(self) -> None:
        """Dismiss on click anywhere in toast."""
        self.dismiss()
    
    def dismiss(self) -> None:
        """Dismiss the toast with animation."""
        # Cancel timer if active
        if self._dismiss_timer:
            self._dismiss_timer.stop()
            self._dismiss_timer = None
        
        # Add slide-out animation class
        self.remove_class("toast-slide-in")
        self.add_class("toast-slide-out")
        
        # Remove after animation completes
        self.set_timer(0.3, self._remove)
    
    def _remove(self) -> None:
        """Remove the toast from DOM."""
        # Notify parent of dismissal
        self.post_message(ToastDismissed(self.toast_id))
        self.remove()


class ToastManager(Widget):
    """Manager for displaying and stacking toast notifications.
    
    Features:
    - Queue management
    - Automatic positioning
    - Maximum toast limit
    """
    
    DEFAULT_CLASSES = "toast-manager"
    MAX_TOASTS = 5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_toasts: List[ToastNotification] = []
        
    def compose(self) -> ComposeResult:
        """Compose the toast container."""
        with Vertical(id="toast-stack", classes="toast-stack"):
            pass
    
    def show_toast(
        self,
        message: str,
        severity: ToastSeverity = "info",
        duration: float = ToastNotification.DEFAULT_DURATION
    ) -> str:
        """Show a new toast notification.
        
        Returns:
            Toast ID for tracking
        """
        # Remove oldest toast if at limit
        if len(self.active_toasts) >= self.MAX_TOASTS:
            oldest = self.active_toasts[0]
            oldest.dismiss()
        
        # Create new toast
        toast_id = f"toast-{datetime.now().timestamp()}"
        toast = ToastNotification(
            message=message,
            severity=severity,
            duration=duration,
            toast_id=toast_id
        )
        
        # Add to stack
        stack = self.query_one("#toast-stack", Vertical)
        stack.mount(toast)
        self.active_toasts.append(toast)
        
        logger.info(f"Showing toast: {message} ({severity})")
        return toast_id
    
    def on_toast_dismissed(self, message: ToastDismissed) -> None:
        """Handle toast dismissal."""
        # Remove from active list
        self.active_toasts = [
            t for t in self.active_toasts 
            if t.toast_id != message.toast_id
        ]
    
    def dismiss_all(self) -> None:
        """Dismiss all active toasts."""
        for toast in self.active_toasts:
            toast.dismiss()
    
    def show_success(self, message: str, duration: float = 3.0) -> str:
        """Show a success toast."""
        return self.show_toast(message, "success", duration)
    
    def show_error(self, message: str, duration: float = 6.0) -> str:
        """Show an error toast."""
        return self.show_toast(message, "error", duration)
    
    def show_warning(self, message: str, duration: float = 5.0) -> str:
        """Show a warning toast."""
        return self.show_toast(message, "warning", duration)
    
    def show_info(self, message: str, duration: float = 4.0) -> str:
        """Show an info toast."""
        return self.show_toast(message, "info", duration)


class ToastMixin:
    """Mixin to add toast notification support to any screen/app."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._toast_manager: Optional[ToastManager] = None
    
    def on_mount(self) -> None:
        """Create toast manager on mount."""
        super().on_mount()
        if not self._toast_manager:
            self._toast_manager = ToastManager()
            # Mount to screen for proper positioning
            self.screen.mount(self._toast_manager)
    
    def toast(
        self,
        message: str,
        severity: ToastSeverity = "info",
        duration: float = ToastNotification.DEFAULT_DURATION
    ) -> None:
        """Show a toast notification."""
        if self._toast_manager:
            self._toast_manager.show_toast(message, severity, duration)
    
    def toast_success(self, message: str) -> None:
        """Show a success toast."""
        self.toast(message, "success", 3.0)
    
    def toast_error(self, message: str) -> None:
        """Show an error toast."""
        self.toast(message, "error", 6.0)
    
    def toast_warning(self, message: str) -> None:
        """Show a warning toast."""
        self.toast(message, "warning", 5.0)
    
    def toast_info(self, message: str) -> None:
        """Show an info toast."""
        self.toast(message, "info", 4.0)