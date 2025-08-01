"""
Tests for Toast Notification Widget.
Tests notification creation, stacking, auto-dismiss, and user interactions.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from textual.widgets import Button, Static
from textual.containers import Container, Vertical

from tldw_chatbook.Widgets.toast_notification import (
    ToastNotification,
    ToastManager,
    ToastDismissed,
    ToastMixin
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestToastNotification(EmbeddingsTestBase):
    """Test individual toast notification behavior."""
    
    @pytest.mark.asyncio
    async def test_toast_creation(self):
        """Test creating a toast notification with different severities."""
        # Test each severity level
        severities = ['success', 'info', 'warning', 'error']
        
        for severity in severities:
            toast = ToastNotification(
                message=f"Test {severity} message",
                severity=severity,
                duration=5.0
            )
            
            assert toast.message == f"Test {severity} message"
            assert toast.severity == severity
            assert toast.duration == 5.0
            assert toast.toast_id.startswith("toast-")
    
    @pytest.mark.asyncio
    async def test_toast_compose(self):
        """Test toast UI composition."""
        toast = ToastNotification(
            message="Test message",
            severity="success",
            duration=3.0
        )
        
        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check container has correct class
            container = pilot.app.query_one(".toast-container")
            assert "toast-success" in container.classes
            
            # Check icon
            icon = pilot.app.query_one(".toast-icon")
            assert icon.renderable == "✅"
            
            # Check message
            message = pilot.app.query_one(".toast-message")
            assert message.renderable == "Test message"
            
            # Check close button
            close_btn = pilot.app.query_one("#toast-close", Button)
            assert close_btn.label == "×"
    
    @pytest.mark.asyncio
    async def test_toast_auto_dismiss(self):
        """Test toast auto-dismiss functionality."""
        toast = ToastNotification(
            message="Auto dismiss test",
            severity="info",
            duration=0.5  # Short duration for testing
        )
        
        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Toast should be visible initially
            assert toast.display == True
            
            # Wait for auto-dismiss
            await asyncio.sleep(0.6)
            await pilot.pause()
            
            # Toast should start dismissing
            assert "toast-slide-out" in toast.classes
    
    @pytest.mark.asyncio
    async def test_toast_manual_dismiss(self):
        """Test manual dismissal via close button."""
        toast = ToastNotification(
            message="Manual dismiss test",
            severity="warning",
            duration=10.0  # Long duration
        )
        
        app = WidgetTestApp(toast)
        dismissed_messages = []
        
        # Capture dismiss messages
        def on_dismissed(message):
            if isinstance(message, ToastDismissed):
                dismissed_messages.append(message)
        
        app.on_toast_dismissed = on_dismissed
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click close button
            await pilot.click("#toast-close")
            await pilot.pause()
            
            # Should add slide-out class
            assert "toast-slide-out" in toast.classes
    
    @pytest.mark.asyncio
    async def test_toast_click_dismiss(self):
        """Test dismissal by clicking anywhere on toast."""
        toast = ToastNotification(
            message="Click to dismiss",
            severity="error",
            duration=10.0
        )
        
        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click on the toast itself
            await pilot.click(".toast-container")
            await pilot.pause()
            
            # Should start dismissing
            assert "toast-slide-out" in toast.classes


class TestToastManager(EmbeddingsTestBase):
    """Test toast manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_show_toast(self):
        """Test showing toasts through manager."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Show a toast
            toast_id = manager.show_toast("Test message", "success", 3.0)
            await pilot.pause()
            
            # Check toast was created
            assert len(manager.active_toasts) == 1
            assert manager.active_toasts[0].toast_id == toast_id
            
            # Check toast is in DOM
            toast = pilot.app.query_one(f".toast-notification")
            assert toast is not None
    
    @pytest.mark.asyncio
    async def test_manager_multiple_toasts(self):
        """Test managing multiple toasts."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Show multiple toasts
            ids = []
            for i in range(3):
                toast_id = manager.show_toast(f"Toast {i}", "info", 5.0)
                ids.append(toast_id)
                await pilot.pause(0.1)
            
            # All toasts should be active
            assert len(manager.active_toasts) == 3
            
            # Check they're stacked in DOM
            toasts = pilot.app.query(".toast-notification")
            assert len(toasts) == 3
    
    @pytest.mark.asyncio
    async def test_manager_max_toasts(self):
        """Test maximum toast limit."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Show more than max toasts
            for i in range(ToastManager.MAX_TOASTS + 2):
                manager.show_toast(f"Toast {i}", "info", 10.0)
                await pilot.pause(0.05)
            
            # Should only have MAX_TOASTS active
            assert len(manager.active_toasts) <= ToastManager.MAX_TOASTS
    
    @pytest.mark.asyncio
    async def test_manager_convenience_methods(self):
        """Test convenience methods for different severities."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Test each convenience method
            manager.show_success("Success message", 1.0)
            await pilot.pause(0.1)
            assert manager.active_toasts[-1].severity == "success"
            
            manager.show_error("Error message", 1.0)
            await pilot.pause(0.1)
            assert manager.active_toasts[-1].severity == "error"
            
            manager.show_warning("Warning message", 1.0)
            await pilot.pause(0.1)
            assert manager.active_toasts[-1].severity == "warning"
            
            manager.show_info("Info message", 1.0)
            await pilot.pause(0.1)
            assert manager.active_toasts[-1].severity == "info"
    
    @pytest.mark.asyncio
    async def test_manager_dismiss_all(self):
        """Test dismissing all toasts at once."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Show multiple toasts
            for i in range(3):
                manager.show_toast(f"Toast {i}", "info", 10.0)
            await pilot.pause(0.1)
            
            assert len(manager.active_toasts) == 3
            
            # Dismiss all
            manager.dismiss_all()
            await pilot.pause(0.1)
            
            # All should be dismissing
            for toast in manager.active_toasts:
                assert "toast-slide-out" in toast.classes


class TestToastMixin(EmbeddingsTestBase):
    """Test the ToastMixin for easy integration."""
    
    class TestScreen(ToastMixin):
        """Test screen with toast support."""
        def compose(self):
            yield Container()
    
    @pytest.mark.asyncio
    async def test_mixin_initialization(self):
        """Test mixin properly initializes toast manager."""
        screen = self.TestScreen()
        
        # Initially no manager
        assert screen._toast_manager is None
        
        # Manager created on mount
        app = WidgetTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Manager should be created
            assert screen._toast_manager is not None
            assert isinstance(screen._toast_manager, ToastManager)
    
    @pytest.mark.asyncio
    async def test_mixin_toast_methods(self):
        """Test mixin provides toast methods."""
        screen = self.TestScreen()
        app = WidgetTestApp(screen)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Test general toast method
            screen.toast("Test message", "success", 2.0)
            await pilot.pause(0.1)
            
            # Check toast was created
            toasts = pilot.app.query(".toast-notification")
            assert len(toasts) == 1
            
            # Test convenience methods
            screen.toast_success("Success!")
            screen.toast_error("Error!")
            screen.toast_warning("Warning!")
            screen.toast_info("Info!")
            await pilot.pause(0.1)
            
            # Should have 5 toasts total
            toasts = pilot.app.query(".toast-notification")
            assert len(toasts) == 5


class TestToastIntegration(EmbeddingsTestBase):
    """Test toast integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_toast_with_long_message(self):
        """Test toast with long message wrapping."""
        long_message = "This is a very long message that should wrap properly in the toast notification without breaking the layout"
        
        toast = ToastNotification(
            message=long_message,
            severity="info",
            duration=5.0
        )
        
        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check message is displayed
            message_widget = pilot.app.query_one(".toast-message")
            assert long_message in str(message_widget.renderable)
    
    @pytest.mark.asyncio
    async def test_toast_dismiss_event_handling(self):
        """Test proper event handling for dismissal."""
        manager = ToastManager()
        app = WidgetTestApp(manager)
        
        dismissed_count = 0
        
        def count_dismissals(message):
            nonlocal dismissed_count
            if isinstance(message, ToastDismissed):
                dismissed_count += 1
        
        # Mock message handling
        original_on_toast_dismissed = manager.on_toast_dismissed
        manager.on_toast_dismissed = lambda msg: (count_dismissals(msg), original_on_toast_dismissed(msg))
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Show and dismiss a toast
            toast_id = manager.show_toast("Test", "info", 0.3)
            await pilot.pause(0.1)
            
            # Wait for auto-dismiss
            await asyncio.sleep(0.5)
            await pilot.pause(0.1)
            
            # Should have received dismiss event
            assert dismissed_count >= 1