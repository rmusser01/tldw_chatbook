"""
Tests for the current toast notification widgets.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.toast_notification import (
    ToastDismissed,
    ToastManager,
    ToastNotification,
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


def _static_text(widget: Static) -> str:
    return str(widget.render())


class TestToastNotification(EmbeddingsTestBase):
    """Test individual toast behavior."""

    @pytest.mark.asyncio
    async def test_toast_creation(self):
        toast = ToastNotification("Test success message", severity="success", duration=5.0)

        assert toast.message == "Test success message"
        assert toast.severity == "success"
        assert toast.duration == 5.0
        assert toast.toast_id.startswith("toast-")

    @pytest.mark.asyncio
    async def test_toast_compose(self):
        toast = ToastNotification("Test message", severity="success", duration=3.0)

        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()

            container = pilot.app.query_one(".toast-container")
            icon = pilot.app.query_one(".toast-icon", Static)
            message = pilot.app.query_one(".toast-message", Static)
            close_btn = pilot.app.query_one("#toast-close", Button)

            assert "toast-success" in container.classes
            assert _static_text(icon) == "✅"
            assert _static_text(message) == "Test message"
            assert close_btn.label == "×"

    @pytest.mark.asyncio
    async def test_toast_auto_dismiss_adds_slide_out(self):
        toast = ToastNotification("Auto dismiss", severity="info", duration=0.2)

        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "toast-slide-in" in toast.classes

            await asyncio.sleep(0.25)
            await pilot.pause(0.1)

            assert "toast-slide-out" in toast.classes

    @pytest.mark.asyncio
    async def test_toast_close_button_dismisses(self):
        toast = ToastNotification("Dismiss me", severity="warning", duration=10.0)

        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.click("#toast-close")
            await pilot.pause()

            assert "toast-slide-out" in toast.classes


class TestToastManager(EmbeddingsTestBase):
    """Test toast manager behavior."""

    @pytest.mark.asyncio
    async def test_manager_show_toast(self):
        manager = ToastManager()
        app = WidgetTestApp(manager)

        async with app.run_test() as pilot:
            await pilot.pause()

            toast_id = manager.show_toast("Test message", "success", 3.0)
            await pilot.pause()

            assert len(manager.active_toasts) == 1
            assert manager.active_toasts[0].toast_id == toast_id
            assert len(pilot.app.query(".toast-notification")) == 1

    @pytest.mark.asyncio
    async def test_manager_respects_max_toasts(self):
        manager = ToastManager()
        app = WidgetTestApp(manager)

        async with app.run_test() as pilot:
            await pilot.pause()

            for index in range(ToastManager.MAX_TOASTS + 2):
                manager.show_toast(f"Toast {index}", "info", 10.0)
                await pilot.pause(0.05)

            assert len(manager.active_toasts) <= ToastManager.MAX_TOASTS

    @pytest.mark.asyncio
    async def test_manager_convenience_methods(self):
        manager = ToastManager()
        app = WidgetTestApp(manager)

        async with app.run_test() as pilot:
            await pilot.pause()

            manager.show_success("Success", 1.0)
            manager.show_error("Error", 1.0)
            manager.show_warning("Warning", 1.0)
            manager.show_info("Info", 1.0)
            await pilot.pause()

            severities = [toast.severity for toast in manager.active_toasts]
            assert severities == ["success", "error", "warning", "info"]

    @pytest.mark.asyncio
    async def test_manager_dismiss_all_marks_each_toast(self):
        manager = ToastManager()
        app = WidgetTestApp(manager)

        async with app.run_test() as pilot:
            await pilot.pause()

            for index in range(3):
                manager.show_toast(f"Toast {index}", "info", 10.0)
            await pilot.pause()

            manager.dismiss_all()
            await pilot.pause()

            assert all("toast-slide-out" in toast.classes for toast in manager.active_toasts)

    def test_manager_handles_dismiss_message(self):
        manager = ToastManager()
        toast = ToastNotification("Tracked", toast_id="toast-123")
        manager.active_toasts = [toast]

        manager.on_toast_dismissed(ToastDismissed("toast-123"))

        assert manager.active_toasts == []


class TestToastIntegration(EmbeddingsTestBase):
    """Integration coverage for long messages and stacking."""

    @pytest.mark.asyncio
    async def test_toast_with_long_message(self):
        long_message = (
            "This is a very long message that should remain readable in the toast "
            "notification without breaking the layout."
        )
        toast = ToastNotification(long_message, severity="info", duration=5.0)

        app = WidgetTestApp(toast)
        async with app.run_test() as pilot:
            await pilot.pause()

            message_widget = pilot.app.query_one(".toast-message", Static)
            assert long_message in _static_text(message_widget)

    @pytest.mark.asyncio
    async def test_manager_auto_dismiss_updates_active_toasts(self):
        manager = ToastManager()
        app = WidgetTestApp(manager)

        async with app.run_test() as pilot:
            await pilot.pause()

            toast_id = manager.show_toast("Short lived", "info", 0.2)
            await pilot.pause(0.1)
            assert any(toast.toast_id == toast_id for toast in manager.active_toasts)

            await asyncio.sleep(0.25)
            await pilot.pause(0.4)

            manager.on_toast_dismissed(ToastDismissed(toast_id))
            assert all(toast.toast_id != toast_id for toast in manager.active_toasts)
