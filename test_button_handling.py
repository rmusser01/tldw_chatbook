#!/usr/bin/env python3
"""Test script to verify button handling in screen navigation mode."""

import asyncio
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Static
from textual.containers import Container
from loguru import logger

class TestScreen(Screen):
    """Test screen with buttons."""
    
    def compose(self) -> ComposeResult:
        """Compose the test screen."""
        with Container():
            yield Static("Test Screen", id="title")
            yield Button("Toggle Left", id="toggle-left")
            yield Button("Toggle Right", id="toggle-right")
            yield Button("Send Message", id="send-button")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses at screen level."""
        button_id = event.button.id
        logger.info(f"Screen handling button: {button_id}")
        
        if button_id == "toggle-left":
            self.notify("Left sidebar toggled")
        elif button_id == "toggle-right":
            self.notify("Right sidebar toggled")
        elif button_id == "send-button":
            self.notify("Message sent")
        
        # Stop event propagation
        event.stop()

class TestApp(App):
    """Test app with screen navigation."""
    
    def on_mount(self) -> None:
        """Mount the test screen."""
        self.push_screen(TestScreen())
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """App-level button handler."""
        logger.warning(f"Button event reached app level: {event.button.id}")
        # This should not be reached if screen handles the event

if __name__ == "__main__":
    app = TestApp()
    app.run()