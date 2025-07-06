#!/usr/bin/env python3
"""Test script to verify splash screen functionality."""

import time
from textual.app import App, ComposeResult
from textual.widgets import Static
from tldw_chatbook.Widgets.splash_screen import SplashScreen, SplashScreenClosed
from textual import on

class SplashTestApp(App):
    """Test app to verify splash screen works."""
    
    def compose(self) -> ComposeResult:
        """Compose with splash screen."""
        # Create splash screen with matrix effect
        self.splash = SplashScreen(
            card_name="matrix",
            duration=3.0,
            skip_on_keypress=True,
            show_progress=True
        )
        yield self.splash
    
    @on(SplashScreenClosed)
    async def on_splash_closed(self, event: SplashScreenClosed) -> None:
        """Handle splash screen closing."""
        await self.splash.remove()
        await self.mount(Static("Main app content loaded successfully!"))
        
    def on_mount(self) -> None:
        """Update progress during mount."""
        # Simulate initialization progress
        self.set_timer(0.5, lambda: self.splash.update_progress(0.25, "Loading configuration..."))
        self.set_timer(1.0, lambda: self.splash.update_progress(0.50, "Initializing database..."))
        self.set_timer(1.5, lambda: self.splash.update_progress(0.75, "Loading UI components..."))
        self.set_timer(2.0, lambda: self.splash.update_progress(1.0, "Ready!"))

if __name__ == "__main__":
    app = SplashTestApp()
    app.run()