#!/usr/bin/env python3
"""Test script to verify embeddings download button fix."""

import asyncio
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Button, Label
from textual.containers import Container

class TestEmbeddingsDownload(App):
    """Test app to verify button event handling."""
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Test Embeddings Download Button")
            yield Button("Download Model", id="embeddings-download-model")
            yield Button("Load Model", id="embeddings-load-model")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """App-level button handler (should not be reached if widget handles it)."""
        button_id = event.button.id
        if button_id == "embeddings-download-model":
            print("ERROR: Button event reached app level - widget handler failed!")
        else:
            print(f"App handled button: {button_id}")
    
    async def on_mount(self) -> None:
        """Mount handler."""
        print("Test app mounted")

class EmbeddingsWidget(Container):
    """Widget with button handlers using @on decorator."""
    
    def compose(self) -> ComposeResult:
        yield Button("Download Model", id="embeddings-download-model")
        yield Button("Load Model", id="embeddings-load-model")
    
    @on(Button.Pressed, "#embeddings-download-model")
    async def on_download_model(self, event: Button.Pressed) -> None:
        """Handle download button - should stop event propagation."""
        event.stop()  # This prevents the event from bubbling up
        print("SUCCESS: Download button handled at widget level!")
        self.app.exit()
    
    @on(Button.Pressed, "#embeddings-load-model") 
    async def on_load_model(self, event: Button.Pressed) -> None:
        """Handle load button."""
        event.stop()
        print("Load button handled at widget level")

if __name__ == "__main__":
    print("Testing embeddings download button fix...")
    print("Expected: 'SUCCESS: Download button handled at widget level!'")
    print("If you see 'ERROR: Button event reached app level', the fix didn't work.\n")
    
    app = TestEmbeddingsDownload()
    app.run()