#!/usr/bin/env python3
"""Test script to verify embeddings download button works."""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Button, Label
from textual import on, work
from loguru import logger

class TestApp(App):
    def compose(self) -> ComposeResult:
        yield Label("Test Embeddings Download Button")
        yield Button("Download Model", id="embeddings-download-model")
        yield Label("Status: Ready", id="status")
    
    @on(Button.Pressed, "#embeddings-download-model")
    def on_download_model(self) -> None:
        """Test button handler."""
        status = self.query_one("#status", Label)
        status.update("Button pressed - handler called!")
        logger.info("Button handler executed")
        
        # Test worker
        self.run_worker(self._test_worker, thread=True, name="test_worker")
    
    def _test_worker(self) -> None:
        """Test worker function."""
        import time
        logger.info("Worker started")
        time.sleep(1)
        logger.info("Worker completed")
        
        # Update UI from worker
        status = self.query_one("#status", Label)
        self.call_from_thread(status.update, "Worker completed!")

if __name__ == "__main__":
    app = TestApp()
    app.run()