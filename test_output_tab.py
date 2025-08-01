#!/usr/bin/env python3
"""Test script to verify Output tab visibility in EmbeddingsCreationContent"""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import TabbedContent, TabPane, Label, Button
from tldw_chatbook.UI.Embeddings_Creation_Content import EmbeddingsCreationContent

class TestOutputTabApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    
    #test-container {
        width: 90%;
        height: 90%;
        border: solid $primary;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="test-container"):
            yield EmbeddingsCreationContent(app_instance=self)

if __name__ == "__main__":
    app = TestOutputTabApp()
    app.run()