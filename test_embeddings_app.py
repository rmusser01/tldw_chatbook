#!/usr/bin/env python3
"""Minimal test app to verify embeddings navigation."""

from textual.app import App, ComposeResult
from textual.widgets import Button, Label, Static
from textual.containers import Container
from textual import on
from loguru import logger

# Import our windows
from tldw_chatbook.UI.Embeddings_Window import EmbeddingsWindow
from tldw_chatbook.UI.Embeddings_Management_Window import EmbeddingsManagementWindow

class TestEmbeddingsApp(App):
    """Test app for embeddings functionality."""
    
    CSS = """
    #embeddings-view-create {
        display: block;
    }
    
    #embeddings-view-manage {
        display: none;
    }
    
    .embeddings-view-area {
        height: 100%;
        width: 100%;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.embeddings_active_view = "embeddings-view-create"
    
    def compose(self) -> ComposeResult:
        yield Static("Test Embeddings Navigation", id="title")
        yield Button("Show Create View", id="show-create")
        yield Button("Show Manage View", id="show-manage")
        yield Container(id="view-container")
    
    def on_mount(self) -> None:
        """Mount the embeddings window."""
        container = self.query_one("#view-container")
        
        # Create a mock app instance
        mock_app = type('MockApp', (), {
            'chachanotes_db': None,
            'media_db': None,
        })()
        
        # Create the embeddings window
        embeddings_window = EmbeddingsWindow(mock_app, id="embeddings-window")
        container.mount(embeddings_window)
    
    @on(Button.Pressed, "#show-create")
    def show_create_view(self) -> None:
        """Show create view."""
        logger.info("Showing create view")
        self._switch_view("embeddings-view-create")
    
    @on(Button.Pressed, "#show-manage")
    def show_manage_view(self) -> None:
        """Show manage view."""
        logger.info("Showing manage view")
        self._switch_view("embeddings-view-manage")
    
    def _switch_view(self, view_id: str) -> None:
        """Switch to a specific view."""
        try:
            # Hide all views
            for vid in ["embeddings-view-create", "embeddings-view-manage"]:
                view = self.query_one(f"#{vid}")
                view.styles.display = "none"
            
            # Show selected view
            view = self.query_one(f"#{view_id}")
            view.styles.display = "block"
            
            self.notify(f"Switched to {view_id}")
        except Exception as e:
            logger.error(f"Failed to switch view: {e}")
            self.notify(f"Error: {str(e)}", severity="error")

if __name__ == "__main__":
    app = TestEmbeddingsApp()
    app.run()