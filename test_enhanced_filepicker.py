#!/usr/bin/env python3
"""Test script to demonstrate the enhanced file picker features."""

from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Label
from textual.containers import Container

from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, EnhancedFileSave
from tldw_chatbook.Third_Party.textual_fspicker import Filters


class FilePickerTestApp(App):
    """A test app to demonstrate the enhanced file picker."""
    
    CSS = """
    Container {
        align: center middle;
        height: 100%;
    }
    
    Button {
        margin: 1;
    }
    
    Label {
        margin: 1;
        padding: 1;
        border: solid $primary;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.selected_file = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("Enhanced File Picker Demo", id="info")
            yield Button("Open File Dialog", id="open-btn", variant="primary")
            yield Button("Save File Dialog", id="save-btn", variant="success")
            yield Label("No file selected", id="result")
        yield Footer()
    
    def on_mount(self):
        """Set up key bindings display."""
        info = self.query_one("#info", Label)
        info.update(
            "Enhanced File Picker Demo\\n\\n"
            "Features:\\n"
            "• Show/hide hidden files (Ctrl+H or button)\\n"
            "• Sort by name, size, type, or date\\n"
            "• Search files in real-time\\n"
            "• Recent locations (Ctrl+R)\\n"
            "• Breadcrumb navigation\\n"
            "• Keyboard shortcuts (F5 to refresh)"
        )
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "open-btn":
            # Define some filters for common file types
            filters = Filters(
                ("Python Files", "*.py"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*"),
            )
            
            # Show the enhanced file open dialog
            file_path = await self.push_screen(
                EnhancedFileOpen(
                    location=Path.home(),
                    title="Select a File",
                    filters=filters
                )
            )
            
            if file_path:
                self.selected_file = file_path
                result = self.query_one("#result", Label)
                result.update(f"Selected: {file_path}")
            
        elif event.button.id == "save-btn":
            # Define filters for saving
            filters = Filters(
                ("Python Files", "*.py"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*"),
            )
            
            # Show the enhanced file save dialog
            file_path = await self.push_screen(
                EnhancedFileSave(
                    location=Path.cwd(),
                    title="Save File As",
                    filters=filters,
                    default_filename="untitled.txt"
                )
            )
            
            if file_path:
                self.selected_file = file_path
                result = self.query_one("#result", Label)
                result.update(f"Save to: {file_path}")


if __name__ == "__main__":
    app = FilePickerTestApp()
    app.run()