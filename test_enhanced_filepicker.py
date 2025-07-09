#!/usr/bin/env python3
"""
Test script for the enhanced file picker
"""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Label, Footer
from pathlib import Path

# Import the enhanced file picker
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, EnhancedFileSave
from tldw_chatbook.Third_Party.textual_fspicker import Filters


class FilePickerTestApp(App):
    """Test app for enhanced file picker"""
    
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
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("o", "open_file", "Open File"),
        ("s", "save_file", "Save File"),
        ("i", "open_image", "Open Image"),
        ("c", "open_code", "Open Code"),
    ]
    
    def __init__(self):
        super().__init__()
        self.last_selected = "No file selected yet"
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Enhanced File Picker Test", id="title")
            yield Button("Open File (Generic)", id="open-generic")
            yield Button("Open Image", id="open-image")
            yield Button("Open Code File", id="open-code")
            yield Button("Save File", id="save-file")
            yield Label(self.last_selected, id="result")
        yield Footer()
    
    async def on_button_pressed(self, event):
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "open-generic":
            await self.action_open_file()
        elif button_id == "open-image":
            await self.action_open_image()
        elif button_id == "open-code":
            await self.action_open_code()
        elif button_id == "save-file":
            await self.action_save_file()
    
    async def action_open_file(self):
        """Open a generic file"""
        def on_file_selected(path):
            if path:
                self.last_selected = f"Selected: {path}"
                self.query_one("#result", Label).update(self.last_selected)
                self.notify(f"File selected: {path.name}")
        
        await self.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Open File",
                context="test_generic"
            ),
            on_file_selected
        )
    
    async def action_open_image(self):
        """Open an image file"""
        def on_file_selected(path):
            if path:
                self.last_selected = f"Image: {path}"
                self.query_one("#result", Label).update(self.last_selected)
                self.notify(f"Image selected: {path.name}")
        
        # Create image filters
        filters = Filters(
            ("Image Files", lambda p: p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")),
            ("PNG files", lambda p: p.suffix.lower() == ".png"),
            ("JPEG files", lambda p: p.suffix.lower() in (".jpg", ".jpeg")),
            ("All files", lambda p: True)
        )
        
        await self.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select Image",
                filters=filters,
                context="test_images"
            ),
            on_file_selected
        )
    
    async def action_open_code(self):
        """Open a code file"""
        def on_file_selected(path):
            if path:
                self.last_selected = f"Code: {path}"
                self.query_one("#result", Label).update(self.last_selected)
                self.notify(f"Code file selected: {path.name}")
        
        # Create code filters
        filters = Filters(
            ("Python files", lambda p: p.suffix.lower() == ".py"),
            ("JavaScript files", lambda p: p.suffix.lower() in (".js", ".ts")),
            ("Code files", lambda p: p.suffix.lower() in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".h")),
            ("All files", lambda p: True)
        )
        
        await self.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select Code File",
                filters=filters,
                context="test_code"
            ),
            on_file_selected
        )
    
    async def action_save_file(self):
        """Save a file"""
        def on_file_selected(path):
            if path:
                self.last_selected = f"Save to: {path}"
                self.query_one("#result", Label).update(self.last_selected)
                self.notify(f"File will be saved to: {path}")
        
        await self.push_screen(
            EnhancedFileSave(
                location=".",
                title="Save File",
                default_filename="test_file.txt",
                context="test_save"
            ),
            on_file_selected
        )


if __name__ == "__main__":
    app = FilePickerTestApp()
    app.run()