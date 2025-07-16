#!/usr/bin/env python3
"""Quick test to verify Copy Note button functionality."""

import asyncio
from textual.app import App
from textual.widgets import Input, TextArea, Button
from textual.containers import Container, VerticalScroll

class TestCopyNoteApp(App):
    CSS = """
    Container {
        padding: 1;
    }
    """
    
    def compose(self):
        with Container():
            yield Input(id="chat-notes-title-input", placeholder="Note title...")
            yield TextArea(id="chat-notes-content-textarea", text="Test note content")
            yield Button("Copy Note", id="chat-notes-copy-button")
    
    async def on_button_pressed(self, event):
        if event.button.id == "chat-notes-copy-button":
            title_input = self.query_one("#chat-notes-title-input", Input)
            content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)
            
            title = title_input.value.strip()
            content = content_textarea.text.strip()
            
            if title and content:
                formatted_note = f"# {title}\n\n{content}"
            elif title:
                formatted_note = f"# {title}"
            elif content:
                formatted_note = content
            else:
                self.notify("No content to copy")
                return
            
            self.copy_to_clipboard(formatted_note)
            self.notify("Copied to clipboard!")
            self.exit()

if __name__ == "__main__":
    app = TestCopyNoteApp()
    app.run()