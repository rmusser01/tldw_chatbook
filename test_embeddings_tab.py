#!/usr/bin/env python3
"""Test script to verify embeddings tab switching."""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Button, Label
from textual.containers import Container, Horizontal
from textual.reactive import reactive

class TestApp(App):
    current_tab = reactive("chat")
    
    def compose(self) -> ComposeResult:
        yield Horizontal(
            Button("Chat", id="tab-chat"),
            Button("Embeddings", id="tab-embeddings"),
            id="tabs"
        )
        yield Container(
            Label("Chat Window", id="chat-window", classes="window"),
            Label("Embeddings Window", id="embeddings-window", classes="window"),
            id="content"
        )
    
    def on_mount(self):
        self.hide_inactive_windows()
    
    def hide_inactive_windows(self):
        """Hide all windows except the current tab."""
        for window in self.query(".window"):
            is_active = window.id == f"{self.current_tab}-window"
            window.display = is_active
            print(f"Window {window.id}: display={is_active}")
    
    def watch_current_tab(self, old_tab, new_tab):
        """Show/hide windows based on current tab."""
        print(f"\nSwitching from {old_tab} to {new_tab}")
        
        # Hide old window
        if old_tab:
            try:
                old_window = self.query_one(f"#{old_tab}-window")
                old_window.display = False
                print(f"Hidden {old_tab}-window")
            except:
                print(f"Could not find {old_tab}-window")
        
        # Show new window
        try:
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True
            print(f"Shown {new_tab}-window")
        except:
            print(f"Could not find {new_tab}-window")
    
    def on_button_pressed(self, event):
        """Handle tab button presses."""
        if event.button.id.startswith("tab-"):
            new_tab = event.button.id.replace("tab-", "")
            print(f"\nButton pressed: {event.button.id} -> tab: {new_tab}")
            self.current_tab = new_tab

if __name__ == "__main__":
    app = TestApp()
    app.run()