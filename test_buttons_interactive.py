#!/usr/bin/env python3
"""Interactive test for button functionality in the app."""

import asyncio
import time
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Static, RichLog
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive

class TestChatScreen(Screen):
    """Test screen simulating the chat interface."""
    
    left_sidebar_collapsed = reactive(False)
    right_sidebar_collapsed = reactive(False)
    message_sent = reactive("")
    
    def compose(self) -> ComposeResult:
        """Compose the test screen."""
        with Vertical():
            yield Static("Test Chat Screen", id="title")
            
            with Horizontal():
                # Left sidebar toggle
                yield Button("◀", id="toggle-chat-left-sidebar", classes="sidebar-toggle")
                
                # Input area
                yield Static("Type message here...", id="input-area")
                
                # Right sidebar toggle  
                yield Button("▶", id="toggle-chat-right-sidebar", classes="sidebar-toggle")
                
                # Send button
                yield Button("Send", id="send-chat", classes="send-button")
            
            # Status display
            yield RichLog(id="status-log", highlight=True, markup=True)
    
    def on_mount(self) -> None:
        """Initialize the screen."""
        log = self.query_one("#status-log", RichLog)
        log.write("[green]Screen mounted successfully[/green]")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses at screen level."""
        button_id = event.button.id
        log = self.query_one("#status-log", RichLog)
        
        log.write(f"[yellow]Button pressed: {button_id}[/yellow]")
        
        if button_id == "toggle-chat-left-sidebar":
            self.left_sidebar_collapsed = not self.left_sidebar_collapsed
            status = "collapsed" if self.left_sidebar_collapsed else "expanded"
            log.write(f"[blue]Left sidebar {status}[/blue]")
            self.notify(f"Left sidebar {status}")
            
        elif button_id == "toggle-chat-right-sidebar":
            self.right_sidebar_collapsed = not self.right_sidebar_collapsed
            status = "collapsed" if self.right_sidebar_collapsed else "expanded"
            log.write(f"[blue]Right sidebar {status}[/blue]")
            self.notify(f"Right sidebar {status}")
            
        elif button_id == "send-chat":
            timestamp = time.strftime("%H:%M:%S")
            self.message_sent = f"Message sent at {timestamp}"
            log.write(f"[green]Message sent at {timestamp}[/green]")
            self.notify("Message sent!")
        
        # Stop event propagation
        event.stop()
    
    def watch_left_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to left sidebar state changes."""
        log = self.query_one("#status-log", RichLog)
        log.write(f"[dim]Left sidebar watcher triggered: {collapsed}[/dim]")
    
    def watch_right_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to right sidebar state changes."""
        log = self.query_one("#status-log", RichLog)
        log.write(f"[dim]Right sidebar watcher triggered: {collapsed}[/dim]")

class TestApp(App):
    """Test app with screen navigation."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #title {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        text-align: center;
        padding: 1;
    }
    
    Horizontal {
        height: 3;
        padding: 0 1;
    }
    
    .sidebar-toggle {
        width: 3;
        margin: 0 1;
    }
    
    #input-area {
        width: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    
    .send-button {
        width: 8;
        background: $success;
    }
    
    #status-log {
        height: 1fr;
        border: solid $primary;
        margin: 1;
    }
    """
    
    def on_mount(self) -> None:
        """Mount the test screen."""
        self.push_screen(TestChatScreen())
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """App-level button handler - should not be reached."""
        self.bell()
        self.notify(f"WARNING: Button {event.button.id} reached app level!", severity="warning")

if __name__ == "__main__":
    app = TestApp()
    app.run()