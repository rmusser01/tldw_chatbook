#!/usr/bin/env python3
"""
Simple test of Evals window in isolation
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class SimpleTestApp(App):
    CSS = """
    Screen {
        background: $surface;
    }
    
    #test-container {
        width: 100%;
        height: 100%;
        border: solid blue;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="test-container"):
            yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")
    
    async def on_mount(self):
        # Check what's actually visible
        await asyncio.sleep(0.5)
        window = self.query_one(EvalsWindow)
        print(f"\n=== EVALS WINDOW STATE ===")
        print(f"Window visible: {window.visible}")
        print(f"Window display: {window.styles.display}")
        print(f"Window height: {window.styles.height}")
        print(f"Window width: {window.styles.width}")
        
        # Check container
        container = self.query_one("#test-container")
        print(f"\nContainer visible: {container.visible}")
        print(f"Container display: {container.styles.display}")
        print(f"Container height: {container.styles.height}")
        print(f"Container width: {container.styles.width}")
        
        # Check header
        try:
            header = window.query_one(".evals-header")
            print(f"\nHeader visible: {header.visible}")
            print(f"Header display: {header.styles.display}")
            
            # Get header text
            for static in header.query("Static"):
                print(f"Header text: '{static.renderable}'")
        except:
            print("\nHeader not found!")

if __name__ == "__main__":
    app = SimpleTestApp()
    app.run()