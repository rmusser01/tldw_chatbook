#!/usr/bin/env python3
"""
Debug tool to inspect the live Evals window state
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.widgets import Button, Static, TextArea
from textual.containers import Container, Vertical

class DebugApp(App):
    CSS = """
    #output {
        height: 100%;
        width: 100%;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Inspect Evals Window", id="inspect")
            yield TextArea(id="output", read_only=True)
    
    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "inspect":
            await self.inspect_app()
    
    async def inspect_app(self):
        """Connect to running app and inspect"""
        output = []
        
        # Import the app
        from tldw_chatbook.app import TldwCli
        
        # Create a test instance
        app = TldwCli()
        
        async with app.run_test() as pilot:
            # Wait for app to load
            await pilot.pause(delay=2.0)
            
            # Press space to skip splash if needed
            await pilot.press("space")
            await pilot.pause(delay=0.5)
            
            output.append("=== APP INITIALIZED ===\n")
            
            # Switch to evals tab
            try:
                # Set tab directly
                app.current_tab = "evals"
                await pilot.pause(delay=1.0)
                output.append("Switched to evals tab\n")
            except Exception as e:
                output.append(f"Failed to switch tab: {e}\n")
            
            # Check for the window
            try:
                window = app.query_one("#evals-window")
                output.append(f"\n=== EVALS WINDOW FOUND ===")
                output.append(f"Type: {window.__class__.__name__}")
                output.append(f"Display: {window.display}")
                output.append(f"Visible: {window.visible}")
                output.append(f"Size: {window.size}")
                output.append(f"Styles.display: {window.styles.display}")
                output.append(f"Styles.visibility: {window.styles.visibility}")
                output.append(f"Styles.height: {window.styles.height}")
                output.append(f"Styles.width: {window.styles.width}")
                
                # Check if it's a PlaceholderWindow
                if hasattr(window, '_actual_window'):
                    output.append(f"\n=== ACTUAL WINDOW ===")
                    if window._actual_window:
                        actual = window._actual_window
                        output.append(f"Type: {actual.__class__.__name__}")
                        output.append(f"Display: {actual.display}")
                        output.append(f"Visible: {actual.visible}")
                        output.append(f"Size: {actual.size}")
                        output.append(f"Children count: {len(list(actual.children))}")
                        
                        # Check for key elements
                        try:
                            header = actual.query_one(".evals-header")
                            output.append(f"\nHeader found: {header.visible}")
                        except:
                            output.append("\nHeader NOT FOUND")
                        
                        try:
                            scroll = actual.query_one(".evals-scroll-container")
                            output.append(f"Scroll container found: {scroll.visible}")
                            output.append(f"Scroll container children: {len(list(scroll.children))}")
                        except:
                            output.append("Scroll container NOT FOUND")
                    else:
                        output.append("_actual_window is None!")
                
                # Get all visible Static widgets
                output.append(f"\n=== VISIBLE TEXT ===")
                all_statics = window.query("Static")
                visible_count = 0
                for static in all_statics[:10]:  # First 10
                    if static.visible:
                        text = str(static.renderable)[:50]
                        if text.strip():
                            output.append(f"- {text}")
                            visible_count += 1
                output.append(f"Total visible Static widgets: {visible_count}")
                
            except Exception as e:
                output.append(f"\nError inspecting window: {e}")
            
            # Check CSS
            output.append(f"\n=== CSS CHECK ===")
            try:
                from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
                evals = app.query_one(EvalsWindow)
                output.append(f"EvalsWindow found directly")
                output.append(f"Has DEFAULT_CSS: {hasattr(evals, 'DEFAULT_CSS')}")
                if hasattr(evals, 'DEFAULT_CSS'):
                    output.append(f"CSS length: {len(evals.DEFAULT_CSS)}")
            except Exception as e:
                output.append(f"Could not find EvalsWindow: {e}")
        
        # Update output
        text_area = self.query_one("#output", TextArea)
        text_area.text = "\n".join(str(x) for x in output)

if __name__ == "__main__":
    app = DebugApp()
    app.run()