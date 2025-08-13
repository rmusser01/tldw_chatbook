#!/usr/bin/env python3
"""
Test Evals window in headless mode
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class HeadlessTestApp(App):
    def compose(self) -> ComposeResult:
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")

async def test_headless():
    """Test in headless mode"""
    app = HeadlessTestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get the window
        window = app.query_one(EvalsWindow)
        
        print("\n=== EVALS WINDOW STRUCTURE ===")
        print(f"Window exists: {window is not None}")
        
        if window:
            # Check visibility
            print(f"Window visible: {window.visible}")
            print(f"Window display: {window.styles.display}")
            
            # Check if it has content
            children = list(window.children)
            print(f"Direct children count: {len(children)}")
            
            # Count all descendants
            all_widgets = list(window.query("*"))
            print(f"Total widgets: {len(all_widgets)}")
            
            # Check for key elements
            elements = {
                ".evals-header": "Header",
                ".evals-scroll-container": "Scroll container",
                "#task-select": "Task selector",
                "#model-select": "Model selector",
                "#run-button": "Run button",
                "#results-table": "Results table"
            }
            
            print("\n=== KEY ELEMENTS ===")
            for selector, name in elements.items():
                try:
                    elem = window.query_one(selector)
                    print(f"✅ {name}: Found (display={elem.styles.display})")
                except:
                    print(f"❌ {name}: NOT FOUND")
            
            # Check what's in the header
            try:
                header = window.query_one(".evals-header")
                header_text = []
                for static in header.query("Static"):
                    header_text.append(static.renderable)
                print(f"\nHeader text: {header_text}")
            except:
                print("\nCouldn't get header text")
            
            # Check if the window has proper size
            print(f"\n=== SIZE INFO ===")
            print(f"Window size: {window.size}")
            print(f"Window region: {window.region}")
            
            # Check if compose was called
            print(f"\n=== COMPOSE CHECK ===")
            print(f"Has compose method: {hasattr(window, 'compose')}")
            print(f"Has _initialized flag: {hasattr(window, '_initialized')}")
            if hasattr(window, '_initialized'):
                print(f"Is initialized: {window._initialized}")

if __name__ == "__main__":
    asyncio.run(test_headless())