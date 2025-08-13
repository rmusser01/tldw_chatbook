#!/usr/bin/env python3
"""
Debug the Evals window composition
"""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Static
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class DebugApp(App):
    def compose(self) -> ComposeResult:
        yield EvalsWindow(app_instance=self)
    
    def on_mount(self):
        # Debug: print widget tree
        print("\n=== WIDGET TREE ===")
        self._print_tree(self, 0)
        
    def _print_tree(self, widget, level):
        indent = "  " * level
        print(f"{indent}{widget.__class__.__name__} (id={widget.id}, display={widget.styles.display}, visible={widget.visible})")
        for child in widget.children:
            self._print_tree(child, level + 1)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")

async def debug_compose():
    """Debug what's being composed"""
    app = DebugApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get the window
        window = app.query_one(EvalsWindow)
        
        print("\n=== EVALS WINDOW DEBUG ===")
        print(f"Window exists: {window is not None}")
        if window:
            print(f"Window ID: {window.id}")
            print(f"Window visible: {window.visible}")
            print(f"Window display: {window.styles.display}")
            print(f"Window children count: {len(list(window.children))}")
            
            # Check specific elements
            print("\n=== CHECKING KEY ELEMENTS ===")
            elements_to_check = [
                ".evals-header",
                ".evals-scroll-container",
                ".evals-content",
                "#task-select",
                "#model-select",
                "#run-button"
            ]
            
            for selector in elements_to_check:
                try:
                    elem = window.query_one(selector)
                    print(f"✅ {selector}: display={elem.styles.display}, visible={elem.visible}")
                except:
                    print(f"❌ {selector}: NOT FOUND")
            
            # Check CSS
            print("\n=== CSS CHECK ===")
            print(f"Has DEFAULT_CSS: {hasattr(window, 'DEFAULT_CSS')}")
            if hasattr(window, 'DEFAULT_CSS'):
                print(f"CSS length: {len(window.DEFAULT_CSS)} chars")
                
            # List all children at top level
            print("\n=== TOP-LEVEL CHILDREN ===")
            for child in window.children:
                print(f"- {child.__class__.__name__} (id={child.id}, classes={child.classes})")

if __name__ == "__main__":
    asyncio.run(debug_compose())