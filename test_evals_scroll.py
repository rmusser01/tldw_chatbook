#!/usr/bin/env python3
"""
Test scrolling functionality of Evals window
"""

import asyncio
from textual.app import App, ComposeResult
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestEvalsApp(App):
    def compose(self) -> ComposeResult:
        window = EvalsWindow(app_instance=self)
        yield window
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")
    
    async def on_mount(self):
        """Test scrolling after mount"""
        await asyncio.sleep(0.5)
        
        window = self.query_one(EvalsWindow)
        scroll_container = window.query_one(".evals-scroll-container")
        
        print(f"\n=== SCROLL TEST ===")
        print(f"Scroll container class: {scroll_container.__class__.__name__}")
        print(f"Can scroll: {scroll_container.can_scroll}")
        print(f"Scrollable size: {scroll_container.scrollable_content_region}")
        print(f"Visible size: {scroll_container.size}")
        print(f"Scroll offset: {scroll_container.scroll_offset}")
        
        # Try to scroll down
        print("\nAttempting to scroll down...")
        scroll_container.scroll_down()
        await asyncio.sleep(0.1)
        print(f"New scroll offset: {scroll_container.scroll_offset}")
        
        # Check if content is visible
        all_sections = list(window.query(".config-section"))
        print(f"\nConfig sections found: {len(all_sections)}")
        for i, section in enumerate(all_sections):
            print(f"  Section {i}: visible={section.display}, size={section.size}")

if __name__ == "__main__":
    TestEvalsApp().run()