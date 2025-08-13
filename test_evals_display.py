#!/usr/bin/env python3
"""
Test if Evals window displays correctly and is scrollable
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestEvalsApp(App):
    def compose(self) -> ComposeResult:
        window = EvalsWindow(app_instance=self)
        yield window
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")
    
    async def on_mount(self):
        """Auto-test after mounting"""
        await asyncio.sleep(1)
        
        # Try to scroll the content
        try:
            window = self.query_one(EvalsWindow)
            scroll_container = window.query_one(".evals-scroll-container")
            
            # Report what we found
            print("\n=== EVALS WINDOW TEST RESULTS ===")
            print(f"Window visible: {window.display}")
            print(f"Window size: {window.size}")
            print(f"Scroll container found: {scroll_container is not None}")
            if scroll_container:
                print(f"Scroll container size: {scroll_container.size}")
                print(f"Can scroll: {scroll_container.can_scroll}")
                print(f"Scrollable height: {scroll_container.scrollable_content_region}")
            
            # Check for key widgets
            widgets_to_check = [
                ("#task-select", "Task dropdown"),
                ("#model-select", "Model dropdown"),
                ("#temperature-input", "Temperature input"),
                ("#max-tokens-input", "Max tokens input"),
                ("#run-button", "Run button"),
                (".config-section", "Config sections"),
                (".form-row", "Form rows")
            ]
            
            for selector, name in widgets_to_check:
                try:
                    widgets = list(window.query(selector))
                    if widgets:
                        print(f"✅ {name}: {len(widgets)} found")
                        if widgets[0].display == False:
                            print(f"  ⚠️ But display is False!")
                    else:
                        print(f"❌ {name}: Not found")
                except Exception as e:
                    print(f"❌ {name}: Error - {e}")
                    
            print("=================================\n")
            
        except Exception as e:
            print(f"Error during test: {e}")
        
        # Keep app running for a moment
        await asyncio.sleep(2)
        self.exit()

if __name__ == "__main__":
    app = TestEvalsApp()
    app.run()