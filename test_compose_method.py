#!/usr/bin/env python3
"""
Test if compose method is working correctly
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Select, Button, Label, Input
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestApp(App):
    def compose(self) -> ComposeResult:
        window = EvalsWindow(app_instance=self)
        yield window
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")

async def test_compose():
    app = TestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        window = app.query_one(EvalsWindow)
        
        # Count widgets
        all_widgets = list(window.query("*"))
        selects = list(window.query("Select"))
        inputs = list(window.query("Input"))
        buttons = list(window.query("Button"))
        containers = list(window.query("Container"))
        
        print(f"Total widgets: {len(all_widgets)}")
        print(f"Containers: {len(containers)}")
        print(f"Selects: {len(selects)}")
        print(f"Inputs: {len(inputs)}")
        print(f"Buttons: {len(buttons)}")
        
        # Check specific IDs
        ids_to_check = ["#task-select", "#model-select", "#temperature-input", "#run-button"]
        for widget_id in ids_to_check:
            try:
                widget = window.query_one(widget_id)
                print(f"✅ Found {widget_id}: {widget.__class__.__name__}")
            except:
                print(f"❌ Missing {widget_id}")

if __name__ == "__main__":
    asyncio.run(test_compose())