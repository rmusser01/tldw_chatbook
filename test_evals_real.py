#!/usr/bin/env python3
"""
Test script to verify the Evals UI actually works
"""

import sys
import asyncio
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.containers import Container
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestEvalsApp(App):
    """Test app to verify Evals UI works"""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the app"""
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        """Mock notify method"""
        print(f"[{severity.upper()}] {message}")

async def main():
    """Run the test"""
    app = TestEvalsApp()
    
    # Run for a short time to test
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Check if it loaded
        window = app.query_one(EvalsWindow)
        if window and window.orchestrator:
            print("✅ Evals UI loaded successfully!")
            print(f"   Orchestrator: {window.orchestrator}")
            print(f"   Database: {window.orchestrator.db}")
            
            # Check data
            try:
                tasks = window.orchestrator.db.list_tasks()
                models = window.orchestrator.db.list_models()
                print(f"   Tasks: {len(tasks)}")
                print(f"   Models: {len(models)}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
        else:
            print("❌ Failed to load Evals UI")

if __name__ == "__main__":
    print("Testing Evals UI...")
    asyncio.run(main())