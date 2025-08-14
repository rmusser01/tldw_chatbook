#!/usr/bin/env python3
"""Test script to verify the Media Ingest tab loads in the main app."""

import sys
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ingest_tab():
    """Test that the ingest tab loads correctly."""
    from tldw_chatbook.app import TldwCli
    from textual.pilot import Pilot
    
    app = TldwCli()
    
    async with app.run_test() as pilot: 
        # Wait for app to load
        await pilot.pause(0.5)
        
        # Try to switch to ingest tab
        try:
            # Click on ingest tab button
            await pilot.click("#nav-button-ingest")
            await pilot.pause(0.5)
            
            # Check if the ingest window is visible
            ingest_window = app.query_one("#ingest-window")
            assert ingest_window is not None, "Ingest window not found"
            assert ingest_window.display == True, "Ingest window not displayed"
            
            print("✅ Media Ingest tab loaded successfully!")
            
            # Check for TabbedContent
            tabbed_content = ingest_window.query("TabbedContent")
            if tabbed_content:
                print("✅ TabbedContent widget found!")
            else:
                print("❌ TabbedContent widget not found")
                
        except Exception as e:
            print(f"❌ Error loading ingest tab: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ingest_tab())