#!/usr/bin/env python3
"""
Test that the Evals UI actually renders in the main app
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Disable splash screen for testing
os.environ['TLDW_SPLASH_ENABLED'] = 'false'

from textual.app import App
from tldw_chatbook.app import TldwCli
from tldw_chatbook import config

# Disable splash screen in config
config.cli_config['splash_screen'] = {'enabled': False}

async def test_evals_in_app():
    """Test Evals UI in the actual app"""
    
    # Create the actual app
    app = TldwCli()
    
    print("=" * 80)
    print("TESTING EVALS UI IN ACTUAL APP")
    print("=" * 80)
    
    async with app.run_test() as pilot:
        # Wait for app to initialize and splash screen to finish
        print("Waiting for app initialization...")
        await pilot.pause(delay=2.0)  # Wait for splash screen
        
        # Press a key to skip splash if needed
        await pilot.press("space")
        await pilot.pause(delay=0.5)
        
        print("\n1. CHECKING TAB EXISTS")
        print("-" * 40)
        
        # Check that the Evals tab exists
        try:
            # Click on the Evals tab
            await pilot.click("#evals-tab-button")
            await pilot.pause(delay=1.0)  # Give time for lazy loading
            
            print("✅ Clicked on Evals tab")
        except Exception as e:
            print(f"❌ FAILED: Could not click Evals tab - {e}")
            # Try alternative method
            print("Trying to set current_tab directly...")
            app.current_tab = "evals"
            await pilot.pause(delay=1.0)
        
        print("\n2. CHECKING WINDOW INITIALIZATION")
        print("-" * 40)
        
        # Check if the PlaceholderWindow exists
        try:
            placeholder = app.query_one("#evals-window")
            if placeholder:
                print("✅ Found evals-window container")
                
                # Check if it's initialized
                if hasattr(placeholder, 'is_initialized'):
                    if placeholder.is_initialized:
                        print("✅ Window is initialized")
                    else:
                        print("⚠️  Window not yet initialized, triggering initialization...")
                        # Trigger initialization
                        if hasattr(placeholder, 'initialize'):
                            placeholder.initialize()
                            await pilot.pause(delay=0.5)
                            print("✅ Window initialized manually")
            else:
                print("❌ FAILED: No evals-window found")
        except Exception as e:
            print(f"❌ FAILED: Error finding window - {e}")
        
        print("\n3. CHECKING UI ELEMENTS")
        print("-" * 40)
        
        # Now check for actual UI elements
        critical_elements = [
            "#task-select",
            "#model-select", 
            "#run-button",
            "#results-table"
        ]
        
        found_count = 0
        for elem_id in critical_elements:
            try:
                elem = app.query_one(elem_id)
                if elem:
                    print(f"✅ Found {elem_id}")
                    found_count += 1
                else:
                    print(f"❌ FAILED: {elem_id} not found")
            except Exception as e:
                print(f"❌ FAILED: {elem_id} - {e}")
        
        print("\n4. CHECKING ACTUAL WINDOW")
        print("-" * 40)
        
        # Try to get the actual EvalsWindow
        try:
            from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
            evals_window = app.query_one(EvalsWindow)
            if evals_window:
                print("✅ Found actual EvalsWindow instance")
                
                # Check if it has the orchestrator
                if hasattr(evals_window, 'orchestrator'):
                    if evals_window.orchestrator:
                        print("✅ Orchestrator is initialized")
                    else:
                        print("❌ FAILED: Orchestrator is None")
                
                # Check if compose was called
                children = list(evals_window.children)
                if len(children) > 0:
                    print(f"✅ Window has {len(children)} children")
                else:
                    print("❌ FAILED: Window has no children (compose not called?)")
            else:
                print("❌ FAILED: No EvalsWindow found")
        except Exception as e:
            print(f"❌ FAILED: Could not find EvalsWindow - {e}")
        
        print("\n5. CHECKING VISIBLE CONTENT")
        print("-" * 40)
        
        # Check for visible text that should be in the UI
        try:
            # Look for the header text
            all_statics = app.query("Static")
            header_found = False
            for static in all_statics:
                if "Evaluation Lab" in static.renderable:
                    print("✅ Found Evaluation Lab header")
                    header_found = True
                    break
            
            if not header_found:
                print("❌ FAILED: Header text not found")
                
        except Exception as e:
            print(f"❌ FAILED: Error checking visible content - {e}")
        
        print("\n" + "=" * 80)
        if found_count >= 3:
            print("✅ EVALS UI IS RENDERING IN THE APP!")
        else:
            print(f"❌ EVALS UI NOT FULLY RENDERING (only {found_count}/4 critical elements found)")
        print("=" * 80)

if __name__ == "__main__":
    print("Testing Evals UI in actual app...")
    asyncio.run(test_evals_in_app())