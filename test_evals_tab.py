#!/usr/bin/env python3
"""
Test clicking on the Evals tab in the real app
"""

import asyncio
from textual.pilot import Pilot
from tldw_chatbook.app import TldwCli

async def test_evals_tab():
    """Test the Evals tab in the actual app"""
    app = TldwCli()
    
    async with app.run_test() as pilot: 
        # Wait for app to load
        await pilot.pause(delay=0.5)
        
        # Try to click on the Evals tab
        try:
            # Click the Evals tab button
            await pilot.click("#tab-evals")
            await pilot.pause(delay=0.5)
            
            # Check if Evals window loaded
            from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
            evals_window = app.query_one(EvalsWindow)
            
            if evals_window:
                print("✅ Evals tab clicked and window loaded!")
                
                # Check orchestrator
                if evals_window.orchestrator:
                    print(f"✅ Orchestrator initialized: {evals_window.orchestrator}")
                    
                    # Check database
                    if hasattr(evals_window.orchestrator, 'db'):
                        print(f"✅ Database initialized: {evals_window.orchestrator.db}")
                        
                        # Try to list tasks
                        try:
                            tasks = evals_window.orchestrator.db.list_tasks()
                            print(f"✅ Tasks loaded: {len(tasks)} tasks")
                        except AttributeError as e:
                            print(f"❌ Error: {e}")
                            print(f"   DB methods: {dir(evals_window.orchestrator.db)}")
                else:
                    print("❌ Orchestrator not initialized")
            else:
                print("❌ Evals window not found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing Evals tab in real app...")
    asyncio.run(test_evals_tab())