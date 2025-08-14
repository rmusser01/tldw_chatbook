#!/usr/bin/env python3
"""Test that MediaIngestWindow loads successfully."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_instantiation():
    """Test basic instantiation."""
    from tldw_chatbook.UI.MediaIngestWindow import MediaIngestWindow
    
    class MockApp:
        pass
    
    try:
        window = MediaIngestWindow(MockApp(), id="test-window")
        print("✅ MediaIngestWindow instantiated successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate: {e}")
        return False

def test_with_app():
    """Test loading in actual app context."""
    import asyncio
    from tldw_chatbook.app import TldwCli
    
    async def run_test():
        app = TldwCli()
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Try to navigate to ingest tab
            try:
                await pilot.click("#tab-ingest")
                await pilot.pause(0.5)
                
                # Check if window loaded
                ingest_window = app.query("#ingest-window")
                if ingest_window:
                    print("✅ Ingest window loaded in app!")
                    
                    # Check for TabbedContent
                    tabbed = app.query("TabbedContent")
                    if tabbed:
                        print("✅ TabbedContent found in ingest window!")
                    return True
                else:
                    print("❌ Ingest window not found")
                    return False
                    
            except Exception as e:
                print(f"❌ Error navigating to ingest tab: {e}")
                return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    print("Testing MediaIngestWindow...")
    
    # Test 1: Basic instantiation
    if test_instantiation():
        print("\nTest 1 passed: Basic instantiation")
    else:
        print("\nTest 1 failed")
        sys.exit(1)
    
    # Test 2: In app context
    print("\nTesting in app context...")
    if test_with_app():
        print("\nTest 2 passed: App integration")
    else:
        print("\nTest 2 failed")
        sys.exit(1)
    
    print("\n✅ All tests passed!")