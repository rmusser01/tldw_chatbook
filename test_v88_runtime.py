#!/usr/bin/env python3
"""Test MediaWindowV88 runtime behavior."""

import sys
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_media_window():
    """Test the MediaWindowV88 functionality."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
    
    print("Creating app instance...")
    app = TldwCli()
    
    # Initialize databases
    if not app.media_db:
        print("✗ Media database not initialized")
        return False
    
    print("✓ Media database initialized")
    
    # Create MediaWindowV88 instance
    print("\nCreating MediaWindowV88...")
    media_window = MediaWindowV88(app)
    
    # Check app.call_from_thread availability
    if hasattr(app, 'call_from_thread'):
        print("✓ app.call_from_thread available")
    else:
        print("✗ app.call_from_thread NOT available")
        return False
    
    # Test that components are created
    print("\nTesting component creation...")
    try:
        # Mount the window to test compose
        from textual.app import App
        test_app = App()
        
        # Check if compose works
        components = list(media_window.compose())
        print(f"✓ Compose created {len(components)} components")
        
        # Verify component types
        component_names = [c.__class__.__name__ for c in components]
        expected = ["NavigationColumn", "Container"]
        
        for exp in expected:
            if any(exp in name for name in component_names):
                print(f"  ✓ {exp} component found")
            else:
                print(f"  ✗ {exp} component NOT found")
        
    except Exception as e:
        print(f"✗ Error during compose: {e}")
        return False
    
    # Test search functionality
    print("\nTesting search method...")
    try:
        results, total = await media_window.search_media_async(
            query="test",
            page=1,
            per_page=10
        )
        print(f"✓ Search executed: {len(results)} results, {total} total")
    except Exception as e:
        print(f"✗ Search failed: {e}")
    
    print("\n✅ All runtime tests completed")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_media_window())
    sys.exit(0 if result else 1)