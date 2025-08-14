#!/usr/bin/env python3
"""Quick test to verify MediaIngestWindow loads without errors."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from tldw_chatbook.UI.MediaIngestWindow import MediaIngestWindow
    
    # Create a mock app instance
    class MockApp:
        pass
    
    try:
        # Try to instantiate the window
        window = MediaIngestWindow(MockApp())
        print("✅ MediaIngestWindow instantiated successfully!")
        
        # Try to compose it
        composed = list(window.compose())
        print(f"✅ MediaIngestWindow composed {len(composed)} widgets!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()