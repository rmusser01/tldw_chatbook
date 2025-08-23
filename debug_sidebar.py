#!/usr/bin/env python3
"""Debug script to isolate enhanced sidebar composition issues."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tldw_chatbook.Widgets.enhanced_settings_sidebar import EnhancedSettingsSidebar

def test_composition():
    """Test the composition methods directly."""
    try:
        # Mock config
        config = {
            "chat_defaults": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "system_prompt": "You are a helpful assistant.",
                "top_p": 0.95,
            }
        }
        
        print("Creating EnhancedSettingsSidebar...")
        sidebar = EnhancedSettingsSidebar(
            id_prefix="chat",
            config=config,
            id="test-sidebar"
        )
        
        print("Testing _compose_features_content...")
        try:
            for i, widget in enumerate(sidebar._compose_features_content()):
                print(f"  Widget {i}: {widget.__class__.__name__}")
        except Exception as e:
            print(f"ERROR in _compose_features_content: {e}")
            import traceback
            traceback.print_exc()
        
        print("Testing _compose_advanced_content...")
        try:
            for i, widget in enumerate(sidebar._compose_advanced_content()):
                print(f"  Widget {i}: {widget.__class__.__name__}")
        except Exception as e:
            print(f"ERROR in _compose_advanced_content: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR creating sidebar: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_composition()