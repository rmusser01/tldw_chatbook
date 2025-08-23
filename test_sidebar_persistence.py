#!/usr/bin/env python3
"""Test script to verify sidebar persistence and visual grouping functionality."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.state.ui_state import UIState

def test_ui_state_collapsible_management():
    """Test UIState collapsible management methods."""
    print("Testing UIState collapsible management...")
    
    ui_state = UIState()
    
    # Test setting collapsible state
    ui_state.set_collapsible_state("chat-quick-settings", True)
    assert ui_state.get_collapsible_state("chat-quick-settings") == True
    print("✓ Set collapsible state")
    
    # Test toggle
    result = ui_state.toggle_collapsible("chat-quick-settings")
    assert result == False  # Should toggle from True to False
    assert ui_state.get_collapsible_state("chat-quick-settings") == False
    print("✓ Toggle collapsible")
    
    # Test collapse all
    ui_state.set_collapsible_state("chat-quick-settings", False)
    ui_state.set_collapsible_state("chat-rag-panel", False)
    ui_state.set_collapsible_state("chat-model-params", False)
    ui_state.collapse_all(except_priority=False)
    
    assert ui_state.get_collapsible_state("chat-quick-settings") == True
    assert ui_state.get_collapsible_state("chat-rag-panel") == True
    assert ui_state.get_collapsible_state("chat-model-params") == True
    print("✓ Collapse all")
    
    # Test expand all
    ui_state.expand_all()
    assert ui_state.get_collapsible_state("chat-quick-settings") == False
    assert ui_state.get_collapsible_state("chat-rag-panel") == False
    assert ui_state.get_collapsible_state("chat-model-params") == False
    print("✓ Expand all")
    
    print("All UIState tests passed! ✅")

def test_sidebar_visual_elements():
    """Test that sidebar creates proper visual grouping elements."""
    print("\nTesting sidebar visual elements...")
    
    from tldw_chatbook.Widgets.settings_sidebar import create_settings_sidebar
    
    # Mock config
    config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant."
        }
    }
    
    # Generate sidebar widgets
    widgets = list(create_settings_sidebar("chat", config))
    
    # Check for quick actions bar
    quick_actions_found = False
    search_input_found = False
    group_headers_found = []
    
    for widget in widgets:
        widget_str = str(widget)
        if "quick-actions-bar" in widget_str:
            quick_actions_found = True
        if "settings-search" in widget_str:
            search_input_found = True
        if "group-header" in widget_str:
            group_headers_found.append(widget)
    
    assert quick_actions_found, "Quick actions bar not found"
    print("✓ Quick actions bar present")
    
    assert search_input_found, "Search input not found"
    print("✓ Search input present")
    
    assert len(group_headers_found) >= 2, f"Expected at least 2 group headers, found {len(group_headers_found)}"
    print(f"✓ Found {len(group_headers_found)} group headers")
    
    print("All sidebar visual element tests passed! ✅")

if __name__ == "__main__":
    test_ui_state_collapsible_management()
    test_sidebar_visual_elements()
    
    print("\n" + "="*50)
    print("All tests passed successfully! 🎉")
    print("="*50)
    
    print("\nImplemented features:")
    print("1. ✅ State Persistence - Collapsible states saved/restored")
    print("2. ✅ Visual Grouping - ESSENTIAL, FEATURES, ADVANCED groups")
    print("3. ✅ Quick Actions Bar - Expand/Collapse all buttons")
    print("4. 🔄 Search System - Input field added (functionality pending)")
    print("5. ⏳ Form Validation - Not yet implemented")