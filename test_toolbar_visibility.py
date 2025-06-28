#!/usr/bin/env python3
"""Simple test to verify the enhanced file picker toolbar appears."""

import asyncio
from pathlib import Path
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

async def test_compose():
    """Test that the compose method works and includes the toolbar."""
    filters = Filters(
        ("Python Files", "*.py"),
        ("All Files", "*.*"),
    )
    
    dialog = EnhancedFileOpen(
        location=Path.home(),
        title="Test Enhanced File Picker",
        filters=filters
    )
    
    # Test compose
    print("Testing compose method...")
    try:
        widgets = list(dialog.compose())
        print(f"Found {len(widgets)} widgets")
        
        # Check if FilePickerToolbar is in the widgets
        for widget in widgets:
            print(f"  - {type(widget).__name__}")
            if hasattr(widget, 'id'):
                print(f"    ID: {widget.id}")
        
        # Check for toolbar
        toolbar_found = any("FilePickerToolbar" in str(type(widget).__name__) for widget in widgets)
        print(f"\nToolbar found: {toolbar_found}")
        
    except Exception as e:
        print(f"Error during compose: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_compose())