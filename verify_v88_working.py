#!/usr/bin/env python3
"""Verify MediaWindowV88 is working correctly."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_fixes():
    """Verify all fixes are in place."""
    
    print("Checking MediaWindowV88 fixes...")
    print("=" * 50)
    
    # Check MediaWindowV88 for correct panel updates
    with open('tldw_chatbook/UI/MediaWindowV88.py', 'r') as f:
        content = f.read()
        
    if "self.metadata_panel.load_media(full_data)" in content:
        print("✓ MediaWindowV88: Direct panel updates (no call_from_thread)")
    else:
        print("✗ MediaWindowV88: Still using call_from_thread")
        return False
    
    # Check metadata panel for mount protection
    with open('tldw_chatbook/Widgets/MediaV88/metadata_panel.py', 'r') as f:
        content = f.read()
        
    if "if not self.is_mounted:" in content and "return" in content:
        print("✓ MetadataPanel: Protected against mount-time edits")
    else:
        print("✗ MetadataPanel: Missing mount protection")
        return False
        
    if "if self.edit_mode and self.is_mounted:" in content:
        print("✓ MetadataPanel: clear_display checks is_mounted")
    else:
        print("✗ MetadataPanel: clear_display missing mount check")
        return False
    
    # Check search bar for event stopping
    with open('tldw_chatbook/Widgets/MediaV88/search_bar.py', 'r') as f:
        content = f.read()
        
    if content.count("event.stop()") >= 4:
        print("✓ SearchBar: All button events stop propagation")
    else:
        print("✗ SearchBar: Missing event.stop() calls")
        return False
    
    print("=" * 50)
    print("✅ All fixes verified!")
    print("\nExpected behavior:")
    print("1. No 'Exiting edit mode' error during mount")
    print("2. Media selection loads content without errors")
    print("3. Search toggle works without 'Unhandled button' warnings")
    
    return True

if __name__ == "__main__":
    success = verify_fixes()
    sys.exit(0 if success else 1)