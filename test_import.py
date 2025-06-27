#!/usr/bin/env python3
"""
Simple test script to verify tldw_chatbook can be imported after installation.
Run this after installing the package to verify everything is working.
"""

import sys

def test_import():
    """Test that the package can be imported and has expected attributes."""
    print("Testing tldw_chatbook import...")
    
    try:
        import tldw_chatbook
        print(f"✓ Successfully imported tldw_chatbook")
        print(f"  Version: {tldw_chatbook.__version__}")
        print(f"  Author: {tldw_chatbook.__author__}")
        print(f"  License: {tldw_chatbook.__license__}")
    except ImportError as e:
        print(f"✗ Failed to import tldw_chatbook: {e}")
        return False
    
    # Test main app import
    try:
        from tldw_chatbook.app import TldwCli
        print("✓ Successfully imported TldwCli app class")
    except ImportError as e:
        print(f"✗ Failed to import TldwCli: {e}")
        return False
    
    # Test entry point function
    try:
        from tldw_chatbook.app import main_cli_runner
        print("✓ Successfully imported main_cli_runner entry point")
    except ImportError as e:
        print(f"✗ Failed to import main_cli_runner: {e}")
        return False
    
    # Test CSS files are included
    try:
        import tldw_chatbook.css
        import os
        css_path = os.path.dirname(tldw_chatbook.css.__file__)
        tcss_files = [f for f in os.listdir(css_path) if f.endswith('.tcss')]
        if tcss_files:
            print(f"✓ Found {len(tcss_files)} CSS files")
        else:
            print("✗ No CSS files found")
            return False
    except Exception as e:
        print(f"✗ Failed to check CSS files: {e}")
        return False
    
    # Test Config_Files are included
    try:
        import tldw_chatbook.Config_Files
        config_path = os.path.dirname(tldw_chatbook.Config_Files.__file__)
        json_files = [f for f in os.listdir(config_path) if f.endswith('.json')]
        if json_files:
            print(f"✓ Found {len(json_files)} JSON template files")
        else:
            print("✗ No JSON template files found")
            return False
    except Exception as e:
        print(f"✗ Failed to check Config_Files: {e}")
        return False
    
    print("\nAll import tests passed! ✓")
    return True

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)