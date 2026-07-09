#!/usr/bin/env python3
"""
verify_command_palette.py

Simple verification script for command palette functionality.
This script tests the structural aspects that can be verified without running the full app.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test that all command palette providers can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from tldw_chatbook.app import (
            ThemeProvider, TabNavigationProvider, LLMProviderProvider,
            QuickActionsProvider, SettingsProvider, CharacterProvider,
            MediaProvider, DeveloperProvider, TldwCli
        )
        from tldw_chatbook.Constants import TAB_CHAT
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_app_configuration():
    """Test that app is configured with command providers."""
    print("\n🔍 Testing app configuration...")
    
    try:
        from tldw_chatbook.app import TldwCli
        
        # Check COMMANDS
        if hasattr(TldwCli, 'COMMANDS'):
            commands_count = len(TldwCli.COMMANDS)
            print(f"✅ Found {commands_count} command providers")
            
            if commands_count >= 8:  # Should have 8+ providers
                print("✅ Expected number of providers registered")
            else:
                print(f"⚠️ Expected 8+ providers, found {commands_count}")
        else:
            print("❌ No COMMANDS attribute found")
            return False
            
        # Check BINDINGS  
        if hasattr(TldwCli, 'BINDINGS'):
            bindings = TldwCli.BINDINGS
            ctrl_p_bindings = [b for b in bindings if "ctrl+p" in str(b.key).lower()]
            
            if ctrl_p_bindings:
                print("✅ Ctrl+P binding found")
            else:
                print("❌ Ctrl+P binding not found")
                return False
        else:
            print("❌ No BINDINGS attribute found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ App configuration test failed: {e}")
        return False

def test_provider_structure():
    """Test that providers have required methods."""
    print("\n🔍 Testing provider structure...")
    
    try:
        from tldw_chatbook.app import ThemeProvider
        from unittest.mock import MagicMock
        
        # Create provider with mock screen
        mock_screen = MagicMock()
        provider = ThemeProvider(mock_screen)
        
        # Check required methods exist
        required_methods = ['search', 'discover', 'switch_theme', 'show_theme_submenu']
        for method_name in required_methods:
            if hasattr(provider, method_name):
                print(f"✅ {method_name} method found")
            else:
                print(f"❌ {method_name} method missing")
                return False
                
        # Check methods are callable
        for method_name in required_methods:
            method = getattr(provider, method_name)
            if callable(method):
                print(f"✅ {method_name} is callable")
            else:
                print(f"❌ {method_name} is not callable")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Provider structure test failed: {e}")
        return False

def test_constants():
    """Test that required constants are available."""
    print("\n🔍 Testing constants...")
    
    try:
        from tldw_chatbook.Constants import (
            TAB_CHAT, TAB_CCP, TAB_MEDIA, TAB_SEARCH,
            TAB_INGEST, TAB_TOOLS_SETTINGS, TAB_LLM, TAB_LOGS,
            TAB_STATS, TAB_EVALS, TAB_CODING, ALL_TABS
        )

        expected_tabs = [
            TAB_CHAT, TAB_CCP, TAB_MEDIA, TAB_SEARCH,
            TAB_INGEST, TAB_TOOLS_SETTINGS, TAB_LLM, TAB_LOGS,
            TAB_STATS, TAB_EVALS, TAB_CODING
        ]

        if len(expected_tabs) == 11:
            print("✅ All 11 tab constants found")
        else:
            print(f"❌ Expected 11 tabs, found {len(expected_tabs)}")
            return False

        if "notes" not in ALL_TABS:
            print("✅ ALL_TABS no longer carries the retired Notes tab")
        else:
            print("❌ ALL_TABS still carries the retired 'notes' tab id")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def test_theme_system():
    """Test that theme system is available."""
    print("\n🔍 Testing theme system...")
    
    try:
        from tldw_chatbook.css.Themes.themes import ALL_THEMES
        
        if ALL_THEMES and len(ALL_THEMES) > 0:
            print(f"✅ Found {len(ALL_THEMES)} themes available")
            
            # Check some themes have names
            named_themes = [t for t in ALL_THEMES if hasattr(t, 'name')]
            if named_themes:
                print(f"✅ {len(named_themes)} themes have names")
            else:
                print("⚠️ No themes have name attribute")
                
        else:
            print("❌ No themes found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Theme system test failed: {e}")
        return False

def test_config_integration():
    """Test that config system works for themes."""
    print("\n🔍 Testing config integration...")
    
    try:
        from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
        
        # Test reading a setting (should not crash)
        default_theme = get_cli_setting("general", "default_theme", "textual-dark")
        print(f"✅ Config reading works, default theme: {default_theme}")
        
        # Test that save function exists and is callable
        if callable(save_setting_to_cli_config):
            print("✅ Config saving function available")
        else:
            print("❌ Config saving function not callable")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Command Palette Verification Script")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_configuration, 
        test_provider_structure,
        test_constants,
        test_theme_system,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"💥 Test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All command palette verification tests passed!")
        print("\n📋 Manual testing steps:")
        print("1. Run: python -m tldw_chatbook.app")
        print("2. Press Ctrl+P to open command palette")
        print("3. Test theme switching, tab navigation, quick actions")
        print("4. Verify commands execute without errors")
        return True
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)