#!/usr/bin/env python3
"""
Test script to verify tldw_chatbook package imports correctly.

This script should be run after installing the package to ensure
all core modules can be imported without errors.
"""

import sys
import traceback
from importlib import import_module

# List of core modules that should be importable
CORE_MODULES = [
    "tldw_chatbook",
    "tldw_chatbook.app",
    "tldw_chatbook.config",
    "tldw_chatbook.Constants",
    "tldw_chatbook.Chat.Chat_Functions",
    "tldw_chatbook.UI.Chat_Window",
    "tldw_chatbook.DB.ChaChaNotes_DB",
    "tldw_chatbook.Utils.Utils",
    "tldw_chatbook.Widgets.chat_message_enhanced",
    "tldw_chatbook.LLM_Calls.LLM_API_Calls",
]

# Optional modules that may fail due to missing dependencies
OPTIONAL_MODULES = [
    "tldw_chatbook.RAG_Search.chunking_service",
    "tldw_chatbook.Embeddings.Embeddings_Lib",
    "tldw_chatbook.Local_Ingestion.transcription_service",
    "tldw_chatbook.TTS.TTS_Generation",
]


def test_imports():
    """Test importing core and optional modules."""
    print("Testing tldw_chatbook package imports...")
    print("=" * 60)
    
    # Test version info
    try:
        import tldw_chatbook
        print(f"✓ Package version: {tldw_chatbook.__version__}")
        print(f"✓ Author: {tldw_chatbook.__author__}")
        print(f"✓ License: {tldw_chatbook.__license__}")
    except Exception as e:
        print(f"✗ Failed to import main package: {e}")
        return False
    
    print("\nTesting core modules:")
    print("-" * 40)
    
    failed_core = []
    for module_name in CORE_MODULES:
        try:
            module = import_module(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {type(e).__name__}: {e}")
            failed_core.append(module_name)
    
    print("\nTesting optional modules:")
    print("-" * 40)
    
    failed_optional = []
    for module_name in OPTIONAL_MODULES:
        try:
            module = import_module(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"⚠ {module_name}: Missing optional dependency")
            failed_optional.append(module_name)
        except Exception as e:
            print(f"✗ {module_name}: {type(e).__name__}: {e}")
            failed_optional.append(module_name)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    if not failed_core:
        print("✓ All core modules imported successfully!")
    else:
        print(f"✗ {len(failed_core)} core modules failed to import:")
        for module in failed_core:
            print(f"  - {module}")
    
    if failed_optional:
        print(f"⚠ {len(failed_optional)} optional modules require additional dependencies")
    
    # Test CLI entry point
    print("\nTesting CLI entry point:")
    print("-" * 40)
    try:
        from tldw_chatbook.app import main_cli_runner
        print("✓ CLI entry point found: tldw_chatbook.app:main_cli_runner")
    except Exception as e:
        print(f"✗ CLI entry point error: {e}")
        return False
    
    return len(failed_core) == 0


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)