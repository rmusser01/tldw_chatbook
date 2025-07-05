#!/usr/bin/env python3
"""Test script to verify startup optimizations are working"""

import time
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def measure_import_time():
    """Measure time to import the app module"""
    start_time = time.perf_counter()
    
    # Import config first to measure DB init time
    config_start = time.perf_counter()
    from tldw_chatbook import config
    config_time = time.perf_counter() - config_start
    print(f"Config import time: {config_time:.3f}s")
    
    # Now import app
    app_start = time.perf_counter()
    from tldw_chatbook.app import TldwCli
    app_time = time.perf_counter() - app_start
    print(f"App import time: {app_time:.3f}s")
    
    total_time = time.perf_counter() - start_time
    print(f"Total import time: {total_time:.3f}s")
    
    return total_time

def test_lazy_loading():
    """Test that databases are not initialized until accessed"""
    from tldw_chatbook import config
    
    print("\nTesting lazy database loading:")
    print(f"chachanotes_db is None: {config.chachanotes_db is None}")
    print(f"prompts_db is None: {config.prompts_db is None}")
    print(f"media_db is None: {config.media_db is None}")
    
    # Access one database
    print("\nAccessing chachanotes_db...")
    start = time.perf_counter()
    db = config.get_chachanotes_db_lazy()
    load_time = time.perf_counter() - start
    print(f"First access took: {load_time:.3f}s")
    print(f"DB is None: {db is None}")
    
    # Second access should be instant
    start = time.perf_counter()
    db2 = config.get_chachanotes_db_lazy()
    cached_time = time.perf_counter() - start
    print(f"Second access took: {cached_time:.3f}s (should be ~0)")

def test_window_creation():
    """Test that only initial window is created"""
    print("\nTesting window creation:")
    from tldw_chatbook.app import TldwCli
    
    app = TldwCli()
    print(f"Initial tab: {app._initial_tab_value}")
    print(f"Window mapping exists: {hasattr(app, '_window_mapping')}")
    if hasattr(app, '_created_windows'):
        print(f"Created windows: {app._created_windows}")
    else:
        print("_created_windows not set (app not composed yet)")

if __name__ == "__main__":
    print("=== Testing Startup Optimizations ===\n")
    
    # Test 1: Import times
    import_time = measure_import_time()
    
    # Test 2: Lazy database loading
    test_lazy_loading()
    
    # Test 3: Window creation
    test_window_creation()
    
    print("\n=== Tests Complete ===")
    print(f"\nIf import time is < 1s and databases load lazily, optimizations are working!")