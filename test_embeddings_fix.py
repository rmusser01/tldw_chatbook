#!/usr/bin/env python3
"""Test embeddings download button fix."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Check if the worker method exists and is not duplicated
def test_worker_method():
    """Check that _download_model_worker exists and is not duplicated."""
    print("Test 1: Checking _download_model_worker method...")
    
    with open("tldw_chatbook/UI/Embeddings_Management_Window.py", "r") as f:
        content = f.read()
    
    # Count occurrences of the method definition
    count = content.count("def _download_model_worker(")
    print(f"  Found {count} definition(s) of _download_model_worker")
    
    if count == 1:
        print("  ✓ PASS: Only one definition found")
        return True
    else:
        print("  ✗ FAIL: Expected 1 definition, found", count)
        return False

# Test 2: Check button handler registration
def test_button_handler():
    """Check that the download button handler is properly registered."""
    print("\nTest 2: Checking button handler registration...")
    
    with open("tldw_chatbook/UI/Embeddings_Management_Window.py", "r") as f:
        content = f.read()
    
    # Check for the button handler
    handler_found = '@on(Button.Pressed, "#embeddings-download-model")' in content
    print(f"  Button handler decorated: {handler_found}")
    
    # Check that the handler is synchronous (not async)
    if handler_found:
        # Find the handler method
        handler_start = content.find('@on(Button.Pressed, "#embeddings-download-model")')
        handler_def = content.find('def on_download_model(', handler_start)
        if handler_def > handler_start:
            # Check if it's async
            async_check = content[handler_def-10:handler_def]
            is_async = 'async ' in async_check
            print(f"  Handler is async: {is_async}")
            
            if not is_async:
                print("  ✓ PASS: Button handler is synchronous")
                return True
            else:
                print("  ✗ FAIL: Button handler should not be async")
                return False
    
    print("  ✗ FAIL: Button handler not found")
    return False

# Test 3: Check worker call
def test_worker_call():
    """Check that the worker is called correctly."""
    print("\nTest 3: Checking worker call...")
    
    with open("tldw_chatbook/UI/Embeddings_Management_Window.py", "r") as f:
        content = f.read()
    
    # Check for the correct worker call
    correct_call = 'self.run_worker(\n                self._download_model_worker,' in content
    print(f"  Correct worker call found: {correct_call}")
    
    if correct_call:
        print("  ✓ PASS: Worker is called correctly")
        return True
    else:
        print("  ✗ FAIL: Worker call not found or incorrect")
        return False

# Test 4: Check message handling
def test_message_handling():
    """Check that message handlers are properly defined."""
    print("\nTest 4: Checking message handling...")
    
    with open("tldw_chatbook/UI/Embeddings_Management_Window.py", "r") as f:
        content = f.read()
    
    # Check for message classes
    has_download_status = 'class DownloadStatusMessage(Message):' in content
    has_set_loading = 'class SetLoadingMessage(Message):' in content
    
    # Check for message handlers
    has_download_handler = '@on(DownloadStatusMessage)' in content
    has_loading_handler = '@on(SetLoadingMessage)' in content
    
    print(f"  DownloadStatusMessage class: {has_download_status}")
    print(f"  SetLoadingMessage class: {has_set_loading}")
    print(f"  Download status handler: {has_download_handler}")
    print(f"  Loading handler: {has_loading_handler}")
    
    if all([has_download_status, has_set_loading, has_download_handler, has_loading_handler]):
        print("  ✓ PASS: All message handling components found")
        return True
    else:
        print("  ✗ FAIL: Missing message handling components")
        return False

# Run all tests
def main():
    """Run all tests."""
    print("Testing Embeddings Download Button Fix")
    print("=" * 50)
    
    tests = [
        test_worker_method,
        test_button_handler,
        test_worker_call,
        test_message_handling
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("\n✓ All tests passed! The embeddings download button should work.")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)