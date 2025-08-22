#!/usr/bin/env python3
"""Test to understand the actual chat log structure."""

from tldw_chatbook.Utils.chat_diagnostics import ChatDiagnostics
from textual.widgets import Static, TextArea
from textual.containers import Container, Vertical

# Create a mock chat structure
class MockChatLog(Container):
    def __init__(self):
        super().__init__(id="chat-log")
        
class MockMainContent(Container):
    def __init__(self):
        super().__init__(id="chat-main-content")

# Test different scenarios
def test_log_finding():
    print("\n=== Testing Chat Log Finding ===\n")
    
    # Scenario 1: Direct chat-log
    mock_window = Container()
    chat_log = MockChatLog()
    mock_window._add_child(chat_log)
    
    # Try to find it
    found = None
    for child in mock_window.children:
        if hasattr(child, 'id') and child.id == "chat-log":
            found = child
            break
    
    if found:
        print("✓ Found chat-log by ID")
    else:
        print("✗ Could not find chat-log")
    
    # Scenario 2: Nested in main-content
    mock_window2 = Container()
    main_content = MockMainContent()
    chat_log2 = MockChatLog()
    main_content._add_child(chat_log2)
    mock_window2._add_child(main_content)
    
    # Try to find it
    found2 = None
    for child in mock_window2.children:
        if hasattr(child, 'id') and child.id == "chat-main-content":
            # Look inside main content
            for subchild in child.children:
                if hasattr(subchild, 'id') and subchild.id == "chat-log":
                    found2 = subchild
                    break
    
    if found2:
        print("✓ Found nested chat-log")
    else:
        print("✗ Could not find nested chat-log")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_log_finding()