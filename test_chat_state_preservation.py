#!/usr/bin/env python3
"""Test script for chat screen state preservation."""

import asyncio
from datetime import datetime
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState, MessageData


def test_state_serialization():
    """Test that state can be serialized and deserialized correctly."""
    print("Testing state serialization...")
    
    # Create a test state
    state = ChatScreenState()
    
    # Add some tabs
    tab1 = TabState(
        tab_id="abc12345",
        title="Chat with GPT",
        input_text="Hello, how are you?",
        cursor_position=18,
        scroll_position=100,
        is_active=True
    )
    
    tab2 = TabState(
        tab_id="def67890",
        title="Code Review",
        input_text="Can you review this code?",
        cursor_position=25,
        scroll_position=0,
        is_active=False
    )
    
    state.tabs = [tab1, tab2]
    state.active_tab_id = "abc12345"
    state.tab_order = ["abc12345", "def67890"]
    state.left_sidebar_collapsed = True
    state.right_sidebar_collapsed = False
    state.last_saved = datetime.now()
    
    # Add some messages to tab1
    msg1 = MessageData(
        message_id="msg1",
        role="user",
        content="Hello!",
        timestamp=datetime.now()
    )
    msg2 = MessageData(
        message_id="msg2",
        role="assistant",
        content="Hi there! How can I help you today?",
        timestamp=datetime.now()
    )
    tab1.messages = [msg1, msg2]
    
    # Serialize to dict
    state_dict = state.to_dict()
    print(f"✓ Serialized state with {len(state.tabs)} tabs")
    
    # Deserialize from dict
    restored_state = ChatScreenState.from_dict(state_dict)
    print(f"✓ Deserialized state with {len(restored_state.tabs)} tabs")
    
    # Validate
    assert len(restored_state.tabs) == 2
    assert restored_state.active_tab_id == "abc12345"
    assert restored_state.tabs[0].input_text == "Hello, how are you?"
    assert restored_state.tabs[0].messages[0].content == "Hello!"
    assert restored_state.left_sidebar_collapsed == True
    print("✓ State validation passed")
    
    # Test validation method
    assert restored_state.validate() == True
    print("✓ State consistency check passed")
    
    # Test getting active tab
    active_tab = restored_state.get_active_tab()
    assert active_tab is not None
    assert active_tab.tab_id == "abc12345"
    print("✓ Active tab retrieval works")
    
    print("\n✅ All state serialization tests passed!")
    return True


def test_tab_operations():
    """Test tab management operations."""
    print("\nTesting tab operations...")
    
    state = ChatScreenState()
    
    # Add tabs
    tab1 = TabState(tab_id="tab1", title="Tab 1")
    tab2 = TabState(tab_id="tab2", title="Tab 2")
    
    state.add_tab(tab1)
    state.add_tab(tab2)
    assert len(state.tabs) == 2
    assert len(state.tab_order) == 2
    print("✓ Tab addition works")
    
    # Set active tab
    state.active_tab_id = "tab1"
    active = state.get_active_tab()
    assert active.tab_id == "tab1"
    print("✓ Active tab management works")
    
    # Remove tab
    assert state.remove_tab("tab2") == True
    assert len(state.tabs) == 1
    assert "tab2" not in state.tab_order
    print("✓ Tab removal works")
    
    # Update tab order
    state.add_tab(tab2)
    state.update_tab_order(["tab2", "tab1"])
    assert state.tab_order == ["tab2", "tab1"]
    print("✓ Tab order update works")
    
    print("\n✅ All tab operation tests passed!")
    return True


def test_message_data():
    """Test message data serialization."""
    print("\nTesting message data...")
    
    msg = MessageData(
        message_id="test123",
        role="user",
        content="Test message",
        timestamp=datetime.now(),
        attachments=[{"type": "image", "path": "/tmp/image.png"}],
        metadata={"edited": True}
    )
    
    # Serialize
    msg_dict = msg.to_dict()
    assert msg_dict["message_id"] == "test123"
    assert msg_dict["role"] == "user"
    print("✓ Message serialization works")
    
    # Deserialize
    restored_msg = MessageData.from_dict(msg_dict)
    assert restored_msg.message_id == "test123"
    assert restored_msg.attachments[0]["type"] == "image"
    assert restored_msg.metadata["edited"] == True
    print("✓ Message deserialization works")
    
    print("\n✅ All message data tests passed!")
    return True


def test_state_snapshot():
    """Test state snapshot creation."""
    print("\nTesting state snapshot...")
    
    original = ChatScreenState()
    tab = TabState(tab_id="test", title="Test Tab", input_text="Original text")
    original.add_tab(tab)
    original.active_tab_id = "test"
    
    # Create snapshot
    snapshot = original.create_snapshot()
    
    # Modify original
    original.tabs[0].input_text = "Modified text"
    
    # Check snapshot is unchanged
    assert snapshot.tabs[0].input_text == "Original text"
    assert original.tabs[0].input_text == "Modified text"
    print("✓ Snapshot creates independent copy")
    
    print("\n✅ Snapshot test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Chat Screen State Preservation Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    try:
        all_passed &= test_state_serialization()
        all_passed &= test_tab_operations()
        all_passed &= test_message_data()
        all_passed &= test_state_snapshot()
        
        if all_passed:
            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED SUCCESSFULLY!")
            print("=" * 50)
            print("\nThe chat state preservation system is working correctly.")
            print("Users can now navigate away from chat and return without losing their work.")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())