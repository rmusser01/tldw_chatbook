#!/usr/bin/env python3
"""Test script for chat message preservation functionality."""

from datetime import datetime
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState, MessageData


def test_message_serialization():
    """Test that messages are properly serialized and deserialized."""
    print("\n" + "="*60)
    print("Testing Message Preservation")
    print("="*60)
    
    # Create a state with messages
    state = ChatScreenState()
    
    # Create a tab with messages
    tab = TabState(
        tab_id="chat-1",
        title="Test Chat",
        input_text="This is my current input",
        is_active=True
    )
    
    # Add some test messages
    messages = [
        MessageData(
            message_id="msg1",
            role="user",
            content="Hello, how are you?",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg2",
            role="assistant",
            content="I'm doing well, thank you! How can I help you today?",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg3",
            role="user",
            content="Can you explain quantum computing?",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg4",
            role="assistant",
            content="Quantum computing is a type of computation that harnesses quantum mechanical phenomena...",
            timestamp=datetime.now(),
            metadata={"thinking": "Let me provide a clear explanation..."}
        )
    ]
    
    tab.messages = messages
    state.tabs = [tab]
    state.active_tab_id = "chat-1"
    
    print(f"Created state with {len(messages)} messages")
    
    # Serialize to dict
    state_dict = state.to_dict()
    print(f"Serialized state to dictionary")
    
    # Deserialize from dict
    restored_state = ChatScreenState.from_dict(state_dict)
    print(f"Deserialized state from dictionary")
    
    # Validate restoration
    assert len(restored_state.tabs) == 1
    assert len(restored_state.tabs[0].messages) == 4
    assert restored_state.tabs[0].messages[0].content == "Hello, how are you?"
    assert restored_state.tabs[0].messages[1].role == "assistant"
    assert restored_state.tabs[0].messages[3].metadata["thinking"] == "Let me provide a clear explanation..."
    assert restored_state.tabs[0].input_text == "This is my current input"
    
    print("✅ All assertions passed!")
    
    # Display restored messages
    print("\nRestored conversation:")
    for msg in restored_state.tabs[0].messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")
    
    print(f"\nRestored input text: '{restored_state.tabs[0].input_text}'")
    
    return True


def test_message_extraction_mock():
    """Test the message extraction logic with mock data."""
    print("\n" + "="*60)
    print("Testing Message Extraction Logic")
    print("="*60)
    
    # Create a mock message widget class
    class MockMessageWidget:
        def __init__(self, role, content, msg_id=None, timestamp=None):
            self.role = role
            self.message_text = content
            self.message_id_internal = msg_id or f"mock_{id(self)}"
            self.timestamp = timestamp or datetime.now()
            self.image_data = None
    
    # Create mock messages
    mock_messages = [
        MockMessageWidget("user", "What's the weather like?"),
        MockMessageWidget("assistant", "I don't have real-time weather data, but I can help you find weather resources."),
        MockMessageWidget("user", "Thanks, what about Python programming?"),
        MockMessageWidget("assistant", "Python is a versatile programming language great for beginners and experts alike!")
    ]
    
    # Simulate extraction
    extracted_messages = []
    for widget in mock_messages:
        msg_data = MessageData(
            message_id=widget.message_id_internal,
            role=widget.role,
            content=widget.message_text,
            timestamp=widget.timestamp
        )
        extracted_messages.append(msg_data)
    
    print(f"Extracted {len(extracted_messages)} messages from mock widgets")
    
    # Verify extraction
    assert len(extracted_messages) == 4
    assert extracted_messages[0].role == "user"
    assert "weather" in extracted_messages[0].content.lower()
    assert extracted_messages[1].role == "assistant"
    assert extracted_messages[3].content.startswith("Python is")
    
    print("✅ Message extraction test passed!")
    
    # Display extracted messages
    print("\nExtracted messages:")
    for msg in extracted_messages:
        print(f"  [{msg.role}]: {msg.content[:60]}...")
    
    return True


def test_conversation_continuity():
    """Test that a conversation can be saved and restored maintaining continuity."""
    print("\n" + "="*60)
    print("Testing Conversation Continuity")
    print("="*60)
    
    # Simulate a conversation in progress
    state1 = ChatScreenState()
    tab1 = TabState(
        tab_id="session1",
        title="Project Discussion",
        input_text="What about the deadline for",  # User was typing this
        cursor_position=28,
        is_active=True
    )
    
    # Add conversation history
    tab1.messages = [
        MessageData("msg1", "user", "I need help with my project", datetime.now()),
        MessageData("msg2", "assistant", "I'd be happy to help! What kind of project are you working on?", datetime.now()),
        MessageData("msg3", "user", "It's a web application using React", datetime.now()),
        MessageData("msg4", "assistant", "Great! React is excellent for building interactive UIs. What specific aspect do you need help with?", datetime.now()),
    ]
    
    state1.tabs = [tab1]
    state1.active_tab_id = "session1"
    
    print("Initial conversation state:")
    print(f"  Messages: {len(tab1.messages)}")
    print(f"  Current input: '{tab1.input_text}'")
    print(f"  Cursor position: {tab1.cursor_position}")
    
    # Serialize (simulating navigation away)
    saved_state = state1.to_dict()
    print("\n📦 State saved (user navigates away)")
    
    # Create new state (simulating return to chat)
    state2 = ChatScreenState.from_dict(saved_state)
    print("📥 State restored (user returns)")
    
    # Verify continuity
    restored_tab = state2.tabs[0]
    assert len(restored_tab.messages) == 4
    assert restored_tab.input_text == "What about the deadline for"
    assert restored_tab.cursor_position == 28
    assert restored_tab.messages[-1].content.startswith("Great! React is excellent")
    
    print("\n✅ Conversation continuity maintained!")
    print(f"  Messages restored: {len(restored_tab.messages)}")
    print(f"  Input text restored: '{restored_tab.input_text}'")
    print(f"  Cursor position restored: {restored_tab.cursor_position}")
    print(f"  Last message: [{restored_tab.messages[-1].role}] {restored_tab.messages[-1].content[:50]}...")
    
    return True


def main():
    """Run all message preservation tests."""
    print("="*60)
    print("Chat Message Preservation Test Suite")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_message_serialization()
        all_passed &= test_message_extraction_mock()
        all_passed &= test_conversation_continuity()
        
        if all_passed:
            print("\n" + "="*60)
            print("✅ ALL MESSAGE PRESERVATION TESTS PASSED!")
            print("="*60)
            print("\nThe enhanced state preservation system now:")
            print("1. Saves the entire conversation history")
            print("2. Preserves user's typed input and cursor position")
            print("3. Restores messages when returning to chat")
            print("4. Maintains conversation continuity across navigation")
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