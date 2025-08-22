#!/usr/bin/env python3
"""
Complete test of state preservation flow following Textual best practices.

This test validates:
1. State saving when navigating away
2. State restoration when returning
3. Message preservation
4. Input text preservation
5. UI state preservation
"""

from datetime import datetime
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState, MessageData


def test_complete_flow():
    """Test the complete state preservation flow."""
    print("\n" + "="*60)
    print("Complete State Preservation Flow Test")
    print("="*60)
    
    # Step 1: Create initial state (simulating user interaction)
    print("\n1. Creating initial state with user data...")
    
    initial_state = ChatScreenState()
    
    # Create a tab with conversation
    tab = TabState(
        tab_id="default",
        title="Python Help",
        input_text="How do I handle exceptions in async functions?",
        cursor_position=45,
        is_active=True
    )
    
    # Add conversation history
    tab.messages = [
        MessageData(
            message_id="msg_1",
            role="user",
            content="Hello, I need help with Python async/await",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg_2",
            role="assistant",
            content="I'll be happy to help you with Python async/await! This is a powerful feature for concurrent programming. What specific aspect would you like to explore?",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg_3",
            role="user",
            content="How do error handling and try/except work with async functions?",
            timestamp=datetime.now()
        ),
        MessageData(
            message_id="msg_4",
            role="assistant",
            content="Great question! Error handling in async functions works similarly to regular functions, but with some important considerations...",
            timestamp=datetime.now(),
            metadata={"truncated": True}  # Simulating a longer response
        )
    ]
    
    # Add UI state
    initial_state.tabs = [tab]
    initial_state.active_tab_id = "default"
    initial_state.tab_order = ["default"]
    initial_state.left_sidebar_collapsed = True
    initial_state.right_sidebar_collapsed = False
    initial_state.show_timestamps = True
    initial_state.last_saved = datetime.now()
    
    print(f"  - Created tab: {tab.title}")
    print(f"  - Added {len(tab.messages)} messages")
    print(f"  - Input text: '{tab.input_text}'")
    print(f"  - UI state: left_sidebar={initial_state.left_sidebar_collapsed}, timestamps={initial_state.show_timestamps}")
    
    # Step 2: Validate the state
    print("\n2. Validating state consistency...")
    
    if not initial_state.validate():
        print("  ❌ State validation failed!")
        return False
    
    print("  ✅ State validation passed")
    
    # Step 3: Serialize state (simulating navigation away)
    print("\n3. Serializing state (user navigates away)...")
    
    state_dict = initial_state.to_dict()
    
    print(f"  - Serialized to dictionary with {len(state_dict)} keys")
    print(f"  - State version: {state_dict.get('version')}")
    print(f"  - Preserved tabs: {len(state_dict.get('tabs', []))}")
    
    # Verify critical data is in dict
    assert 'tabs' in state_dict
    assert 'active_tab_id' in state_dict
    assert 'tab_order' in state_dict
    assert len(state_dict['tabs']) == 1
    assert state_dict['tabs'][0]['input_text'] == "How do I handle exceptions in async functions?"
    assert len(state_dict['tabs'][0]['messages']) == 4
    
    print("  ✅ All critical data preserved in serialization")
    
    # Step 4: Deserialize state (simulating return to screen)
    print("\n4. Deserializing state (user returns)...")
    
    restored_state = ChatScreenState.from_dict(state_dict)
    
    print(f"  - Restored {len(restored_state.tabs)} tabs")
    print(f"  - Active tab: {restored_state.active_tab_id}")
    print(f"  - Tab order: {restored_state.tab_order}")
    
    # Step 5: Validate restored state
    print("\n5. Validating restored state...")
    
    if not restored_state.validate():
        print("  ❌ Restored state validation failed!")
        return False
    
    print("  ✅ Restored state validation passed")
    
    # Step 6: Verify all data was preserved
    print("\n6. Verifying data integrity...")
    
    restored_tab = restored_state.get_active_tab()
    
    if not restored_tab:
        print("  ❌ No active tab found!")
        return False
    
    checks = [
        ("Tab ID", restored_tab.tab_id == "default"),
        ("Tab title", restored_tab.title == "Python Help"),
        ("Input text", restored_tab.input_text == "How do I handle exceptions in async functions?"),
        ("Cursor position", restored_tab.cursor_position == 45),
        ("Message count", len(restored_tab.messages) == 4),
        ("First message", restored_tab.messages[0].content == "Hello, I need help with Python async/await"),
        ("Last message role", restored_tab.messages[-1].role == "assistant"),
        ("Message metadata", restored_tab.messages[-1].metadata.get("truncated") == True),
        ("Left sidebar", restored_state.left_sidebar_collapsed == True),
        ("Right sidebar", restored_state.right_sidebar_collapsed == False),
        ("Show timestamps", restored_state.show_timestamps == True),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        all_passed &= result
    
    # Step 7: Test state modification and re-save
    print("\n7. Testing state modification...")
    
    # Simulate user continuing to type
    restored_tab.input_text += " Should I use try/except or try/finally?"
    restored_tab.cursor_position = len(restored_tab.input_text)
    
    # Add a new message
    restored_tab.messages.append(
        MessageData(
            message_id="msg_5",
            role="user",
            content=restored_tab.input_text,
            timestamp=datetime.now()
        )
    )
    
    print(f"  - Modified input text to: '{restored_tab.input_text[:50]}...'")
    print(f"  - Added new message (total: {len(restored_tab.messages)})")
    
    # Re-serialize
    modified_dict = restored_state.to_dict()
    re_restored = ChatScreenState.from_dict(modified_dict)
    
    if re_restored.validate() and len(re_restored.tabs[0].messages) == 5:
        print("  ✅ Modifications preserved correctly")
    else:
        print("  ❌ Modifications not preserved")
        all_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ COMPLETE STATE FLOW TEST PASSED!")
        print("\nSummary:")
        print("- State validation with auto-population works")
        print("- Full conversation history preserved")
        print("- Input text and cursor position maintained")
        print("- UI preferences saved and restored")
        print("- State modifications handled correctly")
        print("\nThe state preservation system is working correctly!")
    else:
        print("❌ Some checks failed")
    print("="*60)
    
    return all_passed


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Empty state
    print("\n1. Empty state handling...")
    empty_state = ChatScreenState()
    if empty_state.validate():
        print("  ✅ Empty state validates")
    else:
        print("  ❌ Empty state fails validation")
        all_passed = False
    
    # Test 2: State with no input text
    print("\n2. State with no input text...")
    state_no_input = ChatScreenState()
    tab_no_input = TabState(tab_id="test", title="Test", input_text="", is_active=True)
    state_no_input.tabs = [tab_no_input]
    state_no_input.active_tab_id = "test"
    
    dict_no_input = state_no_input.to_dict()
    restored_no_input = ChatScreenState.from_dict(dict_no_input)
    
    if restored_no_input.validate() and restored_no_input.tabs[0].input_text == "":
        print("  ✅ Empty input text preserved")
    else:
        print("  ❌ Empty input text not preserved")
        all_passed = False
    
    # Test 3: Very long input text
    print("\n3. Very long input text...")
    long_text = "x" * 10000
    state_long = ChatScreenState()
    tab_long = TabState(tab_id="long", title="Long", input_text=long_text, is_active=True)
    state_long.tabs = [tab_long]
    state_long.active_tab_id = "long"
    
    dict_long = state_long.to_dict()
    restored_long = ChatScreenState.from_dict(dict_long)
    
    if restored_long.tabs[0].input_text == long_text:
        print(f"  ✅ Long text preserved ({len(long_text)} chars)")
    else:
        print("  ❌ Long text not preserved")
        all_passed = False
    
    # Test 4: Unicode and special characters
    print("\n4. Unicode and special characters...")
    unicode_text = "Hello 世界! 🎉 Special chars: <>&\"'\n\ttab"
    state_unicode = ChatScreenState()
    tab_unicode = TabState(tab_id="unicode", title="Unicode", input_text=unicode_text, is_active=True)
    
    # Add message with unicode
    tab_unicode.messages = [
        MessageData("u1", "user", "Question with emoji 🤔", datetime.now()),
        MessageData("u2", "assistant", "Response with 中文 and émojis 🎯", datetime.now())
    ]
    
    state_unicode.tabs = [tab_unicode]
    state_unicode.active_tab_id = "unicode"
    
    dict_unicode = state_unicode.to_dict()
    restored_unicode = ChatScreenState.from_dict(dict_unicode)
    
    if (restored_unicode.tabs[0].input_text == unicode_text and 
        restored_unicode.tabs[0].messages[0].content == "Question with emoji 🤔"):
        print("  ✅ Unicode preserved correctly")
    else:
        print("  ❌ Unicode not preserved")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL EDGE CASES PASSED!")
    else:
        print("❌ Some edge cases failed")
    print("="*60)
    
    return all_passed


def main():
    """Run all tests."""
    print("="*60)
    print("Chat State Preservation Test Suite")
    print("Following Textual Best Practices")
    print("="*60)
    
    flow_passed = test_complete_flow()
    edge_passed = test_edge_cases()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Complete Flow Test: {'✅ PASSED' if flow_passed else '❌ FAILED'}")
    print(f"Edge Cases Test: {'✅ PASSED' if edge_passed else '❌ FAILED'}")
    
    if flow_passed and edge_passed:
        print("\n🎉 ALL TESTS PASSED! State preservation is working correctly.")
        print("\nThe implementation follows Textual best practices:")
        print("- Uses dataclasses for state management")
        print("- Implements proper serialization/deserialization")
        print("- Validates state consistency")
        print("- Handles edge cases gracefully")
        print("- Preserves all user data accurately")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())