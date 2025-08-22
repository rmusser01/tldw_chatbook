#!/usr/bin/env python3
"""Test the fixed state preservation."""

from datetime import datetime
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState, MessageData


def test_validation_fix():
    """Test that validation now handles empty tab_order."""
    print("\n" + "="*60)
    print("Testing Fixed State Validation")
    print("="*60)
    
    # Create state with a tab but no tab_order (simulating non-tabbed interface)
    state = ChatScreenState()
    tab = TabState(
        tab_id="default",
        title="Chat",
        input_text="Hello world",
        is_active=True
    )
    state.tabs = [tab]
    state.active_tab_id = "default"
    # Intentionally leave tab_order empty to test auto-population
    
    print(f"Created state with tab but empty tab_order")
    print(f"Tabs: {[t.tab_id for t in state.tabs]}")
    print(f"Tab order before validation: {state.tab_order}")
    
    # Validate should now auto-populate tab_order
    is_valid = state.validate()
    
    print(f"Validation result: {is_valid}")
    print(f"Tab order after validation: {state.tab_order}")
    
    assert is_valid == True, "Validation should pass with auto-population"
    assert state.tab_order == ["default"], "Tab order should be auto-populated"
    
    print("✅ Validation fix working!")
    
    # Test serialization/deserialization preserves the fix
    state_dict = state.to_dict()
    restored = ChatScreenState.from_dict(state_dict)
    
    assert restored.validate() == True
    assert restored.tab_order == ["default"]
    
    print("✅ State survives serialization!")
    
    return True


def test_input_restoration():
    """Test that input text is properly saved and restored."""
    print("\n" + "="*60)
    print("Testing Input Text Restoration")
    print("="*60)
    
    # Create state with input text
    state = ChatScreenState()
    tab = TabState(
        tab_id="default",
        title="Chat",
        input_text="This is my important message that I don't want to lose",
        cursor_position=25,
        is_active=True
    )
    
    # Add some messages too
    tab.messages = [
        MessageData("msg1", "user", "Previous message", datetime.now()),
        MessageData("msg2", "assistant", "Response", datetime.now())
    ]
    
    state.tabs = [tab]
    state.active_tab_id = "default"
    state.tab_order = ["default"]  # Set it properly this time
    
    print(f"Original input: '{tab.input_text}'")
    print(f"Cursor position: {tab.cursor_position}")
    print(f"Messages: {len(tab.messages)}")
    
    # Serialize
    state_dict = state.to_dict()
    
    # Deserialize
    restored = ChatScreenState.from_dict(state_dict)
    
    # Validate restoration
    assert restored.validate() == True
    assert len(restored.tabs) == 1
    
    restored_tab = restored.tabs[0]
    assert restored_tab.input_text == "This is my important message that I don't want to lose"
    assert restored_tab.cursor_position == 25
    assert len(restored_tab.messages) == 2
    
    print(f"Restored input: '{restored_tab.input_text}'")
    print(f"Restored cursor: {restored_tab.cursor_position}")
    print(f"Restored messages: {len(restored_tab.messages)}")
    
    print("✅ Input text restoration working!")
    
    return True


def test_non_tabbed_state():
    """Test state for non-tabbed interface."""
    print("\n" + "="*60)
    print("Testing Non-Tabbed Interface State")
    print("="*60)
    
    # Simulate what _save_non_tabbed_state creates
    state = ChatScreenState()
    default_tab = TabState(
        tab_id="default",
        title="Chat",
        input_text="User was typing this",
        is_active=True
    )
    
    # Add messages
    default_tab.messages = [
        MessageData("msg1", "user", "Hello", datetime.now()),
        MessageData("msg2", "assistant", "Hi there!", datetime.now())
    ]
    
    state.tabs = [default_tab]
    state.active_tab_id = "default"
    state.tab_order = ["default"]  # This was missing before!
    
    print("Created non-tabbed state")
    print(f"Tab ID: {default_tab.tab_id}")
    print(f"Input: '{default_tab.input_text}'")
    print(f"Messages: {len(default_tab.messages)}")
    
    # Validate
    assert state.validate() == True
    print("✅ Validation passes")
    
    # Serialize and restore
    state_dict = state.to_dict()
    restored = ChatScreenState.from_dict(state_dict)
    
    assert restored.validate() == True
    assert restored.get_active_tab() is not None
    assert restored.get_active_tab().input_text == "User was typing this"
    assert len(restored.get_active_tab().messages) == 2
    
    print("✅ Non-tabbed state preservation working!")
    
    return True


def main():
    """Run all fixed state preservation tests."""
    print("="*60)
    print("Fixed State Preservation Test Suite")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_validation_fix()
        all_passed &= test_input_restoration()
        all_passed &= test_non_tabbed_state()
        
        if all_passed:
            print("\n" + "="*60)
            print("✅ ALL FIXED STATE TESTS PASSED!")
            print("="*60)
            print("\nThe fixes address:")
            print("1. Validation no longer fails on empty tab_order")
            print("2. Tab order is auto-populated for single tabs")
            print("3. Input text and messages are properly preserved")
            print("4. Non-tabbed interfaces work correctly")
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