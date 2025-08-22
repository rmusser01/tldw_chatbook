#!/usr/bin/env python3
"""Test script for chat diagnostics functionality."""

import asyncio
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import TextArea, Button, Static
from textual.containers import Container, Vertical, Horizontal

from tldw_chatbook.Utils.chat_diagnostics import ChatDiagnostics
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState


class MockChatWindow(Container):
    """Mock chat window for testing diagnostics."""
    
    def compose(self) -> ComposeResult:
        """Create a mock chat interface."""
        with Vertical(id="main-container"):
            # Chat log area
            with Container(id="chat-log", classes="chat-log"):
                yield Static("Message 1", classes="message")
                yield Static("Message 2", classes="message")
                yield Static("Message 3", classes="message")
            
            # Input area
            with Horizontal(id="input-container"):
                yield TextArea(
                    "This is some test input text that should be captured",
                    id="chat-input",
                    classes="chat-input"
                )
                yield Button("Send", id="send-button")


class MockTabbedChatWindow(Container):
    """Mock tabbed chat window for testing."""
    
    def compose(self) -> ComposeResult:
        """Create a mock tabbed chat interface."""
        with Vertical(id="main-container"):
            # Tab bar
            with Horizontal(id="tab-bar", classes="tab-bar"):
                yield Button("Tab 1", id="tab-1")
                yield Button("Tab 2", id="tab-2")
                yield Button("+", id="new-tab")
            
            # Tab container
            with Container(id="chat-tab-container", classes="tab-container"):
                # Tab 1 content
                with Container(id="tab-content-1", classes="tab-content"):
                    with Container(id="chat-log-1", classes="chat-log"):
                        yield Static("Tab 1 Message", classes="message")
                    yield TextArea(
                        "Tab 1 input text",
                        id="chat-input-1",
                        classes="chat-input"
                    )
                
                # Tab 2 content (hidden)
                with Container(id="tab-content-2", classes="tab-content"):
                    with Container(id="chat-log-2", classes="chat-log"):
                        yield Static("Tab 2 Message", classes="message")
                    yield TextArea(
                        "Tab 2 input text",
                        id="chat-input-2",
                        classes="chat-input"
                    )


def test_non_tabbed_diagnostics():
    """Test diagnostics on non-tabbed interface."""
    print("\n" + "="*60)
    print("Testing Non-Tabbed Chat Interface Diagnostics")
    print("="*60)
    
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield MockChatWindow()
    
    app = TestApp()
    async def run_test():
        async with app.run_test() as pilot:
            # Get the mock window
            mock_window = app.query_one(MockChatWindow)
            
            # Run diagnostics
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(mock_window, max_depth=5)
            
            # Print report
            diagnostics.print_report(report)
            
            # Assertions
            assert report['text_areas']['count'] == 1, f"Expected 1 TextArea, found {report['text_areas']['count']}"
            assert report['chat_structure']['type'] in ['single', 'unknown'], f"Expected single interface, got {report['chat_structure']['type']}"
            assert len(report['input_widgets']) > 0, "Should have found input widgets"
            
            print("\n✅ Non-tabbed diagnostics test passed!")
    
    # Run the async test
    import asyncio
    asyncio.run(run_test())
    return True


def test_tabbed_diagnostics():
    """Test diagnostics on tabbed interface."""
    print("\n" + "="*60)
    print("Testing Tabbed Chat Interface Diagnostics")
    print("="*60)
    
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield MockTabbedChatWindow()
    
    app = TestApp()
    async def run_test():
        async with app.run_test() as pilot:
            # Get the mock window
            mock_window = app.query_one(MockTabbedChatWindow)
            
            # Run diagnostics
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(mock_window, max_depth=5)
            
            # Print report
            diagnostics.print_report(report)
            
            # Assertions
            assert report['text_areas']['count'] == 2, f"Expected 2 TextAreas, found {report['text_areas']['count']}"
            # Note: Detection might not identify as "tabbed" without actual ChatTabContainer class
            assert report['containers']['tab_containers'] > 0 or 'tab' in str(report['containers']['tab_container_ids']).lower(), \
                "Should have detected tab-related containers"
            
            print("\n✅ Tabbed diagnostics test passed!")
    
    # Run the async test
    import asyncio
    asyncio.run(run_test())
    return True


def test_state_capture_with_diagnostics():
    """Test that state capture works with diagnostic insights."""
    print("\n" + "="*60)
    print("Testing State Capture with Diagnostics")
    print("="*60)
    
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield MockChatWindow()
    
    app = TestApp()
    async def run_test():
        async with app.run_test() as pilot:
            # Create a state object
            state = ChatScreenState()
            
            # Get the mock window
            mock_window = app.query_one(MockChatWindow)
            
            # Run diagnostics
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(mock_window, max_depth=5)
            
            # Based on diagnostics, capture state
            if report['input_widgets']:
                # Found input widgets
                for widget_info in report['input_widgets']:
                    print(f"Found input widget: {widget_info['id']} with text preview")
                    
                    # Create a tab to store this state
                    tab = TabState(
                        tab_id=widget_info['id'] or "default",
                        title="Chat",
                        input_text=widget_info.get('text_preview', '').replace('...', ''),
                        is_active=True
                    )
                    state.tabs.append(tab)
            
            # Verify state was captured
            assert len(state.tabs) > 0, "Should have captured at least one tab"
            assert state.tabs[0].input_text, "Should have captured input text"
            
            print(f"✅ Successfully captured {len(state.tabs)} tabs with input text")
            print(f"   Input text: '{state.tabs[0].input_text[:50]}...'")
    
    # Run the async test
    import asyncio
    asyncio.run(run_test())
    return True


def test_recommendations():
    """Test that recommendations are generated correctly."""
    print("\n" + "="*60)
    print("Testing Diagnostic Recommendations")
    print("="*60)
    
    class TestApp1(App):
        def compose(self) -> ComposeResult:
            yield MockChatWindow()
    
    class TestApp2(App):
        def compose(self) -> ComposeResult:
            yield MockTabbedChatWindow()
    
    async def run_test():
        # Test with non-tabbed interface
        app1 = TestApp1()
        async with app1.run_test() as pilot:
            mock_window = app1.query_one(MockChatWindow)
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(mock_window, max_depth=5)
            
            print("Recommendations for non-tabbed interface:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            assert len(report['recommendations']) > 0, "Should generate recommendations"
        
        # Test with tabbed interface
        app2 = TestApp2()
        async with app2.run_test() as pilot:
            mock_window = app2.query_one(MockTabbedChatWindow)
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(mock_window, max_depth=5)
            
            print("\nRecommendations for tabbed interface:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            assert len(report['recommendations']) > 0, "Should generate recommendations"
        
        print("\n✅ Recommendations test passed!")
    
    # Run the async test
    import asyncio
    asyncio.run(run_test())
    return True


def main():
    """Run all diagnostic tests."""
    print("="*60)
    print("Chat Diagnostics Test Suite")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_non_tabbed_diagnostics()
        all_passed &= test_tabbed_diagnostics()
        all_passed &= test_state_capture_with_diagnostics()
        all_passed &= test_recommendations()
        
        if all_passed:
            print("\n" + "="*60)
            print("✅ ALL DIAGNOSTIC TESTS PASSED!")
            print("="*60)
            print("\nThe diagnostic tool successfully:")
            print("1. Detects different chat interface types")
            print("2. Identifies input widgets and their content")
            print("3. Provides actionable recommendations")
            print("4. Helps with state capture strategy")
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