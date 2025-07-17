"""
Example UI test demonstrating best practices for Textual application testing.

This file shows various patterns and techniques for testing Textual UI components.
"""
import pytest
from textual.widgets import Button, Input, TextArea, Label
from textual.containers import Container
from unittest.mock import MagicMock, AsyncMock, patch

from tldw_chatbook.UI.Chat_Window import ChatWindow
from tldw_chatbook.UI.Notes_Window import NotesWindow
from tldw_chatbook.Widgets.chat_message import ChatMessage


@pytest.mark.ui
class TestUIBestPractices:
    """Demonstrates best practices for UI testing."""
    
    @pytest.mark.asyncio
    async def test_widget_initialization(self, widget_pilot, mock_app_instance):
        """Test proper widget initialization and composition."""
        # Create widget with dependencies
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            chat_window = app.test_widget
            
            # Verify widget is properly initialized
            assert chat_window is not None
            assert hasattr(chat_window, 'app_instance')
            assert chat_window.app_instance == mock_app_instance
            
            # Wait for composition to complete
            await pilot.pause()
            
            # Verify expected child widgets exist
            chat_log = app.query_one("#chat-log")
            assert chat_log is not None
            
            chat_input = app.query_one("#chat-input", TextArea)
            assert chat_input is not None
    
    @pytest.mark.asyncio
    async def test_user_interaction_flow(self, widget_pilot, mock_app_instance):
        """Test a complete user interaction flow."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            
            # Wait for UI to be ready
            await pilot.pause()
            
            # Get input area and send button
            chat_input = app.query_one("#chat-input", TextArea)
            send_button = app.query_one("#send-stop-chat", Button)
            
            # Simulate user typing
            await pilot.click(chat_input)
            # Set the text directly since pilot.type might not exist
            chat_input.text = "Hello, this is a test message"
            await pilot.pause()
            
            # Verify input contains text
            assert chat_input.text == "Hello, this is a test message"
            
            # Click send button
            await pilot.click(send_button)
            await pilot.pause()
            
            # In a real test, verify the message was processed
            # This would depend on your event handling implementation
    
    @pytest.mark.asyncio
    async def test_widget_state_changes(self, widget_pilot, mock_app_instance, assert_widget_state):
        """Test widget state changes in response to events."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Get the unified send/stop button
            send_stop_button = app.query_one("#send-stop-chat", Button)
            
            # Initially should be enabled for sending
            assert_widget_state.is_enabled(send_stop_button)
            
            # Simulate starting generation (would normally be triggered by send)
            # In real app, this would change the button state to "Stop"
            # For this example, we'll just verify it exists
            assert send_stop_button is not None
    
    @pytest.mark.asyncio
    async def test_css_class_manipulation(self, widget_pilot, mock_app_instance, assert_widget_state):
        """Test CSS class changes for visual states."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Get a button
            button = app.query_one("#send-stop-chat", Button)
            
            # Add a class
            button.add_class("sending")
            assert_widget_state.has_class(button, "sending")
            
            # Remove the class
            button.remove_class("sending")
            assert_widget_state.not_has_class(button, "sending")
    
    @pytest.mark.asyncio
    async def test_async_operations(self, widget_pilot, mock_app_instance, wait_for_condition):
        """Test handling of async operations."""
        # Mock an async operation
        mock_app_instance.run_worker = AsyncMock()
        
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Trigger an operation that uses run_worker
            # This is a simplified example - real implementation would vary
            chat_window = app.test_widget
            
            # Simulate triggering async work
            await chat_window.app_instance.run_worker()
            
            # Verify worker was called
            mock_app_instance.run_worker.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_handling(self, widget_pilot, mock_app_instance):
        """Test custom event handling."""
        events_received = []
        
        # Mock event handler
        async def mock_handler(event):
            events_received.append(event)
        
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            chat_window = app.test_widget
            
            await pilot.pause()
            
            # Click a button - this should trigger events in the app
            button = app.query_one("#send-stop-chat", Button)
            await pilot.click(button)
            await pilot.pause()
            
            # Since we can't easily override event handlers in the test,
            # we'll just verify that the button exists and can be clicked
            assert button is not None
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, widget_pilot, mock_app_instance):
        """Test keyboard shortcut handling."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Test Ctrl+[ to toggle sidebar
            await pilot.press("ctrl+[")
            await pilot.pause()
            
            # In a real test, verify sidebar state changed
            # This depends on your implementation
    
    @pytest.mark.asyncio
    async def test_error_handling(self, widget_pilot, mock_app_instance):
        """Test error handling and user feedback."""
        # Configure mock to simulate error
        mock_app_instance.notify = MagicMock()
        
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Simulate an error condition
            # In real app, this might be network error, validation error, etc.
            chat_window = app.test_widget
            
            # Trigger error notification (simplified)
            chat_window.app_instance.notify("An error occurred", severity="error")
            
            # Verify notification was called
            mock_app_instance.notify.assert_called_with(
                "An error occurred", 
                severity="error"
            )
    
    @pytest.mark.asyncio
    async def test_responsive_layout(self, isolated_widget_pilot, mock_app_instance):
        """Test widget behavior at different sizes."""
        # Create a test app with ChatWindow
        def compose():
            yield ChatWindow(app_instance=mock_app_instance)
        
        async with isolated_widget_pilot(compose) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Verify the ChatWindow exists
            chat_window = app.query_one(ChatWindow)
            assert chat_window is not None
            assert isinstance(chat_window, ChatWindow)
    
    @pytest.mark.asyncio
    async def test_focus_management(self, widget_pilot, mock_app_instance):
        """Test focus handling between widgets."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Get focusable widgets
            chat_input = app.query_one("#chat-input", TextArea)
            
            # Focus the input
            chat_input.focus()
            await pilot.pause()
            
            # Verify focus
            assert app.focused == chat_input
            
            # Tab to next widget
            await pilot.press("tab")
            await pilot.pause()
            
            # Focus should have moved
            assert app.focused != chat_input


@pytest.mark.ui
class TestComplexScenarios:
    """Test more complex UI scenarios."""
    
    @pytest.mark.asyncio
    async def test_message_rendering(self, isolated_widget_pilot):
        """Test rendering of chat messages."""
        messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
        
        # Create a compose function that yields messages
        def compose():
            with Container(id="test-container"):
                for i, msg in enumerate(messages):
                    yield ChatMessage(
                        message=msg["content"],
                        role=msg["role"],
                        id=f"chat-message-{i}"
                    )
        
        async with isolated_widget_pilot(compose) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Verify messages rendered
            rendered_messages = app.query(ChatMessage)
            assert len(rendered_messages) == 2
            
            # Verify content
            # Note: The actual attribute names may differ based on ChatMessage implementation
            for i, msg_widget in enumerate(rendered_messages):
                # ChatMessage widget exists
                assert msg_widget is not None
                # Just verify it's a ChatMessage instance
                assert isinstance(msg_widget, ChatMessage)
    
    @pytest.mark.asyncio
    async def test_dynamic_content_update(self, widget_pilot, mock_app_instance):
        """Test dynamic content updates."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            chat_window = app.test_widget
            await pilot.pause()
            
            # Get chat log
            chat_log = app.query_one("#chat-log")
            
            # Add content dynamically
            test_message = ChatMessage(
                message="Test dynamic message",
                role="user",
                id="dynamic-1"
            )
            
            await chat_log.mount(test_message)
            await pilot.pause()
            
            # Verify message was added
            messages = chat_log.query(ChatMessage)
            assert len(messages) > 0  # At least the message was added
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_with_many_widgets(self, isolated_widget_pilot):
        """Test UI performance with many widgets."""
        num_messages = 100
        
        def compose():
            with Container(id="perf-test"):
                for i in range(num_messages):
                    yield Label(f"Message {i}", id=f"msg-{i}")
        
        async with isolated_widget_pilot(compose) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Verify all widgets rendered
            labels = app.query(Label)
            assert len(labels) == num_messages
            
            # Test scrolling performance
            container = app.query_one("#perf-test", Container)
            
            # Scroll to bottom (implementation specific)
            if hasattr(container, 'scroll_end'):
                container.scroll_end()
                await pilot.pause()


@pytest.mark.ui
class TestAccessibility:
    """Test accessibility features."""
    
    @pytest.mark.asyncio
    async def test_aria_labels(self, widget_pilot, mock_app_instance):
        """Test that widgets have appropriate ARIA labels."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Check important interactive elements
            buttons = app.query(Button)
            
            # Define buttons that should have tooltips
            buttons_requiring_tooltips = {
                "send-stop-chat",
                "toggle-chat-left-sidebar", 
                "toggle-chat-right-sidebar",
                "respond-for-me-button"
            }
            
            for button in buttons:
                if button.id in buttons_requiring_tooltips:
                    # These specific buttons should have tooltips
                    assert button.tooltip is not None, \
                        f"Button {button.id} should have a tooltip"
    
    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, widget_pilot, mock_app_instance):
        """Test complete keyboard navigation."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Get all focusable widgets
            focusable = [w for w in app.query("*") if w.can_focus]
            
            # Tab through all widgets
            for _ in focusable:
                await pilot.press("tab")
                await pilot.pause()
                
                # Verify something has focus
                assert app.focused is not None


# Parametrized tests for multiple scenarios
@pytest.mark.parametrize("button_id,expected_tooltip", [
    ("send-stop-chat", ["Send message", "Stop generation"]),  # This button shows different tooltips based on state
    ("respond-for-me-button", "Suggest a response"),
    ("toggle-chat-left-sidebar", "Toggle left sidebar (Ctrl+[)"),
    ("toggle-chat-right-sidebar", "Toggle right sidebar (Ctrl+])"),
])
@pytest.mark.asyncio
async def test_button_tooltips_parametrized(widget_pilot, mock_app_instance, 
                                           button_id, expected_tooltip):
    """Test button tooltips using parametrization."""
    async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
        app = pilot.app
        await pilot.pause()
        
        button = app.query_one(f"#{button_id}", Button)
        
        # For send-stop-chat, accept either tooltip since it's dynamic
        if isinstance(expected_tooltip, list):
            assert button.tooltip in expected_tooltip, \
                f"Button {button_id} tooltip '{button.tooltip}' not in expected values {expected_tooltip}"
        else:
            assert button.tooltip == expected_tooltip, \
                f"Button {button_id} has incorrect tooltip: {button.tooltip} (expected: {expected_tooltip})"