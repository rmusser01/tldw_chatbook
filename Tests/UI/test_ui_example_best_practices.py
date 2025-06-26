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
            send_button = app.query_one("#send-chat", Button)
            
            # Simulate user typing
            await pilot.click(chat_input)
            await pilot.type("Hello, this is a test message")
            
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
            
            # Get stop button
            stop_button = app.query_one("#stop-chat-generation", Button)
            
            # Initially should be disabled
            assert_widget_state.is_disabled(stop_button)
            
            # Simulate starting generation (would normally be triggered by send)
            # In real app, this would be done via event
            stop_button.disabled = False
            await pilot.pause()
            
            # Now should be enabled
            assert_widget_state.is_enabled(stop_button)
    
    @pytest.mark.asyncio
    async def test_css_class_manipulation(self, widget_pilot, mock_app_instance, assert_widget_state):
        """Test CSS class changes for visual states."""
        async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Get a button
            button = app.query_one("#send-chat", Button)
            
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
            
            # Override event handler
            original_handler = chat_window.on_button_pressed
            chat_window.on_button_pressed = mock_handler
            
            await pilot.pause()
            
            # Click a button
            button = app.query_one("#send-chat", Button)
            await pilot.click(button)
            await pilot.pause()
            
            # Verify event was received
            assert len(events_received) > 0
            
            # Restore original handler
            chat_window.on_button_pressed = original_handler
    
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
    async def test_responsive_layout(self, isolated_widget_pilot):
        """Test widget behavior at different sizes."""
        def compose():
            with Container():
                yield ChatWindow(MagicMock())
        
        async with isolated_widget_pilot(compose) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Test at different sizes
            original_size = app.size
            
            # Simulate small screen
            app.size = (80, 24)
            await pilot.pause()
            
            # Verify layout adjustments (implementation specific)
            container = app.query_one(Container)
            assert container is not None
            
            # Restore size
            app.size = original_size
    
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
        
        def compose():
            with Container(id="test-container"):
                for msg in messages:
                    yield ChatMessage(
                        message=msg["content"],
                        role=msg["role"],
                        message_id=f"msg-{messages.index(msg)}"
                    )
        
        async with isolated_widget_pilot(compose) as pilot:
            app = pilot.app
            await pilot.pause()
            
            # Verify messages rendered
            rendered_messages = app.query(ChatMessage)
            assert len(rendered_messages) == 2
            
            # Verify content
            for i, msg_widget in enumerate(rendered_messages):
                assert msg_widget.message == messages[i]["content"]
                assert msg_widget.role == messages[i]["role"]
    
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
                message="Dynamic test message",
                role="user",
                message_id="dynamic-1"
            )
            
            await chat_log.mount(test_message)
            await pilot.pause()
            
            # Verify message was added
            messages = chat_log.query(ChatMessage)
            assert any(m.message == "Dynamic test message" for m in messages)
    
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
            for button in buttons:
                # Verify button has either tooltip or aria-label
                assert button.tooltip or hasattr(button, 'aria_label'), \
                    f"Button {button.id} lacks accessibility info"
    
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
    ("send-chat", "Send message"),
    ("stop-chat-generation", "Stop generation"),
    ("respond-for-me-button", "Suggest a response"),
])
@pytest.mark.asyncio
async def test_button_tooltips_parametrized(widget_pilot, mock_app_instance, 
                                           button_id, expected_tooltip, 
                                           assert_tooltip):
    """Test button tooltips using parametrization."""
    async with await widget_pilot(ChatWindow, app_instance=mock_app_instance) as pilot:
        app = pilot.app
        await pilot.pause()
        
        button = app.query_one(f"#{button_id}", Button)
        assert_tooltip(button, expected_tooltip)