"""Test message actions in ChatV99."""

import pytest
from textual.pilot import Pilot
from unittest.mock import Mock, patch
import pyperclip

from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatMessage
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced
from tldw_chatbook.chat_v99.widgets.message_list import MessageList


class TestMessageActions:
    """Test message action buttons and functionality."""
    
    @pytest.mark.asyncio
    async def test_message_action_buttons_visible_on_hover(self):
        """Test that action buttons appear on hover."""
        async with ChatV99App().run_test() as pilot:
            # Add a message
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            message_list.add_user_message("Test message")
            await pilot.pause(0.1)
            
            # Get message item
            message_items = message_list.query("MessageItemEnhanced")
            assert len(message_items) > 0
            
            # Check action buttons exist
            message_item = message_items[0]
            action_buttons = message_item.query(".action-button")
            assert len(action_buttons) == 6  # Edit, Copy, Regenerate, Continue, Pin, Delete
    
    @pytest.mark.asyncio
    async def test_copy_action(self):
        """Test copy to clipboard action."""
        async with ChatV99App().run_test() as pilot:
            # Add a message
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            test_content = "Test content to copy"
            message_list.add_user_message(test_content)
            await pilot.pause(0.1)
            
            # Get copy button
            message_item = message_list.query("MessageItemEnhanced")[0]
            copy_button = message_item.query_one("#copy-btn")
            
            # Mock pyperclip at the actual import location (imported inside the function)
            with patch('pyperclip.copy') as mock_copy:
                # Click copy button
                await pilot.click(copy_button)
                await pilot.pause(0.1)
                
                # Check content was copied
                mock_copy.assert_called_once_with(test_content)
    
    @pytest.mark.asyncio
    async def test_delete_action(self):
        """Test delete message action."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Add messages
            msg1 = message_list.add_user_message("Message 1")
            msg2 = message_list.add_assistant_message("Response 1")
            msg3 = message_list.add_user_message("Message 2")
            
            # Update session
            app.current_session.messages = [msg1, msg2, msg3]
            await pilot.pause(0.1)
            
            # Delete middle message
            message_items = message_list.query("MessageItemEnhanced")
            delete_button = message_items[1].query_one("#delete-btn")
            
            await pilot.click(delete_button)
            await pilot.pause(0.5)  # Wait for event handling and reactive update
            
            # Check message was deleted
            assert len(app.current_session.messages) == 2
            assert msg2 not in app.current_session.messages
    
    @pytest.mark.asyncio
    async def test_regenerate_action(self):
        """Test regenerate response action."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            screen = pilot.app.screen
            message_list = screen.query_one("#message-list", MessageList)
            
            # Add user and assistant messages
            user_msg = message_list.add_user_message("Tell me a joke")
            assistant_msg = message_list.add_assistant_message("Why did the chicken cross the road?")
            app.current_session.messages = [user_msg, assistant_msg]
            await pilot.pause(0.1)
            
            # Mock LLM worker
            with patch.object(screen, 'process_message') as mock_process:
                # Click regenerate on assistant message
                message_items = message_list.query("MessageItemEnhanced")
                regenerate_button = message_items[1].query_one("#regenerate-btn")
                
                await pilot.click(regenerate_button)
                await pilot.pause(0.1)
                
                # Check that process_message was called with original prompt
                mock_process.assert_called_once_with("Tell me a joke")
                
                # Check assistant message was removed
                assert len(app.current_session.messages) == 1
    
    @pytest.mark.asyncio
    async def test_continue_action(self):
        """Test continue generation action."""
        async with ChatV99App().run_test() as pilot:
            screen = pilot.app.screen
            message_list = screen.query_one("#message-list", MessageList)
            
            # Add assistant message
            message_list.add_assistant_message("Once upon a time...")
            await pilot.pause(0.1)
            
            # Mock process_message
            with patch.object(screen, 'process_message') as mock_process:
                # Click continue button
                message_item = message_list.query("MessageItemEnhanced")[0]
                continue_button = message_item.query_one("#continue-btn")
                
                await pilot.click(continue_button)
                await pilot.pause(0.1)
                
                # Check continue prompt was sent
                mock_process.assert_called_once_with("Continue from where you left off.")
    
    @pytest.mark.asyncio
    async def test_pin_action_visual_feedback(self):
        """Test pin button visual feedback."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Add a message
            message_list.add_user_message("Important message")
            await pilot.pause(0.1)
            
            # Get pin button
            message_item = message_list.query("MessageItemEnhanced")[0]
            pin_button = message_item.query_one("#pin-btn")
            
            # Check initial state
            assert pin_button.label == "📌"
            
            # Click pin button
            await pilot.click(pin_button)
            await pilot.pause(0.1)
            
            # Check visual change
            assert pin_button.label == "📍"
            
            # Click again to unpin
            await pilot.click(pin_button)
            await pilot.pause(0.1)
            
            # Check reverted
            assert pin_button.label == "📌"
    
    @pytest.mark.asyncio
    async def test_action_buttons_not_shown_during_streaming(self):
        """Test that action buttons don't appear while streaming."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Create streaming message
            streaming_msg = ChatMessage(role="assistant", content="Streaming...")
            message_item = MessageItemEnhanced(streaming_msg, is_streaming=True)
            
            # Check no action buttons during streaming
            action_buttons = message_item.query(".action-button")
            assert len(action_buttons) == 0
            
            # Finalize streaming
            message_item.finalize_streaming()
            await pilot.pause(0.1)
            
            # Check action buttons appear after streaming
            action_buttons = message_item.query(".action-button")
            assert len(action_buttons) == 6


class TestMessageEditing:
    """Test message editing functionality."""
    
    @pytest.mark.asyncio
    async def test_edit_mode_activation(self):
        """Test entering edit mode for a message."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Add a message
            message_list.add_user_message("Original message")
            await pilot.pause(0.1)
            
            # Click edit button
            message_item = message_list.query("MessageItemEnhanced")[0]
            edit_button = message_item.query_one("#edit-btn")
            
            await pilot.click(edit_button)
            await pilot.pause(0.1)
            
            # For now, just check notification appears
            # Full edit mode would require input field replacement
            # This is a placeholder for future implementation


class TestTokenCounting:
    """Test token counting display."""
    
    @pytest.mark.asyncio
    async def test_token_count_display(self):
        """Test that token count is displayed for messages."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Add a longer message
            long_message = "This is a longer message " * 20  # ~100 chars
            message_list.add_user_message(long_message)
            await pilot.pause(0.1)
            
            # Check token count is displayed
            message_item = message_list.query("MessageItemEnhanced")[0]
            token_display = message_item.query(".message-token-count")
            
            # Should show token count (approximate)
            assert len(token_display) > 0
            token_text = token_display[0].renderable
            assert "tokens" in str(token_text).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])