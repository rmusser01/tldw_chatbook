"""
Test message feedback functionality.
"""

import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage


class TestMessageFeedback:
    """Test message feedback database operations."""
    
    @pytest.fixture
    def test_db(self):
        """Create test database with default character."""
        db = CharactersRAGDB(":memory:", "test_client")
        # Create a default character that conversations require
        db.add_character_card({
            'name': 'Default Character',
            'description': 'Default test character',
            'personality': 'Test personality',
            'scenario': 'Test scenario',
            'first_mes': 'Hello',
            'mes_example': 'Example message'
        })
        return db
    
    def test_update_message_feedback_thumbs_up(self, test_db):
        """Test updating message with thumbs up feedback."""
        # Create a conversation and message
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'User',
            'content': 'Test message'
        })
        
        # Update with thumbs up
        success = test_db.update_message_feedback(msg_id, "1;", expected_version=1)
        assert success is True
        
        # Verify feedback was saved
        message = test_db.get_message_by_id(msg_id)
        assert message['feedback'] == "1;"
        assert message['version'] == 2
    
    def test_update_message_feedback_thumbs_down(self, test_db):
        """Test updating message with thumbs down feedback."""
        # Create a conversation and message
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'Assistant',
            'content': 'AI response'
        })
        
        # Update with thumbs down
        success = test_db.update_message_feedback(msg_id, "2;", expected_version=1)
        assert success is True
        
        # Verify feedback was saved
        message = test_db.get_message_by_id(msg_id)
        assert message['feedback'] == "2;"
        assert message['version'] == 2
    
    def test_update_message_feedback_clear(self, test_db):
        """Test clearing message feedback."""
        # Create message with feedback
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'Assistant',
            'content': 'AI response'
        })
        
        # First set feedback
        test_db.update_message_feedback(msg_id, "1;", expected_version=1)
        
        # Then clear it
        success = test_db.update_message_feedback(msg_id, None, expected_version=2)
        assert success is True
        
        # Verify feedback was cleared
        message = test_db.get_message_by_id(msg_id)
        assert message['feedback'] is None
        assert message['version'] == 3
    
    def test_update_message_feedback_invalid_format(self, test_db):
        """Test that invalid feedback format raises InputError."""
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'User',
            'content': 'Test message'
        })
        
        # Test invalid formats
        with pytest.raises(InputError, match="Invalid feedback format"):
            test_db.update_message_feedback(msg_id, "3;", expected_version=1)
        
        with pytest.raises(InputError, match="Invalid feedback format"):
            test_db.update_message_feedback(msg_id, "thumbs up", expected_version=1)
        
        with pytest.raises(InputError, match="Invalid feedback format"):
            test_db.update_message_feedback(msg_id, "1", expected_version=1)  # Missing semicolon
    
    def test_update_message_feedback_version_conflict(self, test_db):
        """Test version conflict when updating feedback."""
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'User',
            'content': 'Test message'
        })
        
        # Try to update with wrong version
        with pytest.raises(ConflictError):
            test_db.update_message_feedback(msg_id, "1;", expected_version=999)
    
    def test_feedback_with_future_comments(self, test_db):
        """Test that feedback format supports future extension with comments."""
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'Assistant',
            'content': 'AI response'
        })
        
        # Test extended format (future use)
        extended_feedback = "1;Great response, very helpful!"
        success = test_db.update_message_feedback(msg_id, extended_feedback, expected_version=1)
        assert success is True
        
        message = test_db.get_message_by_id(msg_id)
        assert message['feedback'] == extended_feedback


class TestChatMessageWidget:
    """Test ChatMessage widget feedback display."""
    
    def test_chat_message_with_feedback(self):
        """Test ChatMessage widget initialization with feedback."""
        # Test with thumbs up
        msg = ChatMessage(
            message="Test message",
            role="Assistant",
            feedback="1;"
        )
        assert msg.feedback == "1;"
        
        # Test with thumbs down
        msg2 = ChatMessage(
            message="Test message",
            role="Assistant",
            feedback="2;"
        )
        assert msg2.feedback == "2;"
        
        # Test without feedback
        msg3 = ChatMessage(
            message="Test message",
            role="User"
        )
        assert msg3.feedback is None
    
    def test_chat_message_feedback_buttons_display(self):
        """Test that feedback state affects button display in compose."""
        # This would require a more complex test with the Textual test framework
        # For now, just verify the widget accepts feedback parameter
        msg = ChatMessage(
            message="AI response",
            role="Assistant",
            message_id="test-123",
            message_version=1,
            feedback="1;"
        )
        
        # In compose(), the buttons should show:
        # thumb_up_label = "👍✓" (checked)
        # thumb_down_label = "👎" (unchecked)
        # This is handled by the compose method logic


class TestFeedbackDialog:
    """Test the feedback dialog functionality."""
    
    @pytest.fixture
    def test_db(self):
        """Create test database with default character."""
        db = CharactersRAGDB(":memory:", "test_client")
        # Create a default character that conversations require
        db.add_character_card({
            'name': 'Default Character',
            'description': 'Default test character',
            'personality': 'Test personality',
            'scenario': 'Test scenario',
            'first_mes': 'Hello',
            'mes_example': 'Example message'
        })
        return db
    
    def test_feedback_dialog_initialization(self):
        """Test FeedbackDialog initialization."""
        from tldw_chatbook.Widgets.feedback_dialog import FeedbackDialog
        
        # Test thumbs up dialog
        dialog = FeedbackDialog(
            feedback_type="1",
            existing_comment="Great response!",
            callback=lambda x: None
        )
        assert dialog.feedback_type == "1"
        assert dialog.existing_comment == "Great response!"
        
        # Test thumbs down dialog
        dialog2 = FeedbackDialog(
            feedback_type="2",
            existing_comment="",
            callback=lambda x: None
        )
        assert dialog2.feedback_type == "2"
        assert dialog2.existing_comment == ""
    
    def test_feedback_format_with_comments(self, test_db):
        """Test feedback storage with comments."""
        conv_id = test_db.add_conversation({
            'title': 'Test Conversation',
            'rating': 5,
            'character_id': 1  # Use the default character
        })
        
        msg_id = test_db.add_message({
            'conversation_id': conv_id,
            'sender': 'Assistant',
            'content': 'AI response'
        })
        
        # Test with comment
        feedback_with_comment = "1;This is a very helpful response!"
        success = test_db.update_message_feedback(msg_id, feedback_with_comment, expected_version=1)
        assert success is True
        
        message = test_db.get_message_by_id(msg_id)
        assert message['feedback'] == feedback_with_comment
        
        # Extract and verify parts
        parts = message['feedback'].split(";", 1)
        assert parts[0] == "1"
        assert parts[1] == "This is a very helpful response!"