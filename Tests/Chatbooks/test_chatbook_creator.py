# test_chatbook_creator.py
# Unit tests for chatbook creator

import pytest
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open, Mock
import sqlite3

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator


class TestChatbookCreator:
    """Test ChatbookCreator functionality."""
    
    @pytest.fixture
    def temp_db_paths(self, tmp_path):
        """Create temporary database paths."""
        db_dir = tmp_path / "databases"
        db_dir.mkdir()
        
        paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Create empty database files with minimal schema
        for name, path in paths.items():
            conn = sqlite3.connect(path)
            if name == 'ChaChaNotes':
                # Create minimal schema for ChaChaNotes
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY,
                        conversation_name TEXT,
                        title TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        character_id INTEGER,
                        media_id INTEGER,
                        deleted_at TEXT,
                        is_deleted INTEGER DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY,
                        conversation_id INTEGER,
                        role TEXT,
                        content TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        created_at TEXT,
                        keywords TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS characters (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        personality TEXT,
                        scenario TEXT,
                        greeting_message TEXT,
                        example_messages TEXT
                    )
                """)
            elif name == 'Prompts':
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prompts (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        author TEXT,
                        details TEXT,
                        system_prompt TEXT,
                        user_prompt TEXT
                    )
                """)
            conn.commit()
            conn.close()
            
        return paths
    
    @pytest.fixture
    def chatbook_creator(self, temp_db_paths):
        """Create a ChatbookCreator instance with test database paths."""
        return ChatbookCreator(db_paths=temp_db_paths)
    
    def test_creator_initialization(self, chatbook_creator, temp_db_paths):
        """Test ChatbookCreator initialization."""
        assert chatbook_creator.db_paths == temp_db_paths
        assert chatbook_creator.temp_dir.exists()
        assert "chatbooks" in str(chatbook_creator.temp_dir)
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_create_chatbook_minimal(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
        """Test creating a minimal chatbook."""
        # Setup mocks
        mock_chacha_db.return_value = MagicMock()
        mock_prompts_db.return_value = MagicMock()
        
        output_path = tmp_path / "test_chatbook.zip"
        
        # Create chatbook with empty content
        content_selections = {
            ContentType.CONVERSATION: [],
            ContentType.NOTE: [],
            ContentType.CHARACTER: []
        }
        
        success, message = chatbook_creator.create_chatbook(
            name="Test Chatbook",
            description="A test chatbook",
            content_selections=content_selections,
            output_path=output_path
        )
        
        # Since we have empty databases, this should succeed but with no content
        assert success is True
        assert output_path.exists()
        
        # Verify contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            assert 'manifest.json' in zf.namelist()
            
            # Check manifest
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Test Chatbook"
            assert manifest_data['description'] == "A test chatbook"
            assert manifest_data['version'] == "1.0"
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_create_chatbook_with_sample_data(self, mock_prompts_db, mock_chacha_db, chatbook_creator, temp_db_paths, tmp_path):
        """Test creating a chatbook with sample data."""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_prompts_db.return_value = MagicMock()
        
        # Mock conversation data
        mock_db_instance.get_conversation_by_id.return_value = {
            'id': 1,
            'conversation_name': 'Test Conversation',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'character_id': 1
        }
        mock_db_instance.get_messages_for_conversation.return_value = [
            {'id': 1, 'sender': 'user', 'message': 'Hello', 'timestamp': datetime.now().isoformat()},
            {'id': 2, 'sender': 'assistant', 'message': 'Hi there!', 'timestamp': datetime.now().isoformat()}
        ]
        
        # Mock note data
        mock_db_instance.get_note_by_id.return_value = {
            'id': 1,
            'title': 'Test Note',
            'content': 'This is test content',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'keywords': 'test,sample'
        }
        
        # Mock character data
        mock_db_instance.get_character_details.return_value = {
            'id': 1,
            'name': 'Test Character',
            'description': 'A test character',
            'personality': 'Helpful',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        mock_db_instance.get_character_card_details.return_value = {}
        
        # Create chatbook
        output_path = tmp_path / "test_chatbook_with_data.zip"
        content_selections = {
            ContentType.CONVERSATION: ["1"],
            ContentType.NOTE: ["1"],
            ContentType.CHARACTER: ["1"]
        }
        
        success, message = chatbook_creator.create_chatbook(
            name="Test Chatbook with Data",
            description="A test chatbook containing sample data",
            content_selections=content_selections,
            output_path=output_path,
            author="Test Author",
            tags=["test", "sample"],
            categories=["testing"]
        )
        
        assert success is True
        assert output_path.exists()
        
        # Verify contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            
            # Check metadata
            assert manifest_data['name'] == "Test Chatbook with Data"
            assert manifest_data['author'] == "Test Author"
            assert manifest_data['tags'] == ["test", "sample"]
            assert manifest_data['categories'] == ["testing"]
            
            # Check content items
            assert len(manifest_data['content_items']) >= 3
            
            # Verify content files exist
            namelist = zf.namelist()
            assert any('conversations/' in name for name in namelist)
            assert any('notes/' in name for name in namelist)
            assert any('characters/' in name for name in namelist)
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_create_chatbook_error_handling(self, mock_chacha_db, chatbook_creator, tmp_path):
        """Test error handling during chatbook creation."""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.get_conversation_by_id.return_value = None  # Conversation not found
        
        output_path = tmp_path / "test_error.zip"
        
        # Try with invalid content type
        content_selections = {
            ContentType.CONVERSATION: ["999"]  # Non-existent ID
        }
        
        success, message = chatbook_creator.create_chatbook(
            name="Error Test",
            description="Testing error handling",
            content_selections=content_selections,
            output_path=output_path
        )
        
        # Should still succeed but with no conversations
        assert success is True
        assert output_path.exists()
    
    @patch('zipfile.ZipFile')
    def test_create_chatbook_zip_error(self, mock_zipfile, chatbook_creator, tmp_path):
        """Test handling of ZIP creation errors."""
        # Mock ZIP file to raise error
        mock_zipfile.side_effect = Exception("ZIP creation failed")
        
        output_path = tmp_path / "test_zip_error.zip"
        content_selections = {}
        
        success, message = chatbook_creator.create_chatbook(
            name="ZIP Error Test",
            description="Testing ZIP error",
            content_selections=content_selections,
            output_path=output_path
        )
        
        assert success is False
        assert "error" in message.lower()
    
    def test_chatbook_with_media_settings(self, chatbook_creator, tmp_path):
        """Test creating chatbook with media settings."""
        output_path = tmp_path / "test_media.zip"
        
        success, message = chatbook_creator.create_chatbook(
            name="Media Test",
            description="Testing media settings",
            content_selections={},
            output_path=output_path,
            include_media=True,
            media_quality="original",
            include_embeddings=True
        )
        
        assert success is True
        
        # Check manifest for media settings
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data.get('include_media') is True
            assert manifest_data.get('media_quality') == "original"
            assert manifest_data.get('include_embeddings') is True