# test_chatbook_creator.py
# Unit tests for chatbook creator

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_models import ContentType, ChatbookVersion


class TestChatbookCreator:
    """Test ChatbookCreator functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_dbs(self):
        """Create mock database instances."""
        # Mock ChaChaNotes DB
        mock_chachanotes = Mock()
        mock_chachanotes.get_conversation_by_id.return_value = {
            'id': 1,
            'conversation_name': 'Test Conversation',
            'created_at': '2024-01-01T12:00:00',
            'updated_at': '2024-01-02T13:00:00',
            'character_id': 123
        }
        mock_chachanotes.get_messages_for_conversation.return_value = [
            {
                'id': 1,
                'sender': 'user',
                'message': 'Hello',
                'timestamp': '2024-01-01T12:00:00'
            },
            {
                'id': 2,
                'sender': 'assistant',
                'message': 'Hi there!',
                'timestamp': '2024-01-01T12:00:05'
            }
        ]
        mock_chachanotes.get_note_by_id.return_value = {
            'id': 1,
            'title': 'Test Note',
            'content': '# Test Note\n\nThis is a test note.',
            'created_at': '2024-01-01T12:00:00',
            'updated_at': '2024-01-02T13:00:00',
            'keywords': 'test,example'
        }
        mock_chachanotes.get_character_details.return_value = {
            'id': 123,
            'name': 'Test Character',
            'description': 'A test character',
            'personality': 'Friendly',
            'created_at': '2024-01-01T12:00:00',
            'updated_at': '2024-01-02T13:00:00'
        }
        mock_chachanotes.get_character_card_details.return_value = {
            'version': 'v3',
            'data': {'name': 'Test Character'}
        }
        
        # Mock Prompts DB
        mock_prompts = Mock()
        mock_prompts.get_prompt_by_id.return_value = {
            'id': 1,
            'name': 'Test Prompt',
            'details': 'A test prompt',
            'system_prompt': 'You are a helpful assistant.',
            'user_prompt': None,
            'created_at': '2024-01-01T12:00:00',
            'updated_at': '2024-01-02T13:00:00'
        }
        
        return {
            'chachanotes': mock_chachanotes,
            'prompts': mock_prompts
        }
    
    @pytest.fixture
    def creator(self, temp_dir):
        """Create a ChatbookCreator instance."""
        db_paths = {
            'chachanotes': str(temp_dir / 'chachanotes.db'),
            'prompts': str(temp_dir / 'prompts.db'),
            'media': str(temp_dir / 'media.db')
        }
        return ChatbookCreator(db_paths)
    
    def test_creator_initialization(self, temp_dir):
        """Test ChatbookCreator initialization."""
        db_paths = {
            'chachanotes': str(temp_dir / 'chachanotes.db'),
            'prompts': str(temp_dir / 'prompts.db')
        }
        
        creator = ChatbookCreator(db_paths)
        
        assert creator.db_paths == db_paths
        assert creator.temp_dir.exists()
        assert "chatbooks" in str(creator.temp_dir)
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_create_chatbook_basic(self, mock_prompts_class, mock_chachanotes_class, 
                                   creator, temp_dir, mock_dbs):
        """Test basic chatbook creation."""
        # Setup mocks
        mock_chachanotes_class.return_value = mock_dbs['chachanotes']
        mock_prompts_class.return_value = mock_dbs['prompts']
        
        # Create chatbook
        output_path = temp_dir / "test_chatbook.zip"
        success, message = creator.create_chatbook(
            name="Test Chatbook",
            description="A test chatbook",
            content_selections={
                ContentType.CONVERSATION: ["1"],
                ContentType.NOTE: ["1"]
            },
            output_path=output_path,
            author="Test Author",
            tags=["test", "example"]
        )
        
        assert success is True
        assert "successfully" in message
        assert output_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert 'manifest.json' in files
            assert 'README.md' in files
            assert 'content/conversations/conversation_1.json' in files
            assert 'content/notes/Test Note.md' in files
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_collect_conversations(self, mock_db_class, creator, temp_dir, mock_dbs):
        """Test conversation collection."""
        mock_db_class.return_value = mock_dbs['chachanotes']
        
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookContent
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        content = ChatbookContent()
        
        # Collect conversations
        creator._collect_conversations(["1"], work_dir, manifest, content)
        
        # Verify content was collected
        assert len(content.conversations) == 1
        assert content.conversations[0]['id'] == 1
        assert content.conversations[0]['name'] == 'Test Conversation'
        assert len(content.conversations[0]['messages']) == 2
        
        # Verify manifest was updated
        assert len(manifest.content_items) == 1
        assert manifest.content_items[0].type == ContentType.CONVERSATION
        assert manifest.content_items[0].title == 'Test Conversation'
        
        # Verify file was created
        conv_file = work_dir / "content" / "conversations" / "conversation_1.json"
        assert conv_file.exists()
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_collect_notes(self, mock_db_class, creator, temp_dir, mock_dbs):
        """Test note collection."""
        mock_db_class.return_value = mock_dbs['chachanotes']
        
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookContent
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        content = ChatbookContent()
        
        # Collect notes
        creator._collect_notes(["1"], work_dir, manifest, content)
        
        # Verify content was collected
        assert len(content.notes) == 1
        assert content.notes[0]['title'] == 'Test Note'
        assert content.notes[0]['tags'] == ['test', 'example']
        
        # Verify manifest was updated
        assert len(manifest.content_items) == 1
        assert manifest.content_items[0].type == ContentType.NOTE
        assert manifest.content_items[0].tags == ['test', 'example']
        
        # Verify markdown file was created
        note_file = work_dir / "content" / "notes" / "Test Note.md"
        assert note_file.exists()
        
        # Check markdown content
        with open(note_file, 'r') as f:
            content_text = f.read()
            assert "---" in content_text  # Frontmatter
            assert "title: Test Note" in content_text
            assert "# Test Note" in content_text
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_collect_characters(self, mock_db_class, creator, temp_dir, mock_dbs):
        """Test character collection."""
        mock_db_class.return_value = mock_dbs['chachanotes']
        
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookContent
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        content = ChatbookContent()
        
        # Collect characters
        creator._collect_characters(["123"], work_dir, manifest, content)
        
        # Verify content was collected
        assert len(content.characters) == 1
        assert content.characters[0]['name'] == 'Test Character'
        assert content.characters[0]['card'] == {'version': 'v3', 'data': {'name': 'Test Character'}}
        
        # Verify manifest was updated
        assert len(manifest.content_items) == 1
        assert manifest.content_items[0].type == ContentType.CHARACTER
        assert manifest.content_items[0].description == 'A test character'
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_collect_prompts(self, mock_db_class, creator, temp_dir, mock_dbs):
        """Test prompt collection."""
        mock_db_class.return_value = mock_dbs['prompts']
        
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookContent
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        content = ChatbookContent()
        
        # Collect prompts
        creator._collect_prompts(["1"], work_dir, manifest, content)
        
        # Verify content was collected
        assert len(content.prompts) == 1
        assert content.prompts[0]['name'] == 'Test Prompt'
        assert content.prompts[0]['content'] == 'You are a helpful assistant.'
        
        # Verify manifest was updated
        assert len(manifest.content_items) == 1
        assert manifest.content_items[0].type == ContentType.PROMPT
    
    def test_discover_relationships(self, creator):
        """Test relationship discovery."""
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookContent, ContentItem
        
        # Create manifest with items
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        
        # Add character to manifest
        manifest.content_items.append(ContentItem(
            id="123",
            type=ContentType.CHARACTER,
            title="Test Character"
        ))
        
        # Create content with conversation referencing character
        content = ChatbookContent()
        content.conversations = [{
            'id': '1',
            'name': 'Test Conv',
            'character_id': 123
        }]
        
        # Discover relationships
        creator._discover_relationships(manifest, content)
        
        # Verify relationship was created
        assert len(manifest.relationships) == 1
        assert manifest.relationships[0].source_id == '1'
        assert manifest.relationships[0].target_id == '123'
        assert manifest.relationships[0].relationship_type == 'uses_character'
    
    def test_create_readme(self, creator, temp_dir):
        """Test README creation."""
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest
        
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="A comprehensive test chatbook",
            author="Test Author",
            tags=["test", "example", "demo"],
            license="MIT"
        )
        manifest.total_conversations = 5
        manifest.total_notes = 10
        manifest.total_characters = 2
        
        # Create README
        creator._create_readme(work_dir, manifest)
        
        readme_path = work_dir / "README.md"
        assert readme_path.exists()
        
        # Check README content
        with open(readme_path, 'r') as f:
            content = f.read()
            assert "# Test Chatbook" in content
            assert "A comprehensive test chatbook" in content
            assert "**Author:** Test Author" in content
            assert "**Conversations:** 5" in content
            assert "**Notes:** 10" in content
            assert "**Characters:** 2" in content
            assert "test, example, demo" in content
            assert "MIT" in content
    
    def test_create_zip_archive(self, creator, temp_dir):
        """Test ZIP archive creation."""
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        
        # Create some test files
        (work_dir / "manifest.json").write_text('{"version": "1.0"}')
        (work_dir / "README.md").write_text("# Test Chatbook")
        content_dir = work_dir / "content" / "notes"
        content_dir.mkdir(parents=True)
        (content_dir / "note1.md").write_text("Test note content")
        
        # Create ZIP
        output_path = temp_dir / "output.zip"
        creator._create_zip_archive(work_dir, output_path)
        
        assert output_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert "manifest.json" in files
            assert "README.md" in files
            assert "content/notes/note1.md" in files
    
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_error_handling(self, mock_db_class, creator, temp_dir):
        """Test error handling during creation."""
        # Setup mock to raise exception
        mock_db = Mock()
        mock_db.get_conversation_by_id.side_effect = Exception("Database error")
        mock_db_class.return_value = mock_db
        
        output_path = temp_dir / "error_test.zip"
        success, message = creator.create_chatbook(
            name="Error Test",
            description="Testing error handling",
            content_selections={ContentType.CONVERSATION: ["1"]},
            output_path=output_path
        )
        
        assert success is False
        assert "Error creating chatbook" in message
        assert not output_path.exists()
    
    def test_add_character_dependency(self, creator):
        """Test character dependency tracking."""
        from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ContentItem
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test"
        )
        
        # Add character dependency - character not in manifest
        creator._add_character_dependency(123, manifest)
        # Should just log warning, not crash
        
        # Add character to manifest
        manifest.content_items.append(ContentItem(
            id="123",
            type=ContentType.CHARACTER,
            title="Test Character"
        ))
        
        # Try adding again - should not duplicate
        creator._add_character_dependency(123, manifest)
        assert len([item for item in manifest.content_items if item.id == "123"]) == 1