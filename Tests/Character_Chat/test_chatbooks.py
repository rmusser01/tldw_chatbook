# test_chatbooks.py
# Tests for chatbook functionality using existing test infrastructure

import pytest
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from hypothesis import given, strategies as st
from hypothesis.strategies import composite
import string

# Import test utilities from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from db_test_utilities import TestDatabaseSchema, DatabasePopulator, setup_test_db
from test_utilities import TestDataFactory, chacha_db_factory, media_db_factory

# Import chatbook modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion, ConflictResolution
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_conflict_resolver import ChatbookConflictResolver


# ===========================================
# Chatbook Model Tests
# ===========================================

class TestChatbookModels:
    """Test chatbook data models."""
    
    def test_content_item_creation(self):
        """Test creating a content item."""
        item = ContentItem(
            id="test_1",
            type=ContentType.CONVERSATION,
            title="Test Conversation",
            description="A test conversation",
            created_at=datetime.now(),
            tags=["test", "conversation"]
        )
        
        assert item.id == "test_1"
        assert item.type == ContentType.CONVERSATION
        assert item.title == "Test Conversation"
        assert len(item.tags) == 2
    
    def test_content_item_serialization(self):
        """Test content item serialization."""
        item = ContentItem(
            id="test_1",
            type=ContentType.NOTE,
            title="Test Note",
            created_at=datetime.now()
        )
        
        # Convert to dict
        data = item.to_dict()
        assert data["id"] == "test_1"
        assert data["type"] == "note"
        assert data["title"] == "Test Note"
        
        # Restore from dict
        restored = ContentItem.from_dict(data)
        assert restored.id == item.id
        assert restored.type == item.type
        assert restored.title == item.title
    
    def test_chatbook_manifest_creation(self):
        """Test creating a chatbook manifest."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="A test chatbook",
            author="Test Author",
            content_items=[
                ContentItem(id="1", type=ContentType.CONVERSATION, title="Conv 1"),
                ContentItem(id="2", type=ContentType.NOTE, title="Note 1")
            ],
            relationships=[
                Relationship("1", "2", "references")
            ]
        )
        
        assert manifest.name == "Test Chatbook"
        assert len(manifest.content_items) == 2
        assert len(manifest.relationships) == 1
        assert manifest.total_conversations == 1
        assert manifest.total_notes == 1
    
    def test_chatbook_content_access(self):
        """Test accessing chatbook content."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=[
                ContentItem(id="conv_1", type=ContentType.CONVERSATION, title="Conv"),
                ContentItem(id="note_1", type=ContentType.NOTE, title="Note")
            ]
        )
        
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Test get by type
        conversations = chatbook.get_content_by_type(ContentType.CONVERSATION)
        assert len(conversations) == 1
        assert conversations[0].id == "conv_1"
        
        # Test get by ID
        item = chatbook.get_content_by_id("note_1")
        assert item is not None
        assert item.type == ContentType.NOTE


# ===========================================
# Chatbook Creator Tests
# ===========================================

class TestChatbookCreator:
    """Test chatbook creation functionality."""
    
    @pytest.fixture
    def mock_databases(self, chacha_db_factory, media_db_factory, tmp_path):
        """Create mock databases with test data."""
        # Create databases
        chachanotes_db = chacha_db_factory("test_client", str(tmp_path / "chacha.db"))
        media_db = media_db_factory("test_client", str(tmp_path / "media.db"))
        
        # Add test data using DatabasePopulator
        populator = DatabasePopulator(chachanotes_db.connection)
        
        # Add conversations with messages
        conv1_id = populator.add_conversation("Blue Whale Research", "test_client")
        populator.add_message(conv1_id, "user", "Tell me about blue whales")
        populator.add_message(conv1_id, "assistant", "Blue whales are the largest animals...")
        
        conv2_id = populator.add_conversation("General Chat", "test_client")
        populator.add_message(conv2_id, "user", "Hello!")
        
        # Add notes
        note1_id = populator.add_note("Blue Whale Facts", "Blue whales can grow up to 100 feet", "test_client")
        populator.add_keywords_to_note(note1_id, ["blue whale", "ocean", "marine life"])
        
        # Add character
        char_id = populator.add_character("Marine Biologist", "test_client", 
                                         description="Expert in marine life",
                                         personality="Scientific and informative")
        
        chachanotes_db.connection.commit()
        
        return {
            'chachanotes': chachanotes_db,
            'media': media_db,
            'test_ids': {
                'conversation_ids': [conv1_id, conv2_id],
                'note_ids': [note1_id],
                'character_ids': [char_id]
            }
        }
    
    def test_chatbook_creator_initialization(self, mock_databases):
        """Test creating a ChatbookCreator instance."""
        creator = ChatbookCreator(
            chachanotes_db=mock_databases['chachanotes'],
            media_db=mock_databases['media'],
            prompts_db=None,
            include_media=False,
            include_embeddings=False
        )
        
        assert creator.chachanotes_db is not None
        assert creator.media_db is not None
        assert creator.include_media is False
    
    def test_collect_conversations(self, mock_databases):
        """Test collecting conversations for chatbook."""
        creator = ChatbookCreator(
            chachanotes_db=mock_databases['chachanotes'],
            media_db=mock_databases['media']
        )
        
        # Collect specific conversations
        conv_ids = mock_databases['test_ids']['conversation_ids']
        items = creator._collect_conversations(conv_ids[:1])  # Just the first one
        
        assert len(items) == 1
        assert items[0].type == ContentType.CONVERSATION
        assert "Blue Whale" in items[0].title
    
    def test_collect_notes(self, mock_databases):
        """Test collecting notes for chatbook."""
        creator = ChatbookCreator(
            chachanotes_db=mock_databases['chachanotes'],
            media_db=mock_databases['media']
        )
        
        note_ids = mock_databases['test_ids']['note_ids']
        items = creator._collect_notes(note_ids)
        
        assert len(items) == 1
        assert items[0].type == ContentType.NOTE
        assert items[0].title == "Blue Whale Facts"
    
    @pytest.mark.asyncio
    async def test_create_chatbook(self, mock_databases, tmp_path):
        """Test creating a complete chatbook."""
        creator = ChatbookCreator(
            chachanotes_db=mock_databases['chachanotes'],
            media_db=mock_databases['media']
        )
        
        output_path = tmp_path / "test_chatbook.zip"
        
        # Create chatbook with selected content
        test_ids = mock_databases['test_ids']
        await creator.create_chatbook(
            name="Blue Whale Research Pack",
            description="A collection about blue whales",
            output_path=str(output_path),
            conversation_ids=test_ids['conversation_ids'][:1],
            note_ids=test_ids['note_ids'],
            character_ids=test_ids['character_ids']
        )
        
        # Verify the chatbook was created
        assert output_path.exists()
        
        # Verify contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            assert 'manifest.json' in zf.namelist()
            
            # Check manifest
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Blue Whale Research Pack"
            assert len(manifest_data['content_items']) >= 2  # At least note and conversation


# ===========================================
# Chatbook Importer Tests
# ===========================================

class TestChatbookImporter:
    """Test chatbook import functionality."""
    
    @pytest.fixture
    def sample_chatbook(self, tmp_path):
        """Create a sample chatbook for testing."""
        chatbook_path = tmp_path / "sample.zip"
        
        # Create manifest
        manifest = {
            "version": "1.0",
            "name": "Sample Chatbook",
            "description": "Test chatbook",
            "author": "Test",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "conv_1",
                    "type": "conversation",
                    "title": "Sample Conversation",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/conversations/conversation_conv_1.json"
                },
                {
                    "id": "note_1",
                    "type": "note",
                    "title": "Sample Note",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/notes/Sample Note.md"
                }
            ],
            "relationships": [],
            "include_media": False,
            "include_embeddings": False,
            "statistics": {
                "total_conversations": 1,
                "total_notes": 1,
                "total_characters": 0,
                "total_media_items": 0,
                "total_size_bytes": 1024
            }
        }
        
        # Create conversation content
        conversation = {
            "id": "conv_1",
            "title": "Sample Conversation",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "created_at": datetime.now().isoformat(),
            "metadata": {}
        }
        
        # Create note content
        note_content = "# Sample Note\n\nThis is a test note."
        
        # Create ZIP file
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))
            zf.writestr('content/conversations/conversation_conv_1.json', 
                       json.dumps(conversation, indent=2))
            zf.writestr('content/notes/Sample Note.md', note_content)
        
        return chatbook_path
    
    def test_chatbook_importer_initialization(self, chacha_db_factory, tmp_path):
        """Test creating a ChatbookImporter instance."""
        db = chacha_db_factory("test_client", str(tmp_path / "chacha.db"))
        
        importer = ChatbookImporter(
            chachanotes_db=db,
            media_db=None,
            prompts_db=None
        )
        
        assert importer.chachanotes_db is not None
        assert importer.conflict_resolver is not None
    
    @pytest.mark.asyncio
    async def test_preview_chatbook(self, chacha_db_factory, sample_chatbook, tmp_path):
        """Test previewing chatbook contents."""
        db = chacha_db_factory("test_client", str(tmp_path / "chacha.db"))
        importer = ChatbookImporter(chachanotes_db=db)
        
        # Preview the chatbook
        preview = await importer.preview_chatbook(str(sample_chatbook))
        
        assert preview['name'] == "Sample Chatbook"
        assert preview['total_items'] == 2
        assert preview['conversations'] == 1
        assert preview['notes'] == 1
        assert len(preview['conflicts']) == 0  # No conflicts in empty DB
    
    @pytest.mark.asyncio
    async def test_import_chatbook_no_conflicts(self, chacha_db_factory, sample_chatbook, tmp_path):
        """Test importing a chatbook with no conflicts."""
        db = chacha_db_factory("test_client", str(tmp_path / "chacha.db"))
        importer = ChatbookImporter(chachanotes_db=db)
        
        # Import the chatbook
        result = await importer.import_chatbook(
            chatbook_path=str(sample_chatbook),
            conflict_resolution=ConflictResolution.SKIP
        )
        
        assert result.status == "completed"
        assert result.imported_conversations == 1
        assert result.imported_notes == 1
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_import_chatbook_with_conflicts(self, chacha_db_factory, sample_chatbook, tmp_path):
        """Test importing a chatbook with conflicts."""
        db = chacha_db_factory("test_client", str(tmp_path / "chacha.db"))
        
        # Add existing data that will conflict
        populator = DatabasePopulator(db.connection)
        populator.add_conversation("Sample Conversation", "test_client")
        db.connection.commit()
        
        importer = ChatbookImporter(chachanotes_db=db)
        
        # Import with rename strategy
        result = await importer.import_chatbook(
            chatbook_path=str(sample_chatbook),
            conflict_resolution=ConflictResolution.RENAME
        )
        
        assert result.status == "completed"
        # Should still import but with renamed items
        assert result.imported_conversations >= 1


# ===========================================
# Property-Based Tests
# ===========================================

# Custom strategies for property-based testing
@composite
def content_type_strategy(draw):
    """Generate valid ContentType values."""
    return draw(st.sampled_from(list(ContentType)))


@composite
def content_item_strategy(draw):
    """Generate valid ContentItem instances."""
    return ContentItem(
        id=draw(st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + string.digits + "_-")),
        type=draw(content_type_strategy()),
        title=draw(st.text(min_size=1, max_size=200)),
        description=draw(st.one_of(st.none(), st.text(max_size=500))),
        created_at=draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1)))),
        tags=draw(st.lists(st.text(min_size=1, max_size=50), max_size=10))
    )


class TestChatbookProperties:
    """Property-based tests for chatbooks."""
    
    @given(content_item_strategy())
    def test_content_item_roundtrip(self, item):
        """Test that ContentItem survives dict roundtrip."""
        data = item.to_dict()
        restored = ContentItem.from_dict(data)
        
        assert restored.id == item.id
        assert restored.type == item.type
        assert restored.title == item.title
        assert restored.description == item.description
        assert restored.tags == item.tags
    
    @given(st.lists(content_item_strategy(), min_size=1, max_size=20))
    def test_chatbook_manifest_statistics(self, items):
        """Test that manifest statistics are calculated correctly."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items
        )
        
        # Verify statistics match content
        conv_count = sum(1 for item in items if item.type == ContentType.CONVERSATION)
        note_count = sum(1 for item in items if item.type == ContentType.NOTE)
        char_count = sum(1 for item in items if item.type == ContentType.CHARACTER)
        
        assert manifest.total_conversations == conv_count
        assert manifest.total_notes == note_count
        assert manifest.total_characters == char_count


# ===========================================
# Integration Tests
# ===========================================

class TestChatbookIntegration:
    """Integration tests for full chatbook workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_export_import_cycle(self, chacha_db_factory, media_db_factory, tmp_path):
        """Test complete export and import cycle."""
        # Setup source database
        source_db = chacha_db_factory("source_client", str(tmp_path / "source.db"))
        source_media = media_db_factory("source_client", str(tmp_path / "source_media.db"))
        
        # Populate with test data
        populator = DatabasePopulator(source_db.connection)
        conv_id = populator.add_conversation("Export Test", "source_client")
        populator.add_message(conv_id, "user", "Test message")
        note_id = populator.add_note("Export Note", "Test content", "source_client")
        source_db.connection.commit()
        
        # Export to chatbook
        creator = ChatbookCreator(source_db, source_media)
        export_path = tmp_path / "export.zip"
        
        await creator.create_chatbook(
            name="Integration Test",
            description="Test export",
            output_path=str(export_path),
            conversation_ids=[conv_id],
            note_ids=[note_id]
        )
        
        assert export_path.exists()
        
        # Setup target database
        target_db = chacha_db_factory("target_client", str(tmp_path / "target.db"))
        
        # Import chatbook
        importer = ChatbookImporter(target_db)
        result = await importer.import_chatbook(str(export_path))
        
        assert result.status == "completed"
        assert result.imported_conversations == 1
        assert result.imported_notes == 1
        assert len(result.errors) == 0