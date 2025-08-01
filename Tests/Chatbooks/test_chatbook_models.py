# test_chatbook_models.py
# Unit tests for chatbook data models

import pytest
from datetime import datetime
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookVersion, ContentType, ContentItem, Relationship,
    ChatbookManifest, ChatbookContent, Chatbook
)


class TestContentItem:
    """Test ContentItem model."""
    
    def test_content_item_creation(self):
        """Test creating a ContentItem."""
        item = ContentItem(
            id="conv_123",
            type=ContentType.CONVERSATION,
            title="Test Conversation",
            description="A test conversation",
            created_at=datetime(2024, 1, 1, 12, 0),
            tags=["test", "example"],
            metadata={"messages": 5}
        )
        
        assert item.id == "conv_123"
        assert item.type == ContentType.CONVERSATION
        assert item.title == "Test Conversation"
        assert item.description == "A test conversation"
        assert item.tags == ["test", "example"]
        assert item.metadata["messages"] == 5
    
    def test_content_item_to_dict(self):
        """Test converting ContentItem to dictionary."""
        created = datetime(2024, 1, 1, 12, 0)
        updated = datetime(2024, 1, 2, 13, 0)
        
        item = ContentItem(
            id="note_456",
            type=ContentType.NOTE,
            title="Test Note",
            created_at=created,
            updated_at=updated,
            file_path="content/notes/test.md"
        )
        
        data = item.to_dict()
        
        assert data["id"] == "note_456"
        assert data["type"] == "note"
        assert data["title"] == "Test Note"
        assert data["created_at"] == created.isoformat()
        assert data["updated_at"] == updated.isoformat()
        assert data["file_path"] == "content/notes/test.md"
    
    def test_content_item_from_dict(self):
        """Test creating ContentItem from dictionary."""
        data = {
            "id": "char_789",
            "type": "character",
            "title": "Test Character",
            "description": "A test character",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T13:00:00",
            "tags": ["character", "test"],
            "metadata": {"personality": "friendly"},
            "file_path": "content/characters/char_789.json"
        }
        
        item = ContentItem.from_dict(data)
        
        assert item.id == "char_789"
        assert item.type == ContentType.CHARACTER
        assert item.title == "Test Character"
        assert item.description == "A test character"
        assert item.created_at == datetime(2024, 1, 1, 12, 0)
        assert item.updated_at == datetime(2024, 1, 2, 13, 0)
        assert item.tags == ["character", "test"]
        assert item.metadata["personality"] == "friendly"
    
    def test_content_item_minimal(self):
        """Test ContentItem with minimal required fields."""
        item = ContentItem(
            id="1",
            type=ContentType.PROMPT,
            title="Minimal Item"
        )
        
        assert item.id == "1"
        assert item.type == ContentType.PROMPT
        assert item.title == "Minimal Item"
        assert item.description is None
        assert item.created_at is None
        assert item.updated_at is None
        assert item.tags == []
        assert item.metadata == {}
        assert item.file_path is None


class TestRelationship:
    """Test Relationship model."""
    
    def test_relationship_creation(self):
        """Test creating a Relationship."""
        rel = Relationship(
            source_id="conv_123",
            target_id="char_456",
            relationship_type="uses_character",
            metadata={"primary": True}
        )
        
        assert rel.source_id == "conv_123"
        assert rel.target_id == "char_456"
        assert rel.relationship_type == "uses_character"
        assert rel.metadata["primary"] is True
    
    def test_relationship_to_dict(self):
        """Test converting Relationship to dictionary."""
        rel = Relationship(
            source_id="note_1",
            target_id="note_2",
            relationship_type="references"
        )
        
        data = rel.to_dict()
        
        assert data["source_id"] == "note_1"
        assert data["target_id"] == "note_2"
        assert data["relationship_type"] == "references"
        assert data["metadata"] == {}


class TestChatbookManifest:
    """Test ChatbookManifest model."""
    
    def test_manifest_creation(self):
        """Test creating a ChatbookManifest."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="A test chatbook for unit testing",
            author="Test Author"
        )
        
        assert manifest.version == ChatbookVersion.V1
        assert manifest.name == "Test Chatbook"
        assert manifest.description == "A test chatbook for unit testing"
        assert manifest.author == "Test Author"
        assert isinstance(manifest.created_at, datetime)
        assert isinstance(manifest.updated_at, datetime)
        assert manifest.content_items == []
        assert manifest.relationships == []
    
    def test_manifest_with_content(self):
        """Test manifest with content items and relationships."""
        # Create content items
        conv = ContentItem(
            id="conv_1",
            type=ContentType.CONVERSATION,
            title="Conversation 1"
        )
        char = ContentItem(
            id="char_1",
            type=ContentType.CHARACTER,
            title="Character 1"
        )
        
        # Create relationship
        rel = Relationship(
            source_id="conv_1",
            target_id="char_1",
            relationship_type="uses_character"
        )
        
        # Create manifest
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="Test",
            content_items=[conv, char],
            relationships=[rel]
        )
        
        assert len(manifest.content_items) == 2
        assert len(manifest.relationships) == 1
        assert manifest.content_items[0].id == "conv_1"
        assert manifest.relationships[0].relationship_type == "uses_character"
    
    def test_manifest_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Export Test",
            description="Testing export",
            author="Tester",
            tags=["test", "export"],
            categories=["testing"],
            language="en",
            license="MIT"
        )
        
        # Add some statistics
        manifest.total_conversations = 5
        manifest.total_notes = 10
        manifest.total_characters = 2
        manifest.total_size_bytes = 1024000
        
        data = manifest.to_dict()
        
        assert data["version"] == "1.0"
        assert data["name"] == "Export Test"
        assert data["author"] == "Tester"
        assert data["tags"] == ["test", "export"]
        assert data["categories"] == ["testing"]
        assert data["language"] == "en"
        assert data["license"] == "MIT"
        
        stats = data["statistics"]
        assert stats["total_conversations"] == 5
        assert stats["total_notes"] == 10
        assert stats["total_characters"] == 2
        assert stats["total_size_bytes"] == 1024000
    
    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "version": "1.0",
            "name": "Import Test",
            "description": "Testing import",
            "author": "Importer",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T13:00:00",
            "content_items": [
                {
                    "id": "1",
                    "type": "note",
                    "title": "Note 1"
                }
            ],
            "relationships": [
                {
                    "source_id": "1",
                    "target_id": "2",
                    "relationship_type": "references",
                    "metadata": {}
                }
            ],
            "include_media": True,
            "include_embeddings": False,
            "media_quality": "compressed",
            "statistics": {
                "total_conversations": 3,
                "total_notes": 7,
                "total_characters": 1,
                "total_size_bytes": 512000
            },
            "tags": ["import", "test"],
            "categories": ["testing"],
            "language": "en",
            "license": "GPL"
        }
        
        manifest = ChatbookManifest.from_dict(data)
        
        assert manifest.version == ChatbookVersion.V1
        assert manifest.name == "Import Test"
        assert manifest.author == "Importer"
        assert manifest.created_at == datetime(2024, 1, 1, 12, 0)
        assert manifest.updated_at == datetime(2024, 1, 2, 13, 0)
        assert len(manifest.content_items) == 1
        assert len(manifest.relationships) == 1
        assert manifest.include_media is True
        assert manifest.include_embeddings is False
        assert manifest.media_quality == "compressed"
        assert manifest.total_conversations == 3
        assert manifest.total_notes == 7
        assert manifest.license == "GPL"


class TestChatbook:
    """Test Chatbook model."""
    
    def test_chatbook_creation(self):
        """Test creating a Chatbook."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="Test"
        )
        content = ChatbookContent()
        
        chatbook = Chatbook(
            manifest=manifest,
            content=content,
            base_path=Path("/test/path")
        )
        
        assert chatbook.manifest.name == "Test Chatbook"
        assert isinstance(chatbook.content, ChatbookContent)
        assert chatbook.base_path == Path("/test/path")
    
    def test_get_content_by_type(self):
        """Test getting content by type."""
        # Create content items
        conv1 = ContentItem(id="1", type=ContentType.CONVERSATION, title="Conv 1")
        conv2 = ContentItem(id="2", type=ContentType.CONVERSATION, title="Conv 2")
        note1 = ContentItem(id="3", type=ContentType.NOTE, title="Note 1")
        char1 = ContentItem(id="4", type=ContentType.CHARACTER, title="Char 1")
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=[conv1, conv2, note1, char1]
        )
        
        chatbook = Chatbook(manifest=manifest, content=ChatbookContent())
        
        # Test getting conversations
        conversations = chatbook.get_content_by_type(ContentType.CONVERSATION)
        assert len(conversations) == 2
        assert all(item.type == ContentType.CONVERSATION for item in conversations)
        
        # Test getting notes
        notes = chatbook.get_content_by_type(ContentType.NOTE)
        assert len(notes) == 1
        assert notes[0].title == "Note 1"
        
        # Test getting characters
        characters = chatbook.get_content_by_type(ContentType.CHARACTER)
        assert len(characters) == 1
        assert characters[0].id == "4"
    
    def test_get_content_by_id(self):
        """Test getting content by ID."""
        items = [
            ContentItem(id="conv_123", type=ContentType.CONVERSATION, title="Test Conv"),
            ContentItem(id="note_456", type=ContentType.NOTE, title="Test Note"),
            ContentItem(id="char_789", type=ContentType.CHARACTER, title="Test Char")
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items
        )
        
        chatbook = Chatbook(manifest=manifest, content=ChatbookContent())
        
        # Test finding existing items
        conv = chatbook.get_content_by_id("conv_123")
        assert conv is not None
        assert conv.title == "Test Conv"
        
        note = chatbook.get_content_by_id("note_456")
        assert note is not None
        assert note.type == ContentType.NOTE
        
        # Test non-existent item
        missing = chatbook.get_content_by_id("missing_999")
        assert missing is None
    
    def test_get_related_content(self):
        """Test getting related content."""
        # Create content items
        conv1 = ContentItem(id="conv_1", type=ContentType.CONVERSATION, title="Conv 1")
        char1 = ContentItem(id="char_1", type=ContentType.CHARACTER, title="Char 1")
        note1 = ContentItem(id="note_1", type=ContentType.NOTE, title="Note 1")
        note2 = ContentItem(id="note_2", type=ContentType.NOTE, title="Note 2")
        
        # Create relationships
        rel1 = Relationship("conv_1", "char_1", "uses_character")
        rel2 = Relationship("conv_1", "note_1", "references")
        rel3 = Relationship("note_1", "note_2", "references")
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=[conv1, char1, note1, note2],
            relationships=[rel1, rel2, rel3]
        )
        
        chatbook = Chatbook(manifest=manifest, content=ChatbookContent())
        
        # Test getting items related to conversation
        conv_related = chatbook.get_related_content("conv_1")
        assert len(conv_related) == 2
        related_ids = [item.id for item in conv_related]
        assert "char_1" in related_ids
        assert "note_1" in related_ids
        
        # Test getting items related to note
        note_related = chatbook.get_related_content("note_1")
        assert len(note_related) == 2
        related_ids = [item.id for item in note_related]
        assert "conv_1" in related_ids
        assert "note_2" in related_ids
        
        # Test item with no relationships
        char_related = chatbook.get_related_content("char_1")
        assert len(char_related) == 1
        assert char_related[0].id == "conv_1"


class TestEnums:
    """Test enum values."""
    
    def test_chatbook_version_values(self):
        """Test ChatbookVersion enum values."""
        assert ChatbookVersion.V1.value == "1.0"
        assert ChatbookVersion.V2.value == "2.0"
    
    def test_content_type_values(self):
        """Test ContentType enum values."""
        assert ContentType.CONVERSATION.value == "conversation"
        assert ContentType.NOTE.value == "note"
        assert ContentType.CHARACTER.value == "character"
        assert ContentType.MEDIA.value == "media"
        assert ContentType.EMBEDDING.value == "embedding"
        assert ContentType.PROMPT.value == "prompt"
        assert ContentType.EVALUATION.value == "evaluation"