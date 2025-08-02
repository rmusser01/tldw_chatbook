# test_chatbook_models.py
# Unit tests for chatbook data models

import pytest
from datetime import datetime, timezone
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion
)


class TestContentType:
    """Test ContentType enum."""
    
    def test_content_type_values(self):
        """Test that all content types are defined."""
        assert ContentType.CONVERSATION.value == "conversation"
        assert ContentType.NOTE.value == "note"
        assert ContentType.CHARACTER.value == "character"
        assert ContentType.MEDIA.value == "media"
        assert ContentType.PROMPT.value == "prompt"
        assert ContentType.EMBEDDING.value == "embedding"


class TestContentItem:
    """Test ContentItem model."""
    
    def test_content_item_creation(self):
        """Test creating a content item with required fields."""
        item = ContentItem(
            id="test_123",
            type=ContentType.CONVERSATION,
            title="Test Conversation"
        )
        
        assert item.id == "test_123"
        assert item.type == ContentType.CONVERSATION
        assert item.title == "Test Conversation"
        assert item.description is None
        assert item.created_at is None
        assert item.updated_at is None
        assert item.tags == []
        assert item.metadata == {}
        assert item.file_path is None
    
    def test_content_item_with_all_fields(self):
        """Test creating a content item with all fields."""
        created = datetime.now(timezone.utc)
        updated = datetime.now(timezone.utc)
        
        item = ContentItem(
            id="test_456",
            type=ContentType.NOTE,
            title="Test Note",
            description="A test note description",
            created_at=created,
            updated_at=updated,
            tags=["test", "note"],
            metadata={"author": "test_user", "word_count": 100},
            file_path="content/notes/test_note.md"
        )
        
        assert item.id == "test_456"
        assert item.type == ContentType.NOTE
        assert item.title == "Test Note"
        assert item.description == "A test note description"
        assert item.created_at == created
        assert item.updated_at == updated
        assert item.tags == ["test", "note"]
        assert item.metadata["author"] == "test_user"
        assert item.metadata["word_count"] == 100
        assert item.file_path == "content/notes/test_note.md"
    
    def test_content_item_to_dict(self):
        """Test converting content item to dictionary."""
        created = datetime.now(timezone.utc)
        item = ContentItem(
            id="test_789",
            type=ContentType.CHARACTER,
            title="Test Character",
            created_at=created,
            tags=["character", "test"]
        )
        
        data = item.to_dict()
        
        assert data["id"] == "test_789"
        assert data["type"] == "character"
        assert data["title"] == "Test Character"
        assert data["created_at"] == created.isoformat()
        assert data["tags"] == ["character", "test"]
        assert data["description"] is None
        assert data["updated_at"] is None
        assert data["metadata"] == {}
        assert data["file_path"] is None
    
    def test_content_item_from_dict(self):
        """Test creating content item from dictionary."""
        data = {
            "id": "test_101",
            "type": "media",
            "title": "Test Media",
            "description": "A test media file",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-02T00:00:00+00:00",
            "tags": ["media", "test"],
            "metadata": {"size": 1024, "format": "mp4"},
            "file_path": "content/media/test.mp4"
        }
        
        item = ContentItem.from_dict(data)
        
        assert item.id == "test_101"
        assert item.type == ContentType.MEDIA
        assert item.title == "Test Media"
        assert item.description == "A test media file"
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)
        assert item.tags == ["media", "test"]
        assert item.metadata["size"] == 1024
        assert item.metadata["format"] == "mp4"
        assert item.file_path == "content/media/test.mp4"
    
    def test_content_item_roundtrip(self):
        """Test that content item survives dict roundtrip."""
        original = ContentItem(
            id="test_202",
            type=ContentType.PROMPT,
            title="Test Prompt",
            description="A test prompt",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["prompt", "test"],
            metadata={"tokens": 50},
            file_path="content/prompts/test.json"
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = ContentItem.from_dict(data)
        
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
        assert restored.file_path == original.file_path


class TestRelationship:
    """Test Relationship model."""
    
    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            source_id="conv_1",
            target_id="char_1",
            relationship_type="uses_character"
        )
        
        assert rel.source_id == "conv_1"
        assert rel.target_id == "char_1"
        assert rel.relationship_type == "uses_character"
        assert rel.metadata == {}
    
    def test_relationship_with_metadata(self):
        """Test creating a relationship with metadata."""
        rel = Relationship(
            source_id="note_1",
            target_id="note_2",
            relationship_type="references",
            metadata={"citation_count": 3, "relevance": 0.95}
        )
        
        assert rel.source_id == "note_1"
        assert rel.target_id == "note_2"
        assert rel.relationship_type == "references"
        assert rel.metadata["citation_count"] == 3
        assert rel.metadata["relevance"] == 0.95
    
    def test_relationship_to_dict(self):
        """Test converting relationship to dictionary."""
        rel = Relationship(
            source_id="media_1",
            target_id="conv_1",
            relationship_type="attached_to",
            metadata={"position": 5}
        )
        
        data = rel.to_dict()
        
        assert data["source_id"] == "media_1"
        assert data["target_id"] == "conv_1"
        assert data["relationship_type"] == "attached_to"
        assert data["metadata"]["position"] == 5


class TestChatbookManifest:
    """Test ChatbookManifest model."""
    
    def test_manifest_minimal(self):
        """Test creating a minimal manifest."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="A test chatbook"
        )
        
        assert manifest.version == ChatbookVersion.V1
        assert manifest.name == "Test Chatbook"
        assert manifest.description == "A test chatbook"
        assert manifest.author is None
        assert isinstance(manifest.created_at, datetime)
        assert isinstance(manifest.updated_at, datetime)
        assert manifest.content_items == []
        assert manifest.relationships == []
        assert manifest.include_media is False
        assert manifest.include_embeddings is False
        assert manifest.media_quality == "thumbnail"
    
    def test_manifest_with_content(self):
        """Test creating a manifest with content."""
        items = [
            ContentItem(id="1", type=ContentType.CONVERSATION, title="Conv 1"),
            ContentItem(id="2", type=ContentType.NOTE, title="Note 1"),
            ContentItem(id="3", type=ContentType.CHARACTER, title="Char 1")
        ]
        
        relationships = [
            Relationship("1", "3", "uses_character"),
            Relationship("2", "1", "references")
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="A test chatbook with content",
            author="Test Author",
            content_items=items,
            relationships=relationships,
            include_media=True,
            media_quality="compressed",
            tags=["test", "example"],
            categories=["testing"],
            language="en",
            license="MIT"
        )
        
        assert len(manifest.content_items) == 3
        assert len(manifest.relationships) == 2
        assert manifest.author == "Test Author"
        assert manifest.include_media is True
        assert manifest.media_quality == "compressed"
        assert manifest.tags == ["test", "example"]
        assert manifest.categories == ["testing"]
        assert manifest.language == "en"
        assert manifest.license == "MIT"
    
    def test_manifest_statistics_calculation(self):
        """Test that manifest statistics are set correctly."""
        items = [
            ContentItem(id="1", type=ContentType.CONVERSATION, title="Conv 1"),
            ContentItem(id="2", type=ContentType.CONVERSATION, title="Conv 2"),
            ContentItem(id="3", type=ContentType.NOTE, title="Note 1"),
            ContentItem(id="4", type=ContentType.CHARACTER, title="Char 1"),
            ContentItem(id="5", type=ContentType.MEDIA, title="Media 1"),
            ContentItem(id="6", type=ContentType.PROMPT, title="Prompt 1")
        ]
        
        # Calculate statistics from content items
        conv_count = sum(1 for item in items if item.type == ContentType.CONVERSATION)
        note_count = sum(1 for item in items if item.type == ContentType.NOTE)
        char_count = sum(1 for item in items if item.type == ContentType.CHARACTER)
        media_count = sum(1 for item in items if item.type == ContentType.MEDIA)
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items,
            total_conversations=conv_count,
            total_notes=note_count,
            total_characters=char_count,
            total_media_items=media_count
        )
        
        # Check statistics are set correctly
        assert manifest.total_conversations == 2
        assert manifest.total_notes == 1
        assert manifest.total_characters == 1
        assert manifest.total_media_items == 1
    
    def test_manifest_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="Test description",
            author="Test Author",
            content_items=[
                ContentItem(id="1", type=ContentType.NOTE, title="Note 1")
            ],
            tags=["test"]
        )
        
        data = manifest.to_dict()
        
        assert data["version"] == "1.0"
        assert data["name"] == "Test Chatbook"
        assert data["description"] == "Test description"
        assert data["author"] == "Test Author"
        assert len(data["content_items"]) == 1
        assert data["content_items"][0]["id"] == "1"
        assert data["tags"] == ["test"]
        assert "created_at" in data
        assert "updated_at" in data
        assert "statistics" in data
    
    def test_manifest_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "version": "1.0",
            "name": "Test Chatbook",
            "description": "Test description",
            "author": "Test Author",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-02T00:00:00+00:00",
            "content_items": [
                {
                    "id": "1",
                    "type": "conversation",
                    "title": "Conv 1",
                    "created_at": "2024-01-01T00:00:00+00:00"
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
            "media_quality": "original",
            "statistics": {
                "total_conversations": 1,
                "total_notes": 0,
                "total_characters": 0,
                "total_media_items": 0,
                "total_size_bytes": 1024
            },
            "tags": ["test"],
            "categories": ["example"],
            "language": "en",
            "license": "MIT"
        }
        
        manifest = ChatbookManifest.from_dict(data)
        
        assert manifest.version == ChatbookVersion.V1
        assert manifest.name == "Test Chatbook"
        assert manifest.author == "Test Author"
        assert len(manifest.content_items) == 1
        assert manifest.content_items[0].id == "1"
        assert len(manifest.relationships) == 1
        assert manifest.include_media is True
        assert manifest.media_quality == "original"
        assert manifest.tags == ["test"]
        assert manifest.language == "en"


class TestChatbook:
    """Test Chatbook model."""
    
    def test_chatbook_creation(self):
        """Test creating a chatbook."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="Test"
        )
        content = ChatbookContent()
        
        chatbook = Chatbook(manifest=manifest, content=content)
        
        assert chatbook.manifest == manifest
        assert chatbook.content == content
    
    def test_chatbook_get_content_by_type(self):
        """Test getting content by type."""
        items = [
            ContentItem(id="1", type=ContentType.CONVERSATION, title="Conv 1"),
            ContentItem(id="2", type=ContentType.CONVERSATION, title="Conv 2"),
            ContentItem(id="3", type=ContentType.NOTE, title="Note 1"),
            ContentItem(id="4", type=ContentType.CHARACTER, title="Char 1")
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items
        )
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Get conversations
        conversations = chatbook.get_content_by_type(ContentType.CONVERSATION)
        assert len(conversations) == 2
        assert all(item.type == ContentType.CONVERSATION for item in conversations)
        
        # Get notes
        notes = chatbook.get_content_by_type(ContentType.NOTE)
        assert len(notes) == 1
        assert notes[0].title == "Note 1"
        
        # Get non-existent type
        embeddings = chatbook.get_content_by_type(ContentType.EMBEDDING)
        assert len(embeddings) == 0
    
    def test_chatbook_get_content_by_id(self):
        """Test getting content by ID."""
        items = [
            ContentItem(id="conv_1", type=ContentType.CONVERSATION, title="Conv 1"),
            ContentItem(id="note_1", type=ContentType.NOTE, title="Note 1")
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items
        )
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Get existing item
        conv = chatbook.get_content_by_id("conv_1")
        assert conv is not None
        assert conv.title == "Conv 1"
        
        # Get non-existent item
        missing = chatbook.get_content_by_id("missing_id")
        assert missing is None
    
    def test_chatbook_get_related_content(self):
        """Test getting related content."""
        items = [
            ContentItem(id="conv_1", type=ContentType.CONVERSATION, title="Conv 1"),
            ContentItem(id="char_1", type=ContentType.CHARACTER, title="Char 1"),
            ContentItem(id="note_1", type=ContentType.NOTE, title="Note 1"),
            ContentItem(id="note_2", type=ContentType.NOTE, title="Note 2")
        ]
        
        relationships = [
            Relationship("conv_1", "char_1", "uses_character"),
            Relationship("note_1", "conv_1", "references"),
            Relationship("note_2", "note_1", "references")
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Test",
            content_items=items,
            relationships=relationships
        )
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Get items related to conv_1
        related = chatbook.get_related_content("conv_1")
        related_ids = [item.id for item in related]
        assert "char_1" in related_ids  # Direct relationship
        assert "note_1" in related_ids  # Reverse relationship
        
        # Get items related to note_1
        related = chatbook.get_related_content("note_1")
        related_ids = [item.id for item in related]
        assert "conv_1" in related_ids
        assert "note_2" in related_ids
        
        # Get items related to non-existent ID
        related = chatbook.get_related_content("missing_id")
        assert len(related) == 0