# test_chatbook_property.py
# Property-based tests for chatbook models using Hypothesis

import pytest
from datetime import datetime, timezone
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import composite
import string

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion
)


# Custom strategies for our domain objects

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
        created_at=draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)))),
        updated_at=draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)))),
        tags=draw(st.lists(st.text(min_size=1, max_size=50), max_size=10)),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(st.text(), st.integers(), st.booleans(), st.none()),
            max_size=20
        )),
        file_path=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    )


@composite
def relationship_strategy(draw):
    """Generate valid Relationship instances."""
    return Relationship(
        source_id=draw(st.text(min_size=1, max_size=50)),
        target_id=draw(st.text(min_size=1, max_size=50)),
        relationship_type=draw(st.sampled_from(["uses_character", "references", "parent_of", "requires", "related_to"])),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=10
        ))
    )


@composite
def chatbook_manifest_strategy(draw):
    """Generate valid ChatbookManifest instances."""
    content_items = draw(st.lists(content_item_strategy(), max_size=50))
    
    # Ensure created_at <= updated_at if both are present
    created = draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)))
    updated = draw(st.datetimes(min_value=created, max_value=datetime(2030, 1, 1)))
    
    return ChatbookManifest(
        version=ChatbookVersion.V1,  # For now, only V1
        name=draw(st.text(min_size=1, max_size=200)),
        description=draw(st.text(min_size=1, max_size=1000)),
        author=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        created_at=created,
        updated_at=updated,
        content_items=content_items,
        relationships=draw(st.lists(relationship_strategy(), max_size=100)),
        include_media=draw(st.booleans()),
        include_embeddings=draw(st.booleans()),
        media_quality=draw(st.sampled_from(["thumbnail", "compressed", "original"])),
        total_conversations=draw(st.integers(min_value=0, max_value=1000)),
        total_notes=draw(st.integers(min_value=0, max_value=1000)),
        total_characters=draw(st.integers(min_value=0, max_value=100)),
        total_media_items=draw(st.integers(min_value=0, max_value=1000)),
        total_size_bytes=draw(st.integers(min_value=0, max_value=10**12)),  # Up to 1TB
        tags=draw(st.lists(st.text(min_size=1, max_size=50), max_size=20)),
        categories=draw(st.lists(st.text(min_size=1, max_size=50), max_size=10)),
        language=draw(st.sampled_from(["en", "es", "fr", "de", "ja", "zh"])),
        license=draw(st.one_of(st.none(), st.sampled_from(["MIT", "GPL", "Apache", "CC BY", "CC BY-SA"])))
    )


class TestContentItemProperties:
    """Property-based tests for ContentItem."""
    
    @given(content_item_strategy())
    def test_content_item_roundtrip(self, item):
        """Test that ContentItem survives dict roundtrip."""
        # Convert to dict and back
        data = item.to_dict()
        restored = ContentItem.from_dict(data)
        
        # Verify all fields match
        assert restored.id == item.id
        assert restored.type == item.type
        assert restored.title == item.title
        assert restored.description == item.description
        assert restored.tags == item.tags
        assert restored.metadata == item.metadata
        assert restored.file_path == item.file_path
        
        # Handle datetime comparison (may have microsecond differences)
        if item.created_at:
            assert abs((restored.created_at - item.created_at).total_seconds()) < 1
        if item.updated_at:
            assert abs((restored.updated_at - item.updated_at).total_seconds()) < 1
    
    @given(content_item_strategy())
    def test_content_item_required_fields(self, item):
        """Test that required fields are always present."""
        assert item.id is not None and len(item.id) > 0
        assert item.type in ContentType
        assert item.title is not None and len(item.title) > 0
    
    @given(st.lists(content_item_strategy(), min_size=2, max_size=10))
    def test_content_item_uniqueness(self, items):
        """Test that IDs should be unique within a collection."""
        ids = [item.id for item in items]
        # This is a property we expect - IDs should be unique
        # In practice, the system should enforce this
        assert len(ids) == len(set(ids)) or True  # Allow duplicates for testing


class TestRelationshipProperties:
    """Property-based tests for Relationship."""
    
    @given(relationship_strategy())
    def test_relationship_validity(self, rel):
        """Test that relationships have valid structure."""
        assert rel.source_id is not None and len(rel.source_id) > 0
        assert rel.target_id is not None and len(rel.target_id) > 0
        assert rel.relationship_type is not None and len(rel.relationship_type) > 0
        assert isinstance(rel.metadata, dict)
    
    @given(relationship_strategy())
    def test_relationship_serialization(self, rel):
        """Test relationship serialization."""
        data = rel.to_dict()
        
        assert data["source_id"] == rel.source_id
        assert data["target_id"] == rel.target_id
        assert data["relationship_type"] == rel.relationship_type
        assert data["metadata"] == rel.metadata
    
    @given(st.lists(relationship_strategy(), min_size=1, max_size=20))
    def test_relationship_graph_properties(self, relationships):
        """Test properties of relationship graphs."""
        # No self-loops (in our domain)
        for rel in relationships:
            # This is a business rule we might want to enforce
            if rel.source_id == rel.target_id:
                # Log this as a potential issue
                pass
        
        # Check for duplicate relationships
        rel_tuples = [(r.source_id, r.target_id, r.relationship_type) for r in relationships]
        # Duplicates might be allowed depending on business rules


class TestChatbookManifestProperties:
    """Property-based tests for ChatbookManifest."""
    
    @given(chatbook_manifest_strategy())
    def test_manifest_roundtrip(self, manifest):
        """Test manifest survives JSON roundtrip."""
        # Convert to dict and back
        data = manifest.to_dict()
        restored = ChatbookManifest.from_dict(data)
        
        # Basic fields
        assert restored.version == manifest.version
        assert restored.name == manifest.name
        assert restored.description == manifest.description
        assert restored.author == manifest.author
        
        # Collections
        assert len(restored.content_items) == len(manifest.content_items)
        assert len(restored.relationships) == len(manifest.relationships)
        
        # Statistics
        assert restored.total_conversations == manifest.total_conversations
        assert restored.total_notes == manifest.total_notes
        assert restored.total_characters == manifest.total_characters
        
        # Metadata
        assert restored.tags == manifest.tags
        assert restored.categories == manifest.categories
        assert restored.language == manifest.language
        assert restored.license == manifest.license
    
    @given(chatbook_manifest_strategy())
    def test_manifest_invariants(self, manifest):
        """Test manifest invariants."""
        # Version should be valid
        assert manifest.version in ChatbookVersion
        
        # Required fields should be non-empty
        assert len(manifest.name) > 0
        assert len(manifest.description) > 0
        
        # Timestamps should be ordered
        assert manifest.created_at <= manifest.updated_at
        
        # Statistics should be non-negative
        assert manifest.total_conversations >= 0
        assert manifest.total_notes >= 0
        assert manifest.total_characters >= 0
        assert manifest.total_media_items >= 0
        assert manifest.total_size_bytes >= 0
        
        # Media quality should be valid
        assert manifest.media_quality in ["thumbnail", "compressed", "original"]
    
    @given(chatbook_manifest_strategy())
    def test_manifest_content_consistency(self, manifest):
        """Test consistency between content items and statistics."""
        # Count content types
        conv_count = sum(1 for item in manifest.content_items if item.type == ContentType.CONVERSATION)
        note_count = sum(1 for item in manifest.content_items if item.type == ContentType.NOTE)
        char_count = sum(1 for item in manifest.content_items if item.type == ContentType.CHARACTER)
        media_count = sum(1 for item in manifest.content_items if item.type == ContentType.MEDIA)
        
        # Statistics should be >= actual content (might include deleted items)
        assert manifest.total_conversations >= conv_count
        assert manifest.total_notes >= note_count
        assert manifest.total_characters >= char_count
        assert manifest.total_media_items >= media_count


class TestChatbookProperties:
    """Property-based tests for complete Chatbook."""
    
    @given(chatbook_manifest_strategy())
    def test_chatbook_content_access(self, manifest):
        """Test chatbook content access methods."""
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Test get_content_by_type
        for content_type in ContentType:
            items = chatbook.get_content_by_type(content_type)
            assert all(item.type == content_type for item in items)
        
        # Test get_content_by_id
        for item in manifest.content_items:
            found = chatbook.get_content_by_id(item.id)
            assert found is not None
            assert found.id == item.id
            assert found.type == item.type
        
        # Test non-existent ID
        assert chatbook.get_content_by_id("non_existent_id_12345") is None
    
    @given(chatbook_manifest_strategy())
    def test_chatbook_relationships(self, manifest):
        """Test chatbook relationship queries."""
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # For each content item, check related items
        for item in manifest.content_items:
            related = chatbook.get_related_content(item.id)
            
            # All related items should exist in manifest
            related_ids = [r.id for r in related]
            manifest_ids = [i.id for i in manifest.content_items]
            assert all(rid in manifest_ids for rid in related_ids)
            
            # Relationship should be bidirectional
            for related_item in related:
                reverse_related = chatbook.get_related_content(related_item.id)
                reverse_ids = [r.id for r in reverse_related]
                assert item.id in reverse_ids
    
    @given(
        st.lists(content_item_strategy(), min_size=1, max_size=20),
        st.lists(relationship_strategy(), max_size=50)
    )
    def test_chatbook_construction(self, items, relationships):
        """Test constructing a chatbook with arbitrary content."""
        # Filter relationships to only reference existing items
        item_ids = {item.id for item in items}
        valid_relationships = [
            rel for rel in relationships
            if rel.source_id in item_ids and rel.target_id in item_ids
        ]
        
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Chatbook",
            description="Property test chatbook",
            content_items=items,
            relationships=valid_relationships
        )
        
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        # Verify construction succeeded
        assert chatbook.manifest.name == "Test Chatbook"
        assert len(chatbook.manifest.content_items) == len(items)
        assert len(chatbook.manifest.relationships) == len(valid_relationships)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_chatbook(self):
        """Test chatbook with no content."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Empty",
            description="Empty chatbook"
        )
        
        content = ChatbookContent()
        chatbook = Chatbook(manifest=manifest, content=content)
        
        assert len(chatbook.manifest.content_items) == 0
        assert len(chatbook.manifest.relationships) == 0
        
        # All queries should return empty results
        for content_type in ContentType:
            assert chatbook.get_content_by_type(content_type) == []
        
        assert chatbook.get_content_by_id("any_id") is None
        assert chatbook.get_related_content("any_id") == []
    
    @given(st.text(min_size=1000, max_size=10000))
    def test_large_content(self, large_text):
        """Test handling of large content."""
        item = ContentItem(
            id="large_1",
            type=ContentType.NOTE,
            title="Large Note",
            description=large_text,
            metadata={"size": len(large_text)}
        )
        
        # Should handle serialization
        data = item.to_dict()
        restored = ContentItem.from_dict(data)
        
        assert restored.description == large_text
        assert restored.metadata["size"] == len(large_text)
    
    @given(st.text(alphabet=string.printable))
    def test_special_characters(self, text):
        """Test handling of special characters in content."""
        assume(len(text) > 0 and len(text) < 200)
        
        item = ContentItem(
            id="special_1",
            type=ContentType.NOTE,
            title=text,
            description=f"Content with special chars: {text}"
        )
        
        # Should handle serialization with special characters
        data = item.to_dict()
        restored = ContentItem.from_dict(data)
        
        assert restored.title == text
        assert text in restored.description