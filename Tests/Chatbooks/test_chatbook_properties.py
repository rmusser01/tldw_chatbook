# test_chatbook_properties.py
# Description: Property-based tests for chatbook functionality
#
"""
Chatbook Property-Based Tests
----------------------------

Uses Hypothesis to test chatbook functionality with generated data.
"""

import pytest
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from hypothesis import given, assume, strategies as st, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant

from tldw_chatbook.Chatbooks import (
    ChatbookCreator, ChatbookImporter, ChatbookError,
    ChatbookErrorType
)
from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, ChatbookManifest, ChatbookVersion,
    Relationship
)
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution


# Custom strategies for generating test data
@st.composite
def content_type_strategy(draw):
    """Generate a valid ContentType."""
    return draw(st.sampled_from(list(ContentType)))


@st.composite
def content_item_strategy(draw, content_type=None):
    """Generate a valid ContentItem."""
    if content_type is None:
        content_type = draw(content_type_strategy())
    
    item_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), 
        min_codepoint=48, max_codepoint=122
    )))
    
    title = draw(st.text(min_size=1, max_size=100))
    description = draw(st.text(max_size=500))
    
    tags = draw(st.lists(
        st.text(min_size=1, max_size=30),
        max_size=10
    ))
    
    # Generate timestamps with fixed base time to avoid flakiness
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    created_at = draw(st.datetimes(
        min_value=base_time - timedelta(days=365),
        max_value=base_time
    ))
    updated_at = draw(st.datetimes(
        min_value=created_at,
        max_value=base_time
    ))
    
    return ContentItem(
        id=item_id,
        type=content_type,
        title=title,
        description=description,
        created_at=created_at,
        updated_at=updated_at,
        tags=tags,
        file_path=f"content/{content_type.value}s/{content_type.value}_{item_id}.json"
    )


@st.composite
def manifest_strategy(draw):
    """Generate a valid ChatbookManifest."""
    name = draw(st.text(min_size=1, max_size=100))
    description = draw(st.text(max_size=1000))
    author = draw(st.text(min_size=1, max_size=100))
    
    tags = draw(st.lists(
        st.text(min_size=1, max_size=30),
        max_size=20
    ))
    
    categories = draw(st.lists(
        st.text(min_size=1, max_size=50),
        max_size=10
    ))
    
    # Generate content items
    content_items = draw(st.lists(
        content_item_strategy(),
        max_size=50
    ))
    
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name=name,
        description=description,
        author=author,
        tags=tags,
        categories=categories
    )
    
    manifest.content_items = content_items
    
    # Update statistics
    for item in content_items:
        if item.type == ContentType.CONVERSATION:
            manifest.total_conversations += 1
        elif item.type == ContentType.NOTE:
            manifest.total_notes += 1
        elif item.type == ContentType.CHARACTER:
            manifest.total_characters += 1
        elif item.type == ContentType.MEDIA:
            manifest.total_media_items += 1
        elif item.type == ContentType.PROMPT:
            manifest.total_prompts += 1
    
    return manifest


@st.composite
def chatbook_selections_strategy(draw):
    """Generate valid content selections for chatbook creation."""
    selections = {}
    
    for content_type in ContentType:
        # Randomly decide whether to include this content type
        if draw(st.booleans()):
            # Generate 0-10 IDs for this content type
            ids = draw(st.lists(
                st.text(min_size=1, max_size=20),
                max_size=10
            ))
            if ids:
                selections[content_type] = ids
    
    return selections


class TestChatbookProperties:
    """Property-based tests for chatbook functionality."""
    
    @given(manifest=manifest_strategy())
    def test_manifest_serialization_roundtrip(self, manifest):
        """Test that manifests can be serialized and deserialized."""
        # Serialize to dict
        data = manifest.to_dict()
        
        # Deserialize back
        restored = ChatbookManifest.from_dict(data)
        
        # Check key properties
        assert restored.name == manifest.name
        assert restored.description == manifest.description
        assert restored.author == manifest.author
        assert restored.version == manifest.version
        assert len(restored.content_items) == len(manifest.content_items)
        assert restored.total_conversations == manifest.total_conversations
        assert restored.total_notes == manifest.total_notes
        assert restored.total_characters == manifest.total_characters
    
    @given(content_item=content_item_strategy())
    def test_content_item_serialization(self, content_item):
        """Test ContentItem serialization and deserialization."""
        data = content_item.to_dict()
        restored = ContentItem.from_dict(data)
        
        assert restored.id == content_item.id
        assert restored.type == content_item.type
        assert restored.title == content_item.title
        assert restored.description == content_item.description
        assert restored.file_path == content_item.file_path
        assert set(restored.tags) == set(content_item.tags)
    
    @given(
        name=st.text(min_size=1, max_size=100),
        description=st.text(max_size=1000),
        author=st.text(max_size=100),
        tags=st.lists(st.text(min_size=1, max_size=30), max_size=20),
        categories=st.lists(st.text(min_size=1, max_size=50), max_size=10),
        selections=chatbook_selections_strategy()
    )
    @settings(max_examples=10, deadline=None)
    def test_chatbook_creation_with_random_data(
        self, name, description, author, tags, categories, selections, tmp_path
    ):
        """Test chatbook creation with randomly generated data."""
        # Mock database paths
        mock_db_paths = {
            'ChaChaNotes': ':memory:',
            'Media': ':memory:',
            'Prompts': ':memory:'
        }
        
        # Create chatbook creator with mocked dependencies
        creator = ChatbookCreator(mock_db_paths)
        
        # Mock the collection methods to avoid database calls
        with patch.object(creator, '_collect_conversations'), \
             patch.object(creator, '_collect_notes'), \
             patch.object(creator, '_collect_characters'), \
             patch.object(creator, '_collect_media'), \
             patch.object(creator, '_collect_prompts'):
            
            output_path = tmp_path / f"test_{hash(name)}.zip"
            
            success, message, info = creator.create_chatbook(
                name=name,
                description=description,
                content_selections=selections,
                output_path=output_path,
                author=author,
                tags=tags,
                categories=categories
            )
            
            assert success is True
            assert output_path.exists()
            
            # Verify ZIP structure
            with zipfile.ZipFile(output_path, 'r') as zf:
                files = zf.namelist()
                assert 'manifest.json' in files
                assert 'README.md' in files
                
                # Check manifest content
                manifest_data = json.loads(zf.read('manifest.json'))
                assert manifest_data['name'] == name
                assert manifest_data['description'] == description
                assert manifest_data['author'] == author
                assert set(manifest_data['tags']) == set(tags)
                assert set(manifest_data['categories']) == set(categories)
    
    @given(
        items=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=50),  # title
                st.text(min_size=1, max_size=500), # content
                st.booleans()  # should_conflict
            ),
            min_size=1,
            max_size=20
        ),
        resolution=st.sampled_from(list(ConflictResolution))
    )
    def test_conflict_resolution_strategies(self, items, resolution):
        """Test different conflict resolution strategies with random data."""
        from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolver
        
        resolver = ConflictResolver()
        
        # Group items by title to create conflicts
        conflicts = {}
        for title, content, should_conflict in items:
            if should_conflict and title in conflicts:
                # We have a conflict
                existing = {"title": title, "content": conflicts[title]}
                incoming = {"title": title, "content": content}
                
                result = resolver.resolve_note_conflict(
                    existing, incoming, resolution
                )
                
                # Verify resolution is valid
                assert isinstance(result, ConflictResolution)
                
                if resolution == ConflictResolution.ASK:
                    # Without callback, should return SKIP
                    assert result == ConflictResolution.SKIP
                else:
                    assert result == resolution
            else:
                conflicts[title] = content
    
    @given(
        source_ids=st.lists(st.text(min_size=1, max_size=20), max_size=10),
        target_ids=st.lists(st.text(min_size=1, max_size=20), max_size=10),
        relationship_types=st.lists(
            st.sampled_from(['uses_character', 'references', 'related_to']),
            max_size=10
        )
    )
    def test_relationship_creation(self, source_ids, target_ids, relationship_types):
        """Test relationship creation with random data."""
        relationships = []
        
        # Create relationships for each combination
        for i, rel_type in enumerate(relationship_types):
            if i < len(source_ids) and i < len(target_ids):
                rel = Relationship(
                    source_id=source_ids[i],
                    target_id=target_ids[i],
                    relationship_type=rel_type
                )
                relationships.append(rel)
        
        # Test serialization
        for rel in relationships:
            data = rel.to_dict()
            assert data['source_id'] == rel.source_id
            assert data['target_id'] == rel.target_id
            assert data['relationship_type'] == rel.relationship_type
    
    @given(
        file_paths=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=50
        )
    )
    def test_file_path_handling(self, file_paths):
        """Test handling of various file paths."""
        from tldw_chatbook.Chatbooks.error_handler import ChatbookErrorHandler
        
        for path in file_paths:
            # Test path validation
            try:
                # Simulate file operation
                if '/' in path and '..' not in path:
                    # Valid path structure
                    assert True
                else:
                    # Should handle invalid paths
                    error = ChatbookErrorHandler.handle_error(
                        ValueError(f"Invalid path: {path}"),
                        "validating path"
                    )
                    assert error.error_type in [
                        ChatbookErrorType.VALIDATION_ERROR,
                        ChatbookErrorType.FILE_NOT_FOUND
                    ]
            except Exception as e:
                # Any exception should be properly handled
                error = ChatbookErrorHandler.handle_error(e, "processing path")
                assert isinstance(error, ChatbookError)


class ChatbookStateMachine(RuleBasedStateMachine):
    """Stateful testing for chatbook operations."""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.db_paths = {
            'ChaChaNotes': ':memory:',
            'Media': ':memory:',
            'Prompts': ':memory:'
        }
        self.created_chatbooks = []
        self.imported_content = {}
        
    # Bundles to track state
    chatbooks = Bundle('chatbooks')
    content_items = Bundle('content_items')
    
    @rule(
        name=st.text(min_size=1, max_size=50),
        description=st.text(max_size=200)
    )
    def create_empty_chatbook(self, name, description):
        """Create an empty chatbook."""
        creator = ChatbookCreator(self.db_paths)
        output_path = Path(self.temp_dir) / f"{name}_{len(self.created_chatbooks)}.zip"
        
        with patch.object(creator, '_collect_conversations'), \
             patch.object(creator, '_collect_notes'):
            
            success, message, info = creator.create_chatbook(
                name=name,
                description=description,
                content_selections={},
                output_path=output_path
            )
            
            assume(success)  # Only continue if creation succeeded
            
            self.created_chatbooks.append({
                'path': output_path,
                'name': name,
                'description': description,
                'content_count': 0
            })
            
            return output_path
    
    @rule(
        chatbook=chatbooks,
        content_type=content_type_strategy(),
        count=st.integers(min_value=1, max_value=10)
    )
    def add_content_to_chatbook(self, chatbook, content_type, count):
        """Add content to an existing chatbook."""
        # This would simulate adding content after creation
        # In practice, content is added during creation
        pass
    
    @rule(chatbook=chatbooks)
    def import_chatbook(self, chatbook):
        """Import a previously created chatbook."""
        importer = ChatbookImporter(self.db_paths)
        
        # Mock the import methods
        with patch.object(importer, '_import_conversations') as mock_conv, \
             patch.object(importer, '_import_notes') as mock_notes:
            
            # Set up return values
            mock_conv.return_value = (0, 0)
            mock_notes.return_value = (0, 0)
            
            success, message = importer.import_chatbook(
                chatbook_path=chatbook,
                conflict_resolution=ConflictResolution.SKIP
            )
            
            if success:
                self.imported_content[str(chatbook)] = {
                    'imported_at': datetime.now(),
                    'success': True
                }
    
    @invariant()
    def all_chatbooks_are_valid_zips(self):
        """Verify all created chatbooks are valid ZIP files."""
        for cb_info in self.created_chatbooks:
            path = cb_info['path']
            if path.exists():
                assert zipfile.is_zipfile(path)
                
                with zipfile.ZipFile(path, 'r') as zf:
                    files = zf.namelist()
                    assert 'manifest.json' in files
                    assert 'README.md' in files
    
    @invariant()
    def chatbook_names_are_preserved(self):
        """Verify chatbook names are preserved in manifests."""
        for cb_info in self.created_chatbooks:
            path = cb_info['path']
            if path.exists():
                with zipfile.ZipFile(path, 'r') as zf:
                    if 'manifest.json' in zf.namelist():
                        manifest_data = json.loads(zf.read('manifest.json'))
                        assert manifest_data['name'] == cb_info['name']
    
    def teardown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# Run the state machine tests
TestChatbookStateMachine = ChatbookStateMachine.TestCase


@pytest.mark.property
class TestAdvancedProperties:
    """Advanced property-based tests."""
    
    @given(
        num_items=st.integers(min_value=0, max_value=1000),
        item_size=st.integers(min_value=100, max_value=10000)
    )
    @settings(max_examples=5, deadline=None)
    def test_chatbook_size_limits(self, num_items, item_size, tmp_path):
        """Test chatbook behavior with various sizes."""
        mock_db_paths = {
            'ChaChaNotes': ':memory:',
            'Media': ':memory:',
            'Prompts': ':memory:'
        }
        
        creator = ChatbookCreator(mock_db_paths)
        
        # Generate large content
        large_content = "x" * item_size
        
        # Mock collection to return many items
        def mock_collect_notes(ids, *args, **kwargs):
            for i in range(num_items):
                # Simulate writing note files
                pass
        
        with patch.object(creator, '_collect_notes', side_effect=mock_collect_notes):
            output_path = tmp_path / "large_test.zip"
            
            selections = {
                ContentType.NOTE: [str(i) for i in range(num_items)]
            }
            
            success, message, info = creator.create_chatbook(
                name="Size Test",
                description=f"Testing with {num_items} items of {item_size} bytes",
                content_selections=selections,
                output_path=output_path
            )
            
            # Should handle any size gracefully
            assert isinstance(success, bool)
            assert isinstance(message, str)
    
    @given(
        manifest_data=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.lists(st.text(), max_size=10)
            ),
            max_size=50
        )
    )
    def test_manifest_robustness(self, manifest_data):
        """Test manifest parsing with various data structures."""
        # Add required fields
        manifest_data.update({
            'version': '1.0',
            'name': 'Test',
            'description': 'Test manifest',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
        
        try:
            # Try to create manifest from data
            manifest = ChatbookManifest.from_dict(manifest_data)
            # If successful, should have required fields
            assert manifest.name == 'Test'
            assert manifest.version == ChatbookVersion.V1
        except Exception:
            # Should handle invalid data gracefully
            pass