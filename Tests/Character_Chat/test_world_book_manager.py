"""
Tests for World Book Manager functionality.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


@pytest.fixture
def test_db(tmp_path, request):
    """Create a test database with proper cleanup and isolation."""
    # Use test name in db filename for better isolation
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
    db_path = tmp_path / f"test_world_books_{test_name}.db"
    
    # Clean up any existing files from previous runs
    for suffix in ["", "-wal", "-shm"]:
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not unlink {p}: {e}")
    
    # Just return the path, let each test create its own connection
    yield str(db_path)


@pytest.fixture
def wb_manager(test_db):
    """Create a WorldBookManager instance with fresh database."""
    db = CharactersRAGDB(test_db, "test_client")
    manager = WorldBookManager(db)
    yield manager
    db.close_connection()


class TestWorldBookManager:
    """Test cases for WorldBookManager CRUD operations."""
    
    def test_create_world_book(self, wb_manager):
        """Test creating a world book."""
        wb_id = wb_manager.create_world_book(
            name="Test World",
            description="A test world book",
            scan_depth=5,
            token_budget=1000,
            recursive_scanning=True
        )
        
        assert wb_id > 0
        
        # Verify it was created
        wb = wb_manager.get_world_book(wb_id)
        assert wb is not None
        assert wb['name'] == "Test World"
        assert wb['description'] == "A test world book"
        assert wb['scan_depth'] == 5
        assert wb['token_budget'] == 1000
        assert wb['recursive_scanning'] is True
        assert wb['enabled'] is True
    
    def test_create_world_book_duplicate_name(self, wb_manager):
        """Test creating world books with duplicate names."""
        wb_manager.create_world_book(name="Unique World")
        
        with pytest.raises(ConflictError):
            wb_manager.create_world_book(name="Unique World")
    
    def test_create_world_book_empty_name(self, wb_manager):
        """Test creating world book with empty name."""
        with pytest.raises(InputError):
            wb_manager.create_world_book(name="")
        
        with pytest.raises(InputError):
            wb_manager.create_world_book(name="   ")
    
    def test_update_world_book(self, wb_manager):
        """Test updating a world book."""
        wb_id = wb_manager.create_world_book(name="Original Name")
        
        # Update various fields
        success = wb_manager.update_world_book(
            wb_id,
            name="Updated Name",
            description="New description",
            scan_depth=10,
            enabled=False
        )
        
        assert success is True
        
        # Verify updates
        wb = wb_manager.get_world_book(wb_id)
        assert wb['name'] == "Updated Name"
        assert wb['description'] == "New description"
        assert wb['scan_depth'] == 10
        assert wb['enabled'] is False
        assert wb['version'] == 2  # Version should increment
    
    def test_update_world_book_optimistic_locking(self, wb_manager):
        """Test optimistic locking on updates."""
        wb_id = wb_manager.create_world_book(name="Test Book")
        
        # Update with correct version
        success = wb_manager.update_world_book(wb_id, description="Update 1", expected_version=1)
        assert success is True
        
        # Update with wrong version should fail
        with pytest.raises(ConflictError):
            wb_manager.update_world_book(wb_id, description="Update 2", expected_version=1)
    
    def test_delete_world_book(self, wb_manager):
        """Test soft deleting a world book."""
        wb_id = wb_manager.create_world_book(name="To Delete")
        
        success = wb_manager.delete_world_book(wb_id)
        assert success is True
        
        # Should not be found after deletion
        wb = wb_manager.get_world_book(wb_id)
        assert wb is None
    
    def test_list_world_books(self, wb_manager):
        """Test listing world books."""
        # Create some world books
        wb_manager.create_world_book(name="World A", enabled=True)
        wb_manager.create_world_book(name="World B", enabled=False)
        wb_manager.create_world_book(name="World C", enabled=True)
        
        # List all
        all_books = wb_manager.list_world_books(include_disabled=True)
        assert len(all_books) == 3
        
        # List only enabled
        enabled_books = wb_manager.list_world_books(include_disabled=False)
        assert len(enabled_books) == 2
        assert all(wb['enabled'] for wb in enabled_books)
    
    def test_create_world_book_entry(self, wb_manager):
        """Test creating world book entries."""
        wb_id = wb_manager.create_world_book(name="Entry Test World")
        
        entry_id = wb_manager.create_world_book_entry(
            world_book_id=wb_id,
            keys=["test", "testing"],
            content="This is test content",
            position="before_char",
            selective=True,
            secondary_keys=["specific"],
            case_sensitive=True
        )
        
        assert entry_id > 0
        
        # Verify entry
        entries = wb_manager.get_world_book_entries(wb_id)
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry['keys'] == ["test", "testing"]
        assert entry['content'] == "This is test content"
        assert entry['position'] == "before_char"
        assert entry['selective'] is True
        assert entry['secondary_keys'] == ["specific"]
        assert entry['case_sensitive'] is True
    
    def test_create_entry_validation(self, wb_manager):
        """Test entry creation validation."""
        wb_id = wb_manager.create_world_book(name="Validation Test")
        
        # Empty keys
        with pytest.raises(InputError):
            wb_manager.create_world_book_entry(wb_id, keys=[], content="Content")
        
        # Empty content
        with pytest.raises(InputError):
            wb_manager.create_world_book_entry(wb_id, keys=["key"], content="")
    
    def test_update_world_book_entry(self, wb_manager):
        """Test updating world book entries."""
        wb_id = wb_manager.create_world_book(name="Update Entry Test")
        entry_id = wb_manager.create_world_book_entry(
            wb_id, 
            keys=["original"],
            content="Original content"
        )
        
        # Update various fields
        success = wb_manager.update_world_book_entry(
            entry_id,
            keys=["updated", "new"],
            content="Updated content",
            enabled=False,
            position="after_char"
        )
        
        assert success is True
        
        # Verify updates
        entries = wb_manager.get_world_book_entries(wb_id)
        entry = entries[0]
        assert entry['keys'] == ["updated", "new"]
        assert entry['content'] == "Updated content"
        assert entry['enabled'] is False
        assert entry['position'] == "after_char"
    
    def test_delete_world_book_entry(self, wb_manager):
        """Test deleting world book entries."""
        wb_id = wb_manager.create_world_book(name="Delete Entry Test")
        entry_id = wb_manager.create_world_book_entry(wb_id, keys=["delete"], content="To delete")
        
        success = wb_manager.delete_world_book_entry(entry_id)
        assert success is True
        
        # Verify deletion
        entries = wb_manager.get_world_book_entries(wb_id)
        assert len(entries) == 0
    
    def test_conversation_associations(self, wb_manager):
        """Test associating world books with conversations."""
        # Get the database from the world book manager
        db = wb_manager.db
        
        # Create a conversation first
        # First create a character for the conversation
        character_id = db.add_character_card({
            'name': 'Test Character',
            'description': 'A test character',
            'first_message': 'Hello!',
            'personality': 'Friendly'
        })
        
        # Now create a conversation with required character_id
        conv_id = db.add_conversation({
            'character_id': character_id,
            'title': 'Test Conversation'
        })
        
        # Create world books
        wb1_id = wb_manager.create_world_book(name="Book 1")
        wb2_id = wb_manager.create_world_book(name="Book 2")
        
        # Associate with conversation
        wb_manager.associate_world_book_with_conversation(conv_id, wb1_id, priority=1)
        wb_manager.associate_world_book_with_conversation(conv_id, wb2_id, priority=2)
        
        # Get associated books
        books = wb_manager.get_world_books_for_conversation(conv_id)
        assert len(books) == 2
        
        # Should be ordered by priority (descending)
        assert books[0]['name'] == "Book 2"  # Higher priority
        assert books[1]['name'] == "Book 1"
        
        # Disassociate one
        wb_manager.disassociate_world_book_from_conversation(conv_id, wb1_id)
        
        books = wb_manager.get_world_books_for_conversation(conv_id)
        assert len(books) == 1
        assert books[0]['name'] == "Book 2"
    
    def test_export_world_book(self, wb_manager):
        """Test exporting a world book."""
        # Create world book with entries
        wb_id = wb_manager.create_world_book(
            name="Export Test",
            description="Book for export",
            scan_depth=7,
            token_budget=2000,
            recursive_scanning=True
        )
        
        wb_manager.create_world_book_entry(
            wb_id,
            keys=["magic", "spell"],
            content="Magic is powerful",
            position="before_char",
            selective=True,
            secondary_keys=["wizard"]
        )
        
        wb_manager.create_world_book_entry(
            wb_id,
            keys=["sword"],
            content="A legendary blade",
            position="after_char"
        )
        
        # Export
        export_data = wb_manager.export_world_book(wb_id)
        
        assert export_data['name'] == "Export Test"
        assert export_data['description'] == "Book for export"
        assert export_data['scan_depth'] == 7
        assert export_data['token_budget'] == 2000
        assert export_data['recursive_scanning'] is True
        assert len(export_data['entries']) == 2
        
        # Check entries
        entry1 = export_data['entries'][0]
        assert entry1['keys'] == ["magic", "spell"]
        assert entry1['content'] == "Magic is powerful"
        assert entry1['selective'] is True
        assert entry1['secondary_keys'] == ["wizard"]
    
    def test_import_world_book(self, wb_manager):
        """Test importing a world book."""
        import_data = {
            'name': 'Imported Book',
            'description': 'Book from import',
            'scan_depth': 5,
            'token_budget': 1500,
            'recursive_scanning': False,
            'entries': [
                {
                    'keys': ['dragon', 'wyrm'],
                    'content': 'Ancient dragons rule the skies',
                    'enabled': True,
                    'position': 'before_char',
                    'insertion_order': 0,
                    'selective': False,
                    'secondary_keys': [],
                    'case_sensitive': False
                },
                {
                    'keys': ['castle'],
                    'content': 'The castle stands tall',
                    'enabled': True,
                    'position': 'at_start',
                    'insertion_order': 1,
                    'selective': True,
                    'secondary_keys': ['fortress'],
                    'case_sensitive': True
                }
            ]
        }
        
        wb_id = wb_manager.import_world_book(import_data)
        
        # Verify import
        wb = wb_manager.get_world_book(wb_id)
        assert wb['name'] == 'Imported Book'
        assert wb['scan_depth'] == 5
        assert wb['token_budget'] == 1500
        
        entries = wb_manager.get_world_book_entries(wb_id)
        assert len(entries) == 2
        
        # Check first entry
        assert entries[0]['keys'] == ['dragon', 'wyrm']
        assert entries[0]['content'] == 'Ancient dragons rule the skies'
        
        # Check second entry
        assert entries[1]['keys'] == ['castle']
        assert entries[1]['selective'] is True
        assert entries[1]['secondary_keys'] == ['fortress']
    
    def test_import_with_name_override(self, wb_manager):
        """Test importing with name override to avoid conflicts."""
        # Create existing book
        wb_manager.create_world_book(name="Existing Book")
        
        # Import with same name should work with override
        import_data = {'name': 'Existing Book', 'entries': []}
        
        wb_id = wb_manager.import_world_book(import_data, name_override="Existing Book (Copy)")
        
        wb = wb_manager.get_world_book(wb_id)
        assert wb['name'] == "Existing Book (Copy)"


class TestWorldInfoProcessorIntegration:
    """Test integration between world books and world info processor."""
    
    def test_processor_with_world_books(self, wb_manager):
        """Test world info processor with standalone world books."""
        # Create world books with entries
        wb1_id = wb_manager.create_world_book(name="Book 1", scan_depth=3, token_budget=500)
        wb_manager.create_world_book_entry(
            wb1_id,
            keys=["kingdom"],
            content="The kingdom is vast",
            position="before_char"
        )
        
        wb2_id = wb_manager.create_world_book(name="Book 2", scan_depth=5, token_budget=1000)
        wb_manager.create_world_book_entry(
            wb2_id,
            keys=["hero"],
            content="The hero arrives",
            position="after_char"
        )
        
        # Prepare world books data as it would come from the manager
        world_books = [
            {
                'name': 'Book 1',
                'scan_depth': 3,
                'token_budget': 500,
                'recursive_scanning': False,
                'enabled': True,
                'priority': 1,
                'entries': wb_manager.get_world_book_entries(wb1_id)
            },
            {
                'name': 'Book 2',
                'scan_depth': 5,
                'token_budget': 1000,
                'recursive_scanning': False,
                'enabled': True,
                'priority': 2,
                'entries': wb_manager.get_world_book_entries(wb2_id)
            }
        ]
        
        # Create processor with world books
        processor = WorldInfoProcessor(world_books=world_books)
        
        # Should have both entries
        assert len(processor.entries) == 2
        
        # Settings should use max values
        assert processor.scan_depth == 5  # Max of 3 and 5
        assert processor.token_budget == 1000  # Max of 500 and 1000
        
        # Test processing
        result = processor.process_messages(
            "The hero enters the kingdom",
            []
        )
        
        assert len(result['matched_entries']) == 2
        assert 'before_char' in result['injections']
        assert 'after_char' in result['injections']
    
    def test_processor_with_character_and_world_books(self, wb_manager):
        """Test processor with both character book and world books."""
        # Character data with embedded world info
        character_data = {
            'name': 'Test Character',
            'extensions': {
                'character_book': {
                    'entries': [
                        {
                            'keys': ['character trait'],
                            'content': 'Character specific info',
                            'enabled': True,
                            'position': 'before_char'
                        }
                    ]
                }
            }
        }
        
        # World book
        wb_id = wb_manager.create_world_book(name="Shared Lore")
        wb_manager.create_world_book_entry(
            wb_id,
            keys=["world"],
            content="World background info",
            position="at_start"
        )
        
        world_books = [{
            'name': 'Shared Lore',
            'enabled': True,
            'priority': 0,
            'entries': wb_manager.get_world_book_entries(wb_id)
        }]
        
        # Create processor with both sources
        processor = WorldInfoProcessor(character_data=character_data, world_books=world_books)
        
        assert len(processor.entries) == 2
        
        # Test processing
        result = processor.process_messages(
            "Tell me about the character trait in this world",
            []
        )
        
        assert len(result['matched_entries']) == 2
        assert 'before_char' in result['injections']
        assert 'at_start' in result['injections']