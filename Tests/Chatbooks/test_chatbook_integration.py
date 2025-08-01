# test_chatbook_integration.py
# Integration tests for chatbook export/import cycle

import pytest
import tempfile
import shutil
import sqlite3
import json
import zipfile
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ContentType, ChatbookVersion
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase


class TestChatbookIntegration:
    """Integration tests for complete chatbook export/import cycle."""
    
    @pytest.fixture
    def test_environment(self):
        """Create a complete test environment with databases."""
        temp_dir = tempfile.mkdtemp()
        env_dir = Path(temp_dir)
        
        # Create database directory
        db_dir = env_dir / "databases"
        db_dir.mkdir()
        
        # Create output directory
        output_dir = env_dir / "chatbooks"
        output_dir.mkdir()
        
        # Initialize databases
        db_paths = {
            'chachanotes': str(db_dir / 'chachanotes.db'),
            'prompts': str(db_dir / 'prompts.db'),
            'media': str(db_dir / 'media.db')
        }
        
        # Create and populate databases
        self._setup_test_databases(db_paths)
        
        yield env_dir, db_paths, output_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _setup_test_databases(self, db_paths):
        """Create and populate test databases."""
        # Setup ChaChaNotes database
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'test_setup')
        
        # Add test conversations
        conv1_id = chachanotes_db.add_conversation("Research Discussion", media_id=None)
        conv2_id = chachanotes_db.add_conversation("Project Planning", media_id=None)
        
        # Add messages
        chachanotes_db.add_message(conv1_id, "user", "What do you know about blue whales?")
        chachanotes_db.add_message(conv1_id, "assistant", "Blue whales are the largest animals ever known...")
        chachanotes_db.add_message(conv2_id, "user", "Let's plan the new feature")
        chachanotes_db.add_message(conv2_id, "assistant", "I'll help you plan that...")
        
        # Add test notes
        chachanotes_db.add_note(
            title="Blue Whale Research",
            content="# Blue Whale Research\n\nBlue whales are magnificent creatures...",
            keywords="whales,marine biology,research"
        )
        chachanotes_db.add_note(
            title="Project Requirements",
            content="# Requirements\n\n1. User authentication\n2. Data export",
            keywords="project,requirements"
        )
        
        # Add test character
        chachanotes_db.create_character(
            name="Research Assistant",
            description="Helps with research tasks",
            personality="Analytical and thorough",
            scenario="Research environment",
            greeting_message="Hello! Ready to help with research.",
            example_messages="User: What should we research?\nAssistant: Let's explore the topic systematically."
        )
        
        # Setup Prompts database
        prompts_db = PromptsDatabase(db_paths['prompts'], 'test_setup')
        prompts_db.add_prompt(
            name="Research Prompt",
            author="Test User",
            details="For research tasks",
            system_prompt="You are a research assistant. Be thorough and cite sources.",
            user_prompt=None
        )
    
    def test_full_export_import_cycle(self, test_environment):
        """Test complete export and import cycle."""
        env_dir, db_paths, output_dir = test_environment
        
        # Create chatbook
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "test_chatbook.zip"
        
        # Get IDs for export
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'test_export')
        conversations = chachanotes_db.list_all_active_conversations()
        notes = chachanotes_db.list_notes()
        characters = chachanotes_db.list_all_characters()
        
        conv_ids = [str(c['id']) for c in conversations]
        note_ids = [str(n['id']) for n in notes]
        char_ids = [str(c['id']) for c in characters]
        
        # Export chatbook
        success, message = creator.create_chatbook(
            name="Integration Test Chatbook",
            description="Testing full export/import cycle",
            content_selections={
                ContentType.CONVERSATION: conv_ids,
                ContentType.NOTE: note_ids,
                ContentType.CHARACTER: char_ids
            },
            output_path=output_path,
            author="Test Suite",
            tags=["test", "integration"],
            categories=["testing"]
        )
        
        assert success is True
        assert output_path.exists()
        
        # Verify chatbook contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert 'manifest.json' in files
            assert 'README.md' in files
            assert any('conversations' in f for f in files)
            assert any('notes' in f for f in files)
            assert any('characters' in f for f in files)
            
            # Check manifest
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Integration Test Chatbook"
            assert len(manifest_data['content_items']) == len(conv_ids) + len(note_ids) + len(char_ids)
        
        # Create new database environment for import
        import_db_dir = env_dir / "import_databases"
        import_db_dir.mkdir()
        
        import_db_paths = {
            'chachanotes': str(import_db_dir / 'chachanotes_import.db'),
            'prompts': str(import_db_dir / 'prompts_import.db'),
            'media': str(import_db_dir / 'media_import.db')
        }
        
        # Initialize empty databases
        CharactersRAGDB(import_db_paths['chachanotes'], 'import_init')
        PromptsDatabase(import_db_paths['prompts'], 'import_init')
        
        # Import chatbook
        importer = ChatbookImporter(import_db_paths)
        import_success, import_status = importer.import_chatbook(
            chatbook_path=output_path,
            conflict_resolution=ConflictResolution.RENAME,
            prefix_imported=True
        )
        
        assert import_success is True
        assert import_status.successful_items > 0
        assert import_status.failed_items == 0
        
        # Verify imported content
        import_db = CharactersRAGDB(import_db_paths['chachanotes'], 'verify_import')
        
        # Check conversations
        imported_convs = import_db.list_all_active_conversations()
        assert len(imported_convs) == len(conversations)
        assert all('[Imported]' in c['conversation_name'] for c in imported_convs)
        
        # Check notes
        imported_notes = import_db.list_notes()
        assert len(imported_notes) == len(notes)
        assert all('[Imported]' in n['title'] for n in imported_notes)
        
        # Check characters
        imported_chars = import_db.list_all_characters()
        assert len(imported_chars) == len(characters)
    
    def test_selective_export_import(self, test_environment):
        """Test exporting and importing only selected content."""
        env_dir, db_paths, output_dir = test_environment
        
        # Get specific items to export
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'selective_test')
        conversations = chachanotes_db.list_all_active_conversations()
        notes = chachanotes_db.list_notes()
        
        # Export only first conversation and first note
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "selective_chatbook.zip"
        
        success, message = creator.create_chatbook(
            name="Selective Export",
            description="Only some content",
            content_selections={
                ContentType.CONVERSATION: [str(conversations[0]['id'])],
                ContentType.NOTE: [str(notes[0]['id'])]
            },
            output_path=output_path
        )
        
        assert success is True
        
        # Verify only selected content is in chatbook
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            assert len(manifest_data['content_items']) == 2
            assert manifest_data['statistics']['total_conversations'] == 1
            assert manifest_data['statistics']['total_notes'] == 1
    
    def test_conflict_resolution(self, test_environment):
        """Test import conflict resolution."""
        env_dir, db_paths, output_dir = test_environment
        
        # Create and export a chatbook
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "conflict_test.zip"
        
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'conflict_test')
        notes = chachanotes_db.list_notes()
        
        success, message = creator.create_chatbook(
            name="Conflict Test",
            description="Testing conflicts",
            content_selections={
                ContentType.NOTE: [str(notes[0]['id'])]
            },
            output_path=output_path
        )
        
        assert success is True
        
        # Import once
        importer = ChatbookImporter(db_paths)
        success1, status1 = importer.import_chatbook(
            chatbook_path=output_path,
            conflict_resolution=ConflictResolution.RENAME,
            prefix_imported=False
        )
        
        assert success1 is True
        assert status1.successful_items == 1
        
        # Import again - should handle conflict
        success2, status2 = importer.import_chatbook(
            chatbook_path=output_path,
            conflict_resolution=ConflictResolution.RENAME,
            prefix_imported=False
        )
        
        assert success2 is True
        assert status2.successful_items == 1
        
        # Check that we have multiple versions
        all_notes = chachanotes_db.list_notes()
        note_titles = [n['title'] for n in all_notes]
        
        # Should have original plus renamed imports
        assert len([t for t in note_titles if 'Blue Whale Research' in t]) >= 2
    
    def test_large_chatbook_performance(self, test_environment):
        """Test performance with larger chatbooks."""
        env_dir, db_paths, output_dir = test_environment
        
        # Add more content
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'perf_test')
        
        # Add 100 conversations with messages
        conv_ids = []
        for i in range(100):
            conv_id = chachanotes_db.add_conversation(f"Conversation {i}", media_id=None)
            conv_ids.append(str(conv_id))
            
            # Add 10 messages per conversation
            for j in range(10):
                role = "user" if j % 2 == 0 else "assistant"
                chachanotes_db.add_message(conv_id, role, f"Message {j} in conversation {i}")
        
        # Create large chatbook
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "large_chatbook.zip"
        
        start_time = datetime.now()
        success, message = creator.create_chatbook(
            name="Large Chatbook",
            description="Performance testing",
            content_selections={
                ContentType.CONVERSATION: conv_ids
            },
            output_path=output_path
        )
        export_time = (datetime.now() - start_time).total_seconds()
        
        assert success is True
        print(f"Export time for 100 conversations: {export_time:.2f} seconds")
        
        # Test import performance
        import_db_paths = {k: v.replace('.db', '_import.db') for k, v in db_paths.items()}
        CharactersRAGDB(import_db_paths['chachanotes'], 'import_init')
        
        importer = ChatbookImporter(import_db_paths)
        start_time = datetime.now()
        import_success, import_status = importer.import_chatbook(
            chatbook_path=output_path,
            conflict_resolution=ConflictResolution.RENAME
        )
        import_time = (datetime.now() - start_time).total_seconds()
        
        assert import_success is True
        assert import_status.successful_items == 100
        print(f"Import time for 100 conversations: {import_time:.2f} seconds")
        
        # Both operations should complete reasonably quickly
        assert export_time < 30  # 30 seconds max
        assert import_time < 30  # 30 seconds max
    
    def test_metadata_preservation(self, test_environment):
        """Test that metadata is preserved through export/import."""
        env_dir, db_paths, output_dir = test_environment
        
        # Create chatbook with specific metadata
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "metadata_test.zip"
        
        test_tags = ["research", "whales", "marine-biology"]
        test_categories = ["science", "nature"]
        test_author = "Marine Researcher"
        test_license = "CC BY-SA 4.0"
        
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'metadata_test')
        notes = chachanotes_db.list_notes()
        
        success, message = creator.create_chatbook(
            name="Metadata Test",
            description="Testing metadata preservation",
            content_selections={
                ContentType.NOTE: [str(notes[0]['id'])]
            },
            output_path=output_path,
            author=test_author,
            tags=test_tags,
            categories=test_categories
        )
        
        assert success is True
        
        # Read and verify metadata
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            
            assert manifest_data['author'] == test_author
            assert manifest_data['tags'] == test_tags
            assert manifest_data['categories'] == test_categories
            assert manifest_data['language'] == 'en'
            
            # Check timestamps
            assert 'created_at' in manifest_data
            assert 'updated_at' in manifest_data
            
            # Verify dates are valid ISO format
            datetime.fromisoformat(manifest_data['created_at'])
            datetime.fromisoformat(manifest_data['updated_at'])
    
    def test_relationship_discovery(self, test_environment):
        """Test that relationships between content are discovered."""
        env_dir, db_paths, output_dir = test_environment
        
        # Create conversation with character
        chachanotes_db = CharactersRAGDB(db_paths['chachanotes'], 'relationship_test')
        
        # Get character ID
        characters = chachanotes_db.list_all_characters()
        char_id = characters[0]['id']
        
        # Create conversation using character
        conv_id = chachanotes_db.add_conversation(
            "Character Conversation",
            media_id=None,
            character_id=char_id
        )
        
        # Export both
        creator = ChatbookCreator(db_paths)
        output_path = output_dir / "relationship_test.zip"
        
        success, message = creator.create_chatbook(
            name="Relationship Test",
            description="Testing relationship discovery",
            content_selections={
                ContentType.CONVERSATION: [str(conv_id)],
                ContentType.CHARACTER: [str(char_id)]
            },
            output_path=output_path
        )
        
        assert success is True
        
        # Check relationships in manifest
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            
            assert len(manifest_data['relationships']) > 0
            
            # Find the conversation-character relationship
            rel = manifest_data['relationships'][0]
            assert rel['source_id'] == str(conv_id)
            assert rel['target_id'] == str(char_id)
            assert rel['relationship_type'] == 'uses_character'