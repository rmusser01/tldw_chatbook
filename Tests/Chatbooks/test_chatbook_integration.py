# test_chatbook_integration.py
# Integration tests for chatbook functionality

import pytest
import json
import zipfile
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase


@pytest.mark.integration
class TestChatbookIntegration:
    """Integration tests for complete chatbook workflow."""
    
    @pytest.fixture
    def setup_test_databases(self, tmp_path):
        """Setup test databases with proper schema and test data."""
        db_dir = tmp_path / "databases"
        db_dir.mkdir()
        
        db_paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize ChaChaNotes database
        chacha_db = CharactersRAGDB(db_paths['ChaChaNotes'], "test_client")
        
        # Add test character first (required for conversations)
        char_data = {
            "name": "AI Assistant",
            "description": "Helpful AI assistant",
            "personality": "Professional and informative",
            "scenario": "",
            "greeting_message": "Hello! How can I help you?",
            "example_messages": "",
            "version": 1
        }
        char_id = chacha_db.add_character_card(char_data)
        
        # Add test conversations
        conv1_data = {
            "title": "Export Test Conv 1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "root_id": "conv1_root",
            "character_id": char_id
        }
        conv1_id = chacha_db.add_conversation(conv1_data)
        if conv1_id:
            chacha_db.add_message({
                'conversation_id': conv1_id,
                'sender': 'user',
                'content': 'What is machine learning?'
            })
            chacha_db.add_message({
                'conversation_id': conv1_id,
                'sender': 'assistant',
                'content': 'Machine learning is...'
            })
        
        conv2_data = {
            "title": "Export Test Conv 2",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "root_id": "conv2_root",
            "character_id": char_id  # All conversations require a character
        }
        conv2_id = chacha_db.add_conversation(conv2_data)
        if conv2_id:
            chacha_db.add_message({
                'conversation_id': conv2_id,
                'sender': 'user',
                'content': 'Hello'
            })
            chacha_db.add_message({
                'conversation_id': conv2_id,
                'sender': 'assistant',
                'content': 'Hi there!'
            })
        
        # Add test notes
        note1_id = chacha_db.add_note(
            title="ML Notes",
            content="Notes about machine learning concepts"
        )
        
        note2_id = chacha_db.add_note(
            title="Python Tips",
            content="Useful Python programming tips"
        )
        
        # Character already created above, no need to create another one
        
        # Initialize Media database
        media_db = MediaDatabase(db_paths['Media'], "test_client")
        
        # Add test media
        media_id = media_db.add_media_with_keywords(
            url="https://example.com/video1",
            title="Test Video",
            media_type="video",
            content="This is a test transcription of the video content.",
            keywords=["test", "video", "sample"],
            prompt="Summarize this video",
            analysis_content="A test video for integration testing",
            transcription_model="whisper"
        )
        
        # Initialize Prompts database  
        prompts_db = PromptsDatabase(db_paths['Prompts'], "test_client")
        
        # Add test prompt
        prompt_result = prompts_db.add_prompt(
            name="Test Prompt",
            author="Test Author",
            details="A test prompt for chatbook testing",
            system_prompt="You are a helpful assistant.",
            user_prompt=None,
            keywords=["test", "prompt"]
        )
        
        return {
            'db_paths': db_paths,
            'conv_ids': [conv1_id, conv2_id],
            'note_ids': [note1_id, note2_id],
            'char_id': char_id,
            'media_id': media_id,
            'prompt_id': prompt_result[0] if prompt_result else None
        }
    
    def test_full_export_import_cycle(self, setup_test_databases, tmp_path):
        """Test complete export and import cycle with real databases."""
        db_paths = setup_test_databases['db_paths']
        test_data = setup_test_databases
        
        # Create chatbook
        creator = ChatbookCreator(db_paths=db_paths)
        export_path = tmp_path / "test_export.zip"
        
        content_selections = {
            ContentType.CONVERSATION: [str(id) for id in test_data['conv_ids'] if id],
            ContentType.NOTE: [str(id) for id in test_data['note_ids'] if id],
            ContentType.CHARACTER: [str(test_data['char_id'])] if test_data['char_id'] else []
        }
        
        success, message, dep_info = creator.create_chatbook(
            name="Integration Test Chatbook",
            description="Testing full export/import cycle",
            content_selections=content_selections,
            output_path=export_path,
            author="Test Suite",
            tags=["test", "integration"],
            categories=["testing"]
        )
        
        # Verify chatbook was created
        assert success is True
        assert export_path.exists()
        assert export_path.stat().st_size > 0
        
        # Verify chatbook contents
        with zipfile.ZipFile(export_path, 'r') as zf:
            # Check manifest
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Integration Test Chatbook"
            assert manifest_data['author'] == "Test Suite"
            assert manifest_data['statistics']['total_conversations'] == 2
            assert manifest_data['statistics']['total_notes'] == 2
            assert manifest_data['statistics']['total_characters'] == 1
            
            # Check content directories
            namelist = zf.namelist()
            assert any('conversations/' in name for name in namelist)
            assert any('notes/' in name for name in namelist)
            assert any('characters/' in name for name in namelist)
        
        # Setup new databases for import
        import_db_dir = tmp_path / "import_databases"
        import_db_dir.mkdir()
        
        import_db_paths = {
            'ChaChaNotes': str(import_db_dir / "ChaChaNotes.db"),
            'Media': str(import_db_dir / "Client_Media_DB.db"),
            'Prompts': str(import_db_dir / "Prompts_DB.db"),
            'Evals': str(import_db_dir / "Evals_DB.db"),
            'RAG': str(import_db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(import_db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize import databases
        import_chacha_db = CharactersRAGDB(import_db_paths['ChaChaNotes'], "test_client")
        import_media_db = MediaDatabase(import_db_paths['Media'], "test_client")
        
        # Import chatbook
        importer = ChatbookImporter(db_paths=import_db_paths)
        
        # First preview
        manifest, error = importer.preview_chatbook(export_path)
        assert manifest is not None
        assert error is None
        # Should have 5 items: 2 conv + 2 notes + 1 char
        # The character is auto-included as a dependency of conversations
        assert len(manifest.content_items) == 5
        
        # Import
        status = ImportStatus()
        success, message = importer.import_chatbook(
            chatbook_path=export_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        assert success is True
        # All 5 items should be imported successfully
        assert status.successful_items == 5
        assert len(status.errors) == 0
        
        # Verify imported data
        # Check conversations by searching for them
        conv1_results = import_chacha_db.search_conversations_by_title("Export Test Conv 1")
        assert len(conv1_results) > 0
        assert conv1_results[0]['title'] == "Export Test Conv 1"
        
        conv2_results = import_chacha_db.search_conversations_by_title("Export Test Conv 2")
        assert len(conv2_results) > 0
        assert conv2_results[0]['title'] == "Export Test Conv 2"
        
        # Check notes by searching for them
        note1 = import_chacha_db.search_notes("ML Notes")
        assert len(note1) > 0
        assert note1[0]['title'] == "ML Notes"
        
        note2 = import_chacha_db.search_notes("Python Tips")
        assert len(note2) > 0
        assert note2[0]['title'] == "Python Tips"
        
        # Check character was imported
        # Search for the character by name since we don't know the exact ID
        all_chars = import_chacha_db.list_character_cards()
        imported_char = None
        for char in all_chars:
            if char['name'] == "AI Assistant":
                imported_char = char
                break
        assert imported_char is not None
        assert imported_char['name'] == "AI Assistant"
    
    def test_selective_export_import(self, setup_test_databases, tmp_path):
        """Test exporting and importing only selected content."""
        db_paths = setup_test_databases['db_paths']
        test_data = setup_test_databases
        
        # Create chatbook with only selected items
        creator = ChatbookCreator(db_paths=db_paths)
        export_path = tmp_path / "selective.zip"
        
        # Only export first conversation and first note
        content_selections = {
            ContentType.CONVERSATION: [str(test_data['conv_ids'][0])] if test_data['conv_ids'] else [],
            ContentType.NOTE: [str(test_data['note_ids'][0])] if test_data['note_ids'] else []
        }
        
        success, message, dep_info = creator.create_chatbook(
            name="Selective Export",
            description="Only some content",
            content_selections=content_selections,
            output_path=export_path
        )
        
        assert success is True
        
        # Verify selective export
        with zipfile.ZipFile(export_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['statistics']['total_conversations'] == 1
            assert manifest_data['statistics']['total_notes'] == 1
    
    def test_conflict_resolution_scenarios(self, tmp_path):
        """Test different conflict resolution strategies."""
        # Setup database with existing content
        db_dir = tmp_path / "conflict_test"
        db_dir.mkdir()
        
        db_paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize database and add existing note
        chacha_db = CharactersRAGDB(db_paths['ChaChaNotes'], "test_client")
        existing_note_id = chacha_db.add_note(
            title="Existing Note",
            content="Old content"
        )
        
        # Create a chatbook with same note title
        chatbook_path = tmp_path / "conflicts.zip"
        
        manifest = {
            "version": "1.0",
            "name": "Conflict Test",
            "description": "Testing conflicts",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "note_1",
                    "type": "note",
                    "title": "Existing Note",
                    "file_path": "content/notes/Existing Note.md",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
            ],
            "relationships": [],
            "statistics": {
                "total_conversations": 0,
                "total_notes": 1,
                "total_characters": 0,
                "total_media_items": 0,
                "total_size_bytes": 0
            },
            "total_conversations": 0,
            "total_notes": 1,
            "total_characters": 0,
            "total_media_items": 0
        }
        
        note_content = """---
id: 1
title: Existing Note
created_at: {}
updated_at: {}
---

# Existing Note

New content from import""".format(datetime.now().isoformat(), datetime.now().isoformat())
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            zf.writestr('manifest.json', json.dumps(manifest))
            zf.writestr('content/notes/Existing Note.md', note_content)
        
        # Test SKIP resolution
        importer = ChatbookImporter(db_paths=db_paths)
        status = ImportStatus()
        
        success, message = importer.import_chatbook(
            chatbook_path=chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        # Should complete successfully
        assert success is True
        # With SKIP resolution, the duplicate note should be skipped
        assert status.skipped_items == 1
        assert status.successful_items == 0
        
        # Verify original note is unchanged
        original_note = chacha_db.get_note_by_id(existing_note_id)
        assert original_note['content'] == "Old content"
        
        # Test RENAME resolution
        status2 = ImportStatus()
        
        success, message = importer.import_chatbook(
            chatbook_path=chatbook_path,
            conflict_resolution=ConflictResolution.RENAME,
            import_status=status2
        )
        
        assert success is True
        assert status2.successful_items > 0
        
        # Check that we now have 2 notes (original + 1 renamed import)
        all_notes = chacha_db.list_notes(limit=100)
        assert len(all_notes) == 2
        titles = [n['title'] for n in all_notes]
        # Should have one "Existing Note" and one renamed version
        existing_note_count = sum(1 for title in titles if title == "Existing Note")
        assert existing_note_count == 1
        # Should have one renamed note
        renamed_count = sum(1 for title in titles if "Existing Note (" in title)
        assert renamed_count == 1
    
    def test_large_chatbook_performance(self, tmp_path):
        """Test performance with larger chatbooks."""
        import time
        
        # Setup database
        db_dir = tmp_path / "perf_test"
        db_dir.mkdir()
        
        db_paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize database
        chacha_db = CharactersRAGDB(db_paths['ChaChaNotes'], "test_client")
        
        # First add a character for conversations
        char_data = {
            "name": "Performance Test Assistant",
            "description": "Assistant for performance testing",
            "personality": "Efficient",
            "scenario": "",
            "greeting_message": "Ready for performance testing!",
            "example_messages": "",
            "version": 1
        }
        perf_char_id = chacha_db.add_character_card(char_data)
        
        # Add many conversations and notes
        start_time = time.time()
        
        conv_ids = []
        for i in range(100):
            conv_data = {
                "title": f"Conv {i+1}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "root_id": f"conv{i+1}_root",
                "character_id": perf_char_id
            }
            conv_id = chacha_db.add_conversation(conv_data)
            if conv_id:
                conv_ids.append(str(conv_id))
                for j in range(10):  # 10 messages each
                    role = "user" if j % 2 == 0 else "assistant"
                    chacha_db.add_message({
                        'conversation_id': conv_id,
                        'sender': role,
                        'content': f"Message {j+1} in conv {i+1}"
                    })
        
        note_ids = []
        for i in range(50):
            note_id = chacha_db.add_note(
                title=f"Note {i+1}",
                content=f"Content for note {i+1}" * 10
            )
            if note_id:
                note_ids.append(str(note_id))
        
        populate_time = time.time() - start_time
        
        # Export to chatbook
        creator = ChatbookCreator(db_paths=db_paths)
        export_path = tmp_path / "large.zip"
        
        content_selections = {
            ContentType.CONVERSATION: conv_ids,
            ContentType.NOTE: note_ids,
            ContentType.CHARACTER: [str(perf_char_id)]  # Explicitly include the character
        }
        
        start_time = time.time()
        success, message, dep_info = creator.create_chatbook(
            name="Large Chatbook",
            description="Performance test",
            content_selections=content_selections,
            output_path=export_path
        )
        export_time = time.time() - start_time
        
        assert success is True
        
        # Check size
        size_mb = export_path.stat().st_size / (1024 * 1024)
        
        # Import to new database
        import_db_dir = tmp_path / "import_perf"
        import_db_dir.mkdir()
        
        import_db_paths = {
            'ChaChaNotes': str(import_db_dir / "ChaChaNotes.db"),
            'Media': str(import_db_dir / "Client_Media_DB.db"),
            'Prompts': str(import_db_dir / "Prompts_DB.db"),
            'Evals': str(import_db_dir / "Evals_DB.db"),
            'RAG': str(import_db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(import_db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize import database
        import_chacha_db = CharactersRAGDB(import_db_paths['ChaChaNotes'], "test_client")
        
        importer = ChatbookImporter(db_paths=import_db_paths)
        status = ImportStatus()
        
        start_time = time.time()
        success, message = importer.import_chatbook(
            chatbook_path=export_path,
            import_status=status
        )
        import_time = time.time() - start_time
        
        # Verify results
        assert success is True
        # The character is auto-included when exporting conversations
        # So we should have 100 conversations + 50 notes + 1 character = 151 items
        # However, if import is failing for conversations, we need to check errors
        if status.successful_items != 151:
            logger.error(f"Import errors: {status.errors}")
            logger.error(f"Successful items: {status.successful_items}")
        assert status.successful_items == 151  # 100 conversations + 50 notes + 1 character (auto-included)
        
        # Performance assertions (generous limits)
        assert populate_time < 30.0  # Should populate in < 30s
        assert export_time < 10.0    # Should export in < 10s
        assert import_time < 20.0    # Should import in < 20s
        assert size_mb < 20.0        # Should be < 20MB
    
    def test_chatbook_with_relationships(self, tmp_path):
        """Test exporting and importing relationships between content."""
        # Setup database
        db_dir = tmp_path / "relationship_test"
        db_dir.mkdir()
        
        db_paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Initialize database
        chacha_db = CharactersRAGDB(db_paths['ChaChaNotes'], "test_client")
        
        # Create character
        char_data = {
            "name": "Assistant",
            "description": "Test assistant",
            "personality": "Helpful",
            "scenario": "Testing",
            "greeting_message": "Hello!",
            "example_messages": "",
            "version": 1
        }
        char_id = chacha_db.add_character_card(char_data)
        
        # Create conversation using the character
        conv_data = {
            "title": "Chat with Assistant",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "root_id": "conv_rel_root",
            "character_id": char_id
        }
        conv_id = chacha_db.add_conversation(conv_data)
        
        # Create chatbook
        creator = ChatbookCreator(db_paths=db_paths)
        export_path = tmp_path / "relations.zip"
        
        content_selections = {
            ContentType.CONVERSATION: [str(conv_id)] if conv_id else [],
            ContentType.CHARACTER: [str(char_id)] if char_id else []
        }
        
        success, message, dep_info = creator.create_chatbook(
            name="Relationships Test",
            description="Testing relationship export",
            content_selections=content_selections,
            output_path=export_path
        )
        
        assert success is True
        
        # Verify relationships in manifest
        with zipfile.ZipFile(export_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            
            # Should have relationship between conversation and character
            relationships = manifest_data['relationships']
            assert len(relationships) >= 1
            
            char_rel = next((r for r in relationships if r['relationship_type'] == 'uses_character'), None)
            assert char_rel is not None
            assert char_rel['source_id'] == str(conv_id)
            assert char_rel['target_id'] == str(char_id)