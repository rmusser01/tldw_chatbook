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
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter


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
    def chatbook_creator(self, temp_db_paths, tmp_path, monkeypatch):
        """Create a ChatbookCreator instance with test database paths."""
        test_user_data_dir = tmp_path / "test_data" / "home" / ".local" / "share" / "tldw_cli" / "default_user"
        monkeypatch.setattr(
            "tldw_chatbook.Chatbooks.chatbook_creator.get_user_data_dir",
            lambda: test_user_data_dir,
        )
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
        
        success, message, dependency_info = chatbook_creator.create_chatbook(
            name="Test Chatbook",
            description="A test chatbook",
            content_selections=content_selections,
            output_path=output_path
        )
        
        # Since we have empty databases, this should succeed but with no content
        assert success is True
        assert output_path.exists()
        assert dependency_info["missing_dependencies"] == []
        assert dependency_info["auto_included"] == []
        
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
    def test_create_chatbook_reports_packaging_progress(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
        """progress_callback fires ExportProgress events; packaging counts are monotonic to total."""
        from tldw_chatbook.Chatbooks.chatbook_creator import ExportProgress
        mock_chacha_db.return_value = MagicMock()
        mock_prompts_db.return_value = MagicMock()
        output_path = tmp_path / "cb.zip"
        events = []
        success, _msg, _dep = chatbook_creator.create_chatbook(
            name="P", description="", content_selections={
                ContentType.CONVERSATION: [], ContentType.NOTE: [], ContentType.CHARACTER: [],
            }, output_path=output_path, progress_callback=events.append,
        )
        assert success is True
        packaging = [e for e in events if e.phase == "packaging"]
        assert packaging, "expected at least one packaging progress event"
        assert all(isinstance(e, ExportProgress) for e in packaging)
        currents = [e.current for e in packaging]
        assert currents == sorted(currents) and packaging[-1].current == packaging[-1].total

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_create_chatbook_cancel_during_packaging_leaves_no_output(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
        """cancel_check True during packaging → cancelled result, no destination file, temp cleaned."""
        mock_chacha_db.return_value = MagicMock()
        mock_prompts_db.return_value = MagicMock()
        output_path = tmp_path / "cb.zip"
        calls = {"n": 0}
        def cancel_after_first_package():
            # allow collection to pass; trip on the first packaging checkpoint
            calls["n"] += 1
            return calls["n"] > 1
        success, message, dep = chatbook_creator.create_chatbook(
            name="C", description="", content_selections={
                ContentType.CONVERSATION: [], ContentType.NOTE: [], ContentType.CHARACTER: [],
            }, output_path=output_path, cancel_check=cancel_after_first_package,
        )
        assert success is False
        assert dep.get("cancelled") is True
        assert message == "Export cancelled"
        assert not output_path.exists()
        assert not output_path.with_name(output_path.name + ".partial").exists()

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    @patch('tldw_chatbook.Chatbooks.chatbook_creator.PromptsDatabase')
    def test_create_chatbook_success_leaves_no_partial(self, mock_prompts_db, mock_chacha_db, chatbook_creator, tmp_path):
        """Atomic finalize: a successful export yields a valid zip and no .partial sibling."""
        import zipfile
        mock_chacha_db.return_value = MagicMock()
        mock_prompts_db.return_value = MagicMock()
        output_path = tmp_path / "cb.zip"
        success, _msg, _dep = chatbook_creator.create_chatbook(
            name="S", description="", content_selections={ContentType.CONVERSATION: []},
            output_path=output_path,
        )
        assert success is True
        assert output_path.exists() and zipfile.is_zipfile(output_path)
        assert not output_path.with_name(output_path.name + ".partial").exists()

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
        mock_db_instance.get_character_card_by_id.return_value = {
            'id': 1,
            'name': 'Test Character',
            'description': 'A test character',
            'personality': 'Helpful',
            'avatar_path': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        # Create chatbook
        output_path = tmp_path / "test_chatbook_with_data.zip"
        content_selections = {
            ContentType.CONVERSATION: ["1"],
            ContentType.NOTE: ["1"],
            ContentType.CHARACTER: ["1"]
        }
        
        success, message, dependency_info = chatbook_creator.create_chatbook(
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
        assert dependency_info["missing_dependencies"] == []
        
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
        
        success, message, dependency_info = chatbook_creator.create_chatbook(
            name="Error Test",
            description="Testing error handling",
            content_selections=content_selections,
            output_path=output_path
        )
        
        # Should still succeed but with no conversations
        assert success is True
        assert output_path.exists()
        assert dependency_info["missing_dependencies"] == []

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_create_chatbook_preserves_conversation_citation_artifacts(
        self,
        mock_chacha_db,
        chatbook_creator,
        tmp_path,
    ):
        """Conversation exports preserve citation/evidence payloads and readable snippets."""
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        timestamp = datetime.now().isoformat()
        mock_db_instance.get_conversation_by_id.return_value = {
            'id': 'conv-1',
            'title': 'Incident Chat',
            'created_at': timestamp,
            'updated_at': timestamp,
            'character_id': None,
        }
        mock_db_instance.get_messages_for_conversation.return_value = [
            {
                'id': 'm-user',
                'sender': 'user',
                'message': 'What caused the outage?',
                'timestamp': timestamp,
            },
            {
                'id': 'm-ai',
                'sender': 'assistant',
                'message': 'An expired credential caused the outage. [S1]',
                'timestamp': timestamp,
                'metadata': {
                    'citation_validation': {
                        'status': 'validated',
                        'cited_evidence_ids': ['S1'],
                        'unknown_citation_ids': [],
                        'uncited_evidence_ids': [],
                        'citations': [
                            {
                                'evidence_id': 'S1',
                                'source_id': 'note-incident',
                                'status': 'validated',
                                'quote': 'An expired credential caused the outage. [S1]',
                            }
                        ],
                    },
                    'evidence_bundle': {
                        'bundle_id': 'library-rag:incident',
                        'query': 'What caused the outage?',
                        'source': 'Library Search/RAG',
                        'status': 'available',
                        'references': [
                            {
                                'evidence_id': 'S1',
                                'source_id': 'note-incident',
                                'source_type': 'note',
                                'title': 'Incident Review',
                                'snippet': 'Expired credential caused the outage during deploy.',
                                'authority_label': 'Local Library',
                                'status': 'available',
                            }
                        ],
                    },
                },
            },
        ]

        output_path = tmp_path / "incident_chatbook.zip"

        success, _, dependency_info = chatbook_creator.create_chatbook(
            name="Incident Chatbook",
            description="Export with citations",
            content_selections={ContentType.CONVERSATION: ["conv-1"]},
            output_path=output_path,
        )

        assert success is True
        assert dependency_info["missing_dependencies"] == []

        with zipfile.ZipFile(output_path, 'r') as zf:
            conversation = json.loads(zf.read('content/conversations/conversation_conv-1.json'))
            assistant_message = conversation["messages"][1]
            assert assistant_message["citation_validation"]["status"] == "validated"
            assert assistant_message["citation_validation"]["cited_evidence_ids"] == ["S1"]
            assert assistant_message["evidence_bundle"]["bundle_id"] == "library-rag:incident"
            assert (
                assistant_message["evidence_bundle"]["references"][0]["snippet"]
                == "Expired credential caused the outage during deploy."
            )

            report_text = zf.read(
                'content/conversations/conversation_conv-1_citations.md'
            ).decode("utf-8")
            assert "# Citations and Evidence: Incident Chat" in report_text
            assert "Citation status: validated" in report_text
            assert "Expired credential caused the outage during deploy." in report_text

            manifest_data = json.loads(zf.read('manifest.json'))
            conversation_item = next(
                item
                for item in manifest_data["content_items"]
                if item["id"] == "conv-1" and item["type"] == "conversation"
            )
            assert conversation_item["metadata"]["citation_report_path"] == (
                "content/conversations/conversation_conv-1_citations.md"
            )
            assert conversation_item["metadata"]["citation_message_count"] == 1
            assert conversation_item["metadata"]["evidence_source_count"] == 1
            assert conversation_item["metadata"]["evidence_snippet_count"] == 1

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_create_chatbook_hydrates_citation_artifacts_from_rag_context_store(
        self,
        mock_chacha_db,
        chatbook_creator,
        tmp_path,
    ):
        """Production-shaped DB messages hydrate citation artifacts from the RAG context store."""
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        timestamp = datetime.now().isoformat()
        mock_db_instance.get_conversation_by_id.return_value = {
            'id': 'conv-rag',
            'title': 'RAG Conversation',
            'created_at': timestamp,
            'updated_at': timestamp,
            'character_id': None,
        }
        mock_db_instance.get_messages_for_conversation.return_value = [
            {
                'id': 'msg-rag',
                'conversation_id': 'conv-rag',
                'sender': 'assistant',
                'content': 'Answer grounded by an incident note. [S1]',
                'timestamp': timestamp,
            },
        ]
        rag_store_path = chatbook_creator.temp_dir.parent.parent / "tldw_chatbook_chat_rag_context.json"
        rag_store_path.parent.mkdir(parents=True, exist_ok=True)
        rag_store_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "conversations": {
                        "conv-rag": {
                            "msg-rag": {
                                "conversation_id": "conv-rag",
                                "message_id": "msg-rag",
                                "rag_context": {
                                    "citation_validation": {
                                        "status": "validated",
                                        "cited_evidence_ids": ["S1"],
                                    },
                                    "evidence_bundle": {
                                        "bundle_id": "library-rag:store",
                                        "query": "stored evidence",
                                        "references": [
                                            {
                                                "evidence_id": "S1",
                                                "source_id": "note-store",
                                                "source_type": "note",
                                                "title": "Stored Incident",
                                                "snippet": "Stored snippet from the RAG context store.",
                                                "authority_label": "Local Library",
                                                "status": "available",
                                            }
                                        ],
                                    },
                                },
                                "citations": [
                                    {
                                        "id": "cite-1",
                                        "evidence_id": "S1",
                                        "source_id": "note-store",
                                        "quote": "Answer grounded by an incident note. [S1]",
                                    }
                                ],
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        output_path = tmp_path / "rag_store_chatbook.zip"

        success, _, _ = chatbook_creator.create_chatbook(
            name="RAG Store Chatbook",
            description="Export with stored citations",
            content_selections={ContentType.CONVERSATION: ["conv-rag"]},
            output_path=output_path,
        )

        assert success is True
        with zipfile.ZipFile(output_path, 'r') as zf:
            conversation = json.loads(zf.read('content/conversations/conversation_conv-rag.json'))
            exported = conversation["messages"][0]
            assert exported["citation_validation"]["status"] == "validated"
            assert exported["evidence_bundle"]["bundle_id"] == "library-rag:store"
            assert exported["citations"][0]["source_id"] == "note-store"

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_create_chatbook_sanitizes_citation_report_path_and_markdown(
        self,
        mock_chacha_db,
        chatbook_creator,
        tmp_path,
    ):
        """Unsafe conversation IDs and Markdown content stay contained and escaped."""
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        unsafe_conv_id = "../evil<script>"
        timestamp = datetime.now().isoformat()
        mock_db_instance.get_conversation_by_id.return_value = {
            'id': unsafe_conv_id,
            'title': '<script>alert(1)</script> [bad]',
            'created_at': timestamp,
            'updated_at': timestamp,
            'character_id': None,
        }
        mock_db_instance.get_messages_for_conversation.return_value = [
            {
                'id': 'msg-unsafe',
                'sender': 'assistant',
                'message': 'Uses evidence. [S1]',
                'timestamp': timestamp,
                'metadata': {
                    'citation_validation': {'status': '<b>validated</b>', 'cited_evidence_ids': ['S1']},
                    'evidence_bundle': {
                        'bundle_id': 'bundle<script>',
                        'query': '<img src=x onerror=alert(1)>',
                        'source': '<b>Library</b>',
                        'references': [
                            {
                                'evidence_id': 'S1',
                                'source_id': 'note<script>',
                                'source_type': 'note',
                                'title': '<i>Incident</i>',
                                'snippet': '<script>alert(1)</script> Root cause.',
                                'authority_label': 'Local <Library>',
                                'status': 'available',
                            }
                        ],
                    },
                },
            },
        ]

        output_path = tmp_path / "unsafe_chatbook.zip"

        success, _, _ = chatbook_creator.create_chatbook(
            name="Unsafe Chatbook",
            description="Export with unsafe content",
            content_selections={ContentType.CONVERSATION: [unsafe_conv_id]},
            output_path=output_path,
        )

        assert success is True
        with zipfile.ZipFile(output_path, 'r') as zf:
            names = zf.namelist()
            assert all(".." not in name for name in names)
            assert all("<script>" not in name for name in names)
            report_name = next(name for name in names if name.endswith("_citations.md"))
            assert report_name.startswith("content/conversations/")
            report_text = zf.read(report_name).decode("utf-8")
            assert "<script>" not in report_text
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report_text
            assert "&lt;img src=x onerror=alert(1)&gt;" in report_text

    @patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB')
    def test_create_chatbook_bounds_large_citation_reports(
        self,
        mock_chacha_db,
        chatbook_creator,
        tmp_path,
    ):
        """Large citation exports keep full JSON but bound the readable Markdown report."""
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        timestamp = datetime.now().isoformat()
        mock_db_instance.get_conversation_by_id.return_value = {
            'id': 'conv-large',
            'title': 'Large Citation Chat',
            'created_at': timestamp,
            'updated_at': timestamp,
            'character_id': None,
        }
        references = [
            {
                'evidence_id': f'S{i}',
                'source_id': f'note-{i}',
                'source_type': 'note',
                'title': f'Note {i}',
                'snippet': f'Snippet {i}',
                'authority_label': 'Local Library',
                'status': 'available',
            }
            for i in range(1, 75)
        ]
        mock_db_instance.get_messages_for_conversation.return_value = [
            {
                'id': 'msg-large',
                'sender': 'assistant',
                'message': 'Large answer. [S1]',
                'timestamp': timestamp,
                'metadata': {
                    'citation_validation': {'status': 'validated', 'cited_evidence_ids': ['S1']},
                    'evidence_bundle': {
                        'bundle_id': 'library-rag:large',
                        'query': 'large query',
                        'references': references,
                    },
                },
            },
        ]

        output_path = tmp_path / "large_citations.zip"

        success, _, _ = chatbook_creator.create_chatbook(
            name="Large Citation Chatbook",
            description="Export with many references",
            content_selections={ContentType.CONVERSATION: ["conv-large"]},
            output_path=output_path,
        )

        assert success is True
        with zipfile.ZipFile(output_path, 'r') as zf:
            conversation = json.loads(zf.read('content/conversations/conversation_conv-large.json'))
            assert len(conversation["messages"][0]["evidence_bundle"]["references"]) == 74
            report_text = zf.read(
                'content/conversations/conversation_conv-large_citations.md'
            ).decode("utf-8")
            assert "Report truncated:" in report_text
            assert "Snippet 1" in report_text
            assert "Snippet 74" not in report_text
            manifest_data = json.loads(zf.read('manifest.json'))
            item = next(item for item in manifest_data["content_items"] if item["id"] == "conv-large")
            assert item["metadata"]["citation_report_truncated"] is True
            assert item["metadata"]["evidence_source_count"] == 74
    
    @patch('zipfile.ZipFile')
    def test_create_chatbook_zip_error(self, mock_zipfile, chatbook_creator, tmp_path):
        """Test handling of ZIP creation errors."""
        # Mock ZIP file to raise error
        mock_zipfile.side_effect = Exception("ZIP creation failed")
        
        output_path = tmp_path / "test_zip_error.zip"
        content_selections = {}
        
        success, message, dependency_info = chatbook_creator.create_chatbook(
            name="ZIP Error Test",
            description="Testing ZIP error",
            content_selections=content_selections,
            output_path=output_path
        )

        assert success is False
        assert "error" in message.lower()
        assert dependency_info["auto_included"] == []

    @patch('tldw_chatbook.Chatbooks.chatbook_importer.get_user_data_dir')
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_chatbook_persists_conversation_citation_payloads(
        self,
        mock_chacha_db,
        mock_get_user_data_dir,
        temp_db_paths,
        tmp_path,
    ):
        """Import preserves exported citation fields in the conversation RAG context store."""
        user_data_dir = tmp_path / "user-data"
        mock_get_user_data_dir.return_value = user_data_dir

        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.get_conversation_by_name.return_value = []
        mock_db_instance.add_conversation.return_value = "new-conv"
        mock_db_instance.add_message.return_value = "new-msg"
        mock_db_instance.get_message_by_id.return_value = {
            "id": "new-msg",
            "conversation_id": "new-conv",
        }

        timestamp = datetime.now().isoformat()
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Imported Citation Chatbook",
            description="Import with citations",
        )
        manifest.content_items.append(
            ContentItem(
                id="conv-import",
                type=ContentType.CONVERSATION,
                title="Imported Conversation",
                created_at=datetime.fromisoformat(timestamp),
                updated_at=datetime.fromisoformat(timestamp),
                file_path="content/conversations/conversation_conv-import.json",
            )
        )
        manifest.total_conversations = 1
        chatbook_path = tmp_path / "import_citations.zip"
        conversation_payload = {
            "id": "conv-import",
            "name": "Imported Conversation",
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [
                {
                    "id": "old-msg",
                    "role": "assistant",
                    "content": "Imported answer. [S1]",
                    "timestamp": timestamp,
                    "citation_validation": {
                        "status": "validated",
                        "cited_evidence_ids": ["S1"],
                    },
                    "evidence_bundle": {
                        "bundle_id": "library-rag:import",
                        "query": "import evidence",
                        "references": [
                            {
                                "evidence_id": "S1",
                                "source_id": "note-import",
                                "source_type": "note",
                                "title": "Import Note",
                                "snippet": "Imported evidence snippet.",
                                "authority_label": "Local Library",
                                "status": "available",
                            }
                        ],
                    },
                    "citations": [
                        {
                            "evidence_id": "S1",
                            "source_id": "note-import",
                            "quote": "Imported answer. [S1]",
                        }
                    ],
                }
            ],
        }
        with zipfile.ZipFile(chatbook_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest.to_dict()))
            zf.writestr(
                "content/conversations/conversation_conv-import.json",
                json.dumps(conversation_payload),
            )

        importer = ChatbookImporter(temp_db_paths)
        success, message = importer.import_chatbook(chatbook_path)

        assert success is True, message
        rag_store = json.loads(
            (user_data_dir / "tldw_chatbook_chat_rag_context.json").read_text(encoding="utf-8")
        )
        record = rag_store["conversations"]["new-conv"]["new-msg"]
        assert record["rag_context"]["citation_validation"]["status"] == "validated"
        assert record["rag_context"]["evidence_bundle"]["bundle_id"] == "library-rag:import"
        assert record["citations"][0]["source_id"] == "note-import"
    
    def test_chatbook_with_media_settings(self, chatbook_creator, tmp_path):
        """Test creating chatbook with media settings."""
        output_path = tmp_path / "test_media.zip"
        
        success, message, dependency_info = chatbook_creator.create_chatbook(
            name="Media Test",
            description="Testing media settings",
            content_selections={},
            output_path=output_path,
            include_media=True,
            media_quality="original",
            include_embeddings=True
        )

        assert success is True
        assert dependency_info["missing_dependencies"] == []
        
        # Check manifest for media settings
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data.get('include_media') is True
            assert manifest_data.get('media_quality') == "original"
            assert manifest_data.get('include_embeddings') is True
