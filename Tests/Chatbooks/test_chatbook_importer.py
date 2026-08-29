# test_chatbook_importer.py
# Unit tests for chatbook importer

import pytest
import io
import json
import os
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import sqlite3

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
import tldw_chatbook.Chatbooks.chatbook_importer as importer_module
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    encode_project_context_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


class TestChatbookImporter:
    """Test ChatbookImporter functionality."""

    @pytest.fixture(autouse=True)
    def stub_citation_composition(self, monkeypatch):
        """Keep importer unit tests on their existing mocked database seam."""

        from tldw_chatbook.Chat.chat_conversation_service import (
            ChatConversationService,
        )

        def build_local(db, *, sidecar_path):
            return (
                ChatConversationService(db, rag_context_store_path=sidecar_path),
                None,
                None,
            )

        monkeypatch.setattr(
            "tldw_chatbook.Chatbooks.chatbook_importer.build_local_citation_conversation_service",
            build_local,
        )

    @pytest.fixture
    def temp_db_paths(self, tmp_path):
        """Create temporary database paths with schema."""
        db_dir = tmp_path / "databases"
        db_dir.mkdir()

        paths = {
            "ChaChaNotes": str(db_dir / "ChaChaNotes.db"),
            "Media": str(db_dir / "Client_Media_DB.db"),
            "Prompts": str(db_dir / "Prompts_DB.db"),
            "Evals": str(db_dir / "Evals_DB.db"),
            "RAG": str(db_dir / "RAG_Indexing_DB.db"),
            "Subscriptions": str(db_dir / "Subscriptions_DB.db"),
        }

        # Create database files with schema
        for name, path in paths.items():
            conn = sqlite3.connect(path)
            if name == "ChaChaNotes":
                # Create schema for ChaChaNotes
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        created_at TEXT,
                        character_id INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY,
                        conversation_id INTEGER,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT
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
            elif name == "Prompts":
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
    def chatbook_importer(self, temp_db_paths):
        """Create a ChatbookImporter instance with test database paths."""
        return ChatbookImporter(db_paths=temp_db_paths)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX temp privacy contract")
    def test_importer_uses_secured_canonical_temp_root(
        self,
        temp_db_paths,
        tmp_path,
        monkeypatch,
    ):
        user_data_dir = tmp_path / "runtime-data"
        user_data_dir.mkdir(mode=0o700)
        imports_dir = user_data_dir / "temp" / "imports"
        imports_dir.mkdir(parents=True, mode=0o755)
        monkeypatch.setattr(
            importer_module,
            "get_user_data_dir",
            lambda: user_data_dir,
            raising=False,
        )

        importer = ChatbookImporter(db_paths=temp_db_paths)

        assert importer.temp_dir == imports_dir
        assert importer.temp_dir.stat().st_mode & 0o777 == 0o700

    @pytest.mark.skipif(os.name != "posix", reason="POSIX extraction contract")
    def test_preview_extracts_privately_and_always_cleans_up(
        self,
        chatbook_importer,
        tmp_path,
        monkeypatch,
    ):
        chatbook_path = tmp_path / "private.zip"
        with zipfile.ZipFile(chatbook_path, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("content/private-note.md", "secret")
        observed: dict[str, int] = {}

        def inspect_then_fail(handle):
            extract_dir = Path(handle.name).parent
            note_path = extract_dir / "content" / "private-note.md"
            observed["extract_dir"] = extract_dir.stat().st_mode & 0o777
            observed["content_dir"] = note_path.parent.stat().st_mode & 0o777
            observed["manifest"] = Path(handle.name).stat().st_mode & 0o777
            observed["note"] = note_path.stat().st_mode & 0o777
            raise RuntimeError("stop after privacy inspection")

        monkeypatch.setattr(importer_module.json, "load", inspect_then_fail)
        previous = os.umask(0)
        try:
            manifest, error = chatbook_importer.preview_chatbook(chatbook_path)
        finally:
            os.umask(previous)

        assert manifest is None
        assert "stop after privacy inspection" in error
        assert observed == {
            "extract_dir": 0o700,
            "content_dir": 0o700,
            "manifest": 0o600,
            "note": 0o600,
        }
        assert list(chatbook_importer.temp_dir.iterdir()) == []

    @pytest.mark.parametrize(
        ("members", "limits"),
        [
            ({"a": b"x", "b": b"x"}, {"_MAX_ARCHIVE_MEMBERS": 1}),
            ({"a": b"xx"}, {"_MAX_ARCHIVE_MEMBER_BYTES": 1}),
            (
                {"a": b"xx", "b": b"xx"},
                {"_MAX_ARCHIVE_TOTAL_BYTES": 3},
            ),
            (
                {"a": b"x" * 1_000},
                {"_MAX_ARCHIVE_COMPRESSION_RATIO": 2},
            ),
        ],
    )
    def test_preview_rejects_archive_resource_limits_before_extraction(
        self,
        chatbook_importer,
        tmp_path,
        monkeypatch,
        members,
        limits,
    ):
        archive_path = tmp_path / "bounded.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        for name, value in limits.items():
            monkeypatch.setattr(importer_module, name, value)

        manifest, error = chatbook_importer.preview_chatbook(archive_path)

        assert manifest is None
        assert (
            error
            == "Error previewing chatbook: Chatbook archive exceeds safety limits."
        )
        assert list(chatbook_importer.temp_dir.iterdir()) == []

    def test_extraction_counts_actual_member_bytes_and_cleans_up(
        self,
        chatbook_importer,
        tmp_path,
        monkeypatch,
    ):
        archive_path = tmp_path / "dishonest.zip"
        archive_path.write_bytes(b"fake")
        member = zipfile.ZipInfo("manifest.json")
        member.file_size = 1
        member.compress_size = 1

        class FakeArchive:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def infolist():
                return [member]

            @staticmethod
            def open(*_args, **_kwargs):
                return io.BytesIO(b"xx")

        monkeypatch.setattr(importer_module.zipfile, "ZipFile", FakeArchive)

        manifest, error = chatbook_importer.preview_chatbook(archive_path)

        assert manifest is None
        assert (
            error
            == "Error previewing chatbook: Chatbook archive exceeds safety limits."
        )
        assert list(chatbook_importer.temp_dir.iterdir()) == []

    def test_archive_limit_import_fails_before_database_writes(
        self,
        chatbook_importer,
        temp_db_paths,
        tmp_path,
        monkeypatch,
    ):
        archive_path = tmp_path / "too-many.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("extra", "x")
        monkeypatch.setattr(importer_module, "_MAX_ARCHIVE_MEMBERS", 1)

        success, error = chatbook_importer.import_chatbook(archive_path)

        assert success is False
        assert error == "Fatal error: Chatbook archive exceeds safety limits."
        with sqlite3.connect(temp_db_paths["ChaChaNotes"]) as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                == 0
            )
        assert list(chatbook_importer.temp_dir.iterdir()) == []

    @pytest.fixture
    def sample_chatbook_path(self, tmp_path):
        """Create a sample chatbook for testing."""
        chatbook_path = tmp_path / "sample_chatbook.zip"

        # Create manifest
        manifest = {
            "version": "1.0",
            "name": "Sample Chatbook",
            "description": "A sample chatbook for testing",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "1",
                    "type": "conversation",
                    "title": "Test Conversation",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/conversations/conversation_1.json",
                },
                {
                    "id": "1",
                    "type": "note",
                    "title": "Test Note",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/notes/Test Note.md",
                },
                {
                    "id": "1",
                    "type": "character",
                    "title": "Test Character",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/characters/character_1.json",
                },
            ],
            "relationships": [
                {
                    "source_id": "1",
                    "target_id": "1",
                    "relationship_type": "uses_character",
                    "metadata": {},
                }
            ],
            "include_media": False,
            "include_embeddings": False,
            "media_quality": "thumbnail",
            "statistics": {
                "total_conversations": 1,
                "total_notes": 1,
                "total_characters": 1,
                "total_media_items": 0,
                "total_size_bytes": 1024,
            },
            "tags": ["test", "sample"],
            "categories": ["testing"],
            "language": "en",
            "license": None,
        }

        # Create conversation content
        conversation_content = {
            "id": 1,
            "name": "Test Conversation",
            "title": "Test Conversation",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": "Hi there!",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "character_id": 1,
        }

        # Create note content
        note_content = """# Test Note

This is a test note with some content.

Keywords: test, sample"""

        # Create character content
        character_content = {
            "id": 1,
            "name": "Test Character",
            "description": "A test character",
            "personality": "Helpful and friendly",
            "scenario": "Testing environment",
            "greeting_message": "Hello!",
            "example_messages": "",
        }

        # Create ZIP file
        with zipfile.ZipFile(chatbook_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr(
                "content/conversations/conversation_1.json",
                json.dumps(conversation_content, indent=2),
            )
            zf.writestr("content/notes/Test Note.md", note_content)
            zf.writestr(
                "content/characters/character_1.json",
                json.dumps(character_content, indent=2),
            )

        return chatbook_path

    def test_importer_initialization(self, chatbook_importer, temp_db_paths):
        """Test ChatbookImporter initialization."""
        assert chatbook_importer.db_paths == temp_db_paths
        assert chatbook_importer.temp_dir.exists()
        assert chatbook_importer.conflict_resolver is not None

    def test_temp_dir_derives_from_get_user_data_dir(
        self, chatbook_importer, temp_db_paths
    ):
        """TASK-865: the extraction root must derive from
        ``get_user_data_dir()`` -- not a ``Path.home()/".local"/"share"/
        "tldw_cli"`` literal that omits the per-profile user-folder
        segment. Before the fix, imports silently extracted outside the
        per-user tree (a location a live reproduction confirmed already
        existed on disk in production)."""
        from tldw_chatbook.config import get_user_data_dir

        assert chatbook_importer.temp_dir == get_user_data_dir() / "temp" / "imports"

    def test_temp_dir_shares_a_parent_with_the_chatbook_creators_temp_root(
        self, chatbook_importer
    ):
        """AC #3: the importer's extraction root and the creator's temp
        root must derive to the same parent (``get_user_data_dir() /
        "temp"``), not two different, disagreeing directories."""
        from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator

        creator = ChatbookCreator(db_paths={})

        assert chatbook_importer.temp_dir.parent == creator.temp_dir.parent
        assert chatbook_importer.temp_dir.parent.name == "temp"

    def test_preview_chatbook_valid(self, chatbook_importer, sample_chatbook_path):
        """Test previewing a valid chatbook."""
        manifest, error = chatbook_importer.preview_chatbook(sample_chatbook_path)

        assert manifest is not None
        assert error is None
        assert manifest.name == "Sample Chatbook"
        assert manifest.description == "A sample chatbook for testing"
        assert len(manifest.content_items) == 3

    def test_preview_chatbook_invalid_zip(self, chatbook_importer, tmp_path):
        """Test previewing an invalid ZIP file."""
        invalid_path = tmp_path / "invalid.zip"
        invalid_path.write_text("Not a ZIP file")

        manifest, error = chatbook_importer.preview_chatbook(invalid_path)

        assert manifest is None
        assert error is not None
        assert "zip" in error.lower()

    def test_preview_chatbook_missing_manifest(self, chatbook_importer, tmp_path):
        """Test previewing a chatbook without manifest."""
        invalid_path = tmp_path / "no_manifest.zip"

        with zipfile.ZipFile(invalid_path, "w") as zf:
            zf.writestr("content/test.txt", "test")

        manifest, error = chatbook_importer.preview_chatbook(invalid_path)

        assert manifest is None
        assert error is not None
        assert "manifest.json" in error

    @pytest.mark.parametrize("missing_manifest", [False, True])
    def test_import_chatbook_early_failure_returns_message_string(
        self,
        chatbook_importer,
        tmp_path,
        missing_manifest,
    ):
        """Keep the documented result contract on validation failures."""
        invalid_path = tmp_path / (
            "no_manifest.zip" if missing_manifest else "unsupported.chatbook"
        )
        if missing_manifest:
            with zipfile.ZipFile(invalid_path, "w") as zf:
                zf.writestr("content/test.txt", "test")
        else:
            invalid_path.write_text("not a zip archive")
        status = ImportStatus()

        success, message = chatbook_importer.import_chatbook(
            chatbook_path=invalid_path,
            import_status=status,
        )

        assert success is False
        assert isinstance(message, str)
        assert status.errors == [message]

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_import_chatbook_no_conflicts(
        self, mock_chacha_db, chatbook_importer, sample_chatbook_path
    ):
        """Test importing a chatbook with no conflicts."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        mock_db_instance.get_conversation_by_name.return_value = []
        mock_db_instance.get_note_by_title.return_value = None
        mock_db_instance.get_character_card_by_name.return_value = None

        status = ImportStatus()

        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        assert success is True
        assert status.processed_items > 0
        assert len(status.errors) == 0
        imported_conversation = mock_db_instance.add_conversation.call_args.args[0]
        assert imported_conversation["character_id"] == 1
        assert imported_conversation["assistant_authority_id"] is None

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_import_chatbook_with_conflicts(
        self, mock_chacha_db, chatbook_importer, sample_chatbook_path, temp_db_paths
    ):
        """Test importing with existing data (conflicts)."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        mock_db_instance.get_conversation_by_name.return_value = []
        mock_db_instance.get_note_by_title.return_value = None
        mock_db_instance.get_character_card_by_name.return_value = None

        status = ImportStatus()

        # Import with SKIP resolution
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        assert success is True
        # Since we're using mocks and not simulating actual conflicts,
        # we just verify that the import succeeded
        assert status.processed_items > 0

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_import_chatbook_rename_conflicts(
        self, mock_chacha_db, chatbook_importer, sample_chatbook_path, temp_db_paths
    ):
        """Test importing with RENAME conflict resolution."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        mock_db_instance.get_conversation_by_name.side_effect = lambda name: (
            [{"id": 99, "title": name}] if name == "Test Conversation" else []
        )
        mock_db_instance.get_note_by_title.side_effect = lambda title: (
            {"id": 88, "title": title} if title == "Test Note" else None
        )
        mock_db_instance.get_character_card_by_name.side_effect = lambda name: (
            {"id": 77, "name": name} if name == "Test Character" else None
        )

        status = ImportStatus()

        # Import with RENAME resolution
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.RENAME,
            import_status=status,
        )

        assert success is True
        assert status.successful_items > 0

        # Since we're mocking, just verify that the import was successful
        # In real implementation, it would rename the note

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_import_status_tracking(
        self, mock_chacha_db, chatbook_importer, sample_chatbook_path
    ):
        """Test import status tracking."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        mock_db_instance.get_conversation_by_name.return_value = []
        mock_db_instance.get_note_by_title.return_value = None
        mock_db_instance.get_character_card_by_name.return_value = None

        status = ImportStatus()

        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        assert success is True
        assert status.total_items == 3  # 1 conv + 1 note + 1 char
        assert status.processed_items == 3
        assert status.successful_items > 0

        # Check status dict
        status_dict = status.to_dict()
        assert "total_items" in status_dict
        assert "errors" in status_dict
        assert "warnings" in status_dict

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_import_error_handling(self, mock_chacha_db, chatbook_importer, tmp_path):
        """Test error handling during import."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance

        # Create a chatbook with invalid content
        bad_chatbook = tmp_path / "bad_chatbook.zip"

        manifest = {
            "version": "1.0",
            "name": "Bad Chatbook",
            "description": "Test",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "bad_1",
                    "type": "conversation",
                    "title": "Bad Item",
                    "file_path": "content/missing.json",  # File doesn't exist
                }
            ],
        }

        with zipfile.ZipFile(bad_chatbook, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            # Don't create the content file

        status = ImportStatus()
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=bad_chatbook,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        # Import fails when no items could be imported
        assert success is False
        assert status.failed_items > 0
        assert len(status.warnings) > 0  # File not found generates warnings

    def test_import_with_media_settings(self, chatbook_importer, tmp_path):
        """Test importing chatbook with media settings."""
        chatbook_path = tmp_path / "media_chatbook.zip"

        manifest = {
            "version": "1.0",
            "name": "Media Chatbook",
            "description": "Test with media",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "include_media": True,
            "media_quality": "original",
            "include_embeddings": True,
            "content_items": [],
        }

        with zipfile.ZipFile(chatbook_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        manifest_obj, error = chatbook_importer.preview_chatbook(chatbook_path)

        assert manifest_obj is not None
        assert manifest_obj.include_media is True
        assert manifest_obj.media_quality == "original"
        assert manifest_obj.include_embeddings is True


# ---------------------------------------------------------------------------
# TASK-928: ChatbookImporter's internal db_paths key casing
# ("ChaChaNotes"/"Prompts"/"Media") must agree with
# Chatbooks.database_paths.get_chatbook_database_paths() -- the single
# helper every real UI call site (the canonical Chatbooks destination,
# the import/creation wizards, and the export management window; see
# Tests/Chatbooks/test_chatbook_database_paths.py) uses to build the
# db_paths dict handed to ChatbookImporter. A casing mismatch between the
# two sides is invisible to type checking and to any test that stubs one
# side out from under the other.
# ---------------------------------------------------------------------------


def test_chatbook_importer_key_lookups_match_get_chatbook_database_paths():
    """The importer's actual `self.db_paths.get("...")` lookups (scanned via
    AST, not a hardcoded duplicate list and not a source-text/comment match)
    must all be keys `get_chatbook_database_paths()` actually produces.

    This is deliberately AST-based rather than a plain substring check: a
    substring/text scan would pass even if the string only appeared in a
    comment or docstring, proving nothing about the real lookup contract.
    """
    import ast
    import inspect

    from tldw_chatbook.Chatbooks import chatbook_importer as importer_module
    from tldw_chatbook.Chatbooks.database_paths import get_chatbook_database_paths

    tree = ast.parse(inspect.getsource(importer_module))
    looked_up_keys: set[str] = set()

    class _DbPathsGetVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "db_paths"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                looked_up_keys.add(node.args[0].value)
            self.generic_visit(node)

    _DbPathsGetVisitor().visit(tree)

    assert looked_up_keys, (
        "Expected to find self.db_paths.get(<string literal>) lookups in "
        "chatbook_importer.py -- if this fails, the importer's lookup "
        "pattern changed and this scan needs updating, not deleting."
    )

    produced_keys = set(get_chatbook_database_paths().keys())
    missing = looked_up_keys - produced_keys
    assert not missing, (
        f"ChatbookImporter looks up db_paths key(s) {sorted(missing)} that "
        f"get_chatbook_database_paths() does not produce (it produces "
        f"{sorted(produced_keys)}). The importer's key contract and the "
        "canonical path-resolution helper every real caller uses have "
        "drifted apart -- see TASK-928."
    )


class TestChatbookImporterKeyCasingMismatch:
    """Documents the real, verified behaviour of a db_paths key-casing
    mismatch (TASK-928 AC: "The real behaviour of the current mismatch is
    established and recorded"), and guards against it recurring silently.

    Established live (see task-928's Implementation Notes): a mismatch does
    not raise, and does not silently "succeed" while importing nothing --
    every db_paths.get(...) lookup returns None, each content-type import
    method records a "<Name> database path not configured" error and skips
    that type, and the overall import reports success=False with that error
    surfaced as the failure message.
    """

    @pytest.fixture(autouse=True)
    def stub_citation_composition(self, monkeypatch):
        """Pins the key-casing contract between ChatbookImporter and get_chatbook_database_paths().

        Stubs the citation conversation service builder to keep these tests isolated
        at the mocked database seam. This fixture exists in both TestChatbookImporter
        and TestChatbookImporterKeyCasingMismatch: the former tests the happy path
        where db_paths keys match the importer's literal lookups, the latter tests
        what happens when key casing diverges (TASK-928).

        Args:
            monkeypatch: pytest monkeypatch fixture for replacing build_local_citation_conversation_service.
        """
        from tldw_chatbook.Chat.chat_conversation_service import (
            ChatConversationService,
        )

        def build_local(db, *, sidecar_path):
            return (
                ChatConversationService(db, rag_context_store_path=sidecar_path),
                None,
                None,
            )

        monkeypatch.setattr(
            "tldw_chatbook.Chatbooks.chatbook_importer.build_local_citation_conversation_service",
            build_local,
        )

    @pytest.fixture
    def sample_chatbook_path(self, tmp_path):
        """Create and return a temporary chatbook ZIP file with valid structure.

        Generates a minimal but complete chatbook archive with manifest and one
        conversation, suitable for testing ChatbookImporter behavior.

        Args:
            tmp_path: pytest fixture providing a temporary directory.

        Returns:
            Path object pointing to the created sample.zip file.
        """
        chatbook_path = tmp_path / "sample.zip"
        manifest = {
            "version": "1.0",
            "name": "Sample",
            "description": "A sample chatbook for testing",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "1",
                    "type": "conversation",
                    "title": "Test Conversation",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/conversations/conversation_1.json",
                },
            ],
            "relationships": [],
            "include_media": False,
            "include_embeddings": False,
            "media_quality": "thumbnail",
            "statistics": {
                "total_conversations": 1,
                "total_notes": 0,
                "total_characters": 0,
                "total_media_items": 0,
                "total_size_bytes": 1,
            },
            "tags": [],
            "categories": [],
            "language": "en",
            "license": None,
        }
        conversation_content = {
            "id": 1,
            "name": "Test Conversation",
            "title": "Test Conversation",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "character_id": None,
        }
        with zipfile.ZipFile(chatbook_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr(
                "content/conversations/conversation_1.json",
                json.dumps(conversation_content),
            )
        return chatbook_path

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_correctly_cased_keys_import_successfully(
        self, mock_chacha_db, sample_chatbook_path
    ):
        """Control case: get_chatbook_database_paths()'s actual casing
        imports cleanly, proving the failure below is caused by casing
        alone."""
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.get_conversation_by_name.return_value = []

        importer = ChatbookImporter(
            db_paths={"ChaChaNotes": "unused", "Prompts": "unused", "Media": "unused"}
        )
        status = ImportStatus()
        success, message = importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        assert success is True
        assert status.successful_items == 1
        assert status.errors == []

    @patch("tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB")
    def test_mismatched_lowercase_keys_fail_cleanly_not_silently_not_crashing(
        self, mock_chacha_db, sample_chatbook_path
    ):
        """The pre-consolidation bug shape: db_paths built with lowercase
        keys ("chachanotes"/"prompts"/"media"), as _import_chatbook used to
        before dev's get_chatbook_database_paths() consolidation."""
        mock_chacha_db.return_value = MagicMock()

        importer = ChatbookImporter(
            db_paths={"chachanotes": "unused", "prompts": "unused", "media": "unused"}
        )
        status = ImportStatus()
        success, message = importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status,
        )

        # Not silent: reports failure with a specific, actionable message.
        assert success is False
        assert message == "Failed to import any items from chatbook"
        # Not a crash: import_chatbook returns its normal (bool, str)
        # contract rather than raising.
        assert status.successful_items == 0
        assert status.errors == ["ChaChaNotes database path not configured"]


# ---------------------------------------------------------------------------
# task-4022 (review round 2, I1a): the DB-layer trash-restore fix
# (Client_Media_DB_v2.add_media_with_keywords) made restoring a trashed
# match on re-import OPT-IN (``restore_trashed=True``) rather than
# unconditional. ``ChatbookImporter._import_media`` never passes that flag,
# so a trashed row must stay exactly as inert as it was before task-4022
# existed -- even though its own conflict check (``get_media_by_url``,
# which excludes trashed rows by default) can't see the trashed row and
# therefore never even reaches the SKIP/RENAME branch for it. Real,
# file-backed ``MediaDatabase`` throughout -- no mocks -- per this
# programme's DB-layer testing requirement.
# ---------------------------------------------------------------------------


def test_import_media_skip_conflict_leaves_trashed_row_untouched(tmp_path):
    target_path = tmp_path / "target.sqlite"
    target = MediaDatabase(target_path, client_id="target")
    url = "https://example.com/trashed-article"
    media_id, _, _ = target.add_media_with_keywords(
        title="Original title",
        media_type="article",
        content="original content",
        url=url,
    )
    assert target.mark_as_trash(media_id) is True
    target.close_connection()

    extract_dir = tmp_path / "export"
    media_dir = extract_dir / "content" / "media"
    metadata_dir = media_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    export_media_id = "999"
    (metadata_dir / f"media_{export_media_id}.json").write_text(
        json.dumps(
            {
                "title": "Reimported title",
                "url": url,
                "media_type": "article",
                "metadata": {"media_keywords": []},
            }
        ),
        encoding="utf-8",
    )
    (media_dir / f"media_{export_media_id}.txt").write_text(
        "reimported content", encoding="utf-8"
    )

    manifest = ChatbookManifest(
        version=ChatbookVersion.V1, name="skip-conflict", description="test"
    )
    importer = ChatbookImporter(db_paths={"Media": str(target_path)})
    status = ImportStatus()
    importer._import_media(
        extract_dir, manifest, [export_media_id], ConflictResolution.SKIP, status
    )

    # Its OWN existing-check (get_media_by_url, include_trash=False) can't
    # see the trashed row, so this is NOT `status.skipped_items` -- it
    # falls through to `add_media_with_keywords`, which (without
    # `restore_trashed=True`) reports the trashed match as an ordinary
    # duplicate-skip: `media_id=None` -> `status.failed_items += 1`. The
    # important behavior under test is what happens to the ROW, not which
    # counter absorbs it.
    assert status.successful_items == 0
    assert status.failed_items == 1

    verify = MediaDatabase(target_path, client_id="verify")
    row = verify.get_media_by_id(media_id, include_trash=True)
    assert row["is_trash"] == 1, "SKIP must not resurrect a trashed row"
    assert row["title"] == "Original title"
    assert row["content"] == "original content"
    cursor = verify.execute_query("SELECT COUNT(*) FROM Media")
    assert cursor.fetchone()[0] == 1, "no second row must have been created either"
    verify.close_connection()


def _write_project_context_conflict_chatbook(path: Path) -> None:
    """Write one conflicting conversation with a hostile local-state field."""
    now = datetime.now().isoformat()
    manifest = {
        "version": "1.0",
        "name": "Project context conflict",
        "description": "Importer local-state preservation fixture",
        "created_at": now,
        "updated_at": now,
        "content_items": [
            {
                "id": "incoming-1",
                "type": "conversation",
                "title": "Existing conversation",
                "created_at": now,
                "file_path": "content/conversations/conversation_incoming-1.json",
            }
        ],
        "relationships": [],
        "statistics": {"total_conversations": 1},
    }
    conversation = {
        "id": "incoming-1",
        "name": "Existing conversation",
        "title": "Existing conversation",
        "created_at": now,
        "updated_at": now,
        "character_id": None,
        "messages": [{"role": "user", "content": "imported", "timestamp": now}],
        "console_project_context_json": "must-not-enter-local-state",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(
            "content/conversations/conversation_incoming-1.json",
            json.dumps(conversation),
        )


def _seed_conversation_with_project_context(path: Path) -> tuple[str, str]:
    db = CharactersRAGDB(path, client_id="import-seed")
    conversation_id = db.add_conversation({"title": "Existing conversation"})
    encoded = encode_project_context_json(
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="existing-binding",
            working_folder_locator_fingerprint="existing-locator-fingerprint",
            project_instruction_notice_key="existing-notice-key",
        )
    )
    db.set_conversation_console_project_context(conversation_id, encoded)
    db.close_connection()
    return str(conversation_id), encoded


def test_import_conflict_skip_preserves_existing_console_project_context(
    tmp_path, monkeypatch
) -> None:
    user_data = tmp_path / "user-data"
    monkeypatch.setattr(importer_module, "get_user_data_dir", lambda: user_data)
    db_path = tmp_path / "chachanotes.db"
    existing_id, encoded = _seed_conversation_with_project_context(db_path)
    chatbook_path = tmp_path / "conflict.chatbook.zip"
    _write_project_context_conflict_chatbook(chatbook_path)
    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()

    importer.import_chatbook(
        chatbook_path=chatbook_path,
        conflict_resolution=ConflictResolution.SKIP,
        import_status=status,
    )

    reopened = CharactersRAGDB(db_path, client_id="import-assert")
    rows = (
        reopened.get_connection()
        .execute(
            "SELECT id, console_project_context_json FROM conversations "
            "WHERE title = ? ORDER BY rowid",
            ("Existing conversation",),
        )
        .fetchall()
    )
    assert status.skipped_items == 1
    assert [(row["id"], row["console_project_context_json"]) for row in rows] == [
        (existing_id, encoded)
    ]
    reopened.close_connection()


@pytest.mark.parametrize(
    "resolution",
    [
        ConflictResolution.REPLACE,
        ConflictResolution.RENAME,
        ConflictResolution.MERGE,
    ],
)
def test_import_non_skip_conflicts_create_null_local_state_and_preserve_existing(
    tmp_path, monkeypatch, resolution
) -> None:
    user_data = tmp_path / "user-data"
    monkeypatch.setattr(importer_module, "get_user_data_dir", lambda: user_data)
    db_path = tmp_path / "chachanotes.db"
    existing_id, encoded = _seed_conversation_with_project_context(db_path)
    chatbook_path = tmp_path / "conflict.chatbook.zip"
    _write_project_context_conflict_chatbook(chatbook_path)
    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()

    success, _message = importer.import_chatbook(
        chatbook_path=chatbook_path,
        conflict_resolution=resolution,
        import_status=status,
    )

    reopened = CharactersRAGDB(db_path, client_id="import-assert")
    rows = (
        reopened.get_connection()
        .execute(
            "SELECT id, console_project_context_json FROM conversations ORDER BY rowid"
        )
        .fetchall()
    )
    assert success is True
    assert status.successful_items == 1
    assert len(rows) == 2
    assert rows[0]["id"] == existing_id
    assert rows[0]["console_project_context_json"] == encoded
    assert rows[1]["id"] != existing_id
    assert rows[1]["console_project_context_json"] is None
    reopened.close_connection()
