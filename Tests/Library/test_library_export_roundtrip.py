"""Library chatbook export round-trip: Task 1 resolver -> real service -> real importer.

Exercises the FULL F4 path with no mocking of the chatbook internals:
``resolve_export_selections`` (Task 1) resolves ids from real, file-backed
DBs -> ``LibraryScreen._build_library_export_payload``/
``_run_library_export_via_service`` (Task 3) drive the REAL
``LocalChatbookService`` (the exact static helpers the export worker calls)
-> the resulting zip is inspected directly -> ``ChatbookImporter`` re-
imports it into fresh DBs and item counts are checked.

``LocalChatbookService`` opens its own DB connections from file paths
internally (via ``ChatbookCreator``/``ChatbookImporter``), so this suite
always uses real, ``tmp_path``-backed ``MediaDatabase``/``CharactersRAGDB``
instances -- never ``:memory:`` (a fresh connection to ``:memory:`` is a
distinct, empty database, so the service could never see seeded data).
"""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Chatbooks import LocalChatbookService
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_export_scope import ExportScope, resolve_export_selections
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

# Pinned transcript text -- the round-trip assertion below checks this
# EXACT string survives inside the zip and after re-import, not merely
# that "a media item" (any count) made it across.
_MEDIA_TRANSCRIPT = "EXACT TRANSCRIPT TEXT FOR THE F4 ROUNDTRIP PIN."

pytestmark = pytest.mark.integration


def _seed_source_dbs(tmp_path: Path) -> dict:
    db_dir = tmp_path / "source_dbs"
    db_dir.mkdir()
    db_paths = {
        "ChaChaNotes": str(db_dir / "ChaChaNotes.db"),
        "Media": str(db_dir / "Client_Media_DB.db"),
        "Prompts": str(db_dir / "Prompts_DB.db"),
    }
    chachanotes_db = CharactersRAGDB(db_paths["ChaChaNotes"], "f4-roundtrip-source")
    media_db = MediaDatabase(db_paths["Media"], "f4-roundtrip-source")

    char_id = chachanotes_db.add_character_card(
        {
            "name": "Roundtrip Assistant",
            "description": "A character auto-included as a conversation dependency.",
            "personality": "Helpful",
            "scenario": "",
            "greeting_message": "Hello!",
            "example_messages": "",
            "version": 1,
        }
    )
    conv_id = chachanotes_db.add_conversation(
        {
            "title": "Roundtrip Conversation",
            "root_id": "roundtrip-root",
            "character_id": char_id,
        }
    )
    chachanotes_db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "Hello there"}
    )
    note_id = chachanotes_db.add_note(title="Roundtrip Note", content="Roundtrip note content")

    media_id, _msg, _status = media_db.add_media_with_keywords(
        url="https://example.com/roundtrip-media",
        title="Roundtrip Media",
        media_type="video",
        content=_MEDIA_TRANSCRIPT,
        keywords=["roundtrip"],
    )

    return {
        "db_paths": db_paths,
        "chachanotes_db": chachanotes_db,
        "media_db": media_db,
        "char_id": char_id,
        "conv_id": conv_id,
        "note_id": note_id,
        "media_id": media_id,
    }


def test_library_export_roundtrip_everything_scope_through_real_service_and_importer(
    tmp_path,
):
    """Full path: Task 1 resolver -> real ``LocalChatbookService`` (Task 3's
    own execution helpers) -> zip inspection (manifest counts AND actual
    media text) -> ``ChatbookImporter`` into fresh DBs -> counts match."""
    seeded = _seed_source_dbs(tmp_path)

    scope = ExportScope(kind="everything")
    selections = resolve_export_selections(scope, seeded["media_db"], seeded["chachanotes_db"])
    assert ContentType.MEDIA in selections
    assert ContentType.CONVERSATION in selections
    assert ContentType.NOTE in selections

    payload = LibraryScreen._build_library_export_payload(
        name="Roundtrip Export",
        description="Exported by the F4 roundtrip test",
        selections=selections,
        destination=str(tmp_path / "export.zip"),
        media_quality="thumbnail",
    )
    # Spec-critical invariant, pinned end-to-end (not just at the unit
    # level): a media ID is in scope, so include_media must be True, or
    # ChatbookCreator silently drops all media from the zip below.
    assert payload["include_media"] is True

    registry_path = tmp_path / "chatbooks.json"
    service = LocalChatbookService(seeded["db_paths"], registry_path=registry_path)

    outcome = LibraryScreen._run_library_export_via_service(
        service,
        payload,
        name="Roundtrip Export",
        description="Exported by the F4 roundtrip test",
    )

    assert outcome["success"] is True, outcome["message"]
    assert outcome["registry_recorded"] is True
    export_path = Path(outcome["path"])
    assert export_path.exists()
    assert export_path.stat().st_size > 0

    # (a) Zip inspection: manifest counts AND actual media content text.
    with zipfile.ZipFile(export_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["statistics"]["total_conversations"] == 1
        assert manifest["statistics"]["total_notes"] == 1
        assert manifest["statistics"]["total_media_items"] == 1
        # The character is auto-included as the conversation's dependency.
        assert manifest["statistics"]["total_characters"] == 1

        namelist = zf.namelist()
        media_content_names = [
            name
            for name in namelist
            if name.startswith("content/media/") and name.endswith(".txt")
        ]
        assert len(media_content_names) == 1
        assert zf.read(media_content_names[0]).decode("utf-8") == _MEDIA_TRANSCRIPT

        # Task 155: ``_collect_media`` previously read ``media_type``/
        # ``created_at``/``updated_at`` off the Media row dict, but the
        # real ``MediaDatabase`` columns are ``type``/``ingestion_date``/
        # ``last_modified`` -- every export silently lost the media type
        # and both timestamps. Assert the exported metadata JSON carries
        # the correct type and a real (non-None) timestamp.
        media_metadata_names = [
            name
            for name in namelist
            if name.startswith("content/media/metadata/") and name.endswith(".json")
        ]
        assert len(media_metadata_names) == 1
        media_metadata = json.loads(zf.read(media_metadata_names[0]))
        assert media_metadata["media_type"] == "video"
        assert media_metadata["created_at"] is not None
        assert media_metadata["updated_at"] is not None

    # Registry record was created (zip succeeded).
    listed = asyncio.run(service.list_chatbooks())
    assert any(record["file_path"] == str(export_path) for record in listed)

    # (b) Re-import into FRESH DBs via the real ChatbookImporter.
    import_dir = tmp_path / "import_dbs"
    import_dir.mkdir()
    import_db_paths = {
        "ChaChaNotes": str(import_dir / "ChaChaNotes.db"),
        "Media": str(import_dir / "Client_Media_DB.db"),
        "Prompts": str(import_dir / "Prompts_DB.db"),
    }
    import_chachanotes_db = CharactersRAGDB(import_db_paths["ChaChaNotes"], "f4-roundtrip-import")
    import_media_db = MediaDatabase(import_db_paths["Media"], "f4-roundtrip-import")

    importer = ChatbookImporter(import_db_paths)
    status = ImportStatus()
    success, message = importer.import_chatbook(export_path, import_status=status)

    assert success is True, message
    assert status.errors == []
    # conversation + note + character (auto-included dependency) + media.
    assert status.successful_items == 4

    imported_convs = import_chachanotes_db.search_conversations_by_title(
        "Roundtrip Conversation"
    )
    assert len(imported_convs) == 1
    imported_notes = import_chachanotes_db.search_notes("Roundtrip Note")
    assert len(imported_notes) == 1
    imported_chars = import_chachanotes_db.list_character_cards()
    assert any(char["name"] == "Roundtrip Assistant" for char in imported_chars)

    imported_media = import_media_db.get_media_by_url("https://example.com/roundtrip-media")
    assert imported_media is not None
    assert imported_media["content"] == _MEDIA_TRANSCRIPT
    # The media type must survive the export -> import round-trip too.
    assert imported_media["type"] == "video"


def test_library_export_roundtrip_conversations_only_scope_excludes_media_and_notes(
    tmp_path,
):
    """A conversations-only scope must resolve NO media/notes selections at
    all -- ``include_media`` comes out ``False`` because ``ContentType.
    MEDIA`` is never even in the resolved dict (Task 1's per-source
    scoping), and the zip must contain zero media/notes content items."""
    seeded = _seed_source_dbs(tmp_path)

    scope = ExportScope(kind="conversations")
    selections = resolve_export_selections(scope, seeded["media_db"], seeded["chachanotes_db"])
    assert ContentType.CONVERSATION in selections
    assert ContentType.MEDIA not in selections
    assert ContentType.NOTE not in selections

    payload = LibraryScreen._build_library_export_payload(
        name="Conversations Only",
        description="",
        selections=selections,
        destination=str(tmp_path / "conversations_only.zip"),
        media_quality="thumbnail",
    )
    assert payload["include_media"] is False

    service = LocalChatbookService(seeded["db_paths"], registry_path=tmp_path / "chatbooks.json")
    outcome = LibraryScreen._run_library_export_via_service(
        service, payload, name="Conversations Only", description=""
    )

    assert outcome["success"] is True, outcome["message"]
    with zipfile.ZipFile(Path(outcome["path"]), "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["statistics"]["total_conversations"] == 1
        assert manifest["statistics"]["total_notes"] == 0
        assert manifest["statistics"]["total_media_items"] == 0
        # The character dependency auto-inclusion is per-conversation, not
        # scope-gated -- a narrower "conversations" scope must still pull
        # in the referenced character, exactly like the "everything" scope
        # does above.
        assert manifest["statistics"]["total_characters"] == 1
        assert not any(name.startswith("content/media/") for name in zf.namelist())


def test_library_export_roundtrip_unwritable_destination_fails_with_no_registry_record(
    tmp_path,
):
    """A destination that is already a DIRECTORY (not a file) is
    unwritable as a zip target -- ``ChatbookCreator`` wraps the resulting
    ``IsADirectoryError`` into ``(False, message, ...)`` rather than
    raising, ``LocalChatbookService.export_chatbook`` surfaces
    ``success=False``, and -- per "zip first, registry record only on
    success" -- no ``create_chatbook`` record is ever created."""
    seeded = _seed_source_dbs(tmp_path)

    destination_dir = tmp_path / "not-a-file.zip"
    destination_dir.mkdir()  # A directory, already suffixed .zip.

    scope = ExportScope(kind="notes")
    selections = resolve_export_selections(scope, seeded["media_db"], seeded["chachanotes_db"])
    payload = LibraryScreen._build_library_export_payload(
        name="Should Fail",
        description="",
        selections=selections,
        destination=str(destination_dir),
        media_quality="thumbnail",
    )

    registry_path = tmp_path / "chatbooks.json"
    service = LocalChatbookService(seeded["db_paths"], registry_path=registry_path)

    outcome = LibraryScreen._run_library_export_via_service(
        service, payload, name="Should Fail", description=""
    )

    assert outcome["success"] is False
    assert outcome["registry_recorded"] is False
    assert outcome["message"]

    listed = asyncio.run(service.list_chatbooks())
    assert listed == []
