"""Windows stock Notes, semantic mutations and Chatbook import across the writer."""

import json
import os
import sqlite3
import sys
import zipfile
from contextlib import closing

import pytest

from Tests.private_profile import private_profile_test


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")


def _unexpected_local_write(*args, **kwargs):
    raise AssertionError("the client entered the SQLite mutation")


@private_profile_test
@pytest.mark.asyncio
async def test_console_acceptance_dispatch_and_terminal_receipt_share_writer(request, monkeypatch, tmp_path):
    from dataclasses import replace

    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        TERMINAL_METADATA, TERMINAL_RECEIPT, _acceptance, _start_dispatch,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleAssistantSettlement, ConsoleDispatchResultStatus
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate
    from tldw_chatbook.Chat.library_preparation import LibraryPreparationContribution, LibraryPreparationEvent
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    count_file = tmp_path / "schema-initializations.txt"
    _count_owner_schema_initializations(monkeypatch, count_file)
    host = profile_writer.start_profile_writer(get_user_data_dir())
    db = CharactersRAGDB(get_chachanotes_db_path(), "turn-owner")
    try:
        service = ChatPersistenceService(db)
        accepted = _acceptance("writer-conversation")
        policy = replace(accepted.frozen_authority.policy, source="new_session", policy_revision=None)
        accepted = replace(
            accepted, frozen_authority=replace(accepted.frozen_authority, policy=policy),
            contributions=(LibraryPreparationContribution(
                LibraryPreparationEvent(1, "zero_matches", accepted.attempt_id, 0, ("notes",)),
            ),),
        )
        with monkeypatch.context() as patch:
            for name in ("insert_with_messages", "_cas_state_uncoordinated", "_settle_with_assistant_uncoordinated"):
                patch.setattr(ConsoleDispatchRepository, name, _unexpected_local_write)
            checkpoint = service.commit_durable_turn(
                acceptance=accepted, policy_candidate=ConsoleLibraryPolicyCandidate(policy.auto_retrieve, policy.assistant_access),
                conversation_kwargs={"conversation_title": "Writer dispatch"},
            )
            started = _start_dispatch(service.console_dispatch_repository, checkpoint)
            assert db.add_note("While waiting", "Another client can write.", "while-waiting")
            settled = service.console_dispatch_repository.settle_with_assistant(
                ConsoleAssistantSettlement(
                    started.assistant_message_id, started.state, started.checkpoint_revision,
                    started.user_message_version, started.assistant_message_version,
                    "complete", "Finished", TERMINAL_METADATA, terminal_receipt_id=TERMINAL_RECEIPT,
                ),
            )
        assert settled.status is ConsoleDispatchResultStatus.COMMITTED
        assert db.get_message_by_id(accepted.assistant_message_id)["content"] == "Finished"
        assert service.console_dispatch_repository.read_for_session(accepted.conversation_id).checkpoint is None
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM message_trajectory_metadata WHERE event_kind = 'library_preparation'",
        ).fetchone()[0] == 1
        assert count_file.read_text().splitlines().count("turn-owner") == 1
        # A new logical opener explicitly requests its own initialization,
        # even when its sync identity and migration seed match an existing one.
        second = CharactersRAGDB(get_chachanotes_db_path(), "turn-owner")
        second.close_connection()
        assert count_file.read_text().splitlines().count("turn-owner") == 2
        db.get_connection().execute("CREATE INDEX turn_writer_probe ON notes(title)")
        assert db.add_note("After DDL", "Body", "after-ddl")
        assert count_file.read_text().splitlines().count("turn-owner") == 3
    finally:
        db.close_connection()
        host.close()


def _count_owner_schema_initializations(monkeypatch, count_file):
    """Instrument the real method only in the independently elected interpreter."""
    instrumentation = f"""
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
original_initialize = CharactersRAGDB._initialize_schema
def counted_initialize(db):
    with open({str(count_file)!r}, 'a', encoding='utf-8') as stream:
        stream.write(db.client_id + '\\n')
    return original_initialize(db)
CharactersRAGDB._initialize_schema = counted_initialize
"""
    _instrument_writer(monkeypatch, instrumentation)


def _instrument_writer(monkeypatch, instrumentation):
    from tldw_chatbook.DB import profile_writer

    original_popen = profile_writer.subprocess.Popen

    def launch(argv, **kwargs):
        argv = list(argv)
        code = argv.index("-c") + 1
        argv[code] = argv[code].replace("raise SystemExit(_serve(", f"exec({instrumentation!r});raise SystemExit(_serve(")
        return original_popen(argv, **kwargs)

    monkeypatch.setattr(profile_writer.subprocess, "Popen", launch)


@private_profile_test
@pytest.mark.asyncio
async def test_schema_repair_and_write_share_one_connection_then_allow_maintenance(request, tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE, admission_authority
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Utils.platform_files import os as files

    events = tmp_path / "connections.txt"
    _instrument_writer(monkeypatch, f"""
from tldw_chatbook.DB import ChaChaNotes_DB as notes
from tldw_chatbook.DB.profile_writer import in_profile_writer
original_connect = notes.connect_private_sqlite
def counted_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    if in_profile_writer():
        with open({str(events)!r}, 'a', encoding='utf-8') as stream:
            print('open', file=stream)
    return connection
notes.connect_private_sqlite = counted_connect
""")
    path = get_chachanotes_db_path()
    host = profile_writer.start_profile_writer(get_user_data_dir())
    try:
        first = CharactersRAGDB(path, "first-client")
        second = CharactersRAGDB(path, "second-client")
        first.add_note("First", "Body", "first")
        with closing(sqlite3.connect(path)) as observer:
            observer.execute("DROP INDEX idx_notes_organization_heads_note_subject")
        before = events.read_text().splitlines()
        second.add_note("Second", "Body", "second")
        assert events.read_text().splitlines() == [*before, "open"]
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute(
                "SELECT id, client_id FROM notes WHERE id IN ('first','second') ORDER BY id"
            ).fetchall() == [("first", "first-client"), ("second", "second-client")]
            assert observer.execute(
                "SELECT name FROM sqlite_schema WHERE name='idx_notes_organization_heads_note_subject'"
            ).fetchone()
        replacement = CharactersRAGDB(path.with_name("incoming.sqlite"), "restored-client")
        replacement.add_note("Restored", "Body", "restored")
        replacement.close_connection()
        files.chmod(replacement.db_path, 0o600)
        storage_admission._shutdown()
        authority = admission_authority(bootstrap.default_bootstrap_root())
        with authority.maintenance((UNBOUND_NAMESPACE,), 5):
            os.replace(replacement.db_path, path)
        first.add_note("After restore", "Body", "after-restore")
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT id FROM notes ORDER BY id").fetchall() == [
                ("after-restore",), ("restored",),
            ]
    finally:
        host.close()


@private_profile_test
@pytest.mark.asyncio
async def test_committed_note_with_unconfirmed_native_close_receives_unknown(request, tmp_path, monkeypatch):
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    fault = tmp_path / "decline-close"
    _instrument_writer(monkeypatch, f"""
from pathlib import Path
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
original_close = CharactersRAGDB.close_connection
def decline(db):
    if not Path({str(fault)!r}).exists():
        return original_close(db)
CharactersRAGDB.close_connection = decline
""")
    path = get_chachanotes_db_path()
    host = profile_writer.start_profile_writer(get_user_data_dir())
    try:
        db = CharactersRAGDB(path, "close-owner")
        fault.touch()
        with pytest.raises(profile_writer.ProfileWriterOutcomeUnknown):
            db.add_note("Committed", "Body", "committed")
        fault.unlink()
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT id FROM notes WHERE id='committed'").fetchall() == [("committed",)]
        assert db.add_note("Successor", "Body", "successor") == "successor"
    finally:
        host.close()


@private_profile_test
@pytest.mark.asyncio
async def test_cold_open_refuses_file_replaced_before_service_acknowledgment(request, monkeypatch):
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    path = get_chachanotes_db_path()
    incoming = path.with_name("incoming.sqlite")
    replacement = CharactersRAGDB(incoming, "incoming")
    replacement.add_note("Incoming", "Body", "incoming")
    replacement.close_connection()
    assert not path.exists()
    _instrument_writer(monkeypatch, f"""
import os
from tldw_chatbook.DB.profile_writer_repository import ChaChaWriterDatabase
close = ChaChaWriterDatabase.close_connection
def replace_after_close(holder):
    close(holder)
    if os.path.exists({str(incoming)!r}):
        os.replace({str(incoming)!r}, holder.database.db_path)
ChaChaWriterDatabase.close_connection = replace_after_close
""")
    host = profile_writer.start_profile_writer(get_user_data_dir())
    try:
        with pytest.raises(profile_writer.ProfileWriterOutcomeUnknown):
            CharactersRAGDB(path, "cold-opener")
        with closing(sqlite3.connect(path)) as observer:
            assert observer.execute("SELECT id FROM notes").fetchall() == [("incoming",)]
    finally:
        host.close()


@private_profile_test
@pytest.mark.asyncio
async def test_owner_reinitializes_after_replacement_config_change_and_restart(request, tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    count_file = tmp_path / "schema-initializations.txt"
    _count_owner_schema_initializations(monkeypatch, count_file)
    profile = get_user_data_dir()
    path = get_chachanotes_db_path()
    host = profile_writer.start_profile_writer(profile)
    db = CharactersRAGDB(path, "retained-opener")

    def assert_initializations(expected):
        assert count_file.read_text().splitlines().count("retained-opener") == expected

    try:
        db.add_note("Before", "Original", "before")
        assert_initializations(1)
        replacement = CharactersRAGDB(path.with_name("replacement.sqlite"), "replacement-source")
        replacement.add_note("Replacement", "New file", "replacement")
        replacement.close_connection()
        db.close_connection()
        os.replace(replacement.db_path, path)
        db.add_note("After replacement", "Body", "after-replace")
        assert_initializations(2)
        assert db.get_note_by_id("replacement") and db.get_note_by_id("before") is None
        db.close_connection()
        selector = effective_config_path()
        original = selector.stat()
        os.utime(selector, ns=(original.st_atime_ns, original.st_mtime_ns + 2_000_000_000))
        db.add_note("After config change", "Body", "after-config")
        assert_initializations(3)
        host.close()
        host = profile_writer.start_profile_writer(profile)
        db.add_note("After owner restart", "Body", "after-restart")
        assert_initializations(4)
    finally:
        db.close_connection()
        host.close()


@private_profile_test
@pytest.mark.asyncio
async def test_retained_reader_and_original_core_scope_cannot_write_another_database(request, monkeypatch):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import core_operation
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    path = get_chachanotes_db_path()
    db = CharactersRAGDB(path, "original")
    other_opener = CharactersRAGDB(path, "another-opener")
    replacement = CharactersRAGDB(path.with_name("incoming.sqlite"), "replacement")
    try:
        with core_operation(other_opener), pytest.raises(RecoveryRequired, match="operation_provenance_invalid"):
            db.add_note("Wrong participant", "Rejected", "foreign-participant")
        assert db.get_note_by_id("foreign-participant") is None
        replacement.add_note("Replacement", "New file", "replacement")
        replacement.close_connection()
        original_call = profile_writer.call_bound_profile_writer

        def replace_after_selection(bound, *arguments):
            assert bound is db
            # Finish the original Windows handle, then change the selected native file.
            db.close_connection()
            os.replace(replacement.db_path, path)
            return original_call(bound, *arguments)

        with monkeypatch.context() as patch:
            patch.setattr(profile_writer, "call_bound_profile_writer", replace_after_selection)
            with pytest.raises(RecoveryRequired, match="profile_writer_database_changed"):
                db.add_note("Replaced while being submitted", "Rejected", "wrong-file")
        assert db.get_note_by_id("wrong-file") is None
        assert db.get_note_by_id("replacement")
    finally:
        db.close_connection()
        other_opener.close_connection()
        replacement.close_connection()


@private_profile_test
@pytest.mark.asyncio
async def test_large_semantic_attachment_remains_one_committed_message(request):
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(get_chachanotes_db_path(), "large-message")
    try:
        conversation = db.add_conversation({"title": "Large attachment"})
        payload = bytes(range(256)) * (13 * 1024 * 1024 // 256)
        message = db.add_message_with_semantic_sidecars(
            {"conversation_id": conversation, "sender": "assistant", "content": "Attachment"},
            attachments=[{"position": 1, "data": payload, "mime_type": "application/octet-stream"}],
        )
        conn = db.get_connection()
        assert conn.execute("SELECT data FROM message_attachments WHERE message_id = ?", (message,)).fetchone()[0] == payload
        assert conn.execute(
            "SELECT COUNT(*) FROM console_trace_semantic_revisions WHERE live_message_id = ?", (message,),
        ).fetchone()[0] == 1
    finally:
        db.close_connection()


@private_profile_test
@pytest.mark.asyncio
async def test_profile_notes_write_and_semantic_sidecars_use_owner(request, monkeypatch):
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    path = get_chachanotes_db_path()
    db = CharactersRAGDB(path, "notes-client")
    notes = NotesInteropService(path.parent, "app", global_db_to_use=db)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(CharactersRAGDB, "_add_note_with_cursor", _unexpected_local_write)
            patch.setattr(CharactersRAGDB, "_validated_message_insert", _unexpected_local_write)
            assert db.add_note("One", "First body", "note-1") == "note-1"
            assert db.get_note_by_id("note-1")["client_id"] == "notes-client"
            with pytest.raises(ConflictError) as error:
                db.add_note("Duplicate", "Other body", "note-1")
            assert (error.value.entity, error.value.entity_id) == ("notes", "note-1")
            saved = notes.save_note_with_organization(
                "organizer", title="Two", content="Body", ensure_keywords=("alpha",),
            )
            assert db.get_note_by_id(saved["id"])["client_id"] == "organizer"
            assert [row["keyword"] for row in db.get_keywords_for_note(saved["id"])] == ["alpha"]
            chat = db.add_conversation({"title": "Writer chat"})
            message = db.add_message_with_semantic_sidecars(
                {"conversation_id": chat, "sender": "assistant", "content": "Committed"},
                attachments=[{"position": 1, "data": b"\x00\xff", "mime_type": "image/png"}],
            )
            assert db.get_message_by_id(message)["content"] == "Committed"
            assert db.get_connection().execute(
                "SELECT COUNT(*) FROM console_trace_semantic_revisions WHERE live_message_id = ?",
                (message,),
            ).fetchone()[0] == 1
            assert db.get_connection().execute(
                "SELECT data FROM message_attachments WHERE message_id = ? AND position = 1",
                (message,),
            ).fetchone()[0] == b"\x00\xff"

        conn = db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            with db.transaction() as borrowed:
                assert db.add_note("Borrowed", "Body", "rolled-back", cursor=borrowed)
            assert conn.in_transaction
        finally:
            conn.rollback()
        assert db.get_note_by_id("rolled-back") is None
    finally:
        notes.close_all_user_connections()
        db.close_connection()


@private_profile_test
@pytest.mark.asyncio
async def test_profile_import_returns_each_item_outcome_from_owner(request, tmp_path, monkeypatch):
    from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
    from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion, ContentItem, ContentType
    from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
    from tldw_chatbook.Chatbooks.database_paths import get_chatbook_database_paths
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    paths = get_chatbook_database_paths()
    importer = ChatbookImporter(paths)
    archive = tmp_path / "notes.zip"
    manifest = ChatbookManifest(version=ChatbookVersion.V1, name="Notes", description="Import test")
    manifest.content_items.append(ContentItem("kept", ContentType.NOTE, "Saved note", file_path="content/notes/one.md"))
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("manifest.json", json.dumps(manifest.to_dict()))
        zip_file.writestr("content/notes/one.md", "One note")

    reviewed, error = importer.preview_chatbook(archive)
    assert reviewed and error is None
    reviewed_digest = reviewed.reviewed_archive_sha256
    original = archive.read_bytes()
    real_call = profile_writer.call_profile_writer

    def replace_selected_source(profile, operation, arguments):
        if operation == "chatbook.import":
            assert str(archive) not in repr(arguments)
            archive.write_bytes(b"replaced after the client opened the reviewed ZIP")
        return real_call(profile, operation, arguments)

    status = ImportStatus()
    with monkeypatch.context() as patch:
        patch.setattr(ChatbookImporter, "_import_notes", _unexpected_local_write)
        patch.setattr(profile_writer, "call_profile_writer", replace_selected_source)
        ok, _ = importer.import_chatbook(
            archive, {ContentType.NOTE: ["kept", "absent"]},
            ConflictResolution.SKIP, import_status=status, reviewed_archive_sha256=reviewed_digest,
        )
    assert ok and status.outcome == "partial"
    assert (status.successful_items, status.failed_items, status.processed_items) == (1, 1, 2)
    assert status.result_for(ContentType.NOTE).failed == 1
    assert any("absent" in warning for warning in status.warnings)
    db = CharactersRAGDB(paths["ChaChaNotes"], "reader")
    try:
        assert db.get_note_by_title("Saved note")["content"] == "One note"
    finally:
        db.close_connection()
    with monkeypatch.context() as patch:
        patch.setattr(profile_writer, "call_profile_writer", _unexpected_local_write)
        refused, message = importer.import_chatbook(archive, reviewed_archive_sha256=reviewed_digest)
    assert not refused and "changed" in message.lower()
    archive.write_bytes(original)
    again = ImportStatus()
    ok, _ = importer.import_chatbook(
        archive, {ContentType.NOTE: ["kept"]}, ConflictResolution.SKIP, import_status=again,
    )
    assert ok and again.skipped_items == 1 and again.successful_items == 0
