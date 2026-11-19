"""The original local trace owner holds the single file grant through SQLite completion."""

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from Tests.private_profile import private_profile_test


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")


@private_profile_test
@pytest.mark.asyncio
async def test_original_trace_checkpoint_policy_keeps_its_borrowed_handle_until_restore(request, monkeypatch):
    from Tests.Chat.test_console_trace_settlement import _call
    from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.DB import profile_writer_chacha_grant
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), "trace-policy")
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository, sanitizer=CredentialSanitizer(known_credentials=("local",)))
    released, finish = Event(), Event()
    original_grant = profile_writer_chacha_grant._local_chacha_transaction

    @contextmanager
    def observe_outer_policy(database, kind):
        with original_grant(database, kind) as grant:
            yield grant
        connection = grant.conn
        assert connection is grant.prior is database._local.conn
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
        assert database._connection_quiescence.is_registered(connection)
        with storage._lock:
            assert participants.repository_participant(database).connections[connection] in storage._live_leases
            assert storage._operation_local.operation in storage._operations
        released.set()
        assert finish.wait(20)
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0

    def mark_started(call):
        connection = db.get_connection()
        try:
            connection.execute("PRAGMA wal_autocheckpoint=17")
            result = coordinator.mark_response_started(db, call_id=call, occurred_at="2026-09-01T00:00:00Z")
            assert connection is db.get_connection()
            assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 17
            return result.call_id
        finally:
            db.close_connection()

    try:
        _, _, call = _call(db, repository)
        observer = db.get_connection()
        observer.execute("PRAGMA wal_autocheckpoint=23")
        monkeypatch.setattr(profile_writer_chacha_grant, "_local_chacha_transaction", observe_outer_policy)
        with ThreadPoolExecutor(1) as pool:
            marked = pool.submit(mark_started, call)
            try:
                assert released.wait(20)
                assert db.add_note("Peer", "Concurrent commit", "independent-policy") == "independent-policy"
                assert observer.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 23
            finally:
                finish.set()
            assert marked.result(timeout=20) == call
    finally:
        finish.set()
        db.close_connection()


@private_profile_test
@pytest.mark.asyncio
async def test_original_credential_settlement_fences_other_note_writers_and_keeps_secrets_local(request, monkeypatch):
    from Tests.Chat.test_console_trace_settlement import _call, _request
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.profile_writer_chacha_grant import ChaChaTransactionGrant
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), "local-trace-grant")
    repository = ConsoleTraceRepository()
    secret = "credential-known-only-to-this-settlement"
    coordinator = ConsoleTraceSettlementCoordinator(repository, sanitizer=CredentialSanitizer(known_credentials=(secret,)))
    bound, release, competing = Event(), Event(), Event()
    offered = []
    encode = profile_writer._encode
    bind = ChaChaTransactionGrant.bind

    def capture(value):
        body = encode(value)
        assert secret.encode() not in body
        if value.get("intent") == "grant":
            offered.append(value["data"])
        if value.get("intent") == "call" and value["data"]["operation"] == "chacha.db.add_note":
            competing.set()
        return body

    def stop_before_begin(grant, connection):
        bind(grant, connection)
        bound.set()
        assert release.wait(15)

    try:
        _, _, call_id = _call(db, repository)
        full_request = _request(call_id, response={"role": "assistant", "content": secret})
        monkeypatch.setattr(profile_writer, "_encode", capture)
        monkeypatch.setattr(ChaChaTransactionGrant, "bind", stop_before_begin)
        with ThreadPoolExecutor(max_workers=2) as workers:
            settlement = workers.submit(coordinator.settle, db, full_request)
            try:
                assert bound.wait(15)
                other = workers.submit(db.add_note, "Competing note", "Body", "grant-peer-note")
                assert competing.wait(5)
                assert not other.done()
                assert db.get_note_by_id("grant-peer-note") is None
            finally:
                release.set()
            assert settlement.result(timeout=20).call_id == call_id
            assert other.result(timeout=20) == "grant-peer-note"
        assert len(offered) == 1
        assert offered[0]["kind"] == "chacha_local.trace_settlement"
        assert set(offered[0]) == {"kind", "database"}
        link = repository.get_response_link(db.get_connection().cursor(), call_id)
        assert secret.encode() not in repository.get_artifact(db.get_connection().cursor(), link.artifact_id).sanitized_bytes
    finally:
        release.set()
        db.close_connection()


@pytest.mark.parametrize("borrowed", (False, True), ids=("fresh", "borrowed"))
@private_profile_test
@pytest.mark.asyncio
async def test_original_notes_save_rolls_back_and_retires_only_its_own_handle(request, monkeypatch, borrowed):
    from Tests.Notes.test_note_organization_transaction import USER_ID, _checkpoint, _counts, _save
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.profile_writer_chacha import _database_request
    from tldw_chatbook.DB.profile_writer_chacha_grant import ChaChaTransactionGrant
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), USER_ID)
    handles = []
    bind = ChaChaTransactionGrant.bind

    def capture(grant, connection):
        bind(grant, connection)
        handles.append(connection)

    def fail_after_original_write(stage):
        if stage == "after_note_write":
            assert handles[-1].in_transaction
            with pytest.raises(RecoveryRequired, match="profile_writer_local_transaction_active"):
                profile_writer.call_profile_writer(db.db_path.parent, "chacha.db.add_note", _database_request(
                    db, {"title": "Nested", "content": "Body", "note_id": "reentrant-note"},
                ))
            raise RuntimeError("after_note_write")

    try:
        notes = NotesInteropService(db.db_path.parent, USER_ID, db, failure_injector=fail_after_original_write)
        notes._db_instances[USER_ID] = db
        _checkpoint(db, ready=True)
        before = _counts(db)
        db.close_connection()
        prior = db.get_connection() if borrowed else None
        monkeypatch.setattr(ChaChaTransactionGrant, "bind", capture)
        with pytest.raises(RuntimeError, match="after_note_write"):
            _save(notes)
        (original,) = handles
        if borrowed:
            assert original is prior and not prior.in_transaction
            assert db.get_connection() is prior
        else:
            assert not db._connection_quiescence.is_registered(original)
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                original.execute("SELECT 1")
        assert _counts(db) == before
    finally:
        db.close_connection()


@private_profile_test
@pytest.mark.asyncio
async def test_successor_refuses_chatbook_import_while_original_notes_transaction_is_live(request):
    import _winapi
    import base64
    import json
    import zipfile
    from io import BytesIO

    from Tests.DB.test_profile_writer_transport import _wait_for_owned_server
    from Tests.Notes.test_note_organization_transaction import USER_ID, _checkpoint, _save
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chatbooks.chatbook_importer import ImportStatus
    from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion, ContentItem, ContentType
    from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.config import get_chachanotes_db_path, get_user_data_dir

    profile = get_user_data_dir()
    writing, finish = Event(), Event()
    manifest = ChatbookManifest(version=ChatbookVersion.V1, name="Peer", description="Grant test")
    manifest.content_items.append(ContentItem("peer", ContentType.NOTE, "Peer", file_path="content/notes/peer.md"))
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("manifest.json", json.dumps(manifest.to_dict()))
        zipped.writestr("content/notes/peer.md", "Imported body")
    arguments = {
        "stores": ["ChaChaNotes"], "resolution": ConflictResolution.SKIP.value,
        "selections": {ContentType.NOTE.value: ["peer"]}, "prefix": False, "media": False,
        "status": ImportStatus().to_dict(), "zip_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
    }

    def pause(stage):
        if stage == "after_note_write":
            writing.set()
            assert finish.wait(30)

    host = profile_writer.start_profile_writer(profile)
    process = None
    db = None
    try:
        db = CharactersRAGDB(get_chachanotes_db_path(), USER_ID)
        notes = NotesInteropService(db.db_path.parent, USER_ID, db, failure_injector=pause)
        notes._db_instances[USER_ID] = db
        _checkpoint(db, ready=True)
        _, process = _wait_for_owned_server(profile, terminate=True)
        with ThreadPoolExecutor(1) as pool:
            save = pool.submit(_save, notes)
            try:
                assert writing.wait(20)
                _winapi.TerminateProcess(process, 57)
                assert _winapi.WaitForSingleObject(process, 5000) == _winapi.WAIT_OBJECT_0
                with pytest.raises(RecoveryRequired, match="profile_writer_local_transaction_active"):
                    profile_writer.call_profile_writer(profile, "chatbook.import", arguments)
                assert db.get_note_by_title("Peer") is None
            finally:
                finish.set()
            with pytest.raises(profile_writer.ProfileWriterOutcomeUnknown):
                save.result(timeout=20)
        assert profile_writer.call_profile_writer(profile, "chatbook.import", arguments)["success"]
        assert db.get_note_by_title("Peer")["content"] == "Imported body"
    finally:
        finish.set()
        if db is not None:
            db.close_connection()
        if process is not None:
            _winapi.CloseHandle(process)
        host.close()
