# test_import_rag_context_batching.py
# Description: RED-first coverage for batching legacy RAG-context store writes
# on chatbook import (wave-4 task 9, nonconsole efficiency remediation).
"""
The legacy RAG-context store in ``Chat/chat_conversation_service.py`` is a
JSON sidecar written by ``_save_rag_context_store`` -- one ``json.dumps`` of
the ENTIRE cross-conversation store plus one temp-file-rewrite per call. The
chatbook importer's recovery-mode fallback (``record_imported_legacy_citation_
context`` -> ``record_message_rag_context``) used to call it once per imported
cited message, making an N-cited-message import cost N full-store
serialize+write cycles (O(N x store-bytes) disk traffic).

These tests pin the batched contract:

* an import of a 500-cited-message chatbook triggers exactly ONE store
  serialize+write (spy on ``chat_source_participants.write_text`` -- the
  single file-write seam ``_save_rag_context_store`` uses; one call there is
  one full-store serialize+write);
* the flushed store content is byte-identical to replaying per-message
  ``record_message_rag_context`` over the same records (equivalence pin,
  frozen clock so ``last_modified`` matches);
* single-message ``record_message_rag_context`` still flushes immediately
  (unchanged behavior for genuine one-off recovery callers), while the
  staging API itself never touches the file until ``flush_rag_context_store``;
* an error mid-import still flushes already-staged records (flush-on-error
  direction) -- matching the old per-message path, where records persisted
  before the error stayed persisted.
"""

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

# These tests drive the real importer -> CharactersRAGDB -> config-participant
# path, which the per-test sandbox breaks (admission binds at collection
# time); keep the bootstrap profile like the other full round-trip suites.
pytestmark = pytest.mark.bootstrap_profile

import tldw_chatbook.Chatbooks.chatbook_importer as importer_module
from tldw_chatbook.Backup_Recovery import chat_source_participants
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

NUM_CONVERSATIONS = 2
MESSAGES_PER_CONVERSATION = 250  # 500 cited messages total, per the AC
STORE_FILENAME = "tldw_chatbook_chat_rag_context.json"
_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _cited_message(conversation_index: int, message_index: int) -> dict:
    timestamp = (_BASE_TIME + timedelta(minutes=message_index)).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "role": "user" if message_index % 2 == 0 else "assistant",
        "content": f"conv {conversation_index} message {message_index}",
        "timestamp": timestamp,
        "rag_context": {
            "search_query": f"query-{conversation_index}-{message_index}",
            "chunks": [{"source_id": f"note-{conversation_index}-{message_index}"}],
        },
        "citations": [
            {
                "id": f"cite-{conversation_index}-{message_index}",
                "source_id": f"note-{conversation_index}-{message_index}",
                "quote": "fact",
            }
        ],
    }


def _build_cited_chatbook(tmp_path: Path) -> tuple[Path, list[list[dict]]]:
    """A V1 chatbook whose every message carries rag_context + citations.

    Returns ``(archive_path, per_conversation_message_payloads)`` so tests can
    replay the identical payloads through the per-message writer.
    """
    archive_path = tmp_path / "cited_chatbook.zip"
    created = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    content_items = []
    conversation_files = {}
    payloads = []
    for i in range(NUM_CONVERSATIONS):
        conv_id = f"conv-{i}"
        title = f"Cited Conversation {i}"
        content_items.append(
            {
                "id": conv_id,
                "type": "conversation",
                "title": title,
                "created_at": created,
                "file_path": f"content/conversations/conversation_{i}.json",
            }
        )
        messages = [
            _cited_message(i, j) for j in range(MESSAGES_PER_CONVERSATION)
        ]
        payloads.append(messages)
        conversation_files[f"content/conversations/conversation_{i}.json"] = {
            "id": conv_id,
            "name": title,
            "title": title,
            "created_at": created,
            "messages": messages,
        }

    manifest = {
        "version": "1.0",
        "name": "Cited Messages Chatbook",
        "description": "Synthetic fixture for legacy RAG-context store batching",
        "author": "Test",
        "created_at": created,
        "updated_at": created,
        "content_items": content_items,
        "relationships": [],
        "include_media": False,
        "include_embeddings": False,
        "media_quality": "thumbnail",
        "statistics": {
            "total_conversations": NUM_CONVERSATIONS,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_size_bytes": 0,
        },
        "tags": [],
        "categories": [],
        "language": "en",
        "license": None,
    }

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file_path, content in conversation_files.items():
            zf.writestr(file_path, json.dumps(content, indent=2))

    return archive_path, payloads


@pytest.fixture
def store_write_spy(monkeypatch):
    """Count legacy-store serialize+write cycles.

    ``_save_rag_context_store`` serializes the whole store and hands the text
    to ``chat_source_participants.write_text``; one intercepted call from a
    ``ChatConversationService`` source is therefore exactly one full-store
    serialize+write. Patching the module attribute (not the guarded method on
    the class) keeps the ``@_chat_sources.guarded`` integrity checks intact.
    """
    calls = {"n": 0, "texts": []}
    original = chat_source_participants.write_text

    def counting_write_text(source, text):
        if isinstance(source, ChatConversationService):
            calls["n"] += 1
            calls["texts"].append(text)
        return original(source, text)

    monkeypatch.setattr(chat_source_participants, "write_text", counting_write_text)
    return calls


@pytest.fixture
def recovery_mode_import(tmp_path, monkeypatch):
    """Isolate the importer's citation service in recovery mode.

    Redirects the importer's user-data dir into the test sandbox and stubs the
    citation composition with a bare ``ChatConversationService`` (no migration
    attached), which is exactly the recovery-mode fallback configuration.
    Services created during an import are collected so tests can wrap them
    (e.g. to inject a mid-import failure).
    """
    created: list[ChatConversationService] = []

    def build_local(db, *, sidecar_path):
        service = ChatConversationService(db, rag_context_store_path=sidecar_path)
        created.append(service)
        return (service, None, None)

    monkeypatch.setattr(importer_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        importer_module,
        "build_local_citation_conversation_service",
        build_local,
    )
    return {"user_data_dir": tmp_path, "created_services": created}


def _run_import(db_path: Path, archive_path: Path) -> ImportStatus:
    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()
    success, message = importer.import_chatbook(
        chatbook_path=archive_path,
        conflict_resolution=ConflictResolution.SKIP,
        import_status=status,
    )
    assert success, message
    return status


def test_import_flushes_legacy_rag_context_store_once(
    tmp_path, recovery_mode_import, store_write_spy
):
    """A 500-cited-message import must serialize+write the store exactly once."""
    archive_path, _payloads = _build_cited_chatbook(tmp_path)
    db_path = tmp_path / "databases" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    status = _run_import(db_path, archive_path)

    assert status.successful_items == NUM_CONVERSATIONS
    assert status.failed_items == 0
    store_path = recovery_mode_import["user_data_dir"] / STORE_FILENAME
    assert store_path.exists()
    total_records = sum(
        len(messages)
        for messages in json.loads(store_path.read_text())["conversations"].values()
    )
    assert total_records == NUM_CONVERSATIONS * MESSAGES_PER_CONVERSATION
    assert store_write_spy["n"] == 1, (
        f"legacy RAG-context store was serialized+written "
        f"{store_write_spy['n']} times for {total_records} records; expected 1"
    )


def test_batched_store_content_is_byte_identical_to_per_message_writes(
    tmp_path, recovery_mode_import, store_write_spy, monkeypatch
):
    """The single batched flush must produce the same file the old per-message
    path produced (frozen clock so ``last_modified`` stamps are comparable)."""
    frozen = "2026-10-06T00:00:00Z"
    monkeypatch.setattr(
        ChatConversationService, "_now", staticmethod(lambda: frozen)
    )

    archive_path, payloads = _build_cited_chatbook(tmp_path)
    db_path = tmp_path / "databases" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    status = _run_import(db_path, archive_path)
    assert status.successful_items == NUM_CONVERSATIONS
    assert store_write_spy["n"] == 1

    batched_store = recovery_mode_import["user_data_dir"] / STORE_FILENAME
    assert batched_store.exists()

    # Replay the identical payloads through the per-message writer on a fresh
    # store, using the importer's own extraction helper (un-staged mode is the
    # pre-batching behavior).
    imported = CharactersRAGDB(db_path, "rag-context-replay")
    replay_store = tmp_path / "replay_rag_context.json"
    replay_service = ChatConversationService(
        imported, rag_context_store_path=replay_store
    )
    try:
        store_write_spy["n"] = 0
        replayed = 0
        for i, messages in enumerate(payloads):
            matches = imported.get_conversation_by_name(f"Cited Conversation {i}")
            assert len(matches) == 1
            conv_id = matches[0]["id"]
            rows_by_content = {
                row["content"]: row
                for row in imported.get_messages_for_conversation(
                    conv_id, limit=MESSAGES_PER_CONVERSATION + 10
                )
            }
            assert len(rows_by_content) == MESSAGES_PER_CONVERSATION
            for payload in messages:
                row = rows_by_content[payload["content"]]
                ChatbookImporter._persist_imported_message_citation_context(
                    replay_service,
                    str(conv_id),
                    str(row["id"]),
                    payload,
                )
                replayed += 1
        assert replayed == NUM_CONVERSATIONS * MESSAGES_PER_CONVERSATION
        # The replay exercised the un-batched path: one write per record.
        assert store_write_spy["n"] == replayed
    finally:
        imported.close_connection()

    assert replay_store.read_bytes() == batched_store.read_bytes()


def test_record_message_rag_context_still_flushes_immediately(
    tmp_path, recovery_mode_import, store_write_spy
):
    """Genuine one-off recovery callers keep immediate-flush semantics."""
    db_path = tmp_path / "one-off.db"
    db = CharactersRAGDB(db_path, "one-off-recovery")
    try:
        conv_id = db.add_conversation({"title": "one-off recovery"})
        msg_id = db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "Answer with citation",
                "timestamp": "2026-10-06T00:00:00Z",
            }
        )
        store_path = tmp_path / "one_off_rag_context.json"
        service = ChatConversationService(
            db, rag_context_store_path=store_path
        )

        record = service.record_message_rag_context(
            str(conv_id),
            str(msg_id),
            rag_context={"search_query": "alpha"},
            citations=[{"id": "cite-1", "source_id": "note-1"}],
        )

        assert record["message_id"] == str(msg_id)
        assert store_write_spy["n"] == 1
        assert store_path.exists()
        stored = json.loads(store_path.read_text())
        assert (
            stored["conversations"][str(conv_id)][str(msg_id)]["rag_context"]
            == {"search_query": "alpha"}
        )
        assert service._staged_rag_context_records == {}
    finally:
        db.close_connection()


def test_stage_and_flush_write_store_exactly_once(
    tmp_path, recovery_mode_import, store_write_spy
):
    """The staging API never touches the file until flush; flush writes once
    and is a no-op when nothing is staged."""
    db_path = tmp_path / "staging.db"
    db = CharactersRAGDB(db_path, "staging-recovery")
    try:
        conv_id = db.add_conversation({"title": "staging"})
        msg_ids = [
            db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "assistant",
                    "content": f"answer {i}",
                    "timestamp": "2026-10-06T00:00:00Z",
                }
            )
            for i in range(3)
        ]
        store_path = tmp_path / "staged_rag_context.json"
        service = ChatConversationService(
            db, rag_context_store_path=store_path
        )

        for i, msg_id in enumerate(msg_ids):
            record = service.record_message_rag_context(
                str(conv_id),
                str(msg_id),
                rag_context={"search_query": f"q-{i}"},
                stage=True,
            )
            assert record["message_id"] == str(msg_id)
        assert store_write_spy["n"] == 0
        assert not store_path.exists()

        service.flush_rag_context_store()
        assert store_write_spy["n"] == 1
        stored = json.loads(store_path.read_text())
        assert set(stored["conversations"][str(conv_id)]) == {
            str(msg_id) for msg_id in msg_ids
        }
        assert service._staged_rag_context_records == {}

        # Nothing staged: flush must not write again.
        service.flush_rag_context_store()
        assert store_write_spy["n"] == 1
    finally:
        db.close_connection()


def test_stage_rag_context_record_rejected_in_canonical_mode(tmp_path):
    """Staging cannot bypass the legacy-write prohibition when the canonical
    migration has writes enabled."""
    migration = SimpleNamespace(writes_enabled=True)
    service = ChatConversationService(
        FakeDBWithGetMessage(),
        rag_context_store_path=tmp_path / "canonical.json",
        citation_legacy_migration=migration,
    )

    with pytest.raises(RuntimeError, match="legacy_rag_context_writes_disabled"):
        service.stage_rag_context_record(
            "conv-1",
            "msg-1",
            {
                "rag_context": {"search_query": "alpha"},
                "citations": [],
                "last_modified": "2026-10-06T00:00:00Z",
            },
        )
    assert service._staged_rag_context_records == {}
    assert not service.rag_context_store_path.exists()


def test_mid_import_error_still_flushes_pre_error_staged_records(
    tmp_path, recovery_mode_import, store_write_spy
):
    """Flush-on-error direction: records staged before a mid-import failure
    must still land in the store (one write), matching the old observable
    behavior where records persisted before the error stayed persisted."""
    archive_path = _build_small_two_conversation_chatbook(tmp_path)
    db_path = tmp_path / "databases" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail the 5th citation persistence call: conversation 0's three records
    # plus conversation 1's first record are already staged when it raises.
    calls = {"n": 0}

    def wrap_for_failure(service):
        original = service.record_imported_legacy_citation_context

        def failing(conv_id, message_id, **kwargs):
            calls["n"] += 1
            if calls["n"] == 5:
                raise RuntimeError("simulated citation persistence failure")
            return original(conv_id, message_id, **kwargs)

        service.record_imported_legacy_citation_context = failing

    # The services are constructed lazily during import, so wrap at creation.
    original_build = importer_module.build_local_citation_conversation_service

    def wrapping_build_local(db, *, sidecar_path):
        result = original_build(db, sidecar_path=sidecar_path)
        wrap_for_failure(result[0])
        return result

    importer_module.build_local_citation_conversation_service = wrapping_build_local

    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()
    success, _message = importer.import_chatbook(
        chatbook_path=archive_path,
        conflict_resolution=ConflictResolution.SKIP,
        import_status=status,
    )
    assert success  # one conversation failed, the rest imported
    assert status.successful_items == 1
    assert status.failed_items == 1

    store_path = recovery_mode_import["user_data_dir"] / STORE_FILENAME
    assert store_path.exists()
    conversations = json.loads(store_path.read_text())["conversations"]
    total_records = sum(len(messages) for messages in conversations.values())
    assert total_records == 4, (
        f"expected the 4 records staged before the failure to be flushed; "
        f"got {total_records}"
    )
    assert store_write_spy["n"] == 1


def _build_small_two_conversation_chatbook(tmp_path: Path) -> Path:
    """Two conversations x three cited messages each (error-injection fixture)."""
    archive_path = tmp_path / "small_cited_chatbook.zip"
    created = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    content_items = []
    conversation_files = {}
    for i in range(2):
        conv_id = f"conv-{i}"
        title = f"Small Cited Conversation {i}"
        content_items.append(
            {
                "id": conv_id,
                "type": "conversation",
                "title": title,
                "created_at": created,
                "file_path": f"content/conversations/small_conversation_{i}.json",
            }
        )
        conversation_files[f"content/conversations/small_conversation_{i}.json"] = {
            "id": conv_id,
            "name": title,
            "title": title,
            "created_at": created,
            "messages": [_cited_message(i, j) for j in range(3)],
        }
    manifest = {
        "version": "1.0",
        "name": "Small Cited Chatbook",
        "description": "Error-injection fixture",
        "author": "Test",
        "created_at": created,
        "updated_at": created,
        "content_items": content_items,
        "relationships": [],
        "include_media": False,
        "include_embeddings": False,
        "media_quality": "thumbnail",
        "statistics": {
            "total_conversations": 2,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_size_bytes": 0,
        },
        "tags": [],
        "categories": [],
        "language": "en",
        "license": None,
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file_path, content in conversation_files.items():
            zf.writestr(file_path, json.dumps(content, indent=2))
    return archive_path


class FakeDBWithGetMessage:
    """Minimal DB double: message lookup says the row exists."""

    def get_message_by_id(self, message_id):
        return {"id": message_id, "conversation_id": "conv-1"}
