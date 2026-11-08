"""TASK-36949.4: chat-sync persistence sits behind a Sync_Interop adapter.

ChaChaNotes_DB imports no Sync_Interop module. Its tombstone writers still
stamp each deleted message's Sync record hash inside their own write
transaction, through the DB-layer hashing port; a process with nothing
installed fails closed. The adapter's moved sync update runs its write and its
semantic coordination inside one transaction.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
from tldw_chatbook.DB import transaction_observer as port
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_sync_source import (
    CharactersRAGDBChatSyncSource,
    chat_record_payload_hash,
    chat_sync_source_for,
    install_chat_sync_hashing,
)
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "hashing-port.db", client_id="hash-test")
    yield database
    database.close_connection()


def _live_message(db) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "t"})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "hi"}
    )
    return conversation_id, message_id


def _deleted_and_version(db, message_id) -> tuple[int, int]:
    row = (
        db.get_connection()
        .execute("SELECT deleted, version FROM messages WHERE id = ?", (message_id,))
        .fetchone()
    )
    return row["deleted"], row["version"]


def _delete_intent_base_hashes(db, message_id) -> list[str | None]:
    rows = (
        db.get_connection()
        .execute(
            "SELECT payload FROM sync_log WHERE entity = 'messages' "
            "AND entity_id = ? AND operation = 'delete'",
            (message_id,),
        )
        .fetchall()
    )
    return [json.loads(row["payload"]).get("base_payload_hash") for row in rows]


def _soft_delete(db, message_id):
    return db.soft_delete_message(message_id, expected_version=1)


def _subtree_delete(db, message_id):
    return db.soft_delete_message_subtree(message_id, expected_version=1)


DELETES = pytest.mark.parametrize(
    "delete", [_soft_delete, _subtree_delete], ids=["message", "subtree"]
)


def test_db_module_imports_no_sync_interop():
    """Top-level, TYPE_CHECKING and function-level imports all count."""
    tree = ast.parse((REPO / "tldw_chatbook/DB/ChaChaNotes_DB.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None))
            == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            imported.add(str(node.args[0].value))
    assert not {name for name in imported if "Sync_Interop" in name}


@DELETES
def test_tombstone_hash_runs_inside_the_delete_transaction(db, monkeypatch, delete):
    _conversation_id, message_id = _live_message(db)
    live = db.get_message_by_id(message_id)
    seen: list[bool] = []

    def recording(row):
        seen.append(db.get_connection().in_transaction)
        return chat_record_payload_hash(row)

    monkeypatch.setattr(port, "_chat_record_hasher", recording)
    delete(db, message_id)
    assert seen == [True]
    assert _deleted_and_version(db, message_id) == (1, 2)
    assert _delete_intent_base_hashes(db, message_id) == [
        chat_record_payload_hash(live)
    ]


@DELETES
def test_hasher_failure_rolls_the_tombstone_back(db, monkeypatch, delete):
    _conversation_id, message_id = _live_message(db)

    def boom(_row):
        raise RuntimeError("hasher failed")

    monkeypatch.setattr(port, "_chat_record_hasher", boom)
    with pytest.raises(RuntimeError, match="hasher failed"):
        delete(db, message_id)
    assert _deleted_and_version(db, message_id) == (0, 1)
    assert _delete_intent_base_hashes(db, message_id) == []


@DELETES
def test_unset_port_fails_closed_and_deletes_nothing(db, monkeypatch, delete):
    _conversation_id, message_id = _live_message(db)
    monkeypatch.setattr(port, "_chat_record_hasher", None)
    with pytest.raises(port.ChatRecordHashingNotInstalledError) as info:
        delete(db, message_id)
    assert "install_chat_sync_hashing" in str(info.value)
    assert _deleted_and_version(db, message_id) == (0, 1)
    assert _delete_intent_base_hashes(db, message_id) == []


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(port, "_chat_record_hasher", None)
    install_chat_sync_hashing()
    install_chat_sync_hashing()
    assert port._chat_record_hasher is chat_record_payload_hash


def _sync_record(conversation_id: str, content: str) -> tuple[str, dict, str]:
    payload = {"assistant_generation_state": None, "content": content, "role": "user"}
    return f"{conversation_id}:sync-1", payload, canonical_payload_hash(payload)


def _content_and_version(db, message_id) -> tuple[str, int]:
    row = (
        db.get_connection()
        .execute("SELECT content, version FROM messages WHERE id = ?", (message_id,))
        .fetchone()
    )
    return row["content"], row["version"]


def test_adapter_sync_update_and_coordination_share_one_transaction(db, monkeypatch):
    conversation_id = db.add_conversation({"title": "t"})
    store = CharactersRAGDBChatSyncSource(db)
    store.append_chat_message(*_sync_record(conversation_id, "before"))
    seen: list[bool] = []

    class Recording(SemanticRevisionCoordinator):
        def mutate_message(self, cursor, **kwargs):
            seen.append(cursor.connection.in_transaction)
            return super().mutate_message(cursor, **kwargs)

    monkeypatch.setattr(port, "_factory", Recording)
    store.append_chat_message(*_sync_record(conversation_id, "after"))
    assert seen == [True]
    assert _content_and_version(db, "sync-1") == ("after", 2)


def test_coordinator_failure_rolls_the_adapter_sync_update_back(db, monkeypatch):
    conversation_id = db.add_conversation({"title": "t"})
    store = CharactersRAGDBChatSyncSource(db)
    store.append_chat_message(*_sync_record(conversation_id, "before"))

    class MutateThenFail(SemanticRevisionCoordinator):
        def mutate_message(self, cursor, *, mutate, **kwargs):
            def write_then_fail(mutation_cursor):
                mutate(mutation_cursor)
                raise RuntimeError("coordinator failed after the write")

            return super().mutate_message(cursor, mutate=write_then_fail, **kwargs)

    monkeypatch.setattr(port, "_factory", MutateThenFail)
    with pytest.raises(Exception, match="coordinator failed after the write"):
        store.append_chat_message(*_sync_record(conversation_id, "after"))
    assert _content_and_version(db, "sync-1") == ("before", 1)


def test_source_for_accepts_only_a_characters_rag_db(db):
    assert chat_sync_source_for(db) == CharactersRAGDBChatSyncSource(db)
    assert chat_sync_source_for(None) is None
    assert chat_sync_source_for(object()) is None


_UNCOMPOSED_PROCESS = """
from Tests.network_guard import install
install()
from tldw_chatbook.Chat.console_semantic_revision import install_semantic_coordination
install_semantic_coordination()
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.transaction_observer import ChatRecordHashingNotInstalledError
db = CharactersRAGDB(':memory:', 'uncomposed')
conversation = db.add_conversation({'title': 'bare library use'})
message = db.add_message({'conversation_id': conversation, 'sender': 'user', 'content': 'hi'})
try:
    db.soft_delete_message(message, expected_version=1)
except ChatRecordHashingNotInstalledError:
    pass
else:
    raise AssertionError('an uncomposed process deleted without its Sync proof')
deleted = 'SELECT deleted FROM messages WHERE id = ?'
assert db.get_connection().execute(deleted, (message,)).fetchone()[0] == 0
import tldw_chatbook.Sync_Interop.chat_sync_source  # registers itself on import
assert db.soft_delete_message(message, expected_version=1)
assert db.get_connection().execute(deleted, (message,)).fetchone()[0] == 1
print('fail-closed then registered')
"""


def test_uncomposed_process_fails_closed_until_the_adapter_is_imported():
    result = subprocess.run(
        [sys.executable, "-c", _UNCOMPOSED_PROCESS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "fail-closed then registered" in result.stdout
