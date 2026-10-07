"""B25: bulk path for chat-history import (DB-state equivalence oracle).

Strategy (recorded per plan Step 1): the semantic-revision sidecar write is
NOT safely batchable outside its coordinator -- ``ensure_semantic_revision``
runs ``_claim_write_intent``, graph-epoch fencing and read-back contracts on
the private ``console_trace_semantic_revisions`` shape -- so the chosen
strategy is the plan's fallback: hoist per-message validation ahead of the
transaction, insert the messages with one ``executemany`` inside a single
(immediate) transaction, verify the inserted rows with one chunked read-back
SELECT per 500 ids, and keep the sanctioned per-message sidecar write. The
10,000-message import cap stays in the importer's staged-building validation.

Oracle: a 200-message fixture imported through the old per-message loop into
DB-A and through the new batch path into DB-B must produce row-for-row
identical ``conversations``, ``messages`` and semantic-revision rows. Ids and
timestamps are made deterministic per-DB (``_generate_uuid`` /
``_get_current_utc_timestamp_iso`` / ``new_opaque_id``) so both databases can
be compared exactly; revision ``created_at`` uses SQL-side ``CURRENT_TIMESTAMP``
and is excluded from the comparison.
"""

from __future__ import annotations

import itertools

import pytest

import tldw_chatbook.Chat.console_semantic_revision as console_semantic_revision
import tldw_chatbook.Chat.console_trace_repository as console_trace_repository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

FIXTURE_SIZE = 200
_FROZEN_NOW = "2026-01-01T00:00:00.000000+00:00"

_REVISION_COLUMNS = (
    "revision_id, source_conversation_id, source_message_id, revision_sequence, "
    "normalized_role, content_kind, creation_reason, predecessor_revision_id, "
    "live_message_id, live_locator_retired_at"
)


def _staged_messages(count: int = FIXTURE_SIZE) -> list[dict]:
    staged = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        staged.append(
            {
                "sender": role,
                "role": role,
                "content": f"imported body {index}",
            }
        )
    return staged


def _make_deterministic(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ids/timestamps per DB instance so two databases compare exactly.

    Each database instance gets its OWN counter starting at 1, so the id
    sequences are identical across databases.
    """
    sequence = itertools.count(1)

    def generate_uuid() -> str:
        return f"det-{next(sequence):05d}"

    monkeypatch.setattr(db, "_generate_uuid", generate_uuid)
    monkeypatch.setattr(
        db, "_get_current_utc_timestamp_iso", lambda: _FROZEN_NOW
    )


def _pin_revision_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm a fresh deterministic revision-id counter (per import)."""
    sequence = itertools.count(1)
    for module in (console_trace_repository, console_semantic_revision):
        monkeypatch.setattr(
            module,
            "new_opaque_id",
            lambda s=sequence: f"rev-{next(s):05d}",
        )


def _import_via_per_message_loop(
    db: CharactersRAGDB, staged_messages: list[dict]
) -> str:
    """The pre-change import replay (Character_Chat_Lib ~:3336)."""
    conversation_id = db.add_conversation(
        {
            "title": "Imported Chat",
            "assistant_authority_id": None,
            "thinking_history_policy": "auto",
        }
    )
    assert conversation_id
    parent_id = None
    for staged in staged_messages:
        staged["conversation_id"] = conversation_id
        staged["parent_message_id"] = parent_id
        new_message_id = db.add_message(staged)
        assert new_message_id
        parent_id = str(new_message_id)
    db.set_conversation_active_leaf(conversation_id, parent_id)
    return str(conversation_id)


def _import_via_bulk_path(
    db: CharactersRAGDB, staged_messages: list[dict]
) -> str:
    """The post-change import replay (batch method chains the parents)."""
    conversation_id = db.add_conversation(
        {
            "title": "Imported Chat",
            "assistant_authority_id": None,
            "thinking_history_policy": "auto",
        }
    )
    assert conversation_id
    for staged in staged_messages:
        staged["conversation_id"] = conversation_id
    message_ids = db.add_message_import_batch(staged_messages)
    assert message_ids and all(message_ids)
    db.set_conversation_active_leaf(conversation_id, message_ids[-1])
    return str(conversation_id)


def _dump_state(db: CharactersRAGDB, conversation_id: str) -> tuple:
    connection = db.get_connection()
    conversation = tuple(
        connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    )
    messages = tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY rowid",
            (conversation_id,),
        )
    )
    revisions = tuple(
        tuple(row)
        for row in connection.execute(
            f"SELECT {_REVISION_COLUMNS} FROM console_trace_semantic_revisions "
            "WHERE source_conversation_id = ? "
            "ORDER BY source_message_id, revision_sequence",
            (conversation_id,),
        )
    )
    epoch = connection.execute(
        "SELECT epoch FROM console_trace_graph_epoch WHERE singleton_id = 1"
    ).fetchone()[0]
    return conversation, messages, revisions, epoch


@pytest.fixture
def fresh_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    def make(name: str) -> CharactersRAGDB:
        db = CharactersRAGDB(tmp_path / f"{name}.db", client_id="import-bulk")
        _make_deterministic(db, monkeypatch)
        return db

    yield make
    # connections are closed by each made instance's caller scope; tests close




def test_bulk_import_matches_per_message_path_row_for_row(
    fresh_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_a = _staged_messages()
    staged_b = _staged_messages()

    db_a = fresh_db("db-a")
    _pin_revision_ids(monkeypatch)
    conversation_a = _import_via_per_message_loop(db_a, staged_a)
    state_a = _dump_state(db_a, conversation_a)
    db_a.close_connection()

    db_b = fresh_db("db-b")
    _pin_revision_ids(monkeypatch)
    conversation_b = _import_via_bulk_path(db_b, staged_b)
    state_b = _dump_state(db_b, conversation_b)
    db_b.close_connection()

    conversation, messages, revisions, epoch = state_a
    assert len(messages) == FIXTURE_SIZE
    assert len(revisions) == FIXTURE_SIZE
    assert conversation == state_b[0]
    assert messages == state_b[1], "messages table differs from the loop path"
    assert revisions == state_b[2], (
        "semantic-revision sidecars differ from the loop path"
    )
    assert epoch == state_b[3]


def test_loop_path_is_self_consistent_oracle_sanity(
    fresh_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The oracle harness itself: two loop imports must be identical."""
    db_a = fresh_db("oracle-a")
    _pin_revision_ids(monkeypatch)
    conversation_a = _import_via_per_message_loop(db_a, _staged_messages())
    state_a = _dump_state(db_a, conversation_a)
    db_a.close_connection()

    db_b = fresh_db("oracle-b")
    _pin_revision_ids(monkeypatch)
    conversation_b = _import_via_per_message_loop(db_b, _staged_messages())
    state_b = _dump_state(db_b, conversation_b)
    db_b.close_connection()

    assert state_a == state_b


def test_bulk_import_rejects_malformed_message_before_any_insert(
    fresh_db,
) -> None:
    db = fresh_db("db-invalid")
    conversation_id = db.add_conversation({"title": "Imported Chat"})
    staged = _staged_messages(3)
    staged[1].pop("content")
    for message in staged:
        message["conversation_id"] = conversation_id

    with pytest.raises(Exception) as excinfo:
        db.add_message_import_batch(staged)

    assert type(excinfo.value).__name__ == "InputError"
    # nothing was inserted: the hoisted validation runs before the transaction
    remaining = db.get_connection().execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0]
    assert remaining == 0
    db.close_connection()


def test_bulk_import_preserves_error_semantics_for_duplicate_ids(
    fresh_db,
) -> None:
    """A staged id colliding with a message in ANOTHER conversation.

    (Same-conversation duplicates violate the composite (conversation_id, id)
    constraint first, and the per-message ``add_message`` path raises
    ``CharactersRAGDBError`` for those; the batch preserves that too.)
    """
    db = fresh_db("db-duplicate")
    conversation_id = db.add_conversation({"title": "Imported Chat"})
    other_conversation_id = db.add_conversation({"title": "Other"})
    staged = _staged_messages(3)
    staged[2]["id"] = "pre-existing-id"
    db.add_message(
        {
            "id": "pre-existing-id",
            "conversation_id": other_conversation_id,
            "sender": "user",
            "content": "already here",
            "client_id": db.client_id,
        }
    )
    for message in staged:
        message["conversation_id"] = conversation_id

    with pytest.raises(Exception) as excinfo:
        db.add_message_import_batch(staged)

    assert type(excinfo.value).__name__ == "ConflictError"
    # and the per-message path raises the same shape for the same fixture
    with pytest.raises(Exception) as loop_excinfo:
        for message in staged:
            db.add_message(dict(message))
    assert type(loop_excinfo.value).__name__ == "ConflictError"
    db.close_connection()
