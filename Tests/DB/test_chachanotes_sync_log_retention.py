"""`sync_log` retention: deleted content leaves, reachable proof stays (task-19564).

Why this exists -- the incident, not the rule. `sync_log` stores the COMPLETE
row as JSON, written by 35 triggers, and before this nothing ever deleted a
row. Lane 3 of the 2026-08-21 holistic review soft-deleted a conversation and
read the message body straight back out of `sync_log`; soft-deleting the
MESSAGE left it there too. "Delete" left the user's plaintext in the database
indefinitely, and every edit left a full extra copy behind forever.

The filing recommended retiring the content columns because "both readers have
zero external callers". That is stale, and this module pins the reason:
`read_committed_chat_sync_intent`, `read_committed_chat_delete_intent` and
`list_current_committed_chat_sync_intents` all compare the sync_log payload to
the live `messages` row FIELD BY FIELD, and two of them have live non-test
callers (`ConsoleChatStore.ensure_provider_continuation_durable`, which raises
when the read returns None, and `._reconcile_restored_chat_sync_intents`). So
the log is bounded to the frontier its readers can reach, not emptied of
content -- and `test_retention_does_not_break_the_committed_intent_readers`
below is the guard on that trade.

The first cut covered the six entities the FILING named. `sync_log` is written
for NINE, and Qodo's review of PR #1974 independently found the same three
omissions this branch's own review had recorded: `chat_dictionaries`,
`world_books` and `world_book_entries`, none of which a version rule can
bound. They are covered now, under a second rule, and
`test_every_sync_log_writer_has_a_retention_scope` makes the covered set a
DERIVED fact -- read out of the schema's own writers -- so the next writer
cannot ship without retention the way these three did.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "retention.db", "retention_client")
    try:
        yield database
    finally:
        database.close_connection()


def _needle() -> str:
    return "zqx" + uuid.uuid4().hex[:12]


def _chat_envelope_payload_hash(content: str, role: str) -> str:
    """Hash of the chat envelope payload for a message with no private parts.

    The chat envelope's clear payload is exactly three keys, and
    `assistant_generation_state` is present even when it is None -- the one
    optional key (`provider_continuation_json`) is the one production omits.
    """
    return canonical_payload_hash(
        {
            "assistant_generation_state": None,
            "content": content,
            "role": role,
        }
    )


def _sync_log_hits(db: CharactersRAGDB, needle: str) -> list[dict]:
    """Every sync_log row whose payload still carries `needle`, whatever entity."""
    return [
        {
            "entity": row["entity"],
            "entity_id": row["entity_id"],
            "operation": row["operation"],
            "version": row["version"],
        }
        for row in db.execute_query(
            "SELECT entity, entity_id, operation, version FROM sync_log "
            "WHERE payload LIKE ?",
            (f"%{needle}%",),
        ).fetchall()
    ]


def _entries(db: CharactersRAGDB, entity: str, entity_id: str) -> list[tuple]:
    return [
        (row["operation"], row["version"])
        for row in db.execute_query(
            "SELECT operation, version FROM sync_log "
            "WHERE entity = ? AND entity_id = ? ORDER BY change_id",
            (entity, entity_id),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# the lane's probe, per entity
# ---------------------------------------------------------------------------
def test_soft_deleting_a_message_removes_its_body_from_sync_log(db: CharactersRAGDB):
    """The lane probe, sharpened onto the entity actually being deleted."""
    needle = _needle()
    conversation_id = db.add_conversation({"title": "probe", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"the {needle} body",
        }
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_message(message_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []
    # The tombstone survives -- it is the delete proof and carries no content.
    assert _entries(db, "messages", message_id) == [("delete", 2)]


def test_soft_deleting_a_note_removes_its_body_from_sync_log(db: CharactersRAGDB):
    needle = _needle()
    note_id = db.add_note(title="probe note", content=f"secret {needle}")
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_note(note_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_soft_deleting_a_character_removes_its_text_from_sync_log(
    db: CharactersRAGDB,
):
    needle = _needle()
    card_id = db.add_character_card(
        {"name": "probe card", "description": f"backstory {needle}"}
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_character_card(card_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_soft_deleting_a_conversation_removes_its_title_from_sync_log(
    db: CharactersRAGDB,
):
    """A conversation's own content in `sync_log` is its title.

    Its messages are NOT deleted by this operation -- they stay `deleted = 0`
    and come back on restore -- so their frontier row is retained, exactly as
    `messages.content` is. That residue is deliberate and is what
    `test_a_live_message_keeps_only_the_frontier_its_readers_need` bounds.
    """
    needle = _needle()
    conversation_id = db.add_conversation(
        {"title": f"title {needle}", "character_id": 1}
    )
    assert _sync_log_hits(db, needle) != []

    db.soft_delete_conversation(conversation_id, expected_version=1)

    assert _sync_log_hits(db, needle) == []


def test_hard_deleting_a_message_removes_its_body_from_sync_log(db: CharactersRAGDB):
    needle = _needle()
    conversation_id = db.add_conversation({"title": "probe", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"hard {needle}",
        }
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    assert _sync_log_hits(db, needle) == []
    assert _entries(db, "messages", message_id) == []


def test_hard_deleting_a_conversation_cascades_the_purge_to_its_messages(
    db: CharactersRAGDB,
):
    """The FK cascade fires the child trigger -- verified, not assumed.

    ``messages.conversation_id`` is ``ON DELETE CASCADE``, and SQLite fires the
    child table's AFTER DELETE trigger for a foreign-key action (``PRAGMA
    recursive_triggers`` is 0 here and governs recursion, not this). Without
    that, a hard conversation delete would leave every message body in
    ``sync_log`` with no row left to reach it from.
    """
    needle = _needle()
    conversation_id = db.add_conversation({"title": "cascade", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"cascade {needle}",
        }
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    assert db.execute_query(
        "SELECT COUNT(*) FROM messages WHERE id = ?", (message_id,)
    ).fetchone()[0] == 0
    assert _sync_log_hits(db, needle) == []


# ---------------------------------------------------------------------------
# the size story: edit history stops accumulating
# ---------------------------------------------------------------------------
def test_editing_a_message_does_not_accumulate_old_bodies(db: CharactersRAGDB):
    """Before this, every edit left its previous full text in `sync_log`."""
    conversation_id = db.add_conversation({"title": "edits", "character_id": 1})
    needles = [_needle() for _ in range(5)]
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"draft {needles[0]}",
        }
    )
    for version, needle in enumerate(needles[1:], start=1):
        db.update_message(
            message_id, {"content": f"draft {needle}"}, expected_version=version
        )

    surviving = [needle for needle in needles if _sync_log_hits(db, needle)]

    # Only the frontier the readers need: the current body and the one below
    # it, never the whole history.
    assert surviving == needles[-2:]
    assert _entries(db, "messages", message_id) == [("update", 4), ("update", 5)]


def test_a_live_message_keeps_only_the_frontier_its_readers_need(
    db: CharactersRAGDB,
):
    conversation_id = db.add_conversation({"title": "frontier", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    db.update_message(message_id, {"content": "v3"}, expected_version=2)

    assert _entries(db, "messages", message_id) == [("update", 2), ("update", 3)]


def test_editing_a_note_keeps_only_the_current_version(db: CharactersRAGDB):
    """Non-message entities have no reader at all, so only the frontier stays."""
    note_id = db.add_note(title="n", content="first")
    db.update_note(note_id, {"content": "second"}, expected_version=1)
    db.update_note(note_id, {"content": "third"}, expected_version=2)

    assert _entries(db, "notes", note_id) == [("update", 3)]


# ---------------------------------------------------------------------------
# the trade: retention must not break the readers that keep the content alive
# ---------------------------------------------------------------------------
def test_retention_does_not_break_the_committed_intent_readers(db: CharactersRAGDB):
    """The reason the content columns are retained rather than retired.

    `ensure_provider_continuation_durable` raises when
    `read_committed_chat_sync_intent` returns None, so a retention rule that
    pruned the frontier would turn every continuation checkpoint into a hard
    error while silently disabling Sync v2.

    The envelope payload is built through `_chat_envelope_payload` rather than
    inline: this test hand-rolled `{"content": ..., "role": ...}` and went red
    the moment v48 added `assistant_generation_state` to the envelope contract,
    reporting a retention defect that did not exist (task-21441). The helper
    states the contract once, in the same shape production states it -- see
    `Sync_Interop/envelope_builder.build_chat_message_upsert` and
    `envelope_applier`'s `allowed_keys` -- so the next envelope key breaks it
    in one place with a readable diff.
    """
    conversation_id = db.add_conversation({"title": "intents", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "first"}
    )
    db.update_message(message_id, {"content": "second"}, expected_version=1)
    db.update_message(message_id, {"content": "third"}, expected_version=2)
    row = db.execute_query(
        "SELECT role, version FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    assert row["version"] == 3

    record = db.read_committed_chat_sync_intent(
        message_id=message_id,
        message_version=3,
        payload_hash=_chat_envelope_payload_hash("third", row["role"]),
    )

    assert record is not None
    assert record.content == "third"
    # The base hash comes from the version below the frontier; that row is
    # exactly what the retention rule keeps for live messages.
    assert record.base_payload_hash == _chat_envelope_payload_hash(
        "second", row["role"]
    )

    # And a restore-time reconcile still sees the conversation's intents.
    intents = db.list_current_committed_chat_sync_intents(conversation_id)
    assert [intent["message_id"] for intent in intents] == [message_id]
    assert intents[0]["message_version"] == 3


def test_pruning_cannot_flip_the_single_intent_ambiguity_check(db: CharactersRAGDB):
    """Retention must not turn a REJECTED intent into an accepted one.

    `list_current_committed_chat_sync_intents` accepts a message only when
    `1 = (SELECT COUNT(*) FROM sync_log ...)` at the message's CURRENT version
    -- two rows there mean the history is ambiguous and the intent is refused.
    A prune that could delete one of a duplicate pair would silently promote
    that refusal into an acceptance, changing what a sync proof asserts. The
    rule that makes it safe is that pruning only ever removes rows STRICTLY
    BELOW the frontier; this is the test that says so.
    """
    conversation_id = db.add_conversation({"title": "ambiguous", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    version = db.execute_query(
        "SELECT version FROM messages WHERE id = ?", (message_id,)
    ).fetchone()["version"]

    # Forge a duplicate at the CURRENT version -- the ambiguous state.
    with db.transaction() as conn:
        duplicate = conn.execute(
            "SELECT entity_id, operation, timestamp, client_id, version, payload "
            "FROM sync_log WHERE entity = 'messages' AND entity_id = ? "
            "AND version = ?",
            (message_id, version),
        ).fetchone()
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES ('messages',?,?,?,?,?,?)",
            tuple(duplicate),
        )

    def count_at_current() -> int:
        return db.execute_query(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'messages' "
            "AND entity_id = ? AND version = ?",
            (message_id, version),
        ).fetchone()[0]

    assert count_at_current() == 2
    assert db.list_current_committed_chat_sync_intents(conversation_id) == []

    db.prune_sync_log()

    assert count_at_current() == 2, "the frontier's row count must be untouched"
    assert db.list_current_committed_chat_sync_intents(conversation_id) == []


def test_a_tombstoned_message_still_proves_its_delete_intent(db: CharactersRAGDB):
    conversation_id = db.add_conversation({"title": "tombstone", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "bye"}
    )
    db.soft_delete_message(message_id, expected_version=1)

    record = db.read_committed_chat_delete_intent(
        message_id=message_id,
        message_version=2,
        payload_hash=canonical_payload_hash({"deleted": True}),
    )

    assert record is not None
    assert record.message_id == message_id


# ---------------------------------------------------------------------------
# existing databases, and the maintenance surface
# ---------------------------------------------------------------------------
def test_prune_sync_log_clears_a_backlog_written_before_the_triggers(
    db: CharactersRAGDB,
):
    """The shape an existing user's database is in when it reaches v45.

    Rows are written straight into `sync_log` here, bypassing the triggers, to
    reproduce a pre-v45 backlog; `prune_sync_log` is the same sweep the
    migration performs once on upgrade.
    """
    needle = _needle()
    conversation_id = db.add_conversation({"title": "backlog", "character_id": 1})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "v1"}
    )
    db.update_message(message_id, {"content": "v2"}, expected_version=1)
    db.update_message(message_id, {"content": "v3"}, expected_version=2)
    db.update_message(message_id, {"content": "v4"}, expected_version=3)
    assert _entries(db, "messages", message_id) == [("update", 3), ("update", 4)]

    with db.transaction() as conn:
        # Versions 1 and 2 are what a pre-v45 database would still be holding.
        for version in (1, 2):
            conn.execute(
                "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
                "client_id, version, payload) VALUES ('messages', ?, 'update', "
                "'2020-01-01T00:00:00Z', 'legacy', ?, ?)",
                (
                    message_id,
                    version,
                    json.dumps({"id": message_id, "content": f"old {needle}"}),
                ),
            )
        # An orphan: the entity row is long gone.
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES ('notes', 'vanished', 'create', "
            "'2020-01-01T00:00:00Z', 'legacy', 1, ?)",
            (json.dumps({"id": "vanished", "content": f"orphan {needle}"}),),
        )
    assert _sync_log_hits(db, needle) != []

    removed = db.prune_sync_log()

    assert removed == 3
    assert _sync_log_hits(db, needle) == []
    # The genuine frontier rows are untouched.
    assert _entries(db, "messages", message_id) == [("update", 3), ("update", 4)]


def test_prune_sync_log_is_idempotent(db: CharactersRAGDB):
    conversation_id = db.add_conversation({"title": "idem", "character_id": 1})
    db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "hello"}
    )

    assert db.prune_sync_log() == 0
    assert db.prune_sync_log() == 0


def test_delete_sync_log_entries_before_matches_the_media_database_api(
    db: CharactersRAGDB,
):
    """Parity with `Client_Media_DB_v2`, which ChaChaNotes never had."""
    conversation_id = db.add_conversation({"title": "api", "character_id": 1})
    db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "one"}
    )
    watermark = db.get_latest_sync_log_change_id()
    assert watermark > 0
    live_rows = len(db.get_sync_log_entries())
    assert live_rows > 0

    assert db.delete_sync_log_entries_before(watermark) == live_rows
    assert db.get_sync_log_entries() == []

    with pytest.raises(ValueError):
        db.delete_sync_log_entries_before(-1)


def test_delete_sync_log_entries_rejects_non_integers(db: CharactersRAGDB):
    with pytest.raises(ValueError):
        db.delete_sync_log_entries([1, "two"])
    assert db.delete_sync_log_entries([]) == 0


# ---------------------------------------------------------------------------
# the census: the covered set is derived from the schema's own writers
# ---------------------------------------------------------------------------
# The first cut of v45 covered six entities because the FILING listed six.
# `sync_log` is written for NINE, and Qodo's review of PR #1974 independently
# found the same three omissions. These two tests make the covered set a
# derived fact rather than a remembered one: a tenth writer added later reds
# them until it also has a retention rule. There is deliberately NO allowlist
# -- every writer is covered, so an exemption row would have nothing to hold.
def _sync_log_writer_entities(db: CharactersRAGDB) -> set[str]:
    """Entities the live schema actually emits `sync_log` rows for."""
    entities: set[str] = set()
    for row in db.execute_query(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND sql LIKE ?",
        ("%INSERT INTO sync_log%",),
    ).fetchall():
        match = re.search(r"INTO sync_log\([^)]*\)\s*VALUES\(\s*'([a-z_]+)'", row[0])
        assert match is not None, f"unparsed sync_log writer:\n{row[0]}"
        entities.add(match.group(1))
    return entities


def test_every_sync_log_writer_has_a_retention_scope(db: CharactersRAGDB):
    """`prune_sync_log`'s covered set == the table's writers, both directions."""
    covered = {scope[0] for scope in CharactersRAGDB._SYNC_LOG_RETENTION_SCOPES} | {
        scope[0] for scope in CharactersRAGDB._SYNC_LOG_LATEST_ONLY_SCOPES
    }
    writers = _sync_log_writer_entities(db)

    assert writers, "no sync_log writers found -- the census would be vacuous"
    assert covered == writers, (
        f"sync_log writers with no retention rule: {sorted(writers - covered)}; "
        f"retention scopes for entities nothing writes: {sorted(covered - writers)}"
    )


def test_every_sync_log_writer_has_a_prune_trigger(db: CharactersRAGDB):
    """The schema-side half: the bound must hold without anyone calling anything."""
    triggers = {
        row[0]
        for row in db.execute_query(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }
    missing = {
        entity
        for entity in _sync_log_writer_entities(db)
        if f"sync_log_prune_{entity}" not in triggers
        or f"sync_log_prune_{entity}_hard" not in triggers
    }
    assert missing == set()


# ---------------------------------------------------------------------------
# the latest-only three (Qodo's review of PR #1974)
# ---------------------------------------------------------------------------
def _dictionary_lib():
    from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib

    return Chat_Dictionary_Lib


def _books(db: CharactersRAGDB):
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager

    return WorldBookManager(db)


def test_soft_deleting_a_chat_dictionary_removes_its_text_from_sync_log(
    db: CharactersRAGDB,
):
    """Includes the hazard a version rule cannot see.

    `chat_dictionaries_update_timestamp` runs a nested UPDATE, which fires the
    `sync_update` emitter again. When the timestamp it writes differs from the
    one the outer statement wrote, that lands a FULL-PAYLOAD row at the
    tombstone's OWN version -- so `version < NEW.version`, the rule the other
    six use, would leave the deleted dictionary's plaintext behind. The delete
    here is issued in exactly that shape.
    """
    lib = _dictionary_lib()
    needle = _needle()
    dictionary_id = lib.save_chat_dictionary(
        db, name=f"dict-{needle}", description=f"secret {needle}", content="body"
    )
    assert _sync_log_hits(db, needle) != []

    with db.transaction() as conn:
        conn.execute(
            "UPDATE chat_dictionaries SET deleted = 1, version = version + 1, "
            "last_modified = '2001-01-01 00:00:00' WHERE id = ?",
            (dictionary_id,),
        )

    assert _sync_log_hits(db, needle) == []
    assert _entries(db, "chat_dictionaries", str(dictionary_id)) == [("delete", 2)]


def test_editing_a_chat_dictionary_keeps_only_the_current_version(
    db: CharactersRAGDB,
):
    lib = _dictionary_lib()
    stale, current = _needle(), _needle()
    dictionary_id = lib.save_chat_dictionary(
        db, name="edited-dict", description=f"first {stale}", content="body"
    )
    lib.update_chat_dictionary(db, dictionary_id, description=f"second {current}")

    assert _sync_log_hits(db, stale) == []
    assert len(_sync_log_hits(db, current)) == 1


def test_soft_deleting_a_world_book_removes_its_text_from_sync_log(
    db: CharactersRAGDB,
):
    needle = _needle()
    books = _books(db)
    book_id = books.create_world_book(
        name=f"book-{needle}", description=f"secret {needle}"
    )
    assert _sync_log_hits(db, needle) != []

    books.delete_world_book(book_id)

    assert _sync_log_hits(db, needle) == []
    assert _entries(db, "world_books", str(book_id)) == [("delete", 2)]


def test_hard_deleting_a_world_book_entry_removes_its_lore_from_sync_log(
    db: CharactersRAGDB,
):
    """The sharpest of the three: prose, hard delete, orphaned forever.

    `world_book_entries` has no `version` and no `deleted` column -- every one
    of its sync rows is written at the literal version 1 -- and Personas >
    entry delete calls `delete_world_book_entry`, a hard `DELETE`. Before this
    rule the entry's full `keys` + `content` survived that delete permanently,
    with no entity row left to reach them from.
    """
    needle = _needle()
    books = _books(db)
    book_id = books.create_world_book(name="lorebook")
    entry_id = books.create_world_book_entry(
        book_id, keys=[needle], content=f"the {needle} lore"
    )
    assert _sync_log_hits(db, needle) != []

    books.delete_world_book_entry(entry_id)

    assert _sync_log_hits(db, needle) == []
    # The tombstone survives -- ids only, and it is the entity's ONLY record
    # that the delete happened (there is no soft-deleted row left behind).
    assert _entries(db, "world_book_entries", str(entry_id)) == [("delete", 1)]


def test_editing_a_world_book_entry_does_not_accumulate_old_bodies(
    db: CharactersRAGDB,
):
    """4 edits used to retain 4/4 old bodies -- the version rule is inert here."""
    books = _books(db)
    book_id = books.create_world_book(name="edited-lorebook")
    stale = [_needle() for _ in range(4)]
    entry_id = books.create_world_book_entry(
        book_id, keys=["k"], content=f"body {stale[0]}"
    )
    for needle in stale[1:]:
        books.update_world_book_entry(entry_id, content=f"body {needle}")
    current = _needle()
    books.update_world_book_entry(entry_id, content=f"body {current}")

    for needle in stale:
        assert _sync_log_hits(db, needle) == [], needle
    assert len(_sync_log_hits(db, current)) == 1
    assert _entries(db, "world_book_entries", str(entry_id)) == [("update", 1)]


def test_hard_deleting_a_chat_dictionary_orphans_nothing(db: CharactersRAGDB):
    """A hard DELETE emits no sync row, so nothing would fire the log-side rule."""
    lib = _dictionary_lib()
    needle = _needle()
    dictionary_id = lib.save_chat_dictionary(
        db, name="doomed", description=f"secret {needle}", content="body"
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM chat_dictionaries WHERE id = ?", (dictionary_id,))

    assert _sync_log_hits(db, needle) == []
    assert _entries(db, "chat_dictionaries", str(dictionary_id)) == []


def test_latest_only_retention_is_independent_of_trigger_firing_order(
    tmp_path: Path,
):
    """The claim the whole rule rests on, exercised rather than asserted.

    SQLite does not define the firing order of same-kind triggers, and a
    chat-dictionary soft delete really does fire two emitters. Recreating them
    in a different order genuinely flips which one emits first -- the control
    half below proves that, so the experiment is not vacuous -- and the
    retained content must be the same either way.
    """
    lib = _dictionary_lib()
    emitters = (
        "chat_dictionaries_sync_delete",
        "chat_dictionaries_update_timestamp",
        "chat_dictionaries_sync_update",
    )

    def run(order, with_retention):
        database = CharactersRAGDB(
            tmp_path / f"order-{hash((order, with_retention)) & 0xFFFF}.db",
            "order-client",
        )
        try:
            connection = database.get_connection()
            if not with_retention:
                connection.execute("DROP TRIGGER sync_log_prune_chat_dictionaries")
            definitions = {
                name: connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
                ).fetchone()[0]
                for name in order
            }
            for name in order:
                connection.execute(f"DROP TRIGGER {name}")
            for name in order:
                connection.execute(definitions[name])
            connection.commit()

            dictionary_id = lib.save_chat_dictionary(
                db=database, name="ordered", description="secret-body", content="body"
            )
            with database.transaction() as conn:
                conn.execute(
                    "UPDATE chat_dictionaries SET deleted = 1, "
                    "version = version + 1, "
                    "last_modified = '2001-01-01 00:00:00' WHERE id = ?",
                    (dictionary_id,),
                )
            return [
                (row["operation"], row["version"], "secret-body" in (row["payload"] or ""))
                for row in database.execute_query(
                    "SELECT operation, version, payload FROM sync_log "
                    "WHERE entity = 'chat_dictionaries' ORDER BY change_id"
                ).fetchall()
            ]
        finally:
            database.close_connection()

    reversed_order = tuple(reversed(emitters))
    # Control: without retention the two orders differ, so the permutation is
    # really reaching the firing order and the experiment below has teeth.
    assert run(emitters, False) != run(reversed_order, False)
    # With retention both converge on the tombstone alone, carrying no body.
    assert run(emitters, True) == run(reversed_order, True) == [("delete", 2, False)]


def test_prune_sync_log_clears_a_latest_only_backlog_written_before_the_triggers(
    db: CharactersRAGDB,
):
    """The maintenance sweep must agree with the triggers, entity for entity."""
    lib = _dictionary_lib()
    books = _books(db)
    stale, current, orphaned = _needle(), _needle(), _needle()
    dictionary_id = lib.save_chat_dictionary(
        db, name="swept", description="body", content="body"
    )
    book_id = books.create_world_book(name="swept-book")
    entry_id = books.create_world_book_entry(book_id, keys=["k"], content="lore")

    with db.transaction() as conn:
        for trigger in (
            "sync_log_prune_chat_dictionaries",
            "sync_log_prune_world_books",
            "sync_log_prune_world_book_entries",
            "sync_log_prune_world_book_entries_hard",
        ):
            conn.execute(f"DROP TRIGGER {trigger}")
    # Backlog in the pre-v45 shape: superseded content, plus content that
    # outlives its entity.
    lib.update_chat_dictionary(db, dictionary_id, description=f"stale {stale}")
    lib.update_chat_dictionary(db, dictionary_id, description=f"current {current}")
    books.update_world_book(book_id, description=f"current {current}")
    books.update_world_book_entry(entry_id, content=f"stale {stale}")
    books.update_world_book_entry(entry_id, content=f"orphaned {orphaned}")
    books.delete_world_book_entry(entry_id)
    assert len(_sync_log_hits(db, stale)) == 2
    assert len(_sync_log_hits(db, orphaned)) == 1

    # Six rows: the three superseded `create` rows the setup wrote before the
    # triggers were dropped, the two superseded edits, and the orphaned entry
    # body. Both entities that are still live keep exactly their frontier.
    assert db.prune_sync_log() == 6

    assert _sync_log_hits(db, stale) == []
    assert _sync_log_hits(db, orphaned) == []
    assert sorted(hit["entity"] for hit in _sync_log_hits(db, current)) == [
        "chat_dictionaries",
        "world_books",
    ]
    assert _entries(db, "world_book_entries", str(entry_id)) == [("delete", 1)]
    assert db.prune_sync_log() == 0


def test_prune_sync_log_refuses_an_unvalidated_retention_identifier(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
):
    """The identifiers go through `sql_validation`, not straight into an f-string.

    Qodo flagged the f-string interpolation of `{table}` / `{id_expr}`. Only
    the table and its id column are identifiers now, and both are checked --
    this is the proof the check is wired rather than decorative.
    """
    monkeypatch.setattr(
        CharactersRAGDB,
        "_SYNC_LOG_RETENTION_SCOPES",
        (("messages", "messages) --", "id", False, True),),
    )
    with pytest.raises(CharactersRAGDBError, match="Invalid sync_log retention table"):
        db.prune_sync_log()

    monkeypatch.setattr(
        CharactersRAGDB,
        "_SYNC_LOG_RETENTION_SCOPES",
        (("messages", "messages", "id) --", False, True),),
    )
    with pytest.raises(CharactersRAGDBError, match="Invalid sync_log retention id"):
        db.prune_sync_log()
