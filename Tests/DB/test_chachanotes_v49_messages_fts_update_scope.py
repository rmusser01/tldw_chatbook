"""ChaChaNotes v48 -> v49: scope `messages_au` to the FTS-relevant columns (task-21128).

THE DEFECT
----------
`messages_au` shipped as a bare `AFTER UPDATE ON messages`, so it fired on
every update of the row. `messages_fts` is an external-content fts5 table over
ONE column (`content`), but a chat turn now issues three to four auxiliary
UPDATEs against the assistant row that name no indexed column at all --
`update_message_usage_local` (usage_json), `update_message_metadata_local`
(metadata_json), attachment/variant bookkeeping, a `ranking`-only edit. Each
made the trigger delete the reply's doclist and re-tokenize the whole reply
back into the index.

Measured over one simulated streamed turn with a 400-token reply, on this
schema: FOUR index rewrites, `messages_fts_data` 55 -> 12,636 bytes. After the
step: ONE rewrite, 3,201 bytes. `test_a_streamed_turn_rewrites_the_index_once`
re-runs both arms in-process, varying nothing but the trigger definition, so
the number is re-derived on every run rather than quoted from a scratchpad.

WHY THE COLUMN LIST IS `content, deleted` AND NOT `content`
-----------------------------------------------------------
The filing proposed `AFTER UPDATE OF content`. That shape is a data-exposure
bug, and it is the reason this module's matrix exists rather than a one-line
diff: soft delete is `UPDATE messages SET deleted = 1 ...` and never names
`content`, so the trigger would not fire and the soft-deleted message would
STAY SEARCHABLE -- measured on a scratch matrix before any code was written,
where a direct `messages_fts MATCH` returned the tombstoned rowid. That is the
task-19567 privacy guarantee. `deleted` decides index MEMBERSHIP, so it
belongs in the dependency set alongside every column the index stores;
`test_the_update_of_list_covers_every_fts_relevant_column` derives that set
from the live schema, so widening `messages_fts` without widening the trigger
fails here.

Assertion style follows `test_chachanotes_v47_messages_fts_backfill.py` and
`test_fts_soft_delete_index_witness.py`: the FTS index is queried DIRECTLY
(never through a consumer that re-filters on `deleted`), expectations are
absolute rather than before/after equality snapshots, and the "must not write"
half is witnessed against the index's PHYSICAL storage, because an fts5
'delete' of an unindexed row can corrupt silently with a green
integrity-check (task-21100's review measured that form).

This module carried the repo's EXACT current-schema-version pin while v49 was
the newest step; task-22225 added v49 -> v50, so the pin moved on to
`Tests/DB/test_chachanotes_v50_console_policy_tombstone_cleanup.py`. The pin
belongs to the NEWEST migration's own file, so a schema bump touches the file
that caused it rather than an unrelated older one. What stays here is the
entry-version pin (a v48 database is what THIS step upgrades) and end-state
assertions that read `_CURRENT_SCHEMA_VERSION`: a version literal is only
correct at a fixture's SEEDED starting point, never after an upgrade.

Renumbered from v47->v48 to v48->v49: the Console Library policy step took 48
by merging first, and schema versions must be contiguous. Its content was
re-read rather than predicted before renumbering -- it adds
`messages.assistant_generation_state`, two tables, an index, and a rewrite of
the four `messages_sync_*` triggers, and it leaves `messages_au`/`_ai`/`_ad`
untouched. That is what keeps `V47_MESSAGES_AU` below the correct "before"
baseline and `test_upgrading_a_v48_database_...`'s pre-fix assertion true, so
both are pinned rather than assumed: if a future step DOES touch `messages_au`
before this one, that assertion fails and says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.chachanotes_fts_backfill import (
    backfill_chachanotes_messages_fts,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME

#: The trigger this step replaces, verbatim, for the "before" arm of the
#: write-count probe. Created by v47 and carried unchanged through v48 (the
#: Console Library policy step touches the `messages_sync_*` triggers, not the
#: FTS ones), so it is what any pre-v49 database actually holds.
#: Kept as a literal rather than read out of the v47 `.sql` file on purpose:
#: the probe must keep measuring the shape this task replaced even if that
#: file is later reworded, and a literal is what a reviewer can diff against
#: the migration by eye.
V47_MESSAGES_AU = """
CREATE TRIGGER messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  SELECT 'delete',old.rowid,old.content
  WHERE old.deleted = 0
    AND EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid);

  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;
"""

#: Crossing schema 47 now requires the caller to hand in the legacy
#: automatic-retrieval value, a precondition the v48 Console Library step
#: added (`CharactersRAGDB._migrate_from_v47_to_v48`; a from-scratch database
#: is exempt, an upgraded one is not). Only the v45-seeded fixture below
#: crosses 47, so only it needs this; the value is irrelevant to the FTS
#: trigger under test.
MIGRATION_SEED = ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False)


#: One assistant reply, big enough that re-tokenizing it is visible in the
#: index's physical storage.
REPLY_BODY = " ".join(f"replyneedle{i:04d}" for i in range(400))


def _version(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()[0]


def _trigger_sql(db: CharactersRAGDB, name: str) -> str:
    row = db.execute_query(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (name,),
    ).fetchone()
    assert row is not None, f"trigger {name} is missing"
    return " ".join(row[0].lower().split())


def _fts_rowids(db: CharactersRAGDB, needle: str) -> list[int]:
    """Rowids the FTS index itself returns -- no join, no ``deleted`` filter."""
    return [
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid",
            (needle,),
        ).fetchall()
    ]


def _docsize_rowids(db: CharactersRAGDB) -> set[int]:
    return {
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages_fts_docsize"
        ).fetchall()
    }


def _live_rowids(db: CharactersRAGDB) -> set[int]:
    return {
        row[0]
        for row in db.execute_query(
            "SELECT rowid FROM messages WHERE deleted = 0"
        ).fetchall()
    }


def _fts_storage_image(db: CharactersRAGDB) -> str:
    """A digest of the index's physical storage (`messages_fts_data`).

    Strictly stronger than the (row count, total bytes) witness task-21100
    used: ANY write into the inverted index moves this, including one that
    happens to preserve the byte total. "The trigger did not fire" is exactly
    "this digest did not move".
    """
    digest = hashlib.sha256()
    for rowid, block in db.execute_query(
        "SELECT id, block FROM messages_fts_data ORDER BY id"
    ).fetchall():
        digest.update(str(rowid).encode())
        digest.update(b"|")
        digest.update(block if block is not None else b"")
        digest.update(b"|")
    return digest.hexdigest()


def _fts_data_bytes(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT COALESCE(SUM(LENGTH(block)), 0) FROM messages_fts_data"
    ).fetchone()[0]


def _assert_index_structurally_sound(db: CharactersRAGDB) -> None:
    """FTS5 structural integrity check, NOT the external-content comparison.

    The flag form checks the inverted index's own consistency without
    comparing it to `messages` -- the comparison form legitimately fails
    during the backfill window and at steady state (tombstoned rows are
    deliberately absent). Needs SQLite >= 3.42; on older runtimes the MATCH
    assertions around each call still carry the behavioural check.
    """
    if sqlite3.sqlite_version_info < (3, 42, 0):
        return
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES ('integrity-check', 0)"
        )


def _install_trigger(db: CharactersRAGDB, create_sql: str) -> None:
    """Swap `messages_au` for another shape, inside one transaction.

    `executescript` would COMMIT the surrounding transaction (task-19553), so
    the two statements are issued individually.
    """
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER IF EXISTS messages_au")
        conn.execute(create_sql)


def _rowid_of(db: CharactersRAGDB, message_id: str) -> int:
    return db.execute_query(
        "SELECT rowid FROM messages WHERE id = ?", (message_id,)
    ).fetchone()[0]


def _seed_v45(db_path: Path, bodies: list[str]) -> tuple[list[str], list[int]]:
    """Build a genuine v45 DB, so opening it lands inside v47's backfill window.

    The v46 step issues `'delete-all'` and defers the reinsert to a background
    backfill (task-21100), so every seeded row is LIVE but absent from the
    index right after the upgrade -- the state this module's second matrix
    needs.

    The rows are inserted with explicit SQL rather than through `add_message`
    on purpose. `historical_bootstrap` builds an OLD schema and then hands it
    to the CURRENT code, and that only works while the current writers stay
    compatible with the old shape -- which stopped being true when the v48
    Console Library step added `messages.assistant_generation_state` to
    `add_message`'s unconditional INSERT column list. Against any pre-v48
    fixture that now raises `table messages has no column named
    assistant_generation_state`; it reds NINE tests in
    `test_chachanotes_v47_messages_fts_backfill.py` on pristine dev
    (reproduced at `origin/dev` d20dd733b, unrelated to task-21128 and filed
    separately). Seeding through SQL keeps this module's fixture independent
    of which columns today's writers happen to name, and the trigger under
    test still fires exactly as it would for a real insert.
    """
    with chachanotes_db_at_version(db_path, 45, client_id="v49-seed") as historical:
        conversation_id = historical.add_conversation(
            {"title": "window", "character_id": 1}
        )
        message_ids = [str(uuid.uuid4()) for _ in bodies]
        with historical.transaction() as conn:
            for message_id, body in zip(message_ids, bodies, strict=True):
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, sender, content, "
                    "client_id) VALUES (?, ?, ?, ?, ?)",
                    (message_id, conversation_id, "user", body, "v49-seed"),
                )
        rowids = [_rowid_of(historical, message_id) for message_id in message_ids]
    return message_ids, rowids


@pytest.fixture
def db(tmp_path: Path):
    instance = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="v49-test")
    yield instance
    instance.close_connection()


@pytest.fixture
def conversation(db: CharactersRAGDB) -> str:
    return db.add_conversation({"title": "v49", "character_id": 1})


# ---------------------------------------------------------------------------
# the step itself
# ---------------------------------------------------------------------------
def test_a_fresh_database_is_at_least_v49(db):
    """This step's floor. The exact pin lives in the newest step's file."""
    assert _version(db.get_connection()) >= 49
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 49


def test_fresh_schema_scopes_messages_au_to_content_and_deleted(db):
    au = _trigger_sql(db, "messages_au")
    assert "after update of content, deleted on messages" in au, au
    # v47's guards survive the rescoping, both halves.
    assert "messages_fts_docsize" in au, "the backfill-window membership guard is gone"
    assert "old.deleted = 0" in au, "the task-19567 corruption guard is gone"
    assert "new.deleted = 0" in au, "the task-19567 leak guard is gone"


def test_the_insert_and_delete_triggers_are_untouched(db):
    """`messages_ai`/`messages_ad` have no column list to narrow.

    Scoping the wrong trigger would be silent: an INSERT or DELETE has no
    `UPDATE OF` clause, so an accidental edit there could only remove a guard.
    """
    assert _trigger_sql(db, "messages_ai") == (
        "create trigger messages_ai after insert on messages begin "
        "insert into messages_fts(rowid,content) "
        "select new.rowid,new.content where new.deleted = 0; end"
    )
    assert _trigger_sql(db, "messages_ad") == (
        "create trigger messages_ad after delete on messages begin "
        "insert into messages_fts(messages_fts,rowid,content) "
        "select 'delete',old.rowid,old.content "
        "where exists (select 1 from messages_fts_docsize where rowid = old.rowid); end"
    )


def test_migrate_from_v48_to_v49_requires_version_48(db):
    """A fresh DB is already past 49, so re-entering the step must refuse."""
    with pytest.raises(SchemaError, match="requires schema version"):
        db._migrate_from_v48_to_v49(db.get_connection())


def test_upgrading_a_v48_database_replaces_the_trigger_and_keeps_its_index(
    tmp_path: Path,
):
    """Convergence, and the "did the base schema pin this?" trap.

    `messages_au` already exists in the v4 base schema, so a test that only
    asserted "a trigger named messages_au exists with a docsize guard" would
    pass against a v48 database that this step never touched. The old text is
    captured first and asserted to have CHANGED. The step is DDL-only, so the
    complete index a v48 database carries must be left exactly as it was.
    """
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 48, client_id="v48-complete") as at48:
        conversation_id = at48.add_conversation(
            {"title": "complete", "character_id": 1}
        )
        for i in range(5):
            at48.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"upgradeneedle{i:03d} body",
                }
            )
        before_sql = _trigger_sql(at48, "messages_au")
        before_index = _docsize_rowids(at48)
        before_storage = _fts_storage_image(at48)
        assert "after update on messages" in before_sql, (
            "the v48 fixture is not the pre-fix shape; the bootstrap changed"
        )
        assert len(before_index) == 5

    migrated = CharactersRAGDB(db_path, client_id="v49-upgrade")
    try:
        assert _version(migrated.get_connection()) == (
            CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        after_sql = _trigger_sql(migrated, "messages_au")
        assert after_sql != before_sql, "the migration did not replace the trigger"
        assert "after update of content, deleted on messages" in after_sql
        # DDL only: not one byte of index content moved.
        assert _docsize_rowids(migrated) == before_index
        assert _fts_storage_image(migrated) == before_storage
        assert len(_fts_rowids(migrated, "upgradeneedle002")) == 1
        _assert_index_structurally_sound(migrated)
    finally:
        migrated.close_connection()


def test_a_failure_mid_v49_rewinds_the_whole_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Atomic and re-enterable, the task-19553 rule (v47's test, two steps on)."""
    db_path = tmp_path / "poisoned.db"
    with chachanotes_db_at_version(db_path, 48, client_id="poison-seed") as at48:
        conversation_id = at48.add_conversation({"title": "p", "character_id": 1})
        at48.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "poison49needle body",
            }
        )
        rowids = sorted(_docsize_rowids(at48))
        original_sql = _trigger_sql(at48, "messages_au")

    original = CharactersRAGDB._execute_migration_statements

    def poisoned(self, cursor, script, label):
        if label == "V48→V49":
            script = script + "\nINSERT INTO no_such_table_21128(x) VALUES (1);\n"
        return original(self, cursor, script, label)

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", poisoned)
    with pytest.raises(SchemaError, match="no_such_table_21128"):
        CharactersRAGDB(db_path, client_id="poisoned")

    connection = sqlite3.connect(str(db_path))
    try:
        assert (
            connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (SCHEMA_NAME,),
            ).fetchone()[0]
            == 48
        ), "a failing chain must not bump the stamp"
        surviving = " ".join(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'messages_au'"
            ).fetchone()[0].lower().split()
        )
        assert surviving == original_sql, (
            "the DROP must rewind with the failing chain, or the database is "
            "left stamped 48 with no trigger at all"
        )
    finally:
        connection.close()

    monkeypatch.undo()
    migrated = CharactersRAGDB(db_path, client_id="poison-removed")
    try:
        assert _version(migrated.get_connection()) == (
            CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert "after update of content, deleted" in _trigger_sql(
            migrated, "messages_au"
        )
        assert sorted(_docsize_rowids(migrated)) == rowids
        assert _fts_rowids(migrated, "poison49needle") == rowids
    finally:
        migrated.close_connection()


# ---------------------------------------------------------------------------
# the no-stale-index census
# ---------------------------------------------------------------------------
_UPDATE_OF_RE = re.compile(
    r"after\s+update\s+of\s+(?P<columns>[^)]+?)\s+on\s+messages\b", re.IGNORECASE
)


def test_the_update_of_list_covers_every_fts_relevant_column(db):
    """A narrowed trigger is only safe while the narrowing stays complete.

    The required set is derived from the LIVE schema, never from the migration
    text: every column `messages_fts` indexes (because a change to one must
    reindex) plus `deleted` (because it decides membership). Add a column to
    the fts5 table without widening the trigger and this fails.

    Coverage, stated exactly: the assertion is `required <= listed`, so this
    catches only the TOO-NARROW direction -- a stale list that would leave the
    index silently out of date. It is deliberately blind to an over-broad list,
    which is a performance regression rather than a correctness one, and which
    five other assertions in this module do cover. Measured by mutating the
    shipped list to `content, deleted, usage_json`: 5 red, most directly the
    exact substring pin in
    `test_fresh_schema_scopes_messages_au_to_content_and_deleted` (a third
    column breaks `"after update of content, deleted on messages"`) and the
    `after_rewrites == 1` arm of `test_a_streamed_turn_rewrites_the_index_once`
    (a column an auxiliary flush writes puts the rewrite count back above one).
    """
    indexed_columns = {
        row[1] for row in db.execute_query("PRAGMA table_info(messages_fts)").fetchall()
    }
    assert indexed_columns == {"content"}, (
        "messages_fts no longer indexes exactly `content`; the trigger's "
        f"UPDATE OF list must be widened to cover {sorted(indexed_columns)}"
    )
    required = indexed_columns | {"deleted"}

    match = _UPDATE_OF_RE.search(_trigger_sql(db, "messages_au"))
    assert match is not None, (
        "messages_au has no UPDATE OF column list -- it fires on every write "
        "to the row again (task-21128)"
    )
    listed = {column.strip() for column in match.group("columns").split(",")}
    assert required <= listed, f"UPDATE OF {sorted(listed)} misses {sorted(required - listed)}"


# ---------------------------------------------------------------------------
# matrix A: an INDEXED row (steady state)
# ---------------------------------------------------------------------------
def test_content_edit_reindexes(db, conversation):
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "aaaneedle old"}
    )
    rowid = _rowid_of(db, message_id)

    db.update_message(message_id, {"content": "bbbneedle new"}, expected_version=1)

    assert _fts_rowids(db, "bbbneedle") == [rowid]
    assert _fts_rowids(db, "aaaneedle") == []
    _assert_index_structurally_sound(db)


def test_usage_only_flush_does_not_touch_the_index(db, conversation):
    """The headline: no write, and the row stays findable.

    "No write" is asserted against the index's physical storage, not against
    MATCH results -- a delete-then-reinsert of the same content is invisible
    to MATCH, which is exactly how this cost hid for so long.
    """
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "cccneedle body"}
    )
    rowid = _rowid_of(db, message_id)
    before = _fts_storage_image(db)

    assert db.update_message_usage_local(message_id, json.dumps({"in": 1, "out": 2}))

    assert _fts_storage_image(db) == before, (
        "a usage-only flush rewrote the reply into messages_fts (task-21128)"
    )
    assert _fts_rowids(db, "cccneedle") == [rowid], "the existing index row was lost"
    _assert_index_structurally_sound(db)


def test_metadata_only_flush_does_not_touch_the_index(db, conversation):
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "dddneedle body"}
    )
    rowid = _rowid_of(db, message_id)
    before = _fts_storage_image(db)

    assert db.update_message_metadata_local(message_id, json.dumps({"model": "m"}))

    assert _fts_storage_image(db) == before, (
        "a metadata-only flush rewrote the reply into messages_fts (task-21128)"
    )
    assert _fts_rowids(db, "dddneedle") == [rowid], "the existing index row was lost"
    _assert_index_structurally_sound(db)


def test_soft_delete_still_drops_the_message_from_the_index(db, conversation):
    """The regression `AFTER UPDATE OF content` alone would have shipped.

    Mutation-proved: with `deleted` removed from the trigger's column list
    this assertion returns the tombstoned rowid, because
    `soft_delete_message` issues `UPDATE messages SET deleted = 1 ...` and
    never names `content`. `Tests/DB/test_fts_soft_delete_index_witness.py`
    is the same guarantee reached through the other production entry points.
    """
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "eeeneedle body"}
    )
    rowid = _rowid_of(db, message_id)
    assert _fts_rowids(db, "eeeneedle") == [rowid]

    db.soft_delete_message(message_id, expected_version=1)

    assert _fts_rowids(db, "eeeneedle") == []
    _assert_index_structurally_sound(db)


def test_undelete_puts_the_message_back_into_the_index(db, conversation):
    """The other half of the `deleted` dependency -- and it must not be
    satisfiable by never removing the row in the first place, which is why
    the removal is asserted in between."""
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "fffneedle body"}
    )
    rowid = _rowid_of(db, message_id)
    db.soft_delete_message(message_id, expected_version=1)
    assert _fts_rowids(db, "fffneedle") == []

    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET deleted = 0, version = version + 1 WHERE id = ?",
            (message_id,),
        )

    assert _fts_rowids(db, "fffneedle") == [rowid]
    _assert_index_structurally_sound(db)


def test_hard_delete_removes_an_indexed_row(db, conversation):
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "gggneedle body"}
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    assert _fts_rowids(db, "gggneedle") == []
    _assert_index_structurally_sound(db)


def test_streaming_finalize_indexes_the_finished_reply(db, conversation):
    """The shape a real turn takes: placeholder row, then one content write."""
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "assistant", "content": "..."}
    )
    rowid = _rowid_of(db, message_id)

    db.update_message(
        message_id, {"content": f"hhhneedle {REPLY_BODY}"}, expected_version=1
    )

    assert _fts_rowids(db, "hhhneedle") == [rowid]
    assert _fts_rowids(db, "replyneedle0399") == [rowid]
    _assert_index_structurally_sound(db)


# ---------------------------------------------------------------------------
# matrix A, the EDGES -- a per-cell matrix cannot see a bad handoff
# ---------------------------------------------------------------------------
def test_a_content_edit_after_auxiliary_flushes_still_reindexes(db, conversation):
    """The transition the narrowing could plausibly break.

    Each cell above is green in isolation under several wrong trigger shapes.
    What has to hold is the SEQUENCE: skipping the auxiliary writes must not
    leave the index in a state where the next real content write reindexes
    the wrong text (or nothing at all).
    """
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "assistant", "content": "iiineedle first"}
    )
    rowid = _rowid_of(db, message_id)

    db.update_message_usage_local(message_id, json.dumps({"in": 1}))
    db.update_message_metadata_local(message_id, json.dumps({"model": "m"}))
    db.update_message(message_id, {"content": "jjjneedle second"}, expected_version=1)
    db.update_message_usage_local(message_id, json.dumps({"in": 2}))
    db.update_message(message_id, {"content": "kkkneedle third"}, expected_version=2)

    assert _fts_rowids(db, "kkkneedle") == [rowid]
    assert _fts_rowids(db, "jjjneedle") == []
    assert _fts_rowids(db, "iiineedle") == []
    _assert_index_structurally_sound(db)


def test_soft_delete_after_auxiliary_flushes_still_drops_the_row(db, conversation):
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "lllneedle body"}
    )
    db.update_message_metadata_local(message_id, json.dumps({"model": "m"}))
    db.update_message_usage_local(message_id, json.dumps({"in": 1}))

    db.soft_delete_message(message_id, expected_version=1)

    assert _fts_rowids(db, "lllneedle") == []
    _assert_index_structurally_sound(db)


def test_hard_delete_after_auxiliary_flushes_leaves_a_sound_index(db, conversation):
    message_id = db.add_message(
        {"conversation_id": conversation, "sender": "user", "content": "mmmneedle body"}
    )
    db.update_message_usage_local(message_id, json.dumps({"in": 1}))

    with db.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    assert _fts_rowids(db, "mmmneedle") == []
    _assert_index_structurally_sound(db)


# ---------------------------------------------------------------------------
# matrix B: an UN-BACKFILLED row (task-21100's window)
# ---------------------------------------------------------------------------
def test_the_whole_matrix_is_safe_on_un_backfilled_rows(tmp_path: Path):
    """Every cell again, against rows that are LIVE but absent from the index.

    This is the state a first boot after the v46 upgrade leaves behind, and it
    is where an fts5 'delete' for an unindexed rowid corrupts the index --
    silently, with a green integrity-check, when the index is partly filled
    (task-21100's review measured that). The "must not write" cells are
    therefore witnessed against physical storage, and the arm ends by proving
    the backfill still converges on exactly the live rows.
    """
    db_path = tmp_path / "chachanotes.db"
    bodies = [f"windowneedle{i:03d} body" for i in range(8)]
    message_ids, rowids = _seed_v45(db_path, bodies)

    migrated = CharactersRAGDB(
        db_path,
        client_id="v49-window",
        console_library_migration_seed=MIGRATION_SEED,
    )
    try:
        assert _version(migrated.get_connection()) == (
            CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert _docsize_rowids(migrated) == set(), "the window is not open"

        # usage-only flush on an un-backfilled row: the trigger must not fire
        # at all, so there is no 'delete' to corrupt with and nothing indexed.
        before = _fts_storage_image(migrated)
        migrated.update_message_usage_local(message_ids[0], json.dumps({"in": 1}))
        assert _fts_storage_image(migrated) == before
        assert _docsize_rowids(migrated) == set()
        _assert_index_structurally_sound(migrated)

        # metadata-only flush: same.
        migrated.update_message_metadata_local(message_ids[0], json.dumps({"m": 1}))
        assert _fts_storage_image(migrated) == before
        assert _docsize_rowids(migrated) == set()
        _assert_index_structurally_sound(migrated)

        # content edit: the guarded delete half skips (not in the index), the
        # insert half indexes the new content -- the row is indexed EARLY, and
        # the backfill (keyed on the same membership) then skips it.
        migrated.update_message(
            message_ids[1], {"content": "editedneedle body"}, expected_version=1
        )
        assert _fts_rowids(migrated, "editedneedle") == [rowids[1]]
        assert _fts_rowids(migrated, "windowneedle001") == []
        assert _docsize_rowids(migrated) == {rowids[1]}
        _assert_index_structurally_sound(migrated)

        # soft delete of an un-backfilled row: nothing to remove, no write.
        before = _fts_storage_image(migrated)
        migrated.soft_delete_message(message_ids[2], expected_version=1)
        assert _fts_storage_image(migrated) == before, (
            "soft-deleting an unindexed row wrote into messages_fts_data -- "
            "the v47 membership guard is not holding under the new scope"
        )
        _assert_index_structurally_sound(migrated)

        # undelete of an un-backfilled row: indexed early, like a content edit.
        with migrated.transaction() as conn:
            conn.execute(
                "UPDATE messages SET deleted = 0, version = version + 1 WHERE id = ?",
                (message_ids[2],),
            )
        assert _fts_rowids(migrated, "windowneedle002") == [rowids[2]]
        _assert_index_structurally_sound(migrated)

        # hard delete of an un-backfilled row: messages_ad's membership guard.
        before = _fts_storage_image(migrated)
        with migrated.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (message_ids[3],))
        assert _fts_storage_image(migrated) == before, (
            "hard-deleting an unindexed row wrote into messages_fts_data"
        )
        _assert_index_structurally_sound(migrated)

        # streaming finalize on an un-backfilled placeholder row.
        conversation_id = migrated.execute_query(
            "SELECT conversation_id FROM messages WHERE id = ?", (message_ids[0],)
        ).fetchone()[0]
        streamed = migrated.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "...",
            }
        )
        migrated.update_message(
            streamed, {"content": "streamneedle finished"}, expected_version=1
        )
        assert _fts_rowids(migrated, "streamneedle") == [_rowid_of(migrated, streamed)]
        _assert_index_structurally_sound(migrated)

        # And the backfill still converges on exactly the live rows: 8 seeded,
        # one hard-deleted, two indexed early (the edit and the undelete), one
        # streamed row already indexed by messages_ai.
        assert backfill_chachanotes_messages_fts(migrated, chunk_size=2) == 5
        assert _docsize_rowids(migrated) == _live_rowids(migrated)
        assert _fts_rowids(migrated, "windowneedle007") == [rowids[7]]
        assert _fts_rowids(migrated, "windowneedle003") == []
        _assert_index_structurally_sound(migrated)
        # Idempotent afterwards.
        assert backfill_chachanotes_messages_fts(migrated) == 0
    finally:
        migrated.close_connection()


# ---------------------------------------------------------------------------
# the write-count probe (AC #5)
# ---------------------------------------------------------------------------
def _streamed_turn_index_rewrites(db: CharactersRAGDB) -> tuple[int, int]:
    """Drive one streamed turn; return (rewrites, bytes written into the index).

    A "rewrite" is one turn statement after which the index's physical storage
    moved. The turn is the shape the Console actually issues: the user row, an
    assistant placeholder, the content finalize, then the auxiliary flushes
    that carry usage, metadata and a ranking edit -- three to four UPDATEs
    against a row whose indexed column never changes again.
    """
    conversation_id = db.add_conversation({"title": "turn", "character_id": 1})
    db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "turnneedle ask"}
    )
    assistant_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "assistant", "content": "..."}
    )

    image = _fts_storage_image(db)
    start_bytes = _fts_data_bytes(db)
    rewrites = 0

    def account() -> None:
        nonlocal image, rewrites
        current = _fts_storage_image(db)
        if current != image:
            rewrites += 1
            image = current

    db.update_message(assistant_id, {"content": REPLY_BODY}, expected_version=1)
    account()
    db.update_message_usage_local(assistant_id, json.dumps({"in": 10, "out": 20}))
    account()
    db.update_message_metadata_local(assistant_id, json.dumps({"model": "m"}))
    account()
    db.update_message(assistant_id, {"ranking": 1}, expected_version=2)
    account()

    assert _fts_rowids(db, "replyneedle0399") == [_rowid_of(db, assistant_id)], (
        "the turn did not leave the reply searchable -- the probe is measuring "
        "a broken index, not a cheaper one"
    )
    return rewrites, _fts_data_bytes(db) - start_bytes


def test_a_streamed_turn_rewrites_the_index_once(tmp_path: Path):
    """AC #5, as an in-process A/B that varies only the trigger definition.

    Both arms run the identical turn against an identically built database on
    the same connection shape; the ONLY difference is which `messages_au` is
    installed. Quoting a scratchpad number would not survive a future change
    to the turn's write count, and a single-arm assertion would not show the
    saving is real.
    """
    after_db = CharactersRAGDB(tmp_path / "after.db", client_id="v49-after")
    before_db = CharactersRAGDB(tmp_path / "before.db", client_id="v49-before")
    try:
        _install_trigger(before_db, V47_MESSAGES_AU)
        assert "after update on messages" in _trigger_sql(before_db, "messages_au")

        before_rewrites, before_bytes = _streamed_turn_index_rewrites(before_db)
        after_rewrites, after_bytes = _streamed_turn_index_rewrites(after_db)

        assert before_rewrites == 4, (
            "the pre-fix arm no longer rewrites the index on every auxiliary "
            f"write ({before_rewrites}); this probe has stopped measuring the "
            "defect and must be re-derived"
        )
        assert after_rewrites == 1, (
            f"one streamed turn rewrote messages_fts {after_rewrites} times "
            f"(bytes: {after_bytes} vs {before_bytes} pre-fix)"
        )
        # The amplification, not just the count: the reply is tokenized into
        # the index once instead of four times.
        assert after_bytes * 3 < before_bytes, (
            f"index bytes written per turn barely moved: {after_bytes} vs "
            f"{before_bytes}"
        )
        _assert_index_structurally_sound(after_db)
        _assert_index_structurally_sound(before_db)
    finally:
        after_db.close_connection()
        before_db.close_connection()
