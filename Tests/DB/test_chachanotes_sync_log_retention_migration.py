"""ChaChaNotes v45 -> v46: `sync_log` bounded to its reachable frontier.

task-19564. 35 triggers wrote the complete row as JSON into `sync_log` and
nothing ever removed one, so a soft delete left the user's plaintext in the
database indefinitely and every edit left a full extra copy behind forever.
This migration installs the retention triggers and performs the one-time purge
that an EXISTING database needs -- a fix that only helped new installs would
leave the plaintext in place for everyone who already has it, which is what
`test_upgrading_a_real_v44_database_purges_its_backlog` is here to prevent.

It also repairs the three FTS `*_au` triggers whose DELETE half was unguarded
(task-19567) — `messages_au`, `keyword_collections_au`, `world_books_au`;
`keyword_collections_au` was live-reachable and corrupted the index. The
behavioural coverage for that lives in
`Tests/DB/test_fts_soft_delete_index_witness.py`.

This module carries the repo's EXACT current-schema-version pin. It reached
here by two hops -- from `test_chachanotes_sync_conflict_preservation_migration
.py` (v44) to `Tests/ChaChaNotesDB/test_actor_pack_migration.py` (v45) to here
(v46) -- because the pin belongs to the NEWEST migration's own file, so a
schema bump touches the file that caused it rather than an unrelated older one
(older files assert `>= their own version` instead). Updating the number here
is a deliberate schema-review act.

This step was authored as v44->v45 and renumbered to v45->v46 when TASK-19057
(portable Actor Pack identity) merged to dev claiming v45 first. The seeds
below deliberately still start at a real **v44** database, so every upgrade
assertion now runs THROUGH the Actor Packs step and lands at v46 -- the chain,
not just this one link.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME

# Named `sync_log_prune_<entity>`, NOT `<entity>_sync_log_prune`: the
# `<entity>_sync_%` namespace belongs to the four triggers that WRITE the log,
# and three tests assert that namespace's membership exactly (`_` is a
# single-character wildcard in SQL LIKE, so `conversations_sync_log_prune`
# matches `conversations_sync_%`). Retention is a different concern and gets
# its own prefix.
RETENTION_TRIGGERS = {
    f"sync_log_prune_{entity}{suffix}"
    for entity in (
        "messages",
        "conversations",
        "notes",
        "character_cards",
        "keywords",
        "keyword_collections",
    )
    for suffix in ("", "_hard")
}


def _version(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()[0]


def _triggers(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }


@pytest.fixture
def db(tmp_path: Path):
    instance = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="v46-test")
    yield instance
    instance.close_connection()


def test_schema_version_is_46(db):
    """The one exact current-version pin (see this module's docstring)."""
    assert _version(db.get_connection()) == 46
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 46


def test_fresh_schema_has_every_retention_trigger(db):
    assert RETENTION_TRIGGERS <= _triggers(db.get_connection())


def test_migrate_from_v45_to_v46_requires_version_45(db):
    """A fresh DB is already at 46, so re-entering the step must refuse."""
    with pytest.raises(SchemaError, match="requires schema version"):
        db._migrate_from_v45_to_v46(db.get_connection())


def test_upgrading_a_real_v44_database_purges_its_backlog(tmp_path: Path):
    """The AC that a fix helping only new installs would fail.

    A genuine v44 database is seeded with the shape the shipped code produces:
    a soft-deleted message whose `create` intent still carries its body, a
    superseded edit, and an orphan whose entity row is gone. After the reopen
    they are gone and the frontier is intact.
    """
    db_path = tmp_path / "chachanotes.db"
    kept_body = "frontier body that must survive"
    deleted_body = "tombstoned body that must not"
    superseded_body = "superseded body that must not"

    with chachanotes_db_at_version(db_path, 44) as historical:
        conversation_id = historical.add_conversation(
            {"title": "upgrade", "character_id": 1}
        )
        deleted_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": deleted_body,
            }
        )
        historical.soft_delete_message(deleted_id, expected_version=1)
        edited_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": superseded_body,
            }
        )
        historical.update_message(
            edited_id, {"content": "second"}, expected_version=1
        )
        historical.update_message(edited_id, {"content": kept_body}, expected_version=2)
        with historical.transaction() as conn:
            conn.execute(
                "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
                "client_id, version, payload) VALUES ('notes', 'vanished', 'create',"
                " '2020-01-01T00:00:00Z', 'legacy', 1, ?)",
                (json.dumps({"id": "vanished", "content": "orphan body"}),),
            )
        connection = historical.get_connection()
        # The defect, present on a real v44 database. This `== 44` is a
        # precondition on the BOOTSTRAP FIXTURE's own argument, not a
        # current-schema pin -- it cannot go stale on a schema bump, so it is
        # not the short-circuiting shape task-19568 removed. The
        # post-migration assertion below is dynamic for exactly that reason.
        assert _version(connection) == 44
        assert RETENTION_TRIGGERS.isdisjoint(_triggers(connection))
        payloads_before = [
            row[0]
            for row in connection.execute("SELECT payload FROM sync_log")
        ]
        assert any(deleted_body in payload for payload in payloads_before)
        assert any(superseded_body in payload for payload in payloads_before)
        assert any("orphan body" in payload for payload in payloads_before)

    migrated = CharactersRAGDB(db_path, client_id="v46-upgrade")
    try:
        connection = migrated.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert RETENTION_TRIGGERS <= _triggers(connection)
        payloads_after = [
            row[0] for row in connection.execute("SELECT payload FROM sync_log")
        ]
        assert not any(deleted_body in payload for payload in payloads_after)
        assert not any(superseded_body in payload for payload in payloads_after)
        assert not any("orphan body" in payload for payload in payloads_after)
        # The frontier the intent readers join to is untouched.
        assert any(kept_body in payload for payload in payloads_after)
        # The tombstone survives; it is the delete proof and carries no body.
        assert [
            row[0]
            for row in connection.execute(
                "SELECT operation FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? ORDER BY change_id",
                (deleted_id,),
            )
        ] == ["delete"]
    finally:
        migrated.close_connection()


def test_a_failure_mid_step_rewinds_to_v44_with_nothing_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """This step must be atomic and re-enterable (task-19553's rule).

    The purge here DELETEs user rows, so a step that could commit half of
    itself would destroy content while leaving the stamp at 44 and the
    retention triggers absent -- the database would then re-enter the step
    forever. Poison a statement in the middle and require a full rewind.
    """
    db_path = tmp_path / "poisoned.db"
    needle = "zqxpoisonneedle"
    with chachanotes_db_at_version(db_path, 44) as historical:
        conversation_id = historical.add_conversation(
            {"title": "poison", "character_id": 1}
        )
        message_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": f"body {needle}",
            }
        )
        historical.soft_delete_message(message_id, expected_version=1)
        rows_before = historical.get_connection().execute(
            "SELECT COUNT(*) FROM sync_log"
        ).fetchone()[0]

    original = CharactersRAGDB._execute_migration_statements

    def poisoned(self, cursor, script, label):
        if label == "V45→V46":
            # Appended, so it runs AFTER every trigger create AND after every
            # purge DELETE. A step that could commit part of itself would have
            # already destroyed the sync_log rows by the time this fires --
            # which is exactly what the assertions below would catch.
            script = script + "\nINSERT INTO no_such_table_19564(x) VALUES (1);\n"
        return original(self, cursor, script, label)

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", poisoned)

    with pytest.raises(SchemaError, match="no_such_table_19564"):
        CharactersRAGDB(db_path, client_id="poisoned")

    connection = sqlite3.connect(str(db_path))
    try:
        assert _version(connection) == 44, "a failing step must not bump the stamp"
        assert RETENTION_TRIGGERS.isdisjoint(_triggers(connection))
        assert (
            connection.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
            == rows_before
        ), "no row may be purged by a step that did not complete"
    finally:
        connection.close()

    # Re-enterable: with the poison removed the same file migrates.
    monkeypatch.undo()
    migrated = CharactersRAGDB(db_path, client_id="poison-removed")
    try:
        assert _version(migrated.get_connection()) == (
            CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert RETENTION_TRIGGERS <= _triggers(migrated.get_connection())
        assert migrated.execute_query(
            "SELECT COUNT(*) FROM sync_log WHERE payload LIKE ?", (f"%{needle}%",)
        ).fetchone()[0] == 0
    finally:
        migrated.close_connection()


def test_upgrading_reindexes_only_live_rows_into_messages_fts(tmp_path: Path):
    """The FTS reset must not re-index tombstoned rows (task-19567).

    The obvious way to repair the corrupted index is FTS5 `'rebuild'`, which
    re-derives from the base table with no `deleted` filter -- that would put
    every soft-deleted message back into the index and reintroduce exactly the
    leak the guard exists to prevent. The migration uses `'delete-all'` plus an
    explicit filtered reinsert instead; this is the assertion that says so.
    """
    db_path = tmp_path / "chachanotes.db"
    needle = "zqxupgradeneedle"
    with chachanotes_db_at_version(db_path, 44) as historical:
        conversation_id = historical.add_conversation(
            {"title": "fts", "character_id": 1}
        )
        live_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": f"live {needle}",
            }
        )
        gone_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": f"gone {needle}",
            }
        )
        historical.soft_delete_message(gone_id, expected_version=1)

    migrated = CharactersRAGDB(db_path, client_id="v46-upgrade")
    try:
        rowids = [
            row[0]
            for row in migrated.execute_query(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
                (needle,),
            ).fetchall()
        ]
        live_rowid = migrated.execute_query(
            "SELECT rowid FROM messages WHERE id = ?", (live_id,)
        ).fetchone()[0]
        assert rowids == [live_rowid]
    finally:
        migrated.close_connection()
