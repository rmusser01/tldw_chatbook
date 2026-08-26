"""ChaChaNotes v49 -> v50: Console Library policy rows follow live conversations.

As shipped, the v47 -> v48 step seeded
``console_conversation_library_policy`` from ``SELECT id FROM conversations``
with no ``deleted`` predicate, so every conversation the profile had EVER held
-- tombstones included -- got a permanent row, written inside the boot
version-bump transaction (task-22225).

Two populations exist and they must end in the same state:

* a database that has NOT yet run v48 -- the seed itself now excludes
  soft-deleted conversations, so the rows are never written; and
* a database that ALREADY ran the shipped v48 -- v49 -> v50 removes every
  policy row that has no live conversation behind it.

The second half is why v48 could be edited at all. Editing an applied step
only changes the outcome for databases that have not reached it; the forward
step is what makes the already-migrated profile converge, in the same open.

This module holds the repo's exact current-schema-version pin, which belongs
to the NEWEST migration's own file so that a schema bump touches the file that
caused it (the convention
``Tests/DB/test_chachanotes_console_library_policy_migration.py`` records).
Everything downstream of a completed upgrade reads
``_CURRENT_SCHEMA_VERSION``: a version literal is only correct at a fixture's
SEEDED starting point.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME
MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "DB"
    / "migrations"
    / "chachanotes_v49_to_v50_console_policy_tombstone_cleanup.sql"
)

#: The v47 -> v48 seed EXACTLY as it shipped, so the already-migrated
#: population is replayed rather than hand-built. Keep this text frozen: it is
#: a record of what is in users' databases, not a copy of current production
#: code, and it must not be "fixed" when the production method changes.
_SHIPPED_V48_SEED_SQL = """
    INSERT INTO console_conversation_library_policy(
        conversation_id,
        auto_retrieve_on_send,
        assistant_library_access
    )
    SELECT id, ?, 1
      FROM conversations
"""


def _shipped_v48_seed(
    self: CharactersRAGDB,
    cursor: sqlite3.Cursor,
    auto_retrieve_on_send: int,
) -> None:
    cursor.execute(_SHIPPED_V48_SEED_SQL, (auto_retrieve_on_send,))


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _insert_conversation(
    connection: sqlite3.Connection,
    conversation_id: str,
    *,
    deleted: int = 0,
) -> None:
    connection.execute(
        """
        INSERT INTO conversations(id, root_id, title, deleted, client_id, version)
        VALUES (?, ?, ?, ?, 'tombstone-fixture', 1)
        """,
        (conversation_id, conversation_id, conversation_id, deleted),
    )


def _policy_ids(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT conversation_id FROM console_conversation_library_policy"
            " ORDER BY conversation_id"
        )
    ]


def _policy_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    """Every policy column except ``updated_at``, which is CURRENT_TIMESTAMP."""
    return [
        tuple(row)
        for row in connection.execute(
            "SELECT conversation_id, schema_version, auto_retrieve_on_send,"
            " assistant_library_access, policy_revision"
            " FROM console_conversation_library_policy ORDER BY conversation_id"
        )
    ]


def _raw_policy_rows(db_path: Path) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(str(db_path))
    try:
        return _policy_rows(connection)
    finally:
        connection.close()


def _build_v47_with_a_tombstone(path: Path) -> None:
    """A v47 profile holding one live and one soft-deleted conversation."""
    with chachanotes_db_at_version(path, 47) as at47:
        connection = at47.get_connection()
        _insert_conversation(connection, "live-conversation")
        _insert_conversation(connection, "tombstoned-conversation", deleted=1)
        connection.commit()


def _build_shipped_v48_profile(path: Path) -> None:
    """A profile carrying the over-seeded rows the shipped v48 really wrote."""
    _build_v47_with_a_tombstone(path)
    with patch.object(
        CharactersRAGDB,
        "_seed_console_library_policy_rows",
        _shipped_v48_seed,
    ):
        with chachanotes_db_at_version(path, 49, client_id="shipped-v48") as at49:
            connection = at49.get_connection()
            assert _version(connection) == 49
            assert _policy_ids(connection) == [
                "live-conversation",
                "tombstoned-conversation",
            ], "the shipped-v48 replay did not reproduce the over-seeded state"


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------
def test_schema_version_is_50(tmp_path: Path) -> None:
    """The one exact current-version pin (see this module's docstring)."""
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="v50-pin")
    try:
        assert _version(db.get_connection()) == 50
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 50
    finally:
        db.close_connection()


def test_migrate_from_v49_to_v50_requires_version_49(tmp_path: Path) -> None:
    """A fresh DB is already current, so re-entering the step must refuse."""
    db = CharactersRAGDB(tmp_path / "entry.db", client_id="v50-entry")
    try:
        with pytest.raises(SchemaError, match="requires schema version"):
            db._migrate_from_v49_to_v50(db.get_connection())
    finally:
        db.close_connection()


# ---------------------------------------------------------------------------
# population 1: a database that has not yet run v48
# ---------------------------------------------------------------------------
def test_the_v48_seed_itself_never_writes_a_tombstone_row(tmp_path: Path) -> None:
    """Stop the chain AT 48, where the seed's own output is observable.

    This is the only assertion that can see the seed. Everything downstream
    runs the v49 -> v50 cleanup in the same chain, which deletes exactly the
    rows a broken seed would have written -- so a whole-chain test passes
    identically whether the ``WHERE deleted = 0`` is there or not (measured:
    removing it left every other test in this module green). The end state
    would be right and the boot cost the finding is about would be back.
    """
    db_path = tmp_path / "seed-only.db"
    _build_v47_with_a_tombstone(db_path)

    with chachanotes_db_at_version(db_path, 48, client_id="seed-only") as at48:
        connection = at48.get_connection()
        assert _version(connection) == 48
        assert _policy_ids(connection) == ["live-conversation"], (
            "the v47 -> v48 seed still writes a policy row for a soft-deleted "
            "conversation; the v50 cleanup would hide this everywhere else"
        )


def test_a_v47_upgrade_seeds_only_live_conversations(tmp_path: Path) -> None:
    """End to end, a not-yet-migrated profile gains only the live row."""
    db_path = tmp_path / "fresh-upgrade.db"
    _build_v47_with_a_tombstone(db_path)

    migrated = CharactersRAGDB(db_path, client_id="v50-fresh-upgrade")
    try:
        connection = migrated.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _policy_ids(connection) == ["live-conversation"], (
            "the seed still writes a policy row for a soft-deleted conversation"
        )
        # The live conversation keeps the full seeded contract: assistant
        # Library access Allowed, at revision one.
        assert _policy_rows(connection) == [("live-conversation", 1, 0, 1, 1)]
    finally:
        migrated.close_connection()


# ---------------------------------------------------------------------------
# population 2: a database that already ran the shipped v48
# ---------------------------------------------------------------------------
def test_a_shipped_v48_profile_loses_its_tombstone_policy_rows(
    tmp_path: Path,
) -> None:
    """The forward step is what makes the already-migrated profile converge."""
    db_path = tmp_path / "shipped-v48.db"
    _build_shipped_v48_profile(db_path)

    migrated = CharactersRAGDB(db_path, client_id="v50-cleanup")
    try:
        connection = migrated.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _policy_ids(connection) == ["live-conversation"], (
            "the over-seeded tombstone row survived the v49 -> v50 step"
        )
        # The tombstoned conversation itself is untouched: this step retires
        # dead policy, not user data.
        assert connection.execute(
            "SELECT deleted FROM conversations WHERE id = ?",
            ("tombstoned-conversation",),
        ).fetchone()[0] == 1
    finally:
        migrated.close_connection()


def test_both_populations_converge_on_the_same_policy_rows(tmp_path: Path) -> None:
    """A user who already migrated ends where a user who migrates now ends."""
    already = tmp_path / "already-migrated.db"
    not_yet = tmp_path / "not-yet-migrated.db"
    _build_shipped_v48_profile(already)
    _build_v47_with_a_tombstone(not_yet)

    for path, client in ((already, "converge-already"), (not_yet, "converge-fresh")):
        db = CharactersRAGDB(path, client_id=client)
        try:
            assert _version(db.get_connection()) == (
                CharactersRAGDB._CURRENT_SCHEMA_VERSION
            )
        finally:
            db.close_connection()

    assert _raw_policy_rows(already) == _raw_policy_rows(not_yet)
    assert _raw_policy_rows(already) == [("live-conversation", 1, 0, 1, 1)]


def test_the_cleanup_keeps_user_authored_policy_on_live_conversations(
    tmp_path: Path,
) -> None:
    """Only rows with no live conversation behind them are removed."""
    db_path = tmp_path / "authored.db"
    _build_shipped_v48_profile(db_path)
    with sqlite3.connect(str(db_path)) as seeded:
        seeded.execute(
            "UPDATE console_conversation_library_policy"
            "   SET auto_retrieve_on_send = 1, assistant_library_access = 0,"
            "       policy_revision = 4"
            " WHERE conversation_id = 'live-conversation'"
        )

    migrated = CharactersRAGDB(db_path, client_id="v50-authored")
    try:
        assert _policy_rows(migrated.get_connection()) == [
            ("live-conversation", 1, 1, 0, 4)
        ], "the step rewrote a policy the user authored"
    finally:
        migrated.close_connection()


def test_the_cleanup_is_idempotent(tmp_path: Path) -> None:
    """Re-applying the step's SQL over a migrated database is a no-op."""
    db_path = tmp_path / "idempotent.db"
    _build_shipped_v48_profile(db_path)
    CharactersRAGDB(db_path, client_id="v50-first-run").close_connection()
    after_first = _raw_policy_rows(db_path)

    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.executescript(
            MIGRATION_PATH.read_text(encoding="utf-8")
        )
        connection.commit()
        assert _policy_rows(connection) == after_first
    finally:
        connection.close()

    # And a plain reopen -- the chain no-ops at the current version -- is
    # equally stable.
    reopened = CharactersRAGDB(db_path, client_id="v50-second-run")
    try:
        assert _policy_rows(reopened.get_connection()) == after_first
    finally:
        reopened.close_connection()


def test_a_failure_mid_v50_rewinds_the_whole_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic and re-enterable: the task-19553/21441 partial-apply class."""
    db_path = tmp_path / "poisoned.db"
    _build_shipped_v48_profile(db_path)
    before = _raw_policy_rows(db_path)
    assert len(before) == 2

    original = CharactersRAGDB._execute_migration_statements

    def poisoned(self, cursor, script, label):
        if label == "V49→V50":
            script = script + "\nINSERT INTO no_such_table_22225(x) VALUES (1);\n"
        return original(self, cursor, script, label)

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", poisoned)
    with pytest.raises(SchemaError, match="no_such_table_22225"):
        CharactersRAGDB(db_path, client_id="poisoned")

    connection = sqlite3.connect(str(db_path))
    try:
        assert _version(connection) == 49, "a failing chain must not bump the stamp"
        assert _policy_rows(connection) == before, (
            "a half-applied cleanup left rows deleted at the old version stamp"
        )
    finally:
        connection.close()

    monkeypatch.undo()
    migrated = CharactersRAGDB(db_path, client_id="poison-removed")
    try:
        assert _version(migrated.get_connection()) == (
            CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert _policy_ids(migrated.get_connection()) == ["live-conversation"]
    finally:
        migrated.close_connection()
