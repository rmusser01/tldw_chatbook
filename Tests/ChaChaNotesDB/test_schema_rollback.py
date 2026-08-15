"""Guards for the shared ChaChaNotes rollback registry (schema_rollback.py).

Two enforcement halves for the fixture class that produced task-15730,
task-15765, and task-16197 ("table note_folders already exists"):

* a completeness ratchet — bumping ``_CURRENT_SCHEMA_VERSION`` without
  declaring a rollback entry for the new version fails HERE, by name, with
  instructions, instead of failing three scattered historical fixtures with
  a confusing "already exists" at an unrelated migration's step;
* a rollback-replay sweep — every historical version a fixture could target
  is rewound and replayed through the production migration chain, and the
  result must reach the current version with a schema identical to a fresh
  bootstrap: object inventory (type, name) PLUS per-table column sets.
  Columns are not sqlite_master rows, so an object-only oracle is blind to a
  wrong ``DROP COLUMN`` registry entry — half the registry is column drops.
  Column comparison is by SET, not position: replay legitimately re-appends
  a dropped column at the end of the table (fresh has ``messages.usage_json``
  before ``metadata_json``; a v16..v29 replay has the reverse).
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.schema_rollback import (
    MINIMUM_ROLLBACK_VERSION,
    POST_VERSION_SCHEMA_REMOVALS,
    SCHEMA_NAME,
    rollback_chachanotes_schema,
)


def test_rollback_registry_covers_every_schema_version():
    declared = set(POST_VERSION_SCHEMA_REMOVALS)
    required = set(
        range(
            MINIMUM_ROLLBACK_VERSION + 1,
            CharactersRAGDB._CURRENT_SCHEMA_VERSION + 1,
        )
    )
    assert declared == required, (
        "POST_VERSION_SCHEMA_REMOVALS must declare exactly versions "
        f"{min(required)}..{max(required)}. "
        f"Missing: {sorted(required - declared)}; "
        f"stale: {sorted(declared - required)}. If you just added a schema "
        "migration, add an entry to Tests/ChaChaNotesDB/schema_rollback.py "
        "removing what it creates (an empty tuple only if replaying the "
        "migration over its own baked artifacts is tolerated)."
    )


@pytest.fixture(scope="module")
def fresh_template_db(tmp_path_factory):
    """Bootstrap one current-version DB per module; sweep cases copy it."""
    path = tmp_path_factory.mktemp("chacha_template") / "template.sqlite"
    db = CharactersRAGDB(str(path), client_id="template-client")
    db.close_connection()
    return path


def _schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Object inventory plus per-table column membership.

    Each table contributes a ``("column", "<table>.<column>")`` entry per
    column, so column loss (invisible in sqlite_master) fails the parity
    assertion by name. Membership is a set on purpose — column ORDER may
    legitimately diverge between a replayed and a fresh DB.
    """
    objects = {
        (row[0], row[1])
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }
    for object_type, name in sorted(objects):
        if object_type != "table":
            continue
        objects.update(
            ("column", f"{name}.{column_row[1]}")
            for column_row in conn.execute(f'PRAGMA table_info("{name}")')
        )
    return objects


@pytest.mark.parametrize(
    "target_version",
    range(MINIMUM_ROLLBACK_VERSION, CharactersRAGDB._CURRENT_SCHEMA_VERSION),
)
def test_rollback_then_replay_reaches_current_schema(
    fresh_template_db, tmp_path, target_version
):
    db_path = tmp_path / f"rollback_v{target_version}.sqlite"
    shutil.copy(fresh_template_db, db_path)

    conn = sqlite3.connect(db_path)
    try:
        rollback_chachanotes_schema(conn, target_version)
        conn.commit()
    finally:
        conn.close()

    migrated = CharactersRAGDB(str(db_path), client_id="sweep-client")
    try:
        migrated_conn = migrated.get_connection()
        version = migrated_conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()[0]
        assert version == migrated._CURRENT_SCHEMA_VERSION

        fresh_conn = sqlite3.connect(fresh_template_db)
        try:
            fresh_objects = _schema_objects(fresh_conn)
        finally:
            fresh_conn.close()
        replayed_objects = _schema_objects(migrated_conn)
        assert replayed_objects == fresh_objects, (
            f"rollback to v{target_version} + replay diverged from a fresh "
            f"bootstrap: missing={sorted(fresh_objects - replayed_objects)} "
            f"extra={sorted(replayed_objects - fresh_objects)}"
        )
    finally:
        migrated.close_connection()
