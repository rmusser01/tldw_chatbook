"""The ``notes_au`` FTS undelete guard ships as a real migration (task-19565).

Why this exists: ``notes_au`` originally shipped without the ``deleted = 0``
guard on its FTS ``'delete'`` half (found and patched in the notes-undo work,
commit ``4d4dceebc4``). Because the v4 base script never re-runs on an
existing database, the repair could not ship through a schema bump -- instead
a runtime self-heal re-DROP/re-CREATEd the trigger on **every database
open**, forever, for every user, to fix a defect exactly once per database.
A trigger body that wrong had shipped to every user precisely because
nothing pinned trigger bodies (the trigger census in
``test_trigger_census.py`` now does).

This module pins the replacement: schema v76 carries the guarded body as a
real migration step, and the runtime self-heal is deleted. The red this test
was born under (evidence in the task notes): with the v75->v76 step's
trigger recreation neutered, the legacy body survives the upgrade.
"""

from __future__ import annotations

import sqlite3

from Tests.ChaChaNotesDB.historical_bootstrap import (
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

#: The exact pre-fix ``notes_au`` body as it shipped (commit 4d4dceebc4's
#: pre-image): the FTS 'delete' half as an unguarded ``VALUES``, so a
#: soft-deleted note was never removed from ``notes_fts`` and -- worse --
#: the unconditional 'delete' against an external-content FTS index
#: corrupted it on restore. Databases created by that build carry this body
#: in ``sqlite_master`` at ANY schema version, which is exactly why the fix
#: could not be a normal migration until now: there was no version boundary
#: to hang it on. v76 is that boundary: every existing database passes
#: through v75->v76 exactly once.
LEGACY_UNGUARDED_NOTES_AU_SQL = """
CREATE TRIGGER notes_au
AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts,rowid,title,content)
  VALUES('delete',old.rowid,old.title,old.content);

  INSERT INTO notes_fts(rowid,title,content)
  SELECT new.rowid,new.title,new.content
  WHERE new.deleted = 0;
END;
"""


def _normalized_trigger_sql(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
    ).fetchone()
    assert row is not None, f"trigger {name} is missing entirely"
    return " ".join(str(row[0]).split())


def _fresh_bootstrap_notes_au_body() -> str:
    db = CharactersRAGDB(":memory:", client_id="notes-au-fresh")
    try:
        return _normalized_trigger_sql(db.get_connection(), "notes_au")
    finally:
        db.close_connection()


class TestNotesFtsUndeleteGuardMigration:
    def test_legacy_broken_notes_au_body_is_repaired_by_migration(self, tmp_path):
        """A genuinely-v75 DB carrying the legacy unguarded body is repaired.

        The chain replay through v75->v76 must land the guarded body --
        byte-identical (whitespace-normalized) to what a fresh bootstrap
        creates, so the trigger census sees one shape, not two.
        """
        db_path = tmp_path / "legacy_notes_au.sqlite"
        with chachanotes_db_at_version(db_path, 75) as db:
            conn = db.get_connection()
            # Replace the (correct) bootstrap-time body with the historical
            # broken one, the exact shape a pre-fix database carries.
            conn.execute("DROP TRIGGER notes_au")
            conn.execute(LEGACY_UNGUARDED_NOTES_AU_SQL)
            assert "VALUES('delete'" in _normalized_trigger_sql(conn, "notes_au")

        migrated = open_current_chachanotes_from_legacy(
            db_path, client_id="notes-au-replay"
        )
        try:
            conn = migrated.get_connection()
            version = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            body = _normalized_trigger_sql(conn, "notes_au")
            assert "VALUES('delete'" not in body, (
                "the legacy unguarded FTS 'delete' half survived the v75->v76 "
                "upgrade -- the migration did not repair the trigger body"
            )
            assert "WHERE old.deleted = 0" in body
            assert "WHERE new.deleted = 0" in body
            assert body == _fresh_bootstrap_notes_au_body(), (
                "the migrated notes_au body diverges from a fresh bootstrap's "
                "-- the census would see two shapes for one trigger"
            )
        finally:
            migrated.close_connection()

    def test_migration_repair_works_functionally(self, tmp_path):
        """Soft-delete removes a note from FTS search after the upgrade.

        The historical defect's user-visible shape: a soft-deleted note kept
        answering FTS search (and the unguarded 'delete' corrupted the
        external-content index on restore). Pin the repaired behavior, not
        just the body text.
        """
        db_path = tmp_path / "legacy_notes_au_functional.sqlite"
        with chachanotes_db_at_version(db_path, 75) as db:
            conn = db.get_connection()
            conn.execute("DROP TRIGGER notes_au")
            conn.execute(LEGACY_UNGUARDED_NOTES_AU_SQL)

        migrated = open_current_chachanotes_from_legacy(
            db_path, client_id="notes-au-functional"
        )
        try:
            note_id = migrated.add_note(
                title="needle in the title",
                content="unique-zq-19565 body",
            )
            assert migrated.search_notes("unique-zq-19565"), "note is indexed"
            row = migrated.execute_query(
                "SELECT version FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
            migrated.soft_delete_note(note_id, expected_version=row["version"])
            assert migrated.search_notes("unique-zq-19565") == [], (
                "a soft-deleted note still answers FTS search after the "
                "v75->v76 repair"
            )
            row = migrated.execute_query(
                "SELECT version FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
            migrated.restore_note(note_id, expected_version=row["version"])
            assert migrated.search_notes("unique-zq-19565"), (
                "a restored note is missing from FTS search after the "
                "v75->v76 repair"
            )
        finally:
            migrated.close_connection()

    def test_no_runtime_self_heal_remains(self):
        """The every-open DROP/reCREATE fixup is gone, replaced by v76.

        The self-heal was the proof this task existed: a permanent startup
        repair standing in for a migration. If a future change reintroduces
        a schema-init trigger fixup, it should come back as a migration, not
        as an unconditional open-time rewrite.
        """
        import inspect

        source = inspect.getsource(CharactersRAGDB)
        assert "_ensure_notes_fts_update_trigger_handles_undelete" not in source
        assert source.count("CREATE TRIGGER notes_au") == 1, (
            "notes_au is created in more than one place in ChaChaNotes_DB.py "
            "-- the v4 base script should be its only in-module definition, "
            "with the v75->v76 migration file carrying the upgrade-path copy"
        )
