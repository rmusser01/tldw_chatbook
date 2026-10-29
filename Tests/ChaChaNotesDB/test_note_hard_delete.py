"""Delete forever: one deleted note leaves no trace in ChaChaNotes.

TASK-34000.29 (N-29, part 8b; AC#9, ruling T8b-b). The owner ruled
"yes with confimration" (2026-10-29). ``CharactersRAGDB.hard_delete_note``
purges one soft-deleted note in ONE ``BEGIN IMMEDIATE`` transaction and
refuses a live one. Afterwards the note cannot be found through search,
keywords, folders, versions or the sync log.

Every place these tests look is DISCOVERED from the live schema by
``note_trace_census`` -- no hand-kept table list. The fixture fills every
discovered note-id column before the purge (a precondition, so a table added
later fails here until the fixture -- and the purge -- know it), and the
negative control adds a note-referencing table the purge does not know and
shows the census catches the row left in it.

Base (71f8d8a487): ``hard_delete_note`` does not exist, and a bare
``DELETE FROM notes`` on a tombstone raises "database disk image is
malformed" -- ``notes_ad`` issues the FTS ``'delete'`` for a row the soft
delete already took out of the external-content index.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.note_trace_census import note_id_columns, note_traces
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    NoteNotInTrashError,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository

pytestmark = pytest.mark.unit

_HEX = "a" * 64
#: Words only the purged note ever held (the Backlinker spells its title).
_TOKENS = ("plutonium",)


class _Purgeable:
    """A soft-deleted note referenced from every discovered note-id column."""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "notes.sqlite3"
        self.db = CharactersRAGDB(self.path, client_id="task-34000-29-8b")
        db = self.db
        self.keeper = db.add_note("Keeper note", "shared words survive here")
        self.note_id = db.add_note(
            "Purge zeppelinward", "plutonium secret body links [[Keeper note]]"
        )
        self.linker = db.add_note("Backlinker", "see [[Purge zeppelinward]]")
        self.keyword_ids = [db.add_keyword(word) for word in ("alpha", "beta")]
        for keyword_id in self.keyword_ids:
            db.link_note_to_keyword(self.note_id, keyword_id)
        db.link_note_to_keyword(self.keeper, self.keyword_ids[0])
        folders = LocalNoteFolderRepository(db)
        folder = folders.create_folder(name="Projects", parent_id=None)
        folders.attach_manual(folder_id=folder.folder_id, note_id=self.note_id)
        folders.attach_manual(folder_id=folder.folder_id, note_id=self.keeper)
        version = db.get_note_by_id(self.note_id)["version"]
        db.soft_delete_note(self.note_id, version)
        self.version = version + 1
        with sqlite3.connect(self.path) as connection:
            self.rowid = connection.execute(
                "SELECT rowid FROM notes WHERE id = ?", (self.note_id,)
            ).fetchone()[0]
            # The rows no Library gesture writes directly: a folder-sync
            # suppression, a legacy file-sync conflict that preserved a losing
            # version, a pending organization receipt and a publication
            # intent carrying the note's text, a research owner proof.
            connection.execute(
                "INSERT INTO note_folder_sync_suppressions VALUES (?, 'sync-folder-1', "
                "'2026-10-29T00:00:00Z')",
                (self.note_id,),
            )
            connection.execute(
                "INSERT INTO sync_sessions(session_id, sync_root_folder, "
                "sync_direction, conflict_resolution, status, client_id) "
                "VALUES ('session-1', '/vault', 'bidirectional', 'ask', "
                "'completed', 'c')"
            )
            connection.execute(
                "INSERT INTO sync_conflicts(session_id, note_id, file_path, "
                "conflict_type, losing_side, losing_content) VALUES "
                "('session-1', ?, 'p.md', 'both_changed', 'db', 'plutonium v1')",
                (self.note_id,),
            )
            connection.execute(
                "INSERT INTO note_organization_receipts(receipt_id, note_id, "
                "requested_keywords_json, note_version, organization_version, "
                "state, created_at, updated_at) VALUES ('receipt-1', ?, '[]', 1, "
                "?, 'pending_organization', 't', 't')",
                (self.note_id, _HEX),
            )
            connection.execute(
                "INSERT INTO note_sync_publication_intents(intent_id, "
                "server_profile_id, dataset_id, note_id, operation, entity_version, "
                "request_fingerprint, payload_json, created_at) VALUES "
                "('intent-1', 'p', 'd', ?, 'update', 2, ?, "
                "'{\"content\": \"plutonium\"}', 't')",
                (self.note_id, _HEX),
            )
            connection.execute(
                "INSERT INTO research_quick_note_owner_proofs(note_id, owner_proof) "
                "VALUES (?, ?)",
                (self.note_id, _HEX),
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def traces(self, connection: sqlite3.Connection) -> dict[str, int]:
        return note_traces(
            connection, self.note_id, rowid=self.rowid, tokens=_TOKENS
        )


def test_delete_forever_leaves_no_row_that_names_the_note(tmp_path) -> None:
    """AC#9: no search, keyword, folder, version or sync-log trace survives.

    The precondition reads the same census: every discovered note-id column
    holds the note before the purge, and the sync log names it (its own
    tombstone and the two keyword links).
    """
    seed = _Purgeable(tmp_path)
    with seed.connect() as connection:
        before = seed.traces(connection)
        unfilled = [
            f"{table}.{column}"
            for table, column in note_id_columns(connection)
            if f"{table}.{column}" not in before
        ]
        assert not unfilled, (
            "a discovered note-id column the fixture does not fill -- teach "
            f"the fixture (and hard_delete_note) about it: {unfilled}"
        )
        assert before["sync_log.entity_id"] >= 3, before
        assert before["sync_log.payload"] >= 3, before

    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    with seed.connect() as connection:
        assert seed.traces(connection) == {}
        # Search and keywords, through the store's own readers.
        assert seed.db.search_notes("plutonium") == []
        assert seed.note_id not in {
            row["id"] for row in seed.db.search_notes("zeppelinward")
        }
        assert seed.db.get_keywords_for_note(seed.note_id) == []
        assert seed.db.get_note_by_id(seed.note_id) is None
        # What is not the note's survives: the other notes, their keyword
        # link, the shared keywords, the folder and its other placement.
        survivors = {row[0] for row in connection.execute("SELECT id FROM notes")}
        assert survivors == {seed.keeper, seed.linker}
        assert [row["keyword"] for row in seed.db.get_keywords_for_note(seed.keeper)] == [
            "alpha"
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM keywords WHERE deleted = 0"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ?",
            (seed.keeper,),
        ).fetchone()[0] == 1
    assert [row["id"] for row in seed.db.search_notes("survive")] == [seed.keeper]


def test_a_note_referencing_table_the_purge_does_not_know_is_caught(
    tmp_path,
) -> None:
    """Negative control: the census finds a table it was never told about.

    A table added after this purge was written, holding the note's id, is
    discovered by its column name and its leftover row is reported -- the
    failure a future schema change would see here.
    """
    seed = _Purgeable(tmp_path)
    with seed.connect() as connection:
        connection.execute(
            "CREATE TABLE note_review_flags(flagged_note_id TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO note_review_flags VALUES (?)", (seed.note_id,))
        assert ("note_review_flags", "flagged_note_id") in note_id_columns(connection)

    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    with seed.connect() as connection:
        assert seed.traces(connection) == {"note_review_flags.flagged_note_id": 1}


def test_delete_forever_refuses_a_live_note(tmp_path) -> None:
    """A live note is never purged: typed refusal, nothing written."""
    seed = _Purgeable(tmp_path)
    live = seed.db.get_note_by_id(seed.keeper)
    with seed.connect() as connection:
        before = note_traces(connection, seed.keeper)

    with pytest.raises(NoteNotInTrashError) as refused:
        seed.db.hard_delete_note(seed.keeper, expected_version=live["version"])

    assert isinstance(refused.value, ConflictError)
    assert refused.value.entity_id == seed.keeper
    with seed.connect() as connection:
        assert note_traces(connection, seed.keeper) == before
    assert seed.db.get_note_by_id(seed.keeper)["deleted"] == 0


def test_delete_forever_refuses_a_tombstone_that_changed_since_it_was_shown(
    tmp_path,
) -> None:
    """The confirm named one version of the note; a different one is kept."""
    seed = _Purgeable(tmp_path)

    with pytest.raises(ConflictError):
        seed.db.hard_delete_note(seed.note_id, expected_version=seed.version - 1)

    with seed.connect() as connection:
        assert "notes.id" in seed.traces(connection)


def test_delete_forever_of_a_note_already_gone_reports_nothing_deleted(
    tmp_path,
) -> None:
    seed = _Purgeable(tmp_path)
    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    assert seed.db.hard_delete_note(seed.note_id) is False


def test_delete_forever_is_one_transaction(tmp_path) -> None:
    """A failure at the last statement leaves every row and the index intact."""
    seed = _Purgeable(tmp_path)
    with seed.connect() as connection:
        before = seed.traces(connection)
    connection = seed.db.get_connection()
    connection.execute(
        "CREATE TEMP TRIGGER purge_aborts BEFORE DELETE ON main.notes "
        "BEGIN SELECT RAISE(ABORT, 'injected purge failure'); END"
    )
    try:
        with pytest.raises(Exception, match="injected purge failure"):
            seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)
    finally:
        connection.execute("DROP TRIGGER IF EXISTS temp.purge_aborts")

    with seed.connect() as connection:
        assert seed.traces(connection) == before
        # The rolled-back FTS re-insert left nothing behind either.
        assert "notes_fts:integrity-check" not in before
        assert connection.execute(
            "SELECT COUNT(*) FROM notes_fts_docsize WHERE id = ?", (seed.rowid,)
        ).fetchone()[0] == 0
