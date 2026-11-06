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

import json
import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.note_trace_census import (
    note_id_columns,
    note_traces,
    value_traces,
)
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
#: Review R0, m-3: a title word, a body word and a keyword only this note has.
_TITLE_TOKEN = "zeppelinward"
_CONTENT_TOKEN = "plutonium"
_KEYWORD_TOKEN = "quokkatag"
#: The agent receipt the soft delete cancels into a replay guard (m-1).
_GUARD_RECEIPT = "receipt-guard-1"
#: Review R0, m-2: what a purge KEEPS on purpose, by table.column -- Sync v2
#: organization state that by its v58 contract holds logical ids only, never
#: a title or text (``chachanotes_v57_to_v58_notes_organization_sync.sql``).
#: A pending link intent must still reach the server, in order; a head is the
#: server's acknowledged state a later link tombstone builds on.
_ID_ONLY_TOMBSTONES = {
    "notes_organization_sync_intents.payload_json": 1,
    "notes_organization_heads.payload_json": 1,
}


class _Purgeable:
    """A soft-deleted note referenced from every discovered note-id column.

    Args:
        tmp_path: Directory for the database file.
        backlinked: Whether another note's body links to this one by title.
            The token scan (review R0, m-3) turns it off: that note's own
            words would otherwise spell the purged title.
    """

    def __init__(self, tmp_path: Path, *, backlinked: bool = True) -> None:
        self.path = tmp_path / "notes.sqlite3"
        self.db = CharactersRAGDB(self.path, client_id="task-34000-29-8b")
        db = self.db
        self.keeper = db.add_note("Keeper note", "shared words survive here")
        self.note_id = db.add_note(
            "Purge zeppelinward", "plutonium secret body links [[Keeper note]]"
        )
        self.linker = (
            db.add_note("Backlinker", "see [[Purge zeppelinward]]")
            if backlinked
            else None
        )
        self.keyword_ids = [db.add_keyword(word) for word in ("alpha", _KEYWORD_TOKEN)]
        for keyword_id in self.keyword_ids:
            db.link_note_to_keyword(self.note_id, keyword_id)
        db.link_note_to_keyword(self.keeper, self.keyword_ids[0])
        folders = LocalNoteFolderRepository(db)
        folder = folders.create_folder(name="Projects", parent_id=None)
        folders.attach_manual(folder_id=folder.folder_id, note_id=self.note_id)
        folders.attach_manual(folder_id=folder.folder_id, note_id=self.keeper)
        with sqlite3.connect(self.path) as connection:
            # Review R0, m-1: an agent's organization receipt the delete
            # below cancels, turning it into the replay guard -- a
            # content-free publication intent under the receipt's id.
            connection.execute(
                "INSERT INTO note_organization_receipts(receipt_id, note_id, "
                "requested_keywords_json, note_version, organization_version, "
                "state, created_at, updated_at) VALUES (?, ?, '[]', 1, ?, "
                "'pending_organization', 't', 't')",
                (_GUARD_RECEIPT, self.note_id, _HEX),
            )
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
            # Review R0, m-2: Sync v2 organization state naming the note --
            # a pending keyword-link intent and its acknowledged head. By the
            # v58 contract these hold logical ids only, never the note's text.
            link = json.dumps(
                {
                    "subject_type": "note",
                    "subject_id": self.note_id,
                    "keyword_sync_id": "kw-sync-1",
                }
            )
            connection.execute(
                "INSERT INTO notes_organization_sync_intents(intent_id, "
                "intent_sequence, server_profile_id, dataset_id, domain, "
                "object_id, operation, schema_version, encryption_policy, "
                "payload_json, payload_hash, source_version, created_at) VALUES "
                "('org-intent-1', 1, 'p', 'd', 'notes.keyword_link', 'link-1', "
                "'upsert', 1, 'server_trusted_v1', ?, ?, 1, 't')",
                (link, _HEX),
            )
            connection.execute(
                "INSERT INTO notes_organization_heads(server_profile_id, dataset_id, "
                "domain, object_id, operation, schema_version, encryption_policy, "
                "payload_json, payload_hash, object_revision, object_hash, "
                "server_cursor, deleted, apply_state, applied_at, updated_at) "
                "VALUES ('p', 'd', 'notes.keyword_link', 'link-1', 'upsert', 1, "
                "'server_trusted_v1', ?, ?, 1, ?, 'c1', 0, 'applied', 't', 't')",
                (link, _HEX, _HEX),
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
        assert seed.traces(connection) == _ID_ONLY_TOMBSTONES
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
        assert seed.traces(connection) == {
            **_ID_ONLY_TOMBSTONES,
            "note_review_flags.flagged_note_id": 1,
        }


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


# -- review R0: the replay guard (m-1) and the token scan (m-3) --------------


def _needles(note_id: str) -> tuple[str, ...]:
    """The note's id -- bare, and as a JSON string -- and its three words."""
    return (note_id, json.dumps(note_id), _TITLE_TOKEN, _CONTENT_TOKEN, _KEYWORD_TOKEN)


def _the_tag_itself(where: str) -> bool:
    """The keyword is its own entity, shared with other notes.

    Delete forever removes the NOTE's link to it (``note_keywords`` and that
    link's sync-log rows) and never the tag: the word stays in the keyword's
    own row, its index and its own sync-log entry -- nowhere else.
    """
    table, _, rest = where.partition(".")
    return rest.endswith(f":{_KEYWORD_TOKEN}") and (
        table == "keywords" or table.startswith("keywords_fts") or table == "sync_log"
    )


def test_a_token_scan_of_every_column_finds_only_id_only_tombstones(tmp_path) -> None:
    """Review R0, m-3: EVERY column of EVERY table, as bytes, after a purge.

    The note's id (bare and as JSON), a word of its title, a word of its body
    and its own keyword: before the purge each is found; afterwards nothing
    is, except the id-only Sync v2 organization rows the purge keeps (m-2)
    and the tag's own rows. The FTS index's own blocks are scanned too --
    an FTS ``'delete'`` leaves a word's bytes in older segments until the
    purge's ``'optimize'`` merges them.
    """
    seed = _Purgeable(tmp_path, backlinked=False)
    needles = _needles(seed.note_id)
    with seed.connect() as connection:
        before = value_traces(connection, needles)
    for needle in needles:
        assert any(where.endswith(f":{needle}") for where in before), (needle, before)

    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    with seed.connect() as connection:
        after = value_traces(connection, needles)
        tag_rows = connection.execute(
            "SELECT entity FROM sync_log WHERE instr(payload, ?) > 0",
            (_KEYWORD_TOKEN,),
        ).fetchall()
    kept = {where: rows for where, rows in after.items() if not _the_tag_itself(where)}
    assert kept == {
        f"{where}:{needle}": rows
        for where, rows in _ID_ONLY_TOMBSTONES.items()
        for needle in (seed.note_id, json.dumps(seed.note_id))
    }, kept
    assert {str(row[0]) for row in tag_rows} <= {"keywords"}, tag_rows
    assert f"keywords.keyword:{_KEYWORD_TOKEN}" in after


def test_the_token_scan_catches_a_title_in_a_json_column_it_was_never_told_about(
    tmp_path,
) -> None:
    """Negative control: a new table holding the title inside JSON is found."""
    seed = _Purgeable(tmp_path, backlinked=False)
    with seed.connect() as connection:
        connection.execute("CREATE TABLE note_export_cache(meta_json TEXT)")
        connection.execute(
            "INSERT INTO note_export_cache VALUES (?)",
            (json.dumps({"title": "Purge zeppelinward", "words": 3}),),
        )

    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    with seed.connect() as connection:
        after = value_traces(connection, _needles(seed.note_id))
    assert after.get(f"note_export_cache.meta_json:{_TITLE_TOKEN}") == 1, after


def test_a_purge_keeps_the_cancelled_receipt_unreplayable(tmp_path) -> None:
    """Review R0, m-1: "permanently non-replayable" survives Delete forever.

    The soft delete cancelled the agent's receipt into a content-free
    publication intent. The purge keeps it as an id-only tombstone, so a
    replayed ``save_note`` with that receipt id is still refused -- never
    answered by creating the note again from the agent's request.
    """
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.note_folder_models import NotesOrganizationRepositoryError

    seed = _Purgeable(tmp_path, backlinked=False)
    interop = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="t8b-m1", global_db_to_use=seed.db
    )
    assert seed.db.hard_delete_note(seed.note_id, expected_version=seed.version)

    with pytest.raises(NotesOrganizationRepositoryError) as refused:
        interop.save_note_with_organization(
            "user-1",
            title="Purge zeppelinward",
            content="plutonium secret body",
            receipt_id=_GUARD_RECEIPT,
        )

    assert "cancelled" in str(refused.value)
    with seed.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM notes WHERE title = 'Purge zeppelinward'"
        ).fetchone()[0] == 0
        guard = connection.execute(
            "SELECT note_id, payload_json, cancelled_at FROM "
            "note_sync_publication_intents WHERE intent_id = ?",
            (_GUARD_RECEIPT,),
        ).fetchone()
        after = value_traces(connection, _needles(seed.note_id))
    assert guard[0] == f"purged:{_GUARD_RECEIPT}" and guard[1] == "{}" and guard[2]
    assert not [where for where in after if where.startswith("note_sync_publication")]


def test_note_ids_present_names_live_and_soft_deleted_rows_but_not_a_purged_one(
    tmp_path,
) -> None:
    """Final review C-1: the Sync v2 push's existence seam, one statement.

    A live note and a soft-deleted note have a row; a note deleted for good
    has none, and an id the database never held is absent too. Blank ids
    are ignored, and the deleted-or-not contrast with ``live_note_ids`` is
    pinned beside it so the two never drift into one another.
    """
    seed = _Purgeable(tmp_path, backlinked=False)
    db = seed.db
    trash = db.add_note("Trash note", "kept for now")
    db.soft_delete_note(trash, db.get_note_by_id(trash)["version"])
    assert db.hard_delete_note(seed.note_id, expected_version=seed.version) is True

    asked = [seed.keeper, trash, seed.note_id, "never-existed", ""]
    assert db.note_ids_present(asked) == {seed.keeper, trash}
    assert db.live_note_ids(asked) == {seed.keeper}
    assert db.note_ids_present([]) == set()
