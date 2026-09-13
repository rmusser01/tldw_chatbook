"""ChaChaNotes v72 -> v73: the persisted note-link relation (task-32186).

Backlinks used to be answered by a leading-wildcard ``LIKE`` over
``notes.content`` -- every active body read and sorted on every note open.
V73 persists the relation instead, so the migration has to do two things a
plain ``CREATE TABLE`` does not: build the table, and backfill it from the
bodies an existing database already holds, so an upgrade picks its backlinks
up without the user re-importing anything (task AC #4).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _drop_the_bootstrap_scaffold(db: CharactersRAGDB) -> None:
    """Undo ``_add_forward_write_dependencies`` so V73 genuinely creates it.

    The bootstrap gives a pre-V73 database a bare ``note_links`` so today's
    ``add_note`` can run against it at all. Leaving it in place would make
    this file assert against the scaffold instead of the migration, which is
    the escape ``historical_bootstrap``'s header warns about: the fixture that
    pins a step must drop the artifact the step declares.
    """
    with db.transaction() as cursor:
        cursor.execute("DROP TABLE IF EXISTS note_links")


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def test_fresh_database_ships_the_note_link_relation(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh")
    try:
        connection = db.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert "note_links" in _table_names(connection)
        assert {
            str(row[1])
            for row in connection.execute("PRAGMA index_list('note_links')")
        } >= {"idx_note_links_target"}
    finally:
        db.close_connection()


def test_v72_database_backfills_its_existing_bodies(tmp_path: Path) -> None:
    """An existing vault gets its backlinks without re-importing (AC #4)."""
    path = tmp_path / "genuine-v72.sqlite"
    with chachanotes_db_at_version(path, 72, client_id="v72-fixture") as historical:
        connection = historical.get_connection()
        assert _version(connection) == 72
        target = historical.add_note("Zettelkasten", "hub")
        linker = historical.add_note(
            "Library review", f"See [[Zettelkasten|hub]](note://{target}) for it."
        )
        plain = historical.add_note("Unrelated", "no links here")
        # A soft-deleted body is still a real link; the Trash view can restore
        # the note, and its backlink must come back with it.
        doomed = historical.add_note("Doomed", f"[t](note://{target})")
        detail = historical.get_note_by_id(doomed)
        assert historical.soft_delete_note(doomed, detail["version"]) is True
        _drop_the_bootstrap_scaffold(historical)
        assert "note_links" not in _table_names(connection)

    upgraded = CharactersRAGDB(str(path), client_id="upgraded")
    try:
        connection = upgraded.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        rows = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT source_note_id, target_note_id FROM note_links"
            )
        }
        assert rows == {(linker, target), (doomed, target)}
        assert plain not in {source for source, _ in rows}
        assert [row["id"] for row in upgraded.get_notes_linking_to(target)] == [linker]
    finally:
        upgraded.close_connection()


def test_backfill_skips_a_self_link_and_keeps_an_unresolved_target(
    tmp_path: Path,
) -> None:
    """The backfill records what the body says, minus links to the note itself.

    A target id that no longer names a note is kept on purpose: nothing can
    ask for its backlinks, and dropping it would make the relation disagree
    with the body it came from the moment the target is restored.
    """
    path = tmp_path / "edge-v72.sqlite"
    with chachanotes_db_at_version(path, 72, client_id="v72-edge") as historical:
        looper = historical.add_note("Self", "placeholder")
        historical.update_note(
            looper,
            {"content": f"I link to [me](note://{looper}) and [gone](note://ghost-id)"},
            expected_version=1,
        )
        _drop_the_bootstrap_scaffold(historical)

    upgraded = CharactersRAGDB(str(path), client_id="upgraded-edge")
    try:
        rows = {
            (str(row[0]), str(row[1]))
            for row in upgraded.get_connection().execute(
                "SELECT source_note_id, target_note_id FROM note_links"
            )
        }
        assert rows == {(looper, "ghost-id")}
    finally:
        upgraded.close_connection()


def test_backfill_streams_a_vault_larger_than_one_batch(tmp_path: Path) -> None:
    """The backfill reads in ``fetchmany(500)`` batches on a second cursor and
    writes each with ``executemany`` on the migration cursor. 502 notes puts
    edges in both the full first batch and the short second one, so a loop
    that stops after one batch, or a read cursor the writes disturb, loses
    rows here.
    """
    path = tmp_path / "big-v72.sqlite"
    with chachanotes_db_at_version(path, 72, client_id="v72-big") as historical:
        target = historical.add_note("Hub", "hub")
        linkers = {
            historical.add_note(f"Linker {index:03d}", f"[h](note://{target})")
            for index in range(501)
        }
        _drop_the_bootstrap_scaffold(historical)

    upgraded = CharactersRAGDB(str(path), client_id="upgraded-big")
    try:
        connection = upgraded.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 502
        sources = {
            str(row[0])
            for row in connection.execute(
                "SELECT source_note_id FROM note_links WHERE target_note_id = ?",
                (target,),
            )
        }
        assert sources == linkers
        assert len(sources) == 501
    finally:
        upgraded.close_connection()
