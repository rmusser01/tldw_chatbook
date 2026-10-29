"""Where can a note still be found? Read off the LIVE ChaChaNotes schema.

TASK-34000.29 (N-29, part 8b; ruling T8b-b): the purge-completeness check
keeps NO table list of its own. Every place it looks is discovered from
``sqlite_master``, ``PRAGMA table_info`` and ``PRAGMA foreign_key_list`` on
the database under test, so a table added later that holds a note id is
checked the day it lands, and a purge that forgets it fails here.

Three discoveries, one count:

- **Note-id columns.** A column named ``note_id`` or ``*_note_id``, or the
  child column of any foreign key to ``notes`` -- plus ``notes.id`` itself.
- **Every text column of every ordinary table**, swept for the id as a
  substring: catches the id inside a payload or a composite key that no
  column name announces (``sync_log``'s ``note_keywords`` rows carry it as
  ``entity_id='<id>_<keyword>'`` and inside the JSON payload).
- **The FTS index over notes** (an external-content FTS5 table whose
  ``content`` is ``notes``): its ``_docsize`` row for the note's rowid, the
  index's own instances for that rowid (``fts5vocab``), FTS5's integrity
  check, the index's row total against the live notes it should hold, and
  ``MATCH`` on tokens only the note had.

Used by the DB-level test (``test_note_hard_delete.py``) and by the Pilot
that purges through the real Library (``Tests/UI/test_library_notes_trash_
pages_and_purges_extended.py``), each through a second plain ``sqlite3``
connection -- the store's own connection never vouches for itself.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

_NOTE_ID_COLUMN = re.compile(r"(?:^|_)note_id$")
_NOTES_CONTENT = re.compile(r"content\s*=\s*['\"]?notes['\"]?", re.IGNORECASE)


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (str(name), str(sql or ""))
        for name, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


def _virtual_tables(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (name, sql)
        for name, sql in _tables(connection)
        if sql.upper().startswith("CREATE VIRTUAL TABLE")
    ]


def note_id_columns(connection: sqlite3.Connection) -> tuple[tuple[str, str], ...]:
    """Every ``(table, column)`` that can hold a note id, from the live schema."""
    found: set[tuple[str, str]] = {("notes", "id")}
    for table, _sql in _tables(connection):
        for column in connection.execute(f"PRAGMA table_info({_quoted(table)})"):
            if _NOTE_ID_COLUMN.search(str(column[1])):
                found.add((table, str(column[1])))
        for foreign_key in connection.execute(
            f"PRAGMA foreign_key_list({_quoted(table)})"
        ):
            if str(foreign_key[2]) == "notes":
                found.add((table, str(foreign_key[3])))
    return tuple(sorted(found))


def notes_fts_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    """The external-content FTS tables whose content table is ``notes``."""
    return tuple(
        name for name, sql in _virtual_tables(connection) if _NOTES_CONTENT.search(sql)
    )


def _swept_tables(connection: sqlite3.Connection) -> list[str]:
    """Ordinary tables: not virtual, and not a virtual table's shadow."""
    virtual = [name for name, _sql in _virtual_tables(connection)]
    return [
        name
        for name, sql in _tables(connection)
        if not sql.upper().startswith("CREATE VIRTUAL TABLE")
        and not any(name.startswith(f"{owner}_") for owner in virtual)
    ]


def _sqlite_varints(blob: bytes) -> list[int]:
    """Decode SQLite/FTS5 varints (big-endian 7-bit groups, 9th byte whole)."""
    values: list[int] = []
    position = 0
    while position < len(blob):
        value = 0
        for index in range(9):
            byte = blob[position]
            position += 1
            if index == 8:
                value = (value << 8) | byte
                break
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        values.append(value)
    return values


def _fts_row_total(connection: sqlite3.Connection, fts: str) -> int | None:
    """The FTS5 index's own row count: the first value of its averages record."""
    row = connection.execute(
        f"SELECT block FROM {_quoted(fts + '_data')} WHERE id = 1"
    ).fetchone()
    if row is None or not row[0]:
        return 0
    values = _sqlite_varints(bytes(row[0]))
    return values[0] if values else None


def value_traces(
    connection: sqlite3.Connection, needles: Iterable[str]
) -> dict[str, int]:
    """Count, per ``table.column`` and needle, the rows whose value holds it.

    Review R0, m-3: a token scan, not a column-name scan. EVERY table in
    ``sqlite_master`` -- ordinary, virtual and FTS shadow tables alike -- and
    EVERY column of it (``PRAGMA table_info``), whatever its declared type:
    each value is compared as bytes (``CAST(... AS BLOB)``), so TEXT, JSON
    and BLOB columns are all searched, an id inside a JSON payload or a
    composite key included.

    Args:
        connection: A plain ``sqlite3`` connection to the database file.
        needles: Strings to look for (a note id, words only the note had).

    Returns:
        ``{"<table>.<column>:<needle>": rows}`` for every non-zero finding.
    """
    found: dict[str, int] = {}
    wanted = [needle for needle in needles if needle]
    for table, _sql in _tables(connection):
        columns = [
            str(column[1])
            for column in connection.execute(f"PRAGMA table_info({_quoted(table)})")
        ]
        for column in columns:
            for needle in wanted:
                count = connection.execute(
                    f"SELECT COUNT(*) FROM {_quoted(table)} WHERE "
                    f"instr(CAST({_quoted(column)} AS BLOB), CAST(? AS BLOB)) > 0",
                    (needle,),
                ).fetchone()[0]
                if count:
                    found[f"{table}.{column}:{needle}"] = int(count)
    return found


def note_traces(
    connection: sqlite3.Connection,
    note_id: str,
    *,
    rowid: int | None = None,
    tokens: Iterable[str] = (),
) -> dict[str, int]:
    """Count every place ``note_id`` can still be found; ``{}`` is no trace.

    Args:
        connection: A plain ``sqlite3`` connection to the database file.
        note_id: The note's id.
        rowid: The note's former ``notes`` rowid, for the FTS checks.
        tokens: Words only the note had, matched against each notes FTS table.

    Returns:
        ``{"<where>": count}`` for every non-zero finding.
    """
    traces: dict[str, int] = {}
    for table, column in note_id_columns(connection):
        count = connection.execute(
            f"SELECT COUNT(*) FROM {_quoted(table)} WHERE {_quoted(column)} = ?",
            (note_id,),
        ).fetchone()[0]
        if count:
            traces[f"{table}.{column}"] = int(count)
    for table in _swept_tables(connection):
        for column in connection.execute(f"PRAGMA table_info({_quoted(table)})"):
            name = str(column[1])
            count = connection.execute(
                f"SELECT COUNT(*) FROM {_quoted(table)} "
                f"WHERE instr(CAST({_quoted(name)} AS TEXT), ?) > 0",
                (note_id,),
            ).fetchone()[0]
            if count:
                traces.setdefault(f"{table}.{name}", int(count))
    for fts in notes_fts_tables(connection):
        docsize = f"{fts}_docsize"
        if rowid is not None and connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (docsize,)
        ).fetchone():
            count = connection.execute(
                f"SELECT COUNT(*) FROM {_quoted(docsize)} WHERE id = ?", (rowid,)
            ).fetchone()[0]
            if count:
                traces[f"{docsize}.id"] = int(count)
        if rowid is not None:
            vocab = f"note_trace_{fts}_instances"
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.{_quoted(vocab)} "
                f"USING fts5vocab(main, {_quoted(fts)}, 'instance')"
            )
            try:
                count = connection.execute(
                    f"SELECT COUNT(*) FROM temp.{_quoted(vocab)} WHERE doc = ?",
                    (rowid,),
                ).fetchone()[0]
            finally:
                connection.execute(f"DROP TABLE IF EXISTS temp.{_quoted(vocab)}")
            if count:
                traces[f"{fts}:instances"] = int(count)
        try:
            # The index's own structure. ``rank = 1`` (the index against the
            # content table) only when the content table holds no tombstone:
            # the notes index deliberately leaves them out, so that
            # comparison fails whenever any note is in Recently deleted.
            tombstones = connection.execute(
                "SELECT COUNT(*) FROM notes WHERE deleted <> 0"
            ).fetchone()[0]
            connection.execute(
                f"INSERT INTO {_quoted(fts)}({_quoted(fts)}, rank) "
                "VALUES('integrity-check', ?)",
                (0 if tombstones else 1,),
            )
        except sqlite3.DatabaseError:
            traces[f"{fts}:integrity-check"] = 1
        # The index's row total against the rows it holds and the live notes
        # it should hold. An FTS 'delete' issued for a row the index never
        # held (a purged tombstone, unguarded) decrements this total without
        # touching the rest -- measured: the structural check above passes.
        totals = _fts_row_total(connection, fts)
        indexed = connection.execute(
            f"SELECT COUNT(*) FROM {_quoted(fts + '_docsize')}"
        ).fetchone()[0]
        live = connection.execute(
            "SELECT COUNT(*) FROM notes WHERE deleted = 0"
        ).fetchone()[0]
        if not (totals == indexed == live):
            traces[f"{fts}:totals"] = int(totals if totals is not None else -1)
        for token in tokens:
            try:
                count = connection.execute(
                    f"SELECT COUNT(*) FROM {_quoted(fts)} WHERE {_quoted(fts)} MATCH ?",
                    ('"' + token.replace('"', '""') + '"',),
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                traces[f"{fts}:match:{token}"] = -1
                continue
            if count:
                traces[f"{fts}:match:{token}"] = int(count)
    return traces
