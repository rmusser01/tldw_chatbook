"""Per-object Sync v2 mirror for whole-object domains (notes.note in P2).

Stores the last server-acknowledged revision/hash/cursor per object so the builder
can fill base_object_revision/base_object_hash on updates and tombstones, and the
applier can recognise already-applied envelopes.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MirrorRecord:
    object_revision: int
    object_hash: str
    server_cursor: int


class NotesMirror:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes_object_mirror (
                dataset_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_revision INTEGER NOT NULL,
                object_hash TEXT NOT NULL,
                server_cursor INTEGER NOT NULL,
                PRIMARY KEY (dataset_id, object_id)
            )
            """
        )
        self._conn.commit()

    def record(
        self,
        dataset_id: str,
        object_id: str,
        *,
        object_revision: int,
        object_hash: str,
        server_cursor: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO notes_object_mirror
                (dataset_id, object_id, object_revision, object_hash, server_cursor)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, object_id) DO UPDATE SET
                object_revision=excluded.object_revision,
                object_hash=excluded.object_hash,
                server_cursor=excluded.server_cursor
            """,
            (dataset_id, object_id, object_revision, object_hash, server_cursor),
        )
        self._conn.commit()

    def get(self, dataset_id: str, object_id: str) -> MirrorRecord | None:
        row = self._conn.execute(
            "SELECT object_revision, object_hash, server_cursor FROM notes_object_mirror "
            "WHERE dataset_id=? AND object_id=?",
            (dataset_id, object_id),
        ).fetchone()
        if row is None:
            return None
        return MirrorRecord(
            object_revision=row["object_revision"],
            object_hash=row["object_hash"],
            server_cursor=row["server_cursor"],
        )

    def close(self) -> None:
        self._conn.close()
