"""One installed ChaChaNotes ``.sql`` migration step, run and stamped atomically."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def run_sql_version_step(
    db: Any, conn: sqlite3.Connection, start: int, path: Path
) -> None:
    """Run ``path`` (the caller has checked the entry version) and move the stamp from ``start`` to ``start + 1`` in one transaction.

    Args:
        db: The ``CharactersRAGDB`` being migrated.
        conn: Its migration connection.
        start: The version this step requires on entry.
        path: The installed migration script.

    Raises:
        SchemaError: If a statement or the stamp fails; the
            transaction rolls the step back to its entry state.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError, SchemaError

    label = f"V{start}→V{start + 1}"
    try:
        with db.transaction() as cursor:
            db._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), label
            )
            updated = cursor.execute(
                "UPDATE db_schema_version SET version=? WHERE schema_name=? AND version=?",
                (start + 1, db._SCHEMA_NAME, start),
            )
            if updated.rowcount != 1:
                raise SchemaError(f"{label} version update failed")
    except (OSError, sqlite3.Error, CharactersRAGDBError) as exc:
        raise SchemaError(f"Migration {label} failed: {type(exc).__name__}") from exc
