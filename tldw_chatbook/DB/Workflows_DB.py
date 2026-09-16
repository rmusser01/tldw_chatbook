"""Private SQLite lifecycle and serialized authoring transactions."""
# ruff: noqa: N999 -- DB module name follows the repository convention.

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


class WorkflowsDB:
    """Own one authoring connection, with no execution authority.

    The caller supplies an explicit private path or the exact :memory: token.
    Existing migrations retain file compatibility, including historical run
    tables; authoring never acquires runtime ownership or changes runtime rows.
    """

    SCHEMA_VERSION = 4

    def __init__(self, path: Path) -> None:
        self._lock = RLock()
        self._closed = False
        self._connection = connect_private_sqlite(
            "workflows.local", path, isolation_level=None, check_same_thread=False
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            with self.transaction() as cursor:
                version = cursor.execute("PRAGMA user_version").fetchone()[0]
                if version not in range(self.SCHEMA_VERSION + 1):
                    raise ValueError("Unsupported workflow database schema")
                for source_version in range(version, self.SCHEMA_VERSION):
                    migration = (
                        Path(__file__).with_name("migrations")
                        / f"workflows_v{source_version}_to_v{source_version + 1}.sql"
                    )
                    # executescript would commit before running the migration.
                    for statement in migration.read_text(encoding="utf-8").split(";"):
                        if statement.strip():
                            cursor.execute(statement)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Cursor]:
        """Hold the connection lock through commit/rollback and cursor closure.

        Args:
            write: Reserve the writer before reading mutable state. False
                provides a consistent read snapshot without a writer lock.

        Raises:
            RuntimeError: The store is closed or the caller nests transactions.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Workflow database is closed")
            if self._connection.in_transaction:
                raise RuntimeError("Workflow transactions cannot be nested")
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield cursor
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def close(self) -> None:
        """Drain other threads' transactions and close the owned connection."""
        with self._lock:
            if not self._closed:
                if self._connection.in_transaction:
                    raise RuntimeError("Cannot close inside a workflow transaction")
                self._connection.close()
                self._closed = True
