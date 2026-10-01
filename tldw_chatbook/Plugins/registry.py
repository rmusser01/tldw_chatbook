"""Private, versioned plugin metadata. Stored intent never constitutes trust."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Self

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

if TYPE_CHECKING:
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

SCHEMA_VERSION = 1
PAGE_SIZE = 50
SELECT_INSTALLATIONS = """
SELECT installation_id, revision_digest, activation_default
FROM installations ORDER BY installation_id LIMIT ? OFFSET ?
"""


def validate_page(limit: int, offset: int) -> None:
    """Reject unbounded/invalid page requests before reaching SQLite."""
    if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
        raise ValueError("limit must be between 1 and 50")
    if type(offset) is not int or not 0 <= offset <= 2**63 - 1:
        raise ValueError("offset must be a nonnegative SQLite integer")


def _schema_statements() -> tuple[str, ...]:
    text = (
        files("tldw_chatbook.Plugins")
        .joinpath("migrations/001_initial.sql")
        .read_text(encoding="utf-8")
    )
    statements, pending = [], ""
    for line in text.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending)
            pending = ""
    if pending.strip():
        raise sqlite3.DatabaseError("incomplete plugin migration")
    return tuple(statements)


class PluginRegistry:
    """Own a synchronous SQLite handle; disk writes require a live runtime owner.

    Args:
        path: Registry file under the resolved plugin root, or ``:memory:``.
        owner: Acquired owner of this exact root, required for disk mutation.
        read_only: Explicitly restrict an owner-backed view to browsing.
    """

    def __init__(
        self,
        path: Path,
        *,
        owner: PluginRuntimeOwner | None = None,
        read_only: bool | None = None,
    ) -> None:
        self._memory = str(path) == ":memory:"
        self.path = path if self._memory else path.parent.resolve() / path.name
        self._owner = owner
        self.read_only = (
            (not self._memory and owner is None) if read_only is None else read_only
        )
        self._closed = False
        if not self.read_only:
            self._require_write()
        self._connection = connect_private_sqlite(
            "plugins.registry",
            self.path,
            read_only=self.read_only,
            must_exist=self.read_only,
            isolation_level=None,
            # The body authorizer must run on every execution, including reuse
            # after SQLite automatically rolls back a failed statement.
            cached_statements=0,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys=ON")
            if not self.read_only:
                mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode != ("memory" if self._memory else "wal"):
                    raise sqlite3.DatabaseError("plugin WAL unavailable")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA fullfsync=ON")
                if self._connection.execute("PRAGMA synchronous").fetchone()[0] != 2:
                    raise sqlite3.DatabaseError("plugin durable commit unavailable")
                if self._connection.execute("PRAGMA fullfsync").fetchone()[0] != 1:
                    raise sqlite3.DatabaseError("plugin fullfsync unavailable")
            self._initialize()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _require_write(self) -> None:
        if self._closed or self.read_only:
            raise PermissionError("plugin registry is read only or closed")
        if not self._memory:
            if self._owner is None:
                raise PermissionError("plugin mutation requires runtime ownership")
            self._owner.require_owner(self.path.parent)

    def _initialize(self) -> None:
        connection = self._connection
        connection.execute("BEGIN" if self.read_only else "BEGIN IMMEDIATE")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            objects = connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
            if version == 0 and not objects and not self.read_only:
                for statement in _schema_statements():
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=1")
            elif version != SCHEMA_VERSION:
                raise sqlite3.DatabaseError("unsupported plugin registry schema")
            # Exact DDL validation catches missing constraints/columns/triggers, not
            # merely a forged user_version. The reference has no filesystem owner.
            with PluginRegistry._reference_schema() as reference:
                expected = [
                    tuple(row)
                    for row in reference.execute(
                        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                    )
                ]
            actual = [
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            ]
            if (
                actual != expected
                or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            ):
                raise sqlite3.DatabaseError("invalid plugin registry schema")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise sqlite3.DatabaseError("invalid plugin registry references")
            if not self.read_only:
                self._require_write()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    @contextmanager
    def _reference_schema() -> Iterator[sqlite3.Connection]:
        # Reuse the registered memory seam for schema validation as well.
        connection = connect_private_sqlite("plugins.registry", ":memory:")
        try:
            for statement in _schema_statements():
                connection.execute(statement)
            yield connection
        finally:
            connection.close()

    @property
    def schema_version(self) -> int:
        """Return the schema version of this validated view."""
        return self._connection.execute("PRAGMA user_version").fetchone()[0]

    def _authorize_transaction_body(self, action: int, *_args: object) -> int:
        # executescript first requests COMMIT, even before parsing the script.
        # Deny it, explicit transaction control, and any work following SQLite's
        # automatic rollback (for example a caught INSERT OR ROLLBACK failure).
        if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
            return sqlite3.SQLITE_DENY
        if not self._connection.in_transaction:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Commit atomically and durably before returning; rollback on any failure.

        A single worker must serialize access. Nested transactions are refused.
        Body statements cannot commit/rollback/savepoint or execute scripts.
        This low-level service seam is for trusted coordinator code, not plugins.
        """
        self._require_write()
        if self._connection.in_transaction:
            raise RuntimeError("nested plugin transaction")
        self._connection.execute("BEGIN IMMEDIATE")
        cursor = self._connection.cursor()
        try:
            self._connection.set_authorizer(self._authorize_transaction_body)
            try:
                yield cursor
                self._require_write()
                if not self._connection.in_transaction:
                    raise RuntimeError("plugin transaction ended inside its body")
            finally:
                # Only this wrapper may settle the transaction, after ownership
                # validation. Remove the body guard for its commit or rollback.
                self._connection.set_authorizer(None)
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def list_installations(self, *, limit: int, offset: int) -> tuple[dict, ...]:
        """Return a bounded, deterministic page of untrusted installation intent."""
        validate_page(limit, offset)
        return tuple(
            dict(row)
            for row in self._connection.execute(SELECT_INSTALLATIONS, (limit, offset))
        )

    def close(self) -> None:
        """Release the database handle without deleting recovery evidence."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
