# base_db.py
# Description: Base class for standardized database path handling
#
"""
base_db.py
----------

Base class that provides standardized path handling for all database modules.
This ensures consistent behavior across all DB classes for:
- Path type handling (str vs Path)
- Memory database special case (':memory:')
- Client ID handling
- Private file connection enforcement
"""

import sqlite3
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod
from loguru import logger

from .private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.private_paths import lexical_path


SEMANTIC_MUTATION_GUARD_FUNCTION = "console_semantic_mutation_authorized"


class _SemanticMutationAuthorization:
    """Connection-local authorization read by SQLite mutation triggers.

    The registered SQLite callback consults only this Python object. It never
    executes SQL, avoiding recursive use of a connection from within a trigger.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._message_id: str | None = None
        self._operations: frozenset[str] = frozenset()
        self._transaction_generation = 0
        self._authorized_generation: int | None = None

    def trace_transaction(self, statement: str) -> None:
        """Advance connection-local identity at transaction boundaries."""

        operation = statement.lstrip().split(None, 1)[0].upper()
        if operation in {"BEGIN", "COMMIT", "ROLLBACK"}:
            self._transaction_generation += 1

    def sqlite_authorizer(
        self,
        action: int,
        argument1: str | None,
        argument2: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        """Deny transaction escape while a mutation scope is active."""

        del argument2, database, trigger
        if self._message_id is not None and (
            (
                action == sqlite3.SQLITE_TRANSACTION
                and (argument1 or "").upper() in {"COMMIT", "ROLLBACK"}
            )
            or (
                action == sqlite3.SQLITE_SAVEPOINT
                and (argument1 or "").upper() in {"RELEASE", "ROLLBACK"}
            )
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def _authorize(
        self,
        *,
        message_id: str,
        operations: Collection[str],
    ) -> Iterator[None]:
        """Authorize one coordinator-owned mutation scope on this connection."""

        if not self._connection.in_transaction:
            raise RuntimeError("caller_transaction_required")
        if self._message_id is not None:
            raise RuntimeError("semantic_mutation_authorization_already_active")
        if type(message_id) is not str or not message_id:
            raise ValueError("message_id")
        normalized = frozenset(operations)
        if not normalized or any(
            type(item) is not str or not item for item in normalized
        ):
            raise ValueError("operations")
        self._message_id = message_id
        self._operations = normalized
        self._authorized_generation = self._transaction_generation
        try:
            yield
        finally:
            self._clear()

    def _clear(self) -> None:
        """Clear authorization without retaining mutation identity."""

        self._message_id = None
        self._operations = frozenset()
        self._authorized_generation = None

    def _sqlite_authorized(self, message_id: object, operation: object) -> int:
        """Return one only for the active message and allowlisted operation."""

        return int(
            type(message_id) is str
            and type(operation) is str
            and message_id == self._message_id
            and operation in self._operations
            and self._connection.in_transaction
            and self._authorized_generation == self._transaction_generation
        )

    def _assert_current_transaction(self) -> None:
        """Reject a callback that escaped the authorized transaction."""

        if (
            not self._connection.in_transaction
            or self._authorized_generation != self._transaction_generation
        ):
            raise RuntimeError("semantic_mutation_transaction_changed")


def register_semantic_mutation_guard(
    connection: sqlite3.Connection,
) -> _SemanticMutationAuthorization:
    """Register the fail-closed semantic mutation guard on one connection."""

    authorization = _SemanticMutationAuthorization(connection)
    connection.create_function(
        SEMANTIC_MUTATION_GUARD_FUNCTION,
        2,
        authorization._sqlite_authorized,
    )
    connection.set_trace_callback(authorization.trace_transaction)
    connection.set_authorizer(authorization.sqlite_authorizer)
    return authorization


class BaseDB(ABC):
    """
    Base class for all database modules providing standardized path handling.

    This class ensures consistent handling of:
    - Union[str, Path] type for db_path
    - Special ':memory:' case for in-memory databases
    - Client ID for multi-client support
    - Private file connection enforcement
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str = "default",
        check_integrity_on_startup: bool = False,
        *,
        initialize_schema: bool = True,
    ):
        """
        Initialize the base database with standardized path handling.

        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier for multi-client support
            check_integrity_on_startup: Whether to run integrity check on startup
            initialize_schema: Whether to initialize the subclass schema
        """
        # Standardized path handling
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = lexical_path(db_path)
        else:
            self.is_memory_db = db_path == ":memory:"
            if self.is_memory_db:
                self.db_path = Path(":memory:")  # Symbolic Path for consistency
            else:
                self.db_path = lexical_path(db_path)

        # Store string representation for SQLite connection
        self.db_path_str = ":memory:" if self.is_memory_db else str(self.db_path)

        # Store client ID
        self.client_id = client_id

        # Initialize schema (implemented by subclasses)
        if initialize_schema:
            self._initialize_schema()

        # Run integrity check if requested
        if initialize_schema and check_integrity_on_startup and not self.is_memory_db:
            logger.info(
                f"Running startup integrity check for {self.__class__.__name__}"
            )
            if not self.check_integrity():
                logger.warning(
                    f"Database integrity check failed for {self.db_path_str}. "
                    "Consider running repairs or restoring from backup."
                )
                # Note: We don't raise an exception here to allow the app to continue
                # with potentially degraded functionality. Subclasses can override
                # this behavior if they need stricter integrity enforcement.

        logger.info(
            f"{self.__class__.__name__} initialized with path: {self.db_path_str} [Client: {self.client_id}]"
        )

    @abstractmethod
    def _initialize_schema(self):
        """
        Initialize the database schema.
        Must be implemented by subclasses.
        """
        pass

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory.
        Can be overridden by subclasses for custom connection handling.
        """
        conn = connect_private_sqlite("db.base", self.db_path_str)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        """
        Close database connections if needed.
        Can be overridden by subclasses.
        """
        pass

    def vacuum(self):
        """
        Vacuum the database to reclaim unused space and optimize performance.
        """
        if self.is_memory_db:
            logger.debug("Skipping vacuum for in-memory database")
            return

        try:
            conn = self._get_connection()
            conn.execute("VACUUM")
            conn.close()
            logger.info(f"Successfully vacuumed database: {self.db_path_str}")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise

    def check_integrity(self) -> bool:
        """
        Check the integrity of the database.

        Returns:
            bool: True if integrity check passes, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()

            is_ok = result and result[0] == "ok"
            if is_ok:
                logger.info(f"Database integrity check passed: {self.db_path_str}")
            else:
                logger.error(f"Database integrity check failed: {self.db_path_str}")

            return is_ok
        except Exception as e:
            logger.error(f"Failed to check database integrity: {e}")
            return False
