# Prompts_DB_v2.py
#########################################
# Prompts_DB_v2 Library
# Manages Prompts_DB_v2 operations for specific instances, handling sync metadata internally.
# Requires a client_id during Database initialization.
# Standalone functions require a PromptsDatabase instance passed as an argument.
#
# Manages SQLite database interactions for prompts and related metadata.
#
# This library provides a `PromptsDatabase` class to encapsulate operations for a specific
# SQLite database file. It handles connection management (thread-locally),
# schema initialization and versioning, CRUD operations, Full-Text Search (FTS)
# updates, and internal logging of changes for synchronization purposes via a
# `sync_log` table.
#
# Key Features:
# - Instance-based: Each `PromptsDatabase` object connects to a specific DB file.
# - Client ID Tracking: Requires a `client_id` for attributing changes.
# - Internal Sync Logging: Automatically logs creates, updates, deletes, links,
#   and unlinks to the `sync_log` table for external sync processing.
# - Internal FTS Updates: Manages associated FTS5 tables (`prompts_fts`, `prompt_keywords_fts`)
#   within the Python code during relevant operations.
# - Schema Versioning: Checks and applies schema updates upon initialization.
# - Thread-Safety: Uses thread-local storage for database connections.
# - Soft Deletes: Implements soft deletes (`deleted=1`) for Prompts and Keywords.
# - Transaction Management: Provides a context manager for atomic operations.
# - Standalone Functions: Offers utility functions that operate on a `PromptsDatabase`
#   instance (e.g., searching, fetching related data, exporting).
####
#
import json
import sqlite3
import threading
import time
import uuid
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any, Iterator, Optional, Union

#
# Third-Party Libraries
from loguru import logger
from loguru import logger as logging

#
# Local Imports
from .sql_validation import validate_table_name, validate_column_name
from .sql_logging import preview_params
from .private_sqlite import backup_connection_to_private, connect_private_sqlite
from ..Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Utils.private_paths import PrivatePathError, lexical_path
from tldw_chatbook.Utils.fts5_match_forms import (
    build_and_match_query,
    quote_fts5_token,
)
#
########################################################################################################################
#
# Functions:


# --- Custom Exceptions (Mirrors Media_DB_v2) ---
class DatabaseError(Exception):
    """Base exception for database related errors."""

    pass


class SchemaError(DatabaseError):
    """Exception for schema version mismatches or migration failures."""

    pass


class InputError(ValueError):
    """Custom exception for input validation errors."""

    pass


class ConflictError(DatabaseError):
    """Indicates a conflict due to concurrent modification (version mismatch)."""

    def __init__(
        self,
        message="Conflict detected: Record modified concurrently.",
        entity=None,
        identifier=None,
        *,
        code="conflict",
    ):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier
        self.code = code

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base


class ExpectedVersionConflictError(ConflictError):
    """A conditional write observed a newer persisted version."""

    def __init__(self, message, entity=None, identifier=None):
        super().__init__(
            message,
            entity,
            identifier,
            code="expected_version",
        )


class PromptNameConflictError(ConflictError):
    """A Prompt write would duplicate another active Prompt name."""

    def __init__(self, message, entity=None, identifier=None):
        super().__init__(message, entity, identifier, code="name_conflict")


# --- Database Class ---
class PromptsDatabase:
    _CURRENT_SCHEMA_VERSION = 4
    _PROMPT_HISTORY_INDEX_NAME = "idx_sync_log_prompt_history"
    _PROMPT_HISTORY_INDEX_COLUMNS = (
        ("entity", False),
        ("entity_uuid", False),
        ("change_id", True),
        ("operation", False),
    )
    _PROMPT_HISTORY_INDEX_PREDICATE = (
        "entity = 'prompts' and operation in ('create', 'update')"
    )
    _PROMPT_HISTORY_INDEX_SQL = """
        CREATE INDEX idx_sync_log_prompt_history
        ON sync_log (
            entity,
            entity_uuid,
            change_id DESC,
            operation
        )
        WHERE entity = 'Prompts'
          AND operation IN ('create', 'update')
    """
    _PROMPT_HISTORY_MAX_PAGE_SIZE = 100
    _SQLITE_SIGNED_INTEGER_MAX = (2**63) - 1
    _PROMPT_HISTORY_COUNT_SQL = """
        SELECT COUNT(*)
        FROM sync_log
        WHERE entity = 'Prompts'
          AND entity_uuid = ?
          AND operation IN ('create', 'update')
    """
    # task-261: idle window within which the per-call `SELECT 1` liveness
    # ping is skipped for a recently-used thread-local connection (see
    # `_get_thread_connection`).
    _LIVENESS_PING_IDLE_SECONDS = 30.0
    _PROMPT_BROWSE_SORT_COLUMNS = {
        "last_modified": "p.last_modified",
        "name": "prompt_browse_lower(p.name)",
    }
    _PROMPT_BROWSE_SORT_ORDERS = {"asc": "ASC", "desc": "DESC"}

    _TABLES_SQL_V1 = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (0);

    CREATE TABLE IF NOT EXISTS Prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        author TEXT,
        details TEXT,
        system_prompt TEXT, -- Renamed from 'system'
        user_prompt TEXT,   -- Renamed from 'user'
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    CREATE TABLE IF NOT EXISTS PromptKeywordsTable ( -- Renamed from Keywords to avoid clash if in same scope
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    CREATE TABLE IF NOT EXISTS PromptKeywordLinks ( -- Renamed from PromptKeywords for clarity
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id INTEGER NOT NULL,
        keyword_id INTEGER NOT NULL,
        UNIQUE (prompt_id, keyword_id),
        FOREIGN KEY (prompt_id) REFERENCES Prompts(id) ON DELETE CASCADE,
        FOREIGN KEY (keyword_id) REFERENCES PromptKeywordsTable(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS sync_log (
        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_uuid TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
        timestamp DATETIME NOT NULL,
        client_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        payload TEXT
    );
    """

    _INDICES_SQL_V1 = """
                      CREATE INDEX IF NOT EXISTS idx_prompts_name ON Prompts(name);
                      CREATE INDEX IF NOT EXISTS idx_prompts_author ON Prompts(author);
                      CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_uuid ON Prompts(uuid);
                      CREATE INDEX IF NOT EXISTS idx_prompts_last_modified ON Prompts(last_modified);
                      CREATE INDEX IF NOT EXISTS idx_prompts_deleted ON Prompts(deleted);

                      CREATE UNIQUE INDEX IF NOT EXISTS idx_promptkeywordstable_keyword ON PromptKeywordsTable(keyword);
                      CREATE UNIQUE INDEX IF NOT EXISTS idx_promptkeywordstable_uuid ON PromptKeywordsTable(uuid);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordstable_last_modified ON PromptKeywordsTable(last_modified);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordstable_deleted ON PromptKeywordsTable(deleted);

                      CREATE INDEX IF NOT EXISTS idx_promptkeywordlinks_prompt_id ON PromptKeywordLinks(prompt_id);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordlinks_keyword_id ON PromptKeywordLinks(keyword_id);

                      CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
                      CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
                      CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id); \
                      """

    _TRIGGERS_SQL_V1 = """
    DROP TRIGGER IF EXISTS prompts_validate_sync_update;
    CREATE TRIGGER prompts_validate_sync_update BEFORE UPDATE ON Prompts
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Prompts): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Prompts): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Prompts): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS promptkeywordstable_validate_sync_update;
    CREATE TRIGGER promptkeywordstable_validate_sync_update BEFORE UPDATE ON PromptKeywordsTable
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;
    """

    _FTS_TABLES_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(
        name,
        author,
        details,
        system_prompt,
        user_prompt,
        content='Prompts',
        content_rowid='id'
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS prompt_keywords_fts USING fts5(
        keyword,
        content='PromptKeywordsTable',
        content_rowid='id'
    );
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str,
        check_integrity_on_startup: bool = False,
    ):
        """
        Initializes the PromptsDatabase instance, sets up the connection pool (via threading.local),
        and ensures the database schema is correctly initialized or migrated.

        Args:
            db_path (Union[str, Path]): The path to the SQLite database file or ':memory:'.
            client_id (str): A unique identifier for the client using this database instance.
            check_integrity_on_startup: Whether to run integrity check on startup.

        Raises:
            ValueError: If client_id is empty or None.
            DatabaseError: If database initialization or schema setup fails.
        """
        # Determine if it's an in-memory DB and normalize the path lexically.
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = lexical_path(db_path)
        else:  # Treat as string
            self.is_memory_db = db_path == ":memory:"
            if not self.is_memory_db:
                self.db_path = lexical_path(db_path)
            else:
                # For in-memory DB, we don't need a Path object
                self.db_path = None

        # Store the path as a string for convenience/logging
        self.db_path_str = str(self.db_path) if not self.is_memory_db else ":memory:"

        # Validate client_id
        if not client_id:
            raise ValueError("Client ID cannot be empty or None.")
        self.client_id = client_id

        logging.info(
            f"Initializing PromptsDatabase object for path: {self.db_path_str} [Client ID: {self.client_id}]"
        )

        # Initialize thread-local storage for connections
        self._local = threading.local()

        # Flag to track successful initialization before logging completion
        initialization_successful = False
        try:
            # --- Core Initialization Logic ---
            # This establishes the first connection for the current thread
            # and applies/verifies the schema.
            self._initialize_schema()

            # Run integrity check if requested and not in-memory
            if check_integrity_on_startup and not self.is_memory_db:
                logging.info("Running startup integrity check for PromptsDatabase")
                if not self.check_integrity():
                    logging.warning(
                        f"Database integrity check failed for {self.db_path_str}. "
                        "Consider running repairs or restoring from backup."
                    )
                    # Note: We don't raise an exception here to allow the app to continue
                    # with potentially degraded functionality.

            initialization_successful = (
                True  # Mark as successful if no exception occurred
            )
        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            # Catch specific DB/Schema errors and general SQLite errors during init
            logging.opt(exception=True).critical(
                f"FATAL: Prompts DB Initialization failed for {self.db_path_str}: {e}"
            )
            # Attempt to clean up the connection before raising
            self.close_connection()  # Important to call this if available
            # Re-raise as a DatabaseError to signal catastrophic failure
            raise DatabaseError(f"Prompts Database initialization failed: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during initialization
            logging.opt(exception=True).critical(
                f"FATAL: Unexpected error during Prompts DB Initialization for {self.db_path_str}: {e}"
            )
            # Attempt cleanup
            self.close_connection()  # Important to call this
            # Re-raise as a DatabaseError
            raise DatabaseError(
                f"Unexpected prompts database initialization error: {e}"
            ) from e
        finally:
            # Log completion status based on the flag
            if initialization_successful:
                logging.debug(
                    f"PromptsDatabase initialization completed successfully for {self.db_path_str}"
                )
            else:
                # This path indicates an exception was caught and raised above.
                # Logging here provides context that the __init__ block finished, albeit with failure.
                logging.error(
                    f"PromptsDatabase initialization block finished for {self.db_path_str}, but failed."
                )

    # --- Connection Management ---
    def _get_thread_connection(self) -> sqlite3.Connection:
        """Retrieve or create the current thread's SQLite connection.

        task-22224 EXCEPTION -- this held connection keeps the legacy
        default isolation level for now instead of the store template's
        ``isolation_level = None`` (rule: ``Library_Ingest_Jobs_DB.py``
        module docstring). ``transaction()`` here borrows via
        ``in_transaction`` and several write paths still rely on implicit
        transactions (``execute_query(commit=True)`` commits whatever is
        pending; schema paths mix ``executescript`` into transactions), so
        flipping requires this file's own commit/rollback/write-site census
        first, as done for ``ChaChaNotes_DB`` -- its own task. Do NOT copy
        this pattern into new stores.

        task-261: the ``SELECT 1`` liveness ping is gated behind an idle
        threshold (``_LIVENESS_PING_IDLE_SECONDS``) instead of running on
        every call — connections are thread-local and long-lived, and
        ``close_connection()`` always clears the thread-local reference, so
        a recently-used connection is known-good without a ping. A
        connection idle past the threshold still gets the ping +
        transparent-reopen treatment.

        Returns:
            sqlite3.Connection: The thread-local database connection.

        Raises:
            DatabaseError: If connecting to the database fails.
        """
        conn = getattr(self._local, "conn", None)
        is_closed = conn is None
        if conn:
            last_used = getattr(self._local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used) >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    logging.warning(
                        f"Thread-local connection to {self.db_path_str} was closed. Reopening."
                    )
                    is_closed = True
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._local.conn = None

        if is_closed:
            try:
                conn = connect_private_sqlite(
                    "db.prompts.primary",
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,  # Required for threading.local
                    timeout=10,  # seconds
                )
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode=WAL;")
                # NORMAL is safe under WAL (app-crash-safe; only an OS/power
                # crash can lose the last commit or two, acceptable for this
                # local prompt library) and avoids an fsync on every commit --
                # the default FULL was fsyncing the WAL on every commit
                # despite WAL already being enabled. See
                # Library_Ingest_Jobs_DB.py:57-61 for the original template
                # (task-15465).
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                self._local.conn = conn
                logging.debug(
                    f"Opened/Reopened SQLite connection to {self.db_path_str} [Client: {self.client_id}, Thread: {threading.current_thread().name}]"
                )
            except (sqlite3.Error, PrivatePathError) as e:
                logging.opt(exception=True).error(
                    f"Failed to connect to database at {self.db_path_str}: {e}"
                )
                self._local.conn = None
                raise DatabaseError(
                    f"Failed to connect to database '{self.db_path_str}': {e}"
                ) from e
        self._local.conn_last_used = time.monotonic()
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        return self._get_thread_connection()

    def close_connection(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                conn = self._local.conn
                self._local.conn = None
                conn.close()
                logging.debug(
                    f"Closed connection for thread {threading.current_thread().name}."
                )
            except sqlite3.Error as e:
                logging.warning(f"Error closing connection: {e}")
            finally:
                if hasattr(self._local, "conn"):
                    self._local.conn = None

    def backup_database(self, backup_file_path: str) -> bool:
        """
        Creates a backup of the current database to the specified file path.

        Args:
            backup_file_path (str): The path to save the backup database file.

        Returns:
            bool: True if the backup was successful, False otherwise.
        """
        logger.info(
            f"Starting database backup from '{self.db_path_str}' to '{backup_file_path}'"
        )
        try:
            src_conn = self.get_connection()
            backup_connection_to_private(
                "db.prompts.backup",
                src_conn,
                self.db_path_str,
                backup_file_path,
            )

            logger.info(
                f"Database backup successful from '{self.db_path_str}' to '{backup_file_path}'"
            )
            return True
        except ValueError as ve:
            logger.opt(exception=True).error(f"ValueError during database backup: {ve}")
            return False
        except sqlite3.Error as e:
            logger.opt(exception=True).error(
                f"SQLite error during database backup: {e}"
            )
            return False
        except Exception as e:
            logger.opt(exception=True).error(
                f"Unexpected error during database backup: {e}"
            )
            return False

    def check_integrity(self) -> bool:
        """
        Check the integrity of the database.

        Returns:
            bool: True if integrity check passes, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            is_ok = result and result[0] == "ok"
            if is_ok:
                logging.info(f"Database integrity check passed: {self.db_path_str}")
            else:
                logging.error(f"Database integrity check failed: {self.db_path_str}")

            return is_ok
        except Exception as e:
            logging.error(f"Failed to check database integrity: {e}")
            return False

    # --- Query Execution ---
    def execute_query(
        self, query: str, params: tuple = None, *, commit: bool = False
    ) -> sqlite3.Cursor:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # Lazy + BLOB-safe (`logging` here is loguru, aliased -- see the
            # module-level `from loguru import logger as logging` import --
            # it has no isEnabledFor(); use opt(lazy=True) instead).
            logging.opt(lazy=True).debug(
                "Executing Query: {}",
                lambda: f"{query[:200]}... Params: {preview_params(params)}",
            )
            cursor.execute(query, params or ())
            if commit:
                conn.commit()
                logging.debug("Committed.")
            return cursor
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "sync error" in msg:  # From our custom triggers
                logging.error(f"Sync Validation Failed: {e}")
                raise e
            else:
                logging.opt(exception=True).error(
                    f"Integrity error: {query[:200]}... Error: {e}"
                )
                raise DatabaseError(f"Integrity constraint violation: {e}") from e
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"Query failed: {query[:200]}... Error: {e}"
            )
            raise DatabaseError(f"Query execution failed: {e}") from e

    def execute_many(
        self, query: str, params_list: List[tuple], *, commit: bool = False
    ) -> Optional[sqlite3.Cursor]:
        conn = self.get_connection()
        if not isinstance(params_list, list):
            raise TypeError("params_list must be a list.")
        if not params_list:
            return None
        try:
            cursor = conn.cursor()
            logging.debug(
                f"Executing Many: {query[:150]}... with {len(params_list)} sets."
            )
            cursor.executemany(query, params_list)
            if commit:
                conn.commit()
                logging.debug("Committed Many.")
            return cursor
        except sqlite3.IntegrityError as e:
            logging.opt(exception=True).error(
                f"Integrity error during Execute Many: {query[:150]}... Error: {e}"
            )
            raise DatabaseError(
                f"Integrity constraint violation during batch: {e}"
            ) from e
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"Execute Many failed: {query[:150]}... Error: {e}"
            )
            raise DatabaseError(f"Execute Many failed: {e}") from e
        except TypeError as te:
            logging.opt(exception=True).error(
                f"TypeError during Execute Many: {te}. Check params_list format."
            )
            raise TypeError(f"Parameter list format error: {te}") from te

    # --- Transaction Context ---
    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run database work in a transaction.

        Args:
            immediate: Reserve SQLite's writer slot before yielding when true.

        Yields:
            The active SQLite connection.

        Raises:
            Exception: Re-raises the database operation error after rollback.
        """
        conn = self.get_connection()
        in_outer = conn.in_transaction
        try:
            if not in_outer:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                logging.debug("Started transaction.")
            yield conn  # yield connection
            if not in_outer:
                conn.commit()
                logging.debug("Committed transaction.")
        except Exception as e:
            if not in_outer:
                logging.error(
                    "PromptsDatabase.transaction: rolling back category={}",
                    type(e).__name__,
                )
                try:
                    conn.rollback()
                    logging.debug("Rollback successful.")
                except sqlite3.Error as rb_err:
                    logging.error(
                        "PromptsDatabase.transaction: rollback failed category={}",
                        type(rb_err).__name__,
                    )
            raise e

    # --- Schema Initialization and Migration ---
    def _get_db_version(self, conn: sqlite3.Connection) -> int:
        try:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            result = cursor.fetchone()
            return result["version"] if result else 0
        except sqlite3.Error as e:
            if "no such table: schema_version" in str(e).lower():
                return 0
            else:
                raise DatabaseError(f"Could not determine schema version: {e}") from e

    _SCHEMA_UPDATE_VERSION_SQL_V1 = (
        "UPDATE schema_version SET version = 1 WHERE version = 0;"
    )

    _MIGRATIONS = {
        0: {
            "to_version": 1,
            "function": "_apply_schema_v1",
            "description": "Initial prompts schema",
        },
        1: {
            "to_version": 2,
            "function": "_apply_migration_v1_to_v2",
            "description": "Add structured prompt metadata columns",
        },
        2: {
            "to_version": 3,
            "function": "_apply_migration_v2_to_v3",
            "description": "Add prompt artifact type",
        },
        3: {
            "to_version": 4,
            "function": "_apply_migration_v3_to_v4",
            "description": "Add retained Prompt history index",
        },
    }

    def _apply_schema_v1(self, conn: sqlite3.Connection):
        logging.info(
            f"Applying initial schema (Version 1) to DB: {self.db_path_str}..."
        )
        try:
            core_schema_script_with_version_update = f"""
                {self._TABLES_SQL_V1}
                {self._INDICES_SQL_V1}
                {self._TRIGGERS_SQL_V1}
                {self._SCHEMA_UPDATE_VERSION_SQL_V1}
            """
            with self.transaction():
                logging.debug("[Schema V1] Applying Core Schema + Version Update...")
                conn.executescript(core_schema_script_with_version_update)
                logging.debug(
                    "[Schema V1] Core Schema script (incl. version update) executed."
                )
                # Validation
                cursor = conn.execute("PRAGMA table_info(Prompts)")
                columns = {row["name"] for row in cursor.fetchall()}
                expected_cols = {
                    "id",
                    "name",
                    "author",
                    "details",
                    "system_prompt",
                    "user_prompt",
                    "uuid",
                    "last_modified",
                    "version",
                    "client_id",
                    "deleted",
                }
                if not expected_cols.issubset(columns):
                    missing_cols = expected_cols - columns
                    raise SchemaError(
                        f"Validation Error: Prompts table missing columns: {missing_cols}"
                    )
                logging.debug("[Schema V1] Prompts table structure validated.")
                cursor_check = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                )
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 1:
                    raise SchemaError(
                        "Schema version update did not take effect within transaction."
                    )
            logging.info(
                f"[Schema V1] Core Schema V1 applied and committed for DB: {self.db_path_str}."
            )
            try:
                logging.debug("[Schema V1] Applying FTS Tables...")
                conn.executescript(self._FTS_TABLES_SQL)
                conn.commit()  # Commit FTS creation separately
                logging.info("[Schema V1] FTS Tables created successfully.")
            except sqlite3.Error as fts_err:
                logging.opt(exception=True).error(
                    f"[Schema V1] Failed to create FTS tables: {fts_err}"
                )
                # This might not be fatal if FTS is optional or can be rebuilt.
        except sqlite3.Error as e:
            logging.opt(exception=True).error(f"[Schema V1] Application failed: {e}")
            raise DatabaseError(f"DB schema V1 setup failed: {e}") from e

    def _apply_migration_v1_to_v2(self, conn: sqlite3.Connection):
        logging.info(
            f"Applying prompts migration from version 1 to 2 for DB: {self.db_path_str}..."
        )
        migration_sql = """
        ALTER TABLE Prompts ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy';
        ALTER TABLE Prompts ADD COLUMN prompt_schema_version INTEGER;
        ALTER TABLE Prompts ADD COLUMN prompt_definition TEXT;
        UPDATE schema_version SET version = 2 WHERE version = 1;
        """

        try:
            with self.transaction():
                conn.executescript(migration_sql)

                cursor = conn.execute("PRAGMA table_info(Prompts)")
                columns = {row["name"] for row in cursor.fetchall()}
                expected_cols = {
                    "prompt_format",
                    "prompt_schema_version",
                    "prompt_definition",
                }
                if not expected_cols.issubset(columns):
                    missing_cols = expected_cols - columns
                    raise SchemaError(
                        f"Validation Error: Prompts table missing migrated columns: {missing_cols}"
                    )

                cursor_check = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                )
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 2:
                    raise SchemaError(
                        "Schema version update to 2 did not take effect within transaction."
                    )

            logging.info(
                f"Prompts migration to version 2 applied successfully for DB: {self.db_path_str}."
            )
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"[Migration v1->v2] Failed during migration: {e}"
            )
            raise DatabaseError(f"Migration v1->v2 failed: {e}") from e

    def _apply_migration_v2_to_v3(self, conn: sqlite3.Connection):
        """Add the first-class Prompt/Recipe discriminator atomically."""
        logging.info(
            f"Applying prompts migration from version 2 to 3 for DB: {self.db_path_str}..."
        )
        try:
            with self.transaction():
                conn.execute(
                    """
                    ALTER TABLE Prompts
                    ADD COLUMN artifact_type TEXT NOT NULL DEFAULT 'prompt'
                    CHECK(artifact_type IN ('prompt', 'recipe'))
                    """
                )
                conn.execute("UPDATE schema_version SET version = 3 WHERE version = 2")
                columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(Prompts)")
                }
                if "artifact_type" not in columns:
                    raise SchemaError(
                        "Validation Error: Prompts table missing artifact_type column."
                    )
                version_in_tx = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
                if not version_in_tx or version_in_tx["version"] != 3:
                    raise SchemaError(
                        "Schema version update to 3 did not take effect within transaction."
                    )
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"[Migration v2->v3] Failed during migration: {e}"
            )
            raise DatabaseError(f"Migration v2->v3 failed: {e}") from e

    def _apply_migration_v3_to_v4(self, conn: sqlite3.Connection):
        """Add an index for bounded retained Prompt history reads."""
        logging.info(
            f"Applying prompts migration from version 3 to 4 for DB: {self.db_path_str}..."
        )
        try:
            with self.transaction():
                conn.execute(f"DROP INDEX IF EXISTS {self._PROMPT_HISTORY_INDEX_NAME}")
                conn.execute(self._PROMPT_HISTORY_INDEX_SQL)

                index_row = conn.execute(
                    """
                    SELECT sql
                    FROM sqlite_master
                    WHERE type = 'index' AND name = ?
                    """,
                    (self._PROMPT_HISTORY_INDEX_NAME,),
                ).fetchone()
                if index_row is None:
                    raise SchemaError(
                        "Validation Error: retained Prompt history index is missing."
                    )

                actual_columns = tuple(
                    (row["name"], bool(row["desc"]))
                    for row in conn.execute(
                        f"PRAGMA index_xinfo({self._PROMPT_HISTORY_INDEX_NAME})"
                    )
                    if row["key"]
                )
                if actual_columns != self._PROMPT_HISTORY_INDEX_COLUMNS:
                    raise SchemaError(
                        "Validation Error: retained Prompt history index columns "
                        "do not match the required order."
                    )

                normalized_sql = " ".join(str(index_row["sql"]).lower().split())
                _prefix, separator, actual_predicate = normalized_sql.partition(
                    " where "
                )
                if (
                    not separator
                    or actual_predicate != self._PROMPT_HISTORY_INDEX_PREDICATE
                ):
                    raise SchemaError(
                        "Validation Error: retained Prompt history index predicate "
                        "does not match the required filter."
                    )

                conn.execute("UPDATE schema_version SET version = 4 WHERE version = 3")

                version_in_tx = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
                if not version_in_tx or version_in_tx["version"] != 4:
                    raise SchemaError(
                        "Schema version update to 4 did not take effect within transaction."
                    )
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"[Migration v3->v4] Failed during migration: {e}"
            )
            raise DatabaseError(f"Migration v3->v4 failed: {e}") from e

    def _initialize_schema(self):
        conn = self.get_connection()
        try:
            current_db_version = self._get_db_version(conn)
            target_version = self._CURRENT_SCHEMA_VERSION
            logging.info(
                f"Checking DB schema. Current: {current_db_version}, Code supports: {target_version}"
            )

            if current_db_version == target_version:
                logging.debug("Database schema is up to date.")
                try:  # Ensure FTS tables exist
                    conn.executescript(self._FTS_TABLES_SQL)
                    conn.commit()
                    logging.debug("Verified FTS tables exist.")
                except sqlite3.Error as fts_err:
                    logging.warning(
                        f"Could not verify/create FTS tables on correct schema: {fts_err}"
                    )
                return

            if current_db_version > target_version:
                raise SchemaError(
                    f"DB schema version ({current_db_version}) is newer than supported ({target_version})."
                )

            while current_db_version < target_version:
                migration = self._MIGRATIONS.get(current_db_version)
                if not migration:
                    raise SchemaError(
                        f"No migration path defined from version {current_db_version}."
                    )

                next_version = migration["to_version"]
                migration_func = getattr(self, migration["function"], None)
                if migration_func is None:
                    raise SchemaError(
                        f"Migration function {migration['function']} not found."
                    )

                logging.info(
                    f"Applying migration: {migration['description']} (v{current_db_version} -> v{next_version})"
                )
                migration_func(conn)

                current_db_version = self._get_db_version(conn)
                if current_db_version != next_version:
                    raise SchemaError(
                        f"Migration {migration['function']} failed. Expected version {next_version}, got {current_db_version}."
                    )

            try:
                conn.executescript(self._FTS_TABLES_SQL)
                conn.commit()
                logging.debug("Verified FTS tables exist after migrations.")
            except sqlite3.Error as fts_err:
                logging.warning(
                    f"Could not verify/create FTS tables after migrations: {fts_err}"
                )

            logging.info(
                f"Database schema initialized/migrated to version {target_version}."
            )
        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            logging.opt(exception=True).error(
                f"Schema initialization/migration failed: {e}"
            )
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    # --- Internal Helpers ---
    def _get_current_utc_timestamp_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())

    def _normalize_keyword(self, keyword: str) -> str:
        return re.sub(r"\s+", " ", keyword.strip().lower())

    def _canonicalize_prompt_keywords(self, keywords_list: Any) -> List[str]:
        """Validate and canonicalize a Prompt's requested keyword membership."""
        if not isinstance(keywords_list, list):
            raise InputError("keywords must be a list of strings.")

        normalized_keywords = set()
        for keyword in keywords_list:
            if not isinstance(keyword, str):
                raise InputError("keyword list members must be strings.")
            if keyword.strip():
                normalized_keywords.add(self._normalize_keyword(keyword))
        return sorted(normalized_keywords)

    def _normalize_prompt_format(self, prompt_format: Optional[str]) -> str:
        if prompt_format is None:
            return "legacy"
        if prompt_format not in {"legacy", "structured"}:
            raise InputError("Prompt format must be either 'legacy' or 'structured'.")
        return prompt_format

    def _normalize_artifact_type(self, artifact_type: Optional[str]) -> str:
        """Validate the durable Prompt/Recipe discriminator at the DB boundary."""
        if artifact_type is None:
            return "prompt"
        if not isinstance(artifact_type, str) or artifact_type not in {
            "prompt",
            "recipe",
        }:
            raise InputError("artifact_type must be either 'prompt' or 'recipe'.")
        return artifact_type

    @staticmethod
    def _normalize_expected_version(expected_version: Optional[int]) -> Optional[int]:
        if expected_version is None:
            return None
        if type(expected_version) is not int or expected_version < 1:
            raise InputError("expected_version must be a positive integer or None.")
        return expected_version

    @staticmethod
    def _is_busy_snapshot_error(error: BaseException) -> bool:
        """Identify SQLite's WAL stale-snapshot error without masking other I/O errors."""
        return isinstance(error, sqlite3.OperationalError) and getattr(
            error, "sqlite_errorcode", None
        ) == getattr(sqlite3, "SQLITE_BUSY_SNAPSHOT", 517)

    def _serialize_prompt_definition(self, prompt_definition: Any) -> Optional[str]:
        if prompt_definition is None:
            return None
        if isinstance(prompt_definition, str):
            return prompt_definition
        if isinstance(prompt_definition, (dict, list)):
            return json.dumps(prompt_definition)
        raise InputError(
            "Prompt definition must be a JSON string, dict, list, or None."
        )

    def _get_next_version(
        self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any
    ) -> Optional[Tuple[int, int]]:
        # Validate SQL identifiers to prevent injection
        if not validate_table_name(table, "prompts"):
            raise InputError(f"Invalid table name: {table}")
        if not validate_column_name(id_col, table):
            raise InputError(f"Invalid column name: {id_col}")

        try:
            cursor = conn.execute(
                f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0",
                (id_val,),
            )
            result = cursor.fetchone()
            if result:
                current_version = result["version"]
                if isinstance(current_version, int):
                    return current_version, current_version + 1
                else:
                    logging.error(
                        f"Invalid non-integer version '{current_version}' for {table} {id_col}={id_val}"
                    )
                    return None
        except sqlite3.Error as e:
            logging.error(
                f"DB error fetching version for {table} {id_col}={id_val}: {e}"
            )
            raise DatabaseError(f"Failed to fetch current version: {e}") from e
        return None

    @staticmethod
    def _serialize_sync_payload(payload: Optional[Dict]) -> Optional[str]:
        """Serialize sync payloads consistently, including SQLite datetimes."""
        if payload and isinstance(payload, dict):
            serializable_payload = {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in payload.items()
            }
            return json.dumps(serializable_payload, separators=(",", ":"))
        return json.dumps(payload, separators=(",", ":")) if payload else None

    def _log_sync_event(
        self,
        conn: sqlite3.Connection,
        entity: str,
        entity_uuid: str,
        operation: str,
        version: int,
        payload: Optional[Dict] = None,
    ) -> Optional[int]:
        if not entity or not entity_uuid or not operation:
            logging.error("Sync log attempt with missing entity, uuid, or operation.")
            return None
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        payload_json = self._serialize_sync_payload(payload)
        try:
            cursor = conn.execute(
                """
                         INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
                         VALUES (?, ?, ?, ?, ?, ?, ?)
                         """,
                (
                    entity,
                    entity_uuid,
                    operation,
                    current_time,
                    client_id,
                    version,
                    payload_json,
                ),
            )
            if cursor.lastrowid is None:
                raise DatabaseError("Failed to get change ID for sync event.")
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            logging.error(
                "Prompt sync event write failed category={}", type(e).__name__
            )
            raise DatabaseError("Failed to log Prompt sync event.") from None

    def _finalize_prompt_sync_snapshot(
        self,
        conn: sqlite3.Connection,
        change_id: Optional[int],
        prompt_id: int,
        payload: Dict[str, Any],
    ) -> None:
        """Finalize one pending Prompt event after keyword membership settles."""
        if change_id is None:
            raise DatabaseError(
                "Failed to finalize Prompt sync snapshot: missing change ID."
            )

        try:
            keyword_rows = conn.execute(
                """
                SELECT k.keyword
                FROM PromptKeywordsTable AS k
                JOIN PromptKeywordLinks AS link ON link.keyword_id = k.id
                WHERE link.prompt_id = ? AND k.deleted = 0
                """,
                (prompt_id,),
            ).fetchall()
            snapshot_payload = dict(payload)
            snapshot_payload["keywords"] = sorted(
                {row["keyword"] for row in keyword_rows}
            )
            snapshot_payload["keywords_captured"] = True
            serialized_payload = self._serialize_sync_payload(snapshot_payload)
            cursor = conn.execute(
                """
                UPDATE sync_log
                SET payload = ?
                WHERE change_id = ? AND entity = 'Prompts'
                """,
                (serialized_payload, change_id),
            )
            if cursor.rowcount != 1:
                raise DatabaseError(
                    "Failed to finalize Prompt sync snapshot: event not found."
                )
        except sqlite3.Error as e:
            logging.error(
                "Prompt sync snapshot finalization failed category={}",
                type(e).__name__,
            )
            raise DatabaseError("Failed to finalize Prompt sync snapshot.") from None

    # --- FTS Helper Methods ---
    def _update_fts_prompt(
        self,
        conn: sqlite3.Connection,
        prompt_id: int,
        name: str,
        author: Optional[str],
        details: Optional[str],
        system_prompt: Optional[str],
        user_prompt: Optional[str],
    ):
        try:
            conn.execute(
                "INSERT OR REPLACE INTO prompts_fts (rowid, name, author, details, system_prompt, user_prompt) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    prompt_id,
                    name,
                    author or "",
                    details or "",
                    system_prompt or "",
                    user_prompt or "",
                ),
            )
        except sqlite3.Error as e:
            logging.error("Prompt FTS update failed category={}", type(e).__name__)
            raise DatabaseError("Failed to update Prompt FTS.") from None

    def _delete_fts_prompt(self, conn: sqlite3.Connection, prompt_id: int):
        try:
            conn.execute("DELETE FROM prompts_fts WHERE rowid = ?", (prompt_id,))
        except sqlite3.Error as e:
            logging.error("Prompt FTS delete failed category={}", type(e).__name__)
            raise DatabaseError("Failed to delete Prompt FTS.") from None

    def _update_fts_prompt_keyword(
        self, conn: sqlite3.Connection, keyword_id: int, keyword: str
    ):
        try:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_keywords_fts (rowid, keyword) VALUES (?, ?)",
                (keyword_id, keyword),
            )
        except sqlite3.Error as e:
            logging.error(
                "Prompt keyword FTS update failed category={}", type(e).__name__
            )
            raise DatabaseError("Failed to update Prompt keyword FTS.") from None

    def _delete_fts_prompt_keyword(self, conn: sqlite3.Connection, keyword_id: int):
        try:
            conn.execute(
                "DELETE FROM prompt_keywords_fts WHERE rowid = ?", (keyword_id,)
            )
        except sqlite3.Error as e:
            logging.error(
                "Prompt keyword FTS delete failed category={}", type(e).__name__
            )
            raise DatabaseError("Failed to delete Prompt keyword FTS.") from None

    # --- Public Mutating Methods ---
    def add_keyword(self, keyword_text: str) -> Optional[int]:
        """
        Add a keyword to the database.

        Args:
            keyword_text: The keyword text to add

        Returns:
            The keyword ID if successful, None otherwise
        """
        keyword_id, _ = self._add_keyword_full(keyword_text)
        return keyword_id

    def _add_keyword_full(
        self, keyword_text: str
    ) -> Tuple[Optional[int], Optional[str]]:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword cannot be empty.")
        normalized_keyword = self._normalize_keyword(keyword_text)
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, uuid, deleted, version FROM PromptKeywordsTable WHERE keyword = ?",
                    (normalized_keyword,),
                )
                existing = cursor.fetchone()

                if existing:
                    kw_id, kw_uuid, is_deleted, current_version = (
                        existing["id"],
                        existing["uuid"],
                        existing["deleted"],
                        existing["version"],
                    )
                    if is_deleted:  # Undelete
                        new_version = current_version + 1
                        cursor.execute(
                            "UPDATE PromptKeywordsTable SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                            (
                                current_time,
                                new_version,
                                client_id,
                                kw_id,
                                current_version,
                            ),
                        )
                        if cursor.rowcount == 0:
                            raise ConflictError(
                                "Failed to undelete keyword due to version mismatch or it was not found.",
                                "PromptKeywordsTable",
                                kw_id,
                            )
                        cursor.execute(
                            "SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id,)
                        )
                        payload = dict(cursor.fetchone())
                        self._log_sync_event(
                            conn,
                            "PromptKeywordsTable",
                            kw_uuid,
                            "update",
                            new_version,
                            payload,
                        )
                        self._update_fts_prompt_keyword(conn, kw_id, normalized_keyword)
                        return kw_id, kw_uuid
                    else:  # Already active, just return its ID and UUID
                        logger.debug(
                            "Prompt keyword reused operation=add_keyword "
                            "category=existing"
                        )
                        return kw_id, kw_uuid
                else:  # New keyword
                    new_uuid = self._generate_uuid()
                    new_version = 1
                    cursor.execute(
                        "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, 0)",
                        (
                            normalized_keyword,
                            new_uuid,
                            current_time,
                            new_version,
                            client_id,
                        ),
                    )
                    kw_id = cursor.lastrowid
                    if not kw_id:
                        raise DatabaseError("Failed to get ID for new prompt keyword.")
                    cursor.execute(
                        "SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id,)
                    )
                    payload = dict(cursor.fetchone())
                    self._log_sync_event(
                        conn,
                        "PromptKeywordsTable",
                        new_uuid,
                        "create",
                        new_version,
                        payload,
                    )
                    self._update_fts_prompt_keyword(conn, kw_id, normalized_keyword)
                    return kw_id, new_uuid
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
            logger.error(
                "Prompt keyword write failed operation=add_keyword category={}",
                type(exc).__name__,
            )
            if isinstance(exc, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError("Failed to add or update Prompt keyword.") from None

    def get_active_keyword_by_text(self, keyword_text: str) -> Optional[Dict]:
        """
        Fetches an active (not deleted) keyword by its exact normalized text.

        Args:
            keyword_text: The keyword text to search for.

        Returns:
            A dictionary of the keyword's data if found and active, else None.
        """
        if not keyword_text or not keyword_text.strip():
            return None  # Or raise InputError if strictness is preferred here
        normalized_keyword = self._normalize_keyword(keyword_text)
        query = "SELECT id, uuid, keyword, last_modified, version, client_id FROM PromptKeywordsTable WHERE keyword = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (normalized_keyword,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(
                f"Error fetching active keyword by text '{normalized_keyword}': {e}"
            )
            # Depending on desired strictness, could raise or return None
            # For a simple check, returning None on error is acceptable if the next step handles it.
            return None

    def add_prompt(
        self,
        name: str,
        author: Optional[str],
        details: Optional[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        overwrite: bool = False,
        prompt_format: Optional[str] = None,
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        artifact_type: Optional[str] = None,
        serialize_create: bool = False,
    ) -> Tuple[Optional[int], Optional[str], str]:
        start_time = time.time()

        if not name or not name.strip():
            raise InputError("Prompt name cannot be empty.")
        name = (
            name.strip()
        )  # Use original case for name, but ensure no leading/trailing spaces

        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        normalized_prompt_definition = self._serialize_prompt_definition(
            prompt_definition
        )
        normalized_prompt_format = (
            self._normalize_prompt_format(prompt_format)
            if prompt_format is not None
            else None
        )
        normalized_artifact_type = self._normalize_artifact_type(artifact_type)
        normalized_keywords = (
            self._canonicalize_prompt_keywords(keywords)
            if keywords is not None
            else None
        )

        try:
            with self.transaction(immediate=serialize_create) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, uuid, version, deleted, prompt_format, prompt_schema_version,
                           prompt_definition, artifact_type
                    FROM Prompts
                    WHERE name = ?
                    """,
                    (name,),
                )
                existing = cursor.fetchone()

                prompt_id: Optional[int] = None
                prompt_uuid: Optional[str] = None
                prompt_sync_change_id: Optional[int] = None
                prompt_snapshot_payload: Optional[Dict[str, Any]] = None
                action_taken: str = "skipped"

                if existing:
                    prompt_id, prompt_uuid, current_version, is_deleted = (
                        existing["id"],
                        existing["uuid"],
                        existing["version"],
                        existing["deleted"],
                    )
                    if (
                        is_deleted and not overwrite
                    ):  # Soft-deleted, treat as "exists" if not overwriting
                        return (
                            prompt_id,
                            prompt_uuid,
                            f"Prompt '{name}' exists but is soft-deleted. Use overwrite to restore/update.",
                        )
                    if not overwrite and not is_deleted:
                        raise ConflictError(
                            f"Prompt '{name}' already exists."
                        )  # RAISE ERROR
                        # return prompt_id, prompt_uuid, f"Prompt '{name}' already exists. Skipped."

                    # Overwrite or undelete-and-update
                    action_taken = "updated"
                    new_version = current_version + 1
                    resolved_prompt_format = (
                        normalized_prompt_format
                        if normalized_prompt_format is not None
                        else existing["prompt_format"]
                    )
                    resolved_prompt_schema_version = (
                        prompt_schema_version
                        if prompt_schema_version is not None
                        else existing["prompt_schema_version"]
                    )
                    resolved_prompt_definition = (
                        normalized_prompt_definition
                        if prompt_definition is not None
                        else existing["prompt_definition"]
                    )
                    resolved_artifact_type = (
                        normalized_artifact_type
                        if artifact_type is not None
                        else existing["artifact_type"]
                    )
                    update_data = {
                        "name": name,
                        "author": author,
                        "details": details,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt_format": resolved_prompt_format,
                        "prompt_schema_version": resolved_prompt_schema_version,
                        "prompt_definition": resolved_prompt_definition,
                        "artifact_type": resolved_artifact_type,
                        "last_modified": current_time,
                        "version": new_version,
                        "client_id": client_id,
                        "deleted": 0,
                        "uuid": prompt_uuid,
                    }
                    cursor.execute(
                        """UPDATE Prompts
                                      SET author=?,
                                          details=?,
                                          system_prompt=?,
                                          user_prompt=?,
                                          prompt_format=?,
                                          prompt_schema_version=?,
                                          prompt_definition=?,
                                          artifact_type=?,
                                          last_modified=?,
                                          version=?,
                                          client_id=?,
                                          deleted=0
                                      WHERE id = ?
                                        AND version = ?""",
                        (
                            author,
                            details,
                            system_prompt,
                            user_prompt,
                            resolved_prompt_format,
                            resolved_prompt_schema_version,
                            resolved_prompt_definition,
                            resolved_artifact_type,
                            current_time,
                            new_version,
                            client_id,
                            prompt_id,
                            current_version,
                        ),
                    )
                    if cursor.rowcount == 0:
                        # If it was deleted and overwrite is true, version check might fail if version wasn't for active.
                        # Or, a concurrent update happened.
                        # Re-fetch to check if it was deleted to adjust error message
                        cursor.execute(
                            "SELECT deleted, version FROM Prompts WHERE id=?",
                            (prompt_id,),
                        )
                        refetched = cursor.fetchone()
                        if (
                            refetched
                            and refetched["deleted"]
                            and refetched["version"] == current_version
                        ):
                            # This means it was soft-deleted, and we tried to update with old version.
                            # We need to increment from its current soft-deleted version.
                            # For simplicity, we'll just tell user to handle undelete separately or ensure version matches.
                            # A more complex undelete+update would fetch its true current version first.
                            raise ConflictError(
                                f"Prompt '{name}' (ID: {prompt_id}) was soft-deleted. Undelete first or ensure overwrite logic handles versioning correctly.",
                                "Prompts",
                                prompt_id,
                            )
                        raise ConflictError(
                            f"Failed to update prompt '{name}'.", "Prompts", prompt_id
                        )

                    prompt_snapshot_payload = update_data
                    prompt_sync_change_id = self._log_sync_event(
                        conn, "Prompts", prompt_uuid, "update", new_version, update_data
                    )
                    self._update_fts_prompt(
                        conn,
                        prompt_id,
                        name,
                        author,
                        details,
                        system_prompt,
                        user_prompt,
                    )
                else:  # New prompt
                    action_taken = "added"
                    prompt_uuid = self._generate_uuid()
                    new_version = 1
                    resolved_prompt_format = normalized_prompt_format or "legacy"
                    insert_data = {
                        "name": name,
                        "author": author,
                        "details": details,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt_format": resolved_prompt_format,
                        "prompt_schema_version": prompt_schema_version,
                        "prompt_definition": normalized_prompt_definition,
                        "artifact_type": normalized_artifact_type,
                        "uuid": prompt_uuid,
                        "last_modified": current_time,
                        "version": new_version,
                        "client_id": client_id,
                        "deleted": 0,
                    }
                    cursor.execute(
                        """INSERT INTO Prompts (
                                                name,
                                                author,
                                                details,
                                                system_prompt,
                                                user_prompt,
                                                prompt_format,
                                                prompt_schema_version,
                                                prompt_definition,
                                                artifact_type,
                                                uuid,
                                                last_modified,
                                                version,
                                                client_id,
                                                deleted
                                            )
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                        (
                            name,
                            author,
                            details,
                            system_prompt,
                            user_prompt,
                            resolved_prompt_format,
                            prompt_schema_version,
                            normalized_prompt_definition,
                            normalized_artifact_type,
                            prompt_uuid,
                            current_time,
                            new_version,
                            client_id,
                        ),
                    )
                    prompt_id = cursor.lastrowid
                    if not prompt_id:
                        raise DatabaseError("Failed to get ID for new prompt.")
                    prompt_snapshot_payload = insert_data
                    prompt_sync_change_id = self._log_sync_event(
                        conn, "Prompts", prompt_uuid, "create", new_version, insert_data
                    )
                    self._update_fts_prompt(
                        conn,
                        prompt_id,
                        name,
                        author,
                        details,
                        system_prompt,
                        user_prompt,
                    )

                if (
                    prompt_id and normalized_keywords is not None
                ):  # keywords can be empty list to remove all
                    self.update_keywords_for_prompt(
                        prompt_id, keywords_list=normalized_keywords
                    )  # This is an instance method

                if prompt_id is None or prompt_snapshot_payload is None:
                    raise DatabaseError("Failed to prepare Prompt sync snapshot.")
                self._finalize_prompt_sync_snapshot(
                    conn,
                    prompt_sync_change_id,
                    prompt_id,
                    prompt_snapshot_payload,
                )

                msg = f"Prompt '{name}' {action_taken} successfully."

                # Log success metrics
                duration = time.time() - start_time
                log_histogram(
                    "prompts_db_operation_duration",
                    duration,
                    labels={
                        "operation": "add_prompt",
                        "action": action_taken,
                        "has_keywords": "true" if keywords else "false",
                    },
                )
                log_counter(
                    "prompts_db_operation_count",
                    labels={
                        "operation": "add_prompt",
                        "action": action_taken,
                        "status": "success",
                        "overwrite": str(overwrite),
                    },
                )

                return prompt_id, prompt_uuid, msg

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            if self._is_busy_snapshot_error(e):
                e = ConflictError(
                    "Prompt creation lost a WAL snapshot race.", "Prompts", name
                )
            # Log error metrics
            duration = time.time() - start_time
            error_type = (
                "input_error"
                if isinstance(e, InputError)
                else "conflict"
                if isinstance(e, ConflictError)
                else "database_error"
                if isinstance(e, DatabaseError)
                else "sqlite_error"
            )
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "add_prompt",
                    "action": "error",
                    "has_keywords": "false",
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "add_prompt",
                    "status": "error",
                    "error_type": error_type,
                },
            )

            logger.error(
                "PromptsDatabase.add_prompt: operation failed category={}",
                error_type,
            )
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to process prompt '{name}': {e}") from e

    def update_keywords_for_prompt(self, prompt_id: int, keywords_list: List[str]):
        normalized_new_keywords = self._canonicalize_prompt_keywords(keywords_list)

        try:
            # This method is called within an existing transaction (e.g. from add_prompt)
            # So, use self.get_connection() but don't start a new transaction here.
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get prompt_uuid for logging
            cursor.execute(
                "SELECT uuid FROM Prompts WHERE id = ? AND deleted = 0", (prompt_id,)
            )
            prompt_info = cursor.fetchone()
            if not prompt_info:
                raise InputError(
                    f"Cannot update keywords: Prompt ID {prompt_id} not found or deleted."
                )
            prompt_uuid = prompt_info["uuid"]

            # Get current keywords for the prompt
            cursor.execute(
                """
                           SELECT pkl.keyword_id, pkw.keyword, pkw.uuid as keyword_uuid
                           FROM PromptKeywordLinks pkl
                                    JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                           WHERE pkl.prompt_id = ? AND pkw.deleted = 0
                           """,
                (prompt_id,),
            )
            current_keyword_links = {
                row["keyword_id"]: {"text": row["keyword"], "uuid": row["keyword_uuid"]}
                for row in cursor.fetchall()
            }
            current_keyword_ids = set(current_keyword_links.keys())

            target_keyword_data: Dict[
                int, Dict[str, str]
            ] = {}  # {keyword_id: {'text': text, 'uuid': uuid}}
            if normalized_new_keywords:
                for kw_text in normalized_new_keywords:
                    # _add_keyword_full is an instance method, it will use the existing transaction
                    kw_id, kw_uuid = self._add_keyword_full(kw_text)
                    if kw_id and kw_uuid:
                        target_keyword_data[kw_id] = {"text": kw_text, "uuid": kw_uuid}
                    else:
                        # This should not happen if add_keyword is robust
                        raise DatabaseError(
                            f"Failed to get/add keyword '{kw_text}' during prompt keyword update."
                        )

            target_keyword_ids = set(target_keyword_data.keys())

            ids_to_add = target_keyword_ids - current_keyword_ids
            ids_to_remove = current_keyword_ids - target_keyword_ids
            link_sync_version = 1  # For link/unlink operations, version is on the junction table itself if it had one, or just 1 for the event

            if ids_to_remove:
                remove_placeholders = ",".join("?" * len(ids_to_remove))
                cursor.execute(
                    f"DELETE FROM PromptKeywordLinks WHERE prompt_id = ? AND keyword_id IN ({remove_placeholders})",
                    (prompt_id, *list(ids_to_remove)),
                )
                for removed_id in ids_to_remove:
                    keyword_uuid = current_keyword_links[removed_id]["uuid"]
                    link_composite_uuid = (
                        f"{prompt_uuid}_{keyword_uuid}"  # Composite UUID for the link
                    )
                    payload = {"prompt_uuid": prompt_uuid, "keyword_uuid": keyword_uuid}
                    self._log_sync_event(
                        conn,
                        "PromptKeywordLinks",
                        link_composite_uuid,
                        "unlink",
                        link_sync_version,
                        payload,
                    )

            if ids_to_add:
                insert_params = [(prompt_id, kid) for kid in ids_to_add]
                cursor.executemany(
                    "INSERT OR IGNORE INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)",
                    insert_params,
                )
                for added_id in ids_to_add:
                    keyword_uuid = target_keyword_data[added_id]["uuid"]
                    link_composite_uuid = f"{prompt_uuid}_{keyword_uuid}"
                    payload = {"prompt_uuid": prompt_uuid, "keyword_uuid": keyword_uuid}
                    self._log_sync_event(
                        conn,
                        "PromptKeywordLinks",
                        link_composite_uuid,
                        "link",
                        link_sync_version,
                        payload,
                    )

            if ids_to_add or ids_to_remove:
                logger.debug(
                    "Prompt keyword membership updated added={} removed={}",
                    len(ids_to_add),
                    len(ids_to_remove),
                )
        except (InputError, DatabaseError, sqlite3.Error) as exc:
            logger.error(
                "Prompt keyword membership update failed category={}",
                type(exc).__name__,
            )
            if isinstance(exc, (InputError, DatabaseError)):
                raise
            raise DatabaseError("Keyword update failed.") from None

    def update_prompt_by_id(
        self,
        prompt_id: int,
        update_data: Dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Updates an existing prompt identified by its ID.
        Handles name changes and ensures the new name doesn't conflict with other existing prompts.

        Args:
            prompt_id: The ID of the prompt to update.
            update_data: A dictionary containing fields to update (name, author, details, system_prompt, user_prompt).
                         Keywords are handled separately by `update_keywords_for_prompt`.
            expected_version: The version captured by the caller, when updating an
                existing working copy. The comparison occurs in this transaction.

        Returns:
            A tuple (updated_prompt_uuid, message_string).

        Raises:
            InputError: If required fields like 'name' are missing or invalid in update_data.
            ConflictError: If a name change conflicts with another existing prompt, or version mismatch.
            DatabaseError: For other database issues.
        """
        start_time = time.time()

        if "name" in update_data and (
            not update_data["name"] or not update_data["name"].strip()
        ):
            raise InputError("Prompt name cannot be empty if provided for update.")
        normalized_keywords = None
        if "keywords" in update_data:
            normalized_keywords = self._canonicalize_prompt_keywords(
                update_data["keywords"]
            )
        expected_version = self._normalize_expected_version(expected_version)
        normalized_artifact_type = None
        if "artifact_type" in update_data:
            normalized_artifact_type = self._normalize_artifact_type(
                update_data.get("artifact_type")
            )

        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        normalized_prompt_definition = None
        if "prompt_definition" in update_data:
            normalized_prompt_definition = self._serialize_prompt_definition(
                update_data.get("prompt_definition")
            )

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Get current state of the prompt being updated
                cursor.execute(
                    "SELECT uuid, name, version, deleted FROM Prompts WHERE id = ?",
                    (prompt_id,),
                )
                existing_prompt_state = cursor.fetchone()

                if not existing_prompt_state:
                    return (
                        None,
                        f"Prompt with ID {prompt_id} not found.",
                    )  # Or raise InputError("Prompt not found")

                original_uuid = existing_prompt_state["uuid"]
                original_name = existing_prompt_state["name"]
                current_version = existing_prompt_state["version"]
                is_deleted = existing_prompt_state["deleted"]

                if expected_version is not None and expected_version != int(
                    current_version
                ):
                    raise ExpectedVersionConflictError(
                        "Prompt changed after it was opened.", "Prompts", prompt_id
                    )

                if is_deleted:  # Optional: decide if updating a soft-deleted prompt should undelete it.
                    # For now, let's assume we are updating an active prompt or an explicitly fetched soft-deleted one.
                    # If this method should also undelete, set 'deleted = 0' in the update.
                    pass

                new_name = update_data.get("name", original_name).strip()

                # If name is changing, check for conflict with *other* prompts
                if new_name != original_name:
                    cursor.execute(
                        "SELECT id FROM Prompts WHERE name = ? AND id != ? AND deleted = 0",
                        (new_name, prompt_id),
                    )
                    conflicting_prompt = cursor.fetchone()
                    if conflicting_prompt:
                        raise PromptNameConflictError(
                            f"Another active prompt with name '{new_name}' already exists (ID: {conflicting_prompt['id']})."
                        )

                new_version = current_version + 1

                set_clauses = []
                params = []

                # Build SET clause dynamically
                if (
                    "name" in update_data
                    and update_data["name"].strip() != original_name
                ):  # Only if actually changing
                    set_clauses.append("name = ?")
                    params.append(new_name)
                if "author" in update_data:
                    set_clauses.append("author = ?")
                    params.append(update_data.get("author"))
                if "details" in update_data:
                    set_clauses.append("details = ?")
                    params.append(update_data.get("details"))
                if "system_prompt" in update_data:
                    set_clauses.append("system_prompt = ?")
                    params.append(update_data.get("system_prompt"))
                if "user_prompt" in update_data:
                    set_clauses.append("user_prompt = ?")
                    params.append(update_data.get("user_prompt"))
                if "prompt_format" in update_data:
                    set_clauses.append("prompt_format = ?")
                    params.append(
                        self._normalize_prompt_format(update_data.get("prompt_format"))
                    )
                if "prompt_schema_version" in update_data:
                    set_clauses.append("prompt_schema_version = ?")
                    params.append(update_data.get("prompt_schema_version"))
                if "prompt_definition" in update_data:
                    set_clauses.append("prompt_definition = ?")
                    params.append(normalized_prompt_definition)
                if "artifact_type" in update_data:
                    set_clauses.append("artifact_type = ?")
                    params.append(normalized_artifact_type)

                # Always update these
                set_clauses.extend(
                    ["last_modified = ?", "version = ?", "client_id = ?", "deleted = 0"]
                )  # Ensure it's marked active
                params.extend([current_time, new_version, client_id])

                if not set_clauses:  # Nothing to update besides version/timestamp
                    return original_uuid, "No changes detected to update."

                sql_set_clause = ", ".join(set_clauses)
                update_sql = (
                    f"UPDATE Prompts SET {sql_set_clause} WHERE id = ? AND version = ?"
                )
                params.extend([prompt_id, current_version])

                cursor.execute(update_sql, tuple(params))

                if cursor.rowcount == 0:
                    raise ConflictError(
                        f"Failed to update prompt ID {prompt_id} (version mismatch or record gone).",
                        "Prompts",
                        prompt_id,
                    )

                # Log sync event
                # Fetch the full updated row for payload
                cursor.execute("SELECT * FROM Prompts WHERE id = ?", (prompt_id,))
                updated_payload = dict(cursor.fetchone())
                prompt_sync_change_id = self._log_sync_event(
                    conn,
                    "Prompts",
                    original_uuid,
                    "update",
                    new_version,
                    updated_payload,
                )

                # Update FTS
                self._update_fts_prompt(
                    conn,
                    prompt_id,
                    updated_payload["name"],
                    updated_payload.get("author"),
                    updated_payload.get("details"),
                    updated_payload.get("system_prompt"),
                    updated_payload.get("user_prompt"),
                )

                # Handle keywords if provided in update_data (assuming 'keywords' is a list of strings)
                if normalized_keywords is not None:
                    self.update_keywords_for_prompt(
                        prompt_id, normalized_keywords
                    )  # Call existing method

                self._finalize_prompt_sync_snapshot(
                    conn,
                    prompt_sync_change_id,
                    prompt_id,
                    updated_payload,
                )

                # Log success metrics
                duration = time.time() - start_time
                log_histogram(
                    "prompts_db_operation_duration",
                    duration,
                    labels={
                        "operation": "update_prompt",
                        "fields_updated": str(len(set_clauses)),
                        "has_keywords": "true"
                        if "keywords" in update_data
                        else "false",
                    },
                )
                log_counter(
                    "prompts_db_operation_count",
                    labels={
                        "operation": "update_prompt",
                        "status": "success",
                        "name_changed": "true"
                        if new_name != original_name
                        else "false",
                    },
                )

                return (
                    original_uuid,
                    f"Prompt ID {prompt_id} updated successfully to version {new_version}.",
                )

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            if self._is_busy_snapshot_error(e):
                e = ConflictError(
                    "Prompt update lost a version race.", "Prompts", prompt_id
                )
            # Log error metrics
            duration = time.time() - start_time
            error_type = (
                "input_error"
                if isinstance(e, InputError)
                else "conflict"
                if isinstance(e, ConflictError)
                else "database_error"
                if isinstance(e, DatabaseError)
                else "sqlite_error"
            )
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "update_prompt",
                    "fields_updated": "0",
                    "has_keywords": "false",
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "update_prompt",
                    "status": "error",
                    "error_type": error_type,
                },
            )

            logger.opt(exception=True).error(
                f"Error updating prompt ID {prompt_id}: {e}"
            )
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            raise DatabaseError(f"Failed to update prompt ID {prompt_id}: {e}") from e

    @staticmethod
    def _canonical_prompt_batch_targets(
        targets: tuple["PromptBatchTarget", ...],
    ) -> tuple["PromptBatchTarget", ...]:
        """Validate and order one strict non-empty Prompt mutation batch."""
        if type(targets) is not tuple:
            raise TypeError("targets must be an exact tuple.")
        if not targets:
            raise ValueError("targets must be non-empty.")
        if any(type(target) is not PromptBatchTarget for target in targets):
            raise TypeError("targets must contain only exact PromptBatchTarget values.")

        validated = tuple(
            PromptBatchTarget(target.local_id, target.expected_version)
            for target in targets
        )
        if len({target.local_id for target in validated}) != len(validated):
            raise ValueError("targets must use unique local IDs.")
        return tuple(sorted(validated, key=lambda target: target.local_id))

    def _require_prompt_mutation_transaction_ownership(self) -> None:
        """Reject public mutations that cannot own their durable commit."""
        if self.get_connection().in_transaction:
            raise DatabaseError("Prompt mutation transaction ownership unavailable.")

    @staticmethod
    def _require_canonical_uuid(value: Any, error_message: str) -> str:
        if type(value) is not str:
            raise DatabaseError(error_message)
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, ValueError):
            raise DatabaseError(error_message) from None
        if str(parsed) != value:
            raise DatabaseError(error_message)
        return value

    def _active_prompt_keyword_rows(
        self, conn: sqlite3.Connection, prompt_id: int
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            sorted(
                conn.execute(
                    """
                    SELECT pkw.id, pkw.keyword, pkw.uuid AS keyword_uuid
                    FROM PromptKeywordLinks AS pkl
                    JOIN PromptKeywordsTable AS pkw ON pkl.keyword_id = pkw.id
                    WHERE pkl.prompt_id = ? AND pkw.deleted = 0
                    ORDER BY pkw.keyword COLLATE NOCASE, pkw.id ASC
                    """,
                    (prompt_id,),
                ).fetchall(),
                key=lambda row: row["keyword"],
            )
        )
        if any(
            type(row["keyword"]) is not str or not row["keyword"].strip()
            for row in rows
        ):
            raise DatabaseError("Prompt keyword recovery metadata is unavailable.")
        keywords = [row["keyword"] for row in rows]
        if self._canonicalize_prompt_keywords(keywords) != keywords:
            raise DatabaseError("Prompt keyword recovery metadata is unavailable.")
        for row in rows:
            self._require_canonical_uuid(
                row["keyword_uuid"],
                "Prompt keyword recovery metadata is unavailable.",
            )
        return rows

    def _restore_prompt_keyword_rows(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
    ) -> tuple[sqlite3.Row, ...]:
        tombstone_event = conn.execute(
            """
            SELECT payload
            FROM sync_log
            WHERE entity = 'Prompts'
              AND entity_uuid = ?
              AND operation = 'delete'
              AND version = ?
            ORDER BY change_id DESC
            LIMIT 1
            """,
            (row["uuid"], expected_version),
        ).fetchone()
        try:
            payload = json.loads(tombstone_event["payload"])
        except (TypeError, KeyError, json.JSONDecodeError):
            payload = None
        if (
            type(payload) is not dict
            or payload.get("keywords_captured") is not True
            or type(payload.get("keywords")) is not list
            or any(type(keyword) is not str for keyword in payload["keywords"])
        ):
            raise DatabaseError("Prompt tombstone recovery metadata is unavailable.")

        keywords = self._canonicalize_prompt_keywords(payload["keywords"])
        if payload["keywords"] != keywords:
            raise DatabaseError("Prompt tombstone recovery metadata is unavailable.")
        if conn.execute(
            "SELECT COUNT(*) FROM PromptKeywordLinks WHERE prompt_id = ?",
            (row["id"],),
        ).fetchone()[0]:
            raise DatabaseError("Prompt tombstone recovery state is invalid.")

        keyword_rows = []
        for keyword in keywords:
            keyword_row = conn.execute(
                """
                SELECT id, keyword, uuid AS keyword_uuid, deleted, version
                FROM PromptKeywordsTable
                WHERE keyword = ?
                """,
                (keyword,),
            ).fetchone()
            if (
                keyword_row is None
                or type(keyword_row["version"]) is not int
                or not 1 <= keyword_row["version"] <= self._SQLITE_SIGNED_INTEGER_MAX
                or int(keyword_row["deleted"]) not in (0, 1)
                or (
                    int(keyword_row["deleted"]) == 1
                    and keyword_row["version"] == self._SQLITE_SIGNED_INTEGER_MAX
                )
            ):
                raise DatabaseError("Prompt keyword recovery metadata is unavailable.")
            self._require_canonical_uuid(
                keyword_row["keyword_uuid"],
                "Prompt keyword recovery metadata is unavailable.",
            )
            keyword_rows.append(keyword_row)
        return tuple(keyword_rows)

    def _validate_delete_prompt_row(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
    ) -> None:
        self._require_canonical_uuid(
            row["uuid"], "Prompt delete recovery metadata is unavailable."
        )
        self._active_prompt_keyword_rows(conn, int(row["id"]))
        PromptDeleteReceiptEntry(
            local_id=int(row["id"]),
            title=row["name"],
            artifact_type=row["artifact_type"],
            tombstone_version=expected_version + 1,
        )

    def _validate_restore_prompt_row(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
    ) -> tuple[sqlite3.Row, ...]:
        self._require_canonical_uuid(
            row["uuid"], "Prompt tombstone recovery metadata is unavailable."
        )
        keyword_rows = self._restore_prompt_keyword_rows(
            conn, row=row, expected_version=expected_version
        )
        PromptRestoreResultEntry(
            local_id=int(row["id"]), restored_version=expected_version + 1
        )
        return keyword_rows

    def _delete_prompt_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
    ) -> "PromptDeleteReceiptEntry":
        """Delete one prevalidated row without owning transaction settlement."""
        prompt_id = int(row["id"])
        prompt_uuid = str(row["uuid"])
        keywords = self._active_prompt_keyword_rows(conn, prompt_id)
        current_time = self._get_current_utc_timestamp_str()
        new_version = expected_version + 1
        cursor = conn.execute(
            """
            UPDATE Prompts
            SET deleted = 1, last_modified = ?, version = ?, client_id = ?
            WHERE id = ? AND deleted = 0 AND version = ?
            """,
            (current_time, new_version, self.client_id, prompt_id, expected_version),
        )
        if cursor.rowcount != 1:
            raise ExpectedVersionConflictError("Prompt batch delete conflict.")

        change_id = self._log_sync_event(
            conn,
            "Prompts",
            prompt_uuid,
            "delete",
            new_version,
            {
                "uuid": prompt_uuid,
                "last_modified": current_time,
                "version": new_version,
                "client_id": self.client_id,
                "deleted": 1,
                "keywords": [keyword["keyword"] for keyword in keywords],
                "keywords_captured": True,
            },
        )
        if change_id is None:
            raise DatabaseError("Prompt delete sync event is unavailable.")
        self._delete_fts_prompt(conn, prompt_id)
        if keywords:
            conn.execute(
                "DELETE FROM PromptKeywordLinks WHERE prompt_id = ?", (prompt_id,)
            )
            for keyword in keywords:
                keyword_uuid = str(keyword["keyword_uuid"])
                self._log_sync_event(
                    conn,
                    "PromptKeywordLinks",
                    f"{prompt_uuid}_{keyword_uuid}",
                    "unlink",
                    1,
                    {"prompt_uuid": prompt_uuid, "keyword_uuid": keyword_uuid},
                )
        return PromptDeleteReceiptEntry(
            local_id=prompt_id,
            title=row["name"],
            artifact_type=row["artifact_type"],
            tombstone_version=new_version,
        )

    def _restore_prompt_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
        keyword_rows: tuple[sqlite3.Row, ...],
    ) -> "PromptRestoreResultEntry":
        """Restore one prevalidated row without owning transaction settlement."""
        prompt_id = int(row["id"])
        prompt_uuid = str(row["uuid"])
        current_time = self._get_current_utc_timestamp_str()
        new_version = expected_version + 1
        cursor = conn.execute(
            """
            UPDATE Prompts
            SET deleted = 0, last_modified = ?, version = ?, client_id = ?
            WHERE id = ? AND deleted = 1 AND version = ?
            """,
            (current_time, new_version, self.client_id, prompt_id, expected_version),
        )
        if cursor.rowcount != 1:
            raise ExpectedVersionConflictError("Prompt batch restore conflict.")

        restored_row = conn.execute(
            "SELECT * FROM Prompts WHERE id = ?", (prompt_id,)
        ).fetchone()
        if restored_row is None:
            raise DatabaseError("Prompt restore state is unavailable.")
        restored_payload = dict(restored_row)
        change_id = self._log_sync_event(
            conn,
            "Prompts",
            prompt_uuid,
            "update",
            new_version,
            restored_payload,
        )
        for keyword in keyword_rows:
            if int(keyword["deleted"]) == 1:
                keyword_version = int(keyword["version"])
                keyword_uuid = str(keyword["keyword_uuid"])
                keyword_time = self._get_current_utc_timestamp_str()
                keyword_cursor = conn.execute(
                    """
                    UPDATE PromptKeywordsTable
                    SET deleted = 0, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND deleted = 1 AND version = ?
                    """,
                    (
                        keyword_time,
                        keyword_version + 1,
                        self.client_id,
                        keyword["id"],
                        keyword_version,
                    ),
                )
                if keyword_cursor.rowcount != 1:
                    raise ExpectedVersionConflictError("Prompt batch restore conflict.")
                restored_keyword = conn.execute(
                    "SELECT * FROM PromptKeywordsTable WHERE id = ?", (keyword["id"],)
                ).fetchone()
                if restored_keyword is None:
                    raise DatabaseError("Prompt keyword recovery state is unavailable.")
                self._log_sync_event(
                    conn,
                    "PromptKeywordsTable",
                    keyword_uuid,
                    "update",
                    keyword_version + 1,
                    dict(restored_keyword),
                )
                self._update_fts_prompt_keyword(
                    conn, int(keyword["id"]), str(keyword["keyword"])
                )
            conn.execute(
                """
                INSERT INTO PromptKeywordLinks (prompt_id, keyword_id)
                VALUES (?, ?)
                """,
                (prompt_id, keyword["id"]),
            )
            keyword_uuid = str(keyword["keyword_uuid"])
            self._log_sync_event(
                conn,
                "PromptKeywordLinks",
                f"{prompt_uuid}_{keyword_uuid}",
                "link",
                1,
                {"prompt_uuid": prompt_uuid, "keyword_uuid": keyword_uuid},
            )
        self._finalize_prompt_sync_snapshot(
            conn, change_id, prompt_id, restored_payload
        )
        self._update_fts_prompt(
            conn,
            prompt_id,
            restored_payload["name"],
            restored_payload.get("author"),
            restored_payload.get("details"),
            restored_payload.get("system_prompt"),
            restored_payload.get("user_prompt"),
        )
        return PromptRestoreResultEntry(
            local_id=prompt_id, restored_version=new_version
        )

    def soft_delete_prompts(
        self, targets: tuple["PromptBatchTarget", ...]
    ) -> "PromptBatchDeleteResult":
        """Atomically soft-delete one strict batch of active Prompt rows.

        Args:
            targets: Exact Prompt IDs and expected active-row versions.

        Returns:
            The canonical receipt for the committed batch.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If targets are empty, duplicated, or invalid.
            ExpectedVersionConflictError: If any target is missing or stale.
            DatabaseError: If transaction ownership or persistence fails.
        """
        canonical_targets = self._canonical_prompt_batch_targets(targets)
        try:
            self._require_prompt_mutation_transaction_ownership()
            with self.transaction(immediate=True) as conn:
                prepared = []
                for target in canonical_targets:
                    row = conn.execute(
                        "SELECT * FROM Prompts WHERE id = ?", (target.local_id,)
                    ).fetchone()
                    if (
                        row is None
                        or int(row["deleted"]) != 0
                        or int(row["version"]) != target.expected_version
                    ):
                        raise ExpectedVersionConflictError(
                            "Prompt batch delete conflict."
                        )
                    prepared.append((target, row))
                for target, row in prepared:
                    self._validate_delete_prompt_row(
                        conn,
                        row=row,
                        expected_version=target.expected_version,
                    )
                result = PromptBatchDeleteResult(
                    entries=tuple(
                        self._delete_prompt_in_transaction(
                            conn,
                            row=row,
                            expected_version=target.expected_version,
                        )
                        for target, row in prepared
                    )
                )
        except ExpectedVersionConflictError as exc:
            logger.error(
                "Prompt batch mutation failed operation=delete count={} category={}",
                len(canonical_targets),
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            logger.error(
                "Prompt batch mutation failed operation=delete count={} category={}",
                len(canonical_targets),
                type(exc).__name__,
            )
            raise DatabaseError("Prompt batch delete failed.") from None

        logger.info(
            "Prompt batch mutation committed operation=delete count={}",
            len(result.entries),
        )
        return result

    def restore_deleted_prompts(
        self, targets: tuple["PromptBatchTarget", ...]
    ) -> "PromptBatchRestoreResult":
        """Atomically restore one strict batch of Prompt tombstones.

        Args:
            targets: Exact Prompt IDs and expected tombstone versions.

        Returns:
            The canonical result for the committed batch restore.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If targets are empty, duplicated, or invalid.
            ExpectedVersionConflictError: If any target is missing or stale.
            DatabaseError: If recovery metadata or persistence is unavailable.
        """
        canonical_targets = self._canonical_prompt_batch_targets(targets)
        try:
            self._require_prompt_mutation_transaction_ownership()
            with self.transaction(immediate=True) as conn:
                prepared = []
                for target in canonical_targets:
                    row = conn.execute(
                        "SELECT * FROM Prompts WHERE id = ?", (target.local_id,)
                    ).fetchone()
                    if (
                        row is None
                        or int(row["deleted"]) != 1
                        or int(row["version"]) != target.expected_version
                    ):
                        raise ExpectedVersionConflictError(
                            "Prompt batch restore conflict."
                        )
                    prepared.append((target, row))
                validated = [
                    (
                        target,
                        row,
                        self._validate_restore_prompt_row(
                            conn,
                            row=row,
                            expected_version=target.expected_version,
                        ),
                    )
                    for target, row in prepared
                ]
                result = PromptBatchRestoreResult(
                    entries=tuple(
                        self._restore_prompt_in_transaction(
                            conn,
                            row=row,
                            expected_version=target.expected_version,
                            keyword_rows=keyword_rows,
                        )
                        for target, row, keyword_rows in validated
                    )
                )
        except ExpectedVersionConflictError as exc:
            logger.error(
                "Prompt batch mutation failed operation=restore count={} category={}",
                len(canonical_targets),
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            logger.error(
                "Prompt batch mutation failed operation=restore count={} category={}",
                len(canonical_targets),
                type(exc).__name__,
            )
            raise DatabaseError("Prompt batch restore failed.") from None

        logger.info(
            "Prompt batch mutation committed operation=restore count={}",
            len(result.entries),
        )
        return result

    def soft_delete_prompt(
        self,
        prompt_id_or_name_or_uuid: Union[int, str],
        *,
        expected_version: Optional[int] = None,
    ) -> bool:
        """Soft-delete one legacy ID/name/UUID lookup through the shared core."""
        if expected_version is not None and (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 1
        ):
            raise InputError("expected_version must be a positive integer or None.")

        col_name = "id"
        if isinstance(prompt_id_or_name_or_uuid, str):
            try:
                uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                col_name = "uuid"
            except ValueError:
                col_name = "name"
        if not validate_column_name(col_name, "Prompts"):
            raise InputError(f"Invalid column name: {col_name}")

        deleted = False
        try:
            self._require_prompt_mutation_transaction_ownership()
            with self.transaction(immediate=True) as conn:
                row = conn.execute(
                    f"SELECT * FROM Prompts WHERE {col_name} = ? AND deleted = 0",
                    (prompt_id_or_name_or_uuid,),
                ).fetchone()
                if row is not None:
                    resolved_version = (
                        int(row["version"])
                        if expected_version is None
                        else expected_version
                    )
                    if int(row["version"]) != resolved_version:
                        raise ExpectedVersionConflictError(
                            "Prompt changed after it was opened."
                        )
                    self._validate_delete_prompt_row(
                        conn, row=row, expected_version=resolved_version
                    )
                    self._delete_prompt_in_transaction(
                        conn, row=row, expected_version=resolved_version
                    )
                    deleted = True
        except ExpectedVersionConflictError:
            raise
        except Exception as exc:
            logger.error(
                "Prompt mutation failed operation=delete count=1 category={}",
                type(exc).__name__,
            )
            raise DatabaseError("Prompt delete failed.") from None

        if deleted:
            logger.info("Prompt mutation committed operation=delete count=1")
        return deleted

    def restore_deleted_prompt(
        self,
        prompt_id_or_name_or_uuid: Union[int, str],
        *,
        expected_version: int,
    ) -> Dict[str, Any]:
        """Restore one legacy ID/name/UUID tombstone through the shared core."""
        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 1
        ):
            raise InputError("expected_version must be a positive integer.")

        col_name = "id"
        if isinstance(prompt_id_or_name_or_uuid, str):
            try:
                uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                col_name = "uuid"
            except ValueError:
                col_name = "name"
        if not validate_column_name(col_name, "Prompts"):
            raise InputError(f"Invalid column name: {col_name}")

        try:
            self._require_prompt_mutation_transaction_ownership()
            with self.transaction(immediate=True) as conn:
                row = conn.execute(
                    f"SELECT * FROM Prompts WHERE {col_name} = ?",
                    (prompt_id_or_name_or_uuid,),
                ).fetchone()
                if (
                    row is None
                    or int(row["deleted"]) != 1
                    or int(row["version"]) != expected_version
                ):
                    raise ExpectedVersionConflictError(
                        "Prompt tombstone changed or is no longer deleted."
                    )
                recovery_keywords = self._validate_restore_prompt_row(
                    conn, row=row, expected_version=expected_version
                )
                entry = self._restore_prompt_in_transaction(
                    conn,
                    row=row,
                    expected_version=expected_version,
                    keyword_rows=recovery_keywords,
                )
                restored_row = conn.execute(
                    "SELECT * FROM Prompts WHERE id = ?", (entry.local_id,)
                ).fetchone()
                if restored_row is None:
                    raise DatabaseError("Prompt restore state is unavailable.")
                restored_payload = dict(restored_row)
                restored_payload["keywords"] = [
                    keyword_row["keyword"] for keyword_row in recovery_keywords
                ]
        except ExpectedVersionConflictError:
            raise
        except Exception as exc:
            logger.error(
                "Prompt mutation failed operation=restore count=1 category={}",
                type(exc).__name__,
            )
            raise DatabaseError("Prompt restore failed.") from None

        logger.info("Prompt mutation committed operation=restore count=1")
        return restored_payload

    def soft_delete_keyword(self, keyword_text: str) -> bool:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword to delete cannot be empty.")
        normalized_keyword = self._normalize_keyword(keyword_text)
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, uuid, version FROM PromptKeywordsTable WHERE keyword = ? AND deleted = 0",
                    (normalized_keyword,),
                )
                kw_info = cursor.fetchone()
                if not kw_info:
                    logger.warning(
                        f"Prompt keyword '{normalized_keyword}' not found or already deleted."
                    )
                    return False

                kw_id, kw_uuid, current_version = (
                    kw_info["id"],
                    kw_info["uuid"],
                    kw_info["version"],
                )
                new_version = current_version + 1

                cursor.execute(
                    "UPDATE PromptKeywordsTable SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                    (current_time, new_version, client_id, kw_id, current_version),
                )
                if cursor.rowcount == 0:
                    raise ConflictError("PromptKeywordsTable", kw_id)

                delete_payload = {
                    "uuid": kw_uuid,
                    "last_modified": current_time,
                    "version": new_version,
                    "client_id": client_id,
                    "deleted": 1,
                }
                self._log_sync_event(
                    conn,
                    "PromptKeywordsTable",
                    kw_uuid,
                    "delete",
                    new_version,
                    delete_payload,
                )
                self._delete_fts_prompt_keyword(conn, kw_id)

                # Explicitly unlink from prompts and log events
                cursor.execute(
                    """
                               SELECT p.uuid AS prompt_uuid
                               FROM PromptKeywordLinks pkl
                                        JOIN Prompts p ON pkl.prompt_id = p.id
                               WHERE pkl.keyword_id = ? AND p.deleted = 0
                               """,
                    (kw_id,),
                )
                prompts_to_unlink = cursor.fetchall()

                if prompts_to_unlink:
                    # FK ON DELETE CASCADE will handle actual deletion from PromptKeywordLinks.
                    # Log these unlinks.
                    cursor.execute(
                        "DELETE FROM PromptKeywordLinks WHERE keyword_id = ?", (kw_id,)
                    )
                    link_sync_version = 1
                    for p_to_unlink in prompts_to_unlink:
                        prompt_uuid_val = p_to_unlink["prompt_uuid"]
                        link_composite_uuid = f"{prompt_uuid_val}_{kw_uuid}"
                        unlink_payload = {
                            "prompt_uuid": prompt_uuid_val,
                            "keyword_uuid": kw_uuid,
                        }
                        self._log_sync_event(
                            conn,
                            "PromptKeywordLinks",
                            link_composite_uuid,
                            "unlink",
                            link_sync_version,
                            unlink_payload,
                        )
                    logging.debug(
                        f"Unlinked keyword ID {kw_id} from {len(prompts_to_unlink)} prompts during soft delete."
                    )

                logger.info(
                    f"Soft deleted prompt keyword '{normalized_keyword}' (ID: {kw_id}, UUID: {kw_uuid})."
                )
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.opt(exception=True).error(
                f"Error soft deleting prompt keyword '{keyword_text}': {e}"
            )
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to soft delete prompt keyword: {e}") from e

    # --- Read Methods ---
    def get_all_active_prompt_ids(self) -> List[int]:
        """Return every active local Prompt/Recipe row ID in stable order.

        Returns:
            All non-deleted Prompt IDs ordered by their integer row ID.

        Raises:
            sqlite3.Error: If SQLite cannot execute the uncapped ID query.
        """
        rows = self.get_connection().execute(
            "SELECT id FROM Prompts WHERE deleted = 0 ORDER BY id"
        )
        return [int(row["id"]) for row in rows.fetchall()]

    def fetch_prompt_chatbook_snapshot(
        self, prompt_id: int
    ) -> Optional[Dict[str, Any]]:
        """Read one active Prompt and its keywords from one SQLite snapshot.

        This export-specific seam uses the shared nested-aware transaction
        context while avoiding query helpers whose diagnostics include
        parameters, exception messages, or tracebacks. The detached result
        contains only portable Chatbook fields.

        Args:
            prompt_id: Positive SQLite-range Prompt row ID.

        Returns:
            Portable Prompt fields plus canonical active keywords, or ``None``
            when the row is missing or deleted.

        Raises:
            ValueError: If ``prompt_id`` is not a positive SQLite-range integer.
            DatabaseError: If the coherent snapshot cannot be read.
        """
        if type(prompt_id) is not int or not 1 <= prompt_id <= (2**63 - 1):
            raise ValueError("prompt_id must be a positive integer in SQLite range.")

        try:
            with self.transaction() as conn:
                row = conn.execute(
                    """
                    SELECT name, author, details, system_prompt, user_prompt,
                           artifact_type, prompt_format, prompt_schema_version,
                           prompt_definition
                    FROM Prompts
                    WHERE id = ? AND deleted = 0
                    """,
                    (prompt_id,),
                ).fetchone()
                if row is None:
                    return None
                keyword_rows = conn.execute(
                    """
                    SELECT keyword_table.keyword
                    FROM PromptKeywordsTable AS keyword_table
                    JOIN PromptKeywordLinks AS link
                      ON link.keyword_id = keyword_table.id
                    WHERE link.prompt_id = ? AND keyword_table.deleted = 0
                    ORDER BY keyword_table.keyword COLLATE NOCASE
                    """,
                    (prompt_id,),
                ).fetchall()
                return {
                    "name": row["name"],
                    "author": row["author"],
                    "details": row["details"],
                    "system_prompt": row["system_prompt"],
                    "user_prompt": row["user_prompt"],
                    "keywords": [
                        keyword_row["keyword"] for keyword_row in keyword_rows
                    ],
                    "artifact_type": row["artifact_type"],
                    "prompt_format": row["prompt_format"],
                    "prompt_schema_version": row["prompt_schema_version"],
                    "prompt_definition": row["prompt_definition"],
                }
        except (
            DatabaseError,
            sqlite3.Error,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
        ):
            raise DatabaseError("Failed to read Prompt export snapshot.") from None

    def get_prompt_by_id(
        self, prompt_id: int, include_deleted: bool = False
    ) -> Optional[Dict]:
        start_time = time.time()

        query = "SELECT * FROM Prompts WHERE id = ?"
        params = [prompt_id]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            found_result = dict(result) if result else None

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "get_by_id",
                    "found": "true" if found_result else "false",
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "get_by_id",
                    "status": "success",
                    "found": "true" if found_result else "false",
                },
            )

            return found_result
        except (DatabaseError, sqlite3.Error) as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={"operation": "get_by_id", "found": "false"},
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "get_by_id",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )

            logger.error(f"Error fetching prompt by ID {prompt_id}: {e}")
            raise DatabaseError(f"Failed fetch prompt by ID: {e}") from e

    def get_prompt_by_uuid(
        self, prompt_uuid: str, include_deleted: bool = False
    ) -> Optional[Dict]:
        query = "SELECT * FROM Prompts WHERE uuid = ?"
        params = [prompt_uuid]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by UUID {prompt_uuid}: {e}")
            raise DatabaseError(f"Failed fetch prompt by UUID: {e}") from e

    def get_prompt_by_name(
        self, name: str, include_deleted: bool = False
    ) -> Optional[Dict]:
        start_time = time.time()

        query = "SELECT * FROM Prompts WHERE name = ?"
        params = [name]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            found_result = dict(result) if result else None

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "get_by_name",
                    "found": "true" if found_result else "false",
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "get_by_name",
                    "status": "success",
                    "found": "true" if found_result else "false",
                },
            )

            return found_result
        except (DatabaseError, sqlite3.Error) as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={"operation": "get_by_name", "found": "false"},
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "get_by_name",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )

            logger.error(f"Error fetching prompt by name '{name}': {e}")
            raise DatabaseError(f"Failed fetch prompt by name: {e}") from e

    def list_prompts(
        self, page: int = 1, per_page: int = 10, include_deleted: bool = False
    ) -> Tuple[List[Dict], int, int, int]:
        start_time = time.time()

        if page < 1:
            raise ValueError("Page number must be >= 1")
        if per_page < 1:
            raise ValueError("Per page must be >= 1")
        offset = (page - 1) * per_page

        include_deleted_flag = 1 if include_deleted else 0

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM Prompts WHERE ? = 1 OR deleted = 0",
                    (include_deleted_flag,),
                )
                total_items = cursor.fetchone()[0]

                results_data = []
                if total_items > 0:
                    # `details` (Task 8b D2): the Library list canvas's
                    # secondary line/filter need the prompt's description
                    # without an N+1 per-row `fetch_keywords_for_prompt`-
                    # style fetch for a whole page -- this is a single extra
                    # TEXT column on the same query, not a second query.
                    query = """SELECT id, name, uuid, author, details, last_modified,
                                version, artifact_type,
                                CASE WHEN length(trim(coalesce(system_prompt, ''))) > 0 THEN 1 ELSE 0 END
                                    AS has_system_prompt,
                                CASE WHEN length(trim(coalesce(user_prompt, ''))) > 0 THEN 1 ELSE 0 END
                                    AS has_user_prompt
                                FROM Prompts
                                WHERE ? = 1 OR deleted = 0
                                ORDER BY last_modified DESC, id DESC
                                LIMIT ? OFFSET ?"""
                    cursor.execute(query, (include_deleted_flag, per_page, offset))
                    results_data = [dict(row) for row in cursor.fetchall()]

            total_pages = ceil(total_items / per_page) if total_items > 0 else 0

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "list_prompts",
                    "page": str(page),
                    "per_page": str(per_page),
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "list_prompts",
                    "status": "success",
                    "result_count": str(len(results_data)),
                    "total_items": str(total_items),
                },
            )

            return results_data, total_pages, page, total_items
        except (DatabaseError, sqlite3.Error) as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "list_prompts",
                    "page": str(page),
                    "per_page": str(per_page),
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "list_prompts",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )

            logger.error(f"Error listing prompts: {e}")
            raise DatabaseError(f"Failed to list prompts: {e}") from e

    def browse_prompts(
        self,
        *,
        query: str = "",
        collection_id: int | None = None,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int, int, int]:
        """Browse one exact, stable page of active local Prompts and Recipes.

        Args:
            query: Case-insensitive literal substring matched against name and details.
            collection_id: Optional positive local collection identifier.
            sort_by: ``last_modified`` or ``name``.
            sort_order: ``asc`` or ``desc``.
            page: Requested positive one-based page.
            page_size: Positive page size, capped at 100.

        Returns:
            A tuple of rows, total pages, resolved current page, and total items.

        Raises:
            TypeError: If a textual argument has the wrong type.
            ValueError: If an identifier, sort, page, or page size is invalid.
            DatabaseError: If SQLite cannot complete the browse operation.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if collection_id is not None and (
            type(collection_id) is not int
            or collection_id <= 0
            or collection_id > self._SQLITE_SIGNED_INTEGER_MAX
        ):
            raise ValueError(
                "collection_id must be a positive signed 64-bit integer or None."
            )
        if not isinstance(sort_by, str):
            raise TypeError("sort_by must be a string.")
        normalized_sort = sort_by.strip().lower()
        sort_column = self._PROMPT_BROWSE_SORT_COLUMNS.get(normalized_sort)
        if sort_column is None:
            raise ValueError("sort_by must be 'last_modified' or 'name'.")
        if not isinstance(sort_order, str):
            raise TypeError("sort_order must be a string.")
        normalized_order = sort_order.strip().lower()
        order_sql = self._PROMPT_BROWSE_SORT_ORDERS.get(normalized_order)
        if order_sql is None:
            raise ValueError("sort_order must be 'asc' or 'desc'.")
        if type(page) is not int or page <= 0:
            raise ValueError("page must be a positive integer.")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be a positive integer.")

        normalized_query = query.strip()
        page_size = min(page_size, 100)
        join_sql = ""
        conditions = ["p.deleted = 0"]
        params: list[Any] = []
        if collection_id is not None:
            join_sql = (
                "JOIN LocalPromptCollectionItems AS pci ON pci.prompt_id = p.id "
                "JOIN LocalPromptCollections AS pc "
                "ON pc.collection_id = pci.collection_id AND pc.deleted = 0"
            )
            conditions.append("pci.collection_id = ?")
            params.append(collection_id)
        if normalized_query:
            like_pattern = f"%{self._escape_library_prompt_like(normalized_query)}%"
            conditions.append(
                "(prompt_browse_lower(p.name) LIKE prompt_browse_lower(?) ESCAPE '\\' "
                "OR prompt_browse_lower(coalesce(p.details, '')) "
                "LIKE prompt_browse_lower(?) ESCAPE '\\')"
            )
            params.extend((like_pattern, like_pattern))

        where_sql = " AND ".join(conditions)
        try:
            with self.transaction() as conn:
                if collection_id is not None:
                    collection_table_count = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM sqlite_master
                        WHERE type = 'table' AND name IN (?, ?)
                        """,
                        (
                            "LocalPromptCollections",
                            "LocalPromptCollectionItems",
                        ),
                    ).fetchone()[0]
                    if collection_table_count != 2:
                        return [], 0, 1, 0
                if normalized_query or normalized_sort == "name":
                    conn.create_function(
                        "prompt_browse_lower", 1, str.lower, deterministic=True
                    )
                total_items = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM Prompts AS p {join_sql} "
                        f"WHERE {where_sql}",
                        tuple(params),
                    ).fetchone()[0]
                )
                total_pages = (
                    (total_items + page_size - 1) // page_size if total_items else 0
                )
                current_page = min(page, total_pages) if total_pages else 1
                rows = []
                if total_items:
                    offset = (current_page - 1) * page_size
                    cursor = conn.execute(
                        f"""
                        SELECT p.id, p.name, p.uuid, p.author, p.details,
                               p.last_modified, p.version, p.artifact_type,
                               CASE WHEN length(trim(coalesce(p.system_prompt, ''))) > 0
                                    THEN 1 ELSE 0 END AS has_system_prompt,
                               CASE WHEN length(trim(coalesce(p.user_prompt, ''))) > 0
                                    THEN 1 ELSE 0 END AS has_user_prompt
                        FROM Prompts AS p
                        {join_sql}
                        WHERE {where_sql}
                        ORDER BY {sort_column} {order_sql}, p.id {order_sql}
                        LIMIT ? OFFSET ?
                        """,
                        tuple(params + [page_size, offset]),
                    )
                    rows = [dict(row) for row in cursor.fetchall()]
            return rows, total_pages, current_page, total_items
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to browse prompts: {e}") from e

    # ============================= Library read seams (task-1337) =========================================
    #
    # Additive, read-only queries backing the local Library agent tools.
    # Same discipline as the Media/Notes seams: single-transaction count+page,
    # agent-safe projections (bounded details preview and section presence
    # flags only; full section text is reachable solely through the windowed
    # section reader), escaped-LIKE + tokenized safe-FTS search with
    # exact-name precedence, and exact keyword counts.

    _LIBRARY_PROMPT_PREVIEW_CHARS = 241
    _LIBRARY_PROMPT_KEYWORD_CAP = 20
    _LIBRARY_PROMPT_FTS_TOKEN_LIMIT = 20
    _LIBRARY_PROMPT_SECTIONS = (
        "details",
        "system_prompt",
        "user_prompt",
        "prompt_definition",
    )

    @staticmethod
    def _escape_library_prompt_like(value: str) -> str:
        """Escape LIKE metacharacters so user input matches literally."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _library_prompt_fts_query(cls, raw_query: str) -> Optional[str]:
        """Build a safe FTS5 MATCH query from raw user text (operators inert).

        The AND-of-quoted-tokens form, not a phrase: each token is
        double-quoted and they are space-joined, which is FTS5's implicit
        AND. Returns None when the input contains no usable tokens.
        """
        tokens = re.findall(r"\w+", raw_query, flags=re.UNICODE)
        if not tokens:
            return None
        tokens = tokens[: cls._LIBRARY_PROMPT_FTS_TOKEN_LIMIT]
        return " ".join(quote_fts5_token(token) for token in tokens)

    def _library_keywords_for_prompts(
        self, conn: sqlite3.Connection, prompt_ids: List[int]
    ) -> Dict[int, List[str]]:
        """Fetch active keywords for a page of prompt ids, grouped by prompt id."""
        if not prompt_ids:
            return {}
        placeholders = ",".join("?" * len(prompt_ids))
        query = f"""
            SELECT pkl.prompt_id, k.keyword
            FROM PromptKeywordLinks pkl
            JOIN PromptKeywordsTable k ON pkl.keyword_id = k.id
            WHERE pkl.prompt_id IN ({placeholders}) AND k.deleted = 0
            ORDER BY k.keyword COLLATE NOCASE
        """
        cursor = conn.execute(query, tuple(prompt_ids))
        keywords_by_prompt: Dict[int, List[str]] = {}
        for row in cursor.fetchall():
            keywords_by_prompt.setdefault(row["prompt_id"], []).append(row["keyword"])
        return keywords_by_prompt

    def _library_prompt_item(
        self, row: sqlite3.Row, keywords_by_prompt: Dict[int, List[str]]
    ) -> Dict[str, Any]:
        """Project a Prompts row into the agent-safe library item shape."""
        all_keywords = keywords_by_prompt.get(row["id"], [])
        visible = all_keywords[: self._LIBRARY_PROMPT_KEYWORD_CAP]
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "name": row["name"],
            "author": row["author"],
            "last_modified": row["last_modified"],
            "version": row["version"],
            "details_preview": row["details_preview"],
            "has_system_prompt": row["has_system_prompt"],
            "has_user_prompt": row["has_user_prompt"],
            "has_prompt_definition": row["has_prompt_definition"],
            "keywords": visible,
            "keyword_total": len(all_keywords),
            "keywords_truncated": len(all_keywords) > len(visible),
        }

    @classmethod
    def _library_prompt_page_columns(cls) -> str:
        # The preview bound is a class constant baked into the SQL text so
        # callers can use plain positional parameters throughout.
        return f"""
        id, uuid, name, author, last_modified, version,
        substr(coalesce(details, ''), 1, {cls._LIBRARY_PROMPT_PREVIEW_CHARS}) AS details_preview,
        CASE WHEN length(trim(coalesce(system_prompt, ''))) > 0 THEN 1 ELSE 0 END
            AS has_system_prompt,
        CASE WHEN length(trim(coalesce(user_prompt, ''))) > 0 THEN 1 ELSE 0 END
            AS has_user_prompt,
        CASE WHEN length(trim(coalesce(prompt_definition, ''))) > 0 THEN 1 ELSE 0 END
            AS has_prompt_definition
    """

    @staticmethod
    def _library_prompt_fts_fields(
        conn: sqlite3.Connection, prompt_id: int, fts_query: str
    ) -> List[str]:
        """Per-column FTS probes attributing a combined-FTS hit honestly.

        Only the columns present in ``prompts_fts`` are probed (author hits
        are not part of the Library matched-fields vocabulary and
        prompt_definition is not indexed).
        """
        matched: List[str] = []
        for column in ("name", "details", "system_prompt", "user_prompt"):
            cursor = conn.execute(
                "SELECT 1 FROM prompts_fts WHERE prompts_fts MATCH ? "
                "AND rowid = ? LIMIT 1",
                (f"{column} : {fts_query}", prompt_id),
            )
            if cursor.fetchone():
                matched.append(column)
        return matched

    def list_library_prompts_page(self, *, limit: int, offset: int) -> Dict[str, Any]:
        """Return one page of active library prompts plus the exact active total.

        Active means ``deleted = 0``. Ordering is stable:
        ``last_modified DESC, id DESC``. The count and the page are read in
        one transaction.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip (SQL OFFSET, not Python slicing).

        Returns:
            Dict with ``items`` (agent-safe projections) and ``total``.

        Raises:
            DatabaseError: If a database error occurs.
        """
        try:
            with self.transaction() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) AS count FROM Prompts WHERE deleted = 0"
                ).fetchone()["count"]
                cursor = conn.execute(
                    f"""
                    SELECT {self._library_prompt_page_columns()}
                    FROM Prompts
                    WHERE deleted = 0
                    ORDER BY last_modified DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = cursor.fetchall()
                keywords_by_prompt = self._library_keywords_for_prompts(
                    conn, [row["id"] for row in rows]
                )
            items = [self._library_prompt_item(row, keywords_by_prompt) for row in rows]
            return {"items": items, "total": total}
        except sqlite3.Error as e:
            logger.error(
                f"Error listing library prompts page (limit={limit}, offset={offset}): {e}"
            )
            raise DatabaseError(f"Failed to list library prompts page: {e}") from e

    def search_library_prompts_page(
        self, *, query: str, limit: int, offset: int
    ) -> Dict[str, Any]:
        """Search active library prompts, returning one page plus exact total.

        Match branches (OR, deduplicated by Prompts row): case-insensitive
        exact name, escaped-LIKE substring over name/details/system_prompt/
        user_prompt/prompt_definition, tokenized safe FTS over the same
        fields, and keyword substring via PromptKeywordLinks. Exact-name hits
        rank first, then recency, then id.

        Args:
            query: Raw user search text.
            limit: Maximum number of items to return.
            offset: Number of items to skip.

        Returns:
            Dict with ``items`` (library projections plus ``matched_fields``
            and ``matched_keywords``) and ``total``.

        Raises:
            DatabaseError: If a database error occurs.
        """
        like_pattern = f"%{self._escape_library_prompt_like(query)}%"
        fts_query = self._library_prompt_fts_query(query)
        keyword_branch = (
            "id IN (SELECT pkl.prompt_id FROM PromptKeywordLinks pkl "
            "JOIN PromptKeywordsTable k ON pkl.keyword_id = k.id "
            "WHERE k.deleted = 0 AND k.keyword LIKE ? ESCAPE '\\')"
        )
        section_columns = (
            ("details", 2),
            ("system_prompt", 3),
            ("user_prompt", 4),
            ("prompt_definition", 5),
        )

        branches = [
            "LOWER(name) = LOWER(?)",
            "name LIKE ? ESCAPE '\\'",
        ]
        params: List[Any] = [query, like_pattern]
        for column, _index in section_columns:
            branches.append(f"coalesce({column}, '') LIKE ? ESCAPE '\\'")
            params.append(like_pattern)
        fts_index: Optional[int] = None
        if fts_query is not None:
            fts_index = len(branches)
            branches.append(
                "id IN (SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?)"
            )
            params.append(fts_query)
        keyword_index = len(branches)
        branches.append(keyword_branch)
        params.append(like_pattern)

        where_clause = " OR ".join(f"({branch})" for branch in branches)
        hit_selects = ", ".join(
            f"({branch}) AS hit_{index}" for index, branch in enumerate(branches)
        )
        hit_params = list(params)

        try:
            with self.transaction() as conn:
                total = conn.execute(
                    f"SELECT COUNT(*) AS count FROM Prompts "
                    f"WHERE deleted = 0 AND ({where_clause})",
                    tuple(params),
                ).fetchone()["count"]
                cursor = conn.execute(
                    f"""
                    SELECT {self._library_prompt_page_columns()},
                           {hit_selects}
                    FROM Prompts
                    WHERE deleted = 0 AND ({where_clause})
                    ORDER BY (LOWER(name) = LOWER(?)) DESC,
                             last_modified DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    tuple(hit_params + params + [query, limit, offset]),
                )
                rows = cursor.fetchall()
                keywords_by_prompt = self._library_keywords_for_prompts(
                    conn, [row["id"] for row in rows]
                )
                # FTS can hit across token boundaries where no LIKE branch
                # fires; attribute those rows honestly with per-column probes.
                fts_only_fields: Dict[int, List[str]] = {}
                if fts_index is not None and fts_query is not None:
                    for row in rows:
                        literal_hit = any(
                            row[f"hit_{index}"]
                            for index in range(len(branches))
                            if index != fts_index
                        )
                        if row[f"hit_{fts_index}"] and not literal_hit:
                            fts_only_fields[row["id"]] = (
                                self._library_prompt_fts_fields(
                                    conn, row["id"], fts_query
                                )
                            )
            lowered_query = query.lower()
            items = []
            for row in rows:
                item = self._library_prompt_item(row, keywords_by_prompt)
                matched_fields = set()
                if row["hit_0"] or row["hit_1"]:
                    matched_fields.add("name")
                for column, index in section_columns:
                    if row[f"hit_{index}"]:
                        matched_fields.add(column)
                if fts_index is not None and row[f"hit_{fts_index}"]:
                    matched_fields.update(fts_only_fields.get(row["id"], []))
                if row[f"hit_{keyword_index}"]:
                    matched_fields.add("keywords")
                item["matched_fields"] = sorted(matched_fields)
                item["matched_keywords"] = [
                    keyword
                    for keyword in keywords_by_prompt.get(row["id"], [])
                    if lowered_query in keyword.lower()
                ][: self._LIBRARY_PROMPT_KEYWORD_CAP]
                items.append(item)
            return {"items": items, "total": total}
        except sqlite3.Error as e:
            logger.error(
                "Error searching library prompts "
                f"(query_chars={len(query)}, limit={limit}, offset={offset}): {e}"
            )
            raise DatabaseError(f"Failed to search library prompts: {e}") from e

    def get_library_prompt_overview(self, prompt_uuid: str) -> Optional[Dict[str, Any]]:
        """Return a bounded overview of one active prompt.

        Every present section is independently bounded: its exact total
        character length plus a 241-char preview. Full section text and
        version history are never included.

        Args:
            prompt_uuid: The prompt UUID to read.

        Returns:
            Dict with identity, ``version``, and a ``sections`` map; or None
            when no active prompt matches the UUID.

        Raises:
            DatabaseError: If a database error occurs.
        """
        selects = ", ".join(
            f"length(coalesce({column}, '')) AS {column}_total, "
            f"substr(coalesce({column}, ''), 1, {self._LIBRARY_PROMPT_PREVIEW_CHARS}) "
            f"AS {column}_preview"
            for column in self._LIBRARY_PROMPT_SECTIONS
        )
        try:
            with self.transaction() as conn:
                row = conn.execute(
                    f"""
                    SELECT uuid, name, author, last_modified, version, {selects}
                    FROM Prompts
                    WHERE uuid = ? AND deleted = 0
                    """,
                    (prompt_uuid,),
                ).fetchone()
            if row is None:
                return None
            sections = {}
            for column in self._LIBRARY_PROMPT_SECTIONS:
                total = row[f"{column}_total"] or 0
                if total > 0:
                    sections[column] = {
                        "total_chars": total,
                        "preview": row[f"{column}_preview"] or "",
                    }
            return {
                "uuid": row["uuid"],
                "name": row["name"],
                "author": row["author"],
                "last_modified": row["last_modified"],
                "version": row["version"],
                "sections": sections,
            }
        except sqlite3.Error as e:
            logger.error(
                f"Error reading library prompt overview (prompt_uuid={prompt_uuid!r}): {e}"
            )
            raise DatabaseError(f"Failed to read library prompt overview: {e}") from e

    def get_library_prompt_section(
        self, prompt_uuid: str, *, section: str, start: int, max_chars: int
    ) -> Optional[Dict[str, Any]]:
        """Return a windowed text segment of one prompt section.

        Reads only ``substr(<section>, start + 1, max_chars)`` and
        ``length(<section>)`` — never the whole section, never other
        sections, never version-history rows.

        Args:
            prompt_uuid: The prompt UUID to read.
            section: One of ``details``, ``system_prompt``, ``user_prompt``,
                ``prompt_definition``.
            start: Zero-based character offset into the section.
            max_chars: Maximum number of characters to return.

        Returns:
            Dict with identity, ``section``, ``version``, ``total_chars``,
            ``start``, ``returned_chars``, ``has_more``, and ``text``; or
            None when no active prompt matches the UUID.

        Raises:
            InputError: If ``section`` is not a known prompt section.
            DatabaseError: If a database error occurs.
        """
        if section not in self._LIBRARY_PROMPT_SECTIONS:
            raise InputError(
                f"Unknown prompt section {section!r}; expected one of "
                f"{', '.join(self._LIBRARY_PROMPT_SECTIONS)}."
            )
        try:
            with self.transaction() as conn:
                row = conn.execute(
                    f"""
                    SELECT uuid, name, version,
                           length(coalesce({section}, '')) AS total_chars,
                           substr(coalesce({section}, ''), ?, ?) AS text
                    FROM Prompts
                    WHERE uuid = ? AND deleted = 0
                    """,
                    (start + 1, max_chars, prompt_uuid),
                ).fetchone()
            if row is None:
                return None
            text = row["text"] or ""
            total_chars = row["total_chars"] or 0
            return {
                "uuid": row["uuid"],
                "name": row["name"],
                "section": section,
                "version": row["version"],
                "total_chars": total_chars,
                "start": start,
                "returned_chars": len(text),
                "has_more": start + len(text) < total_chars,
                "text": text,
            }
        except sqlite3.Error as e:
            logger.error(
                "Error reading library prompt section "
                f"(prompt_uuid={prompt_uuid!r}, section={section!r}, "
                f"start={start}, max_chars={max_chars}): {e}"
            )
            raise DatabaseError(f"Failed to read library prompt section: {e}") from e

    def fetch_prompt_details(
        self, prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False
    ) -> Optional[Dict]:
        prompt_data = None
        if isinstance(prompt_id_or_name_or_uuid, int):
            prompt_data = self.get_prompt_by_id(
                prompt_id_or_name_or_uuid, include_deleted
            )
        elif isinstance(prompt_id_or_name_or_uuid, str):
            try:  # Check if UUID
                uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                prompt_data = self.get_prompt_by_uuid(
                    prompt_id_or_name_or_uuid, include_deleted
                )
            except ValueError:  # Assume name
                prompt_data = self.get_prompt_by_name(
                    prompt_id_or_name_or_uuid, include_deleted
                )

        if not prompt_data:
            return None

        # Fetch keywords
        keywords = self.fetch_keywords_for_prompt(
            prompt_data["id"], include_deleted=include_deleted
        )  # Pass prompt_id
        prompt_data_dict = dict(prompt_data)
        prompt_data_dict["keywords"] = keywords
        return prompt_data_dict

    def fetch_all_keywords(self, include_deleted: bool = False) -> List[str]:
        query = "SELECT keyword FROM PromptKeywordsTable"
        if not include_deleted:
            query += " WHERE deleted = 0"
        query += " ORDER BY keyword COLLATE NOCASE"
        try:
            cursor = self.execute_query(query)
            return [row["keyword"] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all prompt keywords: {e}")
            raise DatabaseError("Failed to fetch all prompt keywords") from e

    def fetch_keywords_for_prompt(
        self, prompt_id: int, include_deleted: bool = False
    ) -> List[str]:
        # Note: include_deleted here refers to the keyword itself, not the link or prompt
        query = """SELECT k.keyword FROM PromptKeywordsTable k
                                             JOIN PromptKeywordLinks pkl ON k.id = pkl.keyword_id
                   WHERE pkl.prompt_id = ?"""
        params = [prompt_id]
        if not include_deleted:  # Filter for active keywords
            query += " AND k.deleted = 0"
        query += " ORDER BY k.keyword COLLATE NOCASE"
        try:
            cursor = self.execute_query(query, tuple(params))
            return [row["keyword"] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching keywords for prompt ID {prompt_id}: {e}")
            raise DatabaseError(
                f"Failed to fetch keywords for prompt {prompt_id}"
            ) from e

    def search_prompts(
        self,
        search_query: Optional[str],
        search_fields: Optional[
            List[str]
        ] = None,  # e.g. ['name', 'details', 'keywords']
        page: int = 1,
        results_per_page: int = 20,
        include_deleted: bool = False,
        fts_match_query: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Searches prompts using FTS.

        Args:
            search_query: Plain user search text. Every token must appear
                in ``prompts_fts``/``prompt_keywords_fts``
                (TASK-19558 -- each token is quoted with
                ``Utils.fts5_match_forms.build_and_match_query``, NOT used
                verbatim as a MATCH clause; FTS5 operators typed into it are
                inert, and a typed ``"`` no longer raises). The AND-of-tokens
                form, not a whole-query phrase: a prompt named "lore of the
                dragon reversed" is still found by ``dragon lore``.
                Unsearchable text -- ``None``, punctuation-only, or
                containing a NUL -- matches nothing instead of raising.
            search_fields: Fields to search; defaults to the standard text
                fields when ``search_query`` is set.
            page: 1-indexed page number.
            results_per_page: Page size.
            include_deleted: Whether to include soft-deleted prompts.
            fts_match_query: Optional pre-built FTS5 MATCH expression (e.g.
                Library keyword search's plural/singular-widened query,
                see ``library_fts_query.build_fts_match_query``) that
                overrides the MATCH clause built from ``search_query``.
                This is the ONLY seam through which a caller may supply
                FTS5 syntax; it must already be injection-safe.
        """
        start_time = time.time()

        if page < 1:
            raise ValueError("Page must be >= 1")
        if results_per_page < 1:
            raise ValueError("Results per page must be >= 1")

        if search_query and not search_fields:
            search_fields = [
                "name",
                "details",
                "system_prompt",
                "user_prompt",
                "author",
            ]
        elif not search_fields:
            search_fields = []

        offset = (page - 1) * results_per_page

        base_select = """SELECT p.*,
            CASE WHEN length(trim(coalesce(p.system_prompt, ''))) > 0 THEN 1 ELSE 0 END
                AS has_system_prompt,
            CASE WHEN length(trim(coalesce(p.user_prompt, ''))) > 0 THEN 1 ELSE 0 END
                AS has_user_prompt"""
        count_select = "SELECT COUNT(p.id)"
        from_clause = "FROM Prompts p"
        conditions = []
        params = []

        if not include_deleted:
            conditions.append("p.deleted = 0")

        # --- Robust FTS search using subqueries ---
        if search_query and search_fields:
            matching_prompt_ids = set()
            text_search_fields = {
                "name",
                "author",
                "details",
                "system_prompt",
                "user_prompt",
            }
            # Forward the caller-built MATCH expression when provided;
            # otherwise quote each of the user's tokens and AND them.
            # TASK-19558: the else-branch used to bind `search_query` RAW,
            # so a typed `"` raised OperationalError('unterminated string')
            # and a typed column filter/`OR` executed as FTS5 syntax.
            effective_match_query = (
                fts_match_query
                if fts_match_query
                else build_and_match_query(search_query)
            )
            # "" means the text cannot be searched at all (punctuation only,
            # or containing a NUL that SQLite truncates the bound parameter
            # at). `MATCH ''` is an FTS5 syntax error, so skip both FTS legs
            # and let the id-set stay empty rather than raising into the
            # prompt search box.
            if not effective_match_query:
                return [], 0

            # Search in prompt text fields
            if any(field in text_search_fields for field in search_fields):
                try:
                    cursor = self.execute_query(
                        "SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?",
                        (effective_match_query,),
                    )
                    matching_prompt_ids.update(
                        row["rowid"] for row in cursor.fetchall()
                    )
                except sqlite3.Error as e:
                    logging.opt(exception=True).error(
                        f"FTS search on prompts failed: {e}"
                    )
                    raise DatabaseError(f"FTS search on prompts failed: {e}") from e

            # Search in keywords
            if "keywords" in search_fields:
                try:
                    # 1. Find keyword IDs matching the query
                    kw_cursor = self.execute_query(
                        "SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?",
                        (effective_match_query,),
                    )
                    matching_keyword_ids = {
                        row["rowid"] for row in kw_cursor.fetchall()
                    }

                    # 2. Find prompt IDs linked to those keywords
                    if matching_keyword_ids:
                        placeholders = ",".join("?" * len(matching_keyword_ids))
                        link_cursor = self.execute_query(
                            f"SELECT DISTINCT prompt_id FROM PromptKeywordLinks WHERE keyword_id IN ({placeholders})",
                            tuple(matching_keyword_ids),
                        )
                        matching_prompt_ids.update(
                            row["prompt_id"] for row in link_cursor.fetchall()
                        )
                except sqlite3.Error as e:
                    logging.opt(exception=True).error(
                        f"FTS search on keywords failed: {e}"
                    )
                    raise DatabaseError(f"FTS search on keywords failed: {e}") from e

            if not matching_prompt_ids:
                return [], 0  # No matches found, short-circuit

            # Add the final ID list to the main query conditions
            id_placeholders = ",".join("?" * len(matching_prompt_ids))
            conditions.append(f"p.id IN ({id_placeholders})")
            params.extend(list(matching_prompt_ids))

        # --- Build and Execute Final Query ---
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_by_clause = "ORDER BY p.last_modified DESC, p.id DESC"

        try:
            # Get total count
            count_sql = f"{count_select} {from_clause} {where_clause}"
            total_matches = self.execute_query(count_sql, tuple(params)).fetchone()[0]

            results_list = []
            if total_matches > 0:
                # Get paginated results
                results_sql = f"{base_select} {from_clause} {where_clause} {order_by_clause} LIMIT ? OFFSET ?"
                paginated_params = tuple(params + [results_per_page, offset])
                results_cursor = self.execute_query(results_sql, paginated_params)
                results_list = [dict(row) for row in results_cursor.fetchall()]
                # Attach keywords to each result
                for res_dict in results_list:
                    res_dict["keywords"] = self.fetch_keywords_for_prompt(
                        res_dict["id"], include_deleted=False
                    )

            # Log success metrics
            duration = time.time() - start_time
            search_type = "full_text" if search_query else "list_all"
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "search_prompts",
                    "search_type": search_type,
                    "page": str(page),
                    "result_count": str(len(results_list)),
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "search_prompts",
                    "status": "success",
                    "search_type": search_type,
                    "total_matches": str(total_matches),
                    "searched_keywords": "true"
                    if "keywords" in search_fields
                    else "false",
                },
            )

            return results_list, total_matches
        except (DatabaseError, sqlite3.Error) as e:
            # Log error metrics
            duration = time.time() - start_time
            error_type = (
                "database_error" if isinstance(e, DatabaseError) else "sqlite_error"
            )
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "search_prompts",
                    "search_type": "error",
                    "page": str(page),
                    "result_count": "0",
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "search_prompts",
                    "status": "error",
                    "error_type": error_type,
                },
            )

            logging.opt(exception=True).error(f"DB error during prompt search: {e}")
            raise DatabaseError(f"Failed to search prompts: {e}") from e

    # --- Sync Log Access Methods ---
    @staticmethod
    def _validate_prompt_history_entity_uuid(entity_uuid: str) -> str:
        if not isinstance(entity_uuid, str) or not entity_uuid.strip():
            raise InputError("entity_uuid must be a non-empty string.")
        return entity_uuid

    @staticmethod
    def _validate_prompt_history_positive_int(
        value: int, *, name: str, maximum: int
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > maximum
        ):
            raise InputError(
                f"{name} must be a positive integer no greater than {maximum}."
            )
        return value

    @staticmethod
    def _decode_prompt_history_row(row: sqlite3.Row) -> Dict[str, Any]:
        row_dict = dict(row)
        timestamp = row_dict.get("timestamp")
        if isinstance(timestamp, datetime):
            row_dict["timestamp"] = timestamp.isoformat()
        raw_payload = row_dict.get("payload")
        row_dict["payload_error"] = None
        row_dict["raw_payload"] = None
        if raw_payload is not None:
            try:
                row_dict["payload"] = json.loads(raw_payload)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                row_dict["payload"] = None
                row_dict["payload_error"] = "malformed_json"
                row_dict["raw_payload"] = raw_payload
        return row_dict

    def get_prompt_history_count(self, entity_uuid: str) -> int:
        """Return the exact retained create/update count for one Prompt UUID."""
        validated_uuid = self._validate_prompt_history_entity_uuid(entity_uuid)
        try:
            with self.transaction() as conn:
                row = conn.execute(
                    self._PROMPT_HISTORY_COUNT_SQL, (validated_uuid,)
                ).fetchone()
            return int(row[0]) if row is not None else 0
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"Failed to count retained Prompt history for {validated_uuid}: {e}"
            )
            raise DatabaseError("Failed to count retained Prompt history") from e

    def get_prompt_history_entries(
        self,
        entity_uuid: str,
        page_size: int,
        before_change_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read one bounded retained Prompt history page and its predecessor."""
        validated_uuid = self._validate_prompt_history_entity_uuid(entity_uuid)
        validated_page_size = self._validate_prompt_history_positive_int(
            page_size,
            name="page_size",
            maximum=self._PROMPT_HISTORY_MAX_PAGE_SIZE,
        )
        if before_change_id is not None:
            before_change_id = self._validate_prompt_history_positive_int(
                before_change_id,
                name="before_change_id",
                maximum=self._SQLITE_SIGNED_INTEGER_MAX,
            )

        row_query = """
            SELECT *
            FROM sync_log
            WHERE entity = 'Prompts'
              AND entity_uuid = ?
              AND operation IN ('create', 'update')
        """
        row_params: List[Any] = [validated_uuid]
        if before_change_id is not None:
            row_query += " AND change_id < ?"
            row_params.append(before_change_id)
        row_query += " ORDER BY change_id DESC LIMIT ?"
        row_params.append(validated_page_size + 1)

        try:
            with self.transaction() as conn:
                count_row = conn.execute(
                    self._PROMPT_HISTORY_COUNT_SQL, (validated_uuid,)
                ).fetchone()
                raw_rows = conn.execute(row_query, tuple(row_params)).fetchall()
        except sqlite3.Error as e:
            logging.opt(exception=True).error(
                f"Failed to read retained Prompt history for {validated_uuid}: {e}"
            )
            raise DatabaseError("Failed to read retained Prompt history") from e

        decoded_rows = [self._decode_prompt_history_row(row) for row in raw_rows]
        has_more = len(decoded_rows) > validated_page_size
        items = decoded_rows[:validated_page_size]
        predecessor = decoded_rows[validated_page_size] if has_more else None
        next_before_change_id = items[-1]["change_id"] if has_more and items else None
        return {
            "items": items,
            "predecessor": predecessor,
            "total_count": int(count_row[0]) if count_row is not None else 0,
            "has_more": has_more,
            "next_before_change_id": next_before_change_id,
        }

    def restore_prompt_history_entry(
        self,
        entity_uuid: str,
        *,
        change_id: int,
        version: int,
        expected_version: int,
        snapshot_validator: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Restore one retained Prompt snapshot through the ordinary update path.

        The retained row and current Prompt are deliberately re-resolved after
        acquiring SQLite's write lock.  ``snapshot_validator`` is supplied by
        the source-service boundary so compatibility/capability validation runs
        *inside* that same transaction without teaching the DB layer about UI
        artifact policy.
        """
        validated_uuid = self._validate_prompt_history_entity_uuid(entity_uuid)
        validated_change_id = self._validate_prompt_history_positive_int(
            change_id,
            name="change_id",
            maximum=self._SQLITE_SIGNED_INTEGER_MAX,
        )
        validated_version = self._validate_prompt_history_positive_int(
            version,
            name="version",
            maximum=self._SQLITE_SIGNED_INTEGER_MAX,
        )
        validated_expected_version = self._normalize_expected_version(expected_version)
        if validated_expected_version is None:
            raise InputError("expected_version is required for retained restore.")
        if not callable(snapshot_validator):
            raise InputError("snapshot_validator must be callable.")

        unavailable = {
            "outcome": "snapshot_unavailable",
            "snapshot_unavailable": True,
            "no_change": False,
            "source_version": validated_version,
            "current_version": None,
            "new_version": None,
            "retained_current_keywords": False,
        }
        with self.transaction(immediate=True) as conn:
            snapshot_row = conn.execute(
                """
                SELECT *
                FROM sync_log
                WHERE entity = 'Prompts'
                  AND entity_uuid = ?
                  AND operation IN ('create', 'update')
                  AND change_id = ?
                  AND version = ?
                """,
                (validated_uuid, validated_change_id, validated_version),
            ).fetchone()
            if snapshot_row is None:
                return unavailable

            current_row = conn.execute(
                "SELECT * FROM Prompts WHERE uuid = ?", (validated_uuid,)
            ).fetchone()
            if current_row is None or current_row["deleted"]:
                return {
                    **unavailable,
                    "outcome": "current_unavailable",
                    "snapshot_unavailable": False,
                    "current_version": (
                        int(current_row["version"]) if current_row is not None else None
                    ),
                }
            current_version = int(current_row["version"])
            if current_version != validated_expected_version:
                raise ExpectedVersionConflictError(
                    "Prompt changed after it was opened.",
                    "Prompts",
                    int(current_row["id"]),
                )

            validated_snapshot = snapshot_validator(
                self._decode_prompt_history_row(snapshot_row)
            )
            if not isinstance(validated_snapshot, dict):
                raise InputError("snapshot_validator must return a mapping.")
            update_data = validated_snapshot.get("update_data")
            keywords_captured = validated_snapshot.get("keywords_captured")
            if not isinstance(update_data, dict) or type(keywords_captured) is not bool:
                raise InputError(
                    "snapshot_validator must return update_data and keywords_captured."
                )
            required_fields = (
                "name",
                "author",
                "details",
                "system_prompt",
                "user_prompt",
                "prompt_format",
                "prompt_schema_version",
                "prompt_definition",
                "artifact_type",
            )
            if any(field not in update_data for field in required_fields):
                raise InputError(
                    "Retained snapshot is missing restorable Prompt fields."
                )

            current_keywords = sorted(
                row["keyword"]
                for row in conn.execute(
                    """
                    SELECT keyword
                    FROM PromptKeywordsTable AS keyword_table
                    JOIN PromptKeywordLinks AS link
                      ON link.keyword_id = keyword_table.id
                    WHERE link.prompt_id = ? AND keyword_table.deleted = 0
                    """,
                    (int(current_row["id"]),),
                ).fetchall()
            )
            if keywords_captured:
                desired_keywords = self._canonicalize_prompt_keywords(
                    validated_snapshot.get("keywords")
                )
            else:
                desired_keywords = current_keywords

            candidate = dict(update_data)
            candidate["keywords"] = desired_keywords
            desired_definition = self._serialize_prompt_definition(
                validated_snapshot.get(
                    "durable_prompt_definition", candidate["prompt_definition"]
                )
            )
            desired_values = {
                "name": candidate["name"].strip()
                if isinstance(candidate["name"], str)
                else candidate["name"],
                "author": candidate["author"],
                "details": candidate["details"],
                "system_prompt": candidate["system_prompt"],
                "user_prompt": candidate["user_prompt"],
                "prompt_format": self._normalize_prompt_format(
                    candidate["prompt_format"]
                ),
                "prompt_schema_version": candidate["prompt_schema_version"],
                "prompt_definition": desired_definition,
                "artifact_type": self._normalize_artifact_type(
                    candidate["artifact_type"]
                ),
            }
            if (
                all(
                    current_row[field] == desired_value
                    for field, desired_value in desired_values.items()
                )
                and current_keywords == desired_keywords
            ):
                return {
                    "outcome": "no_change",
                    "snapshot_unavailable": False,
                    "no_change": True,
                    "source_version": validated_version,
                    "current_version": current_version,
                    "new_version": current_version,
                    "retained_current_keywords": not keywords_captured,
                }

            self.update_prompt_by_id(
                int(current_row["id"]),
                candidate,
                expected_version=current_version,
            )
            return {
                "outcome": "restored",
                "snapshot_unavailable": False,
                "no_change": False,
                "source_version": validated_version,
                "current_version": current_version,
                "new_version": current_version + 1,
                "retained_current_keywords": not keywords_captured,
            }

    def get_sync_log_entries(
        self, since_change_id: int = 0, limit: Optional[int] = None
    ) -> List[Dict]:
        query = "SELECT * FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
        params_list = [since_change_id]
        if limit is not None:
            query += " LIMIT ?"
            params_list.append(limit)
        try:
            cursor = self.execute_query(query, tuple(params_list))
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict.get("payload"):
                    try:
                        row_dict["payload"] = json.loads(row_dict["payload"])
                    except json.JSONDecodeError:
                        logging.warning(
                            f"Failed decode JSON payload for sync_log ID {row_dict.get('change_id')}"
                        )
                        row_dict["payload"] = None
                results.append(row_dict)
            return results
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching sync_log entries: {e}")
            raise DatabaseError("Failed to fetch sync_log entries") from e

    def delete_sync_log_entries(self, change_ids: List[int]) -> int:
        if not change_ids:
            return 0
        if not all(isinstance(cid, int) for cid in change_ids):
            raise ValueError("change_ids must be a list of integers.")
        placeholders = ",".join("?" * len(change_ids))
        query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"
        try:
            with self.transaction():  # Ensure commit happens
                cursor = self.execute_query(
                    query, tuple(change_ids), commit=False
                )  # commit handled by transaction
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries: {e}")
            raise DatabaseError("Failed to delete sync log entries") from e

    # --- Additional Query Methods ---
    def get_all_prompts(
        self, include_deleted: bool = False, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        """
        Get all prompts from the database.

        Args:
            include_deleted: Whether to include soft-deleted prompts
            limit: Maximum number of prompts to return
            offset: Number of prompts to skip

        Returns:
            List of prompt dictionaries
        """
        query = "SELECT * FROM Prompts"
        if not include_deleted:
            query += " WHERE deleted = 0"
        query += " ORDER BY name COLLATE NOCASE LIMIT ? OFFSET ?"

        try:
            cursor = self.execute_query(query, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all prompts: {e}")
            raise DatabaseError(f"Failed to fetch all prompts: {e}") from e

    def get_all_keywords(
        self, include_deleted: bool = False, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        """
        Get all keywords from the database.

        Args:
            include_deleted: Whether to include soft-deleted keywords
            limit: Maximum number of keywords to return
            offset: Number of keywords to skip

        Returns:
            List of keyword dictionaries with 'name' key
        """
        query = """
            SELECT id, keyword as name, uuid, last_modified, version, client_id 
            FROM PromptKeywordsTable
        """
        if not include_deleted:
            query += " WHERE deleted = 0"
        query += " ORDER BY keyword COLLATE NOCASE LIMIT ? OFFSET ?"

        try:
            cursor = self.execute_query(query, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all keywords: {e}")
            raise DatabaseError(f"Failed to fetch all keywords: {e}") from e

    def search_prompts_by_keyword(
        self, keyword: str, include_deleted: bool = False
    ) -> List[Dict]:
        """
        Search for prompts that have a specific keyword.

        Args:
            keyword: The keyword to search for
            include_deleted: Whether to include soft-deleted prompts

        Returns:
            List of prompt dictionaries that have the keyword
        """
        start_time = time.time()

        if not keyword or not keyword.strip():
            return []

        normalized_keyword = self._normalize_keyword(keyword)

        query = """
            SELECT DISTINCT p.* 
            FROM Prompts p
            JOIN PromptKeywordLinks pkl ON p.id = pkl.prompt_id
            JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
            WHERE pkw.keyword = ?
        """
        params = [normalized_keyword]

        if not include_deleted:
            query += " AND p.deleted = 0 AND pkw.deleted = 0"

        query += " ORDER BY p.name COLLATE NOCASE"

        try:
            cursor = self.execute_query(query, tuple(params))
            results = [dict(row) for row in cursor.fetchall()]

            # Attach keywords to each result
            for res_dict in results:
                res_dict["keywords"] = self.fetch_keywords_for_prompt(
                    res_dict["id"], include_deleted=False
                )

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={
                    "operation": "search_by_keyword",
                    "result_count": str(len(results)),
                },
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "search_by_keyword",
                    "status": "success",
                    "result_count": str(len(results)),
                },
            )

            return results
        except (DatabaseError, sqlite3.Error) as e:
            # Log error metrics
            duration = time.time() - start_time
            error_type = (
                "database_error" if isinstance(e, DatabaseError) else "sqlite_error"
            )
            log_histogram(
                "prompts_db_operation_duration",
                duration,
                labels={"operation": "search_by_keyword", "result_count": "0"},
            )
            log_counter(
                "prompts_db_operation_count",
                labels={
                    "operation": "search_by_keyword",
                    "status": "error",
                    "error_type": error_type,
                },
            )

            logger.error(f"Error searching prompts by keyword '{keyword}': {e}")
            raise DatabaseError(f"Failed to search prompts by keyword: {e}") from e

    def search_prompts_by_text(
        self, search_text: str, include_deleted: bool = False
    ) -> List[Dict]:
        """
        Full-text search across prompt content.
        Searches in: name, author, details, system_prompt, user_prompt

        Args:
            search_text: Plain user text. Every token is quoted individually
                and the tokens are AND-ed (``build_and_match_query``), so all
                of them must appear but they need not be adjacent -- NOT a
                phrase, and FTS5 operators in it are inert.
            include_deleted: Whether to include soft-deleted prompts

        Returns:
            List of prompt dictionaries matching the search
        """
        if not search_text or not search_text.strip():
            return []

        try:
            # Use FTS to find matching prompt IDs
            # TASK-19558: `search_text` is plain user text; quote each of
            # its tokens and AND them rather than binding it as an FTS5
            # expression (and rather than one whole-query phrase, which
            # would halve recall -- see `build_and_match_query`).
            match_expression = build_and_match_query(search_text)
            if not match_expression:
                return []
            cursor = self.execute_query(
                "SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?",
                (match_expression,),
            )
            matching_ids = [row["rowid"] for row in cursor.fetchall()]

            if not matching_ids:
                return []

            # Build query for full prompt data
            placeholders = ",".join("?" * len(matching_ids))
            query = f"SELECT * FROM Prompts WHERE id IN ({placeholders})"
            params = list(matching_ids)

            if not include_deleted:
                query += " AND deleted = 0"

            query += " ORDER BY name COLLATE NOCASE"

            cursor = self.execute_query(query, tuple(params))
            results = [dict(row) for row in cursor.fetchall()]

            # Attach keywords to each result
            for res_dict in results:
                res_dict["keywords"] = self.fetch_keywords_for_prompt(
                    res_dict["id"], include_deleted=False
                )

            return results
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error in full-text search for '{search_text}': {e}")
            raise DatabaseError(f"Failed to search prompts by text: {e}") from e

    def search_prompts_by_content(
        self, search_text: str, include_deleted: bool = False
    ) -> List[Dict]:
        """
        Alias for search_prompts_by_text for backward compatibility.
        Full-text search across prompt content.

        Args:
            search_text: The text to search for
            include_deleted: Whether to include soft-deleted prompts

        Returns:
            List of prompt dictionaries matching the search
        """
        return self.search_prompts_by_text(search_text, include_deleted)

    def get_prompt_details(
        self, prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False
    ) -> Optional[Dict]:
        """
        Get detailed information about a prompt including its keywords.
        This is an alias for fetch_prompt_details for backward compatibility.

        Args:
            prompt_id_or_name_or_uuid: Prompt identifier (ID, name, or UUID)
            include_deleted: Whether to include soft-deleted prompts

        Returns:
            Dictionary with prompt data and keywords, or None if not found
        """
        return self.fetch_prompt_details(prompt_id_or_name_or_uuid, include_deleted)

    def _add_keyword_with_retry(
        self, keyword_text: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Internal method to add a keyword with retry logic for concurrent access.

        Args:
            keyword_text: The keyword text to add

        Returns:
            Tuple of (keyword_id, keyword_uuid)
        """
        max_retries = 5
        retry_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                return self._add_keyword_full(keyword_text)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Database locked on add_keyword attempt {attempt + 1}, retrying in {retry_delay}s"
                    )
                    import time

                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
            except Exception:
                raise

    def execute_query_with_retry(
        self, query: str, params: tuple = None, *, commit: bool = False
    ) -> sqlite3.Cursor:
        """
        Execute a query with retry logic for database lock errors.

        Args:
            query: SQL query to execute
            params: Query parameters
            commit: Whether to commit after execution

        Returns:
            Cursor object
        """
        max_retries = 5
        retry_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                return self.execute_query(query, params, commit=commit)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Database locked on query attempt {attempt + 1}, retrying in {retry_delay}s"
                    )
                    import time

                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
            except Exception:
                raise

    def export_keywords_to_csv(
        self, file_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Export keywords to CSV file.
        This is a convenience method that calls the standalone function.

        Args:
            file_path: Optional path for the CSV file. If not provided, generates one.

        Returns:
            Tuple of (status_message, file_path)
        """
        # Call the standalone function directly
        return export_prompt_keywords_to_csv(self, file_path)

    def update_prompt(self, prompt_id: int, **kwargs) -> bool:
        """
        Update a prompt by ID.
        This is a convenience method for update_prompt_by_id.

        Args:
            prompt_id: The ID of the prompt to update
            **kwargs: Fields to update (name, author, details, system_prompt, user_prompt)

        Returns:
            True if update was successful, False otherwise
        """
        try:
            uuid, msg = self.update_prompt_by_id(prompt_id, kwargs)
            return uuid is not None
        except Exception as e:
            logger.warning(f"Failed to update prompt {prompt_id}: {e}")
            return False

    def delete_prompt(self, prompt_id_or_name_or_uuid: Union[int, str]) -> bool:
        """
        Delete a prompt (soft delete).
        This is an alias for soft_delete_prompt.

        Args:
            prompt_id_or_name_or_uuid: Prompt identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            return self.soft_delete_prompt(prompt_id_or_name_or_uuid)
        except Exception as e:
            logger.warning(f"Failed to delete prompt {prompt_id_or_name_or_uuid}: {e}")
            return False

    def close(self) -> None:
        """Alias for close_connection() to maintain consistency with BaseDB."""
        self.close_connection()

    def vacuum(self) -> None:
        """Vacuum the database to reclaim unused space and optimize performance."""
        if self.is_memory_db:
            logger.debug("Skipping vacuum for in-memory database")
            return

        try:
            conn = self._get_thread_connection()
            # Vacuum must be run outside of a transaction
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.isolation_level = ""  # Restore default
            logger.info(f"Successfully vacuumed database: {self.db_path_str}")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise DatabaseError(f"Vacuum failed: {e}") from e


# =========================================================================
# Standalone Functions (REQUIRE db_instance passed explicitly)
# =========================================================================
# These functions now operate on a PromptsDatabase instance.


def add_or_update_prompt(
    db_instance: PromptsDatabase,
    name: str,
    author: Optional[str],
    details: Optional[str],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    prompt_format: Optional[str] = None,
    prompt_schema_version: Optional[int] = None,
    prompt_definition: Optional[Any] = None,
    artifact_type: Optional[str] = None,
) -> Tuple[Optional[int], Optional[str], str]:
    """
    Adds a new prompt or updates an existing one (identified by name).
    If the prompt exists (even if soft-deleted), it will be updated/undeleted.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
    # `add_prompt` with overwrite=True handles both add and update logic.
    return db_instance.add_prompt(
        name=name,
        author=author,
        details=details,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        keywords=keywords,
        overwrite=True,  # Key change: always overwrite/update if exists
        prompt_format=prompt_format,
        prompt_schema_version=prompt_schema_version,
        prompt_definition=prompt_definition,
        artifact_type=artifact_type,
    )


def load_prompt_details_for_ui(
    db_instance: PromptsDatabase, prompt_name: str
) -> Tuple[str, str, str, str, str, str]:
    """
    Loads prompt details for UI display, fetching by name.
    Returns empty strings if not found.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
    if not prompt_name:
        return "", "", "", "", "", ""

    details_dict = db_instance.fetch_prompt_details(
        prompt_name, include_deleted=False
    )  # Fetch active by name
    if details_dict:
        return (
            details_dict.get("name", ""),
            details_dict.get("author", "") or "",  # Ensure empty string if None
            details_dict.get("details", "") or "",
            details_dict.get("system_prompt", "") or "",
            details_dict.get("user_prompt", "") or "",
            ", ".join(details_dict.get("keywords", [])),  # keywords should be a list
        )
    return "", "", "", "", "", ""


def export_prompt_keywords_to_csv(
    db_instance: PromptsDatabase, file_path: Optional[str] = None
) -> Tuple[str, str]:
    import csv
    import tempfile
    import os
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")

    logging.debug(f"export_prompt_keywords_to_csv from DB: {db_instance.db_path_str}")
    try:
        # If no file path provided, generate one
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(
                temp_dir, f"prompt_keywords_export_{timestamp}.csv"
            )

        # Query to get keywords with associated prompt info (names, authors, counts)
        # This requires joining Prompts, PromptKeywordsTable, PromptKeywordLinks
        query = """
                SELECT
                    pkw.keyword,
                    GROUP_CONCAT(DISTINCT p.name) AS prompt_names,
                    COUNT(DISTINCT p.id) AS num_prompts,
                    GROUP_CONCAT(DISTINCT p.author) AS authors
                FROM PromptKeywordsTable pkw
                         LEFT JOIN PromptKeywordLinks pkl ON pkw.id = pkl.keyword_id
                         LEFT JOIN Prompts p ON pkl.prompt_id = p.id AND p.deleted = 0 /* Only count links to active prompts */
                WHERE pkw.deleted = 0 /* Only export active keywords */
                GROUP BY pkw.id, pkw.keyword
                ORDER BY pkw.keyword COLLATE NOCASE \
                """
        cursor = db_instance.execute_query(query)
        results = cursor.fetchall()

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Keyword", "Associated Prompts", "Number of Prompts", "Authors"]
            )
            for row in results:
                writer.writerow(
                    [
                        row["keyword"],
                        row["prompt_names"] or "",
                        row["num_prompts"],
                        row["authors"] or "",
                    ]
                )

        status_msg = (
            f"Successfully exported {len(results)} active prompt keywords to CSV."
        )
        logging.info(status_msg)
        return status_msg, file_path

    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Database error exporting keywords: {e}"
        logging.opt(exception=True).error(error_msg)
        return error_msg, "None"
    except Exception as e:
        error_msg = f"Error exporting keywords: {e}"
        logging.opt(exception=True).error(error_msg)
        return error_msg, "None"


def view_prompt_keywords_markdown(db_instance: PromptsDatabase) -> str:
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
    logging.debug(f"view_prompt_keywords_markdown from DB: {db_instance.db_path_str}")
    try:
        query = """
                SELECT pkw.keyword, COUNT(DISTINCT pkl.prompt_id) as prompt_count
                FROM PromptKeywordsTable pkw
                         LEFT JOIN PromptKeywordLinks pkl ON pkw.id = pkl.keyword_id
                         LEFT JOIN Prompts p ON pkl.prompt_id = p.id AND p.deleted = 0
                WHERE pkw.deleted = 0
                GROUP BY pkw.id, pkw.keyword
                ORDER BY pkw.keyword COLLATE NOCASE \
                """
        cursor = db_instance.execute_query(query)
        keywords_data = cursor.fetchall()

        if keywords_data:
            keyword_list_md = [
                f"- {row['keyword']} ({row['prompt_count']} active prompts)"
                for row in keywords_data
            ]
            return "### Current Active Prompt Keywords:\n" + "\n".join(keyword_list_md)
        return "No active keywords found."
    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Error retrieving keywords for markdown view: {e}"
        logging.opt(exception=True).error(error_msg)
        return error_msg


def export_prompts_formatted(
    db_instance: PromptsDatabase,
    export_format: str = "csv",  # 'csv' or 'markdown'
    filter_keywords: Optional[List[str]] = None,
    include_system: bool = True,
    include_user: bool = True,
    include_details: bool = True,
    include_author: bool = True,
    include_associated_keywords: bool = True,  # Renamed for clarity
    markdown_template_name: Optional[str] = "Basic Template",  # Name of template
) -> Tuple[str, str]:
    import csv
    import tempfile
    import os
    import zipfile  # For markdown if multiple files
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")

    logging.debug(
        f"export_prompts_formatted (format: {export_format}) from DB: {db_instance.db_path_str}"
    )

    # --- Fetch Prompts Data ---
    # Build base query parts
    select_fields = ["p.id", "p.name", "p.uuid"]  # Always include id, name, uuid
    if include_author:
        select_fields.append("p.author")
    if include_details:
        select_fields.append("p.details")
    if include_system:
        select_fields.append("p.system_prompt")
    if include_user:
        select_fields.append("p.user_prompt")

    query_sql = f"SELECT DISTINCT {', '.join(select_fields)} FROM Prompts p"
    query_params = []

    # Keyword filtering
    if filter_keywords and len(filter_keywords) > 0:
        normalized_filter_keywords = [
            db_instance._normalize_keyword(k)
            for k in filter_keywords
            if k and k.strip()
        ]
        if normalized_filter_keywords:
            placeholders = ",".join(["?"] * len(normalized_filter_keywords))
            query_sql += f"""
                JOIN PromptKeywordLinks pkl ON p.id = pkl.prompt_id
                JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                WHERE p.deleted = 0 AND pkw.deleted = 0 AND pkw.keyword IN ({placeholders})
            """
            query_params.extend(normalized_filter_keywords)
        else:  # No valid filter keywords, so just filter active prompts
            query_sql += " WHERE p.deleted = 0"
    else:  # No keyword filter, just active prompts
        query_sql += " WHERE p.deleted = 0"

    query_sql += " ORDER BY p.name COLLATE NOCASE"

    try:
        cursor = db_instance.execute_query(query_sql, tuple(query_params))
        prompts_data = [dict(row) for row in cursor.fetchall()]

        if not prompts_data:
            return "No prompts found matching the criteria for export.", "None"

        # Fetch associated keywords for each prompt if needed
        if include_associated_keywords:
            for prompt_dict in prompts_data:
                prompt_dict["keywords_list"] = db_instance.fetch_keywords_for_prompt(
                    prompt_dict["id"], include_deleted=False
                )

        # --- Perform Export ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = "None"

        if export_format == "csv":
            temp_csv_file = os.path.join(
                tempfile.gettempdir(), f"prompts_export_{timestamp}.csv"
            )
            header_row = ["Name", "UUID"]  # Start with common fields
            if include_author:
                header_row.append("Author")
            if include_details:
                header_row.append("Details")
            if include_system:
                header_row.append("System Prompt")
            if include_user:
                header_row.append("User Prompt")
            if include_associated_keywords:
                header_row.append("Keywords")

            with open(temp_csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header_row)
                for p_data in prompts_data:
                    row_to_write = [p_data["name"], p_data["uuid"]]
                    if include_author:
                        row_to_write.append(p_data.get("author", ""))
                    if include_details:
                        row_to_write.append(p_data.get("details", ""))
                    if include_system:
                        row_to_write.append(p_data.get("system_prompt", ""))
                    if include_user:
                        row_to_write.append(p_data.get("user_prompt", ""))
                    if include_associated_keywords:
                        row_to_write.append(", ".join(p_data.get("keywords_list", [])))
                    writer.writerow(row_to_write)
            output_file_path = temp_csv_file
            status_msg = f"Successfully exported {len(prompts_data)} prompts to CSV."

        elif export_format == "markdown":
            temp_zip_dir = tempfile.mkdtemp()
            zip_file_path = os.path.join(
                tempfile.gettempdir(), f"prompts_export_markdown_{timestamp}.zip"
            )

            templates = {
                "Basic Template": """# {name} ({uuid})
{author_section}
{details_section}
{system_section}
{user_section}
{keywords_section}
""",
                "Detailed Template": """# {name}
**UUID**: {uuid}

## Author
{author_section}

## Description
{details_section}

## System Prompt
```
{system_prompt_content}
```

## User Prompt
```
{user_prompt_content}
```

## Keywords
{keywords_section}
""",
            }
            chosen_template_str = templates.get(
                markdown_template_name, templates["Basic Template"]
            )

            with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for p_data in prompts_data:
                    author_sec = (
                        f"**Author**: {p_data['author']}"
                        if include_author and p_data.get("author")
                        else ""
                    )
                    details_sec = (
                        f"**Details**: {p_data['details']}"
                        if include_details and p_data.get("details")
                        else ""
                    )
                    system_sec = (
                        f"**System Prompt**:\n```\n{p_data['system_prompt']}\n```"
                        if include_system and p_data.get("system_prompt")
                        else ""
                    )
                    user_sec = (
                        f"**User Prompt**:\n```\n{p_data['user_prompt']}\n```"
                        if include_user and p_data.get("user_prompt")
                        else ""
                    )
                    keywords_sec = (
                        f"**Keywords**: {', '.join(p_data['keywords_list'])}"
                        if include_associated_keywords and p_data.get("keywords_list")
                        else ""
                    )

                    md_content = chosen_template_str.format(
                        name=p_data["name"],
                        uuid=p_data["uuid"],
                        author_section=author_sec,
                        details_section=details_sec,
                        system_section=system_sec,  # For Basic Template direct injection
                        system_prompt_content=p_data.get(
                            "system_prompt", ""
                        ),  # For Detailed Template
                        user_section=user_sec,  # For Basic Template direct injection
                        user_prompt_content=p_data.get(
                            "user_prompt", ""
                        ),  # For Detailed Template
                        keywords_section=keywords_sec,
                    ).strip()  # Clean up extra newlines if sections are empty

                    safe_filename = re.sub(r"[^\w\-_ \.]", "_", p_data["name"]) + ".md"
                    md_file_path_in_zip_dir = os.path.join(temp_zip_dir, safe_filename)
                    with open(
                        md_file_path_in_zip_dir, "w", encoding="utf-8"
                    ) as md_file:
                        md_file.write(md_content)
                    zipf.write(md_file_path_in_zip_dir, arcname=safe_filename)

            output_file_path = zip_file_path
            status_msg = f"Successfully exported {len(prompts_data)} prompts to Markdown in a ZIP file."
        else:
            raise ValueError(
                f"Unsupported export_format: {export_format}. Must be 'csv' or 'markdown'."
            )

        logging.info(status_msg)
        return status_msg, output_file_path

    except (DatabaseError, sqlite3.Error, ValueError) as e:
        error_msg = f"Error exporting prompts: {e}"
        logging.opt(exception=True).error(error_msg)
        return error_msg, "None"
    except Exception as e:  # Catch any other unexpected error
        error_msg = f"Unexpected error exporting prompts: {e}"
        logging.opt(exception=True).error(error_msg)
        return error_msg, "None"


# Import after the legacy standalone surface is defined. Importing this submodule
# executes Prompt_Management.__init__, whose compatibility adapters import the
# database class and standalone helpers above.
from ..Prompt_Management.prompt_batch_models import (  # noqa: E402
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    PromptDeleteReceiptEntry,
    PromptRestoreResultEntry,
)
