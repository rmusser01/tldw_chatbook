"""SQLite handles owned and retired by the elected writer's storage thread."""

from __future__ import annotations

import sqlite3
import stat
import threading
from contextlib import closing, contextmanager
from pathlib import Path

from tldw_chatbook.Backup_Recovery import bootstrap, participants
from tldw_chatbook.Utils.platform_files import os

from .profile_writer import ProfileWriterOutcomeUnknown, _StaleWriterResource


def _artifacts(path: Path) -> tuple:
    """Check every named SQLite artifact before borrowing an idle connection."""
    from .private_sqlite import _SIDECAR_SUFFIXES

    paths = (path, *(Path(str(path) + suffix) for suffix in _SIDECAR_SUFFIXES))
    if os.name == "nt":
        observed = os.stat_many_for_admission(paths)
        samples = [observed[name] for name in paths]
    else:
        samples = []
        for name in paths:
            try:
                samples.append((os.stat(name, follow_symlinks=False), None))
            except FileNotFoundError:
                samples.append(None)
    result = []
    for sample in samples:
        if sample is None:
            result.append(None)
            continue
        info, descriptor = sample
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise bootstrap.RecoveryRequired("profile_writer_database_unsafe")
        result.append((info.st_dev, info.st_ino, descriptor))
    return tuple(result)


def _schema(connection: sqlite3.Connection) -> tuple:
    with closing(connection.execute("PRAGMA main.schema_version")) as cookie:
        version = cookie.fetchone()[0]
    try:
        with closing(connection.execute("SELECT max(version) FROM main.schema_version")) as versions:
            return version, versions.fetchone()[0]
    except sqlite3.OperationalError as error:
        if error.args == ("no such table: main.schema_version",):
            return version, None
        raise


class WriterSQLiteConnection:
    """One process-owned native resource, independent of any command transaction."""

    private_policy = "db.base"

    def __init__(self):
        self.database = None
        self.connection = self.lease = None
        self.selection = self.artifacts = self.schema = None
        self.failed = False

    def current_connection(self):
        return getattr(self.database._thread_local, "conn", None)

    def close_connection(self):
        if self.artifacts != _artifacts(self.database.db_path):
            raise ProfileWriterOutcomeUnknown("writer database artifacts changed before retirement")
        with closing(self.connection.execute("PRAGMA main.wal_checkpoint(FULL)")) as cursor:
            busy, log, checkpointed = cursor.fetchone()
        if busy or log != checkpointed:
            raise ProfileWriterOutcomeUnknown("writer database checkpoint incomplete")
        self.database.close()

    def bind(self, database, connection):
        from .private_sqlite import _ordinary_connections

        self.database, self.connection = database, connection
        self.lease = _ordinary_connections.get(connection)
        self.idle_lease()

    def idle_lease(self):
        """Positively identify this one native borrower and its creating thread."""
        from .private_sqlite import _ordinary_connections, _validated_owner_policy

        if self.failed:
            raise ProfileWriterOutcomeUnknown("writer connection retirement is uncertain")
        if self.database is None:
            return None
        participant = participants.repository_participant(self.database)
        if (self.connection is None or self.lease is None
            or self.current_connection() is not self.connection
            or _ordinary_connections.get(self.connection) is not self.lease
            or participant.connections.get(self.connection) is not self.lease
            or self.lease.resource_policy is not _validated_owner_policy(self.private_policy)
            or self.lease.resource_path != self.database.db_path
            or self.lease.resource_participant is not participant
            or self.lease.resource_thread is not threading.current_thread()
            or self.lease.resource_close_failed
            or sqlite3.Connection.in_transaction.__get__(self.connection)):
            raise ProfileWriterOutcomeUnknown("writer connection did not settle")
        return self.lease

    def retire(self) -> None:
        """Checkpoint and close on the opener thread before releasing native admission."""
        if self.idle_lease() is None:
            return
        self.failed = True  # A failed native close is never retried.
        try:
            self.close_connection()
            if self.current_connection() is not None:
                raise ProfileWriterOutcomeUnknown("writer database close incomplete")
            try:
                sqlite3.Connection.in_transaction.__get__(self.connection)
            except sqlite3.ProgrammingError:
                pass  # The original native handle itself is closed.
            else:
                raise ProfileWriterOutcomeUnknown("writer database connection still live")
            self.database = self.connection = self.lease = None
            self.failed = False
        except Exception as error:
            raise ProfileWriterOutcomeUnknown("writer database connection did not retire") from error


class AgentRunsWriterDatabase(WriterSQLiteConnection):
    def _open(self, path, selection):
        from .AgentRuns_DB import AgentRunsDB

        database = AgentRunsDB(path, reconcile_on_init=False)
        self.bind(database, database._thread_local.conn)
        self.artifacts = _artifacts(path)
        if self.artifacts[0] is None:
            raise ProfileWriterOutcomeUnknown("writer database disappeared after open")
        self.selection = selection
        self.schema = None

    @contextmanager
    def command(self, profile: Path, selection: tuple, *, opening: bool):
        """Use the original counted scope and release it completely after dispatch."""
        path = profile / "agent_runs.db"
        generation = selection, bootstrap._admission_epoch
        if self.database is not None:
            self.idle_lease()
            try:
                artifacts = _artifacts(path)
            except (OSError, ValueError, bootstrap.RecoveryRequired) as error:
                raise _StaleWriterResource("profile_writer_database_unsafe") from error
            if self.artifacts != artifacts:
                raise _StaleWriterResource("profile_writer_database_changed")
            if self.selection != generation:
                self.retire()
        if self.database is None:
            self._open(path, generation)
            opening = False  # The stock constructor has run its initialization.
        with participants.core_operation(self.database):
            schema = _schema(self.connection)
            if opening or self.schema is not None and self.schema != schema:
                # Every explicit stock opener and any external schema change
                # runs the original migration/repair before another operation.
                self.database.receipt_capability_available = False
                self.database._initialize_schema()
                schema = _schema(self.connection)
            self.schema = schema
            yield self.database


class ChaChaWriterDatabase(WriterSQLiteConnection):
    """Own one connection through schema checking, the domain command and close."""

    private_policy = "db.chachanotes.primary"

    def current_connection(self):
        return getattr(self.database._local, "conn", None)

    def close_connection(self):
        # The stock close runs its own TRUNCATE checkpoint before releasing SQLite.
        self.database.close_connection()

    @contextmanager
    def command(self, database):
        if self.database is not None or self.failed:
            raise ProfileWriterOutcomeUnknown("ChaChaNotes writer connection did not retire")
        self.bind(database, database.get_connection())
        try:
            yield self.connection
        finally:
            # Windows permits file replacement between ordinary ChaChaNotes calls.
            # The physical handle must retire before the client gets its receipt.
            self.retire()
