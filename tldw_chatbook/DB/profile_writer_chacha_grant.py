"""Physical retirement of an original, caller-owned ChaChaNotes transaction."""

from contextlib import closing, contextmanager

from ..Backup_Recovery.bootstrap import RecoveryRequired
from .profile_writer import ProfileWriterOutcomeUnknown
from .profile_writer_grant import _Grant, _local_chacha_transaction


class ChaChaTransactionGrant(_Grant):
    owner_id = private_policy = "db.chachanotes.primary"

    @staticmethod
    def profile(database):
        return database._profile_writer_profile

    @staticmethod
    def connection(database):
        return getattr(database._local, "conn", None)

    @staticmethod
    def source_identity(database, expected):
        return database._profile_writer_path == database.db_path and database._profile_writer_file == expected

    @staticmethod
    def schema(database, connection):
        with closing(connection.execute("PRAGMA main.schema_version")) as cursor:
            return cursor.fetchone()[0], database._get_db_version(connection)

    def _connection(self, conn):
        lease = super()._connection(conn)
        if (self.database._local.profile_writer_file_identity != self.identity
            or not self.database._connection_quiescence.is_registered(conn)):
            raise RecoveryRequired("profile_writer_grant_connection_changed")
        return lease

    @staticmethod
    def close(database):
        database._close_connection_handle(
            database._local.conn, database._local, database._connection_quiescence, strict=True
        )

    def retire(self, connection):
        if self.database.managed_transaction_depth():
            raise ProfileWriterOutcomeUnknown("local Notes transaction remains managed")
        super().retire(connection)
        if self.prior is None and (self.connection(self.database) is not None
                                   or self.database._connection_quiescence.is_registered(connection)):
            self.retired = False
            raise ProfileWriterOutcomeUnknown("local Notes connection remains registered")


@contextmanager
def granted_transaction(transaction, kind, require_source):
    """Release the file only after the original managed transaction has finished."""
    with _local_chacha_transaction(transaction.db, kind) as grant:
        require_source()  # The original receiver and target must survive the IPC wait.
        transaction.writer_grant = grant
        try:
            with transaction as cursor:
                yield cursor
        finally:
            if transaction.cursor is not None or transaction.connection_use_token is not None:
                grant.uncertain()
            if grant.conn is not None and not grant.ambiguous:
                grant.retire(grant.conn)
