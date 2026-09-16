"""Real SQLite behavior for the Windows held-descriptor snapshot route."""

import contextlib
import sqlite3

import pytest

from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Utils.platform_files import os as native_os


class _WindowsRoute:
    """Select only the Windows reader while retaining this host's real IO."""

    name = "nt"

    def __getattr__(self, name):
        return getattr(native_os, name)


class _NativeIdentityProjection:
    """Model a native identity namespace while retaining real file operations."""

    def __getattr__(self, name):
        return getattr(native_os, name)

    @staticmethod
    def _project(info):
        values = list(info)
        values[2] = info.st_dev ^ (1 << 32)
        return native_os.stat_result(
            values,
            {"st_mtime_ns": info.st_mtime_ns, "st_ctime_ns": info.st_ctime_ns},
        )

    def fstat(self, descriptor):
        return self._project(native_os.fstat(descriptor))

    def stat(self, path, *, follow_symlinks=True):
        assert not follow_symlinks
        return self._project(native_os.stat(path, follow_symlinks=False))


@pytest.mark.parametrize("identity_kind", ["stat", "protocol"])
@pytest.mark.parametrize("projected", [False, True], ids=["native", "distinct-stat"])
def test_named_identity_accepts_held_file_and_rejects_replacement(
    tmp_path, monkeypatch, identity_kind, projected
):
    """Named checks and native pins must use the same identity representation."""
    from tldw_chatbook.DB.private_sqlite_protocol import FileIdentity

    path = tmp_path / "named.sqlite3"
    descriptor = native_os.open(
        path, native_os.O_RDWR | native_os.O_CREAT | native_os.O_EXCL, 0o600
    )
    facade = _NativeIdentityProjection() if projected else native_os
    monkeypatch.setattr(private_sqlite, "os", facade)
    try:
        native_os.write(descriptor, _database_bytes("original"))
        identity = facade.fstat(descriptor)
        if identity_kind == "protocol":
            identity = FileIdentity.from_stat(identity)
        private_sqlite.verify_expected_named_identity(path, identity)
        native_os.rename(path, path.with_suffix(".retained"))
        path.write_bytes(_database_bytes("replacement"))
        with pytest.raises(private_sqlite.private_paths.PrivatePathError) as error:
            private_sqlite.verify_expected_named_identity(path, identity)
        assert error.value.result.reason == "private_sqlite_expected_identity_changed"
    finally:
        native_os.close(descriptor)


def _database_bytes(value):
    with contextlib.closing(sqlite3.connect(":memory:")) as database:
        database.execute("CREATE TABLE sample(value TEXT)")
        database.execute("INSERT INTO sample VALUES (?)", (value,))
        database.commit()
        return database.serialize()


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "source.sqlite3"
    descriptor = native_os.open(
        path, native_os.O_RDWR | native_os.O_CREAT | native_os.O_EXCL, 0o600
    )
    native_os.write(descriptor, _database_bytes("original"))
    monkeypatch.setattr(private_sqlite, "os", _WindowsRoute())
    try:
        yield path, descriptor
    finally:
        native_os.close(descriptor)


def _reader(descriptor, **kwargs):
    return private_sqlite.connect_private_sqlite_descriptor(
        "tts.profile_store_descriptor", descriptor, **kwargs
    )


def test_snapshot_uses_held_object_and_preserves_offset(source, monkeypatch):
    path, descriptor = source
    native_os.rename(path, path.with_suffix(".retained"))
    path.write_bytes(_database_bytes("replacement"))
    native_os.lseek(descriptor, 37, native_os.SEEK_SET)
    connect = private_sqlite._SQLITE_CONNECT

    def memory_only(target, **kwargs):
        assert target == ":memory:", "Windows must not reopen a pathname"
        return connect(target, **kwargs)

    monkeypatch.setattr(private_sqlite, "_SQLITE_CONNECT", memory_only)
    with contextlib.closing(_reader(descriptor)) as database:
        assert database.execute("SELECT value FROM sample").fetchone() == ("original",)
    assert native_os.lseek(descriptor, 0, native_os.SEEK_CUR) == 37


@pytest.mark.parametrize("wal_header", [False, True])
def test_snapshot_is_readonly_and_independent_of_later_source_writes(
    source, wal_header
):
    _, descriptor = source
    if wal_header:
        native_os.lseek(descriptor, 18, native_os.SEEK_SET)
        native_os.write(descriptor, b"\x02\x02")
    with contextlib.closing(_reader(descriptor)) as database:
        native_os.lseek(descriptor, 0, native_os.SEEK_SET)
        native_os.write(descriptor, _database_bytes("changed!"))
        database.execute("PRAGMA foreign_keys = ON")
        database.execute("PRAGMA query_only = ON")
        with pytest.raises(sqlite3.DatabaseError):
            database.execute("PRAGMA query_only = OFF")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.execute("UPDATE sample SET value = 'bad'")
        assert database.execute("SELECT value FROM sample").fetchone() == ("original",)


def test_snapshot_rejects_source_changed_while_reading(source, monkeypatch):
    _, descriptor = source
    read = native_os.read
    native_os.lseek(descriptor, 29, native_os.SEEK_SET)

    def changed(fd, count):
        data = read(fd, count)
        info = native_os.fstat(fd)
        native_os.utime(fd, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        return data

    monkeypatch.setattr(private_sqlite.os, "read", changed, raising=False)
    with pytest.raises(ValueError, match="descriptor_snapshot_changed"):
        _reader(descriptor)
    assert native_os.lseek(descriptor, 0, native_os.SEEK_CUR) == 29


def test_snapshot_rejects_oversized_source_before_reading(source, monkeypatch):
    from tldw_chatbook.TTS import profile_migration_journal

    _, descriptor = source
    monkeypatch.setattr(
        profile_migration_journal, "MAX_PROFILE_MIGRATION_ARTIFACT_BYTES", 1
    )

    def unexpected_read(*args):
        raise AssertionError("Oversized descriptor must be refused before reading")

    monkeypatch.setattr(private_sqlite.os, "read", unexpected_read, raising=False)
    with pytest.raises(ValueError, match="descriptor_snapshot_limit"):
        _reader(descriptor)


@pytest.mark.parametrize("authorizer", [None, lambda *args: sqlite3.SQLITE_OK])
def test_replacing_sql_authorizer_cannot_disable_readonly(source, authorizer):
    _, descriptor = source
    with contextlib.closing(_reader(descriptor)) as database:
        database.set_authorizer(authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            database.execute("PRAGMA query_only = 0")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.execute("DELETE FROM sample")


def test_installed_sql_validator_preserves_snapshot_readonly(source):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _restrict_connection

    _, descriptor = source
    with contextlib.closing(_reader(descriptor)) as database:
        _restrict_connection(database)
        assert database.execute("SELECT value FROM sample").fetchone() == ("original",)
        with pytest.raises(sqlite3.DatabaseError):
            database.execute("PRAGMA query_only = OFF")
        with pytest.raises(sqlite3.DatabaseError):
            database.execute("DELETE FROM sample")


@pytest.mark.parametrize("noop_close", [False, True])
def test_deserialize_failure_closes_observed_native_connection(source, noop_close):
    _, descriptor = source

    class FailedDeserialize(sqlite3.Connection):
        def deserialize(self, data):
            raise sqlite3.DatabaseError("injected_deserialize_failure")

        def close(self):
            if not noop_close:
                super().close()

    outcome = private_sqlite._SQLiteDescriptorOutcome()
    with pytest.raises(sqlite3.DatabaseError, match="injected_deserialize_failure"):
        _reader(descriptor, factory=FailedDeserialize, _native_outcome=outcome)
    assert outcome.connection_closed and outcome.duplicate_closed
    assert not outcome.connection_pending
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        outcome.connection.execute("SELECT 1")
    assert not outcome.connector_pending
