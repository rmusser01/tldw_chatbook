"""Real exec contenders guard process-owned SQLite locks across private checks."""

from __future__ import annotations

import inspect
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB import private_sqlite_files as files
from tldw_chatbook.DB import private_sqlite_process as process


def _other_process_can_begin_write(path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0.05, isolation_level=None)
try:
    connection.execute('BEGIN IMMEDIATE')
except sqlite3.OperationalError as error:
    print(error.sqlite_errorcode)
else:
    print(0)
    connection.rollback()
finally:
    connection.close()
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    code = int(result.stdout.strip())
    assert code in {0, sqlite3.SQLITE_BUSY}, f"unexpected child SQLite result: {code}"
    return code == 0


def _exclusive_contender(path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0.05, isolation_level=None)
try:
    connection.execute('BEGIN EXCLUSIVE')
except sqlite3.OperationalError as error:
    print(error.sqlite_errorcode)
else:
    print(0)
    connection.rollback()
finally:
    connection.close()
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() in {"0", str(sqlite3.SQLITE_BUSY)}
    return int(result.stdout.strip())


@pytest.mark.parametrize("journal_mode", ["WAL", "DELETE"])
@pytest.mark.parametrize("other_thread", [False, True])
def test_private_connect_preserves_live_writer_lock(
    tmp_path, journal_mode, other_thread
):
    path = tmp_path.resolve() / "owned.sqlite"
    with closing(sqlite3.connect(path, isolation_level=None)) as owner:
        owner.execute(f"PRAGMA journal_mode={journal_mode}")
        owner.execute("CREATE TABLE sample(value INTEGER)")
        owner.execute("BEGIN IMMEDIATE")
        try:
            assert not _other_process_can_begin_write(path)

            def private_open():
                with closing(
                    private_sqlite.connect_private_sqlite(
                        "db.chachanotes.primary", path
                    )
                ):
                    assert not _other_process_can_begin_write(path)
                assert not _other_process_can_begin_write(path)

            if other_thread:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    worker.submit(private_open).result(timeout=15)
            else:
                private_open()
            assert owner.in_transaction
        finally:
            owner.rollback()
    assert _other_process_can_begin_write(path)


@pytest.mark.parametrize("other_thread", [False, True])
def test_private_connect_preserves_rollback_reader_lock(tmp_path, other_thread):
    path = tmp_path.resolve() / "reader.sqlite"
    with closing(sqlite3.connect(path, isolation_level=None)) as owner:
        owner.execute("PRAGMA journal_mode=DELETE")
        owner.execute("CREATE TABLE sample(value INTEGER)")
        owner.execute("INSERT INTO sample VALUES (1)")
        owner.execute("BEGIN")
        assert owner.execute("SELECT * FROM sample").fetchall() == [(1,)]
        try:
            assert _exclusive_contender(path) == sqlite3.SQLITE_BUSY

            def private_open():
                with closing(
                    private_sqlite.connect_private_sqlite(
                        "db.chachanotes.primary", path
                    )
                ):
                    assert _exclusive_contender(path) == sqlite3.SQLITE_BUSY
                assert _exclusive_contender(path) == sqlite3.SQLITE_BUSY

            if other_thread:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    worker.submit(private_open).result(timeout=15)
            else:
                private_open()
            assert owner.in_transaction
        finally:
            owner.rollback()
    assert _exclusive_contender(path) == 0


def _checkpoint_contender(path, *, update):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0.05, isolation_level=None)
try:
    if sys.argv[2] == 'update':
        connection.execute('UPDATE sample SET value=2')
    print(connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0])
finally:
    connection.close()
""",
            str(path),
            "update" if update else "checkpoint",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() in {"0", "1"}
    return int(result.stdout.strip())


def test_private_connect_preserves_wal_snapshot_and_checkpoint_exclusion(tmp_path):
    path = tmp_path.resolve() / "snapshot.sqlite"
    with closing(sqlite3.connect(path, isolation_level=None)) as reader:
        reader.execute("PRAGMA journal_mode=WAL")
        reader.execute("CREATE TABLE sample(value INTEGER)")
        reader.execute("INSERT INTO sample VALUES (1)")
        reader.execute("BEGIN")
        assert reader.execute("SELECT * FROM sample").fetchall() == [(1,)]
        try:
            # WAL readers permit writers, but keep their snapshot and block truncation.
            assert _checkpoint_contender(path, update=True) == 1
            with closing(
                private_sqlite.connect_private_sqlite("db.chachanotes.primary", path)
            ):
                assert _checkpoint_contender(path, update=False) == 1
                assert reader.execute("SELECT * FROM sample").fetchall() == [(1,)]
            assert _checkpoint_contender(path, update=False) == 1
            assert reader.execute("SELECT * FROM sample").fetchall() == [(1,)]
        finally:
            reader.rollback()
        assert _checkpoint_contender(path, update=False) == 0
        assert reader.execute("SELECT * FROM sample").fetchall() == [(2,)]


def _diagnostic_control(path, inspection, loses_lock):
    # Deliberate unsafe controls use their own DB, never a product assertion DB.
    with closing(sqlite3.connect(path, isolation_level=None)) as owner:
        owner.execute("PRAGMA journal_mode=WAL")
        owner.execute("CREATE TABLE sample(value INTEGER)")
        owner.execute("BEGIN IMMEDIATE")
        observer = None
        try:
            assert not _other_process_can_begin_write(path)
            if inspection == "sqlite":
                observer = sqlite3.connect(path)
            elif inspection.startswith("raw-"):
                suffix = {
                    "raw-main-close": "",
                    "raw-wal-close": "-wal",
                    "raw-shm-close": "-shm",
                }[inspection]
                descriptor = os.open(str(path) + suffix, os.O_RDONLY)
                os.close(descriptor)
            else:
                target = path if inspection == "private-main" else Path(f"{path}-shm")
                files._prepare_artifact(target, writable=True, create_if_missing=False)
            assert owner.in_transaction
            assert _other_process_can_begin_write(path) is loses_lock
        finally:
            if observer is not None:
                observer.close()
            owner.rollback()


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX descriptor-close characterization"
)
@pytest.mark.parametrize(
    ("inspection", "loses_lock"),
    [
        ("sqlite", False),
        ("raw-main-close", False),
        ("raw-wal-close", False),
        ("raw-shm-close", True),
        ("private-main", False),
        ("private-shm", True),
    ],
)
def test_isolated_diagnostic_controls(tmp_path, inspection, loses_lock):
    # Bootstrap only fixed leaf namespaces; never run application startup.
    program = """
import os, sqlite3, subprocess, sys
from contextlib import closing
from pathlib import Path
from types import ModuleType
if sys.argv[3].startswith('private-'):
    package = Path(sys.argv[1])
    for name, directory in (
        ("tldw_chatbook", package),
        ("tldw_chatbook.DB", package / "DB"),
        ("tldw_chatbook.Utils", package / "Utils"),
    ):
        namespace = ModuleType(name)
        namespace.__path__ = [str(directory)]
        namespace.__package__ = name
        sys.modules[name] = namespace
    from tldw_chatbook.DB import private_sqlite_files as files
"""
    program += inspect.getsource(_other_process_can_begin_write)
    program += inspect.getsource(_diagnostic_control)
    program += "\n_diagnostic_control(Path(sys.argv[2]), sys.argv[3], sys.argv[4] == 'yes')\nprint('control passed')\n"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(Path(private_sqlite.__file__).resolve().parent.parent),
            str(tmp_path.resolve() / "discarded-control.sqlite"),
            inspection,
            "yes" if loses_lock else "no",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout == "control passed\n"


@pytest.mark.parametrize("fail_callback", [False, True])
def test_backup_pin_release_preserves_sibling_writer_and_borrowed_handle(
    tmp_path, fail_callback
):
    path = tmp_path.resolve() / "backup-source.sqlite"
    target = tmp_path.resolve() / "backup.sqlite"
    with (
        closing(sqlite3.connect(path, isolation_level=None)) as writer,
        closing(sqlite3.connect(path, isolation_level=None)) as borrowed,
    ):
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value INTEGER)")
        writer.execute("INSERT INTO sample VALUES (1)")
        writer.execute("BEGIN IMMEDIATE")
        calls = []

        def guard():
            calls.append(True)
            assert not _other_process_can_begin_write(path)
            if fail_callback:
                raise RuntimeError("owned callback refusal")

        try:
            assert not _other_process_can_begin_write(path)
            if fail_callback:
                with pytest.raises(RuntimeError, match="owned callback refusal"):
                    private_sqlite.backup_connection_to_private(
                        "db.chachanotes.backup",
                        borrowed,
                        path,
                        target,
                        progress_guard=guard,
                    )
            else:
                private_sqlite.backup_connection_to_private(
                    "db.chachanotes.backup",
                    borrowed,
                    path,
                    target,
                    progress_guard=guard,
                )
                with closing(sqlite3.connect(target)) as backup:
                    assert backup.execute("SELECT * FROM sample").fetchall() == [(1,)]
                    assert backup.execute("PRAGMA integrity_check").fetchone() == (
                        "ok",
                    )
            assert calls
            assert writer.in_transaction
            assert not _other_process_can_begin_write(path)
            assert borrowed.execute("SELECT * FROM sample").fetchall() == [(1,)]
        finally:
            writer.rollback()


@pytest.mark.parametrize("failure_at", ["before_backup", "progress", "final_recheck"])
def test_actual_pin_death_refuses_backup_and_preserves_sibling_lock(
    tmp_path, monkeypatch, failure_at
):
    path = tmp_path.resolve() / "source.sqlite"
    target = tmp_path.resolve() / "target.sqlite"
    leases = []
    real_start = process.HelperLease.start

    def capture(request, **kwargs):
        lease = real_start(request, **kwargs)
        leases.append(lease)
        if kwargs["operation"] == "pin_source" and failure_at == "before_backup":
            lease._child.kill()
            lease._child.wait(timeout=5)
        return lease

    monkeypatch.setattr(process.HelperLease, "start", capture)
    real_backup = private_sqlite._backup_pages

    def kill_after_backup(*args, **kwargs):
        real_backup(*args, **kwargs)
        if failure_at == "final_recheck":
            leases[0]._child.kill()
            leases[0]._child.wait(timeout=5)

    monkeypatch.setattr(private_sqlite, "_backup_pages", kill_after_backup)
    calls = []

    def guard():
        calls.append(True)
        assert not _other_process_can_begin_write(path)
        if failure_at == "progress":
            leases[0]._child.kill()
            leases[0]._child.wait(timeout=5)

    with (
        closing(sqlite3.connect(path, isolation_level=None)) as writer,
        closing(sqlite3.connect(path, isolation_level=None)) as borrowed,
    ):
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value INTEGER)")
        writer.execute("INSERT INTO sample VALUES (1)")
        writer.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(process.HelperProtocolError):
                private_sqlite.backup_connection_to_private(
                    "db.chachanotes.backup",
                    borrowed,
                    path,
                    target,
                    progress_guard=guard,
                )
            assert bool(calls) is (failure_at != "before_backup")
            if failure_at == "before_backup":
                assert not target.exists()
            assert writer.in_transaction
            assert not _other_process_can_begin_write(path)
            assert borrowed.execute("SELECT * FROM sample").fetchall() == [(1,)]
            assert all(lease.cleanup_state == "reaped" for lease in leases)
        finally:
            writer.rollback()


def test_expired_deadline_refuses_before_precreation_and_is_not_sqlite_kwarg(
    tmp_path,
):
    path = tmp_path.resolve() / "deadline.sqlite"
    with pytest.raises(process.HelperTimeoutError):
        private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary",
            path,
            operation_deadline=time.monotonic() - 1,
        )
    assert not path.exists()

    class CustomConnection(sqlite3.Connection):
        pass

    with closing(
        private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary",
            path,
            operation_deadline=time.monotonic() + 5,
            factory=CustomConnection,
            timeout=0.123,
            isolation_level=None,
        )
    ) as connection:
        assert type(connection) is CustomConnection
        assert connection.isolation_level is None
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (123,)


def test_reentrant_backup_callback_refuses_before_nested_target_mutation(
    tmp_path,
):
    source_path = tmp_path.resolve() / "source.sqlite"
    target = tmp_path.resolve() / "target.sqlite"
    nested = tmp_path.resolve() / "nested.sqlite"
    with closing(sqlite3.connect(source_path)) as source:
        source.execute("CREATE TABLE sample(value INTEGER)")
        source.commit()
        calls = []

        def guard():
            with pytest.raises(process.HelperUnavailableError):
                private_sqlite.connect_private_sqlite("db.chachanotes.primary", nested)
            assert not nested.exists()
            calls.append(True)

        private_sqlite.backup_connection_to_private(
            "db.chachanotes.backup",
            source,
            source_path,
            target,
            progress_guard=guard,
        )
        assert calls
        assert source.execute("SELECT COUNT(*) FROM sample").fetchone() == (0,)


def test_copy_borrows_one_two_slot_envelope_for_sequential_preparations(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path.resolve() / "source.sqlite"
    target = tmp_path.resolve() / "target.sqlite"
    with closing(sqlite3.connect(source_path)) as source:
        source.execute("CREATE TABLE sample(value INTEGER)")
    admission = process.HelperAdmission()
    monkeypatch.setattr(private_sqlite, "HELPER_ADMISSION", admission)
    original_start = process.HelperLease.start
    envelopes, leases, peaks = [], [], []
    expires_at = time.monotonic() + 5

    def capture(request, **kwargs):
        lease = original_start(request, **kwargs)
        reservation = kwargs["reservation"]
        assert kwargs["deadline"].expires_at == expires_at
        assert reservation._deadline.expires_at == expires_at
        envelopes.append(reservation)
        leases.append(lease)
        peaks.append(len(reservation._children))
        assert admission._used == {"transient": 2, "retained": 0}
        return lease

    monkeypatch.setattr(process.HelperLease, "start", capture)
    private_sqlite.copy_private_sqlite(
        "settings.bulk_backup", source_path, target, operation_deadline=expires_at
    )
    assert len(envelopes) == 3
    assert all(owner is envelopes[0] for owner in envelopes)
    assert peaks == [1, 2, 2]
    assert admission._used == {"transient": 0, "retained": 0}
    assert all(lease.cleanup_state == "reaped" for lease in leases)
    with closing(sqlite3.connect(target)) as copied:
        assert copied.execute("SELECT COUNT(*) FROM sample").fetchone() == (0,)


@pytest.mark.parametrize(
    ("program", "error"),
    [
        ("import os; os._exit(7)", process.HelperUnavailableError),
        ("import os; os.write(1, b'\\x00')", process.HelperProtocolError),
        ("import time; time.sleep(30)", process.HelperTimeoutError),
    ],
)
def test_normal_seam_real_helper_failure_has_no_local_fallback(
    tmp_path,
    monkeypatch,
    program,
    error,
):
    target = tmp_path.resolve() / "untouched.sqlite"
    real_popen = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = real_popen([sys.executable, "-I", "-S", "-c", program], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    monkeypatch.setattr(
        files, "prepare_batch", lambda *_: pytest.fail("local fallback")
    )
    with pytest.raises(error):
        private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary",
            target,
            operation_deadline=time.monotonic() + 0.2,
        )
    assert not target.exists()
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_public_expected_stat_identity_checked_after_actual_helper(
    tmp_path, monkeypatch
):
    target = tmp_path.resolve() / "target.sqlite"
    replacement = tmp_path.resolve() / "replacement.sqlite"
    target.touch(mode=0o600)
    replacement.touch(mode=0o600)
    identity = target.stat()
    real_prepare = private_sqlite.prepare_in_helper

    def prepare_then_replace(request, **kwargs):
        result = real_prepare(request, **kwargs)
        replacement.replace(target)
        return result

    monkeypatch.setattr(private_sqlite, "prepare_in_helper", prepare_then_replace)
    with pytest.raises(private_sqlite.private_paths.PrivatePathError) as caught:
        private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary",
            target,
            expected_identity=identity,
        )
    assert caught.value.result.reason == "private_sqlite_expected_identity_changed"


def test_reentrant_sqlite_factory_cannot_escape_operation_envelope(tmp_path):
    target = tmp_path.resolve() / "factory.sqlite"
    nested = tmp_path.resolve() / "nested.sqlite"
    returned = object()

    def factory(database, **kwargs):
        assert Path(database) == target
        with pytest.raises(process.HelperUnavailableError):
            private_sqlite.connect_private_sqlite("db.chachanotes.primary", nested)
        return returned

    assert (
        private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary",
            target,
            factory=factory,
        )
        is returned
    )
    assert not nested.exists()


def test_borrowed_connection_sql_admission_is_inside_operation_scope(tmp_path):
    nested = tmp_path.resolve() / "nested.sqlite"
    calls = []

    class SourceConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA database_list":
                with (
                    pytest.raises(process.HelperUnavailableError),
                    closing(
                        private_sqlite.connect_private_sqlite(
                            "db.chachanotes.primary",
                            nested,
                        )
                    ),
                ):
                    pass
                calls.append(True)
            return super().execute(sql, *args, **kwargs)

    with (
        closing(
            sqlite3.connect(tmp_path / "source.sqlite", factory=SourceConnection)
        ) as source,
        closing(sqlite3.connect(tmp_path / "target.sqlite")) as destination,
    ):
        source.execute("CREATE TABLE sample(value INTEGER)")
        private_sqlite.backup_open_connections_to_private(
            "db.chachanotes.backup",
            source,
            destination,
        )
        assert calls
        assert not nested.exists()
        assert source.execute("SELECT COUNT(*) FROM sample").fetchone() == (0,)
        assert destination.execute("SELECT COUNT(*) FROM sample").fetchone() == (0,)


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_restore_final_helper_loss_uses_owned_safety_snapshot_and_closes_sql(
    tmp_path, monkeypatch, recovery_fails
):
    source_path = tmp_path.resolve() / "source.sqlite"
    destination = tmp_path.resolve() / "destination.sqlite"
    safety = tmp_path.resolve() / "safety.sqlite"
    with closing(sqlite3.connect(destination, isolation_level=None)) as setup:
        setup.execute("CREATE TABLE sample(value TEXT)")
        setup.execute("INSERT INTO sample VALUES ('before')")
    with closing(sqlite3.connect(source_path, isolation_level=None)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value TEXT)")
        writer.execute("INSERT INTO sample VALUES ('after')")
        writer.execute("BEGIN IMMEDIATE")
        leases, owned = [], []
        real_start = process.HelperLease.start
        real_connect = private_sqlite._connect_registered_sqlite
        real_backup = private_sqlite._backup_pages
        restore_calls = 0

        class OwnedConnection(sqlite3.Connection):
            close_calls = 0

            def close(self):
                self.close_calls += 1
                super().close()

        def track_connect(*args, **kwargs):
            connection = real_connect(*args, factory=OwnedConnection, **kwargs)
            owned.append(connection)
            return connection

        def capture(request, **kwargs):
            lease = real_start(request, **kwargs)
            leases.append(lease)
            return lease

        def lose_proof_after_commit(source, target, *, restore, **kwargs):
            nonlocal restore_calls
            if restore:
                restore_calls += 1
                if restore_calls == 2 and recovery_fails:
                    raise sqlite3.OperationalError("owned recovery refusal")
            real_backup(source, target, restore=restore, **kwargs)
            if restore and restore_calls == 1:
                leases[0]._child.kill()
                leases[0]._child.wait(timeout=5)

        monkeypatch.setattr(process.HelperLease, "start", capture)
        monkeypatch.setattr(private_sqlite, "_connect_registered_sqlite", track_connect)
        monkeypatch.setattr(private_sqlite, "_backup_pages", lose_proof_after_commit)
        try:
            expected = (
                private_sqlite.SQLiteRestoreIndeterminateError
                if recovery_fails
                else process.HelperProtocolError
            )
            with pytest.raises(expected):
                private_sqlite.restore_private_sqlite(
                    "tts.profile_restore_stage",
                    "tts.profile_restore_stage",
                    source_path,
                    destination,
                    safety,
                )
            assert restore_calls == 2
            assert len(owned) == 3
            assert all(connection.close_calls == 1 for connection in owned)
            assert all(lease.cleanup_state == "reaped" for lease in leases)
            assert writer.in_transaction
            assert not _other_process_can_begin_write(source_path)
            with closing(sqlite3.connect(destination)) as result:
                expected_value = "after" if recovery_fails else "before"
                assert result.execute("SELECT * FROM sample").fetchall() == [
                    (expected_value,)
                ]
                assert result.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            with closing(sqlite3.connect(safety)) as snapshot:
                assert snapshot.execute("SELECT * FROM sample").fetchall() == [
                    ("before",)
                ]
        finally:
            writer.rollback()
