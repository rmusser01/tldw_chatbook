"""Core database maintenance must preserve actual borrower/native lifetimes."""

import select
import sqlite3
import time

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB


from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


CORE_TYPES = (
    CharactersRAGDB,
    MediaDatabase,
    PromptsDatabase,
    LibraryCollectionsDB,
    LibraryIngestJobsDB,
)


def cached_connection(owner):
    if isinstance(owner, LibraryCollectionsDB):
        return getattr(owner._thread_local, "conn", None)
    if isinstance(owner, LibraryIngestJobsDB):
        return owner._conn
    return getattr(owner._local, "conn", None)


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_failed_explicit_native_close_preserves_owner_reference(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = owner_type(tmp_path / "core.sqlite", "core-lifetime-test")
    connection = cached_connection(owner)
    assert connection is not None
    real_close = connection.close

    def fail_close():
        raise sqlite3.OperationalError("injected pre-close failure")

    monkeypatch.setattr(connection, "close", fail_close)
    pause = storage._begin_local_pause()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        owner.close()
        # A refused close must not make the owner lose the only explicit
        # retirement route while native SQLite and normal admission remain live.
        assert cached_connection(owner) is connection
        assert connection.execute("SELECT 42").fetchone()[0] == 42
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
    finally:
        monkeypatch.setattr(connection, "close", real_close)
        real_close()
        owner.close()
        pause.resume()
    assert line(child) == "entered"
    release(child)


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_cached_native_getter_refuses_new_borrower_after_pause(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    owner = owner_type(tmp_path / "gate.sqlite", "gate-test")
    pause = storage._begin_local_pause()
    try:
        getter = (
            owner._held_connection
            if isinstance(owner, LibraryCollectionsDB)
            else owner._get_connection
            if isinstance(owner, LibraryIngestJobsDB)
            else owner.get_connection
        )
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            getter()
    finally:
        pause.resume()
        owner.close()


def get_connection(owner):
    if isinstance(owner, LibraryCollectionsDB):
        return owner._held_connection()
    if isinstance(owner, LibraryIngestJobsDB):
        return owner._get_connection()
    return owner.get_connection()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_admitted_transaction_finishes_but_escaped_cursor_keeps_native_hold(
    tmp_path, local_root, launch, owner_type
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "transaction.sqlite", "transaction-test")
    participant = _repository_participant(owner)
    pause = None
    try:
        with owner.transaction() as work:
            work.execute("CREATE TABLE phase3_result (value TEXT)")
            work.execute("INSERT INTO phase3_result VALUES ('committed')")
            cursor = work.execute("SELECT value FROM phase3_result")
            pause = storage._begin_local_pause()
            participant.close_admission()
            # The exact live transaction can finish its originally admitted scope.
            assert get_connection(owner) is cached_connection(owner)
            assert not participant.drain(time.monotonic() + 0.02)
        # Scope exit commits; it does not revoke reusable native borrowers.
        assert cursor.fetchone()[0] == "committed"
        assert cursor.execute("SELECT COUNT(*) FROM phase3_result").fetchone()[0] == 1
        assert not participant.drain(time.monotonic() + 0.02)
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            with owner.transaction():
                pytest.fail("new transaction entered after pause")
        child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
        assert not select.select([child.stdout], [], [], 0.02)[0]
        cursor.close()
        # Explicit close by the source owner is the borrower retirement boundary.
        owner.close()
        assert participant.drain(time.monotonic() + 1)
        assert pause.drain(time.monotonic() + 1)
        assert line(child) == "entered"
        release(child)
        pause.resume()
        pause = None
        participant.resume()
        assert (
            get_connection(owner)
            .execute("SELECT value FROM phase3_result")
            .fetchone()[0]
            == "committed"
        )
    finally:
        if pause is not None:
            pause.resume()
        participant.resume()
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_owner_close_during_managed_work_preserves_live_transaction(
    tmp_path, owner_type
):
    owner = owner_type(tmp_path / "active.sqlite", "active-test")
    try:
        with owner.transaction() as work:
            work.execute("CREATE TABLE phase3_active (value TEXT)")
            work.execute("INSERT INTO phase3_active VALUES ('kept')")
            owner.close()
            assert (
                work.execute("SELECT value FROM phase3_active").fetchone()[0] == "kept"
            )
        assert (
            get_connection(owner)
            .execute("SELECT value FROM phase3_active")
            .fetchone()[0]
            == "kept"
        )
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_explicit_raw_close_reopens_cached_owner_without_idle_delay(
    tmp_path, owner_type
):
    owner = owner_type(tmp_path / "reopen.sqlite", "reopen-test")
    connection = get_connection(owner)
    try:
        connection.close()
        assert get_connection(owner).execute("SELECT 42").fetchone()[0] == 42
    finally:
        owner.close()


def test_uninstalled_subclass_keeps_ordinary_use_without_participant_authority(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    class OrdinaryReader(CharactersRAGDB):
        pass

    owner = OrdinaryReader(tmp_path / "ordinary.sqlite", "ordinary-test")
    try:
        with owner.transaction() as cursor:
            assert cursor.execute("SELECT 42").fetchone()[0] == 42
        with pytest.raises(ValueError, match="repository_participant_not_installed"):
            _repository_participant(owner)
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES[:-1])
def test_distinct_thread_cache_closes_without_interrupting_same_owner_worker(
    tmp_path, local_root, owner_type
):
    import threading

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "threads.sqlite", "threads-test")
    participant = _repository_participant(owner)
    main_connection = cached_connection(owner)
    entered, finish, idle, retire = (threading.Event() for _ in range(4))
    errors = []

    def work():
        try:
            with owner.transaction() as native:
                entered.set()
                assert finish.wait(5)
                native.execute("CREATE TABLE phase3_thread (value TEXT)")
                native.execute("INSERT INTO phase3_thread VALUES ('finished')")
            connection = cached_connection(owner)
            idle.set()
            assert retire.wait(5)
            if isinstance(native, sqlite3.Cursor):
                native.close()
            owner.close()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        except BaseException as error:
            errors.append(error)
            idle.set()
            owner.close()

    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(5)
    pause = storage._begin_local_pause()
    participant.close_admission()
    try:
        owner.close()
        assert cached_connection(owner) is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            main_connection.execute("SELECT 1")
        finish.set()
        assert idle.wait(5)
        assert not errors
        assert not participant.drain(time.monotonic() + 0.02)
        assert not pause.drain(time.monotonic() + 0.02)
        retire.set()
        worker.join(5)
        assert not worker.is_alive() and not errors
        assert participant.drain(time.monotonic() + 1)
        assert pause.drain(time.monotonic() + 1)
    finally:
        finish.set()
        retire.set()
        worker.join(5)
        pause.resume()
        participant.resume()
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_pause_never_rolls_back_borrowed_raw_transaction(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = owner_type(tmp_path / "borrowed.sqlite", "borrowed-test")
    connection = get_connection(owner)
    connection.execute("CREATE TABLE phase3_borrowed (value TEXT)")
    connection.commit()
    connection.execute("BEGIN")
    connection.execute("INSERT INTO phase3_borrowed VALUES ('caller-owned')")
    pause = storage._begin_local_pause()
    try:
        owner.close()
        assert connection.in_transaction
        assert (
            connection.execute("SELECT value FROM phase3_borrowed").fetchone()[0]
            == "caller-owned"
        )
        assert not pause.drain(time.monotonic() + 0.02)
        connection.rollback()
        owner.close()
    finally:
        pause.resume()
        owner.close()


def test_ingest_shared_handle_close_preserves_operation_on_other_thread(tmp_path):
    import threading

    owner = LibraryIngestJobsDB(tmp_path / "ingest-shared.sqlite")
    entered, finish = threading.Event(), threading.Event()
    errors = []

    def work():
        try:
            with owner.transaction() as connection:
                entered.set()
                assert finish.wait(5)
                assert connection.execute("SELECT 42").fetchone()[0] == 42
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(5)
    try:
        owner.close()
        assert cached_connection(owner) is not None
    finally:
        finish.set()
        worker.join(5)
        owner.close()
    assert not worker.is_alive() and not errors


def test_ingest_source_close_reserves_handle_before_native_close(tmp_path, monkeypatch):
    import threading

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    owner = LibraryIngestJobsDB(tmp_path / "close-race.sqlite")
    connection = get_connection(owner)
    real_close = connection.close
    closing, attempted = threading.Event(), threading.Event()
    errors = []

    def delayed_close():
        closing.set()
        assert attempted.wait(5)
        real_close()

    def late_operation():
        try:
            assert closing.wait(5)
            with pytest.raises(RecoveryRequired, match="core_connection_retiring"):
                with owner.transaction():
                    pytest.fail("a borrower entered a retiring native connection")
        except BaseException as error:
            errors.append(error)
        finally:
            attempted.set()

    worker = threading.Thread(target=late_operation)
    worker.start()
    monkeypatch.setattr(connection, "close", delayed_close)
    try:
        owner.close()
    finally:
        attempted.set()
        worker.join(5)
        monkeypatch.setattr(connection, "close", real_close)
        real_close()
        owner.close()
    assert not worker.is_alive() and not errors


@pytest.mark.parametrize("owner_type", CORE_TYPES[:-1])
def test_failed_liveness_probe_preserves_cached_native_reference(
    tmp_path, monkeypatch, owner_type
):
    owner = owner_type(tmp_path / "liveness.sqlite", "liveness-test")
    connection = get_connection(owner)
    real_execute, real_close = connection.execute, connection.close
    local = (
        owner._thread_local if isinstance(owner, LibraryCollectionsDB) else owner._local
    )
    local.conn_last_used = None

    def fail_ping(sql, *args, **kwargs):
        if sql.strip().lower() == "select 1":
            raise sqlite3.OperationalError("injected ping failure")
        return real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", fail_ping)
    try:
        with pytest.raises(sqlite3.OperationalError):
            get_connection(owner)
        assert cached_connection(owner) is connection
        assert real_execute("SELECT 42").fetchone()[0] == 42
    finally:
        monkeypatch.setattr(connection, "execute", real_execute)
        monkeypatch.setattr(connection, "close", real_close)
        real_close()
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_memory_owner_keeps_native_state_across_local_pause(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = owner_type(":memory:", "memory-test")
    connection = get_connection(owner)
    pause = storage._begin_local_pause()
    try:
        with owner.transaction() as work:
            work.execute("CREATE TABLE phase3_memory (value TEXT)")
            work.execute("INSERT INTO phase3_memory VALUES ('retained')")
        assert get_connection(owner) is connection
        assert (
            connection.execute("SELECT value FROM phase3_memory").fetchone()[0]
            == "retained"
        )
    finally:
        pause.resume()
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES[:3])
def test_nested_and_borrowed_transaction_ownership_remains_with_original_caller(
    tmp_path, owner_type
):
    owner = owner_type(tmp_path / "nested.sqlite", "nested-test")
    connection = get_connection(owner)
    connection.execute("CREATE TABLE phase3_nested (value TEXT)")
    connection.commit()
    try:
        with owner.transaction() as outer:
            outer.execute("INSERT INTO phase3_nested VALUES ('outer')")
            with owner.transaction() as inner:
                inner.execute("INSERT INTO phase3_nested VALUES ('inner')")
            assert connection.in_transaction
        assert (
            connection.execute("SELECT COUNT(*) FROM phase3_nested").fetchone()[0] == 2
        )
        connection.execute("BEGIN")
        connection.execute("INSERT INTO phase3_nested VALUES ('borrowed')")
        with pytest.raises(RuntimeError, match="caller retains rollback"):
            with owner.transaction() as borrowed:
                borrowed.execute("INSERT INTO phase3_nested VALUES ('borrowed-inner')")
                raise RuntimeError("caller retains rollback")
        assert connection.in_transaction
        assert (
            connection.execute("SELECT COUNT(*) FROM phase3_nested").fetchone()[0] == 4
        )
        connection.rollback()
        assert (
            connection.execute("SELECT COUNT(*) FROM phase3_nested").fetchone()[0] == 2
        )
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CORE_TYPES)
def test_connection_registration_rejects_another_selected_path(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _register_core_connection

    owner = owner_type(tmp_path / "selected.sqlite", "selected-test")
    other = owner_type(tmp_path / "other.sqlite", "other-test")
    try:
        with pytest.raises(
            RecoveryRequired, match="core_connection_provenance_invalid"
        ):
            _register_core_connection(owner, get_connection(other))
        assert get_connection(other).execute("SELECT 42").fetchone()[0] == 42
    finally:
        owner.close()
        other.close()


def test_independent_nested_exception_restores_outer_operation_after_pause(tmp_path):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    outer = PromptsDatabase(tmp_path / "outer.sqlite", "outer-test")
    inner = PromptsDatabase(tmp_path / "inner.sqlite", "inner-test")
    pause = None
    try:
        with outer.transaction() as first:
            first.execute("CREATE TABLE phase3_outer (value TEXT)")
            with pytest.raises(RuntimeError, match="inner failed"):
                with inner.transaction() as second:
                    second.execute("CREATE TABLE phase3_inner (value TEXT)")
                    pause = storage._begin_local_pause()
                    assert not pause.drain(time.monotonic() + 0.02)
                    raise RuntimeError("inner failed")
            assert outer.get_connection() is first
            first.execute("INSERT INTO phase3_outer VALUES ('outer survives')")
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                with inner.transaction():
                    pytest.fail(
                        "a new different-participant scope entered during pause"
                    )
        assert (
            first.execute("SELECT value FROM phase3_outer").fetchone()[0]
            == "outer survives"
        )
        outer.close()
        inner.close()
        assert pause.drain(time.monotonic() + 1)
        pause.resume()
        pause = None
        participant = _repository_participant(inner)
        participant.close_admission()
        try:
            with outer.transaction():
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    with inner.transaction():
                        pytest.fail("a closed target participant admitted nested work")
        finally:
            participant.resume()
    finally:
        if pause is not None:
            pause.resume()
        outer.close()
        inner.close()


def test_pause_cancels_independent_nested_admission_and_keeps_outer_live(
    tmp_path, local_root, monkeypatch
):
    import threading

    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    outer = PromptsDatabase(tmp_path / "outer-race.sqlite", "outer-test")
    inner = PromptsDatabase(tmp_path / "inner-race.sqlite", "inner-test")
    entered, finish, retired = (threading.Event() for _ in range(3))
    original = storage.admission_authority
    errors = []

    def delayed_authority(root):
        if any(
            getattr(operation, "path", None) == inner.db_path
            and operation.lease is None
            for operation in tuple(storage._operations)
        ):
            entered.set()
            assert finish.wait(5)
        return original(root)

    monkeypatch.setattr(storage, "admission_authority", delayed_authority)

    def work():
        try:
            with outer.transaction() as connection:
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="storage_locally_paused"
                ):
                    with inner.transaction():
                        pytest.fail("nested admission raced through closed gate")
                assert outer.get_connection() is connection
                connection.execute("CREATE TABLE phase3_race (value TEXT)")
                connection.execute("INSERT INTO phase3_race VALUES ('committed')")
            outer.close()
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()

    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(5)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.02)
        finish.set()
        assert retired.wait(5)
        worker.join(5)
        assert not worker.is_alive() and not errors
        outer.close()
        inner.close()
        assert pause.drain(time.monotonic() + 1)
    finally:
        finish.set()
        worker.join(5)
        pause.resume()
        outer.close()
        inner.close()


def test_dead_affine_worker_handle_is_retained_until_process_exit(tmp_path):
    import subprocess
    import sys

    script = """
import gc, sys, threading, time, weakref
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Backup_Recovery.participants import _repository_participant
owner = LibraryCollectionsDB(Path(sys.argv[2]))
owner.close()
references = []
def work():
    connection = owner._held_connection()
    references.append(weakref.ref(connection))
worker = threading.Thread(target=work)
worker.start()
worker.join(5)
assert not worker.is_alive()
# Adversarial GC cannot replace explicit native retirement or erase the blocker.
gc.collect()
assert references[0]() is not None
participant = _repository_participant(owner)
participant.close_admission()
pause = storage._begin_local_pause()
assert not participant.drain(time.monotonic() + 0.03)
assert not pause.drain(time.monotonic() + 0.03)
print("orphan-retained", flush=True)
# No foreign-thread close or attempt to resurrect the dead source thread.
pause.resume()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "orphan-bootstrap"),
            str(tmp_path / "orphan.sqlite"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "orphan-retained"


def test_raw_unwrapped_native_connection_cannot_claim_core_provenance(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _register_core_connection

    owner = LibraryIngestJobsDB(tmp_path / "provenance.sqlite")
    native = sqlite3.Connection(":memory:")
    try:
        with pytest.raises(
            RecoveryRequired, match="core_connection_provenance_invalid"
        ):
            _register_core_connection(owner, native)
        assert get_connection(owner).execute("SELECT 42").fetchone()[0] == 42
    finally:
        native.close()
        owner.close()


def test_unmatched_factory_native_return_retains_exclusion_until_process_exit(
    tmp_path, launch
):
    import subprocess
    import sys

    root = tmp_path / "unmatched-bootstrap"
    script = """
import sqlite3, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
class OtherNative(sqlite3.Connection):
    def __new__(cls, *args, **kwargs):
        return sqlite3.Connection(str(Path(sys.argv[2])))
connection = connect_private_sqlite("db.library_ingest_jobs", Path(sys.argv[2]), factory=OtherNative)
assert type(connection) is sqlite3.Connection
connection.execute("CREATE TABLE unmatched (value TEXT)")
connection.execute("INSERT INTO unmatched VALUES ('still-live')")
connection.commit()
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + 0.03)
assert connection.execute("SELECT value FROM unmatched").fetchone()[0] == "still-live"
print("unmatched-live", flush=True)
sys.stdin.readline()
connection.close()
# No admitted wrapper observed this unmatched native object's retirement.
assert not pause.drain(time.monotonic() + 0.03)
print("unmatched-unqualified", flush=True)
sys.stdin.readline()
pause.resume()
"""
    writer = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            script,
            str(root),
            str(tmp_path / "unmatched.sqlite"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert line(writer) == "unmatched-live"
        maintainer = launch(root / "admission", "maintenance", ("bootstrap.unbound",))
        assert not select.select([maintainer.stdout], [], [], 0.03)[0]
        writer.stdin.write("close\n")
        writer.stdin.flush()
        assert line(writer) == "unmatched-unqualified"
        assert not select.select([maintainer.stdout], [], [], 0.03)[0]
        writer.stdin.write("exit\n")
        writer.stdin.flush()
        assert writer.wait(timeout=5) == 0, writer.stderr.read()
        assert line(maintainer) == "entered"
        release(maintainer)
    finally:
        if writer.poll() is None:
            writer.kill()
        writer.wait(timeout=5)
        writer.stdin.close()
        writer.stdout.close()
        writer.stderr.close()


def test_live_connection_cannot_be_reassigned_to_another_instance_same_path(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _register_core_connection

    first = PromptsDatabase(tmp_path / "same.sqlite", "first")
    second = PromptsDatabase(tmp_path / "same.sqlite", "second")
    try:
        with pytest.raises(
            RecoveryRequired, match="core_connection_provenance_invalid"
        ):
            _register_core_connection(second, first.get_connection())
    finally:
        first.close()
        second.close()
