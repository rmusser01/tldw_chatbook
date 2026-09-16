"""Native-held installed owner reads never open the original WAL group."""

import hashlib
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_core_owners import application_authority
from tldw_chatbook.Backup_Recovery.models import StorageItem

_WAL = """
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA wal_autocheckpoint=0')
conn.execute('UPDATE ' + sys.argv[2] + " SET title='WAL-only'")
conn.commit()
os._exit(0)
"""


def state(source):
    paths = [
        source.parent,
        *(Path(str(source) + s) for s in ("", "-wal", "-shm", "-journal")),
    ]
    return {
        str(path): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_mode,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in paths
        if path.exists()
    }


@pytest.fixture(params=["research", "core"])
def wal_owner(request, tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    source = root / "owned.db"
    if request.param == "research":
        from tldw_chatbook.Research_Interop.local_research_service import (
            LocalResearchService,
        )
        from tldw_chatbook.Research_Interop.recovery import recovery_adapters

        service = LocalResearchService(source)
        service.create_session(title="original", query="question")
        adapter = recovery_adapters()[0]
        table = "research_sessions"
    else:
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        from tldw_chatbook.DB.recovery_core import core_adapters

        service = CharactersRAGDB(source, "fixture")
        service.add_note("original", "body")
        adapter = core_adapters()[0]
        table = "notes"
    service.close()
    subprocess.run(
        [sys.executable, "-c", _WAL, str(source), table], check=True, timeout=15
    )
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    item = StorageItem(adapter.owner_id, "fixture:owned", source, "included", ())
    return source, stage, authority, adapter, item, table


def test_installed_capture_preserves_wal_group_and_committed_rows(wal_owner):
    source, stage, authority, adapter, item, table = wal_owner
    before = state(source)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, stage / "snapshot.db", Event())
        assert adapter.validate(stage / "snapshot.db") == ()
        assert state(source) == before
    assert state(source) == before
    with closing(sqlite3.connect(stage / "snapshot.db")) as connection:
        query = (
            "SELECT title FROM research_sessions"
            if table == "research_sessions"
            else "SELECT title FROM notes"
        )
        assert connection.execute(query).fetchall() == [("WAL-only",)]
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert not (stage / "snapshot.db-wal").exists()
    assert not (stage / "snapshot.db-shm").exists()


def test_direct_read_reuses_materialization_and_retires_native_connection(wal_owner):
    from tldw_chatbook.Backup_Recovery.admission import _local
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source, stage, authority, adapter, item, _ = wal_owner
    before = state(source)
    required = sum(Path(str(source) + suffix).stat().st_size for suffix in ("", "-wal"))
    owner = (
        "recovery.domain.research"
        if adapter.owner_id == "research.local"
        else "recovery.core.chachanotes"
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 2) as session:
        with session.capture_scope((source,), stage, byte_budget=required):
            connection = connect_private_sqlite(owner, source, read_only=True)
            private = Path(connection.execute("PRAGMA database_list").fetchone()[2])
            cancelled = Event()
            cancelled.set()
            with pytest.raises(InterruptedError, match="cancelled"):
                adapter.capture(item, stage / "cancelled.db", cancelled)
            adapter.capture(item, stage / "one.db", Event())
            adapter.capture(item, stage / "two.db", Event())
            assert _local.capture_scope.copied_bytes == required
            assert len(_local.capture_scope.sqlite_snapshots) == 1
            assert private != source and stage in private.parents
            assert state(source) == before
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        assert not private.exists()
        session._check()
    assert state(source) == before


@pytest.mark.parametrize("limit", ["member", "total"])
def test_reviewed_budget_refuses_before_source_open(wal_owner, limit):
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    source, stage, authority, adapter, item, _ = wal_owner
    before = state(source)
    options = (
        {"limits": ArchiveLimits(member_bytes=1)}
        if limit == "member"
        else {"byte_budget": 1}
    )
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage, **options),
        pytest.raises(ValueError, match="capture_sqlite_limit"),
    ):
        adapter.capture(item, stage / "refused.db", Event())
    assert state(source) == before
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("timing", ["before", "during"])
def test_cancel_preserves_source_and_removes_private_materialization(
    wal_owner, monkeypatch, timing
):
    source, stage, authority, adapter, item, _ = wal_owner
    before = state(source)
    cancel = Event()
    original = os.read
    source_identity = source.stat().st_ino

    def read(fd, count):
        data = original(fd, count)
        if os.fstat(fd).st_ino == source_identity:
            cancel.set()
        return data

    if timing == "before":
        cancel.set()
    else:
        monkeypatch.setattr(os, "read", read)
    with authority.maintenance(("core", "bootstrap.unbound"), 2) as session:
        with (
            session.capture_scope((source,), stage),
            pytest.raises(InterruptedError, match="cancelled"),
        ):
            adapter.capture(item, stage / "cancelled.db", cancel)
        session._check()
    assert state(source) == before
    assert list(stage.iterdir()) == []


def test_changed_source_identity_is_not_recaptured(wal_owner):
    from dataclasses import replace

    source, stage, authority, adapter, item, _ = wal_owner
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage),
    ):
        unselected = source.parent / "unselected.db"
        unselected.write_bytes(source.read_bytes())
        with pytest.raises(ValueError, match="capture_sqlite_source_changed"):
            adapter.capture(
                replace(item, path=unselected), stage / "refused.db", Event()
            )
    assert not (stage / "refused.db").exists()


def test_hot_journal_refused_without_recovery(wal_owner):
    source, stage, authority, adapter, item, _ = wal_owner
    journal = Path(str(source) + "-journal")
    journal.write_bytes(b"not a safe committed SQLite image")
    journal.chmod(0o600)
    before = state(source)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage),
        pytest.raises(ValueError, match="capture_sqlite_hot_journal"),
    ):
        adapter.capture(item, stage / "refused.db", Event())
    assert state(source) == before


@pytest.mark.parametrize("drift", [False, True])
def test_backup_error_rechecks_source_state(wal_owner, monkeypatch, drift):
    from tldw_chatbook.DB import private_sqlite

    source, stage, authority, adapter, item, _ = wal_owner
    before = state(source)

    def fail(*args, **kwargs):
        if drift:
            wal = Path(str(source) + "-wal")
            info = wal.stat()
            os.utime(wal, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        raise RuntimeError("injected backup failure")

    monkeypatch.setattr(private_sqlite, "_backup_pages", fail)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        pytest.raises(
            (RuntimeError, ValueError),
            match="source_changed" if drift else "injected backup",
        ),
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, stage / "failed.db", Event())
    if not drift:
        assert state(source) == before
    assert not list(stage.glob("sqlite-sources-*"))


@pytest.mark.parametrize("handle", ["source", "output", "parent", "post_open"])
def test_ambiguous_materializer_close_quarantines_native_exclusion(wal_owner, handle):
    import select

    source, stage, _, adapter, _, _ = wal_owner
    before = state(source)
    script = """
import sys
from pathlib import Path
from Tests.Backup_Recovery.test_capture_sqlite_materialization import _close_failure_child
_close_failure_child(*map(Path, sys.argv[1:4]), sys.argv[4], sys.argv[5])
"""
    writer_script = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
print('ready', flush=True)
with admission_authority(Path(sys.argv[1])).normal(('core',)):
    print('admitted', flush=True)
"""
    processes = []
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(stage.parent / "bootstrap"),
                str(source),
                str(stage),
                adapter.owner_id,
                handle,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(child)
        assert select.select([child.stdout], [], [], 15)[0]
        line = child.stdout.readline().strip()
        if line != "quarantined":
            _, error = child.communicate(timeout=5)
            pytest.fail(error or line)
        writer = subprocess.Popen(
            [sys.executable, "-c", writer_script, str(stage.parent / "bootstrap")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(writer)
        assert select.select([writer.stdout], [], [], 15)[0]
        assert writer.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            writer.communicate(timeout=0.2)
        _, error = child.communicate(input="exit\n", timeout=10)
        assert child.returncode == 0, error
        output, error = writer.communicate(timeout=10)
        assert writer.returncode == 0 and "admitted" in output, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    assert state(source) == before


def _close_failure_child(root, source, stage, owner_id, handle):
    if handle == "post_open":
        _post_open_close_failure_child(root, source, stage, owner_id)
        return
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        native_files,
    )
    from tldw_chatbook.Backup_Recovery import (
        storage_admission as storage,
    )
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    bootstrap.default_bootstrap_root = lambda: root
    adapter = next(row for row in install_adapters() if row.owner_id == owner_id)
    real_os = os
    observed = {"fd": None, "calls": 0}
    source_identity = source.stat().st_ino
    parent_identity = source.parent.stat().st_ino

    class FaultOS:
        def __getattr__(self, name):
            return getattr(real_os, name)

        def close(self, fd):
            if fd == observed["fd"]:
                observed["calls"] += 1
                raise OSError("attempted ambiguous FD retry")
            info = real_os.fstat(fd)
            candidates = list(stage.glob("sqlite-sources-*/*/source.sqlite3"))
            wanted = (
                handle == "source"
                and info.st_ino == source_identity
                or handle == "parent"
                and info.st_ino == parent_identity
                and bool(candidates)
                or handle == "output"
                and any(path.stat().st_ino == info.st_ino for path in candidates)
            )
            if observed["fd"] is None and wanted:
                real_os.close(fd)
                reused = real_os.open(real_os.devnull, real_os.O_RDONLY)
                if reused != fd:
                    real_os.dup2(reused, fd)
                    real_os.close(reused)
                observed.update(fd=fd, calls=1)
                raise OSError("closed selected FD then failed")
            real_os.close(fd)

    storage.os = bootstrap.os = native_files.os = FaultOS()
    try:
        with (
            admission_authority(root).maintenance(
                ("core", "bootstrap.unbound"), 2
            ) as session,
            session.capture_scope((source,), stage),
        ):
            item = StorageItem(owner_id, "fixture:owned", source, "included", ())
            adapter.capture(item, stage / "failed.db", Event())
    except (OSError, RuntimeError):
        pass
    assert observed["fd"] is not None and observed["calls"] == 1, observed
    real_os.fstat(observed["fd"])
    assert list(stage.glob("sqlite-sources-*"))
    assert storage._failed_capture_holds
    print("quarantined", flush=True)
    sys.stdin.readline()
    assert observed["calls"] == 1
    real_os.fstat(observed["fd"])
    real_os._exit(0)


def _replacement_group(source, root, table):
    root.mkdir(mode=0o700)
    target = root / "source.sqlite3"
    for suffix in ("", "-wal"):
        Path(str(target) + suffix).write_bytes(Path(str(source) + suffix).read_bytes())
    subprocess.run(
        [
            sys.executable,
            "-c",
            _WAL.replace("'WAL-only'", "'replacement'"),
            str(target),
            table,
        ],
        check=True,
        timeout=15,
    )
    return target


@pytest.mark.parametrize("replacement", ["main", "wal", "directory"])
def test_materialized_identity_cannot_be_rebound_to_valid_replacement(
    wal_owner, replacement
):
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, authority, adapter, item, table = wal_owner
    other = _replacement_group(source, stage.parent / "replacement", table)
    before = state(source)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        pytest.raises(ValueError, match="capture_sqlite_(target|staging)_changed"),
        session.capture_scope((source,), stage),
    ):
        assert adapter.validate(source) == ()
        target = _local.capture_scope.sqlite_snapshots[source][1]
        if replacement == "directory":
            target.parent.rename(target.parent.with_name("old-materialization"))
            other.parent.rename(target.parent)
        else:
            suffix = "-wal" if replacement == "wal" else ""
            os.replace(Path(str(other) + suffix), Path(str(target) + suffix))
        adapter.capture(item, stage / "untrusted.db", Event())
    assert state(source) == before


def test_materialized_swap_at_native_open_is_rejected(wal_owner, monkeypatch):
    from tldw_chatbook.Backup_Recovery.admission import _local
    from tldw_chatbook.DB import private_sqlite

    source, stage, authority, adapter, item, table = wal_owner
    other = _replacement_group(source, stage.parent / "replacement", table)
    before = state(source)
    original = private_sqlite.sqlite3.connect
    replaced = []

    def connect(database, *args, **kwargs):
        if not replaced and "source.sqlite3" in str(database):
            target = _local.capture_scope.sqlite_snapshots[source][1]
            os.replace(Path(str(other) + "-wal"), Path(str(target) + "-wal"))
            replaced.append(target)
        return original(database, *args, **kwargs)

    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        pytest.raises(ValueError, match="capture_sqlite_target_changed"),
        session.capture_scope((source,), stage),
    ):
        assert adapter.validate(source) == ()
        monkeypatch.setattr(private_sqlite.sqlite3, "connect", connect)
        adapter.capture(item, stage / "untrusted.db", Event())
    assert replaced
    assert state(source) == before


@pytest.mark.parametrize("short_read", [False, True])
def test_clean_wal_header_binds_empty_private_wal_before_read(
    wal_owner, monkeypatch, short_read
):
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, authority, adapter, item, _ = wal_owner
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert source.read_bytes()[18:20] == b"\x02\x02"
    assert not Path(str(source) + "-wal").exists()
    before = state(source)
    reads = []
    original_read = os.read
    original_inode = source.stat().st_ino

    def read(fd, count):
        if short_read and not reads and os.fstat(fd).st_ino == original_inode:
            reads.append(fd)
            return original_read(fd, min(count, 16))
        return original_read(fd, count)

    monkeypatch.setattr(os, "read", read)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage),
    ):
        assert adapter.validate(source) == ()
        target = _local.capture_scope.sqlite_snapshots[source][1]
        expected_wal = _local.capture_scope.sqlite_targets[target][1][1]
        assert expected_wal[5] == 0
        assert Path(str(target) + "-wal").stat().st_ino == expected_wal[1]
        adapter.capture(item, stage / "clean.db", Event())
        assert adapter.validate(stage / "clean.db") == ()
    assert state(source) == before
    assert bool(reads) is short_read


def test_private_replacement_after_backup_does_not_gain_capture_authority(
    wal_owner, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.admission import _local
    from tldw_chatbook.DB import private_sqlite

    source, stage, authority, adapter, item, table = wal_owner
    other = _replacement_group(source, stage.parent / "replacement", table)
    before = state(source)
    original = private_sqlite._backup_pages

    def backup(*args, **kwargs):
        original(*args, **kwargs)
        target = _local.capture_scope.sqlite_snapshots[source][1]
        os.replace(Path(str(other) + "-wal"), Path(str(target) + "-wal"))

    monkeypatch.setattr(private_sqlite, "_backup_pages", backup)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        pytest.raises(ValueError, match="capture_sqlite_target_changed"),
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, stage / "untrusted.db", Event())
    assert state(source) == before


def _post_open_close_failure_child(root, source, stage, owner_id):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.DB import private_sqlite

    bootstrap.default_bootstrap_root = lambda: root
    adapter = next(row for row in install_adapters() if row.owner_id == owner_id)
    original = private_sqlite.sqlite3.connect
    retained = []
    closes = []

    def connect(database, *args, **kwargs):
        connection = original(database, *args, **kwargs)
        if not retained and "source.sqlite3" in str(database):
            retained.append(connection)
            target = storage._local.capture_scope.sqlite_snapshots[source][1]
            wal = Path(str(target) + "-wal")
            info = wal.stat()
            os.utime(wal, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))

            def fail(self):
                closes.append(self)
                raise OSError("native connection close not established")

            type(connection).close = fail
        return connection

    private_sqlite.sqlite3.connect = connect
    try:
        with (
            admission_authority(root).maintenance(
                ("core", "bootstrap.unbound"), 2
            ) as session,
            session.capture_scope((source,), stage),
        ):
            adapter.capture(
                StorageItem(owner_id, "fixture:owned", source, "included", ()),
                stage / "failed.db",
                Event(),
            )
    except (ValueError, RuntimeError):
        pass
    assert len(retained) == len(closes) == 1
    assert retained[0].in_transaction is False
    assert list(stage.glob("sqlite-sources-*"))
    assert storage._failed_capture_holds
    print("quarantined", flush=True)
    sys.stdin.readline()
    assert len(closes) == 1
    assert retained[0].in_transaction is False
    os._exit(0)


def test_post_open_refusal_with_positive_close_retires_capture_resources(
    wal_owner, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.admission import _local
    from tldw_chatbook.DB import private_sqlite

    source, stage, authority, adapter, item, _ = wal_owner
    before = state(source)
    original = private_sqlite.sqlite3.connect
    retained = []

    def connect(database, *args, **kwargs):
        connection = original(database, *args, **kwargs)
        if not retained and "source.sqlite3" in str(database):
            retained.append(connection)
            target = _local.capture_scope.sqlite_snapshots[source][1]
            wal = Path(str(target) + "-wal")
            info = wal.stat()
            os.utime(wal, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        return connection

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", connect)
    with authority.maintenance(("core", "bootstrap.unbound"), 2) as session:
        with (
            pytest.raises(ValueError, match="capture_sqlite_target_changed"),
            session.capture_scope((source,), stage),
        ):
            adapter.capture(item, stage / "refused.db", Event())
        session._check()
        assert session._scopes[-1].resources == []
        with pytest.raises(sqlite3.ProgrammingError):
            retained[0].execute("SELECT 1")
        assert not list(stage.glob("sqlite-sources-*"))
    assert state(source) == before
