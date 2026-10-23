"""Native binding gate and source-backed SQLite allocation boundary."""

import sqlite3
import asyncio
import os
import subprocess
import threading
import time

import pytest

from Tests.private_profile import private_profile_test

from tldw_chatbook.Workspaces.project_index.allocation import (
    admit_connection,
    allocation_gate,
    retain_uncertain,
    verify_pre_open,
)
from tldw_chatbook.Workspaces.project_index.lifetime import (
    BindingLifetime,
    LifetimeBusy,
)


def namespace(group, digit):
    directory = group / (digit * 64)
    directory.mkdir()
    return directory / "workspace_project_index.db"


def test_two_namespaces_compete_for_one_native_allocation_gate(tmp_path):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    namespace(group, "a")
    namespace(group, "b")
    with allocation_gate(group):
        with pytest.raises(LifetimeBusy):
            BindingLifetime.acquire(group / "allocation-v1.lock")
    lease = BindingLifetime.acquire(group / "allocation-v1.lock", existing_only=True)
    lease.close()


def test_control_gate_borrows_same_live_lease_for_sequential_stores(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    group = tmp_path / ("f" * 64)
    group.mkdir()
    first = namespace(group, "a")
    second = namespace(group, "b")
    with allocation_gate(group) as outer:
        with allocation_gate(group) as borrowed:
            assert borrowed is outer
            one = WorkspaceProjectIndexDB(first)
            with one.connection() as conn:
                assert (
                    conn.execute("SELECT version FROM schema_version").fetchone()[0]
                    == 4
                )
            two = WorkspaceProjectIndexDB(second)
            with two.connection() as conn:
                assert (
                    conn.execute("SELECT version FROM schema_version").fetchone()[0]
                    == 4
                )
    assert first.is_file() and second.is_file()


def test_borrow_refuses_an_overlapping_independent_store(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    group = tmp_path / ("f" * 64)
    group.mkdir()
    first = WorkspaceProjectIndexDB(namespace(group, "a"))
    second = WorkspaceProjectIndexDB(namespace(group, "b"))
    with allocation_gate(group):
        with first.sql_unit():
            with pytest.raises(ValueError, match="overlapping"):
                with second.sql_unit():
                    pass


def test_borrow_refuses_another_thread_and_retains_uncertain_inner(tmp_path):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    result = []
    with allocation_gate(group) as outer:
        with allocation_gate(group) as inner:
            worker = threading.Thread(target=lambda: result.append(_try_gate(group)))
            worker.start()
            worker.join(timeout=3)
            assert not worker.is_alive()
            assert result == ["busy"]
            retain_uncertain(group, inner)
    assert outer is inner
    with pytest.raises(ValueError, match="retirement_unproven"):
        with allocation_gate(group):
            pass


def _try_gate(group):
    try:
        with allocation_gate(group):
            return "acquired"
    except LifetimeBusy:
        return "busy"


@pytest.mark.asyncio
async def test_gate_and_nested_store_refuse_another_async_execution(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    group = tmp_path / ("f" * 64)
    group.mkdir()
    store = WorkspaceProjectIndexDB(namespace(group, "a"))

    async def another_task():
        with pytest.raises(ValueError, match="borrow_refused"):
            with allocation_gate(group):
                pass
        with pytest.raises(ValueError, match="borrow_refused"):
            with store.sql_unit():
                pass

    with store.sql_unit():
        await asyncio.create_task(another_task())


def test_keyword_transactions_reduce_actual_cap_for_other_namespace(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    group = tmp_path / ("f" * 64)
    group.mkdir()
    first = WorkspaceProjectIndexDB(namespace(group, "a"))
    with first.transaction() as conn:
        initial_cap = conn.execute("PRAGMA main.max_page_count").fetchone()[0]
    second = WorkspaceProjectIndexDB(namespace(group, "b"))
    with second.transaction() as conn:
        conn.execute("CREATE TABLE payload(value BLOB)")
        conn.execute("INSERT INTO payload VALUES(?)", (b"x" * 65536,))
    with first.transaction() as conn:
        lower_cap = conn.execute("PRAGMA main.max_page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        old_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    assert old_pages <= lower_cap < initial_cap
    assert (initial_cap - lower_cap) * page_size >= second.db_path.stat().st_size // 2


@pytest.mark.skipif(os.name != "nt", reason="Windows junction control")
def test_preopen_refuses_windows_junction_namespace(tmp_path):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = group / ("a" * 64)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with allocation_gate(group):
        with pytest.raises(ValueError, match="alias"):
            verify_pre_open(junction / "workspace_project_index.db", group)
    assert not (outside / "workspace_project_index.db").exists()


def test_negative_scaled_allowance_refuses_without_clamping_or_writing(tmp_path):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    path = namespace(group, "a")
    with sqlite3.connect(path, isolation_level=None, timeout=0) as conn:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("CREATE TABLE records(value BLOB)")
        previous = conn.execute("PRAGMA max_page_count").fetchone()[0]
        before = path.read_bytes()
        with allocation_gate(group):
            with pytest.raises(ValueError, match="index_cache_limit"):
                admit_connection(conn, path, group, budget=65536)
        assert conn.execute("PRAGMA max_page_count").fetchone()[0] == previous
        assert path.read_bytes() == before


def test_scaled_cap_reserves_other_namespace_and_every_old_page(tmp_path):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    path = namespace(group, "a")
    other = namespace(group, "b")
    other.write_bytes(b"x" * 8192)
    with sqlite3.connect(path, isolation_level=None, timeout=0) as conn:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("CREATE TABLE records(value BLOB)")
        assert conn.execute("PRAGMA page_count").fetchone()[0] == 3
        assert conn.execute("PRAGMA page_size").fetchone()[0] == 4096
        with allocation_gate(group):
            cap = admit_connection(conn, path, group, budget=300000)
            assert cap == 35
            assert conn.execute("PRAGMA max_page_count").fetchone()[0] == 35


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm", "-unknown"])
def test_preopen_refuses_nonempty_or_unknown_sidecar(tmp_path, suffix):
    group = tmp_path / ("f" * 64)
    group.mkdir()
    path = namespace(group, "a")
    sidecar = path.with_name(path.name + suffix)
    sidecar.write_bytes(b"evidence")
    with allocation_gate(group):
        with pytest.raises(ValueError, match="sidecar"):
            verify_pre_open(path, group)
    assert sidecar.read_bytes() == b"evidence"


def test_commit_failure_rolls_back_and_retires_native_before_gate_reuse(tmp_path):
    from Tests.Workspaces.project_index_support import open_semantic_store

    store = open_semantic_store(tmp_path)
    store.close_native()
    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as conn:
            conn.execute("PRAGMA defer_foreign_keys=ON")
            conn.execute(
                "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end)"
                " VALUES('missing.py','x',0,1,1,1)"
            )
    assert getattr(store._thread_local, "conn", None) is None
    with allocation_gate(store.db_path.parent.parent):
        pass
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    store.close_native()


class GateBarrierScanner:
    """Pause before a scanner SQL callback without retaining a SQL unit."""

    def __init__(self, stage):
        from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner

        class Scanner(ProjectScanner):
            def __init__(inner):
                super().__init__()
                inner.ready = threading.Event()
                inner.release = threading.Event()

            def scan(inner, *args, **kwargs):
                if stage == "frontier":
                    inner.ready.set()
                    assert inner.release.wait(5)
                    return super().scan(*args, **kwargs)
                emit = args[4]

                def held(page):
                    inner.ready.set()
                    assert inner.release.wait(5)
                    return emit(page)

                return super().scan(*args[:4], held, *args[5:], **kwargs)

        self.scanner = Scanner()


@pytest.mark.parametrize("stage", ["frontier", "emit"])
@private_profile_test
async def test_keyword_writer_retries_busy_gate_after_finite_unit(
    tmp_path, request, stage
):
    """A real scanner reaches Ready after another unit releases the gate."""
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    barrier = GateBarrierScanner(stage).scanner
    rig = Rig(tmp_path, barrier)
    try:
        (rig.root / "sample.txt").write_text("finite unit marker")
        rig.enable()
        job_id = rig.refresh()
        assert barrier.ready.wait(5)
        group = index_store_path(rig.data, rig.key).parent.parent
        with allocation_gate(group):
            barrier.release.set()
            time.sleep(0.1)
            assert rig.service._runs[job_id].state != "Error"
        run = rig.service._runs[job_id]
        run.supervisor.join(timeout=10)
        assert run.native_retired
        status = rig.service.status(rig.key)
        assert status.state == "Ready"
        assert status.indexed == 1
    finally:
        barrier.release.set()
        await rig.close()


@private_profile_test
async def test_search_under_other_native_unit_returns_safe_unavailable(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexSearchAuthority,
        ProjectSearchResult,
    )

    rig = Rig(tmp_path)
    entered, release = threading.Event(), threading.Event()
    holder = None
    try:
        (rig.root / "sample.txt").write_text("private marker")
        rig.enable()
        rig.refresh()
        scope = rig.settle().scope
        group = index_store_path(rig.data, rig.key).parent.parent

        def hold():
            with allocation_gate(group):
                entered.set()
                assert release.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        assert entered.wait(5)
        auth = ProjectIndexSearchAuthority(
            scope,
            rig.live.root,
            rig.live.chain,
            (),
            1,
            lambda: True,
            None,
            lambda _: pytest.fail("busy search must not Capture"),
        )
        assert rig.service.search(auth, "marker") == ProjectSearchResult(
            "unavailable", (), None, False
        )
    finally:
        release.set()
        if holder is not None:
            holder.join(timeout=5)
            assert not holder.is_alive()
        await rig.close()


@private_profile_test
async def test_status_reports_busy_for_four_second_foreign_native_owner(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    rig = Rig(tmp_path)
    entered, release = threading.Event(), threading.Event()
    holder = None
    timer = None
    try:
        expected = rig.enable()
        group = index_store_path(rig.data, rig.key).parent.parent

        def hold():
            with allocation_gate(group):
                entered.set()
                assert release.wait(6)

        holder = threading.Thread(target=hold)
        holder.start()
        assert entered.wait(5)
        timer = threading.Timer(4.0, release.set)
        timer.start()
        with pytest.raises(LifetimeBusy):
            rig.service.status(rig.key)
        assert rig.service._pending_reads == {}
        assert release.wait(5)
        holder.join(timeout=5)
        assert not holder.is_alive()
        status = rig.service.status(rig.key)
        assert status.scope == expected.scope and status.state == expected.state
    finally:
        release.set()
        if timer is not None:
            timer.join(timeout=5)
        if holder is not None:
            holder.join(timeout=5)
            assert not holder.is_alive()
        await rig.close()


@private_profile_test
async def test_stopping_scanner_while_gate_busy_retires_after_release(
    tmp_path, request
):
    """Cancellation interrupts gate retry and final status settles after release."""
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    barrier = GateBarrierScanner("frontier").scanner
    rig = Rig(tmp_path, barrier)
    try:
        (rig.root / "sample.txt").write_text("cancel marker")
        rig.enable()
        job_id = rig.refresh()
        assert barrier.ready.wait(5)
        group = index_store_path(rig.data, rig.key).parent.parent
        with allocation_gate(group):
            barrier.release.set()
            time.sleep(0.1)
            run = rig.service._runs[job_id]
            run.stop.set()
        run.supervisor.join(timeout=10)
        assert run.native_retired
        assert rig.service.status(rig.key).state == "Interrupted"
    finally:
        barrier.release.set()
        await rig.close()


@private_profile_test
async def test_pending_read_handoff_precedes_writer_reacquisition(
    tmp_path, request, monkeypatch
):
    """A real waiting status read gets a native gate window between writer units."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    import importlib

    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    rig = Rig(tmp_path)
    rig.enable()
    store = rig.store(rig.key)
    run = SimpleNamespace(
        stop=threading.Event(),
        claim=SimpleNamespace(scope=SimpleNamespace(key=rig.key)),
    )
    entered = threading.Event()
    release = threading.Event()
    missed = threading.Event()
    order = []
    errors = []
    result = []
    module = importlib.import_module("tldw_chatbook.DB.Workspace_Project_Index_DB")
    original_gate = module.allocation_gate

    @contextmanager
    def observed_gate(*args, **kwargs):
        try:
            with original_gate(*args, **kwargs) as lease:
                if (
                    threading.current_thread().name == "status-reader"
                    and "reader" not in order
                ):
                    order.append("reader")
                yield lease
        except LifetimeBusy:
            if threading.current_thread().name == "status-reader":
                missed.set()
            raise

    monkeypatch.setattr(module, "allocation_gate", observed_gate)

    def writer():
        try:
            for number in range(4):

                def unit():
                    with store.sql_unit():
                        with store.connection() as conn:
                            assert conn.execute("SELECT 1").fetchone()[0] == 1
                        if number == 0:
                            entered.set()
                            assert release.wait(5)
                        elif number == 1:
                            order.append("writer2")

                rig.service._retry_job_sql(run, unit)
        except BaseException as error:
            errors.append(error)
        finally:
            try:
                store.close_native()
            except BaseException as error:
                errors.append(error)

    def reader():
        try:
            result.append(rig.service.status(rig.key))
        except BaseException as error:
            errors.append(error)

    writer_thread = threading.Thread(target=writer, name="unit-writer")
    reader_thread = threading.Thread(target=reader, name="status-reader")
    try:
        writer_thread.start()
        assert entered.wait(5)
        reader_thread.start()
        assert missed.wait(5)
        time.sleep(0.05)
        release.set()
        writer_thread.join(timeout=8)
        reader_thread.join(timeout=8)
        assert not writer_thread.is_alive() and not reader_thread.is_alive()
        assert not errors
        assert result and result[0].scope.key == rig.key
        assert rig.service._pending_reads == {}
        assert order[:2] == ["reader", "writer2"]
        group = index_store_path(rig.data, rig.key).parent.parent
        with allocation_gate(group, existing_only=True):
            pass
    finally:
        run.stop.set()
        release.set()
        writer_thread.join(timeout=8)
        reader_thread.join(timeout=8)
        await rig.close()


@private_profile_test
async def test_keyword_lookup_during_real_scanner_wait_outside_sql(
    tmp_path, request, monkeypatch
):
    """A running worker request leaves the binding SQL gate free for live lookup."""
    import hashlib

    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ProjectIndexSearchAuthority,
    )

    rig = Rig(tmp_path)
    entered, release = threading.Event(), threading.Event()
    try:
        source = rig.root / "sample.txt"
        source.write_text("live scanner marker")
        rig.enable()
        first_job = rig.refresh()
        ready = rig.settle()
        rig.service._runs[first_job].supervisor.join(timeout=10)
        assert rig.service._runs[first_job].native_retired
        original_unit = rig.service._scanner._unit

        def held_unit(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                assert release.wait(10)
            return original_unit(*args, **kwargs)

        monkeypatch.setattr(rig.service._scanner, "_unit", held_unit)
        rig.refresh()
        assert entered.wait(10)
        assert rig.service._scanner.worker_pid > 0
        assert not rig.service._scanner.retired
        group = index_store_path(rig.data, rig.key).parent.parent
        with allocation_gate(group, existing_only=True):
            pass

        def live_read(candidate):
            data = (rig.root / candidate.relative_path).read_bytes()
            return FileCapture(
                candidate.relative_path,
                candidate.file_identity,
                (rig.root / candidate.relative_path).stat().st_mtime_ns,
                len(data),
                hashlib.sha256(data).hexdigest(),
                data.decode("utf-8"),
            )

        search = ProjectIndexSearchAuthority(
            ready.scope,
            rig.live.root,
            rig.live.chain,
            (),
            rig.live.authority_revision,
            lambda: True,
            None,
            live_read,
        )
        result = rig.service.search(search, "marker")
        assert result.status == "partial"
        assert result.coverage is None
        assert result.hits and "marker" in result.hits[0].excerpt
    finally:
        release.set()
        await rig.close()


def test_group_semantic_peer_qualifies_keyword_target_before_native_open(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from Tests.Workspaces.project_index_support import (
        make_index_key,
        open_semantic_store,
    )
    from Tests.Workspaces.test_project_semantic_settings import settings

    first = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    first.configure_semantic(
        first.load_status(key).scope,
        settings(),
        enabled=True,
        expected_semantic_revision=0,
    )
    first.close_native()
    second = module.WorkspaceProjectIndexDB(namespace(first.db_path.parent.parent, "b"))
    observed = []
    original = module.admit_connection

    def recorded(conn, path, group, **kwargs):
        observed.append(path)
        return original(conn, path, group, **kwargs)

    monkeypatch.setattr(module, "admit_connection", recorded)
    try:
        with second.transaction() as conn:
            conn.execute("CREATE TABLE peer_keyword(value TEXT)")
        assert second.db_path in observed
    finally:
        second.close_native()


def test_pure_keyword_group_uses_keyword_allowance(tmp_path, monkeypatch):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module

    group = tmp_path / ("f" * 64)
    group.mkdir()
    first = module.WorkspaceProjectIndexDB(namespace(group, "a"))
    first.close_native()
    second = module.WorkspaceProjectIndexDB(namespace(group, "b"))
    monkeypatch.setattr(
        module,
        "admit_connection",
        lambda *args, **kwargs: pytest.fail("pure keyword must not qualify source"),
    )
    try:
        with second.transaction() as conn:
            conn.execute("CREATE TABLE peer_keyword(value TEXT)")
    finally:
        second.close_native()


def test_read_to_write_nesting_refuses_before_target_mutation(tmp_path):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module

    group = tmp_path / ("f" * 64)
    group.mkdir()
    store = module.WorkspaceProjectIndexDB(namespace(group, "a"))
    try:
        with store.sql_unit():
            with pytest.raises(ValueError, match="write"):
                with store.transaction() as conn:
                    conn.execute("CREATE TABLE should_not_exist(value TEXT)")
        with store.connection() as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE name='should_not_exist'"
                ).fetchone()
                is None
            )
    finally:
        store.close_native()


def test_missing_group_gate_with_existing_v4_peer_refuses_repeatedly(tmp_path):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from Tests.Workspaces.project_index_support import open_semantic_store

    first = open_semantic_store(tmp_path)
    group = first.db_path.parent.parent
    first.close_native()
    gate = group / "allocation-v1.lock"
    gate.unlink()
    second = namespace(group, "b")
    for _ in range(2):
        with pytest.raises(ValueError, match="index_allocation_control_missing"):
            module.WorkspaceProjectIndexDB(second)
        assert not gate.exists()
        assert not second.exists()


def test_retained_generation_after_revoke_qualifies_peer_and_closes_probes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from Tests.Workspaces.project_index_support import (
        make_index_key,
        open_semantic_store,
    )
    from Tests.Workspaces.test_project_semantic_settings import settings

    first = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = first.load_status(key).scope
    configured = first.configure_semantic(
        scope, settings(), enabled=True, expected_semantic_revision=0
    )
    first.configure_semantic(
        scope,
        None,
        enabled=False,
        expected_semantic_revision=configured.semantic_revision,
    )
    with first.transaction(semantic=True) as conn:
        conn.execute(
            "INSERT INTO semantic_generations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "retained-1",
                key.namespace,
                scope.epoch,
                scope.processing_revision,
                2,
                "huggingface",
                "e5-small-v2",
                "digest-a",
                "digest-b",
                "artifact",
                "test-v1",
                1,
                0,
                None,
                None,
                "Retained",
                0,
            ),
        )
    first.close_native()
    observed = []
    opened = []
    original_admit = module.admit_connection
    original_connect = module.connect_private_sqlite

    def recorded_admit(conn, path, group, **kwargs):
        observed.append(path)
        return original_admit(conn, path, group, **kwargs)

    def recorded_connect(owner, path, **kwargs):
        conn = original_connect(owner, path, **kwargs)
        if kwargs.get("read_only"):
            opened.append(conn)
        return conn

    monkeypatch.setattr(module, "admit_connection", recorded_admit)
    monkeypatch.setattr(module, "connect_private_sqlite", recorded_connect)
    second = module.WorkspaceProjectIndexDB(namespace(first.db_path.parent.parent, "b"))
    try:
        with second.transaction() as conn:
            conn.execute("CREATE TABLE retained_peer(value TEXT)")
        assert second.db_path in observed
        assert opened
        for conn in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(conn)
    finally:
        second.close_native()


def test_project_owner_peer_read_only_open_never_writes_or_creates(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.Workspaces.project_index.models import OWNER_ID

    group = tmp_path / ("f" * 64)
    group.mkdir()
    existing = WorkspaceProjectIndexDB(namespace(group, "a"))
    path = existing.db_path
    existing.close_native()
    conn = connect_private_sqlite(
        OWNER_ID, path, read_only=True, must_exist=True, isolation_level=None
    )
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 4
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE should_not_write(value TEXT)")
    finally:
        conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.in_transaction.__get__(conn)
    missing = group / ("b" * 64) / "workspace_project_index.db"
    missing.parent.mkdir()
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError, OSError)):
        connect_private_sqlite(
            OWNER_ID, missing, read_only=True, must_exist=True, isolation_level=None
        )
    assert not missing.exists()


def test_pre_gate_peer_close_uncertainty_retains_exact_native_and_blocks_retry(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from Tests.Workspaces.project_index_support import open_semantic_store

    first = open_semantic_store(tmp_path)
    group = first.db_path.parent.parent
    first.close_native()
    gate = group / "allocation-v1.lock"
    gate.unlink()
    second = namespace(group, "b")
    original = module.connect_private_sqlite
    retained = {}

    class CloseFails(sqlite3.Connection):
        def close(self):
            raise OSError("injected_peer_close_uncertainty")

    def failing_owner(owner, path, **kwargs):
        conn = original(owner, path, factory=CloseFails, **kwargs)
        retained["connection"] = conn
        return conn

    monkeypatch.setattr(module, "_UNCERTAIN_PEER_CONNECTIONS", {})
    monkeypatch.setattr(module, "connect_private_sqlite", failing_owner)
    try:
        with pytest.raises(
            module.ProjectIndexRetirementUnproven, match="peer_retirement"
        ):
            module.WorkspaceProjectIndexDB(second)
        conn = retained["connection"]
        assert module._UNCERTAIN_PEER_CONNECTIONS[group] is conn
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        assert not gate.exists() and not second.exists()
        monkeypatch.setattr(module, "connect_private_sqlite", original)
        with pytest.raises(
            module.ProjectIndexRetirementUnproven, match="peer_retirement"
        ):
            module.WorkspaceProjectIndexDB(second)
        with pytest.raises(
            module.ProjectIndexRetirementUnproven, match="peer_retirement"
        ):
            module.WorkspaceProjectIndexDB(first.db_path, cleanup_only=True)
        assert not gate.exists() and not second.exists()
    finally:
        if "connection" in retained:
            sqlite3.Connection.close(retained["connection"])


def test_nested_other_store_write_refuses_before_peer_probe(tmp_path, monkeypatch):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module

    group = tmp_path / ("f" * 64)
    group.mkdir()
    first = module.WorkspaceProjectIndexDB(namespace(group, "a"))
    second = module.WorkspaceProjectIndexDB(namespace(group, "b"))
    first.close_native()
    second.close_native()
    original = module.connect_private_sqlite
    probes = []

    def observed(owner, path, **kwargs):
        if kwargs.get("read_only"):
            probes.append(path)
        return original(owner, path, **kwargs)

    try:
        with first.sql_unit(write=True):
            with first.connection() as conn:
                assert conn.execute("SELECT 1").fetchone()[0] == 1
            monkeypatch.setattr(module, "connect_private_sqlite", observed)
            with pytest.raises(ValueError, match="overlapping"):
                with second.transaction() as conn:
                    conn.execute("CREATE TABLE should_not_exist(value TEXT)")
            assert probes == []
    finally:
        first.close_native()
        second.close_native()


def test_disabled_settings_without_generations_keep_keyword_allowance(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from Tests.Workspaces.project_index_support import (
        make_index_key,
        open_semantic_store,
    )
    from Tests.Workspaces.test_project_semantic_settings import settings

    first = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    first.configure_semantic(
        first.load_status(key).scope,
        settings(),
        enabled=False,
        expected_semantic_revision=0,
    )
    first.close_native()
    monkeypatch.setattr(
        module,
        "admit_connection",
        lambda *args, **kwargs: pytest.fail("disabled metadata is keyword-only"),
    )
    second = module.WorkspaceProjectIndexDB(namespace(first.db_path.parent.parent, "b"))
    try:
        with second.transaction() as conn:
            conn.execute("CREATE TABLE keyword_only(value TEXT)")
    finally:
        second.close_native()


def test_gated_peer_close_uncertainty_retains_connection_and_gate(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module
    from tldw_chatbook.Workspaces.project_index import allocation
    from Tests.Workspaces.project_index_support import open_semantic_store

    first = open_semantic_store(tmp_path)
    group = first.db_path.parent.parent
    first.close_native()
    second = namespace(group, "b")
    original = module.connect_private_sqlite
    retained = {}

    class CloseFails(sqlite3.Connection):
        def close(self):
            raise OSError("injected_gated_peer_close_uncertainty")

    def failing_owner(owner, path, **kwargs):
        conn = original(owner, path, factory=CloseFails, **kwargs)
        retained["connection"] = conn
        return conn

    monkeypatch.setattr(module, "_UNCERTAIN_PEER_CONNECTIONS", {})
    monkeypatch.setattr(allocation, "_QUARANTINED", {})
    monkeypatch.setattr(module, "connect_private_sqlite", failing_owner)
    try:
        with pytest.raises(
            module.ProjectIndexRetirementUnproven, match="peer_retirement"
        ):
            module.WorkspaceProjectIndexDB(second)
        conn = retained["connection"]
        assert module._UNCERTAIN_PEER_CONNECTIONS[group] is conn
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        assert group in allocation._QUARANTINED
        with pytest.raises(ValueError, match="retirement_unproven"):
            with allocation_gate(group, existing_only=True):
                pass
        assert not second.exists()
    finally:
        if "connection" in retained:
            sqlite3.Connection.close(retained["connection"])
        if group in allocation._QUARANTINED:
            allocation._QUARANTINED.pop(group).close()
