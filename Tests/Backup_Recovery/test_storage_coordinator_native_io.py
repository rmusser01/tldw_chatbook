"""Real private-child causal controls; no guard/callback replacement."""

import json
import os

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
# ruff: noqa: E402, E721 -- Original import order and concrete type qualification.

import _thread
import ast
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from types import CodeType, FunctionType
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger

logger.remove()
route, outcome = sys.argv[1:]
assert route in {"scope_before_count", "scope_after_count", "operation_check"}
assert outcome == "ordinary" and os.name == "nt"


def shape(code):
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_exceptiontable,
        code.co_stacksize,
        code.co_flags,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            shape(value) if type(value) is CodeType else value
            for value in code.co_consts
        ),
    )


def child(code, qualified_name):
    for name in qualified_name.split("."):
        matches = [
            value
            for value in code.co_consts
            if type(value) is CodeType and value.co_name == name
        ]
        assert len(matches) == 1, (code.co_name, name, len(matches))
        code = matches[0]
    return code


def main():
    base = Path(os.environ["TLDW_CONFIG_PATH"]).absolute().parent.parent
    selector = Path(os.environ["TLDW_CONFIG_PATH"]).absolute()
    data = base / "data"
    selector.write_text(
        '[general]\nusers_name="coordinator"\n[paths]\ndata_dir="'
        + data.as_posix()
        + '"\n',
        encoding="utf-8",
    )
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery import raw_participants as raw, participants
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Utils import windows_files

    db = AgentRunsDB(
        config.get_user_data_dir() / "agent_runs.db",
        client_id="coordinator-native",
        reconcile_on_init=False,
    )
    with db.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = ?", ("table",)
            ).fetchone()[0]
            > 0
        )
    db.close()
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("seeded native SQLite connection remains open")

    root = bootstrap.default_bootstrap_root()
    with storage._lock:
        assert all(key == (os.getpid(), str(root)) for key in storage._startups)
        assert not (storage._live_leases - set(storage._startups.values()))
        assert not storage._pending_acquisitions and not storage._operations
        assert not storage._raw_operations and not storage._retiring_holds
    # Explicitly retire only this private child's original startup owners before
    # the supported intact-profile enrollment, as the existing native fixtures do.
    storage._shutdown()
    authority = admission_authority(root)
    authority.register("profile", (selector.parent, data))
    bind_profile(root, selector, ("profile",), authority.control_root)
    participant = participants._repository_participant(db)
    victim_path = data / "independent-retirement.sqlite"
    victim = storage.acquire_storage(victim_path)
    assert victim.execution_context(victim_path)[1] == ("profile",)

    project = Path.cwd().resolve()
    modules = (
        config,
        bootstrap,
        storage,
        raw,
        participants,
        windows_files,
        sys.modules[AgentRunsDB.__module__],
    )
    sources, compiled, original_sources, source_bytes = {}, {}, {}, {}
    for module in modules:
        assert sys.modules[module.__name__] is module
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(project)
        assert Path(module.__spec__.origin).resolve() == path
        source = path.read_bytes()
        source_bytes[module.__name__] = source
        sources[module.__name__] = hashlib.sha256(source).hexdigest()
        original_sources[module.__name__] = (
            module,
            path,
            module.__spec__,
            module.__spec__.origin,
            module.__file__,
        )
        compiled[module.__name__] = compile(
            source.decode("utf-8"), str(path), "exec", dont_inherit=True
        )

    functions = (
        (storage._acquire_storage, storage, "_acquire_storage"),
        (storage._scope, storage, "_scope"),
        (storage._Operation.check, storage, "_Operation.check"),
        (storage.StorageLease.close, storage, "StorageLease.close"),
        (bootstrap._records, bootstrap, "_records"),
        (bootstrap._registry, bootstrap, "_registry"),
        (windows_files._Native.open_handle, windows_files, "_Native.open_handle"),
    )
    metadata = []
    for function, module, qualified_name in functions:
        assert type(function) is FunctionType and function.__globals__ is vars(module)
        assert (
            Path(function.__code__.co_filename).resolve()
            == original_sources[module.__name__][1]
        )
        assert shape(function.__code__) == shape(
            child(compiled[module.__name__], qualified_name)
        )
        keyword_defaults = function.__kwdefaults__
        metadata.append(
            (
                function,
                module,
                function.__code__,
                function.__defaults__,
                keyword_defaults,
                () if keyword_defaults is None else tuple(keyword_defaults.items()),
                function.__closure__,
            )
        )
    protected = (
        storage.acquire_storage,
        storage._acquire_storage,
        storage._scope,
        storage._Operation.check,
        storage.StorageLease.close,
        storage._begin_local_pause,
        raw._check,
        bootstrap._records,
        bootstrap._registry,
        bootstrap.default_bootstrap_root,
        config._load_settings_guarded,
        config._load_settings_uncached,
        windows_files._Native.open_handle,
    )
    coordinator = storage._lock
    assert type(coordinator) is _thread.RLock
    lock_owned = inspect.getattr_static(_thread.RLock, "_is_owned")
    lock_acquire = inspect.getattr_static(_thread.RLock, "acquire")
    lock_release = inspect.getattr_static(_thread.RLock, "release")
    scope_code, acquire_code = (
        storage._scope.__code__,
        storage._acquire_storage.__code__,
    )
    check_code, native_code = (
        storage._Operation.check.__code__,
        windows_files._Native.open_handle.__code__,
    )
    close_code = storage.StorageLease.close.__code__
    close_ast = next(
        node
        for node in ast.walk(ast.parse(source_bytes[storage.__name__]))
        if type(node) is ast.FunctionDef
        and node.name == "close"
        and node.lineno == close_code.co_firstlineno
    )
    close_lock_line = next(
        node.lineno for node in close_ast.body if type(node) is ast.With
    )
    entered, release, native_returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    close_at_lock, close_done, worker_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    actor, closer, held_frame = None, None, None
    capture, rows, failures, acquisitions, secondary = {}, [], [], [], {}
    scope_starts, native_starts = 0, 0
    row_overflow = 0

    def current():
        return (
            storage._lock is coordinator
            and type(coordinator) is _thread.RLock
            and all(
                sys.modules[name] is module
                and module.__spec__ is spec
                and module.__file__ == filename
                and module.__spec__.origin == origin
                for name, (
                    module,
                    path,
                    spec,
                    origin,
                    filename,
                ) in original_sources.items()
            )
            and inspect.getattr_static(_thread.RLock, "_is_owned") is lock_owned
            and inspect.getattr_static(_thread.RLock, "acquire") is lock_acquire
            and inspect.getattr_static(_thread.RLock, "release") is lock_release
            and all(
                before is after
                for before, after in zip(
                    protected,
                    (
                        storage.acquire_storage,
                        storage._acquire_storage,
                        storage._scope,
                        storage._Operation.check,
                        storage.StorageLease.close,
                        storage._begin_local_pause,
                        raw._check,
                        bootstrap._records,
                        bootstrap._registry,
                        bootstrap.default_bootstrap_root,
                        config._load_settings_guarded,
                        config._load_settings_uncached,
                        windows_files._Native.open_handle,
                    ),
                    strict=True,
                )
            )
            and all(
                function.__code__ is code
                and function.__globals__ is vars(module)
                and function.__defaults__ is defaults
                and function.__kwdefaults__ is keyword_defaults
                and (
                    keyword_defaults is None
                    or len(keyword_defaults) == len(keyword_items)
                    and all(
                        key in keyword_defaults and keyword_defaults[key] is value
                        for key, value in keyword_items
                    )
                )
                and function.__closure__ is closure
                for (
                    function,
                    module,
                    code,
                    defaults,
                    keyword_defaults,
                    keyword_items,
                    closure,
                ) in metadata
            )
        )

    def record(kind, code, line):
        nonlocal row_overflow
        if len(rows) < 64:
            rows.append(
                {
                    "event": kind,
                    "body": code.co_qualname,
                    "line": line,
                    "thread": "read"
                    if threading.current_thread() is actor
                    else "retire",
                }
            )
        else:
            row_overflow += 1

    def frames(code):
        frame = sys._getframe(1)
        for _ in range(12):
            if frame is None:
                break
            if frame.f_code is code:
                return frame
            frame = frame.f_back
        return None

    def observe(kind, code):
        nonlocal scope_starts, native_starts, held_frame
        thread = threading.current_thread()
        if thread is closer and code is close_code:
            frame = frames(code)
            assert frame is not None and frame.f_globals is vars(storage)
            if frame.f_locals.get("self") is victim:
                if kind == "line" and frame.f_lineno == close_lock_line:
                    record(kind, code, frame.f_lineno)
                    close_at_lock.set()
                elif kind in {"start", "return"}:
                    record(kind, code, frame.f_lineno)
            return
        if thread is not actor:
            return
        if code is scope_code and kind == "start":
            scope_starts += 1
        if code is not native_code:
            return
        frame = frames(code)
        assert frame is not None and frame.f_globals is vars(windows_files)
        parent, scope_frame, acquire_frame, operation_frame, reader_frame = (
            frame.f_back,
            None,
            None,
            None,
            None,
        )
        for _ in range(100):
            if parent is None:
                break
            if parent.f_code is scope_code:
                scope_frame = parent
            if parent.f_code is acquire_code:
                acquire_frame = parent
            if parent.f_code is check_code:
                operation_frame = parent
            if parent.f_code in {
                bootstrap._records.__code__,
                bootstrap._registry.__code__,
            }:
                reader_frame = parent
            parent = parent.f_back
        eligible = acquire_frame is not None and (
            (route == "operation_check" and operation_frame is not None)
            or (
                route.startswith("scope_")
                and scope_frame is not None
                and reader_frame is not None
                and scope_starts == (1 if route == "scope_before_count" else 2)
            )
        )
        if not eligible:
            return
        if kind == "return" and held_frame is frame:
            native_returned.set()
            record(kind, code, frame.f_lineno)
            held_frame = None
            return
        if kind != "start":
            return
        native_starts += 1
        if capture:
            return
        assert current()
        attempt = acquire_frame.f_locals["attempt"]
        assert type(attempt) is storage._Acquisition
        assert attempt in storage._pending_acquisitions
        assert (
            attempt.pid == os.getpid()
            and attempt.thread is thread
            and attempt.task is None
        )
        assert acquire_frame.f_locals["root"] == root
        assert acquire_frame.f_locals["selector"] == selector
        assert acquire_frame.f_locals["path"] == db.db_path
        if operation_frame is not None:
            operation = operation_frame.f_locals["self"]
            assert (
                operation is attempt.operation and operation.participant is participant
            )
            assert (
                operation in storage._operations
                and operation.lease in storage._live_leases
            )
        held_frame = frame
        capture.update(
            original_native_boundary=True,
            coordinator_owned_by_read_actor=coordinator._is_owned(),
            scope_ordinal=scope_starts,
            actual_actor=threading.get_ident(),
            attempt_live=True,
            source_path_equal=True,
            original_nested_operation=operation_frame is not None,
        )
        record(kind, code, frame.f_lineno)
        entered.set()
        assert release.wait(10), "test-owned original native read release expired"

    def read():
        lease = None
        try:
            if route == "operation_check":
                with participant.operation():
                    lease = storage.acquire_storage(db.db_path)
                    acquisitions.append(lease.execution_context(db.db_path)[1])
                    lease.close()
                    lease = None
            else:
                lease = storage.acquire_storage(db.db_path)
                acquisitions.append(lease.execution_context(db.db_path)[1])
                lease.close()
                lease = None
        except BaseException as error:
            failures.append(("read", type(error).__name__))
        finally:
            if lease is not None:
                lease.close()
            db.close()
            worker_done.set()

    def retire():
        try:
            acquired = coordinator.acquire(blocking=False)
            secondary["nonblocking_coordinator_acquired"] = acquired
            if acquired:
                coordinator.release()
            victim.close()
        except BaseException as error:
            failures.append(("retire", type(error).__name__))
        finally:
            close_done.set()

    monitor = sys.monitoring
    original_tools = tuple(monitor.get_tool(index) for index in range(6))
    tool = next((index for index in range(6) if monitor.get_tool(index) is None), None)
    assert tool is not None
    observed_codes = (scope_code, native_code, close_code)
    monitor.use_tool_id(tool, "storage-coordinator-native-io-control")
    monitor.register_callback(
        tool, monitor.events.PY_START, lambda code, offset: observe("start", code)
    )
    monitor.register_callback(
        tool,
        monitor.events.PY_RETURN,
        lambda code, offset, value: observe("return", code),
    )
    monitor.register_callback(
        tool, monitor.events.LINE, lambda code, line: observe("line", code)
    )
    for code in observed_codes:
        mask = monitor.events.PY_START | monitor.events.PY_RETURN
        if code is close_code:
            mask |= monitor.events.LINE
        monitor.set_local_events(tool, code, mask)
    assert monitor.get_events(tool) == 0
    actor = threading.Thread(target=read, name="coordinator-original-native-read")
    closer = threading.Thread(target=retire, name="coordinator-independent-retirement")
    observed_close_before_release = False
    try:
        actor.start()
        assert entered.wait(10), "original qualified native read boundary not reached"
        closer.start()
        assert close_at_lock.wait(
            10
        ), "original independent close never reached coordinator"
        observed_close_before_release = close_done.wait(0.1)
    finally:
        release.set()
        if actor.ident is not None:
            actor.join(10)
        if closer.ident is not None:
            closer.join(10)
        for code in observed_codes:
            monitor.set_local_events(tool, code, 0)
        try:
            assert monitor.get_events(tool) == 0
        finally:
            monitor.set_events(tool, 0)
            for event in (
                monitor.events.PY_START,
                monitor.events.PY_RETURN,
                monitor.events.LINE,
            ):
                monitor.register_callback(tool, event, None)
            monitor.free_tool_id(tool)
        victim.close()
        db.close()
    assert not actor.is_alive() and not closer.is_alive()
    assert worker_done.is_set() and close_done.is_set()
    assert (
        native_returned.is_set()
    ), "held original native opener did not return normally"
    assert current() and not failures, failures
    assert tuple(monitor.get_tool(index) for index in range(6)) == original_tools
    assert acquisitions == [("profile",)]
    assert native_starts >= 1 and capture and row_overflow == 0
    for name, (module, path, spec, origin, filename) in original_sources.items():
        assert sys.modules[name] is module and module.__spec__ is spec
        assert module.__spec__.origin == origin and module.__file__ == filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sources[name]
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("seeded native SQLite descriptor reopened")
    with coordinator:
        census = {
            "ordinary": len(storage._live_leases - set(storage._startups.values())),
            "pending": len(storage._pending_acquisitions),
            "operations": len(storage._operations),
            "raw": len(storage._raw_operations),
            "raw_states": len(raw._states),
            "retiring": len(storage._retiring_holds),
        }
    assert not any(census.values()), census
    assert not network_guard.blocked_attempts()
    receipt = {
        "route": route,
        "diagnostic_only": False,
        "coverage": capture,
        "secondary": secondary,
        (
            "independent_close_finished_" "while_native_read_held"
        ): observed_close_before_release,
        "held_original_native_returned": native_returned.is_set(),
        "seeded_native_sqlite_physically_closed": True,
        "threads_physically_settled": True,
        "final_census": census,
        "source_hashes": sources,
        "source_current": True,
        "global_events": 0,
        "monitoring_tool_retired": True,
        "overflow": row_overflow,
        "events": rows,
    }
    (base / "storage-coordinator-native-io.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    assert secondary.get(
        "nonblocking_coordinator_acquired"
    ), "fresh native proof monopolizes global coordinator"
    assert not capture[
        "coordinator_owned_by_read_actor"
    ], "original native proof executes while holding coordinator"
    assert (
        observed_close_before_release
    ), "independent original retirement blocked behind unrelated native read"
    print("retired and reopened")


with user_fixture_default_owner():
    main()
"""


@pytest.mark.skipif(
    os.name != "nt", reason="actual Windows native opener causal control"
)
@pytest.mark.parametrize(
    "route", ["scope_before_count", "scope_after_count", "operation_check"]
)
def test_original_native_storage_proof_allows_independent_retirement(tmp_path, route):
    _run(tmp_path, route, "ordinary", script=_SCRIPT)
    receipt = json.loads((tmp_path / "storage-coordinator-native-io.json").read_text())
    print(
        json.dumps(
            {key: receipt[key] for key in ("route", "secondary", "final_census")},
            sort_keys=True,
        )
    )
