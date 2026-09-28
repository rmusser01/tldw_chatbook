"""Per-run SSH session registry: one session per (run key, binding), failure policy."""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Tools.remote_session_registry import RemoteSessionRegistry
from tldw_chatbook.Tools.remote_session_worker import SessionStartError
from tldw_chatbook.Tools.remote_workspace_transport import TransportFailure, TransportFailureKind


class FakeWorker:
    def __init__(self, start_error=None, idle_since=None, start_hook=None):
        self.start_error, self.idle_since, self.start_hook = start_error, idle_since, start_hook
        self.alive, self.closed = False, False
        self.clean_end = False  # R10: natural death with exit code 0

    def ended_cleanly(self):
        return self.clean_end

    def start(self):
        if self.start_hook:
            self.start_hook()
        if self.start_error:
            raise self.start_error
        self.alive = True

    def close(self):
        self.closed, self.alive = True, False


def test_one_session_per_key_and_binding():
    reg = RemoteSessionRegistry()
    made = []

    def create():
        w = FakeWorker()
        made.append(w)
        return w

    a = reg.acquire(("run-1", "b1"), create)
    b = reg.acquire(("run-1", "b1"), create)
    assert a is b and len(made) == 1


def test_protocol_failure_disables_key_for_the_run():
    reg = RemoteSessionRegistry()

    def create():
        return FakeWorker(start_error=SessionStartError(False, None, "stamp"))

    assert reg.acquire(("run-1", "b1"), create) is None
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("retried a disabled session")) is None


def test_transport_failure_raises_and_is_retried_next_call():
    reg = RemoteSessionRegistry()
    with pytest.raises(SessionStartError):
        reg.acquire(
            ("run-1", "b1"), lambda: FakeWorker(start_error=SessionStartError(True, None, "255"))
        )
    assert reg.acquire(("run-1", "b1"), FakeWorker) is not None


def test_dead_session_is_restarted_once_then_falls_back():
    reg = RemoteSessionRegistry()
    first = reg.acquire(("run-1", "b1"), FakeWorker)
    first.alive = False  # died mid-run
    second = reg.acquire(("run-1", "b1"), FakeWorker)  # one restart allowed
    assert second is not first and second.alive
    second.alive = False  # died again
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("restarted twice")) is None


def test_slow_start_does_not_block_other_bindings():
    reg = RemoteSessionRegistry()
    gate = threading.Event()

    def slow():
        return FakeWorker(start_hook=lambda: gate.wait(5))

    t = threading.Thread(target=reg.acquire, args=(("run-1", "b1"), slow))
    t.start()
    time.sleep(0.1)
    started = time.monotonic()
    try:
        assert reg.acquire(("run-1", "b2"), FakeWorker) is not None
        assert time.monotonic() - started < 1.0
    finally:
        gate.set()
        t.join(5)


def test_close_key_and_idle_reap():
    reg = RemoteSessionRegistry()
    w = reg.acquire(("run-1", "b1"), FakeWorker)
    reg.close_key("run-1")
    assert w.closed
    assert reg._key_locks == {}
    w2 = reg.acquire(("run-2", "b1"), lambda: FakeWorker(idle_since=0.0))
    reg.reap_idle(now=100.0, idle_s=60)
    # reap_idle closes off the calling thread (TASK-33401): bounded poll.
    deadline = time.monotonic() + 5
    while not w2.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert w2.closed


def test_close_key_leaves_other_runs_alone():
    reg = RemoteSessionRegistry()
    keep = reg.acquire(("run-2", "b1"), FakeWorker)
    reg.acquire(("run-1", "b1"), FakeWorker)
    reg.close_key("run-1")
    assert not keep.closed
    assert reg.acquire(("run-2", "b1"), lambda: pytest.fail("restarted")) is keep


def test_module_helpers_close_the_singleton(monkeypatch):
    from tldw_chatbook.Tools import remote_session_registry as module

    monkeypatch.setattr(module, "_REGISTRY", None)
    module.close_all_remote_sessions()  # no registry yet: still shuts one down
    assert module.get_session_registry().acquire(("run-1", "b1"), lambda: pytest.fail("opened")) is None
    monkeypatch.setattr(module, "_REGISTRY", None)
    reg = module.get_session_registry()
    assert module.get_session_registry() is reg
    w = reg.acquire(("run-1", "b1"), FakeWorker)
    module.close_all_remote_sessions()
    assert w.closed
    assert reg._key_locks == {}


def test_clean_end_is_recreated_without_spending_the_restart():
    reg = RemoteSessionRegistry()
    for _ in range(3):  # host idle-exit (exit 0), repeatedly in one run
        w = reg.acquire(("run-1", "b1"), FakeWorker)
        w.alive, w.clean_end = False, True
    died = reg.acquire(("run-1", "b1"), FakeWorker)
    died.alive = False  # a genuine death (e.g. 255) still gets its one restart
    restarted = reg.acquire(("run-1", "b1"), FakeWorker)
    assert restarted is not None
    restarted.alive = False  # a SECOND genuine death spends the key for the run
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("restarted twice")) is None
    assert ("run-1", "b1") in reg._disabled


def test_closed_key_is_tombstoned_and_never_reopens():
    reg = RemoteSessionRegistry()
    w = reg.acquire(("run-1", "b1"), FakeWorker)
    reg.close_key("run-1")
    assert w.closed
    # A straggler after run end goes one-shot: create is never called.
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("reopened")) is None
    assert reg.acquire(("run-1", "b2"), lambda: pytest.fail("reopened")) is None
    assert reg.acquire(("run-2", "b1"), FakeWorker) is not None


def test_module_close_tombstones_even_before_any_session(monkeypatch):
    from tldw_chatbook.Tools import remote_session_registry as module

    monkeypatch.setattr(module, "_REGISTRY", None)
    module.close_remote_sessions("run-1")
    reg = module.get_session_registry()
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("reopened")) is None


def test_tombstones_are_bounded(monkeypatch):
    from tldw_chatbook.Tools import remote_session_registry as module

    monkeypatch.setattr(module, "_CLOSED_KEYS_MAX", 3)
    reg = RemoteSessionRegistry()
    for index in range(5):
        reg.close_key(f"run-{index}")
    assert list(reg._closed_keys) == ["run-2", "run-3", "run-4"]
    # The evicted oldest key may open again; the kept ones may not.
    assert reg.acquire(("run-0", "b1"), FakeWorker) is not None
    assert reg.acquire(("run-4", "b1"), lambda: pytest.fail("reopened")) is None


def test_start_in_flight_when_run_closes_is_closed_not_kept():
    reg = RemoteSessionRegistry()
    made = []

    def create():
        worker = FakeWorker(start_hook=lambda: reg.close_key("run-1"))
        made.append(worker)
        return worker

    assert reg.acquire(("run-1", "b1"), create) is None
    assert made[0].closed
    assert reg._sessions == {}


def test_app_exit_closes_sessions_before_masters_in_one_best_effort_try():
    """Pin the on_unmount shutdown order (AST, not a live app: driving
    ``TldwCli.on_unmount`` needs a fully mounted app and ~20 unrelated
    teardown steps). Sessions close first so their remote parents see stdin
    EOF over a still-open master; both sit in the SAME try so a failure in
    either never blocks the quit."""
    import ast
    from pathlib import Path

    import tldw_chatbook

    tree = ast.parse((Path(tldw_chatbook.__file__).parent / "app.py").read_text("utf-8"))
    unmount = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_unmount"
    )

    def calls_in(stmts):
        found = []
        for stmt in stmts:
            for node in ast.walk(stmt):
                if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "to_thread"):
                    continue
                target = ast.unparse(node.args[0])
                if target in ("close_all_remote_sessions", "get_master_manager().close_all"):
                    found.append((node.lineno, target))
        return sorted(found)

    tries = [
        node
        for node in ast.walk(unmount)
        if isinstance(node, ast.Try) and calls_in(node.body)
    ]
    innermost = min(tries, key=lambda node: node.end_lineno - node.lineno)
    order = [target for _, target in calls_in(innermost.body)]
    assert order == ["close_all_remote_sessions", "get_master_manager().close_all"]
    assert any(
        handler.type is not None and ast.unparse(handler.type) == "Exception"
        for handler in innermost.handlers
    )


def test_close_all_makes_later_acquires_one_shot():
    reg = RemoteSessionRegistry()
    reg.close_all()
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("opened after shutdown")) is None


def test_waiter_on_the_key_lock_goes_one_shot_after_close_all():
    reg = RemoteSessionRegistry()
    in_start, release = threading.Event(), threading.Event()
    made = []

    def slow_create():
        def hook():
            in_start.set()
            release.wait(5)
        worker = FakeWorker(start_hook=hook)
        made.append(worker)
        return worker

    results = {}
    first = threading.Thread(target=lambda: results.__setitem__("first", reg.acquire(("run-1", "b1"), slow_create)))
    first.start()
    assert in_start.wait(5)
    # A second caller now blocks on the per-key lock behind the slow start.
    second = threading.Thread(target=lambda: results.__setitem__("second", reg.acquire(("run-1", "b1"), FakeWorker)))
    second.start()
    time.sleep(0.1)
    reg.close_all()
    release.set()
    first.join(5); second.join(5)
    assert results == {"first": None, "second": None}
    assert made[0].closed, "a start finishing after close_all must be closed, not kept"
    assert reg._sessions == {}


# A pair: part 1 shuts the singleton down like an app test's on_unmount; part 2
# (next in file order) must still get a working one via the conftest reset. Split
# across xdist workers, part 2 still passes (it can never be made flaky by it).
def test_shutdown_singleton_part_1_app_exit():
    from tldw_chatbook.Tools import remote_session_registry as module

    module.close_all_remote_sessions()
    assert module.get_session_registry()._shutdown


def test_shutdown_singleton_part_2_next_test_gets_a_working_registry():
    """The conftest autouse reset drops the shut-down singleton between tests."""
    from tldw_chatbook.Tools import remote_session_registry as module

    reg = module.get_session_registry()
    assert not reg._shutdown
    assert reg.acquire(("run-x", "b1"), FakeWorker) is not None
    reg.close_key("run-x")


def test_waiters_share_one_transport_start_failure():
    """A dead host costs one start, not one per queued caller (TASK-33400)."""
    reg = RemoteSessionRegistry()
    creates = []
    release = threading.Event()
    entered = threading.Barrier(5)

    def create():
        creates.append(1)
        return FakeWorker(
            start_error=SessionStartError(True, None, "255"),
            start_hook=lambda: release.wait(5),
        )

    errors = []

    def caller():
        entered.wait(5)
        try:
            reg.acquire(("run-1", "b1"), create)
        except SessionStartError as error:
            errors.append(error)

    threads = [threading.Thread(target=caller) for _ in range(4)]
    for thread in threads:
        thread.start()
    entered.wait(5)
    time.sleep(0.3)  # all four are inside acquire: one starting, three queued
    release.set()
    for thread in threads:
        thread.join(10)
    assert len(creates) == 1
    assert len(errors) == 4 and all(error.transport for error in errors)


def _mux_error():
    failure = TransportFailure(TransportFailureKind.MUX_ERROR, 255, "mux_client_hello_exchange")
    return SessionStartError(False, failure, "session start failed: mux")


def test_mux_start_failure_goes_one_shot_for_that_call_only():
    """TASK-33402: a stale control socket costs one one-shot call, not the run's warm path."""
    reg = RemoteSessionRegistry()
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), FakeWorker) is not None


def test_repeated_mux_start_failure_disables_the_key_for_the_run():
    reg = RemoteSessionRegistry()
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("mux-disabled key restarted")) is None


class SlowCloseWorker(FakeWorker):
    def __init__(self, delay=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delay, self.closed_on = delay, None

    def close(self):
        self.closed_on = threading.current_thread()
        time.sleep(self.delay)
        super().close()


def test_reap_idle_never_closes_on_the_calling_thread():
    """TASK-33401: a wedged session's close never lands on an unrelated call."""
    reg = RemoteSessionRegistry()
    worker = reg.acquire(("run-1", "b1"), lambda: SlowCloseWorker(delay=2.0, idle_since=0.0))
    started = time.monotonic()
    reg.reap_idle(now=100.0, idle_s=1.0)
    assert time.monotonic() - started < 0.5
    deadline = time.monotonic() + 5
    while worker.closed_on is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.closed_on is not threading.current_thread()


def test_close_all_closes_in_parallel_and_is_bounded(monkeypatch):
    """TASK-33405: app exit costs about the slowest close, never their sum."""
    from tldw_chatbook.Tools import remote_session_registry as registry_module

    reg = RemoteSessionRegistry()
    workers = [
        reg.acquire((f"run-{i}", "b1"), lambda: SlowCloseWorker(delay=1.0)) for i in range(3)
    ]
    started = time.monotonic()
    reg.close_all()
    assert time.monotonic() - started < 2.5  # serial would be ~3 s
    assert all(w.closed for w in workers)

    monkeypatch.setattr(registry_module, "_CLOSE_JOIN_S", 0.5)
    reg = RemoteSessionRegistry()
    reg.acquire(("run-x", "b1"), lambda: SlowCloseWorker(delay=5.0))
    started = time.monotonic()
    reg.close_all()
    assert time.monotonic() - started < 2.0  # a wedged close never holds app exit
