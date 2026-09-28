"""Per-run SSH session registry: one session per (run key, binding), failure policy."""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Tools.remote_session_registry import RemoteSessionRegistry
from tldw_chatbook.Tools.remote_session_worker import SessionStartError


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
    module.close_all_remote_sessions()  # no registry yet: no-op
    assert module._REGISTRY is None
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
