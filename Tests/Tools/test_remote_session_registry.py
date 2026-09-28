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
    module.close_remote_sessions("run-1")  # no registry yet: no-op
    module.close_all_remote_sessions()
    reg = module.get_session_registry()
    assert module.get_session_registry() is reg
    w = reg.acquire(("run-1", "b1"), FakeWorker)
    module.close_all_remote_sessions()
    assert w.closed
