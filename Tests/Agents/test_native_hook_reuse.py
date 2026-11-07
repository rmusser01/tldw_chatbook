"""Native finalization never acts on an identity after its wait is lost."""

import gc
import os
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents import run_hooks
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

pytestmark = pytest.mark.skipif(
    not native_workflow_qualified(), reason="native POSIX group custody required"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("loss_in_finalizer", [False, True])
async def test_consumed_leader_never_signals_or_waits_again(
    monkeypatch, loss_in_finalizer
):
    """Consume a real child first; simulated reuse never receives a PID action."""
    process = subprocess.Popen(  # noqa: ASYNC220 -- actual owned child for wait regression
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    waited, _ = os.waitpid(process.pid, 0)  # noqa: ASYNC222 -- intentional consumed-child fixture
    assert waited == process.pid
    pid = process.pid
    signals, waits, terminals, polls = [], [], [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    protocol = run_hooks._HookProtocol()
    original_observe = run_hooks._native_observe
    calls = 0

    def observe(child):
        nonlocal calls
        assert child == pid
        calls += 1
        if loss_in_finalizer and calls == 1:
            return True
        return original_observe(child)

    monkeypatch.setattr(run_hooks, "_native_observe", observe)
    monkeypatch.setattr(run_hooks.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(run_hooks.os, "waitpid", lambda *args: waits.append(args))
    original_poll = subprocess.Popen._internal_poll

    def poll(self, *args, **kwargs):
        if self.pid == pid:
            polls.append(self.pid)
        return original_poll(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "_internal_poll", poll)
    with pytest.raises(RuntimeError, match="cleanup unproven"):
        await run_hooks._native_capture(
            native,
            process,
            protocol,
            b"{}",
            None,
            1,
            lambda: True,
            native.process_terminal,
        )
    assert terminals == [False]
    assert not signals and not waits
    assert native._retained_process is process
    assert process._child_created is False and process.returncode is None
    reference = weakref.ref(process)
    del native._retained_process
    del process
    gc.collect()
    assert reference() is None and not polls


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["present", "permission"])
async def test_finalizer_can_observe_absence_within_original_deadline(
    monkeypatch, first
):
    import time

    process = SimpleNamespace(
        pid=432100, stdin=None, stdout=None, stderr=None, returncode=None
    )
    terminals, effects = [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    monkeypatch.setattr(run_hooks, "_native_observe", lambda pid: True)

    def waitpid(pid, flags):
        effects.append(("wait", pid))
        return pid, 0

    checks = 0

    def killpg(pid, sig):
        nonlocal checks
        effects.append(("signal", sig))
        if sig == 0:
            checks += 1
            if checks == 1:
                if first == "permission":
                    raise PermissionError
                return
            raise ProcessLookupError

    monkeypatch.setattr(run_hooks.os, "waitpid", waitpid)
    monkeypatch.setattr(run_hooks.os, "killpg", killpg)
    assert await run_hooks._native_cleanup(
        native, process, time.monotonic() + 0.2, lambda: True, native.process_terminal
    )
    assert terminals == [True] and process.returncode == 0
    assert effects == [
        ("signal", 9),
        ("wait", process.pid),
        ("signal", 0),
        ("signal", 0),
    ]


@pytest.mark.asyncio
async def test_finalizer_keeps_expired_group_unknown_without_later_observation(
    monkeypatch,
):
    import asyncio
    import time

    process = SimpleNamespace(
        pid=432101, stdin=None, stdout=None, stderr=None, returncode=None
    )
    terminals, signals = [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    monkeypatch.setattr(run_hooks, "_native_observe", lambda pid: True)
    monkeypatch.setattr(run_hooks.os, "waitpid", lambda pid, flags: (pid, 0))
    monkeypatch.setattr(run_hooks.os, "killpg", lambda pid, sig: signals.append(sig))
    assert not await run_hooks._native_cleanup(
        native, process, time.monotonic() + 0.02, lambda: True, native.process_terminal
    )
    count = len(signals)
    await asyncio.sleep(0.03)
    assert terminals == [False] and len(signals) == count
    assert signals[0] == 9 and all(sig == 0 for sig in signals[1:])


@pytest.mark.asyncio
async def test_owned_late_reap_returns_resource_without_releasing_uncertain_target(
    monkeypatch,
):
    import time

    process = subprocess.Popen(  # noqa: ASYNC220 -- actual owned child for wait regression
        [sys.executable, "-c", "import time; time.sleep(.3)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    terminals, signals = [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    monkeypatch.setattr(run_hooks, "_native_observe", lambda pid: False)
    monkeypatch.setattr(run_hooks.os, "killpg", lambda pid, sig: signals.append(sig))
    try:
        assert not await run_hooks._native_cleanup(
            native,
            process,
            time.monotonic() + 0.03,
            lambda: True,
            native.process_terminal,
        )
        assert terminals == [False] and signals == [9]
        native._late_reaper.join(3)
        assert not native._late_reaper.is_alive()
        assert process.returncode == 0 and terminals == [False]
        assert signals == [9]
    finally:
        if process.returncode is None:
            native._late_reaper.join(3)


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before_signal", "after_signal"])
async def test_repeated_cancellation_does_not_interrupt_original_signal_and_reap(
    monkeypatch, when
):
    import asyncio
    import signal

    process = subprocess.Popen(  # noqa: ASYNC220 -- actual owned child for wait regression
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    actual_signal, actual_wait = os.killpg, os.waitpid
    effects, terminals = [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    task = None

    def group(pid, sig):
        effects.append(("signal", sig))
        if sig == signal.SIGKILL and when == "before_signal":
            task.cancel()
            task.cancel()
        result = actual_signal(pid, sig)
        if sig == signal.SIGKILL and when == "after_signal":
            task.cancel()
            task.cancel()
        return result

    def wait(pid, flags):
        result = actual_wait(pid, flags)
        effects.append(("wait", result[0]))
        return result

    monkeypatch.setattr(run_hooks.os, "killpg", group)
    monkeypatch.setattr(run_hooks.os, "waitpid", wait)
    task = asyncio.create_task(
        run_hooks._native_capture(
            native,
            process,
            run_hooks._HookProtocol(),
            b"{}",
            None,
            0.05,
            lambda: True,
            native.process_terminal,
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert terminals == [True] and process.returncode == -signal.SIGKILL
    assert [event for event in effects if event == ("signal", signal.SIGKILL)] == [
        ("signal", signal.SIGKILL)
    ]
    consumed = effects.index(("wait", process.pid))
    assert all(event != ("signal", signal.SIGKILL) for event in effects[consumed + 1 :])


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["present", "permission"])
@pytest.mark.parametrize("expiry", ["before_probe", "during_probe"])
async def test_expired_absence_never_proves_group_return(monkeypatch, first, expiry):
    process = SimpleNamespace(
        pid=432102, stdin=None, stdout=None, stderr=None, returncode=None
    )
    terminals, effects = [], []
    native = SimpleNamespace(process_terminal=terminals.append)
    clock = [0.0]
    monkeypatch.setattr(run_hooks, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(run_hooks, "_native_observe", lambda pid: True)
    monkeypatch.setattr(run_hooks.os, "waitpid", lambda pid, flags: (pid, 0))

    async def advance(_delay):
        clock[0] = 1.01 if expiry == "before_probe" else 0.75

    monkeypatch.setattr(run_hooks, "asyncio", SimpleNamespace(sleep=advance))

    def killpg(pid, sig):
        effects.append(sig)
        if sig == 0:
            if effects.count(0) == 1:
                clock[0] = 0.5
                if first == "permission":
                    raise PermissionError
                return
            clock[0] = 1.01
            raise ProcessLookupError

    monkeypatch.setattr(run_hooks.os, "killpg", killpg)
    assert not await run_hooks._native_cleanup(
        native, process, 1.0, lambda: True, native.process_terminal
    )
    assert terminals == [False]
    assert effects[0] == 9 and all(sig == 0 for sig in effects[1:])
    if expiry == "before_probe":
        assert effects == [9, 0]
    else:
        assert effects == [9, 0, 0]
