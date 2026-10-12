"""A private-profile child that overruns says where it was (TASK-34789).

``private_profile_test`` gave the child its parent's own timeout, and the
parent's clock starts first: six UI Fast Lane runs of one hung child ended as
a bare parent timeout, the child's log unread. The parent's own limit now
moves later, so the child stops first and writes its stacks to a file the
parent quotes; a child that still cannot finish is asked once more and ended
before the parent's moved limit.
"""

import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests import private_profile
from Tests.private_profile import (
    _CHILD_NODE,
    _CHILD_STACKS,
    _child_deadlines,
    private_profile_test,
)

_CHILD_LIMIT = 3
#: Where a child says it has reached the state its parent test asks it in.
_READY = "TLDW_TEST_PRIVATE_PROFILE_CHILD_READY"


def test_the_child_keeps_its_budget_and_the_parent_waits_for_it():
    # The UI Fast Lane limit, with the parent's own limit moved 30 s later:
    # the child keeps all 180 s, the parent waits until 5 s before 210 s.
    assert _child_deadlines(180.0, 30.0) == (180.0, 205.0)
    # A limit that could not move (thread method, Windows): as before.
    assert _child_deadlines(180.0, 0.0) == (180.0, 180.0)
    assert _child_deadlines(0.0, 30.0) == (0.0, None)
    # Nothing armed: the parent's limit is not touched.
    assert private_profile._extend_own_alarm(0.0) == 0.0


def _stuck_worker(release: threading.Event) -> None:
    release.wait()


async def _waits_on_the_worker() -> None:
    await asyncio.Event().wait()


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
@pytest.mark.timeout(_CHILD_LIMIT, func_only=True)  # the wait itself, not start-up
@private_profile_test
async def test_child_whose_worker_never_answers(request):
    threading.Thread(
        target=_stuck_worker,
        args=(threading.Event(),),
        name="zq-stuck-worker",
        daemon=True,
    ).start()
    await _waits_on_the_worker()


async def test_a_dump_groups_identical_stacks_and_keeps_its_head_within_its_bound(
    monkeypatch,
):
    """40 threads in the same wait share one header; the main thread and the
    await chain lead the dump, and the bound cuts what follows them."""
    release, entered = threading.Event(), threading.Semaphore(0)

    def stuck() -> None:
        entered.release()
        release.wait()

    threads = [
        threading.Thread(target=stuck, name=f"zq-same-{n}", daemon=True)
        for n in range(40)
    ]
    for thread in threads:
        thread.start()
    for _ in threads:
        entered.acquire()
    task = asyncio.create_task(_waits_on_the_worker())
    await asyncio.sleep(0)
    monkeypatch.setattr(private_profile, "_awaiting", task.get_coro())
    try:
        for _ in range(100):  # until each worker has gone from release() to wait()
            text = private_profile._stacks_text("in a test")
            if "--- 40 threads: zq-same-" in text:
                break
            await asyncio.sleep(0.01)
        assert text.startswith("--- stacks in a test\n--- thread MainThread\n")
        assert text.count("--- 40 threads: zq-same-") == 1
        assert "--- thread zq-same-" not in text
        head = text[: text.index("--- 40 threads: ")]
        assert (
            "await asyncio.Event().wait()"
            in head.split("--- the test, outermost await first")[1]
        )

        monkeypatch.setattr(private_profile, "_DUMP_LIMIT", len(head) + 100)
        bounded = private_profile._stacks_text("in a test")
    finally:
        release.set()
        task.cancel()
    assert bounded.endswith(" more characters of this dump cut\n")
    assert "await asyncio.Event().wait()" in bounded
    assert len(bounded) < len(head) + 200


def _as_parent(request, tmp_path, child):
    """A parent-side request for ``child``, as pytest would hand its wrapper."""
    module = request.node.nodeid.rsplit("::", 1)[0]
    return SimpleNamespace(
        node=SimpleNamespace(
            nodeid=f"{module}::{child.__name__}",
            name=child.__name__,
            get_closest_marker=lambda _name: None,
        ),
        function=child,
        config=request.config,
        getfixturevalue=lambda _name: tmp_path,
    )


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX signals")
async def test_a_hung_child_fails_its_parent_with_its_own_timeout_and_stacks(
    request, tmp_path
):
    """A real child, hung in an await behind a stuck thread: the parent's
    failure quotes the child's own timeout, names the thread and the await."""
    child = test_child_whose_worker_never_answers

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert f"Timeout (>{_CHILD_LIMIT}.0s) from pytest-timeout" in text
    assert "(killed" not in text
    stacks = text.split("--- child stacks at its deadline ---")[1]
    stacks = stacks.split("--- child log tail ---")[0]
    assert "--- thread zq-stuck-worker" in stacks and "release.wait()" in stacks
    awaits = stacks.split("--- the test, outermost await first")[1]
    assert "await _waits_on_the_worker()" in awaits
    assert "await asyncio.Event().wait()" in awaits


_NEVER_EXITS = """
import os, signal, time
def asked(*_):
    with open(os.environ[{stacks!r}], "a") as output:
        output.write("zq-asked-before-the-end\\n")
signal.signal(signal.SIGUSR1, asked)
print("zq-child-log-line", flush=True)
for _ in range(1200):  # at most 60 s: a parent ended mid-test leaves no orphan
    time.sleep(0.05)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_that_cannot_finish_is_asked_then_ended_before_the_parents_limit(
    request, tmp_path, monkeypatch
):
    """The parent's own wait: a stand-in child that never exits is asked for
    its stacks, ended, and reported with them, well inside the parent's limit."""
    spawn = asyncio.create_subprocess_exec

    async def never_exits(*_command, **options):
        script = _NEVER_EXITS.format(stacks=_CHILD_STACKS)
        return await spawn(sys.executable, "-c", script, **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never_exits)
    monkeypatch.setattr(
        private_profile, "_child_deadlines", lambda _budget, _lead: (1.0, 5.0)
    )
    child = test_child_whose_worker_never_answers
    started = asyncio.get_running_loop().time()

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    assert asyncio.get_running_loop().time() - started < 30
    text = str(failure.value)
    assert "(killed: still running after 5s; asked for its stacks first)" in text
    assert "(the child wrote no JUnit report)" in text
    assert "--- child stacks at its deadline ---\nzq-asked-before-the-end" in text
    assert "zq-child-log-line" in text


_EXITS_ONCE_ASKED = """
import os, signal, time
def asked(*_):
    with open(os.environ[{stacks!r}], "a") as output:
        output.write("zq-asked\\n")
    os._exit(1)
signal.signal(signal.SIGUSR1, asked)
for _ in range(1200):  # at most 60 s
    time.sleep(0.05)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_that_exits_once_asked_is_not_reported_as_killed(
    request, tmp_path, monkeypatch
):
    spawn = asyncio.create_subprocess_exec

    async def exits_once_asked(*_command, **options):
        script = _EXITS_ONCE_ASKED.format(stacks=_CHILD_STACKS)
        return await spawn(sys.executable, "-c", script, **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", exits_once_asked)
    monkeypatch.setattr(
        private_profile, "_child_deadlines", lambda _budget, _lead: (1.0, 3.0)
    )
    child = test_child_whose_worker_never_answers

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "(killed" not in text
    assert "asked for its stacks, it then exited by itself)" in text
    assert "--- child stacks at its deadline ---\nzq-asked" in text


def _ask_once_the_child_is_ready(monkeypatch, tmp_path) -> None:
    """The parent asks half a second after its child says it is in place.

    The child's start-up takes seconds and varies with load; a fixed deadline
    would ask some runs before the child got where the test needs it.
    """
    ready = tmp_path / "ready"
    monkeypatch.setenv(_READY, str(ready))
    real = private_profile._exited_within
    waits = []

    async def exited_within(process, seconds):
        if not waits:  # the parent's deadline wait
            for _ in range(1200):  # at most 60 s of start-up
                if ready.exists() or process.returncode is not None:
                    break
                await asyncio.sleep(0.05)
        waits.append(seconds)
        return await real(process, seconds)

    monkeypatch.setattr(private_profile, "_exited_within", exited_within)
    monkeypatch.setattr(
        private_profile, "_child_deadlines", lambda _budget, _lead: (600.0, 0.5)
    )


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
@pytest.mark.timeout(_CHILD_LIMIT, func_only=True)
@private_profile_test
async def test_child_that_replaces_its_environment(request, monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    await _waits_on_the_worker()


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX signals")
async def test_a_child_that_replaces_os_environ_still_writes_its_stacks(
    request, tmp_path, monkeypatch
):
    """Review of #3091, item 4a: the dump's path is read at import, not when
    the signal arrives, so a test that replaces os.environ keeps it."""
    monkeypatch.setattr(
        private_profile, "_child_deadlines", lambda _budget, _lead: (1.0, 60.0)
    )
    child = test_child_that_replaces_its_environment

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert f"Timeout (>{_CHILD_LIMIT}.0s) from pytest-timeout" in text
    assert "--- stacks when its own timeout fired" in text


@pytest.fixture
def _hangs_in_setup():
    Path(os.environ[_READY]).touch()
    for _ in range(1200):  # at most 60 s; the parent ends it first
        time.sleep(0.05)


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
@pytest.mark.timeout(120, method="thread")
@private_profile_test
async def test_child_hung_in_a_fixture(request, _hangs_in_setup):
    pass


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_asked_in_fixture_setup_under_the_thread_method_answers(
    request, tmp_path, monkeypatch
):
    """Review of #3091, item 4b: the child answers the parent's ask from
    import on, whatever its timeout method, not only once its test body has
    started under the signal method; before, the ask itself killed it."""
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    child = test_child_hung_in_a_fixture

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "(killed: still running after" in text
    assert "--- stacks when the parent asked" in text
    assert "in _hangs_in_setup" in text.split("--- stacks when the parent asked")[1]


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
@private_profile_test
async def test_child_whose_stacks_file_cannot_be_written(request):
    os.mkdir(os.environ[_CHILD_STACKS])
    Path(os.environ[_READY]).touch()
    await _waits_on_the_worker()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_dump_that_fails_when_asked_does_not_end_the_child(
    request, tmp_path, monkeypatch
):
    """Review of #3091, item 4c: a dump that raises under SIGUSR1 (here its
    file is a directory) is swallowed, as under the child's own timeout, and
    does not end the child's test with an unrelated error."""
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    child = test_child_whose_stacks_file_cannot_be_written

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "(killed: still running after" in text
    assert "IsADirectoryError" not in text


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="POSIX itimers")
async def test_the_child_gets_the_armed_budget_and_the_parent_moves_its_own_limit(
    request, tmp_path, monkeypatch
):
    """Review of #3091, items 5a and 5c, with a real ITIMER_REAL: the child's
    --timeout is what is left of the parent's armed timer, and the parent's
    own timer is moved later by the lead before the child starts."""
    seen = {}
    spawn = asyncio.create_subprocess_exec

    async def record(*command, **options):
        seen["command"] = command
        seen["alarm"] = signal.getitimer(signal.ITIMER_REAL)[0]
        return await spawn(sys.executable, "-c", "raise SystemExit(3)", **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)
    child = test_child_whose_worker_never_answers
    handler = signal.getsignal(signal.SIGALRM)
    left = signal.getitimer(signal.ITIMER_REAL)[0]
    signal.signal(signal.SIGALRM, lambda *_: None)  # stands in for pytest-timeout's
    signal.setitimer(signal.ITIMER_REAL, 100.0)
    try:
        with pytest.raises(pytest.fail.Exception):
            await child(request=_as_parent(request, tmp_path, child))
    finally:
        signal.setitimer(signal.ITIMER_REAL, left)
        signal.signal(signal.SIGALRM, signal.SIG_DFL if handler is None else handler)

    flags = [arg for arg in seen["command"] if arg.startswith("--timeout=")]
    assert len(flags) == 1
    assert 95.0 < float(flags[0].removeprefix("--timeout=")) <= 100.0
    assert 125.0 < seen["alarm"] <= 130.0


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
@private_profile_test
async def test_child_without_a_timeout_marker(request):
    Path(os.environ[_READY]).touch()
    await _waits_on_the_worker()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_an_unmarked_child_asked_with_a_real_signal_writes_its_stacks(
    request, tmp_path, monkeypatch
):
    """Review of #3091, item 5b: the parent's real SIGUSR1 reaches the
    child's own handler, whose dump names the main thread and the await."""
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    child = test_child_without_a_timeout_marker

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "(killed: still running after" in text
    assert "--- stacks when the parent asked" in text
    asked = text.split("--- stacks when the parent asked")[1]
    assert "--- thread MainThread" in asked
    awaits = asked.split("--- the test, outermost await first")[1]
    assert "await _waits_on_the_worker()" in awaits
