"""A private-profile child that overruns says where it was (TASK-34789).

``private_profile_test`` gave the child its parent's own timeout, and the
parent's clock starts first: six UI Fast Lane runs of one hung child ended as
a bare parent timeout, the child's log unread. The parent's own limit now
moves later, so the child stops first and writes its stacks to a file the
parent quotes; a child that still cannot finish is asked once more and ended
before the parent's moved limit.

One real pytest child (a few seconds); every other case is a bare
interpreter standing in for the child, or a direct call, so the file stays
cheap in the PR Fast Lane.
"""

import asyncio
import os
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests import private_profile
from Tests.private_profile import (
    _CHILD_NODE,
    _CHILD_STACKS,
    _child_deadlines,
    private_profile_test,
)

#: What the parent test arms its own timer at for the real child.
_CHILD_LIMIT = 2.0
#: Where a stand-in child says it is in place for its parent's ask.
_READY = "TLDW_TEST_PRIVATE_PROFILE_CHILD_READY"


def test_the_child_keeps_its_budget_and_the_parent_waits_for_it():
    # The UI Fast Lane limit, with the parent's own limit moved 20 s later:
    # the child keeps all 180 s, the parent waits until 5 s before 200 s.
    assert private_profile._CHILD_LEAD_SECONDS == 20.0
    assert _child_deadlines(180.0, 20.0) == (180.0, 195.0)
    # A limit that could not move (thread method, Windows): as before.
    assert _child_deadlines(180.0, 0.0) == (180.0, 180.0)
    assert _child_deadlines(0.0, 20.0) == (0.0, None)
    # Nothing armed: the parent's limit is not touched.
    assert private_profile._extend_own_alarm(0.0) == 0.0


def _stuck_worker(release: threading.Event) -> None:
    release.wait()


async def _waits_on_the_worker() -> None:
    await asyncio.Event().wait()


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


def test_a_dump_that_fails_when_asked_is_swallowed(tmp_path):
    """Review of #3091, item 4c: under SIGUSR1 a dump that raises (here its
    file is a directory) must not raise into whatever the child was doing."""
    private_profile._answer_the_parent(str(tmp_path))


def _as_parent(request, tmp_path, child, marker=None):
    """A parent-side request for ``child``, as pytest would hand its wrapper."""
    module = request.node.nodeid.rsplit("::", 1)[0]
    return SimpleNamespace(
        node=SimpleNamespace(
            nodeid=f"{module}::{child.__name__}",
            name=child.__name__,
            get_closest_marker=lambda _name: marker,
        ),
        function=child,
        config=request.config,
        getfixturevalue=lambda _name: tmp_path,
    )


class _OwnTimer:
    """Arm this test's own ITIMER_REAL, as pytest-timeout would, then restore."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.fired = []

    def __enter__(self):
        self._handler = signal.getsignal(signal.SIGALRM)
        self._left = signal.getitimer(signal.ITIMER_REAL)[0]
        signal.signal(signal.SIGALRM, lambda *_: self.fired.append(True))
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_exc):
        signal.setitimer(signal.ITIMER_REAL, self._left)
        handler = self._handler
        signal.signal(signal.SIGALRM, signal.SIG_DFL if handler is None else handler)


def _watch_the_spawn(monkeypatch, seen, script=None):
    """Record the child's command and this test's timer at the spawn; with a
    ``script``, run that in a bare interpreter instead of the child."""
    spawn = asyncio.create_subprocess_exec

    async def spawned(*command, **options):
        seen["command"] = command
        if hasattr(signal, "getitimer"):
            seen["alarm"] = signal.getitimer(signal.ITIMER_REAL)[0]
        if script is None:
            return await spawn(*command, **options)
        return await spawn(sys.executable, "-c", script, **options)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawned)


def _child_timeout(seen) -> str:
    flags = [arg for arg in seen["command"] if arg.startswith("--timeout=")]
    assert len(flags) == 1
    return flags[0].removeprefix("--timeout=")


# --- the one real child ------------------------------------------------------


@pytest.mark.skipif(_CHILD_NODE not in os.environ, reason="the child of the next test")
# No limit of its own: the parent's --timeout applies, around the call only.
@pytest.mark.timeout(func_only=True)
@private_profile_test
async def test_child_whose_worker_never_answers(request, monkeypatch):
    threading.Thread(
        target=_stuck_worker,
        args=(threading.Event(),),
        name="zq-stuck-worker",
        daemon=True,
    ).start()
    monkeypatch.setattr(os, "environ", {})  # the dump's path survives this
    await _waits_on_the_worker()


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="POSIX signals")
async def test_a_hung_child_fails_its_parent_with_its_own_timeout_and_stacks(
    request, tmp_path, monkeypatch
):
    """A real child, hung in an await behind a stuck thread, with this test's
    own timer armed at 2 s.

    The child's --timeout is what was left of that timer (review of #3091,
    items 5a and 5c); the parent's timer was moved later by the lead before
    the child started, and never fired; the parent's failure quotes the
    child's own timeout and the stacks it wrote at it: the main thread, the
    await chain and the stuck thread, although the test replaced os.environ
    (item 4a).
    """
    seen = {}
    _watch_the_spawn(monkeypatch, seen)
    child = test_child_whose_worker_never_answers

    with (
        _OwnTimer(_CHILD_LIMIT) as timer,
        pytest.raises(pytest.fail.Exception) as failure,
    ):
        await child(request=_as_parent(request, tmp_path, child))

    budget = _child_timeout(seen)
    assert _CHILD_LIMIT - 0.5 < float(budget) <= _CHILD_LIMIT
    moved = _CHILD_LIMIT + private_profile._CHILD_LEAD_SECONDS
    assert moved - 0.5 < seen["alarm"] <= moved
    assert not timer.fired
    text = str(failure.value)
    assert f"Timeout (>{budget}s) from pytest-timeout" in text
    assert "(killed" not in text and "(still running" not in text
    assert "--- child stacks at its deadline ---\n" in text
    stacks = text.split("--- child stacks at its deadline ---\n")[1]
    stacks = stacks.split("--- child log tail ---")[0]
    assert stacks.startswith("--- stacks when its own timeout fired\n")
    assert "--- thread MainThread" in stacks
    assert "--- thread zq-stuck-worker" in stacks and "release.wait()" in stacks
    awaits = stacks.split("--- the test, outermost await first")[1]
    assert "await _waits_on_the_worker()" in awaits
    assert "await asyncio.Event().wait()" in awaits


# --- bare interpreters standing in for the child -----------------------------


def _ask_once_the_child_is_ready(monkeypatch, tmp_path) -> None:
    """The parent asks a tenth of a second after its child says it is in place,
    and gives it half a second to answer.

    A fixed deadline would ask some runs before the stand-in had installed
    what the test is about.
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
        private_profile, "_child_deadlines", lambda _budget, _lead: (600.0, 0.1)
    )
    monkeypatch.setattr(private_profile, "_PARENT_RESERVE_SECONDS", 1.0)


#: Answers the ask with the real handler, which importing Tests.private_profile
#: installs by itself (no wrapper, no pytest-timeout: whatever the timeout
#: method, and before any fixture or test body), then cannot finish.
_ANSWERS_THEN_HANGS = """
import asyncio, os, pathlib
from Tests import private_profile

async def _hangs():
    pathlib.Path(os.environ[{ready!r}]).touch()
    await asyncio.sleep(60)  # at most 60 s: a parent ended mid-test leaves no orphan

async def main():
    task = asyncio.ensure_future(_hangs())
    private_profile._awaiting = task.get_coro()
    print("zq-child-log-line", flush=True)
    await task

asyncio.run(main())
"""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_that_cannot_finish_is_asked_then_ended_before_the_parents_limit(
    request, tmp_path, monkeypatch
):
    """The parent's own wait: a child that never exits is asked for its stacks
    with a real SIGUSR1, answers through the handler installed at import
    (review of #3091, items 4b and 5b), and is ended and reported."""
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    _watch_the_spawn(monkeypatch, {}, _ANSWERS_THEN_HANGS.format(ready=_READY))
    child = test_child_whose_worker_never_answers
    started = asyncio.get_running_loop().time()

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    assert asyncio.get_running_loop().time() - started < 30
    text = str(failure.value)
    assert "(killed: still running after 0s; asked for its stacks first)" in text
    assert "(the child wrote no JUnit report)" in text
    assert "zq-child-log-line" in text
    assert "--- child stacks at its deadline ---\n" in text
    stacks = text.split("--- child stacks at its deadline ---\n")[1]
    assert stacks.startswith("--- stacks when the parent asked\n--- thread MainThread")
    # A -c script has no source lines to show; its frame names the coroutine.
    awaits = stacks.split("--- the test, outermost await first")[1]
    assert ", in _hangs\n" in awaits


_EXITS_ONCE_ASKED = """
import os, pathlib, signal, time
def asked(*_):
    with open(os.environ[{stacks!r}], "a") as output:
        output.write("zq-asked\\n")
    os._exit(1)
signal.signal(signal.SIGUSR1, asked)
pathlib.Path(os.environ[{ready!r}]).touch()
for _ in range(1200):  # at most 60 s
    time.sleep(0.05)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_that_exits_once_asked_is_not_reported_as_killed(
    request, tmp_path, monkeypatch
):
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    script = _EXITS_ONCE_ASKED.format(stacks=_CHILD_STACKS, ready=_READY)
    _watch_the_spawn(monkeypatch, {}, script)
    child = test_child_whose_worker_never_answers

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "(killed" not in text
    assert "asked for its stacks, it then exited by itself)" in text
    assert "--- child stacks at its deadline ---\nzq-asked" in text


_HAS_NO_HANDLER = """
import os, pathlib, time
pathlib.Path(os.environ[{ready!r}]).touch()
for _ in range(1200):  # at most 60 s
    time.sleep(0.05)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals")
async def test_a_child_ended_by_the_ask_itself_is_reported_as_such(
    request, tmp_path, monkeypatch
):
    """Review of #3091, round 2, item R3: with no handler installed yet, the
    ask's default action ends the child; that is not exiting by itself."""
    _ask_once_the_child_is_ready(monkeypatch, tmp_path)
    _watch_the_spawn(monkeypatch, {}, _HAS_NO_HANDLER.format(ready=_READY))
    child = test_child_whose_worker_never_answers

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    text = str(failure.value)
    assert "ended by the ask for its stacks before it could answer)" in text
    assert "exited by itself" not in text and "(killed" not in text
