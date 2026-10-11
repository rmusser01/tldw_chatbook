"""A private-profile child that overruns says where it was (TASK-34789).

``private_profile_test`` gave the child its parent's own timeout, and the
parent's clock starts first: six UI Fast Lane runs of one hung child ended as
a bare parent timeout, the child's log unread. The child now stops first and
writes its stacks to a file the parent quotes; a child that still cannot
finish is asked once more and ended before the parent's own limit.
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

_CHILD_LIMIT = 3


def test_the_child_stops_before_its_parent_and_the_parent_before_its_own_limit():
    # The UI Fast Lane limit. The slowest passing private-profile test in 177
    # CI logs took 126 s there, child start-up included.
    assert _child_deadlines(180.0) == (150.0, 175.0)
    assert _child_deadlines(300.0) == (270.0, 295.0)
    # A short limit leaves the child five sixths of it.
    assert _child_deadlines(30.0) == (25.0, 27.0)
    assert _child_deadlines(0.0) == (0.0, None)


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
while True:
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
    monkeypatch.setattr(private_profile, "_child_deadlines", lambda _left: (1.0, 5.0))
    child = test_child_whose_worker_never_answers
    started = asyncio.get_running_loop().time()

    with pytest.raises(pytest.fail.Exception) as failure:
        await child(request=_as_parent(request, tmp_path, child))

    assert asyncio.get_running_loop().time() - started < 30
    text = str(failure.value)
    assert "(killed: still running after 5s)" in text
    assert "(the child wrote no JUnit report)" in text
    assert "--- child stacks at its deadline ---\nzq-asked-before-the-end" in text
    assert "zq-child-log-line" in text
