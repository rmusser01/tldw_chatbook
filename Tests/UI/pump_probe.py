"""Find, name and release a parked Textual message pump -- for freeze tests.

A helper module, not a test module: no ``test_`` prefix, so pytest never
collects it (the ``Tests/UI/app_factory.py`` precedent). Import it from a
freeze test; ``Tests/UI/test_library_media_export_no_freeze.py`` and
``Tests/Architecture/test_surface_swap_guard.py`` do.

TASK-34000.4. A test that pins "this press must not freeze the app" has two
problems the moment it goes red, and both hang the suite if left alone:

* Pilot's idle wait asks every pump to answer, so it stops on a parked one;
* ``App.run_test``'s exit closes every screen and waits for the pump tasks,
  so the test cannot even tear down.

And the obvious bound does not hold. ``asyncio.wait_for`` cancels the task it
wraps; ``Task.cancel()`` cancels the future that task awaits; a ``gather``
cancels its children; each child is a pump task awaiting the next ``gather``.
In a wait CYCLE -- which is what a pump waiting on its own removal is -- that
recursion never ends. Measured on the L-01 deadlock: both ``task.cancel()``
and ``wait_for``'s own timeout callback died with ``RecursionError``, the
loop logged it, and the await stayed parked until pytest-timeout's SIGALRM.

So a freeze test here polls on wall clock, reads the pump TASKS to say who is
parked and at which await, and on a red run cuts the cycle at one edge before
tear-down (``free_parked_pumps``). ``@pytest.mark.timeout`` stays as the outer
bound: it is the only one that holds whatever the event loop is doing.
"""

from __future__ import annotations

import asyncio
import contextlib

from textual import events

#: Frames that mean "this pump is running somebody's code right now": a
#: message handler, or a ``call_next`` callback (flushed outside dispatch).
_DISPATCH_FRAMES = ("_dispatch_message", "_flush_next_callbacks")


def key(app, name: str, character: str | None = None) -> None:
    """Deliver a key the way the terminal driver does, through the app pump,
    without Pilot's idle wait (the TASK-33621.28 freeze tests' recipe)."""
    event = events.Key(name, character)
    event.set_sender(app)
    app._driver.send_message(event)


@contextlib.contextmanager
def task_factory(factory: str):
    """Run a scenario under the task factory the real app uses.

    ``eager`` is how the app runs: Textual's ``App.run_async`` installs
    ``asyncio.eager_task_factory``, and ``run_test`` does not. Anything else
    leaves the loop's factory as it is.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    if factory == "eager":
        loop.set_task_factory(asyncio.eager_task_factory)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


async def until(predicate, seconds: float) -> bool:
    """Poll ``predicate`` on wall clock; never touches Pilot's idle wait."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def pump_runs(pump, seconds: float = 2.0) -> bool:
    """True when ``pump`` runs a posted callback within ``seconds``."""
    ran = asyncio.Event()
    pump.call_later(ran.set)
    try:
        await asyncio.wait_for(ran.wait(), seconds)
    except TimeoutError:
        return False
    return True


def await_chain(task: asyncio.Task) -> list[str]:
    """The functions a suspended task is waiting inside, outermost first.

    ``Task.get_stack`` returns one frame for a suspended coroutine, so this
    follows ``cr_await`` instead -- the await chain a parked handler sits at
    the bottom of.
    """
    chain: list[str] = []
    awaitable = task.get_coro()
    while awaitable is not None:
        frame = getattr(awaitable, "cr_frame", None)
        if frame is None:
            frame = getattr(awaitable, "gi_frame", None)
        if frame is not None:
            chain.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
        following = getattr(awaitable, "cr_await", None)
        if following is None:
            following = getattr(awaitable, "gi_yieldfrom", None)
        awaitable = following
    return chain


def _dispatch_index(chain: list[str]) -> int | None:
    for index, name in enumerate(chain):
        if name.split(":", 1)[0] in _DISPATCH_FRAMES:
            return index
    return None


def pumps_inside_a_handler(app) -> dict:
    """Every live pump whose task is running a handler or callback right now.

    An idle pump waits in ``_get_message`` and a closing one in
    ``_message_loop_exit``; neither is dispatching, so neither is returned.
    """
    inside = {}
    for node in [app, *app.screen_stack, *app.screen.walk_children()]:
        task = getattr(node, "_task", None)
        if task is None or task.done():
            continue
        chain = await_chain(task)
        if _dispatch_index(chain) is not None:
            inside[node] = chain
    return inside


async def parked_pumps(app, seconds: float = 1.0) -> dict:
    """Pumps that sit at the same await, inside one handler, for ``seconds``.

    Reads the tasks instead of asking the pumps to answer: the await chain
    says WHERE each one sits, and a pump that is merely busy -- a different
    chain a moment later -- is not reported.
    """
    before = pumps_inside_a_handler(app)
    if not before:
        return {}
    await asyncio.sleep(seconds)
    after = pumps_inside_a_handler(app)
    return {node: chain for node, chain in after.items() if before.get(node) == chain}


def describe_parked(parked: dict) -> list[str]:
    """One line per parked pump: who, and the await chain from its handler."""
    return [
        f"{node!r} parked at {' > '.join(chain[_dispatch_index(chain) :])}"
        for node, chain in parked.items()
    ]


def unpark(task: asyncio.Task) -> None:
    """Break a parked task's wait without ``Task.cancel()``.

    Completes the one future the task awaits, which cuts a wait cycle at a
    single edge (see the module docstring for why ``cancel()`` cannot).
    ``CancelledError`` is what Textual's pump treats as "stop": it leaves its
    loop and unmounts.
    """
    waiter = getattr(task, "_fut_waiter", None)
    if waiter is not None and not waiter.done():
        waiter.set_exception(asyncio.CancelledError())


async def free_parked_pumps(app) -> list[str]:
    """Let a red run tear down: release whatever a parked pump is awaiting.

    Returns:
        What was parked, and where, for the failure message.
    """
    parked = await parked_pumps(app)
    for node in parked:
        unpark(node._task)
    await until(lambda: not pumps_inside_a_handler(app), 3.0)
    return describe_parked(parked)
