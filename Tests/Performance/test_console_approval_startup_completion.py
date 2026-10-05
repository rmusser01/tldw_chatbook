"""Pure controls for the original Console sync completion fixture barrier."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import time

import pytest


_HELPER_PATH = (
    Path(__file__).resolve().parents[2] / "Tests/UI/test_console_mcp_approval.py"
)


class _Screen:
    def __init__(self):
        self.workers = []
        self._console_sync_in_progress = False
        self._console_sync_requested = False

    async def _sync_native_console_chat_ui(self, entered, release):
        self._console_sync_in_progress = True
        entered.set()
        try:
            await release.wait()
        finally:
            self._console_sync_in_progress = False


class _Pilot:
    async def pause(self, duration):
        await asyncio.sleep(duration)


def _load_helper():
    source = _HELPER_PATH.read_bytes()
    tree = ast.parse(source)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_await_original_console_sync_completion"
    ]
    assert len(definitions) == 1
    namespace = {"asyncio": asyncio, "time": time, "ChatScreen": _Screen}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), str(_HELPER_PATH), "exec"
        ),
        namespace,
    )
    assert _HELPER_PATH.read_bytes() == source
    return namespace["_await_original_console_sync_completion"]


async def _wait_until_helper_observed_owner(helper_task):
    # The old queue callback can be delivered while the actual owner is held.
    delivered = asyncio.Event()
    asyncio.get_running_loop().call_soon(delivered.set)
    await delivered.wait()
    await asyncio.sleep(0)
    assert not helper_task.done()


async def _case_direct_owner():
    screen, pilot = _Screen(), _Pilot()
    entered, release = asyncio.Event(), asyncio.Event()
    owner = asyncio.create_task(screen._sync_native_console_chat_ui(entered, release))
    barrier = None
    try:
        await entered.wait()
        barrier = asyncio.create_task(
            _load_helper()(screen, pilot, deadline=time.monotonic() + 0.5)
        )
        await _wait_until_helper_observed_owner(barrier)
        release.set()
        await owner
        await barrier
        assert not screen._console_sync_in_progress
    finally:
        release.set()
        await owner
        if barrier is not None and not barrier.done():
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)


async def _case_queued_worker():
    screen, pilot = _Screen(), _Pilot()
    entered, release = asyncio.Event(), asyncio.Event()
    original = screen._sync_native_console_chat_ui(entered, release)
    screen.workers.append(SimpleNamespace(node=screen, _work=original))
    barrier, owner = None, None
    try:
        assert not screen._console_sync_in_progress
        barrier = asyncio.create_task(
            _load_helper()(screen, pilot, deadline=time.monotonic() + 0.5)
        )
        await _wait_until_helper_observed_owner(barrier)
        owner = asyncio.create_task(original)
        await entered.wait()
        release.set()
        await owner
        await barrier
    finally:
        release.set()
        if owner is None:
            original.close()
        else:
            await owner
        if barrier is not None and not barrier.done():
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)


async def _case_foreign_owner():
    screen, foreign, pilot = _Screen(), _Screen(), _Pilot()
    entered, release = asyncio.Event(), asyncio.Event()
    original = foreign._sync_native_console_chat_ui(entered, release)
    screen.workers.append(SimpleNamespace(node=foreign, _work=original))
    owner = asyncio.create_task(original)
    try:
        await entered.wait()
        await _load_helper()(screen, pilot, deadline=time.monotonic() + 0.5)
        assert not owner.done()
        assert foreign._console_sync_in_progress
    finally:
        release.set()
        await owner


async def _case_long_lived_pump():
    screen, pilot = _Screen(), _Pilot()
    entered, release, pump_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sync_finished = asyncio.Event()

    async def pump():
        await screen._sync_native_console_chat_ui(entered, release)
        sync_finished.set()
        await pump_release.wait()

    owner = asyncio.create_task(pump())
    barrier = None
    try:
        await entered.wait()
        barrier = asyncio.create_task(
            _load_helper()(screen, pilot, deadline=time.monotonic() + 0.5)
        )
        await _wait_until_helper_observed_owner(barrier)
        release.set()
        await sync_finished.wait()
        await barrier
        assert not owner.done(), "Barrier must not await the whole message pump Task"
    finally:
        release.set()
        pump_release.set()
        await owner
        if barrier is not None and not barrier.done():
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)


async def _case_deadline_does_not_cancel_owner():
    screen, pilot = _Screen(), _Pilot()
    entered, release = asyncio.Event(), asyncio.Event()
    owner = asyncio.create_task(screen._sync_native_console_chat_ui(entered, release))
    try:
        await entered.wait()
        with pytest.raises(AssertionError, match="within startup budget"):
            await _load_helper()(screen, pilot, deadline=time.monotonic() + 0.01)
        assert not owner.done()
        assert not owner.cancelled()
        assert screen._console_sync_in_progress
    finally:
        release.set()
        await owner


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        _case_direct_owner,
        _case_queued_worker,
        _case_foreign_owner,
        _case_long_lived_pump,
        _case_deadline_does_not_cancel_owner,
    ],
    ids=["direct", "queued", "foreign", "long_lived_pump", "deadline"],
)
def test_fixture_waits_for_exact_original_sync_ownership(case):
    asyncio.run(asyncio.wait_for(case(), timeout=1.0))
