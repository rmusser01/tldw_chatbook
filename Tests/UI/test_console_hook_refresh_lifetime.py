"""Original native hook visits keep one physical presentation-read lifetime."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from tldw_chatbook import config
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.UI.Console_Modules.hooks import (
    ConsoleHooksController,
    HookReviewResult,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchResult,
    ConsolePromptDispatchStatus,
)

pytestmark = pytest.mark.bootstrap_profile
hook_file = _hook_file


async def _until(predicate, seconds=10):
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class _OriginalVisit:
    def __init__(self, owner):
        self.owner = owner
        self.visit_code = HookPermissions.visit_snapshot.__code__
        self.snapshot_code = HookPermissions.snapshot.__code__
        self.raw_code = inspect.unwrap(config._read_raw_cli_config_unlocked).__code__
        self.entered = threading.Event()
        self.release = threading.Event()
        self.visit_entries = 0
        self.visit_frames = set()
        self.snapshot_frames = set()
        self.visit_counts = {}
        self.operations = ()
        self.leases = ()
        self.thread = None

    def observe(self, frame, event, arg):
        if (
            frame.f_code is self.snapshot_code
            and frame.f_locals.get("self") is self.owner
        ):
            if event == "call":
                self.snapshot_frames.add(id(frame))
            elif event == "return":
                self.snapshot_frames.discard(id(frame))
        if event == "call" and frame.f_code is self.visit_code:
            identity = id(frame.f_locals.get("self"))
            self.visit_counts[identity] = self.visit_counts.get(identity, 0) + 1
        if frame.f_code is self.visit_code and frame.f_locals.get("self") is self.owner:
            if event == "call":
                self.visit_entries += 1
                self.visit_frames.add(id(frame))
            elif event == "return":
                self.visit_frames.discard(id(frame))
        if event != "call":
            return
        if frame.f_code is not self.raw_code or self.entered.is_set():
            return
        parent = frame.f_back
        while parent is not None:
            if (
                parent.f_code is self.snapshot_code
                and parent.f_locals.get("self") is self.owner
            ):
                break
            parent = parent.f_back
        if parent is None:
            return
        self.thread = threading.current_thread()
        with storage._lock:
            self.operations = tuple(
                operation
                for operation, state in raw._states.items()
                if state.source is config and state.thread is self.thread
            )
            self.leases = tuple(
                lease
                for operation in self.operations
                for lease in raw._states[operation].leases
            )
        self.entered.set()
        assert self.release.wait(20), "original native visit was not released"

    @contextlib.contextmanager
    def installed(self):
        previous, previous_threads = sys.getprofile(), threading.getprofile()
        threading.setprofile_all_threads(self.observe)
        try:
            yield
        finally:
            self.release.set()
            threading.setprofile_all_threads(previous_threads)
            sys.setprofile(previous)

    def assert_live(self):
        assert self.operations and self.leases
        assert self.thread is not threading.current_thread()
        with storage._lock:
            assert all(operation in raw._states for operation in self.operations)
            assert all(lease in storage._live_leases for lease in self.leases)

    def retired(self):
        with storage._lock:
            return (
                not self.visit_frames
                and not self.snapshot_frames
                and all(operation not in raw._states for operation in self.operations)
                and all(lease not in storage._live_leases for lease in self.leases)
            )

    def assert_retired(self):
        assert (
            not self.snapshot_frames
        ), "accepted snapshot callback has not physically returned"
        assert (
            not self.visit_frames
        ), "accepted visit callback has not physically returned"
        with storage._lock:
            assert all(operation not in raw._states for operation in self.operations)
            assert all(lease not in storage._live_leases for lease in self.leases)


def _controller(owner):
    selected, states = [owner], []

    async def review(*args):
        return HookReviewResult("cancel")

    hooks = ConsoleHooksController(
        hook_permissions_accessor=lambda: selected[0],
        request_review=review,
        current_session=lambda: "chat",
        current_stash=lambda: None,
        on_state=states.append,
        notify=lambda *args: None,
    )
    return hooks, selected, states


@pytest.mark.asyncio
async def test_concurrent_visits_join_one_original_native_reader(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, states = _controller(owner)
    probe, tasks = _OriginalVisit(owner), []
    with probe.installed():
        try:
            tasks.append(asyncio.create_task(hooks.refresh()))
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            tasks.append(asyncio.create_task(hooks.refresh()))
            await _until(lambda: probe.visit_entries > 1, 0.3)
            assert not tasks[
                -1
            ].done(), "concurrent caller did not join the accepted reader"
            assert (
                probe.visit_entries == 1
            ), "concurrent refresh accepted a second physical reader"
        finally:
            probe.release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert await _until(probe.retired)
    assert not any(isinstance(result, BaseException) for result in results)
    probe.assert_retired()
    assert len(states) == 1, "joined visits publish their same result more than once"


@pytest.mark.asyncio
async def test_cancelled_visit_drains_actual_native_custody(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, states = _controller(owner)
    probe, task = _OriginalVisit(owner), None
    with probe.installed():
        try:
            task = asyncio.create_task(hooks.refresh())
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert (
                not task.done()
            ), "refresh cancellation abandoned its admitted native reader"
            probe.assert_live()
        finally:
            probe.release.set()
            results = await asyncio.gather(task, return_exceptions=True) if task else []
            assert await _until(probe.retired), "released native visit did not retire"
    assert len(results) == 1 and isinstance(results[0], asyncio.CancelledError)
    assert states == []
    probe.assert_retired()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["owner", "reader", "accessor", "publisher"])
async def test_actual_visit_source_replacement_refuses_publication(
    hook_file, replacement
):
    owner, other = HookPermissions(), HookPermissions()
    owner.snapshot()
    hooks, selected, states = _controller(owner)
    probe, task, redirected = _OriginalVisit(owner), None, []
    original_accessor, original_publisher = hooks._permissions, hooks._on_state
    with probe.installed():
        try:
            task = asyncio.create_task(hooks.refresh())
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            if replacement == "owner":
                selected[0] = other
            elif replacement == "reader":
                owner.visit_snapshot = other.visit_snapshot
            elif replacement == "accessor":
                hooks._permissions = lambda: owner
            else:
                hooks._on_state = redirected.append
            probe.release.set()
            await task
            assert (
                states == [] and redirected == []
            ), "replaced source published the previous visit"
        finally:
            probe.release.set()
            if task:
                await asyncio.gather(task, return_exceptions=True)
            hooks._permissions, hooks._on_state = original_accessor, original_publisher
            if replacement == "reader":
                del owner.visit_snapshot
    probe.assert_retired()


@pytest.mark.asyncio
async def test_busy_send_defers_indicator_without_skipping_fresh_snapshot(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, _ = _controller(owner)
    probe, tasks = _OriginalVisit(owner), []

    async def dispatch():
        return ConsolePromptDispatchResult(ConsolePromptDispatchStatus.SENT, "chat")

    with probe.installed():
        try:
            tasks.append(
                asyncio.create_task(
                    hooks.dispatch(
                        "fresh",
                        session_id="chat",
                        stash=None,
                        dispatch=dispatch,
                    )
                )
            )
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            assert hooks._busy
            tasks.append(asyncio.create_task(hooks.refresh()))
            assert await _until(
                tasks[-1].done, 0.3
            ), "indicator competes with a Send-owned fresh reader"
            assert probe.visit_entries == 0
        finally:
            probe.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
    probe.assert_retired()


@pytest.mark.asyncio
async def test_manual_review_keeps_its_final_original_visit(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, states = _controller(owner)
    probe, task = _OriginalVisit(owner), None
    with probe.installed():
        try:
            task = asyncio.create_task(hooks.review_current())
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            probe.release.set()
            await task
        finally:
            probe.release.set()
            if task:
                await asyncio.gather(task, return_exceptions=True)
    assert probe.visit_entries == 1
    assert len(states) == 2
    probe.assert_retired()


@pytest.mark.asyncio
async def test_cancelled_joiner_does_not_abandon_the_surviving_native_visit(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, states = _controller(owner)
    probe, tasks = _OriginalVisit(owner), []
    with probe.installed():
        try:
            tasks.append(asyncio.create_task(hooks.refresh()))
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            tasks.append(asyncio.create_task(hooks.refresh()))
            await asyncio.sleep(0.1)
            assert not tasks[-1].done()
            tasks[0].cancel()
            await asyncio.sleep(0)
            tasks[0].cancel()
            await asyncio.sleep(0)
            assert not any(task.done() for task in tasks)
            probe.assert_live()
            probe.release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert isinstance(results[0], asyncio.CancelledError)
            assert results[1] is None
        finally:
            probe.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert await _until(probe.retired)
    assert probe.visit_entries == 1
    assert len(states) == 1
    probe.assert_retired()


@pytest.mark.asyncio
async def test_different_owner_waiter_recaptures_after_original_reader_retirement(
    hook_file,
):
    owner, intermediate, latest = (
        HookPermissions(),
        HookPermissions(),
        HookPermissions(),
    )
    owner.snapshot()
    hooks, selected, states = _controller(owner)
    probe, tasks = _OriginalVisit(owner), []
    with probe.installed():
        try:
            tasks.append(asyncio.create_task(hooks.refresh()))
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            selected[0] = intermediate
            tasks.append(asyncio.create_task(hooks.refresh()))
            await asyncio.sleep(0.1)
            assert not tasks[-1].done()
            assert probe.visit_counts.get(id(intermediate), 0) == 0
            selected[0] = latest
            probe.release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert not any(isinstance(result, BaseException) for result in results)
        finally:
            probe.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert await _until(probe.retired)
    assert probe.visit_counts.get(id(intermediate), 0) == 0
    assert probe.visit_counts.get(id(latest), 0) == 1
    assert len(states) == 1
    probe.assert_retired()


@pytest.mark.asyncio
async def test_deferred_indicator_replay_is_coalesced_after_send_custody(hook_file):
    owner = HookPermissions()
    owner.snapshot()
    hooks, _, _ = _controller(owner)
    probe, tasks, replay = _OriginalVisit(owner), [], []
    hooks._on_send_settled = lambda: replay.append((hooks._busy, probe.retired()))

    async def dispatch():
        return ConsolePromptDispatchResult(ConsolePromptDispatchStatus.SENT, "chat")

    with probe.installed():
        try:
            tasks.append(
                asyncio.create_task(
                    hooks.dispatch(
                        "fresh",
                        session_id="chat",
                        stash=None,
                        dispatch=dispatch,
                    )
                )
            )
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            await hooks.refresh()
            await hooks.refresh()
            assert replay == [] and probe.visit_entries == 0
            probe.release.set()
            await tasks[0]
        finally:
            probe.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert await _until(probe.retired)
    assert replay == [(False, True)]
    assert not hooks._refresh_deferred
    probe.assert_retired()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["direct-send", "manual-review"])
async def test_every_hook_snapshot_cancellation_drains_original_native_body(
    hook_file, route
):
    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(
        pending,
        [row.entry.key for row in pending.rows if row.entry and row.state == "pending"],
    )
    hooks, _, states = _controller(owner)
    probe, task = _OriginalVisit(owner), None

    async def dispatch():
        return ConsolePromptDispatchResult(ConsolePromptDispatchStatus.SENT, "chat")

    with probe.installed():
        try:
            coroutine = (
                hooks.review_current()
                if route == "manual-review"
                else hooks.dispatch(
                    "fresh",
                    session_id="chat",
                    stash=None,
                    dispatch=dispatch,
                )
            )
            task = asyncio.create_task(coroutine)
            assert await _until(probe.entered.is_set)
            probe.assert_live()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert (
                not task.done()
            ), "hook snapshot cancellation abandoned its actual native worker"
            assert hooks._busy
            probe.assert_live()
        finally:
            probe.release.set()
            results = await asyncio.gather(task, return_exceptions=True) if task else []
            assert await _until(probe.retired)
    assert len(results) == 1 and isinstance(results[0], asyncio.CancelledError)
    assert states == [] and not hooks._busy
    probe.assert_retired()


@pytest.mark.asyncio
async def test_different_owner_does_not_inherit_retired_reader_failure():
    entered, release = threading.Event(), threading.Event()
    error = RuntimeError("original reader failure")

    def old_reader():
        entered.set()
        assert release.wait(20)
        raise error

    old = SimpleNamespace(visit_snapshot=old_reader)
    new = SimpleNamespace(visit_snapshot=lambda: "current owner")
    hooks, selected, states = _controller(old)
    tasks = []
    try:
        tasks.append(asyncio.create_task(hooks.refresh()))
        assert await _until(entered.is_set)
        selected[0] = new
        tasks.append(asyncio.create_task(hooks.refresh()))
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results[0] is error
        assert results[1] is None, "new owner inherited the retired owner's failure"
        assert states == ["current owner"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_matching_joiners_share_original_publication_failure():
    entered, release = threading.Event(), threading.Event()
    error = RuntimeError("original publication failure")
    attempts = []

    def reader():
        entered.set()
        assert release.wait(20)
        return "current view"

    hooks, _, _ = _controller(SimpleNamespace(visit_snapshot=reader))

    def publish(value):
        attempts.append(value)
        raise error

    hooks._on_state = publish
    tasks = []
    try:
        tasks.append(asyncio.create_task(hooks.refresh()))
        assert await _until(entered.is_set)
        tasks.append(asyncio.create_task(hooks.refresh()))
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results == [
            error,
            error,
        ], "joined caller hid the original publication failure"
        assert attempts == ["current view"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["sent", "reader-error", "cancelled"])
async def test_presentation_schedule_failure_preserves_original_send_outcome(outcome):
    error = RuntimeError("original snapshot failure")

    def snapshot():
        if outcome == "reader-error":
            raise error
        if outcome == "cancelled":
            raise asyncio.CancelledError("original snapshot cancellation")
        return SimpleNamespace(ready=True)

    hooks, _, _ = _controller(SimpleNamespace(snapshot=snapshot))
    hooks._refresh_deferred = True

    def schedule():
        raise RuntimeError("presentation scheduling failure")

    hooks._on_send_settled = schedule

    async def dispatch():
        return ConsolePromptDispatchResult(ConsolePromptDispatchStatus.SENT, "chat")

    if outcome == "sent":
        result = await hooks.dispatch(
            "fresh", session_id="chat", stash=None, dispatch=dispatch
        )
        assert result.status is ConsolePromptDispatchStatus.SENT
    elif outcome == "reader-error":
        with pytest.raises(RuntimeError) as caught:
            await hooks.dispatch(
                "fresh", session_id="chat", stash=None, dispatch=dispatch
            )
        assert caught.value is error
    else:
        with pytest.raises(asyncio.CancelledError):
            await hooks.dispatch(
                "fresh", session_id="chat", stash=None, dispatch=dispatch
            )
    assert not hooks._busy
