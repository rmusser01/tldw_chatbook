"""Initializer publication stays with its selected source and loop owner."""

import asyncio
import threading

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source_before", "owner_before", "source_held", "owner_held"])
async def test_trust_initializer_refuses_changed_source_without_publication(change):
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    owner = ServiceWiringMixin()
    owner._local_skill_trust_service = None
    owner._local_skill_trust_service_build_lock = asyncio.Lock()
    entered, release = threading.Event(), threading.Event()
    source_current = change != "source_before"
    owner_current = change != "owner_before"
    candidate = object()
    calls = []
    reads = set()
    loop_thread = threading.current_thread()

    def build():
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(3), "test-owned builder release timed out"
        return candidate

    def current_owner(expected=None):
        assert threading.current_thread() is loop_thread
        return owner_current and owner._local_skill_trust_service is expected

    owner._build_local_skill_trust_service = build
    task = asyncio.create_task(owner.ensure_local_skill_trust_service(
        _source_current=lambda: source_current,
        _owner_current=current_owner,
        _read_observers=(reads,),
    ))
    try:
        if change.endswith("held"):
            deadline = asyncio.get_running_loop().time() + 3
            while not entered.is_set() and not task.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.001)
            if task.done():
                await task
            assert len(reads) == 1
            read = next(iter(reads))
            assert read.creator is owner and read.task is task and not read.retired.done()
            if change == "source_held":
                source_current = False
            else:
                owner_current = False
        release.set()
        with pytest.raises(RuntimeError, match="console_skill_trust_(source|owner)_changed"):
            await task
        assert owner._local_skill_trust_service is None
        assert not owner._local_skill_trust_service_build_lock.locked()
        assert not reads
        assert len(calls) == int(change.endswith("held"))
        assert all(thread is not loop_thread for thread in calls)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
