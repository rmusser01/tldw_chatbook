"""Prompt recall keeps the exact original job's physical retirement outcome."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


async def _fresh_history():
    from tldw_chatbook.Chat import prompt_history

    await prompt_history._resolve_default_path_off_loop()
    return prompt_history.PromptHistory()


@private_profile_test
@pytest.mark.asyncio
async def test_history_successful_load_keeps_exact_receipt_for_later_task(request):
    history = await _fresh_history()
    await history.load()
    job = history._load_job
    assert history._loaded and history.persistence_error is None
    assert job._entered and job._outcome.error is None and job.native_retired
    assert job._raw_state.retired
    assert not copy.copy(job).native_retired

    async def later():
        await history.load()
        assert history._load_job is job and job.native_retired

    await asyncio.create_task(later())


@private_profile_test
@pytest.mark.asyncio
async def test_history_failed_read_retires_but_new_prejob_failure_cannot_borrow_it(
    request, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    history = await _fresh_history()
    original = raw.file

    @contextmanager
    def unreadable(operation, path, mode):
        if raw._states[operation].source is history and mode == "r":
            raise OSError("ordinary history read failure")
        with original(operation, path, mode) as stream:
            yield stream

    with monkeypatch.context() as changed:
        changed.setattr(raw, "file", unreadable)
        await history.load()
    previous = history._load_job
    assert not history._loaded and history.persistence_error == "OSError"
    assert previous._entered and isinstance(previous._outcome.error, OSError)
    assert previous.native_retired

    async def no_job():
        raise OSError("failure before original job construction")

    with monkeypatch.context() as changed:
        changed.setattr(history, "_file_job", no_job)
        await history.load()
    assert history._load_job is None and not history._loaded
    assert previous.native_retired
    await history.load()
    assert history._loaded and history._load_job is not previous
    assert history._load_job.native_retired and history.persistence_error is None


@private_profile_test
@pytest.mark.asyncio
async def test_history_unretired_scope_cannot_be_certified_by_later_registry_cleanup(
    request, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    history = await _fresh_history()
    original, retained = raw._retire, []

    def hold_first(state):
        if state.source is history and not retained:
            retained.extend((operation, state) for operation, current in raw._states.items()
                            if current is state)
            return False
        return original(state)

    monkeypatch.setattr(raw, "_retire", hold_first)
    try:
        await history.load()
        job = history._load_job
        # Preserve existing cache delivery: receipt observation changes no
        # legacy result, error, retry or cancellation behavior.
        assert history._loaded and job._outcome.error is None
        assert job._entered and not job.native_retired
        assert len(retained) == 1
        operation, state = retained[0]
        assert job._raw_operation is operation and job._raw_state is state
        assert not state.retired and state.pins
    finally:
        monkeypatch.setattr(raw, "_retire", original)
        for operation, state in retained:
            assert original(state)
            with storage._changed:
                del raw._states[operation]
                storage._raw_operations.discard(operation)
                storage._changed.notify_all()
    assert not job.native_retired  # Absence cannot invent a successful receipt.


@private_profile_test
@pytest.mark.asyncio
async def test_history_queued_cancel_records_atomic_no_entry(request):
    history = await _fresh_history()
    loop = asyncio.get_running_loop()
    original_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    occupied = executor.submit(release.wait, 5)
    loop.set_default_executor(executor)
    task = asyncio.create_task(history.load())
    try:
        for _ in range(200):
            job = history._load_job
            if job is not None and job._state == "queued":
                break
            await asyncio.sleep(0.01)
        assert job is not None and job._state == "queued"
        assert not job.native_retired
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not job._entered and job._raw_state is None
        assert job.native_retired and not history._loaded
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        occupied.result(5)
        executor.shutdown(wait=True)
        loop._default_executor = original_executor


@private_profile_test
@pytest.mark.asyncio
async def test_history_running_cancel_waits_for_original_read_and_retirement(
    request, monkeypatch
):
    from tldw_chatbook.Chat.prompt_history import PromptHistory
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    writer = await _fresh_history()
    assert await writer.append("retained prompt")
    history = PromptHistory(writer.path)
    entered, release = threading.Event(), threading.Event()
    original = raw.file

    @contextmanager
    def held(operation, path, mode):
        with original(operation, path, mode) as stream:
            if raw._states[operation].source is history and mode == "r":
                entered.set()
                assert release.wait(5)
            yield stream

    monkeypatch.setattr(raw, "file", held)
    task = asyncio.create_task(history.load())
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        job = history._load_job
        assert job._entered and not job.native_retired
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and not job.native_retired
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert job.native_retired and job._outcome.cancelled
        assert history._loaded and history.entry_texts() == ("retained prompt",)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@private_profile_test
@pytest.mark.asyncio
async def test_history_native_receipt_does_not_claim_original_executor_tail(request):
    from tldw_chatbook.Backup_Recovery.async_file_participants import FileJob

    history = await _fresh_history()
    entered, release = threading.Event(), threading.Event()
    monitoring = sys.monitoring
    tool = next(index for index in range(5, -1, -1)
                if monitoring.get_tool(index) is None)
    code = FileJob._work.__code__

    def held_return(original_code, offset, value):
        entered.set()
        assert release.wait(5)

    monitoring.use_tool_id(tool, "history-original-producer-tail")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, held_return)
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
        await history.load()
        job = history._load_job
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        assert job.native_retired and not job._producer.done()
        release.set()
        await asyncio.shield(job._producer)
        assert job.native_retired and job._producer.done()
    finally:
        release.set()
        job = history._load_job
        if job is not None and job._producer is not None:
            await asyncio.shield(job._producer)
        monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(tool)
