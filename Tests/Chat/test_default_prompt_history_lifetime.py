"""Default-history callback cancellation and real installed-source refusals."""

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_bootstrap import local_scope as local_scope  # noqa: PLC0414
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
from tldw_chatbook.Chat import prompt_history


@pytest.fixture
def default_history_profile(local_scope, monkeypatch):  # noqa: F811 - imported fixture
    root, configuration, data, _authority = local_scope
    configuration.write_text(
        '[general]\nusers_name = "data"\n[paths]\ndata_dir = '
        + json.dumps(data.parent.as_posix())
        + "\n",
        encoding="utf-8",
    )
    try:
        install_config_source(monkeypatch)
        # Establish the same real prerequisite as installed_history, leaving
        # the actual guarded resolver intact. This is lifetime, not boot timing.
        assert (
            prompt_history.default_prompt_history_path()
            == data / "prompt_history.jsonl"
        )
        assert raw._pinned_io_available(), "the actual source needs native guards"
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()
        yield root, configuration, data
    finally:
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()


@contextmanager
def observe_history_callback(history, hook=None):
    """Observe original code on a newly issued executor, without replacing it."""
    resolver = prompt_history.default_prompt_history_path
    resolver_code = resolver.__code__
    io = prompt_history.PromptHistory._history_io
    io_code = io.__code__
    file_body = raw._file
    file_code = file_body.__wrapped__.__code__
    run = _FileJob.run
    run_code = run.__code__
    observed = SimpleNamespace(
        jobs=[], resolver_actors=[], io_actors=[], file_actors=[]
    )

    def observe(frame, event, argument):
        if event == "call":
            if frame.f_code is resolver_code:
                observed.resolver_actors.append(threading.current_thread())
            elif frame.f_code is run_code:
                job = frame.f_locals["self"]
                if job._source is history and job not in observed.jobs:
                    observed.jobs.append(job)
            elif frame.f_code is io_code and frame.f_locals["self"] is history:
                observed.io_actors.append(threading.current_thread())
            elif frame.f_code is file_code:
                state = raw._states.get(frame.f_locals.get("operation"))
                if state is not None and state.source is history:
                    observed.file_actors.append(threading.current_thread())
        if hook is not None:
            hook(frame, event, argument)

    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    previous_main_profile = sys.getprofile()
    previous_thread_profile = threading.getprofile()
    try:
        loop.set_default_executor(executor)
        sys.setprofile(observe)
        threading.setprofile(observe)
        yield observed, executor
        assert prompt_history.default_prompt_history_path is resolver
        assert resolver.__code__ is resolver_code
        assert prompt_history.PromptHistory._history_io is io
        assert io.__code__ is io_code
        assert raw._file is file_body
        assert file_body.__wrapped__.__code__ is file_code
        assert _FileJob.run is run
        assert run.__code__ is run_code
    finally:
        sys.setprofile(previous_main_profile)
        threading.setprofile(previous_thread_profile)
        executor.shutdown(wait=True)
        loop._default_executor = previous_executor


@pytest.mark.asyncio
async def test_default_history_queued_cancel_never_enters_the_real_resolver(
    default_history_profile,
):
    _root, _configuration, data = default_history_profile
    history = prompt_history.PromptHistory()
    assert history.path is None
    release = threading.Event()
    with observe_history_callback(history) as (observed, executor):
        occupied = executor.submit(release.wait, 5)
        task = asyncio.create_task(history.load())
        try:
            for _ in range(300):
                if observed.jobs and observed.jobs[0]._state == "queued":
                    break
                await asyncio.sleep(0.01)
            assert observed.jobs and observed.jobs[0]._state == "queued"
            job = observed.jobs[0]
            assert job._attempt in storage._pending_acquisitions
            assert history._append_lock.locked()
            assert not observed.resolver_actors
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), 1)
            assert job._state == "closed"
            assert job._attempt not in storage._pending_acquisitions
            assert not history._append_lock.locked()
            assert history.path is None
            assert not history._loaded and history.size == 0
            assert raw._source_participants.get(history) is None
        finally:
            release.set()
            results = await asyncio.gather(task, return_exceptions=True)
            assert all(
                result is None or isinstance(result, asyncio.CancelledError)
                for result in results
            )
            assert occupied.result(5)
        assert not observed.resolver_actors
        assert not observed.io_actors
    assert not (data / "prompt_history.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["resolver", "native_read"])
async def test_default_history_recancel_retains_native_read_and_cache_delivery(
    default_history_profile, boundary
):
    _root, _configuration, data = default_history_profile
    seed = prompt_history.PromptHistory()
    assert await seed.append("original native bytes")
    selected = data / "prompt_history.jsonl"
    original_bytes = selected.read_bytes()
    history = prompt_history.PromptHistory()
    entered, release = threading.Event(), threading.Event()
    decoder = json.loads
    decoder_code = decoder.__code__
    body_code = prompt_history.PromptHistory._history_io.__code__
    resolver_code = prompt_history.default_prompt_history_path.__code__
    resolve_code = prompt_history.PromptHistory._resolve_default_path.__code__

    def hold_original_source(frame, event, _argument):
        caller = frame.f_back
        if event != "call" or caller is None:
            return
        at_resolver = (
            boundary == "resolver"
            and frame.f_code is resolver_code
            and caller.f_code is resolve_code
            and caller.f_locals["self"] is history
        )
        at_read = (
            boundary == "native_read"
            and frame.f_code is decoder_code
            and caller.f_code is body_code
            and caller.f_locals["self"] is history
        )
        if at_resolver or at_read:
            entered.set()
            assert release.wait(8), "original source gate was not released"

    participant = pause = next_read = None
    with observe_history_callback(history, hold_original_source) as (
        observed,
        _executor,
    ):
        task = asyncio.create_task(history.load())
        try:
            for _ in range(300):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert (
                entered.is_set()
            ), "the original source did not reach its callback boundary"
            assert len(observed.jobs) == 1
            job = observed.jobs[0]
            assert job._state == "running"
            assert job._attempt in storage._pending_acquisitions
            states = [
                state for state in raw._states.values() if state.source is history
            ]
            leases = ()
            if boundary == "native_read":
                assert len(states) == 1 and states[0].active
                state = states[0]
                participant = raw._source_participants[history]
                assert state.participant is participant
                assert state.selected == selected and not state.writing
                assert state.route == "prompt_history"
                assert state.thread in observed.io_actors
                assert state.thread in observed.file_actors
                assert state.files and state.descriptors and state.leases
                leases = tuple(state.leases)
                assert all(lease in storage._live_leases for lease in leases)
                participant.close_admission()
                pause = storage._begin_local_pause()
            else:
                assert history.path is None and not states
                assert raw._source_participants.get(history) is None
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done(), "repeated cancellation retired the actual source"
            assert history._append_lock.locked()
            assert not history._loaded and history.size == 0
            assert job._attempt in storage._pending_acquisitions
            if boundary == "native_read":
                assert not participant.drain(time.monotonic() + 0.02)
                assert not pause.drain(time.monotonic() + 0.02)
            next_read = asyncio.create_task(history.get_entry(-1))
            await asyncio.sleep(0.01)
            assert not next_read.done(), "the next recall bypassed the held append lock"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert history._loaded and history.size == 1
            assert (await asyncio.wait_for(asyncio.shield(next_read), 1))[
                "input"
            ] == "original native bytes"
            assert job._state == "closed"
            assert job._attempt not in storage._pending_acquisitions
            assert not any(state.source is history for state in raw._states.values())
            assert all(lease not in storage._live_leases for lease in leases)
            assert not history._append_lock.locked()
            if boundary == "native_read":
                assert participant.drain(time.monotonic() + 1)
                assert pause.drain(time.monotonic() + 1)
            assert selected.read_bytes() == original_bytes
            assert history.path == selected
            assert json.loads is decoder and decoder.__code__ is decoder_code
            assert observed.resolver_actors and observed.io_actors
            assert all(
                actor is not threading.current_thread()
                for actor in observed.resolver_actors + observed.io_actors
            )
        finally:
            release.set()
            tasks = [task] + ([next_read] if next_read is not None else [])
            results = await asyncio.gather(*tasks, return_exceptions=True)
            unexpected = [
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            ]
            if pause is not None:
                pause.resume()
            if participant is not None:
                participant.resume()
            assert all(task.done() for task in tasks)
            if unexpected:
                raise unexpected[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("retarget", ["path", "profile"])
async def test_bound_default_history_refuses_actual_retarget_without_following(
    default_history_profile, monkeypatch, retarget
):
    _root, configuration, data = default_history_profile
    history = prompt_history.PromptHistory()
    assert await history.append("before retarget")
    selected = history.path
    original_bytes = selected.read_bytes()
    participant = raw._source_participants[history]
    history.stash_draft("keep draft")
    other_history = data.parent / "other" / "prompt_history.jsonl"
    if retarget == "path":
        history.path = other_history
    else:
        other_configuration = configuration.with_name("other-config.toml")
        other_configuration.write_text(
            '[general]\nusers_name = "other"\n[paths]\ndata_dir = '
            + json.dumps(data.parent.as_posix())
            + "\n",
            encoding="utf-8",
        )
        other_configuration.chmod(0o600)
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(other_configuration))
    with observe_history_callback(history) as (observed, _executor):
        assert await history.append("must refuse") is False
        assert history.persistence_error == "RecoveryRequired"
        assert observed.resolver_actors
        assert (
            not observed.file_actors
        ), "retargeted default source entered the original file operation"
        assert all(
            actor is not threading.current_thread()
            for actor in observed.resolver_actors
        )
        assert observed.jobs and all(job._state == "closed" for job in observed.jobs)
        assert all(
            job._attempt not in storage._pending_acquisitions for job in observed.jobs
        )
    assert selected.read_bytes() == original_bytes
    assert not other_history.exists()
    assert history.path == (other_history if retarget == "path" else selected)
    assert history.size == 1 and history.current == "keep draft"
    assert raw._source_participants[history] is participant
    assert raw._participants[participant].selected == selected
    assert not any(state.source is history for state in raw._states.values())


@pytest.mark.asyncio
async def test_closed_default_history_refuses_fresh_worker_selection(
    default_history_profile,
):
    _root, _configuration, _data = default_history_profile
    history = prompt_history.PromptHistory()
    assert await history.append("before close")
    selected = history.path
    original_bytes = selected.read_bytes()
    participant = raw._source_participants[history]
    history.stash_draft("keep draft")
    participant.close_admission()
    try:
        with observe_history_callback(history) as (observed, _executor):
            assert await history.append("must refuse") is False
            assert history.persistence_error == "RecoveryRequired"
            assert observed.resolver_actors, "fresh default qualification did not run"
            assert not observed.file_actors, "closed source entered the file operation"
            assert all(
                actor is not threading.current_thread()
                for actor in observed.resolver_actors
            )
            assert len(observed.jobs) == 1 and observed.jobs[0]._state == "closed"
            assert observed.jobs[0]._attempt not in storage._pending_acquisitions
        assert selected.read_bytes() == original_bytes
        assert history.path == selected
        assert history.size == 1 and history.current == "keep draft"
        assert not any(state.source is history for state in raw._states.values())
        assert participant.drain(time.monotonic() + 1)
    finally:
        participant.resume()
