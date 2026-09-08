"""Actual async source cancellation and native lifetime evidence (ADR-126)."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Chat.prompt_history import PromptHistory
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


@pytest.mark.asyncio
async def test_history_pause_does_not_change_cache_or_create_parent(
    tmp_path, local_root
):
    history = PromptHistory(tmp_path / "new" / "prompt_history.jsonl")
    history.stash_draft("draft")
    pause = storage._begin_local_pause()
    try:
        assert await history.append("refused") is False
        assert history.size == 0
        assert not history._loaded
        assert history.current == "draft"
        assert not history.path.parent.exists()
        assert pause.drain(time.monotonic() + 0.2)
    finally:
        pause.resume()


@pytest.mark.asyncio
async def test_history_running_cancel_keeps_serialization_and_cache(
    tmp_path, local_root, monkeypatch
):
    history = PromptHistory(tmp_path / "prompt_history.jsonl", max_entries=1)
    await history.append("first")
    entered, finish = threading.Event(), threading.Event()
    original = json.dumps

    def blocked(value, *args, **kwargs):
        if value.get("input") == "second":
            entered.set()
            assert finish.wait(5)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", blocked)
    task = asyncio.create_task(history.append("second"))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), "cancelled awaiter retired before actual native writer"
        next_write = asyncio.create_task(history.append("third"))
        await asyncio.sleep(0.02)
        assert not next_write.done()
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
    assert await next_write
    assert [
        json.loads(line)["input"] for line in history.path.read_text().splitlines()
    ] == ["third"]
    assert (await history.get_entry(-1))["input"] == "third"


@pytest.mark.asyncio
async def test_history_queued_cancel_retires_without_cache_loss(tmp_path, local_root):
    history = PromptHistory(tmp_path / "new" / "prompt_history.jsonl")
    history._loaded = True
    loop = asyncio.get_running_loop()
    original = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    occupied = executor.submit(gate.wait, 5)
    loop.set_default_executor(executor)
    try:
        task = asyncio.create_task(history.append("never started"))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert history.size == 0
        pause = storage._begin_local_pause()
        try:
            assert not storage._pending_acquisitions
            assert not history.path.parent.exists()
        finally:
            pause.resume()
        gate.set()
        await asyncio.sleep(0.03)
        assert not history.path.exists()
    finally:
        gate.set()
        occupied.result(5)
        executor.shutdown(wait=True)
        loop._default_executor = original


from contextlib import contextmanager
import copy
import os
import select

from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_bootstrap import local_scope


@pytest.fixture
def installed_history(tmp_path, local_root, monkeypatch):
    from tldw_chatbook.Chat import prompt_history

    selected = tmp_path / "prompt_history.jsonl"
    monkeypatch.setattr(prompt_history, "default_prompt_history_path", lambda: selected)
    return PromptHistory(selected, max_entries=2)


@pytest.mark.asyncio
async def test_installed_history_running_cancel_holds_native_until_bookkeeping(
    installed_history, local_root, monkeypatch, launch
):
    history = installed_history
    await history.append("before")
    entered, finish = threading.Event(), threading.Event()
    original = json.dumps

    def blocked(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("input") == "after":
            entered.set()
            assert finish.wait(8)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", blocked)
    task = asyncio.create_task(history.append("after"))
    for _ in range(300):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    participant = raw._raw_participant(history)
    participant.close_admission()
    pause = storage._begin_local_pause()
    hold = storage._holds[(os.getpid(), str(local_root))]
    observer = launch(hold.authority.control_root, "maintenance", hold.names)
    try:
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert not participant.drain(time.monotonic() + 0.02)
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([observer.stdout], [], [], 0.04)[0]
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert history.size == 2
        assert history._entries[-1]["input"] == "after"
        assert participant.drain(time.monotonic() + 1)
        assert pause.drain(time.monotonic() + 1)
        assert line(observer) == "entered"
        release(observer)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        pause.resume()
        participant.resume()


@pytest.mark.asyncio
async def test_pause_after_queue_refuses_worker_without_io(installed_history):
    history = installed_history
    history._loaded = True
    loop = asyncio.get_running_loop()
    original = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    occupied = executor.submit(gate.wait, 5)
    loop.set_default_executor(executor)
    task = asyncio.create_task(history.append("refused"))
    await asyncio.sleep(0.03)
    pause = storage._begin_local_pause()
    try:
        assert storage._pending_acquisitions
        gate.set()
        assert await task is False
        assert history.size == 0
        assert not history.path.exists()
        assert not await history.persistence_safe_point()
        assert not storage._pending_acquisitions
    finally:
        gate.set()
        pause.resume()
        occupied.result(5)
        executor.shutdown(wait=True)
        loop._default_executor = original


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["submit", "shutdown"])
async def test_failed_enqueue_retires_pending(installed_history, monkeypatch, failure):
    history = installed_history
    history._loaded = True
    loop = asyncio.get_running_loop()
    if failure == "submit":

        def reject(*args):
            raise RuntimeError("test enqueue refused")

        monkeypatch.setattr(loop, "run_in_executor", reject)
        assert await history.append("no") is False
    else:
        old = loop._default_executor
        executor = ThreadPoolExecutor(max_workers=1)
        gate = threading.Event()
        occupied = executor.submit(gate.wait, 5)
        loop.set_default_executor(executor)
        try:
            task = asyncio.create_task(history.append("no"))
            await asyncio.sleep(0.03)
            executor.shutdown(wait=False, cancel_futures=True)
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            gate.set()
            occupied.result(5)
            executor.shutdown(wait=True)
            loop._default_executor = old
    assert not storage._pending_acquisitions
    assert not history.path.exists()
    assert history.size == 0


@pytest.mark.asyncio
async def test_job_copy_task_thread_and_one_shot_provenance(installed_history):
    with _FileJob(installed_history, "prompt_history") as job:
        copied = copy.copy(job)
        with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
            await copied.run(None)

        async def foreign_task():
            with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
                await job.run(None)

        await asyncio.create_task(foreign_task())
        errors = []

        def foreign_thread():
            try:
                job._check_creator()
            except bootstrap.RecoveryRequired:
                errors.append(True)

        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join(3)
        assert errors == [True]
        assert (await job.run(None)).error is None
        with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
            await job.run(None)
    assert not storage._pending_acquisitions


@pytest.mark.asyncio
async def test_history_custom_subclass_and_missing_primitives_remain_ordinary(
    installed_history, monkeypatch, tmp_path
):
    class Custom(PromptHistory):
        pass

    for history in (
        PromptHistory(tmp_path / "custom.jsonl"),
        Custom(installed_history.path),
    ):
        assert await history.append("ordinary")
        with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
            raw._raw_participant(history)
    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    assert await installed_history.append("portable")
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(installed_history)


@pytest.mark.asyncio
async def test_history_fixed_selection_does_not_follow_retarget(
    installed_history, monkeypatch, tmp_path
):
    history = installed_history
    selected = history.path
    with _FileJob(history, "prompt_history") as job:
        history.path = tmp_path / "other.jsonl"
        result = await job.run(((("wrong", 0.0),), False))
        assert isinstance(result.error, bootstrap.RecoveryRequired)
    assert not selected.exists()
    assert not history.path.exists()


@pytest.mark.asyncio
async def test_exact_file_binding_refuses_missing_sidecar(local_scope, monkeypatch):
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Chat import prompt_history

    root, config, data, authority = local_scope
    selected = data / "prompt_history.jsonl"
    selected.write_text('{"input":"before","timestamp":0}\n')
    monkeypatch.setattr(prompt_history, "default_prompt_history_path", lambda: selected)
    authority.register("file-only", (selected, config))
    bind_profile(root, config, ("file-only",), root / "admission")
    history = PromptHistory(selected, max_entries=1)
    assert await history.append("no") is False
    assert "before" in selected.read_text()
    assert not selected.with_suffix(".jsonl.tmp").exists()


@pytest.mark.asyncio
async def test_native_raw_scope_cannot_transfer_to_async_task(installed_history):
    with raw._scope(installed_history, "prompt_history") as operation:

        async def copied_context():
            with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
                raw._check(operation)

        await asyncio.create_task(copied_context())
        assert raw._selected(operation) == installed_history.path


@pytest.mark.asyncio
async def test_startup_native_exclusion_and_pending_cover_event_loop_bookkeeping(
    installed_history, local_root, launch, monkeypatch
):
    """Supported process startup holds natives until whole-process pending drain."""
    from Tests.Backup_Recovery.config_test_support import install_config_source

    install_config_source(monkeypatch)
    storage.admit_startup()
    startup = storage._startups[(os.getpid(), str(local_root))]
    hold = storage._holds[startup._key]
    pause = None
    try:
        with _FileJob(installed_history, "prompt_history") as job:
            result = await job.run(((("committed", 0.0),), False))
            assert result.error is None
            observer = launch(hold.authority.control_root, "maintenance", hold.names)
            pause = storage._begin_local_pause()
            assert not pause.drain(time.monotonic() + 0.03)
            assert not select.select([observer.stdout], [], [], 0.05)[0]
            installed_history._entries = [{"input": "committed", "timestamp": 0.0}]
        assert pause.drain(time.monotonic() + 0.2)
        # Task10 has not enabled startup retirement; completed per-source IO
        # cannot cause a supported external maintainer to cross this hold.
        assert not select.select([observer.stdout], [], [], 0.05)[0]
    finally:
        if pause is not None:
            pause.resume()
        storage._startups.pop((os.getpid(), str(local_root))).close()
    assert line(observer) == "entered"
    release(observer)


@pytest.mark.asyncio
async def test_actual_composer_recall_cancels_warm_load_without_retiring_native(
    installed_history, monkeypatch
):
    from textual.app import App
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    history = installed_history
    history.path.write_text('{"input":"stored prompt","timestamp":1}\n')
    # Existing cached entries make real recall eligible while a refresh is pending.
    history._entries = [{"input": "stale", "timestamp": 0}]
    started, finish = threading.Event(), threading.Event()
    original = json.loads

    def gated(value, *args, **kwargs):
        if isinstance(value, str) and '"stored prompt"' in value:
            started.set()
            assert finish.wait(8)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", gated)
    composer = ConsoleComposerBar()
    composer.set_prompt_history(history)

    class ComposerApp(App):
        def compose(self):
            yield composer

    async with ComposerApp().run_test() as pilot:
        try:
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            warm = next(
                w for w in composer.workers if w.group == "console-prompt-history"
            )
            assert composer.recall_history_previous()
            await asyncio.sleep(0.02)
            assert warm.is_cancelled
            assert history._append_lock.locked()
            finish.set()
            for _ in range(200):
                if composer.draft_text() == "stored prompt":
                    break
                await asyncio.sleep(0.01)
            assert composer.draft_text() == "stored prompt"
            assert await history.persistence_safe_point()
        finally:
            finish.set()


import subprocess
import sys


_HISTORY_CLOSE_CHILD = r"""
import asyncio, gc, os, sys, time
from pathlib import Path
from contextlib import contextmanager
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.Chat import prompt_history
root, destination, failure = sys.argv[1:]
bootstrap.default_bootstrap_root = lambda: Path(root)
prompt_history.default_prompt_history_path = lambda: Path(destination)
history = prompt_history.PromptHistory(destination)
original_file = raw._file
original_close = os.close
target = None
@contextmanager
def observed(operation, selected, mode):
    global target
    with original_file(operation, selected, mode) as stream:
        if mode in {'a', 'w'}:
            target = stream.fileno()
        yield stream
def failed_close(fd):
    if fd == target:
        if failure == 'after':
            original_close(fd)
        raise OSError('injected actual file close uncertainty')
    original_close(fd)
raw._file = observed
os.close = failed_close
assert asyncio.run(history.append('uncertain')) is False
os.close = original_close
raw._file = original_file
gc.collect()
assert history.persistence_error is not None
assert asyncio.run(history.persistence_safe_point()) is False
assert not storage._pending_acquisitions
assert storage._raw_operations
assert any(target in state.descriptors for state in raw._states.values())
if failure == 'before':
    os.fstat(target)
participant = raw._raw_participant(history)
participant.close_admission()
pause = storage._begin_local_pause()
assert not participant.drain(time.monotonic() + .03)
assert not pause.drain(time.monotonic() + .03)
print('held', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize("failure", ["before", "after"])
def test_history_uncertain_native_close_keeps_independent_exclusion(
    tmp_path, local_root, launch, failure
):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        UNBOUND_NAMESPACE,
    )

    authority = admission_authority(local_root)
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            _HISTORY_CLOSE_CHILD,
            str(local_root),
            str(tmp_path / "history.jsonl"),
            failure,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        try:
            assert line(child) == "held"
        except AssertionError:
            child.wait(timeout=5)
            pytest.fail(child.stderr.read())
        observer = launch(authority.control_root, "maintenance", (UNBOUND_NAMESPACE,))
        assert not select.select([observer.stdout], [], [], 0.05)[0]
        child.stdin.write("exit\n")
        child.stdin.flush()
        child.wait(timeout=5)
        assert child.returncode == 0, child.stderr.read()
        assert line(observer) == "entered"
        release(observer)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


@pytest.mark.asyncio
async def test_running_history_write_keeps_a_newly_stashed_live_draft(
    installed_history, monkeypatch
):
    history = installed_history
    await history.load()
    started, finish = threading.Event(), threading.Event()
    original = json.dumps

    def gated(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("input") == "sent":
            started.set()
            assert finish.wait(5)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", gated)
    history.stash_draft("previous")
    task = asyncio.create_task(history.append("sent"))
    try:
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        history.stash_draft("new unsent draft")
        finish.set()
        assert await task
        assert history.current == "new unsent draft"
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


_EXECUTOR_CANCEL_CHILD = r"""
import asyncio, json, sys, threading
from pathlib import Path
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
async def main():
    history = PromptHistory(sys.argv[2])
    history._loaded = True
    loop = asyncio.get_running_loop()
    original_executor = loop.run_in_executor
    original_dumps = json.dumps
    started, finish = threading.Event(), threading.Event()
    futures = []
    def captured(*args):
        future = original_executor(*args)
        futures.append(future)
        return future
    def gated(value, *args, **kwargs):
        if isinstance(value, dict) and value.get('input') == 'native complete':
            started.set()
            assert finish.wait(3)
        return original_dumps(value, *args, **kwargs)
    loop.run_in_executor = captured
    json.dumps = gated
    task = asyncio.create_task(history.append('native complete'))
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(.005)
    assert started.is_set()
    futures[-1].cancel()
    timer = threading.Timer(.1, finish.set)
    timer.start()
    try:
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('cancellation not delivered')
        assert history.size == 1
        assert 'native complete' in history.path.read_text()
        assert not storage._pending_acquisitions
        print('retired', flush=True)
    finally:
        finish.set()
        timer.join()
asyncio.run(main())
"""


def test_cancelled_executor_wrapper_still_waits_for_real_worker(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            _EXECUTOR_CANCEL_CHILD,
            str(tmp_path / "bootstrap"),
            str(tmp_path / "history.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=6,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "retired"


@pytest.mark.asyncio
async def test_bound_history_cannot_demote_to_bypass_closed_source_gate(
    installed_history, monkeypatch, tmp_path
):
    from tldw_chatbook.Chat import prompt_history

    history = installed_history
    assert await history.append("before")
    participant = raw._raw_participant(history)
    participant.close_admission()
    monkeypatch.setattr(
        prompt_history,
        "default_prompt_history_path",
        lambda: tmp_path / "new-profile.jsonl",
    )
    assert await history.append("must refuse") is False
    assert history.size == 1
    assert "must refuse" not in history.path.read_text()


@pytest.mark.asyncio
async def test_read_scope_cannot_append_or_accept_unknown_native_mode(
    installed_history,
):
    installed_history.path.write_bytes(b"original durable bytes")
    with raw._scope(installed_history, "prompt_history") as operation:
        with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
            with raw._file(operation, installed_history.path, "a"):
                pytest.fail("read operation appended")
        with pytest.raises(ValueError, match="raw_file_mode_invalid"):
            with raw._file(operation, installed_history.path, "r+"):
                pytest.fail("unknown native mode opened")
    assert installed_history.path.read_bytes() == b"original durable bytes"
