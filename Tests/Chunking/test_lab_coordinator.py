"""Coordinator ordering with real immutable transitions and recording boundaries."""

import asyncio
import importlib
import json
import os
import sqlite3
import sys
import threading
import time
from importlib.util import find_spec

import pytest

from tldw_chatbook.Chunking.lab_autosave import AutosaveWriter
from tldw_chatbook.Chunking.lab_models import RunResult
from tldw_chatbook.Chunking.lab_recovery import export_recovery
from tldw_chatbook.Chunking.lab_state import (
    accept_result,
    capture_batch,
    edit_json,
    install_batch,
    new_session,
    pin_baseline,
    replace_sample,
    replace_template,
    undo_edit,
)
from tldw_chatbook.Chunking.template_runtime import execute_prepared
from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore


def api():
    assert find_spec("tldw_chatbook.Chunking.lab_coordinator"), (
        "Lab lifecycle coordinator is missing"
    )
    return importlib.import_module("tldw_chatbook.Chunking.lab_coordinator")


def outcome(request, status="completed"):
    return RunResult(
        request=request,
        status=status,
        report=execute_prepared(request.recipe, request.sample.text)
        if status == "completed"
        else None,
        started_at="2026-09-04T00:00:00+00:00",
        finished_at="2026-09-04T00:00:00+00:00",
        elapsed_ms=0,
        error=None,
    )


def comparison():
    session = replace_sample(new_session("profile"), "one two three", {"kind": "paste"})
    requests = capture_batch(session, tuple(session.candidates))
    session = accept_result(install_batch(session, requests), outcome(requests[0]))
    return pin_baseline(session)


class RecordingRunner:
    def __init__(self, events, statuses=()):
        self.events = events
        self.requests = []
        self.statuses = statuses
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.stopped = False

    async def run(self, request):
        self.requests.append(request)
        self.events.append(("launch", request.candidate_id))
        self.started.set()
        await self.release.wait()
        return outcome(
            request,
            self.statuses[len(self.requests) - 1] if self.statuses else "completed",
        )

    async def cancel(self):
        self.events.append(("stop",))
        self.stopped = True
        self.release.set()

    async def close(self):
        await self.cancel()


class RecordingStore(CheckpointStore):
    def __init__(self, path, events):
        super().__init__(path, "profile")
        self.events = events
        self.fail_save = False
        self.fail_replace = False

    def save(self, session, *, expected):
        if self.fail_save:
            raise OSError("disk full")
        token = super().save(session, expected=expected)
        self.events.append(("commit", session.batch))
        return token

    def replace(self, *args, **kwargs):
        self.events.append(("replace",))
        if self.fail_replace:
            raise OSError("replace failed")
        return super().replace(*args, **kwargs)


def setup(tmp_path, statuses=()):
    events = []
    store = RecordingStore(tmp_path / "lab.sqlite3", events)
    writer = AutosaveWriter(store)
    runner = RecordingRunner(events, statuses)
    coordinator = api().LabCoordinator(comparison(), writer, runner)
    return coordinator, runner, writer, store, events


@pytest.mark.asyncio
async def test_manifest_commits_before_a_then_b_and_failed_a_does_not_borrow_previous(
    tmp_path,
):
    coordinator, runner, _writer, _store, events = setup(
        tmp_path, ("failed", "completed")
    )
    await coordinator.run(tuple(coordinator.session.candidates))
    assert events[0][0] == "commit"
    manifest = events[0][1]
    assert len(manifest["requests"]) == 2 and manifest["outcomes"] == {}
    assert [
        coordinator.session.candidates[r.candidate_id]["role"] for r in runner.requests
    ] == ["A", "B"]
    assert list(coordinator.session.batch["outcomes"].values()) == [
        "failed",
        "completed",
    ]
    failed = coordinator.session.results[runner.requests[0].run_id]
    assert failed["report"] is None
    assert coordinator.session.candidates[runner.requests[0].candidate_id][
        "previous_run_id"
    ]
    await coordinator.close()


@pytest.mark.asyncio
async def test_edits_during_a_cannot_change_queued_b_and_concurrent_run_is_rejected(
    tmp_path,
):
    coordinator, runner, _writer, _store, _events = setup(tmp_path)
    runner.release.clear()
    task = asyncio.create_task(coordinator.run(tuple(coordinator.session.candidates)))
    await runner.started.wait()
    b = next(
        key
        for key, value in coordinator.session.candidates.items()
        if value["role"] == "B"
    )
    coordinator.set_session(
        edit_json(coordinator.session, b, '{"chunking":{"method":"fixed_size"}}')
    )
    with pytest.raises(RuntimeError):
        await coordinator.run((b,))
    runner.release.set()
    await task
    assert runner.requests[1].recipe == runner.requests[0].recipe
    assert (
        coordinator.session.candidates[b]["draft"]["raw_json"]
        == '{"chunking":{"method":"fixed_size"}}'
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_initial_save_failure_prevents_any_launch_and_retains_manifest(tmp_path):
    coordinator, runner, writer, store, _events = setup(tmp_path)
    store.fail_save = True
    with pytest.raises(OSError):
        await coordinator.run(tuple(coordinator.session.candidates))
    assert not runner.requests
    assert coordinator.session.batch
    assert writer.status.state == "failed"
    store.fail_save = False
    await coordinator.close()


@pytest.mark.asyncio
async def test_undo_removing_a_stops_worker_and_never_launches_b(tmp_path):
    coordinator, runner, _writer, _store, _events = setup(tmp_path)
    runner.release.clear()
    task = asyncio.create_task(coordinator.run(tuple(coordinator.session.candidates)))
    await runner.started.wait()
    coordinator.set_session(undo_edit(coordinator.session))
    assert coordinator.session.batch is None
    with pytest.raises(RuntimeError):
        await coordinator.run(tuple(coordinator.session.candidates))
    await task
    assert runner.stopped and len(runner.requests) == 1
    assert coordinator.session.batch is None
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_restore_stops_before_replace_and_preserves_retry_authority(
    tmp_path,
):
    coordinator, runner, writer, store, events = setup(tmp_path)
    runner.release.clear()
    task = asyncio.create_task(coordinator.run(tuple(coordinator.session.candidates)))
    await runner.started.wait()
    epoch = coordinator.session.epoch
    store.fail_replace = True
    with pytest.raises(OSError):
        await coordinator.replace_recovery(export_recovery(new_session("imported")))
    await task
    assert events.index(("stop",)) < events.index(("replace",))
    assert coordinator.session.epoch == epoch
    assert len(runner.requests) == 1
    token = await writer.flush()
    assert token.epoch == epoch
    await coordinator.close()


@pytest.mark.asyncio
async def test_malformed_restore_leaves_running_session_untouched(tmp_path):
    coordinator, runner, _writer, _store, _events = setup(tmp_path)
    runner.release.clear()
    task = asyncio.create_task(coordinator.run(tuple(coordinator.session.candidates)))
    await runner.started.wait()
    before = coordinator.session
    with pytest.raises(ValueError):
        await coordinator.replace_recovery(b"{malformed")
    assert coordinator.session is before
    assert not runner.stopped
    runner.release.set()
    await task
    await coordinator.close()


@pytest.mark.asyncio
async def test_save_failure_retains_result_in_memory_and_retry_commits_it(tmp_path):
    coordinator, runner, writer, store, _events = setup(tmp_path)
    runner.release.clear()
    b = next(
        key
        for key, value in coordinator.session.candidates.items()
        if value["role"] == "B"
    )
    task = asyncio.create_task(coordinator.run((b,)))
    await runner.started.wait()
    store.fail_save = True
    runner.release.set()
    with pytest.raises(OSError):
        await task
    result = coordinator.session.results[runner.requests[0].run_id]
    assert result["status"] == "completed" and result["report"]["chunks"]
    assert coordinator.save_status.state == "failed"
    store.fail_save = False
    token = await writer.flush()
    assert token.revision == coordinator.session.revision
    await coordinator.close()


@pytest.mark.asyncio
async def test_close_failure_keeps_writer_retryable_and_memory_exportable(tmp_path):
    coordinator, _runner, writer, store, _events = setup(tmp_path)
    store.fail_save = True
    with pytest.raises(OSError):
        await coordinator.close()
    assert export_recovery(coordinator.session)
    store.fail_save = False
    token = await writer.flush()
    assert token.epoch == coordinator.session.epoch
    await coordinator.close()


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_clear_guard_survives_cancel_during_initial_manifest_commit(tmp_path):
    module = api()
    events = []
    save_entered, save_release, clear_entered, clear_release = (
        threading.Event() for _ in range(4)
    )

    class GatedStore(RecordingStore):
        def save(self, session, *, expected):
            save_entered.set()
            assert save_release.wait(5)
            return super().save(session, expected=expected)

        def clear(self, **kwargs):
            clear_entered.set()
            assert clear_release.wait(5)
            return super().clear(**kwargs)

    store = GatedStore(tmp_path / "lab.sqlite3", events)
    runner = RecordingRunner(events)
    coordinator = module.LabCoordinator(comparison(), AutosaveWriter(store), runner)
    run = asyncio.create_task(coordinator.run(tuple(coordinator.session.candidates)))
    await wait_until(save_entered.is_set)
    clear = asyncio.create_task(coordinator.clear())
    await asyncio.sleep(0)
    save_release.set()
    await wait_until(clear_entered.is_set)
    try:
        assert coordinator.guarded
        with pytest.raises(RuntimeError):
            coordinator.set_session(coordinator.session)
        with pytest.raises(RuntimeError):
            await coordinator.run(tuple(coordinator.session.candidates))
        assert not runner.requests
    finally:
        clear_release.set()
        await clear
        await run
        await coordinator.close()


@pytest.mark.asyncio
async def test_canceled_restore_await_keeps_guard_until_commit_and_undo_restores(
    tmp_path,
):
    coordinator, _runner, _writer, store, _events = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = store.replace

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    store.replace = gated
    before = coordinator.session
    task = asyncio.create_task(
        coordinator.replace_recovery(export_recovery(new_session("foreign")))
    )
    await wait_until(entered.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert coordinator.guarded and coordinator.session.epoch == before.epoch
    with pytest.raises(RuntimeError):
        coordinator.set_session(before)
    release.set()
    await wait_until(lambda: not coordinator.guarded)
    assert coordinator.session.epoch != before.epoch
    await coordinator.undo_restore()
    assert coordinator.session.samples == before.samples
    assert coordinator.session.epoch != before.epoch
    await coordinator.close()


@pytest.mark.asyncio
async def test_recovery_admission_replaces_unpersistable_success_with_limited(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chunking import lab_recovery
    from tldw_chatbook.Chunking.lab_models import ExecutionReport

    monkeypatch.setattr(lab_recovery, "MAX_RESULT_BYTES", 4096)
    events = []

    class LargeResultRunner(RecordingRunner):
        async def run(self, request):
            result = await super().run(request)
            return result.model_copy(
                update={
                    "report": ExecutionReport(chunks=(), transformed_text="x" * 5000)
                }
            )

    runner = LargeResultRunner(events)
    writer = AutosaveWriter(RecordingStore(tmp_path / "lab.sqlite3", events))
    coordinator = api().LabCoordinator(new_session("profile"), writer, runner)
    await coordinator.run(tuple(coordinator.session.candidates))
    assert set(coordinator.session.batch["outcomes"].values()) == {"limited"}
    assert all(
        coordinator.session.results[request.run_id]["report"] is None
        for request in runner.requests
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_subscribers_receive_small_copied_events_without_copying_blobs(
    tmp_path, monkeypatch
):
    coordinator, _runner, _writer, _store, _events = setup(tmp_path)
    observed = []
    from tldw_chatbook.Chunking.lab_models import LabSession

    original_copy = LabSession.model_copy

    def refuse_full_copy(self, *, update=None, deep=False):
        assert not deep, "Edit/status events must not copy immutable blobs"
        return original_copy(self, update=update, deep=deep)

    monkeypatch.setattr(LabSession, "model_copy", refuse_full_copy)
    detach = coordinator.subscribe(observed.append)
    await coordinator.run(tuple(coordinator.session.candidates))
    await wait_until(
        lambda: (
            observed
            and observed[-1].save_status.state == "saved"
            and not observed[-1].busy
        )
    )
    assert observed[-1].epoch == coordinator.session.epoch
    assert observed[-1].revision == coordinator.session.revision
    assert not hasattr(observed[-1], "session")
    detach()
    await coordinator.close()


@pytest.mark.asyncio
async def test_capture_install_accept_and_initial_load_never_block_ui_loop(
    tmp_path, monkeypatch
):
    module = api()
    main_thread = threading.get_ident()
    threads = []
    for name in ("capture_batch", "install_batch", "accept_result"):
        original = getattr(module, name)

        def slow(*args, original=original, **kwargs):
            threads.append(threading.get_ident())
            time.sleep(0.11)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, slow)
    coordinator, _runner, _writer, _store, _events = setup(tmp_path)
    await coordinator.run(tuple(coordinator.session.candidates))
    assert threads and main_thread not in threads
    await coordinator.close()
    reader = AutosaveWriter(RecordingStore(tmp_path / "lab.sqlite3", []))
    fresh_runner = RecordingRunner([])
    restored = await module.LabCoordinator.load("profile", reader, fresh_runner)
    assert restored.session.results and not fresh_runner.requests
    await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["replace_recovery", "clear", "close"])
async def test_real_noncooperative_child_reaped_before_authority_transition(
    tmp_path, monkeypatch, action
):
    from tldw_chatbook.Chunking import lab_runner

    events = []
    pidfile = tmp_path / "child.pid"
    code = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(100)"
    )
    monkeypatch.setattr(
        lab_runner, "_worker_command", lambda: [sys.executable, "-c", code]
    )

    def assert_reaped():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
        events.append(("reaped-before-authority",))

    class CheckedStore(RecordingStore):
        def replace(self, *args, **kwargs):
            assert_reaped()
            return super().replace(*args, **kwargs)

        def clear(self, **kwargs):
            assert_reaped()
            return super().clear(**kwargs)

        def close(self):
            assert_reaped()
            return super().close()

    writer = AutosaveWriter(CheckedStore(tmp_path / "lab.sqlite3", events))
    coordinator = api().LabCoordinator(
        comparison(), writer, lab_runner.LocalPreviewRunner(lab_runner.PreviewLimits())
    )
    before = coordinator.session
    run = asyncio.create_task(coordinator.run(tuple(before.candidates)))
    await wait_until(pidfile.exists)
    if action == "replace_recovery":
        await coordinator.replace_recovery(export_recovery(new_session("imported")))
    else:
        await getattr(coordinator, action)()
    await run
    assert ("reaped-before-authority",) in events
    if action != "close":
        assert coordinator.session.epoch != before.epoch
        assert coordinator.session.batch is None
        await coordinator.close()
    else:
        assert set(coordinator.session.batch["outcomes"].values()) == {"canceled"}


@pytest.mark.asyncio
async def test_changed_catalog_and_source_cannot_change_queued_request(tmp_path):
    # A tiny real SQLite catalog is the external mutable loading boundary. The
    # coordinator receives only the detached state, never this connection.
    catalog = sqlite3.connect(":memory:")
    catalog.execute("CREATE TABLE recipes(body TEXT, version INTEGER)")
    catalog.execute(
        "INSERT INTO recipes VALUES (?, ?)", ('{"chunking":{"method":"words"}}', 3)
    )
    source = tmp_path / "source.txt"
    source.write_text("copied original sample")
    session = replace_sample(
        new_session("profile"),
        source.read_text(),
        {"kind": "file", "path": str(source)},
    )
    b = next(iter(session.candidates))
    body, version = catalog.execute("SELECT body, version FROM recipes").fetchone()
    session = replace_template(
        session,
        b,
        json.loads(body),
        record_fields={"name": "Saved", "description": "", "tags": []},
        expected_record={"id": 1, "uuid": "saved", "version": version},
    )
    requests = capture_batch(session, (b,))
    session = pin_baseline(
        accept_result(install_batch(session, requests), outcome(requests[0]))
    )
    events = []
    runner = RecordingRunner(events)
    runner.release.clear()
    coordinator = api().LabCoordinator(
        session,
        AutosaveWriter(RecordingStore(tmp_path / "lab.sqlite3", events)),
        runner,
    )
    task = asyncio.create_task(coordinator.run(tuple(session.candidates)))
    await runner.started.wait()
    changed = '{"chunking":{"method":"fixed_size"}}'
    catalog.execute("UPDATE recipes SET body=?, version=?", (changed, 4))
    source.write_text("externally changed source")
    runner.release.set()
    await task
    queued = runner.requests[1]
    assert queued.template_record["version"] == 3
    assert json.loads(queued.recipe.authored_json)["chunking"]["method"] == "words"
    assert queued.sample.text == "copied original sample"
    assert catalog.execute("SELECT body,version FROM recipes").fetchone() == (
        changed,
        4,
    )
    assert source.read_text() == "externally changed source"
    catalog.close()
    await coordinator.close()


@pytest.mark.asyncio
async def test_real_worker_completion_is_durable_and_reopens_without_dispatch(tmp_path):
    from tldw_chatbook.Chunking.lab_runner import LocalPreviewRunner, PreviewLimits

    path = tmp_path / "lab.sqlite3"
    runner = LocalPreviewRunner(PreviewLimits())
    writer = AutosaveWriter(CheckpointStore(path, "profile"))
    coordinator = await api().LabCoordinator.load("profile", writer, runner)
    coordinator.set_session(
        replace_sample(coordinator.session, "durable real preview", {"kind": "paste"})
    )
    await coordinator.run(tuple(coordinator.session.candidates))
    request = next(iter(coordinator.session.batch["requests"].values()))
    assert coordinator.session.results[request["run_id"]]["status"] == "completed"
    await coordinator.close()
    recording = RecordingRunner([])
    reopened = await api().LabCoordinator.load(
        "profile", AutosaveWriter(CheckpointStore(path, "profile")), recording
    )
    result = reopened.session.results[request["run_id"]]
    assert result["report"]["chunks"][0]["text"] == "durable real preview"
    assert not recording.requests
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "replace_recovery", "close"])
async def test_transition_waits_for_run_not_its_callers_remaining_lifetime(
    tmp_path, action
):
    coordinator, runner, _writer, _store, _events = setup(tmp_path)
    runner.release.clear()
    run_returned = asyncio.Event()
    caller_release = asyncio.Event()

    async def caller():
        await coordinator.run(tuple(coordinator.session.candidates))
        run_returned.set()
        await caller_release.wait()

    caller_task = asyncio.create_task(caller())
    await runner.started.wait()
    if action == "replace_recovery":
        transition = asyncio.create_task(
            coordinator.replace_recovery(export_recovery(new_session("imported")))
        )
    else:
        transition = asyncio.create_task(getattr(coordinator, action)())
    try:
        await asyncio.wait_for(run_returned.wait(), 3)
        assert runner.stopped and not caller_task.done()
        # The caller intentionally stays alive: this wait must not depend on it.
        await asyncio.wait_for(asyncio.shield(transition), 1)
        assert not coordinator.guarded
        assert not caller_task.done()
    finally:
        caller_release.set()
        await asyncio.wait_for(caller_task, 3)
        await asyncio.wait_for(transition, 3)
        await coordinator.close()


@pytest.mark.asyncio
async def test_run_caller_can_join_the_close_already_waiting_for_its_run(tmp_path):
    coordinator, runner, _writer, _store, _events = setup(tmp_path)
    runner.release.clear()
    run_returned = asyncio.Event()

    async def caller():
        await coordinator.run(tuple(coordinator.session.candidates))
        run_returned.set()
        await coordinator.close()

    caller_task = asyncio.create_task(caller())
    await runner.started.wait()
    close = asyncio.create_task(coordinator.close())
    try:
        await asyncio.wait_for(run_returned.wait(), 3)
        await asyncio.wait_for(asyncio.shield(close), 1)
        await asyncio.wait_for(asyncio.shield(caller_task), 1)
        assert not coordinator.guarded
    finally:
        # Break the pre-fix dependency cycle so a RED run releases its real DB.
        if not caller_task.done():
            caller_task.cancel()
        await asyncio.gather(caller_task, return_exceptions=True)
        await asyncio.wait_for(close, 3)
        await coordinator.close()
