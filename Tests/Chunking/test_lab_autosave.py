"""Serialized autosave behavior using real SQLite and explicitly gated storage."""

import asyncio
import sqlite3
import threading
import time

import pytest

from tldw_chatbook.Chunking.lab_autosave import AutosaveWriter
from tldw_chatbook.Chunking.lab_state import (
    capture_batch,
    edit_json,
    install_batch,
    new_session,
    update_view,
)
from tldw_chatbook.DB.Chunking_Lab_DB import (
    CheckpointConflict,
    CheckpointStore,
    RecoverySchemaError,
)


async def wait_until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class ObservedStore(CheckpointStore):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = []
        self.threads = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def load(self):
        self.threads.append(threading.get_ident())
        return super().load()

    def save(self, session, *, expected):
        self.threads.append(threading.get_ident())
        self.entered.set()
        if not self.release.wait(4):
            raise RuntimeError("Test storage gate timed out")
        if self.fail:
            raise OSError("disk unavailable")
        result = super().save(session, expected=expected)
        self.calls.append((time.monotonic(), session.revision))
        return result


@pytest.mark.asyncio
async def test_writer_lazily_loads_and_commits_off_ui_thread(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = ObservedStore(path, "profile")
    writer = AutosaveWriter(store)
    assert not path.exists()
    assert await writer.load() is None
    session = new_session("profile")
    writer.submit(session)
    token = await writer.flush()
    assert token.revision == session.revision
    assert writer.status.state == "saved"
    assert all(thread != threading.get_ident() for thread in store.threads)
    snapshot = writer.status
    writer.submit(update_view(session, {"tab": "Compare"}))
    assert snapshot.state == "saved" and writer.status.state == "saving"
    await writer.close()
    reopened = CheckpointStore(path, "profile")
    assert reopened.load()[0].view["tab"] == "Compare"
    reopened.close()


@pytest.mark.asyncio
async def test_trailing_debounce_coalesces_latest_and_immediate_bypasses_wait(tmp_path):
    store = ObservedStore(tmp_path / "lab.sqlite3", "profile")
    writer = AutosaveWriter(store)
    session = new_session("profile")
    started = time.monotonic()
    for _ in range(4):
        session = update_view(session, {"tab": "Draft"})
        writer.submit(session)
        await asyncio.sleep(0.05)
    assert not store.calls
    await wait_until(lambda: writer.status.state == "saved")
    assert len(store.calls) == 1 and store.calls[0][1] == 4
    assert store.calls[0][0] - started >= 0.4
    store.entered.clear()
    session = update_view(session, {"tab": "Compare"})
    writer.submit(session, immediate=True)
    await wait_until(store.entered.is_set, timeout=0.25)
    await writer.close()


@pytest.mark.asyncio
async def test_continuous_typing_checkpoints_within_one_second(tmp_path):
    store = ObservedStore(tmp_path / "lab.sqlite3", "profile")
    writer = AutosaveWriter(store)
    session = new_session("profile")
    started = time.monotonic()
    for _ in range(24):
        session = update_view(session, {"tab": "Draft"})
        writer.submit(session)
        await asyncio.sleep(0.05)
    assert store.calls, "Continuous edits starved the maximum-wait checkpoint"
    assert store.calls[0][0] - started < 1.15
    assert store.calls[0][1] < session.revision
    await writer.close()


@pytest.mark.asyncio
async def test_old_acknowledgment_cannot_mark_newer_draft_saved(tmp_path):
    store = ObservedStore(tmp_path / "lab.sqlite3", "profile")
    store.release.clear()
    writer = AutosaveWriter(store)
    session = new_session("profile")
    writer.submit(session, immediate=True)
    await wait_until(store.entered.is_set)
    newer = edit_json(session, next(iter(session.candidates)), "{")
    writer.submit(newer)
    store.release.set()
    await wait_until(lambda: writer.status.acknowledged is not None)
    assert writer.status.acknowledged.revision == 0
    assert writer.status.latest_revision == 1
    assert writer.status.state == "saving"
    await writer.close()


@pytest.mark.asyncio
async def test_disk_failure_preserves_memory_and_retry_uses_latest(tmp_path):
    store = ObservedStore(tmp_path / "lab.sqlite3", "profile")
    writer = AutosaveWriter(store)
    session = new_session("profile")
    writer.submit(session)
    original = await writer.flush()
    store.fail = True
    failed = update_view(session, {"tab": "Compare"})
    writer.submit(failed)
    with pytest.raises(OSError):
        await writer.flush()
    assert writer.status.state == "failed"
    assert writer.status.acknowledged == original
    latest = edit_json(failed, next(iter(failed.candidates)), "{latest")
    writer.submit(latest)
    assert writer.status.state == "failed"
    store.fail = False
    token = await writer.flush()
    assert token.revision == latest.revision
    assert writer.status.state == "saved" and writer.status.error is None
    await writer.close()


@pytest.mark.asyncio
async def test_two_writers_conflict_stops_automatic_overwrite(tmp_path):
    path = tmp_path / "lab.sqlite3"
    first = AutosaveWriter(CheckpointStore(path, "profile"))
    second = AutosaveWriter(CheckpointStore(path, "profile"))
    original = new_session("profile")
    first.submit(original)
    await first.flush()
    loaded, _ = await second.load()
    first.submit(update_view(original, {"tab": "Compare"}))
    await first.flush()
    losing = edit_json(loaded, next(iter(loaded.candidates)), "{losing")
    second.submit(losing)
    with pytest.raises(CheckpointConflict):
        await second.flush()
    assert second.status.state == "conflict"
    second.submit(update_view(losing, {"tab": "Draft"}))
    await asyncio.sleep(0.4)
    assert second.status.state == "conflict"
    with pytest.raises(CheckpointConflict):
        await second.close()
    await first.close()
    reopened = CheckpointStore(path, "profile")
    assert reopened.load()[0].view["tab"] == "Compare"
    reopened.close()


@pytest.mark.asyncio
async def test_load_failure_never_grants_empty_overwrite_authority(tmp_path):
    path = tmp_path / "lab.sqlite3"
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version=2")
    writer = AutosaveWriter(CheckpointStore(path, "profile"))
    with pytest.raises(RecoverySchemaError):
        await writer.load()
    assert writer.status.state == "failed"
    writer.submit(new_session("profile"), immediate=True)
    with pytest.raises(RecoverySchemaError):
        await writer.flush()
    with pytest.raises(RecoverySchemaError):
        await writer.close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_clear_waits_for_inflight_write_then_fences_old_submissions(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = ObservedStore(path, "profile")
    writer = AutosaveWriter(store)
    old = new_session("profile")
    writer.submit(old)
    await writer.flush()
    store.entered.clear()
    store.release.clear()
    pending = update_view(old, {"tab": "Compare"})
    writer.submit(pending, immediate=True)
    await wait_until(store.entered.is_set)
    clearing = asyncio.create_task(writer.clear())
    await asyncio.sleep(0)
    with pytest.raises(CheckpointConflict):
        writer.submit(update_view(pending, {"tab": "Draft"}))
    store.release.set()
    fresh, token = await clearing
    assert fresh.epoch != old.epoch and writer.status.acknowledged == token
    with pytest.raises(CheckpointConflict):
        writer.submit(pending, immediate=True)
    await writer.close()
    reopened = CheckpointStore(path, "profile")
    assert reopened.load()[1] == token
    reopened.close()


@pytest.mark.asyncio
async def test_later_typing_cannot_delay_an_already_immediate_checkpoint(tmp_path):
    store = ObservedStore(tmp_path / "lab.sqlite3", "profile")
    writer = AutosaveWriter(store)
    session = new_session("profile")
    writer.submit(session, immediate=True)
    writer.submit(update_view(session, {"tab": "Compare"}))
    try:
        await wait_until(store.entered.is_set, timeout=0.2)
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_canceling_flush_await_keeps_commit_serialized_before_clear(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = ObservedStore(path, "profile")
    writer = AutosaveWriter(store)
    original = new_session("profile")
    writer.submit(original)
    await writer.flush()
    store.release.clear()
    store.entered.clear()
    writer.submit(update_view(original, {"tab": "Compare"}))
    flushing = asyncio.create_task(writer.flush())
    await wait_until(store.entered.is_set)
    flushing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flushing
    clearing = asyncio.create_task(writer.clear())
    await asyncio.sleep(0)
    store.release.set()
    fresh, token = await clearing
    await writer.close()
    reopened = CheckpointStore(path, "profile")
    assert reopened.load()[1] == token
    assert fresh.epoch == token.epoch != original.epoch
    reopened.close()


@pytest.mark.asyncio
async def test_initial_load_autosaves_interrupted_normalization_without_an_edit(
    tmp_path,
):
    path = tmp_path / "lab.sqlite3"
    session = new_session("profile")
    requests = capture_batch(session, tuple(session.candidates))
    session = install_batch(session, requests)
    seed = CheckpointStore(path, "profile")
    seed.save(session, expected=None)
    seed.close()
    writer = AutosaveWriter(CheckpointStore(path, "profile"))
    try:
        restored, token = await writer.load()
        assert restored.revision > token.revision
        await wait_until(lambda: writer.status.state == "saved")
        assert writer.status.acknowledged.revision == restored.revision
    finally:
        await writer.close()
