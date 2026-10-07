"""PERF-10 (TASK-33269): legacy trace maintenance parks when there is no work.

The loop used to call ``run_batch`` -- a ``BEGIN IMMEDIATE`` write transaction
with an admission and, through ``run_owned_db_call``, a fresh connection and
private-SQLite helper spawn -- once a second for the life of the process,
even with nothing to normalize (2026-09-27 audit: 6-9 ms CPU plus ~45 ms of
helper CPU per tick). It now parks after a complete pass, and wakes when an
exchange is written or once per GC interval (other writers advance the graph
epoch without signalling, and a failed GC attempt must be retried).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_legacy_migration import _capture, _message
from tldw_chatbook.Chat import chat_persistence_service
from tldw_chatbook.Chat import console_runtime as runtime_module
from tldw_chatbook.Chat.console_exchange_capture import capture_to_blob
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
from tldw_chatbook.Chat.console_trace_maintenance import (
    LegacyTraceMaintenance,
    PhysicalTraceCompactor,
    TraceGarbageCollector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _install_fake_maintenance(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """A normalization worker that is always complete; no GC classes (test double)."""

    class _Maintenance:
        def __init__(self, _database: object, **_kwargs: object) -> None:
            pass

        def run_batch(self) -> SimpleNamespace:
            calls.append("batch")
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)


#: The real sleep, captured before any test shortens the runtime's sleeps
#: (``asyncio`` is one shared module, so the patch reaches the test too).
_real_sleep = asyncio.sleep


def _fast_runtime(monkeypatch: pytest.MonkeyPatch) -> ConsoleRuntime:
    real_sleep = _real_sleep

    async def fast_sleep(delay: float, *args: object, **kwargs: object) -> object:
        return await real_sleep(min(delay, 0.005), *args, **kwargs)

    monkeypatch.setattr(runtime_module.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(runtime_module, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 0.0)
    app = SimpleNamespace(_ui_ready=True, persona_buddy_controller=None)
    runtime = ConsoleRuntime(app)
    runtime._chat_controller = SimpleNamespace(_active_stream_tasks={})
    return runtime


@pytest.mark.asyncio
async def test_complete_maintenance_parks_instead_of_polling_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to normalize, run_batch is not called again and again.

    Args:
        monkeypatch: Swaps in the fake worker and shortens the loop's sleeps.
    """
    calls: list[str] = []
    _install_fake_maintenance(monkeypatch, calls)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _real_sleep(0.3)
    await runtime.dispose()

    assert 1 <= len(calls) <= 2, f"idle maintenance polled run_batch {len(calls)} times"


@pytest.mark.asyncio
async def test_an_exchange_write_signal_wakes_parked_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new exchange row is picked up promptly after the loop has parked.

    Args:
        monkeypatch: Swaps in the fake worker and shortens the loop's sleeps.
    """
    calls: list[str] = []
    _install_fake_maintenance(monkeypatch, calls)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _real_sleep(0.2)
    parked_calls = len(calls)
    chat_persistence_service.signal_trace_maintenance_work()
    await _real_sleep(0.2)
    await runtime.dispose()

    assert len(calls) > parked_calls, "the work signal did not wake maintenance"


@pytest.mark.asyncio
async def test_a_signal_wake_tells_the_pass_it_has_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-33801: a pass woken by an exchange write skips the idle check.

    Args:
        monkeypatch: Swaps in the fake worker and shortens the loop's sleeps.
    """
    flags: list[bool] = []

    class _Maintenance:
        def __init__(self, _database: object, **_kwargs: object) -> None:
            self.expect_work = True

        def run_batch(self) -> SimpleNamespace:
            flags.append(self.expect_work)
            self.expect_work = False
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _until_parked(flags)
    parked_at = len(flags)
    chat_persistence_service.signal_trace_maintenance_work()
    await _until(lambda: len(flags) > parked_at)
    await runtime.dispose()

    assert flags[parked_at] is True, f"the woken pass was not told it has work: {flags}"


@pytest.mark.asyncio
async def test_work_written_during_an_unparked_pass_is_flagged_for_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-33801: a write landing while a pass runs is flagged for the next one.

    Not only a write that wakes a parked loop: one landing during a pass, or
    the physical cleanup after it, is flagged too (Qodo, #2959).

    Args:
        monkeypatch: Swaps in the fake worker and shortens the loop's sleeps.
    """
    flags: list[bool] = []

    class _Maintenance:
        def __init__(self, _database: object, **_kwargs: object) -> None:
            self.expect_work = True

        def run_batch(self) -> SimpleNamespace:
            flags.append(self.expect_work)
            self.expect_work = False
            if len(flags) == 1:  # a write lands while the first pass runs
                chat_persistence_service.signal_trace_maintenance_work()
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _until(lambda: len(flags) >= 2)
    await runtime.dispose()

    assert flags[1] is True, f"the pass after a mid-pass write was not told: {flags}"


@pytest.mark.asyncio
async def test_one_signal_wakes_every_parked_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parked runtimes both wake on one exchange-write signal.

    Neither consumes the other's wake (Qodo, #2914).

    Args:
        monkeypatch: Swaps in the fake worker and shortens the loop's sleeps.
    """
    calls: list[object] = []

    class _Maintenance:
        def __init__(self, database: object, **_kwargs: object) -> None:
            self._database = database

        def run_batch(self) -> SimpleNamespace:
            calls.append(self._database)
            return SimpleNamespace(logical_complete=True, admitted=True)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = _Maintenance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtimes = [_fast_runtime(monkeypatch), _fast_runtime(monkeypatch)]
    databases = ["db-a", "db-b"]

    for runtime, database in zip(runtimes, databases):
        runtime._schedule_legacy_trace_maintenance(database, object)
    await _real_sleep(0.2)
    parked = {database: calls.count(database) for database in databases}
    chat_persistence_service.signal_trace_maintenance_work()
    await _real_sleep(0.2)
    for runtime in runtimes:
        await runtime.dispose()

    for database in databases:
        assert calls.count(database) > parked[database], f"{database} stayed parked"


def test_a_successful_exchange_append_signals_maintenance() -> None:
    """The only exchange writer raises the work signal after it appends."""

    class _DB:
        def append_message_exchanges_local(self, message_id: str, rows: object) -> None:
            del message_id, rows

    before = chat_persistence_service.trace_maintenance_work_generation()
    service = chat_persistence_service.ChatPersistenceService.__new__(
        chat_persistence_service.ChatPersistenceService
    )
    service.db = _DB()

    assert service.append_message_exchanges(message_id="m-1", rows=[]) is True
    assert chat_persistence_service.trace_maintenance_work_generation() > before


# -- Real database: the runtime loop with the real worker, collector and writer --


def _append_exchange(db: CharactersRAGDB, message_id: str, seq: int) -> None:
    """Append through the production writer, which raises the work signal."""
    capture = _capture(seq)
    service = chat_persistence_service.ChatPersistenceService.__new__(
        chat_persistence_service.ChatPersistenceService
    )
    service.db = db
    assert service.append_message_exchanges(
        message_id=message_id,
        rows=[
            {
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "status": capture.status,
                "abandoned": False,
                "capture_detail": capture.capture_detail.value,
                "capture_blob": capture_to_blob(capture),
                "created_at": capture.created_at,
            }
        ],
    )


async def _until(condition, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if condition():
            return
        await _real_sleep(0.01)
    raise AssertionError("condition not reached")


async def _until_parked(batches: list, quiet: float = 0.3, timeout: float = 10.0) -> None:
    """Wait until run_batch has not been called for ``quiet`` seconds."""
    for _ in range(int(timeout / quiet)):
        seen = len(batches)
        await _real_sleep(quiet)
        if len(batches) == seen:
            return
    raise AssertionError("maintenance never parked")


def _spy(monkeypatch: pytest.MonkeyPatch, owner: type, name: str, calls: list, fail_first=False):
    real = getattr(owner, name)

    def spied(self, *args, **kwargs):
        calls.append(name)
        if fail_first and len(calls) == 1:
            raise RuntimeError("collector unavailable")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, spied)


def _real_db(tmp_path: Path) -> tuple[CharactersRAGDB, str]:
    # File-backed: the loop's database calls run on worker threads, and an
    # in-memory database is private to the thread that opened it.
    db = CharactersRAGDB(tmp_path / "chacha.db", "perf10-parking")
    conversation_id = db.add_conversation({"title": "parking"})
    assert conversation_id is not None
    return db, conversation_id


@pytest.mark.asyncio
async def test_a_real_exchange_append_wakes_parked_maintenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production writer's signal wakes the parked loop, which normalizes the row.

    Args:
        monkeypatch: Spies on run_batch and shortens the loop's sleeps.
        tmp_path: Holds the real ChaChaNotes database.
    """
    db, conversation_id = _real_db(tmp_path)
    first = _message(db, conversation_id, "answer-0")
    _append_exchange(db, first, 0)
    batches: list[str] = []
    _spy(monkeypatch, LegacyTraceMaintenance, "run_batch", batches)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(db, lambda: LegacyTraceNormalizer(db))
    try:
        await _until(lambda: LegacyTraceNormalizer(db).read_calls(first))
        await _until_parked(batches)

        second = _message(db, conversation_id, "answer-1")
        _append_exchange(db, second, 1)
        await _until(lambda: LegacyTraceNormalizer(db).read_calls(second))
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_a_signal_woken_real_pass_opens_only_its_write_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-33801 end to end: a signal-woken real pass opens one transaction.

    The real worker, woken by an exchange written through the production
    writer, opens only its immediate write transaction -- no read-only idle
    check first (Qodo, #2959).

    Args:
        monkeypatch: Records the transactions each pass opens; shortens sleeps.
        tmp_path: Holds the real ChaChaNotes database.
    """
    db, conversation_id = _real_db(tmp_path)
    first = _message(db, conversation_id, "answer-0")
    _append_exchange(db, first, 0)
    passes: list[list[bool]] = []
    # Only the thread running the pass: the test's own polling reads the
    # database concurrently and must not be billed to it.
    in_pass = threading.local()
    real_run_batch = LegacyTraceMaintenance.run_batch

    def bracketed_run_batch(self):
        passes.append([])
        in_pass.active = True
        try:
            return real_run_batch(self)
        finally:
            in_pass.active = False

    real_transaction = db.transaction

    def recording_transaction(*, immediate: bool = False):
        if getattr(in_pass, "active", False):
            passes[-1].append(immediate)
        return real_transaction(immediate=immediate)

    monkeypatch.setattr(LegacyTraceMaintenance, "run_batch", bracketed_run_batch)
    monkeypatch.setattr(db, "transaction", recording_transaction)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(db, lambda: LegacyTraceNormalizer(db))
    try:
        await _until(lambda: LegacyTraceNormalizer(db).read_calls(first))
        # Silence can mean a queued batch or unfinished GC, not parking.
        # Observe the actual scheduler latch before billing its next wake.
        await _until(
            lambda: runtime._legacy_trace_maintenance_task.get_coro()
            .cr_frame.f_locals.get("parked")
            is True,
            timeout=10.0,
        )
        woken_at = len(passes)
        second = _message(db, conversation_id, "answer-1")
        _append_exchange(db, second, 1)
        await _until(lambda: LegacyTraceNormalizer(db).read_calls(second))
    finally:
        await runtime.dispose()

    assert passes[woken_at] == [True], f"the signal-woken pass opened {passes[woken_at]}"


@pytest.mark.asyncio
async def test_a_real_write_landing_mid_pass_flags_the_next_real_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-33801 end to end: a write landing during a real pass is flagged.

    The exchange is appended through the production writer just after the
    first real pass returns -- after the loop read the write generation,
    before its next comparison -- so the next pass (after the physical
    cleanup) must open only its immediate write transaction (Qodo, #2959).

    Args:
        monkeypatch: Records the transactions each pass opens; shortens sleeps.
        tmp_path: Holds the real ChaChaNotes database.
    """
    db, conversation_id = _real_db(tmp_path)
    first = _message(db, conversation_id, "answer-0")
    second = _message(db, conversation_id, "answer-1")
    _append_exchange(db, first, 0)
    passes: list[list[bool]] = []
    in_pass = threading.local()
    real_run_batch = LegacyTraceMaintenance.run_batch

    def bracketed_run_batch(self):
        passes.append([])
        in_pass.active = True
        try:
            return real_run_batch(self)
        finally:
            in_pass.active = False
            if len(passes) == 1:  # lands mid-pass, as the loop sees it
                _append_exchange(db, second, 1)

    real_transaction = db.transaction

    def recording_transaction(*, immediate: bool = False):
        if getattr(in_pass, "active", False):
            passes[-1].append(immediate)
        return real_transaction(immediate=immediate)

    monkeypatch.setattr(LegacyTraceMaintenance, "run_batch", bracketed_run_batch)
    monkeypatch.setattr(db, "transaction", recording_transaction)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(db, lambda: LegacyTraceNormalizer(db))
    try:
        await _until(lambda: LegacyTraceNormalizer(db).read_calls(second))
    finally:
        await runtime.dispose()

    assert passes[1] == [True], f"the pass after a mid-pass write opened {passes[1]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unsignalled-epoch-advance", "failed-collection"])
async def test_parked_maintenance_still_collects_once_per_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Writers other than exchange appends advance the graph epoch without a
    signal, and a failed collection sets no signal either; both are picked up
    at the next GC interval (Qodo, #2914).

    Args:
        monkeypatch: Spies on collection, shortens the GC interval and sleeps.
        tmp_path: Holds the real ChaChaNotes database.
        case: Which unsignalled reason the interval wake must cover.
    """
    db, conversation_id = _real_db(tmp_path)
    _append_exchange(db, _message(db, conversation_id, "answer-0"), 0)
    collects: list[str] = []
    _spy(monkeypatch, TraceGarbageCollector, "collect", collects,
         fail_first=case == "failed-collection")
    # Compaction waits for idle time; finish it so no result stays pending and
    # the loop's next GC check compares graph epochs.
    monkeypatch.setattr(
        PhysicalTraceCompactor,
        "run_after_gc",
        lambda self, result: SimpleNamespace(completed=True, reason_code="completed"),
    )
    monkeypatch.setattr(runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 0.1)
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(db, lambda: LegacyTraceNormalizer(db))
    try:
        await _until(lambda: len(collects) >= 1)
        if case == "unsignalled-epoch-advance":
            await _until_parked(collects, quiet=0.5)
            assert len(collects) == 1, "an unchanged graph was collected again"
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE console_trace_graph_epoch SET epoch = epoch + 1"
                    " WHERE singleton_id = 1"
                )
        await _until(lambda: len(collects) >= 2)
    finally:
        await runtime.dispose()
