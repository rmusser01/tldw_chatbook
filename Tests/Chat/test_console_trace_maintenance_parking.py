"""PERF-10 (TASK-33269): legacy trace maintenance parks when there is no work.

The loop used to call ``run_batch`` -- a ``BEGIN IMMEDIATE`` write transaction
with an admission and, through ``run_owned_db_call``, a fresh connection and
private-SQLite helper spawn -- once a second for the life of the process,
even with nothing to normalize (2026-09-27 audit: 6-9 ms CPU plus ~45 ms of
helper CPU per tick). It now parks after a complete pass, and wakes when an
exchange is written or a due GC pass needs to run.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tldw_chatbook.Chat import chat_persistence_service
from tldw_chatbook.Chat import console_runtime as runtime_module
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


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
    """With nothing to normalize, run_batch is not called again and again."""
    calls: list[str] = []
    _install_fake_maintenance(monkeypatch, calls)
    chat_persistence_service.consume_trace_maintenance_work_signal()
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _real_sleep(0.3)
    await runtime.dispose()

    assert 1 <= len(calls) <= 2, f"idle maintenance polled run_batch {len(calls)} times"


@pytest.mark.asyncio
async def test_an_exchange_write_signal_wakes_parked_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new exchange row is picked up promptly after the loop has parked."""
    calls: list[str] = []
    _install_fake_maintenance(monkeypatch, calls)
    chat_persistence_service.consume_trace_maintenance_work_signal()
    runtime = _fast_runtime(monkeypatch)

    runtime._schedule_legacy_trace_maintenance(object(), object)
    await _real_sleep(0.2)
    parked_calls = len(calls)
    chat_persistence_service.signal_trace_maintenance_work()
    await _real_sleep(0.2)
    await runtime.dispose()

    assert len(calls) > parked_calls, "the work signal did not wake maintenance"
    assert not chat_persistence_service.consume_trace_maintenance_work_signal()


def test_a_successful_exchange_append_signals_maintenance() -> None:
    """The only exchange writer raises the work signal after it appends."""

    class _DB:
        def append_message_exchanges_local(self, message_id: str, rows: object) -> None:
            del message_id, rows

    chat_persistence_service.consume_trace_maintenance_work_signal()
    service = chat_persistence_service.ChatPersistenceService.__new__(
        chat_persistence_service.ChatPersistenceService
    )
    service.db = _DB()

    assert service.append_message_exchanges(message_id="m-1", rows=[]) is True
    assert chat_persistence_service.consume_trace_maintenance_work_signal()
