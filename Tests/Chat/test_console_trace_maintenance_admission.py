"""Busy maintenance never pays for admission just to refuse its callback."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_trace_maintenance import LegacyMaintenanceBatch
from tldw_chatbook.Chat.console_trace_maintenance_runtime import (
    run_legacy_trace_maintenance,
)


@pytest.mark.asyncio
async def test_provider_busy_poll_defers_admission_until_idle():
    runtime = SimpleNamespace(
        _disposed=False,
        _chat_controller=SimpleNamespace(_active_stream_tasks={"turn": object()}),
    )
    polls = 0
    admissions = []

    async def sleep(delay):
        nonlocal polls
        polls += 1
        assert delay == 1.0
        if polls == 2:
            runtime._chat_controller._active_stream_tasks.clear()

    async def owned_call(database, operation, *args, **kwargs):
        admissions.append(bool(runtime._chat_controller._active_stream_tasks))
        if runtime._chat_controller._active_stream_tasks:
            # Keep the existing worker's race guard live: this must refuse
            # without touching our deliberately unusable database.
            return operation(*args, **kwargs)
        runtime._disposed = True
        return LegacyMaintenanceBatch(False, 0, 0, False)

    source = SimpleNamespace(
        asyncio=SimpleNamespace(
            sleep=sleep,
            create_task=asyncio.create_task,
            shield=asyncio.shield,
            CancelledError=asyncio.CancelledError,
        ),
        time=time,
        run_owned_db_call=owned_call,
    )
    await run_legacy_trace_maintenance(runtime, object(), object, _source=source)
    assert admissions == [False]
