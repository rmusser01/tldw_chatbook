"""Logs mount seeds the newest slice synchronously and backfills the rest."""

import asyncio

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Logs_Window import (
    _BACKFILL_CHUNK_RECORDS,
    _MOUNT_SEED_RECORDS,
    LogsWindow,
)

pytestmark = pytest.mark.bootstrap_profile

TOTAL = _MOUNT_SEED_RECORDS + _BACKFILL_CHUNK_RECORDS + 17


class _Harness(App):
    def __init__(self) -> None:
        super().__init__()
        import threading
        from collections import deque

        self._log_records = deque(
            ("INFO", "app", f"line {index}") for index in range(TOTAL)
        )
        self._log_records_seq = TOTAL
        self._log_records_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield LogsWindow(self)


async def _mounted_window():
    app = _Harness()
    async with app.run_test() as pilot:
        window = app.query_one(LogsWindow)
        window.load_from_app()  # the Logs screen calls this from its on_mount
        await pilot.pause()
        return app, pilot, window


@pytest.mark.asyncio
async def test_mount_seeds_only_the_newest_slice():
    app = _Harness()
    async with app.run_test() as pilot:
        window = app.query_one(LogsWindow)
        window.load_from_app()  # the Logs screen calls this from its on_mount
        # Synchronously: only the newest slice is seeded, before the worker.
        assert len(window._records) == _MOUNT_SEED_RECORDS
        assert window._records[-1].message == f"line {TOTAL - 1}", (
            "the newest lines must paint at mount"
        )
        assert window._seeded_through == TOTAL
        await pilot.pause()


@pytest.mark.asyncio
async def test_backfill_restores_the_full_buffer_in_order():
    _app, _pilot, window = await _mounted_window()
    async with asyncio.timeout(10):
        while len(window._records) < TOTAL:
            await asyncio.sleep(0.01)
    assert len(window._records) == TOTAL
    assert window._records[0].message == "line 0", "backfill splices in front"
    assert window._records[-1].message == f"line {TOTAL - 1}"


@pytest.mark.asyncio
async def test_live_records_arriving_during_backfill_keep_their_order():
    app = _Harness()
    async with app.run_test() as pilot:
        window = app.query_one(LogsWindow)
        window.load_from_app()  # the Logs screen calls this from its on_mount
        await pilot.pause()
        window.append_record("ERROR", "live", "live during backfill")
        async with asyncio.timeout(10):
            while len(window._records) < TOTAL + 1:
                await asyncio.sleep(0.01)
        assert window._records[-1].message == "live during backfill"
        assert window._records[-2].message == f"line {TOTAL - 1}"
        assert window._records[0].message == "line 0"
