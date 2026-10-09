"""Host shutdown must retain each finite presentation worker group."""

import asyncio
import threading

import pytest
from textual.app import App

from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.UI.Console_Modules.view_workers import (
    capture_console_view_workers,
    drain_console_view_workers,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group",
    [
        "console-readiness-config",
        "console-agent-history",
        "console-subagent-counts",
        "console-subagent-count-publication",
        "console-citation-counts",
        "console-context-publication",
        "console-hook-refresh",
        "console-manual-unread-load",
        "console-readiness-publication",
    ],
)
async def test_host_drain_retains_finite_presentation_worker(group):
    host = App()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    reads = set()

    def callback():
        entered.set()
        try:
            assert release.wait(5)
        finally:
            returned.set()

    async def work():
        await run_preparation_read(
            callback,
            creator=host,
            session_id=None,
            reads=reads,
            require_current=lambda: None,
        )

    with host._context():
        worker = host.run_worker(work(), group=group, exit_on_error=False)
        drain = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert len(reads) == 1
            read = next(iter(reads))
            assert not read._producer.done()
            captured = capture_console_view_workers(host)
            selected = any(row[0] is worker for row in captured[-1])
            drain = asyncio.create_task(drain_console_view_workers(captured))
            await asyncio.wait({drain}, timeout=0.05)
            returned_early = drain.done()
            assert not returned.is_set() and not read._producer.done()
        finally:
            release.set()
            if drain is not None:
                await drain
            await asyncio.gather(worker._task, return_exceptions=True)
        assert returned.is_set() and read._producer.done() and not reads
        assert selected, f"Host shutdown omitted {group}"
        assert not returned_early, "Host drain returned with the physical callback live"
