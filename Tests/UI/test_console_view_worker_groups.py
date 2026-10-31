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
        "console-flat-conversation-search",
        "console-agent-history",
        "console-subagent-counts",
        "console-subagent-count-publication",
        "console-citation-counts",
        "console-library-activity",
        "console-library-outcome",  # TASK-33621.51
        "console-workspace-display-reads",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("group", ["console-agent-history", "console-subagent-counts", "console-hook-refresh"])
async def test_original_app_runtime_shutdown_waits_for_selected_native_callback(group):
    from tldw_chatbook.app_lifecycle import LifecycleMixin

    host = App()
    host._console_runtime_shutdown_task = None
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    reads = set()

    def callback():
        entered.set()
        try:
            assert release.wait(5)
        finally:
            returned.set()

    with host._context():
        worker = host.run_worker(
            run_preparation_read(callback, creator=host, session_id=None,
                reads=reads, require_current=lambda: None),
            group=group, exit_on_error=False,
        )
        shutdown = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            read, = reads
            shutdown = asyncio.create_task(LifecycleMixin._shutdown_console_runtime(host))
            await asyncio.wait({shutdown}, timeout=0.05)
            returned_early = shutdown.done()
            assert not read._producer.done() and not returned.is_set()
        finally:
            release.set()
            if shutdown is not None:
                await shutdown
            await asyncio.gather(worker._task, return_exceptions=True)
        assert returned.is_set() and read._producer.done() and not reads
        assert not returned_early, "App disposal returned while its callback was live"
