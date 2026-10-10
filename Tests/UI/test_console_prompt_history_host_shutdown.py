"""Host shutdown retains the composer's original finite prompt-history load."""

import asyncio
import json
import threading

import pytest
from textual.app import App

from Tests.Chat.test_default_prompt_history_lifetime import (
    default_history_profile,  # noqa: F401
    local_scope,  # noqa: F401
    observe_history_callback,
)
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.UI.Console_Modules.view_workers import (
    capture_console_view_workers,
    drain_console_view_workers,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


@pytest.mark.asyncio
async def test_host_drain_retains_original_composer_prompt_history_load(
    default_history_profile,  # noqa: F811 - imported fixture
):
    seed = PromptHistory()
    assert await seed.append("original prompt history")
    history = PromptHistory()
    entered, release = threading.Event(), threading.Event()
    decoder_code = json.loads.__code__
    body_code = PromptHistory._history_io.__code__

    def hold_original_read(frame, event, _argument):
        caller = frame.f_back
        if (
            event == "call"
            and frame.f_code is decoder_code
            and caller is not None
            and caller.f_code is body_code
            and caller.f_locals["self"] is history
        ):
            entered.set()
            assert release.wait(8), "original history read was not released"

    host = App()
    composer = ConsoleComposerBar()
    drain = None
    with observe_history_callback(history, hold_original_read) as (observed, _executor):
        with host._context():
            worker = composer.run_worker(
                history.load(), group="console-prompt-history", exit_on_error=False
            )
            try:
                for _ in range(300):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert entered.is_set(), "original native history read never entered"
                assert len(observed.jobs) == 1
                job = observed.jobs[0]
                states = [s for s in raw._states.values() if s.source is history]
                assert len(states) == 1 and states[0].active
                leases = tuple(states[0].leases)
                assert leases and all(lease in storage._live_leases for lease in leases)
                captured = capture_console_view_workers(host)
                selected = any(row[0] is worker for row in captured[-1])
                drain = asyncio.create_task(drain_console_view_workers(captured))
                await asyncio.wait({drain}, timeout=0.03)
                drain.cancel()
                await asyncio.wait({drain}, timeout=0.03)
                drain.cancel()
                await asyncio.wait({drain}, timeout=0.03)
                returned_early = drain.done()
                assert not worker._task.done() and job._state == "running"
                assert history._append_lock.locked()
                assert all(lease in storage._live_leases for lease in leases)
            finally:
                release.set()
                if drain is not None:
                    outcomes = await asyncio.gather(drain, return_exceptions=True)
                    assert all(
                        value is None or isinstance(value, asyncio.CancelledError)
                        for value in outcomes
                    )
                await asyncio.gather(worker._task, return_exceptions=True)
            assert selected, "Host drain omitted the original composer history worker"
            assert (
                not returned_early
            ), "Host drain returned before the native history read"
            assert job._state == "closed"
            assert job._attempt not in storage._pending_acquisitions
            assert not any(s.source is history for s in raw._states.values())
            assert all(lease not in storage._live_leases for lease in leases)
            assert history._loaded and history.size == 1
            assert not history._append_lock.locked()
