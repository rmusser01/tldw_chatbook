"""Stop pauses a composer send's Library search; it does not cancel the send.

TASK-33621.20, found live on the head rebased onto dev ce4aa4a231. A real
composer send stays in the runtime's intake custody (its received intent) for
its whole preparation, Library search included. Dev's 2026-10-15 Stop path
cancels intake custody before it reaches the Library search: Stop during
"Searching Library..." cancelled the whole send (``controller_submit
status=cancelled``), so no card offered Retry / Send once without Library /
Cancel. The send-harness tests did not see it: their sends never enter
intake custody.

This drives the stock received-intent path (real ChatScreen, runtime,
controller, store and file-backed database). Only the provider call and the
Library search service are doubles.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Button

from Tests.UI.test_console_library_auto_outcome import (
    _StalledLibrary,
    _paused_retrieval,
)
from Tests.UI.test_console_received_intent_feedback import (
    _held_received_record,
    _received_console_case,
    _send,
    _until,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("beside", ["custody", "custody-and-a-read"])
async def test_stop_during_a_received_sends_library_search_pauses_it(
    monkeypatch, beside
):
    """Stop ends the search, not the send that is still in intake custody.

    ``custody-and-a-read`` adds what the live run also had in flight: a finite
    preparation read of the same session (live, the transcript's
    ``read_context_presentation_inputs``, slowed by the cold build). Stop
    cancelled that read's task and returned before the Library search.
    """
    async with _received_console_case(
        monkeypatch, f"library-stop-{beside}", durable=True
    ) as case:
        library = _StalledLibrary()
        case.host.app_instance.library_rag_search_service = library
        case.store.stage_session_library_policy(
            case.session.id,
            ConsoleLibraryPolicyCandidate(
                auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            ),
        )
        # Long enough that only Stop, never the budget, can end the wait.
        case.controller._library_preparation_timeout = 120.0
        _send(case, "enter")
        record = await _held_received_record(case)
        case.probe.release.set()
        assert await _until(lambda: library.calls == 1, 20)
        # The live precondition: the send is still in intake custody.
        assert record.received_intent is not None
        assert record.task is not None and not record.task.done()
        read_released = threading.Event()
        read = None
        if beside == "custody-and-a-read":
            read = asyncio.ensure_future(
                run_preparation_read(
                    lambda: read_released.wait(30),
                    creator=case.controller,
                    session_id=case.session.id,
                    reads=case.controller._preparation_reads,
                    require_current=lambda: None,
                )
            )
            assert await _until(lambda: case.controller._preparation_reads, 5)
        stop = case.console.query_one("#console-stop-generation", Button)
        assert await _until(
            lambda: stop.styles.display == "block" and not stop.disabled, 10
        )

        try:
            stop.press()
            paused = await _until(lambda: _paused_retrieval(case.controller), 15)
        finally:
            read_released.set()
        assert paused
        if read is not None:
            # The read was not Stop's to cancel: it finishes on its own.
            assert await asyncio.wait_for(read, 10) is True

        preparation = case.store.preparation_for_session(case.session.id)
        outcome = case.controller._preparation_outcomes[preparation.preparation_id]
        assert outcome.error_code == "library_retrieval_stopped"
        assert case.provider_calls == []
        library.release.set()
