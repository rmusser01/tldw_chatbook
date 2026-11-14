"""A real composer send's Library pause: Stop pauses it, the shelf releases it.

TASK-33621.20, found live on the head rebased onto dev ce4aa4a231. A real
composer send takes dev's received-intent path, which the send-harness tests
never enter. Two of this task's behaviours broke there:

1. The send stays in the runtime's intake custody (its received intent) for
   its whole preparation, Library search included. Dev's 2026-10-15 Stop path
   cancels intake custody before it reaches the Library search: Stop during
   "Searching Library..." cancelled the whole send (``controller_submit
   status=cancelled``), so no card offered Retry / Send once without Library
   / Cancel.
2. The composer keeps showing the sent text as the draft until the turn is
   accepted, so after a pause the shelf's Restore met a non-empty draft and
   refused it as an ambiguous merge ("That unsent turn could not be restored
   safely."): the paused send was never released.

This drives the stock received-intent path (real ChatScreen, runtime,
controller, store and file-backed database). Only the provider call and the
Library search service are doubles.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import zlib

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_automatic_library_preparation import _pending_image
from Tests.UI.test_console_library_auto_outcome import (
    _StalledLibrary,
    _paused_retrieval,
    _shelf_offers_unsent_turn,
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
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _one_pixel_png() -> bytes:
    """A decodable PNG, so the staged image's preview does real work only."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x80\xff"))
        + chunk(b"IEND", b"")
    )


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
        # Settle before the harness's teardown Stop: it cancels the issuing
        # task of any finite read still in flight (a repaint's, here).
        assert await _until(lambda: not case.controller._preparation_reads, 20)


@pytest.mark.parametrize("shelf_action", ["Discard", "Restore"])
async def test_the_shelf_releases_a_received_sends_library_pause(
    monkeypatch, shelf_action
):
    """AC#2 on the stock received-intent path: Restore and Discard release it.

    On this path the paused send's own text is still the conversation's draft
    (the composer keeps showing a sent draft until the turn is accepted).
    Restore refused that as an ambiguous merge ("That unsent turn could not
    be restored safely."), so the paused send stayed, and with it the card.
    """
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *a, **k: None)
    monkeypatch.setattr(controller_module, "vision_block_reason", lambda *a, **k: None)
    async with _received_console_case(
        monkeypatch, f"library-shelf-{shelf_action}", durable=True
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
        image = _pending_image("diagram.png", _one_pixel_png())
        assert case.store.add_pending_attachment(case.session.id, image)
        case.controller._library_preparation_timeout = 120.0
        _send(case, "enter")
        await _held_received_record(case)
        case.probe.release.set()
        assert await _until(lambda: library.calls == 1, 20)
        stop = case.console.query_one("#console-stop-generation", Button)
        assert await _until(
            lambda: stop.styles.display == "block" and not stop.disabled, 10
        )
        stop.press()
        assert await _until(lambda: _paused_retrieval(case.controller), 15)
        assert await _until(lambda: _shelf_offers_unsent_turn(case.console), 20)
        # The precondition on this path: the paused send's text is the draft.
        assert case.store.session_draft(case.session.id) == case.draft
        assert case.store.pending_attachments(case.session.id) == []

        button_id = (
            "#console-prompt-queue-pause"
            if shelf_action == "Discard"
            else "#console-prompt-queue-manage"
        )
        assert str(case.console.query_one(button_id).label) == shelf_action
        case.console.query_one(button_id, Button).press()

        assert await _until(lambda: not _paused_retrieval(case.controller), 20)
        assert await _until(lambda: not _shelf_offers_unsent_turn(case.console), 20)
        assert case.store.session_draft(case.session.id) == case.draft
        if shelf_action == "Restore":
            pending = case.store.pending_attachments(case.session.id)
            assert [item.display_name for item in pending] == ["diagram.png"]
        # The released send leaves the transcript. Waiting for its row to go
        # also keeps teardown clear of that removal: before dev's TASK-34982
        # shield, the harness's shutdown drain could cancel it mid-way and
        # fail with CancelledError (about one run in three on ce4aa4a231).
        transcript = case.console.query_one(ConsoleTranscript)
        assert await _until(
            lambda: (
                len(transcript.query(ConsoleTranscriptMessage)) == 0
                and not transcript._refresh_lock.locked()
            ),
            20,
        )
        library.release.set()
        # Settle before the harness's teardown Stop: it cancels the issuing
        # task of any finite read still in flight (a repaint's, here).
        assert await _until(lambda: not case.controller._preparation_reads, 20)


FOLLOW_UP = "A follow-up typed while the card was up"


@pytest.mark.parametrize("composer_shows", ["the sent text", "a follow-up"])
async def test_the_cards_cancel_takes_a_received_sends_turn_off_the_shelf(
    monkeypatch, composer_shows
):
    """Cancel send puts the turn back in the composer and off the shelf.

    Found live (2026-10-20, head rebased onto dev 2d071d6d71): after the
    card's Cancel send, the shelf still read "Not sent: Library search
    stopped" with Restore / Discard. The composer keeps showing a sent draft
    until the turn is accepted, so it was never empty, and the card's Cancel
    took that for a follow-up to protect: it left the turn on the shelf (with
    its attachments) instead of returning it. Sending the message again then
    left a stale shelf entry whose Restore would queue it a second time.

    ``a follow-up`` is the control: a different draft the user typed is kept,
    and the turn stays on the shelf for Restore.
    """
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *a, **k: None)
    monkeypatch.setattr(controller_module, "vision_block_reason", lambda *a, **k: None)
    async with _received_console_case(
        monkeypatch, f"library-card-cancel-{composer_shows.split()[-1]}", durable=True
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
        image = _pending_image("diagram.png", _one_pixel_png())
        assert case.store.add_pending_attachment(case.session.id, image)
        case.controller._library_preparation_timeout = 120.0
        _send(case, "enter")
        await _held_received_record(case)
        case.probe.release.set()
        assert await _until(lambda: library.calls == 1, 20)
        stop = case.console.query_one("#console-stop-generation", Button)
        assert await _until(
            lambda: stop.styles.display == "block" and not stop.disabled, 10
        )
        stop.press()
        assert await _until(lambda: _paused_retrieval(case.controller), 15)
        assert await _until(lambda: _shelf_offers_unsent_turn(case.console), 20)
        paused = case.store.preparation_for_session(case.session.id)
        cancel = case.console.query_one("#console-trace-cancel", Button)
        assert await _until(lambda: cancel.display and not cancel.disabled, 20)
        # The precondition on this path: the composer still shows the sent text.
        assert case.composer.draft_text() == case.draft
        assert case.store.pending_attachments(case.session.id) == []
        expected_draft = case.draft
        if composer_shows == "a follow-up":
            case.composer.load_draft(FOLLOW_UP)
            expected_draft = FOLLOW_UP

        cancel.press()

        assert await _until(lambda: not _paused_retrieval(case.controller), 20)

        def on_shelf() -> bool:
            return any(
                entry.preparation_id == paused.preparation_id
                for entry in case.runtime._turn_recoveries.values()
            )

        if composer_shows == "the sent text":
            assert await _until(lambda: not on_shelf(), 10), (
                "the cancelled send is still on the unsent-turn shelf"
            )
            assert await _until(lambda: not _shelf_offers_unsent_turn(case.console), 20)
            pending = case.store.pending_attachments(case.session.id)
            assert [item.display_name for item in pending] == ["diagram.png"]
        else:
            # The turn (and its attachment) waits on the shelf for Restore.
            assert await _until(lambda: _shelf_offers_unsent_turn(case.console), 20)
            assert on_shelf()
            assert case.store.pending_attachments(case.session.id) == []
        assert case.composer.draft_text() == expected_draft
        assert case.provider_calls == []
        transcript = case.console.query_one(ConsoleTranscript)
        assert await _until(
            lambda: (
                len(transcript.query(ConsoleTranscriptMessage)) == 0
                and not transcript._refresh_lock.locked()
            ),
            20,
        )
        library.release.set()
        assert await _until(lambda: not case.controller._preparation_reads, 20)
