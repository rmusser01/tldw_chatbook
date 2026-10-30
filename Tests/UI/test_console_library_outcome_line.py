"""Each Automatic Library send shows what its search did, on its own turn.

TASK-33621.51 (review finding G2-09). With Library on Automatic, nothing on a
turn said what retrieval did for it: a grounded turn looked the same as one
sent with 0 matches, or sent once without Library after a timeout, a failure
or a Stop.

These tests drive the real ChatScreen, controller, store, runtime and
transcript over a real ChaChaNotes database. Only the provider and the
Library search service are doubles, and nothing stubs the projection of the
persisted ``library_preparation`` record onto the transcript: a reload opens
the saved conversation in a new session, so the line can only come back from
that record.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.test_console_library_auto_outcome import (
    FIRST,
    REPLY,
    _automatic_console,
    _build,
    _paused_retrieval,
    _StalledLibrary,
)
from Tests.UI.test_console_send_acknowledgement import (
    _painted_lines,
    eager_tasks,
    press,
    until,
)
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.library_preparation import LIBRARY_PREPARATION_EVENT_KIND
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow

pytestmark = pytest.mark.bootstrap_profile

ZERO_LINE = "Library: 0 matches · sent without Library evidence"


class _Library:
    """A Library search that answers at once with fixed rows."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0

    async def search(self, query, source_types, mode, **kwargs):
        self.calls += 1
        return {
            "runtime_backend": "local",
            "results": [
                LibraryRagResultRow.from_result(
                    {
                        "source_id": f"note-{index}",
                        "chunk_id": f"chunk-{index}",
                        "title": f"Note {index}",
                        "content": f"evidence {index}",
                        "score": 0.9,
                        "runtime_backend": "local",
                        "source_type": "notes",
                    }
                )
                for index in range(1, self.count + 1)
            ],
        }


def _text(host) -> str:
    return "\n".join(_painted_lines(host))


def _line_sits_under_the_user_turn(host, line: str) -> bool:
    """The outcome line is painted below the sent message and above its reply."""
    lines = _painted_lines(host)
    found = [index for index, text in enumerate(lines) if line in text]
    sent = [index for index, text in enumerate(lines) if FIRST in text]
    reply = [index for index, text in enumerate(lines) if REPLY in text]
    return (
        len(found) == 1
        and bool(sent)
        and bool(reply)
        and sent[0] < found[0] < reply[-1]
    )


async def _send_first(host, pilot, composer, gateway) -> None:
    composer.load_draft(FIRST)
    await pilot.pause()
    press(host, "enter", "\r")
    await until(lambda: gateway.stream_calls == 1)
    await until(lambda: REPLY in _text(host))


@pytest.mark.asyncio
async def test_a_zero_match_send_shows_its_line_and_keeps_it_after_a_reload():
    """AC#3: the zero-match line renders from the persisted record, live and reloaded."""
    host, gateway = _build()
    host.app_instance.library_rag_search_service = _Library(0)
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _automatic_console(host, pilot)
            await _send_first(host, pilot, composer, gateway)

            # Live: the line appears on the turn that ran the search.
            await until(lambda: _line_sits_under_the_user_turn(host, ZERO_LINE))

            store = controller.store
            sent_session_id = store.active_session_id
            session = next(
                item for item in store.sessions() if item.id == sent_session_id
            )
            conversation_id = session.persisted_conversation_id
            database = host.app_instance.chachanotes_db
            assert [
                row.event_kind
                for row in database.get_trajectory_rows(conversation_id)
                if row.event_kind == LIBRARY_PREPARATION_EVENT_KIND
            ] == [LIBRARY_PREPARATION_EVENT_KIND]

            # Reload: open the saved conversation as a NEW session, so nothing
            # carried in memory from the send can paint the line.
            host.app_instance.chat_conversation_scope_service = (
                ChatConversationScopeService(
                    local_service=ChatConversationService(database),
                    server_service=None,
                )
            )
            reopened = await console._workspace._resume_console_workspace_conversation(
                conversation_id, reuse_existing=False
            )
            assert reopened is True
            assert store.active_session_id != sent_session_id
            await pilot.pause()
            await until(lambda: _line_sits_under_the_user_turn(host, ZERO_LINE))

            # Repaint: a forced full transcript rebuild keeps it.
            await console._sync_native_console_transcript(force=True)
            await pilot.pause()
            assert _line_sits_under_the_user_turn(host, ZERO_LINE)

            # Negative control: without the persisted record, a reload has no
            # line to show -- the record is its only source.
            with database.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM message_trajectory_metadata "
                    "WHERE conversation_id = ? AND event_kind = ?",
                    (conversation_id, LIBRARY_PREPARATION_EVENT_KIND),
                )
            reloaded_session_id = store.active_session_id
            assert (
                await console._workspace._resume_console_workspace_conversation(
                    conversation_id, reuse_existing=False
                )
                is True
            )
            assert store.active_session_id not in {
                sent_session_id,
                reloaded_session_id,
            }
            await console._sync_native_console_transcript(force=True)
            await pilot.pause()
            await until(lambda: FIRST in _text(host) and REPLY in _text(host))
            assert ZERO_LINE not in _text(host)


@pytest.mark.asyncio
async def test_a_grounded_send_says_how_many_sources_it_sent():
    host, gateway = _build()
    library = _Library(2)
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            _console, _controller, composer = await _automatic_console(host, pilot)
            await _send_first(host, pilot, composer, gateway)

            await until(
                lambda: _line_sits_under_the_user_turn(host, "Library: 2 sources sent")
            )
            assert library.calls == 1
            assert ZERO_LINE not in _text(host)


@pytest.mark.asyncio
async def test_a_send_once_without_library_says_why_the_search_paused():
    host, gateway = _build()
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _automatic_console(host, pilot)
            controller._library_preparation_timeout = 0.2
            composer.load_draft(FIRST)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: _paused_retrieval(controller))
            button = console.query_one("#console-trace-send-without-library", Button)
            await until(lambda: bool(button.display))

            button.press()
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in _text(host))

            await until(
                lambda: _line_sits_under_the_user_turn(
                    host, "Library: sent once without Library · search timed out"
                )
            )
            library.release.set()
            await asyncio.sleep(0)
