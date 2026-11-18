"""Selected presenter calls the real mounted transcript, not just its ingest.

The shell/message-controller are presentation fixtures. The actual client is
inert; this test makes no backend, source-authentication or retirement claim.
The codec, transcript methods, row reconciliation and Textual mounts are real.
"""

import asyncio
import sys
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["current", "visit", "projection", "cancel"])
async def test_selected_projection_reconciles_original_rows(request, monkeypatch, outcome):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage, ConsoleMessageRole, ConsoleRunState,
    )
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleProjection
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_execution_projection_values import (
        decode_message_projection, encode_message_projection,
    )
    from tldw_chatbook.UI.Console_Modules import execution_projection as presenter
    from tldw_chatbook.UI.Console_Modules.send_acknowledgement import acknowledgement_for
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    class Host(ConsolidatedCSSApp):
        CSS_PATH = list(APP_STYLESHEETS)

        def compose(self):
            yield ConsoleTranscript(id="console-native-transcript")

    app = Host()
    client = ConsoleExecutionClient(app)
    user = ConsoleChatMessage(id="sent", role=ConsoleMessageRole.USER, content="Question")
    final = ConsoleChatMessage(
        id="final", role=ConsoleMessageRole.ASSISTANT, content="Completed answer", status="complete",
    )

    def publication(messages, revision):
        return ConsoleProjection(
            1, "session", "incarnation", revision,
            tuple(decode_message_projection(encode_message_projection(row)) for row in messages),
            ConsoleRunState(), None,
        )

    async with app.run_test(size=(100, 30)) as pilot:
        screen = app.screen
        screen.app_instance = app
        screen._pending_console_swipe_selection = None
        screen._show_model_thinking = MethodType(lambda self: False, screen)
        screen._message = SimpleNamespace(_screen=screen, app_instance=app)
        # Isolate the shell adaptation; leave the actual transcript method
        # definitions and their receiver checks intact.
        monkeypatch.setattr(presenter, "ChatScreen", type(screen))
        monkeypatch.setattr(presenter, "ConsoleMessageController", SimpleNamespace)
        monkeypatch.setattr(presenter, "_require_sources", lambda: None)
        monkeypatch.setattr(presenter, "_TORN_DOWN", (lambda _: False,))
        monkeypatch.setattr(presenter, "_SCREEN_METHODS", (
            ("query_one", presenter._definition(type(screen).query_one)),
        ))
        client._view, client._visit, client._source_revision = screen, 1, 1
        transcript = screen.query_one(ConsoleTranscript)
        first = client._projection = publication([user], 1)
        assert await presenter.apply_console_process_projection(screen, client, first, visit=1)
        await pilot.pause()
        assert transcript.query_one("#console-message-sent").is_attached
        assert not transcript.query("#console-message-final")

        second = client._projection = publication([user, final], 2)
        assert second.messages[-1].role is ConsoleMessageRole.ASSISTANT
        transcript.selected_message_id = final.id
        refreshes = []

        def refresh_entered(frame):
            if frame.f_locals.get("self") is not transcript:
                return
            refreshes.append(True)
            if outcome == "visit":
                client._visit = 2
            elif outcome == "projection":
                client._projection = replace(second, display_revision=3)
            elif outcome == "cancel":
                raise asyncio.CancelledError

        with _observe(ConsoleTranscript.refresh_messages.__code__, sys.monitoring.events.PY_START, refresh_entered):
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await presenter.apply_console_process_projection(screen, client, second, visit=1)
            else:
                assert await presenter.apply_console_process_projection(screen, client, second, visit=1) is (outcome == "current")
            assert refreshes == [True]
            if outcome != "current":
                assert screen._console_process_transcript_rendered is None
                return
            assert final.id in transcript.mounted_message_content_ids()
            assert transcript.query_one("#console-message-final").is_attached

            # The same values in a new checked publication must not dismiss an
            # actual open message menu or repeat original row reconciliation.
            opener = transcript.query_one("#console-message-action-more-final", Button)
            await transcript._open_message_more_menu(final.id, opener)
            await pilot.pause()
            menu = screen.query_one("#console-message-more-menu")
            before = len(refreshes)
            third = client._projection = publication([user, final], 3)
            assert third.messages[-1] is not second.messages[-1]
            assert await presenter.apply_console_process_projection(screen, client, third, visit=1)
            assert len(refreshes) == before
            assert screen.query_one("#console-message-more-menu") is menu
            assert menu.is_attached

            ack = acknowledgement_for(screen)
            token = ack.begin("session", "Next question")
            assert token is not None
            ack.hold_row("session")
            assert await presenter.apply_console_process_projection(screen, client, third, visit=1)
            assert len(refreshes) == before and menu.is_attached
            assert ack.show_row("session")
            assert await presenter.apply_console_process_projection(screen, client, third, visit=1)
            assert len(refreshes) == before + 1
            pending_id = ack.pending_row_id("session")
            assert transcript.query_one(f"#console-message-{pending_id}").is_attached

            screen._console_chat_store = SimpleNamespace(messages_for_session=lambda _: third.messages)
            with ack.dispatching(token):
                assert ack.custody_callback("session") is not None
                ack.mark_admitted("session")
            echo = ConsoleChatMessage(id="next-echo", role=ConsoleMessageRole.USER, content="Next question")
            fourth = client._projection = publication([user, final, echo], 4)
            assert await presenter.apply_console_process_projection(screen, client, fourth, visit=1)
            assert len(refreshes) == before + 1  # Original stand-in preserves the same rows.
            assert transcript.query_one(f"#console-message-{pending_id}").is_attached
