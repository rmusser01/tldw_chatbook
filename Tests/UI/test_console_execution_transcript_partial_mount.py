"""Actual transcript retries after bookkeeping precedes interrupted DOM work.

Only the selected shell/client qualification is a fixture, as in the adjacent
selected-render test. The row constructors, reconciler, Textual composition,
mount/removal, Markdown children and menu are original. No backend or native
retirement claim is made. These regressions are authored before the repair.
"""

import asyncio
import sys
from types import CodeType, MethodType, SimpleNamespace

import pytest
from textual.widgets import Button, Markdown

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["before_row_mount", "missing_answer_children"])
async def test_selected_retry_requires_actual_composed_content(
    request, monkeypatch, interruption,
):
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
    from tldw_chatbook.UI.Navigation.surface_swap_guard import finish_removal
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    class Host(ConsolidatedCSSApp):
        CSS_PATH = list(APP_STYLESHEETS)

        def __init__(self):
            super().__init__()
            self.app_config = {"chat_defaults": {"assistant_markdown": True}}

        def compose(self):
            yield ConsoleTranscript(id="console-native-transcript")

    app = Host()
    client = ConsoleExecutionClient(app)
    earlier = [
        ConsoleChatMessage(
            id=f"earlier-{index}",
            role=ConsoleMessageRole.USER if index % 2 == 0 else ConsoleMessageRole.ASSISTANT,
            content=f"Earlier message {index}", status="complete",
        )
        for index in range(5)
    ]
    final = ConsoleChatMessage(
        id="final", role=ConsoleMessageRole.ASSISTANT,
        content="```python\nprint('recovered')\n```", status="complete",
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
        monkeypatch.setattr(presenter, "ChatScreen", type(screen))
        monkeypatch.setattr(presenter, "ConsoleMessageController", SimpleNamespace)
        monkeypatch.setattr(presenter, "_require_sources", lambda: None)
        monkeypatch.setattr(presenter, "_TORN_DOWN", (lambda _: False,))
        monkeypatch.setattr(presenter, "_SCREEN_METHODS", (
            ("query_one", presenter._definition(type(screen).query_one)),
        ))
        client._view, client._visit, client._source_revision = screen, 1, 1
        transcript = screen.query_one(ConsoleTranscript)
        first = client._projection = publication(earlier, 1)
        assert await presenter.apply_console_process_projection(screen, client, first, visit=1)
        await pilot.pause()
        assert transcript.mounted_message_content_ids() == frozenset(row.id for row in earlier)

        second = client._projection = publication([*earlier, final], 2)
        transcript.selected_message_id = final.id
        observed = []
        [interruption_code] = [
            code for code in ConsoleTranscript._reconcile_rows.__code__.co_consts
            if type(code) is CodeType and code.co_name == "_mount_pending"
        ]

        def interrupt_original_mount(frame):
            if observed:
                return
            owner = transcript._row_widgets.get("assistant-turn:final")
            if owner is None:
                return
            if frame.f_locals.get("self") is not transcript:
                return
            if owner not in frame.f_locals.get("pending_widgets", ()):
                return
            assert not owner.is_attached
            # Original bookkeeping already exists; do not alter the widget map,
            # signature or any production method to manufacture the condition.
            assert "assistant-turn:final" in transcript._row_signatures
            observed.append(owner)
            raise asyncio.CancelledError

        if interruption == "before_row_mount":
            with _observe(interruption_code, sys.monitoring.events.PY_START, interrupt_original_mount):
                with pytest.raises(asyncio.CancelledError):
                    await presenter.apply_console_process_projection(screen, client, second, visit=1)
            assert getattr(screen, "_console_process_transcript_rendered", None) is None
        else:
            assert await presenter.apply_console_process_projection(screen, client, second, visit=1)
            await pilot.pause()
            owner = transcript._row_widgets["assistant-turn:final"]
            assert owner.answer_widget.query_one(Markdown).is_attached
            # Use original completed DOM removal to reproduce the missing body
            # with a live wrapper. Killing Textual's own composition pump also
            # proved the false render credit, but poisons unrelated app teardown.
            await finish_removal(owner.answer_widget.remove_children())
            assert owner.answer_widget.is_attached and not owner.answer_widget.children
            observed.append(owner)
        assert len(observed) == 1
        assert final.id not in transcript.mounted_message_content_ids()

        # Retry identical values using the original reconciler. A stale map or
        # signature must neither suppress the real mount nor cause move_child
        # to target an unattached widget.
        third = client._projection = publication([*earlier, final], 3)
        assert await presenter.apply_console_process_projection(screen, client, third, visit=1)
        await pilot.pause()
        owner = transcript._row_widgets["assistant-turn:final"]
        answer = transcript.query_one("#console-message-final")
        assert answer is owner.answer_widget and answer.is_attached
        markdown = answer.query_one(Markdown)
        assert markdown.parent is answer and markdown.is_attached
        assert final.id in transcript.mounted_message_content_ids()
        assert len(transcript.query("#console-message-final")) == 1
        assert answer.get_display_text() == final.content

        opener = transcript.query_one("#console-message-action-more-final", Button)
        await transcript._open_message_more_menu(final.id, opener)
        await pilot.pause()
        menu = screen.query_one("#console-message-more-menu")
        fourth = client._projection = publication([*earlier, final], 4)
        assert await presenter.apply_console_process_projection(screen, client, fourth, visit=1)
        assert screen.query_one("#console-message-more-menu") is menu and menu.is_attached

        # A deliberately bounded original window is valid. Recovery may not
        # equate every omitted message with broken content and remount history.
        await transcript.dismiss_message_more_menu(restore_focus=False)
        transcript.selected_message_id = None
        transcript.release_anchor()
        # Keep the original bounded-window latch engaged while this assertion
        # inspects the window, rather than asking viewport hydration to expand it.
        transcript._suppress_boundary_hydration = True
        transcript._set_hidden_prefix(2)
        transcript._hide_tail_from(4)
        await transcript.refresh_messages()
        hidden = (set(transcript._pruned_message_ids), set(transcript._hidden_tail_ids))
        assert hidden[0] and final.id in hidden[1]
        fifth = client._projection = publication([*earlier, final], 5)
        assert await presenter.apply_console_process_projection(screen, client, fifth, visit=1)
        assert hidden == (set(transcript._pruned_message_ids), set(transcript._hidden_tail_ids))
        assert transcript.mounted_message_content_ids() == frozenset({"earlier-2", "earlier-3"})
