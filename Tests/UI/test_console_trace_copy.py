"""Source-pinned trace-copy policy and mounted Console journeys."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
import threading
from time import perf_counter

import pytest
from textual.app import ComposeResult

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Console_Modules.review_selection import (
    ConsoleReviewSelectionController,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

pytestmark = pytest.mark.bootstrap_profile


class TraceStore:
    def __init__(self, messages):
        self.messages = {
            "source": messages,
            "other": [
                ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "OTHER CHAT SENTINEL")
            ],
        }
        self.active_session_id = "source"

    def read_only_messages_for_session(self, session_id):
        return deepcopy(self.messages[session_id])

    def sessions(self):
        return [SimpleNamespace(id=key, title=key) for key in self.messages]


def trace_controller(store, **collaborators):
    jobs, clipboard, notices = [], [], []

    async def noop(*_args):
        return None

    def schedule(work, **_kwargs):
        jobs.append(asyncio.create_task(work))

    options = {"copy_to_clipboard": clipboard.append, **collaborators}
    controller = ConsoleReviewSelectionController(
        store_accessor=lambda: store,
        agent_conversation_id_accessor=lambda: None,
        change_review_provider_accessor=lambda _id: None,
        run_active_accessor=lambda: False,
        run_active_for_root=lambda _root: False,
        workspace_roots_accessor=lambda: None,
        agent_runs_db_accessor=lambda: None,
        capture_policy_bindings_accessor=lambda *_args: None,
        native_messages_accessor=lambda: [],
        run_worker=schedule,
        show_feedback_comment=noop,
        dispatch_prompt=noop,
        marshal_to_ui=lambda *_args: None,
        present_trajectory=lambda *_args: None,
        notify=lambda message, **_kwargs: notices.append(message),
        **options,
    )
    return controller, jobs, clipboard, notices


@pytest.mark.asyncio
async def test_copy_request_keeps_original_session_after_tab_switch():
    message = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "original answer", id="answer"
    )
    store = TraceStore([message])
    controller, jobs, clipboard, _ = trace_controller(store)
    controller.request_trace_copy("source", message_id="answer")
    message.content = "later changed source"
    store.active_session_id = "other"
    await asyncio.gather(*jobs)
    assert len(clipboard) == 1
    assert "original answer" in clipboard[0]
    assert "later changed" not in clipboard[0] and "OTHER CHAT" not in clipboard[0]


@pytest.mark.asyncio
async def test_copy_missing_response_preserves_clipboard():
    controller, jobs, clipboard, notices = trace_controller(TraceStore([]))
    clipboard.append("previous")
    controller.request_trace_copy("source", message_id="removed")
    if jobs:
        await asyncio.gather(*jobs)
    assert clipboard == ["previous"]
    assert notices


@pytest.mark.asyncio
async def test_chat_copy_includes_unmounted_history():
    messages = [
        ConsoleChatMessage(ConsoleMessageRole.USER, f"question {index}")
        for index in range(250)
    ]
    messages.append(
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "x" * 100_000 + "TAIL")
    )
    controller, jobs, clipboard, notices = trace_controller(TraceStore(messages))
    controller.request_trace_copy("source")
    await asyncio.gather(*jobs)
    assert "question 0" in clipboard[0] and "question 249" in clipboard[0]
    assert "x" * 100_000 + "TAIL" in clipboard[0]
    assert "251" in notices[-1]
    assert "sent" in notices[-1].lower()


@pytest.mark.asyncio
async def test_native_store_snapshot_pins_active_branch_without_blocking_admission():
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    history = [
        ConsoleChatMessage(
            ConsoleMessageRole.USER,
            f"history {index}",
            id=f"h{index}",
            persisted_message_id=f"p{index}",
            parent_message_id=f"p{index - 1}" if index else None,
        )
        for index in range(500)
    ]
    first, second = [
        ConsoleChatMessage(
            ConsoleMessageRole.ASSISTANT,
            text,
            id=identifier,
            persisted_message_id=identifier,
            parent_message_id="p499",
        )
        for identifier, text in (
            ("first", "x" * 100_000 + "TAIL"),
            ("second", "INACTIVE BRANCH"),
        )
    ]
    session = store.restore_persisted_session(
        title="Source",
        workspace_id=None,
        persisted_conversation_id="source",
        all_nodes=[*history, first, second],
        active_leaf_persisted_id="first",
    )
    controller, jobs, clipboard, _ = trace_controller(store)
    started = perf_counter()
    controller.request_trace_copy(session.id)
    elapsed = perf_counter() - started
    assert elapsed < 0.1, f"Snapshot admission took {elapsed:.3f}s"
    store.set_active_leaf(session.id, second.id)
    await asyncio.gather(*jobs)
    assert "history 0" in clipboard[0] and "history 499" in clipboard[0]
    assert first.content in clipboard[0] and "INACTIVE BRANCH" not in clipboard[0]


@pytest.mark.asyncio
async def test_late_copy_worker_cannot_overwrite_newer_request(monkeypatch):
    from tldw_chatbook.Chat import console_response_trace

    entered, release = threading.Event(), threading.Event()
    worker_threads = []

    def render(turn):
        worker_threads.append(threading.get_ident())
        if turn.assistant.content == "first":
            entered.set()
            assert release.wait(5)
        return turn.assistant.content

    monkeypatch.setattr(
        console_response_trace, "render_response_trace_markdown", render
    )
    message = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "first", id="answer")
    store = TraceStore([message])
    controller, jobs, clipboard, _ = trace_controller(store)
    controller.request_trace_copy("source", message_id="answer")
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        message.content = "second"
        controller.request_trace_copy("source", message_id="answer")
        await jobs[-1]
        assert clipboard == ["second"]
    finally:
        release.set()
        await asyncio.gather(*jobs)
    assert clipboard == ["second"]
    assert all(thread != threading.get_ident() for thread in worker_threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("later_text", ["new ordinary copy", "initial clipboard"])
async def test_ordinary_copy_supersedes_pending_trace_even_for_same_text(
    monkeypatch, later_text
):
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Chat import console_response_trace

    app = _build_test_app()
    app.copy_to_clipboard("initial clipboard")
    entered, release = threading.Event(), threading.Event()

    def render(_turn):
        entered.set()
        assert release.wait(10)
        return "old pending trace"

    monkeypatch.setattr(
        console_response_trace, "render_response_trace_markdown", render
    )
    controller, jobs, _, _ = trace_controller(
        TraceStore(
            [ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer", id="answer")]
        ),
        copy_to_clipboard=app.copy_to_clipboard,
    )
    controller._clipboard_revision_accessor = lambda: app.clipboard_revision
    controller.request_trace_copy("source", message_id="answer")
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        app.copy_to_clipboard(later_text)
    finally:
        release.set()
        await asyncio.gather(*jobs)
    assert app.clipboard == later_text


class TraceCopyApp(ConsolidatedCSSApp):
    def __init__(self, messages):
        super().__init__()
        self.store = TraceStore(messages)
        self.controller, self.jobs, _, self.notices = trace_controller(self.store)
        self.controller._copy_to_clipboard = self.copy_to_clipboard

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(self.store.messages["source"], session_id="source")
        yield transcript

    def on_console_trace_copy_requested(self, event):
        self.controller.request_trace_copy(
            event.session_id,
            message_id=event.message_id,
            source_message_ids=event.source_message_ids,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", [ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT]
)
@pytest.mark.parametrize("action", ["copy", "create-note", "create-note-go"])
async def test_full_plain_message_preserves_source_without_rendered_chrome(
    role, action
):
    from Tests.UI.test_console_selection_menu import _wait_for_menu
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleTranscriptMessage,
    )

    body = "  # Heading\n**bold** and `code`\nλ last line  "
    message = ConsoleChatMessage(
        role,
        body,
        id="body",
        status="stopped" if role == ConsoleMessageRole.ASSISTANT else "complete",
        attachments=(
            MessageAttachment(
                data=None, mime_type="text/plain", display_name="file.txt", position=0
            ),
        ),
    )

    class SourceBodyApp(TraceCopyApp):
        def __init__(self):
            super().__init__([message])
            self.app_config = {"chat_defaults": {"assistant_markdown": False}}
            self.notes = []

        def on_console_selection_note_requested(self, event):
            self.notes.append((event.quote, event.open_note))

    app = SourceBodyApp()
    async with app.run_test(size=(100, 40)) as pilot:
        _, row = await open_trace_menu(app, pilot, "body")
        assert isinstance(row, ConsoleTranscriptMessage)
        await _wait_for_menu(
            app,
            pilot,
            lambda menu: menu.region.contains_region(
                menu.query_one("#console-selection-full-message").region
            )
            and menu.query_one("#console-selection-full-message").region.height > 0,
        )
        assert await pilot.click("#console-selection-full-message")
        await pilot.pause()
        assert row.get_selection_text(capped=False) == body
        await pilot.click(f"#console-selection-{action}")
        await pilot.pause()
        if action == "copy":
            assert app.clipboard == body
        else:
            assert app.notes == [(body, action == "create-note-go")]


def test_ordinary_drag_after_full_message_keeps_its_rendered_text_domain():
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleTranscriptMessage,
    )

    body = "# Heading\n**bold** and `code`"
    row = ConsoleTranscriptMessage(
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, body)
    )
    end = len(row.get_display_text())
    row.set_selection_range(0, end, full_body=True)
    assert row.get_selection_text(capped=False) == body
    row.set_selection_range(0, end)
    assert row.get_selection_text(capped=False) == row.get_display_text()
    row.clear_selection()
    assert row.get_selection_text(capped=False) == ""


async def open_trace_menu(app, pilot, message_id):
    transcript = app.query_one(ConsoleTranscript)
    row = app.query_one(f"#console-message-{message_id}")
    transcript.selection_manager.begin_drag(row.id, 0)
    transcript.selection_manager.extend_drag(row.id, 3)
    row.set_selection_range(0, 3)
    selection = transcript.selection_manager.finish_drag()
    transcript.post_message(ConsoleTranscript.TranscriptTextSelected(selection, 2, 2))
    await pilot.pause()
    return transcript, row


@pytest.mark.asyncio
async def test_full_message_selection_then_copy_preserves_offscreen_text():
    body = "long message line\n" * 1000 + "END OF MESSAGE"
    message = ConsoleChatMessage(ConsoleMessageRole.USER, body, id="user")
    app = TraceCopyApp([message])
    async with app.run_test(size=(100, 32)) as pilot:
        transcript, row = await open_trace_menu(app, pilot, "user")
        assert row.get_selection_text(capped=False) != body
        await pilot.click("#console-selection-full-message")
        await pilot.pause()
        assert row.get_selection_text(capped=False) == body
        assert app.query_one(ConsoleSelectionMenu)
        assert app.focused.id == "console-selection-copy"
        await pilot.press("enter")
        await pilot.pause()
        assert app.clipboard == body
        assert not app.query(ConsoleSelectionMenu)


@pytest.mark.asyncio
async def test_collapsed_response_and_chat_copy_use_the_same_complete_trace():
    from Tests.Chat.test_console_response_trace import thinking, tool
    from tldw_chatbook.Chat.thinking_blocks import ThinkingEnvelope

    answer = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "the final answer", id="answer"
    )
    answer.thinking = ThinkingEnvelope((thinking("retained thinking"),))
    messages = [
        ConsoleChatMessage(ConsoleMessageRole.USER, "question", id="user"),
        answer,
        tool("read", "full result" * 1000),
    ]
    app = TraceCopyApp(messages)
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, _ = await open_trace_menu(app, pilot, "answer")
        await pilot.click("#console-selection-copy-all")
        await asyncio.gather(*app.jobs)
        await pilot.pause()
        response = app.clipboard
        assert "retained thinking" in response and "full result" * 1000 in response
        assert "question" not in response
        transcript.toggle_tool_output("read")
        await pilot.pause()
        await open_trace_menu(app, pilot, "answer")
        await pilot.click("#console-selection-copy-all")
        await asyncio.gather(*app.jobs)
        await pilot.pause()
        assert app.clipboard == response
        await open_trace_menu(app, pilot, "user")
        assert not app.query("#console-selection-copy-all")
        await pilot.click("#console-selection-copy-all-messages")
        await asyncio.gather(*app.jobs)
        await pilot.pause()
        assert response in app.clipboard and "question" in app.clipboard


@pytest.mark.asyncio
async def test_stale_menu_cannot_copy_another_chat():
    answer = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "source answer", id="answer"
    )
    app = TraceCopyApp([answer])
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, _ = await open_trace_menu(app, pilot, "answer")
        old_menu = app.query_one(ConsoleSelectionMenu)
        app.copy_to_clipboard("previous")
        transcript.set_messages(app.store.messages["other"], session_id="other")
        await pilot.pause()
        # A late message from a removed menu must fail its captured-session gate.
        await app.screen.query(ConsoleSelectionMenu).remove()
        old_menu._copy_all()
        await pilot.pause()
        assert app.clipboard == "previous"


@pytest.mark.asyncio
async def test_stale_chat_copy_menu_refuses_replacement_branch():
    original = [
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "original", id="first")
    ]
    replacement = [
        ConsoleChatMessage(
            ConsoleMessageRole.ASSISTANT, "PRIVATE OTHER BRANCH", id="second"
        )
    ]
    app = TraceCopyApp(original)
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, _ = await open_trace_menu(app, pilot, "first")
        old_menu = app.query_one(ConsoleSelectionMenu)
        app.copy_to_clipboard("previous")
        app.store.messages["source"] = replacement
        transcript.set_messages(replacement, session_id="source")
        await pilot.pause()
        old_menu._copy_all_messages()
        await pilot.pause()
        await asyncio.gather(*app.jobs)
        assert app.clipboard == "previous"
        assert any("branch" in notice.lower() for notice in app.notices)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["complete", "streaming", "stopped", "failed"])
async def test_selected_response_more_menu_copies_readonly_trace(status):
    message = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT, "retained response", id="answer", status=status
    )
    app = TraceCopyApp([message])
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.select_message("answer")
        await pilot.pause()
        await pilot.click("#console-message-action-more-answer")
        await pilot.pause()
        await pilot.click("#console-message-more-copy-all")
        await pilot.pause()
        await asyncio.gather(*app.jobs)
        assert "retained response" in app.clipboard
        assert status in app.clipboard


@pytest.mark.asyncio
async def test_expanded_selection_menu_scrolls_and_restores_after_resize():
    from Tests.UI.test_console_selection_menu import _wait_for_menu

    app = TraceCopyApp(
        [ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer", id="answer")]
    )
    async with app.run_test(size=(80, 9)) as pilot:
        transcript, _ = await open_trace_menu(app, pilot, "answer")
        menu = await _wait_for_menu(
            app,
            pilot,
            lambda candidate: (
                candidate.has_class("shrunk-for-short-owner")
                and transcript.region.contains_region(candidate.region)
            ),
        )
        await pilot.press("up")
        await pilot.wait_for_scheduled_animations()
        assert transcript.region.contains_region(app.focused.region)
        assert transcript.region.contains_region(menu.region)
        await pilot.resize_terminal(100, 40)
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert not menu.has_class("shrunk-for-short-owner")
        assert transcript.region.contains_region(app.focused.region)


@pytest.mark.asyncio
async def test_more_menu_keeps_last_action_reachable_in_short_terminal():
    from tldw_chatbook.Widgets.Console.console_message_more_menu import (
        ConsoleMessageMoreMenu,
    )

    app = TraceCopyApp(
        [ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer", id="answer")]
    )
    chosen = []
    async with app.run_test(size=(80, 8)) as pilot:
        transcript = app.query_one(ConsoleTranscript)

        async def choose(message_id, action_id, *, opener_button_id):
            chosen.append((message_id, action_id))
            await transcript.dismiss_message_more_menu()

        transcript.choose_captured_message_more_action = choose
        transcript.select_message("answer")
        await pilot.pause()
        # Activate the focused action's framework path independently of the
        # primary action row's horizontal layout; this tests the popup viewport.
        transcript.query_one("#console-message-action-more-answer").press()
        await pilot.pause()
        menu = app.query_one(ConsoleMessageMoreMenu)
        last_action = menu.action_ids[-1]
        assert len(menu.action_ids) >= 8
        await pilot.resize_terminal(80, 8)
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert transcript.region.contains_region(menu.region)
        await pilot.press("up")
        await pilot.wait_for_scheduled_animations()
        assert app.focused.id == f"console-message-more-{last_action}"
        assert menu.region.contains_region(app.focused.region)
        await pilot.resize_terminal(160, 40)
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert all(
            menu.region.contains_region(button.region)
            for button in menu.query("Button")
        )
        await pilot.resize_terminal(80, 8)
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert menu.region.contains_region(app.focused.region)
        await pilot.click(app.focused)
        await pilot.pause()
        assert chosen == [("answer", last_action)]
