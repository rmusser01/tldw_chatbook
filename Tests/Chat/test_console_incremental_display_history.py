"""The screen's settled history is reused until its store owner changes."""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.UI.Screens import chat_screen as screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Utils.token_counter import ContextWindowResolution


def test_unchanged_typing_reuses_history_and_late_usage_rebuilds(monkeypatch):
    store = ConsoleChatStore()
    session = store.create_session(title="projection")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="question")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    screen = SimpleNamespace()
    controller = SimpleNamespace(
        run_state_for=lambda _sid: SimpleNamespace(status=ConsoleRunStatus.IDLE),
        _submit_tasks_for_session=lambda _sid: (),
    )
    actual = store.messages_for_session
    reads = []

    def counted(session_id):
        reads.append(session_id)
        return actual(session_id)

    monkeypatch.setattr(store, "messages_for_session", counted)
    first = ChatScreen._console_display_history(screen, store, session.id, controller)
    for _ in range(24):
        assert (
            ChatScreen._console_display_history(screen, store, session.id, controller)
            == first
        )
    assert reads == [session.id]

    store.set_message_usage(
        answer.id,
        ProviderUsage(
            uncached_input=100,
            output=25,
            provider="anthropic",
            model="claude-sonnet-4-6",
        ),
    )
    after = ChatScreen._console_display_history(screen, store, session.id, controller)
    assert len(reads) == 2
    assert after[0] != first[0]
    assert after[1][-1].usage is not None

    store.update_message_content(answer.id, "edited answer")
    edited = ChatScreen._console_display_history(screen, store, session.id, controller)
    assert len(reads) == 3
    assert edited[1][-1].content == "edited answer"


def test_history_projection_refreshes_after_append_stream_and_branch_switch():
    store = ConsoleChatStore()
    session = store.create_session(title="projection mutations")
    question = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    screen = SimpleNamespace()
    controller = SimpleNamespace(
        run_state_for=lambda _sid: SimpleNamespace(status=ConsoleRunStatus.IDLE),
        _submit_tasks_for_session=lambda _sid: (),
    )
    first = ChatScreen._console_display_history(screen, store, session.id, controller)
    store.append_stream_chunk(answer.id, "answer streamed")
    streamed = ChatScreen._console_display_history(
        screen, store, session.id, controller
    )
    assert streamed[0] != first[0]
    assert streamed[1][-1].content == "answer streamed"
    store.mark_message_complete(answer.id)

    followup = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="followup"
    )
    appended = ChatScreen._console_display_history(
        screen, store, session.id, controller
    )
    assert appended[1][-1].id == followup.id

    store.set_active_leaf(session.id, question.id)
    branched = ChatScreen._console_display_history(
        screen, store, session.id, controller
    )
    assert [message.id for message in branched[1]] == [question.id]
    assert branched[2].request_ids == {question.id}


def test_stream_estimate_ttl_precedes_snapshots_and_draft_edits_still_invalidate(
    monkeypatch,
):
    store = ConsoleChatStore()
    session = store.create_session(
        title="stream estimate",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-4o"),
    )
    question = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(answer.id, "first")
    run = SimpleNamespace(status=ConsoleRunStatus.STREAMING)
    controller = SimpleNamespace(
        run_state_for=lambda _sid: run,
        _submit_tasks_for_session=lambda _sid: (),
        _seeded_greeting_text=lambda *_: "",
    )
    composer = SimpleNamespace(draft="")
    composer.draft_text = lambda: composer.draft
    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: store,
        _ensure_console_chat_controller=lambda: controller,
        _console_composer_or_none=lambda: composer,
        _pending_console_launch_context=None,
        _workspace=SimpleNamespace(_current_console_workspace_context=lambda: None),
        _build_console_staged_context_state=lambda _: SimpleNamespace(summary=""),
        _ensure_console_provider_gateway=lambda: SimpleNamespace(
            cached_context_window=lambda _: ContextWindowResolution(
                200_000, "test window", True
            )
        ),
    )
    screen._console_display_history = lambda *args: ChatScreen._console_display_history(
        screen, *args
    )
    clock = [100.0]
    monkeypatch.setattr(screen_module.time, "monotonic", lambda: clock[0])
    original_messages = store.messages_for_session
    reads = []

    def counted(session_id):
        reads.append(session_id)
        return original_messages(session_id)

    monkeypatch.setattr(store, "messages_for_session", counted)

    def estimate():
        return ChatScreen._console_settings_context_estimate_for_session(
            screen, session.id
        )

    first = estimate()
    store.append_stream_chunk(answer.id, " more text")
    clock[0] += 0.2
    assert estimate() is first
    assert len(reads) == 1

    clock[0] += 1.0
    refreshed = estimate()
    assert refreshed is not first
    assert len(reads) == 2

    store.update_message_content(question.id, "edited question")
    assert estimate() is not refreshed
    assert len(reads) == 3

    composer.draft = "aaaa"
    draft_first = estimate()
    composer.draft = "bbbb"
    assert estimate() is not draft_first
    assert len(reads) == 3
