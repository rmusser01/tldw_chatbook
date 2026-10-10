"""The screen's settled history is reused until its store owner changes."""

from tldw_chatbook.UI.Console_Modules.context_spend import ConsoleContextSpendController
from Tests.UI.console_controller_stubs import context_spend_for_test

from tldw_chatbook.UI.Console_Modules import context_spend as context_spend_module
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    ContinuationEventContext,
    FinalContinuation,
    ToolBatchReady,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
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
    first = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
    for _ in range(24):
        assert (
            ConsoleContextSpendController._console_display_history(
                context_spend_for_test(screen), store, session.id, controller
            )
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
    after = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
    assert len(reads) == 2
    assert after[0] != first[0]
    assert after[1][-1].usage is not None

    store.update_message_content(answer.id, "edited answer")
    edited = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
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
    first = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
    store.append_stream_chunk(answer.id, "answer streamed")
    streamed = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
    assert streamed[0] != first[0]
    assert streamed[1][-1].content == "answer streamed"
    store.mark_message_complete(answer.id)

    followup = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="followup"
    )
    appended = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
    )
    assert appended[1][-1].id == followup.id

    store.set_active_leaf(session.id, question.id)
    branched = ConsoleContextSpendController._console_display_history(
        context_spend_for_test(screen), store, session.id, controller
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
    context_spend_for_test(screen)
    screen._context_spend._console_display_history = (
        lambda *args: ConsoleContextSpendController._console_display_history(
            context_spend_for_test(screen), *args
        )
    )
    clock = [100.0]
    monkeypatch.setattr(context_spend_module.time, "monotonic", lambda: clock[0])
    original_messages = store.messages_for_session
    reads = []

    def counted(session_id):
        reads.append(session_id)
        return original_messages(session_id)

    monkeypatch.setattr(store, "messages_for_session", counted)

    def estimate():
        return ConsoleContextSpendController._console_settings_context_estimate_for_session(
            context_spend_for_test(screen), session.id
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


@pytest.mark.parametrize("publication", ["tool_batch", "final"])
def test_continuation_publication_refreshes_warm_history(publication):
    """Missing publication invalidation retains obsolete assistant eligibility."""
    database = CharactersRAGDB(":memory:", "display-continuation-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="continuation projection")
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Use the calculator",
            persist=True,
        )
        owner = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        prefix = "I will calculate it."
        store.append_stream_chunk(owner.id, prefix)
        # Fold before warming: the fold's own revision must not mask a missing
        # continuation-publication revision or schedule an in-memory DB write.
        store.get_message(owner.id)
        context = ContinuationEventContext(
            owner.id, "run-primary", "primary", "persistent"
        )
        active = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model="kimi-k3",
            api_base_url="https://api.moonshot.ai/v1",
            state="active",
            rounds=(
                ContinuationRound(
                    assistant_content=prefix,
                    reasoning_blocks=("private reasoning",),
                    calls=(
                        ContinuationCall(
                            call_id="call-1",
                            name="calculator",
                            arguments='{"expression":"2+2"}',
                            state="pending",
                        ),
                    ),
                ),
            ),
        )
        event = ToolBatchReady(context, active, None)
        expected_content = prefix
        expected_request_ids = {user.id}
        if publication == "final":
            store.persist_provider_continuation_event(event)
            expected_content = "The answer is 4."
            completed_round = replace(
                active.rounds[0],
                calls=(
                    replace(
                        active.rounds[0].calls[0],
                        state="completed",
                        result=ContinuationResult("4"),
                    ),
                ),
            )
            complete = replace(
                active,
                checkpoint_revision=2,
                state="complete",
                rounds=(
                    completed_round,
                    ContinuationRound(
                        expected_content, ("final private reasoning",), ()
                    ),
                ),
            )
            event = FinalContinuation(context, complete, 1, expected_content)
            expected_request_ids = {user.id, owner.id}

        screen = SimpleNamespace()
        controller = SimpleNamespace(
            run_state_for=lambda _sid: SimpleNamespace(
                status=ConsoleRunStatus.STREAMING
            ),
            _submit_tasks_for_session=lambda _sid: (),
        )
        before = ConsoleContextSpendController._console_display_history(
            context_spend_for_test(screen), store, session.id, controller
        )
        revision = store.display_projection_revision(session.id)
        assert before[2].request_ids == (
            {user.id, owner.id} if publication == "tool_batch" else {user.id}
        )

        store.persist_provider_continuation_event(event)
        after = ConsoleContextSpendController._console_display_history(
            context_spend_for_test(screen), store, session.id, controller
        )

        assert store.display_projection_revision(session.id) > revision
        assert after[0] != before[0]
        assert after[2].request_ids == expected_request_ids
        assert after[1][-1].content == expected_content
        assert after[1][-1].assistant_generation_state == (
            "continuation_active" if publication == "tool_batch" else "complete"
        )
    finally:
        database.close_connection()
