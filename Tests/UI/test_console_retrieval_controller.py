"""No-mount contracts for Console retrieval ownership."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController


@pytest.mark.parametrize(
    "method_name",
    [
        "_console_chat_dictionary_applier",
        "_console_world_info_applier",
        "_console_dictionary_attach_worker",
        "_console_dictionary_detach_worker",
        "_console_worldbook_attach_worker",
        "_console_worldbook_detach_worker",
    ],
)
def test_dictionary_and_world_book_operations_belong_to_retrieval(method_name):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert method_name in ConsoleRetrievalController.__dict__
    assert method_name not in ChatScreen.__dict__


def test_view_applier_hooks_resolve_retrieval_only_when_invoked():
    from Tests.UI.console_controller_stubs import (
        stub_fleet_controller,
        stub_library_activity_controller,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace()
    stub_fleet_controller(screen)
    stub_library_activity_controller(screen)
    screen._console_chat_store = None
    hooks = screen.console_view_hooks()
    assert "_retrieval" not in screen.__dict__

    calls = []
    for owner in ("first", "replacement"):
        screen._retrieval = SimpleNamespace(
            _console_chat_dictionary_applier=lambda *args: (
                calls.append((owner, "dictionary", args)) or "dictionary text"
            ),
            _console_world_info_applier=lambda *args: (
                calls.append((owner, "world", args)) or "world text"
            ),
        )
        history = ["earlier text"]
        assert (
            hooks["_chat_dictionary_applier"]("conversation", "text")
            == "dictionary text"
        )
        assert (
            hooks["_world_info_applier"]("conversation", "text", history)
            == "world text"
        )
        assert calls[-2:] == [
            (owner, "dictionary", ("conversation", "text")),
            (owner, "world", ("conversation", "text", history)),
        ]
        assert calls[-1][2][2] is history


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["dictionary", "worldbook"])
@pytest.mark.parametrize("operation", ["attach", "detach"])
async def test_picker_without_conversation_releases_only_its_screen_guard(
    kind, operation
):
    controller, state = _controller()
    notifications = []
    controller.app_instance.notify = lambda message, **kwargs: notifications.append(
        (message, kwargs)
    )

    await getattr(controller, f"_console_{kind}_{operation}_worker")()

    assert notifications == [
        ("Start or load a conversation first.", {"severity": "warning"})
    ]
    assert state.dictionary_dialog_active is (kind != "dictionary")
    assert state.worldbook_dialog_active is (kind != "worldbook")


def _controller() -> tuple[ConsoleRetrievalController, SimpleNamespace]:
    """Build the real controller with observable no-mount boundary doubles."""
    state = SimpleNamespace(
        pending=None,
        pending_auto_open=False,
        sent_notice=3,
        sync_result=True,
        sync_calls=0,
        refresh_calls=0,
        dictionary_dialog_active=True,
        worldbook_dialog_active=True,
    )

    def sync_pending_launch_surfaces() -> bool:
        state.sync_calls += 1
        return state.sync_result

    controller = ConsoleRetrievalController(
        app_instance=SimpleNamespace(),
        active_native_session=lambda: None,
        current_conversation_id=lambda: None,
        clear_evidence_sent_notice=lambda: None,
        consume_pending_launch=lambda: None,
        release_consumed_launch=lambda _launch, _result: None,
        is_mounted=lambda: False,
        sync_retrieval_scope_row=lambda: None,
        sync_control_bar=lambda: None,
        request_control_bar_sync=lambda: None,
        dictionary_scope_service=lambda: None,
        finish_dictionary_dialog=(
            lambda: setattr(state, "dictionary_dialog_active", False)
        ),
        finish_worldbook_dialog=(
            lambda: setattr(state, "worldbook_dialog_active", False)
        ),
        set_library_rag_source_scope=lambda _scope: None,
        set_library_rag_query=lambda _query: None,
        run_library_rag_action=lambda: None,
        push_screen=lambda *_args, **_kwargs: None,
        library_rag_source_scope=lambda: ("notes", "media", "conversations"),
        library_rag_top_k=lambda: 5,
        pending_launch=lambda: state.pending,
        set_pending_launch=lambda value: setattr(state, "pending", value),
        set_pending_auto_open=lambda value: setattr(state, "pending_auto_open", value),
        set_evidence_sent_notice=lambda value: setattr(state, "sent_notice", value),
        sync_pending_launch_surfaces=sync_pending_launch_surfaces,
        refresh_screen=lambda: setattr(state, "refresh_calls", state.refresh_calls + 1),
        has_staged_evidence=lambda: False,
    )
    return controller, state


@pytest.mark.unit
def test_retrieval_controller_initializes_owned_cache_state() -> None:
    """The six extracted state fields start with the historical defaults."""
    controller, _state = _controller()

    assert controller._console_retrieval_scope_cache == {}
    assert controller._console_effective_scope_cache == {}
    assert controller._active_dictionaries_summary is None
    assert controller._last_console_dictionary_scope_ids is None
    assert controller._active_world_books_summary is None
    assert controller._last_console_world_book_scope_ids is None


@pytest.mark.unit
def test_retrieval_status_vocabulary_is_controller_owned() -> None:
    """Status projection stays usable without a mounted ChatScreen."""
    controller, _state = _controller()
    searching = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library retrieval",
        status="searching",
    )

    assert controller._console_rag_source_status(None) == "not staged"
    assert (
        controller._console_rag_source_status(None, sent_source_count=2)
        == "sent with the last message · 2 sources"
    )
    assert (
        controller._console_rag_source_status(searching)
        == "retrieving from Library Search/RAG"
    )


@pytest.mark.unit
def test_staging_updates_state_before_sync_and_recomposes_only_as_fallback() -> None:
    """The controller stages once and keeps full recompose as a fallback."""
    controller, state = _controller()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library retrieval",
        status="staged",
    )

    controller._stage_console_library_rag_launch(launch)

    assert state.pending is launch
    assert state.sent_notice is None
    assert state.sync_calls == 1
    assert state.refresh_calls == 0

    state.sync_result = False
    controller._stage_console_library_rag_launch(launch)

    assert state.sync_calls == 2
    assert state.refresh_calls == 1


@pytest.mark.unit
def test_retrieval_controller_owns_no_automatic_placeholder_cleanup() -> None:
    """Automatic preparation placeholders belong to controller/store state."""
    controller, _state = _controller()

    assert not hasattr(controller, "_clear_console_auto_rag_placeholder")


@pytest.mark.asyncio
async def test_rag_scope_save_keeps_fork_transition_through_final_publication(
    monkeypatch,
) -> None:
    controller, _state = _controller()
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test")
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Question",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Answer",
    )
    controller._chat_store = lambda: store
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_transition(_store, _session, _scope):
        entered.set()
        await release.wait()

    monkeypatch.setattr(
        controller,
        "_apply_console_retrieval_scope_save_transition",
        blocking_transition,
    )
    scope = RagScope(
        items=(ScopeItem("note", "1"),),
        updated_at="2026-08-29T00:00:00Z",
    )
    task = asyncio.create_task(
        controller._apply_console_retrieval_scope_save(session, scope)
    )
    await entered.wait()
    eligibility = store.fork_eligibility(assistant.id)
    assert eligibility.eligible is False
    assert "changing" in eligibility.reason.lower()
    release.set()
    await asyncio.wait_for(task, timeout=2)
    assert session.id not in store._fork_source_transitions
